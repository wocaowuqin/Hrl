import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HRL_Coordinator:
    """
    纯HRL协调器 - V53.0 (移除暴力检测版)
    包含:
    1. 接收 HighLevel 的 Truncated 信号
    2. 强制清空 Agent 的 Subgoal 缓存
    3. 🔥 [关键修复] 移除暴力目标检测，强制 Agent 执行停留动作
    """

    def __init__(self, env, high_agent, low_agent, config=None):
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.config = config or {}

        # 仅保留引用，不主动管理释放
        if hasattr(env, 'resource_mgr'):
            self.resource_mgr = env.resource_mgr
            logger.info("🔗 HRL 已连接 AllResourceManager（仅引用）")
        else:
            self.resource_mgr = None

        self.max_low_steps = self.config.get('max_low_steps', 50)
        # [关键修复] 同步 env.max_subgoal_steps，确保 Low-Level 超时阈值一致
        # 默认50步（与max_low_steps一致），避免Low-Level内部过早超时
        self.env.max_subgoal_steps = self.config.get('max_subgoal_steps', 50)
        self.stats = defaultdict(int)
        self.current_episode = 0
        self.resources_released = False

        logger.info("🚀 纯HRL协调器初始化完成（V53.0 移除暴力检测版）")

    def run_high_low_cycle(self, high_obs, training=True):
        """
        🔥 [V52.0] 核心逻辑：Truncated -> Wipe Agent Memory -> Re-sample
        """
        # 重置释放标志
        if hasattr(self, 'resources_released'):
            self.resources_released = False

        # ============================================================
        # Phase 1: High-Level 决策
        # ============================================================
        # 1. 获取 Mask
        high_mask = self.env.get_high_level_action_mask()

        # 过滤本 episode 已确认不可达/部署失败的节点
        if hasattr(self, '_unreachable_targets') and self._unreachable_targets:
            for idx in self._unreachable_targets:
                if idx < len(high_mask):
                    high_mask[idx] = 0
            logger.debug(f"[Coordinator] 过滤不可达节点: {self._unreachable_targets}")

        if sum(high_mask) == 0:
            logger.error("❌ 无可用高层动作 (Mask全0)")
            return 0.0, True, {'error': 'no_high_actions'}

        # 2. Agent 选择动作
        high_action_idx, high_action_remapped, agent_info = self.high_agent.select_action(
            high_obs, action_mask=high_mask
        )

        # ------------------------------------------------------------
        # 目标解析
        # ------------------------------------------------------------
        target_node = None
        if agent_info and 'subgoal' in agent_info and agent_info['subgoal'] is not None:
            target_node = agent_info['subgoal']
        else:
            valid_indices = np.where(high_mask > 0)[0]
            if high_action_idx < len(valid_indices):
                target_node = int(valid_indices[high_action_idx])
            else:
                target_node = high_action_idx

        # 同步起点
        start_node = agent_info.get('start_node')
        if start_node is None:
            start_node = self.env.current_node_location

        logger.info(f"🎯 [Coordinator] 意图同步: ActualPos={self.env.current_node_location} | "
                    f"AgentPredict={start_node} -> Goal={target_node} (RawIdx={high_action_idx})")

        # [P1修复] 不在此处重置起点，由 set_high_level_goal 统一管理
        # 原代码: self.env.current_node_location = int(start_node)
        self.env.set_high_level_goal(high_action_idx, target_node, start_node)

        # ------------------------------------------------------------
        # High-Level Step (执行决策)
        # ------------------------------------------------------------
        _, _, done, truncated, high_info = self.env.step_high_level(high_action_idx)

        # 🔥 [关键逻辑] 处理 High Level 的立即终止信号
        if done:
            logger.info("🎉 High-Level 判定 Episode 结束")
            return 0.0, True, {'episode_done': True}

        if truncated:
            logger.warning(f"🔄 [Coordinator] High-Level 返回 Truncated (目标 {target_node} 无效/已完成)")
            logger.warning("   -> 触发强制重采机制 (Force Re-sample)")

            # 🔥🔥🔥 [核心修复] 强制清空 Agent 的缓存 🔥🔥🔥
            # 这行代码逼迫 Agent 必须在下一轮重新评估 Mask，而不是使用旧的 current_subgoal

            # 1. 清空当前目标缓存
            if hasattr(self.high_agent, 'current_subgoal'):
                logger.info(f"   🧹 清空 Agent 缓存: current_subgoal ({self.high_agent.current_subgoal}) -> None")
                self.high_agent.current_subgoal = None

            # 2. 重置步数计数器 (使其超过 horizon，触发 update)
            if hasattr(self.high_agent, 'subgoal_steps'):
                horizon = getattr(self.high_agent, 'subgoal_horizon', 10)
                self.high_agent.subgoal_steps = horizon + 1
                logger.info(f"   🧹 重置 Agent 计时: subgoal_steps -> {self.high_agent.subgoal_steps}")

            # 3. 针对不同 Agent 实现的兼容性清空
            if hasattr(self.high_agent, 'subgoal_step_count'):
                self.high_agent.subgoal_step_count = 999

            # 直接返回，不执行下面的 Low Level Loop
            # Coordinator 的 run_episode 会继续循环，重新调用 run_high_low_cycle
            return 0.0, False, {'subgoal_truncated': True}

        # ============================================================
        # Phase 2: Low-Level 执行 (只有在非 Truncated 时才执行)
        # ============================================================
        # [修复A] 统一步数管理：每个 cycle 开始时重置步数，避免 Low-Level 内部超时残留
        self.env.subgoal_step_count = 0
        if hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []

        low_state = self.env.get_state()
        low_step = 0
        low_total_reward = 0.0
        episode_done = False
        info = {}
        low_level_stalled = False
        subgoal_achieved = False
        start_pos_before_low = self.env.current_node_location  # 记录起始位置，用于卡死检测

        logger.info(f"🚀 [Low Exec] 开始执行 Subgoal: {target_node} (MaxSteps={self.max_low_steps})")

        # ==============================================================
        # 🔥 优化：带宽感知最短路径引导
        # 先用图算法算出 current_pos → target_node 的最优路径
        # 然后沿路径逐步执行 step_low_level（保持所有奖励/带宽/信号逻辑）
        # ==============================================================
        planned_path = None
        path_idx = 0
        blocked_edges = set()
        replan_count = 0
        no_progress_count = 0
        if hasattr(self.env, 'low_level_controller') and \
                hasattr(self.env.low_level_controller, 'compute_bw_aware_path'):
            current_pos = self.env.current_node_location
            planned_path = self.env.low_level_controller.compute_bw_aware_path(current_pos, target_node)
            if planned_path and len(planned_path) > 1:
                planned_path = planned_path[1:] + [target_node]
                logger.info(f"🗺️ [PathGuide] 规划路径: {self.env.current_node_location} → "
                            f"{' → '.join(str(n) for n in planned_path[:5])}{'...' if len(planned_path)>5 else ''} "
                            f"(共{len(planned_path)}步)")
            elif planned_path and len(planned_path) == 1:
                planned_path = [target_node]
                logger.info(f"🗺️ [PathGuide] 已在目标节点{target_node}，直接STAY")
            else:
                logger.warning(f"⚠️ [PathGuide] 无法规划路径到{target_node}，回退RL")
                planned_path = None
                self._unreachable_targets.add(target_node)

        while low_step < self.max_low_steps and not episode_done:
            if planned_path is not None and path_idx < len(planned_path):
                low_action = planned_path[path_idx]
                path_idx += 1
            else:
                low_mask = self.env.get_low_level_action_mask()
                if sum(low_mask) == 0:
                    logger.warning(f"⚠️ [Low] 无路可走 (Mask=0) at Node {self.env.current_node_location}")
                    low_level_stalled = True
                    break
                _, low_action, _ = self.low_agent.select_action(
                    low_state, action_mask=low_mask
                )

            pos_before = self.env.current_node_location
            next_state, reward, done, truncated_low, info = self.env.step_low_level(low_action)
            pos_after = self.env.current_node_location

            # 无进展检测（位置没变且不是成功信号）
            is_success = info.get('dest_connected') or info.get('vnf_deployed') \
                or info.get('all_vnf_deployed') or info.get('episode_complete')
            if pos_after == pos_before and not is_success:
                no_progress_count += 1
                failed_edge = info.get('edge')
                if failed_edge:
                    blocked_edges.add(tuple(sorted(failed_edge)))
            else:
                no_progress_count = 0

            # 路径受阻 → 排除失败边重规划（最多3次）
            if info.get('error') in ('no_bandwidth', 'resource_failure'):
                if planned_path is not None:
                    replan_count += 1
                    if replan_count <= 3:
                        current_pos = self.env.current_node_location
                        planned_path = None
                        if hasattr(self.env, 'low_level_controller') and \
                                hasattr(self.env.low_level_controller, 'compute_bw_aware_path'):
                            new_path = self.env.low_level_controller.compute_bw_aware_path(
                                current_pos, target_node, excluded_edges=blocked_edges)
                            if new_path and len(new_path) > 1:
                                planned_path = new_path[1:] + [target_node]
                                path_idx = 0
                                logger.info(f"🔄 [PathGuide] 排除{len(blocked_edges)}条边，"
                                            f"第{replan_count}次重规划: {len(planned_path)}步")
                            else:
                                logger.warning(f"⚠️ [PathGuide] 排除边后无路径，回退RL")
                                self._unreachable_targets.add(target_node)
                    else:
                        logger.warning(f"⚠️ [PathGuide] 重规划{replan_count}次仍受阻，放弃路径引导")
                        planned_path = None

            # 连续无进展 → 目标不可达，直接退出
            if no_progress_count >= 5:
                logger.warning(f"⚠️ [Low] 连续{no_progress_count}步无进展 at {pos_after}，"
                               f"目标{target_node}不可达，退出")

                # 强制加入不可达列表并清空Agent缓存，防止高层死循环选中该节点
                self._unreachable_targets.add(target_node)
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                if hasattr(self.high_agent, 'subgoal_steps'):
                    self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1

                info['stuck'] = True
                break

            if training:
                self._store_transition(
                    self.low_agent, low_state, low_action,
                    reward, next_state, done
                )

            low_total_reward += reward
            low_state = next_state
            low_step += 1

            # ----------------------------------------------------
            # 终止条件检查
            # ----------------------------------------------------
            # 必须由 info 中的明确信号触发终止
            if info.get('subgoal_done', False):
                logger.info(f"✅ [Coordinator] 检测到 Subgoal 完成信号")

                # 即使完成了，也建议清空 Agent 缓存，防止下一轮还想回来
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None

                subgoal_achieved = True
                break

            if info.get('goal_reached', False) or info.get('current_goal_satisfied', False):
                logger.info(f"✅ [Coordinator] 检测到 Goal Reached 信号")
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            # 🔥🔥🔥 [P0修复] 捕获 Low-Level 的阶段性成功信号
            # dest_connected: 连接了一个（非最后）目的地
            # vnf_deployed: 部署了一个（非最后）VNF
            # all_vnf_deployed: 所有 VNF 部署完成
            # episode_complete: 所有目的地已连接（Episode成功）
            if info.get('dest_connected', False) or info.get('vnf_deployed', False) \
                    or info.get('all_vnf_deployed', False):
                # logger.info(f"✅ [Coordinator] 捕获阶段性成功信号: "
                #             f"dest_connected={info.get('dest_connected', False)}, "
                #             f"vnf_deployed={info.get('vnf_deployed', False)}, "
                #             f"all_vnf={info.get('all_vnf_deployed', False)}")
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('episode_complete', False) or info.get('all_destinations_connected', False):
                logger.info(f"🎉 [Coordinator] 捕获 Episode 完成信号!")
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                episode_done = True
                break

            # 🛑 [V53.0 关键修复] 移除暴力检测！
            # ----------------------------------------------------------------------------------
            # 原有的逻辑会检测 "current_loc == target_node" 且 "target_node in connected_dests"。
            # 这会导致 Agent 只要路过目标节点（尚未执行 STAY/CONNECT），就会被判定为完成，
            # 从而跳过了 LowLevelController 中至关重要的 "连接建立" 和 "奖励获取" 步骤。
            # 这正是导致 "树冗余高" 和 "反复徘徊" 的核心原因。
            #
            # 已注释掉以下代码块：
            # ----------------------------------------------------------------------------------
            # current_loc = self.env.current_node_location
            # if current_loc == target_node:
            #     if hasattr(self.env, 'current_tree') and self.env.current_tree:
            #         connected_dests = self.env.current_tree.get('connected_dests', set())
            #         if target_node in connected_dests:
            #             logger.info(f"✅ [Coordinator] 暴力检测: 节点 {target_node} 已在连接树中")
            #             if hasattr(self.high_agent, 'current_subgoal'):
            #                 self.high_agent.current_subgoal = None
            #             subgoal_achieved = True
            #             break
            # ----------------------------------------------------------------------------------

            if done:
                episode_done = True
            if truncated_low:
                logger.info(f"📊 [Low] Cycle结束: 步数={low_step}, 退出原因=truncated, "
                            f"info_keys={list(info.keys())}, pos={self.env.current_node_location}")
                # 🔥 部署失败（非DC节点 / 资源不足）→ 标记为不可达，防止高层反复选它
                if info.get('deploy_fail'):
                    self._unreachable_targets.add(target_node)
                    logger.warning(f"[Coordinator] 节点{target_node}部署失败，加入不可达列表")
                break

        # ============================================================
        # Phase 3: 奖励结算
        # ============================================================
        if low_step > 0 and not subgoal_achieved:
            logger.info(f"📊 [Low] Cycle结束: 步数={low_step}, subgoal_achieved={subgoal_achieved}, "
                        f"pos={self.env.current_node_location}")

        # [关键修复] 卡死与超时检测：只要达到最大步数未完成，强制清空缓存并标记不可达
        actual_end_pos = self.env.current_node_location
        if low_step >= self.max_low_steps and not subgoal_achieved:
            # 无论位置变没变，只要超时未完成，就强制清空HighAgent缓存
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            if hasattr(self.high_agent, 'subgoal_steps'):
                self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1

            # 核心：将超时无法到达的节点加入不可达列表，防止高层再次 Mask 选中
            self._unreachable_targets.add(target_node)

            if actual_end_pos == start_pos_before_low:
                logger.error(f"⛔ [Stuck] Agent走了{low_step}步但位置未变 "
                             f"(pos={actual_end_pos}, goal={target_node})，强制换目标")
                info['stuck'] = True
            else:
                logger.warning(
                    f"⏰ [Timeout] Agent耗尽步数未到达目标{target_node} (停在{actual_end_pos})，已标记不可达并换目标")

        high_done = episode_done
        high_reward = self._compute_reward_from_env_info(info)

        if low_step == 0 or low_level_stalled or info.get('stuck', False):
            logger.error(f"⛔ [Stall] 僵死检测触发 (stuck={info.get('stuck', False)})")
            high_reward = -15.0
            info['stalled'] = True

        # 全局完成检测
        if not high_done and self.env.current_request:
            if hasattr(self.env, 'high_level_controller') and hasattr(self.env.high_level_controller,
                                                                      '_is_all_tasks_completed'):
                try:
                    completed, status = self.env.high_level_controller._is_all_tasks_completed()
                    if completed:
                        logger.info(f"✅ [High Check] 全局任务完成: {status}")
                        high_done = True
                        high_reward = 30.0
                        episode_done = True
                except:
                    pass

        if training:
            high_next_state = None if high_done else self.env.get_state()
            self._store_transition(
                self.high_agent, high_obs, high_action_idx,
                high_reward, high_next_state, high_done
            )

        self._update_stats(high_done, info)

        return low_total_reward, high_done, {
            'high_action': high_action_idx,
            'target_node': target_node,
            'low_steps': low_step,
            'high_reward': high_reward,
            'info': info
        }

    # ... (run_episode, _fallback_success_check, _compute_reward_from_env_info, _store_transition, _update_stats, _reset_episode_stats, get_stats 保持不变) ...
    def run_episode(self, training=True, max_steps=100):
        self.current_episode += 1
        self.resources_released = False

        logger.info(f"\n{'=' * 50}")
        logger.info(f"📚 Episode {self.current_episode}")
        logger.info(f"{'=' * 50}")

        high_obs = self.env.reset()
        self._unreachable_targets = set()  # 每个 episode 清空
        episode_done = False
        total_reward = 0.0
        total_steps = 0

        while not episode_done and total_steps < max_steps:
            cycle_reward, done, info = self.run_high_low_cycle(
                high_obs, training=training
            )

            total_reward += cycle_reward
            total_steps += 1
            episode_done = done

            if not episode_done:
                high_obs = self.env.get_state()

        vnf_success = False
        dest_success = False
        episode_success = False
        completion_status = "未知"

        try:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                vnf_list = self.env.current_request.get('vnf', [])
                dest_list = self.env.current_request.get('dest', [])

                if hasattr(self.env, 'high_level_controller') and \
                        hasattr(self.env.high_level_controller, '_is_all_tasks_completed'):
                    try:
                        completed, status = self.env.high_level_controller._is_all_tasks_completed()
                        episode_success = completed
                        completion_status = status
                    except Exception as e:
                        episode_success = self._fallback_success_check()
                        completion_status = "回退检测"
                else:
                    episode_success = self._fallback_success_check()
                    completion_status = "回退检测"

                if hasattr(self.env, 'next_vnf_idx'):
                    vnf_progress = self.env.next_vnf_idx
                    vnf_success = vnf_progress >= len(vnf_list)

                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_dests = self.env.current_tree.get('connected_dests', set())
                    dest_success = len(connected_dests) >= len(dest_list)

        except Exception as e:
            logger.error(f"❌ 结果判定异常: {e}")
            episode_success = False

        if episode_success:
            logger.info(f"✅ [Episode Success] {completion_status}")
            self.resources_released = True
        else:
            logger.info(f"❌ [Episode Fail] {completion_status}")
            try:
                if (hasattr(self.env, 'resource_mgr') and
                        hasattr(self.env.resource_mgr, '_archive_request')):
                    self.env.resource_mgr._archive_request(
                        success=False, already_rolled_back=False)
                    logger.debug("[Coordinator] 失败Episode资源已回滚")
            except Exception as e:
                logger.error(f"[Coordinator] 资源回滚异常: {e}")

        logger.info(f"📊 Summary: Steps={total_steps} | TotalReward={total_reward:.2f}")

        self._reset_episode_stats()

        return total_reward, {
            'steps': total_steps,
            'success': episode_success,
            'reward': total_reward,
            'subgoals_ok': self.stats.get('subgoals_ok', 0),
            'subgoals_fail': self.stats.get('subgoals_fail', 0),
            'vnf_success': vnf_success,
            'dest_success': dest_success,
            'completion_status': completion_status
        }

    def _fallback_success_check(self):
        if not hasattr(self.env, 'current_request') or not self.env.current_request:
            return True
        try:
            vnf_list = self.env.current_request.get('vnf', [])
            dest_list = self.env.current_request.get('dest', [])

            if hasattr(self.env, 'next_vnf_idx'):
                vnf_success = self.env.next_vnf_idx >= len(vnf_list)
            elif hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'next_vnf_idx'):
                vnf_success = self.env.resource_mgr.next_vnf_idx >= len(vnf_list)
            else:
                vnf_success = False

            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                connected_dests = self.env.current_tree.get('connected_dests', set())
                dest_success = len(connected_dests) >= len(dest_list)
            else:
                dest_success = False

            return vnf_success and dest_success
        except:
            return False

    def _compute_reward_from_env_info(self, info):
        """
        [P1修复] 完善奖励映射，确保所有里程碑事件都有对应奖励
        """
        if not info: return 0.0
        # Episode 完成（最高优先级）
        if info.get('episode_complete', False) or info.get('all_destinations_connected', False):
            return 30.0
        # 所有 VNF 部署完成（里程碑）
        if info.get('all_vnf_deployed', False):
            return 25.0
        # 单个 VNF 部署成功
        if info.get('vnf_deployed', False):
            return 20.0
        # 单个目的地连接成功
        elif info.get('dest_connected', False):
            return 10.0
        # 部署失败
        elif info.get('deploy_fail', False):
            return -10.0
        # 超时
        elif info.get('timeout', False):
            return -5.0
        return 0.0

    def _store_transition(self, agent, state, action, reward, next_state, done):
        if hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward, next_state, done)
        elif hasattr(agent, 'memory') and hasattr(agent.memory, 'store'):
            agent.memory.store(state, action, reward, next_state, done)
        elif hasattr(agent, 'store'):
            agent.store(state, action, reward, next_state, done)

    def _update_stats(self, high_done, info):
        if info:
            if info.get('vnf_deployed', False) or info.get('dest_connected', False):
                self.stats['subgoals_ok'] += 1
            elif info.get('deploy_fail', False) or info.get('timeout', False):
                self.stats['subgoals_fail'] += 1
        if high_done:
            self.stats['episodes_completed'] += 1

    def _reset_episode_stats(self):
        self.stats['subgoals_ok'] = 0
        self.stats['subgoals_fail'] = 0

    def get_stats(self):
        return dict(self.stats)