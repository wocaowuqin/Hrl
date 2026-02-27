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

        logger.info(f"🎯 [Coordinator] 意图同步: Start={start_node} -> Goal={target_node} (RawIdx={high_action_idx})")

        self.env.current_node_location = int(start_node)
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
        low_state = self.env.get_state()
        low_step = 0
        low_total_reward = 0.0
        episode_done = False
        info = {}
        low_level_stalled = False
        subgoal_achieved = False

        logger.info(f"🚀 [Low Exec] 开始执行 Subgoal: {target_node} (MaxSteps={self.max_low_steps})")

        while low_step < self.max_low_steps and not episode_done:
            low_mask = self.env.get_low_level_action_mask()

            if sum(low_mask) == 0:
                logger.warning(f"⚠️ [Low] 无路可走 (Mask=0) at Node {self.env.current_node_location}")
                low_level_stalled = True
                break

            _, low_action, _ = self.low_agent.select_action(
                low_state, action_mask=low_mask
            )

            next_state, reward, done, truncated_low, info = self.env.step_low_level(low_action)

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
                break

        # ============================================================
        # Phase 3: 奖励结算
        # ============================================================
        high_done = episode_done
        high_reward = self._compute_reward_from_env_info(info)

        if low_step == 0 or low_level_stalled:
            logger.error(f"⛔ [Stall] 僵死检测触发")
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
        if not info: return 0.0
        if info.get('vnf_deployed', False):
            return 20.0
        elif info.get('dest_connected', False):
            return 10.0
        elif info.get('deploy_fail', False):
            return -10.0
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