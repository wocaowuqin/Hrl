import numpy as np
import torch
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HRL_Coordinator:
    """
    HRL协调器 - V40.0 完整诊断版本

    关键修复：
    1. 在step_high_level前调用set_high_level_goal
    2. 正确处理Episode完成标志
    3. 添加详细的诊断日志
    """

    def __init__(self, env, high_agent, low_agent, config=None):
        """初始化协调器"""
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent

        # 配置
        self.config = config or {}
        self.max_low_steps = config.get('max_low_steps', 100)
        self.max_high_steps = config.get('max_high_steps', 100)

        # 高层动作维度
        self.high_action_dim = config.get('high_action_dim', None)
        if self.high_action_dim is None and hasattr(high_agent, 'action_dim'):
            self.high_action_dim = high_agent.action_dim

        # 统计
        self.stats = {
            'total_high_steps': 0,
            'total_low_steps': 0,
            'subgoals_completed': 0,
            'subgoals_timeout': 0,
            'subgoals_failed': 0,
            'high_transitions_stored': 0,
            'low_transitions_stored': 0
        }

        logger.info("✅ HRL协调器初始化完成（V40.0诊断版）")
        logger.info(f"   max_low_steps={self.max_low_steps}")

    def reset(self):
        """重置协调器状态"""
        logger.debug("🔄 [Coordinator] 重置")

    def run_full_episode(self, max_steps=1000):
        """
        执行完整Episode - V40.1 修复版

        修复：
        1. 进度统计直接读取 env.next_vnf_idx，解决显示为0的问题
        2. 增加请求检查和空值保护
        """
        logger.info("🎬 [Episode] 开始")

        # ========================================
        # 1. 重置环境
        # ========================================
        self.env.reset()

        # ========================================
        # 🔥 2. 诊断：检查请求信息
        # ========================================
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            dests = self.env.current_request.get('dest', [])
            source = self.env.current_request.get('source', 0)
            total_vnf = len(vnf_list)
            total_dest = len(dests)

            logger.info(f"📋 [Episode] 请求信息:")
            logger.info(f"   源节点: {source}")
            logger.info(f"   VNF链: {vnf_list} (共{total_vnf}个)")
            logger.info(f"   目的地: {dests} (共{total_dest}个)")
        else:
            logger.error("❌ [Episode] 无请求数据！")
            return 0.0, {
                'success': False,
                'error': 'no_request',
                'steps': 0
            }

        obs = self.env.get_state()

        total_reward = 0.0
        episode_steps = 0
        episode_done = False

        # ========================================
        # 3. Episode主循环
        # ========================================
        while not episode_done and episode_steps < max_steps:
            step_reward, episode_done, step_info = self.run_high_low_cycle(obs)

            total_reward += step_reward
            episode_steps += 1

            # ========================================
            # 🔥 诊断：定期打印进度 (修复版)
            # ========================================
            if episode_steps % 10 == 0 or episode_done:
                # 🟢 [修复] 直接读取环境指针（绝对真理）
                if hasattr(self.env, 'next_vnf_idx'):
                    current_vnf_count = self.env.next_vnf_idx
                else:
                    current_vnf_count = self.env._get_total_vnf_progress()

                # 钳位防止越界显示 (例如 4/3)
                current_vnf_count = min(current_vnf_count, total_vnf)

                # 获取目的地连接数（优先使用保存的最终统计）
                current_dest_count = step_info.get('final_dest_count',
                                                   len(self.env.current_tree.get('connected_dests', set())))

                logger.info(f"📊 [Episode进度] 步数={episode_steps}")
                logger.info(f"   VNF进度: {current_vnf_count}/{total_vnf}")
                logger.info(f"   目的地进度: {current_dest_count}/{total_dest}")
                logger.info(f"   累计奖励: {total_reward:.2f}")

            # ========================================
            # 4. 结束检查与结算日志
            # ========================================
            if episode_done:
                # 再次获取最终状态确保日志准确
                if hasattr(self.env, 'next_vnf_idx'):
                    final_vnf = min(self.env.next_vnf_idx, total_vnf)
                else:
                    final_vnf = self.env._get_total_vnf_progress()

                final_dest = len(self.env.current_tree.get('connected_dests', set()))

                logger.info("=" * 70)
                logger.info(f"✅ [Episode] 完成！")
                logger.info(f"   总步数: {episode_steps}")
                logger.info(f"   总奖励: {total_reward:.2f}")
                logger.info(f"   VNF完成: {final_vnf}/{total_vnf}")
                logger.info(f"   目的地完成: {final_dest}/{total_dest}")
                logger.info("=" * 70)
                break

            obs = self.env.get_state()

            # 超时处理
            if episode_steps >= max_steps:
                logger.warning("=" * 70)
                logger.warning(f"⏰ [Episode] 超时！")
                logger.warning(f"   达到最大步数: {max_steps}")

                # 获取当前进度用于日志
                curr_vnf = self.env.next_vnf_idx if hasattr(self.env, 'next_vnf_idx') else 0
                curr_dest = len(self.env.current_tree.get('connected_dests', set()))

                logger.warning(f"   VNF进度: {min(curr_vnf, total_vnf)}/{total_vnf}")
                logger.warning(f"   目的地进度: {curr_dest}/{total_dest}")
                logger.warning("=" * 70)
                break

        # ========================================
        # 5. 返回结果
        # ========================================
        return total_reward, {
            'steps': episode_steps,
            'reward': total_reward,
            'success': episode_done,
            'timeout': episode_steps >= max_steps and not episode_done,
            'high_steps': self.stats['total_high_steps'],
            'low_steps': self.stats['total_low_steps'],
            'subgoals_completed': self.stats['subgoals_completed'],
            'subgoals_timeout': self.stats['subgoals_timeout'],
            'high_transitions': self.stats['high_transitions_stored'],
            'low_transitions': self.stats['low_transitions_stored']
        }

    def run_high_low_cycle(self, high_obs):
        """
        执行一个 High → Low 循环 (V40.1 修复版)

        修复：
        1. 确保返回给外部统计的是环境真实奖励 (low_total_reward)
        2. 仅将塑形奖励 (high_reward) 存入 Replay Buffer
        """

        # ========================================
        # Phase 1: 高层决策
        # ========================================
        high_state = high_obs
        high_mask = self._get_high_mask()

        _, high_action, high_info = self.high_agent.select_action(
            high_obs,
            action_mask=high_mask
        )

        logger.info(f"🎯 [High] 选择动作: {high_action}")
        self.stats['total_high_steps'] += 1

        # ========================================
        # Phase 2: 设置高层目标
        # ========================================
        target_node_id = self._map_high_action_to_node(high_action)

        # 1. 设置目标变量
        self.env.set_high_level_goal(high_action, target_node_id)

        # 2. 执行高层 Step (主要用于瞬移/状态切换)
        _, _, _, truncated, env_info = self.env.step_high_level(high_action)

        if not truncated:
            logger.error("❌ [High] 环境未进入低层模式 (truncated!=True)")
            return 0.0, False, {'error': 'high_action_failed'}

        logger.info(f"✅ [High] 子目标已设置")

        # ========================================
        # Phase 3: 低层循环执行
        # ========================================
        low_step = 0
        low_total_reward = 0.0
        subgoal_done = False
        episode_done = False

        # 初始低层状态
        # (此时已经是瞬移后的状态了)
        low_state = self.env.get_state()

        while low_step < self.max_low_steps and not subgoal_done and not episode_done:
            # 执行一步低层动作
            # 注意：这里直接调用 select_action 和 env.step，避免重复 get_state
            low_mask = self.env.get_low_level_action_mask()

            _, low_action, _ = self.low_agent.select_action(
                low_state,
                action_mask=low_mask
            )

            next_low_state, low_reward, done, truncated, info = self.env.step_low_level(low_action)
            info['low_action'] = low_action

            # 累加统计
            low_total_reward += low_reward
            low_step += 1
            self.stats['total_low_steps'] += 1

            # 存储低层 Transition
            if low_action is not None:
                self._store_transition(
                    self.low_agent,
                    low_state,
                    low_action,
                    low_reward,
                    next_low_state,
                    done or truncated
                )
                self.stats['low_transitions_stored'] += 1

            # 更新状态指针
            low_state = next_low_state

            # 检查子目标状态 (truncated=True 表示子目标完成或超时，但在 Env 内部区分了 timeout)
            # 注意：Env 的 step_low_level 只有在成功时返回 truncated=True (子目标结束)
            # 或者超时时返回 truncated=True

            # 检查是否因为部署成功/连接成功而结束
            if truncated:
                # 检查是否是超时导致的 truncated
                if info.get('timeout'):
                    subgoal_done = True
                    # 超时逻辑在循环外统一处理
                else:
                    subgoal_done = True
                    logger.info(f"✅ [Low] 子目标完成 ({low_step}步)")
                    self.stats['subgoals_completed'] += 1

            # 检查全剧终
            if done:
                episode_done = True
                logger.info(f"🎉 [Low] Episode完成！")

                # 🔥 V40.3修复：在返回前保存最终统计（避免被清空）
                info['final_dest_count'] = len(self.env.current_tree.get('connected_dests', set()))
                info['final_vnf_count'] = self.env._get_total_vnf_progress()
                logger.info(f"📊 [最终统计] VNF={info['final_vnf_count']}, 目的地={info['final_dest_count']}")

                # Episode 结束：计算高层奖励并存储
                high_next_state = next_low_state  # 此时是终止状态
                high_reward = self._compute_high_reward(low_total_reward, low_step, True)

                self._store_transition(
                    self.high_agent,
                    high_state,
                    high_action,
                    high_reward,
                    high_next_state,
                    True  # done
                )
                self.stats['high_transitions_stored'] += 1

                # 🔥 返回环境真实奖励（使用info）
                return low_total_reward, True, info

        # ========================================
        # Phase 4: 子目标结束 (但 Episode 未结束)
        # ========================================

        # 检查是否是超时失败
        subgoal_failed = False
        if low_step >= self.max_low_steps and not subgoal_done:
            logger.warning(f"⏰ [Low] 超时 ({self.max_low_steps}步)")
            subgoal_done = True
            subgoal_failed = True
            self.stats['subgoals_timeout'] += 1
            self.stats['subgoals_failed'] += 1

        # 计算高层奖励 (用于训练 High Agent)
        success = subgoal_done and not subgoal_failed
        high_reward = self._compute_high_reward(low_total_reward, low_step, success)

        # 获取高层的新状态 (即低层结束时的状态)
        high_next_state = low_state

        # 存储 Transition (用 high_reward)
        self._store_transition(
            self.high_agent,
            high_state,
            high_action,
            high_reward,
            high_next_state,
            False  # not done
        )
        self.stats['high_transitions_stored'] += 1

        # 🔥 关键修复：返回给外部统计的是环境真实奖励 (low_total_reward)
        return low_total_reward, False, {
            'high_action': high_action,
            'subgoal_steps': low_step,
            'subgoal_done': subgoal_done,
            'subgoal_failed': subgoal_failed,
            'low_reward': low_total_reward
        }

    def _map_high_action_to_node(self, high_action):
        """
        🔥 [V40.1 修复版] 映射高层动作到目标节点

        修正：直接使用 high_action 作为节点ID (不再取模)，
        前提是 High Agent 的动作空间就是全网节点 (0 ~ N-1)。
        """
        # 1. 安全检查
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            logger.warning("⚠️ [Coordinator] 无当前请求，使用默认节点0")
            return 0

        target_node_id = int(high_action)
        vnf_list = self.env.current_request.get('vnf', [])

        # 2. 判断当前阶段
        # 注意：这里依赖 env.next_vnf_idx 是准确的
        if self.env.next_vnf_idx < len(vnf_list):
            # ========================================
            # 阶段1：VNF部署 - 目标应该是 DC 节点
            # ========================================
            valid_dc = [n for n in getattr(self.env, 'dc_nodes', [])
                        if self.env._is_valid_node(n)]

            # 校验：Agent 选的节点是否在 DC 列表中
            if target_node_id not in valid_dc:
                # 这是一个非法动作（Agent 选了非 DC 节点）
                # 如果使用了 Action Mask，这种情况理论上不应发生
                logger.warning(f"⚠️ [Coordinator] 高层动作 {target_node_id} 不是有效DC节点 {valid_dc}")

                # 兜底策略：如果真的选错了，可以强制修正（可选）
                # target_node_id = valid_dc[0]

            logger.debug(f"🎯 [Coordinator] VNF部署: Action {high_action} -> Node {target_node_id}")

        else:
            # ========================================
            # 阶段2：目的地连接 - 目标应该是剩余目的地
            # ========================================
            dests = self.env.current_request.get('dest', [])
            connected = self.env.current_tree.get('connected_dests', set())
            remaining = list(set(dests) - connected)

            # 校验：Agent 选的节点是否是剩余目的地
            if target_node_id not in remaining:
                logger.warning(f"⚠️ [Coordinator] 高层动作 {target_node_id} 不是有效剩余目的地 {remaining}")
                # 兜底策略（可选）
                # if remaining: target_node_id = remaining[0]

            logger.debug(f"🎯 [Coordinator] 目的连接: Action {high_action} -> Node {target_node_id}")

        return target_node_id

    def _execute_low_level_step(self):
        """执行一步低层动作"""
        low_obs = self.env.get_state()
        low_mask = self.env.get_low_level_action_mask()

        # 🔥 Phase3适配：低层也不需要unconnected_dests
        _, low_action, _ = self.low_agent.select_action(
            low_obs,
            action_mask=low_mask
        )

        next_obs, reward, done, truncated, info = self.env.step_low_level(low_action)
        info['low_action'] = low_action

        return next_obs, reward, done, truncated, info

    def _store_transition(self, agent, state, action, reward, next_state, done):
        """存储transition"""
        if hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward, next_state, done)
        elif hasattr(agent, 'remember'):
            agent.remember(state, action, reward, next_state, done)
        elif hasattr(agent, 'store'):
            agent.store(state, action, reward, next_state, done)

    def _get_high_mask(self):
        """获取高层动作掩码"""
        if hasattr(self.env, 'get_high_level_action_mask'):
            return self.env.get_high_level_action_mask()
        else:
            if self.high_action_dim is not None:
                n_actions = self.high_action_dim
            elif hasattr(self.env, 'n'):
                n_actions = self.env.n
            else:
                n_actions = 28
            return np.ones(n_actions, dtype=np.float32)

    def _compute_high_reward(self, low_reward, steps, success):
        """计算高层奖励"""
        if success:
            high_reward = 10.0 + low_reward
        else:
            high_reward = -5.0 - (steps * 0.1)
        return high_reward

    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()

    def print_stats(self):
        """打印统计"""
        logger.info("=" * 60)
        logger.info("📊 HRL协调器统计")
        logger.info("=" * 60)
        logger.info(f"高层总步数: {self.stats['total_high_steps']}")
        logger.info(f"低层总步数: {self.stats['total_low_steps']}")
        logger.info(f"子目标完成: {self.stats['subgoals_completed']}")
        logger.info(f"子目标超时: {self.stats['subgoals_timeout']}")
        logger.info(f"高层transitions: {self.stats['high_transitions_stored']}")
        logger.info(f"低层transitions: {self.stats['low_transitions_stored']}")

        if self.stats['total_high_steps'] > 0:
            avg_low_per_high = self.stats['total_low_steps'] / self.stats['total_high_steps']
            logger.info(f"平均低层步数/高层: {avg_low_per_high:.2f}")

        total_subgoals = self.stats['subgoals_completed'] + self.stats['subgoals_failed']
        if total_subgoals > 0:
            success_rate = self.stats['subgoals_completed'] / total_subgoals
            logger.info(f"子目标成功率: {success_rate:.2%}")

        logger.info("=" * 60)