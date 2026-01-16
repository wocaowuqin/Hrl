import numpy as np
import torch
from collections import defaultdict, deque
import time
import logging

logger = logging.getLogger(__name__)


class HRL_Coordinator:
    """
    🎮 HRL时序协调器 - 解决时间尺度不匹配问题

    核心功能：
    1. 管理高层和低层的执行时序
    2. 确保 State(t) → High Action → Low Execution → State(t+1) 闭环
    3. 协调高层和低层的状态同步
    4. 统计和监控训练过程
    """

    def __init__(self, env, high_agent, low_agent, config=None):
        """
        初始化协调器

        Args:
            env: SFC_HIRL_Env 环境实例
            high_agent: 高层Agent
            low_agent: 低层Agent
            config: 配置字典
        """
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent

        # 配置参数
        self.config = config or {}
        self.max_low_steps = config.get('max_low_steps', 100)
        self.max_high_steps = config.get('max_high_steps', 100)
        self.use_masking = config.get('use_masking', True)

        # 状态同步缓存
        self.current_high_state = None
        self.current_low_state = None
        self.last_high_action = None
        self.last_high_reward = 0.0

        # 执行统计
        self.stats = defaultdict(int)
        self.episode_stats = {
            'total_reward': 0.0,
            'high_steps': 0,
            'low_steps': 0,
            'vnf_deployments': 0,
            'dest_connections': 0,
            'failures': 0,
            'success': False
        }

        # 历史记录（用于调试）
        self.history = {
            'high_actions': [],
            'low_actions': [],
            'rewards': [],
            'states': []
        }

        self.current_goal = None  # 当前高层目标
        self.current_subgoal = None  # 当前子目标
        self._low_step_count = 0  # 低层步数计数器
        self._last_phase = None  # 上一次的阶段
        self.last_transition = None  # 最近的 transition（供 Trainer 使用）
        self.last_high_action = None  # 最近的高层动作
        logger.info("✅ HRL时序协调器初始化完成")
        logger.info(f"配置: max_low_steps={self.max_low_steps}, max_high_steps={self.max_high_steps}")

    def reset(self):
        """重置协调器状态"""
        self.current_high_state = None
        self.current_low_state = None
        self.last_high_action = None
        self.last_high_reward = 0.0

        # 重置episode统计
        self.episode_stats = {
            'total_reward': 0.0,
            'high_steps': 0,
            'low_steps': 0,
            'vnf_deployments': 0,
            'dest_connections': 0,
            'failures': 0,
            'success': False
        }

        # 清空历史记录
        self.history = {
            'high_actions': [],
            'low_actions': [],
            'rewards': [],
            'states': []
        }

        logger.debug("协调器状态已重置")

    def run_high_low_cycle(self, training_mode=True):
        """
        执行一个完整的高层-低层循环

        Returns:
            high_state: 高层当前状态
            high_action: 高层动作
            high_reward: 高层奖励
            next_high_state: 高层下一个状态（同步后）
            high_done: 高层是否完成
            info: 附加信息
        """
        # ========================================
        # 1. 检查是否需要进行高层决策
        # ========================================
        need_high_decision = self._check_need_high_decision()

        if need_high_decision:
            # ========================================
            # 2. 高层决策阶段
            # ========================================
            high_state = self._get_synchronized_high_state()

            # 获取高层动作掩码
            high_mask = None
            if self.use_masking:
                try:
                    high_mask = self.env.get_high_level_action_mask()
                except AttributeError:
                    logger.warning("环境不支持get_high_level_action_mask，将使用全掩码")
                    high_mask = np.ones(self.env.high_action_space, dtype=bool)

            # 高层Agent选择动作
            high_action = self.high_agent.select_action(
                high_state,
                mask=high_mask,
                training=training_mode
            )

            self.last_high_action = high_action
            self.history['high_actions'].append(high_action)

            # ========================================
            # 3. 执行高层动作（只设定目标）
            # ========================================
            start_time = time.time()

            _, _, high_done, _, high_info = self.env.step_high_level(high_action)

            high_decision_time = time.time() - start_time

            # 记录高层决策
            self.stats['high_decisions'] += 1
            self.episode_stats['high_steps'] += 1

            logger.info(
                f"🔝 [高层决策] 步骤{self.stats['high_decisions']}，动作={high_action}，用时={high_decision_time:.3f}s")

            # 如果高层任务完成（所有目的地已连接）
            if high_done:
                logger.info("✅ 高层任务完成：所有目的地已连接")

                # 获取最终状态
                final_state = self.env.get_high_level_state_graph()

                return (
                    high_state,
                    high_action,
                    100.0,  # 完成奖励
                    final_state,
                    True,
                    {
                        'episode_complete': True,
                        'high_done': True,
                        'message': '所有目的地连接完成'
                    }
                )

            # ========================================
            # 4. 低层执行循环（同步关键）
            # ========================================
            low_execution_result = self._execute_low_level_loop(training_mode)

            # 更新统计
            self.episode_stats['low_steps'] += low_execution_result['steps']
            self.episode_stats['total_reward'] += low_execution_result['total_reward']

            # ========================================
            # 5. 获取同步后的下一个高层状态
            # ========================================
            # 🔥 关键：低层执行完后，状态已更新，现在获取同步后的高层状态
            next_high_state = self._get_synchronized_high_state()

            # ========================================
            # 6. 计算高层奖励
            # ========================================
            high_reward = self._calculate_high_reward(
                low_execution_result['total_reward'],
                low_execution_result['info'],
                low_execution_result['steps']
            )

            self.last_high_reward = high_reward
            self.history['rewards'].append(high_reward)

            # ========================================
            # 7. 返回结果
            # ========================================
            return (
                high_state,
                high_action,
                high_reward,
                next_high_state,
                False,  # high_done
                {
                    **high_info,
                    **low_execution_result['info'],
                    'low_steps': low_execution_result['steps'],
                    'low_total_reward': low_execution_result['total_reward'],
                    'high_reward': high_reward,
                    'execution_time': high_decision_time + low_execution_result['execution_time']
                }
            )
        else:
            # 不需要高层决策，直接返回当前状态
            current_state = self._get_synchronized_high_state()
            return (
                current_state,
                None,
                0.0,
                current_state,
                False,
                {'message': '等待低层执行完成'}
            )

    def _execute_low_level_loop(self, training_mode=True):
        """
        执行低层循环，直到子任务完成或达到最大步数

        Returns:
            dict: 包含总奖励、步数、执行时间等信息
        """
        logger.info("⚙️ [低层执行] 开始执行高层指令...")

        start_time = time.time()
        total_reward = 0.0
        step_count = 0
        low_done = False
        last_info = {}

        # 低层执行循环
        while not low_done and step_count < self.max_low_steps:
            step_count += 1

            # ========================================
            # a. 获取低层状态
            # ========================================
            low_state = self.env.get_state()

            # ========================================
            # b. 获取低层动作掩码
            # ========================================
            low_mask = None
            if self.use_masking:
                try:
                    low_mask = self.env.get_low_level_action_mask()
                except AttributeError:
                    logger.warning("环境不支持get_low_level_action_mask，将使用全掩码")
                    low_mask = np.ones(self.env.n, dtype=bool)

            # ========================================
            # c. 低层Agent选择动作
            # ========================================
            low_action = self.low_agent.select_action(
                low_state,
                mask=low_mask,
                training=training_mode
            )

            self.history['low_actions'].append(low_action)

            # ========================================
            # d. 执行低层动作
            # ========================================
            _, low_reward, low_terminated, low_truncated, low_info = \
                self.env.step_low_level(low_action)

            total_reward += low_reward

            # ========================================
            # e. 检查低层任务是否完成
            # ========================================
            last_info = low_info

            # 成功完成部署
            if low_info.get('deploy_success', False):
                self.episode_stats['vnf_deployments'] += 1
                self.stats['successful_deployments'] += 1
                low_done = True
                logger.info(f"✅ 低层任务完成: VNF部署成功 (步骤{step_count})")

            # 成功连接目的地
            elif low_info.get('connection_success', False):
                self.episode_stats['dest_connections'] += 1
                self.stats['successful_connections'] += 1
                low_done = True
                logger.info(f"✅ 低层任务完成: 目的地连接成功 (步骤{step_count})")

            # 失败情况
            elif low_info.get('deploy_fail', False):
                self.episode_stats['failures'] += 1
                self.stats['failed_deployments'] += 1
                low_done = True
                logger.warning(f"❌ 低层任务失败: VNF部署失败")

            elif low_info.get('connection_fail', False):
                self.episode_stats['failures'] += 1
                self.stats['failed_connections'] += 1
                low_done = True
                logger.warning(f"❌ 低层任务失败: 目的地连接失败")

            elif low_info.get('path_fail', False):
                self.episode_stats['failures'] += 1
                self.stats['failed_paths'] += 1
                low_done = True
                logger.warning(f"❌ 低层任务失败: 路径建立失败")

            # 超时或非法动作
            elif low_info.get('timeout', False) or low_info.get('invalid', False):
                self.episode_stats['failures'] += 1
                self.stats['timeouts'] += 1
                low_done = True
                logger.warning(f"⏰ 低层任务超时或非法动作")

            # 低层自己的终止条件
            elif low_terminated or low_truncated:
                low_done = True
                logger.info(f"🛑 低层任务终止")

            # 显示进度
            if step_count % 20 == 0:
                logger.debug(f"  低层执行中... 步数: {step_count}, 累计奖励: {total_reward:.2f}")

        execution_time = time.time() - start_time

        # 检查是否达到最大步数
        if step_count >= self.max_low_steps and not low_done:
            logger.warning(f"⚠️ 低层执行达到最大步数限制 ({self.max_low_steps})")
            self.episode_stats['failures'] += 1
            self.stats['max_steps_exceeded'] += 1

        logger.info(f"⚙️ [低层执行] 完成，步数={step_count}, 总奖励={total_reward:.2f}, 用时={execution_time:.3f}s")

        return {
            'total_reward': total_reward,
            'steps': step_count,
            'execution_time': execution_time,
            'done': low_done,
            'info': last_info
        }

    def _check_need_high_decision(self):
        """
        检查当前是否需要高层决策

        需要高层决策的条件：
        1. 当前没有正在执行的低层任务
        2. 上一个低层任务已完成
        3. 当前阶段需要新的目标
        """
        # 获取当前环境状态
        current_phase = getattr(self.env, 'current_phase', None)
        current_target = getattr(self.env, 'current_target_node', None)
        current_deployment = getattr(self.env, 'current_deployment_target', None)

        # 如果没有当前阶段，需要高层决策
        if current_phase is None:
            return True

        # 检查VNF部署进度
        if hasattr(self.env, '_get_total_vnf_progress'):
            vnf_progress = self.env._get_total_vnf_progress()
            vnf_list = self.env.current_request.get('vnf', [])

            # VNF未完成且没有部署目标，需要高层决策
            if vnf_progress < len(vnf_list) and current_deployment is None:
                return True

            # VNF已完成且没有连接目标，需要高层决策
            elif vnf_progress >= len(vnf_list) and current_target is None:
                return True

        # 默认情况
        return False

    def _get_synchronized_high_state(self):
        """获取同步后的高层状态"""
        try:
            return self.env.get_high_level_state_graph()
        except Exception as e:
            logger.error(f"获取高层状态失败: {e}")
            # 返回一个默认状态
            return self._create_default_state()

    def _calculate_high_reward(self, low_total_reward, low_info, low_steps):
        """
        计算高层奖励

        策略：
        1. 基础奖励 = 低层累计奖励
        2. 根据任务完成情况给予额外奖励/惩罚
        3. 考虑效率因素（步数越少越好）
        """
        base_reward = low_total_reward

        # 任务成功奖励
        if low_info.get('deploy_success', False):
            base_reward += 30.0  # VNF部署成功额外奖励
        elif low_info.get('connection_success', False):
            base_reward += 50.0  # 目的地连接成功额外奖励

        # 任务失败惩罚
        if low_info.get('deploy_fail', False):
            base_reward -= 40.0  # VNF部署失败惩罚
        elif low_info.get('connection_fail', False):
            base_reward -= 50.0  # 目的地连接失败惩罚
        elif low_info.get('path_fail', False):
            base_reward -= 60.0  # 路径建立失败惩罚

        # 效率奖励：步数越少，奖励越高
        efficiency_factor = max(0, 1.0 - (low_steps / 50.0))  # 50步为基准
        base_reward += 10.0 * efficiency_factor

        return base_reward

    def _create_default_state(self):
        """创建默认状态（容错处理）"""
        import torch
        from torch_geometric.data import Data

        n = getattr(self.env, 'n', 14)

        return Data(
            x=torch.zeros((n, 13), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 2), dtype=torch.float32),
            global_attr=torch.zeros((1, 5), dtype=torch.float32)
        )

    def run_full_episode(self, training_mode=True):
        """
        运行完整的Episode

        Returns:
            dict: Episode结果统计
        """
        logger.info("=" * 60)
        logger.info("🎬 开始运行完整Episode")
        logger.info("=" * 60)

        # 重置环境
        initial_state = self.env.reset()
        self.reset()

        episode_done = False
        total_high_steps = 0
        total_low_steps = 0

        # Episode主循环
        while not episode_done and total_high_steps < self.max_high_steps:
            # 执行高层-低层循环
            result = self.run_high_low_cycle(training_mode)

            high_state, high_action, high_reward, next_high_state, high_done, info = result

            # 更新统计
            total_high_steps += 1
            total_low_steps += info.get('low_steps', 0)

            # 检查是否完成
            if high_done:
                episode_done = True
                self.episode_stats['success'] = True
                logger.info("🎉 Episode完成：所有任务成功完成")
                break

            # 检查是否达到最大步数
            if total_high_steps >= self.max_high_steps:
                logger.warning(f"⚠️ Episode达到最大高层步数限制 ({self.max_high_steps})")
                episode_done = True
                break

        # 最终统计
        self.episode_stats['high_steps'] = total_high_steps
        self.episode_stats['low_steps'] = total_low_steps

        # 打印总结
        self._print_episode_summary()

        return self.episode_stats.copy()

    def _print_episode_summary(self):
        """打印Episode总结"""
        stats = self.episode_stats

        logger.info("=" * 60)
        logger.info("📊 Episode总结报告")
        logger.info("=" * 60)
        logger.info(f"   总奖励: {stats['total_reward']:.2f}")
        logger.info(f"   高层决策步数: {stats['high_steps']}")
        logger.info(f"   低层执行步数: {stats['low_steps']}")
        logger.info(f"   VNF部署次数: {stats['vnf_deployments']}")
        logger.info(f"   目的地连接数: {stats['dest_connections']}")
        logger.info(f"   失败次数: {stats['failures']}")
        logger.info(f"   是否成功: {'✅' if stats['success'] else '❌'}")
        logger.info("=" * 60)

    def get_training_data(self):
        """
        获取训练数据

        Returns:
            dict: 包含高层和低层训练数据
        """
        return {
            'high_agent': {
                'states': self.history.get('high_states', []),
                'actions': self.history['high_actions'],
                'rewards': self.history['rewards'],
                'next_states': self.history.get('next_high_states', [])
            },
            'low_agent': {
                'states': self.history.get('low_states', []),
                'actions': self.history['low_actions'],
                'rewards': self.history.get('low_rewards', [])
            },
            'stats': dict(self.stats),
            'episode_stats': self.episode_stats
        }

    def update_agents(self, high_data=None, low_data=None):
        """
        更新Agent

        Args:
            high_data: 高层训练数据
            low_data: 低层训练数据
        """
        if high_data and hasattr(self.high_agent, 'update'):
            try:
                self.high_agent.update(high_data)
                logger.debug("高层Agent已更新")
            except Exception as e:
                logger.error(f"高层Agent更新失败: {e}")

        if low_data and hasattr(self.low_agent, 'update'):
            try:
                self.low_agent.update(low_data)
                logger.debug("低层Agent已更新")
            except Exception as e:
                logger.error(f"低层Agent更新失败: {e}")

    # ============================================================
    # 🔥 [新增] 核心执行方法
    # ============================================================

    def step(self):
        """
        🎯 执行一次完整的 HRL 决策循环

        核心逻辑:
        1. 检查是否需要高层决策（重新规划子目标）
        2. 如果需要，执行高层决策
        3. 执行低层动作
        4. 返回标准 Gym 接口

        Returns:
            (next_obs, reward, done, truncated, info)
        """
        # ========================================
        # Phase 1: 高层决策（条件触发）
        # ========================================
        if self.current_goal is None or self._should_replan():
            logger.info("🎯 [Coordinator] 触发高层决策")

            # 获取高层状态
            high_obs = self.env.get_high_level_state_graph()

            # 高层选择子目标
            high_action, _, high_info = self.high_agent.select_action(high_obs)

            logger.info(f"🎯 [高层] 选择目标: {high_action}")

            # 设置目标
            self._set_goal(high_action)

            # 执行高层 step（设置环境的目标状态）
            _, high_reward, done, truncated, info = self.env.step_high_level(high_action)

            # 如果高层决策后就结束了（例如所有目标已连接），直接返回
            if done:
                return self.env.get_state(), high_reward, done, truncated, info

        # ========================================
        # Phase 2: 低层执行（向目标移动/部署）
        # ========================================
        low_obs = self.env.get_state()
        low_mask = self.env.get_low_level_action_mask()

        # 低层选择动作
        _, low_action, low_info = self.low_agent.select_action(
            low_obs,
            action_mask=low_mask
        )

        logger.debug(f"🚶 [低层] 执行动作: {low_action}")

        # 执行低层动作
        next_obs, reward, done, truncated, info = self.env.step_low_level(low_action)

        # ========================================
        # Phase 3: 状态更新
        # ========================================
        # 如果子目标完成（truncated=True），清空目标以便下次重新规划
        if truncated:
            logger.info("✅ [Coordinator] 子目标完成，清空目标")
            self.current_goal = None

        # 存储最近的 transition（供 Trainer 使用）
        self.last_transition = (low_obs, low_action, reward, next_obs, done)

        return next_obs, reward, done, truncated, info

    def _should_replan(self):
        """
        判断是否需要重新规划高层目标

        触发条件:
        1. 步数达到阈值（防止卡死）
        2. 当前阶段改变（VNF部署 -> 目的地连接）
        """
        if not hasattr(self, '_low_step_count'):
            self._low_step_count = 0

        self._low_step_count += 1

        # 策略1: 每 10 步强制重新规划
        if self._low_step_count >= 10:
            logger.debug(f"⏰ [Coordinator] 步数达到 {self._low_step_count}，触发重新规划")
            return True

        # 策略2: 阶段改变时重新规划
        current_phase = getattr(self.env, 'current_phase', None)
        if hasattr(self, '_last_phase') and current_phase != self._last_phase:
            logger.debug(f"🔄 [Coordinator] 阶段改变 {self._last_phase} -> {current_phase}")
            self._last_phase = current_phase
            return True

        self._last_phase = current_phase

        return False

    def _set_goal(self, high_action):
        """设置当前高层目标"""
        self.current_goal = high_action
        self._low_step_count = 0
        logger.debug(f"📌 [Coordinator] 设置目标: {high_action}")