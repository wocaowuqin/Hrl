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

        self.last_vnf_progress = 0
        self.last_connected_count = 0

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

        修复：正确处理高层返回的 truncated 信号，确保高层-低层严格切换

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

            # 🔥 修复：正确读取 truncated 信号
            _, _, high_done, high_truncated, high_info = self.env.step_high_level(high_action)

            high_decision_time = time.time() - start_time

            # 记录高层决策
            self.stats['high_decisions'] += 1
            self.episode_stats['high_steps'] += 1

            logger.info(
                f"🔝 [高层决策] 步骤{self.stats['high_decisions']}，动作={high_action}，用时={high_decision_time:.3f}s，truncated={high_truncated}")

            # ========================================
            # 4. 检查高层返回状态
            # ========================================
            # 情况1：高层任务完成（所有目的地已连接）
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

            # 情况2：高层决策结束，进入低层执行阶段
            elif high_truncated:
                logger.info("🔄 高层决策结束，进入低层执行阶段")

                # ========================================
                # 5. 低层执行循环（同步关键）
                # ========================================
                low_execution_result = self._execute_low_level_loop(training_mode)

                # 更新统计
                self.episode_stats['low_steps'] += low_execution_result['steps']
                self.episode_stats['total_reward'] += low_execution_result['total_reward']

                # ========================================
                # 6. 获取同步后的下一个高层状态
                # ========================================
                # 🔥 关键：低层执行完后，状态已更新，现在获取同步后的高层状态
                next_high_state = self._get_synchronized_high_state()

                # ========================================
                # 7. 计算高层奖励
                # ========================================
                high_reward = self._calculate_high_reward(
                    low_execution_result['total_reward'],
                    low_execution_result['info'],
                    low_execution_result['steps']
                )

                self.last_high_reward = high_reward
                self.history['rewards'].append(high_reward)

                # ========================================
                # 8. 返回结果
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
                        'execution_time': high_decision_time + low_execution_result['execution_time'],
                        'high_truncated': high_truncated,
                        'phase': getattr(self.env, 'current_phase', 'unknown')
                    }
                )

            # 情况3：高层既没完成也没结束，继续等待（理论上不应该发生）
            else:
                logger.warning("⚠️ 高层决策既未完成也未结束，可能逻辑错误")
                return (
                    high_state,
                    high_action,
                    0.0,
                    high_state,
                    False,
                    {
                        **high_info,
                        'message': '高层决策状态异常',
                        'high_done': high_done,
                        'high_truncated': high_truncated
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
                {
                    'message': '等待低层执行完成',
                    'phase': getattr(self.env, 'current_phase', 'unknown')
                }
            )

    def _execute_low_level_step(self):
        """
        执行低层动作
        🔥 [修复] 移除 high_action 汇报，防止 Trainer 误判为高层死循环
        """
        low_obs = self.env.get_state()
        low_mask = self.env.get_low_level_action_mask()

        # 1. 选动作
        try:
            _, low_action, low_info = self.low_agent.select_action(
                low_obs,
                action_mask=low_mask
            )
        except Exception as e:
            logger.error(f"❌ [Coordinator] 低层选动作失败: {e}")
            return low_obs, -5.0, False, True, {'error': 'low_select_fail'}

        # 2. 执行动作
        try:
            next_obs, reward, done, truncated, info = self.env.step_low_level(low_action)
        except Exception as e:
            logger.error(f"❌ [Coordinator] 低层执行失败: {e}")
            return low_obs, -10.0, False, True, {'error': 'low_step_fail'}

        # 3. 状态更新
        if truncated:
            logger.info("✅ [Coordinator] 子目标达成，准备重规划")
            self.current_goal = None
            self.stats['subgoals_completed'] += 1

        self.last_transition = (low_obs, low_action, reward, next_obs, done)
        self.stats['total_low_actions'] += 1

        if not isinstance(info, dict): info = {}

        # 🔥🔥🔥 [关键修改] 不要在这里汇报 high_action！🔥🔥🔥
        # info['high_action'] = getattr(self, 'last_high_action', None)  <-- 删掉或注释掉这行
        # 让 Trainer 看到 None，它就知道 "哦，这一步不是高层决策"
        info['high_action'] = None

        return next_obs, reward, done, truncated, info

    def _enhance_low_state(self, low_state):
        """
        增强低层状态信息，帮助Agent理解何时应该执行部署/连接

        Args:
            low_state: 原始低层状态

        Returns:
            增强后的状态
        """
        # 获取当前环境信息
        current_phase = getattr(self.env, 'current_phase', None)
        current_target = getattr(self.env, 'current_target_node', None)
        current_node = getattr(self.env, 'current_node', None)

        # 创建增强信息
        enhanced_info = {
            'phase': current_phase,
            'target_node': current_target,
            'current_node': current_node,
            'at_target': current_node == current_target if current_node and current_target else False
        }

        # 根据环境类型返回增强状态
        if hasattr(low_state, 'enhanced_info'):
            low_state.enhanced_info = enhanced_info
        elif isinstance(low_state, dict):
            low_state['enhanced_info'] = enhanced_info
        elif hasattr(low_state, '__dict__'):
            low_state.enhanced_info = enhanced_info

        return low_state
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
    def step(self, force_goal=None):
        """
        🎯 [死循环终极修复版] 执行 HRL 决策步

        Args:
            force_goal: 如果非 None，表示刚从高层切换下来，强制执行低层逻辑，跳过检查。
        """

        # ============================================================
        # Phase 1: 高层决策 (High-Level Decision)
        # ============================================================
        # 只有在没有强制目标，且 (当前无目标 或 需要重规划) 时才进入
        if force_goal is None and (self.current_goal is None or self._should_replan()):
            logger.info("🎯 [Coordinator] 触发高层决策")
            self.low_step_count = 0
            # 🔥 检查placement状态
            if hasattr(self.env, 'current_tree') and 'placement' in self.env.current_tree:
                placement_count = len(self.env.current_tree['placement'])
                logger.warning(f"🔍 [调试] 当前placement数量: {placement_count}")
                if placement_count > 0:
                    logger.warning(f"   前3个placement: {list(self.env.current_tree['placement'].items())[:3]}")

            # 1. 准备状态
            high_obs = self.env.get_high_level_state_graph()
            high_mask = None
            unconnected_dests = []
            try:
                if hasattr(self.env, 'get_high_level_action_mask'):
                    high_mask = self.env.get_high_level_action_mask()
                if self.env.current_request:
                    dests = self.env.current_request.get('dest', [])
                    connected = self.env.current_tree.get('connected_dests', set())
                    unconnected_dests = [d for d in dests if d not in connected]
            except Exception:
                pass

            # 2. Agent 选择
            high_action, _, high_info = self.high_agent.select_action(
                high_obs,
                action_mask=high_mask,
                unconnected_dests=unconnected_dests
            )
            logger.warning(
                f"🔍 [调试] high_mask类型: {type(high_mask)}, 形状: {high_mask.shape if hasattr(high_mask, 'shape') else len(high_mask)}")
            logger.warning(
                f"🔍 [调试] high_mask[24] = {high_mask[24] if high_mask is not None and len(high_mask) > 24 else 'N/A'}")
            logger.warning(f"🔍 [调试] Agent选择: {high_action}")
            logger.warning(f"🔍 [调试] DC节点列表: {getattr(self.env, 'dc_nodes', 'N/A')}")

            # 🔥 强制验证并修正
            # 🔥 强制验证并修正
            if high_mask is not None and high_action < len(high_mask):
                # 🔥 关键：只在VNF部署阶段且没有subgoal时才验证mask
                phase = getattr(self.env, 'current_phase', 'unknown')
                real_target = high_info.get('subgoal')  # 优先用subgoal

                # 如果没有subgoal，才检查high_action
                if real_target is None:
                    if high_mask[high_action] == 0 or high_mask[high_action] == False:
                        logger.error(f"❌ Agent选择了被屏蔽的节点{high_action}！强制修正...")

                        # 找到所有可用节点
                        valid_nodes = np.where(high_mask > 0)[0] if hasattr(high_mask, '__len__') else []

                        if len(valid_nodes) > 0:
                            # 随机选一个合法节点
                            high_action = int(np.random.choice(valid_nodes))
                            logger.warning(f"✅ 修正为: 节点{high_action}")
                        else:
                            logger.error(f"❌ 没有可用节点！Mask全为0")
                else:
                    # 有subgoal，直接使用它，不验证high_action
                    logger.debug(f"✅ 使用info['subgoal']={real_target}，跳过high_action验证")
                if high_mask[high_action] == 0 or high_mask[high_action] == False:
                    logger.error(f"❌ Agent选择了被屏蔽的节点{high_action}！强制修正...")

                    # 找到所有可用节点
                    valid_nodes = np.where(high_mask > 0)[0] if hasattr(high_mask, '__len__') else []

                    if len(valid_nodes) > 0:
                        # 随机选一个合法节点
                        high_action = int(np.random.choice(valid_nodes))
                        logger.warning(f"✅ 修正为: 节点{high_action}")
                    else:
                        logger.error(f"❌ 没有可用节点！Mask全为0")
            # 🔥🔥🔥 关键修复：正确解析目标节点 🔥🔥🔥
            # 3. 目标解析（优先级：subgoal > unconnected_dests映射 > 直接使用）
            # 🔍 [调试] 在解析real_target_node之前
            logger.warning(f"🔍 [调试] high_info内容: {high_info}")
            logger.warning(f"🔍 [调试] high_info.get('subgoal'): {high_info.get('subgoal')}")
            logger.warning(f"🔍 [调试] unconnected_dests: {unconnected_dests}")
            logger.warning(f"🔍 [调试] current_phase: {getattr(self.env, 'current_phase', 'unknown')}")
            logger.warning(f"🔍 [调试] high_action: {high_action} (type: {type(high_action)})")
            real_target_node = high_info.get('subgoal')  # 优先从info获取

            if real_target_node is None:
                # 如果在目的地连接阶段，映射索引到目的地
                if unconnected_dests and 0 <= high_action < len(unconnected_dests):
                    real_target_node = unconnected_dests[high_action]
                    logger.info(f"🎯 [高层] 目的地连接阶段: 索引{high_action} -> 目的地{real_target_node}")
                else:
                    # VNF部署阶段，high_action就是节点ID
                    real_target_node = high_action
                    logger.info(f"🎯 [高层] VNF部署阶段: 选择节点{real_target_node}")

            logger.info(f"🎯 [高层] 最终目标: {real_target_node}")

            # 🔥 记录实际节点（不是索引）
            self.last_high_action = real_target_node

            # 🔥 设置目标（传入实际节点）
            self._set_goal(real_target_node)

            # 4. 执行高层 Step（传入实际节点）
            try:
                _, high_reward, high_done, high_truncated, step_info = self.env.step_high_level(real_target_node)

                # 同步 Phase 防止误判
                self._last_phase = getattr(self.env, 'current_phase', None)

                # 更新info
                if isinstance(step_info, dict):
                    step_info.update(high_info)
                    step_info['high_action'] = real_target_node  # 🔥 记录实际节点

                # 如果任务完成，直接返回
                if high_done:
                    logger.info("✅ [Coordinator] 高层任务全部完成")
                    self.current_goal = None
                    return self.env.get_state(), high_reward, high_done, False, step_info

                # 🔥 [关键修复] 高层设定完成，准备低层执行
                logger.info("↘️ [Coordinator] 高层设定完成，强制进入低层")
                self.current_goal = real_target_node
                self.stats['total_high_decisions'] += 1

                # 🔥🔥🔥 不要递归！直接继续到Phase 2 🔥🔥🔥
                # 让代码自然流到下面的低层执行部分

            except Exception as e:
                logger.error(f"❌ [Coordinator] 高层执行崩溃: {e}")
                import traceback
                traceback.print_exc()
                self.current_goal = None
                return self.env.get_state(), -10.0, False, True, {
                    'error': 'high_crash',
                    'high_action': real_target_node
                }

        # ============================================================
        # Phase 2: 低层执行 (Low-Level Execution)
        # ============================================================
        # 如果有force_goal，使用它；否则使用current_goal
        target_goal = force_goal if force_goal is not None else self.current_goal

        if target_goal is None:
            logger.warning("⚠️ [Coordinator] 没有目标，跳过低层执行")
            return self.env.get_state(), 0.0, False, False, {
                'error': 'no_goal',
                'high_action': getattr(self, 'last_high_action', None)
            }

        # 获取低层状态和mask
        low_obs = self.env.get_state()
        low_mask = self.env.get_low_level_action_mask()

        # 低层选择动作
        try:
            _, low_action, low_info = self.low_agent.select_action(
                low_obs,
                action_mask=low_mask
            )
            logger.debug(f"🚶 [低层] 执行动作: {low_action}, 目标: {target_goal}")

        except Exception as e:
            logger.error(f"❌ [Coordinator] 低层动作选择失败: {e}")
            return low_obs, -5.0, False, True, {
                'error': 'low_action_selection_failed',
                'high_action': getattr(self, 'last_high_action', None)
            }

        # 执行低层动作
        try:
            next_obs, reward, done, truncated, info = self.env.step_low_level(low_action)

            # 🔥 增加低层步数
            self.low_step_count += 1
            logger.debug(f"📊 [Coordinator] 低层步数: {self.low_step_count}/{self.max_low_steps}")

        except Exception as e:
            logger.error(f"❌ [Coordinator] 低层执行失败: {e}")
            import traceback
            traceback.print_exc()
            return low_obs, -10.0, False, True, {
                'error': 'low_level_execution_failed',
                'high_action': getattr(self, 'last_high_action', None)
            }

        # ========================================
        # Phase 3: 状态更新
        # ========================================
        # 如果子目标完成（truncated=True），清空目标
        if truncated:
            logger.info("✅ [Coordinator] 子目标完成，清空目标")
            self.current_goal = None
            self.stats['subgoals_completed'] += 1
            self.low_step_count = 0  # 🔥 重置低层步数

        # 存储最近的 transition
        self.last_transition = (low_obs, low_action, reward, next_obs, done)

        # 更新统计
        self.stats['total_low_actions'] += 1

        # 🔥 在 info 中加入 high_action
        if not isinstance(info, dict):
            info = {}
        info['high_action'] = getattr(self, 'last_high_action', None)

        return next_obs, reward, done, truncated, info


    def _should_replan(self):
        """
        🔧 判断是否需要重新规划

        重新规划的条件:
        1. 低层超时（执行太多步仍未完成）
        2. 当前子目标已完成（VNF已部署或目的地已连接）
        3. 发生错误
        """
        # 1. 低层超时
        if self.low_step_count >= self.max_low_steps:
            logger.warning(f"⚠️ [Coordinator] 低层超时({self.low_step_count}步)，需要重新规划")
            return True

        # 2. 检查子目标是否完成
        if self.current_goal is not None:
            # 检查VNF部署进度
            if hasattr(self.env, '_get_total_vnf_progress'):
                current_progress = self.env._get_total_vnf_progress()
                if current_progress > self.last_vnf_progress:
                    logger.info(f"✅ [Coordinator] VNF进度提升({self.last_vnf_progress}->{current_progress})，需要新目标")
                    self.last_vnf_progress = current_progress
                    return True

            # 检查目的地连接进度
            if hasattr(self.env, 'current_tree'):
                connected = len(self.env.current_tree.get('connected_dests', set()))
                if connected > self.last_connected_count:
                    logger.info(f"✅ [Coordinator] 目的地连接提升({self.last_connected_count}->{connected})，需要新目标")
                    self.last_connected_count = connected
                    return True

        # 3. 默认不重新规划（让低层继续执行当前目标）
        return False

    def _set_goal(self, high_action):
        """设置当前高层目标"""
        self.current_goal = high_action
        self._low_step_count = 0
        self.stats['total_high_decisions'] += 1
        logger.debug(f"📌 [Coordinator] 设置目标: {high_action}")