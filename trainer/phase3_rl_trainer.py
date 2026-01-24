
# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import random
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.visualizer import SFCVisualizer
logger = logging.getLogger(__name__)
class Phase3RLTrainer:
    """Phase 3: Goal-Conditioned RL Trainer with DAgger + Time Slot System"""
    def __init__(self, env, agent, output_dir, config, coordinator=None):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.coordinator = coordinator

        # 🔥 初始化可视化器
        self.visualizer = None
        if hasattr(env, 'topo'):
            try:
                self.visualizer = SFCVisualizer(env.topo, output_dir)
                logger.info("🎨 可视化器已就绪 (plots 将保存在 outputs/checkpoints/plots)")
            except Exception as e:
                logger.warning(f"⚠️ 可视化器初始化失败: {e}")

        phase3_cfg = config.get("phase3", {})
        self.max_episodes = phase3_cfg.get("episodes", 1000)
        self.save_freq = phase3_cfg.get("save_every", 100)

        # 🔧 新增：训练参数
        self.max_steps_per_episode = phase3_cfg.get("max_steps", 600)
        self.update_frequency = phase3_cfg.get("update_frequency", 4)  # 每4步更新一次
        self.warmup_steps = phase3_cfg.get("warmup_steps", 100)  # 预热步数

        # 1. Epsilon 配置
        epsilon_cfg = phase3_cfg.get("epsilon", {})
        self.epsilon_initial = epsilon_cfg.get("initial", 0.7)
        self.epsilon_final = epsilon_cfg.get("final", 0.01)
        self.epsilon_decay_steps = epsilon_cfg.get("decay_steps", 5000)

        # 2. DAgger 配置
        dagger_cfg = phase3_cfg.get("dagger", {})
        self.use_dagger = dagger_cfg.get("enabled", True)
        self.beta = dagger_cfg.get("initial_beta", 0.8)
        self.beta_final = dagger_cfg.get("final_beta", 0.05)
        self.beta_decay_steps = dagger_cfg.get("decay_steps", 10000)

        # 🔥 新增：时间槽系统配置
        timeslot_cfg = phase3_cfg.get("timeslot", {})
        self.use_timeslot = timeslot_cfg.get("enabled", True)
        self.log_timeslot_info = timeslot_cfg.get("log_timeslot_info", True)
        self.log_timeslot_jumps = timeslot_cfg.get("log_jumps", True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # 统计信息容器
        self.stats = {
            "rewards": [],
            "acceptance_rates": [],
            "blocking_rates": [],
            "resource_levels": [],
            "subgoal_completion_rate": [],
            "time_slots_covered": [],
            "decision_steps": [],
            "requests_per_episode": [],
            "losses": [],
            "high_losses": [],
            "low_losses": []

        }
        self.global_step = 0
        self.total_updates = 0  # 🔧 新增：总更新次数

        # 🔥 新增：时间槽相关统计
        self.timeslot_stats = {
            'total_time_slots': 0,
            'total_decision_steps': 0,
            'avg_steps_per_request': 0,
            'timeslot_jumps': []
        }
        # 🔥 添加诊断开关
        self.enable_diagnosis = config.get('enable_diagnosis', False)
        self.diagnosis_interval = config.get('diagnosis_interval', 10)
    def run(self):
        """🚀 Phase 3 训练主循环 - 完整版含诊断"""

        # 🔥🔥🔥 诊断代码块1：训练开始前诊断 🔥🔥🔥
        logger.info("\n" + "=" * 80)
        logger.info("🔍 训练前环境诊断")
        logger.info("=" * 80)

        logger.info(f"\n1️⃣ 资源管理器结构:")
        logger.info(f"   类型: {type(self.env.resource_mgr).__name__}")
        logger.info(f"   nodes类型: {type(self.env.resource_mgr.nodes)}")

        if isinstance(self.env.resource_mgr.nodes, dict):
            keys = list(self.env.resource_mgr.nodes.keys())[:5]
            logger.info(f"   nodes键: {keys}")

            if 'cpu' in self.env.resource_mgr.nodes:
                logger.info(f"   ⚠️ 结构: 列表字典 {{'cpu': [...], 'mem': [...]}}")
                cpu_list = self.env.resource_mgr.nodes.get('cpu', [])
                mem_list = self.env.resource_mgr.nodes.get('mem', [])
                logger.info(f"   CPU列表长度: {len(cpu_list)}")
                logger.info(f"   前3个节点CPU: {cpu_list[:3]}")
                logger.info(f"   前3个节点Mem: {mem_list[:3]}")
            elif 0 in self.env.resource_mgr.nodes:
                logger.info(f"   ✅ 结构: 节点字典 {{0: {{}}, 1: {{}}, ...}}")
                for i in range(min(3, len(self.env.resource_mgr.nodes))):
                    node = self.env.resource_mgr.nodes.get(i, {})
                    logger.info(f"   节点{i}: CPU={node.get('cpu', 'N/A')}, Mem={node.get('mem', 'N/A')}")

        logger.info(f"\n2️⃣ 环境重置测试:")
        self.env.reset()
        logger.info(f"   请求存在: {self.env.current_request is not None}")
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            logger.info(f"   VNF数量: {len(vnf_list)}")
            logger.info(f"   源节点: {self.env.current_request.get('source')}")
            logger.info(f"   目的节点: {self.env.current_request.get('dest', [])}")

        logger.info(f"\n3️⃣ 高层动作掩码:")
        mask = self.env.get_high_level_action_mask()
        available = np.where(mask)[0]
        logger.info(f"   可用动作数: {len(available)}")
        logger.info(f"   可用节点: {available[:10]}")

        logger.info(f"\n4️⃣ 节点详情 (前10个):")

        # 🔥 正确获取资源
        if isinstance(self.env.resource_mgr.nodes, dict) and 'cpu' in self.env.resource_mgr.nodes:
            cpu_list = self.env.resource_mgr.nodes.get('cpu', [])
            mem_list = self.env.resource_mgr.nodes.get('memory', [])

            for node in range(min(10, self.env.n)):
                is_valid = self.env._is_valid_node(node)
                is_dc = node in getattr(self.env, 'dc_nodes', [])

                cpu = cpu_list[node] if node < len(cpu_list) else 'N/A'
                mem = mem_list[node] if node < len(mem_list) else 'N/A'
                mask_val = mask[node]

                logger.info(f"   节点{node}: 有效={'✅' if is_valid else '❌'}, DC={'✅' if is_dc else '❌'}, "
                            f"CPU={cpu}, Mem={mem}, Mask={mask_val}")
        else:
            # 原来的逻辑
            for node in range(min(10, self.env.n)):
                is_valid = self.env._is_valid_node(node)
                is_dc = node in getattr(self.env, 'dc_nodes', [])
                node_info = self.env.resource_mgr.nodes.get(node, {})
                cpu = node_info.get('cpu', 'N/A')
                mem = node_info.get('mem', 'N/A')
                mask_val = mask[node]

                logger.info(f"   节点{node}: 有效={'✅' if is_valid else '❌'}, DC={'✅' if is_dc else '❌'}, "
                            f"CPU={cpu}, Mem={mem}, Mask={mask_val}")

        logger.info(f"\n5️⃣ 测试高层执行:")
        if len(available) > 0:
            test_action = available[0]
            logger.info(f"   测试节点: {test_action}")

            before_phase = getattr(self.env, 'current_phase', 'unknown')
            before_vnf = self.env._get_total_vnf_progress()

            _, reward, done, trunc, info = self.env.step_high_level(test_action)

            logger.info(f"   执行前: 阶段={before_phase}, VNF进度={before_vnf}")
            logger.info(f"   执行后: reward={reward}, done={done}, trunc={trunc}")
            logger.info(f"   Info: {info}")

            if 'error' in info:
                logger.error(f"\n   ⚠️⚠️⚠️ 检测到错误: {info['error']}")
                logger.error(f"   ⚠️⚠️⚠️ 这可能就是循环的原因!")

        logger.info("=" * 80 + "\n")
        # 🔥🔥🔥 诊断代码块1 结束 🔥🔥🔥

        # ====================================================================
        # 主训练循环
        # ====================================================================
        num_episodes = self.cfg.get('num_episodes', 1000)

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            # 🔥🔥🔥 诊断变量 🔥🔥🔥
            consecutive_same_high_action = 0
            last_high_action = None
            low_timeout_count = 0
            high_error_count = 0
            # 🔥🔥🔥

            while not done:
                # ============================================================
                # 如果有 Coordinator，使用 Coordinator 执行
                # ============================================================
                # ============================================================
                # 如果有 Coordinator，使用 Coordinator 执行
                # ============================================================
                if self.coordinator:
                    result = self.coordinator.step()

                    # 🔥 修复：处理 tuple 返回值
                    if isinstance(result, tuple):
                        # Coordinator.step() 返回 (state, reward, done, truncated, info)
                        state, reward, done, truncated, info = result

                        # 从 info 中提取信息
                        high_action = info.get('high_action') if isinstance(info, dict) else None

                        # 🔥🔥🔥 诊断代码块2：Coordinator结果诊断 🔥🔥🔥
                        if high_action == last_high_action:
                            consecutive_same_high_action += 1
                            if consecutive_same_high_action >= 3:
                                logger.warning(f"\n⚠️ [Episode {episode}, Step {step_count}] 循环警告!")
                                logger.warning(f"   连续{consecutive_same_high_action}次选择节点{high_action}")
                                logger.warning(f"   当前位置: 节点{self.env.current_node_location}")
                                logger.warning(f"   当前阶段: {getattr(self.env, 'current_phase', 'unknown')}")
                                vnf_list = self.env.current_request.get('vnf', []) if self.env.current_request else []
                                logger.warning(f"   VNF进度: {self.env._get_total_vnf_progress()}/{len(vnf_list)}")
                                logger.warning(f"   目标节点: {getattr(self.env, 'current_deployment_target', 'N/A')}")
                                # 检查节点资源（修复版）
                                if high_action is not None:  # 🔥 加上这个检查
                                    if isinstance(self.env.resource_mgr.nodes,
                                                  dict) and 'cpu' in self.env.resource_mgr.nodes:
                                        cpu_list = self.env.resource_mgr.nodes.get('cpu', [])
                                        mem_list = self.env.resource_mgr.nodes.get('memory', [])
                                        cpu = cpu_list[high_action] if high_action < len(cpu_list) else 'N/A'
                                        mem = mem_list[high_action] if high_action < len(mem_list) else 'N/A'
                                    else:
                                        node_info = self.env.resource_mgr.nodes.get(high_action, {})
                                        cpu = node_info.get('cpu', 'N/A')
                                        mem = node_info.get('mem', 'N/A')

                                    logger.warning(f"   节点{high_action}资源: CPU={cpu}, Mem={mem}")
                                    logger.warning(
                                        f"   节点{high_action}是DC节点: {high_action in getattr(self.env, 'dc_nodes', [])}")
                                else:
                                    logger.warning(f"   ⚠️ high_action 是 None! info内容: {info}")
                                # 检查节点资源（修复版）
                                if isinstance(self.env.resource_mgr.nodes,
                                              dict) and 'cpu' in self.env.resource_mgr.nodes:
                                    cpu_list = self.env.resource_mgr.nodes.get('cpu', [])
                                    mem_list = self.env.resource_mgr.nodes.get('memory', [])
                                    cpu = cpu_list[high_action] if high_action < len(cpu_list) else 'N/A'
                                    mem = mem_list[high_action] if high_action < len(mem_list) else 'N/A'
                                else:
                                    node_info = self.env.resource_mgr.nodes.get(high_action, {})
                                    cpu = node_info.get('cpu', 'N/A')
                                    mem = node_info.get('mem', 'N/A')

                                logger.warning(f"   节点{high_action}资源: CPU={cpu}, Mem={mem}")
                                logger.warning(
                                    f"   节点{high_action}是DC节点: {high_action in getattr(self.env, 'dc_nodes', [])}")

                                # 强制中断
                                if consecutive_same_high_action >= 5:
                                    logger.error(f"   ❌ 连续{consecutive_same_high_action}次，强制终止episode")
                                    break
                        else:
                            consecutive_same_high_action = 0

                        last_high_action = high_action

                        # 检测错误
                        if isinstance(info, dict) and 'error' in info:
                            high_error_count += 1
                            logger.error(f"\n⚠️ [Episode {episode}, Step {step_count}] 高层错误!")
                            logger.error(f"   错误信息: {info['error']}")
                            logger.error(f"   累计错误次数: {high_error_count}")

                        # 检测低层超时
                        if isinstance(info, dict) and info.get('low_timeout'):
                            low_timeout_count += 1
                            logger.warning(f"\n⚠️ [Episode {episode}, Step {step_count}] 低层超时!")
                            logger.warning(f"   当前位置: 节点{self.env.current_node_location}")
                            target = getattr(self.env, 'current_deployment_target',
                                             getattr(self.env, 'current_target_node', 'N/A'))
                            logger.warning(f"   目标节点: {target}")
                            logger.warning(f"   累计超时次数: {low_timeout_count}")

                        # 🔥🔥🔥 诊断代码块2 结束 🔥🔥🔥

                        episode_reward += reward
                        # done 已经从 tuple 中提取

                    else:
                        # 兼容字典格式（如果 Coordinator 返回字典）
                        high_action = result.get('high_action')

                        # 循环检测（字典格式）
                        if high_action == last_high_action:
                            consecutive_same_high_action += 1
                            if consecutive_same_high_action >= 3:
                                logger.warning(f"\n⚠️ [Episode {episode}, Step {step_count}] 循环警告!")
                                logger.warning(f"   连续{consecutive_same_high_action}次选择节点{high_action}")

                                if consecutive_same_high_action >= 5:
                                    logger.error(f"   ❌ 强制终止")
                                    break
                        else:
                            consecutive_same_high_action = 0

                        last_high_action = high_action

                        # 检测错误
                        if 'error' in result:
                            high_error_count += 1
                            logger.error(f"\n⚠️ 高层错误: {result['error']}")

                        # 检测低层超时
                        if result.get('low_timeout'):
                            low_timeout_count += 1
                            logger.warning(f"\n⚠️ 低层超时! 累计{low_timeout_count}次")

                        episode_reward += result.get('reward', 0)
                        done = result.get('done', False)

                # ============================================================
                # 如果没有 Coordinator，手动执行高层+低层
                # ============================================================
                else:
                    # 高层决策
                    high_state = self.env.get_high_level_state_graph()
                    high_mask = self.env.get_high_level_action_mask()

                    # Agent 选择动作
                    with torch.no_grad():
                        high_action = self.agent.select_action(high_state, high_mask, explore=True)

                    # 🔥🔥🔥 诊断代码块3：手动模式循环检测 🔥🔥🔥
                    if high_action == last_high_action:
                        consecutive_same_high_action += 1
                        if consecutive_same_high_action >= 3:
                            logger.warning(f"\n⚠️ [Episode {episode}, Step {step_count}] 循环警告!")
                            logger.warning(f"   连续{consecutive_same_high_action}次选择节点{high_action}")

                            if consecutive_same_high_action >= 5:
                                logger.error(f"   ❌ 强制终止")
                                break
                    else:
                        consecutive_same_high_action = 0

                    last_high_action = high_action
                    # 🔥🔥🔥

                    # 执行高层动作
                    _, high_reward, high_done, high_trunc, high_info = self.env.step_high_level(high_action)

                    # 🔥 错误检测
                    if 'error' in high_info:
                        high_error_count += 1
                        logger.error(f"\n⚠️ 高层错误: {high_info['error']}")

                    episode_reward += high_reward

                    # 如果没结束，执行低层
                    if not high_done and not high_trunc:
                        low_done = False
                        low_step = 0
                        max_low_steps = 50

                        while not low_done and low_step < max_low_steps:
                            low_state = self.env.get_state()
                            low_mask = self.env.get_low_level_action_mask()

                            with torch.no_grad():
                                low_action = self.agent.select_action(low_state, low_mask, explore=True)

                            _, low_reward, low_done, low_trunc, low_info = self.env.step_low_level(low_action)

                            episode_reward += low_reward
                            low_step += 1

                            # 🔥 超时检测
                            if low_info.get('timeout'):
                                low_timeout_count += 1
                                logger.warning(f"\n⚠️ 低层超时! 累计{low_timeout_count}次")
                                break

                            if low_trunc:
                                break

                        if low_step >= max_low_steps:
                            logger.warning(f"⚠️ 低层达到最大步数{max_low_steps}")

                    done = high_done

                # ============================================================
                # 通用：步数保护
                # ============================================================
                step_count += 1

                # 🔥🔥🔥 诊断代码块4：步数保护 🔥🔥🔥
                if step_count > 200:
                    logger.error(f"\n❌ [Episode {episode}] 步数超限({step_count})，强制终止")
                    logger.error(f"   当前阶段: {getattr(self.env, 'current_phase', 'unknown')}")
                    logger.error(f"   VNF进度: {self.env._get_total_vnf_progress()}")
                    logger.error(f"   低层超时次数: {low_timeout_count}")
                    logger.error(f"   高层错误次数: {high_error_count}")
                    break
                # 🔥🔥🔥

            # ====================================================================
            # Episode 结束统计
            # ====================================================================
            logger.info(f"\nEpisode {episode}: Reward={episode_reward:.2f}, Steps={step_count}, "
                        f"低层超时={low_timeout_count}次, 高层错误={high_error_count}次")

            # 🔥 每10个episode详细输出
            if episode % 10 == 0:
                if self.env.current_request:
                    vnf_list = self.env.current_request.get('vnf', [])
                    logger.info(f"   VNF进度: {self.env._get_total_vnf_progress()}/{len(vnf_list)}")
                    connected = len(self.env.current_tree.get('connected_dests', set()))
                    dests = len(self.env.current_request.get('dest', []))
                    logger.info(f"   已连接目的地: {connected}/{dests}")

            # ====================================================================
            # 定期保存模型 (每100个episode)
            # ====================================================================
            if episode > 0 and episode % 100 == 0:
                save_path = os.path.join(self.output_dir, f"checkpoint_ep{episode}.pth")
                try:
                    torch.save({
                        'episode': episode,
                        'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
                        'config': self.cfg
                    }, save_path)
                    logger.info(f"💾 保存检查点: {save_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 保存失败: {e}")

        # ====================================================================
        # 训练结束，保存最终模型
        # ====================================================================
        final_path = os.path.join(self.output_dir, "phase3_final.pth")
        try:
            torch.save({
                'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
                'config': self.config
            }, final_path)
            logger.info(f"✅ 训练完成，最终模型: {final_path}")
        except Exception as e:
            logger.error(f"❌ 最终模型保存失败: {e}")
    def _run_episode(self, episode_idx: int):
        """
        🔥 [V32.0 HRL Coordinator 集成版]

        运行一个episode（集成 Coordinator + 黑名单 + DAgger + 时间槽系统 + Loss监控）

        核心逻辑:
        1. 优先使用 HRL Coordinator（如果可用）
        2. Coordinator 自动管理高低层交互
        3. 回退到直接调用 env.step（兼容模式）
        """
        import numpy as np
        import random

        # ========================================
        # 初始化
        # ========================================
        # 🔧 预热检查
        if self.agent.steps_done < self.warmup_steps:
            logger.debug(f"🔥 预热阶段: {self.agent.steps_done}/{self.warmup_steps}")

        max_steps = self.max_steps_per_episode

        # ✅ 重置环境
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, reset_info = reset_result
        else:
            state = reset_result
            reset_info = {}

        # 🔥 获取时间槽信息
        initial_time_slot = reset_info.get('time_slot', 0)
        current_time_slot = initial_time_slot
        request_id = reset_info.get('request_id')
        last_time_slot = current_time_slot

        # 获取初始 mask 和 info
        action_mask = reset_info.get('action_mask')
        blacklist_info = reset_info.get('blacklist_info', {})
        unconnected_dests = self._get_current_destinations()

        # Episode 状态
        done = False
        steps = 0
        decision_steps = 0
        episode_reward = 0

        # 🔥 Loss 统计容器
        episode_losses = []
        episode_high_losses = []
        episode_low_losses = []

        # DAgger 统计
        expert_steps = 0
        masked_expert_steps = 0

        # 经验存储统计
        stored_high_transitions = 0
        stored_low_transitions = 0

        # 初始化 step_info
        step_info = {'success': False, 'request_completed': False}

        # ========================================
        # 🔥 检测是否使用 Coordinator
        # ========================================
        use_coordinator = (self.coordinator is not None)

        if use_coordinator:
            logger.debug(f"✅ Episode {episode_idx}: 使用 HRL Coordinator 模式")
        else:
            logger.debug(f"⚠️ Episode {episode_idx}: 使用回退模式（直接调用 env.step）")

        # ========================================
        # 主循环
        # ========================================
        while not done and steps < max_steps:

            # ============================================================
            # 🔥🔥🔥 方案 A: 使用 HRL Coordinator
            # ============================================================
            if use_coordinator:
                try:
                    # Coordinator 自动管理高低层交互
                    next_state, reward, done, truncated, step_info = self.coordinator.step()

                    # 从 Coordinator 获取执行的动作信息
                    if hasattr(self.coordinator, 'last_transition'):
                        transition = self.coordinator.last_transition
                        if transition and len(transition) == 5:
                            trans_state, low_action, trans_reward, trans_next_state, trans_done = transition

                            # 存储低层经验
                            self.agent.store_transition_low(
                                trans_state, low_action, trans_reward, trans_next_state, trans_done
                            )
                            stored_low_transitions += 1

                    # 如果 Coordinator 触发了高层决策，可能需要单独存储
                    if hasattr(self.coordinator, 'last_high_action'):
                        high_action = self.coordinator.last_high_action
                        if high_action is not None and unconnected_dests:
                            goal = unconnected_dests[high_action] if high_action < len(unconnected_dests) else -1
                            if goal != -1:
                                self.agent.store_transition_high(
                                    state, goal, reward, next_state, done or truncated
                                )
                                stored_high_transitions += 1

                    # 更新状态
                    state = next_state
                    episode_reward += reward
                    steps += 1

                    # 更新时间槽信息
                    new_time_slot = step_info.get('time_slot', current_time_slot)
                    new_decision_steps = step_info.get('decision_steps', decision_steps)

                    if self.use_timeslot and new_time_slot != last_time_slot:
                        if self.log_timeslot_jumps:
                            logger.debug(f"⏰ [Ep {episode_idx}] Time Slot: {last_time_slot} → {new_time_slot}")
                        self.timeslot_stats['timeslot_jumps'].append((last_time_slot, new_time_slot))
                        last_time_slot = new_time_slot

                    current_time_slot = new_time_slot
                    decision_steps = new_decision_steps

                    # 更新目标信息
                    unconnected_dests = self._get_current_destinations()

                except Exception as e:
                    logger.error(f"❌ Coordinator.step 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 发生错误时终止 episode
                    break

            # ============================================================
            # 🔥🔥🔥 方案 B: 回退模式（无 Coordinator）
            # ============================================================
            else:
                # ----------------------------------------
                # 1. 提取 Action Mask
                # ----------------------------------------
                action_mask = None

                # 方式1: 从PyG Data对象中提取
                if hasattr(state, 'action_mask'):
                    action_mask = state.action_mask
                    if hasattr(action_mask, 'cpu'):
                        action_mask = action_mask.cpu().numpy()
                    if action_mask.ndim > 1:
                        action_mask = action_mask.squeeze()

                # 方式2: 从step_info中提取
                elif 'action_mask' in step_info:
                    action_mask = step_info['action_mask']

                # 方式3: 直接调用环境方法
                if action_mask is None and hasattr(self.env, 'get_low_level_action_mask'):
                    action_mask = self.env.get_low_level_action_mask()

                # 🔥 确保mask是numpy数组
                if action_mask is not None:
                    if hasattr(action_mask, 'numpy'):
                        action_mask = action_mask.numpy()
                    if isinstance(action_mask, list):
                        action_mask = np.array(action_mask)

                # ----------------------------------------
                # 2. DAgger 逻辑
                # ----------------------------------------
                beta = self.beta
                use_dagger = self.use_dagger
                use_expert = False
                expert_action = None

                if use_dagger and random.random() < beta:
                    expert_suggestion = self._get_expert_action(state)
                    if action_mask is None:
                        use_expert = True
                        expert_action = expert_suggestion
                    else:
                        valid_actions = np.where(action_mask > 0)[0]
                        if expert_suggestion in valid_actions:
                            use_expert = True
                            expert_action = expert_suggestion
                            expert_steps += 1
                        else:
                            masked_expert_steps += 1

                # ----------------------------------------
                # 3. Agent 选择动作
                # ----------------------------------------
                high_action, low_action, action_info = self.agent.select_action(
                    state=state,
                    unconnected_dests=unconnected_dests,
                    action_mask=action_mask,
                    use_expert=use_expert,
                    expert_action=expert_action,
                    blacklist_info=blacklist_info
                )

                # 🛡️ 防御：如果 Agent 返回 -1 (无效)，终止 episode
                if low_action == -1:
                    logger.warning(f"⚠️ Agent returned -1 (No Valid Actions). Terminating Episode {episode_idx}.")
                    return episode_reward, {
                        'success': False,
                        'blocking_rate': 1.0,
                        'message': 'no_valid_actions',
                        'time_slot': current_time_slot,
                        'decision_steps': decision_steps,
                        'time_slots_covered': current_time_slot - initial_time_slot,
                        'avg_loss': 0.0,
                        'avg_high_loss': 0.0,
                        'avg_low_loss': 0.0
                    }

                # ----------------------------------------
                # 4. 执行动作
                # ----------------------------------------
                step_result = self.env.step(low_action)

                # 解包结果
                if len(step_result) == 5:
                    next_state, reward, done, truncated, step_info = step_result
                else:
                    next_state, reward, done, step_info = step_result
                    truncated = False

                # ----------------------------------------
                # 5. 🔥 检测 need_high_level 信号
                # ----------------------------------------
                if truncated and step_info.get('need_high_level', False):
                    error_type = step_info.get('error', 'unknown')
                    logger.info(f"⚠️ [Episode {episode_idx}] 低层检测到问题: {error_type}")
                    logger.info(f"   → 返回高层重新决策（不终止episode）")

                    # 记录奖励
                    episode_reward += reward

                    # 重置agent分支状态（强制触发高层决策）
                    if hasattr(self.agent, 'current_branch_id'):
                        self.agent.current_branch_id = None
                    if hasattr(self.agent, 'subgoal_steps'):
                        self.agent.subgoal_steps = 999
                    if hasattr(self.agent, 'current_subgoal'):
                        self.agent.current_subgoal = None

                    # 存储经验（失败的尝试也要学习）
                    if action_info.get('high_level_decision', False):
                        goal = unconnected_dests[high_action] if unconnected_dests and high_action < len(
                            unconnected_dests) else -1
                        if goal != -1:
                            self.agent.store_transition_high(state, goal, reward, next_state, False)
                            stored_high_transitions += 1

                    self.agent.store_transition_low(state, low_action, reward, next_state, False)
                    stored_low_transitions += 1

                    # 更新状态
                    state = next_state
                    unconnected_dests = self._get_current_destinations()
                    steps += 1

                    # 继续循环（不终止episode）
                    continue

                # ----------------------------------------
                # 6. 更新时间槽信息
                # ----------------------------------------
                new_time_slot = step_info.get('time_slot', current_time_slot)
                new_decision_steps = step_info.get('decision_steps', decision_steps)

                if self.use_timeslot and new_time_slot != last_time_slot:
                    if self.log_timeslot_jumps:
                        logger.debug(f"⏰ [Ep {episode_idx}] Time Slot: {last_time_slot} → {new_time_slot}")
                    self.timeslot_stats['timeslot_jumps'].append((last_time_slot, new_time_slot))
                    last_time_slot = new_time_slot

                current_time_slot = new_time_slot
                decision_steps = new_decision_steps

                # ----------------------------------------
                # 7. 记录失败原因（黑名单学习）
                # ----------------------------------------
                if not step_info.get('success', True):
                    reason = step_info.get('message', 'unknown')
                    if "资源不足" in reason or "访问超限" in reason:
                        self.agent.record_failure(low_action, reason)

                # ----------------------------------------
                # 8. 存储经验
                # ----------------------------------------
                # High-Level Buffer
                if action_info.get('high_level_decision', False):
                    goal = unconnected_dests[high_action] if unconnected_dests and high_action < len(
                        unconnected_dests) else -1
                    if goal != -1:
                        self.agent.store_transition_high(state, goal, reward, next_state, done or truncated)
                        stored_high_transitions += 1

                # Low-Level Buffer
                self.agent.store_transition_low(state, low_action, reward, next_state, done or truncated)
                stored_low_transitions += 1

                # ----------------------------------------
                # 9. 更新状态
                # ----------------------------------------
                state = next_state
                action_mask = step_info.get('action_mask')
                blacklist_info = step_info.get('blacklist_info', {})
                unconnected_dests = self._get_current_destinations()
                episode_reward += reward
                steps += 1

                if truncated:
                    done = True

            # ============================================================
            # 🔥 定期更新网络（适用于两种模式）
            # ============================================================
            if steps % self.update_frequency == 0:
                # 确保经验缓冲区有足够数据
                has_enough_low_exp = len(self.agent.low_memory) >= self.agent.batch_size

                if has_enough_low_exp:
                    # 调用更新并获取详细的损失信息
                    loss_dict = self.agent.update_policies()

                    if loss_dict:
                        # 记录各种损失
                        high_loss = loss_dict.get('high_loss', 0.0)
                        low_loss = loss_dict.get('low_loss', 0.0)
                        total_loss = loss_dict.get('total_loss', 0.0)

                        # 只记录非零的损失
                        if high_loss > 0:
                            episode_high_losses.append(high_loss)
                        if low_loss > 0:
                            episode_low_losses.append(low_loss)
                        if total_loss > 0:
                            episode_losses.append(total_loss)

                        self.total_updates += 1

                        # 定期打印更新信息
                        if self.total_updates % 100 == 0:
                            logger.debug(
                                f"🔄 Update #{self.total_updates}: HighLoss={high_loss:.6f}, LowLoss={low_loss:.6f}")

                # 如果经验不足，打印警告
                elif self.total_updates < 10 and steps > 50:
                    logger.debug(f"⚠️ 经验不足: High={len(self.agent.high_memory)}, Low={len(self.agent.low_memory)}")

        # ========================================
        # Episode 结束处理
        # ========================================
        # 判断成功与否
        is_success = step_info.get('request_success', None)
        if is_success is None:
            is_success = step_info.get('request_completed', False) or step_info.get('success', False)

        # 检查环境是否已归档
        env_already_archived = False
        if hasattr(self.env, 'current_request'):
            env_already_archived = (self.env.current_request is None)

        # 如果环境未归档，执行归档
        if not env_already_archived:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                req_id = self.env.current_request.get('id', '?')
                if not is_success:
                    logger.info(f"🔄 [Episode清理] 请求 {req_id} 失败，执行回滚...")
                    self.env._archive_request(success=False)
                else:
                    logger.info(f"✅ [Episode清理] 请求 {req_id} 成功，归档资源...")
                    self.env._archive_request(success=True)

                # 清理环境状态
                self.env.current_request = None
                self.env.current_branch_id = None
                self.env.current_tree = {}
                self.env.nodes_on_tree = set()
                self.env.branch_states = {}
                if hasattr(self.env, 'curr_ep_node_allocs'):
                    self.env.curr_ep_node_allocs = []
                if hasattr(self.env, 'curr_ep_link_allocs'):
                    self.env.curr_ep_link_allocs = []
        else:
            logger.info(f"ℹ️ [Episode清理] 环境已归档，跳过Trainer归档")

        # ========================================
        # 构建 Episode Info
        # ========================================
        # 计算平均 Loss
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_high_loss = np.mean(episode_high_losses) if episode_high_losses else 0.0
        avg_low_loss = np.mean(episode_low_losses) if episode_low_losses else 0.0

        episode_info = {
            'steps': steps,
            'success': is_success,
            'blocking_rate': 0.0 if is_success else 1.0,
            'expert_usage': expert_steps / steps if steps > 0 else 0,
            'masked_expert': masked_expert_steps,
            'stored_high': stored_high_transitions,
            'stored_low': stored_low_transitions,
            'avg_loss': avg_loss,
            'avg_high_loss': avg_high_loss,
            'avg_low_loss': avg_low_loss,

            # 时间槽信息
            'current_time_slot': current_time_slot,
            'initial_time_slot': initial_time_slot,
            'time_slots_covered': current_time_slot - initial_time_slot,
            'decision_steps': decision_steps,
            'request_id': request_id,
            'requests_processed': 1,

            # 🔥 新增：标记使用的模式
            'used_coordinator': use_coordinator
        }

        # 更新时间槽统计
        if self.use_timeslot:
            self.timeslot_stats['total_time_slots'] += (current_time_slot - initial_time_slot)
            self.timeslot_stats['total_decision_steps'] += decision_steps

        # ========================================
        # 打印日志
        # ========================================
        status_icon = "✅" if is_success else "❌"
        mode_icon = "🤖" if use_coordinator else "🔧"

        if is_success or episode_idx % 10 == 0:
            logger.info(
                f"{mode_icon} Ep {episode_idx} | {status_icon} | "
                f"Rw: {episode_reward:.1f} | "
                f"Steps: {steps} | "
                f"HiLoss: {avg_high_loss:.4f} | "
                f"LoLoss: {avg_low_loss:.4f} | "
                f"TS: {current_time_slot} | "
                f"DS: {decision_steps}"
            )

            # 调试：打印经验存储情况
            if stored_low_transitions == 0:
                logger.warning(f"⚠️ Episode {episode_idx}: 没有存储任何Low-Level经验!")

        return episode_reward, episode_info
    def _get_current_destinations(self):
        """获取当前未连接的目的地列表"""
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return []
        all_dests = self.env.current_request.get('dest', [])
        connected = self.env.current_tree.get('connected_dests', set())
        return [d for d in all_dests if d not in connected]
    def _get_expert_action(self, state):
        """获取专家动作"""
        if not hasattr(self, 'agent') or not hasattr(self.agent, 'expert'):
            # 如果没有 Expert Wrapper，尝试用环境里的
            if hasattr(self.env, 'expert') and self.env.expert:
                # 这里需要 expert 逻辑，暂时随机兜底
                pass
        return random.randint(0, getattr(self.env, 'n', 28) - 1)

