# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 RL Trainer - Goal-Conditioned HRL + DAgger + 🔥 时间槽系统
===============================================================================
修复内容：
1. ✅ 统计逻辑：改为"全局累计平均"，修复 Acc=1% 的显示问题。
2. 🛡️ 崩溃保护：捕获 Agent 内部错误，防止训练中断。
3. 📊 进度条：显示真实累计 Acc (接纳率) 和 Blk (阻塞率)。
4. 🔥 时间槽系统：支持离散时间模拟、批量请求处理、资源自动释放
===============================================================================
"""

import logging
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

    def __init__(self, env, agent, output_dir, config):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

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

        # 1. Epsilon 配置
        epsilon_cfg = phase3_cfg.get("epsilon", {})
        self.epsilon_initial = epsilon_cfg.get("initial", 0.5)
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
            # 🔥 新增：时间槽统计
            "time_slots_covered": [],
            "decision_steps": [],
            "requests_per_episode": []
        }
        self.global_step = 0

        # 🔥 新增：时间槽相关统计
        self.timeslot_stats = {
            'total_time_slots': 0,
            'total_decision_steps': 0,
            'avg_steps_per_request': 0,
            'timeslot_jumps': []
        }

    def _get_network_resource_level(self):
        """
        🔥 [V10.17 修复版] 动态获取真实容量，不再写死 100.0
        """
        try:
            rm = self.env.resource_mgr
            # 获取 DC 节点列表
            dc_nodes = getattr(self.env, 'dc_nodes', [])

            if not dc_nodes:
                return 0.0

            total_dc_cpu = 0.0
            total_dc_cap = 0.0

            # 1. 尝试获取总容量基准 (优先用 ResourceManager 里的 C_cap)
            # 这是一个保险逻辑：看看 rm.C_cap 是数组还是数字
            c_cap_ref = getattr(rm, 'C_cap', 100.0)

            # 遍历所有 DC 节点
            for node in dc_nodes:
                # --- 获取当前剩余量 (分子) ---
                current_cpu = 0.0
                if isinstance(rm.nodes, dict) and 'cpu' in rm.nodes:
                    if node < len(rm.nodes['cpu']):
                        current_cpu = rm.nodes['cpu'][node]
                elif isinstance(rm.nodes, list):
                    if node < len(rm.nodes):
                        current_cpu = rm.nodes[node].get('cpu', 0)

                # --- 获取该节点总容量 (分母) ---
                # 🔥🔥🔥 之前这里写死成了 total_dc_cap += 100.0，这就是 150% 的罪魁祸首！
                node_cap = 100.0  # 默认兜底

                if hasattr(c_cap_ref, '__getitem__'):  # 如果 C_cap 是数组 [30, 55, 40...]
                    if node < len(c_cap_ref):
                        node_cap = float(c_cap_ref[node])
                elif isinstance(c_cap_ref, (int, float)):  # 如果 C_cap 是标量 100.0
                    node_cap = float(c_cap_ref)

                # 累加
                total_dc_cpu += current_cpu
                total_dc_cap += node_cap

            # 防止除以零
            if total_dc_cap <= 0: return 0.0

            # 计算百分比
            dc_res_pct = (total_dc_cpu / total_dc_cap) * 100.0

            # 再次保险：如果算出来大于 100，强行修正 (说明 C_cap 没取对)
            if dc_res_pct > 100.0:
                # print(f"⚠️ 资源显示异常: {dc_res_pct:.1f}% (分子{total_dc_cpu}/分母{total_dc_cap})")
                return 100.0

            return dc_res_pct

        except Exception as e:
            # print(f"资源监控出错: {e}")
            return 0.0

    def load_timeslot_data(self):
        """
        🔥 新增：加载时间槽数据
        """
        if not self.use_timeslot:
            logger.info("⚠️ 时间槽系统未启用，跳过数据加载")
            return False

        try:
            # 获取数据路径
            path_cfg = self.cfg.get('path', {})
            input_dir = Path(path_cfg.get('input_dir', 'data/input_dir'))

            # 文件名
            requests_file = input_dir / path_cfg.get('requests_file', 'phase3_requests.pkl')
            requests_by_slot_file = input_dir / path_cfg.get('requests_by_slot_file', 'phase3_requests_by_slot.pkl')

            logger.info(f"\n{'=' * 60}")
            logger.info(f"🔥 加载时间槽数据")
            logger.info(f"{'=' * 60}")
            logger.info(f"请求文件: {requests_file}")
            logger.info(f"时间槽文件: {requests_by_slot_file}")

            # 加载数据
            with open(requests_file, 'rb') as f:
                requests = pickle.load(f)

            with open(requests_by_slot_file, 'rb') as f:
                requests_by_slot = pickle.load(f)

            # 加载到环境
            if hasattr(self.env, 'load_requests'):
                self.env.load_requests(requests, requests_by_slot)
                logger.info(f"✅ 时间槽数据加载成功")
                logger.info(f"   总请求数: {len(requests)}")
                logger.info(f"   时间槽数: {len(requests_by_slot)}")
                logger.info(f"{'=' * 60}\n")
                return True
            else:
                logger.warning("⚠️ 环境不支持 load_requests() 方法")
                return False

        except FileNotFoundError as e:
            logger.error(f"❌ 时间槽数据文件不存在: {e}")
            logger.info("提示: 请先运行数据生成脚本:")
            logger.info("  python main_generate_time_slot.py")
            logger.info("  python generate_event_time_slot.py")
            return False
        except Exception as e:
            logger.error(f"❌ 加载时间槽数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        """运行训练主循环"""
        logger.info(f"🚀 Starting Training: DAgger={self.use_dagger}, Beta={self.beta}")

        # 🔥 加载时间槽数据
        if self.use_timeslot:
            if not self.load_timeslot_data():
                logger.error("❌ 时间槽数据加载失败，退出训练")
                return

        # ============================================
        # 🔥 全局累计计数器 (修复 Acc 显示问题)
        # ============================================
        total_episodes = 0
        total_success = 0
        total_failed = 0

        pbar = tqdm(range(self.max_episodes), desc="RL Training")

        for ep in pbar:
            try:
                # 运行一个 Episode
                ep_reward, ep_info = self._run_episode(ep)

                # 1. 获取资源水平
                curr_res_level = self._get_network_resource_level()

                # 2. ✅ 更新全局计数器 (核心修复)
                total_episodes += 1

                # 判断成功标准：只要 env 说是 success 或 request_completed 就算成
                is_success = ep_info.get('success', False)

                if is_success:
                    total_success += 1
                else:
                    total_failed += 1

                # 3. 计算累计指标
                cum_acc = total_success / total_episodes
                cum_blk = total_failed / total_episodes

                # 4. 记录到 Stats (用于绘图)
                self.stats["rewards"].append(ep_reward)
                self.stats["acceptance_rates"].append(1.0 if is_success else 0.0)
                self.stats["blocking_rates"].append(0.0 if is_success else 1.0)
                self.stats["resource_levels"].append(curr_res_level)

                # 🔥 新增：时间槽统计
                if self.use_timeslot:
                    self.stats["time_slots_covered"].append(ep_info.get('time_slots_covered', 0))
                    self.stats["decision_steps"].append(ep_info.get('decision_steps', 0))
                    self.stats["requests_per_episode"].append(ep_info.get('requests_processed', 1))

                # 5. TensorBoard (记录累计值更平滑)
                self.writer.add_scalar("Train/Reward", ep_reward, ep)
                self.writer.add_scalar("Train/CumulativeAcc", cum_acc, ep)
                self.writer.add_scalar("Train/CumulativeBlk", cum_blk, ep)
                self.writer.add_scalar("Train/Resource", curr_res_level, ep)

                # 🔥 新增：时间槽指标
                if self.use_timeslot:
                    self.writer.add_scalar("Train/TimeSlotsCovered", ep_info.get('time_slots_covered', 0), ep)
                    self.writer.add_scalar("Train/DecisionSteps", ep_info.get('decision_steps', 0), ep)
                    self.writer.add_scalar("Train/CurrentTimeSlot", ep_info.get('current_time_slot', 0), ep)

                if hasattr(self.agent, 'epsilon_low'):
                    self.writer.add_scalar("Train/Epsilon", self.agent.epsilon_low, ep)

                # 6. 更新进度条 (显示全局累计值)
                expert_usage_pct = ep_info.get('expert_usage', 0) * 100

                # 🔥 构建进度条显示
                postfix = {
                    "Rw": f"{ep_reward:.0f}",
                    "Exp": f"{expert_usage_pct:.0f}%",
                    "Acc": f"{cum_acc:.1%}",
                    "Blk": f"{cum_blk:.1%}",
                    "Res": f"{curr_res_level:.0f}%"
                }

                # 🔥 如果启用时间槽，添加时间槽信息
                if self.use_timeslot:
                    postfix["TS"] = ep_info.get('current_time_slot', 0)
                    postfix["DS"] = ep_info.get('decision_steps', 0)

                pbar.set_postfix(postfix)

                # 保存模型
                if (ep + 1) % self.save_freq == 0:
                    self.agent.save(str(self.output_dir / f"rl_model_ep{ep + 1}.pth"))

                    # 🔥 打印时间槽统计
                    if self.use_timeslot and self.log_timeslot_info:
                        self._print_timeslot_stats(ep + 1)

            except Exception as e:
                # 🛡️ 崩溃防御：捕获所有异常，不中断训练
                logger.error(f"❌ Episode {ep} CRASHED: {e}")
                import traceback
                traceback.print_exc()
                # 发生异常算作失败
                total_episodes += 1
                total_failed += 1
                continue

        # 训练结束保存
        self.agent.save(str(self.output_dir / "rl_model_final.pth"))
        logger.info(f"✅ Training Complete. Final Acc: {total_success / total_episodes:.2%}")

        # 🔥 打印最终时间槽统计
        if self.use_timeslot:
            self._print_final_timeslot_stats()

    def _run_episode(self, episode_idx: int):
        """运行一个episode（集成黑名单 + DAgger + 🔥 时间槽系统）"""
        import numpy as np
        import random

        # 获取最大步数
        max_steps = getattr(self, 'max_steps_per_episode', getattr(self.env, 'max_steps', 600))

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

        # 🔥 时间槽跳转检测
        last_time_slot = current_time_slot

        # 获取 mask 和 info
        action_mask = reset_info.get('action_mask')
        blacklist_info = reset_info.get('blacklist_info', {})
        unconnected_dests = self._get_current_destinations()

        done = False
        steps = 0
        decision_steps = 0  # 🔥 决策步数（不是时间！）
        episode_reward = 0

        # DAgger 统计
        expert_steps = 0
        masked_expert_steps = 0

        # 初始化 step_info
        step_info = {'success': False, 'request_completed': False}

        while not done and steps < max_steps:
            # DAgger 逻辑
            beta = getattr(self, 'beta', 0.0)
            use_dagger = getattr(self, 'use_dagger', False)
            use_expert = False
            expert_action = None

            # 🔥🔥🔥 关键修复：从 state 中提取 action_mask 🔥🔥🔥
            action_mask = None

            # 方式1: 从PyG Data对象中提取（你的环境用这种）
            if hasattr(state, 'action_mask'):
                action_mask = state.action_mask
                # 转换为numpy数组
                if hasattr(action_mask, 'cpu'):
                    action_mask = action_mask.cpu().numpy()
                # 去掉多余维度 [1, N] -> [N]
                if action_mask.ndim > 1:
                    action_mask = action_mask.squeeze()

            # 方式2: 从step_info中提取（后备）
            elif 'action_mask' in step_info:
                action_mask = step_info['action_mask']

            # 方式3: 直接调用环境方法（最后兜底）
            if action_mask is None and hasattr(self.env, 'get_low_level_action_mask'):
                action_mask = self.env.get_low_level_action_mask()

            # 🔥 确保mask是numpy数组
            if action_mask is not None:
                if hasattr(action_mask, 'numpy'):
                    action_mask = action_mask.numpy()
                if isinstance(action_mask, list):
                    action_mask = np.array(action_mask)

            # 专家介入判断
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

            # ✅ Agent 选择动作（现在action_mask不是None了）
            high_action, low_action, action_info = self.agent.select_action(
                state=state,
                unconnected_dests=unconnected_dests,
                action_mask=action_mask,  # 🔥 现在这个不是None了
                use_expert=use_expert,
                expert_action=expert_action,
                blacklist_info=blacklist_info
            )

            # 🛡️ 防御：如果 Agent 返回 -1 (无效)，手动处理
            if low_action == -1:
                # 强制结束当前 Episode，视为失败
                logger.warning(f"⚠️ Agent returned -1 (No Valid Actions). Terminating Episode {episode_idx}.")
                return episode_reward, {
                    'success': False,
                    'blocking_rate': 1.0,
                    'message': 'no_valid_actions',
                    'time_slot': current_time_slot,
                    'decision_steps': decision_steps,
                    'time_slots_covered': current_time_slot - initial_time_slot
                }

            # 执行动作
            step_result = self.env.step(low_action)

            # 解包结果
            if len(step_result) == 5:
                next_state, reward, done, truncated, step_info = step_result
            else:
                next_state, reward, done, step_info = step_result
                truncated = False

            # 🔥 更新时间槽信息
            new_time_slot = step_info.get('time_slot', current_time_slot)
            new_decision_steps = step_info.get('decision_steps', decision_steps)

            # 🔥 检测时间槽跳转
            if self.use_timeslot and new_time_slot != last_time_slot:
                if self.log_timeslot_jumps:
                    logger.debug(f"⏰ [Ep {episode_idx}] Time Slot: {last_time_slot} → {new_time_slot}")
                self.timeslot_stats['timeslot_jumps'].append((last_time_slot, new_time_slot))
                last_time_slot = new_time_slot

            current_time_slot = new_time_slot
            decision_steps = new_decision_steps

            # 记录失败原因用于黑名单学习
            if not step_info.get('success', True):
                reason = step_info.get('message', 'unknown')
                if "资源不足" in reason or "访问超限" in reason:
                    self.agent.record_failure(low_action, reason)

            # 存储经验
            if action_info.get('source', '').startswith('agent'):
                # High-Level Buffer
                if action_info.get('high_level_decision', False):
                    goal = unconnected_dests[high_action] if unconnected_dests and high_action < len(
                        unconnected_dests) else -1
                    self.agent.store_transition_high(state, goal, reward, next_state, done or truncated)

                # Low-Level Buffer
                self.agent.store_transition_low(state, low_action, reward, next_state, done or truncated)

            # 更新状态
            state = next_state
            action_mask = step_info.get('action_mask')
            blacklist_info = step_info.get('blacklist_info', {})
            unconnected_dests = self._get_current_destinations()
            episode_reward += reward
            steps += 1

            # 定期更新网络
            if steps % 4 == 0:
                self.agent.update_policies()

            if truncated: done = True

        # ============================================================
        # 🔥🔥🔥 关键修复：Episode 结束统计
        # ============================================================
        # 1. 从info中获取明确的成功状态
        is_success = step_info.get('request_success', None)

        # 2. 如果环境没有明确设置，则用旧逻辑判断
        if is_success is None:
            is_success = step_info.get('request_completed', False) or step_info.get('success', False)

        # 3. 检查环境是否已经处理了归档
        env_already_archived = False
        if hasattr(self.env, 'current_request'):
            # 如果 current_request 已经是 None，说明环境已经归档了
            env_already_archived = (self.env.current_request is None)

        # 4. 只在请求未被归档时才调用 _archive_request
        if not env_already_archived:
            # 环境还没归档，由Trainer来归档
            if hasattr(self.env, 'current_request') and self.env.current_request:
                req_id = self.env.current_request.get('id', '?')

                if not is_success:
                    logger.info(f"🔄 [Episode清理] 请求 {req_id} 失败，执行回滚...")
                    self.env._archive_request(success=False)
                else:
                    logger.info(f"✅ [Episode清理] 请求 {req_id} 成功，归档资源...")
                    self.env._archive_request(success=True)

                # 重置环境状态以便下一个 Episode
                self.env.current_request = None
                self.env.current_branch_id = None
                self.env.current_tree = {}
                self.env.nodes_on_tree = set()
                self.env.branch_states = {}
                # 清空本轮账本
                if hasattr(self.env, 'curr_ep_node_allocs'):
                    self.env.curr_ep_node_allocs = []
                if hasattr(self.env, 'curr_ep_link_allocs'):
                    self.env.curr_ep_link_allocs = []
        else:
            # 环境已经归档，Trainer不需要再次归档
            logger.info(f"ℹ️ [Episode清理] 环境已归档，跳过Trainer归档")

        # ============================================================
        # 🔥 构建完整的 episode_info（包含时间槽信息）
        # ============================================================
        episode_info = {
            'steps': steps,
            'success': is_success,
            'blocking_rate': 0.0 if is_success else 1.0,
            'expert_usage': expert_steps / steps if steps > 0 else 0,
            'masked_expert': masked_expert_steps,

            # 🔥 时间槽信息
            'current_time_slot': current_time_slot,
            'initial_time_slot': initial_time_slot,
            'time_slots_covered': current_time_slot - initial_time_slot,
            'decision_steps': decision_steps,
            'request_id': request_id,
            'requests_processed': 1  # 默认每个episode处理1个请求
        }

        # 🔥 更新时间槽统计
        if self.use_timeslot:
            self.timeslot_stats['total_time_slots'] += (current_time_slot - initial_time_slot)
            self.timeslot_stats['total_decision_steps'] += decision_steps

        # 简单日志
        status_icon = "✅" if is_success else "❌"
        if is_success or episode_idx % 10 == 0:  # 减少日志刷屏
            if self.use_timeslot:
                logger.info(
                    f"Ep {episode_idx} | {status_icon} | "
                    f"Rw: {episode_reward:.1f} | "
                    f"Steps: {steps} | "
                    f"TS: {current_time_slot} | "
                    f"DS: {decision_steps}"
                )
            else:
                logger.info(f"Ep {episode_idx} | {status_icon} | Rw: {episode_reward:.1f} | Steps: {steps}")

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

    def _print_timeslot_stats(self, episode):
        """
        🔥 新增：打印时间槽统计信息
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"⏰ 时间槽统计 @ Episode {episode}")
        logger.info(f"{'=' * 60}")

        if self.timeslot_stats['total_decision_steps'] > 0:
            avg_steps = (self.timeslot_stats['total_decision_steps'] /
                         max(1, len(self.stats['decision_steps'])))
            logger.info(f"平均决策步数: {avg_steps:.1f}")

        if len(self.stats['time_slots_covered']) > 0:
            avg_slots = np.mean(self.stats['time_slots_covered'][-100:])
            logger.info(f"平均时间槽跨度: {avg_slots:.1f}")

        if len(self.timeslot_stats['timeslot_jumps']) > 0:
            logger.info(f"时间槽跳转次数: {len(self.timeslot_stats['timeslot_jumps'])}")

        logger.info(f"{'=' * 60}\n")

    def _print_final_timeslot_stats(self):
        """
        🔥 新增：打印最终时间槽统计
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎉 最终时间槽统计")
        logger.info(f"{'=' * 60}")

        total_episodes = len(self.stats['decision_steps'])

        if total_episodes > 0:
            avg_decision_steps = np.mean(self.stats['decision_steps'])
            avg_time_slots = np.mean(self.stats['time_slots_covered'])

            logger.info(f"总Episodes: {total_episodes}")
            logger.info(f"平均决策步数: {avg_decision_steps:.1f}")
            logger.info(f"平均时间槽跨度: {avg_time_slots:.1f}")
            logger.info(f"总时间槽跳转: {len(self.timeslot_stats['timeslot_jumps'])}")

            if self.timeslot_stats['total_decision_steps'] > 0:
                efficiency = (self.timeslot_stats['total_time_slots'] /
                              self.timeslot_stats['total_decision_steps'])
                logger.info(f"时间槽效率: {efficiency:.2f} (时间槽/决策步)")

        logger.info(f"{'=' * 60}\n")