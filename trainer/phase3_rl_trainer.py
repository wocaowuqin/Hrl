# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 RL Trainer - 极简日志版

特点：
1. 专注于关键指标：成功率、资源利用率、树长。
2. 减少冗余日志输出。
"""
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
from envs.modules.HRL_Coordinator import visualize_multicast_trees
from trainer.training_analyzer import TrainingAnalyzer

logger = logging.getLogger(__name__)


class Phase3RLTrainer:
    """Phase 3: RL Trainer with HRL Coordinator (Clean Logs)"""

    def __init__(self, env, agent, output_dir, config, coordinator):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config

        if coordinator is None:
            raise ValueError("❌ Coordinator必须传入！")
        self.coordinator = coordinator

        # 初始化可视化器
        self.visualizer = None
        if hasattr(env, 'topo'):
            try:
                self.visualizer = SFCVisualizer(env.topo, output_dir)
            except Exception:
                pass

        # 训练参数
        phase3_cfg = config.get("phase3", {})
        self.max_episodes = phase3_cfg.get("episodes", 1000)
        self.save_freq = phase3_cfg.get("save_every", 100)
        self.max_steps_per_episode = phase3_cfg.get("max_steps", 600)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "runs"))

        # 统计信息
        self.stats = {
            "rewards": [],
            "episode_lengths": [],
            "success_rate": [],
            "resource_utilization": [],
            "tree_lengths": []
        }

        logger.info("✅ Trainer初始化完成 (极简日志版)")

        # 训练分析器
        self.analyzer = TrainingAnalyzer(output_dir=str(self.output_dir))

    def run(self):
        """🚀 训练主循环"""

        logger.info("\n" + "=" * 40)
        logger.info(f"🎬 开始训练 ({self.max_episodes} eps)")
        logger.info("=" * 40)

        # 简单的环境重置
        self.env.reset()

        num_episodes = self.cfg.get('num_episodes', self.max_episodes)
        success_count = 0

        # 使用 tqdm 显示进度条，替代刷屏日志
        pbar = tqdm(range(num_episodes), desc="Training", unit="ep")

        trees_data = []  # 收集每个episode的树数据供可视化

        for episode in pbar:
            # 执行 Episode
            total_reward, info = self.coordinator.run_episode(
                max_steps=self.max_steps_per_episode
            )

            # --- 收集树快照供可视化 ---
            if info.get('tree_snapshot') and info.get('req_snapshot'):
                trees_data.append({
                    'ep':      episode + 1,
                    'success': info.get('success', False),
                    'req':     info['req_snapshot'],
                    'tree':    info['tree_snapshot'],
                    'chain':   info.get('chain_nodes', []),
                })

            # --- 统计数据采集 ---
            self.stats['rewards'].append(total_reward)
            self.stats['episode_lengths'].append(info.get('steps', 0))

            # 1. 资源利用率
            try:
                res_util = self.env.get_resource_utilization()
            except Exception:
                res_util = 0.0
            self.stats['resource_utilization'].append(res_util)

            # 2. 树长 & 成功计数
            tree_len = 0
            if info.get('success', False):
                success_count += 1
                try:
                    tree_len = len(self.env.current_tree.get('tree', {}))
                except Exception:
                    tree_len = 0
            self.stats['tree_lengths'].append(tree_len)

            # 3. 成功率
            success_rate = success_count / (episode + 1)
            self.stats['success_rate'].append(success_rate)

            # --- 训练 Agent（先更新权重和 epsilon，再记录快照）---
            # update_policies() 是正确入口，包含 _update_epsilon() 调用
            losses = {}
            if hasattr(self.agent, 'update_policies'):
                try:
                    losses = self.agent.update_policies() or {}
                except Exception as e:
                    logger.debug(f"agent.update_policies() 异常: {e}")
            elif hasattr(self.agent, 'learn'):
                try:
                    self.agent.learn()
                except Exception as e:
                    logger.debug(f"agent.learn() 异常: {e}")
            elif hasattr(self.agent, 'update'):
                try:
                    self.agent.update()
                except Exception as e:
                    logger.debug(f"agent.update() 异常: {e}")

            # --- 分析器记录（learn 之后，epsilon 已更新）---
            self.analyzer.record(
                episode=episode,
                info=info,
                res_util=res_util,
                env=self.env,
                coordinator=self.coordinator,
                agent=self.agent,
            )

            # --- 采集剩余资源量 ---
            try:
                total_cpu_avail = sum(
                    self.env.resource_mgr.pool.get_available_cpu(i)
                    for i in range(self.env.n)
                )
                avg_cpu_avail = total_cpu_avail / self.env.n
            except Exception:
                avg_cpu_avail = 0.0

            try:
                bw_vals = []
                for u in range(self.env.n):
                    for v in self.env.resource_mgr.get_neighbors(u):
                        if v > u:
                            bw_vals.append(self.env.resource_mgr.pool.get_available_bandwidth(u, v))
                avg_bw_avail = sum(bw_vals) / len(bw_vals) if bw_vals else 0.0
            except Exception:
                avg_bw_avail = 0.0

            # --- 进度条更新 ---
            eps_low = getattr(self.agent, 'epsilon_low', None)
            high_loss = losses.get('high_loss', 0.0)
            low_loss = losses.get('low_loss', 0.0)
            postfix = {
                'Suc': f"{success_rate:.1%}",
                'Rwd': f"{total_reward:.1f}",
                'CPU': f"{avg_cpu_avail:.1f}",
                'BW': f"{avg_bw_avail:.1f}",
                'HLoss': f"{high_loss:.3f}",
                'LLoss': f"{low_loss:.3f}",
            }
            if eps_low is not None:
                postfix['ε'] = f"{eps_low:.3f}"
            pbar.set_postfix(postfix)

            # --- TensorBoard 记录 ---
            self.writer.add_scalar('Episode/Reward', total_reward, episode)
            self.writer.add_scalar('Episode/SuccessRate', success_rate, episode)
            self.writer.add_scalar('Episode/ResourceUtil', res_util, episode)
            self.writer.add_scalar('Resource/AvgCPU', avg_cpu_avail, episode)
            self.writer.add_scalar('Resource/AvgBW', avg_bw_avail, episode)
            if high_loss > 0:
                self.writer.add_scalar('Loss/HighLevel', high_loss, episode)
            if low_loss > 0:
                self.writer.add_scalar('Loss/LowLevel', low_loss, episode)
            if tree_len > 0:
                self.writer.add_scalar('Episode/TreeLength', tree_len, episode)

            # --- 定期日志 (每10轮详细一点) ---
            if episode > 0 and episode % 10 == 0:
                # 计算最近10次的平均树长
                recent_trees = [l for l in self.stats['tree_lengths'][-10:] if l > 0]
                avg_tree_len = np.mean(recent_trees) if recent_trees else 0.0

                # 读取 epsilon（如果 agent 支持）
                eps_high = getattr(self.agent, 'epsilon_high', None)
                eps_low  = getattr(self.agent, 'epsilon_low',  None)
                steps    = getattr(self.agent, 'steps_done',   None)
                eps_str  = ""
                if eps_high is not None:
                    eps_str = f" | ε_h={eps_high:.3f} ε_l={eps_low:.3f} steps={steps}"

                logger.info(
                    f"Ep {episode}: Rate={success_rate:.2%} | "
                    f"Rwd={total_reward:.1f} | "
                    f"Util={res_util:.2f} | "
                    f"CPU剩余={avg_cpu_avail:.1f} BW剩余={avg_bw_avail:.1f} | "
                    f"HLoss={high_loss:.4f} LLoss={low_loss:.4f} | "
                    f"TreeLen={avg_tree_len:.1f}{eps_str}"
                )

            # --- 保存模型 ---
            if episode > 0 and episode % self.save_freq == 0:
                self._save_checkpoint(episode)

        # ====================================================================
        # 训练结束
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("🎉 训练完成")
        logger.info("=" * 40)

        self._save_final_model(num_episodes)

        # 生成多播树可视化
        if trees_data:
            try:
                import os
                vis_dir = os.path.join(str(self.output_dir.parent), 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, 'multicast_trees_vis.png')
                visualize_multicast_trees(trees_data, save_path=vis_path)
                logger.info(f'🎨 多播树可视化已保存: {vis_path}')
            except Exception as e:
                logger.warning(f'⚠️ 可视化生成失败: {e}')

        # 打印最终统计
        valid_tree_lens = [l for l in self.stats['tree_lengths'] if l > 0]
        avg_tree_final = np.mean(valid_tree_lens) if valid_tree_lens else 0

        print("\n📊 最终统计结果:")
        print(f"   ✅ 最终成功率: {success_rate:.2%}")
        print(f"   💰 平均奖励:   {np.mean(self.stats['rewards']):.2f}")
        print(f"   🔋 平均资源利用率: {np.mean(self.stats['resource_utilization']):.2f}")
        print(f"   🌳 平均树长 (成功): {avg_tree_final:.2f}")
        print("=" * 40)

        # 生成失败分析报告
        self.analyzer.report()

    def _save_checkpoint(self, episode):
        """保存检查点"""
        save_path = self.output_dir / f"checkpoint_ep{episode}.pth"
        try:
            torch.save({
                'episode': episode,
                'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
                'config': self.cfg,
                'stats': self.stats
            }, save_path)
        except Exception:
            pass

    def _save_final_model(self, episode):
        """保存最终模型"""
        final_path = self.output_dir / "final_model.pth"
        try:
            torch.save({
                'episode': episode,
                'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
                'config': self.cfg,
                'stats': self.stats
            }, final_path)
            logger.info(f"💾 模型已保存: {final_path}")
        except Exception as e:
            logger.warning(f"⚠️ 保存失败: {e}")