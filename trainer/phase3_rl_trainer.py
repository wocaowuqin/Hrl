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
from envs.modules.HRL_Coordinator import visualize_sfc_tree_publication
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
            "tree_lengths": [],
            "sharing_ratios": [],    # [SDG-HRL] 边共享率
            "tree_bias_vals": [],    # [SDG-HRL] tree_bias 学习轨迹
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
            if info.get('req_snapshot'):
                trees_data.append({
                    'ep':           episode + 1,
                    'success':      info.get('success', False),
                    'req':          info['req_snapshot'],
                    'tree':         info.get('tree_snapshot'),
                    'chain':        info.get('chain_nodes', []),
                    'sfc_snapshot': info.get('sfc_snapshot'),  # DAG结构
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
                    _sfc_snap = info.get('sfc_snapshot') or {}
                    _all_edges = set()
                    for _seg in _sfc_snap.get('spine_paths', []):
                        for _i in range(len(_seg)-1):
                            _all_edges.add(tuple(sorted((_seg[_i], _seg[_i+1]))))
                    for _bp in _sfc_snap.get('branch_paths', {}).values():
                        for _i in range(len(_bp)-1):
                            _all_edges.add(tuple(sorted((_bp[_i], _bp[_i+1]))))
                    tree_len = len(_all_edges) if _all_edges else                         len(self.env.current_tree.get('tree', {}))
                except Exception:
                    tree_len = len(self.env.current_tree.get('tree', {}))
            self.stats['tree_lengths'].append(tree_len)

            # ── [SDG-HRL] 边共享率：同一条物理边被多条路径复用的比例 ────────
            # 共享率高 → 树更紧凑，说明 tree_bias 归纳偏置在起作用
            sharing_ratio = 0.0
            if info.get('success', False):
                try:
                    _sfc = info.get('sfc_snapshot') or {}
                    _all_edges_list = []
                    for _seg in _sfc.get('spine_paths', []):
                        for _i in range(len(_seg) - 1):
                            _all_edges_list.append(tuple(sorted((_seg[_i], _seg[_i+1]))))
                    for _bp in _sfc.get('branch_paths', {}).values():
                        for _i in range(len(_bp) - 1):
                            _all_edges_list.append(tuple(sorted((_bp[_i], _bp[_i+1]))))
                    if _all_edges_list:
                        _unique = len(set(_all_edges_list))
                        _total  = len(_all_edges_list)
                        sharing_ratio = 1.0 - (_unique / _total) if _total > 0 else 0.0
                except Exception:
                    sharing_ratio = 0.0
            self.stats['sharing_ratios'].append(sharing_ratio)

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

            # --- 采集剩余资源量（只统计DC节点，非DC节点CPU/MEM不被VNF消耗，算入会虚低）---
            try:
                _dc_nodes = getattr(self.env.resource_mgr, 'dc_nodes', list(range(self.env.n)))
                _n_dc = max(1, len(_dc_nodes))
                total_cpu_avail = sum(
                    self.env.resource_mgr.pool.get_available_cpu(i)
                    for i in _dc_nodes
                )
                avg_cpu_avail = total_cpu_avail / _n_dc
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

            # ── [SDG-HRL] 核心指标监控 ──────────────────────────────────
            try:
                _enc = getattr(self.agent, 'encoder', None)
                if _enc is not None:
                    if hasattr(_enc, 'encoder'): _enc = _enc.encoder
                    if hasattr(_enc, 'gat') and hasattr(_enc.gat, 'encoder'):
                        _enc = _enc.gat.encoder
                    if hasattr(_enc, 'tree_bias'):
                        _tb_val = _enc.tree_bias.item()
                        self.stats['tree_bias_vals'].append(_tb_val)
                        self.writer.add_scalar('SDG_HRL/tree_bias', _tb_val, episode)
            except Exception:
                pass

            # 边共享率（每episode记录，成功时才有意义）
            self.writer.add_scalar('SDG_HRL/sharing_ratio', sharing_ratio, episode)
            if sharing_ratio > 0:
                self.writer.add_scalar('SDG_HRL/sharing_ratio_nonzero', sharing_ratio, episode)

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

                # 计算最近10次共享率均值（只算成功episode）
                recent_sharing = [s for s in self.stats['sharing_ratios'][-10:] if s > 0]
                avg_sharing = np.mean(recent_sharing) if recent_sharing else 0.0

                # tree_bias 当前值
                _tb_str = ""
                if self.stats['tree_bias_vals']:
                    _tb_str = f" | tree_bias={self.stats['tree_bias_vals'][-1]:.6f}"

                logger.info(
                    f"Ep {episode}: Rate={success_rate:.2%} | "
                    f"Rwd={total_reward:.1f} | "
                    f"Util={res_util:.2f} | "
                    f"CPU={avg_cpu_avail:.1f} BW={avg_bw_avail:.1f} | "
                    f"HLoss={high_loss:.4f} LLoss={low_loss:.4f} | "
                    f"TreeLen={avg_tree_len:.1f} Sharing={avg_sharing:.3f}"
                    f"{_tb_str}{eps_str}"
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

        # 生成多播树可视化 + 路径打印
        if trees_data:
            try:
                import os
                vis_dir = os.path.join(str(self.output_dir.parent), 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                for _td in trees_data:
                    _ep   = _td.get('ep', '?')
                    _succ = _td.get('success', False)
                    _sfc  = _td.get('sfc_snapshot') or {}
                    _req  = _td.get('req') or {}
                    # ── 路径打印 ──────────────────────────────────────
                    # print(f"\n{'='*55}")
                    # print(f"Ep {_ep} {'✓' if _succ else '✗'}  "
                    #       f"src={_req.get('source','?')}  "
                    #       f"dest={_req.get('dest','?')}")
                    # print(f"  chain : {_sfc.get('chain_nodes','?')}")
                    # for _k, _seg in enumerate(_sfc.get('spine_paths', [])):
                    #     print(f"  spine[{_k}]: {_seg}")
                    # _br = _sfc.get('branch_roots', {})
                    # for _d, _path in _sfc.get('branch_paths', {}).items():
                    #     _root = _br.get(_d, _br.get(str(_d), '?'))
                    #     print(f"  branch dest={_d} root={_root}: {_path}")
                    # ── 单图可视化 ────────────────────────────────────
                    _ep_str = str(_ep).zfill(4)
                    _vis_path = os.path.join(vis_dir, f'sfc_tree_ep{_ep_str}.png')
                    try:
                        visualize_sfc_tree_publication(_td, save_path=_vis_path)
                    except Exception as _ve:
                        logger.warning(f'可视化 ep{_ep} 失败: {_ve}')
                logger.info(f'🎨 可视化已保存到: {vis_dir}/')
            except Exception as e:
                logger.warning(f'⚠️ 可视化生成失败: {e}')

        # 打印最终统计
        valid_tree_lens = [l for l in self.stats['tree_lengths'] if l > 0]
        avg_tree_final = np.mean(valid_tree_lens) if valid_tree_lens else 0

        valid_sharing = [s for s in self.stats['sharing_ratios'] if s > 0]
        avg_sharing_final = np.mean(valid_sharing) if valid_sharing else 0.0
        tb_start = self.stats['tree_bias_vals'][0]  if self.stats['tree_bias_vals'] else 0.5
        tb_end   = self.stats['tree_bias_vals'][-1] if self.stats['tree_bias_vals'] else 0.5

        print("\n📊 最终统计结果:")
        print(f"   ✅ 最终成功率:      {success_rate:.2%}")
        print(f"   💰 平均奖励:        {np.mean(self.stats['rewards']):.2f}")
        print(f"   🔋 平均资源利用率:  {np.mean(self.stats['resource_utilization']):.2f}")
        print(f"   🌳 平均树长 (成功): {avg_tree_final:.2f}")
        print(f"   🔗 平均边共享率:    {avg_sharing_final:.3f}  (越高→树越紧凑)")
        print(f"   🎯 tree_bias 轨迹:  {tb_start:.6f} → {tb_end:.6f}  (偏离0.5说明GNN在学习树结构)")
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