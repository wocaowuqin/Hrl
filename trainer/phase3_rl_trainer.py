# core/trainer/phase3_rl_trainer.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 RL Trainer - TA-HRL v4 极简日志版

特点：
1. 专注于关键指标：成功率、资源利用率、树长。
2. 剥离了旧的 GAT Verify 工具，兼容最新架构。
"""
import logging
import os
import csv
import json
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
            "sharing_ratios": [],    # 边共享率
            "tree_bias_vals": [],    # tree_bias 学习轨迹
        }

        logger.info("✅ Trainer初始化完成 (TA-HRL v4 极简版)")
        self.analyzer = TrainingAnalyzer(output_dir=str(self.output_dir))

        # ── 数据集输出初始化 ───────────────────────────────────────────────
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_csv_path = self.dataset_dir / "episodes.csv"
        self._dataset_csv_file = open(self._dataset_csv_path, 'w', newline='', encoding='utf-8')
        self._dataset_csv_writer = csv.DictWriter(self._dataset_csv_file, fieldnames=[
            'episode', 'success', 'fail_reason',
            'total_reward', 'steps',
            'vnf_done', 'vnf_total', 'dest_done', 'dest_total',
            'tree_len', 'sharing_ratio',
            'cpu_util_pct', 'bw_util_pct', 'res_util',
            'cpu_used_abs', 'mem_used_abs', 'bw_used_abs',
            'accept_rate', 'block_rate',
            'epsilon_high', 'epsilon_low',
            'high_loss', 'low_loss',
            'request_id', 'arrival_time',
        ])
        self._dataset_csv_writer.writeheader()
        self._dataset_records = []   # 同时收集为 list-of-dict，训练结束存 JSON/pkl
        logger.info(f"📂 数据集将输出到: {self.dataset_dir}")

        # ── 加载 Phase2 IL 预训练权重 ──────────────────────────────────────
        # 只加载网络权重，不加载 optimizer state：
        # Phase2 用 IL lr(3e-4)，Phase3 用 RL lr，optimizer state 不兼容。
        il_ckpt_path = (
            config.get('phase3', {}).get('il_checkpoint') or
            config.get('il_checkpoint') or
            str(Path(output_dir).parent / 'il_output' / 'il_model_best.pth')
        )
        self._load_il_checkpoint(il_ckpt_path)

    def run(self):
        """🚀 训练主循环"""
        # 诊断：检查encoder是否在optimizer_low里
        logger.info("=== Encoder诊断 ===")
        logger.info(f"encoder is None: {self.agent.encoder is None}")
        if self.agent.encoder is not None:
            enc_params = list(self.agent.encoder.parameters())
            logger.info(f"encoder参数数量: {len(enc_params)}")
            logger.info(f"tree_bias requires_grad: {self.agent.encoder.tree_bias.requires_grad}")
            logger.info(f"tree_bias初始值: {self.agent.encoder.tree_bias.item():.6f}")

            # 检查optimizer_low里有没有encoder参数
            opt_param_ids = set()
            for pg in self.agent.optimizer_low.param_groups:
                for p in pg['params']:
                    opt_param_ids.add(id(p))
            enc_in_opt = sum(1 for p in enc_params if id(p) in opt_param_ids)
            logger.info(f"encoder参数在optimizer_low中: {enc_in_opt}/{len(enc_params)}")
            logger.info(f"optimizer_low参数组数量: {len(self.agent.optimizer_low.param_groups)}")


        logger.info("\n" + "=" * 40)
        logger.info(f"🎬 开始训练 ({self.max_episodes} eps)")
        logger.info("=" * 40)

        self.env.reset()

        num_episodes = self.cfg.get('num_episodes', self.max_episodes)
        success_count = 0

        pbar = tqdm(range(num_episodes), desc="Training", unit="ep")
        trees_data = []

        for episode in pbar:
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
                    'sfc_snapshot': info.get('sfc_snapshot'),
                })

            # --- 统计数据采集 ---
            self.stats['rewards'].append(total_reward)
            self.stats['episode_lengths'].append(info.get('steps', 0))

            try:
                res_util = self.env.get_resource_utilization()
            except Exception:
                res_util = 0.0
            self.stats['resource_utilization'].append(res_util)

            # 树长 & 成功计数
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
                    tree_len = len(_all_edges) if _all_edges else len(self.env.current_tree.get('tree', {}))
                except Exception:
                    tree_len = len(self.env.current_tree.get('tree', {}))
            self.stats['tree_lengths'].append(tree_len)

            # 边共享率
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

            success_rate = success_count / (episode + 1)
            self.stats['success_rate'].append(success_rate)

            # --- 训练 Agent ---
            losses = {}
            if hasattr(self.agent, 'update_policies'):
                try:
                    losses = self.agent.update_policies() or {}
                except Exception as e:
                    logger.debug(f"agent.update_policies() 异常: {e}")

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
                dc_nodes = getattr(self.env, 'dc_nodes', list(range(self.env.n)))
                cpu_used = sum(max(0.0, self.env.resource_mgr.pool.cpu_cap[i] - self.env.resource_mgr.pool.get_available_cpu(i)) for i in dc_nodes)
                cpu_cap = sum(self.env.resource_mgr.pool.cpu_cap[i] for i in dc_nodes)
                avg_cpu_avail = cpu_used / max(cpu_cap, 1.0) * 100.0
            except Exception:
                avg_cpu_avail = 0.0

            try:
                bw_used_total = 0.0
                bw_cap_total  = 0.0
                for u in range(self.env.n):
                    for v in self.env.resource_mgr.get_neighbors(u):
                        if v > u:
                            key = tuple(sorted((u, v)))
                            cap   = self.env.resource_mgr.pool.bw_cap.get(key, 0.0)
                            avail = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                            bw_used_total += max(0.0, cap - avail)
                            bw_cap_total  += cap
                avg_bw_avail = bw_used_total / max(bw_cap_total, 1.0) * 100.0
            except Exception:
                avg_bw_avail = 0.0
                bw_used_total = 0.0

            # --- 累计 MEM 消耗绝对量 ---
            try:
                dc_nodes = getattr(self.env, 'dc_nodes', list(range(self.env.n)))
                mem_used_abs = sum(
                    max(0.0, self.env.resource_mgr.pool.mem_cap[i]
                        - self.env.resource_mgr.pool.get_available_mem(i))
                    for i in dc_nodes
                )
            except Exception:
                mem_used_abs = 0.0

            # --- 进度条更新 ---
            eps_low = getattr(self.agent, 'epsilon_low', None)
            high_loss = losses.get('high_loss', 0.0)
            low_loss = losses.get('low_loss', 0.0)
            postfix = {
                'Suc': f"{success_rate:.1%}",
                'Rwd': f"{total_reward:.1f}",
                'CPU': f"{avg_cpu_avail:.1f}%",
                'BW': f"{avg_bw_avail:.1f}%",
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
            if high_loss > 0: self.writer.add_scalar('Loss/HighLevel', high_loss, episode)
            if low_loss > 0: self.writer.add_scalar('Loss/LowLevel', low_loss, episode)
            if tree_len > 0: self.writer.add_scalar('Episode/TreeLength', tree_len, episode)

            try:
                _enc = getattr(self.agent, 'encoder', None)
                if _enc is not None and hasattr(_enc, 'tree_bias'):
                    _tb_raw = _enc.tree_bias.item()
                    # 换算为有效权重 (Sigmoid)
                    _tb_eff = torch.sigmoid(torch.tensor(_tb_raw)).item()
                    self.stats['tree_bias_vals'].append(_tb_eff)
                    self.writer.add_scalar('TA_HRL/tree_bias_effective', _tb_eff, episode)
                    self.writer.add_scalar('TA_HRL/tree_bias_raw', _tb_raw, episode)
            except Exception:
                pass

            self.writer.add_scalar('TA_HRL/sharing_ratio', sharing_ratio, episode)

            # ── 数据集记录（每个episode写一行）─────────────────────────────
            try:
                _eps_h = getattr(self.agent, 'epsilon_high', None)
                _eps_l = getattr(self.agent, 'epsilon_low',  None)
                _req   = getattr(self.env, 'current_request', None) or {}
                _tree  = getattr(self.env, 'current_tree', None) or {}
                _vnf_done  = getattr(self.env, 'next_vnf_idx', 0)
                _vnf_total = len(_req.get('vnf', []))
                _dest_done = len(_tree.get('connected_dests', set()))
                _dest_total= len(_req.get('dest', []))
                _row = {
                    'episode':      episode,
                    'success':      int(bool(info.get('success', False))),
                    'fail_reason':  info.get('reason', '') or info.get('error', ''),
                    'total_reward': round(float(total_reward), 4),
                    'steps':        info.get('steps', 0),
                    'vnf_done':     _vnf_done,
                    'vnf_total':    _vnf_total,
                    'dest_done':    _dest_done,
                    'dest_total':   _dest_total,
                    'tree_len':     tree_len,
                    'sharing_ratio': round(float(sharing_ratio), 4),
                    'cpu_util_pct': round(float(avg_cpu_avail), 2),
                    'bw_util_pct':  round(float(avg_bw_avail), 2),
                    'res_util':     round(float(res_util), 4),
                    'cpu_used_abs': round(float(cpu_used), 4),
                    'mem_used_abs': round(float(mem_used_abs), 4),
                    'bw_used_abs':  round(float(bw_used_total), 4),
                    'accept_rate':  round(float(success_rate), 4),
                    'block_rate':   round(1.0 - float(success_rate), 4),
                    'epsilon_high': round(float(_eps_h), 4) if _eps_h is not None else '',
                    'epsilon_low':  round(float(_eps_l), 4) if _eps_l is not None else '',
                    'high_loss':    round(float(losses.get('high_loss', 0.0)), 6),
                    'low_loss':     round(float(losses.get('low_loss',  0.0)), 6),
                    'request_id':   _req.get('id', ''),
                    'arrival_time': _req.get('arrival_time', ''),
                }
                self._dataset_csv_writer.writerow(_row)
                self._dataset_records.append(_row)
                # 每100ep flush一次，防止异常丢数据
                if episode % 100 == 0:
                    self._dataset_csv_file.flush()
            except Exception as _de:
                logger.debug(f"数据集记录失败 ep{episode}: {_de}")

            if episode > 0 and episode % 10 == 0:
                recent_trees = [l for l in self.stats['tree_lengths'][-10:] if l > 0]
                avg_tree_len = np.mean(recent_trees) if recent_trees else 0.0

                eps_high = getattr(self.agent, 'epsilon_high', None)
                eps_low  = getattr(self.agent, 'epsilon_low',  None)
                steps    = getattr(self.agent, 'steps_done',   None)
                eps_str  = f" | ε_h={eps_high:.3f} ε_l={eps_low:.3f} steps={steps}" if eps_high is not None else ""

                recent_sharing = [s for s in self.stats['sharing_ratios'][-10:] if s > 0]
                avg_sharing = np.mean(recent_sharing) if recent_sharing else 0.0

                _tb_str = f" | tree_bias(eff)={self.stats['tree_bias_vals'][-1]:.4f}" if self.stats[
                    'tree_bias_vals'] else ""
                logger.info(
                    f"Ep {episode}: Rate={success_rate:.2%} | "
                    f"Rwd={total_reward:.1f} | Util={res_util:.2f} | "
                    f"CPU={avg_cpu_avail:.1f}% BW={avg_bw_avail:.1f}% | "
                    f"HLoss={high_loss:.4f} LLoss={low_loss:.4f} | "
                    f"TreeLen={avg_tree_len:.1f} Sharing={avg_sharing:.3f}"
                    f"{_tb_str}{eps_str}"
                )

            if episode > 0 and episode % self.save_freq == 0:
                self._save_checkpoint(episode)

        logger.info("\n" + "=" * 40)
        logger.info("🎉 训练完成")
        logger.info("=" * 40)

        self._save_final_model(num_episodes)

        if trees_data:
            try:
                import os
                vis_dir = os.path.join(str(self.output_dir.parent), 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                for _td in trees_data:
                    _ep = _td.get('ep', '?')
                    _ep_str = str(_ep).zfill(4)
                    _vis_path = os.path.join(vis_dir, f'sfc_tree_ep{_ep_str}.png')
                    try:
                        visualize_sfc_tree_publication(_td, save_path=_vis_path)
                    except Exception as _ve:
                        logger.warning(f'可视化 ep{_ep} 失败: {_ve}')
                logger.info(f'🎨 可视化已保存到: {vis_dir}/')
            except Exception as e:
                logger.warning(f'⚠️ 可视化生成失败: {e}')

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
        print(f"   🔗 平均边共享率:    {avg_sharing_final:.3f}")
        print(f"   🎯 tree_bias(有效权重) 轨迹:  {tb_start:.4f} → {tb_end:.4f}  (>0.5说明模型主动利用了多播树结构)")
        print("=" * 40)

        self.analyzer.report()

        # ── 训练结束：保存完整数据集 ──────────────────────────────────────
        try:
            self._dataset_csv_file.flush()
            self._dataset_csv_file.close()
            logger.info(f"✅ CSV 数据集已保存: {self._dataset_csv_path}")

            # JSON（便于跨语言读取）
            json_path = self.dataset_dir / "episodes.json"
            with open(json_path, 'w', encoding='utf-8') as _jf:
                json.dump(self._dataset_records, _jf, ensure_ascii=False, indent=2)
            logger.info(f"✅ JSON 数据集已保存: {json_path}")

            # Pickle（保留完整精度，便于Python直接 pd.read_pickle）
            pkl_path = self.dataset_dir / "episodes.pkl"
            with open(pkl_path, 'wb') as _pf:
                pickle.dump(self._dataset_records, _pf)
            logger.info(f"✅ Pickle 数据集已保存: {pkl_path}")

            # 统计摘要
            n_total   = len(self._dataset_records)
            n_success = sum(r['success'] for r in self._dataset_records)
            logger.info(f"📊 数据集统计: 共 {n_total} 条, 成功 {n_success} ({n_success/max(n_total,1):.1%})")
        except Exception as _se:
            logger.warning(f"⚠️ 数据集保存失败: {_se}")

    def _load_il_checkpoint(self, ckpt_path: str):
        """加载 Phase2 IL 预训练权重到 agent，跳过 optimizer state。
        加载成功后立即同步 target 网络，避免 Q 值估计初期震荡。
        """
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning(
                f"⚠️  Phase2 checkpoint 未找到: {ckpt_path}\n"
                "   Phase3 将从随机初始化开始训练（模仿学习收益丢失）。\n"
                "   请在 config['phase3']['il_checkpoint'] 或 "
                "config['il_checkpoint'] 中指定正确路径。"
            )
            return

        try:
            ckpt = torch.load(ckpt_path, map_location=self.agent.device)
            loaded = []

            # high_policy
            if 'high_policy' in ckpt and hasattr(self.agent, 'high_policy'):
                missing, unexpected = self.agent.high_policy.load_state_dict(
                    ckpt['high_policy'], strict=False)
                loaded.append(f"high_policy(missing={len(missing)}, unexpected={len(unexpected)})")
                # 同步 target
                if hasattr(self.agent, 'target_high_policy'):
                    self.agent.target_high_policy.load_state_dict(
                        self.agent.high_policy.state_dict())
                    loaded.append("target_high_policy←synced")

            # low_policy
            if 'low_policy' in ckpt and hasattr(self.agent, 'low_policy'):
                missing, unexpected = self.agent.low_policy.load_state_dict(
                    ckpt['low_policy'], strict=False)
                loaded.append(f"low_policy(missing={len(missing)}, unexpected={len(unexpected)})")
                # 同步 target
                if hasattr(self.agent, 'target_low_policy'):
                    self.agent.target_low_policy.load_state_dict(
                        self.agent.low_policy.state_dict())
                    loaded.append("target_low_policy←synced")

            # encoder
            if 'encoder' in ckpt and hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                missing, unexpected = self.agent.encoder.load_state_dict(
                    ckpt['encoder'], strict=False)
                loaded.append(f"encoder(missing={len(missing)}, unexpected={len(unexpected)})")

            logger.info(f"✅ Phase2 IL 权重加载成功: {ckpt_path}")
            logger.info(f"   加载项: {', '.join(loaded)}")

        except Exception as e:
            logger.error(f"❌ Phase2 checkpoint 加载失败: {e}，Phase3 从随机初始化开始。")

    def _get_agent_state(self):
        """手动提取HRLAgent权重（HRLAgent不继承nn.Module，无state_dict()）"""
        state = {}
        if hasattr(self.agent, 'high_policy') and hasattr(self.agent.high_policy, 'state_dict'):
            state['high_policy'] = self.agent.high_policy.state_dict()
        if hasattr(self.agent, 'target_high_policy') and hasattr(self.agent.target_high_policy, 'state_dict'):
            state['target_high_policy'] = self.agent.target_high_policy.state_dict()
        if hasattr(self.agent, 'low_policy') and hasattr(self.agent.low_policy, 'state_dict'):
            state['low_policy'] = self.agent.low_policy.state_dict()
        if hasattr(self.agent, 'target_low_policy') and hasattr(self.agent.target_low_policy, 'state_dict'):
            state['target_low_policy'] = self.agent.target_low_policy.state_dict()
        if hasattr(self.agent, 'encoder') and hasattr(self.agent.encoder, 'state_dict'):
            state['encoder'] = self.agent.encoder.state_dict()
        if hasattr(self.agent, 'optimizer_high'):
            try:
                state['optimizer_high'] = self.agent.optimizer_high.state_dict()
            except Exception:
                pass
        if hasattr(self.agent, 'optimizer_low'):
            try:
                state['optimizer_low'] = self.agent.optimizer_low.state_dict()
            except Exception:
                pass
        return state if state else None

    def _save_checkpoint(self, episode):
        save_path = self.output_dir / f"checkpoint_ep{episode}.pth"
        try:
            torch.save({
                'episode': episode,
                'agent_state': self._get_agent_state(),
                'config': self.cfg,
                'stats': self.stats
            }, save_path)
        except Exception:
            pass

    def _save_final_model(self, episode):
        final_path = self.output_dir / "final_model.pth"
        try:
            torch.save({
                'episode': episode,
                'agent_state': self._get_agent_state(),
                'config': self.cfg,
                'stats': self.stats
            }, final_path)
            logger.info(f"💾 模型已保存: {final_path}")
        except Exception as e:
            logger.warning(f"⚠️ 保存失败: {e}")