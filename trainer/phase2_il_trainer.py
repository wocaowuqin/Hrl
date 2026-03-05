#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 Imitation Learning Trainer (HRL 适配版)
====================================================
功能升级：
1. ✅ 支持 HRLAgent 双策略架构 (High + Low)
2. ✅ 自动从专家路径中提取 Subgoal Index (High-Level Label)
3. ✅ 双重 Loss 联合训练 (High-Level 预测目标索引 + Low-Level 预测路径)
====================================================
"""

import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch
from pathlib import Path
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import platform
from torch_geometric.loader import DataLoader as PyGDataLoader
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return False


class ExpertDataset(Dataset):
    def __init__(self, expert_data_path: str):
        self.samples = []
        self._load_and_convert(expert_data_path)

    def _load_and_convert(self, data_path: str):
        logger.info(f"📂 加载专家数据: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        if isinstance(raw_data, dict):
            transitions = raw_data.get('success', raw_data.get('data', []))
        elif isinstance(raw_data, list):
            transitions = raw_data
        else:
            raise ValueError(f"未知的数据格式: {type(raw_data)}")

        if len(transitions) == 0:
            logger.error("❌ 没有可用的训练数据！")
            return

        converted = 0
        skipped = 0

        for i, trans in enumerate(transitions):
            try:
                # 检查是否包含 path 信息 (这是 HRL 训练必需的)
                action_data = trans.get('action')
                if isinstance(action_data, dict) and 'path' in action_data:
                    converted_samples = self._convert_path_to_steps(trans)
                    self.samples.extend(converted_samples)
                    converted += len(converted_samples)
                else:
                    # 如果只是简单的 state->action 对，无法推断 subgoal，跳过
                    skipped += 1
            except Exception as e:
                skipped += 1

        logger.info(f"✅ 数据转换完成:")
        logger.info(f"  - 生成样本数: {converted} (Step级别)")
        logger.info(f"  - 跳过样本数: {skipped} (格式不符)")
        logger.info(f"  - 总训练样本: {len(self.samples)}")

    def _convert_path_to_steps(self, trans: Dict) -> List[Dict]:
        """
        将一条完整路径拆解为多个 step 级训练样本。

        Phase1 新格式：action 里直接存了 high_label，无需重新推断。
        兼容旧格式：action 里只有 path，则从 dest_list 推断。
        """
        action = trans['action']
        path = action['path']
        req  = trans.get('request', {})

        if not path or len(path) < 2:
            return []

        # ── 优先读 Phase1 直接存的 high_label ────────────────
        if 'high_label' in action:
            high_action_idx = int(action['high_label'])
        else:
            # 旧格式兼容：从 dest_list 推断
            subgoal_node = int(path[-1])
            if subgoal_node >= 28:
                subgoal_node %= 28
            dest_list = req.get('dest', [])
            if subgoal_node in dest_list:
                high_action_idx = dest_list.index(subgoal_node)
                if high_action_idx >= 10:
                    high_action_idx = 0
            else:
                high_action_idx = 0

        # ── 生成每一步的样本 ──────────────────────────────────
        steps = []
        state = trans.get('state')
        for step_idx in range(len(path) - 1):
            curr_node = int(path[step_idx])
            next_node = int(path[step_idx + 1])
            if curr_node >= 28: curr_node %= 28
            if next_node >= 28: next_node %= 28

            steps.append({
                'state':      state,
                'high_label': high_action_idx,
                'low_label':  next_node,
            })

        return steps

    def __len__(self):
        return len(self.samples)

    # 修改 phase2_il_trainer.py 中的 ExpertDataset.__getitem__
    def __getitem__(self, index):
        sample = self.samples[index]
        state = sample['state']

        # 🔥 [核心修复] 如果旧的专家数据里没存 mask，实时补一个全 1 的掩码防止崩溃
        # 或者如果你能访问 env，可以调用 env.get_low_level_action_mask()
        if not hasattr(state, 'action_mask'):
            # 这里的 28 是你的动作空间维度（根据日志提示）
            state.action_mask = torch.ones((1, 28), dtype=torch.float32)

        return {
            'state': state,
            'high_label': torch.tensor(sample['high_label'], dtype=torch.long),
            'low_label': torch.tensor(sample['low_label'], dtype=torch.long)
        }


class Phase2ILTrainer:
    def __init__(self, env, agent, expert_data_path: str, output_dir: str, config: dict):
        self.env = env
        self.agent = agent
        self.cfg = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loss_high_sum = 0.0
        self._loss_low_sum = 0.0
        self._loss_count = 0
        # 训练参数
        phase2_cfg = config.get('phase2', {})
        self.epochs = phase2_cfg.get('epochs', 200)
        self.batch_size = phase2_cfg.get('batch_size', 64)
        self.validation_split = phase2_cfg.get('validation_split', 0.1)
        self.device = agent.device

        # 🔥 [关键] 检测 Agent 类型并获取 High/Low Policy
        self.is_hrl = hasattr(agent, 'high_policy') and hasattr(agent, 'low_policy')

        if self.is_hrl:
            logger.info("✅ Phase 2: 检测到 HRL Agent，准备进行双层策略训练")
            self.model_high = agent.high_policy
            self.model_low = agent.low_policy
            self.optimizer_high = agent.optimizer_high
            self.optimizer_low = agent.optimizer_low

            # 🔥 [优化] 添加学习率调度器（移除verbose参数以兼容旧版PyTorch）
            self.scheduler_high = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_high, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )
            self.scheduler_low = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_low, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )
        else:
            logger.warning("⚠️ Phase 2: 检测到旧版 Agent，仅训练 PolicyNet")
            self.model = agent.policy_net
            self.optimizer = agent.optimizer
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        # 数据加载
        self.num_workers = 0 if platform.system() == 'Windows' else 4
        self._prepare_data(expert_data_path)

        # 🔥 修复: Phase2 时 agent.encoder=None → _get_graph_embedding 返回 zeros(1,dim)
        # 为 agent 创建并赋予一个可训练的 GNN encoder，用于将图批次正确映射到 (B, state_dim)
        node_feat_dim = config.get('node_feat_dim',   17)
        state_dim     = config.get('hrl', {}).get('state_dim', 128)

        if agent.encoder is None:
            import torch.nn.functional as F
            from torch_geometric.nn import global_mean_pool, SAGEConv

            class _SimpleGNNEncoder(nn.Module):
                def __init__(self, in_d, hid_d, out_d):
                    super().__init__()
                    self.conv1 = SAGEConv(in_d,  hid_d)
                    self.conv2 = SAGEConv(hid_d, hid_d)
                    self.proj  = nn.Linear(hid_d, out_d)

                def forward(self, x, edge_index, batch=None):
                    x = F.relu(self.conv1(x, edge_index))
                    x = F.relu(self.conv2(x, edge_index))
                    if batch is None:
                        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                    return self.proj(global_mean_pool(x, batch))

            gnn_enc = _SimpleGNNEncoder(node_feat_dim, state_dim, state_dim).to(self.device)
            agent.encoder = gnn_enc
            logger.info(f"✅ Phase2: 创建内嵌 GNNEncoder ({node_feat_dim}->{state_dim}->{state_dim})")
            if self.is_hrl:
                self.optimizer_high.add_param_group({'params': gnn_enc.parameters()})
                logger.info("   📎 encoder 参数已加入 optimizer_high")

        # 早停
        self.early_stopping = EarlyStopping(patience=20)

    def _prepare_data(self, data_path):
        full_dataset = ExpertDataset(data_path)
        if len(full_dataset) == 0:
            self.train_loader = None
            return

        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate_fn,
            drop_last=True  # 避免最后一个 Batch 只有 1 个样本导致 BatchNorm 报错
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn,
            drop_last=True
        )

    def _collate_fn(self, batch):
        states = []
        high_labels = []
        low_labels = []

        for item in batch:
            state = item.get('state')
            if state is None: continue

            states.append(state)
            high_labels.append(item['high_label'])
            low_labels.append(item['low_label'])

        if not states: return None

        graph_batch = Batch.from_data_list(states)
        high_labels = torch.tensor(high_labels, dtype=torch.long)
        low_labels = torch.tensor(low_labels, dtype=torch.long)

        return graph_batch, high_labels, low_labels

    def run(self):
        """主运行循环（修复版：加入验证、早停、LR调度）"""
        if not self.train_loader:
            logger.error("❌ 数据未就绪，停止训练")
            return

        logger.info("🚀 开始 Phase 2 模仿学习 (HRL Mode)...")
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)

            # ── 验证 ─────────────────────────────────────────
            val_loss = self._validate_epoch(epoch)

            # ── LR 调度（根据验证loss降lr）────────────────────
            if self.is_hrl:
                self.scheduler_high.step(val_loss)
                self.scheduler_low.step(val_loss)
                cur_lr = self.optimizer_high.param_groups[0]['lr']
            else:
                cur_lr = self.optimizer.param_groups[0]['lr']

            # ── 日志 ──────────────────────────────────────────
            if epoch % 10 == 0:
                avg_h = self._loss_high_sum / max(1, self._loss_count)
                avg_l = self._loss_low_sum / max(1, self._loss_count)
                logger.info(
                    f"Epoch {epoch:>4} | TrainL={train_loss:.4f} | ValL={val_loss:.4f} | "
                    f"High={avg_h:.4f} Low={avg_l:.4f} | lr={cur_lr:.2e}"
                )

            # ── 保存最优模型 ───────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"  💾 新最优验证Loss={val_loss:.4f}，已保存 il_model_best.pth")

            if epoch % 50 == 0:
                self._save_checkpoint(epoch)

            # ── 早停 ──────────────────────────────────────────
            if self.early_stopping(val_loss):
                logger.info(f"⏹️  早停触发 (epoch={epoch}, best_val={best_val_loss:.4f})")
                break

        # 保存最终模型
        self._save_checkpoint("final")
        logger.info(f"✅ Phase 2 完成 | 最佳验证Loss={best_val_loss:.4f}")
        logger.info("   💡 建议Phase3加载 il_model_best.pth 而非 il_model_final.pth")

    def _train_epoch(self, epoch):
        """
        🔥 [IL V3.1 终极自愈版]
        1. 修复 Encoder 属性缺失：改用 Agent 封装的 _get_graph_embedding 接口
        2. 修复掩码缺失：加入 hasattr 检查
        3. 强化 Loss：保留 15.0 倍非法动作惩罚
        """
        self.model_high.train()
        self.model_low.train()

        # 初始化统计变量 (解决 _loss_count 报错)
        self._loss_high_sum = 0.0
        self._loss_low_sum = 0.0
        self._loss_count = 0
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_data in pbar:
            # 1. 自动适配 dict/tuple 数据解包
            if isinstance(batch_data, dict):
                states = batch_data['state'].to(self.device)
                high_labels = batch_data['high_label'].to(self.device)
                low_labels = batch_data['low_label'].to(self.device)
            else:
                states, high_labels, low_labels = batch_data
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)

            self.optimizer_high.zero_grad()
            self.optimizer_low.zero_grad()

            # 2. 🔥【核心修复】调用 Agent 内部稳定的嵌入接口
            # agent._get_graph_embedding 内部处理了对 self.encoder 的调用和 None 检查
            graph_emb = self.agent._get_graph_embedding(states)

            # 3. 前向预测
            high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)
            low_logits, _ = self.model_low(graph_emb, subgoal_emb)

            # 4. 动作掩码处理 (支持旧数据兼容)
            if hasattr(states, 'action_mask'):
                action_masks = states.action_mask.float()
            else:
                # 如果 states 里没有掩码，使用全 1 掩码（不惩罚但保证逻辑运行）
                action_masks = torch.ones((states.num_graphs, low_logits.size(-1)), device=self.device)

            # 5. 损失计算
            loss_high = self.criterion(high_logits, high_labels)
            loss_low_bc = self.criterion(low_logits, low_labels)

            # 非法动作抑制：惩罚模型在 Masked 为 0 的节点上分配的概率
            low_probs = torch.softmax(low_logits, dim=-1)
            illegal_penalty = (low_probs * (1.0 - action_masks)).sum(dim=-1).mean()

            # 综合损失 (保留 15.0x 非法惩罚权重)
            loss = loss_high * 0.5 + loss_low_bc + 15.0 * illegal_penalty

            # 6. 反向传播与优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_high.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_low.parameters(), 1.0)
            self.optimizer_high.step()
            self.optimizer_low.step()

            # 记录统计
            self._loss_high_sum += loss_high.item()
            self._loss_low_sum += loss_low_bc.item()
            self._loss_count += 1
            total_loss += loss.item()

            pbar.set_postfix({'L': f"{loss.item():.3f}", 'P': f"{illegal_penalty.item():.3f}"})

        return total_loss / max(1, self._loss_count)
    def _validate_epoch(self, epoch):
        # 🔥 [调试] 打印分离的Loss
        if hasattr(self, '_loss_count') and self._loss_count > 0:
            avg_high = self._loss_high_sum / self._loss_count
            avg_low = self._loss_low_sum / self._loss_count
            if epoch % 10 == 0:  # 每10个epoch打印
                logger.info(f"   📊 分离Loss: High={avg_high:.4f}, Low={avg_low:.4f}")
            # 重置
            self._loss_high_sum = 0
            self._loss_low_sum = 0
            self._loss_count = 0

        total_loss = 0
        count = 0

        if self.is_hrl:
            self.model_high.eval()
            self.model_low.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                if not batch: continue
                states, high_labels, low_labels = batch
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)

                if self.is_hrl:
                    if self.agent.encoder:
                        graph_emb = self.agent.encoder(states.x, states.edge_index, states.batch)
                    else:
                        graph_emb = self.agent._get_graph_embedding(states)

                    high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)
                    low_logits, _ = self.model_low(graph_emb, subgoal_emb)

                    loss = self.criterion(high_logits, high_labels) * 0.5 + \
                           self.criterion(low_logits, low_labels)
                else:
                    loss = torch.tensor(0.0)

                total_loss += loss.item()
                count += 1

        if self.is_hrl:
            self.model_high.train()
            self.model_low.train()

        return total_loss / max(1, count)

    def _save_checkpoint(self, tag):
        path = self.output_dir / f"il_model_{tag}.pth"

        save_dict = {
            'config': self.cfg,
        }

        if self.is_hrl:
            save_dict.update({
                'high_policy': self.model_high.state_dict(),
                'low_policy': self.model_low.state_dict(),
                'optimizer_high': self.optimizer_high.state_dict(),
                'optimizer_low': self.optimizer_low.state_dict(),
            })
        else:
            save_dict['model_state_dict'] = self.model.state_dict()

        torch.save(save_dict, path)
        logger.info(f"💾 模型已保存: {path}")