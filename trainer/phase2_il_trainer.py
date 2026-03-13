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
    def __init__(self, patience: int = 30, min_delta: float = 0.0001):
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
        self.max_dest = 28  # 默认值，load后会更新
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
                action_data = trans.get('action')
                if isinstance(action_data, dict) and 'path' in action_data:
                    converted_samples = self._convert_path_to_steps(trans)
                    self.samples.extend(converted_samples)
                    converted += len(converted_samples)
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1

        logger.info(f"✅ 数据转换完成:")
        logger.info(f"  - 生成样本数: {converted} (Step级别)")
        logger.info(f"  - 跳过样本数: {skipped} (格式不符)")
        logger.info(f"  - 总训练样本: {len(self.samples)}")
        # 统计实际最大high_label值，确定高层输出维度
        if self.samples:
            self.max_dest = max(s['high_label'] for s in self.samples) + 1
            logger.info(f"  - 高层类别数(max_dest): {self.max_dest}")

    def _convert_path_to_steps(self, trans: Dict) -> List[Dict]:
        action = trans['action']
        path = action['path']
        req = trans.get('request', {})

        if not path or len(path) < 2:
            return []

        # high_label = 路径终点节点ID（子目标节点，0~27）
        if 'high_label' in action:
            # Phase1新格式：直接用subgoal_node作为high_label
            subgoal_node = int(action.get('subgoal_node', path[-1]))
            high_action_idx = subgoal_node
        else:
            high_action_idx = int(path[-1])
            if high_action_idx >= 28:
                high_action_idx %= 28

        steps = []
        state = trans.get('state')
        for step_idx in range(len(path) - 1):
            curr_node = int(path[step_idx])
            next_node = int(path[step_idx + 1])
            if curr_node >= 28: curr_node %= 28
            if next_node >= 28: next_node %= 28

            steps.append({
                'state': state,
                'high_label': high_action_idx,
                'low_label': next_node,
                'req': req,   # 请求特征，供_phase2_graph_embedding构建req_vec用
            })

        return steps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        state = sample['state']

        return {
            'state': state,
            'high_label': torch.tensor(sample['high_label'], dtype=torch.long),
            'low_label': torch.tensor(sample['low_label'], dtype=torch.long),
            'req': sample.get('req'),   # 透传请求特征
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

        phase2_cfg = config.get('phase2', {})
        self.epochs = phase2_cfg.get('epochs', 150)
        self.batch_size = phase2_cfg.get('batch_size', 64)
        self.validation_split = phase2_cfg.get('validation_split', 0.1)
        self.device = agent.device

        self.is_hrl = hasattr(agent, 'high_policy') and hasattr(agent, 'low_policy')

        if self.is_hrl:
            logger.info("✅ Phase 2: 检测到 HRL Agent，准备进行双层策略训练")
            self.model_high = agent.high_policy
            self.model_low = agent.low_policy

            il_lr = config.get('phase2', {}).get('lr', 3e-4)
            self.optimizer_high = torch.optim.Adam(
                agent.high_policy.parameters(), lr=il_lr
            )

            low_params = list(agent.low_policy.parameters())
            if hasattr(agent, 'encoder') and agent.encoder is not None:
                low_params += list(agent.encoder.parameters())
                agent.encoder.train()
                logger.info("   📎 [修复] TreeTransformerEncoder 参数已加入 optimizer_low")

            self.optimizer_low = torch.optim.Adam(
                low_params, lr=il_lr
            )
            logger.info(f"✅ Phase2: 独立创建优化器 lr={il_lr:.2e}")

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

        self.criterion = nn.CrossEntropyLoss()  # 高层loss，先占位

        self.num_workers = 0 if platform.system() == 'Windows' else 4
        self._prepare_data(expert_data_path)

        # 低层loss加权：平衡高频/低频节点（必须在_prepare_data之后）
        from collections import Counter
        n_nodes = env.n
        weights = torch.ones(n_nodes, dtype=torch.float)
        if getattr(self, 'train_loader', None) is not None:
            low_label_counts = Counter(
                s['low_label'] for s in self.train_loader.dataset.dataset.samples
            )
            total = sum(low_label_counts.values())
            for node_id, cnt in low_label_counts.items():
                if node_id < n_nodes:
                    weights[node_id] = total / (n_nodes * cnt)
            weights = weights / weights.sum() * n_nodes
            logger.info(f"✅ 低层损失加权完成，最大权重节点: "
                        f"{weights.argmax().item()} ({weights.max().item():.2f}x)")
        self.criterion_low = nn.CrossEntropyLoss(weight=weights.to(self.device))
        # 固定高层类别数，避免每batch动态切片导致梯度不稳定
        self.n_dest = 28
        logger.info(f"✅ Phase2: 高层固定类别数 n_dest={self.n_dest}")
        from core.gnn.tree_transformer_encoder import TreeTransformerEncoder
        from torch_geometric.nn import global_mean_pool

        node_feat_dim = config.get('node_feat_dim', 21)
        edge_feat_dim = config.get('edge_feat_dim', 5)
        hidden_dim = config.get('hrl', {}).get('hidden_dim', 128)
        num_heads = config.get('hrl', {}).get('num_heads', 4)

        if agent.encoder is None:
            gnn_enc = TreeTransformerEncoder(
                node_dim=node_feat_dim,
                edge_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(self.device)
            agent.encoder = gnn_enc
            logger.info(f"✅ Phase2: 使用 TreeTransformerEncoder "
                        f"(node={node_feat_dim}, edge={edge_feat_dim}, hidden={hidden_dim})")
            if self.is_hrl:
                # 🔧 [修复] encoder应与low_policy联合优化，与agent_base.py保持一致
                # 原来挂在optimizer_high会导致IL/RL切换时优化器绑定错乱
                self.optimizer_low.add_param_group({'params': gnn_enc.parameters()})
                logger.info("   📎 [修复] TreeTransformerEncoder 参数已加入 optimizer_low")

        n_nodes = env.n

        def _phase2_graph_embedding(pyg_batch, req_vec=None):
            enc = agent.encoder
            node_emb = enc(pyg_batch.x, pyg_batch.edge_index,
                           edge_attr=getattr(pyg_batch, 'edge_attr', None),
                           batch=getattr(pyg_batch, 'batch', None),
                           tree_edge_index=getattr(pyg_batch, 'tree_edge_index', None),
                           dest_mask=getattr(pyg_batch, 'dest_mask', None),
                           req_vec=req_vec)   # 🆕 传入请求特征，IL/RL行为一致

            graph_emb = global_mean_pool(node_emb, pyg_batch.batch)

            B = graph_emb.size(0)
            H = node_emb.size(-1)
            # 🔧 [修复] 用-1推断避免节点数硬编码导致的shape崩溃
            node_emb_3d = node_emb.view(B, -1, H)

            return graph_emb, node_emb_3d

        self._phase2_graph_embedding = _phase2_graph_embedding

        self.early_stopping = EarlyStopping(patience=30)

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
            drop_last=True
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
        reqs = []

        for item in batch:
            state = item.get('state')
            if state is None: continue

            states.append(state)
            high_labels.append(item['high_label'])
            low_labels.append(item['low_label'])
            reqs.append(item.get('req'))

        if not states: return None

        # ── tree_edge_index / dest_mask 无条件补齐 ────────────────────────
        # 当 batch 内所有样本的同一字段都是 None 时，PyG 的 Batch.from_data_list
        # 不会合并成 Tensor，而是留成 list[None]。encoder 收到 list[None] 后
        # 执行 tree_edge_index.size(1) 会抛 AttributeError。
        # 修复：无条件补齐，彻底消除 list[None] 的可能。
        #   - tree_edge_index 缺失 → self-loop（与 encoder 空树退化行为一致）
        #   - dest_mask 缺失 → 全 False（无目标感知，退化为普通节点特征）
        for s in states:
            if getattr(s, 'tree_edge_index', None) is None:
                nodes = torch.arange(s.x.size(0))
                s.tree_edge_index = torch.stack([nodes, nodes], dim=0)
            if getattr(s, 'dest_mask', None) is None:
                s.dest_mask = torch.zeros(s.x.size(0), dtype=torch.bool)

        graph_batch = Batch.from_data_list(states)
        high_labels = torch.tensor(high_labels, dtype=torch.long)
        low_labels  = torch.tensor(low_labels,  dtype=torch.long)

        # 构建 req_vec [B, 3]，字段缺失时填0
        req_vecs = []
        for r in reqs:
            if r is not None:
                bw  = float(r.get('bw_origin', r.get('bw', 0.0)))
                cpu = float(np.mean(r.get('cpu_origin', r.get('cpu', [0.0]))) if r.get('cpu_origin', r.get('cpu')) else 0.0)
                mem = float(np.mean(r.get('memory_origin', r.get('memory', [0.0]))) if r.get('memory_origin', r.get('memory')) else 0.0)
                req_vecs.append([bw, cpu, mem])
            else:
                req_vecs.append([0.0, 0.0, 0.0])
        req_tensor = torch.tensor(req_vecs, dtype=torch.float32)  # [B, 3]

        return graph_batch, high_labels, low_labels, req_tensor

    def run(self):
        if not self.train_loader:
            logger.error("❌ 数据未就绪，停止训练")
            return

        logger.info("🚀 开始 Phase 2 模仿学习 (HRL Mode)...")
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)

            val_loss = self._validate_epoch(epoch)

            if self.is_hrl:
                self.scheduler_high.step(val_loss)
                self.scheduler_low.step(val_loss)
                cur_lr = self.optimizer_high.param_groups[0]['lr']
            else:
                cur_lr = self.optimizer.param_groups[0]['lr']

            if epoch % 10 == 0:
                avg_h = self._loss_high_sum / max(1, self._loss_count)
                avg_l = self._loss_low_sum / max(1, self._loss_count)
                logger.info(
                    f"Epoch {epoch:>4} | TrainL={train_loss:.4f} | ValL={val_loss:.4f} | "
                    f"High={avg_h:.4f} Low={avg_l:.4f} | lr={cur_lr:.2e}"
                )

                self._loss_high_sum = 0
                self._loss_low_sum = 0
                self._loss_count = 0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"  💾 新最优验证Loss={val_loss:.4f}，已保存 il_model_best.pth")

            if epoch % 50 == 0:
                self._save_checkpoint(epoch)

            if self.early_stopping(val_loss):
                logger.info(f"⏹️  早停触发 (epoch={epoch}, best_val={best_val_loss:.4f})")
                break

        self._save_checkpoint("final")
        logger.info(f"✅ Phase 2 完成 | 最佳验证Loss={best_val_loss:.4f}")
        logger.info("   💡 建议Phase3加载 il_model_best.pth 而非 il_model_final.pth")

    def _train_epoch(self, epoch):
        self.model_high.train()
        self.model_low.train()

        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_data in pbar:
            if isinstance(batch_data, dict):
                states = batch_data['state'].to(self.device)
                high_labels = batch_data['high_label'].to(self.device)
                low_labels = batch_data['low_label'].to(self.device)
                req_tensor = batch_data.get('req_vec')
            else:
                states, high_labels, low_labels, req_tensor = batch_data
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)
            req_vec = req_tensor.to(self.device) if req_tensor is not None else None

            self.optimizer_high.zero_grad()
            self.optimizer_low.zero_grad()

            graph_emb, node_emb_3d = self._phase2_graph_embedding(states, req_vec=req_vec)

            high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)

            # ── action_mask：从 Batch 里取出，对齐到 low_logits 维度 ─────
            # action_mask 在 Batch 后形状 [B, 28]（各图 [1,28] cat 而成）
            # low_logits 维度 [B, action_dim]（action_dim 可能≠28）
            # 若维度不匹配：截断/右侧补0对齐，保证传入 model_low 的 mask 有效
            if hasattr(states, 'action_mask'):
                action_masks = states.action_mask.float().to(self.device)   # [B, 28]
                B_cur = node_emb_3d.size(0)
                action_dim = self.model_low.action_dim
                if action_masks.size(0) != B_cur:
                    action_masks = torch.ones(B_cur, action_dim, device=self.device)
                elif action_masks.size(1) != action_dim:
                    if action_masks.size(1) > action_dim:
                        action_masks = action_masks[:, :action_dim]
                    else:
                        pad = torch.zeros(B_cur, action_dim - action_masks.size(1), device=self.device)
                        action_masks = torch.cat([action_masks, pad], dim=1)
            else:
                action_masks = torch.ones(node_emb_3d.size(0), self.model_low.action_dim, device=self.device)

            # ── 传入 action_mask，非法节点被 -inf mask 后再算 loss ────────
            low_logits, _ = self.model_low(node_emb_3d, subgoal_emb, action_mask=action_masks)

            # ── 高层标签对齐：确保 high_labels < high_logits.size(1) ──────
            n_classes = high_logits.size(1)
            if high_labels.max() >= n_classes:
                high_labels = high_labels.clamp(0, n_classes - 1)
            loss_high = self.criterion(high_logits, high_labels)
            loss_low_bc = self.criterion_low(low_logits, low_labels)

            # illegal_penalty 依赖运行时 action_mask，Phase2专家数据里
            # action_mask 全为1（静态离线数据无法知道哪些动作非法），
            # 惩罚项永远为0，已移除。非法节点惩罚由 criterion_low 的频率加权和
            # model_low.forward 里的 -inf masking 共同实现。
            loss = loss_high * 0.5 + loss_low_bc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_high.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model_low.parameters(), 1.0)
            # encoder 梯度也需裁剪，防止 TransformerConv 注意力权重在早期爆炸
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.encoder.parameters(), 1.0)
            self.optimizer_high.step()
            self.optimizer_low.step()

            self._loss_high_sum += loss_high.item()
            self._loss_low_sum += loss_low_bc.item()
            self._loss_count += 1
            total_loss += loss.item()

            pbar.set_postfix({'L': f"{loss.item():.3f}", 'H': f"{loss_high.item():.3f}", 'Low': f"{loss_low_bc.item():.3f}"})

        return total_loss / max(1, len(self.train_loader))

    def _validate_epoch(self, epoch):
        total_loss = 0
        count = 0

        if self.is_hrl:
            self.model_high.eval()
            self.model_low.eval()
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                self.agent.encoder.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                if not batch: continue
                states, high_labels, low_labels, req_tensor = batch
                states = states.to(self.device)
                high_labels = high_labels.to(self.device)
                low_labels = low_labels.to(self.device)
                req_vec = req_tensor.to(self.device) if req_tensor is not None else None

                if self.is_hrl:
                    graph_emb, node_emb_3d = self._phase2_graph_embedding(states, req_vec=req_vec)

                    high_logits, subgoal_emb, _ = self.model_high(graph_emb, return_subgoal=True)

                    if hasattr(states, 'action_mask'):
                        val_action_masks = states.action_mask.float().to(self.device)
                    else:
                        val_action_masks = torch.ones(
                            node_emb_3d.size(0), self.model_low.action_dim, device=self.device)
                    low_logits, _ = self.model_low(node_emb_3d, subgoal_emb, action_mask=val_action_masks)

                    n_cls = high_logits.size(1)
                    if high_labels.max() >= n_cls:
                        high_labels = high_labels.clamp(0, n_cls - 1)
                    loss = self.criterion(high_logits, high_labels) * 0.5 + \
                           self.criterion_low(low_logits, low_labels)
                else:
                    loss = torch.tensor(0.0)

                total_loss += loss.item()
                count += 1

        if self.is_hrl:
            self.model_high.train()
            self.model_low.train()
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                self.agent.encoder.train()

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
                'n_goals':   self.model_high.num_goals,
                'n_actions': self.model_low.action_dim,
            })
            if hasattr(self.agent, 'encoder') and self.agent.encoder is not None:
                save_dict['encoder'] = self.agent.encoder.state_dict()
                logger.info("   📎 Encoder 权重已保存至 checkpoint")
        else:
            save_dict['model_state_dict'] = self.model.state_dict()

        torch.save(save_dict, path)
        logger.info(f"💾 模型已保存: {path}")