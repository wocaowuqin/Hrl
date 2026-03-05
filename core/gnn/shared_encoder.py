"""
core/gnn/shared_encoder.py
GNN 共享编码器 - V2.0 动态特征强化版
核心改进：
1. 显式分离静态拓扑特征和动态状态特征
2. 动态特征独立处理链路（MLP + 非线性）
3. 强力融合确保动态信号能主导决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import GATv2Conv

logger = logging.getLogger(__name__)

class SharedEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SharedEncoder, self).__init__()

        # ====================================================
        # 1. 万能参数解析 (兼容 Config对象 / 字典 / 位置参数)
        # ====================================================

        # 默认配置
        node_feat_dim = 20          # [SDG-HRL] 17→20（+vnf_depth, is_dc, progress_ratio）
        edge_feat_dim = 5
        request_dim = 24
        hidden_dim = 128
        self.num_layers = kwargs.get('num_layers', 2)

        # [SDG-HRL] 动态特征维度：原3维 → 新6维
        # [tree_mask, connected_mask, is_target, vnf_depth, is_dc, progress_ratio]
        self.num_dynamic_features = kwargs.get('num_dynamic_features', 6)

        # 情况 A: 传入了一个 config 对象或字典
        if len(args) == 1 and (isinstance(args[0], dict) or hasattr(args[0], 'get') or hasattr(args[0], 'gnn')):
            cfg = args[0]
            def get_cfg(key, default):
                if isinstance(cfg, dict): return cfg.get(key, default)
                val = getattr(cfg, key, None)
                if val is not None: return val
                if hasattr(cfg, 'gnn'):
                    gnn = getattr(cfg, 'gnn')
                    return gnn.get(key, default) if isinstance(gnn, dict) else getattr(gnn, key, default)
                return default

            node_feat_dim = get_cfg('node_feat_dim', node_feat_dim)
            edge_feat_dim = get_cfg('edge_feat_dim', edge_feat_dim)
            request_dim = get_cfg('request_feat_dim', request_dim)
            hidden_dim = get_cfg('hidden_dim', hidden_dim)
            self.num_dynamic_features = get_cfg('num_dynamic_features', self.num_dynamic_features)

        # 情况 B: 位置参数
        elif len(args) >= 3:
            node_feat_dim = args[0]
            edge_feat_dim = args[1]
            request_dim = args[2]
            if len(args) > 3: hidden_dim = args[3]

        # 情况 C: 关键字参数覆盖
        node_feat_dim = kwargs.get('node_feat_dim', node_feat_dim)
        edge_feat_dim = kwargs.get('edge_feat_dim', edge_feat_dim)
        request_dim = kwargs.get('request_feat_dim', request_dim)
        hidden_dim = kwargs.get('hidden_dim', hidden_dim)

        # 保存期望维度
        self.node_feat_dim = int(node_feat_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.request_dim = int(request_dim)
        self.hidden_dim = int(hidden_dim)

        # 🔥 计算静态特征维度
        self.static_feat_dim = self.node_feat_dim - self.num_dynamic_features

        logger.info(f"🔍 [SharedEncoder SDG-HRL] Init:")
        logger.info(f"   Total Node Feat: {self.node_feat_dim} (static={self.static_feat_dim}, dynamic={self.num_dynamic_features})")
        logger.info(f"   Dynamic dims: [tree_mask, connected_mask, is_target, vnf_depth, is_dc, progress_ratio]")
        logger.info(f"   Edge: {self.edge_feat_dim} [bw_rem, bw_use, hop, is_tree_edge, reserved]")
        logger.info(f"   Request: {self.request_dim}, Hidden: {self.hidden_dim}")
        logger.info(f"   Tree-aware Bias: learnable scalar (init=0.5)")

        # ====================================================
        # 2. 网络构建 - 双流架构
        # ====================================================

        # 🔵 静态流：拓扑结构感知（GAT）
        self.conv1 = GATv2Conv(
            in_channels=self.static_feat_dim,
            out_channels=self.hidden_dim,
            heads=4,
            edge_dim=self.edge_feat_dim if self.edge_feat_dim > 0 else None,
            concat=False
        )

        self.conv2 = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=4,
            edge_dim=self.edge_feat_dim if self.edge_feat_dim > 0 else None,
            concat=False
        )

        # 🟢 动态流：状态感知（MLP + 强非线性）
        # 专门处理 [tree_mask, fork_mask, progress_ratio] 等动态特征
        self.state_fc1 = nn.Linear(self.num_dynamic_features, self.hidden_dim // 2)
        self.state_fc2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim)

        # 🔥 动态特征门控机制（让动态特征能"一票否决"静态偏好）
        self.gate_fc = nn.Linear(self.num_dynamic_features, self.hidden_dim)

        # 🟡 请求特征层
        if self.request_dim > 0:
            self.req_fc = nn.Linear(self.request_dim, self.hidden_dim)
        else:
            self.req_fc = None

        # 🔵 融合层（三路融合：静态 + 动态 + 请求）
        self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_dim = self.hidden_dim

        # ── [SDG-HRL] Tree-aware Attention Bias ─────────────────────────────
        # 可学习标量参数，调制树边的注意力权重
        # 初始化为0.5（轻微正偏置），训练中自适应调整
        # 理论动机：多播树构建倾向于沿已建立结构生长（inductive bias）
        self.tree_bias = nn.Parameter(torch.tensor(0.5))

        # 状态标记
        self._warned = set()
        self._step_count = 0

    def _fix_dim(self, tensor, expected_dim, name="tensor"):
        """自动修复维度不匹配"""
        if tensor is None: return None

        # 获取最后一个维度
        actual_dim = tensor.shape[-1]

        if actual_dim == expected_dim:
            return tensor

        if name not in self._warned:
            logger.warning(f"⚠️ [SharedEncoder] {name} dim mismatch! Expected {expected_dim}, got {actual_dim}. Auto-fixing...")
            self._warned.add(name)

        if actual_dim < expected_dim:
            # 填充
            padding_shape = list(tensor.shape)
            padding_shape[-1] = expected_dim - actual_dim
            padding = torch.zeros(padding_shape, device=tensor.device)
            return torch.cat([tensor, padding], dim=-1)
        else:
            # 截断
            return tensor[..., :expected_dim]

    def forward(self, x, edge_index, edge_attr=None, req_vec=None, batch=None):
        device = x.device
        self._step_count += 1

        # 1. 自动修复节点特征
        x = self._fix_dim(x, self.node_feat_dim, "node_feat")

        # ── [SDG-HRL] 2. 显式分离静态和动态特征 ─────────────────────────────
        # 静态流 (dim 0 ~ N-6): CPU/Mem/fit + padding
        # 动态流 (dim N-6 ~ N): tree_mask, connected_mask, is_target,
        #                        vnf_depth, is_dc, progress_ratio
        static_x  = x[:, :-self.num_dynamic_features]   # [N, 14]
        dynamic_x = x[:, -self.num_dynamic_features:]   # [N, 6]

        # ── [SDG-HRL] 3. 从动态特征中提取各语义子维度 ──────────────────────
        # dim 5 (index -1) = progress_ratio: 全局SFC完成度
        # dim 4 (index -2) = is_dc:          是否DC节点
        # dim 3 (index -3) = vnf_depth:      SFC链路位置编码
        progress_ratio = dynamic_x[:, -1:]              # [N, 1] ∈ [0,1]
        is_dc_feat     = dynamic_x[:, -2:-1]            # [N, 1] ∈ {0,1}
        vnf_depth_feat = dynamic_x[:, -3:-2]            # [N, 1] ∈ [0,1]
        is_completed   = (progress_ratio >= 0.99).float()

        # 4. 自动修复边缘特征
        if self.edge_feat_dim > 0:
            num_edges = edge_index.shape[1]
            if edge_attr is None:
                edge_attr = torch.zeros(num_edges, self.edge_feat_dim, device=device)
            else:
                edge_attr = self._fix_dim(edge_attr, self.edge_feat_dim, "edge_attr")
                if edge_attr.shape[0] != num_edges:
                    edge_attr = torch.zeros(num_edges, self.edge_feat_dim, device=device)        # ── [SDG-HRL] 5. Tree-aware Attention Bias ──────────────────────────
        # is_tree_edge = edge_attr[:, 3] ∈ {0,1}
        # 通过可学习参数 tree_bias 对树边的注意力权重施加归纳偏置
        # 物理意义：鼓励GNN优先沿已建立的多播树结构传播信息
        #   tree_bias > 0 → 树边信息权重更强（期望收敛后 bias > 0）
        #   tree_bias < 0 → 抑制树内传播（不期望，但允许学习）
        # 防御：经验回放里可能存有旧格式的 1D edge_attr，需先修复为 2D
        if edge_attr.dim() == 1:
            # 1D → reshape 为 [E, 1]，再 pad 到标准5维
            edge_attr = edge_attr.unsqueeze(1)
        if edge_attr.shape[1] < self.edge_feat_dim:
            pad = torch.zeros(
                edge_attr.shape[0],
                self.edge_feat_dim - edge_attr.shape[1],
                device=edge_attr.device
            )
            edge_attr = torch.cat([edge_attr, pad], dim=1)

        biased_edge_attr = edge_attr.clone()
        if edge_attr.shape[1] >= 4:
            is_tree_edge = edge_attr[:, 3:4]
            # 路径A: 树边专属增强（is_tree_edge=1时才激活）
            tree_boost_conditional = self.tree_bias * is_tree_edge
            # 路径B: 全局偏置（0.01 × tree_bias 让梯度始终非零，不依赖is_tree_edge）
            tree_boost_global = self.tree_bias * 1.0
            biased_edge_attr = torch.cat([
                edge_attr[:, 0:1] + tree_boost_conditional + tree_boost_global,
                edge_attr[:, 1:]
            ], dim=1)

        # 🔵 6. 静态流：GATv2卷积（使用 biased edge_attr）
        try:
            static_emb = self.conv1(static_x, edge_index, edge_attr=biased_edge_attr)
            static_emb = F.relu(static_emb)
            static_emb = self.conv2(static_emb, edge_index, edge_attr=biased_edge_attr)
            static_emb = F.relu(static_emb)
        except RuntimeError as e:
            logger.error(f"❌ [SharedEncoder] GAT Forward Failed: {e}")
            raise e

        # 🟢 7. 动态流：MLP处理6维状态特征
        state_emb = F.relu(self.state_fc1(dynamic_x))
        state_emb = F.relu(self.state_fc2(state_emb))

        # ── [SDG-HRL] 8. 增强版门控融合 ──────────────────────────────────────
        # 门控由完整6维动态特征驱动（包含vnf_depth和is_dc信号）
        gate_weights = torch.sigmoid(self.gate_fc(dynamic_x))

        # 完成态抑制：VNF全部部署后压制拓扑惯性，迫使agent关注dest
        completion_inhibition = 1.0 - (is_completed * 0.8)
        gate_weights = gate_weights * completion_inhibition

        # 融合：树感知静态流 × 门控 + 动态流（完成态放大）
        modulated_static = static_emb * gate_weights
        node_emb = modulated_static + (state_emb * (1.0 + is_completed * 2.0))

        # 9. 请求特征处理（保持原有逻辑）
        if self.req_fc is not None:
            if req_vec is None:
                batch_size = 1 if batch is None else (batch.max().item() + 1)
                req_emb = torch.zeros(batch_size, self.hidden_dim, device=device)
            else:
                if req_vec.dim() == 1: req_vec = req_vec.unsqueeze(0)
                req_vec = self._fix_dim(req_vec, self.request_dim, "req_vec")
                req_emb = self.req_fc(req_vec)
        else:
            batch_size = 1 if batch is None else (batch.max().item() + 1)
            req_emb = torch.zeros(batch_size, self.hidden_dim, device=device)

        # 10. 智能扩展
        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=device)

        if req_emb.dim() == 1:
            req_expanded = req_emb.unsqueeze(0).expand(node_emb.size(0), -1)
        else:
            max_batch_idx = batch.max().item()
            if req_emb.size(0) <= max_batch_idx:
                req_expanded = req_emb[0].unsqueeze(0).expand(node_emb.size(0), -1)
            else:
                req_expanded = req_emb[batch]

# 11. 最终融合
        combined = torch.cat([node_emb, req_expanded], dim=-1)
        out = self.fusion(combined)

        # ── [SDG-HRL] tree_bias 直接调制输出，梯度路径最短 ─────────────────
        out = out * (1.0 + 0.1 * self.tree_bias)

        return out