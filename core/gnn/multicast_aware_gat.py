import torch
import torch.nn as nn
from core.gnn.shared_encoder import SharedEncoder


class MulticastAwareGAT(nn.Module):
    """
    修复后的 MulticastAwareGAT (适配 Vectorized Wrapper)
    """

    def __init__(self,
                 node_feat_dim,
                 edge_feat_dim,
                 request_dim,
                 hidden_dim,
                 action_dim=None,
                 num_layers=3,
                 num_heads=4,
                 dropout=0.0,
                 **kwargs):
        super().__init__()

        # 1. 实例化共享 Encoder
        self.encoder = SharedEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            request_feat_dim=request_dim,  # 注意参数名对齐
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=num_heads
        )

        self.hidden_dim = hidden_dim
        self.output_layer = None

    def forward(self, x, edge_index, edge_attr, req_vec, batch=None, dest_indices=None, **kwargs):
        """
        Args:
            x: [N, F_node]
            edge_index: [2, E]
            edge_attr: [E, F_edge]
            req_vec: [Batch, F_req] 或 [F_req]
            batch: [N]
            dest_indices: (Wrapper 传进来的，SharedEncoder 暂时不用，用 **kwargs 接收防止报错)
        """

        # 1. 调用 Encoder 获取节点嵌入
        # 注意：这里 SharedEncoder 内部已经做了 Request Fusion
        # z 的形状: [N, hidden_dim]
        z = self.encoder(x, edge_index, edge_attr=edge_attr, req_vec=req_vec, batch=batch)

        # 2. 🔥 关键修复：返回 tuple 以匹配 Wrapper 的解包 (node_embeddings, _, _)
        # 后两个返回值预留给可能的 edge_weights 或 attention_weights，目前给 None
        return z, None, None