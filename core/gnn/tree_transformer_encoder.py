import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool


class TreeTransformerEncoder(nn.Module):
    """
    Tree-Aware Graph Transformer Encoder
    结合全局拓扑注意力与局部树结构注意力的顶级编码器
    """

    def __init__(self, node_dim=21, edge_dim=5, hidden_dim=128, num_heads=4):
        super().__init__()

        # 1. Topology Transformer Stream (拓扑流：感知全局物理网络)
        self.topo_transformer1 = TransformerConv(
            in_channels=node_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            beta=True  # 启用门控残差连接
        )
        self.topo_transformer2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            beta=True
        )

        # 2. Tree Attention Stream (树流：强化已被选入多播树的节点间的通信)
        self.tree_transformer1 = TransformerConv(
            in_channels=node_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            beta=True
        )
        self.tree_transformer2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            beta=True
        )

        # 3. Destination-Aware Cross Attention (目标制导交叉注意力)
        self.dest_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 4. 融合与降维层
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 结构自适应权重
        self.tree_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr=None, batch=None, tree_edge_index=None, dest_mask=None):
        # --- A. Topology Stream ---
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        topo_emb = self.topo_transformer1(x, edge_index, edge_attr)
        topo_emb = torch.nn.functional.gelu(topo_emb)
        topo_emb = self.topo_transformer2(topo_emb, edge_index, edge_attr)

        # --- B. Tree Stream ---
        if tree_edge_index is None or tree_edge_index.size(1) == 0:
            device = x.device
            nodes = torch.arange(x.size(0), device=device)
            tree_edge_index = torch.stack([nodes, nodes], dim=0)

        tree_emb = self.tree_transformer1(x, tree_edge_index)
        tree_emb = torch.nn.functional.gelu(tree_emb)
        tree_emb = self.tree_transformer2(tree_emb, tree_edge_index)

        # --- C. Dual Stream Fusion ---
        fused_emb = torch.cat([topo_emb, tree_emb * self.tree_bias], dim=-1)
        fused_emb = self.fusion_norm(fused_emb)
        node_emb = self.fusion_mlp(fused_emb)  # [N, hidden_dim]

        # --- D. Destination-Aware Pooling (Steiner Tree 启发式) ---
        if dest_mask is not None and dest_mask.any():
            dest_pool = node_emb[dest_mask].mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, hidden_dim]
            query = node_emb.unsqueeze(0)  # [1, N, hidden_dim]

            attn_out, _ = self.dest_cross_attn(query=query, key=dest_pool, value=dest_pool)
            node_emb = node_emb + attn_out.squeeze(0)  # 残差连接注入目标信息

        return node_emb

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None, tree_edge_index=None, dest_mask=None):
        node_embeddings = self.forward(x, edge_index, edge_attr, batch, tree_edge_index, dest_mask)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(node_embeddings, batch)