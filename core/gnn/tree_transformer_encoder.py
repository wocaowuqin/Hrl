import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

class TreeTransformerEncoder(nn.Module):
    def __init__(self, node_dim=21, edge_dim=5, hidden_dim=128, num_heads=4, req_dim=0):
        """
        Args:
            req_dim: 请求特征维度。>0 时启用请求融合（推荐传入 bw+cpu+mem 等请求信息）。
                     默认0保持向后兼容。
        """
        super().__init__()
        self.req_dim = req_dim

        self.topo_input_norm = nn.LayerNorm(node_dim)
        self.tree_input_norm = nn.LayerNorm(node_dim)

        self.topo_transformer1 = TransformerConv(
            in_channels=node_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            beta=True
        )
        self.topo_transformer2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            beta=True
        )

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

        self.dest_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 🆕 请求特征融合：将 bw/cpu/mem 需求融入节点表示
        # 让 Agent 明确知道当前请求需要多少资源，而不是只靠节点剩余量间接推断
        if req_dim > 0:
            self.req_fc = nn.Linear(req_dim, hidden_dim)
            self.req_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.req_norm = nn.LayerNorm(hidden_dim)
        else:
            self.req_fc = None

        self.tree_bias = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5，初始均等融合

    def forward(self, x, edge_index, edge_attr=None, batch=None, tree_edge_index=None, dest_mask=None, req_vec=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        x_topo = self.topo_input_norm(x)

        topo_emb = self.topo_transformer1(x_topo, edge_index, edge_attr)
        topo_emb = torch.nn.functional.gelu(topo_emb)
        topo_emb = self.topo_transformer2(topo_emb, edge_index, edge_attr)

        if tree_edge_index is None or tree_edge_index.size(1) == 0:
            device = x.device
            nodes = torch.arange(x.size(0), device=device)
            tree_edge_index = torch.stack([nodes, nodes], dim=0)

        x_tree = self.tree_input_norm(x)

        tree_emb = self.tree_transformer1(x_tree, tree_edge_index)
        tree_emb = torch.nn.functional.gelu(tree_emb)
        tree_emb = self.tree_transformer2(tree_emb, tree_edge_index)

        gate = torch.sigmoid(self.tree_bias)
        fused_emb = gate * tree_emb + (1.0 - gate) * topo_emb
        fused_emb = self.fusion_norm(fused_emb)
        node_emb = self.fusion_mlp(fused_emb)

        # 🆕 请求特征融合：把 bw/cpu/mem 需求广播到每个节点
        if self.req_fc is not None and req_vec is not None:
            if req_vec.dim() == 1:
                req_vec = req_vec.unsqueeze(0)          # [1, req_dim]
            req_emb = self.req_fc(req_vec)              # [1 or B, hidden_dim]
            # 广播到每个节点：按 batch 索引展开，无 batch 时直接 expand
            if batch is not None:
                req_expanded = req_emb[batch]           # [N, hidden_dim]
            else:
                req_expanded = req_emb.expand(node_emb.size(0), -1)  # [N, hidden_dim]
            node_emb = self.req_norm(
                self.req_fusion(torch.cat([node_emb, req_expanded], dim=-1))
            )

        if dest_mask is not None and dest_mask.any():
            dest_pool = node_emb[dest_mask].mean(dim=0, keepdim=True).unsqueeze(0)
            query = node_emb.unsqueeze(0)

            attn_out, _ = self.dest_cross_attn(query=query, key=dest_pool, value=dest_pool)
            node_emb = node_emb + attn_out.squeeze(0)

        return node_emb

    def get_graph_embedding(self, x, edge_index, edge_attr=None, batch=None, tree_edge_index=None, dest_mask=None, req_vec=None):
        node_embeddings = self.forward(x, edge_index, edge_attr, batch, tree_edge_index, dest_mask, req_vec)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(node_embeddings, batch)