#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Level Policy (第一步修复版 - 修复网络架构与起点选择)
"""

import torch
import torch.nn as nn
import logging
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


def deep_get(cfg, keys, default=None):
    if cfg is None: return default
    if isinstance(cfg, dict):
        for k in keys:
            if k in cfg and cfg[k] is not None: return cfg[k]
        for v in cfg.values():
            if isinstance(v, dict):
                found = deep_get(v, keys, None)
                if found is not None: return found
        return default
    for k in keys:
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None: return val
    return default


class HighLevelPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()

        use_cuda = deep_get(config, ["use_cuda"], False)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.hidden_dim = deep_get(config, ["hidden_dim"], 128)
        self.goal_dim = deep_get(config, ["goal_dim"], 64)
        # 修复点：读取 num_nodes 确保输出维度为 28，而不是默认的 10
        self.num_goals = config.get('environment', {}).get('num_nodes', 28)
        self.gnn_output_dim = deep_get(config, ["gnn_output_dim"], self.hidden_dim)
        dropout = deep_get(config, ["dropout"], 0.1)

        # 1. State Projection (共享特征空间 - 用于Goal Selection)
        self.state_projection = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Goal Head (Q Network - 选终点)
        self.q_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_goals)
        )

        # 3. Start Head (Selector - 选起点)
        # 🔥 修复：输入维度是 hidden_dim * 2
        self.start_selector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        # 🔥 新增：起点专用投影网络（确保维度匹配）
        self.start_projection = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4. Subgoal Generator
        self.subgoal_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.goal_dim),
            nn.Tanh()
        )

        self.to(self.device)

    def forward(self, graph_emb, return_subgoal=True, return_value=False):
        z = self.state_projection(graph_emb)
        q_values = self.q_network(z)

        subgoal_emb = None
        if return_subgoal:
            subgoal_emb = self.subgoal_generator(z)
            if torch.isnan(subgoal_emb).any():
                subgoal_emb = torch.zeros_like(subgoal_emb)

        return q_values, subgoal_emb, None

    def select_goal(self, state_emb, valid_goals_mask, epsilon=0.1):
        """Epsilon-Greedy Goal Selection"""
        import numpy as np
        with torch.no_grad():
            q_values, goal_emb, _ = self.forward(state_emb, return_subgoal=True)

        if valid_goals_mask is not None:
            if isinstance(valid_goals_mask, np.ndarray):
                valid_goals_mask = torch.FloatTensor(valid_goals_mask).to(q_values.device)
            masked_q_values = q_values.clone()
            masked_q_values[valid_goals_mask == 0] = -1e9
        else:
            masked_q_values = q_values

        if np.random.rand() < epsilon:
            if valid_goals_mask is not None:
                valid_indices = torch.nonzero(valid_goals_mask[0] > 0, as_tuple=False).squeeze()
                if valid_indices.numel() == 0:
                    goal_idx = torch.tensor(0)
                elif valid_indices.numel() == 1:
                    goal_idx = valid_indices
                else:
                    goal_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
            else:
                goal_idx = torch.randint(0, q_values.size(1), (1,))
        else:
            goal_idx = torch.argmax(masked_q_values[0])

        return goal_idx, goal_emb

    def select_start_node(self, node_embeddings, target_emb, tree_mask, sample=True):
        """
        🔥 [修复版] 选择起点 (支持采样和梯度流)

        关键修复：
        1. ✅ 使用专门的投影网络处理节点嵌入
        2. ✅ 确保target_emb维度正确
        3. ✅ 修复mask处理逻辑
        """
        # 1. 统一投影到Hidden Space
        # 🔥 使用新增的 start_projection
        node_proj = self.start_projection(node_embeddings)

        num_nodes = node_proj.size(0)

        # 2. 确保target_emb维度正确 [1, hidden_dim] -> [num_nodes, hidden_dim]
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)

        # 🔥 关键修复：如果target_emb维度不对，进行修正
        if target_emb.size(1) != self.hidden_dim:
            # 尝试投影到正确维度
            if hasattr(self, 'state_projection'):
                target_emb = self.state_projection(target_emb)
            else:
                # 使用线性层修正 (Lazy initialization)
                if not hasattr(self, 'target_proj_fix'):
                    self.target_proj_fix = nn.Linear(target_emb.size(1), self.hidden_dim).to(target_emb.device)
                target_emb = self.target_proj_fix(target_emb)

        # 扩展到所有节点
        target_expanded = target_emb.expand(num_nodes, -1)

        # 3. 拼接 [Node, Target] -> [num_nodes, hidden_dim * 2]
        combined = torch.cat([node_proj, target_expanded], dim=1)

        # 4. 计算分数
        scores = self.start_selector(combined).squeeze(-1)  # [num_nodes]

        # 5. Mask处理
        if tree_mask is not None:
            # 确保mask维度正确
            if tree_mask.numel() == 1:
                # 单个值，应用到所有节点
                scores = scores * tree_mask.item()
            elif tree_mask.size(0) == num_nodes:
                scores = scores.masked_fill(tree_mask == 0, -1e9)

        # 6. 生成概率分布
        probs = torch.softmax(scores, dim=0)
        dist = Categorical(probs)

        # 7. 决策
        if sample:
            start_node = dist.sample()
            log_prob = dist.log_prob(start_node)
            return start_node.item(), log_prob
        else:
            start_node = torch.argmax(probs)
            return start_node.item(), None