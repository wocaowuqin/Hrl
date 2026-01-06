#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Level Policy (重构版)
职责：从未连接的目标节点中选择下一个要连接的节点（subgoal selection）
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def deep_get(cfg, keys, default=None):
    """从 config 的多个可能路径中读取值"""
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        for v in cfg.values():
            if isinstance(v, dict):
                found = deep_get(v, keys, None)
                if found is not None:
                    return found
        return default

    for k in keys:
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None:
                return val

    return default


class HighLevelPolicy(nn.Module):
    """
    High-Level Policy Network (重构版)

    职责：
    1. 从未连接的目标节点中选择下一个要连接的节点
    2. 生成subgoal embedding供Low-Level使用

    输入：GNN编码后的图嵌入
    输出：
        - Q值（用于目标选择）
        - Subgoal embedding（给Low-Level）
    """

    def __init__(self, config):
        super().__init__()

        # 设备配置
        use_cuda = deep_get(config, ["use_cuda"], False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        # 维度配置
        self.hidden_dim = deep_get(config, ["hidden_dim"], 128)
        self.goal_dim = deep_get(config, ["goal_dim"], 64)

        # 从环境配置读取目标数量
        env_cfg = config.get('environment', {})
        self.num_goals = env_cfg.get('nb_high_level_goals', 10)

        # GNN输出维度
        self.gnn_output_dim = deep_get(config, ["gnn_output_dim"], self.hidden_dim)

        dropout = deep_get(config, ["dropout"], 0.1)

        logger.info(f"HighLevelPolicy配置:")
        logger.info(f"  GNN输出维度: {self.gnn_output_dim}")
        logger.info(f"  Goal维度: {self.goal_dim}")
        logger.info(f"  目标数量: {self.num_goals}")

        # ============================================
        # 1. State Projection（状态投影）
        # ============================================
        self.state_projection = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============================================
        # 2. Q Network（目标选择）
        # ============================================
        self.q_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_goals)
        )

        # ============================================
        # 3. Subgoal Generator（生成goal embedding）
        # ============================================
        self.subgoal_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.goal_dim),
            nn.Tanh()  # 归一化到[-1, 1]
        )

        # ============================================
        # 4. Value Network（状态价值评估）
        # ============================================
        self.value_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        self.to(self.device)

        logger.info(f"✅ HighLevelPolicy初始化完成，设备: {self.device}")

    def forward(
            self,
            graph_emb: torch.Tensor,
            return_subgoal: bool = True,
            return_value: bool = False
    ) -> tuple:
        """
        前向传播

        Args:
            graph_emb: 图嵌入 [batch, gnn_output_dim]
            return_subgoal: 是否返回subgoal embedding
            return_value: 是否返回value

        Returns:
            q_values: 目标Q值 [batch, num_goals]
            subgoal_emb: 子目标嵌入 [batch, goal_dim] (可选)
            value: 状态价值 [batch, 1] (可选)
        """
        # 投影状态
        z = self.state_projection(graph_emb)

        # Q值（用于选择目标）
        q_values = self.q_network(z)

        # 可选返回
        subgoal_emb = None
        value = None

        if return_subgoal:
            subgoal_emb = self.subgoal_generator(z)

            # 处理NaN
            if torch.isnan(subgoal_emb).any():
                logger.warning("⚠️ Subgoal包含NaN，使用零向量")
                subgoal_emb = torch.zeros_like(subgoal_emb)

        if return_value:
            value = self.value_network(z)

        return q_values, subgoal_emb, value

    def select_goal(
            self,
            graph_emb: torch.Tensor,
            valid_goals: Optional[torch.Tensor] = None,
            epsilon: float = 0.0
    ) -> tuple:
        """
        选择目标（epsilon-greedy）

        Args:
            graph_emb: 图嵌入 [batch, gnn_output_dim]
            valid_goals: 有效目标mask [batch, num_goals]
            epsilon: 探索率

        Returns:
            goal_idx: 选择的目标索引 [batch]
            subgoal_emb: 对应的subgoal embedding [batch, goal_dim]
        """
        batch_size = graph_emb.size(0)

        # 前向传播
        q_values, subgoal_emb, _ = self.forward(
            graph_emb,
            return_subgoal=True,
            return_value=False
        )

        # 应用mask
        if valid_goals is not None:
            q_values = q_values.masked_fill(valid_goals == 0, float('-inf'))

        # Epsilon-greedy
        if self.training and torch.rand(1).item() < epsilon:
            # 随机选择（从有效目标中）
            if valid_goals is not None:
                valid_indices = (valid_goals > 0).nonzero(as_tuple=True)[1]
                if len(valid_indices) > 0:
                    # 每个batch随机选一个
                    goal_idx = valid_indices[
                        torch.randint(0, len(valid_indices), (batch_size,))
                    ]
                else:
                    goal_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            else:
                goal_idx = torch.randint(0, self.num_goals, (batch_size,), device=self.device)
        else:
            # Greedy选择
            goal_idx = torch.argmax(q_values, dim=1)

        return goal_idx, subgoal_emb


# ============================================
# 向后兼容：保留旧接口
# ============================================

class HierarchicalPolicy(nn.Module):
    """
    向后兼容的HierarchicalPolicy

    这是为了不破坏现有代码而保留的接口
    实际上会调用重构后的HighLevelPolicy
    """

    def __init__(self, config):
        super().__init__()

        logger.warning("⚠️ HierarchicalPolicy已弃用，请使用HighLevelPolicy")
        logger.warning("   当前使用兼容模式，功能可能受限")

        # 使用新的HighLevelPolicy
        self.high_policy = HighLevelPolicy(config)

        # 复制属性以保持兼容
        self.device = self.high_policy.device
        self.gnn_output_dim = self.high_policy.gnn_output_dim
        self.num_goals = self.high_policy.num_goals

    def forward(
            self,
            graph_emb: torch.Tensor,
            goal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        兼容旧接口

        Args:
            graph_emb: 图嵌入 [batch, gnn_output_dim]
            goal_mask: 目标mask [batch, num_goals]

        Returns:
            logits: 动作logits [batch, num_goals]
        """
        # 调用新接口
        q_values, _, _ = self.high_policy(graph_emb, return_subgoal=False)

        # 应用mask
        if goal_mask is not None:
            q_values = q_values.masked_fill(goal_mask == 0, float('-inf'))

        return q_values