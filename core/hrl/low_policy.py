#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-Level Policy (重构版)
职责：给定subgoal，选择下一跳节点执行路径规划
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GoalConditionedLowLevelPolicy(nn.Module):
    """
    Goal-Conditioned Low-Level Policy (重构版)

    职责：
    1. 接收当前状态 + subgoal embedding
    2. 输出下一跳节点的Q值

    关键特点：
    - Goal-Conditioned（目标条件化）
    - 学习如何到达subgoal
    - 接收内在奖励（接近/远离subgoal）
    """

    def __init__(self, config):
        super().__init__()

        # 设备配置
        use_cuda = config.get('use_cuda', False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        # 维度配置
        self.state_dim = config.get('state_dim', 128)  # GNN encoder输出
        self.goal_dim = config.get('goal_dim', 64)  # Goal embedding维度
        self.hidden_dim = config.get('hidden_dim', 128)

        # 从环境配置读取动作数量
        env_cfg = config.get('environment', {})
        self.action_dim = env_cfg.get('nb_low_level_actions', 28)

        dropout = config.get('dropout', 0.1)

        logger.info(f"GoalConditionedLowLevelPolicy配置:")
        logger.info(f"  State dim: {self.state_dim}")
        logger.info(f"  Goal dim: {self.goal_dim}")
        logger.info(f"  Action dim: {self.action_dim}")

        # ============================================
        # 1. State Projection（状态投影）
        # ============================================
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============================================
        # 2. Goal Projection（目标投影）
        # ============================================
        self.goal_projection = nn.Sequential(
            nn.Linear(self.goal_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============================================
        # 3. [TA-HGRL] Goal-Conditioned Attention
        #    Goal(Query) × Nodes(Key,Value) → 目标制导注意力
        #    让subgoal主动"审视"全网节点，聚焦最优下一跳
        # ============================================
        self.goal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # ============================================
        # 4. Actor（Q Network）
        #    输入: cat[attn_context, goal_proj] → hidden*2
        # ============================================
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # ============================================
        # 5. Critic（Value Network）
        # ============================================
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

        self.to(self.device)

        logger.info(f"✅ GoalConditionedLowLevelPolicy初始化完成，设备: {self.device}")

    def forward(
            self,
            state_emb: torch.Tensor,
            goal_emb: Optional[torch.Tensor] = None,
            action_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        [TA-HGRL] Goal-Conditioned Attention Forward

        Args:
            state_emb: 状态嵌入 [batch, state_dim]  (全图均值或节点序列)
            goal_emb:  Goal embedding [batch, goal_dim]
            action_mask: [batch, action_dim]  1=可选 0=禁止

        Returns:
            logits: [batch, action_dim]
            value:  [batch, 1]
        """
        B = state_emb.size(0)

        # 1. 投影: Linear作用于最后维度，兼容 [B,H] 和 [B,N,H]
        state_proj = self.state_projection(state_emb)   # [B,H] 或 [B,N,H]
        goal_proj  = (self.goal_projection(goal_emb)
                      if goal_emb is not None
                      else torch.zeros(B, self.hidden_dim, device=state_emb.device))  # [B,H]

        # 2. [TA-HGRL Fix] Goal-Conditioned Cross-Attention
        #    Query = goal [B,1,H]
        #    Key/Value = 全网N个节点特征序列 [B,N,H]（或退化为[B,1,H]）
        query = goal_proj.unsqueeze(1)                           # [B, 1, H]
        kv    = (state_proj if state_proj.dim() == 3            # [B, N, H] ✓
                 else state_proj.unsqueeze(1))                   # [B, 1, H] 兜底
        attn_out, _ = self.goal_attention(query=query, key=kv, value=kv)
        attn_context = self.attn_norm(attn_out.squeeze(1) + goal_proj)  # 残差 [B,H]

        # 3. 拼接 → Actor/Critic
        fused  = torch.cat([attn_context, goal_proj], dim=-1)   # [B, H*2]
        logits = self.actor(fused)

        # 4. Mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        value = self.critic(fused)
        return logits, value

    def select_action(self, state_emb, goal_emb, action_mask=None, epsilon=0.1):
        """
        🔥 [修复版] 选择动作（低层动作）

        Args:
            state_emb: (batch, state_dim) 状态嵌入
            goal_emb: (batch, goal_dim) 目标嵌入
            action_mask: (batch, n_actions) 动作mask，1=可选，0=禁止
            epsilon: 探索率

        Returns:
            action: int, 选择的动作
            q_value: float, 对应的Q值
        """
        import torch
        import numpy as np

        # 1. 前向传播获取Q值
        with torch.no_grad():
            policy_output = self.forward(state_emb, goal_emb)

            # 兼容返回tuple的情况
            if isinstance(policy_output, tuple):
                q_values = policy_output[0]  # (batch, n_actions)
            else:
                q_values = policy_output

        # 2. 🔥 关键修复：应用Mask（在argmax之前）
        if action_mask is not None:
            # 确保mask在正确设备上
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask).to(q_values.device)

            # 确保mask维度匹配
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            if action_mask.size(1) != q_values.size(1):
                # 维度不匹配，截断或填充
                if action_mask.size(1) < q_values.size(1):
                    padding = torch.zeros(
                        action_mask.size(0),
                        q_values.size(1) - action_mask.size(1),
                        device=action_mask.device
                    )
                    action_mask = torch.cat([action_mask, padding], dim=1)
                else:
                    action_mask = action_mask[:, :q_values.size(1)]

            # 🔥 Q值屏蔽法：mask==0的位置设为-1e9
            masked_q_values = q_values.clone()
            masked_q_values[action_mask == 0] = -1e9
        else:
            masked_q_values = q_values

        # 3. 动作选择（epsilon-greedy）
        if np.random.rand() < epsilon:
            # 探索：从有效动作中随机选择
            if action_mask is not None:
                valid_indices = torch.nonzero(action_mask[0] > 0, as_tuple=False).squeeze()
                if valid_indices.numel() == 0:
                    action = torch.tensor(0)
                elif valid_indices.numel() == 1:
                    action = valid_indices
                else:
                    action = valid_indices[torch.randint(len(valid_indices), (1,))]
            else:
                action = torch.randint(0, q_values.size(1), (1,))
        else:
            # 利用：选择Q值最大的
            action = torch.argmax(masked_q_values[0])

        # 获取对应的Q值
        q_value = masked_q_values[0, action].item()

        return action, q_value


# ============================================
# 向后兼容：保留旧接口
# ============================================

class LowLevelPolicy(nn.Module):
    """
    低层策略网络 (向后兼容版)

    这是为了不破坏现有代码而保留的接口
    实际上会调用重构后的GoalConditionedLowLevelPolicy
    但可以在不提供goal的情况下使用
    """

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()

        logger.warning("⚠️ LowLevelPolicy使用旧接口，建议迁移到GoalConditionedLowLevelPolicy")

        # 创建配置
        config = {
            'state_dim': input_dim,
            'goal_dim': 64,  # 默认goal维度
            'hidden_dim': hidden_dim,
            'environment': {
                'nb_low_level_actions': action_dim
            },
            'use_cuda': False,
            'dropout': 0.1
        }

        # 使用新的GoalConditionedLowLevelPolicy
        self.policy = GoalConditionedLowLevelPolicy(config)

        # 简单的Actor和Critic接口（向后兼容）
        # 这些实际上会调用policy的对应部分
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def forward(self, state_emb):
        """
        兼容旧接口（不使用goal）

        Args:
            state_emb: 状态嵌入 [batch, input_dim]

        Returns:
            logits: 动作logits [batch, action_dim]
            value: 状态价值 [batch, 1]
        """
        # 调用新接口，但不提供goal
        return self.policy.forward(state_emb, goal_emb=None, action_mask=None)