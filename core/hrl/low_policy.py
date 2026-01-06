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
        # 3. Fusion（状态-目标融合）
        # ============================================
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ============================================
        # 4. Actor（Q Network - 动作选择）
        # ============================================
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # ============================================
        # 5. Critic（Value Network - 状态评估）
        # ============================================
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
        前向传播

        Args:
            state_emb: 状态嵌入 [batch, state_dim]
            goal_emb: Goal embedding [batch, goal_dim] (可选)
            action_mask: 动作mask [batch, action_dim] (可选)

        Returns:
            logits: 动作logits [batch, action_dim]
            value: 状态价值 [batch, 1]
        """
        # 投影状态
        state_proj = self.state_projection(state_emb)

        # 如果提供了goal，进行融合
        if goal_emb is not None:
            # 投影goal
            goal_proj = self.goal_projection(goal_emb)

            # 融合state和goal
            fused = torch.cat([state_proj, goal_proj], dim=1)
            fused = self.fusion(fused)
        else:
            # 没有goal时，直接使用state
            fused = state_proj

        # Actor输出（Q值）
        logits = self.actor(fused)

        # 应用mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # Critic输出（状态价值）
        value = self.critic(fused)

        return logits, value

    def select_action(
            self,
            state_emb: torch.Tensor,
            goal_emb: Optional[torch.Tensor] = None,
            action_mask: Optional[torch.Tensor] = None,
            epsilon: float = 0.0
    ) -> tuple:
        """
        选择动作（epsilon-greedy）

        Args:
            state_emb: 状态嵌入 [batch, state_dim]
            goal_emb: Goal embedding [batch, goal_dim] (可选)
            action_mask: 动作mask [batch, action_dim] (可选)
            epsilon: 探索率

        Returns:
            action: 选择的动作 [batch]
            q_value: 对应的Q值 [batch]
        """
        batch_size = state_emb.size(0)

        # 前向传播
        logits, _ = self.forward(state_emb, goal_emb, action_mask)

        # Epsilon-greedy
        if self.training and torch.rand(1).item() < epsilon:
            # 随机选择（从有效动作中）
            if action_mask is not None:
                valid_actions = (action_mask > 0).nonzero(as_tuple=True)[1]
                if len(valid_actions) > 0:
                    action = valid_actions[
                        torch.randint(0, len(valid_actions), (batch_size,))
                    ]
                else:
                    action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            else:
                action = torch.randint(0, self.action_dim, (batch_size,), device=self.device)
        else:
            # Greedy选择
            action = torch.argmax(logits, dim=1)

        # 获取对应的Q值
        q_value = logits.gather(1, action.unsqueeze(1)).squeeze(1)

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