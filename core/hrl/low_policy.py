#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low-Level Policy (TA-HRL v4 重构版)
职责：给定 subgoal (destination / target)，结合 Tree-Aware Encoder 输出的全图节点特征，
使用 Goal-Conditioned Attention 机制选择最优下一跳节点。
"""

import torch
import torch.nn as nn
import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)


class GoalConditionedLowLevelPolicy(nn.Module):
    """
    Goal-Conditioned Low-Level Policy (TA-HRL v4)

    核心机制：
    用目标 (Goal) 作为 Query，全图所有节点的特征作为 Key 和 Value，
    计算注意力权重，从而使得策略网络全局感知“去往目标的最优方向”。
    """

    def __init__(self, config):
        super().__init__()

        # 设备配置
        use_cuda = config.get('use_cuda', False)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        # 维度配置
        self.state_dim = config.get('state_dim', 128)  # GNN encoder输出的维度
        self.goal_dim = config.get('goal_dim', 64)  # Goal embedding维度
        self.hidden_dim = config.get('hidden_dim', 128)

        # 从环境配置读取动作空间维度 (全网节点数)
        env_cfg = config.get('environment', config.get('env', {}))
        self.action_dim = env_cfg.get('nb_low_level_actions', 50)
        dropout = config.get('dropout', 0.1)

        # 状态投影层
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # 目标投影层
        self.goal_projection = nn.Sequential(
            nn.Linear(self.goal_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # 🎯 目标条件交叉注意力 (Destination-Conditioned Cross-Attention)
        self.goal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # Actor: 输出各个节点的动作 Q 值 (DQN) 或 Logits
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # Critic: 评估当前状态-目标对的 Value
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state_emb: torch.Tensor, goal_emb: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        前向传播
        :param state_emb: 全局节点特征序列 [Batch, Num_nodes, State_dim]
        :param goal_emb: 当前子目标特征 [Batch, Goal_dim]
        :param action_mask: 动作合法性掩码 [Batch, Num_nodes]
        :return: (logits/q_values, state_value)
        """
        B = state_emb.size(0)

        # 1. 投影到统一的隐藏维度
        state_proj = self.state_projection(state_emb)  # [B, N, H]
        goal_proj = self.goal_projection(goal_emb) if goal_emb is not None else torch.zeros(B, self.hidden_dim,
                                                                                            device=state_emb.device)  # [B, H]

        # 2. 🎯 注意力机制融合 (Goal 去审视 Nodes)
        query = goal_proj.unsqueeze(1)  # [B, 1, H]
        attn_out, _ = self.goal_attention(query=query, key=state_proj, value=state_proj)

        # 残差连接与归一化
        attn_context = self.attn_norm(attn_out.squeeze(1) + goal_proj)  # [B, H]

        # 3. 拼接上下文和原始目标特征
        fused = torch.cat([attn_context, goal_proj], dim=-1)  # [B, H*2]

        # 4. 计算各动作的得分 (Q-values)
        logits = self.actor(fused)

        # 5. 应用严格的动作掩码 (Masking)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        # 6. 计算状态价值
        value = self.critic(fused)

        return logits, value

    def select_action(self, state_emb: torch.Tensor, goal_emb: Optional[torch.Tensor] = None,
                      action_mask: Optional[torch.Tensor] = None, epsilon: float = 0.0, **kwargs):
        """
        为 agent_action.py 提供兼容的动作选择接口
        在推断 (Inference) 阶段直接输出具有最高 Q 值 / Logits 的合法动作
        支持 Epsilon-Greedy 探索
        返回的 action 必须是一个 torch.Tensor 以兼容外部的 .item() 调用
        """
        with torch.no_grad():
            logits, value = self.forward(state_emb, goal_emb, action_mask)

            # 引入 epsilon-greedy 探索机制
            if epsilon > 0.0 and random.random() < epsilon:
                # 随机探索
                if action_mask is not None:
                    valid_actions = (action_mask.squeeze() > 0).nonzero(as_tuple=True)[0]
                    if len(valid_actions) > 0:
                        idx = random.randint(0, len(valid_actions) - 1)
                        action = valid_actions[idx]  # 这里直接提取是 0-dim tensor
                    else:
                        # 兜底机制：如果没有合法动作，只能走最高分
                        action = torch.argmax(logits, dim=-1).squeeze()
                else:
                    action = torch.tensor(random.randint(0, logits.size(-1) - 1), device=logits.device)
            else:
                # 贪心选择最高分的动作索引，squeeze() 后是一个 0-dim tensor
                action = torch.argmax(logits, dim=-1).squeeze()

            return action, value

    def reset_parameters(self):
        """重置网络参数（用于异常自愈）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)


# ============================================
# 向后兼容：保留旧接口以防其它模块报错
# ============================================

class LowLevelPolicy(nn.Module):
    """
    低层策略网络 (向后兼容版)
    """

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        logger.warning("⚠️ 正在使用向后兼容的 LowLevelPolicy 接口，底层已映射为 GoalConditionedLowLevelPolicy")

        config = {
            'state_dim': input_dim,
            'goal_dim': 64,
            'hidden_dim': hidden_dim,
            'environment': {
                'nb_low_level_actions': action_dim
            },
            'use_cuda': False,
            'dropout': 0.1
        }
        self.policy = GoalConditionedLowLevelPolicy(config)
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def forward(self, state, action_mask=None):
        # 兼容旧的前向传播 (没有传入 goal_emb 的情况)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        logits, value = self.policy(state, None, action_mask)
        return logits, value, value

    def select_action(self, state, action_mask=None, epsilon=0.0, **kwargs):
        # 向后兼容接口的推断，支持 epsilon 参数传入
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return self.policy.select_action(state, None, action_mask, epsilon=epsilon, **kwargs)