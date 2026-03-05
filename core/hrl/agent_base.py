#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentBase — 初始化 + 网络构建 + 训练控制 + 保存/加载
"""

import torch
import torch.optim as optim
import numpy as np
import logging
from collections import deque
from typing import Dict

logger = logging.getLogger(__name__)


class HRLAgentBase:
    """
    负责：
    - __init__: 所有网络/优化器/buffer/状态变量初始化
    - train / eval / save / load
    - reset_network_parameters
    - register_encoder_to_optimizer
    """

    def __init__(self, config, encoder=None, phase=3, goal_strategy='adaptive', env=None, **kwargs):
        self.config = config
        self.phase = int(phase)
        self.goal_strategy = goal_strategy
        self.env = env

        # 设备
        use_cuda = config.get('use_cuda', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        # ── 环境参数 ──────────────────────────────────────────────────────
        env_cfg = config.get('environment', config.get('env', {}))
        self.n_actions = kwargs.get('low_action_dim', env_cfg.get('nb_low_level_actions', 50))
        self.n_goals   = kwargs.get('high_action_dim', env_cfg.get('nb_high_level_goals', 28))  # [SDG-HRL] goal=节点ID

        # ── HRL参数 ───────────────────────────────────────────────────────
        hrl_cfg = config.get('hrl', {})
        self.state_dim  = hrl_cfg.get('state_dim', 128)
        self.goal_dim   = hrl_cfg.get('goal_dim', 64)
        self.hidden_dim = hrl_cfg.get('hidden_dim', 128)

        self.subgoal_horizon        = hrl_cfg.get('subgoal_horizon', 40)  # [SDG-HRL]
        self.intrinsic_reward_weight = hrl_cfg.get('intrinsic_reward_weight', 0.3)
        self.adaptive_epsilon        = hrl_cfg.get('adaptive_epsilon', True)
        self.min_epsilon_low         = 0.01

        # ── Encoder ───────────────────────────────────────────────────────
        if encoder is not None:
            self.encoder = encoder.to(self.device)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("✅ 使用预训练Encoder")
        else:
            self.encoder = None
            logger.info("⚠️ 未提供Encoder")

        # ── High-Level Policy ─────────────────────────────────────────────
        from core.hrl.high_policy import HighLevelPolicy
        high_config = {
            'use_cuda':       config.get('use_cuda', False),
            'hidden_dim':     self.hidden_dim,
            'goal_dim':       self.goal_dim,
            'gnn_output_dim': self.state_dim,
            'environment':    {'nb_high_level_goals': self.n_goals},
            'dropout':        config.get('dropout', 0.1),
        }
        self.high_policy        = HighLevelPolicy(high_config).to(self.device)
        self.target_high_policy = HighLevelPolicy(high_config).to(self.device)
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())

        # ── Low-Level Policy ──────────────────────────────────────────────
        from core.hrl.low_policy import GoalConditionedLowLevelPolicy
        low_config = {
            'use_cuda':   config.get('use_cuda', False),
            'state_dim':  self.state_dim,
            'goal_dim':   self.goal_dim,
            'hidden_dim': self.hidden_dim,
            'environment': {'nb_low_level_actions': self.n_actions},
            'dropout':    config.get('dropout', 0.1),
        }
        self.low_policy        = GoalConditionedLowLevelPolicy(low_config).to(self.device)
        self.target_low_policy = GoalConditionedLowLevelPolicy(low_config).to(self.device)
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())

        # ── Optimizers ────────────────────────────────────────────────────
        training_cfg = config.get('training', {})
        lr_high = training_cfg.get('lr_high', training_cfg.get('learning_rate', 1e-4))
        lr_low  = training_cfg.get('lr_low',  training_cfg.get('learning_rate', 1e-4))
        self.optimizer_high = optim.Adam(self.high_policy.parameters(), lr=lr_high)
        self.optimizer_low  = optim.Adam(self.low_policy.parameters(),  lr=lr_low)

        # ── 训练超参 ──────────────────────────────────────────────────────
        self.batch_size        = int(training_cfg.get('batch_size', 32))
        self.gamma             = float(training_cfg.get('gamma', 0.99))
        self.target_update_freq = int(training_cfg.get('target_update_freq', 1000))
        self.clip_grad_norm    = training_cfg.get('clip_grad_norm', 1.0)
        self.tau               = training_cfg.get('tau', 0.005)
        self.huber_delta       = training_cfg.get('huber_delta', 1.0)

        # ── Epsilon ───────────────────────────────────────────────────────
        epsilon_cfg = training_cfg.get('epsilon', {})
        self.epsilon_high_start = float(epsilon_cfg.get('initial_high', epsilon_cfg.get('initial', 0.3)))
        self.epsilon_high_end   = float(epsilon_cfg.get('final_high',   epsilon_cfg.get('final',   0.10)))
        self.epsilon_high       = self.epsilon_high_start
        self.epsilon_low_start  = float(epsilon_cfg.get('initial_low',  epsilon_cfg.get('initial', 0.3)))
        self.epsilon_low_end    = float(epsilon_cfg.get('final_low',    epsilon_cfg.get('final',   0.10)))
        self.epsilon_low        = self.epsilon_low_start
        self.epsilon_decay      = float(epsilon_cfg.get('decay_steps', 50000))  # 保留兼容
        # [Fix] episode-based衰减：在N个episode内从initial线性衰减到final
        # 默认200ep，与path_guide窗口（ep<200）对齐，确保探索覆盖学习期
        self.epsilon_decay_episodes = int(epsilon_cfg.get('decay_episodes', 200))
        self.total_episodes     = 0

        # ── Replay Buffer ─────────────────────────────────────────────────
        buffer_size = int(training_cfg.get('buffer_size', 50000))
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prioritized_buffer import PrioritizedReplayBuffer
            from elite_buffer import EliteBuffer
            self.high_memory    = PrioritizedReplayBuffer(buffer_size // 2)
            self.low_memory     = PrioritizedReplayBuffer(buffer_size)
            self.elite_buffer   = EliteBuffer(capacity=5000)
            self._use_per       = True
            self._best_reward   = -1e9
            self._ep_transitions = []
            logger.info("✅ [SDG-HRL] PER + EliteBuffer 已启用")
        except ImportError as _e:
            logger.warning(f"⚠️ PER未找到，回退到deque: {_e}")
            self.high_memory    = deque(maxlen=buffer_size // 2)
            self.low_memory     = deque(maxlen=buffer_size)
            self.elite_buffer   = None
            self._use_per       = False
            self._best_reward   = -1e9
            self._ep_transitions = []
        self.success_memory = deque(maxlen=20000)  # [SDG-HRL] 成功经验池

        # ── 状态变量 ──────────────────────────────────────────────────────
        self.current_subgoal     = None
        self.current_subgoal_emb = None
        self.current_goal_emb    = None
        self.subgoal_steps       = 0
        self.subgoal_step_count  = 0   # 向后兼容
        self.current_start_node  = None
        self.steps_done          = 0
        self.update_count        = 0
        self._training           = True

        # ── Goal Embedding 模块 ───────────────────────────────────────────
        from core.hrl.goal_embedding import (
            AdaptiveSubgoalEmbedding,
            EnhancedRelativeGoalEmbedding,
            IterativeHybridGoalEmbedding,
        )
        if self.goal_strategy == 'adaptive':
            self.goal_embedding = AdaptiveSubgoalEmbedding(
                state_dim=self.state_dim, goal_dim=self.goal_dim).to(self.device)
        elif self.goal_strategy == 'hybrid':
            self.goal_embedding = IterativeHybridGoalEmbedding(
                local_state_dim=self.state_dim, goal_dim=self.goal_dim).to(self.device)
        else:
            self.goal_embedding = EnhancedRelativeGoalEmbedding(
                node_feat_dim=self.state_dim, goal_dim=self.goal_dim).to(self.device)

        # ── 统计 ──────────────────────────────────────────────────────────
        self.high_loss_history = deque(maxlen=100)
        self.low_loss_history  = deque(maxlen=100)
        self.gradient_norms    = deque(maxlen=100)

    # ── 训练/评估模式 ──────────────────────────────────────────────────────

    def train(self):
        self._training = True
        self.high_policy.train()
        self.low_policy.train()

    def eval(self):
        self._training = False
        self.high_policy.eval()
        self.low_policy.eval()

    # ── 保存/加载 ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'high_policy':    self.high_policy.state_dict(),
            'low_policy':     self.low_policy.state_dict(),
            'optimizer_high': self.optimizer_high.state_dict(),
            'optimizer_low':  self.optimizer_low.state_dict(),
            'epsilon_high':   self.epsilon_high,
            'epsilon_low':    self.epsilon_low,
            'steps_done':     self.steps_done,
            'config':         self.config,
        }, path)
        logger.info(f"✅ 模型已保存: {path}")

    def load(self, path: str):
        import os
        if not os.path.exists(path):
            logger.warning(f"⚠️ 模型文件不存在: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        if 'high_policy' in ckpt:
            self.high_policy.load_state_dict(ckpt['high_policy'])
            self.target_high_policy.load_state_dict(ckpt['high_policy'])
        if 'low_policy' in ckpt:
            self.low_policy.load_state_dict(ckpt['low_policy'])
            self.target_low_policy.load_state_dict(ckpt['low_policy'])
        if 'optimizer_high' in ckpt:
            self.optimizer_high.load_state_dict(ckpt['optimizer_high'])
        if 'optimizer_low' in ckpt:
            self.optimizer_low.load_state_dict(ckpt['optimizer_low'])
        self.epsilon_high = ckpt.get('epsilon_high', self.epsilon_high)
        self.epsilon_low  = ckpt.get('epsilon_low',  self.epsilon_low)
        self.steps_done   = ckpt.get('steps_done',   self.steps_done)
        logger.info(f"✅ 模型已加载: {path}")

    # ── 工具 ──────────────────────────────────────────────────────────────

    def register_encoder_to_optimizer(self, lr=None):
        """将encoder参数加入optimizer_low，使tree_bias可训练"""
        if self.encoder is None:
            logger.warning("[SDG-HRL] encoder=None，无法注册到optimizer")
            return False
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
        _lr = lr or self.optimizer_low.param_groups[0]['lr']
        self.optimizer_low.add_param_group({
            'params': list(self.encoder.parameters()),
            'lr': _lr,
            'name': 'sdg_hrl_encoder',
        })
        n = sum(p.numel() for p in self.encoder.parameters())
        logger.info(f"✅ [SDG-HRL] encoder注册到optimizer_low (lr={_lr:.2e}, params={n:,})")
        return True

    def reset_network_parameters(self):
        """训练异常时自愈：重置网络参数和优化器"""
        logger.warning("🔄 [Auto-Fix] 正在重置网络参数...")
        if hasattr(self.high_policy, 'reset_parameters'):
            self.high_policy.reset_parameters()
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())
        if hasattr(self.low_policy, 'reset_parameters'):
            self.low_policy.reset_parameters()
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())
        training_cfg = self.config.get('training', {})
        self.optimizer_high = optim.Adam(self.high_policy.parameters(),
                                         lr=training_cfg.get('lr_high', 1e-4))
        self.optimizer_low  = optim.Adam(self.low_policy.parameters(),
                                         lr=training_cfg.get('lr_low',  1e-4))
        logger.info("✅ 网络参数重置完成")