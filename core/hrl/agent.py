#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRL Agent (重构版 - 完全修复)
整合High-Level和Low-Level策略，提供统一的分层决策接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging
from collections import deque
from typing import Tuple, Dict, Any, Optional

from torch_geometric.nn import global_mean_pool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HRLAgent:
    """
    Hierarchical RL Agent (重构版)

    职责：
    - 整合High-Level和Low-Level策略
    - 管理分层决策流程
    - 处理分层经验回放
    - 支持DAgger（专家指导）

    架构：
    High-Level: 选择目标节点（subgoal selection）
    Low-Level: 执行路径规划（goal-conditioned）
    """

    def __init__(self, config, encoder=None, phase=3, goal_strategy='adaptive', **kwargs):
        self.config = config
        self.phase = int(phase)
        self.goal_strategy = goal_strategy

        # 设备配置
        use_cuda = config.get('use_cuda', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        # ============================================
        # 环境参数
        # ============================================
        env_cfg = config.get('environment', config.get('env', {}))
        self.n_actions = kwargs.get('low_action_dim', env_cfg.get('nb_low_level_actions', 50))
        self.n_goals = kwargs.get('high_action_dim', env_cfg.get('nb_high_level_goals', 10))

        # ============================================
        # HRL参数
        # ============================================
        hrl_cfg = config.get('hrl', {})
        self.state_dim = hrl_cfg.get('state_dim', 128)
        self.goal_dim = hrl_cfg.get('goal_dim', 64)
        self.hidden_dim = hrl_cfg.get('hidden_dim', 128)

        self.subgoal_horizon = hrl_cfg.get('subgoal_horizon', 20)
        self.intrinsic_reward_weight = hrl_cfg.get('intrinsic_reward_weight', 0.3)

        # ============================================
        # Encoder（GNN特征提取）
        # ============================================
        if encoder is not None:
            self.encoder = encoder.to(self.device)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("✅ 使用预训练Encoder")
        else:
            self.encoder = None
            logger.info("⚠️ 未提供Encoder，将在运行时创建")

        # ============================================
        # High-Level Policy
        # ============================================
        from core.hrl.high_policy import HighLevelPolicy

        high_config = {
            'use_cuda': config.get('use_cuda', False),
            'hidden_dim': self.hidden_dim,
            'goal_dim': self.goal_dim,
            'gnn_output_dim': self.state_dim,
            'environment': {
                'nb_high_level_goals': self.n_goals
            },
            'dropout': config.get('dropout', 0.1)
        }

        self.high_policy = HighLevelPolicy(high_config).to(self.device)

        # ============================================
        # Low-Level Policy
        # ============================================
        from core.hrl.low_policy import GoalConditionedLowLevelPolicy

        low_config = {
            'use_cuda': config.get('use_cuda', False),
            'state_dim': self.state_dim,
            'goal_dim': self.goal_dim,
            'hidden_dim': self.hidden_dim,
            'environment': {
                'nb_low_level_actions': self.n_actions
            },
            'dropout': config.get('dropout', 0.1)
        }

        self.low_policy = GoalConditionedLowLevelPolicy(low_config).to(self.device)

        # ============================================
        # Target Networks
        # ============================================
        self.target_high_policy = HighLevelPolicy(high_config).to(self.device)
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())

        self.target_low_policy = GoalConditionedLowLevelPolicy(low_config).to(self.device)
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())

        # ============================================
        # Optimizers
        # ============================================
        training_cfg = config.get('training', {})

        lr_high = training_cfg.get('lr_high', training_cfg.get('learning_rate', 1e-4))
        lr_low = training_cfg.get('lr_low', training_cfg.get('learning_rate', 1e-4))

        self.optimizer_high = optim.Adam(self.high_policy.parameters(), lr=lr_high)
        self.optimizer_low = optim.Adam(self.low_policy.parameters(), lr=lr_low)

        # ============================================
        # 训练参数
        # ============================================
        self.batch_size = int(training_cfg.get('batch_size', 32))
        self.gamma = float(training_cfg.get('gamma', 0.99))
        self.target_update_freq = int(training_cfg.get('target_update_freq', 1000))

        # Epsilon配置
        epsilon_cfg = training_cfg.get('epsilon', {})

        self.epsilon_high_start = float(epsilon_cfg.get('initial_high', epsilon_cfg.get('initial', 0.3)))
        self.epsilon_high_end = float(epsilon_cfg.get('final_high', epsilon_cfg.get('final', 0.05)))
        self.epsilon_high = self.epsilon_high_start

        self.epsilon_low_start = float(epsilon_cfg.get('initial_low', epsilon_cfg.get('initial', 0.3)))
        self.epsilon_low_end = float(epsilon_cfg.get('final_low', epsilon_cfg.get('final', 0.05)))
        self.epsilon_low = self.epsilon_low_start

        self.epsilon_decay = float(epsilon_cfg.get('decay_steps', 50000))

        # ============================================
        # 经验回放
        # ============================================
        buffer_size = int(training_cfg.get('buffer_size', 50000))

        self.high_memory = deque(maxlen=buffer_size // 10)
        self.low_memory = deque(maxlen=buffer_size)

        # ============================================
        # ============================================
        # 状态管理
        # ============================================
        self.current_subgoal = None  # 整数（节点ID）
        self.current_subgoal_emb = None  # Tensor形式的subgoal embedding
        self.current_goal_emb = None  # Goal embedding
        self.subgoal_steps = 0

        # 向后兼容：旧代码可能使用 subgoal_step_count
        self.subgoal_step_count = 0

        self.steps_done = 0
        self.update_count = 0

        self._training = True

        # ============================================
        # 向后兼容：添加 goal_embedding
        # ============================================
        from core.hrl.goal_embedding import (
            AdaptiveSubgoalEmbedding,
            EnhancedRelativeGoalEmbedding,
            IterativeHybridGoalEmbedding
        )

        if self.goal_strategy == 'adaptive':
            self.goal_embedding = AdaptiveSubgoalEmbedding(
                state_dim=self.state_dim,
                goal_dim=self.goal_dim
            ).to(self.device)
        elif self.goal_strategy == 'hybrid':
            self.goal_embedding = IterativeHybridGoalEmbedding(
                local_state_dim=self.state_dim,
                goal_dim=self.goal_dim
            ).to(self.device)
        else:  # 'relative'
            self.goal_embedding = EnhancedRelativeGoalEmbedding(
                node_feat_dim=self.state_dim,
                goal_dim=self.goal_dim
            ).to(self.device)
        # ✅ 自适应epsilon
        self.adaptive_epsilon = config.get('hrl', {}).get('adaptive_epsilon', True)
        self.min_epsilon_low = 0.01

    def select_action(
            self,
            state: Dict,
            unconnected_dests: Optional[list] = None,
            action_mask: Optional[np.ndarray] = None,
            use_expert: bool = False,
            expert_action: Optional[int] = None,
            blacklist_info: Optional[dict] = None  # ✅ 新增参数
    ) -> Tuple[int, int, Dict]:
        """
        分层动作选择（黑名单感知）

        Args:
            state: 环境状态
            unconnected_dests: 未连接的目的节点列表
            action_mask: Low-Level动作mask
            use_expert: 是否使用专家
            expert_action: 专家动作
            blacklist_info: 黑名单信息（新增）

        Returns:
            high_action: High-Level动作（目标索引）
            low_action: Low-Level动作（下一跳节点）
            info: 信息字典
        """
        info = {
            'high_level_decision': False,
            'subgoal': self.current_subgoal,
            'subgoal_steps': self.subgoal_steps,
            'source': 'agent',
            'blacklist_total': 0
        }

        try:
            # ✅ 根据黑名单调整epsilon
            if self.adaptive_epsilon and blacklist_info:
                blacklist_count = blacklist_info.get('total', 0)
                blacklist_ratio = blacklist_count / self.n_actions if self.n_actions > 0 else 0

                # 黑名单多 → 降低探索（避免踩坑）
                adaptive_factor = 1.0 - blacklist_ratio * 0.3
                self.epsilon_low = max(
                    self.min_epsilon_low,
                    self.epsilon_low_start * adaptive_factor
                )

                info['adaptive_epsilon'] = self.epsilon_low
                info['blacklist_ratio'] = blacklist_ratio

            # 记录黑名单信息
            if blacklist_info:
                info['blacklist_total'] = blacklist_info.get('total', 0)
                info['blacklist_nodes'] = blacklist_info.get('nodes', [])

            # ============================================
            # 1. 判断是否需要新的subgoal
            # ============================================
            need_new_subgoal = self._need_new_subgoal(state, unconnected_dests)

            if need_new_subgoal:
                # ============================================
                # High-Level Decision
                # ============================================
                if use_expert and expert_action is not None:
                    # ✅ 专家建议也要检查Mask
                    if action_mask is None or (
                            0 <= expert_action < len(action_mask) and action_mask[expert_action] > 0):
                        self.current_subgoal = self._extract_subgoal_from_expert(
                            expert_action, unconnected_dests
                        )
                        info['source'] = 'expert_high'
                    else:
                        # 专家建议被Mask，用Agent
                        logger.warning(f"⚠️ 专家High建议{expert_action}被Mask，改用Agent")
                        self.current_subgoal = self._select_subgoal(state, unconnected_dests)
                        info['source'] = 'agent_high_fallback'
                else:
                    # Agent模式：使用High-Level策略
                    self.current_subgoal = self._select_subgoal(state, unconnected_dests)
                    info['source'] = 'agent_high'

                # 生成goal embedding
                self._generate_goal_embedding(state)

                self.subgoal_steps = 0
                info['high_level_decision'] = True
                info['subgoal'] = self.current_subgoal

                logger.debug(f"🎯 新子目标: {self.current_subgoal}")

            # ============================================
            # 2. Low-Level Execution（黑名单感知）
            # ============================================
            if use_expert and expert_action is not None and not need_new_subgoal:
                # ✅ 专家Low动作也要检查Mask
                if action_mask is None or (0 <= expert_action < len(action_mask) and action_mask[expert_action] > 0):
                    low_action = expert_action
                    info['source'] = 'expert_low'
                else:
                    logger.warning(f"⚠️ 专家Low动作{expert_action}被Mask，改用Agent")
                    low_action = self._select_low_action_with_blacklist(
                        state, action_mask, blacklist_info
                    )
                    info['source'] = 'agent_low_fallback'
            else:
                # Agent模式：使用Low-Level策略（黑名单感知）
                low_action = self._select_low_action_with_blacklist(
                    state, action_mask, blacklist_info
                )
                info['source'] = 'agent_low'

            # 记录选择的动作
            info['low_action'] = low_action
            if action_mask is not None:
                info['action_mask_sum'] = action_mask.sum()

            # 如果是黑名单中的节点，记录警告
            if blacklist_info and low_action in blacklist_info.get('nodes', []):
                info['blacklisted_action'] = True
                logger.warning(f"⚠️ Agent选择了黑名单中的节点 {low_action}")

            self.subgoal_steps += 1
            self.subgoal_step_count = self.subgoal_steps  # 向后兼容
            self.steps_done += 1

            # High action（目标在unconnected中的索引）
            high_action = 0
            if unconnected_dests and self.current_subgoal is not None:
                if self.current_subgoal in unconnected_dests:
                    high_action = unconnected_dests.index(self.current_subgoal)

            return high_action, low_action, info

        except Exception as e:
            logger.error(f"[Select Action] Error: {e}")
            import traceback
            traceback.print_exc()

            # Fallback：使用动作掩码选择有效动作
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    low_action = random.choice(valid_actions)
                else:
                    low_action = random.randint(0, self.n_actions - 1)
            else:
                low_action = random.randint(0, self.n_actions - 1)

            return 0, low_action, info
    def _need_new_subgoal(self, state: Dict, unconnected_dests: Optional[list]) -> bool:
        """判断是否需要新的subgoal"""
        # 1. 没有subgoal
        if self.current_subgoal is None:
            return True

        # 2. 没有未连接节点
        if not unconnected_dests or len(unconnected_dests) == 0:
            return False

        # 3. Subgoal已连接
        if self.current_subgoal not in unconnected_dests:
            return True

        # 4. Subgoal超时
        if self.subgoal_steps >= self.subgoal_horizon:
            logger.debug(f"⚠️ Subgoal超时 (steps={self.subgoal_steps})")
            return True

        # 5. 已到达subgoal
        current_pos = state.get('current_position', -1)
        if current_pos == self.current_subgoal:
            return True

        return False

    def _select_subgoal(self, state: Dict, unconnected_dests: list) -> int:
        """High-Level策略选择subgoal"""
        if not unconnected_dests or len(unconnected_dests) == 0:
            return None

        # 获取图嵌入
        graph_emb = self._get_graph_embedding(state)

        # 创建mask（只允许选择未连接节点）
        valid_goals = torch.zeros(1, self.n_goals, device=self.device)
        for i, dest in enumerate(unconnected_dests):
            if i < self.n_goals:
                valid_goals[0, i] = 1

        # High-Level策略选择
        with torch.no_grad():
            goal_idx, goal_emb = self.high_policy.select_goal(
                graph_emb,
                valid_goals,
                epsilon=self.epsilon_high
            )

        # 保存goal embedding
        self.current_goal_emb = goal_emb

        # 映射回实际节点
        goal_idx = goal_idx.item()
        if goal_idx < len(unconnected_dests):
            subgoal = unconnected_dests[goal_idx]
        else:
            subgoal = unconnected_dests[0]

        return subgoal

    def _generate_goal_embedding(self, state: Dict):
        """生成goal embedding"""
        try:
            graph_emb = self._get_graph_embedding(state)

            with torch.no_grad():
                _, goal_emb, _ = self.high_policy(graph_emb, return_subgoal=True)

            self.current_goal_emb = goal_emb

        except Exception as e:
            logger.error(f"[Goal Embedding] Error: {e}")
            self.current_goal_emb = torch.zeros(1, self.goal_dim, device=self.device)

    def _select_low_action(self, state: Dict, action_mask: Optional[np.ndarray]) -> int:
        """Low-Level策略选择动作"""
        # 获取图嵌入
        graph_emb = self._get_graph_embedding(state)

        # Goal embedding
        if self.current_goal_emb is None:
            self._generate_goal_embedding(state)

        # 转换mask
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            mask_tensor = None

        # Low-Level策略选择
        with torch.no_grad():
            action, _ = self.low_policy.select_action(
                graph_emb,
                self.current_goal_emb,
                mask_tensor,
                epsilon=self.epsilon_low
            )

        return action.item()

    def _get_graph_embedding(self, state):
        """
        🔥 [核心修复] 获取图嵌入，支持 Batch 处理
        """
        # 1. 尝试使用真实的 Encoder (如果有)
        if self.encoder is not None:
            try:
                # 情况 A: PyG Batch 对象 (训练时)
                if hasattr(state, 'batch') and state.batch is not None:
                    return self.encoder(state.x, state.edge_index, state.batch)

                # 情况 B: 单个 Data 对象 (推理时)
                # 构造一个全0的 batch 向量
                batch = torch.zeros(state.x.size(0), dtype=torch.long, device=self.device)
                return self.encoder(state.x, state.edge_index, batch)

            except Exception as e:
                logger.error(f"Encoder forward failed: {e}")
                # 如果出错，向下执行 Fallback

        # 2. Fallback (仅用于防止崩溃，输出随机噪声)
        # 必须返回正确的 Batch Size，否则 Loss 计算会报错
        batch_size = 1
        if hasattr(state, 'num_graphs'):
            # PyG Batch 对象包含 num_graphs 属性
            batch_size = state.num_graphs
        elif hasattr(state, 'batch') and state.batch is not None:
            batch_size = state.batch.max().item() + 1

        # logger.warning(f"Using random embedding fallback (Batch Size: {batch_size})")
        return torch.randn(batch_size, self.state_dim, device=self.device)
    def _extract_subgoal_from_expert(self, expert_action: int, unconnected_dests: list) -> int:
        """从专家动作提取subgoal"""
        if unconnected_dests and expert_action in unconnected_dests:
            return expert_action

        if unconnected_dests and len(unconnected_dests) > 0:
            return unconnected_dests[0]

        return None

    # ============================================
    # 向后兼容方法
    # ============================================

    def _generate_and_encode_subgoal(self, state: Dict):
        """
        向后兼容：生成并编码subgoal

        这是旧版本的方法名，调用新的 _generate_goal_embedding
        同时使用 goal_embedding 模块生成更好的嵌入

        Args:
            state: 环境状态

        注意:
        - current_subgoal 应该是整数（节点ID），由 _select_subgoal 设置
        - current_subgoal_emb 是tensor形式的embedding
        - current_goal_emb 是goal embedding
        """
        try:
            # 获取图嵌入
            graph_emb = self._get_graph_embedding(state)

            # 使用 goal_embedding 生成subgoal embedding
            with torch.no_grad():
                if self.goal_strategy == 'adaptive':
                    complexity = torch.tensor([[0.5]], device=self.device)
                    subgoal_emb, info = self.goal_embedding(graph_emb, complexity)
                    # AdaptiveSubgoalEmbedding 返回 (subgoal, info)
                    # 需要手动生成 goal_emb
                    if subgoal_emb.shape[-1] >= self.goal_dim:
                        goal_emb = subgoal_emb[..., :self.goal_dim]
                    else:
                        # 如果 subgoal 维度小于 goal_dim，填充
                        padding = torch.zeros(
                            subgoal_emb.size(0),
                            self.goal_dim - subgoal_emb.size(-1),
                            device=self.device
                        )
                        goal_emb = torch.cat([subgoal_emb, padding], dim=-1)

                elif self.goal_strategy == 'hybrid':
                    subgoal_emb, goal_emb, _ = self.goal_embedding(graph_emb)

                else:  # 'relative'
                    target_emb = torch.randn_like(graph_emb)  # 临时目标
                    goal_emb, info = self.goal_embedding(graph_emb, target_emb)
                    # EnhancedRelativeGoalEmbedding 返回 (goal_emb, info)
                    # 使用 goal_emb 作为 subgoal_emb
                    subgoal_emb = goal_emb

            # 确保维度正确
            if subgoal_emb.shape[-1] != self.goal_dim:
                subgoal_emb = subgoal_emb[..., :self.goal_dim]
            if goal_emb.shape[-1] != self.goal_dim:
                goal_emb = goal_emb[..., :self.goal_dim]

            # 设置embeddings
            # 注意：current_subgoal 是整数（节点ID），由 _select_subgoal 设置
            # current_subgoal_emb 是tensor形式的embedding
            self.current_subgoal_emb = subgoal_emb  # tensor
            self.current_goal_emb = goal_emb
            self.subgoal_step_count = 0

        except Exception as e:
            logger.error(f"[Generate Subgoal] Error: {e}")
            # Fallback
            self.current_subgoal_emb = torch.zeros(1, self.goal_dim, device=self.device)
            self.current_goal_emb = torch.zeros(1, self.goal_dim, device=self.device)
            self.subgoal_step_count = 0

    def store_transition_high(
            self, state: Dict, goal: int, reward: float, next_state: Dict, done: bool
    ):
        """存储High-Level经验 (修复版: 防止索引越界)"""

        # 🔥 [关键修复] 确保 goal 索引在有效范围内
        # High-Level Q网络只有 n_goals 个输出 (例如10)
        # 如果 goal 是物理节点ID (例如24)，会导致索引越界
        goal_idx = goal
        if isinstance(goal, (int, np.integer)):
            if goal >= self.n_goals:
                # 使用模运算映射到有效范围 [0, n_goals-1]
                # 这是一个兜底策略，防止程序崩溃
                goal_idx = goal % self.n_goals
                # logger.debug(f"⚠️ Goal映射: {goal} -> {goal_idx} (Max: {self.n_goals})")
            elif goal < 0:
                goal_idx = 0  # 默认值

        self.high_memory.append({
            'state': state,
            'goal': goal_idx,  # ✅ 存储修正后的索引
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def store_transition_low(
            self, state: Dict, action: int, reward: float, next_state: Dict, done: bool
    ):
        """存储Low-Level经验"""
        self.low_memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal_emb': self.current_goal_emb
        })

    # ============================================
    # 向后兼容：保留旧接口
    # ============================================

    def store_transition(self, state, action, reward, next_state, done, goal=None, next_valid_actions=None):
        """向后兼容的store_transition"""
        # 分发到对应的存储函数
        if isinstance(action, (list, tuple)) and len(action) == 2:
            high_action, low_action = action
            # 分别存储
            if goal is not None:
                self.store_transition_high(state, goal, reward, next_state, done)
            self.store_transition_low(state, low_action, reward, next_state, done)
        else:
            # 默认存储到Low-Level
            self.store_transition_low(state, action, reward, next_state, done)

    def update(self) -> float:
        """向后兼容的update接口"""
        losses = self.update_policies()

        # 返回总loss
        total_loss = losses.get('high_loss', 0.0) + losses.get('low_loss', 0.0)
        return total_loss

    def update_policies(self) -> Dict[str, float]:
        """更新策略（新接口）"""
        losses = {}

        # High-Level更新
        if len(self.high_memory) >= self.batch_size // 4:
            high_loss = self._update_high_level()
            losses['high_loss'] = high_loss

        # Low-Level更新
        if len(self.low_memory) >= self.batch_size:
            low_loss = self._update_low_level()
            losses['low_loss'] = low_loss

        # 更新target networks
        self._update_target_networks()

        # 更新epsilon
        self._update_epsilon()

        return losses

    def _update_high_level(self) -> float:
        """更新High-Level策略 (Double DQN)"""
        # 1. 检查样本数量
        if len(self.high_memory) < self.batch_size:
            return 0.0

        try:
            # 2. 采样
            batch = random.sample(self.high_memory, self.batch_size)

            # 3. 准备数据
            # 提取 Graph Embedding
            # 注意：如果 graph embedding 计算较慢，这里会是瓶颈
            state_embs = [self._get_graph_embedding(x['state']) for x in batch]
            next_state_embs = [self._get_graph_embedding(x['next_state']) for x in batch]

            # 堆叠并确保在正确的设备上
            state_tensor = torch.cat(state_embs, dim=0).to(self.device)
            next_state_tensor = torch.cat(next_state_embs, dim=0).to(self.device)

            goals = torch.tensor([x['goal'] for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward'] for x in batch], device=self.device).float().unsqueeze(1)
            dones = torch.tensor([x['done'] for x in batch], device=self.device).float().unsqueeze(1)

            # 4. 计算 Current Q
            # 🚀 优化：传入 return_subgoal=False，只计算 Q 值，不生成 Subgoal，节省算力
            # HighPolicy forward 返回: (q_values, subgoal_emb, value)
            curr_q_values, _, _ = self.high_policy(state_tensor, return_subgoal=False)
            curr_q = curr_q_values.gather(1, goals)

            # 5. 计算 Target Q (Double DQN)
            with torch.no_grad():
                # Online Net 选动作
                next_q_online, _, _ = self.high_policy(next_state_tensor, return_subgoal=False)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Target Net 评价值
                next_q_target, _, _ = self.target_high_policy(next_state_tensor, return_subgoal=False)
                next_q = next_q_target.gather(1, next_actions)

                target_q = rewards + (1 - dones) * self.gamma * next_q

            # 6. 计算 Loss & 更新
            loss = F.smooth_l1_loss(curr_q, target_q)

            self.optimizer_high.zero_grad()
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.high_policy.parameters(), 1.0)
            self.optimizer_high.step()

            return loss.item()

        except Exception as e:
            logger.error(f"[Update High Level] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _update_low_level(self) -> float:
        """更新Low-Level策略 (Goal-Conditioned Double DQN)"""
        # 1. 检查样本数量
        if len(self.low_memory) < self.batch_size:
            return 0.0

        try:
            # 2. 采样
            batch = random.sample(self.low_memory, self.batch_size)

            # 3. 准备数据
            state_embs = [self._get_graph_embedding(x['state']) for x in batch]
            next_state_embs = [self._get_graph_embedding(x['next_state']) for x in batch]

            state_tensor = torch.cat(state_embs, dim=0).to(self.device)
            next_state_tensor = torch.cat(next_state_embs, dim=0).to(self.device)

            actions = torch.tensor([x['action'] for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward'] for x in batch], device=self.device).float().unsqueeze(1)
            dones = torch.tensor([x['done'] for x in batch], device=self.device).float().unsqueeze(1)

            # =======================================================
            # 🔥 [关键修复] 过滤掉 action = -1 的无效数据
            # =======================================================
            valid_mask = (actions >= 0).squeeze()

            # 如果整个 batch 都是无效动作（极端情况），跳过
            if valid_mask.sum() == 0:
                return 0.0

            # 只保留有效的数据
            state_tensor = state_tensor[valid_mask]
            next_state_tensor = next_state_tensor[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            dones = dones[valid_mask]

            # 重新过滤 batch 列表，以便后续处理 goal_embs
            # valid_mask 是 tensor，转回 numpy 或 list 索引处理比较麻烦
            # 更简单的做法是：先处理好 tensor，再处理 goal_embs
            # =======================================================

            # 处理 Goal Embedding
            goal_embs = []
            # 注意：这里我们得重新遍历 batch，或者用 mask 筛选
            # 简单起见，我们直接遍历 batch 并应用同样的 mask 逻辑
            # 但上面的 valid_mask 是基于 actions tensor 的
            # 所以我们可以这样做：

            valid_indices = torch.nonzero(valid_mask).squeeze().cpu().tolist()
            if not isinstance(valid_indices, list):  # 单个元素时可能是 int
                valid_indices = [valid_indices]

            for idx in valid_indices:
                x = batch[idx]
                g = x.get('goal_emb')
                if g is None:
                    g = torch.zeros(1, self.goal_dim, device=self.device)
                else:
                    g = g.to(self.device)  # 确保在设备上
                    if g.dim() == 1: g = g.unsqueeze(0)

                    # 维度安全检查
                    if g.size(1) != self.goal_dim:
                        if g.size(1) > self.goal_dim:
                            g = g[:, :self.goal_dim]
                        else:
                            padding = torch.zeros(g.size(0), self.goal_dim - g.size(1), device=self.device)
                            g = torch.cat([g, padding], dim=1)
                goal_embs.append(g)

            goal_tensor = torch.cat(goal_embs, dim=0)

            # 4. 计算 Current Q
            # LowPolicy forward 返回: (logits, value)
            curr_logits, _ = self.low_policy(state_tensor, goal_tensor)

            # 🔥 现在 actions 里没有 -1 了，gather 不会报错
            curr_q = curr_logits.gather(1, actions)

            # 5. 计算 Target Q (Double DQN)
            with torch.no_grad():
                # Online Net 选动作
                next_logits, _ = self.low_policy(next_state_tensor, goal_tensor)
                next_actions = next_logits.argmax(dim=1, keepdim=True)

                # Target Net 评价值
                target_logits, _ = self.target_low_policy(next_state_tensor, goal_tensor)
                next_q = target_logits.gather(1, next_actions)

                target_q = rewards + (1 - dones) * self.gamma * next_q

            # 6. 计算 Loss & 更新
            loss = F.smooth_l1_loss(curr_q, target_q)

            self.optimizer_low.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.low_policy.parameters(), 1.0)
            self.optimizer_low.step()

            return loss.item()

        except Exception as e:
            logger.error(f"[Update Low Level] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    def _update_target_networks(self):
        """更新target networks"""
        self.update_count += 1

        if self.update_count % self.target_update_freq == 0:
            self.target_high_policy.load_state_dict(self.high_policy.state_dict())
            self.target_low_policy.load_state_dict(self.low_policy.state_dict())

    def _update_epsilon(self):
        """更新epsilon"""
        progress = min(self.steps_done / self.epsilon_decay, 1.0)

        self.epsilon_high = self.epsilon_high_start + (
                self.epsilon_high_end - self.epsilon_high_start
        ) * progress

        self.epsilon_low = self.epsilon_low_start + (
                self.epsilon_low_end - self.epsilon_low_start
        ) * progress

    def update_epsilon(self):
        """向后兼容的epsilon更新"""
        self._update_epsilon()

    def train(self):
        """训练模式"""
        self._training = True
        self.high_policy.train()
        self.low_policy.train()

    def eval(self):
        """评估模式"""
        self._training = False
        self.high_policy.eval()
        self.low_policy.eval()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'high_policy': self.high_policy.state_dict(),
            'low_policy': self.low_policy.state_dict(),
            'optimizer_high': self.optimizer_high.state_dict(),
            'optimizer_low': self.optimizer_low.state_dict(),
            'epsilon_high': self.epsilon_high,
            'epsilon_low': self.epsilon_low,
            'steps_done': self.steps_done,
            'config': self.config
        }, path)

        logger.info(f"✅ 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        import os
        if not os.path.exists(path):
            logger.warning(f"⚠️ 模型文件不存在: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        if 'high_policy' in checkpoint:
            self.high_policy.load_state_dict(checkpoint['high_policy'])
            self.target_high_policy.load_state_dict(checkpoint['high_policy'])

        if 'low_policy' in checkpoint:
            self.low_policy.load_state_dict(checkpoint['low_policy'])
            self.target_low_policy.load_state_dict(checkpoint['low_policy'])

        if 'optimizer_high' in checkpoint:
            self.optimizer_high.load_state_dict(checkpoint['optimizer_high'])
        if 'optimizer_low' in checkpoint:
            self.optimizer_low.load_state_dict(checkpoint['optimizer_low'])

        if 'epsilon_high' in checkpoint:
            self.epsilon_high = checkpoint['epsilon_high']
        if 'epsilon_low' in checkpoint:
            self.epsilon_low = checkpoint['epsilon_low']
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']

        logger.info(f"✅ 模型已加载: {path}")

    def _select_low_action_with_blacklist(
            self,
            state: Any,
            action_mask: Optional[np.ndarray] = None,
            blacklist_info: Optional[dict] = None
    ) -> int:
        try:
            # 1. 获取基础动作和权重
            if action_mask is not None and len(action_mask) > 0:
                valid_mask = action_mask > 0
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) == 0: return random.randint(0, self.n_actions - 1)
                action_weights = action_mask[valid_actions].copy()
            else:
                valid_actions = np.arange(self.n_actions)
                action_weights = np.ones(len(valid_actions))

            # 2. 黑名单权重惩罚
            if blacklist_info:
                blacklist_nodes = blacklist_info.get('nodes', [])
                for i, action in enumerate(valid_actions):
                    if action in blacklist_nodes: action_weights[i] *= 0.1

            # 3. 策略选择
            if random.random() < self.epsilon_low:
                # 探索：基于权重的概率采样
                p = action_weights / action_weights.sum()
                return int(np.random.choice(valid_actions, p=p))
            else:
                # 利用：Q网络决策
                if self.low_policy is not None:
                    state_emb = self._extract_state_embedding(state)
                    if self.current_goal_emb is None: self._generate_goal_embedding(state)

                    with torch.no_grad():
                        # 🔥 修复重点：处理 low_policy 返回的元组 (logits, value)
                        policy_output = self.low_policy(state_emb, self.current_goal_emb.to(state_emb.device))

                        if isinstance(policy_output, tuple):
                            q_values = policy_output[0].cpu().numpy().flatten()
                        else:
                            q_values = policy_output.cpu().numpy().flatten()

                    # Q值与Mask权重融合
                    combined_scores = q_values[valid_actions] * action_weights
                    return int(valid_actions[np.argmax(combined_scores)])

                return int(valid_actions[np.argmax(action_weights)])

        except Exception as e:
            logger.error(f"[Select Low Action] Error: {e}")
            return random.randint(0, self.n_actions - 1)

    def _extract_state_embedding(self, state):
        """🔥 修复 Data 对象无法直接投影的问题"""
        try:
            # A. 处理 PyG Data 对象
            if hasattr(state, 'x') and hasattr(state, 'edge_index'):
                if self.encoder is not None:
                    return self._get_graph_embedding(state)

                # 如果没有 Encoder，手动池化节点特征
                x = state.x
                state_tensor = torch.mean(x, dim=0, keepdim=True)  # [1, node_feat_dim]

            # B. 处理已经是 Tensor 或 Numpy 的情况
            else:
                state_tensor = self._prepare_state(state)
                if state_tensor.dim() == 1: state_tensor = state_tensor.unsqueeze(0)

            # 🔥 统一投影到 hidden_dim (128)
            if state_tensor.size(-1) != self.state_dim:
                if not hasattr(self, '_state_projection'):
                    self._state_projection = nn.Linear(state_tensor.size(-1), self.state_dim).to(self.device)

                with torch.no_grad():
                    # 确保输入是 Tensor 而非 Data 对象
                    state_tensor = self._state_projection(state_tensor.to(self.device))

            return state_tensor

        except Exception as e:
            logger.error(f"提取state embedding失败: {e}")
            return torch.zeros(1, self.state_dim, device=self.device)

    def _prepare_state(self, state):
        """辅助方法：将状态转换为tensor"""
        if isinstance(state, dict):
            # PyG Data 格式
            return state
        elif isinstance(state, np.ndarray):
            return torch.FloatTensor(state).unsqueeze(0)
        else:
            # 已经是tensor
            return state
    # ============================================
    # 4. 新增 record_failure 方法
    # ============================================
    def record_failure(self, node_id: int, reason: str):
        """
        记录节点失败（用于学习黑名单模式）

        Args:
            node_id: 失败的节点ID
            reason: 失败原因
        """
        if node_id not in self.failed_nodes_counter:
            self.failed_nodes_counter[node_id] = {
                'count': 0,
                'reasons': [],
                'last_failed': self.steps_done
            }

        self.failed_nodes_counter[node_id]['count'] += 1
        self.failed_nodes_counter[node_id]['reasons'].append(reason)
        self.failed_nodes_counter[node_id]['last_failed'] = self.steps_done

        self.blacklist_history.append({
            'step': self.steps_done,
            'node': node_id,
            'reason': reason,
            'epsilon': self.epsilon_low
        })

        logger.debug(f"📝 记录失败: 节点{node_id}, 原因:{reason}")

    # ============================================
    # 5. 新增 get_blacklist_learning_stats 方法
    # ============================================
    def get_blacklist_learning_stats(self) -> dict:
        """
        获取黑名单学习统计

        Returns:
            包含失败统计的字典
        """
        if not self.failed_nodes_counter:
            return {
                'total_failures': 0,
                'unique_failed_nodes': 0,
                'top_failed_nodes': [],
                'blacklist_history_size': 0
            }

        # 按失败次数排序
        sorted_nodes = sorted(
            self.failed_nodes_counter.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]  # 取前10个

        return {
            'total_failures': sum(info['count'] for info in self.failed_nodes_counter.values()),
            'unique_failed_nodes': len(self.failed_nodes_counter),
            'top_failed_nodes': [
                {
                    'node': node,
                    'count': info['count'],
                    'reasons': info['reasons'][-3:]  # 最近3个原因
                }
                for node, info in sorted_nodes
            ],
            'blacklist_history_size': len(self.blacklist_history)
        }

# ============================================
# 向后兼容：保留旧接口
# ============================================

class GoalConditionedHRLAgent(HRLAgent):
    """向后兼容的GoalConditionedHRLAgent"""

    def __init__(self, config, phase=3, goal_strategy='adaptive', **kwargs):
        logger.warning("⚠️ GoalConditionedHRLAgent已重构为HRLAgent，使用兼容模式")
        super().__init__(config, phase=phase, goal_strategy=goal_strategy, **kwargs)


# ... (agent.py 的其他代码保持不变) ...

# ============================================
# Helper Function
# ============================================

def create_goal_conditioned_agent(config, phase=3, goal_strategy='adaptive', encoder=None, **kwargs):
    """
    创建 HRL Agent 的工厂函数

    Args:
        config: 配置字典
        phase: 训练阶段 (默认: 3)
        goal_strategy: Goal embedding 策略
        encoder: 预训练的 GNN Encoder (可选) 🔥 [关键新增]
        **kwargs: 其他参数
    """
    # 实例化 HRLAgent 并透传 encoder
    return HRLAgent(
        config=config,
        encoder=encoder,  # 🔥 把 encoder 传进去
        phase=phase,
        goal_strategy=goal_strategy,
        **kwargs
    )


