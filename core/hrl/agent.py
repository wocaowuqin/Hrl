# !/usr/bin/env python3
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
#logger.setLevel(logging.WARNING)


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

    def __init__(self, config, encoder=None, phase=3, goal_strategy='adaptive', env=None, **kwargs):
        self.config = config
        self.phase = int(phase)
        self.goal_strategy = goal_strategy
        self.env = env
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
        # 状态管理
        # ============================================
        self.current_subgoal = None  # 整数（节点ID）
        self.current_subgoal_emb = None  # Tensor形式的subgoal embedding
        self.current_goal_emb = None  # Goal embedding
        self.subgoal_steps = 0
        self.current_start_node = None  # 🔥 新增：当前起点节点

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

        # 🔥 新增：训练稳定性参数
        self.clip_grad_norm = training_cfg.get('clip_grad_norm', 1.0)
        self.tau = training_cfg.get('tau', 0.005)  # 软更新系数
        self.huber_delta = training_cfg.get('huber_delta', 1.0)  # Huber loss delta

        # 🔥 新增：损失统计
        self.high_loss_history = deque(maxlen=100)
        self.low_loss_history = deque(maxlen=100)
        self.gradient_norms = deque(maxlen=100)

        # 🔥 新增：黑名单学习相关
        self.failed_nodes_counter = {}  # 记录节点失败次数
        self.blacklist_history = []  # 黑名单历史记录

    def select_action(
            self,
            state: Dict,
            unconnected_dests: Optional[list] = None,
            action_mask: Optional[np.ndarray] = None,
            use_expert: bool = False,
            expert_action: Optional[int] = None,
            blacklist_info: Optional[dict] = None
    ) -> Tuple[int, int, Dict]:
        """
        🔥 [V40.20 最终修复版] Start-End 联合决策逻辑
        """
        # 初始化 info 字典
        info = {
            'high_level_decision': False,
            'subgoal': self.current_subgoal,
            'subgoal_steps': self.subgoal_steps,
            'source': 'agent',
            'start_node': None
        }

        try:
            # ============================================
            # 1. High-Level Decision (决定“去哪”和“从哪走”)
            # ============================================
            need_new_subgoal = self._need_new_subgoal(state, unconnected_dests)

            if need_new_subgoal:
                # === A. 选终点 (Goal Head) ===
                # 注意：_select_subgoal 内部负责更新 self.current_subgoal
                self.current_subgoal = self._select_subgoal(state, unconnected_dests, action_mask)
                self.subgoal_steps = 0

                # 更新 info
                info['high_level_decision'] = True
                info['subgoal'] = self.current_subgoal
                info['source'] = 'agent_high'

                logger.debug(f"🎯 [High] 新子目标: {self.current_subgoal}")

                # === B. 🔥🔥🔥 选起点 (Start Head) 🔥🔥🔥 ===
                # 关键修复：调用 _force_select_start_node 强制更新环境位置
                start_node = self._force_select_start_node(state, self.current_subgoal)
                info['start_node'] = start_node

                # 日志确认代码已生效
                logger.info(f"✅ [SelectAction] High-Level 完成: Start={start_node} -> Goal={self.current_subgoal}")

            # ============================================
            # 2. Low-Level Execution (决定“下一步怎么走”)
            # ============================================
            low_action = self._select_low_action_with_blacklist(state, action_mask, blacklist_info)

            # 更新计数器
            self.subgoal_steps += 1
            self.subgoal_step_count = self.subgoal_steps
            # self.steps_done += 1 # 如果外部 Trainer 会增加 step，这里可以注释掉，避免重复计数

            # 更新 info
            info['low_action'] = low_action
            if blacklist_info and low_action in blacklist_info.get('nodes', []):
                info['blacklisted_action'] = True

            # ============================================
            # 3. 计算返回值 (High Action Index)
            # ============================================
            high_action = 0
            if unconnected_dests is not None:
                # 路由模式 (Phase 2/3 Routing): 返回 unconnected_dests 的索引
                if self.current_subgoal in unconnected_dests:
                    high_action = unconnected_dests.index(self.current_subgoal)
            else:
                # VNF 模式 (Phase 3 Placement): 直接返回节点 ID 或 DC ID
                high_action = self.current_subgoal if self.current_subgoal is not None else 0

            return high_action, low_action, info

        except Exception as e:
            logger.error(f"❌ [Select Action] 发生未捕获异常: {e}")
            import traceback
            traceback.print_exc()
            # 发生严重错误时的兜底返回
            return 0, 0, info

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

    def _select_subgoal(self, state: Dict, unconnected_dests: list, action_mask: np.ndarray = None) -> int:
        """
        🔥 [终极修复版]
        1. 修正了 Index Aliasing 问题 (不再映射 unconnected_dests)
        2. 包含完整的 Brain Scan 和 Mask 诊断
        """
        # Get graph embedding
        graph_emb = self._get_graph_embedding(state)

        # ============================================================
        # 1. 获取原始 Q 值
        # ============================================================
        with torch.no_grad():
            q_values, goal_emb, _ = self.high_policy(graph_emb, return_subgoal=True)
            # q_values shape: [1, n_nodes] usually

        # =================================================================================
        # 🕵️‍♂️🔥 [调试模块 A] Brain Scan (查看原始网络想法)
        # =================================================================================
        try:
            q_list = q_values.squeeze().cpu().numpy().tolist()
            # 找到前3名
            top_indices = np.argsort(q_list)[-3:][::-1]
            top_scores = [q_list[i] for i in top_indices]

            # 专门监控当前可能出问题的节点 (比如 7)
            monitor_nodes = [7]
            monitor_str = " | ".join([f"Node{n}={q_list[n]:.2f}" for n in monitor_nodes if n < len(q_list)])

            logger.warning(
                f"\n🧠 [BRAIN] Raw Top3: {top_indices} (Scores: {[f'{s:.2f}' for s in top_scores]}) | {monitor_str}")
        except Exception as e:
            logger.warning(f"🧠 [BRAIN] Log Error: {e}")

        # ============================================================
        # 2. Mask 处理 (施加惩罚)
        # ============================================================
        masked_q_values = q_values.clone()
        effective_mask = torch.ones_like(q_values)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, device=self.device).float()

            # 维度对齐检查
            if mask_tensor.shape[-1] != q_values.shape[-1]:
                logger.error(f"❌ [MASK] 维度不匹配! Mask={mask_tensor.shape[-1]}, Q={q_values.shape[-1]}")
                # 紧急回退：全1 mask，避免 crash，但会打印错误
                mask_tensor = torch.ones_like(q_values)

            # 确保 mask 形状正确
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)

            # 🔥 核心 Mask 操作：将非法节点 Q 值设为负无穷
            huge_negative = torch.tensor(-1e9, device=self.device)
            masked_q_values = torch.where(mask_tensor > 0, q_values, huge_negative)
            effective_mask = mask_tensor

            # 🕵️‍♂️ [调试模块 B] Mask 检查
            try:
                invalid_indices = torch.where(mask_tensor.squeeze() == 0)[0].tolist()
                logger.warning(f"🎭 [MASK] 被封杀节点数: {len(invalid_indices)}")
                if 7 in invalid_indices:
                    logger.warning(f"   ✅ Node 7 已被正确 Mask (Q应该变为 -1e9)")
                else:
                    logger.error(f"   ❌ 警告: Node 7 未被 Mask! 它仍然是可选的!")
            except:
                pass

        # ============================================================
        # 3. 动作选择 (Action Selection)
        # ============================================================
        goal_idx = 0

        # Epsilon-Greedy
        if random.random() < self.epsilon_high:
            # 随机探索：只从 mask=1 的节点中选
            valid_indices = torch.nonzero(effective_mask.squeeze() > 0).flatten()
            if len(valid_indices) > 0:
                rand_pos = torch.randint(0, len(valid_indices), (1,)).item()
                goal_idx = valid_indices[rand_pos].item()
                logger.warning(f"🎲 [EXPLORE] 随机选中: {goal_idx}")
            else:
                logger.error(f"💀 [FAIL] 无路可走(Mask全0)，强制返回 0")
                goal_idx = 0
        else:
            # 贪婪选择
            goal_idx = masked_q_values.argmax(dim=1).item()

            # --- 安全检查 (Sanity Check) ---
            if action_mask is not None and len(action_mask) > goal_idx:
                if action_mask[goal_idx] == 0:
                    logger.error(
                        f"💀 [CRITICAL] 贪婪策略选中了被Mask的节点 {goal_idx}! MaskedQ={masked_q_values[0][goal_idx]}")

                    # 🚑 紧急修复机制：如果网络选了死路，强制选第一个活路
                    valid_indices = np.where(action_mask > 0)[0]
                    if len(valid_indices) > 0:
                        goal_idx = valid_indices[0]
                        logger.warning(f"🔧 [FIX] 强制重定向到第一个合法节点: {goal_idx}")

        # ============================================================
        # 4. 后处理 (❌ BUG 修复区)
        # ============================================================

        # ❌ [已删除] 旧的错误逻辑 (Index Aliasing Source)
        # if unconnected_dests is not None:
        #     if goal_idx < len(unconnected_dests):
        #         subgoal = unconnected_dests[goal_idx]  <-- 罪魁祸首

        # ✅ [新逻辑] 直接使用物理索引
        # 神经网络输出的就是全图的节点 ID，不需要映射！
        subgoal = int(goal_idx)

        logger.info(f"🎯 [DECISION] Final Subgoal: {subgoal} (RawIdx: {goal_idx})")

        # Update Goal Embedding (用于传给 Low Level Policy)
        if hasattr(state, 'x') and state.x is not None and subgoal < state.x.size(0):
            node_feat = state.x[subgoal]
            # Padding check
            if node_feat.size(0) > self.goal_dim:
                node_feat = node_feat[:self.goal_dim]
            elif node_feat.size(0) < self.goal_dim:
                padding = torch.zeros(self.goal_dim - node_feat.size(0), device=self.device)
                node_feat = torch.cat([node_feat, padding])
            self.current_goal_emb = node_feat.unsqueeze(0)
        else:
            # Fallback
            self.current_goal_emb = goal_emb

        return subgoal
    def _select_start_node(self, state: Dict, goal_idx: int, graph_emb: torch.Tensor) -> int:
        """
        🔥 [V41.3 修复版] 使用 Start Selector 选择起点节点 (支持 Tuple 解包)

        Args:
            state: 当前状态（包含图数据）
            goal_idx: 目标节点索引
            graph_emb: 图的全局嵌入 [1, hidden_dim]

        Returns:
            start_node_idx: 起点节点索引（物理节点ID）
        """
        try:
            # 🔥🔥🔥 1. 自动解包 Tuple (obs, info)
            if isinstance(state, tuple):
                real_state = state[0]
            else:
                real_state = state

            # 2. 获取节点嵌入（原始GNN输出）
            if hasattr(real_state, 'x') and hasattr(real_state, 'edge_index'):
                with torch.no_grad():
                    if self.encoder is not None:
                        # 使用GNN编码器获取原始节点嵌入
                        node_embeddings = self.encoder(real_state.x, real_state.edge_index)
                        # shape: [num_nodes, gnn_output_dim]
                    else:
                        # 兜底：直接使用state.x
                        node_embeddings = real_state.x
            else:
                # 只有在解包后依然没有 x 属性时才报警告
                # logger.warning("⚠️ State没有x或edge_index，使用默认起点")
                return self._get_default_start_node(real_state)

            # 3. 获取目标节点的投影embedding
            # 使用 high_policy 的 state_projection 投影 graph_emb
            with torch.no_grad():
                target_emb = self.high_policy.state_projection(graph_emb)
                # shape: [1, hidden_dim]

            # 4. 构建树mask (传入解包后的 real_state)
            tree_mask = self._build_tree_mask(real_state)

            # 5. 检查mask有效性
            if tree_mask.sum() == 0:
                # logger.warning("⚠️ 树mask为空，使用默认起点")
                return self._get_default_start_node(real_state)

            # 6. 调用 HighLevelPolicy 的 select_start_node
            with torch.no_grad():
                start_node_idx, log_prob = self.high_policy.select_start_node(
                    node_embeddings=node_embeddings,
                    target_emb=target_emb,
                    tree_mask=tree_mask,
                    sample=True  # 训练时采样，测试时可以改为False
                )

            logger.debug(f"🎯 Start Selector: {start_node_idx} → Goal: {goal_idx}")

            return start_node_idx

        except Exception as e:
            # logger.error(f"❌ [Select Start Node] Error: {e}")
            # import traceback
            # traceback.print_exc()

            # 兜底：返回默认起点
            return self._get_default_start_node(state)
    def _get_default_start_node(self, state: Dict) -> int:
        """
        获取默认起点

        Args:
            state: 当前状态

        Returns:
            default_start: 默认起点索引
        """
        # 优先级1: 从state中获取source
        if hasattr(state, 'current_request') and state.current_request:
            return state.current_request.get('source', 0)

        # 优先级2: 从环境获取
        if hasattr(self, 'env') and hasattr(self.env, 'current_request'):
            if self.env.current_request:
                return self.env.current_request.get('source', 0)

        # 兜底：返回0
        return 0

    def _build_tree_mask(self, state: Dict) -> torch.Tensor:
        """
        🔥 构建树节点mask (支持 Tuple 解包)
        """
        # 1. 解包
        if isinstance(state, tuple):
            real_state = state[0]
        else:
            real_state = state

        # 2. 获取节点数
        if hasattr(real_state, 'x'):
            num_nodes = real_state.x.size(0)
        else:
            num_nodes = self.n_actions

        tree_mask = torch.zeros(num_nodes, device=self.device)

        # 3. 获取树节点信息（多种来源尝试）
        nodes_on_tree = None

        # 来源1: state.nodes_on_tree
        if hasattr(real_state, 'nodes_on_tree'):
            nodes_on_tree = real_state.nodes_on_tree

        # 来源2: state.tree_nodes
        elif hasattr(real_state, 'tree_nodes'):
            nodes_on_tree = real_state.tree_nodes

        # 来源3: 从环境获取
        elif hasattr(self, 'env'):
            if hasattr(self.env, 'nodes_on_tree'):
                nodes_on_tree = self.env.nodes_on_tree
            elif hasattr(self.env, 'current_tree') and self.env.current_tree:
                # ... (原有逻辑不变) ...
                placement = self.env.current_tree.get('placement', {})
                tree_edges = self.env.current_tree.get('tree', {})
                nodes_on_tree = set()
                for key in placement.keys():
                    if isinstance(key, tuple):
                        nodes_on_tree.add(key[0])
                    elif isinstance(key, int):
                        nodes_on_tree.add(key)
                for edge in tree_edges.keys():
                    if isinstance(edge, tuple) and len(edge) == 2:
                        nodes_on_tree.add(edge[0]);
                        nodes_on_tree.add(edge[1])
                nodes_on_tree = list(nodes_on_tree)

        # 4. 设置mask
        if nodes_on_tree is None or len(nodes_on_tree) == 0:
            # 树为空：只能从source出发
            source_node = self._get_default_start_node(real_state)
            if source_node < num_nodes:
                tree_mask[source_node] = 1
            else:
                tree_mask[0] = 1
        else:
            # 树不为空：可以从树上任意节点出发
            for node_idx in nodes_on_tree:
                if isinstance(node_idx, int) and 0 <= node_idx < num_nodes:
                    tree_mask[node_idx] = 1

        # 5. 验证
        if tree_mask.sum() == 0:
            tree_mask[:] = 1

        return tree_mask
    def _generate_goal_embedding(self, state: Dict):
        """生成goal embedding"""
        try:
            graph_emb = self._get_graph_embedding(state)

            with torch.no_grad():
                _, goal_emb, _ = self.high_policy(graph_emb, return_subgoal=True)

            self.current_goal_emb = goal_emb
            logger.warning(f"🔍 [_generate_goal_embedding] 设置goal_emb:")
            logger.warning(f"   - shape: {goal_emb.shape}")
            logger.warning(f"   - 前5维: {goal_emb[0, :5].cpu().numpy()}")
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
        🔥 [V41.2 修复版] 获取图嵌入 (自动处理 Tuple 和 Batch)
        """
        # 1. 自动解包 Tuple (obs, info)
        if isinstance(state, tuple):
            real_state = state[0]
        else:
            real_state = state

        # 2. 检查 Encoder
        if self.encoder is None:
            return torch.zeros(1, self.hidden_dim, device=self.device)

        # 3. 计算 Embedding
        try:
            # 情况 A: PyG Batch 对象 (训练时)
            if hasattr(real_state, 'batch') and real_state.batch is not None:
                return self.encoder(real_state.x, real_state.edge_index, real_state.batch)

            # 情况 B: 单个 Data 对象 (推理时)
            if hasattr(real_state, 'x'):
                # 构造一个全0的 batch 向量
                batch = torch.zeros(real_state.x.size(0), dtype=torch.long, device=self.device)
                return self.encoder(real_state.x, real_state.edge_index, batch)

        except Exception as e:
            # logger.warning(f"Encoder forward failed: {e}")
            pass

        # 4. 兜底返回 (防止崩溃)
        return torch.zeros(1, self.hidden_dim, device=self.device)
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
            logger.warning(f"🔍 [_generate_subgoal_embedding] 设置goal_emb:")
            logger.warning(f"   - shape: {goal_emb.shape}")
            logger.warning(f"   - 前5维: {goal_emb[0, :5].cpu().numpy()}")
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
        """存储Low-Level经验 (集成强力缩放)"""
        # 🔥 [关键修复] 强力奖励缩放 + 截断
        # 原始奖励可能高达 2500+，我们将其压缩到 [-5, 10] 区间
        scaled_reward = reward * 0.01  # 缩小100倍 (2500 -> 25.0)

        # 硬截断，防止极端值破坏 Q 值估计
        max_reward = 10.0
        min_reward = -5.0
        scaled_reward = max(min_reward, min(max_reward, scaled_reward))

        # 🔥 [可选] 添加 Intrinsic Reward (好奇心奖励)
        if self.config.get('hrl', {}).get('use_intrinsic_reward', False):
            try:
                # 简单的状态差异作为好奇心
                with torch.no_grad():
                    state_emb = self._extract_state_embedding(state)
                    next_state_emb = self._extract_state_embedding(next_state)
                    prediction_error = F.mse_loss(state_emb, next_state_emb).item()
                    # 限制内在奖励幅度
                    intrinsic_reward = min(0.1, prediction_error * 0.5)
                    scaled_reward += intrinsic_reward
            except Exception:
                pass

        self.low_memory.append({
            'state': state,
            'action': action,
            'reward': scaled_reward,  # 使用缩放后的奖励
            'next_state': next_state,
            'done': done,
            'goal_emb': self.current_goal_emb
        })

        # 缓冲区监控 (调试用)
        if len(self.low_memory) % 5000 == 0:
            logger.info(f"📊 Low Buffer Size: {len(self.low_memory)}")

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
        """更新策略（集成监控与调度）"""
        losses = {}
        self.update_count += 1

        # High-Level更新
        high_loss = 0.0
        if len(self.high_memory) >= self.batch_size // 4:
            high_loss = self._update_high_level()
            losses['high_loss'] = high_loss
            if high_loss > 0:
                self.high_loss_history.append(high_loss)

        # Low-Level更新
        low_loss = 0.0
        if len(self.low_memory) >= self.batch_size:
            low_loss = self._update_low_level()
            losses['low_loss'] = low_loss
            if low_loss > 0:
                self.low_loss_history.append(low_loss)

        losses['total_loss'] = losses.get('high_loss', 0) + low_loss

        # 🔥 软更新 target networks (更稳定)
        self._soft_update_target_networks()

        # 定期硬更新 target networks
        if self.update_count % self.target_update_freq == 0:
            self._hard_update_target_networks()

        # 更新 epsilon
        self._update_epsilon()

        # 🔥 监控训练状态
        if self.update_count % 100 == 0:
            self._log_training_stats()

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

                # 🔥 限制目标Q值范围
                target_q = torch.clamp(target_q, -10.0, 50.0)

            # 6. 计算 Loss & 更新
            # 🔥 使用Huber Loss提高稳定性
            loss = F.smooth_l1_loss(curr_q, target_q, reduction='mean')

            # 检查 Loss 有效性
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("❌ High-Level Loss 出现NaN/Inf，跳过更新")
                return 0.0

            self.optimizer_high.zero_grad()
            loss.backward()

            # 🔥 梯度监控
            total_grad_norm = 0.0
            for param in self.high_policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            self.gradient_norms.append(total_grad_norm)

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.high_policy.parameters(), self.clip_grad_norm)
            self.optimizer_high.step()

            return loss.item()

        except Exception as e:
            logger.error(f"[Update High Level] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _update_low_level(self) -> float:
        """更新Low-Level策略 (集成 Q值截断、梯度裁剪、自适应LR)"""
        if len(self.low_memory) < self.batch_size:
            return 0.0

        try:
            # 2. 采样
            batch = random.sample(self.low_memory, self.batch_size)

            # 3. 准备数据
            # 使用新添加的 _extract_state_embedding 方法
            state_embs = [self._extract_state_embedding(x['state']) for x in batch]
            next_state_embs = [self._extract_state_embedding(x['next_state']) for x in batch]

            state_tensor = torch.cat(state_embs, dim=0).to(self.device)
            next_state_tensor = torch.cat(next_state_embs, dim=0).to(self.device)

            actions = torch.tensor([x['action'] for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward'] for x in batch], device=self.device).float().unsqueeze(1)
            dones = torch.tensor([x['done'] for x in batch], device=self.device).float().unsqueeze(1)

            # 🔥 [修复] 过滤无效动作 (-1)
            valid_mask = (actions >= 0).squeeze()
            if valid_mask.sum() == 0: return 0.0

            state_tensor = state_tensor[valid_mask]
            next_state_tensor = next_state_tensor[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            dones = dones[valid_mask]

            # 🔥 [修复] Reward 二次检查 (防止存入后的异常值)
            rewards = torch.clamp(rewards, -10.0, 10.0)

            # 处理 Goal Embedding
            goal_embs = []
            valid_indices = torch.nonzero(valid_mask).squeeze().cpu().tolist()
            if not isinstance(valid_indices, list): valid_indices = [valid_indices]

            for idx in valid_indices:
                x = batch[idx]
                g = x.get('goal_emb')
                if g is None:
                    g = torch.zeros(1, self.goal_dim, device=self.device)
                else:
                    g = g.to(self.device)
                    if g.dim() == 1: g = g.unsqueeze(0)
                    if g.size(1) != self.goal_dim:
                        if g.size(1) > self.goal_dim:
                            g = g[:, :self.goal_dim]
                        else:
                            padding = torch.zeros(g.size(0), self.goal_dim - g.size(1), device=self.device)
                            g = torch.cat([g, padding], dim=1)
                goal_embs.append(g)

            goal_tensor = torch.cat(goal_embs, dim=0).to(self.device)

            # 4. 计算 Current Q
            policy_output = self.low_policy(state_tensor, goal_tensor)
            if isinstance(policy_output, tuple):
                curr_q_values = policy_output[0]
            else:
                curr_q_values = policy_output

            curr_q = curr_q_values.gather(1, actions)

            # 🔥 [监控] Q 值异常检测
            if torch.isnan(curr_q).any() or torch.isinf(curr_q).any():
                logger.error("❌ Q 值出现 NaN/Inf，触发重置！")
                self.reset_network_parameters()
                return 0.0

            # 5. 计算 Target Q (Double DQN)
            with torch.no_grad():
                # Online Net 选动作
                next_output = self.low_policy(next_state_tensor, goal_tensor)
                next_q_online = next_output[0] if isinstance(next_output, tuple) else next_output
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Target Net 评价值
                target_output = self.target_low_policy(next_state_tensor, goal_tensor)
                next_q_target = target_output[0] if isinstance(target_output, tuple) else target_output
                next_q = next_q_target.gather(1, next_actions)

                # 🔥 [关键修复] Target Q 截断
                # 理论最大 Q ≈ 10 / (1-0.99) = 1000
                # 这里限制在 [-20, 100] 防止过估计
                next_q = torch.clamp(next_q, -20.0, 100.0)

                target_q = rewards + (1 - dones) * self.gamma * next_q

                # 🔥 [关键修复] 最终 Target 二次截断
                target_q = torch.clamp(target_q, -20.0, 120.0)

            # 6. 计算 Loss & 更新
            # 🔥 [修复] 使用 Huber Loss (Smooth L1 Loss) 提高稳定性
            loss = F.smooth_l1_loss(curr_q, target_q, reduction='mean')

            # 检查 Loss 有效性
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("❌ Low-Level Loss 出现NaN/Inf，跳过更新")
                return 0.0

            self.optimizer_low.zero_grad()
            loss.backward()

            # 🔥 [修复] 梯度监控与裁剪
            total_grad_norm = 0.0
            for param in self.low_policy.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()

            self.gradient_norms.append(total_grad_norm)

            # 严格裁剪梯度
            nn.utils.clip_grad_norm_(self.low_policy.parameters(), self.clip_grad_norm)

            self.optimizer_low.step()

            # 🔥 [修复] 自适应学习率微调 (简单版)
            loss_val = loss.item()
            if loss_val < 1e-4:  # Loss 太小，可能是学习率太低或陷入局部极小
                for param_group in self.optimizer_low.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.02, 1e-3)
            elif loss_val > 5.0:  # Loss 太大
                for param_group in self.optimizer_low.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.98, 1e-5)

            return loss_val

        except Exception as e:
            logger.error(f"[Update Low Level] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _soft_update_target_networks(self):
        """软更新target networks（更稳定）"""
        # 软更新High-Level
        for target_param, param in zip(self.target_high_policy.parameters(), self.high_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # 软更新Low-Level
        for target_param, param in zip(self.target_low_policy.parameters(), self.low_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update_target_networks(self):
        """硬更新target networks"""
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

    def _log_training_stats(self):
        """记录训练统计"""
        if len(self.high_loss_history) > 0 and len(self.low_loss_history) > 0:
            avg_high_loss = np.mean(self.high_loss_history)
            avg_low_loss = np.mean(self.low_loss_history)
            avg_grad_norm = np.mean(self.gradient_norms) if self.gradient_norms else 0.0

            logger.debug(f"📊 训练统计: HighLoss={avg_high_loss:.4f}, LowLoss={avg_low_loss:.4f}, "
                         f"GradNorm={avg_grad_norm:.2f}, ε_low={self.epsilon_low:.3f}")

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
        """
        🔥 [V2.1 最终修复版] 低层动作选择（黑名单感知 + 严格Mask处理）

        关键修复：
        1. ✅ Q值屏蔽法（不是权重融合）
        2. ✅ Mask在argmax之前应用
        3. ✅ 探索时从有效动作采样
        4. 🔥 新增：最终验证（三重保险）
        """
        try:
            # 1. 获取状态嵌入
            state_emb = self._extract_state_embedding(state)
            if self.current_goal_emb is None:
                self._generate_goal_embedding(state)

            # 2. 🔥 准备完整的Mask（融合action_mask和blacklist）
            if action_mask is not None and len(action_mask) > 0:
                # 确保mask长度正确
                if len(action_mask) < self.n_actions:
                    # 如果mask太短，填充0
                    full_mask = np.zeros(self.n_actions, dtype=np.float32)
                    full_mask[:len(action_mask)] = action_mask
                else:
                    full_mask = action_mask[:self.n_actions].copy()
            else:
                # 没有mask，默认全部允许
                full_mask = np.ones(self.n_actions, dtype=np.float32)

            # 3. 应用黑名单（降低权重而非完全禁止）
            if blacklist_info:
                blacklist_nodes = blacklist_info.get('nodes', [])
                for node in blacklist_nodes:
                    if 0 <= node < self.n_actions:
                        full_mask[node] *= 0.1  # 降低到10%

            # 4. 检查是否有有效动作
            valid_actions = np.where(full_mask > 0)[0]
            if len(valid_actions) == 0:
                logger.critical("🚨 [Agent] 完全没有有效动作！")
                # 完全没有有效动作，返回0作为STAY
                return 0

            # 5. 动作选择（epsilon-greedy）
            if random.random() < self.epsilon_low:
                # ========== 探索：基于mask权重的概率采样 ==========
                action_weights = full_mask[valid_actions]
                p = action_weights / action_weights.sum()
                action = int(np.random.choice(valid_actions, p=p))

                # 🔥 探索也要验证
                if full_mask[action] <= 0:
                    logger.warning(f"⚠️ [Agent] 探索选了无效动作{action}，重新采样")
                    action = int(valid_actions[0])

                return action

            else:
                # ========== 利用：Q网络决策 ==========
                if self.low_policy is not None:
                    with torch.no_grad():
                        # 前向传播获取Q值
                        policy_output = self.low_policy(
                            state_emb,
                            self.current_goal_emb.to(state_emb.device)
                        )

                        # 处理可能的tuple返回
                        if isinstance(policy_output, tuple):
                            q_values = policy_output[0].cpu().numpy().flatten()
                        else:
                            q_values = policy_output.cpu().numpy().flatten()

                        # 确保Q值长度正确
                        if len(q_values) < self.n_actions:
                            # Q值太短，填充最小值
                            full_q = np.full(self.n_actions, -1e9, dtype=np.float32)
                            full_q[:len(q_values)] = q_values
                            q_values = full_q
                        elif len(q_values) > self.n_actions:
                            q_values = q_values[:self.n_actions]

                    # 🔥🔥🔥 关键修复1：屏蔽<=0的位置（不只是==0）
                    masked_q_values = q_values.copy()
                    masked_q_values[full_mask <= 0] = -1e9  # ← 改成 <= 0

                    # argmax选择最大Q值
                    action = int(np.argmax(masked_q_values))

                    # 🔥🔥🔥 关键修复2：最终验证（三重保险）
                    if full_mask[action] <= 0:
                        logger.critical(f"🚨 [Agent] 仍选了无效动作{action}！")
                        logger.critical(f"   Mask值: {full_mask[action]}")
                        logger.critical(f"   Q值: {q_values[action]}")
                        logger.critical(f"   Masked Q值: {masked_q_values[action]}")

                        # 强制选择有效动作
                        if len(valid_actions) > 0:
                            action = int(valid_actions[0])
                            logger.warning(f"   强制修正为{action}")
                        else:
                            action = 0
                            logger.critical("   没有有效动作，返回0")

                    return action

                else:
                    # 没有policy，随机从有效动作选择
                    return int(np.random.choice(valid_actions))

        except Exception as e:
            logger.error(f"[Select Low Action] Error: {e}")
            import traceback
            traceback.print_exc()

            # 紧急fallback
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    return int(valid_actions[0])

            return 0

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

    def _extract_state_embedding(self, state):
        """🔥 [新增] 提取状态嵌入（兼容性方法）"""
        try:
            return self._get_graph_embedding(state)
        except Exception as e:
            logger.error(f"[Extract State Embedding] Error: {e}")
            # 返回随机嵌入作为 fallback，确保维度匹配
            return torch.randn(1, self.state_dim, device=self.device)

    def reset_network_parameters(self):
        """🔥 [新增] 重置网络参数（用于训练异常时的自愈）"""
        logger.warning("🔄 [Auto-Fix] 正在重置网络参数...")

        # 重置 High-Level 网络
        if hasattr(self.high_policy, 'reset_parameters'):
            self.high_policy.reset_parameters()
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())

        # 重置 Low-Level 网络
        if hasattr(self.low_policy, 'reset_parameters'):
            self.low_policy.reset_parameters()
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())

        # 重置优化器 (恢复初始学习率)
        training_cfg = self.config.get('training', {})
        lr_high = training_cfg.get('lr_high', 1e-4)
        lr_low = training_cfg.get('lr_low', 1e-4)

        self.optimizer_high = optim.Adam(self.high_policy.parameters(), lr=lr_high)
        self.optimizer_low = optim.Adam(self.low_policy.parameters(), lr=lr_low)

        logger.info("✅ 网络参数重置完成")

    def _force_select_start_node(self, state, subgoal):
        """
        🔥 [V41.1 修复版] 强制使用高层策略选择起点 (支持 tuple 解包)
        """
        try:
            # 🔥🔥🔥 核心修复：检测并解包 tuple state (obs, info)
            if isinstance(state, tuple):
                # 通常格式为 (data, info)
                real_state = state[0]
            else:
                real_state = state

            # 1. 获取节点嵌入（原始GNN输出）
            if self.encoder is not None:
                # 检查 real_state 是否有 x 属性
                if hasattr(real_state, 'x') and real_state.x is not None:
                    batch_vector = torch.zeros(real_state.x.size(0), dtype=torch.long, device=self.device)
                    node_embeddings = self.encoder(real_state.x, real_state.edge_index, batch_vector)
                else:
                    # 如果没有 x 属性，可能是非 PyG 数据，记录警告并返回
                    # logger.warning(f"State类型 {type(real_state)} 缺少x属性，无法使用Encoder")
                    return self._get_default_start_node(real_state)
            else:
                # 备用模式：没有 Encoder，直接使用 state.x
                if hasattr(real_state, 'x'):
                    node_embeddings = real_state.x
                else:
                    return self._get_default_start_node(real_state)

            # 2. 获取目标节点特征 (Target Embedding)
            if isinstance(subgoal, int) and 0 <= subgoal < node_embeddings.size(0):
                target_raw = node_embeddings[subgoal].unsqueeze(0)
            else:
                # 目标节点无效，使用零向量
                target_raw = torch.zeros(1, node_embeddings.size(1), device=self.device)

            # 3. 获取目标的投影嵌入
            # HighPolicy 内部会处理投影，这里传入原始特征即可
            target_emb = target_raw

            # 4. 构建起点选择 mask
            num_nodes = node_embeddings.size(0)
            start_mask = torch.ones(num_nodes, device=self.device)

            # 基本规则：不能从目标节点本身出发
            if isinstance(subgoal, int) and 0 <= subgoal < num_nodes:
                start_mask[subgoal] = 0

            # 5. 调用高层策略选择起点
            is_training = getattr(self, '_training', True)
            start_node, log_prob = self.high_policy.select_start_node(
                node_embeddings=node_embeddings,
                target_emb=target_emb,
                tree_mask=start_mask,
                sample=is_training
            )

            # 6. 更新环境当前位置
            if hasattr(self, 'env') and self.env is not None:
                self.env.current_node_location = start_node
                # 打印日志确认
                logger.info(f"🤖 [Start Head] 选中起点: {start_node} (目标: {subgoal})")

            logger.info(f"🚀 [Start Selection] 激活！起点: {start_node} → 目标: {subgoal}")

            return start_node

        except Exception as e:
            # 只有在真正出错时才打印 Error
            # logger.error(f"❌ 起点选择失败: {e}")
            # import traceback
            # traceback.print_exc()

            # 兜底：返回默认起点
            default_start = self._get_default_start_node(state)
            if hasattr(self, 'env') and self.env is not None:
                self.env.current_node_location = default_start
            return default_start
# ============================================
# 向后兼容：保留旧接口
# ============================================

class GoalConditionedHRLAgent(HRLAgent):
    """向后兼容的GoalConditionedHRLAgent"""

    def __init__(self, config, phase=3, goal_strategy='adaptive', **kwargs):
        logger.warning("⚠️ GoalConditionedHRLAgent已重构为HRLAgent，使用兼容模式")
        super().__init__(config, phase=phase, goal_strategy=goal_strategy, **kwargs)


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