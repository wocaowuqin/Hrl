#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentAction — 动作选择 + 子目标管理 + 嵌入计算
"""

import torch
import numpy as np
import random
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HRLAgentAction:
    """
    负责：
    - select_action: 对外统一接口
    - _need_new_subgoal: 触发条件判断
    - _select_subgoal: High-Level epsilon-greedy
    - _select_start_node / _get_default_start_node / _build_tree_mask
    - _select_low_action: Low-Level执行
    - _get_local_embedding / _get_graph_embedding: 嵌入计算
    - _generate_goal_embedding / _generate_and_encode_subgoal: 向后兼容
    """

    def select_action(
            self,
            state: Dict,
            unconnected_dests: Optional[list] = None,
            action_mask: Optional[np.ndarray] = None,
            use_expert: bool = False,
            expert_action: Optional[int] = None,
    ) -> Tuple[int, int, Dict]:
        """[V55.6] 分层动作选择入口"""
        info = {
            'high_level_decision': False,
            'subgoal': self.current_subgoal,
            'subgoal_steps': self.subgoal_steps,
            'source': 'agent',
            'start_node': None,
        }
        try:
            # 1. High-Level
            if self._need_new_subgoal(state, unconnected_dests):
                self.current_subgoal = self._select_subgoal(state, unconnected_dests, action_mask)
                self.subgoal_steps = 0
                info.update({
                    'high_level_decision': True,
                    'subgoal': self.current_subgoal,
                    'source': 'agent_high',
                    'start_node': int(getattr(self.env, 'current_node_location', 0))
                                  if self.env is not None else 0,
                })

            # 2. Low-Level
            low_action = self._select_low_action(state, action_mask)
            self.subgoal_steps += 1
            self.subgoal_step_count = self.subgoal_steps
            info['low_action'] = low_action

            # 3. 返回值
            if unconnected_dests is not None:
                high_action = (unconnected_dests.index(self.current_subgoal)
                               if self.current_subgoal in unconnected_dests else 0)
            else:
                high_action = self.current_subgoal if self.current_subgoal is not None else 0

            return high_action, low_action, info

        except Exception as e:
            logger.error(f"❌ [Select Action] 异常: {e}")
            import traceback; traceback.print_exc()
            return 0, 0, info

    # ── 子目标触发 ────────────────────────────────────────────────────────

    def _need_new_subgoal(self, state: Dict, unconnected_dests: Optional[list]) -> bool:
        if self.current_subgoal is None:
            return True
        if not unconnected_dests:
            return False
        if self.current_subgoal not in unconnected_dests:
            logger.debug(f"[NeedNew] 子目标 {self.current_subgoal} 已连接")
            return True
        if self.subgoal_steps >= self.subgoal_horizon:
            logger.debug(f"[NeedNew] 超时 steps={self.subgoal_steps}")
            return True
        if state.get('current_position', -1) == self.current_subgoal:
            return True

        # VNF阶段：资源不足检测
        if self.env is not None and getattr(self.env, 'current_phase', None) == 'vnf_deployment':
            try:
                vnf_list = self.env.current_request.get('vnf', [])
                vnf_idx  = getattr(self.env, 'next_vnf_idx', 0)
                if vnf_idx < len(vnf_list):
                    cpu_req = self.env.current_request.get('cpu_origin', [])[vnf_idx]
                    mem_req = self.env.current_request.get('memory_origin', [])[vnf_idx]
                    avail_cpu = self.env.resource_mgr.pool.get_available_cpu(self.current_subgoal)
                    avail_mem = self.env.resource_mgr.pool.get_available_memory(self.current_subgoal)
                    if avail_cpu < cpu_req or avail_mem < mem_req:
                        logger.warning(f"[NeedNew] 节点{self.current_subgoal}资源不足，强制重选")
                        return True
            except Exception as e:
                logger.error(f"[NeedNew] 资源检查异常: {e}")
            # 非DC节点检查
            dc_nodes = getattr(self.env, 'dc_nodes', [])
            if self.current_subgoal not in dc_nodes:
                logger.warning(f"[NeedNew] {self.current_subgoal} 不是DC节点")
                return True

        return False

    # ── High-Level 子目标选择 ─────────────────────────────────────────────

    def _select_subgoal(self, state, unconnected_dests, action_mask=None) -> int:
        graph_emb = self._get_graph_embedding(state)
        with torch.no_grad():
            q_values, goal_emb, _ = self.high_policy(graph_emb, return_subgoal=True)

        # Mask处理
        effective_mask = torch.ones_like(q_values)
        masked_q = q_values.clone()
        if action_mask is not None:
            mt = torch.tensor(action_mask, device=self.device).float()
            if mt.shape[-1] != q_values.shape[-1]:
                logger.error(f"❌ [MASK] 维度不匹配! Mask={mt.shape[-1]}, Q={q_values.shape[-1]}")
                mt = torch.ones_like(q_values)
            if mt.dim() == 1:
                mt = mt.unsqueeze(0)
            masked_q = torch.where(mt > 0, q_values, torch.tensor(-1e9, device=self.device))
            effective_mask = mt

        valid_indices = torch.nonzero(effective_mask.squeeze() > 0).flatten()

        if random.random() < self.epsilon_high:
            # 探索
            if len(valid_indices) > 0:
                goal_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()].item()
                if action_mask is not None and action_mask[goal_idx] == 0:
                    goal_idx = valid_indices[0].item()
                logger.debug(f"[EXPLORE] 随机选中: {goal_idx}")
            else:
                logger.error("💀 [FAIL] Mask全0，强制返回0")
                goal_idx = 0
        else:
            # 贪婪
            goal_idx = masked_q.argmax(dim=1).item()
            if action_mask is not None and len(action_mask) > goal_idx and action_mask[goal_idx] == 0:
                logger.critical(f"💀 贪婪选中被Mask节点{goal_idx}，强制修正")
                if len(valid_indices) > 0:
                    goal_idx = valid_indices[0].item()

        subgoal = int(goal_idx)
        # 更新goal embedding
        if hasattr(state, 'x') and state.x is not None and subgoal < state.x.size(0):
            nf = state.x[subgoal]
            if nf.size(0) > self.goal_dim:
                nf = nf[:self.goal_dim]
            elif nf.size(0) < self.goal_dim:
                nf = torch.cat([nf, torch.zeros(self.goal_dim - nf.size(0), device=self.device)])
            self.current_goal_emb = nf.unsqueeze(0)
        else:
            self.current_goal_emb = goal_emb
        return subgoal

    # ── 起点选择 ─────────────────────────────────────────────────────────

    def _select_start_node(self, state, goal_idx: int, graph_emb) -> int:
        try:
            real_state = state[0] if isinstance(state, tuple) else state
            if not (hasattr(real_state, 'x') and hasattr(real_state, 'edge_index')):
                return self._get_default_start_node(real_state)

            with torch.no_grad():
                node_embeddings = (self.encoder(real_state.x, real_state.edge_index)
                                   if self.encoder is not None else real_state.x)
                target_emb = self.high_policy.state_projection(graph_emb)

            tree_mask = self._build_tree_mask(real_state)
            if tree_mask.sum() == 0:
                return self._get_default_start_node(real_state)

            with torch.no_grad():
                start_node_idx, _ = self.high_policy.select_start_node(
                    node_embeddings=node_embeddings,
                    target_emb=target_emb,
                    tree_mask=tree_mask,
                    sample=True,
                )
            logger.debug(f"🎯 Start Selector: {start_node_idx} → Goal: {goal_idx}")
            return start_node_idx
        except Exception:
            return self._get_default_start_node(state)

    def _get_default_start_node(self, state) -> int:
        logger.debug("[DefaultStart] 被调用")
        try:
            real_state = state[0] if isinstance(state, tuple) else state
            if hasattr(real_state, 'current_request') and real_state.current_request:
                src = real_state.current_request.get('source')
                if src is not None:
                    return int(src)
        except Exception:
            pass
        try:
            if self.env is not None and hasattr(self.env, 'current_request') and self.env.current_request:
                src = self.env.current_request.get('source')
                if src is not None:
                    return int(src)
        except Exception:
            pass
        logger.warning("[DefaultStart] 未找到源节点，返回0")
        return 0

    def _build_tree_mask(self, state) -> torch.Tensor:
        real_state = state[0] if isinstance(state, tuple) else state
        num_nodes = real_state.x.size(0) if hasattr(real_state, 'x') else self.n_actions
        tree_mask = torch.zeros(num_nodes, device=self.device)

        nodes_on_tree = None
        if hasattr(real_state, 'nodes_on_tree'):
            nodes_on_tree = real_state.nodes_on_tree
        elif hasattr(real_state, 'tree_nodes'):
            nodes_on_tree = real_state.tree_nodes
        elif self.env is not None:
            if hasattr(self.env, 'nodes_on_tree'):
                nodes_on_tree = self.env.nodes_on_tree
            elif hasattr(self.env, 'current_tree') and self.env.current_tree:
                placement  = self.env.current_tree.get('placement', {})
                tree_edges = self.env.current_tree.get('tree', {})
                s = set()
                for k in placement:
                    s.add(k[0] if isinstance(k, tuple) else k)
                for e in tree_edges:
                    if isinstance(e, tuple) and len(e) == 2:
                        s.update(e)
                nodes_on_tree = list(s)

        if not nodes_on_tree:
            src = self._get_default_start_node(real_state)
            tree_mask[src if src < num_nodes else 0] = 1
        else:
            for n in nodes_on_tree:
                if isinstance(n, int) and 0 <= n < num_nodes:
                    tree_mask[n] = 1
        if tree_mask.sum() == 0:
            tree_mask[:] = 1
        return tree_mask

    # ── Low-Level 动作选择 ────────────────────────────────────────────────

    def _select_low_action(self, state, action_mask: Optional[np.ndarray]) -> int:
        # [SDG-HRL] 局部嵌入比全图均值更适合下一跳选择
        graph_emb = self._get_local_embedding(state)
        if self.current_goal_emb is None:
            self._generate_goal_embedding(state)
        mask_tensor = (torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                       if action_mask is not None else None)
        with torch.no_grad():
            action, _ = self.low_policy.select_action(
                graph_emb, self.current_goal_emb, mask_tensor, epsilon=self.epsilon_low)
        return action.item()

    # ── 嵌入计算 ─────────────────────────────────────────────────────────

    def _build_req_vec(self):
        """从 env.current_request 构建请求特征向量 [1, 3]（bw, avg_cpu, avg_mem）。
        若 encoder 未启用 req_fc（req_dim=0）则返回 None，避免无谓计算。"""
        if self.encoder is None or not getattr(self.encoder, 'req_fc', None):
            return None
        try:
            req = (self.env.current_request
                   if self.env is not None and hasattr(self.env, 'current_request')
                   else None)
            if req is None:
                return None
            bw  = float(req.get('bw_origin', req.get('bw', 0.0)))
            cpu_list = req.get('cpu_origin', req.get('cpu', []))
            mem_list = req.get('memory_origin', req.get('memory', []))
            avg_cpu = float(np.mean(cpu_list)) if len(cpu_list) > 0 else 0.0
            avg_mem = float(np.mean(mem_list)) if len(mem_list) > 0 else 0.0
            return torch.tensor([[bw, avg_cpu, avg_mem]], dtype=torch.float32,
                                 device=self.device)
        except Exception:
            return None

    def _build_dest_mask(self, state, n_nodes: int):
        """从 env.unconnected_dests 构建目标节点掩码 [N]（bool Tensor）。"""
        try:
            dests = None
            if self.env is not None and hasattr(self.env, 'unconnected_dests'):
                dests = self.env.unconnected_dests
            if not dests:
                real_state = state[0] if isinstance(state, tuple) else state
                dests = getattr(real_state, 'unconnected_dests', None)
            if dests:
                mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
                for d in dests:
                    if isinstance(d, int) and 0 <= d < n_nodes:
                        mask[d] = True
                return mask if mask.any() else None
        except Exception:
            pass
        return None

    def _get_local_embedding(self, state):
        """[TA-HGRL Fix] 返回完整节点序列 [1, N, H]，保留N维供Attention审视"""
        if self.encoder is None:
            return self._get_graph_embedding(state)
        real_state = state[0] if isinstance(state, tuple) else state
        if not hasattr(real_state, 'x'):
            return self._get_graph_embedding(state)
        try:
            with torch.no_grad():
                _x  = real_state.x.to(self.device)
                _ei = (real_state.edge_index.to(self.device)
                       if real_state.edge_index is not None
                       else (self.env.edge_index.to(self.device)
                             if self.env is not None
                             and hasattr(self.env, 'edge_index')
                             and self.env.edge_index is not None else None))
                if _ei is None:
                    return self._get_graph_embedding(state)
                _ea = getattr(real_state, 'edge_attr', None)
                _ea = (_ea.to(self.device) if _ea is not None
                       else torch.zeros(_ei.shape[1], 5, device=self.device))
                _b    = torch.zeros(_x.size(0), dtype=torch.long, device=self.device)
                _tei  = getattr(real_state, 'tree_edge_index', None)
                if _tei is not None: _tei = _tei.to(self.device)
                _req  = self._build_req_vec()
                _dest = self._build_dest_mask(state, _x.size(0))
                node_out = self.encoder(_x, _ei, _ea, batch=_b,
                                        tree_edge_index=_tei,
                                        dest_mask=_dest,
                                        req_vec=_req)            # [N, H]

            # [TA-HGRL Fix] 保留N维：返回 [1, N, H]，让Goal-Attention看到全网节点
            return node_out.unsqueeze(0)
        except Exception:
            return self._get_graph_embedding(state)

    def _get_graph_embedding(self, state, req=None):
        """全图均值嵌入（兜底用）
        req: 可选，传入历史请求字典（high-level训练时传入），
        鲁棒以免用当前 env.current_request 污染历史 embedding
        """
        real_state = state[0] if isinstance(state, tuple) else state
        if self.encoder is None:
            return torch.zeros(1, self.hidden_dim, device=self.device)
        try:
            _tei = getattr(real_state, 'tree_edge_index', None)
            if _tei is not None: _tei = _tei.to(self.device)
            # 优先用传入的历史req，其次才读 env.current_request
            if req is not None and (self.encoder is not None and getattr(self.encoder, 'req_fc', None)):
                try:
                    bw      = float(req.get('bw_origin', req.get('bw', 0.0)))
                    cpu_lst = req.get('cpu_origin', req.get('cpu', []))
                    mem_lst = req.get('memory_origin', req.get('memory', []))
                    avg_cpu = float(np.mean(cpu_lst)) if len(cpu_lst) > 0 else 0.0
                    avg_mem = float(np.mean(mem_lst)) if len(mem_lst) > 0 else 0.0
                    _req = torch.tensor([[bw, avg_cpu, avg_mem]],
                                        dtype=torch.float32, device=self.device)
                except Exception:
                    _req = self._build_req_vec()
            else:
                _req = self._build_req_vec()
            _dest = (self._build_dest_mask(state, real_state.x.size(0))
                     if hasattr(real_state, 'x') else None)
            if hasattr(real_state, 'batch') and real_state.batch is not None:
                out = self.encoder(real_state.x, real_state.edge_index,
                                   tree_edge_index=_tei,
                                   dest_mask=_dest, req_vec=_req)
                return out.mean(dim=0, keepdim=True)
            if hasattr(real_state, 'x'):
                b = torch.zeros(real_state.x.size(0), dtype=torch.long, device=self.device)
                out = self.encoder(real_state.x, real_state.edge_index,
                                   batch=b, tree_edge_index=_tei,
                                   dest_mask=_dest, req_vec=_req)
                return out.mean(dim=0, keepdim=True)
        except Exception:
            pass
        return torch.zeros(1, self.hidden_dim, device=self.device)

    def _extract_state_embedding(self, state):
        try:
            return self._get_graph_embedding(state)
        except Exception as e:
            logger.error(f"[Extract State Embedding] Error: {e}")
            return torch.randn(1, self.state_dim, device=self.device)

    # ── Goal Embedding 生成 ───────────────────────────────────────────────

    def _generate_goal_embedding(self, state):
        try:
            graph_emb = self._get_graph_embedding(state)
            with torch.no_grad():
                _, goal_emb, _ = self.high_policy(graph_emb, return_subgoal=True)
            self.current_goal_emb = goal_emb
            logger.debug(f"[_generate_goal_embedding] shape={goal_emb.shape}")
        except Exception as e:
            logger.error(f"[Goal Embedding] Error: {e}")
            self.current_goal_emb = torch.zeros(1, self.goal_dim, device=self.device)

    def _generate_and_encode_subgoal(self, state):
        """向后兼容：旧版方法名，调用新版逻辑"""
        try:
            graph_emb = self._get_graph_embedding(state)
            with torch.no_grad():
                if self.goal_strategy == 'adaptive':
                    complexity  = torch.tensor([[0.5]], device=self.device)
                    subgoal_emb, _ = self.goal_embedding(graph_emb, complexity)
                    goal_emb = subgoal_emb[..., :self.goal_dim]
                elif self.goal_strategy == 'hybrid':
                    subgoal_emb, goal_emb, _ = self.goal_embedding(graph_emb)
                else:
                    goal_emb, _ = self.goal_embedding(graph_emb, torch.randn_like(graph_emb))
                    subgoal_emb = goal_emb

            # 对齐维度
            def _trim(t): return t[..., :self.goal_dim] if t.shape[-1] >= self.goal_dim else \
                torch.cat([t, torch.zeros(*t.shape[:-1], self.goal_dim - t.shape[-1], device=self.device)], dim=-1)
            self.current_subgoal_emb = _trim(subgoal_emb)
            self.current_goal_emb    = _trim(goal_emb)
            self.subgoal_step_count  = 0
            logger.debug(f"[_generate_and_encode_subgoal] shape={goal_emb.shape}")
        except Exception as e:
            logger.error(f"[Generate Subgoal] Error: {e}")
            self.current_subgoal_emb = torch.zeros(1, self.goal_dim, device=self.device)
            self.current_goal_emb    = torch.zeros(1, self.goal_dim, device=self.device)
            self.subgoal_step_count  = 0