#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentTrain — 策略更新（Double DQN）+ epsilon衰减 + soft update
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HRLAgentTrain:
    """
    负责：
    - update / update_policies: 训练调度
    - _update_high_level: High-Level Double DQN
    - _update_low_level:  Low-Level Double DQN + success mix + Q监控
    - _soft_update_target_networks / _hard_update_target_networks
    - _update_epsilon / update_epsilon
    - _log_training_stats
    """

    def update(self) -> float:
        """向后兼容的update接口"""
        losses = self.update_policies()
        return losses.get('high_loss', 0.0) + losses.get('low_loss', 0.0)

    def update_policies(self) -> Dict[str, float]:
        losses = {}
        self.update_count += 1

        if len(self.high_memory) >= self.batch_size // 4:
            hl = self._update_high_level()
            losses['high_loss'] = hl
            if hl > 0:
                self.high_loss_history.append(hl)

        if len(self.low_memory) >= self.batch_size // 2:
            ll = self._update_low_level()
            losses['low_loss'] = ll
            if ll > 0:
                self.low_loss_history.append(ll)

        losses['total_loss'] = losses.get('high_loss', 0) + losses.get('low_loss', 0)

        # [SDG-HRL] 只用soft update，删除hard update防止Q值震荡
        self._soft_update_target_networks()
        self._update_epsilon()

        if self.update_count % 100 == 0:
            self._log_training_stats()

        return losses

    # ── High-Level ────────────────────────────────────────────────────────

    def _update_high_level(self) -> float:
        if len(self.high_memory) < self.batch_size:
            return 0.0
        try:
            if getattr(self, '_use_per', False):
                batch, _, _ = self.high_memory.sample(self.batch_size, beta=0.4)
                if not batch:
                    return 0.0
            else:
                batch = random.sample(self.high_memory, self.batch_size)

            state_tensor      = torch.cat([self._get_graph_embedding(x['state'])      for x in batch]).to(self.device)
            next_state_tensor = torch.cat([self._get_graph_embedding(x['next_state']) for x in batch]).to(self.device)
            goals   = torch.tensor([x['goal']   for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward'] for x in batch], device=self.device).float().unsqueeze(1)
            dones   = torch.tensor([x['done']   for x in batch], device=self.device).float().unsqueeze(1)

            curr_q_values, _, _ = self.high_policy(state_tensor, return_subgoal=False)
            curr_q = curr_q_values.gather(1, goals)

            with torch.no_grad():
                next_q_online, _, _ = self.high_policy(next_state_tensor, return_subgoal=False)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target, _, _ = self.target_high_policy(next_state_tensor, return_subgoal=False)
                next_q   = next_q_target.gather(1, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * next_q
                target_q = torch.clamp(target_q, -30.0, 150.0)  # [SDG-HRL]

            # [Loss Fix] reward归一化后loss应在0~3，scale=30让reward范围≈[-1,5]
            _reward_scale = 30.0
            curr_q_scaled  = curr_q  / _reward_scale
            target_q_scaled = target_q / _reward_scale
            loss = F.smooth_l1_loss(curr_q_scaled, target_q_scaled)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("❌ High-Level Loss NaN/Inf，跳过")
                return 0.0

            self.optimizer_high.zero_grad()
            loss.backward()
            # 梯度监控
            self.gradient_norms.append(
                sum(p.grad.norm().item() for p in self.high_policy.parameters() if p.grad is not None))
            nn.utils.clip_grad_norm_(self.high_policy.parameters(), self.clip_grad_norm)
            self.optimizer_high.step()
            return loss.item()

        except Exception as e:
            logger.error(f"[Update High Level] Error: {e}")
            import traceback; traceback.print_exc()
            return 0.0

    # ── Low-Level ─────────────────────────────────────────────────────────

    def _update_low_level(self) -> float:
        if len(self.low_memory) < self.batch_size // 2:
            return 0.0
        try:
            # [SDG-HRL] 混合采样：60%PER + 20%成功池 + 20%Elite
            _per_indices = None
            _per_weights = None
            if getattr(self, '_use_per', False) and len(self.low_memory) >= self.batch_size // 2:
                per_size = int(self.batch_size * 0.6)
                suc_size = int(self.batch_size * 0.2)
                eli_size = self.batch_size - per_size - suc_size
                per_batch, _per_indices, _per_weights = self.low_memory.sample(per_size, beta=0.4)
                suc_batch = (random.sample(self.success_memory, min(suc_size, len(self.success_memory)))
                             if len(self.success_memory) >= suc_size else [])
                eli_batch = (self.elite_buffer.sample(eli_size)
                             if getattr(self, 'elite_buffer', None) and len(self.elite_buffer) > 0 else [])
                batch = per_batch + suc_batch + eli_batch
                if len(batch) < self.batch_size // 2:
                    batch = per_batch  # fallback
                random.shuffle(batch)
            else:
                success_size = self.batch_size // 4
                if len(self.success_memory) >= success_size:
                    success_batch = random.sample(self.success_memory, success_size)
                    normal_batch  = random.sample(self.low_memory, self.batch_size - success_size)
                    batch = success_batch + normal_batch
                    random.shuffle(batch)
                else:
                    batch = random.sample(self.low_memory, self.batch_size)

            # ── 状态嵌入 ────────────────────────────────────────────────
            if self.encoder is not None:
                _topo_ei = (self.env.edge_index.to(self.device)
                            if self.env is not None
                            and hasattr(self.env, 'edge_index')
                            and self.env.edge_index is not None else None)

                def _fix_ea(ei, state_obj, fdim=5):
                    n = ei.shape[1]
                    ea = getattr(state_obj, 'edge_attr', None)
                    if ea is None: return torch.zeros(n, fdim, device=self.device)
                    ea = ea.to(self.device)
                    if ea.dim() == 1: ea = ea.unsqueeze(1)
                    if ea.shape[0] != n: return torch.zeros(n, fdim, device=self.device)
                    if ea.shape[1] < fdim:
                        ea = torch.cat([ea, torch.zeros(n, fdim - ea.shape[1], device=self.device)], dim=1)
                    return ea

                # 节点数N（用于统一tensor形状）
                _N = (self.env.n if self.env is not None and hasattr(self.env, 'n')
                      else 28)

                def _encode(state_obj, detach=False):
                    s = state_obj[0] if isinstance(state_obj, tuple) else state_obj
                    try:
                        if hasattr(s, 'x') and hasattr(s, 'edge_index'):
                            ei = s.edge_index.to(self.device) if s.edge_index is not None else _topo_ei
                            if ei is not None:
                                ea = _fix_ea(ei, s)
                                b  = torch.zeros(s.x.size(0), dtype=torch.long, device=self.device)
                                _tei = getattr(s, 'tree_edge_index', None)
                                if _tei is not None: _tei = _tei.to(self.device)
                                out = self.encoder(s.x.to(self.device), ei, ea, batch=b,
                                                   tree_edge_index=_tei)
                                if detach: out = out.detach()
                                # [TA-HGRL Fix] 返回 [1, N, H]
                                if out.size(0) == _N:
                                    return out.unsqueeze(0)          # [1, N, H] 正常路径
                                # N不匹配时做均值兜底，保证cat不报错
                                return out.mean(dim=0, keepdim=True).unsqueeze(0).expand(1, _N, -1)
                    except Exception:
                        pass
                    # fallback: [1, N, H] 零填充
                    return torch.zeros(1, _N, self.hidden_dim, device=self.device)

                state_tensor      = torch.cat([_encode(x['state'])           for x in batch])
                next_state_tensor = torch.cat([_encode(x['next_state'], True) for x in batch]).detach()
                # state_tensor: [B, N, H]
            else:
                # encoder未使用時のfallback: mean emb を [1,N,H] に broadcast
                _N_fb = (self.env.n if self.env is not None and hasattr(self.env, 'n') else 28)
                def _enc_fb(s):
                    e = self._extract_state_embedding(s).to(self.device)  # [1, H]
                    return e.unsqueeze(1).expand(1, _N_fb, -1)            # [1, N, H]
                state_tensor      = torch.cat([_enc_fb(x['state'])      for x in batch])
                next_state_tensor = torch.cat([_enc_fb(x['next_state']) for x in batch]).detach()

            actions = torch.tensor([x['action']  for x in batch], device=self.device).long().unsqueeze(1)
            rewards = torch.tensor([x['reward']  for x in batch], device=self.device).float().unsqueeze(1)
            dones   = torch.tensor([x['done']    for x in batch], device=self.device).float().unsqueeze(1)

            # 过滤无效动作
            valid_mask = (actions >= 0).squeeze()
            if valid_mask.sum() == 0: return 0.0
            state_tensor      = state_tensor[valid_mask]
            next_state_tensor = next_state_tensor[valid_mask]
            actions = actions[valid_mask]
            rewards = torch.clamp(rewards[valid_mask], -20.0, 150.0)
            dones   = dones[valid_mask]

            # Goal embedding
            valid_idx = torch.nonzero(valid_mask).squeeze().cpu().tolist()
            if not isinstance(valid_idx, list): valid_idx = [valid_idx]
            goal_embs = []
            for idx in valid_idx:
                g = batch[idx].get('goal_emb')
                if g is None:
                    g = torch.zeros(1, self.goal_dim, device=self.device)
                else:
                    g = g.to(self.device)
                    if g.dim() == 1: g = g.unsqueeze(0)
                    if g.size(1) != self.goal_dim:
                        g = (g[:, :self.goal_dim] if g.size(1) > self.goal_dim
                             else torch.cat([g, torch.zeros(g.size(0), self.goal_dim - g.size(1), device=self.device)], 1))
                goal_embs.append(g)
            goal_tensor = torch.cat(goal_embs).to(self.device)

            # Current Q
            out = self.low_policy(state_tensor, goal_tensor)
            curr_q_values = out[0] if isinstance(out, tuple) else out
            curr_q = curr_q_values.gather(1, actions)

            if torch.isnan(curr_q).any() or torch.isinf(curr_q).any():
                logger.error("❌ Q值NaN/Inf，触发重置")
                self.reset_network_parameters()
                return 0.0

            # Target Q (Double DQN)
            with torch.no_grad():
                next_out_on = self.low_policy(next_state_tensor, goal_tensor)
                next_q_on   = next_out_on[0] if isinstance(next_out_on, tuple) else next_out_on
                next_acts   = next_q_on.argmax(dim=1, keepdim=True)

                next_out_tg = self.target_low_policy(next_state_tensor, goal_tensor)
                next_q_tg   = next_out_tg[0] if isinstance(next_out_tg, tuple) else next_out_tg
                next_q      = torch.clamp(next_q_tg.gather(1, next_acts), -20.0, 100.0)

                target_q = torch.clamp(rewards + (1 - dones) * self.gamma * next_q, -30.0, 150.0)

            # [Loss Fix] reward归一化 + PER importance sampling weights
            _reward_scale = 30.0
            curr_q_scaled   = curr_q   / _reward_scale
            target_q_scaled = target_q / _reward_scale
            elementwise_loss = F.smooth_l1_loss(
                curr_q_scaled, target_q_scaled, reduction='none')
            if getattr(self, '_use_per', False) and _per_weights is not None:
                # valid_idx はbatch全体のインデックス、_per_weightsはper_sizeのみ
                # per_sizeの範囲内のインデックスのみweightを適用、それ以外は1.0
                _per_size = len(_per_weights)
                _w_full = torch.ones(len(batch), 1, device=self.device)
                for _i, _bi in enumerate(_per_weights):
                    _w_full[_i, 0] = float(_bi)
                # valid_maskでフィルタ後のweight
                _w_valid = _w_full[valid_mask]
                loss = (elementwise_loss * _w_valid).mean()
            else:
                loss = elementwise_loss.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("❌ Low-Level Loss NaN/Inf，跳过")
                return 0.0

            self.optimizer_low.zero_grad()
            loss.backward()

            # [SDG-HRL] PER: 用TD-error更新优先级
            if getattr(self, '_use_per', False) and _per_indices is not None:
                with torch.no_grad():
                    _td = (curr_q - target_q).abs().detach().cpu().numpy().flatten()
                    _n  = min(len(_per_indices), len(_td))
                    self.low_memory.update_priorities(_per_indices[:_n], _td[:_n])

            # tree_bias监控
            try:
                if self.encoder is not None and hasattr(self.encoder, 'tree_bias'):
                    logger.debug(f"[tree_bias] grad={self.encoder.tree_bias.grad} "
                                 f"val={self.encoder.tree_bias.item():.6f}")
            except Exception as _e:
                logger.debug(f"[tree_bias] error: {_e}")

            # 梯度监控+裁剪
            self.gradient_norms.append(
                sum(p.grad.norm().item() for p in self.low_policy.parameters() if p.grad is not None))
            nn.utils.clip_grad_norm_(self.low_policy.parameters(), self.clip_grad_norm)
            self.optimizer_low.step()

            # Q监控
            with torch.no_grad():
                if not hasattr(self, '_q_stats'): self._q_stats = {'count': 0}
                self._q_stats['count'] += 1
                if self._q_stats['count'] % 200 == 0:
                    logger.info(
                        f"[Q-Monitor] step={self._q_stats['count']} | "
                        f"CurrQ: mean={curr_q.mean():.2f} std={curr_q.std():.2f} "
                        f"min={curr_q.min():.2f} max={curr_q.max():.2f} | "
                        f"TargetQ mean={target_q.mean():.2f} | "
                        f"AllQ_max={curr_q_values.max():.2f} | Loss={loss.item():.4f}"
                    )

            # 自适应学习率
            lv = loss.item()
            # [Loss Fix] 阈值与归一化后的loss量级对齐（期望0.1~2.0）
            for pg in self.optimizer_low.param_groups:
                if lv < 0.01:
                    pg['lr'] = min(pg['lr'] * 1.02, 1e-3)
                elif lv > 0.5:
                    pg['lr'] = max(pg['lr'] * 0.98, 1e-5)
            return lv

        except Exception as e:
            logger.error(f"[Update Low Level] Error: {e}")
            import traceback; traceback.print_exc()
            return 0.0

    # ── Target Network ────────────────────────────────────────────────────

    def _soft_update_target_networks(self):
        """软更新（tau=0.005），比hard update更稳定"""
        for tp, p in zip(self.target_high_policy.parameters(), self.high_policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.target_low_policy.parameters(), self.low_policy.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _hard_update_target_networks(self):
        """硬更新（保留备用，训练中不调用）"""
        self.target_high_policy.load_state_dict(self.high_policy.state_dict())
        self.target_low_policy.load_state_dict(self.low_policy.state_dict())

    # ── Epsilon ───────────────────────────────────────────────────────────

    def _update_epsilon(self):
        # [Fix] 改为基于episode衰减，避免low-level steps突增导致ε崩塌
        # steps_done在ep180后因绕路/重试暴增，导致ε非预期加速下降
        total_ep = getattr(self, 'total_episodes', 0)
        decay_episodes = getattr(self, 'epsilon_decay_episodes', 200)
        progress = min(total_ep / max(1, decay_episodes), 1.0)
        self.epsilon_high = self.epsilon_high_start + (self.epsilon_high_end - self.epsilon_high_start) * progress
        self.epsilon_low  = self.epsilon_low_start  + (self.epsilon_low_end  - self.epsilon_low_start)  * progress

    def on_episode_end(self):
        """每个episode结束时调用，驱动ε衰减"""
        self.total_episodes = getattr(self, 'total_episodes', 0) + 1
        self._update_epsilon()

    def update_epsilon(self):
        """向后兼容（保留，但不再是主要驱动）"""
        pass  # 由on_episode_end驱动，此处空置防止steps驱动干扰

    # ── 统计日志 ──────────────────────────────────────────────────────────

    def _log_training_stats(self):
        if self.high_loss_history and self.low_loss_history:
            logger.debug(
                f"📊 训练统计: HighLoss={np.mean(self.high_loss_history):.4f} "
                f"LowLoss={np.mean(self.low_loss_history):.4f} "
                f"GradNorm={np.mean(self.gradient_norms) if self.gradient_norms else 0:.2f} "
                f"ε_low={self.epsilon_low:.3f}"
            )