#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRLAgentMemory — 经验存储（High/Low/Success buffer）
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HRLAgentMemory:
    """
    负责：
    - store_transition_high: 存储高层经验
    - store_transition_low:  存储低层经验 + success_memory + 内在奖励
    - store_transition:      向后兼容接口
    """

    def store_transition_high(
            self, state: Dict, goal: int, reward: float, next_state: Dict, done: bool
    ):
        """存储High-Level经验，goal用clamp防止越界"""
        goal_idx = max(0, min(int(goal), self.n_goals - 1)) if isinstance(goal, (int, np.integer)) else 0
        scaled_reward = max(-20.0, min(150.0, float(reward)))  # [SDG-HRL] 不缩放
        transition_high = {
            'state':      state,
            'goal':       goal_idx,
            'reward':     scaled_reward,
            'next_state': next_state,
            'done':       done,
        }
        if getattr(self, '_use_per', False):
            self.high_memory.add(transition_high)
        else:
            self.high_memory.append(transition_high)

    def store_transition_low(
            self, state: Dict, action: int, reward: float, next_state: Dict, done: bool
    ):
        """存储Low-Level经验"""
        # [SDG-HRL] 不缩放，直接截断
        scaled_reward = max(-20.0, min(150.0, float(reward)))

        # 可选内在奖励
        if self.config.get('hrl', {}).get('use_intrinsic_reward', False):
            try:
                with torch.no_grad():
                    se  = self._extract_state_embedding(state)
                    nse = self._extract_state_embedding(next_state)
                    err = F.mse_loss(se, nse).item()
                    scaled_reward += min(0.3, err * 0.3)  # [SDG-HRL]
            except Exception:
                pass

        transition = {
            'state':      state,
            'action':     action,
            'reward':     scaled_reward,
            'next_state': next_state,
            'done':       done,
            'goal_emb':   self.current_goal_emb,
        }
        # [SDG-HRL] PER: 新transition用最高优先级入队
        if getattr(self, '_use_per', False):
            self.low_memory.add(transition)
        else:
            self.low_memory.append(transition)

        # 成功经验保留池
        if done and scaled_reward > 50.0:
            self.success_memory.append(transition)

        # EliteBuffer: 记录episode transitions
        self._ep_transitions.append(transition)
        if done and getattr(self, 'elite_buffer', None) is not None:
            ep_reward = sum(t['reward'] for t in self._ep_transitions)
            self._best_reward = max(self._best_reward, ep_reward)
            if ep_reward >= self._best_reward * 0.8:
                self.elite_buffer.add_episode(self._ep_transitions, ep_reward)
            self._ep_transitions = []

        self.steps_done += 1  # 驱动ε衰减

        # buffer大小监控
        _buf_len = len(self.low_memory)
        _buf_max = getattr(self.low_memory, 'maxlen', getattr(self.low_memory, 'capacity', 0)) or 0
        if 0 < _buf_len < _buf_max and _buf_len % 10000 == 0:
            logger.info(f"📊 Low Buffer: {_buf_len}/{_buf_max}")
        elif _buf_max > 0 and _buf_len >= _buf_max and not getattr(self, '_low_buf_full_logged', False):
            logger.info(f"📊 Low Buffer已满: {_buf_len}/{_buf_max}")
            self._low_buf_full_logged = True

    def store_transition(self, state, action, reward, next_state, done,
                         goal=None, next_valid_actions=None):
        """向后兼容接口"""
        if isinstance(action, (list, tuple)) and len(action) == 2:
            high_action, low_action = action
            if goal is not None:
                self.store_transition_high(state, goal, reward, next_state, done)
            self.store_transition_low(state, low_action, reward, next_state, done)
        else:
            self.store_transition_low(state, action, reward, next_state, done)