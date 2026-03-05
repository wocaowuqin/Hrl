#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prioritized Experience Replay Buffer
"""
import numpy as np
import random


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    按TD-error优先采样，让模型更多学习难样本。
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, priority=None):
        if priority is None:
            priority = float(self.priorities.max()) if self.buffer else 1.0
        priority = max(priority, 1e-5)  # 防止优先级为0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        if n == 0:
            return [], [], np.array([])

        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, min(batch_size, n), p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(float(td)) + 1e-5

    def __len__(self):
        return len(self.buffer)

    @property
    def maxlen(self):
        return self.capacity