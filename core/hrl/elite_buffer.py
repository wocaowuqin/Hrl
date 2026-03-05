#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elite Buffer - 保存高质量episode，防止灾难性遗忘
"""
import random


class EliteBuffer:
    """
    保存高奖励episode的经验，混合进训练batch防止遗忘好策略。
    """

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []  # list of (transitions, reward)

    def add_episode(self, transitions, episode_reward):
        """添加一个episode的所有transition"""
        if not transitions:
            return
        self.buffer.append((list(transitions), float(episode_reward)))
        # 按奖励降序，只保留最好的
        self.buffer.sort(key=lambda x: x[1], reverse=True)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[:self.capacity]

    def sample(self, batch_size):
        """从elite episodes中随机采样transitions"""
        if not self.buffer:
            return []
        # 随机选几个episode，从中抽取transition
        n_eps = min(batch_size, len(self.buffer))
        selected = random.sample(self.buffer, n_eps)
        transitions = []
        for eps_transitions, _ in selected:
            if eps_transitions:
                transitions.append(random.choice(eps_transitions))
        return transitions

    def __len__(self):
        return sum(len(t) for t, _ in self.buffer)

    @property
    def n_episodes(self):
        return len(self.buffer)