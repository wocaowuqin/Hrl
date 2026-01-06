#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class ParetoSet:
    """
    Pareto 支配集合，用于 Beam Search 剪枝
    维护一组非支配状态 (Resource Vector, Cost)
    """
    def __init__(self):
        # 存储元组: (resource_vec, cost)
        # resource_vec: np.array([cpu_sum, mem_sum, bw_sum])
        self.states = []

    def is_dominated(self, res_vec: np.ndarray, cost: float) -> bool:
        """
        检查新状态 (res_vec, cost) 是否被现有状态支配。
        支配条件：现有状态 s 满足 s.res <= res_vec 且 s.cost <= cost
        """
        for r, c in self.states:
            # 如果存在一个状态，资源消耗更少(或相等)且代价更低(或相等)，则当前状态被支配
            if c <= cost and np.all(r <= res_vec):
                return True
        return False

    def insert(self, res_vec: np.ndarray, cost: float):
        """
        插入新状态，并移除集合中被新状态支配的旧状态。
        """
        # 过滤掉被新状态支配的旧状态 (即新状态比旧状态更优)
        # 如果 res_vec <= r 且 cost <= c，则旧状态 (r, c) 被新状态支配，应当移除
        self.states = [
            (r, c) for r, c in self.states
            if not (cost <= c and np.all(res_vec <= r))
        ]
        self.states.append((res_vec, cost))