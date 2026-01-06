#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VNF放置策略 - 终极优化版
集成：NumPy向量化 + 需求重排 + 状态感知
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VNFPlacementStrategy:
    """VNF放置策略基类"""

    def __init__(self, node_num: int, type_num: int):
        self.node_num = node_num
        self.type_num = type_num

    def place_vnf_chain(self, *args, **kwargs) -> Optional[Dict]:
        raise NotImplementedError


class OptimizedPlacementStrategy(VNFPlacementStrategy):
    """
    高性能放置策略
    1. VNF 链重排 (First Fit Decreasing)
    2. NumPy 向量化并行检查
    3. 资源碎片整理 (Best Fit)
    """

    def place_vnf_chain(
            self,
            chain: List[int],
            cpu_reqs: List[float],
            mem_reqs: List[float],
            candidate_nodes: List[int],
            existing_hvt: np.ndarray,
            cpu_delta: np.ndarray,
            mem_delta: np.ndarray,
            vnf_delta: np.ndarray,
            state: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Args:
            chain: VNF类型列表
            cpu_reqs: CPU需求
            mem_reqs: 内存需求
            candidate_nodes: 候选节点列表 (1-based)
            existing_hvt: 现有VNF实例 (0/1矩阵)
            cpu_delta, mem_delta, vnf_delta: 增量数组 (In/Out)
            state: 全局网络状态 (用于精确资源检查)
        """
        # 0. 预处理：候选节点转 0-based 索引
        c_indices = np.array(candidate_nodes) - 1
        if len(c_indices) == 0:
            return None

        # 1. [算法优化] VNF 链重排：按 (CPU+MEM) 降序排列
        # 优先处理大需求，尽早失败，避免无效计算
        combined_req = np.array(cpu_reqs) + np.array(mem_reqs)
        order = np.argsort(-combined_req)

        # 2. [实现优化] 状态预取 (向量化视图)
        # 计算当前候选节点的"实时剩余资源"
        if state is not None:
            # 剩余 = 容量 - 已用 - 本次增量
            curr_cpu = state['cpu'][c_indices] - state.get('cpu_load', 0) - cpu_delta[c_indices]
            curr_mem = state['mem'][c_indices] - state.get('mem_load', 0) - mem_delta[c_indices]
        else:
            # 回退模式：仅检查增量不超限 (假设容量无限)
            curr_cpu = 1000.0 - cpu_delta[c_indices]
            curr_mem = 1000.0 - mem_delta[c_indices]

        placement = {}

        # 按重排后的顺序逐个放置 VNF
        for idx in order:
            vnf_type = chain[idx]
            req_c = cpu_reqs[idx]
            req_m = mem_reqs[idx]
            vnf_t = vnf_type - 1  # 0-based type

            # --- A. 优先检查复用 (向量化) ---
            # 检查候选节点中是否已部署该 VNF (existing_hvt > 0)
            # existing_hvt shape: (N, K) -> 切片 (M, 1)
            reuse_mask = existing_hvt[c_indices, vnf_t] > 0

            if np.any(reuse_mask):
                # 贪心复用：选择第一个可复用的节点
                valid_indices = c_indices[reuse_mask]
                chosen_node = valid_indices[0]
                placement[(chosen_node + 1, vnf_type)] = chosen_node + 1
                continue  # 完成当前VNF，处理下一个

            # --- B. 新放置检查 (向量化) ---
            # 1. 本次请求尚未在该节点占用同类 VNF (避免冲突)
            not_occupied_mask = vnf_delta[c_indices, vnf_t] == 0

            # 2. 资源足够 (一行代码检查所有节点)
            res_mask = (curr_cpu >= req_c - 1e-7) & (curr_mem >= req_m - 1e-7)

            # 3. 合并掩码
            valid_mask = not_occupied_mask & res_mask

            if not np.any(valid_mask):
                return None  # 放置失败，整个链回滚

            # --- C. 评分与选择 (Best Fit) ---
            # 选择剩余资源最少的节点（碎片整理），或最多的（负载均衡）
            # 这里使用 Load Balancing (剩余资源越多越好)，提高成功率
            valid_indices = c_indices[valid_mask]
            valid_cpu = curr_cpu[valid_mask]
            valid_mem = curr_mem[valid_mask]

            scores = valid_cpu + valid_mem
            best_local_idx = np.argmax(scores)  # 选择资源最充裕的

            chosen_node = valid_indices[best_local_idx]

            # --- D. 更新状态 ---
            # 1. 更新输出增量
            cpu_delta[chosen_node] += req_c
            mem_delta[chosen_node] += req_m
            vnf_delta[chosen_node, vnf_t] = 1

            # 2. 更新当前视图 (供下一个 VNF 使用)
            # 找到 chosen_node 在 c_indices 中的位置
            chosen_ptr = np.where(c_indices == chosen_node)[0][0]
            curr_cpu[chosen_ptr] -= req_c
            curr_mem[chosen_ptr] -= req_m

            placement[(chosen_node + 1, vnf_type)] = chosen_node + 1

        return placement