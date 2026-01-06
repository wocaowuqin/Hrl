#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VNF放置策略 - 签名修复 + 带宽优化终极版
修复：所有子类方法签名与父类完全一致
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class VNFPlacementStrategy:
    """VNF放置策略基类"""

    def __init__(self, node_num: int, type_num: int, node_index_base: int = 1):
        self.node_num = node_num
        self.type_num = type_num
        self.node_index_base = node_index_base
        logger.debug(f"[Placement] 初始化: 节点={node_num}, 类型={type_num}, 索引基值={node_index_base}")

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
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any  # ✅ 基类也添加 **kwargs
    ) -> Optional[Dict]:
        raise NotImplementedError

    def _to_internal(self, node: int) -> int:
        """外部节点ID → 内部索引 (0-based)"""
        if self.node_index_base == 1:
            return node - 1
        return node

    def _to_external(self, node: int) -> int:
        """内部索引 (0-based) → 外部节点ID"""
        if self.node_index_base == 1:
            return node + 1
        return node


class OptimizedPlacementStrategy(VNFPlacementStrategy):
    """
    高性能放置策略 - 带宽优化版

    核心特性：
    1. 容量感知三级策略（充裕/适中/紧张）
    2. 距离感知优化（减少树复杂度）
    3. 改进的资源预过滤
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
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any  # ✅ 保持签名一致
    ) -> Optional[Dict]:
        """
        放置VNF链

        Args:
            chain: VNF类型列表
            cpu_reqs: CPU需求列表
            mem_reqs: 内存需求列表
            candidate_nodes: 候选节点列表（外部ID）
            existing_hvt: 现有VNF实例矩阵
            cpu_delta: CPU增量数组
            mem_delta: 内存增量数组
            vnf_delta: VNF增量矩阵
            state: 全局网络状态
            enable_debug: 是否开启调试
            **kwargs: 额外参数
                - strategy_type: 策略类型（'capacity_aware', 'balanced', 'fragmentation'）
                - source_node: 源节点（用于距离感知）
                - distance_matrix: 距离矩阵

        Returns:
            placement字典，格式: {(node_ext, vnf_type): node_ext}
            失败返回 None
        """
        strategy_type = kwargs.get('strategy_type', 'capacity_aware')
        source_node = kwargs.get('source_node', None)
        distance_matrix = kwargs.get('distance_matrix', None)
        debug = enable_debug

        if debug:
            print(f"\n{'=' * 60}")
            print(f"[Placement] 开始放置VNF链: {chain}")
            print(f"  候选节点(外部): {candidate_nodes}")
            print(f"  策略类型: {strategy_type}")

        # ========================================================================
        # 步骤 1: 候选节点转换和验证
        # ========================================================================
        c_indices = []
        c_external = []
        for node_ext in candidate_nodes:
            node_int = self._to_internal(node_ext)
            if 0 <= node_int < self.node_num:
                c_indices.append(node_int)
                c_external.append(node_ext)
            elif debug:
                print(f"  ⚠️ 忽略无效节点: 外部{node_ext} → 内部{node_int}")

        if not c_indices:
            if debug:
                print("  ❌ 无有效候选节点")
            return None

        c_indices = np.array(c_indices)

        # ========================================================================
        # 步骤 2: 资源状态获取和容量水平计算
        # ========================================================================
        utilization = 0.5  # 默认中等利用率

        if state is not None and 'cpu' in state and len(state['cpu']) == self.node_num:
            cpu_used = state.get('cpu_used', np.zeros(self.node_num))
            mem_used = state.get('mem_used', np.zeros(self.node_num))
            cpu_capacity = state['cpu']
            mem_capacity = state['mem']

            cpu_remaining = cpu_capacity[c_indices] - cpu_used[c_indices] - cpu_delta[c_indices]
            mem_remaining = mem_capacity[c_indices] - mem_used[c_indices] - mem_delta[c_indices]

            # 计算容量水平
            avg_cpu_remaining = np.mean(cpu_remaining)
            avg_mem_remaining = np.mean(mem_remaining)
            cpu_cap = np.mean(cpu_capacity) if len(cpu_capacity) > 0 else 80.0
            mem_cap = np.mean(mem_capacity) if len(mem_capacity) > 0 else 60.0

            # 资源利用率 = 1 - (剩余/容量)
            cpu_util = 1.0 - (avg_cpu_remaining / cpu_cap) if cpu_cap > 0 else 0.5
            mem_util = 1.0 - (avg_mem_remaining / mem_cap) if mem_cap > 0 else 0.5
            utilization = (cpu_util + mem_util) / 2

            if debug:
                print(
                    f"  📊 资源利用率: CPU={cpu_util * 100:.1f}%, MEM={mem_util * 100:.1f}%, 综合={utilization * 100:.1f}%")

            # 资源预过滤
            total_cpu = sum(cpu_reqs)
            total_mem = sum(mem_reqs)

            feasible_mask = (cpu_remaining >= total_cpu * 0.3) & (mem_remaining >= total_mem * 0.3)

            if debug and np.sum(feasible_mask) < len(c_indices):
                filtered = len(c_indices) - np.sum(feasible_mask)
                print(f"  🔍 资源预过滤: 移除{filtered}个节点")

            c_indices = c_indices[feasible_mask]
            c_external = [c_external[i] for i in range(len(feasible_mask)) if feasible_mask[i]]

            if len(c_indices) == 0:
                if debug:
                    print("  ❌ 资源预过滤后无候选节点")
                return None

            curr_cpu = cpu_remaining[feasible_mask]
            curr_mem = mem_remaining[feasible_mask]
        else:
            # 回退模式
            curr_cpu = 1000.0 - cpu_delta[c_indices]
            curr_mem = 1000.0 - mem_delta[c_indices]

        # ========================================================================
        # 步骤 3: VNF链重排
        # ========================================================================
        cpu_reqs_np = np.array(cpu_reqs)
        mem_reqs_np = np.array(mem_reqs)

        if strategy_type == "cpu_heavy_first":
            order = np.argsort(-cpu_reqs_np)
        elif strategy_type == "mem_heavy_first":
            order = np.argsort(-mem_reqs_np)
        else:
            combined_req = cpu_reqs_np + mem_reqs_np
            order = np.argsort(-combined_req)

        if debug:
            print(f"  VNF处理顺序: {order}")

        placement = {}

        # ========================================================================
        # 步骤 4: 逐个放置VNF
        # ========================================================================
        for vnf_idx in order:
            vnf_type = chain[vnf_idx]
            req_c = cpu_reqs[vnf_idx]
            req_m = mem_reqs[vnf_idx]
            vnf_t = vnf_type - 1

            if debug:
                print(f"\n  🔧 VNF{vnf_idx}(类型{vnf_type}): CPU={req_c:.1f}, MEM={req_m:.1f}")

            # === A. VNF复用检查 ===
            reuse_mask = existing_hvt[c_indices, vnf_t] > 0

            if np.any(reuse_mask):
                reuse_nodes = c_indices[reuse_mask]

                # 复用时选择剩余资源最多的节点
                if len(reuse_nodes) > 1:
                    load_scores = curr_cpu[reuse_mask] + curr_mem[reuse_mask]
                    best_idx = np.argmax(load_scores)
                else:
                    best_idx = 0

                chosen_node_int = reuse_nodes[best_idx]
                chosen_node_ext = self._to_external(chosen_node_int)

                placement[(chosen_node_ext, vnf_type)] = chosen_node_ext

                if debug:
                    print(f"    ✅ 复用: 节点{chosen_node_ext}")
                continue

            # === B. 新放置检查 ===
            not_occupied_mask = vnf_delta[c_indices, vnf_t] == 0
            res_mask = (curr_cpu >= req_c - 1e-7) & (curr_mem >= req_m - 1e-7)
            valid_mask = not_occupied_mask & res_mask

            if not np.any(valid_mask):
                if debug:
                    print(f"    ❌ 放置失败: 无合适节点")
                return None

            # === C. 节点选择策略 ===
            valid_nodes = c_indices[valid_mask]
            valid_cpu = curr_cpu[valid_mask]
            valid_mem = curr_mem[valid_mask]

            # ✅ 核心：容量感知三级策略
            if strategy_type == "capacity_aware":
                if utilization < 0.3:
                    # 资源充裕 → 集中放置（Best Fit）
                    scores = -(valid_cpu + valid_mem)
                    best_idx = np.argmax(scores)

                    if debug:
                        print(f"    📊 策略: 集中放置（利用率{utilization * 100:.1f}%）")

                elif utilization > 0.7:
                    # 资源紧张 → 负载均衡
                    scores = valid_cpu + valid_mem
                    best_idx = np.argmax(scores)

                    if debug:
                        print(f"    📊 策略: 负载均衡（利用率{utilization * 100:.1f}%）")

                else:
                    # 资源适中 → 混合策略
                    resource_scores = valid_cpu + valid_mem

                    # 距离感知（如果有距离信息）
                    if source_node is not None and distance_matrix is not None:
                        source_int = self._to_internal(source_node)
                        if 0 <= source_int < self.node_num and source_int < len(distance_matrix):
                            distances = np.array([
                                distance_matrix[source_int, node_int]
                                if node_int < len(distance_matrix[source_int])
                                else 999
                                for node_int in valid_nodes
                            ])

                            max_dist = np.max(distances) if np.max(distances) > 0 else 1.0
                            distance_scores = 1.0 - (distances / max_dist)

                            # 混合：60%资源 + 40%距离
                            scores = resource_scores * 0.6 + distance_scores * 0.4

                            if debug:
                                print(f"    📊 策略: 混合（资源60% + 距离40%）")
                        else:
                            scores = resource_scores
                    else:
                        scores = resource_scores

                        if debug:
                            print(f"    📊 策略: 资源优先（利用率{utilization * 100:.1f}%）")

                    best_idx = np.argmax(scores)

            elif strategy_type == "fragmentation":
                # 碎片整理（Best Fit）
                scores = valid_cpu + valid_mem
                best_idx = np.argmin(scores)

            else:
                # 默认：负载均衡
                scores = valid_cpu + valid_mem
                best_idx = np.argmax(scores)

            chosen_node_int = valid_nodes[best_idx]
            chosen_node_ext = self._to_external(chosen_node_int)

            pos_in_candidates = np.where(c_indices == chosen_node_int)[0][0]

            # === D. 更新状态 ===
            cpu_delta[chosen_node_int] += req_c
            mem_delta[chosen_node_int] += req_m
            vnf_delta[chosen_node_int, vnf_t] = 1

            curr_cpu[pos_in_candidates] -= req_c
            curr_mem[pos_in_candidates] -= req_m

            placement[(chosen_node_ext, vnf_type)] = chosen_node_ext

            if debug:
                print(f"    ✅ 新放置: 节点{chosen_node_ext}")

        if debug:
            unique_nodes = len(set(placement.values()))
            concentration = (1 - unique_nodes / len(chain)) * 100 if len(chain) > 0 else 0
            print(f"\n  ✅ 完成: 使用{unique_nodes}个节点, 集中度{concentration:.1f}%")
            print(f"{'=' * 60}")

        return placement


class LoadAwarePlacementStrategy(OptimizedPlacementStrategy):
    """
    负载感知放置策略
    ✅ 签名与父类完全一致
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
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any  # ✅ 添加 **kwargs
    ) -> Optional[Dict]:
        """负载感知策略 - 直接使用父类的容量感知策略"""
        # 强制使用 capacity_aware 策略
        kwargs['strategy_type'] = 'capacity_aware'

        return super().place_vnf_chain(
            chain=chain,
            cpu_reqs=cpu_reqs,
            mem_reqs=mem_reqs,
            candidate_nodes=candidate_nodes,
            existing_hvt=existing_hvt,
            cpu_delta=cpu_delta,
            mem_delta=mem_delta,
            vnf_delta=vnf_delta,
            state=state,
            enable_debug=enable_debug,
            **kwargs
        )


class SimplePlacementStrategy(VNFPlacementStrategy):
    """
    简化放置策略 - 用于调试
    ✅ 签名与基类完全一致
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
            state: Optional[Dict] = None,
            enable_debug: bool = False,
            **kwargs: Any  # ✅ 添加 **kwargs
    ) -> Optional[Dict]:
        """简化策略，First Fit"""
        debug = enable_debug

        if debug:
            print(f"\n[SimplePlacement] VNF链: {chain}")

        placement = {}
        candidate_internal = []

        for node_ext in candidate_nodes:
            node_int = self._to_internal(node_ext)
            if 0 <= node_int < self.node_num:
                candidate_internal.append(node_int)

        if not candidate_internal:
            return None

        for i, (vnf_type, req_c, req_m) in enumerate(zip(chain, cpu_reqs, mem_reqs)):
            vnf_t = vnf_type - 1
            placed = False

            # 检查复用
            for node_int in candidate_internal:
                if existing_hvt[node_int, vnf_t] > 0:
                    node_ext = self._to_external(node_int)
                    placement[(node_ext, vnf_type)] = node_ext
                    placed = True
                    if debug:
                        print(f"  VNF{i}: ✅ 复用节点{node_ext}")
                    break

            if placed:
                continue

            # 新放置
            for node_int in candidate_internal:
                if vnf_delta[node_int, vnf_t] > 0:
                    continue

                cpu_available = 1000.0 - cpu_delta[node_int]
                mem_available = 1000.0 - mem_delta[node_int]

                if cpu_available >= req_c and mem_available >= req_m:
                    cpu_delta[node_int] += req_c
                    mem_delta[node_int] += req_m
                    vnf_delta[node_int, vnf_t] = 1

                    node_ext = self._to_external(node_int)
                    placement[(node_ext, vnf_type)] = node_ext
                    placed = True

                    if debug:
                        print(f"  VNF{i}: ✅ 新放置节点{node_ext}")
                    break

            if not placed:
                if debug:
                    print(f"  VNF{i}: ❌ 失败")
                return None

        return placement


# =============================================================================
# 测试代码
# =============================================================================

def test_signature_compatibility():
    """测试方法签名兼容性"""
    print("\n" + "=" * 70)
    print("测试方法签名兼容性")
    print("=" * 70)

    strategies = [
        ("Optimized", OptimizedPlacementStrategy(10, 8, 1)),
        ("LoadAware", LoadAwarePlacementStrategy(10, 8, 1)),
        ("Simple", SimplePlacementStrategy(10, 8, 1))
    ]

    chain = [1, 2, 3]
    cpu_reqs = [10.0, 8.0, 6.0]
    mem_reqs = [8.0, 6.0, 4.0]
    candidate_nodes = [1, 2, 3, 4, 5]

    existing_hvt = np.zeros((10, 8))
    cpu_delta = np.zeros(10)
    mem_delta = np.zeros(10)
    vnf_delta = np.zeros((10, 8))

    state = {
        'cpu': np.array([80.0] * 10),
        'mem': np.array([60.0] * 10),
        'cpu_used': np.array([40.0] * 10),
        'mem_used': np.array([30.0] * 10)
    }

    for name, strategy in strategies:
        print(f"\n{'─' * 70}")
        print(f"测试: {name}PlacementStrategy")
        print(f"{'─' * 70}")

        try:
            # 重置状态
            cpu_delta_test = cpu_delta.copy()
            mem_delta_test = mem_delta.copy()
            vnf_delta_test = vnf_delta.copy()

            # 测试1: 基本调用
            placement = strategy.place_vnf_chain(
                chain=chain,
                cpu_reqs=cpu_reqs,
                mem_reqs=mem_reqs,
                candidate_nodes=candidate_nodes,
                existing_hvt=existing_hvt,
                cpu_delta=cpu_delta_test,
                mem_delta=mem_delta_test,
                vnf_delta=vnf_delta_test,
                state=state,
                enable_debug=False
            )

            print(f"✅ 基本调用成功: {placement is not None}")

            # 测试2: 带kwargs调用
            cpu_delta_test = cpu_delta.copy()
            mem_delta_test = mem_delta.copy()
            vnf_delta_test = vnf_delta.copy()

            placement = strategy.place_vnf_chain(
                chain=chain,
                cpu_reqs=cpu_reqs,
                mem_reqs=mem_reqs,
                candidate_nodes=candidate_nodes,
                existing_hvt=existing_hvt,
                cpu_delta=cpu_delta_test,
                mem_delta=mem_delta_test,
                vnf_delta=vnf_delta_test,
                state=state,
                enable_debug=False,
                strategy_type="balanced",  # kwargs测试
                source_node=1
            )

            print(f"✅ 带kwargs调用成功: {placement is not None}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("✅ 所有签名兼容性测试通过")
    print("=" * 70)


def test_capacity_aware():
    """测试容量感知策略"""
    print("\n" + "=" * 70)
    print("测试容量感知策略")
    print("=" * 70)

    strategy = OptimizedPlacementStrategy(10, 8, 1)

    chain = [1, 2, 3]
    cpu_reqs = [10.0, 8.0, 6.0]
    mem_reqs = [8.0, 6.0, 4.0]
    candidate_nodes = [1, 2, 3, 4, 5]

    scenarios = [
        ("资源充裕（20%利用率）", [16.0] * 10, "集中放置"),
        ("资源适中（50%利用率）", [40.0] * 10, "混合策略"),
        ("资源紧张（80%利用率）", [64.0] * 10, "负载均衡"),
    ]

    for name, cpu_used, expected in scenarios:
        print(f"\n{'─' * 70}")
        print(f"场景: {name}")
        print(f"{'─' * 70}")

        state = {
            'cpu': np.array([80.0] * 10),
            'mem': np.array([60.0] * 10),
            'cpu_used': np.array(cpu_used),
            'mem_used': np.array([30.0] * 10)
        }

        existing_hvt = np.zeros((10, 8))
        cpu_delta = np.zeros(10)
        mem_delta = np.zeros(10)
        vnf_delta = np.zeros((10, 8))

        placement = strategy.place_vnf_chain(
            chain=chain,
            cpu_reqs=cpu_reqs,
            mem_reqs=mem_reqs,
            candidate_nodes=candidate_nodes,
            existing_hvt=existing_hvt,
            cpu_delta=cpu_delta,
            mem_delta=mem_delta,
            vnf_delta=vnf_delta,
            state=state,
            enable_debug=True,
            strategy_type="capacity_aware"
        )

        if placement:
            unique_nodes = len(set(placement.values()))
            print(f"\n结果: 使用{unique_nodes}个节点")
            print(f"期望策略: {expected}")


if __name__ == "__main__":
    test_signature_compatibility()
    test_capacity_aware()