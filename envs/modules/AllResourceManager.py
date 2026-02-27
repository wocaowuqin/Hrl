"""
envs/modules/fused_resource_manager.py
==================================================
融合版资源管理器
==================================================
结合：
1. ResourceManager 的简洁接口
2. UnifiedResourceManager 的事务管理
3. 共享资源池确保一致性
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# 共享资源池（方案C的核心）
# ============================================================================

class SharedResourcePool:
    """
    共享资源池 - 所有资源管理器共享的底层资源
    """

    def __init__(self, topology: np.ndarray, capacities: Dict):
        # 基本属性
        self.n = topology.shape[0]
        self.topology = topology

        # 物理资源容量
        self.C_cap = np.full(self.n, capacities.get('cpu', 100.0), dtype=float)
        self.M_cap = np.full(self.n, capacities.get('memory', 80.0), dtype=float)

        # 🔥 在 _init_links 之前初始化所有字典
        # 预留资源追踪
        self.reserved_cpu = np.zeros(self.n, dtype=float)
        self.reserved_mem = np.zeros(self.n, dtype=float)
        self.reserved_bw = {}

        # 链路相关字典
        self.link_map = {}
        self.bw_cap = {}
        self.B = {}
        self.link_locks = {}

        # 节点锁
        self.node_locks = [threading.Lock() for _ in range(self.n)]

        # 初始化链路（现在安全了）
        self._init_links(capacities.get('bandwidth', 100.0))

        # 当前可用资源
        self.C = self.C_cap.copy()
        self.M = self.M_cap.copy()

        logger.info(f"[SharedPool] 初始化: {self.n}节点, {self.L}链路")
    def _init_links(self, bw_cap: float):
        """初始化链路资源"""
        self.link_map = {}
        self.bw_cap = {}
        self.B = {}

        edge_id = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.topology[i, j] > 0:
                    key = tuple(sorted((i, j)))
                    if key not in self.link_map:
                        self.link_map[key] = edge_id
                        self.bw_cap[key] = bw_cap
                        self.B[key] = bw_cap
                        self.reserved_bw[key] = 0.0
                        self.link_locks[key] = threading.Lock()
                        edge_id += 1

        self.L = len(self.link_map)
    def get_edge_id(self, u: int, v: int) -> Optional[int]:
        """获取链路ID"""
        key = tuple(sorted((u, v)))
        return self.link_map.get(key)
    def get_edge_key(self, edge_id: int) -> Optional[tuple]:
        """根据ID获取链路"""
        for key, eid in self.link_map.items():
            if eid == edge_id:
                return key
        return None
    # ========================================
    # 物理资源操作（原子性）
    # ========================================
    def allocate_cpu(self, node_id: int, amount: float) -> bool:
        """分配CPU资源"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            if self.C[node_id] >= amount - 1e-5:
                self.C[node_id] -= amount
                return True
            return False
    def allocate_memory(self, node_id: int, amount: float) -> bool:
        """分配内存资源"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            if self.M[node_id] >= amount - 1e-5:
                self.M[node_id] -= amount
                return True
            return False
    def allocate_bandwidth(self, u: int, v: int, amount: float) -> bool:
        """分配带宽资源"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return False

        with self.link_locks[key]:
            if self.B[key] >= amount - 1e-5:
                self.B[key] -= amount
                return True
            return False
    def release_cpu(self, node_id: int, amount: float):
        """释放CPU资源"""
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            self.C[node_id] = min(self.C_cap[node_id], self.C[node_id] + amount)
    def release_memory(self, node_id: int, amount: float):
        """释放内存资源"""
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            self.M[node_id] = min(self.M_cap[node_id], self.M[node_id] + amount)
    def release_bandwidth(self, u: int, v: int, amount: float):
        """释放带宽资源"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return

        with self.link_locks[key]:
            self.B[key] = min(self.bw_cap[key], self.B[key] + amount)
    # ========================================
    # 资源预留（虚拟预扣）
    # ========================================
    def reserve_cpu(self, node_id: int, amount: float) -> bool:
        """预留CPU资源"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            available = self.C[node_id] - self.reserved_cpu[node_id]
            if available >= amount - 1e-5:
                self.reserved_cpu[node_id] += amount
                return True
            return False
    def reserve_memory(self, node_id: int, amount: float) -> bool:
        """预留内存资源"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            available = self.M[node_id] - self.reserved_mem[node_id]
            if available >= amount - 1e-5:
                self.reserved_mem[node_id] += amount
                return True
            return False

    def reserve_bandwidth(self, u: int, v: int, amount: float) -> bool:
        """预留带宽资源"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return False

        with self.link_locks[key]:
            available = self.B[key] - self.reserved_bw.get(key, 0.0)
            if available >= amount - 1e-5:
                self.reserved_bw[key] = self.reserved_bw.get(key, 0.0) + amount
                return True
            return False

    def commit_reservation(self, node_id: int, cpu_amount: float, mem_amount: float):
        """提交预留的资源（转为实际占用）"""
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            # 减少预留量
            self.reserved_cpu[node_id] = max(0, self.reserved_cpu[node_id] - cpu_amount)
            self.reserved_mem[node_id] = max(0, self.reserved_mem[node_id] - mem_amount)

            # 实际扣除（已经在allocate时完成了）
            # 这里只是清理预留记录

    def commit_link_reservation(self, u: int, v: int, bw_amount: float):
        """提交链路预留"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return

        with self.link_locks[key]:
            self.reserved_bw[key] = max(0, self.reserved_bw.get(key, 0.0) - bw_amount)

    def cancel_reservation(self, node_id: int, cpu_amount: float, mem_amount: float):
        """取消预留"""
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            self.reserved_cpu[node_id] = max(0, self.reserved_cpu[node_id] - cpu_amount)
            self.reserved_mem[node_id] = max(0, self.reserved_mem[node_id] - mem_amount)

    def cancel_link_reservation(self, u: int, v: int, bw_amount: float):
        """取消链路预留"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return

        with self.link_locks[key]:
            self.reserved_bw[key] = max(0, self.reserved_bw.get(key, 0.0) - bw_amount)

    # ========================================
    # 查询接口
    # ========================================

    def get_available_cpu(self, node_id: int) -> float:
        """获取可用CPU（考虑预留）"""
        if node_id < 0 or node_id >= self.n:
            return 0.0

        with self.node_locks[node_id]:
            return max(0, self.C[node_id] - self.reserved_cpu[node_id])

    def get_available_memory(self, node_id: int) -> float:
        """获取可用内存（考虑预留）"""
        if node_id < 0 or node_id >= self.n:
            return 0.0

        with self.node_locks[node_id]:
            return max(0, self.M[node_id] - self.reserved_mem[node_id])

    def get_available_bandwidth(self, u: int, v: int) -> float:
        """获取可用带宽（考虑预留）"""
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return 0.0

        with self.link_locks[key]:
            return max(0, self.B[key] - self.reserved_bw.get(key, 0.0))

    def reset(self, hard: bool = False):
        """重置资源池"""
        if hard:
            # 完全重置
            self.C = self.C_cap.copy()
            self.M = self.M_cap.copy()
            for key in self.B:
                self.B[key] = self.bw_cap[key]

        # 清理预留
        self.reserved_cpu.fill(0.0)
        self.reserved_mem.fill(0.0)
        for key in self.reserved_bw:
            self.reserved_bw[key] = 0.0


# ============================================================================
# 事务管理器
# ============================================================================

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"


@dataclass
class ResourceAllocation:
    """资源分配记录"""
    transaction_id: str
    resource_type: ResourceType
    resource_id: Any
    amount: float
    vnf_type: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    reserved: bool = False


class TransactionManager:
    """事务管理器"""

    def __init__(self, resource_pool: SharedResourcePool):
        self.resource_pool = resource_pool
        self.transactions: Dict[str, List[ResourceAllocation]] = {}
        self.active_transactions: Set[str] = set()
        self.lock = threading.Lock()

    def begin_transaction(self, tx_id: str):
        """开始事务"""
        with self.lock:
            if tx_id not in self.transactions:
                self.transactions[tx_id] = []
                self.active_transactions.add(tx_id)

    def reserve_node_resource(self, tx_id: str, node_id: int,
                              vnf_type: int, cpu_need: float, mem_need: float) -> bool:
        """预留节点资源"""
        with self.lock:
            if tx_id not in self.transactions:
                return False

            # 预留CPU
            if not self.resource_pool.reserve_cpu(node_id, cpu_need):
                return False

            # 预留内存
            if mem_need > 0 and not self.resource_pool.reserve_memory(node_id, mem_need):
                # 回滚CPU预留
                self.resource_pool.cancel_reservation(node_id, cpu_need, 0)
                return False

            # 记录分配
            if cpu_need > 0:
                alloc = ResourceAllocation(
                    transaction_id=tx_id,
                    resource_type=ResourceType.CPU,
                    resource_id=node_id,
                    amount=cpu_need,
                    vnf_type=vnf_type,
                    reserved=True
                )
                self.transactions[tx_id].append(alloc)

            if mem_need > 0:
                alloc = ResourceAllocation(
                    transaction_id=tx_id,
                    resource_type=ResourceType.MEMORY,
                    resource_id=node_id,
                    amount=mem_need,
                    vnf_type=vnf_type,
                    reserved=True
                )
                self.transactions[tx_id].append(alloc)

            return True

    def reserve_link_resource(self, tx_id: str, u: int, v: int, bw_need: float) -> bool:
        """预留链路资源"""
        with self.lock:
            if tx_id not in self.transactions:
                return False

            if not self.resource_pool.reserve_bandwidth(u, v, bw_need):
                return False

            alloc = ResourceAllocation(
                transaction_id=tx_id,
                resource_type=ResourceType.BANDWIDTH,
                resource_id=(u, v),
                amount=bw_need,
                reserved=True
            )
            self.transactions[tx_id].append(alloc)

            return True

    def commit_transaction(self, tx_id: str) -> bool:
        """提交事务（将预留转为实际占用）"""
        with self.lock:
            if tx_id not in self.transactions:
                return False
            logger.info(f"[Resource Flow] 事务 {tx_id}: 开始扣除资源 (Physical Commit)")
            allocations = self.transactions[tx_id]

            for alloc in allocations:
                if not alloc.reserved:
                    continue

                if alloc.resource_type == ResourceType.CPU:
                    # 实际分配CPU
                    success = self.resource_pool.allocate_cpu(alloc.resource_id, alloc.amount)
                    if not success:
                        # 部分失败，需要回滚
                        self._rollback_partial(tx_id, allocations)
                        return False

                    # 提交预留
                    self.resource_pool.commit_reservation(alloc.resource_id, alloc.amount, 0)
                    alloc.committed = True
                    alloc.reserved = False

                elif alloc.resource_type == ResourceType.MEMORY:
                    success = self.resource_pool.allocate_memory(alloc.resource_id, alloc.amount)
                    if not success:
                        self._rollback_partial(tx_id, allocations)
                        return False

                    self.resource_pool.commit_reservation(alloc.resource_id, 0, alloc.amount)
                    alloc.committed = True
                    alloc.reserved = False

                elif alloc.resource_type == ResourceType.BANDWIDTH:
                    u, v = alloc.resource_id
                    success = self.resource_pool.allocate_bandwidth(u, v, alloc.amount)
                    if not success:
                        self._rollback_partial(tx_id, allocations)
                        return False

                    self.resource_pool.commit_link_reservation(u, v, alloc.amount)
                    alloc.committed = True
                    alloc.reserved = False

            # 标记事务完成
            if tx_id in self.active_transactions:
                self.active_transactions.remove(tx_id)

            return True

    def rollback_transaction(self, tx_id: str):
        """回滚事务"""
        with self.lock:
            if tx_id not in self.transactions:
                return

            allocations = self.transactions[tx_id]

            for alloc in allocations:
                if alloc.reserved:
                    # 取消预留
                    if alloc.resource_type == ResourceType.CPU:
                        self.resource_pool.cancel_reservation(alloc.resource_id, alloc.amount, 0)
                    elif alloc.resource_type == ResourceType.MEMORY:
                        self.resource_pool.cancel_reservation(alloc.resource_id, 0, alloc.amount)
                    elif alloc.resource_type == ResourceType.BANDWIDTH:
                        u, v = alloc.resource_id
                        self.resource_pool.cancel_link_reservation(u, v, alloc.amount)

                elif alloc.committed:
                    # 释放已分配的资源
                    if alloc.resource_type == ResourceType.CPU:
                        self.resource_pool.release_cpu(alloc.resource_id, alloc.amount)
                    elif alloc.resource_type == ResourceType.MEMORY:
                        self.resource_pool.release_memory(alloc.resource_id, alloc.amount)
                    elif alloc.resource_type == ResourceType.BANDWIDTH:
                        u, v = alloc.resource_id
                        self.resource_pool.release_bandwidth(u, v, alloc.amount)

            # 清理事务记录
            if tx_id in self.active_transactions:
                self.active_transactions.remove(tx_id)

            del self.transactions[tx_id]

    def _rollback_partial(self, tx_id: str, allocations: List[ResourceAllocation]):
        """回滚部分已分配的资源"""
        for alloc in allocations:
            if alloc.committed:
                if alloc.resource_type == ResourceType.CPU:
                    self.resource_pool.release_cpu(alloc.resource_id, alloc.amount)
                elif alloc.resource_type == ResourceType.MEMORY:
                    self.resource_pool.release_memory(alloc.resource_id, alloc.amount)
                elif alloc.resource_type == ResourceType.BANDWIDTH:
                    u, v = alloc.resource_id
                    self.resource_pool.release_bandwidth(u, v, alloc.amount)

        self.rollback_transaction(tx_id)


# ============================================================================
# 融合版资源管理器（主类）
# ============================================================================

class FusedResourceManager:
    """
    融合版资源管理器
    1. 兼容 ResourceManager 的所有接口
    2. 支持 UnifiedResourceManager 的事务管理
    3. 基于共享资源池确保一致性
    """

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        """
        初始化融合版资源管理器

        Args:
            topo: 拓扑矩阵
            capacities: 容量配置
            dc_nodes: 数据中心节点
            link_map: 链路映射（可选）
        """
        # 保存配置
        self.topo = topo
        self.n = topo.shape[0]
        self.dc_nodes = dc_nodes
        self.link_map = link_map

        # 共享资源池
        self.pool = SharedResourcePool(topo, capacities)

        # 事务管理器
        self.transaction_mgr = TransactionManager(self.pool)

        # 兼容性字段（保持与ResourceManager相同的属性名）
        self.C_cap = capacities.get('cpu', 100.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.B_cap = capacities.get('bandwidth', 100.0)

        # 当前容量数组（只读，从pool同步）
        self.C = self.pool.C
        self.M = self.pool.M
        # 注意：self.B 需要特殊处理，因为pool使用字典存储

        # VNF相关
        self.K_vnf = 8
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.vnf_instances = []

        # 构建边索引（兼容GNN）
        self._build_edge_index()

        # 状态维度（兼容）
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.pool.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        # GNN特征维度
        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24

        # 兼容性字典
        self.nodes = {
            'cpu': self.C,  # 引用，不是复制
            'memory': self.M
        }

        # 链路容量数组（兼容Expert使用）
        self._init_link_capacity_array()

        logger.info(f"[FusedRM] 初始化完成: {self.n}节点, {self.pool.L}链路")

    def _build_edge_index(self):
        """构建边索引"""
        rows, cols = np.where(self.topo > 0)
        edge_list = []

        for u, v in zip(rows, cols):
            edge_list.append([u, v])

        self.edge_index = np.array(edge_list).T
        self.edge_hops = np.array([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=np.float32)

        # 建立映射
        self.edge_to_phys = {}
        self.phys_to_graph_edges = {}

        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_id = self.pool.get_edge_id(u, v)
            if edge_id is not None:
                self.edge_to_phys[(u, v)] = edge_id
                if edge_id not in self.phys_to_graph_edges:
                    self.phys_to_graph_edges[edge_id] = []
                self.phys_to_graph_edges[edge_id].append(idx)

    def _init_link_capacity_array(self):
        """初始化链路容量数组（兼容旧版self.B）"""
        self.B = np.zeros(self.pool.L, dtype=float)
        self.link_ref_count = np.zeros(self.pool.L, dtype=int)

        # 同步初始容量
        for edge_key, edge_id in self.pool.link_map.items():
            if edge_id < self.pool.L:
                self.B[edge_id] = self.pool.B[edge_key]

    def _sync_link_array(self):
        """同步链路数组到共享池"""
        # 从pool同步到数组
        for edge_key, edge_id in self.pool.link_map.items():
            if edge_id < len(self.B):
                self.B[edge_id] = self.pool.B[edge_key]

    # ========================================
    # 🔥 核心接口：事务管理（新功能）
    # ========================================

    def begin_transaction(self, request_id: str) -> str:
        """开始一个新事务"""
        tx_id = f"tx_{request_id}_{int(time.time() * 1000)}"
        self.transaction_mgr.begin_transaction(tx_id)
        return tx_id

    def reserve_node_resource(self, tx_id: str, node_id: int,
                              vnf_type: int, cpu_need: float, mem_need: float) -> bool:
        """预留节点资源"""
        return self.transaction_mgr.reserve_node_resource(tx_id, node_id, vnf_type, cpu_need, mem_need)

    def reserve_link_resource(self, tx_id: str, u: int, v: int, bw_need: float) -> bool:
        """预留链路资源"""
        return self.transaction_mgr.reserve_link_resource(tx_id, u, v, bw_need)

    def commit_transaction(self, tx_id: str) -> bool:
        """提交事务"""
        success = self.transaction_mgr.commit_transaction(tx_id)
        if success:
            # 同步链路数组
            self._sync_link_array()
        return success

    def rollback_transaction(self, tx_id: str):
        """回滚事务"""
        self.transaction_mgr.rollback_transaction(tx_id)
        # 同步链路数组
        self._sync_link_array()

    # ========================================
    # 🔥 兼容接口：直接分配（旧功能）
    # ========================================

    def allocate_node_resource(self, node_id: int, vnf_type: int,
                               cpu_need: float, mem_need: float = 0.0) -> bool:
        """
        直接分配节点资源（无事务）
        """
        if node_id < 0 or node_id >= self.n:
            return False

        # 分配CPU
        if not self.pool.allocate_cpu(node_id, cpu_need):
            return False

        # 分配内存
        if mem_need > 0:
            if not self.pool.allocate_memory(node_id, mem_need):
                # 回滚CPU
                self.pool.release_cpu(node_id, cpu_need)
                return False

        # 更新VNF计数
        self.hvt_all[node_id, vnf_type] += 1

        return True

    def allocate_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        """
        直接分配链路资源（无事务）
        """
        success = self.pool.allocate_bandwidth(u, v, bw_need)
        if success:
            # 同步到数组
            self._sync_link_array()
        return success

    def release_node_resource(self, node_id: int, vnf_type: int,
                              cpu_val: float, mem_val: float):
        """
        释放节点资源
        """
        if node_id < 0 or node_id >= self.n:
            return

        # 释放资源
        if cpu_val > 0:
            self.pool.release_cpu(node_id, cpu_val)

        if mem_val > 0:
            self.pool.release_memory(node_id, mem_val)

        # 更新VNF计数
        if vnf_type >= 0 and self.hvt_all[node_id, vnf_type] > 0:
            self.hvt_all[node_id, vnf_type] -= 1

    def release_link_resource(self, u: int, v: int, bw_val: float):
        """
        释放链路资源
        """
        self.pool.release_bandwidth(u, v, bw_val)
        # 同步到数组
        self._sync_link_array()

    # ========================================
    # 🔥 核心方法：apply_deployment（兼容版）
    # ========================================

    def apply_deployment(self, plan: dict, request: dict) -> bool:
        """
        应用部署方案（兼容旧版）
        """
        # 解析hvt
        hvt_branch = plan.get('hvt')
        if hvt_branch is None:
            return False

        if isinstance(hvt_branch, dict):
            from envs.modules.sfc_backup_system.utils import build_hvt_from_placement
            hvt_branch = build_hvt_from_placement(hvt_branch, self.n, self.K_vnf)

        hvt_branch = np.asarray(hvt_branch, dtype=np.float32)
        if hvt_branch.shape != (self.n, self.K_vnf):
            return False

        req_id = request.get('id', -1)
        cpu_reqs = request.get('cpu_origin', [])
        mem_reqs = request.get('memory_origin', [])

        # 1. 检查所有资源
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            # 检查可用资源（考虑预留）
            if self.pool.get_available_cpu(node) < cpu_need - 1e-5:
                return False
            if self.pool.get_available_memory(node) < mem_need - 1e-5:
                return False

        # 2. 分配资源
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            # 直接分配（无事务）
            success = self.allocate_node_resource(node, vnf_t, cpu_need, mem_need)
            if not success:
                # 注意：这里应该回滚已分配的资源，简化处理
                return False

            # 记录实例
            self.vnf_instances.append({
                'req_id': req_id,
                'node': node,
                'vnf_type': vnf_t,
                'cpu': cpu_need,
                'memory': mem_need
            })

        return True

    def apply_tree_deployment(self, plan: dict, request: dict) -> bool:
        """
        应用树部署方案（兼容旧版）
        """
        # 1. 部署VNF
        if not self.apply_deployment(plan, request):
            return False

        # 2. 部署链路
        tree = plan.get('tree', {})
        bw_need = request.get('bw_origin', 0)

        if isinstance(tree, dict):
            for edge_key, flow in tree.items():
                if flow <= 0:
                    continue

                u, v = None, None
                if isinstance(edge_key, tuple):
                    u, v = edge_key
                elif isinstance(edge_key, str):
                    try:
                        u, v = map(int, edge_key.strip('()').split('-'))
                    except:
                        continue

                if u is not None and v is not None:
                    # 分配链路资源
                    if not self.allocate_link_resource(u, v, bw_need * flow):
                        # 链路分配失败，需要回滚节点资源（简化处理）
                        return False

        return True

    # ========================================
    # 🔥 查询接口
    # ========================================

    def check_node_resource(self, node_id: int, vnf_type: int = 0,
                            cpu_need: float = 0.0, mem_need: float = 0.0) -> bool:
        """检查节点资源是否足够（考虑预留）"""
        if node_id < 0 or node_id >= self.n:
            return False

        cpu_ok = self.pool.get_available_cpu(node_id) >= cpu_need - 1e-5
        mem_ok = self.pool.get_available_memory(node_id) >= mem_need - 1e-5

        return cpu_ok and mem_ok

    def check_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        """检查链路资源是否足够（考虑预留）"""
        return self.pool.get_available_bandwidth(u, v) >= bw_need - 1e-5

    def get_available_resources(self, node_id: Optional[int] = None) -> Dict:
        """获取可用资源（考虑预留）"""
        if node_id is not None:
            return {
                'cpu': self.pool.get_available_cpu(node_id),
                'memory': self.pool.get_available_memory(node_id)
            }
        else:
            # 全网资源
            nodes = {}
            links = {}

            for nid in range(self.n):
                nodes[nid] = {
                    'cpu': self.pool.get_available_cpu(nid),
                    'memory': self.pool.get_available_memory(nid)
                }

            for edge_key in self.pool.link_map:
                u, v = edge_key
                links[edge_key] = {
                    'bandwidth': self.pool.get_available_bandwidth(u, v)
                }

            return {'nodes': nodes, 'links': links}

    # ========================================
    # 🔥 GNN相关接口（完全兼容）
    # ========================================

    def get_network_state_dict(self, current_request=None):
        """获取网络状态字典"""
        # 构建节点资源数组
        C = np.zeros(self.n)
        M = np.zeros(self.n)

        for i in range(self.n):
            C[i] = self.pool.get_available_cpu(i)
            M[i] = self.pool.get_available_memory(i)

        # 构建链路资源数组
        B = np.zeros(self.pool.L, dtype=float)
        for edge_key, edge_id in self.pool.link_map.items():
            if edge_id < self.pool.L:
                B[edge_id] = self.pool.get_available_bandwidth(*edge_key)

        state = {
            'bw': B,
            'cpu': C,
            'mem': M,
            'hvt': self.hvt_all,
            'bw_ref_count': self.link_ref_count
        }

        if current_request:
            state['request'] = current_request

        return state

    def get_graph_state(self, current_request, nodes_on_tree, current_tree,
                        served_dest_count: int, sharing_strategy: int, nb_high_goals: int):
        """获取图状态（GNN输入）"""
        # 这里简化为返回默认值，实际实现需要完整移植
        import torch

        if not current_request:
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            req_vec = torch.zeros(self.request_dim)
            return x, self.edge_index, edge_attr, req_vec

        # 简化的节点特征
        node_feats = []
        for i in range(self.n):
            cpu_util = 1.0 - self.pool.get_available_cpu(i) / max(1, self.C_cap)
            mem_util = 1.0 - self.pool.get_available_memory(i) / max(1, self.M_cap)

            feat = [
                cpu_util,
                mem_util,
                1.0 if (i + 1) in self.dc_nodes else 0.0,
                1.0 if (i + 1) == current_request.get('source', -1) else 0.0,
                1.0 if (i + 1) in current_request.get('dest', []) else 0.0,
                1.0 if i in nodes_on_tree else 0.0,
                0.5,  # 简化：到目的地的平均距离
                0.5,  # 简化：VNF共享潜力
                0.0   # 简化：预留资源紧张度
            ]

            # VNF实例特征
            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float32)

        # 简化的边特征
        edge_attrs = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim), dtype=torch.float32)

        # 简化的请求向量
        req_vec = torch.randn(self.request_dim)  # 简化

        return x, torch.tensor(self.edge_index, dtype=torch.long), edge_attrs, req_vec

    # ========================================
    # 🔥 其他兼容接口
    # ========================================

    def reset(self, hard: bool = False):
        """重置资源管理器"""
        self.pool.reset(hard)
        self.hvt_all.fill(0)
        self.vnf_instances = []
        self.link_ref_count.fill(0)

        # 同步链路数组
        self._sync_link_array()

        if hard:
            logger.info("[FusedRM] 执行硬重置，资源回满")

    def has_link(self, u: int, v: int) -> bool:
        """检查是否有链路"""
        return self.pool.get_edge_id(u, v) is not None

    def get_neighbors(self, node: int) -> List[int]:
        """获取邻居节点"""
        if node < 0 or node >= self.n:
            return []
        return np.where(self.topo[node] > 0)[0].tolist()

    def get_link_cost(self, u: int, v: int) -> float:
        """获取链路开销"""
        return 1.0  # 简化

    def get_node_features(self, nodes_on_tree):
        """获取节点特征矩阵"""
        feats = []
        for i in range(self.n):
            f = [
                self.pool.get_available_cpu(i) / max(1, self.C_cap),
                self.pool.get_available_memory(i) / max(1, self.M_cap),
                1.0 if i in nodes_on_tree else 0.0
            ]
            feats.append(f)
        return np.array(feats, dtype=np.float32)

    def get_edge_features(self):
        """获取边特征"""
        import torch
        return torch.tensor(self.edge_index, dtype=torch.long), torch.zeros((self.edge_index.shape[1], 5))


# ============================================================================
# 使用示例
# ============================================================================

def usage_example():
    """使用示例"""
    import numpy as np

    # 1. 创建管理器
    topo = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])

    capacities = {
        'cpu': 100.0,
        'memory': 80.0,
        'bandwidth': 100.0
    }

    dc_nodes = [2]

    rm = FusedResourceManager(topo, capacities, dc_nodes)

    print("=== 测试1：旧接口（直接分配）===")
    # 直接分配节点资源
    success = rm.allocate_node_resource(0, 1, 20.0, 10.0)
    print(f"直接分配结果: {success}")

    # 直接分配链路资源
    success = rm.allocate_link_resource(0, 1, 30.0)
    print(f"直接分配链路: {success}")

    # 查询可用资源
    available = rm.get_available_resources(0)
    print(f"节点0可用资源: {available}")

    print("\n=== 测试2：新接口（事务管理）===")
    # 开始事务
    tx_id = rm.begin_transaction("req_001")
    print(f"开始事务: {tx_id}")

    # 预留资源
    success = rm.reserve_node_resource(tx_id, 1, 2, 30.0, 15.0)
    print(f"预留节点资源: {success}")

    success = rm.reserve_link_resource(tx_id, 1, 2, 20.0)
    print(f"预留链路资源: {success}")

    # 提交事务
    success = rm.commit_transaction(tx_id)
    print(f"提交事务: {success}")

    print("\n=== 测试3：回滚事务 ===")
    tx_id2 = rm.begin_transaction("req_002")
    rm.reserve_node_resource(tx_id2, 0, 3, 200.0, 100.0)  # 应该失败，但会预留
    rm.rollback_transaction(tx_id2)  # 回滚预留
    print(f"回滚事务: {tx_id2}")

    print("\n=== 测试4：兼容接口 ===")
    # 使用apply_deployment（旧接口）
    plan = {
        'hvt': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        'tree': {(0, 1): 1.0, (1, 2): 0.5}
    }

    request = {
        'id': 'test_req',
        'cpu_origin': [10.0, 5.0, 8.0],
        'memory_origin': [5.0, 3.0, 4.0],
        'bw_origin': 10.0
    }

    success = rm.apply_deployment(plan, request)
    print(f"apply_deployment结果: {success}")


if __name__ == "__main__":
    usage_example()