"""
envs/modules/fused_resource_manager.py
==================================================
融合版资源管理器 - 修复版
==================================================
修复内容：
1. 预留时立即扣除资源（避免双重扣除Bug）
2. commit时不再调用allocate（避免重复检查）
3. cancel时归还资源（确保回滚正确）
4. 统一资源查询逻辑
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import traceback
import copy

logger = logging.getLogger(__name__)

class SharedResourcePool:
    """
    共享资源池 - 所有资源管理器共享的底层资源
    🔥 修复版：预留时立即扣除资源，避免并发冲突
    """

    def __init__(self, topology: np.ndarray, capacities: Dict):
        # 基本属性
        self.n = topology.shape[0]
        self.topology = topology

        # 物理资源容量
        self.C_cap = np.full(self.n, capacities.get('cpu', 100.0), dtype=float)
        self.M_cap = np.full(self.n, capacities.get('memory', 80.0), dtype=float)

        # 🔥 预留资源追踪（标记哪些资源是预留状态）
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

        # 初始化链路
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
        """分配CPU资源（直接模式，不走事务）"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            if self.C[node_id] >= amount - 1e-5:
                self.C[node_id] -= amount
                return True
            return False

    def allocate_memory(self, node_id: int, amount: float) -> bool:
        """分配内存资源（直接模式，不走事务）"""
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            if self.M[node_id] >= amount - 1e-5:
                self.M[node_id] -= amount
                return True
            return False

    def allocate_bandwidth(self, u: int, v: int, amount: float) -> bool:
        """分配带宽资源（直接模式，不走事务）"""
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
    # 🔥 修复：资源预留（立即扣除）
    # ========================================
    def reserve_cpu(self, node_id: int, amount: float) -> bool:
        """
        🔥 修复版：预留CPU资源（立即扣除）

        旧版问题：只标记不扣除，导致提交时可能失败
        新版修复：预留时立即扣除，确保提交一定成功
        """
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            # 🔥 修复：直接检查并扣除可用资源
            if self.C[node_id] >= amount - 1e-5:
                self.C[node_id] -= amount
                self.reserved_cpu[node_id] += amount  # 标记为预留状态
                return True
            return False

    def reserve_memory(self, node_id: int, amount: float) -> bool:
        """
        🔥 修复版：预留内存资源（立即扣除）
        """
        if node_id < 0 or node_id >= self.n:
            return False

        with self.node_locks[node_id]:
            # 🔥 修复：直接检查并扣除可用资源
            if self.M[node_id] >= amount - 1e-5:
                self.M[node_id] -= amount
                self.reserved_mem[node_id] += amount  # 标记为预留状态
                return True
            return False

    def reserve_bandwidth(self, u: int, v: int, amount: float) -> bool:
        """
        🔥 修复版：预留带宽资源（立即扣除）
        """
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return False

        with self.link_locks[key]:
            # 🔥 修复：直接检查并扣除可用资源
            if self.B[key] >= amount - 1e-5:
                self.B[key] -= amount
                self.reserved_bw[key] = self.reserved_bw.get(key, 0.0) + amount
                return True
            return False

    def commit_reservation(self, node_id: int, cpu_amount: float, mem_amount: float):
        """
        🔥 修复版：提交预留（只清除标记）

        旧版问题：注释说"资源已在allocate时扣除"，但预留时没扣
        新版修复：资源已在reserve时扣除，这里只清除预留标记
        """
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            # 🔥 修复：只清除预留标记，资源已在reserve时扣除
            self.reserved_cpu[node_id] = max(0, self.reserved_cpu[node_id] - cpu_amount)
            self.reserved_mem[node_id] = max(0, self.reserved_mem[node_id] - mem_amount)

    def commit_link_reservation(self, u: int, v: int, bw_amount: float):
        """
        🔥 修复版：提交链路预留（只清除标记）
        """
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return

        with self.link_locks[key]:
            # 🔥 修复：只清除预留标记
            self.reserved_bw[key] = max(0, self.reserved_bw.get(key, 0.0) - bw_amount)

    def cancel_reservation(self, node_id: int, cpu_amount: float, mem_amount: float):
        """
        🔥 修复版：取消预留（归还资源）

        旧版问题：只清除标记，不归还资源
        新版修复：归还资源并清除标记
        """
        if node_id < 0 or node_id >= self.n:
            return

        with self.node_locks[node_id]:
            # 🔥 修复：归还资源
            if cpu_amount > 0:
                self.C[node_id] = min(self.C_cap[node_id], self.C[node_id] + cpu_amount)
                self.reserved_cpu[node_id] = max(0, self.reserved_cpu[node_id] - cpu_amount)

            if mem_amount > 0:
                self.M[node_id] = min(self.M_cap[node_id], self.M[node_id] + mem_amount)
                self.reserved_mem[node_id] = max(0, self.reserved_mem[node_id] - mem_amount)

    def cancel_link_reservation(self, u: int, v: int, bw_amount: float):
        """
        🔥 修复版：取消链路预留（归还资源）
        """
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return

        with self.link_locks[key]:
            # 🔥 修复：归还资源
            self.B[key] = min(self.bw_cap[key], self.B[key] + bw_amount)
            self.reserved_bw[key] = max(0, self.reserved_bw.get(key, 0.0) - bw_amount)

    # ========================================
    # 🔥 修复：查询接口（语义统一）
    # ========================================

    def get_available_cpu(self, node_id: int) -> float:
        """
        🔥 修复版：获取可用CPU

        语义：返回真实可用量（预留的资源已经扣除了）
        """
        if node_id < 0 or node_id >= self.n:
            return 0.0

        with self.node_locks[node_id]:
            # 🔥 修复：预留资源已扣除，直接返回C
            return max(0, self.C[node_id])

    def get_available_memory(self, node_id: int) -> float:
        """
        🔥 修复版：获取可用内存
        """
        if node_id < 0 or node_id >= self.n:
            return 0.0

        with self.node_locks[node_id]:
            # 🔥 修复：预留资源已扣除，直接返回M
            return max(0, self.M[node_id])

    def get_available_bandwidth(self, u: int, v: int) -> float:
        """
        🔥 修复版：获取可用带宽
        """
        key = tuple(sorted((u, v)))
        if key not in self.B:
            return 0.0

        with self.link_locks[key]:
            # 🔥 修复：预留资源已扣除，直接返回B
            return max(0, self.B[key])

    def reset(self, hard: bool = False):
        """重置资源池"""
        if hard:
            # 完全重置
            self.C = self.C_cap.copy()
            self.M = self.M_cap.copy()
            for key in self.B:
                self.B[key] = self.bw_cap[key]

        # 清理预留标记
        self.reserved_cpu.fill(0.0)
        self.reserved_mem.fill(0.0)
        for key in self.reserved_bw:
            self.reserved_bw[key] = 0.0
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
    """
    事务管理器 - 修复版
    🔥 修复：commit时不再调用allocate，避免重复扣除
    """

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
        """
        🔥 修复版：提交事务（不再调用allocate）

        旧版问题：
        1. reserve时只标记不扣除
        2. commit时调用allocate再次检查和扣除
        3. 导致预留可能被绕过

        新版修复：
        1. reserve时已经扣除资源
        2. commit时只清除预留标记
        3. 确保预留一定成功
        """
        with self.lock:
            if tx_id not in self.transactions:
                return False

            logger.info(f"[Resource Flow] 事务 {tx_id}: 提交事务 (Commit)")
            allocations = self.transactions[tx_id]

            for alloc in allocations:
                if not alloc.reserved:
                    continue

                # 🔥 修复：不再调用allocate，直接commit标记
                if alloc.resource_type == ResourceType.CPU:
                    self.resource_pool.commit_reservation(alloc.resource_id, alloc.amount, 0)
                    alloc.committed = True
                    alloc.reserved = False

                elif alloc.resource_type == ResourceType.MEMORY:
                    self.resource_pool.commit_reservation(alloc.resource_id, 0, alloc.amount)
                    alloc.committed = True
                    alloc.reserved = False

                elif alloc.resource_type == ResourceType.BANDWIDTH:
                    u, v = alloc.resource_id
                    self.resource_pool.commit_link_reservation(u, v, alloc.amount)
                    alloc.committed = True
                    alloc.reserved = False

            # 标记事务完成
            if tx_id in self.active_transactions:
                self.active_transactions.remove(tx_id)

            logger.info(f"✅ [Resource Flow] 事务 {tx_id}: 提交成功")
            return True

    def rollback_transaction(self, tx_id: str):
        """
        🔥 修复版：回滚事务（正确归还资源）
        """
        with self.lock:
            if tx_id not in self.transactions:
                return

            logger.info(f"♻️ [Resource Flow] 事务 {tx_id}: 回滚")
            allocations = self.transactions[tx_id]

            for alloc in allocations:
                if alloc.reserved:
                    # 🔥 修复：取消预留会归还资源
                    if alloc.resource_type == ResourceType.CPU:
                        self.resource_pool.cancel_reservation(alloc.resource_id, alloc.amount, 0)
                    elif alloc.resource_type == ResourceType.MEMORY:
                        self.resource_pool.cancel_reservation(alloc.resource_id, 0, alloc.amount)
                    elif alloc.resource_type == ResourceType.BANDWIDTH:
                        u, v = alloc.resource_id
                        self.resource_pool.cancel_link_reservation(u, v, alloc.amount)

                elif alloc.committed:
                    # 释放已提交的资源（这种情况不应该发生，但保留兼容）
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
            logger.info(f"✅ [Resource Flow] 事务 {tx_id}: 回滚完成")

    def _rollback_partial(self, tx_id: str, allocations: List[ResourceAllocation]):
        """回滚部分已分配的资源（兼容旧版）"""
        logger.warning(f"⚠️ 部分回滚: {tx_id}")
        self.rollback_transaction(tx_id)
# 请求生命周期管理器
class RequestLifecycleManager:
    """
    请求生命周期管理器
    负责跟踪请求的生命周期，并在过期时自动释放资源

    主要功能：
    1. 注册请求及其资源分配
    2. 跟踪请求的过期时间
    3. 定期检查并释放过期请求的资源
    4. 提供请求状态查询
    5. 统计和监控
    """

    def __init__(self, resource_manager):
        """
        初始化生命周期管理器

        Args:
            resource_manager: 资源管理器实例，用于释放资源
        """
        self.resource_manager = resource_manager
        self.active_requests = {}  # 活跃请求字典 {req_id: request_info}
        self.expired_requests = {}  # 已过期请求（用于审计）
        self.lock = threading.RLock()  # 可重入锁，用于线程安全

        # 统计信息
        self.stats = {
            'total_registered': 0,
            'total_expired': 0,
            'total_failed': 0,
            'total_succeeded': 0,
            'total_cpu_released': 0.0,
            'total_mem_released': 0.0,
            'total_bw_released': 0.0,
            'max_concurrent_requests': 0,
            'avg_request_lifetime': 0.0
        }

        # 清理间隔（秒）
        self.cleanup_interval = 1.0
        self.last_cleanup_time = time.time()

        logger.info("[LifecycleManager] 初始化完成")

    def register_request(self, request: dict, resources_allocated: dict) -> bool:
        """
        注册请求到生命周期管理

        Args:
            request: 请求字典，必须包含：
                    - id: 请求ID
                    - arrival_time: 到达时间
                    - lifetime: 生存时间（秒）
            resources_allocated: 已分配的资源信息，格式：
                    {
                        'placement': {(node_id, vnf_type): {'cpu_used': x, 'mem_used': y}, ...},
                        'tree': {(u, v): flow_amount, ...}
                    }

        Returns:
            bool: 是否注册成功
        """
        req_id = request.get('id')
        if not req_id:
            logger.error(f"[LifecycleManager] 请求缺少ID: {request}")
            return False

        arrival_time = request.get('arrival_time', time.time())
        lifetime = request.get('lifetime', 5.0)  # 默认5秒

        # 计算过期时间
        expire_time = arrival_time + lifetime

        with self.lock:
            # 检查是否已存在
            if req_id in self.active_requests:
                logger.warning(f"[LifecycleManager] 请求 {req_id} 已存在，跳过注册")
                return False

            # 构建请求记录
            request_record = {
                'request': copy.deepcopy(request),
                'resources_allocated': copy.deepcopy(resources_allocated),
                'arrival_time': arrival_time,
                'lifetime': lifetime,
                'expire_time': expire_time,
                'status': 'active',
                'register_time': time.time()
            }

            # 添加到活跃请求
            self.active_requests[req_id] = request_record

            # 更新统计
            self.stats['total_registered'] += 1

            # 更新最大并发数
            concurrent = len(self.active_requests)
            if concurrent > self.stats['max_concurrent_requests']:
                self.stats['max_concurrent_requests'] = concurrent

            logger.info(f"[LifecycleManager] 注册请求 {req_id}: "
                        f"到达时间={arrival_time:.2f}s, "
                        f"生命周期={lifetime:.2f}s, "
                        f"过期时间={expire_time:.2f}s")

            return True

    def mark_request_failed(self, req_id: str):
        """标记请求为失败状态"""
        with self.lock:
            if req_id in self.active_requests:
                self.active_requests[req_id]['status'] = 'failed'
                self.stats['total_failed'] += 1
                logger.info(f"[LifecycleManager] 请求 {req_id} 标记为失败")

    def check_and_release_expired(self, current_time: float = None) -> List[str]:
        """
        检查并释放过期的请求

        Args:
            current_time: 当前时间，如果为None则使用time.time()

        Returns:
            List[str]: 已释放的请求ID列表
        """
        if current_time is None:
            current_time = time.time()

        # 控制清理频率
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return []

        self.last_cleanup_time = current_time
        expired_req_ids = []

        with self.lock:
            # 找出所有过期请求
            for req_id, req_info in list(self.active_requests.items()):
                if current_time > req_info['expire_time']:
                    expired_req_ids.append(req_id)

            # 释放过期请求的资源
            for req_id in expired_req_ids:
                self._release_request_resources(req_id, current_time)

        return expired_req_ids

    def _release_request_resources(self, req_id: str, current_time: float):
        """
        释放请求占用的资源（内部方法）

        Args:
            req_id: 请求ID
            current_time: 当前时间
        """
        if req_id not in self.active_requests:
            return

        req_info = self.active_requests[req_id]
        resources = req_info.get('resources_allocated', {})

        # 统计释放的资源量
        cpu_released = 0.0
        mem_released = 0.0
        bw_released = 0.0

        # 1. 释放节点资源（VNF实例）
        placement = resources.get('placement', {})
        for key, alloc_info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node_id = key[0]
                vnf_type = key[1]

                # 获取资源使用量
                cpu_used = alloc_info.get('cpu_used', 1.0)
                mem_used = alloc_info.get('mem_used', 1.0)

                # 调用资源管理器释放资源
                try:
                    if hasattr(self.resource_manager, 'release_node_resource'):
                        self.resource_manager.release_node_resource(
                            node_id, vnf_type, cpu_used, mem_used
                        )
                        cpu_released += cpu_used
                        mem_released += mem_used
                except Exception as e:
                    logger.error(f"[LifecycleManager] 释放节点资源失败 {req_id}: {e}")

        # 2. 释放链路资源
        tree_edges = resources.get('tree', {})
        request = req_info.get('request', {})
        bw_needed = request.get('bw_origin', 1.0)

        for edge_key, flow in tree_edges.items():
            if isinstance(edge_key, tuple) and len(edge_key) >= 2:
                u, v = edge_key[0], edge_key[1]

                # 计算实际带宽使用量
                bw_used = bw_needed * flow

                # 调用资源管理器释放资源
                try:
                    if hasattr(self.resource_manager, 'release_link_resource'):
                        self.resource_manager.release_link_resource(u, v, bw_used)
                        bw_released += bw_used
                except Exception as e:
                    logger.error(f"[LifecycleManager] 释放链路资源失败 {req_id}: {e}")

        # 3. 更新统计和状态
        with self.lock:
            # 更新请求状态
            req_info['status'] = 'expired'
            req_info['release_time'] = current_time
            req_info['actual_lifetime'] = current_time - req_info['arrival_time']

            # 移到过期记录
            self.expired_requests[req_id] = req_info
            del self.active_requests[req_id]

            # 更新统计
            self.stats['total_expired'] += 1
            self.stats['total_cpu_released'] += cpu_released
            self.stats['total_mem_released'] += mem_released
            self.stats['total_bw_released'] += bw_released

            # 计算平均生命周期
            if self.stats['total_expired'] > 0:
                total_lifetime = sum(
                    r.get('actual_lifetime', r['lifetime'])
                    for r in self.expired_requests.values()
                )
                self.stats['avg_request_lifetime'] = total_lifetime / self.stats['total_expired']

        # 记录日志
        logger.info(f"[LifecycleManager] 释放请求 {req_id} 资源: "
                    f"CPU={cpu_released:.1f}, "
                    f"Mem={mem_released:.1f}, "
                    f"BW={bw_released:.1f}, "
                    f"实际生命周期={req_info.get('actual_lifetime', 0):.2f}s")

        # 打印详细信息
        print(f"⏰ [生命周期] 请求 {req_id} 已过期释放")
        print(f"   到达时间: {req_info['arrival_time']:.2f}s")
        print(f"   计划生命周期: {req_info['lifetime']:.2f}s")
        print(f"   实际生命周期: {req_info.get('actual_lifetime', 0):.2f}s")
        print(f"   释放资源: CPU={cpu_released:.1f}, Mem={mem_released:.1f}, BW={bw_released:.1f}")

    def force_release_request(self, req_id: str) -> bool:
        """
        强制释放请求（用于请求失败等场景）

        Args:
            req_id: 请求ID

        Returns:
            bool: 是否释放成功
        """
        with self.lock:
            if req_id not in self.active_requests:
                logger.warning(f"[LifecycleManager] 请求 {req_id} 不存在，无法强制释放")
                return False

            # 标记为失败
            self.active_requests[req_id]['status'] = 'force_released'
            self.stats['total_failed'] += 1

            # 释放资源
            self._release_request_resources(req_id, time.time())

            logger.info(f"[LifecycleManager] 强制释放请求 {req_id}")
            return True

    def get_request_status(self, req_id: str) -> Optional[dict]:
        """
        获取请求状态

        Args:
            req_id: 请求ID

        Returns:
            dict: 请求状态信息，或None如果不存在
        """
        with self.lock:
            if req_id in self.active_requests:
                info = self.active_requests[req_id]
                remaining = max(0, info['expire_time'] - time.time())
                return {
                    'req_id': req_id,
                    'status': info['status'],
                    'arrival_time': info['arrival_time'],
                    'lifetime': info['lifetime'],
                    'expire_time': info['expire_time'],
                    'remaining_time': remaining,
                    'resources_allocated': info.get('resources_allocated', {})
                }
            elif req_id in self.expired_requests:
                info = self.expired_requests[req_id]
                return {
                    'req_id': req_id,
                    'status': info['status'],
                    'arrival_time': info['arrival_time'],
                    'lifetime': info['lifetime'],
                    'actual_lifetime': info.get('actual_lifetime', 0),
                    'release_time': info.get('release_time'),
                    'expired': True
                }

        return None

    def get_active_requests_count(self) -> int:
        """获取活跃请求数量"""
        with self.lock:
            return len(self.active_requests)

    def get_all_active_requests(self) -> List[dict]:
        """获取所有活跃请求信息"""
        with self.lock:
            result = []
            for req_id, info in self.active_requests.items():
                remaining = max(0, info['expire_time'] - time.time())
                result.append({
                    'req_id': req_id,
                    'arrival_time': info['arrival_time'],
                    'expire_time': info['expire_time'],
                    'remaining_time': remaining,
                    'status': info['status']
                })
            return result

    def get_statistics(self) -> dict:
        """获取统计信息"""
        with self.lock:
            stats = copy.deepcopy(self.stats)
            stats['current_active_requests'] = len(self.active_requests)
            stats['total_expired_requests'] = len(self.expired_requests)
            stats['cleanup_interval'] = self.cleanup_interval

            # 计算资源利用率
            if stats['total_cpu_released'] > 0:
                stats['avg_cpu_per_request'] = stats['total_cpu_released'] / max(1, stats['total_expired'])
                stats['avg_mem_per_request'] = stats['total_mem_released'] / max(1, stats['total_expired'])
                stats['avg_bw_per_request'] = stats['total_bw_released'] / max(1, stats['total_expired'])

            return stats

    def reset_statistics(self):
        """重置统计信息"""
        with self.lock:
            self.stats = {
                'total_registered': 0,
                'total_expired': 0,
                'total_failed': 0,
                'total_succeeded': 0,
                'total_cpu_released': 0.0,
                'total_mem_released': 0.0,
                'total_bw_released': 0.0,
                'max_concurrent_requests': 0,
                'avg_request_lifetime': 0.0
            }
            logger.info("[LifecycleManager] 统计信息已重置")

    def cleanup_all(self):
        """清理所有请求（用于重置场景）"""
        with self.lock:
            current_time = time.time()
            req_ids = list(self.active_requests.keys())

            for req_id in req_ids:
                self._release_request_resources(req_id, current_time)

            logger.info(f"[LifecycleManager] 清理所有请求，共 {len(req_ids)} 个")

    def set_cleanup_interval(self, interval: float):
        """
        设置清理间隔

        Args:
            interval: 清理间隔（秒）
        """
        if interval > 0:
            self.cleanup_interval = interval
            logger.info(f"[LifecycleManager] 清理间隔设置为 {interval} 秒")
# 融合版资源管理器
class FusedResourceManager:
    """
    融合版资源管理器 - 修复版
    🔥 修复内容：
    1. 预留机制修复（立即扣除）
    2. 事务提交修复（不重复扣除）
    3. 回滚机制修复（正确归还）
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
        self.tools = {}
        # 🔥 修复版共享资源池
        self.pool = SharedResourcePool(topo, capacities)

        # 🔥 修复版事务管理器
        self.transaction_mgr = TransactionManager(self.pool)

        # 兼容性字段
        self.C_cap = capacities.get('cpu', 100.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.B_cap = capacities.get('bandwidth', 100.0)

        # 当前容量数组（引用pool）
        self.C = self.pool.C
        self.M = self.pool.M

        # VNF相关
        self.K_vnf = 8
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.vnf_instances = []

        # 构建边索引
        self._build_edge_index()

        # 状态维度
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.pool.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        # GNN特征维度
        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24

        # 兼容性字典
        self.nodes = {
            'cpu': self.C,
            'memory': self.M
        }

        # 链路容量数组
        self._init_link_capacity_array()

        # 集成方法所需的状态变量
        self.current_request = None
        self.current_tree = {}
        self.current_phase = 'idle'
        self.next_vnf_idx = 0
        self.total_requests_accepted = 0
        self.served_dest_count = 0
        self.nodes_on_tree = set()
        #初始化请求生命周期管理器
        self.request_manager = RequestLifecycleManager(self)
        logger.info(f"[FusedRM] 初始化完成: {self.n}节点, {self.pool.L}链路 (修复版)")

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
        """初始化链路容量数组"""
        self.B = np.zeros(self.pool.L, dtype=float)
        self.link_ref_count = np.zeros(self.pool.L, dtype=int)

        # 同步初始容量
        for edge_key, edge_id in self.pool.link_map.items():
            if edge_id < self.pool.L:
                self.B[edge_id] = self.pool.B[edge_key]

    def _sync_link_array(self):
        """同步链路数组到共享池"""
        for edge_key, edge_id in self.pool.link_map.items():
            if edge_id < len(self.B):
                self.B[edge_id] = self.pool.B[edge_key]

    # ========================================
    # 🔥 核心接口：事务管理（修复版）
    # ========================================

    def begin_transaction(self, request_id: str) -> str:
        """开始一个新事务"""
        tx_id = f"tx_{request_id}_{int(time.time() * 1000)}"
        self.transaction_mgr.begin_transaction(tx_id)
        logger.debug(f"[Transaction] 开始事务: {tx_id}")
        return tx_id

    def reserve_node_resource(self, tx_id: str, node_id: int,
                              vnf_type: int, cpu_need: float, mem_need: float) -> bool:
        """预留节点资源（修复版：立即扣除）"""
        return self.transaction_mgr.reserve_node_resource(tx_id, node_id, vnf_type, cpu_need, mem_need)

    def reserve_link_resource(self, tx_id: str, u: int, v: int, bw_need: float) -> bool:
        """预留链路资源（修复版：立即扣除）"""
        return self.transaction_mgr.reserve_link_resource(tx_id, u, v, bw_need)

    def commit_transaction(self, tx_id: str) -> bool:
        """提交事务（修复版：不重复扣除）"""
        success = self.transaction_mgr.commit_transaction(tx_id)
        if success:
            self._sync_link_array()
        return success

    def rollback_transaction(self, tx_id: str):
        """回滚事务（修复版：正确归还）"""
        self.transaction_mgr.rollback_transaction(tx_id)
        self._sync_link_array()

    # ========================================
    # 兼容接口：直接分配（旧功能）
    # ========================================

    def allocate_node_resource(self, node_id: int, vnf_type: int,
                               cpu_need: float, mem_need: float = 0.0) -> bool:
        """直接分配节点资源（无事务）"""
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
        """直接分配链路资源（无事务）"""
        success = self.pool.allocate_bandwidth(u, v, bw_need)
        if success:
            self._sync_link_array()
        return success

    def allocate_bandwidth(self, u: int, v: int, bw: float) -> bool:
        """
        [桥接方法] 分配带宽
        原因：LowLevelController 调用的是 allocate_bandwidth，
        这里将其转发给现有的 allocate_link_resource 以保持逻辑统一。
        """
        # 调用已有的方法，它会自动处理 pool.allocate 和 link_array 同步
        return self.allocate_link_resource(u, v, bw)

    def release_bandwidth(self, u: int, v: int, bw: float):
        """
        [桥接方法] 释放带宽
        """
        self.release_link_resource(u, v, bw)

    def get_available_bandwidth(self, u: int, v: int) -> float:
        """
        [桥接方法] 获取可用带宽
        直接透传给底层的 pool
        """
        if hasattr(self, 'pool'):
            return self.pool.get_available_bandwidth(u, v)
        return 0.0
    def release_node_resource(self, node_id: int, vnf_type: int,
                              cpu_val: float, mem_val: float):
        """释放节点资源"""
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
        """释放链路资源"""
        self.pool.release_bandwidth(u, v, bw_val)
        self._sync_link_array()

    # ========================================
    # 核心方法：apply_deployment
    # ========================================

    def apply_deployment(self, plan: dict, request: dict) -> bool:
        """
        应用部署方案
        🔥 修复版：移除sfc_backup_system依赖
        """
        hvt_branch = plan.get('hvt')
        if hvt_branch is None:
            return False

        # 🔥 移除sfc_backup_system依赖，直接处理dict格式
        if isinstance(hvt_branch, dict):
            # 如果是dict格式的placement，转换为hvt矩阵
            hvt_matrix = np.zeros((self.n, self.K_vnf), dtype=np.float32)
            for key, info in hvt_branch.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    node_id, vnf_type = key[0], key[1]
                    if 0 <= node_id < self.n and 0 <= vnf_type < self.K_vnf:
                        hvt_matrix[node_id, vnf_type] = 1.0
            hvt_branch = hvt_matrix

        hvt_branch = np.asarray(hvt_branch, dtype=np.float32)
        if hvt_branch.shape != (self.n, self.K_vnf):
            return False

        req_id = request.get('id', -1)
        cpu_reqs = request.get('cpu_origin', [])
        mem_reqs = request.get('memory_origin', [])

        # 检查所有资源
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            if self.pool.get_available_cpu(node) < cpu_need - 1e-5:
                return False
            if self.pool.get_available_memory(node) < mem_need - 1e-5:
                return False

        # 分配资源
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            success = self.allocate_node_resource(node, vnf_t, cpu_need, mem_need)
            if not success:
                return False

            self.vnf_instances.append({
                'req_id': req_id,
                'node': node,
                'vnf_type': vnf_t,
                'cpu': cpu_need,
                'memory': mem_need
            })

        return True

    def apply_tree_deployment(self, plan: dict, request: dict) -> bool:
        """应用树部署方案"""
        if not self.apply_deployment(plan, request):
            return False

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
                    if not self.allocate_link_resource(u, v, bw_need * flow):
                        return False

        return True

    # ========================================
    # 查询接口
    # ========================================

    def check_node_resource(self, node_id: int, vnf_type: int = 0,
                            cpu_need: float = 0.0, mem_need: float = 0.0) -> bool:
        """检查节点资源是否足够"""
        if node_id < 0 or node_id >= self.n:
            return False

        cpu_ok = self.pool.get_available_cpu(node_id) >= cpu_need - 1e-5
        mem_ok = self.pool.get_available_memory(node_id) >= mem_need - 1e-5

        return cpu_ok and mem_ok

    def check_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        """检查链路资源是否足够"""
        return self.pool.get_available_bandwidth(u, v) >= bw_need - 1e-5

    def get_available_resources(self, node_id: Optional[int] = None) -> Dict:
        """获取可用资源"""
        if node_id is not None:
            return {
                'cpu': self.pool.get_available_cpu(node_id),
                'memory': self.pool.get_available_memory(node_id)
            }
        else:
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
    # GNN相关接口
    # ========================================

    def get_network_state_dict(self, current_request=None):
        """获取网络状态字典"""
        C = np.zeros(self.n)
        M = np.zeros(self.n)

        for i in range(self.n):
            C[i] = self.pool.get_available_cpu(i)
            M[i] = self.pool.get_available_memory(i)

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
        import torch

        if not current_request:
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            req_vec = torch.zeros(self.request_dim)
            return x, self.edge_index, edge_attr, req_vec

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
                0.5,
                0.5,
                0.0
            ]

            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float32)
        edge_attrs = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim), dtype=torch.float32)
        req_vec = torch.randn(self.request_dim)

        return x, torch.tensor(self.edge_index, dtype=torch.long), edge_attrs, req_vec

    # ========================================
    # 其他兼容接口
    # ========================================

    def reset(self, hard: bool = False):
        """重置资源管理器"""
        self.pool.reset(hard)
        self.hvt_all.fill(0)
        self.vnf_instances = []
        self.link_ref_count.fill(0)
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
        return 1.0

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

    # ========================================
    # 集成的高级资源管理方法
    # ========================================

    def _get_total_vnf_progress(self):
        """获取当前VNF进度"""
        return self.next_vnf_idx

    def _check_node_resources(self, node_id: int, vnf_idx: int = None) -> bool:
        """检查资源（含虚拟预扣）"""
        try:
            if self.current_request is None:
                return True

            vnf_list = self.current_request.get('vnf', [])

            if vnf_idx is None:
                vnf_idx = self._get_total_vnf_progress()

            if vnf_idx >= len(vnf_list):
                return True

            cpu_reqs = self.current_request.get('cpu_origin', []) or \
                       self.current_request.get('vnf_cpu', [])
            mem_reqs = self.current_request.get('memory_origin', []) or \
                       self.current_request.get('mem_origin', [])

            req_cpu = float(cpu_reqs[vnf_idx]) if vnf_idx < len(cpu_reqs) else 1.0
            req_mem = float(mem_reqs[vnf_idx]) if vnf_idx < len(mem_reqs) else 1.0

            avail_cpu = self.pool.get_available_cpu(node_id)
            avail_mem = self.pool.get_available_memory(node_id)

            logger.debug(f"🔍 [资源检查] 节点{node_id}, VNF[{vnf_idx}]")
            logger.debug(f"   可用资源: CPU={avail_cpu:.1f}, Mem={avail_mem:.1f}")
            logger.debug(f"   VNF需求: CPU={req_cpu:.1f}, Mem={req_mem:.1f}")

            cpu_ok = avail_cpu >= (req_cpu * 1.05)
            mem_ok = avail_mem >= (req_mem * 1.05)

            return cpu_ok and mem_ok

        except Exception as e:
            logger.error(f"❌ [资源检查] 节点{node_id}检查失败: {e}")
            traceback.print_exc()
            return False

    def _try_deploy(self, node):
        """尝试部署VNF"""
        if self.current_request is None:
            logger.error("❌ [部署] 没有当前请求")
            return False

        if self.current_phase != 'vnf_deployment':
            return True

        vnf_list = self.current_request.get('vnf', [])
        if len(vnf_list) == 0:
            logger.info("✅ [部署] 没有VNF需要部署")
            return True

        if hasattr(self, 'next_vnf_idx'):
            next_vnf_idx = self.next_vnf_idx
        else:
            next_vnf_idx = self._get_total_vnf_progress()

        if next_vnf_idx >= len(vnf_list):
            return True

        next_vnf_type = vnf_list[next_vnf_idx]

        logger.info(f"\n🔍 [部署检查] 节点{node}, VNF[{next_vnf_idx}]类型{next_vnf_type}")

        cpu_reqs = self.current_request.get('cpu_origin', []) or self.current_request.get('vnf_cpu', [])
        mem_reqs = self.current_request.get('memory_origin', []) or self.current_request.get('vnf_mem', [])

        required_cpu = cpu_reqs[next_vnf_idx] if next_vnf_idx < len(cpu_reqs) else 10.0
        required_mem = mem_reqs[next_vnf_idx] if next_vnf_idx < len(mem_reqs) else 10.0

        logger.info(f"   VNF需求: CPU={required_cpu}, Mem={required_mem}")

        avail_cpu = self.pool.get_available_cpu(node)
        avail_mem = self.pool.get_available_memory(node)

        logger.info(f"   节点资源: CPU={avail_cpu}, Mem={avail_mem}")

        if avail_cpu < required_cpu or avail_mem < required_mem:
            logger.error(f"❌ [部署失败] 节点{node}资源不足")
            return False

        if hasattr(self, 'dc_nodes') and node not in self.dc_nodes:
            logger.error(f"❌ [部署失败] 节点{node}不是DC节点")
            return False

        logger.info(f"✅ [部署成功] 节点{node}部署VNF[{next_vnf_idx}]类型{next_vnf_type}")

        if hasattr(self, 'allocate_node_resource'):
            success = self.allocate_node_resource(node, next_vnf_type, required_cpu, required_mem)
            if not success:
                logger.error(f"❌ [部署失败] 资源分配失败")
                return False

            placement_key = (node, next_vnf_idx)

            if 'placement' not in self.current_tree:
                self.current_tree['placement'] = {}

            self.current_tree['placement'][placement_key] = {
                'node': node,
                'vnf_type': next_vnf_type,
                'cpu_used': required_cpu,
                'mem_used': required_mem
            }

        return True


    def _get_path_vnf_progress(self, current_node):
        """获取当前路径上已部署的VNF数量"""
        if not self.current_request:
            return 0

        tree_edges = self.current_tree.get('tree', {})
        placement = self.current_tree.get('placement', {})
        source = self.current_request.get('source')
        vnf_list = self.current_request.get('vnf', [])

        if current_node == source:
            progress = 0
            for i in range(len(vnf_list)):
                if (source, i) in placement:
                    progress += 1
                else:
                    break
            return progress

        from collections import deque, defaultdict
        adj = defaultdict(list)
        for u, v in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        parent_map = {source: None}
        queue = deque([source])
        visited = {source}

        path_found = False
        while queue:
            curr = queue.popleft()
            if curr == current_node:
                path_found = True
                break
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    parent_map[nbr] = curr
                    queue.append(nbr)

        if not path_found:
            return 0

        path_nodes = set()
        curr = current_node
        while curr is not None:
            path_nodes.add(curr)
            curr = parent_map.get(curr)

        current_progress = 0
        for i in range(len(vnf_list)):
            found_this_vnf = False
            for node in path_nodes:
                if (node, i) in placement:
                    found_this_vnf = True
                    break

            if found_this_vnf:
                current_progress += 1
            else:
                break

        return current_progress

    def _archive_request(self, success=False, already_rolled_back=False):
        """
        🔥 [V16.5 分离版] 仅负责归档，不再重置状态
        状态重置工作移交至 complete_current_request

        修正内容：
        ✅ 移除了内部对 request_manager 的调用，避免与 LowLevelController 冲突
        ✅ 修复了 "missing 1 required positional argument" 报错
        """
        if self.current_request is None:
            return

        req = self.current_request
        req_id = req.get('id', id(req))

        # --- 可视化逻辑 (保持不变) ---
        if hasattr(self, 'enable_visualization') and self.enable_visualization and hasattr(self, 'visualizer'):
            try:
                subdir = 'success' if success else 'fail'
                save_path = f'visualization/{subdir}/request_{req_id}.png'
                self.visualizer.visualize_request_tree(
                    request=self.current_request,
                    save_path=save_path,
                    show=False
                )
            except Exception as e:
                pass

        if success:
            # =====================================================================
            # 成功分支：保存账本 (仅保存数据，不触发托管)
            # =====================================================================
            # 1. 将资源分配快照保存到 request 对象中
            req['resources_allocated'] = {
                'placement': copy.deepcopy(self.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.current_tree.get('tree', {}))
            }

            # 🔥🔥🔥【关键修改：移除生命周期托管调用】🔥🔥🔥
            # 原因：LowLevelController 已经在 Step 2 中正确调用了 register_request。
            # 这里如果保留，不仅参数错误(缺少resources_allocated)，还会导致重复注册。
            # -------------------------------------------------------------
            # if hasattr(self, 'request_manager') and self.request_manager:
            #     try:
            #         # ❌ 这里的代码引发了 "missing argument" 错误，现已屏蔽
            #         # self.request_manager.register_request(req)
            #         pass
            #     except Exception as e:
            #         print(f"⚠️ [生命周期] 请求 {req_id} 添加失败: {e}")
            # -------------------------------------------------------------

            self.total_requests_accepted += 1
            if hasattr(self, 'served_dest_count'):
                self.served_dest_count += len(req.get('dest', []))

            # 简化日志，表明只做了归档
            print(f"✅ [归档成功] 请求 {req_id} 账本已保存 (等待控制器托管)")

        else:
            # =====================================================================
            # 失败分支：回滚虚拟资源
            # =====================================================================
            if already_rolled_back:
                print(f"ℹ️ [归档失败] 请求 {req_id} 失败（资源已回滚）")
            else:
                print(f"❌ [归档失败] 请求 {req_id} 失败，开始回滚虚拟资源...")

                # 调用工具回滚
                if hasattr(self, 'tools') and hasattr(self.tools, 'rollback_request_resources'):
                    self.tools.rollback_request_resources(req)

                # 额外回滚当前树占用的虚拟资源
                placement = self.current_tree.get('placement', {})
                tree_edges = self.current_tree.get('tree', {})

                restored_cpu = 0.0
                restored_bw = 0.0

                # 回滚节点
                for key, info in placement.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        node = key[0]
                        vnf_type = key[1]
                        if isinstance(info, dict):
                            c = info.get('cpu_used', 1.0)
                            m = info.get('mem_used', 1.0)
                        else:
                            c, m = 1.0, 1.0

                        if hasattr(self, 'release_node_resource'):
                            try:
                                self.release_node_resource(node, vnf_type, c, m)
                                restored_cpu += c
                            except Exception:
                                pass

                # 回滚链路
                bw = req.get('bw_origin', 1.0)
                for edge_key in tree_edges.keys():
                    u, v = edge_key
                    if hasattr(self, 'release_link_resource'):
                        try:
                            self.release_link_resource(u, v, bw)
                            restored_bw += bw
                        except Exception:
                            pass

                if restored_cpu > 0 or restored_bw > 0:
                    print(f"♻️ [虚拟资源回滚] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

    def complete_current_request(self):
        """
        🔥 [新增] 完成并清理当前请求
        专门用于 step_low_level 结束后的状态重置
        """
        if self.current_request is None:
            # 如果已经被清理过，只是报个警告，不影响流程
            logger.warning("⚠️ [Complete] 没有当前请求 (可能已被清理)")
            return False

        req_id = self.current_request.get('id', 'unknown')
        # logger.info(f"🏁 [Complete] 清理请求 {req_id} 上下文")

        # 重置所有临时状态
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.current_request = None
        self.current_branch_id = None
        self.nodes_on_tree = set()

        # 重置VNF指针 (重要)
        if hasattr(self, 'next_vnf_idx'):
            self.next_vnf_idx = 0

        return True

    def check_and_release_expired(self, current_time):
        """
        检查并释放过期的请求 - 带详细日志版本
        """
        expired_req_ids = []

        # 🔥 记录释放前的资源状态
        res_before = self.tools.get_resource_utilization() if hasattr(self, 'env') else None

        # 遍历所有活跃请求
        for req_id, req_info in list(self.active_requests.items()):
            if current_time > req_info['expire_time']:
                expired_req_ids.append(req_id)
                logger.info(f"⏰ [Resource Flow] 请求 {req_id}: 请求生命周期到达，归还资源 (Expired & Released)")
                # 🔥 详细日志
                expire_time = req_info['expire_time']
                arrival_time = req_info['arrival_time']
                print(f"   ⏱️ 请求 {req_id} 已过期: "
                      f"到达={arrival_time:.2f}s, "
                      f"过期={expire_time:.2f}s, "
                      f"当前={current_time:.2f}s")

        # 释放过期请求的资源
        for req_id in expired_req_ids:
            self._release_request_resources(req_id, current_time)

        # 🔥 释放后的资源状态
        if expired_req_ids:
            res_after = self.tools.get_resource_utilization() if hasattr(self, 'env') else None
            print(f"♻️ [过期释放] 释放了 {len(expired_req_ids)} 个请求")
            if res_before is not None and res_after is not None:
                change = res_after - res_before
                print(f"   资源变化: {res_before:.1f}% → {res_after:.1f}% "
                      f"({'+' if change > 0 else ''}{change:.1f}%)")
            print(f"   请求ID: {expired_req_ids}")

        return expired_req_ids
