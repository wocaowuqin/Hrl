import torch
"""
envs/modules/fused_resource_manager.py
==================================================
融合版资源管理器 - 重构版 (职责分离)
==================================================
- SharedResourcePool: 物理资源原子操作
- TransactionManager: 事务预留/提交/回滚
- RequestLifecycleManager: 请求生命周期管理
- RequestHandler: 业务辅助方法（部署、归档、清理）
- FusedResourceManager: 外观类，组合上述组件，提供统一接口
"""

import numpy as np
import time
import logging
import threading
import copy
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== 资源类型枚举 ====================
class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"


# ==================== 资源分配记录 ====================
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


# ==================== 共享资源池 ====================
class SharedResourcePool:
    """共享资源池 - 底层物理资源管理（原子操作）"""

    def __init__(self, topology: np.ndarray, capacities: Dict):
        self.n = topology.shape[0]
        self.topology = topology

        # 物理容量
        self.cpu_cap = np.full(self.n, capacities.get('cpu', 100.0), dtype=float)
        self.mem_cap = np.full(self.n, capacities.get('memory', 80.0), dtype=float)

        # 当前可用资源
        self.cpu_avail = self.cpu_cap.copy()
        self.mem_avail = self.mem_cap.copy()

        # 预留标记
        self.cpu_reserved = np.zeros(self.n, dtype=float)
        self.mem_reserved = np.zeros(self.n, dtype=float)

        # 链路相关
        self.link_map = {}          # (u,v) -> edge_id
        self.bw_cap = {}            # (u,v) -> 带宽容量
        self.bw_avail = {}          # (u,v) -> 当前可用带宽
        self.bw_reserved = {}       # (u,v) -> 预留带宽
        self.link_locks = {}        # (u,v) -> 锁

        # 节点锁
        self.node_locks = [threading.RLock() for _ in range(self.n)]

        self._init_links(capacities.get('bandwidth', 100.0))
        logger.debug(f"[SharedPool] 初始化: {self.n} 节点, {len(self.link_map)} 链路")

    def _init_links(self, bw_cap: float):
        edge_id = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.topology[i, j] > 0:
                    key = tuple(sorted((i, j)))
                    if key not in self.link_map:
                        self.link_map[key] = edge_id
                        self.bw_cap[key] = bw_cap
                        self.bw_avail[key] = bw_cap
                        self.bw_reserved[key] = 0.0
                        self.link_locks[key] = threading.RLock()
                        edge_id += 1
        self.L = len(self.link_map)

    # ---------- 节点资源操作 ----------
    def allocate_cpu(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.cpu_avail[node] >= amount - 1e-5:
                self.cpu_avail[node] -= amount
                return True
            return False

    def allocate_memory(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.mem_avail[node] >= amount - 1e-5:
                self.mem_avail[node] -= amount
                return True
            return False

    def release_cpu(self, node: int, amount: float):
        with self.node_locks[node]:
            self.cpu_avail[node] = min(self.cpu_cap[node], self.cpu_avail[node] + amount)

    def release_memory(self, node: int, amount: float):
        with self.node_locks[node]:
            self.mem_avail[node] = min(self.mem_cap[node], self.mem_avail[node] + amount)

    def reserve_cpu(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.cpu_avail[node] >= amount - 1e-5:
                self.cpu_avail[node] -= amount
                self.cpu_reserved[node] += amount
                return True
            return False

    def reserve_memory(self, node: int, amount: float) -> bool:
        with self.node_locks[node]:
            if self.mem_avail[node] >= amount - 1e-5:
                self.mem_avail[node] -= amount
                self.mem_reserved[node] += amount
                return True
            return False

    def commit_reservation(self, node: int, cpu_amount: float, mem_amount: float):
        with self.node_locks[node]:
            self.cpu_reserved[node] = max(0, self.cpu_reserved[node] - cpu_amount)
            self.mem_reserved[node] = max(0, self.mem_reserved[node] - mem_amount)

    def cancel_reservation(self, node: int, cpu_amount: float, mem_amount: float):
        with self.node_locks[node]:
            if cpu_amount > 0:
                self.cpu_avail[node] = min(self.cpu_cap[node], self.cpu_avail[node] + cpu_amount)
                self.cpu_reserved[node] = max(0, self.cpu_reserved[node] - cpu_amount)
            if mem_amount > 0:
                self.mem_avail[node] = min(self.mem_cap[node], self.mem_avail[node] + mem_amount)
                self.mem_reserved[node] = max(0, self.mem_reserved[node] - mem_amount)

    # ---------- 链路资源操作 ----------
    def allocate_bandwidth(self, u: int, v: int, amount: float) -> bool:
        key = tuple(sorted((u, v)))
        if key not in self.bw_avail:
            return False
        with self.link_locks[key]:
            if self.bw_avail[key] >= amount - 1e-5:
                self.bw_avail[key] -= amount
                return True
            return False

    def release_bandwidth(self, u: int, v: int, amount: float):
        key = tuple(sorted((u, v)))
        if key not in self.bw_avail:
            return
        with self.link_locks[key]:
            self.bw_avail[key] = min(self.bw_cap[key], self.bw_avail[key] + amount)

    def reserve_bandwidth(self, u: int, v: int, amount: float) -> bool:
        key = tuple(sorted((u, v)))
        if key not in self.bw_avail:
            return False
        with self.link_locks[key]:
            if self.bw_avail[key] >= amount - 1e-5:
                self.bw_avail[key] -= amount
                self.bw_reserved[key] += amount
                return True
            return False

    def commit_link_reservation(self, u: int, v: int, amount: float):
        key = tuple(sorted((u, v)))
        if key not in self.bw_reserved:
            return
        with self.link_locks[key]:
            self.bw_reserved[key] = max(0, self.bw_reserved[key] - amount)

    def cancel_link_reservation(self, u: int, v: int, amount: float):
        key = tuple(sorted((u, v)))
        if key not in self.bw_avail:
            return
        with self.link_locks[key]:
            self.bw_avail[key] = min(self.bw_cap[key], self.bw_avail[key] + amount)
            self.bw_reserved[key] = max(0, self.bw_reserved[key] - amount)

    # ---------- 查询接口 ----------
    def get_available_cpu(self, node: int) -> float:
        with self.node_locks[node]:
            return max(0, self.cpu_avail[node])

    def get_available_memory(self, node: int) -> float:
        with self.node_locks[node]:
            return max(0, self.mem_avail[node])

    def get_available_bandwidth(self, u: int, v: int) -> float:
        key = tuple(sorted((u, v)))
        if key not in self.bw_avail:
            return 0.0
        with self.link_locks[key]:
            return max(0, self.bw_avail[key])

    def get_edge_id(self, u: int, v: int) -> Optional[int]:
        return self.link_map.get(tuple(sorted((u, v))))

    def get_edge_key(self, edge_id: int) -> Optional[tuple]:
        for key, eid in self.link_map.items():
            if eid == edge_id:
                return key
        return None

    def reset(self, hard: bool = False):
        if hard:
            self.cpu_avail = self.cpu_cap.copy()
            self.mem_avail = self.mem_cap.copy()
            for key in self.bw_avail:
                self.bw_avail[key] = self.bw_cap[key]
        self.cpu_reserved.fill(0)
        self.mem_reserved.fill(0)
        for key in self.bw_reserved:
            self.bw_reserved[key] = 0.0


# ==================== 事务管理器 ====================
class TransactionManager:
    """事务管理器 - 预留/提交/回滚"""

    def __init__(self, pool: SharedResourcePool):
        self.pool = pool
        self.transactions: Dict[str, List[ResourceAllocation]] = {}
        self.active_transactions: Set[str] = set()
        self.lock = threading.RLock()

    def begin_transaction(self, tx_id: str):
        with self.lock:
            if tx_id not in self.transactions:
                self.transactions[tx_id] = []
                self.active_transactions.add(tx_id)

    def reserve_node(self, tx_id: str, node: int, vnf_type: int,
                     cpu_need: float, mem_need: float) -> bool:
        with self.lock:
            if tx_id not in self.active_transactions:
                return False
            if cpu_need > 0 and not self.pool.reserve_cpu(node, cpu_need):
                return False
            if mem_need > 0 and not self.pool.reserve_memory(node, mem_need):
                if cpu_need > 0:
                    self.pool.cancel_reservation(node, cpu_need, 0)
                return False
            if cpu_need > 0:
                self.transactions[tx_id].append(ResourceAllocation(
                    transaction_id=tx_id,
                    resource_type=ResourceType.CPU,
                    resource_id=node,
                    amount=cpu_need,
                    vnf_type=vnf_type,
                    reserved=True
                ))
            if mem_need > 0:
                self.transactions[tx_id].append(ResourceAllocation(
                    transaction_id=tx_id,
                    resource_type=ResourceType.MEMORY,
                    resource_id=node,
                    amount=mem_need,
                    vnf_type=vnf_type,
                    reserved=True
                ))
            return True

    def reserve_link(self, tx_id: str, u: int, v: int, bw_need: float) -> bool:
        with self.lock:
            if tx_id not in self.active_transactions:
                return False
            if not self.pool.reserve_bandwidth(u, v, bw_need):
                return False
            self.transactions[tx_id].append(ResourceAllocation(
                transaction_id=tx_id,
                resource_type=ResourceType.BANDWIDTH,
                resource_id=(u, v),
                amount=bw_need,
                reserved=True
            ))
            return True

    def commit(self, tx_id: str) -> bool:
        with self.lock:
            if tx_id not in self.active_transactions:
                return False
            for alloc in self.transactions[tx_id]:
                if not alloc.reserved:
                    continue
                if alloc.resource_type == ResourceType.CPU:
                    self.pool.commit_reservation(alloc.resource_id, alloc.amount, 0)
                elif alloc.resource_type == ResourceType.MEMORY:
                    self.pool.commit_reservation(alloc.resource_id, 0, alloc.amount)
                elif alloc.resource_type == ResourceType.BANDWIDTH:
                    u, v = alloc.resource_id
                    self.pool.commit_link_reservation(u, v, alloc.amount)
                alloc.committed = True
                alloc.reserved = False
            self.active_transactions.remove(tx_id)
            logger.debug(f"[Transaction] 提交事务 {tx_id}")
            return True

    def rollback(self, tx_id: str):
        with self.lock:
            if tx_id not in self.transactions:
                return
            for alloc in self.transactions[tx_id]:
                if alloc.reserved:
                    if alloc.resource_type == ResourceType.CPU:
                        self.pool.cancel_reservation(alloc.resource_id, alloc.amount, 0)
                    elif alloc.resource_type == ResourceType.MEMORY:
                        self.pool.cancel_reservation(alloc.resource_id, 0, alloc.amount)
                    elif alloc.resource_type == ResourceType.BANDWIDTH:
                        u, v = alloc.resource_id
                        self.pool.cancel_link_reservation(u, v, alloc.amount)
                elif alloc.committed:
                    # 安全释放
                    if alloc.resource_type == ResourceType.CPU:
                        self.pool.release_cpu(alloc.resource_id, alloc.amount)
                    elif alloc.resource_type == ResourceType.MEMORY:
                        self.pool.release_memory(alloc.resource_id, alloc.amount)
                    elif alloc.resource_type == ResourceType.BANDWIDTH:
                        u, v = alloc.resource_id
                        self.pool.release_bandwidth(u, v, alloc.amount)
            if tx_id in self.active_transactions:
                self.active_transactions.remove(tx_id)
            del self.transactions[tx_id]
            logger.debug(f"[Transaction] 回滚事务 {tx_id}")


# ==================== 请求生命周期管理器 ====================
class RequestLifecycleManager:
    """
    请求生命周期管理器 - 纯仿真时间版
    - 完全依赖外部传入的 current_time（仿真时间）
    - 移除了 cleanup_interval 节流，每个时间步都会检查
    - register_request 必须传入 arrival_time 和 lifetime
    - 所有释放操作都基于记录的资源量
    """

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.active_requests: Dict[str, dict] = {}
        self.expired_requests: Dict[str, dict] = {}
        self.lock = threading.RLock()
        self.stats = {
            'total_registered': 0,
            'total_expired': 0,
            'total_failed': 0,
            'total_cpu_released': 0.0,
            'total_mem_released': 0.0,
            'total_bw_released': 0.0,
        }

    def register_request(self, request: dict, resources_allocated: dict) -> bool:
        """
        注册请求及其占用的资源
        要求 request 必须包含 'id', 'arrival_time', 'lifetime'
        """
        req_id = request.get('id')
        if not req_id:
            logger.error("[Lifecycle] 请求缺少ID")
            return False

        arrival = request.get('arrival_time')
        lifetime = request.get('lifetime')
        if arrival is None or lifetime is None:
            logger.error("[Lifecycle] 请求缺少 arrival_time 或 lifetime")
            return False

        expire_time = arrival + lifetime

        with self.lock:
            if req_id in self.active_requests:
                logger.warning(f"[Lifecycle] 请求 {req_id} 已存在，跳过")
                return False

            self.active_requests[req_id] = {
                'request': copy.deepcopy(request),
                'resources': copy.deepcopy(resources_allocated),
                'arrival_time': arrival,
                'lifetime': lifetime,
                'expire_time': expire_time,
                'status': 'active'
            }
            self.stats['total_registered'] += 1
            logger.debug(f"[Lifecycle] 注册请求 {req_id}, 过期时间 {expire_time:.2f}")
            return True

    def check_and_release_expired(self, current_time: float) -> List[str]:
        """
        检查并释放所有过期的请求
        current_time: 当前仿真时间
        返回已释放的请求ID列表
        """
        expired = []
        with self.lock:
            if self.active_requests:
                expires = {rid: info['expire_time']
                           for rid, info in self.active_requests.items()}
                logger.debug(f"[Lifecycle] t={current_time:.3f} | 活跃请求: "
                             f"{', '.join(f'{r}→exp{e:.2f}' for r,e in expires.items())}")
            for req_id, info in list(self.active_requests.items()):
                if current_time > info['expire_time']:
                    expired.append(req_id)
            for req_id in expired:
                self._release_request(req_id, current_time)
        return expired

    def _release_request(self, req_id: str, current_time: float):
        """释放单个请求的资源（内部方法）"""
        if req_id not in self.active_requests:
            return
        info = self.active_requests[req_id]
        resources = info['resources']
        cpu_rel = mem_rel = bw_rel = 0.0

        # 释放节点资源
        placement = resources.get('placement', {})
        for key, alloc in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node = key[0]
                vnf_idx = key[1]
                cpu_used = alloc.get('cpu_used', 1.0)
                mem_used = alloc.get('mem_used', 1.0)
                # ✅ 从 info['request'] 获取请求数据
                vnf_list = info['request'].get('vnf', [])
                vnf_type = vnf_list[vnf_idx] if vnf_idx < len(vnf_list) else vnf_idx
                self.resource_manager.release_node_resource(node, vnf_type, cpu_used, mem_used)

        # 释放链路资源
        tree = resources.get('tree', {})
        bw_needed = info['request'].get('bw_origin', 1.0)
        for edge_key, flow in tree.items():
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                if flow == 0.0:
                    continue  # flow=0表示仅记录路径，未实际分配BW，跳过释放
                bw_used = bw_needed * flow  # flow=1.0时释放实际BW
                try:
                    self.resource_manager.release_link_resource(u, v, bw_used)
                    bw_rel += bw_used
                except Exception as e:
                    logger.error(f"[Lifecycle] 释放链路资源失败 {req_id}: {e}")

        # 更新统计
        self.stats['total_expired'] += 1
        self.stats['total_cpu_released'] += cpu_rel
        self.stats['total_mem_released'] += mem_rel
        self.stats['total_bw_released'] += bw_rel

        # 移到过期记录
        info['status'] = 'expired'
        info['release_time'] = current_time
        info['actual_lifetime'] = current_time - info['arrival_time']
        self.expired_requests[req_id] = info
        del self.active_requests[req_id]
        logger.debug(f"[Lifecycle] 释放请求 {req_id}: CPU={cpu_rel:.1f}, MEM={mem_rel:.1f}, BW={bw_rel:.1f}")

    def force_release(self, req_id: str, current_time: float) -> bool:
        """
        强制释放请求（用于请求失败等场景）
        current_time: 当前仿真时间
        """
        with self.lock:
            if req_id not in self.active_requests:
                return False
            self._release_request(req_id, current_time)
            return True

    def cleanup_all(self, current_time: float):
        """
        清理所有活跃请求（用于环境重置）
        current_time: 当前仿真时间（通常为重置时刻）
        """
        with self.lock:
            req_ids = list(self.active_requests.keys())
            for req_id in req_ids:
                self._release_request(req_id, current_time)
            logger.debug(f"[Lifecycle] 清理所有请求，共 {len(req_ids)} 个")

    def get_stats(self) -> dict:
        with self.lock:
            stats = copy.deepcopy(self.stats)
            stats['active_count'] = len(self.active_requests)
            return stats


# ==================== 请求处理器（辅助方法） ====================
class RequestHandler:
    """
    请求处理器 - 负责与请求相关的业务逻辑
    包括部署、归档、状态查询等辅助方法
    """

    def __init__(self, resource_mgr):
        self.rm = resource_mgr  # FusedResourceManager 实例

    def _try_deploy(self, node: int) -> bool:
        """尝试部署当前VNF（兼容旧代码）"""
        if self.rm.current_request is None:
            return False
        if self.rm.current_phase != 'vnf_deployment':
            return True

        vnf_list = self.rm.current_request.get('vnf', [])
        if not vnf_list:
            return True

        idx = self.rm.next_vnf_idx
        if idx >= len(vnf_list):
            return True

        vnf_type = vnf_list[idx]
        cpu_reqs = self.rm.current_request.get('cpu_origin', []) or self.rm.current_request.get('vnf_cpu', [])
        mem_reqs = self.rm.current_request.get('memory_origin', []) or self.rm.current_request.get('vnf_mem', [])
        cpu_need = cpu_reqs[idx] if idx < len(cpu_reqs) else 10.0
        mem_need = mem_reqs[idx] if idx < len(mem_reqs) else 10.0

        if node not in self.rm.dc_nodes:
            logger.warning(f"[_try_deploy] 节点{node}不是DC节点")
            return False

        if not self.rm.check_node_resource(node, vnf_type, cpu_need, mem_need):
            logger.warning(f"[_try_deploy] 节点{node}资源不足")
            return False

        if not self.rm.allocate_node_resource(node, vnf_type, cpu_need, mem_need):
            return False

        if 'placement' not in self.rm.current_tree:
            self.rm.current_tree['placement'] = {}
        self.rm.current_tree['placement'][(node, idx)] = {
            'node': node,
            'vnf_type': vnf_type,
            'cpu_used': cpu_need,
            'mem_used': mem_need
        }
        logger.debug(f"[_try_deploy] 节点{node}部署VNF[{idx}]成功")
        return True

    def _archive_request(self, success=False, already_rolled_back=False):
        """归档请求（兼容旧代码）"""
        if self.rm.current_request is None:
            return
        req_id = self.rm.current_request.get('id', id(self.rm.current_request))

        if success:
            self.rm.current_request['resources_allocated'] = {
                'placement': copy.deepcopy(self.rm.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.rm.current_tree.get('tree', {}))
            }
            self.rm.total_requests_accepted += 1
            if hasattr(self.rm, 'served_dest_count'):
                self.rm.served_dest_count += len(self.rm.current_request.get('dest', []))
            logger.debug(f"[Archive] 请求 {req_id} 成功归档")
        else:
            if not already_rolled_back:
                logger.debug(f"[Archive] 请求 {req_id} 失败，回滚资源")
                placement = self.rm.current_tree.get('placement', {})
                tree = self.rm.current_tree.get('tree', {})
                bw = self.rm.current_request.get('bw_origin', 1.0)

                for key, alloc in placement.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        node, vnf_t = key[0], key[1]
                        cpu = alloc.get('cpu_used', 1.0)
                        mem = alloc.get('mem_used', 1.0)
                        self.rm.release_node_resource(node, vnf_t, cpu, mem)
                for edge_key, flow in tree.items():
                    if isinstance(edge_key, tuple) and len(edge_key) == 2:
                        u, v = edge_key
                        if flow == 0.0:
                            continue  # 未实际分配，跳过
                        self.rm.release_link_resource(u, v, bw * flow)
            logger.debug(f"[Archive] 请求 {req_id} 失败归档")

    def complete_current_request(self):
        """完成并清理当前请求（由外部调用）"""
        if self.rm.current_request is None:
            return False
        req_id = self.rm.current_request.get('id', 'unknown')
        self.rm.current_tree = {
            'hvt': np.zeros((self.rm.n, self.rm.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.rm.current_request = None
        self.rm.current_branch_id = None
        self.rm.nodes_on_tree = set()
        self.rm.next_vnf_idx = 0
        logger.debug(f"[Complete] 清理请求 {req_id}")
        return True

    def apply_deployment(self, plan: dict, request: dict) -> bool:
        """应用部署方案（兼容旧代码）"""
        hvt_branch = plan.get('hvt')
        if hvt_branch is None:
            return False

        if isinstance(hvt_branch, dict):
            hvt_matrix = np.zeros((self.rm.n, self.rm.K_vnf), dtype=np.float32)
            for key, info in hvt_branch.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    node, vnf_t = key[0], key[1]
                    if 0 <= node < self.rm.n and 0 <= vnf_t < self.rm.K_vnf:
                        hvt_matrix[node, vnf_t] = 1.0
            hvt_branch = hvt_matrix

        hvt_branch = np.asarray(hvt_branch, dtype=np.float32)
        if hvt_branch.shape != (self.rm.n, self.rm.K_vnf):
            return False

        cpu_reqs = request.get('cpu_origin', [])
        mem_reqs = request.get('memory_origin', [])

        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)
            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0
            if not self.rm.check_node_resource(node, vnf_t, cpu_need, mem_need):
                return False

        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)
            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0
            if not self.rm.allocate_node_resource(node, vnf_t, cpu_need, mem_need):
                return False
            self.rm.vnf_instances.append({
                'req_id': request.get('id', -1),
                'node': node,
                'vnf_type': vnf_t,
                'cpu': cpu_need,
                'memory': mem_need
            })
        return True

    def apply_tree_deployment(self, plan: dict, request: dict) -> bool:
        if not self.apply_deployment(plan, request):
            return False
        tree = plan.get('tree', {})
        bw_need = request.get('bw_origin', 0)
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
                if not self.rm.allocate_link_resource(u, v, bw_need * flow):
                    return False
        return True

    def get_network_state_dict(self, current_request=None):
        C = np.array([self.rm.pool.get_available_cpu(i) for i in range(self.rm.n)])
        M = np.array([self.rm.pool.get_available_memory(i) for i in range(self.rm.n)])
        B = np.zeros(self.rm.pool.L, dtype=float)
        for edge_key, edge_id in self.rm.pool.link_map.items():
            if edge_id < self.rm.pool.L:
                B[edge_id] = self.rm.pool.get_available_bandwidth(*edge_key)
        state = {
            'bw': B,
            'cpu': C,
            'mem': M,
            'hvt': self.rm.hvt_all,
            'bw_ref_count': np.zeros(self.rm.pool.L, dtype=int)
        }
        if current_request:
            state['request'] = current_request
        return state


# ==================== 融合版资源管理器（外观类） ====================
class FusedResourceManager:
    """
    融合版资源管理器 - 外观类
    组合核心组件，提供统一接口，保持与原有代码兼容
    """

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        self.topo = topo
        self.n = topo.shape[0]
        self.dc_nodes = dc_nodes
        self.link_map = link_map
        self.tools = {}  # 保留兼容性

        # 核心组件
        self.pool = SharedResourcePool(topo, capacities)
        self.transaction_mgr = TransactionManager(self.pool)
        self.request_manager = RequestLifecycleManager(self)
        self.handler = RequestHandler(self)  # 业务辅助方法

        # 兼容性字段
        self.C_cap = capacities.get('cpu', 100.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.B_cap = capacities.get('bandwidth', 100.0)

        # VNF相关
        self.K_vnf = 8
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.vnf_instances = []

        # 构建边索引 + 边特征
        self._build_edge_index()
        self._build_edge_attr()

        # 状态维度（兼容GNN）
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.pool.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request
        self.node_feat_dim = 6 + self.K_vnf + 3 + 3  # [SDG-HRL] +vnf_depth, +progress, +phase_flag → 20
        self.edge_feat_dim = 5
        self.request_dim = 24

        # 当前请求相关（由外部设置）
        self.current_request = None
        self.current_tree = {}
        self.current_phase = 'idle'
        self.next_vnf_idx = 0
        self.nodes_on_tree = set()
        self.total_requests_accepted = 0
        self.served_dest_count = 0

        logger.debug(f"[FusedRM] 初始化完成: {self.n}节点, {self.pool.L}链路 (外观版)")

    def _build_edge_index(self):
        rows, cols = np.where(self.topo > 0)
        self.edge_index = np.array([rows, cols], dtype=np.int64)
        self.edge_hops = np.array([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=np.float32)
        self.edge_to_phys = {}
        self.phys_to_graph_edges = {}
        for idx, (u, v) in enumerate(zip(rows, cols)):
            eid = self.pool.get_edge_id(u, v)
            if eid is not None:
                self.edge_to_phys[(u, v)] = eid
                if eid not in self.phys_to_graph_edges:
                    self.phys_to_graph_edges[eid] = []
                self.phys_to_graph_edges[eid].append(idx)

    def _build_edge_attr(self):
        """
        构建静态边特征矩阵（初始化时调用一次）
        edge_attr: [E, 5]
          [0] bw_remaining     — 归一化可用带宽（动态，初始=1.0）
          [1] bw_utilization   — 带宽利用率（动态，初始=0.0）
          [2] hop_weight_norm  — 拓扑跳数权重归一化（静态）
          [3] is_tree_edge     — 是否被当前树占用（动态，初始=0.0）
          [4] reserved         — 预留带宽比例（动态，初始=0.0）
        """
        rows, cols = self.edge_index[0], self.edge_index[1]
        E = len(rows)
        max_bw  = max(1.0, max(self.pool.bw_cap.values()) if self.pool.bw_cap else 1.0)
        max_hop = max(1.0, float(self.edge_hops.max()) if len(self.edge_hops) > 0 else 1.0)

        attr = np.zeros((E, 5), dtype=np.float32)
        for idx, (u, v) in enumerate(zip(rows, cols)):
            cap   = self.pool.bw_cap.get((u, v), self.pool.bw_cap.get((v, u), max_bw))
            avail = self.pool.bw_avail.get((u, v), self.pool.bw_avail.get((v, u), cap))
            # 维度与 shared_encoder 期望一致:
            # [0] bw_remaining (avail/cap)
            # [1] bw_utilization (1 - avail/cap)
            # [2] hop_weight_norm
            # [3] is_tree_edge (动态更新，初始0)
            # [4] reserved (bw_reserved/cap)
            util = 1.0 - avail / max(1.0, cap)
            attr[idx, 0] = avail / max(1.0, cap)         # bw_remaining
            attr[idx, 1] = util                          # bw_utilization
            attr[idx, 2] = self.edge_hops[idx] / max_hop # hop_weight_norm
            attr[idx, 3] = 0.0                           # is_tree_edge (动态)
            attr[idx, 4] = 0.0                           # reserved (动态)
        self.edge_attr = attr
        self._edge_attr_max_bw  = max_bw
        self._edge_attr_max_hop = max_hop
        logger.debug(f"[FusedRM] edge_attr 构建完成: shape={attr.shape}")

    def build_dynamic_edge_attr(self):
        """
        [SDG-HRL] 动态刷新 edge_attr（每step调用，更新带宽占用和树使用情况）
        返回 torch.Tensor [E, 5]，供 get_state() 直接使用
        """
        rows, cols = self.edge_index[0], self.edge_index[1]
        E   = len(rows)
        attr = self.edge_attr.copy()  # 在静态基础上更新动态部分

        # 当前树占用的边集合
        tree_edges = set()
        if hasattr(self, 'current_tree') and self.current_tree:
            for (u, v) in self.current_tree.get('tree', {}).keys():
                tree_edges.add((u, v))
                tree_edges.add((v, u))

        for idx, (u, v) in enumerate(zip(rows, cols)):
            cap   = self.pool.bw_cap.get((u, v), self.pool.bw_cap.get((v, u), self._edge_attr_max_bw))
            avail = self.pool.get_available_bandwidth(u, v)
            util  = 1.0 - avail / max(1.0, cap)
            attr[idx, 0] = avail / max(1.0, cap)         # bw_remaining
            attr[idx, 1] = util                          # bw_utilization
            attr[idx, 3] = 1.0 if (u, v) in tree_edges else 0.0  # is_tree_edge
            # 🆕 补全 reserved 维度（之前始终为0，GNN看不到预留信息）
            reserved = self.pool.bw_reserved.get((u, v), self.pool.bw_reserved.get((v, u), 0.0))
            attr[idx, 4] = reserved / max(1.0, cap)      # reserved ratio

        return torch.from_numpy(attr).float()

    # ---------- 事务接口 ----------
    def begin_transaction(self, request_id: str) -> str:
        return self.transaction_mgr.begin_transaction(f"tx_{request_id}_{int(time.time()*1000)}")

    def reserve_node_resource(self, tx_id: str, node: int, vnf_type: int,
                              cpu_need: float, mem_need: float) -> bool:
        return self.transaction_mgr.reserve_node(tx_id, node, vnf_type, cpu_need, mem_need)

    def reserve_link_resource(self, tx_id: str, u: int, v: int, bw_need: float) -> bool:
        return self.transaction_mgr.reserve_link(tx_id, u, v, bw_need)

    def commit_transaction(self, tx_id: str) -> bool:
        return self.transaction_mgr.commit(tx_id)

    def rollback_transaction(self, tx_id: str):
        self.transaction_mgr.rollback(tx_id)

    # ---------- 直接分配接口 ----------
    def allocate_node_resource(self, node: int, vnf_type: int,
                               cpu_need: float, mem_need: float = 0.0) -> bool:
        if node < 0 or node >= self.n:
            return False
        # 🚀 VNF复用：该节点已有此类型VNF实例，免费复用，不扣物理资源
        if vnf_type >= 0 and self.hvt_all[node, vnf_type] > 0:
            self.hvt_all[node, vnf_type] += 1
            return True
        # 首次实例化，真正扣除物理资源
        if not self.pool.allocate_cpu(node, cpu_need):
            return False
        if mem_need > 0 and not self.pool.allocate_memory(node, mem_need):
            self.pool.release_cpu(node, cpu_need)
            return False
        self.hvt_all[node, vnf_type] += 1
        # 记录首次实际扣除的CPU/MEM，释放时用此值而非当前请求值
        if not hasattr(self, 'vnf_instance_cost'):
            self.vnf_instance_cost = {}
        self.vnf_instance_cost[(node, vnf_type)] = (cpu_need, mem_need)
        return True

    def allocate_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        return self.pool.allocate_bandwidth(u, v, bw_need)

    def allocate_bandwidth(self, u: int, v: int, bw: float) -> bool:
        return self.allocate_link_resource(u, v, bw)

    def release_node_resource(self, node: int, vnf_type: int, cpu_val: float, mem_val: float):
        if node < 0 or node >= self.n:
            return
        if vnf_type >= 0 and self.hvt_all[node, vnf_type] > 0:
            if self.hvt_all[node, vnf_type] > 1:
                # 还有其他请求在用此VNF实例，只减计数，不归还物理资源
                self.hvt_all[node, vnf_type] -= 1
                return
            else:
                # 最后一个请求离开，归还首次分配时实际扣除的资源
                self.hvt_all[node, vnf_type] = 0
                if hasattr(self, 'vnf_instance_cost'):
                    actual_cpu, actual_mem = self.vnf_instance_cost.pop(
                        (node, vnf_type), (cpu_val, mem_val))
                else:
                    actual_cpu, actual_mem = cpu_val, mem_val
                if actual_cpu > 0:
                    self.pool.release_cpu(node, actual_cpu)
                if actual_mem > 0:
                    self.pool.release_memory(node, actual_mem)
        else:
            # hvt_all为0但仍被调用释放（兜底）
            if cpu_val > 0:
                self.pool.release_cpu(node, cpu_val)
            if mem_val > 0:
                self.pool.release_memory(node, mem_val)

    def release_link_resource(self, u: int, v: int, bw_val: float):
        self.pool.release_bandwidth(u, v, bw_val)

    def release_bandwidth(self, u: int, v: int, bw: float):
        self.release_link_resource(u, v, bw)

    def get_available_bandwidth(self, u: int, v: int) -> float:
        return self.pool.get_available_bandwidth(u, v)

    def check_node_resource(self, node: int, vnf_type: int = 0,
                            cpu_need: float = 0.0, mem_need: float = 0.0) -> bool:
        # 🚀 VNF复用：实例已存在则免费，直接返回True
        if vnf_type >= 0 and self.hvt_all[node, vnf_type] > 0:
            return True
        cpu_ok = self.pool.get_available_cpu(node) >= cpu_need - 1e-5
        mem_ok = self.pool.get_available_memory(node) >= mem_need - 1e-5
        return cpu_ok and mem_ok

    def check_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        return self.pool.get_available_bandwidth(u, v) >= bw_need - 1e-5

    # ---------- 业务辅助方法（委托给 handler）----------
    def _try_deploy(self, node: int) -> bool:
        return self.handler._try_deploy(node)

    def _archive_request(self, success=False, already_rolled_back=False):
        self.handler._archive_request(success, already_rolled_back)

    def complete_current_request(self):
        return self.handler.complete_current_request()

    def apply_deployment(self, plan: dict, request: dict) -> bool:
        return self.handler.apply_deployment(plan, request)

    def apply_tree_deployment(self, plan: dict, request: dict) -> bool:
        return self.handler.apply_tree_deployment(plan, request)

    def get_network_state_dict(self, current_request=None):
        return self.handler.get_network_state_dict(current_request)

    # ---------- 查询接口 ----------
    def get_available_resources(self, node_id: Optional[int] = None) -> Dict:
        if node_id is not None:
            return {
                'cpu': self.pool.get_available_cpu(node_id),
                'memory': self.pool.get_available_memory(node_id)
            }
        nodes = {i: {'cpu': self.pool.get_available_cpu(i),
                     'memory': self.pool.get_available_memory(i)} for i in range(self.n)}
        links = {key: {'bandwidth': self.pool.get_available_bandwidth(*key)} for key in self.pool.link_map}
        return {'nodes': nodes, 'links': links}

    def get_neighbors(self, node: int) -> List[int]:
        if node < 0 or node >= self.n:
            return []
        return np.where(self.topo[node] > 0)[0].tolist()

    def has_link(self, u: int, v: int) -> bool:
        return self.pool.get_edge_id(u, v) is not None

    def get_link_cost(self, u: int, v: int) -> float:
        return 1.0

    def get_node_features(self, nodes_on_tree):
        feats = []
        for i in range(self.n):
            feats.append([
                self.pool.get_available_cpu(i) / max(1, self.C_cap),
                self.pool.get_available_memory(i) / max(1, self.M_cap),
                1.0 if i in nodes_on_tree else 0.0
            ])
        return np.array(feats, dtype=np.float32)

    def episode_reset(self):
        """
        🔥 Episode 级轻量重置（在线模拟专用）

        只清理未提交的事务预留（防止上一个 Episode 的残留锁住资源），
        保留：
          - lifecycle 中的活跃请求（让其自然过期）
          - hvt_all 和 vnf_instances（lifecycle 释放时需要这两个表）
          - pool 的 avail（CPU/MEM/BW 真实占用量）

        调用时机：每个 Episode（请求）结束后，准备处理下一个请求前。
        不调用时机：仿真开始/完全重置（用 reset(hard=True)）。
        """
        # 只清 reserved（未提交事务的预留残留），不动 avail 和 lifecycle
        # hvt_all、vnf_instances 保持不变，lifecycle 释放时会自己维护
        self.pool.reset(hard=False)
        logger.debug("[FusedRM] Episode软重置完成 (lifecycle/hvt/instances保留)")

    def reset(self, hard: bool = False, current_time: float = 0.0):
        """
        重置入口，自动根据模式选择行为：
          online_mode=True  + hard=False → episode_reset()（保留lifecycle）
          online_mode=False 或 hard=True → 完整重置（cleanup_all）
        """
        online = (hasattr(self, 'env') and
                  getattr(self.env, 'online_mode', False))

        if online and not hard:
            self.episode_reset()
        else:
            self.pool.reset(hard)
            self.hvt_all.fill(0)
            if hasattr(self, 'vnf_instance_cost'):
                self.vnf_instance_cost.clear()
            self.vnf_instances.clear()
            if hasattr(self, 'env') and hasattr(self.env, 'current_time'):
                ct = self.env.current_time
            else:
                ct = current_time
            self.request_manager.cleanup_all(current_time=ct)
            logger.debug(f"[FusedRM] 完整重置完成 (hard={hard})")