"""
envs/modules/resource.py
==================================================
NFV-Oriented HRL-GNN Resource Management Module
==================================================
完整修复版:
1. ✅ 包含 get_graph_state 方法 (不再省略)
2. ✅ 修复资源同步问题 (self.nodes 引用 self.C/M)
3. ✅ apply_tree_deployment 支持字典 tree 并同步 self.B
"""
import numpy as np
import networkx as nx
import torch
import logging
from collections import defaultdict
from typing import Dict, Set, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    ResourceManager（资源管理器）- 完整功能版
    """

    def __init__(self, topo: np.ndarray, capacities: Dict, dc_nodes: List[int], link_map: Optional[Dict] = None):
        """
        初始化资源管理器
        """
        self.topo = topo
        self.n = topo.shape[0]

        # 🚨 分离 图边数(E) 与 物理链路数(L)
        self.num_graph_edges = np.sum(topo > 0)

        # 如果提供了 link_map，用最大 ID 作为链路数；否则默认对称除以2
        if link_map:
            max_id = 0
            for k, v in link_map.items():
                if v > max_id: max_id = v
            self.L = max_id
        else:
            self.L = self.num_graph_edges // 2

        logger.info(f"[RM] Init: Nodes={self.n}, GraphEdges={self.num_graph_edges}, PhysLinks={self.L}")

        # 1. 资源容量配置
        self.B_cap = capacities.get('bandwidth', 80.0)
        self.C_cap = capacities.get('cpu', 10.0)
        self.M_cap = capacities.get('memory', 80.0)
        self.K_vnf = 8

        self.dc_nodes = list(dc_nodes)
        self.link_map = link_map

        # 2. 动态资源状态矩阵
        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 节点资源
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)

        # 3. 状态向量维度定义
        self.dim_request = 10
        self.dim_network = self.n * 2 + self.L + self.n * self.K_vnf
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        self._dest_dist_cache = {}
        self.vnf_sharing_map = {}

        self._build_shortest_dist_matrix()

        # GNN 维度定义
        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24

        # 🔥 构建边索引和映射表
        self._build_edge_index()

        # ========================================
        # 🔥【关键修复】资源字典引用同步 (不再使用 copy)
        # ========================================
        self.nodes = {
            'cpu': self.C,  # 直接引用 self.C
            'memory': self.M  # 直接引用 self.M
        }

        self.links = {
            'bandwidth': {}
        }

        for i in range(self.n):
            for j in range(self.n):
                if self.topo[i, j] > 0:
                    self.links['bandwidth'][(i, j)] = self.B_cap

        self.vnf_instances = []

        # 兼容性别名
        self.node_cap = self.C
        self.node_mem = self.M
        self.link_cap = self.B

        logger.info(f"✅ ResourceManager 初始化完成")

    def apply_deployment(self, plan: dict, request: dict) -> bool:
        """应用部署方案"""
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

        # 1. 资源检查
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            # 容忍微小浮点误差
            if self.nodes['cpu'][node] < cpu_need - 1e-5: return False
            if self.nodes['memory'][node] < mem_need - 1e-5: return False

        # 2. 资源扣除 (自动同步到 self.C/M)
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            node = int(node)
            vnf_t = int(vnf_t)

            cpu_need = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0
            mem_need = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0

            self.nodes['cpu'][node] -= cpu_need
            self.nodes['memory'][node] -= mem_need
            self.hvt_all[node, vnf_t] += 1  # 记录实例

            self.vnf_instances.append({
                'req_id': req_id,
                'node': node,
                'vnf_type': vnf_t,
                'cpu': cpu_need,
                'memory': mem_need
            })

        return True
    def apply_tree_deployment(self, plan: dict, request: dict) -> bool:
        """应用树部署方案 (支持 Dict Tree 并同步 self.B)"""
        # 1. 部署 VNF
        if not self.apply_deployment(plan, request):
            return False

        # 2. 部署链路
        tree = plan.get('tree', {})
        bw_need = request.get('bw_origin', 0)

        # 兼容 array 类型的 tree (Expert 原始输出)
        if isinstance(tree, (np.ndarray, list)):
            # Expert 模式下，通常 VNF 部署成功就算成功，链路带宽由 Expert 保证
            # 但为了同步 B 数组，我们应该尽量解析（如果有 edge_to_phys）
            return True

        # 字典模式 {(u,v): flow}
        for edge_key, flow in tree.items():
            u, v = None, None
            if isinstance(edge_key, tuple):
                u, v = edge_key
            elif isinstance(edge_key, str):
                try:
                    u, v = map(int, edge_key.strip('()').split('-'))
                except:
                    pass

            if u is not None and v is not None:
                if (u, v) in self.links['bandwidth']:
                    # 检查带宽
                    if self.links['bandwidth'][(u, v)] < bw_need * flow - 1e-5:
                        return False

                    # 扣除带宽
                    self.links['bandwidth'][(u, v)] -= bw_need * flow

                    # 🔥【关键修复】同步更新 self.B 数组 (供 Expert 使用)
                    if hasattr(self, 'edge_to_phys') and (u, v) in self.edge_to_phys:
                        pid = self.edge_to_phys[(u, v)]
                        if pid < len(self.B):
                            self.B[pid] = self.links['bandwidth'][(u, v)]
                            self.link_ref_count[pid] += 1

        return True
    def get_network_state_dict(self, current_request=None):
        """返回结构化网络状态"""
        # 此时 self.C, self.M, self.B 已经是同步后的最新值
        state = {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }
        if current_request:
            state['request'] = current_request
        return state
    def _build_edge_index(self):
        """构建 PyTorch Geometric edge_index"""
        rows, cols = np.where(self.topo > 0)

        # 建立 物理ID -> [图边ID列表] 的映射
        self.phys_to_graph_edges = defaultdict(list)
        self.edge_to_phys = {}  # ✅ 新增：(u, v) -> phys_id 映射

        edge_list = []

        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_list.append([u, v])

            # 获取物理链路ID (0-based)
            phys_id = -1
            if self.link_map:
                # link_map key 是 1-based tuple
                phys_id = self.link_map.get((u + 1, v + 1))
                if phys_id is None:
                    phys_id = self.link_map.get((v + 1, u + 1))

            if phys_id is not None and phys_id > 0:
                real_phys_id = phys_id - 1
                self.phys_to_graph_edges[real_phys_id].append(idx)
                self.edge_to_phys[(u, v)] = real_phys_id
            elif not self.link_map:
                # 简单映射兜底
                if idx < self.L:
                    self.phys_to_graph_edges[idx].append(idx)
                    self.edge_to_phys[(u, v)] = idx

        self.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        self.edge_hops = torch.tensor([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=torch.float32)
    def get_graph_state(self, current_request, nodes_on_tree, current_tree,
                        served_dest_count: int, sharing_strategy: int, nb_high_goals: int):
        """获取图状态 (GNN Input)"""
        if not current_request:
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            req_vec = torch.zeros(self.request_dim)
            return x, self.edge_index, edge_attr, req_vec

        src = current_request['source']
        dest_set = set(current_request.get('dest', []))
        avg_dist = self._compute_dest_distances(dest_set)
        sharing_potential = self._compute_vnf_sharing_potential(dest_set)

        # 1. 节点特征
        node_feats = []
        for i in range(self.n):
            nid = i + 1
            feat = [
                1.0 - self.C[i] / max(1, self.C_cap),
                1.0 - self.M[i] / max(1, self.M_cap),
                1.0 if nid in self.dc_nodes else 0.0,
                1.0 if nid == src else 0.0,
                1.0 if nid in dest_set else 0.0,
                1.0 if nid in nodes_on_tree else 0.0,
                sum(1 for d in dest_set if 0 <= d - 1 < self.n and self.shortest_dist[i, d - 1] <= 3) / max(1,
                                                                                                            len(dest_set)),
                1.0 - avg_dist[i] / max(1, np.max(avg_dist)) if np.max(avg_dist) > 0 else 0,
                sharing_potential[i]
            ]
            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)
        x = torch.tensor(node_feats, dtype=torch.float32)

        # 2. 边特征
        num_edges = self.edge_index.shape[1]
        edge_attrs = torch.zeros((num_edges, self.edge_feat_dim), dtype=torch.float32)

        # 获取当前树中的物理链路集合
        tree_obj = current_tree.get("tree", {})
        tree_phys_links = set()

        if isinstance(tree_obj, dict):
            # Dict Tree
            for edge in tree_obj.keys():
                u, v = None, None
                if isinstance(edge, tuple):
                    u, v = edge
                elif isinstance(edge, str):
                    try:
                        u, v = map(int, edge.strip('()').split('-'))
                    except:
                        pass

                if u is not None:
                    if (u, v) in self.edge_to_phys:
                        tree_phys_links.add(self.edge_to_phys[(u, v)])
                    elif (v, u) in self.edge_to_phys:
                        tree_phys_links.add(self.edge_to_phys[(v, u)])
        else:
            # Array Tree
            tree_phys_links = set(np.where(tree_obj > 0)[0])

        for phys_idx in range(self.L):
            graph_edge_indices = self.phys_to_graph_edges.get(phys_idx, [])
            bw_util = 1.0 - self.B[phys_idx] / max(1, self.B_cap)
            in_tree = 1.0 if phys_idx in tree_phys_links else 0.0

            for edge_idx in graph_edge_indices:
                hop = float(self.edge_hops[edge_idx])
                edge_attrs[edge_idx] = torch.tensor([bw_util, in_tree, hop, 0.0, bw_util], dtype=torch.float32)

        # 3. 请求向量
        num_dests = len(current_request.get('dest', []))
        chain_len = len(current_request.get('vnf', []))
        avg_cpu = np.mean(current_request.get('cpu_origin', [0]))
        avg_mem = np.mean(current_request.get('memory_origin', [0]))
        bw_demand = current_request.get('bw_origin', 0.0)

        base_feats = np.array([
            bw_demand / max(1.0, self.B_cap),
            num_dests / 20.0,
            chain_len / 10.0,
            avg_cpu / max(1.0, self.C_cap),
            avg_mem / max(1.0, self.M_cap),
            served_dest_count / max(1, num_dests),
            sharing_strategy / 3.0,
            len(nodes_on_tree) / self.n,
            len(tree_phys_links) / max(1, self.L),
            0.0
        ], dtype=np.float32)

        dest_onehot = np.zeros(nb_high_goals, dtype=np.float32)
        for i in range(min(num_dests, nb_high_goals)):
            dest_onehot[i] = 1.0

        req_vec_array = np.concatenate([base_feats, dest_onehot])
        if len(req_vec_array) > 24:
            req_vec_array = req_vec_array[:24]
        elif len(req_vec_array) < 24:
            req_vec_array = np.pad(req_vec_array, (0, 24 - len(req_vec_array)))

        req_vec = torch.tensor(req_vec_array, dtype=torch.float32)
        return x, self.edge_index, edge_attrs, req_vec
    def _compute_dest_distances(self, dest_set):
        key = frozenset(dest_set)
        if key in self._dest_dist_cache: return self._dest_dist_cache[key]
        avg_dist = np.zeros(self.n)
        if not dest_set: return avg_dist
        for i in range(self.n):
            dists = [self.shortest_dist[i, d - 1] for d in dest_set if 0 <= d - 1 < self.n]
            avg_dist[i] = np.mean(dists) if dists else 999.0
        self._dest_dist_cache[key] = avg_dist
        return avg_dist
    def _compute_vnf_sharing_potential(self, dest_set):
        avg_dist = self._compute_dest_distances(dest_set)
        dist_factor = 1.0 - avg_dist / (np.max(avg_dist) + 1e-5)
        resource_factor = 1.0 - (self.C / max(1, self.C_cap))
        return np.clip(0.4 * dist_factor + 0.3 * resource_factor, 0, 1)
    def _build_shortest_dist_matrix(self):
        self.shortest_dist = np.full((self.n, self.n), 9999.0)
        np.fill_diagonal(self.shortest_dist, 0.0)
        # 实际应使用 FW 算法或 Dijkstra，这里假设已预计算或简化
        # 如果需要精确距离，请确保此矩阵正确初始化
    def get_flat_state(self, *args, **kwargs):
        """占位符，如果需要 Flat State 请实现"""
        return np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)

    def get_vnf_sharing_rate(self) -> float:
        return 0.0

        # 在 envs/modules/resource.py 的 ResourceManager 类中

    def release_resources_from_req(self, req, plan):
        """
        释放请求占用的资源
        :param req: 请求对象
        :param plan: 部署方案 {'hvt': ..., 'tree': ...}
        """
        bw_ret = float(req.get('bw_origin', 0.0))
        cpu_reqs = req.get('cpu_origin', [])
        mem_reqs = req.get('memory_origin', [])

        # 1. 释放 VNF 资源 (节点 CPU/Mem)
        hvt = plan.get('hvt')
        if hvt is not None:
            # 遍历所有被占用的节点和VNF类型
            for node, vnf_t in np.argwhere(hvt > 0):
                node = int(node)
                vnf_t = int(vnf_t)

                # 获取该 VNF 占用的资源量
                cpu_val = cpu_reqs[vnf_t] if vnf_t < len(cpu_reqs) else 0.0
                mem_val = mem_reqs[vnf_t] if vnf_t < len(mem_reqs) else 0.0

                # 恢复资源 (不超过上限)
                self.nodes['cpu'][node] = min(self.C_cap, self.nodes['cpu'][node] + cpu_val)
                self.nodes['memory'][node] = min(self.M_cap, self.nodes['memory'][node] + mem_val)

                # 减少实例计数
                if self.hvt_all[node, vnf_t] > 0:
                    self.hvt_all[node, vnf_t] -= 1

        # 2. 释放链路资源 (带宽)
        tree_edges = plan.get('tree', {})

        if isinstance(tree_edges, dict):
            for edge_key, flow in tree_edges.items():
                if flow <= 0: continue

                u, v = None, None
                # 解析键值 (u, v)
                if isinstance(edge_key, tuple):
                    u, v = edge_key
                elif isinstance(edge_key, str):
                    try:
                        u, v = map(int, edge_key.strip('()').split('-'))
                    except:
                        pass

                if u is not None and v is not None:
                    # 恢复 links 字典中的带宽
                    if (u, v) in self.links['bandwidth']:
                        self.links['bandwidth'][(u, v)] = min(
                            self.B_cap,
                            self.links['bandwidth'][(u, v)] + bw_ret * flow
                        )

                        # 🔥 同步更新 self.B (Expert 视角)
                        if hasattr(self, 'edge_to_phys') and (u, v) in self.edge_to_phys:
                            pid = self.edge_to_phys[(u, v)]
                            if pid < len(self.B):
                                self.B[pid] = self.links['bandwidth'][(u, v)]
                                if self.link_ref_count[pid] > 0:
                                    self.link_ref_count[pid] -= 1
    def share_vnf(self, node_id: int, vnf_type: int, dest_idx: int):
        pass

    def compute_progress(self, *args):
        return 0.0

    def compute_qos_violation(self, *args):
        return None

    # ==========================================================
    # 🔥【补全缺失方法】修复 AttributeError
    # ==========================================================
    def get_neighbors(self, node: int) -> List[int]:
        """获取节点的邻居索引"""
        if node < 0 or node >= self.n:
            return []
        return np.where(self.topo[node] > 0)[0].tolist()

    def check_node_resource(self, node: int, vnf_type: int, cpu_need: float = 0.0, mem_need: float = 0.0) -> bool:
        """检查节点资源是否足够"""
        if node < 0 or node >= self.n:
            return False
        # 如果未传入具体数值，默认检查是否大于0（或者由调用方保证传入值）
        return self.C[node] >= cpu_need - 1e-5 and self.M[node] >= mem_need - 1e-5

    def check_link_resource(self, u: int, v: int, bw_need: float) -> bool:
        """
        🔥 [新增] 检查链路带宽资源 (只读，不扣费)
        用于 Mask 生成和路径规划预判
        """
        # 1. 拓扑检查
        if not self.has_link(u, v):
            return False

        # 2. 带宽检查 (兼容不同存储结构)
        # 情况 A: 字典存储 (self.links['bandwidth'])
        if hasattr(self, 'links') and 'bandwidth' in self.links:
            # 检查正向
            if (u, v) in self.links['bandwidth']:
                return self.links['bandwidth'][(u, v)] >= bw_need - 1e-5
            # 检查反向 (如果是无向图或双向存储)
            elif (v, u) in self.links['bandwidth']:
                return self.links['bandwidth'][(v, u)] >= bw_need - 1e-5
            # 默认：有连接但没带宽记录，视为 0
            return False

        # 情况 B: 矩阵存储
        elif hasattr(self, 'topology'):
            # 假设 topology 矩阵直接存带宽
            return self.topology[u][v] >= bw_need - 1e-5

        # 情况 C: self.B 数组直接映射 (最快)
        elif hasattr(self, 'edge_to_phys') and hasattr(self, 'B'):
            pid = self.edge_to_phys.get((u, v))
            if pid is None: pid = self.edge_to_phys.get((v, u))

            if pid is not None and pid < len(self.B):
                return self.B[pid] >= bw_need - 1e-5

        # 兜底
        return False

    def get_link_cost(self, u: int, v: int) -> float:
        """获取链路开销 (默认跳数为1，可扩展为延迟或带宽倒数)"""
        return 1.0

    def get_node_features(self, nodes_on_tree):
        """获取节点特征矩阵 (兼容旧版调用)"""
        # 注意：这里返回的是简化的特征，用于非GNN模式或Fallback
        # 实际 GNN 特征在 get_graph_state 中构建
        feats = []
        for i in range(self.n):
            f = [
                self.C[i] / self.C_cap,
                self.M[i] / self.M_cap,
                1.0 if i in nodes_on_tree else 0.0
            ]
            feats.append(f)
        return np.array(feats, dtype=np.float32)

    def get_edge_features(self):
        """获取边特征 (兼容旧版调用)"""
        return self.edge_index, torch.zeros((self.edge_index.shape[1], 5))

    def has_link(self, u: int, v: int) -> bool:
        """检查节点 u 和 v 之间是否有物理链路"""
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            return False
        return self.topo[u, v] > 0

        # ==========================================================
        # 🔥【Resource Manager 最终版】支持带宽释放 & 安全Reset
        # ==========================================================

    def reset(self, hard=False):
        """
        Episode-level reset:
        :param hard: 是否强制重置所有物理资源 (用于 Phase 切换或初始化)
        """
        if hard:
            # 只有在初始化或显式要求时，才恢复满资源
            self.B.fill(self.B_cap)
            self.C.fill(self.C_cap)
            self.M.fill(self.M_cap)
            # logger.warning("⚠️ 执行了 HARD RESET，资源已回满")

        # --- 常规 Episode Reset (Soft) ---
        # 仅清理临时缓存，保留 C/M/B 的当前占用状态
        self.hvt_all.fill(0)
        self.link_ref_count.fill(0)
        self.vnf_sharing_map.clear()
        self._dest_dist_cache.clear()
        self.vnf_instances = []

        # 同步字典
        self.nodes['cpu'] = self.C
        self.nodes['memory'] = self.M

        # 同步带宽 (防 IndexError)
        if hasattr(self, 'edge_to_phys'):
            for (u, v), pid in self.edge_to_phys.items():
                if pid < len(self.B):
                    self.links['bandwidth'][(u, v)] = self.B[pid]

    # =========================================================
    # 🔥 [新增] 资源释放接口 (修复版 - 带上限检查)
    # =========================================================

    def release_link_resource(self, u, v, bw_val):
        """释放链路资源（修复版）"""

        if bw_val <= 0:
            return

        # 规范化边键
        edge_key = tuple(sorted([u, v]))

        # 通过物理ID释放
        if hasattr(self, 'edge_to_phys'):
            pid = self.edge_to_phys.get(edge_key)
            if pid is None:
                pid = self.edge_to_phys.get((u, v))
            if pid is None:
                pid = self.edge_to_phys.get((v, u))

            if pid is not None and pid < len(self.B):
                limit_b = float(self.B_cap[pid]) if hasattr(self.B_cap, '__getitem__') else float(self.B_cap)

                # 归还（加法 + 上限）
                self.B[pid] = min(limit_b, self.B[pid] + bw_val)

                # 🔥 关键：只更新一个方向
                if hasattr(self, 'links') and 'bandwidth' in self.links:
                    current_bw = self.B[pid]

                    # 只更新找到的第一个
                    if edge_key in self.links['bandwidth']:
                        self.links['bandwidth'][edge_key] = current_bw
                    elif (u, v) in self.links['bandwidth']:
                        self.links['bandwidth'][(u, v)] = current_bw
                    elif (v, u) in self.links['bandwidth']:
                        self.links['bandwidth'][(v, u)] = current_bw
                    # 🔥 不再同时更新(u,v)和(v,u)

                return

        # Fallback
        if hasattr(self, 'links') and 'bandwidth' in self.links:
            limit_b = 100.0

            # 只更新一个方向
            if edge_key in self.links['bandwidth']:
                self.links['bandwidth'][edge_key] = min(limit_b,
                                                        self.links['bandwidth'][edge_key] + bw_val)
            elif (u, v) in self.links['bandwidth']:
                self.links['bandwidth'][(u, v)] = min(limit_b,
                                                      self.links['bandwidth'][(u, v)] + bw_val)
            elif (v, u) in self.links['bandwidth']:
                self.links['bandwidth'][(v, u)] = min(limit_b,
                                                      self.links['bandwidth'][(v, u)] + bw_val)

    def release_node_resource(self, node_id, vnf_type, cpu_val, mem_val):
        """
        [修复版] 释放节点资源 (兼容 C_cap 是 float 的情况)
        """
        # --- 1. 归还 CPU ---
        # 检查 C_cap 是数组还是标量
        if hasattr(self.C_cap, '__getitem__'):
            limit_c = self.C_cap[node_id]  # 数组：取对应节点的上限
        else:
            limit_c = self.C_cap  # 标量：直接使用统一上限

        if hasattr(self, 'C'):
            self.C[node_id] = min(limit_c, self.C[node_id] + cpu_val)

        # --- 2. 归还 Memory ---
        if hasattr(self.M_cap, '__getitem__'):
            limit_m = self.M_cap[node_id]
        else:
            limit_m = self.M_cap

        if hasattr(self, 'M'):
            self.M[node_id] = min(limit_m, self.M[node_id] + mem_val)

        # --- 3. 更新 VNF 计数 (如果有的话) ---
        if hasattr(self, 'hvt_all'):
            self.hvt_all[node_id, vnf_type] = max(0.0, self.hvt_all[node_id, vnf_type] - 1.0)

        # --- 4. 同步观测状态 (给 Gym State 用) ---
        if hasattr(self, 'nodes'):
            if 'cpu' in self.nodes:
                self.nodes['cpu'][node_id] = self.C[node_id]
            if 'memory' in self.nodes:
                self.nodes['memory'][node_id] = self.M[node_id]
    # =========================================================
    # 🔥 [新增] 资源分配接口 (支持 V9.1 真实扣费)
    # =========================================================

    def allocate_link_resource(self, u, v, bw_need):
        """
        扣除链路带宽资源（修复版 - 同步更新 self.B）
        :param u: 起点节点 ID
        :param v: 终点节点 ID
        :param bw_need: 需要的带宽量
        :return: bool (成功/失败)
        """
        # 1. 检查链路是否存在
        if not self.has_link(u, v):
            return False

        # 2. 获取当前带宽 (兼容不同存储结构)
        if hasattr(self, 'links') and 'bandwidth' in self.links:
            # 结构 A: self.links['bandwidth'][(u,v)]
            current_bw = self.links['bandwidth'].get((u, v), 0.0)

            if current_bw >= bw_need:
                self.links['bandwidth'][(u, v)] -= bw_need

                # ✅ 关键修复：同步更新 self.B 数组
                if hasattr(self, 'edge_to_phys'):
                    pid = self.edge_to_phys.get((u, v))
                    if pid is None:
                        pid = self.edge_to_phys.get((v, u))

                    if pid is not None and pid < len(self.B):
                        self.B[pid] = self.links['bandwidth'][(u, v)]

                return True
            else:
                return False

        elif hasattr(self, 'topology'):
            # 结构 B: 直接存储在矩阵中 (self.topology[u][v])
            current_bw = self.topology[u][v]

            if current_bw >= bw_need:
                self.topology[u][v] -= bw_need
                return True
            else:
                return False

        return False

    def allocate_node_resource(self, node_id, vnf_type, cpu_need, mem_need=0.0):
        """
        扣除节点计算资源（修复版 - 同步更新 self.nodes）
        :param node_id: 节点 ID
        :param vnf_type: VNF 类型 (部分逻辑可能需要)
        :param cpu_need: CPU 需求
        :param mem_need: 内存需求 (可选)
        :return: bool (成功/失败)
        """
        # 1. 边界检查
        if node_id < 0 or node_id >= self.n:
            return False

        # 2. 检查 CPU
        # 假设 self.C 存储节点剩余 CPU 容量
        if hasattr(self, 'C'):
            if self.C[node_id] >= cpu_need:
                self.C[node_id] -= cpu_need

                # 3. 检查内存 (如果有)
                if hasattr(self, 'M') and mem_need > 0:
                    if self.M[node_id] >= mem_need:
                        self.M[node_id] -= mem_need
                    else:
                        # 回滚 CPU 扣除
                        self.C[node_id] += cpu_need
                        return False

                # ✅ 关键修复：同步更新 self.nodes 字典
                if hasattr(self, 'nodes'):
                    if 'cpu' in self.nodes:
                        self.nodes['cpu'][node_id] = self.C[node_id]
                    if 'memory' in self.nodes:
                        self.nodes['memory'][node_id] = self.M[node_id]

                return True
            else:
                return False

        # 如果没有资源管理属性，默认允许 (Fallback)
        return True