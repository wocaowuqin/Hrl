# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的资源管理器 (Fixed Scope & Logic Version)

修复记录:
1. ✅ [CRITICAL] 修复 check_global_feasibility 缩进错误，使其能被外部调用
2. ✅ [CRITICAL] 修复 _encode_request 返回全0向量的问题，增加特征维度
3. ✅ [LOGIC] check_global_feasibility 改为宽松逻辑（单路径可达即放行）
4. ✅ [LOGIC] apply_tree_deployment 增加 phase 参数，支持 Phase1 宽松模式
"""

import numpy as np
import networkx as nx
import logging
import threading
import copy
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceManager:
    def __init__(self, *args, **kwargs):
        """
        初始化资源管理器
        支持:
        1. (topo, capacities, dc_nodes) - 新版
        2. (node_num, link_num, type_num, ...) - 旧版
        """
        self.init_mode = "unknown"
        self.node_index_base = kwargs.get('node_index_base', 1)
        self._lock = threading.RLock()

        if len(args) >= 4 and isinstance(args[0], int):
            self._init_legacy(*args, **kwargs)
        elif len(args) >= 2:
            self._init_modern(*args, **kwargs)
        else:
            if 'topo' in kwargs:
                self._init_modern(**kwargs)
            else:
                raise TypeError("Invalid arguments for ResourceManager init")

    def _init_legacy(self, node_num, link_num, type_num, cap_cpu, cap_mem=80.0, cap_bw=80.0, **kwargs):
        self.init_mode = "legacy"
        logger.info("[RM] Using legacy initialization")

        self.n = int(node_num)
        self.topo = np.ones((self.n, self.n), dtype=np.float32)
        np.fill_diagonal(self.topo, 0)

        self.L_provided = int(link_num)
        self.K_vnf = int(type_num)
        self.num_graph_edges = int(np.sum(self.topo > 0))
        self.C_cap = float(cap_cpu)
        self.M_cap = float(cap_mem)
        self.B_cap = float(cap_bw)
        self.dc_nodes = list(range(min(10, self.n)))

        self.link_map = self._create_link_map(self.topo)
        max_lid = max(self.link_map.values()) if self.link_map else 0
        self.L = max(self.L_provided, max_lid)

        self._init_common()

    def _init_modern(self, topo, capacities, dc_nodes, link_map=None, **kwargs):
        self.init_mode = "modern"
        logger.info("[RM] Using modern initialization")

        self.topo = topo
        self.n = topo.shape[0]
        self.num_graph_edges = int(np.sum(topo > 0))

        if link_map:
            self.link_map = link_map
        else:
            self.link_map = self._create_link_map(topo)

        self.L = max(self.link_map.values()) if self.link_map else 0

        self.C_cap = float(capacities.get('cpu', 80.0))
        self.M_cap = float(capacities.get('memory', 60.0))
        self.B_cap = float(capacities.get('bandwidth', 80.0))
        self.K_vnf = 8
        self.dc_nodes = list(dc_nodes)

        self._init_common()

    def _init_common(self):
        self.C = np.full(self.n, self.C_cap, dtype=np.float32)
        self.M = np.full(self.n, self.M_cap, dtype=np.float32)

        if self.L > 0:
            self.B = np.full(self.L, self.B_cap, dtype=np.float32)
        else:
            self.B = np.array([], dtype=np.float32)

        self.link_ref_count = np.zeros(self.L, dtype=np.int32)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=np.float32)

        self.nodes = {'cpu': self.C, 'memory': self.M}
        self.links = {'bandwidth': {}}

        for (u, v), lid in self.link_map.items():
            idx = lid - 1
            if 0 <= idx < self.L:
                self.links['bandwidth'][(u, v)] = self.B[idx]

        self.initial_C = self.C.copy()
        self.initial_M = self.M.copy()
        self.initial_B = self.B.copy()

        self.vnf_instances = []
        self.active_requests = {}
        self.vnf_sharing_map = defaultdict(set)

        try:
            self.shortest_dist = self._build_shortest_dist_matrix()
            self.edge_index = self._build_edge_index()
        except Exception as e:
            logger.warning(f"GNN结构初始化部分失败: {e}")
            self.edge_index = np.zeros((2, 0))

        self.node_feat_dim = 6 + self.K_vnf + 3
        self.edge_feat_dim = 5
        self.request_dim = 24  # 确保与 _encode_request 一致

        self.initial_state_template = {
            'cpu': self.C.copy(),
            'mem': self.M.copy(),
            'bw': self.B.copy(),
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'bw_ref_count': np.zeros(self.L, dtype=np.int32)
        }
        logger.info(f"✅ ResourceManager Ready: Nodes={self.n}, Links={self.L}")

    # ========================================================================
    # 修复点 1: 辅助函数移出 _init_common，作为类方法
    # ========================================================================
    def _build_bw_feasible_subgraph(self, bw_req):
        """
        构建一个仅包含满足带宽需求链路的临时 NetworkX 图
        """
        G = nx.Graph()
        for (u, v), lid in self.link_map.items():
            idx = lid - 1
            if self.B[idx] >= bw_req - 1e-5:
                G.add_edge(u, v)
        return G

        # =========================================================================
        # 2️⃣ 核心修正：全局可行性检查 (带 ID 自动转换)
        # =========================================================================
    def check_global_feasibility(self, request, state):
        """
        检查全局可行性 (修正版)
        1. 自动转换 1-based 节点 ID 到 0-based 内部索引
        2. 在连通子图中检查 source -> dests 的可达性
        """
        # 1. 转换 Source ID
        raw_source = request.get('source')
        if raw_source is None:
            return False
        # 强制从外部视角转换 (如果 node_index_base=1，这里会自动 -1)
        source = self._normalize_node(raw_source, from_external=True)

        # 2. 转换 Dest IDs
        raw_dests = request.get('destinations') or request.get('dest')
        if not raw_dests:
            return False
        # 对列表中的每个 dest 做转换
        dests = [self._normalize_node(d, from_external=True) for d in raw_dests]

        # 3. 检查转换后的 ID 是否越界 (双重保险)
        if source < 0 or source >= self.n:
            return False  # 转换后依然不合法，直接拒

        # 4. 构建带宽满足的子图
        bw_req = request.get('bw_origin', 0)
        G_bw = self._build_bw_feasible_subgraph(bw_req)

        # 5. 基础检查：源节点是否孤立
        if not G_bw.has_node(source):
            return False

        # 6. 宽松连通性检查 (只要能通一个 dest 就算可行)
        for d in dests:
            if 0 <= d < self.n and G_bw.has_node(d):
                if nx.has_path(G_bw, source, d):
                    return True

        return False

    def create_initial_state(self):
        return copy.deepcopy(self.initial_state_template)

    def normalize_state(self, state):
        normalized = {}
        for key in ['cpu', 'mem', 'bw', 'hvt']:
            normalized[key] = state.get(key, self.initial_state_template[key].copy())
        normalized['bw_ref_count'] = state.get('bw_ref_count',
                                               self.initial_state_template['bw_ref_count'].copy())
        return normalized

    # ========================================================================
    # 核心资源操作
    # ========================================================================

    def reset(self):
        with self._lock:
            self.C[:] = self.initial_C
            self.M[:] = self.initial_M
            self.B[:] = self.initial_B
            self.hvt_all.fill(0)
            self.link_ref_count.fill(0)
            self.vnf_instances.clear()
            self.active_requests.clear()
            self.vnf_sharing_map.clear()

            for (u, v), lid in self.link_map.items():
                idx = lid - 1
                if 0 <= idx < self.L:
                    self.links['bandwidth'][(u, v)] = self.B[idx]

    def check_node_resources(self, node: int, cpu_req: float, mem_req: float, external_index: bool = False) -> bool:
        internal_node = self._normalize_node(node, external_index)
        if internal_node < 0 or internal_node >= self.n: return False
        return (self.C[internal_node] >= cpu_req - 1e-5) and (self.M[internal_node] >= mem_req - 1e-5)

    def check_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        lid = self.link_map.get((u, v))
        if lid is None: return False
        idx = lid - 1
        if idx < 0 or idx >= self.L: return False
        return self.B[idx] >= bw_req - 1e-5

    def update_resources(self, node_id: int, cpu_delta: float, mem_delta: float,
                         strict: bool = True, external_index: bool = False) -> bool:
        with self._lock:
            internal_node = self._normalize_node(node_id, external_index)
            if internal_node < 0 or internal_node >= self.n: return False

            new_cpu = self.C[internal_node] + cpu_delta
            new_mem = self.M[internal_node] + mem_delta

            if strict:
                if new_cpu < -1e-5 or new_mem < -1e-5: return False
                if new_cpu > self.C_cap + 1e-5: return False

            self.C[internal_node] = np.clip(new_cpu, 0.0, self.C_cap)
            self.M[internal_node] = np.clip(new_mem, 0.0, self.M_cap)
            return True

    def allocate_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        with self._lock:
            if (u, v) not in self.link_map: return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if idx < 0 or idx >= len(self.B): return False

            if self.B[idx] < bw_req - 1e-5: return False

            self.B[idx] = max(0.0, self.B[idx] - bw_req)
            self.link_ref_count[idx] += 1
            self.links['bandwidth'][(u, v)] = self.B[idx]
            return True

    def release_link_bandwidth(self, u: int, v: int, bw_req: float) -> bool:
        with self._lock:
            if (u, v) not in self.link_map: return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if 0 <= idx < self.L:
                self.B[idx] = min(self.B_cap, self.B[idx] + bw_req)
                self.link_ref_count[idx] = max(0, self.link_ref_count[idx] - 1)
                self.links['bandwidth'][(u, v)] = self.B[idx]
                return True
            return False

    # 兼容接口
    def consume_bandwidth(self, u, v, bw):
        return self.allocate_link_bandwidth(u, v, bw)

    def release_bandwidth(self, u, v, bw):
        return self.release_link_bandwidth(u, v, bw)

    def allocate_node_resources(self, n, c, m, vt=None):
        return self.update_resources(n, -c, -m, strict=True)

    def release_node_resources(self, n, c, m, vt=None):
        return self.update_resources(n, c, m, strict=False)

    def check_tree_bandwidth(self, tree, bw_req):
        """部署前带宽预检查（按物理链路聚合）"""
        from collections import defaultdict
        link_demand = defaultdict(float)

        # 1. 聚合需求到物理链路
        for edge, _ in tree.items():
            u, v = self._parse_edge(edge)
            if u is None or v is None:
                continue

            if (u, v) not in self.link_map:
                return False
            lid = self.link_map[(u, v)]
            idx = lid - 1
            if idx < 0 or idx >= self.L:
                return False

            link_demand[idx] += bw_req  # ✅ 累加（关键修复）

        # 2. 统一检查聚合后的需求
        for idx, demand in link_demand.items():
            if self.B[idx] < demand - 1e-5:
                return False

        return True

    # ========================================================================
    # 修复点 3: 增加 phase 参数，支持 Phase 1 宽松部署
    # ========================================================================
    def apply_tree_deployment(self, plan: Dict, request: Dict, phase: str = "phase2") -> bool:
        with self._lock:
            req_id = request.get('id', -1)

            # 1. 尝试部署节点 (任何阶段都需要节点资源扣除，以更新状态图)
            if not self.apply_deployment(plan, request):
                return False

            # --- Phase 1 特殊处理 ---
            # 如果是专家数据采集阶段，我们主要关注“节点是否能放进去”
            # 链路部分可能因为专家给的是逻辑树/抽象边而导致物理聚合失败
            # 因此，Phase 1 只要节点成功，就视为 Success，避免数据被丢弃
            if phase == "phase1":
                return True

            # --- Phase 2 严格部署 ---
            tree = plan.get('tree', {})
            bw_req = request.get('bw_origin', 0)

            # Step 1: 预检查整棵树 (使用聚合检查)
            if not self.check_tree_bandwidth(tree, bw_req):
                # 失败回滚：只回滚刚刚部署的节点
                self._rollback_ops([
                    v for v in self.vnf_instances if v.get('req_id') == req_id
                ])
                return False

            # Step 2: 实际分配 (先聚合，再扣除)
            link_demand = defaultdict(float)
            for edge, _ in tree.items():
                u, v = self._parse_edge(edge)
                if u is None or v is None: continue
                link_demand[(u, v)] += bw_req

            deployed_links = []
            success = True

            for (u, v), demand in link_demand.items():
                if not self.allocate_link_bandwidth(u, v, demand):
                    success = False
                    break
                deployed_links.append((u, v, demand))

            if not success:
                # 回滚已分配的链路
                for u, v, bw in deployed_links:
                    self.release_link_bandwidth(u, v, bw)
                # 回滚节点
                self._rollback_ops([
                    v for v in self.vnf_instances if v.get('req_id') == req_id
                ])
                return False

            # 3. 记录成功
            if req_id != -1:
                if req_id not in self.active_requests: self.active_requests[req_id] = {}
                self.active_requests[req_id]['links'] = deployed_links
            return True

    # ========================================================================
    # 辅助与部署基础逻辑
    # ========================================================================

    def apply_deployment(self, plan: Dict, request: Dict) -> bool:
        with self._lock:
            parsed_ops = self._parse_deployment_ops(plan, request)
            if parsed_ops is None: return False

            for op in parsed_ops:
                if not self.check_node_resources(op['node'], op['cpu'], op['mem']):
                    return False

            executed_ops = []
            req_id = request.get('id', -1)

            for op in parsed_ops:
                node, c, m, vt = op['node'], op['cpu'], op['mem'], op['vnf_type']
                if not self.update_resources(node, -c, -m, strict=True):
                    self._rollback_ops(executed_ops)
                    return False

                hvt_inc = False
                if 0 <= vt < self.K_vnf:
                    self.hvt_all[node, vt] += 1.0
                    hvt_inc = True

                record = {
                    'req_id': req_id, 'node': node, 'cpu': c, 'memory': m,
                    'vnf_type': vt, 'hvt_inc': hvt_inc
                }
                executed_ops.append(record)
                self.vnf_instances.append(record)
            return True

    def _parse_deployment_ops(self, plan, request):
        placement = plan.get('placement', {})
        vnf_types = request.get('vnf', [])
        cpu_reqs = request.get('cpu_origin', [])
        mem_reqs = request.get('memory_origin', [])

        ops = []
        for key, node_id in placement.items():
            try:
                v_idx = -1
                if 'vnf_' in key and '_type_' in key:
                    v_idx = int(key.split('_')[1])
                elif key.startswith('vnf_'):
                    v_idx = int(key.split('_')[1])
                elif key.isdigit():
                    v_idx = int(key)

                if v_idx >= 0 and v_idx < len(vnf_types):
                    v_type = vnf_types[v_idx]
                    c_req = cpu_reqs[v_idx]
                    m_req = mem_reqs[v_idx]
                    internal_node = self._normalize_node(node_id, from_external=True)
                    ops.append({'node': internal_node, 'cpu': c_req, 'mem': m_req, 'vnf_type': v_type})
            except:
                continue
        return ops

    def _rollback_ops(self, executed_ops):
        for op in reversed(executed_ops):
            self.update_resources(op['node'], op['cpu'], op['memory'], strict=False)
            if op['hvt_inc']:
                vt = op['vnf_type']
                self.hvt_all[op['node'], vt] = max(0, self.hvt_all[op['node'], vt] - 1.0)
            if op in self.vnf_instances:
                self.vnf_instances.remove(op)

    def remove_request(self, req_id: int) -> bool:
        with self._lock:
            # 允许在 active_requests 中不存在时尝试移除（为了处理 apply_deployment 后的回滚）
            has_records = req_id in self.active_requests or any(v.get('req_id') == req_id for v in self.vnf_instances)
            if not has_records:
                return False

            if req_id in self.active_requests:
                links = self.active_requests[req_id].get('links', [])
                for u, v, bw in links:
                    self.release_link_bandwidth(u, v, bw)
                del self.active_requests[req_id]

            to_remove = [v for v in self.vnf_instances if v.get('req_id') == req_id]
            self._rollback_ops(to_remove)
            return True

    def _normalize_node(self, node: int, from_external: bool = False) -> int:
        if from_external and self.node_index_base == 1: return node - 1
        return node

    def _parse_edge(self, edge):
        try:
            if isinstance(edge, str): return map(int, edge.strip("()").replace(" ", "").split(","))
            return int(edge[0]), int(edge[1])
        except:
            return None, None

    def _create_link_map(self, topo):
        lm = {}
        lid = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if topo[i, j] > 0:
                    lm[(i, j)] = lid;
                    lm[(j, i)] = lid
                    lid += 1
        return lm

    def _build_shortest_dist_matrix(self):
        try:
            G = nx.from_numpy_array(self.topo)
            return nx.floyd_warshall_numpy(G)
        except:
            return np.full((self.n, self.n), 999.0)

    def _build_edge_index(self):
        rows, cols = np.nonzero(self.topo)
        return np.array([rows, cols], dtype=np.int64)

    def get_gnn_state(self, current_request=None, **kwargs):
        x = self._build_node_features(current_request)
        req = self._encode_request(current_request) if current_request else np.zeros(self.request_dim)
        return {'x': x.astype(np.float32), 'edge_index': self.edge_index, 'request': req.astype(np.float32)}

    def _build_node_features(self, req):
        feats = []
        for i in range(self.n):
            base = [self.C[i] / self.C_cap, self.M[i] / self.M_cap, 1.0 if i in self.dc_nodes else 0.0, 0.0, 0.0, 0.0]
            feats.append(np.concatenate([base, self.hvt_all[i], [0, 0, 0]]))
        return np.array(feats)

    # ========================================================================
    # 修复点 2: _encode_request 填充实际特征
    # ========================================================================
    def _encode_request(self, req):
        vec = np.zeros(self.request_dim, dtype=np.float32)
        if not req:
            return vec

        # 1. Source (Normalized)
        vec[0] = req.get('source', 0) / self.n

        # 2. Destination Count (Normalized)
        dests = req.get('destinations') or req.get('dest', [])
        vec[1] = len(dests) / self.n

        # 3. Bandwidth (Normalized)
        vec[2] = req.get('bw_origin', 0) / self.B_cap

        # 4. CPU & Memory (Avg Normalized)
        cpus = req.get('cpu_origin', [])
        mems = req.get('memory_origin', [])

        # 避免除以零
        count = len(cpus) if len(cpus) > 0 else 1
        vec[3] = sum(cpus) / (self.C_cap * count)
        vec[4] = sum(mems) / (self.M_cap * count)

        # 剩余维度保留为0或按需扩展
        return vec