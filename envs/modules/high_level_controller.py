import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class HighLevelController:
    """
    🎮 高层交互控制器 - HQDQN 对齐优化版
    """

    def __init__(self, env):
        self.env = env
        logger.info("✅ HighLevelController initialized (HQDQN对齐版)")
        self._pending_dist_cache = {}
        self._last_dests_update = -1

    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        self.env.last_high_action_idx = high_action_idx
        target_node_id = int(target_node_id)  # 确保是int

        # ============================================================
        # 🔥 [防御] VNF阶段: target必须是DC节点；在此处做最后一道拦截
        # ============================================================
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
            if current_vnf_idx < len(vnf_list):
                dc_nodes = getattr(self.env, 'dc_nodes', set())
                if dc_nodes and target_node_id not in dc_nodes:
                    # 从mask中找第一个合法DC节点替换
                    mask = self.get_high_level_action_mask()
                    dc_fallback = next(
                        (n for n in range(self.env.n) if mask[n] > 0 and n in dc_nodes),
                        None
                    )
                    if dc_fallback is not None:
                        logger.warning(
                            f"🛡️ [High.set_goal] VNF阶段目标 {target_node_id} 非DC节点，"
                            f"强制替换为 {dc_fallback}"
                        )
                        target_node_id = dc_fallback
                    else:
                        logger.error(f"❌ [High.set_goal] 无可用DC节点可替换目标 {target_node_id}")

        self.env.current_subgoal_node = target_node_id
        self.env.subgoal_step_count = 0

        actual_location = self.env.current_node_location
        if start_node_id is not None:
            start_node_id = int(start_node_id)
            if start_node_id != actual_location:
                # ── [FIX SFC] dest阶段允许合法瞬移回 last_vnf ────────────────────────────────────
                # 原来一律拒绝不一致的 start_node，导致 Coordinator 设置的 last_vnf 回位被忽略，
                # 每个目的地从不同位置出发，SFC主干断裂，可视化里 VNF 节点缺失。
                # 修复：若 start_node == chain_nodes[-1]，说明是 Coordinator 主动回位，执行合法瞬移。
                _phase = getattr(self.env, 'current_phase', None)
                _chain = getattr(self.env, 'chain_nodes', [])
                if _phase == 'destination_connection' and _chain and start_node_id == _chain[-1]:
                    self.env.current_node_location = start_node_id
                    logger.debug(f"🔄 [High] dest阶段合法瞬移: {actual_location}→last_vnf={start_node_id}")
                else:
                    logger.warning(f"⚠️ [High] Agent预测起点({start_node_id})≠实际位置({actual_location})，"
                                   f"以实际位置为准，不执行瞬移")
                # ─────────────────────────────────────────────────────────────────
            else:
                logger.debug(f"📍 [High] Agent起点确认: {start_node_id}")

        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            self.env.current_vnf_to_deploy = self.env.next_vnf_idx

            if self.env.next_vnf_idx < len(vnf_list):
                self.env.current_phase = 'vnf_deployment'
                self.env.current_deployment_target = target_node_id
                self.env.current_target_node = None
            else:
                self.env.current_phase = 'destination_connection'
                self.env.current_target_node = target_node_id
                self.env.current_deployment_target = None

        return self.get_high_level_state_graph()

    def step_high_level(self, action_idx):
        # ================================================================
        # 🔥 [FIX] mask计算前先同步phase状态，防止phase尚未切换就用旧mask判断
        # ================================================================
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
            # 确保phase与vnf进度一致（防止低层完成VNF后phase未同步）
            if current_vnf_idx >= len(vnf_list):
                if getattr(self.env, 'current_phase', None) == 'vnf_deployment':
                    self.env.current_phase = 'destination_connection'
                    self.env.current_deployment_target = None
                    logger.info(f"🔄 [High.step] 检测到VNF已全完成，phase强制切换→destination_connection")

        mask = self.get_high_level_action_mask()
        if np.sum(mask) == 0:
            phase = getattr(self.env, 'current_phase', 'unknown')
            vnf_idx = getattr(self.env, 'next_vnf_idx', -1)
            pending = []
            if self.env.current_request:
                all_d = set(self.env.current_request.get('dest', []))
                conn_d = self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set()
                pending = list(all_d - conn_d)
            logger.warning(
                f"❌ [High] 无可行高层动作 | phase={phase} | vnf_idx={vnf_idx} | pending_dests={pending}"
            )
            return None, -10.0, True, False, {'no_valid_action': True, 'phase': phase}

        target_node = getattr(self.env, 'current_subgoal_node', int(action_idx))
        target_node = int(target_node)
        self.current_high_action = target_node

        connected = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            try:
                connected = {int(x) for x in self.env.current_tree.get('connected_dests', set())}
            except:
                pass

        all_dests = set()
        if self.env.current_request:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
            except:
                pass

        if target_node in connected:
            if all_dests.issubset(connected):
                logger.info("🎉 [High] 所有目的地已连接，Episode完美结束！")
                reward = 20.0
                if hasattr(self.env, 'low_level_controller') and hasattr(self.env.low_level_controller,
                                                                         '_calculate_tree_metrics'):
                    metrics = self.env.low_level_controller._calculate_tree_metrics()
                    reward += -3.0 * metrics.get('redundancy', 0.0)
                    reward += -1.5 * metrics.get('tree_n_edges', 0.0)
                return None, reward, True, False, {'all_done': True}
            else:
                logger.warning(f"🔄 [High] 目标 {target_node} 已完成 -> Truncated")
                return None, 5.0, False, True, {
                    'subgoal_completed': True,
                    'warning': 'selected_completed_target'
                }

        step_penalty = -0.1

        return None, step_penalty, False, False, {
            'target_node': target_node,
            'status': 'executing'
        }

    def _is_all_completed(self):
        if not self.env.current_request: return True
        vnf_list = self.env.current_request.get('vnf', [])
        if self.env.next_vnf_idx < len(vnf_list): return False
        connected = self.env.current_tree.get('connected_dests', set())
        all_dests = set(self.env.current_request.get('dest', []))
        if not all_dests.issubset(connected): return False
        return True

    def get_high_level_action_mask(self):
        n = self.env.n
        mask = np.zeros(n, dtype=np.float32)

        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return np.ones(n, dtype=np.float32)

        vnf_list = self.env.current_request.get('vnf', [])
        current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)

        # 阶段 1: VNF 部署
        if current_vnf_idx < len(vnf_list):
            cpu_list = self.env.current_request.get('cpu_origin', []) or \
                       self.env.current_request.get('vnf_cpu', [])
            mem_list = self.env.current_request.get('memory_origin', []) or \
                       self.env.current_request.get('vnf_mem', [])
            req_cpu = float(cpu_list[current_vnf_idx]) if current_vnf_idx < len(cpu_list) else 10.0
            req_mem = float(mem_list[current_vnf_idx]) if current_vnf_idx < len(mem_list) else 10.0

            # ── [FIX SFC] VNF部署节点排除：source / dest / 已部署节点
            # VNF不能部署在source节点（会造成Source=VNF重叠）
            # VNF不能部署在dest节点（会造成VNF=Dest重叠，SFC语义错误）
            # VNF不能重复部署到同一节点
            _source = self.env.current_request.get('source', -1) if self.env.current_request else -1
            _dests = set(int(d) for d in self.env.current_request.get('dest', [])) if self.env.current_request else set()
            _already = set(getattr(self.env, 'chain_nodes', []))

            if hasattr(self.env, 'dc_nodes'):
                _llc = getattr(self.env, 'low_level_controller', None)
                # 前进约束：第k个VNF离source的距离 必须 > 第k-1个VNF离source的距离
                # 保证spine沿 source→dest 方向单调推进，不回绕
                _prev_vnf = (list(_already)[-1] if _already
                             else (_source if _source != -1 else None))
                _dist_prev_from_src = (_llc._get_hop_distance(_source, _prev_vnf)
                                       if _llc and _prev_vnf is not None and _source != -1
                                       else 0)
                _strict_mask = np.zeros(n, dtype=np.float32)
                _loose_mask  = np.zeros(n, dtype=np.float32)
                for node in self.env.dc_nodes:
                    if 0 <= node < n:
                        if node == _source:  continue
                        if node in _dests:   continue
                        if node in _already: continue
                        avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                        avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                        if avail_cpu >= req_cpu and avail_mem >= req_mem:
                            _dist_node = (_llc._get_hop_distance(_source, node)
                                          if _llc and _source != -1 else 0)
                            # 严格前进：离source更远
                            if _dist_node > _dist_prev_from_src:
                                _strict_mask[node] = 1.0
                            # 宽松：至少一样远（fallback用）
                            if _dist_node >= _dist_prev_from_src:
                                _loose_mask[node] = 1.0
                # 优先严格前进；若无节点满足则退回宽松；再无则不限制
                if np.sum(_strict_mask) > 0:
                    mask = _strict_mask
                elif np.sum(_loose_mask) > 0:
                    mask = _loose_mask
                else:
                    # 极端情况：全部DC都在source附近，放开限制
                    for node in self.env.dc_nodes:
                        if 0 <= node < n and node != _source and node not in _dests and node not in _already:
                            avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                            avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                            if avail_cpu >= req_cpu and avail_mem >= req_mem:
                                mask[node] = 1.0

            if np.sum(mask) == 0:
                logger.warning("⚠️ [Mask] 所有DC节点资源不足，返回全0")
                return np.zeros(n, dtype=np.float32)
            else:
                open_dc = [i for i in range(n) if mask[i] > 0]
                logger.debug(f"[Mask] VNF阶段开放DC节点: {open_dc}")

        # 阶段 2: 目的地连接
        else:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
                connected_set = set()
                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_set = {int(x) for x in self.env.current_tree.get('connected_dests', set())}
            except:
                all_dests, connected_set = set(), set()

            true_pending = list(all_dests - connected_set)
            if not true_pending:
                return np.zeros(n, dtype=np.float32)

            # 目的地阶段：纯拓扑连通性判断，带宽约束由低层执行时处理
            llc = getattr(self.env, 'low_level_controller', None)
            last_vnf = self.env.chain_nodes[-1] if getattr(self.env, 'chain_nodes', []) else None
            for node in true_pending:
                if 0 <= node < n:
                    if llc is not None and last_vnf is not None:
                        if llc._get_hop_distance(last_vnf, node) < 9999:
                            mask[node] = 1.0
                    else:
                        mask[node] = 1.0

            for done_node in connected_set:
                if 0 <= done_node < n:
                    mask[done_node] = 0.0

        return mask

    def get_high_level_state_graph(self):
        n = self.env.n

        if not self.env.current_request:
            return Data(
                x=torch.zeros((n, 11), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                global_attr=torch.zeros((1, 5), dtype=torch.float32)
            )

        req = self.env.current_request
        vnf_list = req.get('vnf', [])
        source = req.get('source')
        dests = req.get('dest', [])

        connected_dests = self.env.current_tree.get('connected_dests', set())
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        is_vnf_phase = next_vnf_idx < len(vnf_list)

        req_cpu, req_mem = 0.0, 0.0
        if is_vnf_phase:
            cpu_list = req.get('cpu_origin') or req.get('vnf_cpu', [])
            mem_list = req.get('memory_origin') or req.get('vnf_mem', [])
            if next_vnf_idx < len(cpu_list):
                req_cpu = float(cpu_list[next_vnf_idx])
                req_mem = float(mem_list[next_vnf_idx]) if next_vnf_idx < len(mem_list) else 1.0

        node_vnf_counts = [0] * n
        placement = self.env.current_tree.get('placement', {})
        for placement_key, info in placement.items():
            node_id = None
            if isinstance(placement_key, tuple) and len(placement_key) >= 1:
                node_id = placement_key[0]
            elif isinstance(info, dict):
                node_id = info.get('node')
            if node_id is not None and 0 <= node_id < n:
                node_vnf_counts[node_id] += 1

        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'hvt_all'):
            hvt = self.env.resource_mgr.hvt_all
            for node_id in range(n):
                global_count = int(np.sum(hvt[node_id]))
                node_vnf_counts[node_id] = max(node_vnf_counts[node_id], global_count)

        dc_nodes = getattr(self.env, 'dc_nodes', set())

        x = []
        for node in range(n):
            try:
                avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                norm_cpu = avail_cpu / 100.0
                norm_mem = avail_mem / 100.0
            except:
                norm_cpu, norm_mem = 0.5, 0.5
                avail_cpu, avail_mem = 50.0, 50.0

            features = [norm_cpu, norm_mem]
            features.append(1.0 if node == source else 0.0)

            is_dest = node in dests
            is_connected = node in connected_dests
            if is_dest:
                if is_connected:
                    is_pending = -1.0
                else:
                    is_pending = 0.5 if is_vnf_phase else 2.0
            else:
                is_pending = 0.0
            features.append(is_pending)

            features.append(1.0 if is_connected else 0.0)
            features.append(1.0 if node in nodes_on_tree else 0.0)

            if is_vnf_phase and req_cpu > 0:
                cpu_match = min(avail_cpu / max(req_cpu, 0.1), 1.0)
                mem_match = min(avail_mem / max(req_mem, 0.1), 1.0)
                match_score = 0.7 * cpu_match + 0.3 * mem_match
                if node in dc_nodes:
                    match_score = min(match_score + 0.1, 1.0)
            else:
                match_score = 0.0
            features.append(match_score)

            try:
                degree = len(self.env.resource_mgr.get_neighbors(node))
                norm_degree = min(degree / 10.0, 1.0)
            except:
                norm_degree = 0.5
            features.append(norm_degree)

            features.append(min(node_vnf_counts[node] / 5.0, 1.0))
            features.append(0.0)

            if is_vnf_phase:
                if node in dc_nodes and avail_cpu >= req_cpu and avail_mem >= req_mem:
                    load_penalty = node_vnf_counts[node] * 0.3
                    phase_guide = max(1.5 - load_penalty, 0.1)
                elif node in dc_nodes:
                    phase_guide = -1.0
                else:
                    phase_guide = -0.3
            else:
                if is_dest:
                    if is_connected:
                        phase_guide = -3.0
                    else:
                        phase_guide = 3.0
                elif node == source:
                    phase_guide = 0.5
                else:
                    pending_dests = [d for d in dests if d not in connected_dests]
                    if pending_dests:
                        min_dist = float('inf')
                        for dest in pending_dests:
                            if node == dest:
                                min_dist = 0
                                break
                            has_direct_link = False
                            if hasattr(self.env, 'resource_mgr'):
                                try:
                                    neighbors = self.env.resource_mgr.get_neighbors(node)
                                    has_direct_link = dest in neighbors
                                except:
                                    if hasattr(self.env, 'topology'):
                                        has_direct_link = self.env.topology[node][dest] > 0
                            if has_direct_link:
                                dist = 1
                            else:
                                dist = abs(node - dest) + 2
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist == 0:
                            phase_guide = 2.5
                        elif min_dist == 1:
                            phase_guide = 1.0
                        elif min_dist <= 3:
                            phase_guide = 0.0
                        else:
                            phase_guide = -0.5
                    else:
                        phase_guide = -1.0
            features.append(phase_guide)
            x.append(features)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        edge_index_list = []
        edge_attr_list = []

        for u in range(n):
            try:
                if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                    neighbors = self.env.resource_mgr.get_neighbors(u)
                else:
                    neighbors = []
                    if hasattr(self.env, 'topology'):
                        for v in range(n):
                            if v != u and self.env.topology[u][v] > 0:
                                neighbors.append(v)
                    else:
                        continue

                for v in neighbors:
                    if u < v:
                        edge_index_list.append([u, v])
                        edge_index_list.append([v, u])
                        try:
                            if hasattr(self.env.resource_mgr, 'pool'):
                                pool = self.env.resource_mgr.pool
                                available_bw = pool.get_available_bandwidth(u, v)
                                norm_bw = min(available_bw / 100.0, 1.0)

                                is_in_tree = 0.0
                                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                                    tree = self.env.current_tree.get('tree', {})
                                    edge_key = (min(u, v), max(u, v))
                                    if edge_key in tree:
                                        is_in_tree = 1.0

                                edge_attr_list.append([norm_bw, is_in_tree])
                                edge_attr_list.append([norm_bw, is_in_tree])
                            else:
                                edge_attr_list.append([0.5, 0.0])
                                edge_attr_list.append([0.5, 0.0])
                        except:
                            edge_attr_list.append([0.5, 0.0])
                            edge_attr_list.append([0.5, 0.0])
            except:
                continue

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        bw_req = req.get('bw_origin', 0.0)
        norm_bw_req = min(bw_req / 10.0, 1.0)
        vnf_progress = next_vnf_idx / max(1, len(vnf_list))
        dest_progress = len(connected_dests) / max(1, len(dests))
        phase_feat = 0.0 if is_vnf_phase else 1.0
        total_avail_cpu = sum([self.env.resource_mgr.pool.get_available_cpu(i) for i in range(n)])
        resource_tension = 1.0 - (total_avail_cpu / (n * 100.0))

        global_attr = torch.tensor([[
            norm_bw_req, vnf_progress, dest_progress, phase_feat, resource_tension
        ]], dtype=torch.float32)

        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, global_attr=global_attr)

    def _validate_start_node(self, start_node, target_node):
        if not self._is_valid_node(start_node):
            return False, f"无效节点ID: {start_node}"
        if not self._is_valid_node(target_node):
            return False, f"无效目标节点ID: {target_node}"
        if self.env.current_phase != 'vnf_deployment' and start_node == target_node:
            return False, f"起点和终点相同: {start_node}"
        if self.env.current_phase == 'destination_connection':
            nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
            if start_node not in nodes_on_tree:
                source = self.env.current_request.get('source', 0)
                try:
                    hop = self._get_hop_distance(source, start_node)
                    if hop >= 9999:
                        return False, f"起点{start_node}不可达"
                except:
                    pass
        return True, "有效"

    def _is_valid_node(self, node):
        try:
            node = int(node)
            if node < 0 or node >= self.env.n:
                return False
            if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                try:
                    _ = self.env.resource_mgr.get_neighbors(node)
                    return True
                except:
                    return False
            return True
        except:
            return False

    def _get_hop_distance(self, node1, node2):
        if node1 == node2: return 0
        if not self._is_valid_node(node1) or not self._is_valid_node(node2): return 9999
        try:
            G = nx.Graph()
            for u in range(self.env.n):
                try:
                    if hasattr(self.env, 'resource_mgr'):
                        neighbors = self.env.resource_mgr.get_neighbors(u)
                    elif hasattr(self.env, 'topology'):
                        neighbors = [v for v in range(self.env.n) if v != u and self.env.topology[u][v] > 0]
                    else:
                        continue
                    for v in neighbors:
                        if self._is_valid_node(v):
                            G.add_edge(u, v)
                except:
                    continue
            if G.has_node(node1) and G.has_node(node2):
                return nx.shortest_path_length(G, node1, node2)
            return 9999
        except:
            return 9999

    def _get_total_vnf_progress(self):
        if not self.env.current_request: return 0
        vnf_list = self.env.current_request.get('vnf', [])
        return min(getattr(self.env, 'next_vnf_idx', 0), len(vnf_list))

    def _is_all_tasks_completed(self):
        if not self.env.current_request:
            return True, "无请求"
        vnf_list = self.env.current_request.get('vnf', [])
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_done = next_vnf_idx >= len(vnf_list)
        dests = set(self.env.current_request.get('dest', []))
        connected = set(self.env.current_tree.get('connected_dests', set()))
        dests_done = dests.issubset(connected)
        if vnf_done and dests_done:
            return True, "所有任务完成"
        elif vnf_done:
            return False, f"VNF完成，还有{len(dests) - len(connected)}个目的地待连接"
        elif dests_done:
            return False, f"目的地完成，还有{len(vnf_list) - next_vnf_idx}个VNF待部署"
        else:
            return False, f"VNF:{next_vnf_idx}/{len(vnf_list)}，Dest:{len(connected)}/{len(dests)}"