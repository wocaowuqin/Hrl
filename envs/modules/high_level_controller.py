import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class HighLevelController:
    """
    🎮 高层交互控制器 - 收敛优化版
    优化1: mask严格可行性（不全开兜底）
    优化3: 高层步惩罚-0.1
    优化5: Feature[9]置0减噪
    """

    def __init__(self, env):
        self.env = env
        logger.info("✅ HighLevelController initialized (收敛优化版)")
        self._pending_dist_cache = {}
        self._last_dests_update = -1

    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        self.env.last_high_action_idx = high_action_idx
        self.env.current_subgoal_node = target_node_id
        self.env.subgoal_step_count = 0

        # [P1修复] 只记录起点，不强制瞬移
        # 如果 start_node_id 与实际物理位置不符，以实际位置为准
        actual_location = self.env.current_node_location
        if start_node_id is not None:
            if int(start_node_id) != actual_location:
                logger.warning(f"⚠️ [High] Agent预测起点({start_node_id})≠实际位置({actual_location})，"
                               f"以实际位置为准，不执行瞬移")
            else:
                logger.info(f"📍 [High] Agent起点确认: {start_node_id}")

        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            self.env.current_vnf_to_deploy = self.env.next_vnf_idx

            if self.env.next_vnf_idx < len(vnf_list):
                self.env.current_phase = 'vnf_deployment'
                self.env.current_deployment_target = target_node_id
                self.env.current_target_node = None
                #logger.info(f"🎯 [Env] VNF部署阶段: VNF[{self.env.next_vnf_idx}] → 节点{target_node_id}")
            else:
                self.env.current_phase = 'destination_connection'
                self.env.current_target_node = target_node_id
                self.env.current_deployment_target = None
                #logger.info(f"🎯 [Env] 目的地连接阶段: 目标节点{target_node_id}")
        #else:
            #logger.warning("⚠️ 设定目标时没有活跃请求")

        return self.get_high_level_state_graph()

    # ==================================================================
    # 🔥 优化1 + 优化3: step_high_level
    # ==================================================================
    def step_high_level(self, action_idx):
        """
        🔥 [收敛优化版 + P1修复]
        优化1: 开头检查mask全0 → Episode失败
        优化3: 标准路径加 -0.5 步惩罚
        P1修复: 使用 current_subgoal_node 而非 action_idx 作为目标节点
        """
        # 🔥🔥🔥 [优化1] 无可行动作 → 直接失败
        mask = self.get_high_level_action_mask()
        if np.sum(mask) == 0:
            logger.warning("❌ [High] 无可行高层动作，Episode失败")
            return None, -10.0, True, False, {'no_valid_action': True}

        # [P1修复] 使用 set_high_level_goal 中已正确解析的目标节点
        # 原代码: target_node = int(action_idx)  ← action_idx可能是Agent输出索引，非真实节点ID
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
                return None, 20.0, True, False, {'all_done': True}
            else:
                logger.warning(f"🔄 [High] 目标 {target_node} 已完成 -> Truncated")
                return None, 5.0, False, True, {
                    'subgoal_completed': True,
                    'warning': 'selected_completed_target'
                }

        self.env.current_goal_node = target_node

        # 🔥🔥🔥 [优化3] 步惩罚，防止频繁换目标
        step_penalty = -0.5

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

    # ==================================================================
    # 🔥 优化1: get_high_level_action_mask (严格失败)
    # ==================================================================
    def get_high_level_action_mask(self):
        """
        🔥 [收敛优化版] 严格可行性控制
        优化1: 资源不足时返回全0，不再全开兜底
        """
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

            if hasattr(self.env, 'dc_nodes'):
                for node in self.env.dc_nodes:
                    if 0 <= node < n:
                        avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                        avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                        if avail_cpu >= req_cpu and avail_mem >= req_mem:
                            mask[node] = 1.0

            # 🔥🔥🔥 [优化1] 严格失败：不再全开兜底
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
                # [P0修复] 所有目的地已连接，不返回全0死锁
                # 检查VNF是否也完成 → 如果是，开放 source 触发全局完成检测
                vnf_done = current_vnf_idx >= len(vnf_list)
                if vnf_done:
                    logger.info("🎉 [Mask] 所有任务已完成，开放source节点触发Episode结束")
                    source = self.env.current_request.get('source', 0)
                    if source is not None and 0 <= int(source) < n:
                        mask[int(source)] = 1.0
                    if np.sum(mask) == 0:
                        # source无效，开放任意已连接节点
                        for d in connected_set:
                            if 0 <= d < n:
                                mask[d] = 1.0
                                break
                    return mask
                else:
                    # VNF 未完成但目的地全连接（异常状态）
                    logger.warning("⚠️ [Mask] 目的地全连接但VNF未完成，返回全0")
                    return np.zeros(n, dtype=np.float32)

            for node in true_pending:
                if 0 <= node < n:
                    mask[node] = 1.0

            for done_node in connected_set:
                if 0 <= done_node < n:
                    mask[done_node] = 0.0

        return mask

    # ==================================================================
    # 优化5: get_high_level_state_graph (Feature[9]置0)
    # ==================================================================
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

        # VNF负载统计
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

        # =============================
        # 构建节点特征 [N, 11]
        # =============================
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

            # [0-1] 资源
            features = [norm_cpu, norm_mem]

            # [2] 源节点
            features.append(1.0 if node == source else 0.0)

            # [3] Pending Dest
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

            # [4] 已连接
            features.append(1.0 if is_connected else 0.0)

            # [5] 在树上
            features.append(1.0 if node in nodes_on_tree else 0.0)

            # [6] 资源匹配度
            if is_vnf_phase and req_cpu > 0:
                cpu_match = min(avail_cpu / max(req_cpu, 0.1), 1.0)
                mem_match = min(avail_mem / max(req_mem, 0.1), 1.0)
                match_score = 0.7 * cpu_match + 0.3 * mem_match
                if node in dc_nodes:
                    match_score = min(match_score + 0.1, 1.0)
            else:
                match_score = 0.0
            features.append(match_score)

            # [7] 度数
            try:
                degree = len(self.env.resource_mgr.get_neighbors(node))
                norm_degree = min(degree / 10.0, 1.0)
            except:
                norm_degree = 0.5
            features.append(norm_degree)

            # [8] 历史负载
            features.append(min(node_vnf_counts[node] / 5.0, 1.0))

            # 🔥🔥🔥 [优化5] Feature[9] 置0，减少噪声
            features.append(0.0)

            # [10] 阶段指导（带负载惩罚）
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

            if is_dest:
                status_str = "✅已完成" if is_connected else "❌未完成"
                # logger.info(f"🧐 [Node {node}] {status_str} | "
                #             f"Feat[3]:{features[3]:.1f} | "
                #             f"Feat[10]:{features[10]:.1f} | "
                #             f"CPU:{features[0]:.2f}")
            x.append(features)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        # 边特征
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

        # 全局特征
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

    # ==================================================================
    # 辅助方法（不变）
    # ==================================================================
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