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
                    # [Bug4修复] set_high_level_goal 调用时 phase/next_vnf_idx 可能尚未
                    # 切换到最新状态（特别是低层刚完成VNF但高层还未同步时），
                    # 直接调 get_high_level_action_mask 可能在旧phase下产生错误mask。
                    # 修复：先强制同步phase状态，再计算mask，保证mask在正确context下运行。
                    if current_vnf_idx >= len(vnf_list):
                        self.env.current_phase = 'destination_connection'
                        self.env.current_deployment_target = None
                    else:
                        self.env.current_phase = 'vnf_deployment'
                    mask = self.get_high_level_action_mask()
                    # [Bug5修复] 按离当前位置最近的合法DC替换，而非按节点编号取第一个
                    # 原来从小到大取第一个会导致所有被拦截的VNF都堆在节点0
                    _cur_loc = getattr(self.env, 'current_node_location', 0)
                    _llc_ref = getattr(self.env, 'low_level_controller', None)
                    _best_fallback = None
                    _best_dist = 9999
                    for _n in range(self.env.n):
                        if mask[_n] > 0 and _n in dc_nodes:
                            _d = _llc_ref._get_hop_distance(_cur_loc, _n) if _llc_ref else 9999
                            if _d < _best_dist:
                                _best_dist = _d
                                _best_fallback = _n
                    if _best_fallback is not None:
                        logger.warning(
                            f"🛡️ [High.set_goal] VNF阶段目标 {target_node_id} 非DC节点，"
                            f"强制替换为最近合法DC {_best_fallback}（hop={_best_dist}）"
                        )
                        target_node_id = _best_fallback
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

        # [Bug1修复] current_subgoal_node 是上一轮 set_high_level_goal 写入的值，
        # step_high_level 调用时 set_high_level_goal 还没有执行，
        # 读到的是上一轮目标 → target_node in connected 判断完全错位。
        # 修复：直接用本次 action_idx 作为 target_node。
        target_node = int(action_idx)
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
            # 同一(node, vnf_idx)不能重复部署，但同节点不同vnf_idx是合法的
            _source = self.env.current_request.get('source', -1) if self.env.current_request else -1
            _dests = set(int(d) for d in self.env.current_request.get('dest', [])) if self.env.current_request else set()
            # [Bug修复] 原来 _already = set(chain_nodes) 做节点级排除，
            # 导致同一节点部署第二个不同VNF类型时被错误封禁。
            # placement key 是 (node, vnf_idx)，合法情况：同一DC节点可承载多个不同VNF。
            # 修复：只排除"当前这个vnf_idx已经部署在该节点"的节点，
            # 允许同节点部署不同vnf_idx。
            _placement = self.env.current_tree.get('placement', {}) if self.env.current_tree else {}
            _already = {key[0] for key in _placement
                        if isinstance(key, tuple) and len(key) >= 2
                        and key[1] == current_vnf_idx}

            # [唯一性约束] 同节点不允许部署相同VNF类型
            # 预计算：每个节点上已部署的vnf_type集合
            _current_vnf_type = vnf_list[current_vnf_idx]
            _node_vnf_types = {}  # node -> set of deployed vnf_types
            for (pnode, _), pinfo in _placement.items():
                _node_vnf_types.setdefault(pnode, set()).add(pinfo.get('vnf_type'))
            # [Bug9修复] set迭代顺序不确定，直接用有序的chain_nodes列表取最后一个
            _chain_nodes_ordered = getattr(self.env, 'chain_nodes', [])

            if hasattr(self.env, 'dc_nodes'):
                _llc = getattr(self.env, 'low_level_controller', None)
                # ── [方案B] SFC顺序约束 ──────────────────────────────────────────────
                # 约束：dist(candidate→nearest_dest) < dist(prev_vnf→nearest_dest)
                # 语义：candidate必须比prev_vnf更靠近dest，保证prev_vnf→candidate→dest物理顺序
                # ──────────────────────────────────────────────────────────────────────
                # [Bug9修复] 用_chain_nodes_ordered[-1]而非list(set)[-1]
                _prev_vnf = (_chain_nodes_ordered[-1] if _chain_nodes_ordered
                             else (_source if _source != -1 else None))

                # [Bug2修复] _llc不可用时无法计算顺序约束，直接走纯资源约束 fallback
                # 第一个VNF（_chain_nodes_ordered为空）且source有效时：prev_vnf=source，
                # 顺序约束正常工作；source==-1时跳过顺序约束只做资源过滤
                _can_use_order_constraint = (_llc is not None and _prev_vnf is not None and bool(_dests))

                if _can_use_order_constraint:
                    # 一次性预计算prev_vnf到各dest的距离，避免mask循环里反复重建图（Bug7前置优化）
                    _dist_prev_to_dests = {}
                    _nearest_dest_dist_prev = 9999
                    for _d in _dests:
                        _dd = _llc._get_hop_distance(_prev_vnf, _d)
                        _dist_prev_to_dests[_d] = _dd
                        if _dd < _nearest_dest_dist_prev:
                            _nearest_dest_dist_prev = _dd
                else:
                    _dist_prev_to_dests = {}
                    _nearest_dest_dist_prev = 9999

                _strict_mask = np.zeros(n, dtype=np.float32)
                _loose_mask  = np.zeros(n, dtype=np.float32)
                for node in self.env.dc_nodes:
                    if 0 <= node < n:
                        if node == _source:  continue
                        if node in _dests:   continue
                        if node in _already: continue
                        # [唯一性约束] 该节点已有相同vnf_type，跳过
                        if _current_vnf_type in _node_vnf_types.get(node, set()): continue
                        avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                        avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                        if avail_cpu >= req_cpu and avail_mem >= req_mem:
                            if not _can_use_order_constraint:
                                # _llc不可用：跳过顺序约束，只保留资源过滤
                                _loose_mask[node] = 1.0
                                continue

                            _prev_to_node = _llc._get_hop_distance(_prev_vnf, node)

                            # 计算candidate到最近dest的距离
                            _nearest_dest_dist_node = 9999
                            for _d in _dests:
                                _dd = _llc._get_hop_distance(node, _d)
                                if _dd < _nearest_dest_dist_node:
                                    _nearest_dest_dist_node = _dd

                            # 严格约束：candidate在prev_vnf→某dest的最短路径上
                            _on_shortest_path = False
                            for _d in _dests:
                                _d_prev = _dist_prev_to_dests.get(_d, 9999)
                                _d_node = _llc._get_hop_distance(node, _d)
                                if _prev_to_node + _d_node == _d_prev:
                                    _on_shortest_path = True
                                    break

                            if _on_shortest_path:
                                _strict_mask[node] = 1.0
                            # 宽松fallback：至少比prev_vnf更靠近dest
                            if _nearest_dest_dist_node < _nearest_dest_dist_prev:
                                _loose_mask[node] = 1.0

                # 优先严格前进；若无节点满足则退回宽松；再无则纯资源约束
                if np.sum(_strict_mask) > 0:
                    mask = _strict_mask
                elif np.sum(_loose_mask) > 0:
                    mask = _loose_mask
                else:
                    # 极端情况：放开位置限制，只保留资源约束
                    for node in self.env.dc_nodes:
                        if 0 <= node < n and node != _source and node not in _dests and node not in _already:
                            if _current_vnf_type in _node_vnf_types.get(node, set()): continue
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
        # [Bug3修复] 高层与低层共用同一encoder（main.py: high_agent=agent=low_agent），
        # 高层状态图节点特征维度必须与encoder的node_dim一致，否则维度崩溃。
        # 动态从env.config读取，与low_level_controller._build_node_features保持同步。
        _node_feat_dim = 11  # fallback：保留原有维度（独立encoder场景）
        if hasattr(self.env, 'config'):
            _node_feat_dim = self.env.config.get('gnn', {}).get('node_feat_dim', 11)

        if not self.env.current_request:
            return Data(
                x=torch.zeros((n, _node_feat_dim), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 5), dtype=torch.float32),
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
                # [Bug10修复] 用实际cap归一化，与low_level_controller._build_node_features一致
                # M_cap=80时除100会导致值>1，且与低层特征分布不一致
                _c_cap = max(1.0, float(getattr(self.env.resource_mgr, 'C_cap', 100.0)))
                _m_cap = max(1.0, float(getattr(self.env.resource_mgr, 'M_cap', 100.0)))
                norm_cpu = min(avail_cpu / _c_cap, 1.0)
                norm_mem = min(avail_mem / _m_cap, 1.0)
            except:
                norm_cpu, norm_mem = 0.5, 0.5
                avail_cpu, avail_mem = 50.0, 50.0
                _c_cap, _m_cap = 100.0, 100.0

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
                        # [Bug6修复] abs(node-dest)+2 在US Backbone拓扑里毫无意义（节点编号≠物理距离）
                        # 改用llc._get_hop_distance计算真实跳数，llc不可用时退化到9999
                        _llc_ref = getattr(self.env, 'low_level_controller', None)
                        min_dist = float('inf')
                        for dest in pending_dests:
                            if _llc_ref is not None:
                                dist = _llc_ref._get_hop_distance(node, dest)
                            else:
                                dist = 9999
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist == 0:
                            phase_guide = 2.5
                        elif min_dist == 1:
                            phase_guide = 1.5
                        elif min_dist <= 3:
                            phase_guide = 0.5
                        elif min_dist <= 6:
                            phase_guide = 0.0
                        else:
                            phase_guide = -0.5
                    else:
                        phase_guide = -1.0
            features.append(phase_guide)
            # 当前高层特征共11维，低层encoder期望 _node_feat_dim（=24）维
            # [Bug3修复] 补充13个语义有意义的维度，与低层特征对齐：
            # dim11: is_dc（是否DC节点）
            # dim12: is_current（是否当前agent位置）
            # dim13: hop_to_target（归一化跳数，到当前高层subgoal目标）
            # dim14-16: hvt[0..2]（全局历史访问次数，归一化）
            # dim17: on_tree（已在dim5，重复置0保持兼容，低层on_tree语义相同）
            # dim18: connected_dest（是否connected的dest，已在dim4）
            # dim19: is_target（是否当前目标，高层subgoal）
            # dim20: vnf_depth（VNF链进度 next_vnf_idx / max_vnf）
            # dim21: progress（dest连接进度）
            # dim22: phase_flag（0=vnf阶段 1=dest阶段，已在global_attr）
            # dim23: hop_to_tree（到最近树上节点的跳数，归一化）
            # 注：高层不关心邻边BW（dim21-23在低层），用进度/距离替代更有意义
            features.append(1.0 if node in dc_nodes else 0.0)         # dim11: is_dc
            cur_loc = getattr(self.env, 'current_node_location', -1)
            features.append(1.0 if node == cur_loc else 0.0)           # dim12: is_current
            _subgoal = getattr(self.env, 'current_subgoal_node', None)
            if _subgoal is not None:
                _llc_ref2 = getattr(self.env, 'low_level_controller', None)
                _h = _llc_ref2._get_hop_distance(node, _subgoal) if _llc_ref2 else 9999
                features.append(min(_h / 10.0, 1.0))                   # dim13: hop_to_target
            else:
                features.append(0.5)
            # dim14-16: hvt（全局历史部署密度）
            if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'hvt_all'):
                _hvt = self.env.resource_mgr.hvt_all[node]
                features.append(min(float(_hvt[0]) / 5.0, 1.0))       # dim14
                features.append(min(float(_hvt[1]) / 5.0, 1.0) if len(_hvt) > 1 else 0.0)  # dim15
                features.append(min(float(_hvt[2]) / 5.0, 1.0) if len(_hvt) > 2 else 0.0)  # dim16
            else:
                features.extend([0.0, 0.0, 0.0])
            # dim17: is_target（是否当前高层subgoal目标节点）
            features.append(1.0 if node == _subgoal else 0.0)          # dim17
            # dim18: vnf_depth（VNF链进度）
            _vnf_total = max(1, len(vnf_list))
            features.append(float(next_vnf_idx) / _vnf_total)          # dim18
            # dim19: dest_progress（目的地连接进度）
            _dest_total = max(1, len(dests))
            features.append(len(connected_dests) / _dest_total)        # dim19
            # dim20: phase_flag（0=VNF阶段，1=dest阶段）
            features.append(0.0 if is_vnf_phase else 1.0)              # dim20
            # dim21-23: 保留为0（对齐低层邻边BW特征位置，高层不使用）
            features.extend([0.0, 0.0, 0.0])                           # dim21-23
            # 最终确保维度与_node_feat_dim对齐（截断或padding）
            if len(features) < _node_feat_dim:
                features.extend([0.0] * (_node_feat_dim - len(features)))
            elif len(features) > _node_feat_dim:
                features = features[:_node_feat_dim]
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
                                cap = pool.bw_cap.get((u, v), pool.bw_cap.get((v, u), 100.0))
                                available_bw = pool.get_available_bandwidth(u, v)
                                norm_bw = min(available_bw / max(1.0, cap), 1.0)
                                bw_util = 1.0 - norm_bw

                                # hop_weight_norm
                                hop_w = 1.0
                                if hasattr(self.env.resource_mgr, 'topo'):
                                    raw_hop = float(self.env.resource_mgr.topo[u, v])
                                    max_hop = max(1.0, float(self.env.resource_mgr.topo.max()))
                                    hop_w = raw_hop / max_hop

                                is_in_tree = 0.0
                                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                                    tree = self.env.current_tree.get('tree', {})
                                    edge_key = (min(u, v), max(u, v))
                                    if edge_key in tree:
                                        is_in_tree = 1.0

                                # reserved ratio
                                reserved = pool.bw_reserved.get((u, v), pool.bw_reserved.get((v, u), 0.0))
                                reserved_ratio = reserved / max(1.0, cap)

                                # 5维：[bw_remaining, bw_utilization, hop_weight_norm, is_tree_edge, reserved]
                                feat = [norm_bw, bw_util, hop_w, is_in_tree, reserved_ratio]
                                edge_attr_list.append(feat)
                                edge_attr_list.append(feat)
                            else:
                                edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
                                edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
                        except:
                            edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
                            edge_attr_list.append([0.5, 0.5, 1.0, 0.0, 0.0])
            except:
                continue

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float32)

        bw_req = req.get('bw_origin', 0.0)
        norm_bw_req = min(bw_req / 10.0, 1.0)
        vnf_progress = next_vnf_idx / max(1, len(vnf_list))
        dest_progress = len(connected_dests) / max(1, len(dests))
        phase_feat = 0.0 if is_vnf_phase else 1.0
        # [Bug8修复] resource_tension只看CPU，MEM紧张时不可见
        # 改为取CPU和MEM紧张度的最大值，任一资源紧张都会反映到特征里
        # [语义修复] VNF只部署在DC节点，非DC节点CPU/MEM从不消耗，
        # 用全部n个节点计算会稀释紧张度（虚低）。改为只统计DC节点。
        _c_cap_g = max(1.0, float(getattr(self.env.resource_mgr, 'C_cap', 100.0)))
        _m_cap_g = max(1.0, float(getattr(self.env.resource_mgr, 'M_cap', 100.0)))
        _dc_nodes = getattr(self.env.resource_mgr, 'dc_nodes', list(range(n)))
        _n_dc = max(1, len(_dc_nodes))
        total_avail_cpu = sum(self.env.resource_mgr.pool.get_available_cpu(i) for i in _dc_nodes)
        total_avail_mem = sum(self.env.resource_mgr.pool.get_available_memory(i) for i in _dc_nodes)
        cpu_tension = 1.0 - (total_avail_cpu / (_n_dc * _c_cap_g))
        mem_tension = 1.0 - (total_avail_mem / (_n_dc * _m_cap_g))
        resource_tension = max(cpu_tension, mem_tension)

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
        # [Bug7修复] 每次调用都重建全图极其低效（mask计算里最多~60次调用）
        # __init__里的_pending_dist_cache和_last_dests_update从未接上，现在接上：
        # cache按episode失效：用current_request的id作为key前缀
        _req_id = id(self.env.current_request) if self.env.current_request else 0
        _cache_key = (_req_id, node1, node2)
        if _cache_key in self._pending_dist_cache:
            return self._pending_dist_cache[_cache_key]
        # 反向也查一下（无向图对称）
        _cache_key_rev = (_req_id, node2, node1)
        if _cache_key_rev in self._pending_dist_cache:
            return self._pending_dist_cache[_cache_key_rev]
        # 新请求时清空cache防止无限增长
        if self._last_dests_update != _req_id:
            self._pending_dist_cache.clear()
            self._last_dests_update = _req_id
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
                dist = nx.shortest_path_length(G, node1, node2)
            else:
                dist = 9999
            self._pending_dist_cache[_cache_key] = dist
            return dist
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