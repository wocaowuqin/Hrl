"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - TA-HRL v4 升级版 (Tree-Aware & Steiner Routing)
====================================
架构级修复与升级:
  [1] TA-HRL v4: 注入 hop_to_tree 距离感知，引导多播树边复用。
  [2] TA-HRL v4: 注入 dest_mask 目标感知，引导全局最优 Steiner 分叉点。
  [3] DAG Mask & Tabu List: 严格距离掩码禁止反向游走，动态禁忌表防止三角死锁。
  [4] Reward Gradient: +1.5(靠近目标), -2.0(远离目标), -1.5(死路退回)。
"""

import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data
import copy

logger = logging.getLogger(__name__)


class LowLevelController:
    """低层执行控制器 - TA-HRL v4 顶级架构优化版"""

    def __init__(self, env):
        self.env = env

        if not hasattr(self.env, 'resource_mgr'):
            logger.error("❌ LowLevelController: 未找到 resource_mgr")
            raise RuntimeError("resource_mgr 必须配置")

        if not hasattr(self.env, 'request_manager'):
            if hasattr(self.env.resource_mgr, 'request_manager'):
                self.env.request_manager = self.env.resource_mgr.request_manager
            else:
                try:
                    req_mgr_class = self.env.resource_mgr.__class__.__module__
                    import sys
                    mod = sys.modules[req_mgr_class]
                    if hasattr(mod, 'RequestLifecycleManager'):
                        RM_Class = getattr(mod, 'RequestLifecycleManager')
                        self.env.request_manager = RM_Class(self.env.resource_mgr)
                        self.env.resource_mgr.request_manager = self.env.request_manager
                except Exception as e:
                    logger.error(f"❌ [Init] 无法创建 request_manager: {e}")

        # 预计算最短路径距离矩阵（一次性）
        self._hop_dist_cache = None
        self._build_hop_distance_cache()

        logger.debug("✅ [LowLevelController] 初始化完成（TA-HRL v4 架构）")

    # ==================================================================
    # 分叉点选择
    # ==================================================================
    def _get_best_branch_point(self, dest_node, chain_nodes):
        """
        从已部署 VNF 中选距 dest_node 跳数最近的作为 branch 起点。
        tie-break: 距离相同时选 chain_nodes 中 index 更大的（更靠近末端 VNF）。
        """
        if not chain_nodes:
            return chain_nodes[-1] if chain_nodes else None
        best_vnf = chain_nodes[-1]
        best_dist = 9999
        best_ci = -1
        for ci, vnf in enumerate(chain_nodes):
            d = self._get_hop_distance(vnf, dest_node)
            if d < best_dist or (d == best_dist and ci > best_ci):
                best_dist = d
                best_ci = ci
                best_vnf = vnf
        logger.debug(f"[BranchPoint] dest={dest_node} → vnf={best_vnf}(dist={best_dist})")
        return best_vnf

    # ==================================================================
    # Hop Distance 缓存
    # ==================================================================
    def _build_hop_distance_cache(self):
        try:
            n = self.env.n
            G = nx.Graph()
            for u in range(n):
                neighbors = self.env.resource_mgr.get_neighbors(u)
                for v in neighbors:
                    G.add_edge(u, v)
            self._hop_dist_cache = dict(nx.all_pairs_shortest_path_length(G))
            logger.debug(f"✅ [HopDist] 预计算完成: {n}节点, {G.number_of_edges()}边")
        except Exception as e:
            logger.warning(f"⚠️ [HopDist] 预计算失败: {e}")
            self._hop_dist_cache = None

    def _get_hop_distance(self, u, v):
        if u == v:
            return 0
        if self._hop_dist_cache is not None:
            try:
                return self._hop_dist_cache[u][v]
            except KeyError:
                return 9999
        try:
            G = nx.Graph()
            for node in range(self.env.n):
                for nbr in self.env.resource_mgr.get_neighbors(node):
                    G.add_edge(node, nbr)
            return nx.shortest_path_length(G, u, v)
        except:
            return 9999

    def compute_bw_aware_path(self, source, target, excluded_edges=None):
        if source == target:
            return [source]

        if excluded_edges is None:
            excluded_edges = set()

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        tree_edges = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            for edge_key in self.env.current_tree.get('tree', {}).keys():
                tree_edges.add(edge_key)

        G = nx.Graph()
        for u in range(self.env.n):
            for v in self.env.resource_mgr.get_neighbors(u):
                edge_key = tuple(sorted((u, v)))
                if edge_key in excluded_edges:
                    continue
                if edge_key in tree_edges:
                    G.add_edge(u, v, weight=0.1) # 共享边代价极低，鼓励复用
                else:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                    if avail_bw >= bw_req * 1.01:
                        G.add_edge(u, v, weight=1.0)

        if source not in G or target not in G:
            return None

        try:
            return nx.shortest_path(G, source, target, weight='weight')
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            return None

    # ==================================================================
    # 核心步进
    # ==================================================================
    def step_low_level(self, action):
        if self.env.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_req'}

        if not hasattr(self.env, 'subgoal_step_count'):
            self.env.subgoal_step_count = 0

        # 全局失败计数：连续低层超时/无进展次数
        if not hasattr(self.env, '_consecutive_timeout_count'):
            self.env._consecutive_timeout_count = 0
        if not hasattr(self.env, '_bw_fail_count'):
            self.env._bw_fail_count = 0

        # 仅在子目标刚切换（_need_reset_to_last_vnf标志）时执行一次回位
        if self.env.current_phase == 'destination_connection' and \
                getattr(self.env, '_need_reset_to_last_vnf', False):
            if hasattr(self.env, 'chain_nodes') and len(self.env.chain_nodes) > 0:
                last_vnf = self.env.chain_nodes[-1]
                if self.env.current_node_location != last_vnf:
                    self.env.current_node_location = last_vnf
                    logger.debug(f"🔄 [Low] 瞬移回位至last_vnf={last_vnf}")
                    if hasattr(self.env, 'current_path_trace'):
                        self.env.current_path_trace = [last_vnf]  # 放入起点防止自己与自己成环
            self.env._need_reset_to_last_vnf = False

        self.env.subgoal_step_count += 1

        current_node = self.env.current_node_location
        target_action = int(action)
        is_stay = (target_action == current_node)

        at_target = False
        if is_stay:
            if self.env.current_phase == 'vnf_deployment':
                at_target = (current_node == getattr(self.env, 'current_deployment_target', None))
            elif self.env.current_phase == 'destination_connection':
                at_target = (current_node == getattr(self.env, 'current_target_node', None))

        if not at_target and self.env.subgoal_step_count > getattr(self.env, 'max_subgoal_steps', 25):
            self.env.subgoal_step_count = 0
            self.env._need_reset_to_last_vnf = False
            self.env._consecutive_timeout_count += 1

            _ph = getattr(self.env, 'current_phase', '?')
            _tgt = (getattr(self.env, 'current_deployment_target', None)
                    if _ph == 'vnf_deployment'
                    else getattr(self.env, 'current_target_node', None))
            _dist = self._get_hop_distance(current_node, _tgt) if _tgt is not None else '?'
            _conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0
            _alld = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
            _max_steps = getattr(self.env, 'max_subgoal_steps', 25)

            logger.warning(
                f"⏰ [Low] 超时 at Node {current_node} "
                f"(连续超时: {self.env._consecutive_timeout_count}次) | "
                f"phase={_ph} target={_tgt} hop_dist={_dist} "
                f"steps_limit={_max_steps} dest_progress={_conn}/{_alld}"
            )

            if self.env._consecutive_timeout_count >= 3:
                _vt = len(self.env.current_request.get('vnf', [])) if self.env.current_request else 0
                _vd = getattr(self.env, 'next_vnf_idx', 0)
                _dt = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
                _dc_conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0

                logger.warning(
                    f"❌ [Low] 连续超时{self.env._consecutive_timeout_count}次，Episode失败 | "
                    f"VNF={_vd}/{_vt} Dest={_dc_conn}/{_dt} | "
                    f"phase={_ph} cur={current_node} target={_tgt} hop_dist={_dist}"
                )
                self.env._consecutive_timeout_count = 0
                return self.get_state(), -20.0, True, False, {
                    'fail': True, 'reason': 'consecutive_timeout'
                }
            return self.get_state(), -5.0, False, True, {'timeout': True}

        if self.env.current_phase == 'vnf_deployment':
            return self._handle_vnf_deployment(current_node, target_action, is_stay)
        elif self.env.current_phase == 'destination_connection':
            return self._handle_destination_connection(current_node, target_action, is_stay)

        return self.get_state(), -10.0, True, False, {'error': 'unknown_phase'}

    # ==================================================================
    # VNF部署
    # ==================================================================
    def _handle_vnf_deployment(self, current_node, target_action, is_stay):
        target_goal = getattr(self.env, 'current_deployment_target', None)

        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        if target_goal is not None and current_node == target_goal:
            if hasattr(self.env, 'resource_mgr'):
                self.env.resource_mgr.current_request = self.env.current_request
                self.env.resource_mgr.current_tree = self.env.current_tree
                self.env.resource_mgr.current_phase = self.env.current_phase
                self.env.resource_mgr.next_vnf_idx = self.env.next_vnf_idx

            deploy_success = False
            if hasattr(self.env.resource_mgr, '_try_deploy'):
                deploy_success = self.env.resource_mgr._try_deploy(target_goal)
            elif hasattr(self.env, '_try_deploy'):
                deploy_success = self.env._try_deploy(target_goal)

            if deploy_success:
                self.env._consecutive_timeout_count = 0
                if not hasattr(self.env, 'chain_nodes'):
                    self.env.chain_nodes = []
                self.env.chain_nodes.append(current_node)

                try:
                    import networkx as _nx
                    _sfc = getattr(self.env, 'current_sfc', None)
                    if _sfc is not None:
                        _prev = (self.env.current_request.get('source')
                                 if not _sfc['chain_nodes']
                                 else _sfc['chain_nodes'][-1])
                        _G_topo = _nx.Graph()
                        for _u in range(self.env.n):
                            for _v in self.env.resource_mgr.get_neighbors(_u):
                                _G_topo.add_edge(_u, _v)
                        if _prev != current_node:
                            _seg = _nx.shortest_path(_G_topo, _prev, current_node)
                        else:
                            _seg = [current_node]
                        _sfc['spine_paths'].append(_seg)
                        _sfc['chain_nodes'].append(current_node)
                        logger.debug(f"[SFC-DAG] spine段: {_prev}→{current_node} = {_seg}")
                except Exception as _e:
                    logger.warning(f"[SFC-DAG] spine记录失败: {_e}")

                self.env.next_vnf_idx += 1
                vnf_list = self.env.current_request.get('vnf', [])
                current_count = self.env.next_vnf_idx

                if current_count >= len(vnf_list):
                    self.env.sfc_upstream_nodes = set(self.env.chain_nodes[:-1])
                    info = {'phase': 'vnf_complete', 'all_vnf_deployed': True,
                            'deployed_count': current_count, 'total_vnf': len(vnf_list)}
                    self._reset_vnf_phase_only()
                    return self.get_state(), 20.0, False, True, info
                else:
                    info = {'vnf_deployed': True, 'vnf_idx': self.env.next_vnf_idx - 1,
                            'deployed_count': current_count, 'total_vnf': len(vnf_list)}
                    return self.get_state(), 10.0, False, True, info
            else:
                self.env._consecutive_timeout_count = getattr(self.env, '_consecutive_timeout_count', 0) + 1
                info = {'deploy_fail': True, 'vnf_idx': self.env.next_vnf_idx, 'reason': 'resource_insufficient'}
                self._reset_vnf_phase_only()
                if self.env._consecutive_timeout_count >= 3:
                    self.env._consecutive_timeout_count = 0
                    return self.get_state(), -20.0, True, False, {**info, 'fail': True}
                return self.get_state(), -5.0, False, True, info
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 目的地连接
    # ==================================================================
    def _handle_destination_connection(self, current_node, target_action, is_stay):
        target_goal = getattr(self.env, 'current_target_node', None)

        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        if current_node == target_goal:
            if 'connected_dests' not in self.env.current_tree:
                self.env.current_tree['connected_dests'] = set()

            if target_goal not in self.env.current_tree['connected_dests']:
                self.env.current_tree['connected_dests'].add(target_goal)

                try:
                    import networkx as _nx
                    _sfc = getattr(self.env, 'current_sfc', None)
                    if _sfc is not None and _sfc['chain_nodes']:
                        _last_vnf = _sfc['chain_nodes'][-1]
                        if 'branch_roots' not in _sfc:
                            _sfc['branch_roots'] = {}
                        _sfc['branch_roots'][target_goal] = _last_vnf
                        _G_topo = _nx.Graph()
                        for _u in range(self.env.n):
                            for _v in self.env.resource_mgr.get_neighbors(_u):
                                _G_topo.add_edge(_u, _v)
                        _bseg = _nx.shortest_path(_G_topo, _last_vnf, target_goal)
                        _sfc['branch_paths'][target_goal] = _bseg
                        _bw = self.env.current_request.get('bw_origin', 0.0)
                        for _j in range(len(_bseg) - 1):
                            _ek = tuple(sorted((_bseg[_j], _bseg[_j+1])))
                            if 'tree' not in self.env.current_tree:
                                self.env.current_tree['tree'] = {}
                            if _ek not in self.env.current_tree['tree']:
                                self.env.current_tree['tree'][_ek] = _bw
                except Exception as _e:
                    logger.warning(f"[SFC-DAG] branch记录失败: {_e}")

            steps_used = getattr(self.env, 'subgoal_step_count', 50)
            _tgt_node = getattr(self.env, 'current_target_node', None)
            if _tgt_node is not None:
                _min_hops = self._get_hop_distance(self.env.current_node_location, _tgt_node) + 1
            else:
                _min_hops = 1
            step_reward = max(20.0 - max(0, steps_used - _min_hops) * 1.5, -5.0)

            try:
                all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
                connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
            except:
                all_dests, connected = set(), set()

            if all_dests.issubset(connected) and len(all_dests) > 0:
                logger.debug(f"✅ [Episode完成] 多播树建树成功！ connected={len(connected)}/{len(all_dests)}")
                self.env._consecutive_timeout_count = 0
                self.env._bw_fail_count = 0
                self._archive_episode_success_only()
                self._add_request_to_lifecycle_manager()

                _sfc = getattr(self.env, 'current_sfc', None)
                _chain = getattr(self.env, 'chain_nodes', [])
                return self.get_state(), 100.0, True, False, {
                    'episode_complete': True, 'all_destinations_connected': True,
                    'success': True, 'connected_count': len(connected),
                    'chain_nodes': _chain
                }
            else:
                self.env.subgoal_step_count = 0
                self.env.current_target_node = None

                if hasattr(self.env, 'chain_nodes') and len(self.env.chain_nodes) > 0:
                    self.env._need_reset_to_last_vnf = True
                if hasattr(self.env, 'current_path_trace'):
                    # 提前放入起点，防止下个目标一出来就被判走回头路
                    if self.env.chain_nodes:
                        self.env.current_path_trace = [self.env.chain_nodes[-1]]
                    else:
                        self.env.current_path_trace = []

                return self.get_state(), step_reward, False, True, {'dest_connected': True}
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 移动处理（带共享边奖励逻辑 & 陡峭梯度）
    # ==================================================================
    def _handle_movement(self, current_node, target_action, target_goal):
        next_node = int(target_action)

        if next_node == current_node:
            if current_node != target_goal:
                neighbors = self.env.resource_mgr.get_neighbors(current_node)
                bw_req = self.env.current_request.get('bw_origin', 0.0)
                valid_neighbors = [n for n in neighbors if self.env.resource_mgr.pool.get_available_bandwidth(current_node, n) >= bw_req]
                if not valid_neighbors:
                    return self.get_state(), -10.0, True, False, {'error': 'trapped'}
                return self.get_state(), -1.0, False, False, {'warning': 'stay'}

        bw_req = self.env.current_request.get('bw_origin', 0.0)
        edge_key = tuple(sorted((current_node, next_node)))

        is_new_edge = edge_key not in self.env.current_tree.get('tree', {})

        if is_new_edge:
            try:
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node)
                has_bw = avail_bw >= bw_req
            except:
                has_bw = False
            if not has_bw:
                self.env._bw_fail_count = getattr(self.env, '_bw_fail_count', 0) + 1
                if self.env._bw_fail_count >= 10:
                    self.env._bw_fail_count = 0
                    return self.get_state(), -20.0, True, False, {
                        'fail': True, 'reason': 'bandwidth_exhausted'
                    }
                return self.get_state(), -2.0, False, False, {'error': 'no_bandwidth'}

        if is_new_edge:
            reward = -1.2
            action_type = "NewPath"
        else:
            reward = 0.0
            action_type = "Reuse"

        # 🚀 距离梯度奖励强化 (+1.5, -2.0)
        if target_goal is not None:
            try:
                dist_before = self._get_hop_distance(current_node, target_goal)
                dist_after = self._get_hop_distance(next_node, target_goal)
                if dist_before < 9999 and dist_after < 9999:
                    delta = dist_before - dist_after
                    if delta > 0:
                        reward += 1.5 * delta   # 强力正反馈
                    elif delta < 0:
                        reward += 2.0 * delta   # 强力惩罚（delta为负数，此处+=其实是相减）
            except:
                pass

        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        step_count = len(self.env.current_path_trace)
        reward += -0.05 * (step_count / 10.0)

        # 🚀 死胡同防抖惩罚 (-1.5)
        if next_node in self.env.current_path_trace:
            reward -= 1.5
            action_type = "Fallback_Revisit"

        self.env.current_node_location = next_node

        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree:
                self.env.current_tree['tree'] = {}
            self.env.current_tree['tree'][edge_key] = bw_req
            self.env.nodes_on_tree.add(current_node)
            self.env.nodes_on_tree.add(next_node)

        if 'tree_usage' not in self.env.current_tree:
            self.env.current_tree['tree_usage'] = {}
        self.env.current_tree['tree_usage'][edge_key] = self.env.current_tree['tree_usage'].get(edge_key, 0) + 1

        self.env.current_path_trace.append(next_node)

        return self.get_state(), reward, False, False, {'moved': True, 'type': action_type}

    # ==================================================================
    # 动作掩码 (DAG 掩码 + 死路软释放)
    # ==================================================================
    def get_low_level_action_mask(self):
        mask = np.zeros(self.env.n, dtype=np.float32)
        current = self.env.current_node_location
        neighbors = self.env.resource_mgr.get_neighbors(current)

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        tree_edges = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            for edge_key in self.env.current_tree.get('tree', {}).keys():
                tree_edges.add(edge_key)

        phase = getattr(self.env, 'current_phase', None)

        # 1. 带宽基础筛选
        for nbr in neighbors:
            edge_key = tuple(sorted((current, nbr)))
            if edge_key in tree_edges:
                mask[nbr] = 1.0 # 复用边无条件开放
            else:
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                if avail_bw >= bw_req:
                    mask[nbr] = 1.0

        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        if target is not None and current == target:
            mask[:] = 0.0
            mask[current] = 1.0
            return mask
        else:
            mask[current] = 0.0

        # 🚀 约束 A: DAG Distance 掩码 (严禁反向游走，仅允许前向和平移)
        if target is not None:
            d_current = self._get_hop_distance(current, target)
            for nbr in neighbors:
                if mask[nbr] > 0:
                    d_next = self._get_hop_distance(nbr, target)
                    if d_next > d_current:
                        mask[nbr] = 0.0

        # 🚀 约束 B: Path Tabu 当前路径禁忌表 (严禁在同一次寻路中交叉形成环)
        current_path_set = set(getattr(self.env, 'current_path_trace', []))
        for nbr in neighbors:
            if mask[nbr] > 0 and nbr in current_path_set and nbr != target:
                mask[nbr] = 0.0

        # 🚀 死路软释放 (Soft Fallback): 如果前面把路全封死了，放开退路，但触发严厉的 Reward 惩罚
        if np.sum(mask) == 0:
            for nbr in neighbors:
                edge_key = tuple(sorted((current, nbr)))
                if edge_key in tree_edges or self.env.resource_mgr.pool.get_available_bandwidth(current, nbr) >= bw_req:
                    mask[nbr] = 1.0

            # 震荡防止: 只要还有其他可走邻居，优先不走上一步退回
            trace = getattr(self.env, 'current_path_trace', [])
            if len(trace) >= 2:
                prev_node = trace[-2]
                if prev_node in neighbors and np.sum(mask) > 1:
                    mask[prev_node] = 0.0

        # 如果真的完全没路走（物理隔离）
        if np.sum(mask) == 0:
            mask[current] = 1.0

        return mask

    # ==================================================================
    # 状态构建 (21维特征 + dest_mask)
    # ==================================================================
    def get_state(self):
        rm = self.env.resource_mgr
        K_vnf    = rm.K_vnf
        C_cap    = max(1, rm.C_cap)
        M_cap    = max(1, rm.M_cap)
        n        = self.env.n

        current_vnf_demand = 0.0
        if self.env.current_request:
            vnf_list  = self.env.current_request.get('vnf', [])
            idx       = getattr(self.env, 'next_vnf_idx', 0)
            if idx < len(vnf_list):
                cpu_reqs = self.env.current_request.get('cpu_origin', [10.0])
                current_vnf_demand = cpu_reqs[idx] if idx < len(cpu_reqs) else 10.0

        target_node = None
        if self.env.current_phase == 'vnf_deployment':
            target_node = getattr(self.env, 'current_deployment_target', None)
        elif self.env.current_phase == 'destination_connection':
            target_node = getattr(self.env, 'current_target_node', None)
        target_node_int = int(target_node) if target_node is not None else -1

        current_node = getattr(self.env, 'current_node_location', -1)
        _dc = getattr(self.env, 'dc_nodes', None) or getattr(getattr(self.env, 'resource_mgr', None), 'dc_nodes', None) or getattr(getattr(getattr(self.env, 'resource_mgr', None), 'pool', None), 'dc_nodes', None)
        dc_nodes = set(_dc) if _dc else set()

        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
        connected_dests = (self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set())
        hvt_all = rm.hvt_all

        # 🚀 目标感知掩码 (Dest Mask)
        try:
            all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
            remaining_dests = all_dests - connected_dests
        except Exception:
            remaining_dests = set()

        dest_mask = torch.zeros(n, dtype=torch.bool)
        for d in remaining_dests:
            if 0 <= d < n:
                dest_mask[int(d)] = True

        max_hops = max(1, n - 1)
        def _hop(u, v):
            if u < 0 or v < 0 or u == v: return 0.0
            try: return self._get_hop_distance(u, v) / max_hops
            except Exception: return 1.0

        vnf_list_total = (self.env.current_request.get('vnf', []) if self.env.current_request else [])
        total_vnf = max(1, len(vnf_list_total))
        cur_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_depth_norm = min(1.0, cur_vnf_idx / total_vnf)

        subgoal_steps = getattr(self.env, 'subgoal_step_count', 0)
        subgoal_horizon = getattr(self.env, 'subgoal_horizon', 40)
        progress_ratio = min(1.0, subgoal_steps / max(1, subgoal_horizon))

        phase = getattr(self.env, 'current_phase', 'other')
        if phase == 'vnf_deployment': phase_flag = 0.0
        elif phase == 'destination_connection': phase_flag = 1.0
        else: phase_flag = 0.5

        # 🚀 21 维状态特征矩阵 (20维原有 + 1维距离多播树)
        features = np.zeros((n, 21), dtype=np.float32)
        for node in range(n):
            avail_cpu  = rm.pool.get_available_cpu(node)
            avail_mem  = rm.pool.get_available_memory(node)
            fit_factor = 1.0 if avail_cpu >= current_vnf_demand else -1.0

            features[node, 0] = avail_cpu / C_cap
            features[node, 1] = avail_mem / M_cap
            features[node, 2] = fit_factor
            features[node, 3] = 1.0 if node in dc_nodes else 0.0
            features[node, 4] = 1.0 if node == current_node else 0.0
            features[node, 5] = _hop(node, target_node_int)

            if 0 <= node < hvt_all.shape[0]:
                features[node, 6:6 + K_vnf] = hvt_all[node, :K_vnf].astype(np.float32)

            features[node, 6 + K_vnf]     = 1.0 if node in nodes_on_tree else 0.0
            features[node, 6 + K_vnf + 1] = 1.0 if node in connected_dests else 0.0
            features[node, 6 + K_vnf + 2] = 1.0 if node == target_node_int else 0.0
            features[node, 6 + K_vnf + 3] = vnf_depth_norm
            features[node, 6 + K_vnf + 4] = progress_ratio
            features[node, 6 + K_vnf + 5] = phase_flag

            # 🚀 核心创新: hop_to_tree 距离已有多播树干的最短距离
            if len(nodes_on_tree) > 0:
                features[node, 20] = min([self._get_hop_distance(node, t) for t in nodes_on_tree]) / max_hops
            else:
                features[node, 20] = 1.0

        x_tensor = torch.from_numpy(features).float()
        low_mask  = self.get_low_level_action_mask()

        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'build_dynamic_edge_attr'):
            edge_attr_tensor = self.env.resource_mgr.build_dynamic_edge_attr()
        elif hasattr(self.env, 'edge_attr') and self.env.edge_attr is not None:
            edge_attr_tensor = self.env.edge_attr
        else:
            edge_attr_tensor = None

        tree_edge_index = None
        tree_dict = getattr(self.env, 'current_tree', None)
        if tree_dict:
            tree_edges_raw = tree_dict.get('tree', {})
            if tree_edges_raw:
                rows, cols = [], []
                for (u, v) in tree_edges_raw.keys():
                    rows += [u, v]; cols += [v, u]
                tree_edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # 🚀 返回 PyG Data (加入了 dest_mask)
        return Data(
            x=x_tensor,
            edge_index=self.env.edge_index if hasattr(self.env, 'edge_index') else None,
            edge_attr=edge_attr_tensor,
            tree_edge_index=tree_edge_index,
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0),
            dest_mask=dest_mask
        )

    # ==================================================================
    # 辅助方法
    # ==================================================================
    def _calculate_tree_metrics(self):
        tree_edges = self.env.current_tree.get('tree', {})
        active_nodes = set()
        for u, v in tree_edges.keys():
            active_nodes.add(u); active_nodes.add(v)
        n_e, n_n = len(tree_edges), len(active_nodes)
        redundancy = max(0, n_e - max(1, n_n - 1)) / max(1, n_n - 1) if n_n > 1 else 0.0
        return {'tree_n_nodes': n_n, 'tree_n_edges': n_e,
                'redundancy': round(redundancy, 2),
                'avg_degree': round((2*n_e)/max(1,n_n), 2),
                'efficiency': round(n_e/max(1, getattr(self.env, 'subgoal_step_count', 1)), 2)}

    def _should_release_immediately(self):
        if hasattr(self.env, 'force_online_mode') and self.env.force_online_mode:
            return False
        if not hasattr(self.env, 'request_manager') or self.env.request_manager is None:
            return True
        return False

    def _archive_episode_success_only(self):
        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, '_archive_request'):
            try:
                self.env.resource_mgr._archive_request(success=True, already_rolled_back=False)
                return
            except:
                pass
        self._manual_save_resources_only()

    def _manual_save_resources_only(self):
        if self.env.current_request is None: return
        if 'resources_allocated' not in self.env.current_request:
            self.env.current_request['resources_allocated'] = {
                'placement': copy.deepcopy(self.env.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.env.current_tree.get('tree', {}))
            }

    def _add_request_to_lifecycle_manager(self):
        if not hasattr(self.env, 'request_manager') or not self.env.request_manager: return False
        if self.env.current_request is None: return False
        req_id = self.env.current_request.get('id', id(self.env.current_request))
        if req_id in self.env.request_manager.active_requests: return True
        resources = self._collect_allocated_resources()
        try:
            return self.env.request_manager.register_request(
                request={'id': req_id,
                         'arrival_time': self.env.current_request.get('arrival_time', 0),
                         'lifetime': self.env.current_request.get('lifetime', 5.0)},
                resources_allocated=resources)
        except:
            return False

    def _collect_allocated_resources(self):
        resources = {
            'placement': {}, 'tree': {},
            'bandwidth': self.env.current_request.get('bw_origin', 1.0),
            'cpu_requirements': self.env.current_request.get('cpu_origin', []),
            'memory_requirements': self.env.current_request.get('memory_origin', [])
        }
        if self.env.current_tree:
            if 'placement' in self.env.current_tree:
                resources['placement'] = copy.deepcopy(self.env.current_tree['placement'])
            if 'tree' in self.env.current_tree:
                resources['tree'] = copy.deepcopy(self.env.current_tree['tree'])
        if 'resources_allocated' in self.env.current_request:
            req_res = self.env.current_request['resources_allocated']
            if 'placement' in req_res:
                resources['placement'].update(copy.deepcopy(req_res['placement']))
            if 'tree' in req_res:
                resources['tree'].update(copy.deepcopy(req_res['tree']))
        return resources

    def _sync_sfc_to_tree(self):
        _sfc = getattr(self.env, 'current_sfc', None)
        if not _sfc:
            return
        _bw = self.env.current_request.get('bw_origin', 0.0) if self.env.current_request else 0.0
        if 'tree' not in self.env.current_tree:
            self.env.current_tree['tree'] = {}
        for _seg in _sfc['spine_paths']:
            for _j in range(len(_seg) - 1):
                _ek = tuple(sorted((_seg[_j], _seg[_j+1])))
                if _ek not in self.env.current_tree['tree']:
                    self.env.current_tree['tree'][_ek] = _bw
                self.env.nodes_on_tree.add(_seg[_j])
                self.env.nodes_on_tree.add(_seg[_j+1])

    def _collect_sfc_edges(self):
        _sfc = getattr(self.env, 'current_sfc', None)
        if not _sfc:
            return {}
        _bw = self.env.current_request.get('bw_origin', 0.0) if self.env.current_request else 0.0
        edges = {}
        all_segs = list(_sfc.get('spine_paths', [])) + list(_sfc.get('branch_paths', {}).values())
        for _seg in all_segs:
            for _j in range(len(_seg) - 1):
                _ek = tuple(sorted((_seg[_j], _seg[_j+1])))
                edges[_ek] = _bw
        return edges

    def _reset_phase_state(self):
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.current_target_node = None
        self.env.subgoal_step_count = 0
        self.env._need_reset_to_last_vnf = False
        if hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        self.env.chain_nodes = []
        self.env.sfc_upstream_nodes = set()

    def _reset_vnf_phase_only(self):
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0
        self.env._consecutive_timeout_count = 0