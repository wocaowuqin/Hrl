"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - TA-HRL v4 升级版 (Tree-Aware & Steiner Routing)
====================================
架构级修复与升级:
  [1] TA-HRL v4: 注入 hop_to_tree 距离感知，引导多播树边复用。
  [2] TA-HRL v4: 注入 dest_mask 目标感知，引导全局最优 Steiner 分叉点。
  [3] DAG Mask & Tabu List: 严格距离掩码禁止反向游走，动态禁忌表防止三角死锁。
  [4] Reward Gradient: +2.0(靠近目标), -3.0(远离目标), -3.0(死路退回), -1.0(等距振荡), -5.0(连续无进展)

带宽孤岛修复 (v4.1):
  [修改1] compute_bw_aware_path: 增加 Widest Path 降级兜底，严格BW路径失败时改走带宽最宽路径。
  [修改2] _handle_destination_connection: 带宽孤岛不再直接Episode失败，
          先尝试绕路，绕路无解时截断子目标让高层重调度，连续5次失败才终止Episode。
  [修改3] get_low_level_action_mask 死路软释放: 优先复用树边（不消耗新带宽），
          树边也不通才放开有剩余带宽的新边，减少无效带宽消耗。
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
            # 🆕 [修改1] 降级：Widest Path 兜底（最大最小带宽路径）
            # 当严格带宽约束找不到路时，退而找剩余带宽最大的路径
            G_fallback = nx.Graph()
            for u in range(self.env.n):
                for v in self.env.resource_mgr.get_neighbors(u):
                    edge_key = tuple(sorted((u, v)))
                    if edge_key in excluded_edges:
                        continue
                    if edge_key in tree_edges:
                        G_fallback.add_edge(u, v, weight=0.01)  # 树边：最优先
                    else:
                        avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                        if avail_bw > 0:
                            # 带宽越大 weight 越小，shortest_path 会优先走带宽最宽的路
                            G_fallback.add_edge(u, v, weight=1.0 / (avail_bw + 0.01))
            try:
                path = nx.shortest_path(G_fallback, source, target, weight='weight')
                logger.debug(f"[WidestPath] 严格BW路径失败，降级找到带宽最宽路径: {source}→{target} = {path}")
                return path
            except Exception:
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

            _trace = getattr(self.env, 'current_path_trace', [])
            _trace_set = set(_trace)
            _nbrs = self.env.resource_mgr.get_neighbors(current_node)
            _mask = [n for n in _nbrs if n not in _trace_set]
            # 打印每个邻居的距离和被屏蔽原因
            _nbr_detail = []
            _d_cur = self._get_hop_distance(current_node, _tgt) if _tgt is not None else 99
            _tree_edges = set()
            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                _tree_edges = set(self.env.current_tree.get('tree', {}).keys())
            for _n in _nbrs:
                _dn = self._get_hop_distance(_n, _tgt) if _tgt is not None else 99
                _ek = tuple(sorted((current_node, _n)))
                _reasons = []
                if _n in _trace_set: _reasons.append('tabu')
                if _dn > _d_cur and _ek not in _tree_edges: _reasons.append('constraint_A')
                _nbr_detail.append(f"{_n}(d={_dn},{'|'.join(_reasons) if _reasons else 'OK'})")
            # 区分失败原因：走满40步 vs 物理资源拦截（trace很短）
            _steps_used = len(_trace)
            if _steps_used >= _max_steps * 0.8:
                _fail_type = "⏰ 迷宫超时"
                _res_info = ""
            else:
                # 诊断是哪种资源不足
                _res_parts = []
                _bw_req = self.env.current_request.get('bw_origin', 0.0) if self.env.current_request else 0.0
                _vnf_list = self.env.current_request.get('vnf', []) if self.env.current_request else []
                _vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
                _cpu_reqs = self.env.current_request.get('cpu_origin', []) if self.env.current_request else []
                _mem_reqs = self.env.current_request.get('mem_origin', []) if self.env.current_request else []
                _req_cpu = _cpu_reqs[_vnf_idx] if _vnf_idx < len(_cpu_reqs) else 0.0
                _req_mem = _mem_reqs[_vnf_idx] if _vnf_idx < len(_mem_reqs) else 0.0
                # 检查target节点的CPU/MEM
                if _tgt is not None:
                    try:
                        _avail_cpu = self.env.resource_mgr.pool.get_available_cpu(_tgt)
                        _avail_mem = self.env.resource_mgr.pool.get_available_memory(_tgt)
                        if _req_cpu > 0 and _avail_cpu < _req_cpu:
                            _res_parts.append(f"CPU不足({_avail_cpu:.0f}<{_req_cpu:.0f})")
                        if _req_mem > 0 and _avail_mem < _req_mem:
                            _res_parts.append(f"MEM不足({_avail_mem:.0f}<{_req_mem:.0f})")
                        # 检查所有邻居链路的BW，找出不足的
                        _bw_blocked = []
                        for _n in _nbrs:
                            _avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, _n)
                            if _bw_req > 0 and _avail_bw < _bw_req:
                                _bw_blocked.append(f"{current_node}-{_n}({_avail_bw:.1f})")
                        if _bw_blocked:
                            _res_parts.append(f"BW不足需要{_bw_req:.1f}:{','.join(_bw_blocked)}")
                    except Exception:
                        pass
                # 如果还是未知，检查target所有接入链路BW和节点资源
                if not _res_parts and _tgt is not None:
                    try:
                        _tgt_nbrs = self.env.resource_mgr.get_neighbors(_tgt)
                        _tgt_bw_blocked = []
                        for _tn in _tgt_nbrs:
                            _abw = self.env.resource_mgr.pool.get_available_bandwidth(_tn, _tgt)
                            if _bw_req > 0 and _abw < _bw_req:
                                _tgt_bw_blocked.append(f"{_tn}-{_tgt}({_abw:.1f})")
                        if len(_tgt_bw_blocked) == len(_tgt_nbrs):
                            _res_parts.append(f"target={_tgt}带宽孤岛(需{_bw_req:.1f})")
                        elif _tgt_bw_blocked:
                            _res_parts.append(f"target={_tgt}部分链路BW不足:{','.join(_tgt_bw_blocked)}")
                        # 检查当前节点到所有邻居的BW
                        _cur_bw_blocked = []
                        for _cn in _nbrs:
                            _abw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, _cn)
                            if _bw_req > 0 and _abw < _bw_req:
                                _cur_bw_blocked.append(f"{current_node}-{_cn}({_abw:.1f})")
                        if _cur_bw_blocked:
                            _res_parts.append(f"当前节点出链路BW不足:{','.join(_cur_bw_blocked)}")
                    except Exception as _e:
                        _res_parts.append(f"资源查询异常:{_e}")
                _res_info = f" [{', '.join(_res_parts) if _res_parts else '原因待查'}]"
                _fail_type = "💥 资源拦截"
            logger.warning(
                f"{_fail_type}{_res_info} at Node {current_node} "
                f"(连续: {self.env._consecutive_timeout_count}次, steps_used={_steps_used}/{_max_steps}) | "
                f"phase={_ph} target={_tgt} hop_dist={_dist} "
                f"dest_progress={_conn}/{_alld} | "
                f"tabu_free={_mask} nbr_detail={_nbr_detail}"
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
                self._archive_episode_fail()  # 🆕 回滚BW
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
                    try:
                        avail_cpu = self.env.resource_mgr.pool.get_available_cpu(target_goal)
                        avail_mem = self.env.resource_mgr.pool.get_available_memory(target_goal)
                    except Exception:
                        avail_cpu, avail_mem = 0.0, 0.0

                    req_cpu, req_mem = 0.0, 0.0
                    if self.env.current_request:
                        vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
                        cpu_reqs = self.env.current_request.get('cpu_origin', [])
                        mem_reqs = self.env.current_request.get('memory_origin', [])
                        if vnf_idx < len(cpu_reqs): req_cpu = cpu_reqs[vnf_idx]
                        if vnf_idx < len(mem_reqs): req_mem = mem_reqs[vnf_idx]

                    self.env._consecutive_timeout_count = getattr(self.env, '_consecutive_timeout_count', 0) + 1

                    logger.warning(
                        f"❌ [VNF-FAIL] 节点={target_goal} | "
                        f"可用CPU={avail_cpu:.1f} 需要={req_cpu:.1f} | "
                        f"可用MEM={avail_mem:.1f} 需要={req_mem:.1f} | "
                        f"连续失败={self.env._consecutive_timeout_count}"
                    )

                    info = {'deploy_fail': True, 'vnf_idx': self.env.next_vnf_idx,
                            'reason': 'resource_insufficient',
                            'avail_cpu': avail_cpu, 'avail_mem': avail_mem}
                    self._reset_vnf_phase_only()
                    if self.env._consecutive_timeout_count >= 3:
                        self.env._consecutive_timeout_count = 0
                        self._archive_episode_fail()  # 🆕 回滚BW
                        return self.get_state(), -20.0, True, False, {**info, 'fail': True}
                    return self.get_state(), -5.0, False, True, info
            else:
                return self.get_state(), -5.0, False, False, {'warning': 'wait_for_stay'}
        # ==================================================================
        # 目的地连接
        # ==================================================================
    def _handle_destination_connection(self, current_node, target_action, is_stay):
        target_goal = getattr(self.env, 'current_target_node', None)

        # 🚀 [修改2] 带宽孤岛检测：目标节点所有接入链路已满时，尝试绕路而非直接失败
        if target_goal is not None and current_node != target_goal:
            _bw_req = self.env.current_request.get('bw_origin', 0.0)
            _tree_edges = self.env.current_tree.get('tree', {})
            _target_alive = False
            _reachable_via_tree = False
            for _n in self.env.resource_mgr.get_neighbors(target_goal):
                _ek = tuple(sorted((_n, target_goal)))
                _in_tree = _ek in _tree_edges
                if _in_tree:
                    _target_alive = True
                    _reachable_via_tree = True
                    break
                if self.env.resource_mgr.pool.get_available_bandwidth(_n, target_goal) >= _bw_req:
                    _target_alive = True
                    break
            if not _target_alive:
                # 尝试通过 compute_bw_aware_path 找带宽最宽的绕路（Widest Path 兜底）
                _fallback_path = self.compute_bw_aware_path(current_node, target_goal)
                if _fallback_path and len(_fallback_path) > 1:
                    # 找到绕路方案，不快速失败，继续让Agent移动
                    logger.info(f"🔀 [Low] 带宽孤岛: target={target_goal} 直连已满，绕路方案: {_fallback_path}")
                else:
                    logger.warning(f"🏝️ [Low] 带宽孤岛: target={target_goal} 所有接入链路已满，快速失败")
                    # 🆕 不直接让整个Episode失败，改为跳过当前目标让高层重新调度
                    self.env._consecutive_timeout_count += 1
                    if self.env._consecutive_timeout_count >= 5:
                        # 连续5次孤岛无法解决，才真正失败
                        self.env._consecutive_timeout_count = 0
                        self._archive_episode_fail()  # 🆕 回滚BW
                        return self.get_state(), -20.0, True, False, {
                            'fail': True, 'reason': 'bandwidth_exhausted'
                        }
                    # 截断当前子目标，让高层重选（Truncated=True）
                    return self.get_state(), -10.0, False, True, {
                        'bandwidth_island': True,
                        'skipped_target': target_goal,
                        'reason': 'bandwidth_exhausted'
                    }

        # ── [DestFix] 每次切换到新dest目标时强制瞬移回last_vnf并清空禁忌表 ──
        _chain = getattr(self.env, 'chain_nodes', [])
        if _chain:
            last_vnf = _chain[-1]
            _prev_target = getattr(self.env, '_last_dest_target', None)
            if _prev_target != target_goal:
                self.env.current_node_location = last_vnf
                self.env._last_dest_target = target_goal
                self.env.current_path_trace = [last_vnf]  # 完全清空禁忌表
                logger.debug(f"[DestFix] 目标{_prev_target}→{target_goal}，回位last_vnf={last_vnf}，清空禁忌表")
                current_node = last_vnf
                is_stay = (int(target_action) == current_node)
        # ──────────────────────────────────────────────────────────────────

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
                        # [SFC修复] branch路径用当前已建树作为图，而非全拓扑
                        # 这样branch只能走agent实际移动过的路径，不会走spine捷径
                        _G_actual = _nx.Graph()
                        _G_actual.add_edges_from(list(self.env.current_tree.get('tree', {}).keys()))
                        # 如果当前树里last_vnf→dest不连通，fallback用全拓扑
                        try:
                            _bseg = _nx.shortest_path(_G_actual, _last_vnf, target_goal)
                        except (_nx.NetworkXNoPath, _nx.NodeNotFound):
                            _bseg = _nx.shortest_path(_G_topo, _last_vnf, target_goal)
                        _sfc['branch_paths'][target_goal] = _bseg
                        _bw = self.env.current_request.get('bw_origin', 0.0)
                        for _j in range(len(_bseg) - 1):
                            _ek = tuple(sorted((_bseg[_j], _bseg[_j+1])))
                            if 'tree' not in self.env.current_tree:
                                self.env.current_tree['tree'] = {}
                            if _ek not in self.env.current_tree['tree']:
                                self.env.current_tree['tree'][_ek] = 0.0  # 仅记录路径，未实际分配BW
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

                # ── [带宽消耗统计] Episode成功完成时打印 ──────────────────────────
                try:
                    _bw_req = self.env.current_request.get('bw_origin', 0.0)
                    _tree_edges = self.env.current_tree.get('tree', {})
                    _n_edges = len(_tree_edges)
                    _total_bw_consumed = _bw_req * _n_edges  # 每条新树边消耗 bw_req

                    # 统计每条边的剩余带宽
                    _edge_details = []
                    for (_u, _v), _ratio in _tree_edges.items():
                        try:
                            _avail = self.env.resource_mgr.pool.get_available_bandwidth(_u, _v)
                            _edge_details.append(f"({_u}-{_v}: 剩余{_avail:.1f})")
                        except Exception:
                            _edge_details.append(f"({_u}-{_v}: 查询失败)")

                    # 全局带宽利用率
                    _total_avail = 0.0
                    _total_cap = 0.0
                    for _u in range(self.env.n):
                        for _v in self.env.resource_mgr.get_neighbors(_u):
                            if _u < _v:
                                try:
                                    _avail = self.env.resource_mgr.pool.get_available_bandwidth(_u, _v)
                                    _cap = getattr(self.env.resource_mgr.pool, 'B_cap', 100.0)
                                    _total_avail += _avail
                                    _total_cap += _cap
                                except Exception:
                                    pass
                    _global_util = (1.0 - _total_avail / max(1.0, _total_cap)) * 100.0

                    logger.info(
                        f"📊 [BW统计] 请求bw={_bw_req:.1f} | 树边数={_n_edges} | "
                        f"本次请求消耗带宽≈{_total_bw_consumed:.1f} | "
                        f"全局BW利用率={_global_util:.1f}% | "
                        f"树边剩余BW: {' '.join(_edge_details)}"
                    )
                except Exception as _bw_e:
                    logger.warning(f"⚠️ [BW统计] 统计失败: {_bw_e}")
                # ──────────────────────────────────────────────────────────────

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
                    # 切换dest目标时完全清空禁忌表，只保留last_vnf作为起点
                    if self.env.chain_nodes:
                        self.env.current_path_trace = [self.env.chain_nodes[-1]]
                    else:
                        self.env.current_path_trace = []
                # 同时清除_last_dest_target，确保DestFix在新目标上强制瞬移
                self.env._last_dest_target = None

                return self.get_state(), step_reward, False, True, {'dest_connected': True}
        else:
            return self.get_state(), -5.0, False, False, {'warning': 'wait_for_stay'}

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
                    self._archive_episode_fail()  # 🆕 回滚BW
                    return self.get_state(), -10.0, True, False, {'error': 'trapped'}
                return self.get_state(), -5.0, False, False, {'warning': 'stay'}

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
                    self._archive_episode_fail()  # 🆕 回滚BW
                    return self.get_state(), -20.0, True, False, {
                        'fail': True, 'reason': 'bandwidth_exhausted'
                    }
                return self.get_state(), -2.0, False, False, {'error': 'no_bandwidth'}

        if is_new_edge:
            reward = -0.8          # 原 -1.2 → -0.8，降低新边惩罚，鼓励直接靠近目标
            action_type = "NewPath"
            # 🆕 热点边惩罚：新边利用率越高惩罚越重，引导Agent绕开拥塞链路
            try:
                _cap = self.env.resource_mgr.pool.bw_cap.get(
                    edge_key, self.env.resource_mgr.pool.bw_cap.get(
                        (next_node, current_node), 100.0))
                _avail = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node)
                _util = 1.0 - _avail / max(1.0, _cap)
                if _util > 0.7:
                    _hotspot_penalty = -2.0 * (_util - 0.7) / 0.3  # 70%→0, 100%→-2.0 线性
                    reward += _hotspot_penalty
                    action_type = f"NewPath_Hotspot({_util:.0%})"
            except Exception:
                pass
        else:
            reward = 0.0
            action_type = "Reuse"

        # =============== 距离梯度奖励强化（+2.0靠近，-3.0远离） ===============
        dist_before = None
        dist_after = None
        if target_goal is not None:
            try:
                dist_before = self._get_hop_distance(current_node, target_goal)
                dist_after = self._get_hop_distance(next_node, target_goal)
                if dist_before < 9999 and dist_after < 9999:
                    delta = dist_before - dist_after
                    if delta > 0:
                        reward += 2.0 * delta   # 靠近奖励增强
                    elif delta < 0:
                        reward += 3.0 * delta   # 远离惩罚增强（delta为负）
            except:
                pass

        # =============== 等距移动惩罚（距离不变，但不是停留）加强至 -3.0 ===============
        if dist_before is not None and dist_after is not None and dist_after == dist_before and next_node != current_node:
            reward -= 3.0   # 原 -2.0 → -3.0
            action_type = "Oscillation"

        # =============== 无进展惩罚（连续5步未减少距离） ===============
        # 初始化/获取无进展跟踪变量
        if not hasattr(self.env, '_last_target_goal'):
            self.env._last_target_goal = None
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None

        # 检测目标是否变化，若变化则重置计数
        if target_goal != self.env._last_target_goal:
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None
            self.env._last_target_goal = target_goal

        if target_goal is not None and dist_after is not None:
            if self.env._last_dist_to_target is not None:
                if dist_after >= self.env._last_dist_to_target:
                    self.env._stuck_steps += 1
                else:
                    self.env._stuck_steps = 0
            self.env._last_dist_to_target = dist_after

            if self.env._stuck_steps >= 5:
                reward -= 5.0
                self.env._stuck_steps = 0   # 惩罚后重置，避免连续叠加
                action_type = "StuckPenalty"
        else:
            # 无目标时重置计数
            self.env._stuck_steps = 0
            self.env._last_dist_to_target = None
        # ==============================================================

        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        reward -= 1.0  # 固定步数惩罚
        # 🚀 死胡同防抖惩罚（加强至 -3.0）
        if next_node in self.env.current_path_trace:
            reward -= 3.0
            action_type = "Fallback_Revisit"

        self.env.current_node_location = next_node

        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree:
                self.env.current_tree['tree'] = {}
            # [SFC修复] VNF部署阶段：如果next_node是dest节点，不把这条边加入树
            # 防止spine路径经过dest节点，导致source→dest存在不经过完整VNF链的捷径
            _phase = getattr(self.env, 'current_phase', None)
            _dests = set()
            if self.env.current_request:
                _dests = set(int(d) for d in self.env.current_request.get('dest', []))
            _skip_edge = (_phase == 'vnf_deployment' and next_node in _dests)
            if not _skip_edge:
                self.env.current_tree['tree'][edge_key] = 1.0  # 存比例1.0，释放时bw*1.0=实际BW
                self.env.nodes_on_tree.add(current_node)
                self.env.nodes_on_tree.add(next_node)
            else:
                logger.debug(f"[SFC修复] spine边({current_node},{next_node})经过dest节点，跳过加入树")

        if 'tree_usage' not in self.env.current_tree:
            self.env.current_tree['tree_usage'] = {}
        self.env.current_tree['tree_usage'][edge_key] = self.env.current_tree['tree_usage'].get(edge_key, 0) + 1

        self.env.current_path_trace.append(next_node)
        _max_tabu = min(20, self.env.n)
        if len(self.env.current_path_trace) > _max_tabu:
            self.env.current_path_trace = self.env.current_path_trace[-_max_tabu:]

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

        # 🚀 约束A (智能树感知版): 禁止远离目标，但允许沿树边逆行复用
        # 先检查是否有前进方向可走，如果全被禁忌封死则不施加方向约束
        if target is not None and current != target:
            d_current = self._get_hop_distance(current, target)
            # 检查有没有不远离目标的邻居可走
            _forward_free = [
                nbr for nbr in neighbors
                if mask[nbr] > 0
                and self._get_hop_distance(nbr, target) <= d_current
            ]
            if _forward_free:  # 有前进方向可走，才施加约束
                for nbr in neighbors:
                    if mask[nbr] > 0:
                        d_next = self._get_hop_distance(nbr, target)
                        edge_key = tuple(sorted((current, nbr)))
                        if d_next > d_current  and edge_key not in tree_edges:
                            mask[nbr] = 0.0

        # 近距离强制约束已删除：与禁忌表叠加会封死所有出路导致超时

        # 🚀 约束 B: Path Tabu + 最优下一跳释放
        current_path_set = set(getattr(self.env, 'current_path_trace', []))
        if target is not None:
            _best_d = min(
                (self._get_hop_distance(nbr, target) for nbr in neighbors),
                default=99
            )
            _best_nbrs = {
                nbr for nbr in neighbors
                if self._get_hop_distance(nbr, target) == _best_d
            }
        else:
            _best_nbrs = set()

        for nbr in neighbors:
            if mask[nbr] > 0 and nbr in current_path_set and nbr != target:
                if nbr in _best_nbrs:
                    # 最优下一跳在禁忌表里：只有没有其他非禁忌的前进方向时才释放
                    _other_forward = [
                        n for n in neighbors
                        if mask[n] > 0 and n not in current_path_set
                        and self._get_hop_distance(n, target) <= _best_d
                    ]
                    if _other_forward:
                        mask[nbr] = 0.0  # 有其他前进路，正常封禁
                    # else: 没有其他路，保留可走
                else:
                    mask[nbr] = 0.0

        # 🚀 [修改3] 死路软释放 (Soft Fallback)：优先树边复用，再才放开新边
        # 当所有路径被禁忌封死时，分两优先级恢复
        if np.sum(mask) == 0:
            # 清空禁忌表，只保留当前节点防止原地stay
            self.env.current_path_trace = [current]
            # 优先级1：树边复用（不消耗新带宽，成本最低）
            for nbr in neighbors:
                edge_key = tuple(sorted((current, nbr)))
                if edge_key in tree_edges:
                    mask[nbr] = 1.0
            # 优先级2：树边也走不通，才放开有带宽的新边
            if np.sum(mask) == 0:
                for nbr in neighbors:
                    edge_key = tuple(sorted((current, nbr)))
                    if self.env.resource_mgr.pool.get_available_bandwidth(current, nbr) >= bw_req:
                        mask[nbr] = 1.0

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

            # 🚀 核心创新: hop_to_tree (支持消融：env._ablation_hop=True 时清零)
            if not getattr(self.env, '_ablation_hop', False):
                if len(nodes_on_tree) > 0:
                    features[node, 20] = min([self._get_hop_distance(node, t) for t in nodes_on_tree]) / max_hops
                else:
                    features[node, 20] = 1.0
            # else: 保持 0.0（np.zeros 已初始化）

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

    def _archive_episode_fail(self):
        """
        🆕 Episode失败时统一回滚BW资源
        之前失败路径直接return，从不调用_archive_request(success=False)，
        导致已分配的BW永远不归还，造成BW虚高（CPU=3.9% BW=36%的根本原因）
        """
        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, '_archive_request'):
            try:
                self.env.resource_mgr._archive_request(success=False, already_rolled_back=False)
                logger.debug("[FailRollback] BW资源已回滚")
                return
            except Exception as e:
                logger.warning(f"[FailRollback] _archive_request失败: {e}，尝试手动回滚")
        # 兜底：手动释放树里每条边的BW
        if self.env.current_request and self.env.current_tree:
            bw = self.env.current_request.get('bw_origin', 0.0)
            tree = self.env.current_tree.get('tree', {})
            released = 0
            for (u, v), flow in tree.items():
                if flow > 0.0:
                    try:
                        self.env.resource_mgr.pool.release_bandwidth(u, v, bw * flow)
                        released += 1
                    except Exception:
                        pass
            logger.debug(f"[FailRollback] 手动回滚 {released} 条树边BW")

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
                request=self.env.current_request,
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
                    self.env.current_tree['tree'][_ek] = 0.0  # 仅记录路径，未实际分配BW
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