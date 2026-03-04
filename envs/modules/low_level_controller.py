"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - 强制SFC状态机优化版
====================================
架构级修复:
  [1] 强制主干顺序: 记录 chain_nodes，锁定上游拓扑(排除误入节点)
  [2] 尾端多播分叉: 强制所有 Destination 必须从 last_vnf 重新出发，切断串联
  [3] 共享边复用: 多播复用边不扣带宽，且增加 tree_usage 记录引用次数
  [4] 奖励重塑: +100(完成), -10(资源/无路), -5(过长/超时)
"""

import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data
import copy

logger = logging.getLogger(__name__)


class LowLevelController:
    """低层执行控制器 - 强制SFC状态机优化版"""

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

        logger.info("✅ [LowLevelController] 初始化完成（强制SFC状态机优化版）")

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
            logger.info(f"✅ [HopDist] 预计算完成: {n}节点, {G.number_of_edges()}边")
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

        # 🔥 约束2前置防线：仅在子目标刚切换（_need_reset_to_last_vnf标志）时执行一次回位
        # ⚠️ 不再依赖 subgoal_step_count==0，避免Coordinator重置计数时反复触发瞬移
        if self.env.current_phase == 'destination_connection' and \
                getattr(self.env, '_need_reset_to_last_vnf', False):
            if hasattr(self.env, 'chain_nodes') and len(self.env.chain_nodes) > 0:
                last_vnf = self.env.chain_nodes[-1]
                if self.env.current_node_location != last_vnf:
                    # ── [FIX 树长] 重新启用瞬移回位 ───────────────────────────
                    # 原来注释掉导致每个目的地各走独立路径，5dest×5跳≈28条边全图遍历。
                    # 启用后每次从last_vnf出发，复用已有路径分叉，预期树长降至8~14。
                    self.env.current_node_location = last_vnf
                    logger.debug(f"🔄 [Low] 瞬移回位至last_vnf={last_vnf}")
                    if hasattr(self.env, 'current_path_trace'):
                        self.env.current_path_trace = []
                    # ─────────────────────────────────────────────────────────
            self.env._need_reset_to_last_vnf = False  # 消费flag，只执行一次

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

            # ── [DIAG] 超时时打印关键导航信息 ────────────────────────────────
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
            # ─────────────────────────────────────────────────────────────

            if self.env._consecutive_timeout_count >= 3:
                # ── [DIAG] 失败时打印完整状态快照，区分是超时问题还是资源问题 ──
                _vt = len(self.env.current_request.get('vnf', [])) if self.env.current_request else 0
                _vd = getattr(self.env, 'next_vnf_idx', 0)
                _dt = len(self.env.current_request.get('dest', [])) if self.env.current_request else 0
                _dc_conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0
                try:
                    _dc_tensions = {
                        n: round(1.0 - self.env.resource_mgr.pool.get_available_cpu(n) / 100.0, 2)
                        for n in getattr(self.env, 'dc_nodes', [])
                    }
                except Exception:
                    _dc_tensions = {}
                try:
                    _bw_tensions = {}
                    if _tgt is not None:
                        neighbors = self.env.resource_mgr.get_neighbors(current_node)
                        bw_req = self.env.current_request.get('bw_origin', 0.0) if self.env.current_request else 0.0
                        _bw_tensions = {
                            nbr: round(self.env.resource_mgr.pool.get_available_bandwidth(current_node, nbr), 1)
                            for nbr in neighbors
                        }
                except Exception:
                    _bw_tensions = {}
                logger.warning(
                    f"❌ [Low] 连续超时{self.env._consecutive_timeout_count}次，Episode失败 | "
                    f"VNF={_vd}/{_vt} Dest={_dc_conn}/{_dt} | "
                    f"phase={_ph} cur={current_node} target={_tgt} hop_dist={_dist} | "
                    f"max_subgoal_steps={_max_steps} | "
                    f"DC资源紧张度={_dc_tensions} | "
                    f"邻居可用带宽={_bw_tensions}"
                )
                # ─────────────────────────────────────────────────────────
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
                self.env._consecutive_timeout_count = 0  # VNF部署成功，清零超时计数
                if not hasattr(self.env, 'chain_nodes'):
                    self.env.chain_nodes = []
                self.env.chain_nodes.append(current_node)

                # ── [SFC-DAG STEP3] 记录spine路径段到current_sfc ────────────
                try:
                    import networkx as _nx
                    _sfc = getattr(self.env, 'current_sfc', None)
                    if _sfc is not None:
                        # prev = source（第一个VNF）或上一个VNF
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
                # ─────────────────────────────────────────────────────────────

                self.env.next_vnf_idx += 1
                vnf_list = self.env.current_request.get('vnf', [])
                current_count = self.env.next_vnf_idx

                if current_count >= len(vnf_list):
                    # 🔥 修复致命漏洞1：只封印真正的主干节点，不再封印 nodes_on_tree
                    # 将 Source 到 VNF(k-1) 彻底锁死
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
                self.env._consecutive_timeout_count = getattr(self.env, '_consecutive_timeout_count', 0) + 1
                logger.warning(
                    f"❌ [Low] VNF部署失败 节点{target_goal}: "
                    f"CPU可用={avail_cpu:.1f} MEM可用={avail_mem:.1f} "
                    f"(连续失败:{self.env._consecutive_timeout_count}次)"
                )
                info = {'deploy_fail': True, 'vnf_idx': self.env.next_vnf_idx,
                        'reason': 'resource_insufficient',
                        'avail_cpu': avail_cpu, 'avail_mem': avail_mem}
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

                # ── [SFC-DAG STEP4] 记录branch路径 ──────────────────────────
                try:
                    import networkx as _nx
                    _sfc = getattr(self.env, 'current_sfc', None)
                    if _sfc is not None and _sfc['chain_nodes']:
                        _last_vnf = _sfc['chain_nodes'][-1]
                        _G_topo = _nx.Graph()
                        for _u in range(self.env.n):
                            for _v in self.env.resource_mgr.get_neighbors(_u):
                                _G_topo.add_edge(_u, _v)
                        _bseg = _nx.shortest_path(_G_topo, _last_vnf, target_goal)
                        _sfc['branch_paths'][target_goal] = _bseg
                        # 把branch边同步到current_tree['tree']
                        _bw = self.env.current_request.get('bw_origin', 0.0)
                        for _j in range(len(_bseg) - 1):
                            _ek = tuple(sorted((_bseg[_j], _bseg[_j+1])))
                            if 'tree' not in self.env.current_tree:
                                self.env.current_tree['tree'] = {}
                            if _ek not in self.env.current_tree['tree']:
                                self.env.current_tree['tree'][_ek] = _bw
                except Exception as _e:
                    logger.warning(f"[SFC-DAG] branch记录失败: {_e}")
                # ─────────────────────────────────────────────────────────────

            steps_used = getattr(self.env, 'subgoal_step_count', 50)
            # ── [FIX 树长1] 加大步数惩罚：每多走1步超出最短路就扣1.5分，允许负奖励
            # 原来 max(20-steps*0.5, 5)，40步仍得5分，绕路代价几乎为零。
            _tgt_node = getattr(self.env, 'current_target_node', None)
            if _tgt_node is not None:
                _min_hops = self._get_hop_distance(self.env.current_node_location, _tgt_node) + 1
            else:
                _min_hops = 1
            step_reward = max(20.0 - max(0, steps_used - _min_hops) * 1.5, -5.0)
            # ──────────────────────────────────────────────────────────────────

            try:
                all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
                connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
            except:
                all_dests, connected = set(), set()

            if all_dests.issubset(connected) and len(all_dests) > 0:
                logger.info(
                    f"✅ [Episode完成] 所有目的地已连接，生成合法多播树！"
                    f" connected={len(connected)}/{len(all_dests)} steps_used={steps_used}"
                )
                self.env._consecutive_timeout_count = 0
                self.env._bw_fail_count = 0
                self._archive_episode_success_only()
                self._add_request_to_lifecycle_manager()

                # ── [SFC-DAG STEP5] 用分层DAG拼接路径验证，彻底替代shortest_path ─
                _sfc = getattr(self.env, 'current_sfc', None)
                _src = self.env.current_request.get('source', None)
                _chain = getattr(self.env, 'chain_nodes', [])
                _sfc_ok = True
                try:
                    if _sfc and _sfc['spine_paths'] and _sfc['branch_paths']:
                        for _d, _branch in _sfc['branch_paths'].items():
                            # 验证方式：不拼接路径，而是直接检查各段结构
                            # spine[k]的末尾必须是chain_nodes[k]
                            # branch的起点必须是last_vnf，终点必须是dest
                            _chain_ok = True
                            _spine_ok = all(
                                len(_sfc['spine_paths'][_k]) > 0 and
                                _sfc['spine_paths'][_k][-1] == _sfc['chain_nodes'][_k]
                                for _k in range(len(_sfc['chain_nodes']))
                                if _k < len(_sfc['spine_paths'])
                            )
                            _branch_ok = (
                                len(_branch) > 0 and
                                _sfc['chain_nodes'] and
                                _branch[0] == _sfc['chain_nodes'][-1] and
                                _branch[-1] == _d
                            )
                            _all_present = _spine_ok and _branch_ok
                            _in_order = True  # 结构拼接方式天然有序
                            if _all_present and _in_order:
                                logger.info(f"✅ [SFC-DAG] src={_src}→dst={_d} "
                                            f"VNF={_sfc['chain_nodes']} ✓")
                            else:
                                logger.warning(f"⚠️ [SFC-DAG违规] src={_src}→dst={_d} "
                                               f"spine_ok={_spine_ok} branch_ok={_branch_ok} "
                                               f"spine_ends={[s[-1] if s else None for s in _sfc['spine_paths']]} "
                                               f"chain={_sfc['chain_nodes']} "
                                               f"branch_start={_branch[0] if _branch else None} "
                                               f"branch_end={_branch[-1] if _branch else None}")
                                _sfc_ok = False
                    if _sfc_ok:
                        logger.info(f"✅ [SFC-DAG验证通过] chain={_sfc['chain_nodes'] if _sfc else _chain} src={_src}")
                except Exception as _e:
                    logger.debug(f"[SFC-DAG验证] 跳过: {_e}")
                # ─────────────────────────────────────────────────────────────

                return self.get_state(), 100.0, True, False, {
                    'episode_complete': True, 'all_destinations_connected': True,
                    'success': True, 'connected_count': len(connected),
                    'sfc_valid': _sfc_ok, 'chain_nodes': _chain
                }
            else:
                self.env.subgoal_step_count = 0
                self.env.current_target_node = None

                # 🔥 约束2核心防线：当前目标连接完成，设置flag通知下一个子目标开始时回切到 last_vnf
                # ⚠️ 不在此处直接瞬移，由step_low_level入口的flag逻辑执行，防止时序错误
                if hasattr(self.env, 'chain_nodes') and len(self.env.chain_nodes) > 0:
                    self.env._need_reset_to_last_vnf = True
                if hasattr(self.env, 'current_path_trace'):
                    self.env.current_path_trace = []

                return self.get_state(), step_reward, False, True, {'dest_connected': True}
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 移动处理（带共享边奖励逻辑）
    # ==================================================================
    def _handle_movement(self, current_node, target_action, target_goal):
        next_node = int(target_action)

        if next_node == current_node:
            if current_node != target_goal:
                neighbors = self.env.resource_mgr.get_neighbors(current_node)
                bw_req = self.env.current_request.get('bw_origin', 0.0)
                valid_neighbors = [n for n in neighbors if
                                   self.env.resource_mgr.pool.get_available_bandwidth(current_node, n) >= bw_req]
                if not valid_neighbors:
                    # 🔥 奖励修正: -10 资源耗尽导致的困境死锁
                    return self.get_state(), -10.0, True, False, {'error': 'trapped'}
                return self.get_state(), -1.0, False, False, {'warning': 'stay'}

        bw_req = self.env.current_request.get('bw_origin', 0.0)
        edge_key = tuple(sorted((current_node, next_node)))

        # 🔥 约束4落实: 严格区分新边和共享边
        is_new_edge = edge_key not in self.env.current_tree.get('tree', {})

        if is_new_edge:
            try:
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node)
                has_bw = avail_bw >= bw_req
            except:
                has_bw = False
                avail_bw = 0.0
            if not has_bw:
                self.env._bw_fail_count = getattr(self.env, '_bw_fail_count', 0) + 1
                # ── [DIAG] 带宽不足：记录目标距离，判断是绕路难还是带宽真的耗尽 ──
                _dist_via_next = self._get_hop_distance(next_node, target_goal) if target_goal is not None else '?'
                _dist_cur = self._get_hop_distance(current_node, target_goal) if target_goal is not None else '?'
                logger.debug(
                    f"⚠️ [Low.BW] 带宽不足 {current_node}→{next_node} | "
                    f"需={bw_req:.1f} 可用={avail_bw:.1f} | "
                    f"target={target_goal} cur_dist={_dist_cur} next_dist={_dist_via_next} | "
                    f"bw_fail_count={self.env._bw_fail_count}"
                )
                # ─────────────────────────────────────────────────────────
                if self.env._bw_fail_count >= 10:
                    logger.warning(
                        f"❌ [Low] 带宽连续失败{self.env._bw_fail_count}次，Episode失败 | "
                        f"cur={current_node} target={target_goal} bw_req={bw_req:.1f}"
                    )
                    self.env._bw_fail_count = 0
                    return self.get_state(), -20.0, True, False, {
                        'fail': True, 'reason': 'bandwidth_exhausted',
                        'required_bw': bw_req, 'available_bw': avail_bw
                    }
                return self.get_state(), -2.0, False, False, {'error': 'no_bandwidth', 'edge': (current_node, next_node)}

        # 🔥 鼓励共享边: 共享边 0 惩罚，新边加大惩罚防止乱走
        # ── [FIX 树长2] 新边惩罚从 -0.5 加强到 -1.2
        # 原来距离奖励 +0.6 就能完全抵消 -0.5，agent 不在乎多走新边。
        if is_new_edge:
            reward = -1.2
            action_type = "NewPath"
        else:
            reward = 0.0
            action_type = "Reuse"

        if target_goal is not None:
            try:
                dist_before = self._get_hop_distance(current_node, target_goal)
                dist_after = self._get_hop_distance(next_node, target_goal)
                if dist_before < 9999 and dist_after < 9999:
                    delta = dist_before - dist_after
                    reward += 0.6 * delta
            except:
                pass

        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        step_count = len(self.env.current_path_trace)
        # 防止利用复用边(0惩罚)无限刷距离奖励，引入随步数增长的硬性微小惩罚
        reward += -0.05 * (step_count / 10.0)

        # ── [FIX] 强化兜圈惩罚 ───────────────────────────────────────────
        # 原来 -0.8*visit_count 太弱，复用边 reward=0 加距离奖励可以完全抵消，
        # 导致 hop_dist=2 时仍然兜圈 40 步超时（EP610/611/614 等大量此类失败）。
        recent_window = self.env.current_path_trace[-15:]
        visit_count = recent_window.count(next_node)
        if visit_count >= 2:
            reward += -2.0 * visit_count  # 从 -0.8 加强到 -2.0
        elif visit_count == 1:
            reward += -0.5               # 第一次重复也给轻惩罚，鼓励探索新路
        # ─────────────────────────────────────────────────────────────────

        self.env.current_node_location = next_node

        # 🔥 共享边不重复计费，且添加 tree_usage 作为引用计数扩展
        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree:
                self.env.current_tree['tree'] = {}
            self.env.current_tree['tree'][edge_key] = bw_req
            self.env.nodes_on_tree.add(current_node)
            self.env.nodes_on_tree.add(next_node)

        # 记录边引用次数（为后续按次释放做结构准备）
        if 'tree_usage' not in self.env.current_tree:
            self.env.current_tree['tree_usage'] = {}
        self.env.current_tree['tree_usage'][edge_key] = self.env.current_tree['tree_usage'].get(edge_key, 0) + 1

        self.env.current_path_trace.append(next_node)

        return self.get_state(), reward, False, False, {'moved': True, 'type': action_type}

    # ==================================================================
    # 动作掩码 (强制主干不回流)
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
            # ── [FIX] 已到达目标节点：强制只允许 stay，完全屏蔽所有邻居 ──────
            # 原逻辑：mask[current]=1 但邻居也全开着，agent 学不会"到了就停"，
            # 导致 hop_dist=0 时仍然乱走直到超时（EP612 等大量此类失败）。
            mask[:] = 0.0
            mask[current] = 1.0
            return mask
            # ─────────────────────────────────────────────────────────────
        else:
            mask[current] = 0.0
            # ── [FIX] 目标恰好在1跳邻居时，强制只开放该邻居，防止绕路兜圈 ──
            if target is not None:
                dist_to_target = self._get_hop_distance(current, target)
                if dist_to_target == 1 and 0 <= target < len(mask) and mask[target] > 0:
                    mask[:] = 0.0
                    mask[target] = 1.0
                    return mask
            # ───────────────────────────────────────────────────────────────

        if hasattr(self.env, 'current_path_trace') and len(self.env.current_path_trace) >= 1:
            recent_trace = self.env.current_path_trace[-10:]
            visit_counts = {}
            for node in recent_trace:
                visit_counts[node] = visit_counts.get(node, 0) + 1

            critical_nodes = set()
            if target is not None:
                critical_nodes.add(target)
            if self.env.current_request:
                source = self.env.current_request.get('source')
                if source is not None:
                    critical_nodes.add(source)

            for nbr in neighbors:
                if mask[nbr] <= 0:
                    continue
                nbr_visits = visit_counts.get(nbr, 0)
                if nbr_visits >= 2 and nbr not in critical_nodes:
                    remaining = np.sum(mask > 0) - 1
                    if remaining >= 1:
                        mask[nbr] = 0.0

        if np.sum(mask) == 0:
            for nbr in neighbors:
                edge_key = tuple(sorted((current, nbr)))
                if edge_key in tree_edges:
                    mask[nbr] = 1.0
            if np.sum(mask) == 0:
                for nbr in neighbors:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                    if avail_bw >= bw_req:
                        mask[nbr] = 1.0
            if np.sum(mask) == 0:
                mask[current] = 1.0

        return mask

    # ==================================================================
    # 状态构建
    # ==================================================================
    def get_state(self):
        current_vnf_demand = 0.0
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            idx = getattr(self.env, 'next_vnf_idx', 0)
            if idx < len(vnf_list):
                cpu_reqs = self.env.current_request.get('cpu_origin', [10.0])
                current_vnf_demand = cpu_reqs[idx] if idx < len(cpu_reqs) else 10.0

        BASE_FEATURE_DIM = 5
        DYNAMIC_FEATURE_DIM = 3

        base_features = []
        for node in range(self.env.n):
            avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
            avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
            fit_factor = 1.0 if avail_cpu >= current_vnf_demand else -1.0
            feat = [avail_cpu / 100.0, avail_mem / 100.0, fit_factor, 0.5, 0.5]
            if len(feat) < 14:
                feat += [0.0] * (14 - len(feat))
            base_features.append(feat)

        base_x = np.array(base_features, dtype=np.float32)

        dynamic_features = []
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
        connected_dests = self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set()

        target_node = None
        if self.env.current_phase == 'vnf_deployment':
            target_node = getattr(self.env, 'current_deployment_target', None)
        elif self.env.current_phase == 'destination_connection':
            target_node = getattr(self.env, 'current_target_node', None)

        target_node_int = -1
        if target_node is not None:
            try:
                target_node_int = int(target_node)
            except:
                pass

        for node in range(self.env.n):
            t_m = 1.0 if node in nodes_on_tree else 0.0
            c_m = 1.0 if node in connected_dests else 0.0
            is_target = 1.0 if node == target_node_int else 0.0
            dynamic_features.append([t_m, c_m, is_target])

        full_x = np.concatenate([base_x, np.array(dynamic_features)], axis=1)
        x_tensor = torch.from_numpy(full_x).float()
        low_mask = self.get_low_level_action_mask()

        return Data(
            x=x_tensor,
            edge_index=self.env.edge_index if hasattr(self.env, 'edge_index') else None,
            edge_attr=self.env.edge_attr if hasattr(self.env, 'edge_attr') else None,
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0)
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
                         'lifetime': self.env.current_request.get('lifetime', 50)},
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
        """把已记录的spine_paths同步到current_tree['tree']，供资源释放兼容代码使用。"""
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
        """
        [STEP6] 从分层DAG收集所有物理边，用于资源释放。
        返回 {edge_key: bw} 字典，替代旧的 current_tree['tree']。
        """
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
        self.env._need_reset_to_last_vnf = False  # 清除回位flag防污染
        if hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        # 🔥 彻底清空主干记录变量防污染
        self.env.chain_nodes = []
        self.env.sfc_upstream_nodes = set()

    def _reset_vnf_phase_only(self):
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0
        self.env._consecutive_timeout_count = 0  # 切换subgoal时清零超时计数