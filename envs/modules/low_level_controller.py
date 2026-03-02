"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - 路径收敛优化版
====================================
优化清单:
  [1] binary mask (去掉2.0)
  [2] 距离引导 reward（核心）
  [3] 步数递增惩罚
  [4] 到达奖励与步数挂钩
  [5] 防绕圈递增惩罚
  [6] max_subgoal_steps 默认25
"""

import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data
import copy

logger = logging.getLogger(__name__)


class LowLevelController:
    """低层执行控制器 - 路径收敛优化版"""

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

        logger.info("✅ [LowLevelController] 初始化完成（路径收敛优化版）")

    # ==================================================================
    # Hop Distance 缓存
    # ==================================================================
    def _build_hop_distance_cache(self):
        """用networkx预计算全对最短路径，避免每步重复计算"""
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
        """获取两节点间最短跳数"""
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
        """
        带宽感知最短路径：优先复用已建树边，其次选带宽充足的新边
        excluded_edges: set of edge tuples to exclude (used after failed attempts)
        返回: 路径节点列表 [source, ..., target] 或 None（不可达）
        """
        if source == target:
            return [source]

        if excluded_edges is None:
            excluded_edges = set()

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        # 获取已建树边集合
        tree_edges = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            for edge_key in self.env.current_tree.get('tree', {}).keys():
                tree_edges.add(edge_key)

        # 构建带权图
        G = nx.Graph()
        for u in range(self.env.n):
            for v in self.env.resource_mgr.get_neighbors(u):
                edge_key = tuple(sorted((u, v)))
                # 排除已知失败边
                if edge_key in excluded_edges:
                    continue
                if edge_key in tree_edges:
                    G.add_edge(u, v, weight=0.1)
                else:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(u, v)
                    # 使用 1.01x 余量，防止浮点精度导致规划/执行不一致
                    if avail_bw >= bw_req * 1.01:
                        G.add_edge(u, v, weight=1.0)

        # 🔥 修复: 先检查 source/target 是否在图里
        #    当节点的所有邻边 BW 都耗尽时，该节点不会被加入 G
        #    此时直接返回 None（BW 不足不可达），而不是抛出异常
        if source not in G:
            logger.debug(f"⚠️ [PathFind] source={source} 所有邻边BW不足，不可达")
            return None
        if target not in G:
            logger.debug(f"⚠️ [PathFind] target={target} 所有邻边BW不足，不可达")
            return None

        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            logger.debug(f"⚠️ [PathFind] 无可达路径: {source} → {target} (bw_req={bw_req})")
            return None
        except Exception as e:
            logger.warning(f"⚠️ [PathFind] 路径计算异常: {e}")
            return None

    # ==================================================================
    # 核心步进
    # ==================================================================
    def step_low_level(self, action):
        if self.env.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_req'}

        if not hasattr(self.env, 'subgoal_step_count'):
            self.env.subgoal_step_count = 0
        self.env.subgoal_step_count += 1

        current_node = self.env.current_node_location
        target_action = int(action)
        is_stay = (target_action == current_node)

        # [关键修复] 到达目标的STAY优先于超时检查
        # 如果Agent已经在目标节点且执行STAY，即使超出步数也应该完成连接/部署
        at_target = False
        if is_stay:
            if self.env.current_phase == 'vnf_deployment':
                at_target = (current_node == getattr(self.env, 'current_deployment_target', None))
            elif self.env.current_phase == 'destination_connection':
                at_target = (current_node == getattr(self.env, 'current_target_node', None))

        if not at_target and self.env.subgoal_step_count > getattr(self.env, 'max_subgoal_steps', 25):
            # 不在目标节点，真正超时
            self.env.subgoal_step_count = 0
            logger.warning(f"⏰ [Low] Subgoal超时 at Node {current_node}, "
                           f"phase={self.env.current_phase}, target="
                           f"{getattr(self.env, 'current_target_node', None) or getattr(self.env, 'current_deployment_target', None)}")
            return self.get_state(), -1.0, False, True, {'timeout': True}

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
                self.env.next_vnf_idx += 1
                vnf_list = self.env.current_request.get('vnf', [])
                current_count = self.env.next_vnf_idx

                if current_count >= len(vnf_list):
                    info = {'phase': 'vnf_complete', 'all_vnf_deployed': True,
                            'deployed_count': current_count, 'total_vnf': len(vnf_list)}
                    self._reset_vnf_phase_only()
                    return self.get_state(), 20.0, False, True, info
                else:
                    info = {'vnf_deployed': True, 'vnf_idx': self.env.next_vnf_idx - 1,
                            'deployed_count': current_count, 'total_vnf': len(vnf_list)}
                    return self.get_state(), 10.0, False, True, info
            else:
                info = {'deploy_fail': True, 'vnf_idx': self.env.next_vnf_idx,
                        'reason': 'resource_insufficient'}
                self._reset_vnf_phase_only()
                return self.get_state(), -5.0, False, True, info
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 目的地连接（到达奖励与步数挂钩）
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

            # 🔥 到达奖励与步数挂钩：走得越快奖励越高
            steps_used = getattr(self.env, 'subgoal_step_count', 50)
            step_reward = max(20.0 - steps_used * 0.5, 5.0)

            try:
                all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
                connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
            except:
                all_dests, connected = set(), set()

            if all_dests.issubset(connected) and len(all_dests) > 0:
                logger.info("✅ [Episode完成] 所有目的地已连接")
                self._archive_episode_success_only()
                self._add_request_to_lifecycle_manager()
                return self.get_state(), 50.0, True, False, {
                    'episode_complete': True, 'all_destinations_connected': True,
                    'success': True, 'connected_count': len(connected)
                }
            else:
                self.env.subgoal_step_count = 0
                self.env.current_target_node = None
                return self.get_state(), step_reward, False, True, {'dest_connected': True}
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    # ==================================================================
    # 移动处理（距离引导 + 步数递增 + 防绕圈）
    # ==================================================================
    def _handle_movement(self, current_node, target_action, target_goal):
        """
        收敛强化版:
        1. 新边 -0.3 / 复用 -0.05
        2. 距离引导 ±0.6*delta（核心）
        3. 步数递增惩罚
        4. 绕圈递增惩罚
        """
        next_node = int(target_action)

        # 原地不动
        if next_node == current_node:
            if current_node != target_goal:
                neighbors = self.env.resource_mgr.get_neighbors(current_node)
                bw_req = self.env.current_request.get('bw_origin', 0.0)
                valid_neighbors = [n for n in neighbors if
                                   self.env.resource_mgr.pool.get_available_bandwidth(current_node, n) >= bw_req]
                if not valid_neighbors:
                    return self.get_state(), -10.0, True, False, {'error': 'trapped'}
                return self.get_state(), -1.0, False, False, {'warning': 'stay'}

        # 带宽检查
        bw_req = self.env.current_request.get('bw_origin', 0.0)
        edge_key = tuple(sorted((current_node, next_node)))
        is_new_edge = edge_key not in self.env.current_tree.get('tree', {})

        if is_new_edge:
            # 新边：必须检查带宽
            try:
                has_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node) >= bw_req
            except:
                has_bw = False
            if not has_bw:
                return self.get_state(), -2.0, False, False, {'error': 'no_bandwidth', 'edge': (current_node, next_node)}
        # 已建树边：带宽已在首次建边时分配，复用免检

        # [1] 新旧边奖励
        if is_new_edge:
            reward = -0.3
            action_type = "NewPath"
        else:
            reward = -0.05
            action_type = "Reuse"

        # [2] 距离引导（最关键）
        if target_goal is not None:
            try:
                dist_before = self._get_hop_distance(current_node, target_goal)
                dist_after = self._get_hop_distance(next_node, target_goal)
                if dist_before < 9999 and dist_after < 9999:
                    delta = dist_before - dist_after  # 正=靠近目标
                    reward += 0.6 * delta
            except:
                pass

        # [3] 步数递增惩罚
        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        step_count = len(self.env.current_path_trace)
        reward += -0.05 * (step_count / 10.0)

        # [4] 绕圈递增惩罚
        recent_window = self.env.current_path_trace[-15:]
        visit_count = recent_window.count(next_node)
        if visit_count >= 2:
            reward += -0.8 * visit_count

        # 物理状态更新
        self.env.current_node_location = next_node
        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree:
                self.env.current_tree['tree'] = {}
            self.env.current_tree['tree'][edge_key] = bw_req
            self.env.nodes_on_tree.add(current_node)
            self.env.nodes_on_tree.add(next_node)

        self.env.current_path_trace.append(next_node)

        return self.get_state(), reward, False, False, {'moved': True, 'type': action_type}

    # ==================================================================
    # 动作掩码 (binary mask)
    # ==================================================================
    def get_low_level_action_mask(self):
        mask = np.zeros(self.env.n, dtype=np.float32)
        current = self.env.current_node_location
        neighbors = self.env.resource_mgr.get_neighbors(current)

        bw_req = 0.0
        if self.env.current_request:
            bw_req = self.env.current_request.get('bw_origin', 0.0)

        # 获取已建树边（复用已建树边不需要额外带宽）
        tree_edges = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            for edge_key in self.env.current_tree.get('tree', {}).keys():
                tree_edges.add(edge_key)

        phase = getattr(self.env, 'current_phase', None)

        for nbr in neighbors:
            edge_key = tuple(sorted((current, nbr)))
            if edge_key in tree_edges:
                # 已建树边：免费复用，始终可走
                mask[nbr] = 1.0
            else:
                # 新边：需要检查带宽
                avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                if avail_bw >= bw_req:
                    mask[nbr] = 1.0

        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        if target is not None and current == target:
            mask[current] = 1.0
        else:
            mask[current] = 0.0

        # 防绕圈
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

        # 兜底
        if np.sum(mask) == 0:
            # 优先：复用已建树边（免费）
            for nbr in neighbors:
                edge_key = tuple(sorted((current, nbr)))
                if edge_key in tree_edges:
                    mask[nbr] = 1.0
            # 其次：检查带宽
            if np.sum(mask) == 0:
                for nbr in neighbors:
                    avail_bw = self.env.resource_mgr.pool.get_available_bandwidth(current, nbr)
                    if avail_bw >= bw_req:
                        mask[nbr] = 1.0
            # 最终兜底：原地等待
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

        BASE_FEATURE_DIM = 5   # [cpu, mem, fit_factor, 0.5, 0.5]
        DYNAMIC_FEATURE_DIM = 3 # [tree_member, connected, is_target]
        TOTAL_FEATURE_DIM = BASE_FEATURE_DIM + DYNAMIC_FEATURE_DIM  # 不再依赖padding到硬编码14

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

    def _reset_phase_state(self):
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.current_target_node = None
        self.env.subgoal_step_count = 0
        if hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []

    def _reset_vnf_phase_only(self):
        """
        [P2修复] VNF 部署成功后的轻量重置
        不再清除 current_phase —— 阶段切换由 set_high_level_goal 统一管理
        只清除部署目标和步数计数器
        """
        # 原代码: self.env.current_phase = None  ← 这会导致下次 step 无法识别阶段
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0