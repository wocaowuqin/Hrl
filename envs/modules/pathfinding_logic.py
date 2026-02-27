# -*- coding: utf-8 -*-
import heapq
import logging
import networkx as nx
from collections import deque

# 初始化日志记录器
logger = logging.getLogger(__name__)


class PathFindingMixin:
    """
    寻路逻辑模块
    包含了 A* 搜索、树上路径查找、分支点选择以及下一跳计算逻辑。

    依赖主类提供的属性:
    - self.n (int): 节点总数
    - self.nodes_on_tree (set/list): 当前树上的节点
    - self.current_tree (dict): 当前树结构，需包含 key 'tree'
    - self.resource_mgr (object): 资源管理器，需有 has_link, check_link_resource, get_neighbors 方法
    - self.current_request (dict): 当前请求信息
    - self.topology_mgr (object, optional): 拓扑管理器
    - self.topology_matrix (numpy.array, optional): 拓扑矩阵
    """

    def _a_star_search_with_tree_awareness(self, start, goal):
        """
        🔥 [智能A*搜索 V2.1] 添加超时机制，防止长时间搜索
        结合了树感知的启发式搜索。
        """
        if start == goal:
            return [start]

        # 缓存检查
        cache_key = (start, goal, frozenset(self.nodes_on_tree)) if hasattr(self, 'nodes_on_tree') else None
        if cache_key and hasattr(self, '_path_cache') and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # 🔥 检查失败缓存
        if hasattr(self, '_failed_paths_cache'):
            if (start, goal) in self._failed_paths_cache:
                return None

        # 检查是否已经在同一棵树上
        if hasattr(self, 'nodes_on_tree') and start in self.nodes_on_tree and goal in self.nodes_on_tree:
            tree_path = self._find_path_on_tree(start, goal)
            if tree_path:
                if hasattr(self, '_path_cache') and cache_key:
                    self._path_cache[cache_key] = tree_path
                return tree_path

        bw_req = self.current_request.get('bw_origin', 1.0) if hasattr(self,
                                                                       'current_request') and self.current_request else 1.0
        tree_edges = self.current_tree.get('tree', {}) if hasattr(self, 'current_tree') else {}

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        def heuristic(n):
            base_dist = self._get_distance(n, goal)
            tree_bonus = -5 if hasattr(self, 'nodes_on_tree') and n in self.nodes_on_tree else 0
            visit_penalty = 0
            if hasattr(self, '_node_visit_count'):
                visit_penalty = self._node_visit_count.get(n, 0) * 2
            return max(0, base_dist + tree_bonus + visit_penalty)

        f_score = {start: heuristic(start)}

        # 🔥 添加访问计数
        visited_count = 0
        max_visits = 30  # 最多访问30个节点

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # 🔥 超时检查
            visited_count += 1
            if visited_count > max_visits:
                # 缓存失败结果
                if not hasattr(self, '_failed_paths_cache'):
                    self._failed_paths_cache = set()
                self._failed_paths_cache.add((start, goal))
                return None

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                if hasattr(self, '_path_cache') and cache_key:
                    self._path_cache[cache_key] = path
                return path

            # 获取邻居
            neighbors = []
            # 假设 self.n 存在，否则需处理异常
            n_count = self.n if hasattr(self, 'n') else 0

            for v in range(n_count):
                if v != current and self.resource_mgr.has_link(current, v):
                    edge = tuple(sorted([current, v]))
                    is_on_tree = (edge in tree_edges)

                    if not is_on_tree:
                        if hasattr(self.resource_mgr, 'check_link_resource'):
                            if not self.resource_mgr.check_link_resource(current, v, bw_req):
                                continue

                    neighbors.append(v)

            for neighbor in neighbors:
                move_cost = 1.0
                if hasattr(self, 'nodes_on_tree'):
                    if neighbor not in self.nodes_on_tree:
                        move_cost = 2.0
                    elif current in self.nodes_on_tree and neighbor in self.nodes_on_tree:
                        move_cost = 0.5

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # 搜索失败，缓存结果
        if not hasattr(self, '_failed_paths_cache'):
            self._failed_paths_cache = set()
        self._failed_paths_cache.add((start, goal))

        if hasattr(self, 'action_stats'):
            print(f"⚠️ [A*失败] 从{start}到{goal}找不到带宽充足的路径（需要带宽{bw_req}）")

        return None

    def _select_best_fork_node(self, remaining_dests):
        """
        智能选择分支节点：基于A*路径和树结构
        """
        if not remaining_dests or not hasattr(self, 'nodes_on_tree'):
            return None

        tree_nodes = list(self.nodes_on_tree)
        if not tree_nodes:
            # 如果没有树节点，从源点开始
            return self.current_request.get('source', 0) if hasattr(self, 'current_request') else 0

        best_node = None
        best_score = float('inf')

        for tree_node in tree_nodes:
            # 计算从该树节点到所有剩余目的地的总路径长度
            total_path_length = 0
            reachable_count = 0

            for dest in remaining_dests:
                path = self._a_star_search_with_tree_awareness(tree_node, dest)
                if path:
                    total_path_length += len(path) - 1
                    reachable_count += 1

            if reachable_count == len(remaining_dests):
                # 所有目的地都可从该节点到达
                # 考虑节点访问次数（避免热点）
                visit_penalty = 0
                if hasattr(self, '_node_visit_count'):
                    visit_penalty = self._node_visit_count.get(tree_node, 0) * 5

                score = total_path_length + visit_penalty

                if score < best_score:
                    best_score = score
                    best_node = tree_node

        # 如果找不到最佳节点，选择离源点最近的树节点
        if best_node is None:
            source = self.current_request.get('source', 0) if hasattr(self, 'current_request') else 0
            distances = [(self._get_distance(node, source), node) for node in tree_nodes]
            distances.sort()
            best_node = distances[0][1] if distances else tree_nodes[0]

        print(f"🌳 [智能分支] 选择节点{best_node}作为分支点，可到达{len(remaining_dests)}个目的地")
        return best_node

    def _find_path_on_tree(self, start, goal):
        """
        在当前树上寻找路径（不建新边）
        """
        if start == goal:
            return [start]

        # 构建树上邻接表
        tree_adj = {}
        tree = self.current_tree.get('tree', {}) if hasattr(self, 'current_tree') else {}
        for (u, v), bw in tree.items():
            tree_adj.setdefault(u, []).append(v)
            tree_adj.setdefault(v, []).append(u)

        # BFS搜索
        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _get_distance(self, u, v):
        """[辅助方法] 计算距离，防止报错"""
        if u == v: return 0
        try:
            # 优先用 TopologyMgr
            if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'get_distance'):
                return self.topology_mgr.get_distance(u, v)

            # 备用 NetworkX
            if not hasattr(self, '_nx_graph'):
                if hasattr(self, 'topology_matrix'):
                    self._nx_graph = nx.from_numpy_array(self.topology_matrix)
                else:
                    return 50  # 无法计算时给个默认值
            return nx.shortest_path_length(self._nx_graph, u, v)
        except Exception:
            return 50  # 出错兜底

    def get_next_hop_to_target(self, current, target):
        """
        🧭 [V36.0] 智能获取下一跳节点
        优先级：最短路径 > 启发式距离 > 随机邻居
        """
        # 情况1：已在目标
        if current == target:
            return current

        # 情况2：构建网络图 (注意：这里每次调用都重新构建图可能比较耗时，建议缓存 G)
        G = nx.Graph()
        if hasattr(self, 'n') and hasattr(self, 'resource_mgr'):
            for u in range(self.n):
                neighbors = self.resource_mgr.get_neighbors(u)
                for v in neighbors:
                    if self.resource_mgr.has_link(u, v):
                        G.add_edge(u, v)

        # 情况3：尝试最短路径
        try:
            path = nx.shortest_path(G, current, target)
            if len(path) > 1:
                next_hop = path[1]
                logger.debug(f"🧭 [路径] 最短路径: {current}→{next_hop}→...→{target}")
                return next_hop
            else:
                return current
        except nx.NetworkXNoPath:
            logger.warning(f"⚠️ [路径] 无路径: {current}→{target}")
        except Exception as e:
            logger.error(f"⚠️ [路径] 搜索异常: {e}")

        # 情况4：启发式选择（距离目标最近的邻居）
        neighbors = self.resource_mgr.get_neighbors(current) if hasattr(self, 'resource_mgr') else []
        if not neighbors:
            logger.error(f"❌ [路径] 节点{current}无邻居！")
            return current

        best_neighbor = None
        best_distance = float('inf')

        for nbr in neighbors:
            try:
                nbr_path = nx.shortest_path(G, nbr, target)
                dist = len(nbr_path) - 1
            except nx.NetworkXNoPath:
                dist = float('inf')
            except Exception:
                dist = float('inf')

            if dist < best_distance:
                best_distance = dist
                best_neighbor = nbr

        if best_neighbor is not None and best_distance != float('inf'):
            logger.debug(f"🧭 [路径] 启发式: {current}→{best_neighbor} (距{target}还有{best_distance}跳)")
            return best_neighbor

        # 情况5：无奈之选（返回第一个邻居）
        logger.warning(f"⚠️ [路径] 目标{target}完全不可达，随机选择邻居{neighbors[0]}")
        return neighbors[0]