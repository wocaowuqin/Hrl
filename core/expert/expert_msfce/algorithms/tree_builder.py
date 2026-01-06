#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree Construction Algorithm - Trajectory Fix
Fixes: Phase 1 saving 0 samples due to missing trajectory data
"""

import time
import copy
import logging
import hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, deque

# ✅ Import optimization components
try:
    from ..utils.pareto import ParetoSet
    from .placement import OptimizedPlacementStrategy
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Tree State Data Class
# ============================================================================
class TreeState:
    def __init__(self, nodes=None, edges=None, covered_dests=None,
                 placement=None, traj=None, sum_cpu=0.0, sum_mem=0.0, sum_bw=0.0,
                 structure_hash=0, link_count=0):
        self.nodes = nodes if nodes else set()
        self.edges = edges if edges else []
        self.covered_dests = covered_dests if covered_dests else set()
        self.placement = placement if placement else {}
        self.traj = traj if traj else []  # ✅ Added trajectory tracking
        self.sum_cpu = sum_cpu
        self.sum_mem = sum_mem
        self.sum_bw = sum_bw
        self.structure_hash = structure_hash
        self.link_count = link_count

    def copy(self):
        return TreeState(
            nodes=self.nodes.copy(),
            edges=list(self.edges),
            covered_dests=self.covered_dests.copy(),
            placement=self.placement.copy(),
            traj=list(self.traj),  # ✅ Deep copy trajectory
            sum_cpu=self.sum_cpu,
            sum_mem=self.sum_mem,
            sum_bw=self.sum_bw,
            structure_hash=self.structure_hash,
            link_count=self.link_count
        )

    def update_hash_with_path(self, path_edges: List[Tuple[int, int]]):
        for u, v in path_edges:
            edge_tuple = tuple(sorted((u, v)))
            self.structure_hash = (self.structure_hash * 1315423911 ^ hash(edge_tuple))

    def get_hash(self) -> str:
        dests_hash = 0
        for d in self.covered_dests:
            dests_hash ^= hash(d)
        return f"{self.structure_hash}_{dests_hash}"


# ============================================================================
# Path Engine (Local Enhanced Version)
# ============================================================================
class PathEngine:
    def __init__(self, topo: np.ndarray, dist_matrix: Optional[np.ndarray] = None, link_cache: Any = None):
        self.topo = topo
        self.n = topo.shape[0]

        self._link_cache = {}
        if link_cache is not None:
            if isinstance(link_cache, dict):
                self._link_cache = link_cache
            elif hasattr(link_cache, 'cache') and isinstance(link_cache.cache, dict):
                self._link_cache = link_cache.cache
            elif hasattr(link_cache, '_cache') and isinstance(link_cache._cache, dict):
                self._link_cache = link_cache._cache

        if not self._link_cache:
            self._link_cache = self._build_link_cache(topo)

        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix = self._compute_hop_distances()

    def _build_link_cache(self, topo):
        cache = {}
        lid = 1
        rows, cols = np.where(np.triu(topo) > 0)
        for u, v in zip(rows, cols):
            cache[(u + 1, v + 1)] = lid
            cache[(v + 1, u + 1)] = lid
            lid += 1
        return cache

    def _compute_hop_distances(self) -> np.ndarray:
        try:
            graph = sp.csr_matrix(self.topo)
            result = shortest_path(csgraph=graph, directed=False, unweighted=True, return_predecessors=False)
            if isinstance(result, tuple): result = result[0]
            result[np.isinf(result)] = 9999
            return result.astype(int)
        except Exception:
            return np.full((self.n, self.n), 9999)

    def get_paths(self, source_nodes: Set[int], target: int, k: int = 5) -> List[Dict]:
        paths = []
        for src in source_nodes:
            if src == target: continue

            path_nodes = self._bfs(src, target)
            if path_nodes:
                edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
                links = []
                for u, v in edges:
                    lid = self._link_cache.get((u, v))
                    if lid: links.append(lid)

                dist = 9999
                if hasattr(self, 'dist_matrix') and self.dist_matrix is not None:
                    if 1 <= src <= self.n and 1 <= target <= self.n:
                        dist = self.dist_matrix[src - 1, target - 1]

                paths.append({
                    'nodes': path_nodes,
                    'edges': edges,
                    'links': links,
                    'distance': dist,
                    'hops': len(path_nodes) - 1,
                    'source': src
                })
        paths.sort(key=lambda p: (p['distance'], p['hops']))
        return paths[:k]

    def _bfs(self, src, dst):
        visited = {src}
        queue = deque([(src, [src])])

        while queue:
            node, path = queue.popleft()
            if node < 1 or node > self.n: continue
            neighbors = np.where(self.topo[node - 1] > 0)[0] + 1

            for n in neighbors:
                if n not in visited:
                    if n == dst: return path + [n]
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None


# ============================================================================
# Main Class: TreeBuilder
# ============================================================================
class TreeBuilder:
    def __init__(
            self,
            node_num: int,
            link_num: int,
            type_num: int,
            config,
            path_engine,
            resource_manager,
            placement_strategy=None
    ):
        self.node_num = node_num
        self.link_num = link_num
        self.type_num = type_num
        self.config = config
        self.resource_manager = resource_manager

        self.stats = {
            'trees_evaluated': 0, 'trees_pruned_pareto': 0,
            'trees_pruned_hash': 0, 'trees_pruned_beam': 0
        }

        topo_data = None
        link_cache_dict = None
        dist_matrix = None

        if hasattr(path_engine, 'topo') and isinstance(path_engine.topo, np.ndarray):
            topo_data = path_engine.topo

        if hasattr(path_engine, '_link_cache'):
            link_cache_dict = self._extract_dict_safe(path_engine._link_cache)
        elif hasattr(path_engine, 'link_map'):
            link_cache_dict = self._extract_dict_safe(path_engine.link_map)

        if hasattr(path_engine, 'dist_matrix'):
            dist_matrix = path_engine.dist_matrix
        elif hasattr(path_engine, '_distance_matrix'):
            dist_matrix = path_engine._distance_matrix

        if topo_data is None:
            logger.warning("⚠️ PathEngine missing 'topo', attempting reconstruction...")
            if link_cache_dict and len(link_cache_dict) > 0:
                logger.info(f"-> Reconstructing from link_cache ({len(link_cache_dict)} entries)")
                topo_data = np.zeros((node_num, node_num), dtype=int)
                for key in link_cache_dict:
                    if isinstance(key, tuple) and len(key) == 2:
                        u, v = key
                        if 1 <= u <= node_num and 1 <= v <= node_num:
                            topo_data[u - 1, v - 1] = 1
                            topo_data[v - 1, u - 1] = 1
            elif dist_matrix is not None:
                logger.info("-> Reconstructing from dist_matrix")
                topo_data = np.where(dist_matrix == 1, 1, 0)
            elif isinstance(path_engine, np.ndarray):
                topo_data = path_engine

        if topo_data is None:
            logger.error("❌ Failed to extract topology. Creating empty topology.")
            topo_data = np.zeros((node_num, node_num), dtype=int)

        self.path_engine = PathEngine(topo_data, dist_matrix, link_cache_dict)
        self.dist_matrix = self.path_engine.dist_matrix
        self.link_lookup = self.path_engine._link_cache

        self.placement_strategy = OptimizedPlacementStrategy(node_num, type_num)
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.beam_size = config.candidate_set_size

    def _extract_dict_safe(self, obj):
        if isinstance(obj, dict): return obj
        if hasattr(obj, 'cache') and isinstance(obj.cache, dict): return obj.cache
        if hasattr(obj, '_cache') and isinstance(obj._cache, dict): return obj._cache
        try:
            return {k: getattr(obj, k, None) for k in obj.keys()}
        except:
            return {}

    def _compute_links_fast(self, nodes: List[int]) -> List[int]:
        links = []
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            lid = self.link_lookup.get((u, v))
            if lid is None: lid = self.link_lookup.get((v, u))
            if lid: links.append(lid)
        return links

    def _check_resource_feasibility_safe(self, state, delta, request, links_to_deduct):
        if np.any(state['cpu'] < delta['cpu'] - 1e-5): return False
        if np.any(state['mem'] < delta['mem'] - 1e-5): return False
        bw_req = request.get('bw_origin', 0)
        if bw_req > 0:
            for lid in links_to_deduct:
                idx = lid - 1
                if idx < 0 or idx >= len(state['bw']): continue
                if state['bw'][idx] < bw_req - 1e-5:
                    return False
        return True

    def _apply_path_to_tree(self, tree_struct: Dict, info: Dict, request: Dict, state: Dict,
                            real_deploy: bool = False, resource_delta: Dict = None) -> bool:
        nodes = info['nodes']
        links_to_deduct = []
        new_links_indices = set()

        path_links = info.get('links')
        if path_links is None:
            path_links = self._compute_links_fast(nodes)

        for lid in path_links:
            idx = lid - 1
            if 0 <= idx < self.link_num:
                if tree_struct['tree'][idx] == 0 and idx not in new_links_indices:
                    new_links_indices.add(idx)
                    if real_deploy:
                        links_to_deduct.append(lid)

        if real_deploy and resource_delta:
            if not self._check_resource_feasibility_safe(state, resource_delta, request, links_to_deduct):
                return False

        for idx in new_links_indices:
            tree_struct['tree'][idx] = 1

        tree_struct['nodes'].update(nodes)
        tree_struct['paths_map'][nodes[-1]] = nodes

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        if 'link_count' not in tree_struct: tree_struct['link_count'] = 0
        tree_struct['link_count'] += len(new_links_indices)
        tree_struct['node_count'] = len(tree_struct['nodes'])

        if real_deploy and resource_delta:
            state['cpu'] = np.maximum(state['cpu'] - resource_delta['cpu'], 0.0)
            state['mem'] = np.maximum(state['mem'] - resource_delta['mem'], 0.0)
            bw_req = request.get('bw_origin', 0.0)
            if bw_req > 0:
                for lid in links_to_deduct:
                    idx = lid - 1
                    if idx < len(state['bw']):
                        state['bw'][idx] = max(0.0, state['bw'][idx] - bw_req)
        return True

    def _init_tree_struct(self, request):
        return {
            'tree': np.zeros(self.link_num, dtype=np.uint8),
            'hvt': np.zeros((self.node_num, self.type_num), dtype=int),
            'nodes': {request['source']},
            'paths_map': {},
            'covered_dests': set(),
            'link_count': 0,
            'node_count': 1,
            'structure_hash': 0,
            'sum_cpu': 0.0,
            'sum_mem': 0.0,
            'sum_bw': 0.0,
            'traj': []  # ✅ Initialized trajectory list
        }

    def _select_next_dest(self, source_nodes: Set[int], unadded: Set[int]) -> Optional[int]:
        if not unadded: return None
        unadded_list = sorted(list(unadded))

        if not source_nodes: return unadded_list[0]
        if self.dist_matrix is None: return unadded_list[0]

        try:
            src_indices = [s - 1 for s in source_nodes if 1 <= s <= self.node_num]
            dst_indices = [d - 1 for d in unadded_list if 1 <= d <= self.node_num]
            if not src_indices or not dst_indices: return unadded_list[0]

            sub_matrix = self.dist_matrix[np.ix_(src_indices, dst_indices)]
            min_dists = np.min(sub_matrix, axis=0)
            best_local_idx = np.argmin(min_dists)
            return unadded_list[best_local_idx]
        except Exception:
            return unadded_list[0]

    def _fast_state_copy(self, state: Dict) -> Dict:
        new_state = state.copy()
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                new_state[k] = v.copy()
        return new_state

    def construct_tree(self, request: Dict, network_state: Dict, forced_first_dest_idx: Optional[int] = None) -> Tuple[
        Optional[Dict], List, List]:
        self.stats = {k: 0 for k in self.stats}
        source = request['source']
        destinations = set(request['dest'])
        pareto_map = defaultdict(ParetoSet)
        seen_trees = set()

        initial_tree = self._init_tree_struct(request)
        initial_tree['structure_hash'] = 0
        candidate_trees = [(initial_tree, 0.0)]
        final_completed = []

        for step in range(len(destinations)):
            next_candidates = []

            for tree, current_cost in candidate_trees:
                unadded = destinations - tree['covered_dests']
                if not unadded:
                    final_completed.append((tree, current_cost))
                    continue

                target_dest = self._select_next_dest(tree['nodes'], unadded)
                if not target_dest: continue

                paths = self.path_engine.get_paths(tree['nodes'], target_dest, k=5)
                if not paths: continue

                path_candidates = [(p['hops'], p) for p in paths]
                path_candidates.sort(key=lambda x: x[0])

                for _, path in path_candidates[:3]:
                    self.stats['trees_evaluated'] += 1
                    branch_state = self._fast_state_copy(network_state)

                    branch_tree = {
                        'tree': tree['tree'].copy(),
                        'hvt': tree['hvt'].copy(),
                        'nodes': tree['nodes'].copy(),
                        'paths_map': tree['paths_map'].copy(),
                        'covered_dests': tree['covered_dests'].copy(),
                        'link_count': tree['link_count'],
                        'node_count': tree['node_count'],
                        'structure_hash': tree.get('structure_hash', 0),
                        'sum_cpu': tree.get('sum_cpu', 0.0),
                        'sum_mem': tree.get('sum_mem', 0.0),
                        'sum_bw': tree.get('sum_bw', 0.0),
                        'traj': list(tree.get('traj', []))  # ✅ Deep copy trajectory
                    }

                    existing_hvt = branch_tree['hvt'].copy()
                    cpu_delta = np.zeros(self.node_num)
                    mem_delta = np.zeros(self.node_num)
                    vnf_delta = np.zeros((self.node_num, self.type_num))

                    vnf_types = request.get('vnf', [])
                    cpu_reqs = request.get('cpu_origin', [])
                    mem_reqs = request.get('memory_origin', [])

                    placement_res = self.placement_strategy.place_vnf_chain(
                        vnf_types, cpu_reqs, mem_reqs, path['nodes'],
                        existing_hvt, cpu_delta, mem_delta, vnf_delta, branch_state
                    )

                    if not placement_res: continue

                    resource_delta = {'cpu': cpu_delta, 'mem': mem_delta, 'vnf': vnf_delta}
                    info = {'nodes': path['nodes'], 'links': path.get('links'), 'hvt': vnf_delta}

                    success = self._apply_path_to_tree(
                        branch_tree, info, request, branch_state,
                        real_deploy=True, resource_delta=resource_delta
                    )

                    if not success: continue

                    d_node = float(np.sum(cpu_delta) + np.sum(mem_delta))
                    d_links = branch_tree['link_count'] - tree['link_count']
                    d_bw = float(d_links * request.get('bw_origin', 0))

                    branch_tree['sum_cpu'] += float(np.sum(cpu_delta))
                    branch_tree['sum_mem'] += float(np.sum(mem_delta))
                    branch_tree['sum_bw'] += d_bw

                    new_cost = current_cost + (self.alpha * d_node + self.gamma * d_bw)
                    branch_tree['covered_dests'].add(target_dest)

                    # ✅ Record Trajectory Step
                    # Step format: (dest_idx, action_data, res_delta)
                    try:
                        d_idx = request['dest'].index(target_dest)
                        action_data = {
                            'path': path['nodes'],
                            'links': path.get('links', []),
                            'placement': placement_res
                        }
                        branch_tree['traj'].append((d_idx, action_data, resource_delta))
                    except ValueError:
                        pass  # Should not happen

                    for u, v in path['edges']:
                        edge_hash = hash(tuple(sorted((u, v))))
                        branch_tree['structure_hash'] = (branch_tree['structure_hash'] * 1315423911 ^ edge_hash)

                    if branch_tree['structure_hash'] in seen_trees:
                        self.stats['trees_pruned_hash'] += 1
                        continue
                    seen_trees.add(branch_tree['structure_hash'])

                    res_vec = np.array([branch_tree['sum_cpu'], branch_tree['sum_mem'], branch_tree['sum_bw']])
                    dest_key = (frozenset(branch_tree['covered_dests']), branch_tree['link_count'])

                    if pareto_map[dest_key].is_dominated(res_vec, new_cost):
                        self.stats['trees_pruned_pareto'] += 1
                        continue
                    pareto_map[dest_key].insert(res_vec, new_cost)

                    next_candidates.append((branch_tree, new_cost))

            if not next_candidates: break

            next_candidates.sort(key=lambda x: x[1])
            candidate_trees = next_candidates[:self.beam_size]
            self.stats['trees_pruned_beam'] += (len(next_candidates) - len(candidate_trees))

        all_candidates = final_completed + candidate_trees
        if not all_candidates:
            return None, [], list(request['dest'])

        all_candidates.sort(key=lambda x: (-len(x[0]['covered_dests']), x[1]))
        best_tree, _ = all_candidates[0]

        final_tree: Dict[str, Any] = best_tree
        final_tree['added_dest_indices'] = [request['dest'].index(d) for d in final_tree['covered_dests']]
        # ✅ Trajectory is now populated in final_tree['traj']

        # ⚠️ CRITICAL FIX: The second return value must be the trajectory list
        return final_tree, final_tree['traj'], list(destinations - final_tree['covered_dests'])

    def get_stats(self):
        return self.stats.copy()