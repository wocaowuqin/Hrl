# backup_policy_fixed.py
"""
修复版 BackupPolicy

修复内容:
1. never_fail 策略添加资源检查
2. 改进 set_current_tree 处理邻接关系
3. 添加更多调试信息
4. 优化策略选择逻辑
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from envs.modules.sfc_backup_system.utils import ensure_list, build_tree_vec, build_hvt_from_placement
from envs.modules.sfc_backup_system.tree_cache import TreeCache
from envs.modules.sfc_backup_system.path_finder import PathFinder
from envs.modules.sfc_backup_system.path_eval import evaluate_path_score
from envs.modules.sfc_backup_system.vnf_placement import VNFPlacement

logger = logging.getLogger(__name__)


class BackupPolicy:
    """
    完整增强版 BackupPolicy (修复版)
    """

    def __init__(self, expert, n: int, L: int, K_vnf: int,
                 dc_nodes: Optional[List[int]] = None):

        self.expert = expert
        self.n = n
        self.L = L
        self.K_vnf = K_vnf

        self.dc_nodes = set(dc_nodes or [])
        self.current_tree: Dict = {"nodes": [], "adjacency": {}}
        self.current_request: Dict = {}

        self.path_finder = PathFinder(expert, n)
        self.tree_cache = TreeCache()

        # 性能统计
        self.stats = {
            "total_calls": 0,
            "strategy_success": {
                "resource_aware": 0,
                "smart_greedy": 0,
                "minimal": 0,
                "never_fail": 0
            },
            "total_failures": 0,
            "avg_score": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            # 🔧 新增: 详细失败原因
            "failure_reasons": {}
        }

    # --------------------------
    # 🔧 修复1: 改进的 Tree 设置方法
    # --------------------------
    def set_current_tree(self, tree):
        """
        设置当前树结构

        支持多种输入格式:
        1. List[int] - 节点列表
        2. Set[int] - 节点集合
        3. Dict - 完整树结构 {"nodes": [...], "adjacency": {...}}
        """
        if tree is None:
            self.current_tree = {"nodes": [], "adjacency": {}}
            logger.warning("BackupPolicy: set_current_tree received None")
            return

        # 处理不同输入类型
        if isinstance(tree, (list, set)):
            nodes = list(tree)
            self.current_tree = {
                "nodes": nodes,
                "adjacency": self._infer_adjacency(nodes)  # 🔧 新增: 推断邻接关系
            }
        elif isinstance(tree, dict):
            self.current_tree = {
                "nodes": list(tree.get("nodes", [])),
                "adjacency": dict(tree.get("adjacency", {})),
                "root": tree.get("root")
            }
        else:
            logger.warning(f"BackupPolicy: Unknown tree type {type(tree)}")
            self.current_tree = {"nodes": [], "adjacency": {}}

        # 缓存失效
        self.tree_cache.invalidate()

        logger.debug(f"[BackupPolicy] Set tree: {len(self.current_tree['nodes'])} nodes")

    def _infer_adjacency(self, nodes: List[int]) -> Dict[int, List[int]]:
        """
        🔧 新增: 从节点列表推断简单的线性邻接关系
        这是一个简化版本，假设节点按路径顺序排列
        """
        if len(nodes) <= 1:
            return {}

        adjacency = {}
        for i, node in enumerate(nodes):
            neighbors = []
            if i > 0:
                neighbors.append(nodes[i - 1])
            if i < len(nodes) - 1:
                neighbors.append(nodes[i + 1])
            if neighbors:
                adjacency[node] = neighbors

        return adjacency

    def set_current_request(self, request: Dict):
        self.current_request = request or {}

    def update_tree(self, tree):
        self.set_current_tree(tree)

    def update_request(self, request: Dict):
        self.set_current_request(request)

    # --------------------------
    # 工具函数
    # --------------------------
    def _is_dc_node(self, nd: int) -> bool:
        if self.dc_nodes:
            return int(nd) in self.dc_nodes
        return int(nd) % 3 == 0

    def _select_top_k(self, node_list: List[int], dst: int, k: int = 5) -> List[int]:
        if not node_list:
            return []
        if len(node_list) <= k:
            return list(node_list)

        scored = []
        adjacency = self.current_tree.get("adjacency", {})

        for nd in node_list:
            cost = self.tree_cache.get(nd, dst)

            if cost is None:
                self.stats["cache_misses"] += 1
                cost = self._bfs_tree_distance(nd, dst, adjacency)
                self.tree_cache.set(nd, dst, cost)
            else:
                self.stats["cache_hits"] += 1

            scored.append((cost, nd))

        scored.sort(key=lambda x: x[0])
        return [nd for _, nd in scored[:k]]

    def _bfs_tree_distance(self, src: int, dst: int, adjacency: Dict) -> int:
        if src == dst:
            return 0
        if not adjacency or src not in adjacency:
            return 999

        q = deque([(src, 0)])
        visited = {src}
        while q:
            u, dist = q.popleft()
            if u == dst:
                return dist
            for v in adjacency.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append((v, dist + 1))
        return 999

    # --------------------------
    # 网络状态标准化
    # --------------------------
    def _prepare_network_state(self, network_state: Dict) -> Dict:
        state = dict(network_state or {})
        state.setdefault("cpu", {})
        state.setdefault("mem", {})
        state.setdefault("bw", {})
        state.setdefault("hvt", {})

        req = self.current_request
        state.setdefault("cpu_demand", ensure_list(req.get("cpu_origin", [])))
        state.setdefault("mem_demand", ensure_list(req.get("memory_origin", [])))
        state.setdefault("bw_demand", float(req.get("bw_origin", 0)))
        state.setdefault("vnf_seq", ensure_list(req.get("vnf") or req.get("vnf_list") or []))
        state.setdefault("tree_nodes", list(self.current_tree.get("nodes", [])))
        state.setdefault("source", int(req.get("source", 1)))
        return state

    # --------------------------
    # 🔧 修复2: 增强的资源检查
    # --------------------------
    def _check_bandwidth_feasible(self, links: List[int], state: Dict) -> bool:
        """检查链路带宽是否满足需求"""
        bw_resources = state.get('bw')
        bw_demand = state.get("bw_demand", 0)

        if bw_demand is None or bw_demand <= 1e-6:
            return True

        if bw_resources is None:
            return True  # 没有资源信息时保守通过

        for lid in links:
            idx = int(lid) - 1

            if isinstance(bw_resources, (np.ndarray, list)):
                if idx < 0 or idx >= len(bw_resources):
                    continue
                avail_bw = float(bw_resources[idx])
            elif isinstance(bw_resources, dict):
                avail_bw = float(bw_resources.get(idx, 0))
            else:
                continue

            if avail_bw < bw_demand:
                logger.debug(f"[BW Check] Link {lid}: avail={avail_bw:.2f} < demand={bw_demand:.2f}")
                return False

        return True

    def _check_node_resources_feasible(self, nodes: List[int], placement: Dict[int, int], state: Dict) -> bool:
        """检查节点资源"""
        cpu_demand = state.get("cpu_demand", [])
        mem_demand = state.get("mem_demand", [])
        cpu_res = state.get("cpu", {})
        mem_res = state.get("mem", {})

        for vnf_idx, node in placement.items():
            node_idx = int(node) - 1
            if vnf_idx >= len(cpu_demand) or vnf_idx >= len(mem_demand):
                continue

            cpu_need = cpu_demand[vnf_idx]
            mem_need = mem_demand[vnf_idx]

            cpu_avail = cpu_res.get(node_idx, 0) if isinstance(cpu_res, dict) else (
                cpu_res[node_idx] if node_idx < len(cpu_res) else 0
            )
            mem_avail = mem_res.get(node_idx, 0) if isinstance(mem_res, dict) else (
                mem_res[node_idx] if node_idx < len(mem_res) else 0
            )

            if cpu_avail < cpu_need or mem_avail < mem_need:
                logger.debug(
                    f"[Node Check] Node {node}: cpu={cpu_avail:.2f}<{cpu_need:.2f} or mem={mem_avail:.2f}<{mem_need:.2f}")
                return False
        return True

    # --------------------------
    # 策略 1: 资源感知
    # --------------------------
    def _resource_aware_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes:
            return {"feasible": False, "reason": "no_tree_nodes"}

        candidates = self._select_top_k(tree_nodes, dst, k=8)

        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links:
                continue

            if not self._check_bandwidth_feasible(links, state):
                continue

            placement = VNFPlacement.resource_aware(state["vnf_seq"], nodes, state, self._is_dc_node)
            if placement is None:
                continue

            if not self._check_node_resources_feasible(nodes, placement, state):
                continue

            score = evaluate_path_score(nodes, links, state, self._is_dc_node)
            return {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": score}

        return {"feasible": False, "reason": "no_feasible_path"}

    # --------------------------
    # 策略 2: Smart Greedy
    # --------------------------
    def _smart_greedy_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes:
            return {"feasible": False, "reason": "no_tree_nodes"}

        candidates = self._select_top_k(tree_nodes, dst, k=6)
        best_plan = None
        best_score = -1e9

        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links:
                continue

            score = evaluate_path_score(nodes, links, state, self._is_dc_node)
            if score > best_score:
                best_score = score
                vnf_seq = state["vnf_seq"]
                placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}
                best_plan = {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": score}

        return best_plan or {"feasible": False, "reason": "no_path_found"}

    # --------------------------
    # 策略 3: Minimal
    # --------------------------
    def _minimal_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes:
            return {"feasible": False, "reason": "no_tree_nodes"}

        candidates = self._select_top_k(tree_nodes, dst, k=6)
        best_plan = None
        best_hops = float('inf')

        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links:
                continue

            hops = len(nodes) - 1
            if hops < best_hops:
                best_hops = hops
                vnf_seq = state["vnf_seq"]
                placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}
                best_plan = {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": -hops}

        return best_plan or {"feasible": False, "reason": "no_path_found"}

    # --------------------------
    # 🔧 修复3: 增强 Never-Fail 策略（添加资源检查）
    # --------------------------
    def _never_fail_strategy(self, dst: int, state: Dict) -> Dict:
        """多层降级策略 - 修复版"""
        src = state.get("source", 1)
        node_num = getattr(self.expert, "node_num", self.n)
        tree_nodes = state.get("tree_nodes", [])
        bw_demand = state.get("bw_demand", 0)

        # Layer 1: k-paths from tree nodes (带资源检查)
        for nd in tree_nodes:
            for k in range(1, self.path_finder.max_k + 1):
                try:
                    nodes, _, links = self.path_finder.get_k_path(nd, dst, k)

                    if nodes and links:
                        # 🔧 修复: 添加带宽检查
                        if bw_demand > 0 and not self._check_bandwidth_feasible(links, state):
                            continue

                        vnf_seq = state.get("vnf_seq", [])
                        placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}

                        return {
                            "feasible": True,
                            "nodes": nodes,
                            "links": links,
                            "placement": placement,
                            "score": -len(nodes)
                        }
                except Exception as e:
                    logger.debug(f"Layer1 failed: {e}")
                    pass

        # Layer 2: DC relay (带资源检查)
        dc_nodes = [n for n in range(1, node_num + 1) if self._is_dc_node(n) and n not in {src, dst}]
        for relay in dc_nodes[:10]:  # 限制搜索数量
            try:
                nodes, links = self.path_finder.compose_via_relay(src, relay, dst)
                if nodes and links:
                    # 🔧 修复: 添加带宽检查
                    if bw_demand > 0 and not self._check_bandwidth_feasible(links, state):
                        continue

                    vnf_seq = state.get("vnf_seq", [])
                    placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}

                    return {
                        "feasible": True,
                        "nodes": nodes,
                        "links": links,
                        "placement": placement,
                        "score": -1.2 * len(nodes)
                    }
            except Exception as e:
                logger.debug(f"Layer2 DC relay failed: {e}")
                pass

        # Layer 3: 任意中继 (最后尝试，仍然检查资源)
        for relay in range(1, min(node_num + 1, 30)):  # 限制搜索范围
            if relay in {src, dst}:
                continue
            try:
                nodes, links = self.path_finder.compose_via_relay(src, relay, dst)
                if nodes and links:
                    # 🔧 修复: 添加带宽检查
                    if bw_demand > 0 and not self._check_bandwidth_feasible(links, state):
                        continue

                    vnf_seq = state.get("vnf_seq", [])
                    placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}

                    return {
                        "feasible": True,
                        "nodes": nodes,
                        "links": links,
                        "placement": placement,
                        "score": -1.5 * len(nodes)
                    }
            except Exception as e:
                logger.debug(f"Layer3 relay failed: {e}")
                pass

        # 🔧 新增: Layer 4 - 无资源检查的最后尝试
        # 只在资源检查全部失败后才使用
        logger.warning(f"[Never-Fail] All resource-checked paths failed for dst={dst}, trying without check")

        for relay in range(1, min(node_num + 1, 20)):
            if relay in {src, dst}:
                continue
            try:
                nodes, links = self.path_finder.compose_via_relay(src, relay, dst)
                if nodes and links:
                    vnf_seq = state.get("vnf_seq", [])
                    placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}

                    return {
                        "feasible": True,
                        "nodes": nodes,
                        "links": links,
                        "placement": placement,
                        "score": -2.0 * len(nodes),  # 更低的分数
                        "warning": "no_resource_check"
                    }
            except Exception:
                pass

        return {"feasible": False, "nodes": [], "links": [], "placement": {}, "score": -1e9, "reason": "all_failed"}

    # --------------------------
    # 动态策略选择
    # --------------------------
    def _select_strategies(self, state: Dict) -> List[Tuple]:
        vnf_count = len(state.get("vnf_seq", []))
        cpu_res = state.get("cpu", {})

        # 计算平均CPU利用率
        if cpu_res is not None:
            if isinstance(cpu_res, dict):
                total_cpu = sum(cpu_res.values())
                max_cpu = len(cpu_res) * 2000
            else:
                total_cpu = sum(cpu_res)
                max_cpu = len(cpu_res) * 2000
            avg_cpu_util = 1.0 - (total_cpu / max(1, max_cpu))
        else:
            avg_cpu_util = 0.5

        # 根据VNF数量和资源利用率选择策略顺序
        if vnf_count >= 3 and avg_cpu_util > 0.7:
            # 资源紧张，优先资源感知
            return [
                (self._resource_aware_strategy, "resource_aware", 1.0),
                (self._smart_greedy_strategy, "smart_greedy", 0.8),
                (self._minimal_strategy, "minimal", 0.6)
            ]

        if vnf_count == 0 or avg_cpu_util < 0.3:
            # 资源充足，优先最短路径
            return [
                (self._minimal_strategy, "minimal", 1.0),
                (self._smart_greedy_strategy, "smart_greedy", 0.9),
                (self._resource_aware_strategy, "resource_aware", 0.7)
            ]

        # 默认：平衡策略
        return [
            (self._smart_greedy_strategy, "smart_greedy", 1.0),
            (self._resource_aware_strategy, "resource_aware", 0.9),
            (self._minimal_strategy, "minimal", 0.8)
        ]

    # --------------------------
    # 结果标准化
    # --------------------------
    def _normalize_plan(self, plan: Dict) -> Dict:
        if not plan.get("feasible", False):
            return {
                "feasible": False,
                "nodes": [],
                "links": [],
                "placement": {},
                "tree": np.zeros(self.L),
                "hvt": np.zeros((self.n, self.K_vnf)),
                "score": -1e9,
                "backup_type": plan.get("backup_type", "none"),
                "error": plan.get("reason", plan.get("error", "unknown"))
            }

        nodes = plan.get("nodes", [])
        links = plan.get("links", [])
        placement = plan.get("placement", {})

        return {
            "feasible": True,
            "nodes": nodes,
            "links": links,
            "placement": placement,
            "tree": build_tree_vec(links, self.L),
            "hvt": build_hvt_from_placement(placement, self.n, self.K_vnf),
            "new_path_full": nodes,
            "backup_type": plan.get("backup_type", "hybrid"),
            "score": float(plan.get("score", 0.0)),
            "warning": plan.get("warning")
        }

    # --------------------------
    # 主接口：get_backup_plan
    # --------------------------
    def get_backup_plan(self, goal_dest_idx: int, network_state: Dict[str, Any]) -> Dict[str, Any]:
        self.stats["total_calls"] += 1
        req = self.current_request

        if not req:
            self._record_failure("no_request")
            return {"feasible": False, "error": "no_request"}

        try:
            dest_list = req.get("dest", [])
            if goal_dest_idx < 0 or goal_dest_idx >= len(dest_list):
                raise IndexError(f"goal_dest_idx={goal_dest_idx} out of range [0, {len(dest_list)})")
            dst = int(dest_list[goal_dest_idx])
        except Exception as e:
            self._record_failure("invalid_dest")
            return {"feasible": False, "error": f"invalid_dest: {e}"}

        try:
            state = self._prepare_network_state(network_state)
        except Exception as e:
            self._record_failure("state_prep_failed")
            return {"feasible": False, "error": str(e)}

        # 尝试各策略
        strategies = self._select_strategies(state)

        for func, name, priority in strategies:
            try:
                plan = func(dst, state)
                if plan.get("feasible", False):
                    plan["score"] = plan.get("score", 0) * priority
                    plan["backup_type"] = name
                    self.stats["strategy_success"][name] += 1

                    # 更新平均分数
                    tc = self.stats["total_calls"]
                    prev_avg = self.stats["avg_score"]
                    self.stats["avg_score"] = (prev_avg * (tc - 1) + plan.get("score", 0)) / tc

                    logger.debug(f"[BackupPolicy] Strategy {name} succeeded for dst={dst}")
                    return self._normalize_plan(plan)
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")

        # Never Fail
        try:
            plan = self._never_fail_strategy(dst, state)
            plan["backup_type"] = "never_fail"
            if plan.get("feasible"):
                self.stats["strategy_success"]["never_fail"] += 1
            else:
                self._record_failure("never_fail_failed")
            return self._normalize_plan(plan)
        except Exception as e:
            self._record_failure(f"exception: {e}")
            return {"feasible": False, "error": str(e)}

    def _record_failure(self, reason: str):
        """记录失败原因"""
        self.stats["total_failures"] += 1
        self.stats["failure_reasons"][reason] = self.stats["failure_reasons"].get(reason, 0) + 1

    # --------------------------
    # 性能统计接口
    # --------------------------
    def get_statistics(self) -> Dict:
        stats = dict(self.stats)
        total = stats["total_calls"]
        if total > 0:
            stats["success_rate"] = 1.0 - (stats["total_failures"] / total)
            stats["cache_hit_rate"] = stats["cache_hits"] / max(1, stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
        return stats

    def reset_statistics(self):
        self.stats = {
            "total_calls": 0,
            "strategy_success": {"resource_aware": 0, "smart_greedy": 0, "minimal": 0, "never_fail": 0},
            "total_failures": 0,
            "avg_score": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failure_reasons": {}
        }

    def print_statistics(self):
        """打印详细统计"""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("BackupPolicy Statistics")
        print("=" * 50)
        print(f"Total Calls:    {stats['total_calls']}")
        print(f"Success Rate:   {stats['success_rate']:.2%}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"Avg Score:      {stats['avg_score']:.3f}")
        print("\nStrategy Success Counts:")
        for name, count in stats['strategy_success'].items():
            print(f"  {name}: {count}")
        print("\nFailure Reasons:")
        for reason, count in stats.get('failure_reasons', {}).items():
            print(f"  {reason}: {count}")
        print("=" * 50 + "\n")