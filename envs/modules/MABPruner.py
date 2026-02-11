# -*- coding: utf-8 -*-
"""
MABPruner.py - 修复版
====================
修复内容：
1. _release_current_resources_temporarily() - 修复资源泄漏Bug
2. _try_reserve_resources() - 添加详细日志
3. Essential边提取 - 增强placement解析
"""
import logging
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set

logger = logging.getLogger(__name__)


# ============================================================================
# 内部策略类: MABPruningHelper
# ============================================================================
class MABPruningHelper:
    """
    MAB辅助剪枝模块 (策略核心)
    负责管理边的统计信息 (Play Count, Avg Reward) 并执行 UCB1 选择
    """

    def __init__(self, exploration_param=1.4, policy='ucb1'):
        self.exploration_param = exploration_param
        self.policy = policy
        self.edge_stats = {}  # {(u, v): {'n': 0, 'mu': 0.0, ...}}
        self.global_stats = {
            'total_evaluations': 0,
            'successful_prunings': 0,
            'total_reward': 0.0
        }

    def reset(self):
        self.edge_stats.clear()
        self.global_stats = {'total_evaluations': 0, 'successful_prunings': 0, 'total_reward': 0.0}

    def _normalize_edge(self, edge: Tuple[int, int]) -> Tuple[int, int]:
        """归一化边：确保 (u, v) 和 (v, u) 统一为 (min, max)"""
        return tuple(sorted(edge))

    def initialize_edges(self, candidate_edges: Set[Tuple[int, int]]):
        for edge in candidate_edges:
            edge_key = self._normalize_edge(edge)
            if edge_key not in self.edge_stats:
                self.edge_stats[edge_key] = {
                    'n': 0, 'mu': 0.0, 'total_reward': 0.0,
                    'alpha': 1.0, 'beta': 1.0  # Beta分布参数
                }

    def select_edge(self, candidate_edges: Set[Tuple[int, int]], total_global_steps: int) -> Optional[Tuple[int, int]]:
        candidates = [self._normalize_edge(e) for e in candidate_edges]
        if not candidates:
            return None

        best_edge = None
        best_ucb = -np.inf

        for edge in candidates:
            stats = self.edge_stats.get(edge)
            if not stats: continue

            n_i = stats['n']
            mu_i = stats['mu']

            if n_i == 0:
                return edge  # 优先探索

            # UCB1 公式
            ucb_value = mu_i + self.exploration_param * np.sqrt(np.log(total_global_steps + 1) / n_i)

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_edge = edge

        return best_edge

    def update_edge_reward(self, edge: Tuple[int, int], reward: float, total_steps: int):
        edge_key = self._normalize_edge(edge)
        if edge_key not in self.edge_stats: return

        stats = self.edge_stats[edge_key]
        stats['n'] += 1
        stats['total_reward'] += reward
        stats['mu'] = stats['total_reward'] / stats['n']

        # 更新全局统计
        self.global_stats['total_evaluations'] += 1
        self.global_stats['total_reward'] += reward
        if reward > 0:
            self.global_stats['successful_prunings'] += 1

    def compute_reward(self, **kwargs):
        """计算剪枝奖励"""
        # 1. 约束检查
        if not kwargs.get('constraints_satisfied', True):
            return -5.0  # 剪断了关键路径，重罚

        # 2. 计算边数节省
        size_before = 0
        size_after = 0

        if 'tree_before' in kwargs: size_before = len(kwargs['tree_before'])
        if 'tree_after' in kwargs: size_after = len(kwargs['tree_after'])

        edges_saved = size_before - size_after
        bw_unit = kwargs.get('bw_req', 1.0)

        # 3. 奖励公式
        reward = 1.0 * (edges_saved * bw_unit)
        if edges_saved > 0:
            reward += 0.5  # 成功剪枝的额外激励

        return reward

    def print_stats(self):
        s = self.global_stats
        logger.info(
            f"📊 [MAB Stats] Eval: {s['total_evaluations']}, Success: {s['successful_prunings']}, Reward: {s['total_reward']:.2f}")


# ============================================================================
# 主 Mixin 类: PruningModuleMixin
# ============================================================================
class PruningModuleMixin:
    """
    🔥 [剪枝功能模块 Mixin]
    集成 MAB 智能剪枝策略与资源结算逻辑。

    使用方法:
    1. 在主环境类中继承此 Mixin。
    2. 在 __init__ 中调用 self.init_pruning_module()。
    3. 在 step() 结束或需要结算时调用 self._finalize_request_with_pruning()。

    依赖属性 (由主类提供):
    - self.current_request (dict)
    - self.current_tree (dict)
    - self.resource_mgr (ResourceManager 实例)
    """

    def init_pruning_module(self, use_mab: bool = True, mab_rounds: int = 10, exploration: float = 1.4):
        """初始化剪枝模块"""
        self.use_mab_pruning = use_mab
        self.mab_rounds = mab_rounds

        # 实例化策略核心
        self.mab_pruner = MABPruningHelper(exploration_param=exploration)

        # 本地动作统计
        self.mab_action_stats = {
            'total_selections': 0, 'positive_rewards': 0,
            'negative_rewards': 0, 'successful_prunes': 0
        }
        logger.info(f"✅ [PruningModule] 初始化完成: MAB={use_mab}")

    def _finalize_request_with_pruning(self) -> bool:
        """
        🔥 [核心入口] 结算请求：释放 -> 剪枝 -> 预留 -> 提交
        """
        if self.current_request is None: return False

        req_id = self.current_request.get('id', 'unknown')
        logger.info(f"🔄 [结算流程] 请求 {req_id} 开始结算 (MAB模式: {self.use_mab_pruning})")

        # 1. 临时释放资源 (为了重新整理)
        self._release_current_resources_temporarily()

        # 2. 执行剪枝逻辑 (尝试 Plan A)
        try:
            pruned_tree, valid_nodes, prune_success, mab_info = self._prune_redundant_branches_with_vnf_mab()
        except Exception as e:
            logger.error(f"❌ MAB剪枝异常，回退传统模式: {e}")
            pruned_tree, valid_nodes, prune_success, _ = self._prune_redundant_branches_with_vnf()

        # 3. 开启事务进行预留
        tx_id = self.resource_mgr.begin_transaction(req_id)
        final_tree = None
        plan_success = False

        try:
            # --- Plan A: 尝试剪枝后的树 ---
            if prune_success:
                logger.info("尝试 Plan A (剪枝方案)...")
                if self._try_reserve_resources(tx_id, self.current_tree.get('placement', {}), pruned_tree, valid_nodes):
                    final_tree = pruned_tree
                    plan_success = True
                    logger.info("✅ Plan A 资源预留成功")
                else:
                    logger.warning("⚠️ Plan A 失败，回滚并准备 Plan B")
                    self.resource_mgr.rollback_transaction(tx_id)
                    tx_id = self.resource_mgr.begin_transaction(req_id)  # 开启新事务

            # --- Plan B: 回退原始树 (如果 Plan A 失败) ---
            if not plan_success:
                logger.info("尝试 Plan B (原始方案)...")
                current_tree_edges = self.current_tree.get('tree', {})
                # 获取原始树的所有节点作为 valid_nodes
                all_nodes = set()
                for u, v in current_tree_edges.keys(): all_nodes.update([u, v])

                if self._try_reserve_resources(tx_id, self.current_tree.get('placement', {}), current_tree_edges,
                                               all_nodes):
                    final_tree = current_tree_edges
                    logger.info("✅ Plan B (原始方案) 资源预留成功")
                else:
                    raise Exception("❌ Plan B 也失败 (资源不足或冲突)")

            # 4. 提交事务
            if self.resource_mgr.commit_transaction(tx_id):
                self.current_tree['tree'] = final_tree
                # 如果主类有归档方法则调用
                if hasattr(self, '_archive_request'): self._archive_request(success=True)

                # 打印 MAB 统计
                if self.use_mab_pruning: self.mab_pruner.print_stats()
                return True
            else:
                logger.error("❌ 事务提交失败")
                return False

        except Exception as e:
            logger.error(f"❌ [结算崩溃] {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.resource_mgr.rollback_transaction(tx_id)
            return False

    def _prune_redundant_branches_with_vnf_mab(self):
        """
        🔥 [MAB增强版] 剪枝冗余分支
        Phase 1: 反向回溯提取Essential边
        Phase 2: MAB迭代剪除候选边
        Phase 3: 构建最终树
        """
        if not self.current_request:
            return {}, set(), False, {}

        req = self.current_request
        source = req.get('source')
        current_tree_edges = self.current_tree.get('tree', {})
        placement = self.current_tree.get('placement', {})
        bw_req = req.get('bw_origin', 1.0)

        if not current_tree_edges:
            return {}, {source}, False, {}

        # === Phase 1: Essential边提取 (BFS + 反向回溯) ===
        adj = defaultdict(list)
        for u, v in current_tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        parent = {source: None}
        queue = deque([source])
        visited = {source}

        while queue:
            curr = queue.popleft()
            for nbr in adj.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = curr
                    queue.append(nbr)

        # 🔥 修复：增强placement解析
        targets = set(req.get('dest', []))
        for k, info in placement.items():
            node_id = None

            # 优先从info字典获取
            if isinstance(info, dict):
                node_id = info.get('node')

            # 回退到key解析
            if node_id is None:
                if isinstance(k, tuple) and len(k) >= 1:
                    node_id = k[0]
                elif isinstance(k, int):
                    node_id = k

            if node_id is not None:
                targets.add(node_id)
            else:
                logger.warning(f"⚠️ 无法解析placement节点: key={k}, info={info}")

        essential_edges = set()
        valid_nodes = {source}
        critical_nodes = targets.copy()

        for t in targets:
            if t not in visited:
                logger.warning(f"⚠️ 目标节点 {t} 不可达")
                continue

            curr = t
            valid_nodes.add(curr)
            while curr != source and curr in parent:
                p = parent[curr]
                if p is None: break
                # 使用 helper 归一化
                edge = self.mab_pruner._normalize_edge((p, curr))
                essential_edges.add(edge)
                valid_nodes.add(p)
                curr = p

        # === Phase 2: MAB 剪枝 ===
        # 如果未启用 MAB，直接返回 Essential Tree
        if not self.use_mab_pruning:
            pruned_tree = {k: v for k, v in current_tree_edges.items()
                           if self.mab_pruner._normalize_edge(k) in essential_edges}
            return pruned_tree, valid_nodes, True, {'method': 'backward_only'}

        # 确定候选边 (所有边 - Essential)
        all_edges_norm = {self.mab_pruner._normalize_edge(e) for e in current_tree_edges.keys()}
        candidate_edges = all_edges_norm - essential_edges

        if not candidate_edges:
            pruned_tree = {k: v for k, v in current_tree_edges.items()
                           if self.mab_pruner._normalize_edge(k) in essential_edges}
            return pruned_tree, valid_nodes, True, {'method': 'no_candidates'}

        # MAB 循环
        self.mab_pruner.initialize_edges(candidate_edges)
        edges_to_keep = set(candidate_edges)  # 初始保留所有
        edges_removed = set()

        original_tree_dict = current_tree_edges.copy()

        for _ in range(self.mab_rounds):
            if not edges_to_keep: break

            # MAB 选择
            selected_edge = self.mab_pruner.select_edge(
                edges_to_keep, self.mab_action_stats['total_selections']
            )
            if not selected_edge: break

            self.mab_action_stats['total_selections'] += 1

            # 模拟剪除
            temp_tree = {}
            for (u, v), data in original_tree_dict.items():
                edge_norm = self.mab_pruner._normalize_edge((u, v))
                # 保留规则: 是 Essential 或 (是候选边 且 不是当前选中的)
                if edge_norm in essential_edges or (edge_norm in candidate_edges and edge_norm != selected_edge):
                    temp_tree[(u, v)] = data

            # 验证连通性
            is_connected = self._verify_tree_connectivity(temp_tree, source, critical_nodes)

            # 计算奖励
            reward = self.mab_pruner.compute_reward(
                tree_before=original_tree_dict,
                tree_after=temp_tree,
                bw_req=bw_req,
                constraints_satisfied=is_connected
            )

            # 更新 MAB
            self.mab_pruner.update_edge_reward(selected_edge, reward, self.mab_action_stats['total_selections'])

            if reward > 0:
                self.mab_action_stats['positive_rewards'] += 1
                edges_removed.add(selected_edge)
                edges_to_keep.remove(selected_edge)  # 确认剪除，从候选池移除
            else:
                self.mab_action_stats['negative_rewards'] += 1
                # 负奖励保留边，这里简单处理：不移除 edge_to_keep，但因为 UCB 值低，下轮可能不会选
                pass

        # === Phase 3: 构建最终树 ===
        final_tree = {}
        final_valid_nodes = {source}

        for (u, v), data in current_tree_edges.items():
            edge_norm = self.mab_pruner._normalize_edge((u, v))
            should_keep = False

            if edge_norm in essential_edges:
                should_keep = True
            elif edge_norm in candidate_edges:
                if edge_norm not in edges_removed:
                    should_keep = True

            if should_keep:
                final_tree[(u, v)] = data
                final_valid_nodes.add(u);
                final_valid_nodes.add(v)

        return final_tree, final_valid_nodes, True, {
            'method': 'mab_enhanced', 'removed': len(edges_removed), 'final_edges': len(final_tree)
        }

    def _prune_redundant_branches_with_vnf(self):
        """
        [兜底] 传统反向回溯剪枝 (只保留 Essential)
        """
        if not self.current_request: return {}, set(), False, {}

        # 简化逻辑：复用 Phase 1 代码 (为了代码复用，实际可以直接调用 _prune_mab 并强制 use_mab=False)
        # 这里为了独立性，快速实现一遍 BFS
        req = self.current_request
        source = req.get('source')
        current_tree = self.current_tree.get('tree', {})
        placement = self.current_tree.get('placement', {})

        if not current_tree: return {}, {source}, False, {}

        adj = defaultdict(list)
        for u, v in current_tree.keys(): adj[u].append(v); adj[v].append(u)

        parent = {source: None};
        queue = deque([source]);
        visited = {source}
        while queue:
            curr = queue.popleft()
            for nbr in adj.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr);
                    parent[nbr] = curr;
                    queue.append(nbr)

        # 关键节点
        targets = set(req.get('dest', []))
        for k in placement: targets.add(k[0] if isinstance(k, tuple) else k)

        essential_edges = set()
        valid_nodes = {source}
        for t in targets:
            if t not in visited: continue
            curr = t
            valid_nodes.add(curr)
            while curr != source and curr in parent:
                p = parent[curr]
                if p is None: break
                # 🔥 修复：统一使用归一化方法
                essential_edges.add(self.mab_pruner._normalize_edge((p, curr)))
                valid_nodes.add(p)
                curr = p

        final_tree = {k: v for k, v in current_tree.items()
                      if self.mab_pruner._normalize_edge(k) in essential_edges}
        return final_tree, valid_nodes, True, {}

    def _try_reserve_resources(self, tx_id, placement, tree_edges, valid_nodes):
        """
        🔥 修复版：尝试预留资源，添加详细日志
        """
        req = self.current_request
        bw = req.get('bw_origin', 1.0)

        # 1. 预留节点资源
        reserved_nodes = 0
        for key, info in placement.items():
            # 解析 Key/Value
            node_id, vnf_type = None, None
            cpu, mem = 1.0, 1.0

            # 尝试从 info 字典获取
            if isinstance(info, dict):
                node_id = info.get('node')
                vnf_type = info.get('vnf_type')
                cpu = info.get('cpu_used', 1.0)
                mem = info.get('mem_used', 1.0)

            # 尝试从 key 获取 (如果 info 解析失败)
            if node_id is None and isinstance(key, tuple):
                node_id, vnf_type = key[0], key[1]

            # 校验
            if node_id is None or node_id not in valid_nodes: continue

            # 🔥 修复：添加详细日志
            if not self.resource_mgr.reserve_node_resource(tx_id, node_id, vnf_type, cpu, mem):
                logger.warning(f"❌ 预留节点资源失败: node={node_id}, vnf={vnf_type}, "
                               f"cpu={cpu:.1f}, mem={mem:.1f}")
                return False

            reserved_nodes += 1

        logger.debug(f"✅ 预留了 {reserved_nodes} 个节点资源")

        # 2. 预留链路资源
        reserved_links = 0
        for (u, v) in tree_edges.keys():
            # 🔥 修复：添加详细日志
            if not self.resource_mgr.reserve_link_resource(tx_id, u, v, bw):
                logger.warning(f"❌ 预留链路资源失败: {u}-{v}, bw={bw:.1f}")
                return False

            reserved_links += 1

        logger.debug(f"✅ 预留了 {reserved_links} 条链路资源")

        return True

    def _verify_tree_connectivity(self, tree_edges, source, critical_nodes):
        """验证树连通性"""
        if not tree_edges: return False
        adj = defaultdict(list)
        for u, v in tree_edges.keys(): adj[u].append(v); adj[v].append(u)

        visited = set()
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node in visited: continue
            visited.add(node)
            for nbr in adj.get(node, []):
                if nbr not in visited: queue.append(nbr)

        return all(n in visited for n in critical_nodes)

    def _release_current_resources_temporarily(self):
        """
        🔥🔥🔥 修复版：临时释放资源
        修复内容：正确读取并释放实际占用的资源量
        """
        tree = self.current_tree.get('tree', {})
        placement = self.current_tree.get('placement', {})
        bw = self.current_request.get('bw_origin', 1.0)

        # 释放链路资源
        released_links = 0
        for u, v in tree.keys():
            self.resource_mgr.release_link_resource(u, v, bw)
            released_links += 1

        # 🔥 修复：释放节点资源 - 读取实际占用量
        released_nodes = 0
        for key, info in placement.items():
            node_id = None
            vnf_type = 0
            cpu_used = 1.0
            mem_used = 1.0

            # Step 1: 优先从info字典获取完整信息
            if isinstance(info, dict):
                node_id = info.get('node')
                vnf_type = info.get('vnf_type', 0)
                cpu_used = info.get('cpu_used', 1.0)  # 🔥 从info读取实际值
                mem_used = info.get('mem_used', 1.0)  # 🔥 从info读取实际值

            # Step 2: 如果info没有node，尝试从key获取
            if node_id is None:
                if isinstance(key, tuple):
                    if len(key) >= 2:
                        node_id, vnf_type = key[0], key[1]
                    elif len(key) >= 1:
                        node_id = key[0]
                elif isinstance(key, int):
                    node_id = key

            # Step 3: 如果资源量是默认值，尝试从请求中获取
            if cpu_used == 1.0 and mem_used == 1.0:
                cpu_list = self.current_request.get('cpu_origin', [])
                mem_list = self.current_request.get('memory_origin', [])

                if isinstance(vnf_type, int) and vnf_type < len(cpu_list):
                    cpu_used = cpu_list[vnf_type]
                    mem_used = mem_list[vnf_type] if vnf_type < len(mem_list) else 1.0

            # Step 4: 执行释放
            if node_id is not None:
                logger.debug(f"释放节点资源: node={node_id}, vnf={vnf_type}, "
                             f"cpu={cpu_used:.1f}, mem={mem_used:.1f}")
                self.resource_mgr.release_node_resource(node_id, vnf_type, cpu_used, mem_used)
                released_nodes += 1
            else:
                logger.warning(f"⚠️ 无法解析placement: key={key}, info={info}")

        logger.info(f"♻️ [临时释放] 节点: {released_nodes}, 链路: {released_links}")