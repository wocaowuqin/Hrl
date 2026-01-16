import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
class MABPruningHelper:
    """
    MAB辅助剪枝模块 (基于 Shanto2025 思想)

    职责：
    1. 管理候选边的MAB统计（play count, avg reward）
    2. 实现UCB1/Thompson Sampling选择策略
    3. 基于反馈更新边的统计信息
    """

    def __init__(self, exploration_param=1.4, policy='ucb1'):
        self.exploration_param = exploration_param
        self.policy = policy

        # 边的统计信息: {(u, v): {'n': play_count, 'mu': avg_reward}}
        self.edge_stats = {}

        # 历史记录 (用于调试或延迟分析)
        self.pruning_history = []
        self.global_stats = {
            'total_evaluations': 0,
            'successful_prunings': 0,
            'total_reward': 0.0
        }

        logger.info(f"✅ MABPruningHelper初始化: policy={policy}, exploration={exploration_param}")

    def reset(self) -> None:
        """重置所有统计"""
        self.edge_stats.clear()
        self.pruning_history.clear()

        # 🔥 重置 global_stats
        self.global_stats = {
            'total_evaluations': 0,
            'successful_prunings': 0,
            'total_reward': 0.0
        }

        logger.debug("MABPruningHelper重置")

    def initialize_edges(self, candidate_edges: Set[Tuple[int, int]]) -> None:
        """
        初始化候选边（完整版）

        Args:
            candidate_edges: 候选边集合
        """
        for edge in candidate_edges:
            edge_key = self._normalize_edge(edge)
            if edge_key not in self.edge_stats:
                self.edge_stats[edge_key] = {
                    # 基础统计
                    'n': 0,  # 尝试次数
                    'mu': 0.0,  # 平均奖励
                    'total_reward': 0.0,  # 累积奖励（用于计算mu）

                    # Beta分布参数（Thompson Sampling）
                    'alpha': 1.0,
                    'beta': 1.0,

                    # 辅助信息
                    'last_selected': 0,  # 最后选择时间
                    'successes': 0,  # 成功次数
                    'failures': 0,  # 失败次数
                }

        logger.debug(f"初始化{len(candidate_edges)}条候选边")

    def select_edge(self, candidate_edges: Set[Tuple[int, int]], total_global_steps: int) -> Optional[Tuple[int, int]]:
        """统一的选择入口"""
        candidates = [tuple(sorted(e)) for e in candidate_edges]
        if not candidates:
            return None

        if self.policy == 'thompson':
            return self._select_edge_thompson(candidates)
        else:
            return self._select_edge_ucb1(candidates, total_global_steps)

    def _select_edge_ucb1(self, candidates: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
        """UCB1策略: value = mu + c * sqrt(ln(t) / n)"""
        best_edge = None
        best_ucb = -np.inf

        for edge in candidates:
            stats = self.edge_stats[edge]
            n_i = stats['n']
            mu_i = stats['mu']

            if n_i == 0:
                # 优先探索未尝试的边 (赋予无穷大UCB)
                return edge

            # UCB1公式
            ucb_value = mu_i + self.exploration_param * np.sqrt(np.log(t + 1) / n_i)

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_edge = edge

        return best_edge

    def _select_edge_thompson(self, candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Thompson Sampling策略: Sample from Beta(alpha, beta)"""
        best_edge = None
        best_sample = -np.inf

        for edge in candidates:
            stats = self.edge_stats[edge]
            # 从Beta分布采样
            sample = np.random.beta(stats['alpha'], stats['beta'])

            if sample > best_sample:
                best_sample = sample
                best_edge = edge

        return best_edge

        # 在 class MABPruningHelper 中:

        # 🔥🔥🔥 [修改这里] 增加 step=None 参数以兼容调用 🔥🔥🔥

    def update_edge_reward(self, edge: Tuple[int, int], reward: float,
                           total_steps: int) -> None:
        """更新边的统计信息"""
        edge_key = self._normalize_edge(edge)
        if edge_key not in self.edge_stats:
            logger.warning(f"尝试更新未初始化的边: {edge_key}")
            return

        stats = self.edge_stats[edge_key]
        n = stats['n']

        # 更新计数和均值
        stats['n'] = n + 1
        stats['total_reward'] += reward
        stats['mu'] = stats['total_reward'] / stats['n']

        # 更新Beta分布参数
        normalized_reward = np.clip(reward, -1, 1)
        if normalized_reward > 0:
            stats['alpha'] += 1 + normalized_reward
        else:
            stats['beta'] += 1 - normalized_reward

        # 🔥 更新 global_stats
        self.global_stats['total_evaluations'] += 1
        self.global_stats['total_reward'] += reward
        if reward > 0:
            self.global_stats['successful_prunings'] += 1

        # 记录历史
        self.pruning_history.append({
            'edge': edge_key,
            'reward': reward,
            'play_count': stats['n'],
            'avg_reward': stats['mu'],
            'alpha': stats['alpha'],
            'beta': stats['beta'],
            'step': total_steps
        })
    def compute_reward(self, **kwargs):
        """
        智能参数适配：支持传入 (size, size) 或 (tree_dict, tree_dict)
        """
        # 1. 提取约束满足情况
        constraints_satisfied = kwargs.get('constraints_satisfied', True)
        if not constraints_satisfied:
            return -5.0

        # 2. 提取树的大小 (兼容两种调用方式)
        if 'tree_before_size' in kwargs and 'tree_after_size' in kwargs:
            # 方式 A: 直接传大小
            size_before = kwargs['tree_before_size']
            size_after = kwargs['tree_after_size']
        elif 'tree_before' in kwargs and 'tree_after' in kwargs:
            # 方式 B: 传字典 (你的代码目前是这种)
            size_before = len(kwargs['tree_before'])
            size_after = len(kwargs['tree_after'])
        else:
            # 兜底
            return 0.0

        # 3. 提取带宽单位 (兼容 bw_unit 和 bw_req)
        bw_unit = kwargs.get('bw_unit', kwargs.get('bw_req', 1.0))

        # 4. 计算奖励
        edges_saved = size_before - size_after
        reward = 1.0 * (edges_saved * bw_unit)

        # 额外奖励：如果成功减少了边，给予固定奖励鼓励
        if edges_saved > 0:
            reward += 0.5

        return reward
    def _normalize_edge(self, edge: Tuple[int, int]) -> Tuple[int, int]:
        """
        🔥 [补丁] 归一化边：确保 (u, v) 和 (v, u) 统一为 (min, max)
        """
        return tuple(sorted(edge))

    def print_stats(self) -> None:
        """打印统计信息"""
        if not self.edge_stats:
            logger.info("📊 MAB统计: 无数据")
            return

        # 🔥 使用 global_stats
        total_evaluations = self.global_stats.get('total_evaluations', 0)
        total_reward = self.global_stats.get('total_reward', 0.0)
        successful = self.global_stats.get('successful_prunings', 0)

        logger.info("=" * 60)
        logger.info("📊 MAB剪枝统计摘要:")
        logger.info(f"  总边数: {len(self.edge_stats)}")
        logger.info(f"  总尝试次数: {total_evaluations}")
        logger.info(f"  成功剪枝次数: {successful}")

        if total_evaluations > 0:
            success_rate = (successful / total_evaluations) * 100
            avg_reward = total_reward / total_evaluations
            logger.info(f"  成功率: {success_rate:.1f}%")
            logger.info(f"  平均奖励: {avg_reward:.3f}")

        logger.info(f"  历史记录数: {len(self.pruning_history)}")

        # Top edges
        try:
            top_edges = self.get_top_edges(5, 'mu')
            if top_edges:
                logger.info("\n  🏆 Top 5边 (按平均奖励):")
                for i, (edge, stats) in enumerate(top_edges):
                    logger.info(f"    {i + 1}. 边{edge}: n={stats['n']}, "
                                f"μ={stats['mu']:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ 打印Top边失败: {e}")

        logger.info("=" * 60)

    def get_top_edges(self, n: int = 5, by: str = 'mu') -> List[Tuple[Tuple[int, int], Dict]]:
        """
        获取排名前n的边

        Args:
            n: 返回前n个边
            by: 排序依据 ('mu' 按平均奖励, 'n' 按尝试次数)

        Returns:
            [(edge, stats), ...]: 排序后的边列表
        """
        if not self.edge_stats:
            return []

        if by == 'mu':
            # 按平均奖励排序
            sorted_edges = sorted(
                self.edge_stats.items(),
                key=lambda x: x[1].get('mu', 0),
                reverse=True
            )
        elif by == 'n':
            # 按尝试次数排序
            sorted_edges = sorted(
                self.edge_stats.items(),
                key=lambda x: x[1].get('n', 0),
                reverse=True
            )
        else:
            sorted_edges = list(self.edge_stats.items())

        return sorted_edges[:n]