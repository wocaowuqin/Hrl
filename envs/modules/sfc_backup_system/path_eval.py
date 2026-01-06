# sfc_backup_system/path_eval.py
import logging
from typing import List, Dict, Callable, Optional, Tuple  # ✅ 添加 Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PathEvaluator:
    """
    路径评分器 - 可配置的评分策略

    评分维度:
    1. 跳数 (hop_count)
    2. 平均带宽 (avg_bandwidth)
    3. 瓶颈带宽 (min_bandwidth)
    4. DC 节点覆盖率 (dc_ratio)
    5. 树节点重叠率 (overlap_ratio)
    6. 节点资源充足度 (node_resources)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: 评分权重字典,格式:
                {
                    'hop': -0.5,
                    'avg_bw': 0.02,
                    'min_bw': 0.03,
                    'dc_ratio': 0.15,
                    'overlap': 0.15,
                    'node_res': 0.1
                }
        """
        # ✅ 默认权重
        self.default_weights = {
            'hop': -0.5,  # 跳数惩罚
            'avg_bw': 0.02,  # 平均带宽奖励
            'min_bw': 0.03,  # 瓶颈带宽奖励 (更重要)
            'dc_ratio': 0.15,  # DC 覆盖率奖励
            'overlap': 0.15,  # 树重叠奖励
            'node_res': 0.1  # 节点资源奖励
        }

        # ✅ 合并用户权重
        self.weights = self.default_weights.copy()
        if weights:
            self.weights.update(weights)

        logger.info(f"[PathEvaluator] Initialized with weights: {self.weights}")

    def evaluate(self, nodes: List[int], links: List[int],
                 network_state: Dict, is_dc_fn: Callable[[int], bool]) -> float:
        """
        完整路径评分

        Args:
            nodes: 路径节点列表 (1-based)
            links: 路径链路列表 (1-based)
            network_state: 网络状态
            is_dc_fn: DC 节点判断函数

        Returns:
            评分 (越高越好)
        """
        if not nodes:
            return -1e9

        # ✅ 计算各个维度的分数
        metrics = self._compute_metrics(nodes, links, network_state, is_dc_fn)

        # ✅ 加权求和
        score = (
                self.weights['hop'] * metrics['hop_count'] +
                self.weights['avg_bw'] * metrics['avg_bw'] +
                self.weights['min_bw'] * metrics['min_bw'] +
                self.weights['dc_ratio'] * metrics['dc_ratio'] +
                self.weights['overlap'] * metrics['overlap_ratio'] +
                self.weights['node_res'] * metrics['node_res_score']
        )

        logger.debug(
            f"[PathEval] Score={score:.3f} | "
            f"hops={metrics['hop_count']}, "
            f"min_bw={metrics['min_bw']:.2f}, "
            f"dc_ratio={metrics['dc_ratio']:.2%}, "
            f"overlap={metrics['overlap_ratio']:.2%}"
        )

        return float(score)

    def _compute_metrics(self, nodes: List[int], links: List[int],
                         network_state: Dict, is_dc_fn: Callable[[int], bool]) -> Dict:
        """计算所有评分维度的指标"""

        metrics = {}

        # ===== 1. 跳数 =====
        metrics['hop_count'] = len(nodes) - 1

        # ===== 2. 带宽指标 =====
        bw_values = self._get_bandwidth_values(links, network_state)

        if bw_values:
            metrics['avg_bw'] = np.mean(bw_values)
            metrics['min_bw'] = np.min(bw_values)
            metrics['std_bw'] = np.std(bw_values)  # 带宽波动
        else:
            metrics['avg_bw'] = 0.0
            metrics['min_bw'] = 0.0
            metrics['std_bw'] = 0.0

        # ===== 3. DC 节点覆盖率 =====
        dc_count = sum(1 for nd in nodes if is_dc_fn(nd))
        metrics['dc_count'] = dc_count
        metrics['dc_ratio'] = dc_count / len(nodes)

        # ===== 4. 树节点重叠率 =====
        tree_nodes = set(network_state.get("tree_nodes", []))
        overlap_count = sum(1 for nd in nodes if nd in tree_nodes)
        metrics['overlap_count'] = overlap_count
        metrics['overlap_ratio'] = overlap_count / len(nodes)

        # ===== 5. 节点资源充足度 =====
        metrics['node_res_score'] = self._compute_node_resource_score(
            nodes, network_state
        )

        # ===== 6. VNF 需求适配度 =====
        vnf_num = len(network_state.get("vnf_seq", []))
        metrics['vnf_count'] = vnf_num
        metrics['vnf_dc_ratio'] = min(dc_count / max(1, vnf_num), 1.0)

        return metrics

    def _get_bandwidth_values(self, links: List[int], network_state: Dict) -> List[float]:
        """
        获取链路带宽值
        支持多种网络状态格式 (修复 NumPy 歧义报错)
        """
        # 尝试多种键名
        possible_keys = ['bw', 'link_resources', 'bandwidth', 'link_bw']

        bw_resources = None
        for key in possible_keys:
            res = network_state.get(key)
            # [关键修复] 使用 is not None 判断，避免 NumPy 数组报错
            if res is not None:
                bw_resources = res
                break

        # 如果没找到，或者找到的是空列表/数组
        if bw_resources is None or len(bw_resources) == 0:
            # logger.debug("[PathEval] No bandwidth resources found") # 可选日志
            return []

        bw_values = []
        for link_id in links:
            val = 0.0
            try:
                # 尝试处理 Dict
                if isinstance(bw_resources, dict):
                    # 优先尝试 0-based (通常代码内部用 0-based)
                    val = bw_resources.get(link_id - 1)
                    # 如果没拿到，尝试 1-based
                    if val is None:
                        val = bw_resources.get(link_id)

                    # 如果还是 None，默认为 0
                    if val is None:
                        val = 0.0

                # 尝试处理 List 或 NumPy Array
                else:
                    # 假设是 array-like，使用 0-based 索引
                    idx = int(link_id) - 1
                    if 0 <= idx < len(bw_resources):
                        val = bw_resources[idx]
                    else:
                        val = 0.0

                bw_values.append(float(val))

            except Exception:
                bw_values.append(0.0)

        return bw_values

    def _compute_node_resource_score(self, nodes: List[int], network_state: Dict) -> float:
        """
        计算节点资源充足度评分
        考虑 CPU 和内存的平均可用率
        """
        cpu_res = network_state.get("cpu", {})
        mem_res = network_state.get("mem", {})

        if not cpu_res and not mem_res:
            return 0.0

        cpu_scores = []
        mem_scores = []

        for node in nodes:
            node_idx = node - 1  # 转为 0-based

            # CPU 可用率
            if cpu_res:
                cpu_avail = cpu_res.get(node_idx, 0)
                cpu_capacity = 2000  # 默认容量 (可配置)
                cpu_scores.append(cpu_avail / cpu_capacity)

            # 内存可用率
            if mem_res:
                mem_avail = mem_res.get(node_idx, 0)
                mem_capacity = 1100  # 默认容量 (可配置)
                mem_scores.append(mem_avail / mem_capacity)

        # 综合评分
        all_scores = cpu_scores + mem_scores
        return float(np.mean(all_scores)) if all_scores else 0.0

    def evaluate_with_details(self, nodes: List[int], links: List[int],
                              network_state: Dict, is_dc_fn: Callable[[int], bool]) -> Dict:
        """
        返回详细的评分信息

        Returns:
            {
                'score': float,
                'metrics': Dict,
                'breakdown': Dict  # 各维度的加权贡献
            }
        """
        if not nodes:
            return {
                'score': -1e9,
                'metrics': {},
                'breakdown': {}
            }

        metrics = self._compute_metrics(nodes, links, network_state, is_dc_fn)

        # 计算各维度贡献
        breakdown = {
            'hop': self.weights['hop'] * metrics['hop_count'],
            'avg_bw': self.weights['avg_bw'] * metrics['avg_bw'],
            'min_bw': self.weights['min_bw'] * metrics['min_bw'],
            'dc_ratio': self.weights['dc_ratio'] * metrics['dc_ratio'],
            'overlap': self.weights['overlap'] * metrics['overlap_ratio'],
            'node_res': self.weights['node_res'] * metrics['node_res_score']
        }

        total_score = sum(breakdown.values())

        return {
            'score': float(total_score),
            'metrics': metrics,
            'breakdown': breakdown
        }

    def compare_paths(self, path1: Tuple[List[int], List[int]],
                      path2: Tuple[List[int], List[int]],
                      network_state: Dict, is_dc_fn: Callable[[int], bool]) -> Dict:
        """
        比较两条路径

        Returns:
            {
                'better_path': 1 or 2,
                'score_diff': float,
                'path1_details': Dict,
                'path2_details': Dict
            }
        """
        nodes1, links1 = path1
        nodes2, links2 = path2

        details1 = self.evaluate_with_details(nodes1, links1, network_state, is_dc_fn)
        details2 = self.evaluate_with_details(nodes2, links2, network_state, is_dc_fn)

        score1 = details1['score']
        score2 = details2['score']

        return {
            'better_path': 1 if score1 > score2 else 2,
            'score_diff': abs(score1 - score2),
            'score1': score1,
            'score2': score2,
            'path1_details': details1,
            'path2_details': details2
        }


# ===== 便捷函数 (保持向后兼容) =====

_default_evaluator = None


def get_default_evaluator() -> PathEvaluator:
    """获取默认评分器 (单例)"""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = PathEvaluator()
    return _default_evaluator


def evaluate_path_score(nodes: List[int], links: List[int],
                        network_state: Dict, is_dc_fn: Callable[[int], bool]) -> float:
    """
    默认评分函数 (向后兼容)

    Heuristic:
      - prefer fewer hops
      - prefer higher average bandwidth
      - consider min(bw) as bottleneck
      - reward DC coverage and overlap with existing tree
    """
    evaluator = get_default_evaluator()
    return evaluator.evaluate(nodes, links, network_state, is_dc_fn)


def evaluate_path_with_details(nodes: List[int], links: List[int],
                               network_state: Dict, is_dc_fn: Callable[[int], bool]) -> Dict:
    """详细评分函数 (新增)"""
    evaluator = get_default_evaluator()
    return evaluator.evaluate_with_details(nodes, links, network_state, is_dc_fn)


def set_default_weights(weights: Dict[str, float]):
    """设置默认评分权重"""
    global _default_evaluator
    _default_evaluator = PathEvaluator(weights=weights)
    logger.info(f"[PathEval] Updated default weights: {weights}")


# ===== 预定义评分策略 =====

class ScoringPresets:
    """预定义的评分策略"""

    # 策略1: 最小跳数优先
    MINIMAL_HOP = {
        'hop': -1.0,
        'avg_bw': 0.01,
        'min_bw': 0.01,
        'dc_ratio': 0.05,
        'overlap': 0.1,
        'node_res': 0.05
    }

    # 策略2: 带宽优先
    BANDWIDTH_FIRST = {
        'hop': -0.2,
        'avg_bw': 0.3,
        'min_bw': 0.5,
        'dc_ratio': 0.1,
        'overlap': 0.1,
        'node_res': 0.1
    }

    # 策略3: 资源均衡
    RESOURCE_BALANCED = {
        'hop': -0.3,
        'avg_bw': 0.15,
        'min_bw': 0.15,
        'dc_ratio': 0.2,
        'overlap': 0.2,
        'node_res': 0.2
    }

    # 策略4: VNF 部署优先
    VNF_DEPLOYMENT = {
        'hop': -0.2,
        'avg_bw': 0.1,
        'min_bw': 0.1,
        'dc_ratio': 0.4,
        'overlap': 0.1,
        'node_res': 0.3
    }

    # 策略5: 树扩展优先
    TREE_EXTENSION = {
        'hop': -0.3,
        'avg_bw': 0.1,
        'min_bw': 0.1,
        'dc_ratio': 0.1,
        'overlap': 0.5,
        'node_res': 0.1
    }


def create_evaluator(preset: str = 'default') -> PathEvaluator:
    """
    创建评分器

    Args:
        preset: 预设策略名称
            - 'default': 默认均衡策略
            - 'minimal_hop': 最小跳数优先
            - 'bandwidth': 带宽优先
            - 'balanced': 资源均衡
            - 'vnf': VNF 部署优先
            - 'tree': 树扩展优先
    """
    preset_map = {
        'default': None,
        'minimal_hop': ScoringPresets.MINIMAL_HOP,
        'bandwidth': ScoringPresets.BANDWIDTH_FIRST,
        'balanced': ScoringPresets.RESOURCE_BALANCED,
        'vnf': ScoringPresets.VNF_DEPLOYMENT,
        'tree': ScoringPresets.TREE_EXTENSION
    }

    weights = preset_map.get(preset.lower())
    evaluator = PathEvaluator(weights=weights)

    logger.info(f"[PathEval] Created evaluator with preset: {preset}")
    return evaluator


# ===== 高级评分函数 =====

def rank_paths(paths: List[Tuple[List[int], List[int]]],
               network_state: Dict, is_dc_fn: Callable[[int], bool],
               evaluator: Optional[PathEvaluator] = None) -> List[Tuple[float, int]]:
    """
    对多条路径进行排序

    Args:
        paths: 路径列表 [(nodes, links), ...]
        network_state: 网络状态
        is_dc_fn: DC 判断函数
        evaluator: 评分器 (可选)

    Returns:
        [(score, path_index), ...] 按分数降序排列
    """
    if evaluator is None:
        evaluator = get_default_evaluator()

    scores = []
    for i, (nodes, links) in enumerate(paths):
        score = evaluator.evaluate(nodes, links, network_state, is_dc_fn)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    logger.info(
        f"[PathEval] Ranked {len(paths)} paths, "
        f"best_score={scores[0][0]:.3f}, worst_score={scores[-1][0]:.3f}"
    )

    return scores


def select_best_path(paths: List[Tuple[List[int], List[int]]],
                     network_state: Dict, is_dc_fn: Callable[[int], bool],
                     evaluator: Optional[PathEvaluator] = None) -> Optional[Tuple[List[int], List[int]]]:
    """
    从多条路径中选择最佳路径

    Returns:
        (nodes, links) 或 None
    """
    if not paths:
        return None

    ranked = rank_paths(paths, network_state, is_dc_fn, evaluator)
    best_score, best_idx = ranked[0]

    logger.info(f"[PathEval] Selected path {best_idx} with score={best_score:.3f}")

    return paths[best_idx]