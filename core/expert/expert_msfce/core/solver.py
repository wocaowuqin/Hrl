#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSFCE主求解器"""

import time
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..utils.config import SolverConfig
from ..utils.metrics import MetricsCollector
from .path_engine import PathEngine
from .cache_manager import CacheManager, LinkCache
from .resource_manager import ResourceManager
from ..algorithms.tree_builder import TreeBuilder
from ..algorithms.placement import VNFPlacementStrategy

logger = logging.getLogger(__name__)


class MSFCE_Solver:
    """MSFCE专家算法求解器（模块化版）"""

    def __init__(
            self,
            path_db_file: Path,
            topology_matrix: np.ndarray,
            dc_nodes: List[int],
            capacities: Dict,
            config: Optional[SolverConfig] = None
    ):
        logger.info("=" * 60)
        logger.info("Initializing MSFCE Solver (Modular Version)")
        logger.info("=" * 60)

        self.config = config or SolverConfig()
        self._recall_failed_req_ids = set()

        # 基础参数
        self.node_num = int(topology_matrix.shape[0])
        self.type_num = 8
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        # DC节点处理
        if dc_nodes and min(dc_nodes) == 0:
            logger.info("Converting DC nodes from 0-based to 1-based")
            self.DC = {n + 1 for n in dc_nodes}
        else:
            self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        # 资源容量
        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        # K路径
        self.k_path = int(self.config.k_path)

        # 初始化模块
        logger.info("Initializing modules...")

        # 1. 链路缓存
        self.link_cache = LinkCache()
        self.link_cache.build_from_link_map(self.link_map)

        # 2. 路径引擎
        logger.info("Initializing path engine...")
        self.path_engine = PathEngine(
            path_db_file, self.node_num, self.k_path, self.link_cache
        )
        self.path_engine.validate_cache()

        # 3. 缓存管理器
        self.cache_manager = CacheManager(self.config.max_cache_size)

        # 4. 资源管理器
        self.resource_manager = ResourceManager(
            self.node_num, self.link_num, self.type_num,
            self.cap_cpu, self.cap_mem, self.cap_bw
        )

        # 5. VNF放置策略
        self.placement_strategy = VNFPlacementStrategy(self.node_num, self.type_num)

        # 6. 树构建器
        self.tree_builder = TreeBuilder(
            self.node_num, self.link_num, self.type_num, self.config,
            self.path_engine, self.resource_manager, self.placement_strategy
        )
        self.tree_builder.DC = self.DC
        self.tree_builder.dc_num = self.dc_num

        # 7. 指标收集器
        self.metrics_collector = MetricsCollector()

        # 兼容旧接口
        self.metrics = self.metrics_collector.metrics
        self.initial_state_template = self.resource_manager.initial_state_template

        logger.info("=" * 60)
        logger.info("✓ MSFCE Solver initialized successfully")
        logger.info(f"  Nodes: {self.node_num}, Links: {self.link_num}")
        logger.info(f"  DC nodes: {len(self.DC)}, VNF types: {self.type_num}")
        logger.info(f"  K-path: {self.k_path}")
        logger.info("=" * 60)

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        """构建链路映射"""
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def solve_request_for_expert(
            self,
            request: Dict,
            network_state: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], List]:
        """
        求解单个请求

        Returns:
            (tree, trajectory)
        """
        start_time = time.time()

        try:
            # 准备网络状态
            if network_state is not None:
                current_state = {}
                for k, v in network_state.items():
                    if isinstance(v, np.ndarray):
                        current_state[k] = v.astype(np.float64).copy()
                    else:
                        current_state[k] = copy.deepcopy(v)
            else:
                current_state = self.resource_manager.create_initial_state()

            # 请求ID对齐（0-based → 1-based）
            req_internal = copy.deepcopy(request)
            if req_internal['source'] == 0 or any(d == 0 for d in req_internal['dest']):
                req_internal['source'] += 1
                req_internal['dest'] = [d + 1 for d in req_internal['dest']]

            current_state['request'] = req_internal

            # 资源可行性检查
            if not self.resource_manager.check_global_feasibility(req_internal, current_state):
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(False, processing_time)
                return None, []

            # 构建树
            tree, traj, failed_dests = self.tree_builder.construct_tree(
                req_internal, current_state
            )

            processing_time = time.time() - start_time

            if tree is not None:
                self.metrics_collector.record_request(True, processing_time)
                return tree, traj
            else:
                self.metrics_collector.record_request(False, processing_time)
                return None, []

        except Exception as e:
            logger.exception(f"Error in solve_request_for_expert: {e}")
            self.metrics_collector.record_request(False, time.time() - start_time)
            return None, []

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self.metrics_collector.get_stats()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        stats = self.cache_manager.get_stats()
        stats['path_cache_entries'] = len(self.path_engine._path_cache)
        return stats

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("SOLVER STATISTICS")
        print("=" * 70)

        metrics = self.get_metrics()
        print(f"Total Requests:  {metrics['total_requests']}")
        print(f"Accepted:        {metrics['accepted']}")
        print(f"Rejected:        {metrics['rejected']}")
        print(f"Acceptance Rate: {metrics['acceptance_rate']:.2%}")

        if metrics.get('avg_processing_time'):
            print(f"Avg Time:        {metrics['avg_processing_time'] * 1000:.2f} ms")
            print(f"P95 Time:        {metrics['p95_processing_time'] * 1000:.2f} ms")

        cache_stats = self.get_cache_stats()
        print(f"\nCache Hit Rate:  {cache_stats['hit_rate']:.2%}")
        print(f"Path Cache Size: {cache_stats['path_cache_entries']}")

        print("=" * 70 + "\n")

    def clear_cache(self):
        """清空路径评分缓存"""
        self.cache_manager.clear_path_eval_cache()
        logger.info("Path evaluation cache cleared")

    def export_metrics(self, path: Optional[Path] = None):
        """导出性能指标到 CSV"""
        import csv
        if path is None:
            path = Path('expert_metrics.csv')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Requests', self.metrics['total_requests']])
            writer.writerow(['Accepted', self.metrics['accepted']])
            writer.writerow(['Rejected', self.metrics['rejected']])

            accept_rate = self.metrics['accepted'] / max(1, self.metrics['total_requests'])
            writer.writerow(['Accept Rate', f"{accept_rate:.2%}"])

            writer.writerow([])
            writer.writerow(['Failure Reason', 'Count'])
            for reason, count in self.metrics.get('failure_reasons', {}).items():
                writer.writerow([reason, count])

            if self.metrics.get('processing_times'):
                writer.writerow([])
                writer.writerow(['Avg Processing Time (s)',
                                 np.mean(self.metrics['processing_times'])])

        logger.info(f"Metrics exported to {path}")

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'total_requests': self.metrics['total_requests'],
            'acceptance_rate': self.metrics['accepted'] / max(1, self.metrics['total_requests']),
            'cache_hit_rate': self.metrics['cache_hits'] /
                              max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'failure_reasons': self.metrics.get('failure_reasons', {}),
        }

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report.update({
                'avg_processing_time': float(np.mean(times)),
                'max_processing_time': float(max(times)),
                'min_processing_time': float(min(times)),
            })

        return report

    def get_detailed_performance_report(self) -> Dict:
        """详细性能报告"""
        report = self.get_performance_report()

        cache_stats = self.get_cache_stats()
        cache_eff = {
            'cache_size': cache_stats['path_eval_cache_size'],
            'cache_max_size': self.config.max_cache_size,
            'cache_utilization': cache_stats['path_eval_cache_size'] / max(1, self.config.max_cache_size)
        }
        report['cache_efficiency'] = cache_eff

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report['recent_performance'] = {
                'last_10_avg': float(np.mean(times[-10:])) if len(times) >= 10 else 0.0,
                'trend': 'improving' if len(times) > 1 and times[-1] < times[0] else 'stable'
            }

        return report

    def validate_cache(self) -> bool:
        """验证缓存完整性"""
        return self.path_engine.validate_cache()

    def _normalize_state(self, state: Dict) -> Dict:
        """规范化网络状态（兼容方法）"""
        return self.resource_manager.normalize_state(state)