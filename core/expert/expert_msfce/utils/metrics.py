#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能指标模块"""

from typing import Dict, List
import numpy as np


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'failure_reasons': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'errors': 0,
        }

    def record_request(self, accepted: bool, processing_time: float):
        """记录请求处理结果"""
        self.metrics['total_requests'] += 1
        if accepted:
            self.metrics['accepted'] += 1
        else:
            self.metrics['rejected'] += 1
        self.metrics['processing_times'].append(processing_time)

    def record_cache_access(self, hit: bool):
        """记录缓存访问"""
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1

    def record_failure(self, reason: str):
        """记录失败原因"""
        if reason not in self.metrics['failure_reasons']:
            self.metrics['failure_reasons'][reason] = 0
        self.metrics['failure_reasons'][reason] += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.metrics.copy()

        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['p95_processing_time'] = np.percentile(stats['processing_times'], 95)
            stats['p99_processing_time'] = np.percentile(stats['processing_times'], 99)
        else:
            stats['avg_processing_time'] = 0
            stats['p95_processing_time'] = 0
            stats['p99_processing_time'] = 0

        total_cache = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / max(1, total_cache)

        total_req = stats['total_requests']
        stats['acceptance_rate'] = stats['accepted'] / max(1, total_req)

        return stats

    def reset(self):
        """重置所有指标"""
        self.__init__()