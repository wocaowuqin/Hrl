#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缓存管理模块"""

import logging
from collections import OrderedDict
from typing import Dict, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""

    def __init__(self, max_cache_size: int = 5000):
        self.max_cache_size = max_cache_size

        # LRU 缓存（用于路径评分）
        self._path_eval_cache = OrderedDict()

        # 统计
        self.cache_hits = 0
        self.cache_misses = 0

    def get_path_eval(self, cache_key: Tuple) -> Tuple:
        """获取缓存的路径评分"""
        if cache_key in self._path_eval_cache:
            # LRU: 移动到末尾
            value = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = value
            self.cache_hits += 1
            return value
        else:
            self.cache_misses += 1
            return None

    def set_path_eval(self, cache_key: Tuple, value: Tuple):
        """设置路径评分缓存"""
        self._path_eval_cache[cache_key] = value

        # LRU 淘汰
        if len(self._path_eval_cache) > self.max_cache_size:
            self._path_eval_cache.popitem(last=False)

    def clear_path_eval_cache(self):
        """清空路径评分缓存"""
        self._path_eval_cache.clear()

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total)

        return {
            'path_eval_cache_size': len(self._path_eval_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def reset_stats(self):
        """重置统计信息"""
        self.cache_hits = 0
        self.cache_misses = 0


class LinkCache:
    """链路查找缓存"""

    def __init__(self):
        self._link_cache = {}  # {(u, v): link_id}

    def build_from_link_map(self, link_map: Dict):
        """从link_map构建缓存"""
        self._link_cache.clear()

        for edge, lid in link_map.items():
            self._link_cache[edge] = lid

        logger.info(f"✓ Link lookup table: {len(self._link_cache)} entries")

    def get_link_id(self, u: int, v: int) -> int:
        """获取链路ID（支持双向查询）"""
        return self._link_cache.get((u, v)) or self._link_cache.get((v, u))

    def __getitem__(self, edge: Tuple[int, int]) -> int:
        """支持字典式访问"""
        return self._link_cache.get(edge)

    def get(self, edge: Tuple[int, int], default=None):
        """支持get方法"""
        return self._link_cache.get(edge, default)