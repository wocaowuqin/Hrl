#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径计算引擎"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import scipy.io as sio
from pathlib import Path

logger = logging.getLogger(__name__)


class PathEngine:
    """路径计算和查询引擎"""

    def __init__(self, path_db_file: Path, node_num: int, k_path: int, link_cache: Dict):
        self.path_db_file = path_db_file
        self.node_num = node_num
        self.k_path = k_path
        self._link_cache = link_cache

        # 缓存系统
        self._path_cache = {}  # {(src, dst, k): (nodes, dist, links)}
        self._distance_matrix = None  # n x n

        # 加载 Path DB
        self._load_path_db()

        # 预计算所有路径
        self._precompute_all_paths()

        # 预计算距离矩阵
        self._precompute_distance_matrix()

    def _load_path_db(self):
        """加载路径数据库"""
        if not Path(self.path_db_file).exists():
            raise FileNotFoundError(f"Path DB missing: {self.path_db_file}")

        try:
            mat = sio.loadmat(self.path_db_file)
            self.path_db = mat['Paths']
            logger.info(f"Loaded Path DB from {self.path_db_file}")
        except Exception as e:
            raise RuntimeError(f"Path DB load failed: {e}")

    def _precompute_all_paths(self):
        """预计算所有路径"""
        total_paths = 0
        failed_paths = 0

        for src in range(1, self.node_num + 1):
            for dst in range(1, self.node_num + 1):
                if src == dst:
                    self._path_cache[(src, dst, 1)] = ([src], 0, [])
                    total_paths += 1
                    continue

                for k in range(1, self.k_path + 1):
                    try:
                        nodes, dist, links = self._load_path_from_db(src, dst, k)
                        if nodes:
                            self._path_cache[(src, dst, k)] = (nodes, dist, links)
                            total_paths += 1
                        else:
                            failed_paths += 1
                    except Exception as e:
                        logger.debug(f"Failed to load path ({src},{dst},k={k}): {e}")
                        failed_paths += 1

        logger.info(f"✓ Path cache: {total_paths} valid paths, {failed_paths} failed")

    def _load_path_from_db(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """从PathDB加载单条路径"""
        try:
            pinfo = self.path_db[src - 1, dst - 1]

            if 'paths' not in pinfo.dtype.names:
                return [], 0, []

            raw_paths = pinfo['paths']
            if raw_paths.size == 0:
                return [], 0, []

            idx = k - 1
            path_arr = None

            if raw_paths.dtype == 'O':
                flat_data = raw_paths.flatten()
                if idx < len(flat_data):
                    path_arr = flat_data[idx]
            elif raw_paths.ndim == 2:
                if idx < raw_paths.shape[0]:
                    path_arr = raw_paths[idx]
            elif raw_paths.ndim == 1 and idx == 0:
                path_arr = raw_paths

            if path_arr is None:
                return [], 0, []

            path_arr_flat = np.array(path_arr).flatten()

            # 获取距离信息
            dist_k = -1
            if 'pathsdistance' in pinfo.dtype.names:
                raw_dists = pinfo['pathsdistance'].flatten()
                if idx < len(raw_dists):
                    dist_k = int(raw_dists[idx])

            # 截取路径
            if dist_k >= 0 and (dist_k + 1) <= len(path_arr_flat):
                path_segment = path_arr_flat[:dist_k + 1]
            else:
                path_segment = path_arr_flat

            # 过滤0值
            path_nodes = [int(x) for x in path_segment if int(x) > 0]

            if len(path_nodes) < 2:
                return [], 0, []

            # 计算链路
            links = self._compute_links_fast(path_nodes)

            return path_nodes, len(path_nodes) - 1, links

        except Exception as e:
            return [], 0, []

    def _compute_links_fast(self, path_nodes: List[int]) -> List[int]:
        """快速计算路径的链路ID列表"""
        links = []

        if len(path_nodes) <= 1:
            return links

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            link_id = self._link_cache.get((u, v)) or self._link_cache.get((v, u))

            if link_id is not None:
                links.append(link_id)

        return links

    def _precompute_distance_matrix(self):
        """预计算最短距离矩阵"""
        n = self.node_num
        self._distance_matrix = np.full((n, n), 9999, dtype=int)
        np.fill_diagonal(self._distance_matrix, 0)

        computed_count = 0

        for src in range(1, n + 1):
            for dst in range(1, n + 1):
                if src == dst:
                    continue

                cache_key = (src, dst, 1)
                if cache_key in self._path_cache:
                    _, dist, _ = self._path_cache[cache_key]
                    self._distance_matrix[src - 1, dst - 1] = dist
                    computed_count += 1

        logger.info(f"✓ Distance matrix: {computed_count}/{n * (n - 1)} entries")

    def get_path_info(self, src: int, dst: int, k: int) -> Tuple[List[int], int, List[int]]:
        """
        获取路径信息（O(1)时间复杂度）

        Returns:
            (path_nodes, distance, link_ids) 或 ([], 0, [])
        """
        if src == dst:
            return [src], 0, []

        if not (1 <= src <= self.node_num and 1 <= dst <= self.node_num):
            logger.warning(f"Invalid nodes: src={src}, dst={dst}")
            return [], 0, []

        if not (1 <= k <= self.k_path):
            return [], 0, []

        cache_key = (src, dst, k)

        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        else:
            logger.warning(f"Cache miss for ({src},{dst},k={k})")
            return [], 0, []

    def get_shortest_distance(self, src: int, dst: int) -> int:
        """快速获取最短距离（O(1)时间复杂度）"""
        if self._distance_matrix is None:
            return 9999

        if 1 <= src <= self.node_num and 1 <= dst <= self.node_num:
            return int(self._distance_matrix[src - 1, dst - 1])

        return 9999

    def get_max_hops(self, src: int, dst: int) -> int:
        """获取最大跳数"""
        try:
            cache_key = (src, dst, self.k_path)
            if cache_key in self._path_cache:
                _, dist, _ = self._path_cache[cache_key]
                return dist

            return self.get_shortest_distance(src, dst) * 2
        except:
            return 10

    def validate_cache(self) -> bool:
        """验证缓存完整性"""
        expected_paths = self.node_num * (self.node_num - 1) * self.k_path
        actual_paths = len(self._path_cache) - self.node_num

        coverage = actual_paths / expected_paths

        if coverage > 0.8:
            logger.info(f"✓ Cache validation PASSED: {coverage:.2%} coverage")
            return True
        else:
            logger.warning(f"✗ Cache validation WARNING: {coverage:.2%} coverage")
            return False