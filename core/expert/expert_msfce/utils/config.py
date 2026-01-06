#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置管理模块"""

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """集中式配置管理"""
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    candidate_set_size: int = 8
    lookahead_depth: int = 1
    k_path: int = 5
    max_cache_size: int = 5000
    max_iterations: int = 500
    max_time_seconds: float = 60.0
    max_candidates: int = 30
    otv_link_weight: float = 0.2
    otv_node_weight: float = 0.8
    otv_norm_link: float = 90.0
    otv_norm_node: float = 8.0

    def __post_init__(self):
        """参数验证"""
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("Alpha, beta, gamma must be between 0 and 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            logger.warning("Score weights do not sum to 1.0")


def parse_mat_request(req_obj) -> Dict:
    """解析请求（兼容 Python Dict 和 MATLAB 格式）"""
    if isinstance(req_obj, dict):
        return req_obj

    try:
        return {
            'id': int(req_obj['id'][0, 0]),
            'source': int(req_obj['source'][0, 0]),
            'dest': [int(d) for d in req_obj['dest'].flatten()],
            'vnf': [int(v) for v in req_obj['vnf'].flatten()],
            'bw_origin': float(req_obj['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
            'arrival_time': int(req_obj.get('arrival_time', [[0]])[0, 0]),
            'leave_time': int(req_obj.get('leave_time', [[0]])[0, 0]),
        }
    except:
        return {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }