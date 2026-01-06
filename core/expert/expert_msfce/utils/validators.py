#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证工具模块"""

from typing import Dict
import numpy as np


def validate_request(request: Dict) -> bool:
    """验证请求格式"""
    required_fields = ['id', 'source', 'dest', 'vnf', 'bw_origin', 'cpu_origin', 'memory_origin']

    for field in required_fields:
        if field not in request:
            return False

    if not request['dest']:
        return False

    if len(request['vnf']) != len(request['cpu_origin']):
        return False

    if len(request['vnf']) != len(request['memory_origin']):
        return False

    return True


def validate_state(state: Dict, node_num: int, link_num: int) -> bool:
    """验证网络状态"""
    if 'cpu' not in state or 'mem' not in state or 'bw' not in state:
        return False

    if len(state['cpu']) != node_num:
        return False

    if len(state['mem']) != node_num:
        return False

    if len(state['bw']) != link_num:
        return False

    return True


def check_resource_availability(
        cpu_delta: np.ndarray,
        mem_delta: np.ndarray,
        rem_cpu: np.ndarray,
        rem_mem: np.ndarray,
        bw_req: float,
        rem_bw: np.ndarray,
        links_used: list,
        tolerance: float = 1e-7
) -> tuple:
    """
    检查资源可用性（向量化版本）

    Returns:
        (is_feasible, reason)
    """
    # CPU 和 Memory 检查（向量化）
    violations = (cpu_delta > rem_cpu + tolerance) | (mem_delta > rem_mem + tolerance)
    if np.any(violations):
        return False, "CPU_or_MEM_violation"

    # 带宽检查
    if links_used and bw_req > tolerance:
        unique_links = set(links_used)
        valid_indices = [lid - 1 for lid in unique_links if 0 < lid <= len(rem_bw)]

        if valid_indices and np.any(bw_req > rem_bw[valid_indices] + tolerance):
            return False, "BW_violation"

    return True, ""