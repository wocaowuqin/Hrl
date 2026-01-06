#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSFCE Expert Algorithm - Modular Version

优化版本特性：
1. ✅ 模块化架构，职责清晰
2. ✅ 三级缓存系统：路径缓存 + 链路缓存 + 距离矩阵
3. ✅ O(1) 路径查询和距离查询
4. ✅ 向量化资源检查
5. ✅ 完整的性能指标收集
"""

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)

# 导出主要类和函数
from core.expert.expert_msfce.core.solver import MSFCE_Solver
from core.expert.expert_msfce.utils import SolverConfig, parse_mat_request
from core.expert.expert_msfce.utils.metrics import MetricsCollector
from core.expert.expert_msfce.utils.validators import validate_request, validate_state

__version__ = "2.0.0"
__all__ = [
    'MSFCE_Solver',
    'SolverConfig',
    'parse_mat_request',
    'MetricsCollector',
    'validate_request',
    'validate_state'
]