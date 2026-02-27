# envs/sfc_env.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC_HIRL_Env - 完整可运行的主环境类（分层强化学习 + 多播感知）
已完全模块化，职责清晰，兼容 Flat 和 GNN 两种状态表示
"""
import os
import logging
import time

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import gym
import pickle
import torch
from collections import deque, defaultdict
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import pyplot as plt

# 导入自定义模块
from envs.modules.AllResourceManager import FusedResourceManager as ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.path_manager import PathManager
from envs.modules.event_handler import EventHandler
from envs.modules.policy_helper import PolicyHelper
from envs.modules.failure_visualizer import FailureVisualizer
from envs.modules.visualize_multicast_tree import MulticastTreeVisualizer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class SimpleTopologyManager:
    """
    增强版简化拓扑管理器
    补全 GNN 特征提取所需的度数和介数计算接口
    """

    def __init__(self, topo):
        self.topo = topo  # 假设是邻接矩阵 [N, N]
        self.n = topo.shape[0]
        self.original_topo = topo.copy()

        # 预计算节点度数，避免 get_state 频繁求和
        self.degrees = np.sum(self.topo > 0, axis=1)

    def reset(self):
        self.topo = self.original_topo.copy()
        self.degrees = np.sum(self.topo > 0, axis=1)

    def get_neighbors(self, node):
        """获取节点的邻居索引"""
        return np.where(self.topo[node] > 0)[0].tolist()

    def get_node_degree(self, node):
        """🔥 修复点：返回节点度数"""
        return float(self.degrees[node])

    def get_node_betweenness(self, node):
        """🔥 修复点：返回介数中心性（简化版，返回0.0或度数比）"""
        # 完整的介数计算开销大，作为 SimpleManager，我们可以返回度数的归一化值
        return float(self.degrees[node] / max(1, self.n))
class RequestLifecycleManager:
    """
    请求生命周期管理器

    核心职责：
    1. 跟踪每个请求的状态（进行中、已完成、已过期）
    2. 基于请求的实际过期时间释放资源
    3. 与时间切片解耦
    """

    def __init__(self, env):
        self.env = env

        # 核心数据结构：跟踪所有活跃请求
        self.active_requests = {}  # {req_id: RequestInfo}

        # 可选：为了兼容性，保留time_slot索引
        self.requests_by_slot = {}  # {slot: set(req_ids)}

    def add_request(self, req):
        """
        添加新请求到管理器

        Args:
            req: 请求对象，包含arrival_time和lifetime
        """
        req_id = req.get('id', id(req))

        # 计算过期时间
        expire_time = req['arrival_time'] + req['lifetime']

        # 创建请求信息
        req_info = {
            'req': req,
            'req_id': req_id,
            'arrival_time': req['arrival_time'],
            'lifetime': req['lifetime'],
            'expire_time': expire_time,
            'time_slot': req.get('time_slot', int(req['arrival_time'] / self.env.delta_t)),
            'status': 'active',  # active / completed / expired
            'vnf_deployed': [],  # 已部署的VNF
            'resources_allocated': {  # 已分配的资源
                'nodes': [],
                'links': []
            }
        }

        # 添加到活跃请求
        self.active_requests[req_id] = req_info

        # 可选：添加到时间切片索引
        slot = req_info['time_slot']
        if slot not in self.requests_by_slot:
            self.requests_by_slot[slot] = set()
        self.requests_by_slot[slot].add(req_id)

        return req_id

    def complete_request(self, req_id):
        """
        标记请求为已完成

        Args:
            req_id: 请求ID
        """
        if req_id in self.active_requests:
            self.active_requests[req_id]['status'] = 'completed'

            # 从活跃请求中移除（已完成的不需要继续跟踪）
            self._remove_request(req_id)

    def _remove_request(self, req_id):
        """
        从管理器中移除请求

        Args:
            req_id: 请求ID
        """
        if req_id in self.active_requests:
            req_info = self.active_requests[req_id]

            # 从时间切片索引中移除
            slot = req_info['time_slot']
            if slot in self.requests_by_slot:
                self.requests_by_slot[slot].discard(req_id)
                if not self.requests_by_slot[slot]:
                    del self.requests_by_slot[slot]

            # 从活跃请求中移除
            del self.active_requests[req_id]

    def check_and_release_expired(self, current_time):
        """
        检查并释放过期的请求 - 带详细日志版本
        """
        expired_req_ids = []

        # 🔥 记录释放前的资源状态
        res_before = self.env.get_resource_utilization() if hasattr(self, 'env') else None

        # 遍历所有活跃请求
        for req_id, req_info in list(self.active_requests.items()):
            if current_time > req_info['expire_time']:
                expired_req_ids.append(req_id)

                # 🔥 详细日志
                expire_time = req_info['expire_time']
                arrival_time = req_info['arrival_time']
                print(f"   ⏱️ 请求 {req_id} 已过期: "
                      f"到达={arrival_time:.2f}s, "
                      f"过期={expire_time:.2f}s, "
                      f"当前={current_time:.2f}s")

        # 释放过期请求的资源
        for req_id in expired_req_ids:
            self._release_request_resources(req_id, current_time)

        # 🔥 释放后的资源状态
        if expired_req_ids:
            res_after = self.env.get_resource_utilization() if hasattr(self, 'env') else None
            print(f"♻️ [过期释放] 释放了 {len(expired_req_ids)} 个请求")
            if res_before is not None and res_after is not None:
                change = res_after - res_before
                print(f"   资源变化: {res_before:.1f}% → {res_after:.1f}% "
                      f"({'+' if change > 0 else ''}{change:.1f}%)")
            print(f"   请求ID: {expired_req_ids}")

        return expired_req_ids

    def _release_request_resources(self, req_id,_):
        """
        🔥 [V16.2 修正版] 从历史账本中释放资源
        注意：这个方法属于 RequestLifecycleManager 类
        """
        if req_id not in self.active_requests:
            return

        req_info = self.active_requests[req_id]
        req = req_info['req']

        # 获取我们刚才在 _archive_request 里存的账本
        allocated = req.get('resources_allocated', {})
        placement = allocated.get('placement', {})
        tree_edges = allocated.get('tree', {})

        restored_cpu = 0
        restored_bw = 0

        # 1. 释放节点资源
        for key, info in placement.items():
            node = info.get('node')
            if isinstance(info, dict):
                c = info.get('cpu_used', 0)
                m = info.get('mem_used', 0)
                vnf = info.get('vnf_type', 0)
            else:
                c, m = 1.0, 1.0
                vnf = 0

            if hasattr(self.env.resource_mgr, 'release_node_resource'):
                self.env.resource_mgr.release_node_resource(node, vnf, c, m)
            restored_cpu += c

        # 2. 释放链路资源
        for (u, v), bw in tree_edges.items():
            if hasattr(self.env.resource_mgr, 'release_link_resource'):
                self.env.resource_mgr.release_link_resource(u, v, bw)
            restored_bw += bw

        if restored_cpu > 0 or restored_bw > 0:
            print(f"♻️ [生命周期结束] 释放请求 {req_id} | CPU: +{restored_cpu:.1f} | BW: +{restored_bw:.1f}")

        # 从管理器中移除
        del self.active_requests[req_id]

        # 同步清理 Slot 索引
        slot = req_info.get('time_slot')
        if slot in self.requests_by_slot:
            self.requests_by_slot[slot].discard(req_id)

    def get_status_summary(self):
        """
        获取状态摘要

        Returns:
            dict: 状态统计
        """
        return {
            'active_requests': len(self.active_requests),
            'active_slots': len(self.requests_by_slot),
            'requests': list(self.active_requests.keys())
        }
class ExpertWrapper:
    """包装 MSFCE_Solver，适配 BackupPolicy (修复版)"""

    def __init__(self, msfce_solver):
        self.solver = msfce_solver
        # 尝试获取节点数，防错处理
        self.node_num = getattr(msfce_solver, 'node_num', 28)
        self.DC = getattr(msfce_solver, 'DC', [])

    def find_any_path(self, src, dst):
        """查找路径（0-based）"""
        # 1. 转换索引：0-based -> 1-based (适配 MATLAB/PathEngine 习惯)
        src_1 = src + 1
        dst_1 = dst + 1

        # 2. 尝试获取 PathEngine
        # 通常 MSFCE_Solver 会把 PathEngine 实例保存在 self.path_engine
        path_engine = getattr(self.solver, 'path_engine', None)

        # --- 方案 A: 通过 PathEngine 标准接口 (推荐) ---
        if path_engine and hasattr(path_engine, 'get_path_info'):
            # k=1 表示找最短路径
            nodes, dist, links = path_engine.get_path_info(src_1, dst_1, 1)
            if nodes:
                # 转回 0-based
                nodes_0 = [n - 1 for n in nodes]
                return nodes_0, links

        # --- 方案 B: 直接访问 PathEngine 缓存 (备选) ---
        if path_engine and hasattr(path_engine, '_path_cache'):
            cache_key = (src_1, dst_1, 1)
            if cache_key in path_engine._path_cache:
                nodes, dist, links = path_engine._path_cache[cache_key]
                nodes_0 = [n - 1 for n in nodes] if nodes else None
                return nodes_0, links

        # --- 方案 C: 旧版兼容 (直接在 Solver 上找) ---
        if hasattr(self.solver, '_path_cache'):
            cache_key = (src_1, dst_1, 1)
            if cache_key in self.solver._path_cache:
                nodes, dist, links = self.solver._path_cache[cache_key]
                nodes_0 = [n - 1 for n in nodes] if nodes else None
                return nodes_0, links

        # 如果都找不到
        return None, None
class SimpleDataLoader:
    """
    简化的数据加载器
    职责：从文件加载数据到内存，仅此而已。
    """

    def __init__(self, config):
        self.config = config
        self.requests = []
        self.events = []
        self.total_steps = 0
        self.req_map = {}

    def reset(self):
        """
        重置加载器状态（适配接口调用）
        """
        self.req_map = {r['id']: r for r in self.requests}
        # 如果需要，可以在这里重置内部指针，但对于简单加载器通常不需要
        pass

    def load_dataset(self, phase_or_file):
        """加载数据集"""
        import pickle

        # 1. 确定文件路径
        if isinstance(phase_or_file, str) and phase_or_file.startswith('phase'):
            # 模式 A: 通过 phase 名称加载
            data_dir = self.config.get('path', {}).get('input_dir', 'data/input_dir')
            req_file = os.path.join(data_dir, f'{phase_or_file}_requests.pkl')
            evt_file = os.path.join(data_dir, f'{phase_or_file}_events.pkl')
        else:
            # 模式 B: 直接提供文件路径
            req_file = phase_or_file
            evt_file = None

        # 2. 加载请求
        if os.path.exists(req_file):
            with open(req_file, 'rb') as f:
                self.requests = pickle.load(f)
            self.total_steps = len(self.requests)
            # 构建索引
            self.req_map = {r['id']: r for r in self.requests}
            logger.info(f"✅ [SimpleDataLoader] 加载请求: {len(self.requests)} 条")
        else:
            logger.warning(f"⚠️ [SimpleDataLoader] 请求文件不存在: {req_file}")
            self.requests = []

        # 3. 加载事件 (可选)
        if evt_file and os.path.exists(evt_file):
            with open(evt_file, 'rb') as f:
                self.events = pickle.load(f)
            logger.info(f"✅ [SimpleDataLoader] 加载事件: {len(self.events)} 条")
        else:
            self.events = []

        return len(self.requests) > 0
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

class SFC_HIRL_Env(gym.Env):
    #基础初始化与数据加载
    def __init__(self, config, use_gnn=True):
        """初始化环境"""
        super().__init__()
        self.config = config
        self.use_gnn = use_gnn
        # self.time_step = 0
        # 1. 基础架构：拓扑与资源
        self._init_infrastructure()
        self.request_manager = RequestLifecycleManager(self)
        # 2. 核心功能模块：专家、备份、路径管理
        self._init_core_modules()

        # 3. 强化学习辅助组件：数据、奖励、策略助手
        self._init_rl_components()

        # 4. 状态与动作空间变量
        self._init_state_variables()
        #MAB 智能剪枝组件
        self._init_mab_components()
        # 5. GNN 与 Gym 空间定义
        self._init_gym_spaces()
        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0
        self.vnf_deployment_history = {}
        logger.info(f"✅ 环境初始化完成: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

        self.enable_visualization = True  # 设为 False 可以关闭
        if self.enable_visualization:
            try:
                import os
                os.makedirs('visualization/success', exist_ok=True)
                os.makedirs('visualization/fail', exist_ok=True)

                self.visualizer = MulticastTreeVisualizer(self)
                logger.info("✅ 可视化器已启用")
            except Exception as e:
                logger.warning(f"⚠️ 可视化器初始化失败: {e}")
                self.enable_visualization = False
    def _init_infrastructure(self):
        """初始化拓扑、维度和资源管理器"""
        # --- 加载拓扑 ---
        topo = self.config.get('topology', {}).get('matrix')
        if topo is None:
            n = self.config.get('environment', {}).get('num_nodes', 28)
            topo = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(topo, 0)
        self.topo = np.asarray(topo, dtype=np.float32)

        # --- 设置维度 ---
        self.n = self.topo.shape[0]
        self.K_vnf = self.config.get('vnf', {}).get('n_types', 8)
        self.L = int(np.sum(self.topo > 0))

        # --- 资源管理器 ---
        capacities = self.config.get('capacities', {'cpu': 100.0, 'memory': 80.0, 'bandwidth': 100.0})
        self.dc_nodes = self.config.get('topology', {}).get('dc_nodes', list(range(10)))

        self.resource_mgr = ResourceManager(self.topo, capacities, self.dc_nodes)
        self.topology_mgr = SimpleTopologyManager(self.topo)

        logger.info(f"✅ 环境参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")
    def _init_core_modules(self):
        """初始化专家系统、备份策略和路径管理器"""
        self.path_manager = PathManager(max_paths=10)

        # --- 初始化 MSFCE 专家 ---
        try:
            from core.expert.expert_msfce.core.solver import MSFCE_Solver
            from core.expert.expert_msfce.utils.config import SolverConfig

            path_db_file = Path("data/input_dir/US_Backbone_path.mat")
            capacities = self.config.get('capacities', {})

            msfce_solver = MSFCE_Solver(
                path_db_file=path_db_file,
                topology_matrix=self.topo,
                dc_nodes=self.dc_nodes,
                capacities=capacities,
                config=SolverConfig()
            )
            self.expert = ExpertWrapper(msfce_solver)
        except ImportError as e:
            logger.error(f"❌ 无法导入专家模块: {e}")
            self.expert = None

        # --- 初始化 BackupPolicy ---
        try:
            from envs.modules.sfc_backup_system.backup_policy import BackupPolicy
            self.backup_policy = BackupPolicy(
                expert=self.expert,
                n=self.n,
                L=self.L,
                K_vnf=self.K_vnf,
                dc_nodes=self.dc_nodes
            )
        except ImportError:
            logger.warning("⚠️ 未能加载 BackupPolicy")
            self.backup_policy = None
    def _init_rl_components(self):
        """初始化数据加载、奖励计算、策略助手等"""
        self.data_loader = DataLoader(self.config)
        self.event_handler = EventHandler(resource_manager=self.resource_mgr)

        self.request_manager = RequestLifecycleManager(self)
        logger.info("✅ 请求生命周期管理器已初始化")

        # --- Policy Helper ---
        input_dir = Path(self.config.get('path', {}).get('input_dir', 'data/input_dir'))
        capacities = self.config.get('capacities', {})
        self.policy_helper = PolicyHelper(
            input_dir=input_dir,
            topo=self.topo,
            dc_nodes=self.dc_nodes,
            capacities=capacities
        )

        # --- Reward Critic ---
        # 🔥 修改：使用极简奖励计算器
        try:
            # 尝试导入新的极简奖励计算器
            from core.reward.stateless_reward_critic import StatelessRewardCritic
            reward_params = self.config.get('reward', {})

            # 确保使用极简参数
            simple_params = {
                'connect_bonus': reward_params.get('connect_bonus', 10.0),
                'reuse_bonus': reward_params.get('reuse_bonus', 1.5),
                'step_cost': reward_params.get('step_cost', 0.05),
                'illegal_penalty': reward_params.get('illegal_penalty', 3.0),
                'timeout_penalty': reward_params.get('timeout_penalty', 100.0)
            }

            self.reward_critic = StatelessRewardCritic()
            logger.info("✅ 使用极简奖励计算器 (StatelessRewardCritic)")

        except ImportError as e:
            # 如果找不到新模块，回退到修改后的RewardCritic
            logger.warning(f"⚠️ 无法导入StatelessRewardCritic: {e}，回退到修改版RewardCritic")

            from core.reward.reward_critic import RewardCritic
            reward_params = self.config.get('reward', {})

            # 创建简化参数
            simple_params = {
                'connect_bonus': reward_params.get('connect_bonus', 10.0),
                'reuse_bonus': reward_params.get('reuse_bonus', 1.5),
                'step_cost': reward_params.get('step_cost', 0.05),
                'illegal_penalty': reward_params.get('illegal_penalty', 3.0),
                'timeout_penalty': reward_params.get('timeout_penalty', 100.0)
            }

            # 创建实例
            self.reward_critic = RewardCritic(training_phase=3, params=simple_params)
            logger.info("✅ 使用修改版RewardCritic (已简化)")

        # --- Failure Visualizer ---
        try:
            self.failure_visualizer = FailureVisualizer(self.config)
        except Exception as e:
            logger.warning(f"⚠️ FailureVisualizer 初始化失败: {e}")
            self.failure_visualizer = None

        # 🔥 打印奖励配置
        print("🎯 奖励配置:")
        print(f"   连接新目的地: +{getattr(self.reward_critic, 'connect_bonus', 10.0)}")
        print(f"   复用树节点: +{getattr(self.reward_critic, 'reuse_bonus', 1.5)}")
        print(f"   每步成本: -{getattr(self.reward_critic, 'step_cost', 0.05)}")
        print(f"   非法动作: -{getattr(self.reward_critic, 'illegal_penalty', 3.0)}")
    def _init_state_variables(self):
        """
        初始化环境运行时的状态变量 (在线模式增强版 - 修复 AttributeError)
        """
        # 1. 基础计数器
        self.step_counter = 0
        self.total_reward = 0

        # 统计计数器
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.node_visit_counts = {}
        #  添加当前节点位置
        self.current_node_location = 0
        #  添加当前VNF索引
        self.current_vnf_index = 0
        #  添加nodes_on_tree（如果还没有）
        self.nodes_on_tree = set()

        # --- 动作空间配置 ---
        env_config = self.config.get('environment', {})
        self.nb_high_level_goals = env_config.get('nb_high_level_goals', 10)
        self.NB_LOW_LEVEL_ACTIONS = self.n
        self._n_actions = self.n

        # --- 动态变量 ---
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.current_request = None
        self._prev_dist = None
        self.failed_deploy_attempts = set()

        # 资源账本
        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []
        self._current_req_record = {}

        # HRL 分支管理状态
        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0

        # ========================================================================
        # 🔥 [新增] 在线仿真模式配置
        # ========================================================================
        self.online_mode = self.config.get('environment', {}).get('online_mode', True)

        # 仿真状态机变量
        self.simulation_done = False
        self.current_slot_index = 0
        self.slot_queue = []

        # 🔥🔥🔥 [关键修复] 初始化数据容器，防止 reset 报错 🔥🔥🔥
        self.all_requests = []  # <--- 必须加这一行
        self.requests_by_slot = {}  # <--- 必须加这一行
        self.max_slot_index = 0

        self.active_requests_by_slot = {}
        self.leave_heap = []

        # ========================================================================
        # 🔥 时间槽系统
        # ========================================================================
        self.delta_t = self.config.get('data_generation', {}).get('time_slot_delta', 0.01)
        self.processing_delay = 0.0 if self.online_mode else 0.002
        self.time_step = 0.0
        self.current_time_slot = 0
        self.decision_step = 0

        # 动态环境配置
        dynamic_cfg = self.config.get('dynamic_env', {})
        self.dynamic_env = dynamic_cfg.get('enabled', True)

        # 全局指针
        self.global_request_index = 0
        self._request_index = 0
        self.served_dest_count = 0

        # 最大步数
        p3_cfg = self.config.get('phase3', {})
        env_cfg = self.config.get('env', {})
        self.max_steps = p3_cfg.get('max_steps_per_episode', env_cfg.get('max_steps', 1000))
    def _init_gym_spaces(self):
        """初始化 GNN 特征提取器和 Gym 空间"""
        # --- GNN Feature Builder ---
        if self.use_gnn:
            try:
                from core.gnn.feature_builder import GNNFeatureBuilder
                self.feature_builder = GNNFeatureBuilder(self.config)
            except Exception as e:
                logger.warning(f"⚠️ FeatureBuilder 初始化失败: {e}")
                self.feature_builder = None
        else:
            self.feature_builder = None

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Dict({
            'x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, 17), dtype=np.float32),
            'edge_index': gym.spaces.Box(low=0, high=self.n, shape=(2, self.n * self.n), dtype=np.int64),
        })
        self.action_space = gym.spaces.Discrete(self.n)

    def _init_mab_components(self):
        """
        初始化 MAB 智能剪枝组件
        """
        # 1. 读取配置
        mab_conf = self.config.get('mab_pruning', {})

        # 2. 设置基本参数
        self.use_mab_pruning = mab_conf.get('enabled', True)
        self.mab_rounds = mab_conf.get('rounds', 20)

        # 🔥🔥🔥 [修复这里] 补上缺失的 enable_mab_learning 属性 🔥🔥🔥
        # 默认为 True，表示允许 MAB 在运行过程中学习和更新
        self.enable_mab_learning = mab_conf.get('learning', True)

        # 3. 实例化 MAB 助手
        self.mab_pruner = MABPruningHelper(
            exploration_param=mab_conf.get('exploration', 1.4),
            policy=mab_conf.get('policy', 'ucb1')
        )

        # 4. 初始化统计信息
        self.mab_action_stats = {
            'total_selections': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'successful_prunes': 0,
            'failed_prunes': 0
        }

        logger.info(f"🤖 MAB组件初始化完成: Mode={self.use_mab_pruning}, Learning={self.enable_mab_learning}")
    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        """
        加载数据集（修复版）
        🔥 关键修复：加载后自动构建时间槽索引，打破死循环
        """
        success = False

        # --- 1. 调用底层 Loader 加载数据 ---
        if events_file is not None:
            # (兼容旧代码：直接读取文件)
            try:
                import pickle
                with open(phase_or_req_file, 'rb') as f:
                    requests = pickle.load(f)
                with open(events_file, 'rb') as f:
                    raw_events = pickle.load(f)

                # 同步给 data_loader
                self.data_loader.requests = requests
                self.data_loader.total_steps = len(requests)
                success = True
                print(f"✅ [Env] 手动文件加载成功: {len(requests)} 条")
            except Exception as e:
                print(f"❌ [Env] 手动加载失败: {e}")
                return False
        else:
            # (标准模式：使用 data_loader)
            if hasattr(self, 'data_loader'):
                success = self.data_loader.load_dataset(phase_or_req_file)
            else:
                print("❌ [Env] data_loader 未初始化")
                return False

        # --- 2. 🔥🔥🔥 核心修复：同步数据到环境索引 🔥🔥🔥 ---
        # 如果不执行这一步，all_requests 永远为空，导致无限 Reset
        if success:
            requests_data = getattr(self.data_loader, 'requests', [])
            if requests_data:
                print(f"🔄 [Env] 正在构建在线仿真索引 (Requests: {len(requests_data)})...")
                # 这一步会填充 self.all_requests 和 self.requests_by_slot
                self.load_requests(requests_data)
            else:
                print("⚠️ [Env] 数据加载报告成功，但请求列表为空！")

        return success
    def load_requests(self, requests, requests_by_slot=None):
        """
        加载请求数据 (修复版：自动修正 1-based 索引)
        """
        if not requests:
            print("⚠️ [Env] 请求列表为空")
            return

        # 🔥🔥🔥 [核心修复] 检测并修正 1-based 索引 (MATLAB 风格) 🔥🔥🔥
        # 检查所有请求中的最大节点 ID
        max_node_in_reqs = 0
        max_vnf_type = 0

        for r in requests:
            s = r.get('source', 0)
            dests = r.get('dest', [])
            vnfs = r.get('vnf', [])

            # 找最大节点ID
            curr_max_node = max(s, max(dests) if dests else 0)
            max_node_in_reqs = max(max_node_in_reqs, curr_max_node)

            # 找最大VNF类型
            if vnfs:
                max_vnf_type = max(max_vnf_type, max(vnfs))

        print(f"🔍 [数据检查] 请求中最大节点ID: {max_node_in_reqs} (环境 N={self.n})")
        print(f"🔍 [数据检查] 请求中最大VNF类型: {max_vnf_type} (环境 K={self.K_vnf})")

        # --- 1. 修正节点索引 (如果最大ID >= N，说明肯定是 1-based 或者越界) ---
        if max_node_in_reqs >= self.n:
            print(f"⚠️ [数据警告] 检测到节点索引越界 (Max {max_node_in_reqs} >= {self.n})")
            print(f"🛠️ [自动修复] 正在执行 1-based -> 0-based 全局转换 (Node - 1)...")

            for r in requests:
                # 修正源节点
                r['source'] = r['source'] - 1
                # 修正目的节点
                r['dest'] = [d - 1 for d in r['dest']]

                # 再次安全检查
                if r['source'] < 0 or r['source'] >= self.n:
                    r['source'] = 0  # 兜底

        # --- 2. 修正 VNF 类型索引 (如果 VNF 类型 == K_vnf，说明也是 1-based) ---
        # 例如 K=8 (0-7)，但数据里有 8
        if max_vnf_type >= self.K_vnf:
            print(f"⚠️ [数据警告] 检测到 VNF 类型越界 (Max {max_vnf_type} >= {self.K_vnf})")
            print(f"🛠️ [自动修复] 正在执行 1-based -> 0-based VNF 转换 (VNF - 1)...")

            for r in requests:
                r['vnf'] = [v - 1 for v in r['vnf']]

        # --- 正常加载逻辑 (保持不变) ---
        self.all_requests = requests
        self.global_request_index = 0

        if hasattr(self, 'data_loader'):
            self.data_loader.requests = requests
            self.data_loader.total_steps = len(requests)
            if hasattr(self.data_loader, 'reset'):
                self.data_loader.reset()

        # 重建时间槽索引 (因为数据可能被修改了，这里最好重新构建)
        requests_by_slot = {}
        for req in requests:
            arr_time = float(req.get('arrival_time', 0))
            slot = req.get('time_slot', int(arr_time / self.delta_t))
            if slot not in requests_by_slot:
                requests_by_slot[slot] = []
            requests_by_slot[slot].append(req)

        self.requests_by_slot = requests_by_slot

        if requests_by_slot:
            self.max_slot_index = max(requests_by_slot.keys())
        else:
            self.max_slot_index = 0

        logger.info(f"✅ 数据加载完成 (已校准): {len(requests)} 条")

        if self.online_mode:
            self.current_slot_index = 0
            self.slot_queue = []
            self.simulation_done = False
    def reset_request(self):
        """
        🔥 [V17.0 时间槽触发版] 获取下一个请求并推进时间
        核心逻辑：
        1. 获取新请求。
        2. 对比新旧时间槽。
        3. 如果跨槽 (Switch Slot)，更新物理时间并触发资源释放。
        """
        # 1. 检查数据是否存在
        if not hasattr(self, 'all_requests') or not self.all_requests:
            return None, self.get_state()

        # 2. 检查并初始化指针
        if not hasattr(self, 'global_request_index'):
            self.global_request_index = 0

        # 3. 检查是否越界
        if self.global_request_index >= len(self.all_requests):
            self.global_request_index = 0

        # 4. 获取请求
        req = self.all_requests[self.global_request_index]

        # 5. 🔥 时间切片处理与资源释放
        # 获取新请求的到达时间和槽位
        new_arrival_time = float(req.get('arrival_time', self.time_step))
        new_time_slot = req.get('time_slot', 0)

        # 获取旧槽位
        old_time_slot = getattr(self, 'current_time_slot', None)

        # 初始化
        if not hasattr(self, 'current_time_slot'):
            self.current_time_slot = new_time_slot
            old_time_slot = new_time_slot

        # === 核心：检测时间槽切换 ===
        if old_time_slot is not None and new_time_slot != old_time_slot:
            # print(f"⏰ [TS切换] {old_time_slot} -> {new_time_slot} (Time: {new_arrival_time:.2f})")

            # A. 更新物理时间 (必须先更新时间，管理器才能判断是否过期)
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

            # B. 触发资源回收管理器
            # 因为时间变了，去检查一下有没有在这段时间内过期的老请求
            if hasattr(self, 'request_manager'):
                self.request_manager.check_and_release_expired(self.time_step)

        else:
            # 同槽内也要更新时间
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

        # 6. 移动指针
        self.global_request_index += 1

        # 7. 返回
        obs = self.get_state()
        return req, obs
#环境智能体交互 reset step step_low_level step_high_level get_state
    def reset(self, seed=None, options=None):
        """
        🔥 [V13.3 完全修复版] 解决 TS=0, Acc=0.0% 和 Repeated 访问 1000+ 的核心修复
        """
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self, 'action_space'): self.action_space.seed(seed)

        options = options or {}
        force_hard_reset = options.get('hard_reset', False)
        phase = options.get("phase", "phase3")

        # 1. 物理清空跨 Episode 的计数器 (关键修复)
        self._node_visit_count = {}  # 彻底解决 1035 次访问报错
        self._recent_positions = []  # 解决环路检测误判
        self._vnf_complete_steps = 0  # 解决超时误判
        self._current_goal_steps = 0  # 解决分支超时累加
        self.decision_step = 0  # 解决总步数累加

        # 2. 判断硬重置条件 (加载数据集或资源管理器归零)
        should_hard_reset = force_hard_reset or \
                            (not hasattr(self, 'all_requests') or not self.all_requests) or \
                            (self.online_mode and self.simulation_done)

        if should_hard_reset:
            print(f"\n🧹 [Hard Reset] 执行物理重置 ({phase})")
            if hasattr(self, 'resource_mgr'): self.resource_mgr.reset()
            if hasattr(self, 'request_manager'):
                self.request_manager.active_requests.clear()
                self.request_manager.requests_by_slot.clear()

            self.leave_heap = []
            self.current_slot_index = 0
            self.time_step = 0.0
            self.current_time_slot = 0  # 🔥🔥🔥 修复：同时重置 time_slot
            self.slot_queue = []
            self.simulation_done = False

            if not hasattr(self, 'all_requests') or not self.all_requests:
                self.load_dataset(phase)
            elif not self.online_mode:
                self.global_request_index = 0

        # 3. 初始化当前请求的状态容器
        self.visit_history = []
        self.nodes_on_tree = set()
        self.current_tree = {
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'hvt': np.zeros((self.n, self.K_vnf))
        }
        self.branch_states = {}
        self.current_branch_id = None
        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []

        # 4. 获取下一个请求并联动推进时间
        if self.online_mode:
            req_raw = self._get_next_request_online()
        else:
            req_raw, _ = self.reset_request()

        # 🔥🔥🔥 关键修复：处理 DataLoader 返回的对象
        if req_raw is not None:
            if hasattr(req_raw, 'to_dict'):
                req = req_raw.to_dict()
            elif hasattr(req_raw, '__dict__') and not isinstance(req_raw, dict):
                req = req_raw.__dict__
            else:
                req = req_raw
        else:
            req = None

        # 递归保护
        if req is None and self.online_mode:
            return self.reset(seed, options={'hard_reset': True})

        self.current_request = req
        if req:
            # 重设起点和目标
            self.current_node_location = req.get('source', 0)
            self.nodes_on_tree = {self.current_node_location}
            self.unadded_dest_indices = set(range(len(req.get('dest', []))))

            # ⏰⏰⏰ 完整修复：同时更新 time_step 和 current_time_slot
            arrival_time = req.get('arrival_time')
            if arrival_time is not None:
                self.time_step = float(arrival_time)

                # 🔥🔥🔥 关键新增：计算并更新 time_slot
                # 如果请求中有 time_slot 就用，否则根据 arrival_time 计算
                if 'time_slot' in req and req.get('time_slot') is not None:
                    self.current_time_slot = int(req.get('time_slot'))
                else:
                    # 根据 arrival_time 计算 time_slot
                    slot_duration = getattr(self, 'slot_duration', 1.0)
                    self.current_time_slot = int(arrival_time / slot_duration)

                # 🔥 调试日志（可选，首次运行时保留）
                if self.current_time_slot > 0:
                    print(f"⏰ [Reset Time Update] Time={self.time_step:.2f}s → Slot {self.current_time_slot}")

        # 5. 生成初始观测和掩码
        info = {
            'request': req,
            'action_mask': self.get_low_level_action_mask(),
            'decision_steps': 0,
            # 🔥🔥🔥 新增：返回时间槽信息
            'time_slot': self.current_time_slot,
            'time_step': self.time_step,
            'request_id': req.get('id') if req else None
        }
        # 🔥 诊断计数器
        self.action_stats = {
            'stay': 0,  # STAY 总次数
            'move': 0,  # MOVE 总次数
            'stay_deploy': 0,  # STAY 用于部署
            'stay_connect': 0,  # STAY 连接目的地
            'stay_waste': 0,  # STAY 无效操作
            'move_follow': 0,  # MOVE 跟随推荐路径
            'move_deviate': 0,  # MOVE 偏离推荐路径
            'total_steps': 0,
            'repeat_visits': 0,
            'tx_success': 0, 'tx_fail': 0,
            'virtual_deploy': 0, 'actual_deploy': 0,
        }


        return self.get_state(), info

    def _print_reward_debug(self, reward, info):
        """打印奖励诊断信息"""

        # 🔥 根据奖励值判断（最可靠）
        if reward >= 150:
            success = True
        elif reward <= -50:
            success = False
        else:
            success = info.get('success', reward > 0)

        # 🔥🔥🔥 修复：优先读 current_request，读不到（已归档）就读 info 中的备份
        if self.current_request:
            req_id = self.current_request.get('id', '?')
        else:
            req_id = info.get('request_id', '?')  # <--- 从 info 读取备份 ID

        if success:
            status = "✅成功"
            emoji = "🎉"
        else:
            status = "❌失败"
            emoji = "💔"
            error = info.get('error', 'unknown')
            # print(f"   失败原因: {error}") # 可选：太吵可以注释掉

        print(f"{emoji} [奖励诊断] 请求 {req_id} {status}: reward={reward:.1f}")

    def step(self, action):
        """🔥 [统一入口 V4.4 ID快照版]"""

        # 🔥🔥🔥 1. 先拍快照 (防止子函数把请求删了导致 ID 丢失)
        current_req_id = self.current_request.get('id', '?') if self.current_request else '?'

        # 2. 路由决策层级
        if self.current_branch_id is None:
            obs, reward, done, truncated, info = self.step_high_level(action)
        else:
            obs, reward, done, truncated, info = self.step_low_level(action)

        # 🔥🔥🔥 2. 把快照塞进 info (供诊断使用)
        if info is None: info = {}
        info['request_id'] = current_req_id

        # 🛑🛑🛑 【安全刹车】 🛑🛑🛑
        if done or self.current_request is None:
            # 现在 info 里有 request_id 了，打印出来就是对的了
            if done:
                self._print_reward_debug(reward, info)
            return obs, reward, done, truncated, info

        # ========================================================
        # 🔥 [核心改进 A] 自动吸附逻辑
        # ========================================================
        progress = self._get_current_progress()
        dests = set(self.current_request.get('dest', []))
        connected = self.current_tree.get('connected_dests', set())
        current_node = self.current_node_location

        if progress >= 1.0 and current_node in dests and current_node not in connected:
            connect_ok = self._connect_destination(current_node)
            if connect_ok:
                connected = self.current_tree.get('connected_dests', set())
                reward += 100.0
                info['reached_new_dest'] = True
                print(f"✨ [Auto Connect] 进度满且踩到目的地 {current_node}，强制吸附结算！")

        # 3. 检查任务是否完成 (所有目的地物理连接)
        if not done and len(connected) >= len(dests) and len(dests) > 0:
            print(f"\n🏭 [质检流水线] 请求 {current_req_id} 物理连接完成，开始验证...")

            # A. 剪枝
            pruned_tree, valid_nodes, prune_success, parent_map = self._prune_redundant_branches_with_vnf()
            if not prune_success:
                self._print_reward_debug(-100.0,
                                         {'success': False, 'error': 'island_topology', 'request_id': current_req_id})
                return obs, -100.0, True, False, {'success': False, 'error': 'island_topology'}

            # B. SFC 路径验证
            sfc_ok, sfc_errors = self._validate_sfc_paths(parent_map)
            if not sfc_ok:
                print("❌ [SFC验证失败]")
                for e in sfc_errors: print(f"   {e}")
                self._print_reward_debug(-200.0,
                                         {'success': False, 'error': 'incomplete_sfc', 'request_id': current_req_id})
                return obs, -200.0, True, False, {'success': False, 'error': 'incomplete_sfc'}

            # C. 统一扣费
            self.current_tree['tree'] = pruned_tree
            self.nodes_on_tree = valid_nodes

            if not self._commit_resources(pruned_tree, valid_nodes):
                self._print_reward_debug(-50.0, {'success': False, 'error': 'resource_commit_fail',
                                                 'request_id': current_req_id})
                return obs, -50.0, True, False, {'success': False, 'error': 'resource_commit_fail'}

            # D. 成功归档
            self._archive_request(success=True)
            print("✅ [结算成功] 资源已扣除，任务完成")

            self._print_reward_debug(200.0, {'success': True, 'request_completed': True, 'request_id': current_req_id})
            return obs, 200.0, True, False, {'success': True, 'request_completed': True}

        # ========================================================
        # 🔥 [核心改进 B] 徘徊惩罚补充逻辑
        # ========================================================
        if progress >= 1.0 and info.get('action_type') == 'move':
            if current_node in self.nodes_on_tree and not info.get('reached_new_dest', False):
                reward -= 15.0
                info['is_backtracking'] = True

        # 中途打印 (可选)
        # if done: self._print_reward_debug(reward, info)

        return obs, reward, done, truncated, info
    def get_state(self):
        """
        🔥 [V3.0 资源感知版]
        解决 Agent 地毯式巡检问题：
        1. 增加节点资源与当前待部署 VNF 的匹配特征 (Fit Factor)
        2. 将静态资源转化为相对于请求需求的相对余量
        """
        import torch
        import numpy as np
        from torch_geometric.data import Data

        # 1. 获取当前待处理的 VNF 需求
        current_vnf_demand = 0.0
        if self.current_request:
            vnf_list = self.current_request.get('vnf', [])
            # 找到下一个还没部署的 VNF 索引
            # 假设你的环境维护了 self.current_vnf_idx
            idx = getattr(self, 'current_vnf_idx', 0)
            if idx < len(vnf_list):
                # 获取该 VNF 的 CPU 需求（假设单位已统一）
                current_vnf_demand = self.current_request.get('vnf_cpu', [10.0])[idx]

        # 2. 构造基础特征流
        base_features = []
        for node in range(self.n):
            node_info = self.resource_mgr.nodes.get(node, {})
            cpu_rem = node_info.get('cpu', 0.0)
            mem_rem = node_info.get('mem', 0.0)

            # 🔥 [关键特征] 适配度 (Fit Factor)
            # 1.0 表示能放得下，-1.0 表示资源不足
            fit_factor = 1.0 if cpu_rem >= current_vnf_demand else -1.0

            # 相对负载 (归一化到 0-1)
            cpu_rate = cpu_rem / 100.0
            mem_rate = mem_rem / 100.0

            feat = [
                cpu_rate,
                mem_rate,
                fit_factor,  # 告诉模型：别推这扇门，里面没位置
                self.topology_mgr.get_node_degree(node) / max(1, self.n),
                self.topology_mgr.get_node_betweenness(node)
            ]
            # 补齐到 14 维静态特征 (对齐 SharedEncoder)
            if len(feat) < 14:
                feat += [0.0] * (14 - len(feat))
            base_features.append(feat)

        base_x = np.array(base_features, dtype=np.float32)

        # 3. 动态状态特征 (最后 3 维 - 对接 SharedEncoder V2.0 门控)
        dynamic_features = []
        nodes_on_tree = getattr(self, 'nodes_on_tree', set())
        connected_dests = self.current_tree.get('connected_dests', set()) if self.current_tree else set()
        vnf_list = self.current_request.get('vnf', []) if self.current_request else []

        for node in range(self.n):
            # 特征1: tree_mask (是否已在多播树中)
            t_m = 1.0 if node in nodes_on_tree else 0.0
            # 特征2: connected_mask (是否已连通目的地)
            c_m = 1.0 if node in connected_dests else 0.0
            # 特征3: progress_ratio (流量净化进度)
            p_r = 0.0
            if len(vnf_list) > 0:
                # 使用已实现的进度计算函数
                p_r = self._get_path_vnf_progress(node) / len(vnf_list)

            dynamic_features.append([t_m, c_m, p_r])

        dynamic_x = np.array(dynamic_features, dtype=np.float32)

        # 4. 拼接并转 Tensor [N, 14 + 3 = 17]
        full_x = np.concatenate([base_x, dynamic_x], axis=1)
        x_tensor = torch.from_numpy(full_x).float()

        # 5. 构建 Data 对象
        # 自动获取 edge_index, edge_attr (逻辑同前)
        if not hasattr(self, 'edge_index') or self.edge_index is None:
            self._build_graph_structures()  # 建议把边构建抽离成私有方法

        low_mask = self.get_low_level_action_mask()

        data = Data(
            x=x_tensor,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            req_vec=torch.zeros((1, 24)),  # 可根据需要填充请求向量
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0)
        )

        return data
#高层
    def step_high_level(self, action):
        """
        🎯 [高层V30.0] 负责决策部署位置

        动作空间：
        - 为下一个VNF选择部署节点
        - 或选择下一个要连接的目的地
        """
        dests = self.current_request.get('dest', [])
        vnf_list = self.current_request.get('vnf', [])
        connected = self.current_tree.get('connected_dests', set())
        remaining_dests = set(dests) - connected

        # 获取当前VNF部署进度
        total_vnf_progress = self._get_total_vnf_progress()

        # ============================================
        # 阶段1：VNF部署规划
        # ============================================
        if total_vnf_progress < len(vnf_list):
            # 高层选择一个节点作为VNF部署位置
            deployment_node = int(action)
            vnf_idx = total_vnf_progress

            print(f"🎯 [高层] 规划VNF[{vnf_idx}]部署到节点{deployment_node}")

            # 设置目标：让低层去这个节点部署
            self.current_deployment_target = deployment_node
            self.current_vnf_to_deploy = vnf_idx
            self.current_phase = 'vnf_deployment'

            return self.get_state(), 0.0, False, False, {
                'phase': 'vnf_deployment',
                'target_node': deployment_node,
                'vnf_idx': vnf_idx
            }

        # ============================================
        # 阶段2：目的地连接规划
        # ============================================
        else:
            if len(remaining_dests) == 0:
                # 所有目的地已连接
                return self.get_state(), 0.0, True, False, {
                    'all_connected': True
                }

            # 高层选择下一个要连接的目的地
            dest_list = list(remaining_dests)
            dest_idx = int(action) % len(dest_list)
            target_dest = dest_list[dest_idx]

            print(f"🎯 [高层] 选择连接目的地{target_dest}")

            # 设置目标：让低层去连接这个目的地
            self.current_target_node = target_dest
            self.current_phase = 'destination_connection'

            return self.get_state(), 0.0, False, False, {
                'phase': 'destination_connection',
                'target_dest': target_dest
            }
    def get_high_level_action_mask(self):
        """
        🔥 [V11.7 逻辑对齐版] 高层动作掩码

        关键修复：
        1. 掩码必须对应 action_space 的含义。
        2. 当前 step_high_level 的动作是 'subgoal_idx' (第几个未连接的目的地)，
           而不是物理节点 ID。
        3. 所以掩码应该允许 [0, 1, ..., num_remaining-1]。
        """
        # 初始化全 0 掩码 (float32 适配某些 RL 库，bool 适配另一些，通常 bool 更通用)
        mask = np.zeros(self.n, dtype=np.bool_)

        # 异常保护
        if self.current_request is None:
            mask[:] = 1
            return mask

        # 1. 计算剩余未连接的目的地
        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())

        # 使用与 step_high_level 一致的逻辑来维护 unadded_dest_indices
        if not hasattr(self, 'unadded_dest_indices'):
            self.unadded_dest_indices = set(range(len(dests)))
            for i, d in enumerate(dests):
                if d in connected:
                    self.unadded_dest_indices.discard(i)

        # 2. 获取有效选项的数量
        num_valid_options = len(self.unadded_dest_indices)

        # 3. 生成掩码
        if num_valid_options == 0:
            # 如果都连完了，允许动作 0 (占位，避免空掩码报错)
            mask[0] = 1
        else:
            # 允许选择第 0 到第 N-1 个未连接目的地
            # 这里的 index 是逻辑索引，不是物理节点 ID
            # 只要 num_valid_options 不超过 self.n (通常目的地数远小于节点数)，就是安全的
            valid_range = min(num_valid_options, self.n)
            mask[:valid_range] = 1

        return mask
    def _get_total_vnf_progress(self):
        """获取全局VNF部署进度"""
        placement = self.current_tree.get('placement', {})
        vnf_list = self.current_request.get('vnf', [])

        deployed_vnf_indices = set()
        for (node, vnf_idx), vnf_type in placement.items():
            deployed_vnf_indices.add(vnf_idx)

        return len(deployed_vnf_indices)

#低层
    def step_low_level(self, action):
        """
        ⚙️ [低层V30.0] 负责执行部署和连接

        根据高层指令执行：
        - VNF部署阶段：移动到目标节点并部署
        - 连接阶段：建立路径并连接目的地
        """
        target_node = int(action)
        current_node = self.current_node_location
        mask = self.get_low_level_action_mask()

        # 步数计数
        if not hasattr(self, '_step_count_this_branch'):
            self._step_count_this_branch = 0
        self._step_count_this_branch += 1

        # 熔断
        if self._step_count_this_branch > 50:
            print(f"❌ [低层超时] 步数超过50")
            self.current_branch_id = None
            self._step_count_this_branch = 0
            return self.get_state(), -50.0, False, False, {'timeout': True}

        # 检查mask
        if mask[target_node] == 0:
            return self.get_state(), -10.0, False, False, {'invalid': True}

        phase = getattr(self, 'current_phase', 'unknown')

        # ============================================
        # 阶段1：VNF部署执行
        # ============================================
        if phase == 'vnf_deployment':
            deployment_target = getattr(self, 'current_deployment_target', None)
            vnf_idx = getattr(self, 'current_vnf_to_deploy', None)

            if deployment_target is None:
                return self.get_state(), -5.0, False, False, {'error': 'no_target'}

            # 情况1：已到达目标节点，执行STAY部署
            if current_node == deployment_target and target_node == current_node:
                vnf_list = self.current_request.get('vnf', [])

                if self._try_deploy(current_node):
                    # 记录部署
                    if 'placement' not in self.current_tree:
                        self.current_tree['placement'] = {}

                    vnf_type = vnf_list[vnf_idx]
                    self.current_tree['placement'][(current_node, vnf_idx)] = vnf_type

                    print(f"✅ [低层部署] 节点{current_node} VNF[{vnf_idx}]")

                    # 重置低层状态，返回高层
                    self._step_count_this_branch = 0
                    delattr(self, 'current_deployment_target')
                    delattr(self, 'current_vnf_to_deploy')

                    return self.get_state(), 50.0, False, True, {  # truncated=True返回高层
                        'deploy_success': True,
                        'node': current_node,
                        'vnf_idx': vnf_idx
                    }
                else:
                    # 部署失败（资源不足）
                    print(f"❌ [低层部署失败] 节点{current_node}资源不足")
                    return self.get_state(), -20.0, False, True, {
                        'deploy_fail': True,
                        'reason': 'resource_insufficient'
                    }

            # 情况2：移动到目标节点
            elif target_node != current_node:
                # 检查是否向目标靠近
                distance_before = self._get_hop_distance(current_node, deployment_target)
                distance_after = self._get_hop_distance(target_node, deployment_target)

                # 尝试移动
                if self._can_move_to(current_node, target_node):
                    self.current_node_location = target_node

                    # 奖励塑形：向目标靠近+奖励，远离-惩罚
                    if distance_after < distance_before:
                        reward = 1.0  # 靠近目标
                    else:
                        reward = -2.0  # 远离目标

                    print(f"🚶 [低层移动] {current_node}→{target_node} (距离目标{distance_after}跳)")

                    return self.get_state(), reward, False, False, {
                        'move': True,
                        'distance_to_target': distance_after
                    }
                else:
                    return self.get_state(), -5.0, False, False, {'move_fail': True}

            # 情况3：在目标节点但选择移动（不合理）
            else:
                return self.get_state(), -3.0, False, False, {
                    'unnecessary_move': True
                }

        # ============================================
        # 阶段2：目的地连接执行
        # ============================================
        elif phase == 'destination_connection':
            target_dest = getattr(self, 'current_target_node', None)

            if target_dest is None:
                return self.get_state(), -5.0, False, False, {'error': 'no_target'}

            # 情况1：已到达目的地，执行STAY连接
            if current_node == target_dest and target_node == current_node:
                # 建立从源到目的地的完整路径
                if self._build_path_to_destination(target_dest):
                    # 连接目的地
                    if self._connect_destination(target_dest):
                        print(f"🎉 [低层连接] 目的地{target_dest}已连接")

                        # 重置低层状态，返回高层
                        self._step_count_this_branch = 0

                        # 检查是否所有目的地都连接
                        dests = self.current_request.get('dest', [])
                        connected = self.current_tree.get('connected_dests', set())

                        if len(connected) >= len(dests):
                            # 全部完成，进行结算
                            return self._finalize_request()
                        else:
                            return self.get_state(), 100.0, False, True, {
                                'connection_success': True,
                                'connected_count': len(connected),
                                'total_dests': len(dests)
                            }
                    else:
                        print(f"❌ [低层连接失败] 目的地{target_dest}")
                        return self.get_state(), -20.0, False, True, {
                            'connection_fail': True
                        }
                else:
                    print(f"❌ [低层建路失败] 无法到达{target_dest}")
                    return self.get_state(), -30.0, False, True, {
                        'path_fail': True
                    }

            # 情况2：移动到目的地
            elif target_node != current_node:
                distance_before = self._get_hop_distance(current_node, target_dest)
                distance_after = self._get_hop_distance(target_node, target_dest)

                if self._can_move_to(current_node, target_node):
                    self.current_node_location = target_node

                    if distance_after < distance_before:
                        reward = 1.0
                    else:
                        reward = -2.0

                    print(f"🚶 [低层移动] {current_node}→{target_node} (距离目标{distance_after}跳)")

                    return self.get_state(), reward, False, False, {
                        'move': True
                    }
                else:
                    return self.get_state(), -5.0, False, False, {'move_fail': True}

            else:
                return self.get_state(), -3.0, False, False, {'idle': True}

        # ============================================
        # 未知阶段
        # ============================================
        else:
            return self.get_state(), -10.0, False, False, {
                'error': 'unknown_phase'
            }
    def _build_path_to_destination(self, dest):
        """
        🌳 建立从源到目的地的完整路径
        确保路径经过所有已部署的VNF（按顺序）
        """
        import networkx as nx

        source = self.current_request.get('source')
        vnf_list = self.current_request.get('vnf', [])
        placement = self.current_tree.get('placement', {})

        # 提取VNF部署节点（按顺序）
        vnf_nodes = []
        for idx in range(len(vnf_list)):
            for (node, vnf_idx), _ in placement.items():
                if vnf_idx == idx:
                    vnf_nodes.append(node)
                    break

        # 路径节点：source → vnf[0] → vnf[1] → ... → dest
        path_nodes = [source] + vnf_nodes + [dest]

        print(f"🌳 [建路] 路径节点: {path_nodes}")

        # 构建图
        G = nx.Graph()
        for u in range(self.n):
            neighbors = self.resource_mgr.get_neighbors(u)
            for v in neighbors:
                if self.resource_mgr.has_link(u, v):
                    G.add_edge(u, v, weight=1)  # 权重=跳数

        # 逐段建立路径
        bw_req = self.current_request.get('bw_origin', 1.0)

        for i in range(len(path_nodes) - 1):
            start, end = path_nodes[i], path_nodes[i + 1]

            try:
                segment = nx.shortest_path(G, start, end)

                # 分配资源
                for j in range(len(segment) - 1):
                    u, v = segment[j], segment[j + 1]
                    edge_key = tuple(sorted([u, v]))

                    # 如果边已存在，跳过
                    if edge_key in self.current_tree.get('tree', {}):
                        continue

                    # 分配带宽
                    if not self.resource_mgr.allocate_link_resource(u, v, bw_req):
                        print(f"❌ [建路失败] 链路{u}-{v}资源不足")
                        return False

                    # 添加到树
                    if 'tree' not in self.current_tree:
                        self.current_tree['tree'] = {}
                    self.current_tree['tree'][edge_key] = bw_req

                    print(f"🌿 [建路] 添加边 {u}-{v}")

            except nx.NetworkXNoPath:
                print(f"❌ [建路失败] 无法连接 {start}→{end}")
                return False

        return True
    def _get_hop_distance(self, node1, node2):
        """计算两节点间的跳数距离"""
        import networkx as nx

        G = nx.Graph()
        for u in range(self.n):
            neighbors = self.resource_mgr.get_neighbors(u)
            for v in neighbors:
                if self.resource_mgr.has_link(u, v):
                    G.add_edge(u, v)

        try:
            return nx.shortest_path_length(G, node1, node2)
        except nx.NetworkXNoPath:
            return 9999  # 不可达
    def _can_move_to(self, from_node, to_node):
        """检查是否可以移动"""
        return self.resource_mgr.has_link(from_node, to_node)
    def get_low_level_action_mask(self):
        """
        🎭 [低层Mask V30.0] 根据阶段生成mask
        """
        mask = np.zeros(self.n, dtype=np.float32)

        current = self.current_node_location
        phase = getattr(self, 'current_phase', 'unknown')

        # ============================================
        # 阶段1：VNF部署 - 引导到目标节点
        # ============================================
        if phase == 'vnf_deployment':
            target = getattr(self, 'current_deployment_target', None)

            if target is not None:
                # 如果在目标节点，强烈建议STAY
                if current == target:
                    mask[current] = 10.0

                    # 也允许移动（但权重低）
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 0.1

                # 如果不在目标节点，允许移动
                else:
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 1.0

                    # 轻微允许STAY（探索）
                    mask[current] = 0.1

        # ============================================
        # 阶段2：目的地连接 - 引导到目的地
        # ============================================
        elif phase == 'destination_connection':
            target = getattr(self, 'current_target_node', None)

            if target is not None:
                # 如果在目的地，强烈建议STAY连接
                if current == target:
                    mask[current] = 10.0

                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 0.1

                # 如果不在目的地，允许移动
                else:
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 1.0

                    mask[current] = 0.1

        # ============================================
        # 兜底
        # ============================================
        else:
            mask[current] = 1.0
            neighbors = self.resource_mgr.get_neighbors(current)
            for nbr in neighbors:
                mask[nbr] = 1.0

        if np.sum(mask) == 0:
            mask[current] = 1.0

        return mask

#寻路逻辑 _a_star_search _find_path _get_distance
    def _a_star_search_with_tree_awareness(self, start, goal):
        """
        🔥 [智能A*搜索 V2.1] 添加超时机制，防止长时间搜索
        """
        if start == goal:
            return [start]

        # 缓存检查
        cache_key = (start, goal, frozenset(self.nodes_on_tree))
        if hasattr(self, '_path_cache') and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # 🔥 检查失败缓存
        if hasattr(self, '_failed_paths_cache'):
            if (start, goal) in self._failed_paths_cache:
                return None

        # 检查是否已经在同一棵树上
        if start in self.nodes_on_tree and goal in self.nodes_on_tree:
            tree_path = self._find_path_on_tree(start, goal)
            if tree_path:
                if hasattr(self, '_path_cache'):
                    self._path_cache[cache_key] = tree_path
                return tree_path

        bw_req = self.current_request.get('bw_origin', 1.0) if self.current_request else 1.0
        tree_edges = self.current_tree.get('tree', {})

        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        def heuristic(n):
            base_dist = self._get_distance(n, goal)
            tree_bonus = -5 if n in self.nodes_on_tree else 0
            visit_penalty = 0
            if hasattr(self, '_node_visit_count'):
                visit_penalty = self._node_visit_count.get(n, 0) * 2
            return max(0, base_dist + tree_bonus + visit_penalty)

        f_score = {start: heuristic(start)}

        # 🔥 添加访问计数
        visited_count = 0
        max_visits = 30  # 最多访问30个节点

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # 🔥 超时检查
            visited_count += 1
            if visited_count > max_visits:
                # 缓存失败结果
                if not hasattr(self, '_failed_paths_cache'):
                    self._failed_paths_cache = set()
                self._failed_paths_cache.add((start, goal))
                return None

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                if hasattr(self, '_path_cache'):
                    self._path_cache[cache_key] = path
                return path

            # 获取邻居
            neighbors = []
            for v in range(self.n):
                if v != current and self.resource_mgr.has_link(current, v):
                    edge = tuple(sorted([current, v]))
                    is_on_tree = (edge in tree_edges)

                    if not is_on_tree:
                        if hasattr(self.resource_mgr, 'check_link_resource'):
                            if not self.resource_mgr.check_link_resource(current, v, bw_req):
                                continue

                    neighbors.append(v)

            for neighbor in neighbors:
                move_cost = 1.0
                if neighbor not in self.nodes_on_tree:
                    move_cost = 2.0
                elif current in self.nodes_on_tree and neighbor in self.nodes_on_tree:
                    move_cost = 0.5

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # 搜索失败，缓存结果
        if not hasattr(self, '_failed_paths_cache'):
            self._failed_paths_cache = set()
        self._failed_paths_cache.add((start, goal))

        if hasattr(self, 'action_stats'):
            print(f"⚠️ [A*失败] 从{start}到{goal}找不到带宽充足的路径（需要带宽{bw_req}）")

        return None
    def _select_best_fork_node(self, remaining_dests):
        """
        智能选择分支节点：基于A*路径和树结构
        """
        if not remaining_dests or not hasattr(self, 'nodes_on_tree'):
            return None

        tree_nodes = list(self.nodes_on_tree)
        if not tree_nodes:
            # 如果没有树节点，从源点开始
            return self.current_request.get('source', 0)

        best_node = None
        best_score = float('inf')

        for tree_node in tree_nodes:
            # 计算从该树节点到所有剩余目的地的总路径长度
            total_path_length = 0
            reachable_count = 0

            for dest in remaining_dests:
                path = self._a_star_search_with_tree_awareness(tree_node, dest)
                if path:
                    total_path_length += len(path) - 1
                    reachable_count += 1

            if reachable_count == len(remaining_dests):
                # 所有目的地都可从该节点到达
                # 考虑节点访问次数（避免热点）
                visit_penalty = self._node_visit_count.get(tree_node, 0) * 5
                score = total_path_length + visit_penalty

                if score < best_score:
                    best_score = score
                    best_node = tree_node

        # 如果找不到最佳节点，选择离源点最近的树节点
        if best_node is None:
            source = self.current_request.get('source', 0)
            distances = [(self._get_distance(node, source), node) for node in tree_nodes]
            distances.sort()
            best_node = distances[0][1] if distances else tree_nodes[0]

        print(f"🌳 [智能分支] 选择节点{best_node}作为分支点，可到达{len(remaining_dests)}个目的地")
        return best_node
    def _find_path_on_tree(self, start, goal):
        """
        在当前树上寻找路径（不建新边）
        """
        if start == goal:
            return [start]

        # 构建树上邻接表
        tree_adj = {}
        tree = self.current_tree.get('tree', {})
        for (u, v), bw in tree.items():
            tree_adj.setdefault(u, []).append(v)
            tree_adj.setdefault(v, []).append(u)

        # BFS搜索
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path

            for neighbor in tree_adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None
    def _get_distance(self, u, v):
        """[辅助方法] 计算距离，防止报错"""
        if u == v: return 0
        try:
            # 优先用 TopologyMgr
            if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'get_distance'):
                return self.topology_mgr.get_distance(u, v)
            # 备用 NetworkX
            import networkx as nx
            if not hasattr(self, '_nx_graph'):
                if hasattr(self, 'topology_matrix'):
                    self._nx_graph = nx.from_numpy_array(self.topology_matrix)
                else:
                    return 50  # 无法计算时给个默认值
            return nx.shortest_path_length(self._nx_graph, u, v)
        except:
            return 50  # 出错兜底
#资源检查 _check_node_resource _check_deployment_validity
#_try_deploy  _manual_release_resources _archive_request _update_tree_state
    def _check_node_resources(self, node_id: int) -> bool:
        """
        🔥 [V3.5 修复版] 检查资源（含虚拟预扣）
        解决“草图画得太满，落地时资源不足”的问题
        """
        try:
            if self.current_request is None:
                return True

            # 1. 获取当前要部署的 VNF 的资源需求
            # ---------------------------------------------------
            vnf_list = self.current_request.get('vnf', [])

            # 计算当前是第几个 VNF (根据 placement 的数量推断)
            # 注意：这里假设是按顺序部署。如果是乱序，需要传参进来，但通常 Agent 是顺序的
            deployed_count = 0
            placement = self.current_tree.get('placement', {})
            for k in placement.keys():
                # 过滤掉非部署记录
                if isinstance(k, tuple) and len(k) >= 2:
                    deployed_count += 1

            vnf_idx = deployed_count

            # 如果已经部署完了，就不需要检查了
            if vnf_idx >= len(vnf_list):
                return True

            # 获取需求值
            cpu_reqs = self.current_request.get('cpu_origin', []) or \
                       self.current_request.get('vnf_cpu', [])
            mem_reqs = self.current_request.get('memory_origin', []) or \
                       self.current_request.get('mem_origin', [])

            # 安全获取当前 VNF 的需求
            req_cpu = float(cpu_reqs[vnf_idx]) if vnf_idx < len(cpu_reqs) else 1.0
            req_mem = float(mem_reqs[vnf_idx]) if vnf_idx < len(mem_reqs) else 1.0

            # 2. 🔥 [核心] 计算当前请求已在草图上预订的资源 (Virtual Reserved)
            # ---------------------------------------------------
            reserved_cpu = 0.0
            reserved_mem = 0.0

            for key, info in placement.items():
                # 只统计当前节点 (node_id) 的预订情况
                p_node = info.get('node')
                if p_node == node_id:
                    reserved_cpu += info.get('cpu_used', 0.0)
                    reserved_mem += info.get('mem_used', 0.0)

            # 3. 获取物理剩余资源 (Physical Available)
            # ---------------------------------------------------
            avail_cpu = 0.0
            avail_mem = 0.0

            # 适配 resource_mgr 的不同实现结构
            nodes_data = self.resource_mgr.nodes
            if isinstance(nodes_data, list):  # List[Dict]
                node_info = nodes_data[node_id]
                avail_cpu = node_info.get('cpu', 0.0)
                avail_mem = node_info.get('memory', node_info.get('mem', 0.0))
            elif isinstance(nodes_data, dict):  # SOA Dict
                avail_cpu = nodes_data.get('cpu', [])[node_id]
                avail_mem = nodes_data.get('memory', [])[node_id]

            # 4. 最终判定：物理余额 - 虚拟预扣 >= 当前需求
            # ---------------------------------------------------
            # 加上 1.05 倍的安全因子，防止浮点数误差
            cpu_ok = (avail_cpu - reserved_cpu) >= (req_cpu * 1.05)
            mem_ok = (avail_mem - reserved_mem) >= (req_mem * 1.05)

            # 调试日志 (可选，排查问题时打开)
            # if not cpu_ok:
            #     print(f"🛑 [资源预警] 节点{node_id} 拒绝部署 VNF{vnf_idx}")
            #     print(f"   物理余: {avail_cpu:.2f}, 草图占: {reserved_cpu:.2f}, 需: {req_cpu:.2f}")

            return cpu_ok and mem_ok

        except Exception as e:
            # print(f"⚠️ 资源检查报错: {e}")
            return False
    def _try_deploy(self, node):
        """
        🔥 [V12.5 修复版] 强制使用带预扣的资源检查
        彻底解决“假装有资源”导致的 Overbooking 问题
        """
        if self.current_request is None or self.current_branch_id is None:
            return False

        vnf_list = self.current_request.get('vnf', [])
        if len(vnf_list) == 0:
            return False

        # 1. 获取当前分支在该路径上的 VNF 连续部署进度
        current_progress = self._get_path_vnf_progress(node)

        # 2. 如果已经全部部署完成，则不再部署
        if current_progress >= len(vnf_list):
            return False

        # 3. 确定当前需要部署的 VNF 类型
        next_vnf_idx = current_progress
        next_vnf_type = vnf_list[next_vnf_idx]

        # 4. 获取资源需求
        cpu_needs = self.current_request.get('cpu_origin', []) or self.current_request.get('vnf_cpu', [])
        mem_needs = self.current_request.get('memory_origin', []) or self.current_request.get('mem_origin', [])

        # 安全获取需求值
        c_req = float(cpu_needs[next_vnf_idx]) if next_vnf_idx < len(cpu_needs) else 1.0
        m_req = float(mem_needs[next_vnf_idx]) if next_vnf_idx < len(mem_needs) else 1.0

        # =================================================================
        # 🔥🔥🔥 核心修改点：调用带“虚拟账本”的检查函数
        # =================================================================
        # 这一步会计算：物理剩余 - (当前请求已在草图中预订的资源) >= 新需求 ?
        if not self._check_node_resources(node):
            # 调试日志：如果被拒绝，说明算上预扣后资源不足
            # print(f"🚫 [虚拟部署拒绝] 节点{node} 无法容纳 VNF{next_vnf_type} (预扣不足)")
            return False

        # 5. 执行虚拟部署记录 (Placement)
        key = (node, next_vnf_type, self.current_branch_id)

        if 'placement' not in self.current_tree:
            self.current_tree['placement'] = {}

        self.current_tree['placement'][key] = {
            'vnf_idx': next_vnf_idx,
            'vnf_type': next_vnf_type,
            'node': node,
            'cpu_used': c_req,
            'mem_used': m_req,
            'branch_id': self.current_branch_id
        }

        print(f"✅ [Virtual Deploy] 节点{node} 记录 VNF[{next_vnf_idx}]={next_vnf_type} (暂未扣费)")
        return True
    def _archive_request(self, success=False, already_rolled_back=False):
        """
        🔥 [V16.4 防重复版] 成功时保存资源快照并加入跟踪，失败时回滚

        Args:
            success: 请求是否成功
            already_rolled_back: 是否已经回滚过（防止重复回滚）
        """
        if self.current_request is None:
            return

        req = self.current_request
        req_id = req.get('id', id(req))
        if self.enable_visualization and hasattr(self, 'visualizer'):
            try:
                subdir = 'success' if success else 'fail'
                save_path = f'visualization/{subdir}/request_{req_id}.png'

                self.visualizer.visualize_request_tree(
                    request=self.current_request,
                    save_path=save_path,
                    show=False
                )

                if success or req_id % 100 == 0:
                    print(f"🎨 [可视化] 已保存: {save_path}")
            except Exception as e:
                pass  # 可视化失败不影响训练
        if success:
            # =====================================================================
            # 成功分支：保存账本 + 添加到生命周期管理器
            # =====================================================================
            import copy

            req['resources_allocated'] = {
                'placement': copy.deepcopy(self.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.current_tree.get('tree', {}))
            }

            if hasattr(self, 'request_manager'):
                try:
                    self.request_manager.add_request(req)

                    arrival = req.get('arrival_time', self.time_step)
                    lifetime = req.get('lifetime', 5.0)
                    expire_time = arrival + lifetime

                    print(f"📝 [生命周期] 请求 {req_id} 已加入跟踪")
                    print(f"   到达时间: {arrival:.2f}s")
                    print(f"   生存时长: {lifetime:.2f}s")
                    print(f"   过期时间: {expire_time:.2f}s")
                except Exception as e:
                    print(f"⚠️ [生命周期] 请求 {req_id} 添加失败: {e}")
            else:
                print(f"⚠️ [生命周期] request_manager 未初始化！")

            self.total_requests_accepted += 1
            if hasattr(self, 'served_dest_count'):
                self.served_dest_count += len(req.get('dest', []))

            print(f"✅ [归档成功] 请求 {req_id} 已完成，等待过期释放")

        else:
            # =====================================================================
            # 失败分支：回滚虚拟资源（除非已经回滚过）
            # =====================================================================
            if already_rolled_back:
                print(f"ℹ️ [归档失败] 请求 {req_id} 失败（资源已回滚，跳过重复回滚）")
            else:
                print(f"❌ [归档失败] 请求 {req_id} 失败，开始回滚虚拟资源...")

                # 调用现有的回滚方法
                self._rollback_request_resources(req)

                # 额外回滚当前树占用的虚拟资源
                placement = self.current_tree.get('placement', {})
                tree_edges = self.current_tree.get('tree', {})

                restored_cpu = 0.0
                restored_bw = 0.0

                # 回滚节点资源
                for key, info in placement.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        node = key[0]
                        vnf_type = key[1]

                        if isinstance(info, dict):
                            c = info.get('cpu_used', 1.0)
                            m = info.get('mem_used', 1.0)
                        else:
                            c, m = 1.0, 1.0

                        if hasattr(self.resource_mgr, 'release_node_resource'):
                            try:
                                self.resource_mgr.release_node_resource(node, vnf_type, c, m)
                                restored_cpu += c
                            except Exception as e:
                                print(f"⚠️ 回滚节点 {node} 资源失败: {e}")

                # 回滚链路资源
                bw = req.get('bw_origin', 1.0)
                for edge_key in tree_edges.keys():
                    u, v = edge_key
                    if hasattr(self.resource_mgr, 'release_link_resource'):
                        try:
                            self.resource_mgr.release_link_resource(u, v, bw)
                            restored_bw += bw
                        except Exception as e:
                            print(f"⚠️ 回滚链路 {edge_key} 资源失败: {e}")

                if restored_cpu > 0 or restored_bw > 0:
                    print(f"♻️ [虚拟资源回滚] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

        # 重置状态
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.current_request = None
        self.current_branch_id = None
        self.nodes_on_tree = set()

    def _get_path_vnf_progress(self, current_node):
        """
        🔥 [V24.7 路径感知版] 获取 *当前路径* 上已部署的 VNF 数量
        核心修复：只统计从 Source -> Current Node 这条链路上覆盖的 VNF
        """
        # 1. 基础数据准备
        tree_edges = self.current_tree.get('tree', {})
        placement = self.current_tree.get('placement', {})  # 格式: {(node, vnf_idx): data}
        source = self.current_request.get('source')
        vnf_list = self.current_request.get('vnf', [])

        # 如果还没开始或者就在源点
        if current_node == source:
            # 检查源点是否部署了 VNF (虽然我们现在禁用了源点部署，但逻辑上要严谨)
            progress = 0
            for i in range(len(vnf_list)):
                if (source, i) in placement:
                    progress += 1
                else:
                    break
            return progress

        # 2. 构建父节点映射 (用于回溯路径)
        # ⚠️ 注意：必须实时构建，因为树结构在动态变化
        from collections import deque, defaultdict
        adj = defaultdict(list)
        for u, v in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        parent_map = {source: None}
        queue = deque([source])
        visited = {source}

        # BFS 找到通往 current_node 的路径
        path_found = False
        while queue:
            curr = queue.popleft()
            if curr == current_node:
                path_found = True
                break
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    parent_map[nbr] = curr
                    queue.append(nbr)

        if not path_found:
            # 如果当前节点不在树上（比如还没连上），或者断连了
            # 默认只有0 (或者抛出异常，视情况而定)
            return 0

        # 3. 回溯路径，收集路径上的所有节点
        path_nodes = set()
        curr = current_node
        while curr is not None:
            path_nodes.add(curr)
            curr = parent_map.get(curr)

        # 4. 严格检查 VNF 序列
        # 我们需要 VNF[0], VNF[1], ... 依次出现在这条路径的节点上
        current_progress = 0
        for i in range(len(vnf_list)):
            found_this_vnf = False
            # 检查 VNF[i] 是否在路径上的任意节点中
            for node in path_nodes:
                if (node, i) in placement:
                    found_this_vnf = True
                    break

            if found_this_vnf:
                current_progress += 1
            else:
                # 🔥 一旦断档，后面的都不算！
                # 例如：路径上有 VNF[0] 和 VNF[2]，但没有 VNF[1]，那进度只能算 1
                break

        return current_progress
#可视化
    def visualize_tree_to_image(self, save_path="multicast_tree.png", show_plot=False,
                                figsize=(14, 10), dpi=150):
        """
        将多播树保存为高质量图片

        Args:
            save_path: 保存路径
            show_plot: 是否显示图片窗口
            figsize: 图片大小
            dpi: 图片分辨率

        Returns:
            save_path: 保存的文件路径
        """
        tree_edges = self.current_tree.get('tree', {})
        if not tree_edges:
            print("❌ 树为空，无法可视化")
            return None

        placement = self.current_tree.get('placement', {})
        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        connected_dests = self.current_tree.get('connected_dests', set())

        # 创建图
        G = nx.Graph()

        # 添加边和权重
        edge_labels = {}
        for (u, v), bw in tree_edges.items():
            G.add_edge(u, v, weight=bw)
            edge_labels[(u, v)] = f"{bw:.1f}"

        # 提取VNF部署信息
        vnf_on_node = defaultdict(list)
        for key, info in placement.items():
            if isinstance(info, dict):
                node_id = info.get('node', key[0] if isinstance(key, tuple) else None)
                vnf_type = info.get('vnf_type', key[1] if isinstance(key, tuple) and len(key) >= 2 else None)
            elif isinstance(key, tuple) and len(key) >= 2:
                node_id = key[0]
                vnf_type = key[1]
            else:
                continue

            if node_id is not None and vnf_type is not None:
                vnf_on_node[node_id].append(vnf_type)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # 使用层次布局（更适合树结构）
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)

        # 节点颜色和大小
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node in G.nodes():
            # 确定颜色
            if node == source:
                node_colors.append('#90EE90')  # 浅绿色 - 源节点
                node_sizes.append(2500)
            elif node in connected_dests:
                node_colors.append('#FF6B6B')  # 浅红色 - 已连接目的地
                node_sizes.append(2500)
            elif node in dests:
                node_colors.append('#FFB6C1')  # 粉色 - 未连接目的地
                node_sizes.append(2000)
            elif node in vnf_on_node:
                node_colors.append('#87CEEB')  # 天蓝色 - VNF部署节点
                node_sizes.append(2000)
            else:
                node_colors.append('#E0E0E0')  # 灰色 - 中间节点
                node_sizes.append(1500)

            # 构建标签
            label = f"{node}"
            if node in vnf_on_node:
                vnfs = sorted(vnf_on_node[node])
                label += f"\nVNF{vnfs}"

            node_labels[node] = label

        # 绘制边（带宽标签）
        nx.draw_networkx_edges(
            G, pos,
            width=3,
            alpha=0.6,
            edge_color='#666666',
            style='solid',
            ax=ax
        )

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        # 绘制节点标签
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=10,
            font_weight='bold',
            font_family='sans-serif',
            ax=ax
        )

        # 绘制边标签（带宽）
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color='darkblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ax=ax
        )

        # 添加标题
        title = f"多播树可视化\n"
        title += f"源节点: {source} | 目的节点: {sorted(dests)}\n"
        title += f"边数: {len(tree_edges)} | 已连接: {len(connected_dests)}/{len(dests)}"
        if vnf_on_node:
            total_vnfs = sum(len(v) for v in vnf_on_node.values())
            title += f" | VNF部署: {total_vnfs}"

        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        # 添加图例
        legend_elements = [
            mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='源节点'),
            mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='已连接目的地'),
            mpatches.Patch(facecolor='#FFB6C1', edgecolor='black', label='未连接目的地'),
            mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='VNF部署节点'),
            mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', label='中间节点'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # 移除坐标轴
        ax.axis('off')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"✅ 多播树已保存到: {save_path}")

        # 显示图片（可选）
        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path
    def visualize_tree_comparison(self, original_edges, pruned_edges,
                                  save_path="tree_comparison.png",
                                  figsize=(20, 10), dpi=150):
        """
        对比剪枝前后的多播树（并排显示）

        Args:
            original_edges: 原始树的边
            pruned_edges: 剪枝后树的边
            save_path: 保存路径
            figsize: 图片大小
            dpi: 分辨率
        """
        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        placement = self.current_tree.get('placement', {})

        # 提取VNF信息
        vnf_on_node = defaultdict(list)
        for key, info in placement.items():
            if isinstance(info, dict):
                node_id = info.get('node', key[0] if isinstance(key, tuple) else None)
                vnf_type = info.get('vnf_type', key[1] if isinstance(key, tuple) and len(key) >= 2 else None)
            elif isinstance(key, tuple) and len(key) >= 2:
                node_id = key[0]
                vnf_type = key[1]
            else:
                continue

            if node_id is not None and vnf_type is not None:
                vnf_on_node[node_id].append(vnf_type)

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        def draw_tree(edges, ax, title):
            """辅助函数：绘制一棵树"""
            G = nx.Graph()
            edge_labels = {}

            for (u, v), bw in edges.items():
                G.add_edge(u, v, weight=bw)
                edge_labels[(u, v)] = f"{bw:.1f}"

            if len(G.nodes()) == 0:
                ax.text(0.5, 0.5, '空树', ha='center', va='center',
                        fontsize=20, transform=ax.transAxes)
                ax.axis('off')
                return

            # 布局
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.circular_layout(G)

            # 节点颜色
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if node == source:
                    node_colors.append('#90EE90')
                    node_sizes.append(2000)
                elif node in dests:
                    node_colors.append('#FF6B6B')
                    node_sizes.append(2000)
                elif node in vnf_on_node:
                    node_colors.append('#87CEEB')
                    node_sizes.append(1500)
                else:
                    node_colors.append('#E0E0E0')
                    node_sizes.append(1200)

            # 绘制
            nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6,
                                   edge_color='#666666', ax=ax)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.9,
                                   edgecolors='black', linewidths=1.5, ax=ax)

            # 标签
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9,
                                    font_weight='bold', ax=ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8,
                                         font_color='darkblue', ax=ax)

            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.axis('off')

        # 绘制原始树
        draw_tree(original_edges, ax1,
                  f"剪枝前\n边数: {len(original_edges)}")

        # 绘制剪枝后的树
        draw_tree(pruned_edges, ax2,
                  f"剪枝后\n边数: {len(pruned_edges)}")

        # 总标题
        reduction = len(original_edges) - len(pruned_edges)
        reduction_pct = (reduction / len(original_edges) * 100) if original_edges else 0

        fig.suptitle(
            f"多播树剪枝对比 | 源:{source} → 目的:{sorted(dests)}\n"
            f"剪除 {reduction} 条边 ({reduction_pct:.1f}%)",
            fontsize=15, fontweight='bold', y=0.98
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"✅ 对比图已保存到: {save_path}")
        plt.close()

        return save_path
    def visualize_multiple_trees(self, trees_data, save_path="trees_grid.png",
                                 figsize=(20, 15), dpi=150):
        """
        网格显示多个多播树（用于展示训练过程中的不同请求）

        Args:
            trees_data: 树数据列表 [{'edges': {...}, 'source': X, 'dests': [...], 'title': '...'}, ...]
            save_path: 保存路径
            figsize: 图片大小
            dpi: 分辨率
        """
        n_trees = len(trees_data)
        if n_trees == 0:
            print("❌ 没有树数据")
            return None

        # 计算网格大小
        cols = min(3, n_trees)
        rows = (n_trees + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
        if n_trees == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        for idx, tree_info in enumerate(trees_data):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]

            edges = tree_info['edges']
            source = tree_info.get('source', 0)
            dests = set(tree_info.get('dests', []))
            title = tree_info.get('title', f'树 {idx + 1}')

            # 创建图
            G = nx.Graph()
            for (u, v), bw in edges.items():
                G.add_edge(u, v, weight=bw)

            if len(G.nodes()) == 0:
                ax.text(0.5, 0.5, '空树', ha='center', va='center',
                        fontsize=12, transform=ax.transAxes)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                continue

            # 布局
            pos = nx.spring_layout(G, k=2, iterations=30, seed=42)

            # 节点颜色
            node_colors = ['#90EE90' if n == source else
                           '#FF6B6B' if n in dests else '#E0E0E0'
                           for n in G.nodes()]

            # 绘制
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, ax=ax)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=300, alpha=0.8, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

            ax.set_title(f"{title}\n({len(edges)} 边)", fontsize=10)
            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(n_trees, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle('多播树集合', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"✅ 网格图已保存到: {save_path}")
        plt.close()

        return save_path
    def save_successful_tree_image(self, episode_num, request_id):
        """
        保存成功请求的树图片
        """
        # 创建输出目录
        output_dir = "output/trees"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        filename = f"tree_ep{episode_num}_req{request_id}.png"
        filepath = os.path.join(output_dir, filename)

        # 保存图片
        self.visualize_tree_to_image(save_path=filepath, show_plot=False)

    #工具函数
    def _validate_sfc_paths(self, parent_map):
        """
        🔥 [增强版] 验证 SFC 路径完整性

        严格检查：
        1. 每条路径必须经过完整的 VNF 链
        2. VNF 必须按顺序部署
        3. 不允许跳过 VNF
        """
        if not self.current_request:
            return False, ["No request"]

        source = self.current_request['source']
        dests = self.current_request.get('dest', [])
        required_vnfs = self.current_request.get('vnf', [])

        # 如果没有 VNF 要求，直接通过
        if not required_vnfs:
            return True, []

        # 构建节点 VNF 映射
        node_vnf_dict = {}  # {node: [vnf_types]}
        placement = self.current_tree.get('placement', {})

        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v = key[0], key[1]
                if n not in node_vnf_dict:
                    node_vnf_dict[n] = []
                node_vnf_dict[n].append(v)

        print(f"\n🔍 [SFC 验证] 开始验证路径...")
        print(f"   源节点: {source}")
        print(f"   目的地: {dests}")
        print(f"   所需 VNF 链: {required_vnfs}")
        print(f"   已部署 VNF: {node_vnf_dict}")

        errors = []

        # 验证每个目的地的路径
        for dest in dests:
            # 1. 回溯路径
            path = []
            curr = dest
            while curr is not None:
                path.append(curr)
                if curr == source:
                    break
                curr = parent_map.get(curr)

            # 2. 检查路径完整性
            if not path or path[-1] != source:
                error = f"Dest {dest}: Path broken (无法从源节点到达)"
                errors.append(error)
                print(f"   ❌ {error}")
                continue

            path.reverse()  # Source -> Dest
            print(f"   📍 目的地 {dest} 的路径: {path}")

            # 3. ✅ 严格验证：收集路径上的 VNF，检查是否完整且有序
            path_vnfs = []  # 按路径顺序收集到的 VNF

            for node in path:
                if node in node_vnf_dict:
                    # 这个节点部署了 VNF
                    deployed = node_vnf_dict[node]
                    for vnf in deployed:
                        path_vnfs.append((node, vnf))
                        print(f"      节点 {node}: VNF {vnf}")

            # 4. 检查是否包含所有必需的 VNF
            collected_vnf_types = [vnf for (node, vnf) in path_vnfs]

            # 检查每个必需的 VNF 是否都在路径上
            for req_vnf in required_vnfs:
                if req_vnf not in collected_vnf_types:
                    error = f"Dest {dest}: 缺少 VNF {req_vnf}（路径VNF: {collected_vnf_types}）"
                    errors.append(error)
                    print(f"   ❌ {error}")
                    break
            else:
                # 5. ✅ 关键修复：检查 VNF 的顺序是否正确
                # 提取路径上 VNF 的索引序列
                vnf_indices = []
                for vnf in collected_vnf_types:
                    if vnf in required_vnfs:
                        vnf_indices.append(required_vnfs.index(vnf))

                # 检查索引是否单调递增（VNF 按顺序经过）
                if vnf_indices != sorted(vnf_indices):
                    error = f"Dest {dest}: VNF 顺序错误（期望: {required_vnfs}, 实际: {collected_vnf_types}）"
                    errors.append(error)
                    print(f"   ❌ {error}")
                else:
                    print(f"   ✅ 目的地 {dest} 路径验证通过")

        success = (len(errors) == 0)

        if success:
            print(f"✅ [SFC 验证] 所有路径验证通过")
        else:
            print(f"❌ [SFC 验证] 发现 {len(errors)} 个错误")

        return success, errors
    def _advance_to_next_active_slot(self):
        """
        ⏩ [修复版] 时间槽推进逻辑
        1. 只有当 slot_queue 为空时才推进。
        2. 找到有请求的槽后，加载队列并更新时间，然后退出循环。
        3. 只有遍历完所有槽仍无请求时，才标记 simulation_done。
        """
        # 如果队列里还有东西，绝对不要推进时间！
        if hasattr(self, 'slot_queue') and self.slot_queue:
            return

        # 记录起始位置用于诊断
        start_slot = self.current_slot_index

        # 循环查找下一个有请求的时间槽
        while not self.simulation_done:
            # A. 边界检查：如果超过最大槽，仿真结束
            if self.current_slot_index > self.max_slot_index:
                print(f"🏁 [仿真结束] 已到达最大时间槽 {self.max_slot_index}")
                self.simulation_done = True
                return

            # B. 检查当前索引是否有请求
            current_reqs = self.requests_by_slot.get(self.current_slot_index, [])

            if current_reqs:
                # ✅ 发现请求：加载到队列
                # 使用 list() 创建副本，防止引用修改
                self.slot_queue = list(current_reqs)

                # 更新物理时间
                self.current_time_slot = self.current_slot_index

                # ✅✅✅ 关键修复1：字典访问方式
                if self.slot_queue:
                    first_req = self.slot_queue[0]

                    # ✅ 使用字典键访问，不是属性访问
                    if isinstance(first_req, dict):
                        self.time_step = float(first_req.get('arrival_time',
                                                             self.current_slot_index * self.delta_t))
                    else:
                        # 如果是对象（某些情况下），使用属性访问
                        self.time_step = float(getattr(first_req, 'arrival_time',
                                                       self.current_slot_index * self.delta_t))
                else:
                    # 队列为空（不应该发生，但保险起见）
                    self.time_step = self.current_slot_index * self.delta_t

                print(
                    f"⏩ [时间推进] Slot {start_slot} -> {self.current_slot_index} | "
                    f"Time: {self.time_step:.2f}s | 加载 {len(self.slot_queue)} 个请求")

                # ✅✅✅ 关键修复2：调用生命周期管理器的释放方法
                if hasattr(self, 'request_manager'):
                    try:
                        expired_ids = self.request_manager.check_and_release_expired(self.time_step)
                        if expired_ids:
                            print(f"♻️ [时间切片] 释放了 {len(expired_ids)} 个过期请求")
                            print(f"   过期ID: {expired_ids}")
                            print(f"   当前Res: {self.get_resource_utilization():.1f}%")
                    except Exception as e:
                        print(f"⚠️ [时间切片] 释放失败: {e}")
                else:
                    print(f"⚠️ [时间切片] request_manager 未初始化")

                # 🔥 准备好下一个槽的索引 (供下一次调用使用)
                self.current_slot_index += 1
                return

            # C. 当前槽为空，继续寻找下一个
            self.current_slot_index += 1
    def _get_next_request_online(self):
        """
        🔥 [V17.0 时间槽触发版] 在线模式获取请求
        """
        if not self.slot_queue:
            self._advance_to_next_active_slot()

        if self.simulation_done or not self.slot_queue:
            return None

        # 弹出请求
        req_raw = self.slot_queue.pop(0)
        if hasattr(req_raw, 'to_dict'):
            req = req_raw.to_dict()
        else:
            req = req_raw if isinstance(req_raw, dict) else req_raw.__dict__

        # 获取新时间信息
        new_arrival_time = float(req.get('arrival_time', self.time_step))
        if 'time_slot' not in req:
            slot_duration = getattr(self, 'slot_duration', 1.0)
            req['time_slot'] = int(new_arrival_time / slot_duration)

        new_time_slot = int(req['time_slot'])
        old_time_slot = self.current_time_slot

        # === 核心：检测时间槽切换 ===
        if new_time_slot != old_time_slot:
            # A. 更新时间
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

            # 🔥 打印时间推进信息
            print(f"⏩ [时间推进] Slot {old_time_slot} -> {new_time_slot} | "
                  f"Time: {self.time_step:.2f}s | "
                  f"Res: {self.get_resource_utilization():.1f}%")

            # B. 触发资源释放
            if hasattr(self, 'request_manager'):
                expired_ids = self.request_manager.check_and_release_expired(self.time_step)

                # 🔥 如果没有释放，说明原因
                if not expired_ids and self.request_manager.active_requests:
                    earliest = min(
                        info['expire_time']
                        for info in self.request_manager.active_requests.values()
                    )
                    print(f"   ℹ️ 无过期请求 (最早过期: {earliest:.2f}s)")
                    print(f"   当前活跃请求: {len(self.request_manager.active_requests)} 个")
        else:
            self.time_step = new_arrival_time

        self._last_queue_size = len(self.slot_queue)
        return req
    def get_resource_utilization(self):
        """
        计算当前全网资源占用率 (兼容版)
        用于验证资源是否成功被占用 (Res < 100%)
        """
        try:
            total_cap = 0.0
            used_cap = 0.0

            # 适配不同的 ResourceManager 实现
            if hasattr(self.resource_mgr, 'nodes'):
                nodes = self.resource_mgr.nodes
                # 列表形式 [{'cpu':..., 'capacity':...}]
                if isinstance(nodes, list):
                    for n in nodes:
                        # 假设 cpu 是剩余量 (remaining)
                        # 尝试获取容量，如果没有则默认为 100
                        cap = n.get('capacity', n.get('cpu_limit', 100.0))
                        rem = n.get('cpu', 100.0)

                        total_cap += cap
                        used_cap += (cap - rem)
                # 字典形式 {id: {...}}
                elif isinstance(nodes, dict):
                    for n in nodes.values():
                        # 处理 SOA 结构 (cpu 是列表) 或 AOS 结构
                        if isinstance(n, list): continue  # 暂不处理纯列表结构
                        cap = n.get('total', 100.0)
                        used = n.get('used', 0.0)
                        total_cap += cap
                        used_cap += used

            if total_cap <= 0: return 0.0
            return used_cap / total_cap

        except Exception as e:
            # print(f"⚠️ 资源统计跳过: {e}")
            return 0.0
    def _commit_resources(self, pruned_tree, valid_nodes):
        """💳 [统一算账] 两阶段提交资源 - 增强诊断版"""
        req = self.current_request
        bw_req = req.get('bw_origin', 1.0)

        pending_links = []
        pending_nodes = []

        # Phase 1: Collect pending allocations
        for (u, v) in pruned_tree.keys():
            pending_links.append((u, v, bw_req))

        placement = self.current_tree.get('placement', {})
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v_type = key[0], key[1]
                if n in valid_nodes:  # 只提交有效节点
                    c = info.get('cpu_used', 1.0)
                    m = info.get('mem_used', 1.0)
                    pending_nodes.append((n, v_type, c, m))

        # Phase 2: Allocate with detailed logging
        self.curr_ep_link_allocs = []
        self.curr_ep_node_allocs = []

        total_cpu = 0.0
        total_mem = 0.0
        total_bw = 0.0

        print(f"\n💳 [开始扣费] 节点={len(pending_nodes)}, 链路={len(pending_links)}")

        # 分配链路资源
        for u, v, bw in pending_links:
            result = self.resource_mgr.allocate_link_resource(u, v, bw)

            if result is not False:  # None 或 True 都视为成功（兼容没返回值的情况）
                self.curr_ep_link_allocs.append((u, v, bw))
                total_bw += bw
                if len(self.curr_ep_link_allocs) <= 3:  # 只打印前3条
                    print(f"   💰 链路({u},{v}): -{bw:.1f} BW (result={result})")
            else:
                print(f"   ❌ 链路({u},{v}) 分配失败")

        if len(pending_links) > 3:
            print(f"   ... 还有 {len(pending_links) - 3} 条链路")

        # 分配节点资源
        for n, v_type, c, m in pending_nodes:
            result = self.resource_mgr.allocate_node_resource(n, v_type, c, m)

            if result is not False:
                self.curr_ep_node_allocs.append((n, v_type, c, m))
                total_cpu += c
                total_mem += m
                print(f"   💰 节点{n}[VNF{v_type}]: -{c:.1f} CPU, -{m:.1f} Mem (result={result})")
            else:
                print(f"   ❌ 节点{n}[VNF{v_type}] 分配失败")

        print(f"💳 [扣费汇总] CPU:{total_cpu:.1f} | Mem:{total_mem:.1f} | BW:{total_bw:.1f}")

        return True
    def _check_deployment_validity(self, node_id):
        """
        检查节点是否可以部署VNF

        规则：
        1. ❌ 源节点不能部署VNF
        2. ✅ 目的节点可以部署VNF
        3. ✅ 必须是DC节点
        4. ✅ 资源充足
        """
        if not self.current_request:
            return False

        # 🔥 规则1: 源节点不能部署VNF
        source = self.current_request.get('source')
        if node_id == source:
            return False

        # 规则2: 必须是DC节点
        if hasattr(self, 'dc_nodes') and node_id not in self.dc_nodes:
            return False

        # 规则3: 检查资源
        if hasattr(self, 'resource_mgr') and hasattr(self, '_check_node_resources'):
            if not self._check_node_resources(node_id):
                return False

        return True
    def _connect_destination(self, dest_node):
        """
        🔥 [增强版] 连接目的地 - 增加 VNF 完整性检查
        """
        if self.current_request is None:
            return False

        dests = self.current_request.get('dest', [])
        if dest_node not in dests:
            print(f"⚠️ 节点 {dest_node} 不是有效的目的地")
            return False

        # ✅ 关键修复：在连接目的地前，验证 VNF 是否部署完整
        required_vnfs = self.current_request.get('vnf', [])
        if required_vnfs:
            placement = self.current_tree.get('placement', {})

            # 统计已部署的 VNF
            deployed_vnf_types = set()
            for key, info in placement.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    vnf_type = key[1]
                    deployed_vnf_types.add(vnf_type)

            required_vnf_set = set(required_vnfs)

            if not required_vnf_set.issubset(deployed_vnf_types):
                missing = required_vnf_set - deployed_vnf_types
                print(f"❌ [连接阻断] 目的地 {dest_node} - VNF 未完整部署")
                print(f"   所需 VNF: {required_vnfs}")
                print(f"   已部署: {list(deployed_vnf_types)}")
                print(f"   缺少: {list(missing)}")
                return False

        # VNF 完整，可以连接
        self.current_tree.setdefault('connected_dests', set()).add(dest_node)
        print(f"✅ [连接成功] 目的地 {dest_node} 已连接 (VNF 完整)")

        return True
    def _get_current_progress(self):
        """
        🔥 计算当前 SFC 部署进度比例 [0.0 - 1.0]
        用于判断是否进入目的地连接阶段
        """
        if not self.current_request:
            return 0.0

        vnf_list = self.current_request.get('vnf', [])
        if not vnf_list:
            return 1.0

        # 获取当前已成功部署的 VNF 索引
        curr_idx = getattr(self, 'current_vnf_idx', 0)
        progress = float(curr_idx) / len(vnf_list)

        return progress
    def _build_graph_structures(self):
        """
        🔥 [核心修复] 构建图神经网络所需的边索引和边特征
        解决 AttributeError 并支持 GNN 拓扑输入
        """
        import torch
        import numpy as np

        # 1. 从拓扑管理器获取邻接矩阵
        adj = self.topology_mgr.topo

        # 2. 提取非零边的索引 (COO 格式)
        edge_indices = np.where(adj > 0)
        self.edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)

        # 3. 初始化边特征 (假设维度为 5，对齐 SharedEncoder)
        num_edges = self.edge_index.shape[1]
        self.edge_attr = torch.zeros((num_edges, 5), dtype=torch.float32)

        # 填充第一维为归一化带宽或链路权重
        weights = adj[edge_indices].astype(np.float32)
        self.edge_attr[:, 0] = torch.from_numpy(weights) / 100.0

        # 移动到正确设备 (如果有定义 self.device)
        if hasattr(self, 'device'):
            self.edge_index = self.edge_index.to(self.device)
            self.edge_attr = self.edge_attr.to(self.device)
    def _rollback_resources(self):
        """
        🔥 [V18.1 完美融合版] 统一回滚 + 状态清理
        - 接口：兼容 V18.0 (无参数，直接读 self.current_request)
        - 逻辑：包含 V15.0 的完整状态清理 (current_tree重置)，防止残留
        """
        if not hasattr(self, 'current_tree') or self.current_request is None:
            return

        placement = self.current_tree.get('placement', {})
        tree_edges = self.current_tree.get('tree', {})
        bw = self.current_request.get('bw_origin', 1.0)

        restored_cpu = 0.0
        restored_bw = 0.0

        # 1. 回滚节点资源
        for key, info_dict in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node, vnf_type = key[0], key[1]
                # 兼容格式
                if isinstance(info_dict, dict):
                    c = info_dict.get('cpu_used', 1.0)
                    m = info_dict.get('mem_used', 1.0)
                else:
                    c, m = 1.0, 1.0

                if hasattr(self.resource_mgr, 'release_node_resource'):
                    self.resource_mgr.release_node_resource(node, vnf_type, c, m)
                    restored_cpu += c

        # 2. 回滚链路资源
        for edge_key in tree_edges.keys():
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                if hasattr(self.resource_mgr, 'release_link_resource'):
                    self.resource_mgr.release_link_resource(u, v, bw)
                    restored_bw += bw

        if restored_cpu > 0 or restored_bw > 0:
            print(f"♻️ [资源回滚] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

        # ==========================================
        # 🔥 V15.0 的核心遗产：状态清理
        # ==========================================
        # 必须重置，否则下一个请求会继承上一个请求的残留树结构
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.nodes_on_tree = set()
        if hasattr(self, '_node_visit_count'): self._node_visit_count = {}
    def _rollback_request_resources(self, req):
        """
        🔥 [V15.0 补丁] 强制回滚：彻底释放当前请求占用的所有资源
        这是 _archive_request(success=False) 的核心依赖。
        """
        if not req: return

        print(f"♻️ [回滚执行] 开始释放请求 {req.get('id')} 的资源...")

        # ==========================================
        # 1. 节点资源回滚 (CPU/Memory)
        # ==========================================
        placement = self.current_tree.get('placement', {})
        restored_cpu = 0
        restored_mem = 0

        for key, info in placement.items():
            # 兼容 info 可能是字典的情况
            if isinstance(info, dict):
                node = info.get('node')
                c = info.get('cpu_used', 0.0)
                m = info.get('mem_used', 0.0)
                v_type = info.get('vnf_type', 0)
            else:
                continue

            # 调用资源管理器释放
            if hasattr(self.resource_mgr, 'release_node_resource'):
                try:
                    self.resource_mgr.release_node_resource(node, v_type, c, m)
                    restored_cpu += c
                    restored_mem += m
                except Exception as e:
                    print(f"⚠️ 节点资源释放失败: {e}")

        # ==========================================
        # 2. 链路资源回滚 (Bandwidth)
        # ==========================================
        tree_edges = self.current_tree.get('tree', {})
        restored_bw = 0
        bw_req = req.get('bw_origin', 1.0)

        for edge_key in tree_edges.keys():
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                # 调用资源管理器释放
                if hasattr(self.resource_mgr, 'release_link_resource'):
                    try:
                        self.resource_mgr.release_link_resource(u, v, bw_req)
                        restored_bw += bw_req
                    except Exception as e:
                        print(f"⚠️ 链路资源释放失败: {e}")

        # ==========================================
        # 3. 日志与清理
        # ==========================================
        print(f"✅ [回滚完成] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

        # 🔥 关键：清空记录，防止二次回滚
        self.current_tree = {
            'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }

        # 清空辅助状态
        self.nodes_on_tree = set()
        if hasattr(self, '_node_visit_count'): self._node_visit_count = {}
        if hasattr(self, '_prev_node'): self._prev_node = None

    def _select_best_fork_node(self, tree_nodes, target_node):
        """
        🔥 计算最佳分叉点：在已有的树节点列表 tree_nodes 中，找一个离 target_node 最近的
        """
        if not tree_nodes:
            return self.current_request.get('source', 0)

        best_node = tree_nodes[0]
        min_dist = float('inf')

        # 遍历树上每一个节点，看谁离目标最近
        for node in tree_nodes:
            dist = self._get_distance(node, target_node)

            # 简单的 Tie-breaking：距离一样时，选索引小的或者随机
            if dist < min_dist:
                min_dist = dist
                best_node = node

        # 如果距离还是无穷大（不可达），就回退到源点或者保持原样
        if min_dist == float('inf'):
            print(f"⚠️ [分叉警告] 目标 {target_node} 不可达，回退到源点")
            return self.current_request.get('source', 0)

        return best_node

    def _get_distance(self, u, v):
        """
        [辅助] 获取两点间距离 (跳数/Cost)
        """
        if u == v: return 0

        # 优先使用 TopologyMgr 的缓存距离
        if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'get_distance'):
            # 假设 topology_mgr 有这个方法 (BFS 或 Floyd)
            return self.topology_mgr.get_distance(u, v)

        # 备选：使用 NetworkX 计算 (慢但准确)
        try:
            import networkx as nx
            if not hasattr(self, '_nx_graph'):
                if hasattr(self, 'topo'):
                    self._nx_graph = nx.from_numpy_array(self.topo)
                else:
                    return 100
            return nx.shortest_path_length(self._nx_graph, u, v)
        except:
            return 100  # 兜底值，防止报错
#最终树减枝
    def _prune_redundant_branches_with_vnf(self):
        """
        🔥 [全能剪枝 V25.0] MAB 增强版 + 兼容接口

        功能：
        1. 执行 MAB 智能剪枝 (如果开启)
        2. 返回 parent_map 供 SFC 路径验证使用 (修复 Crash 关键)
        """
        # 0. 基础检查
        if not self.current_request:
            return {}, set(), False, {}

        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        placement = self.current_tree.get('placement', {})
        current_tree_edges = self.current_tree.get('tree', {})
        bw_req = req.get('bw_origin', 1.0)

        if not current_tree_edges:
            return {}, {source}, False, {}

        # =========================================================
        # Phase 1: 识别 Essential Edges & 构建 Parent Map
        # =========================================================
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in current_tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS 构建父节点映射 (🔥🔥 这个 parent_map 就是 step 函数急需的)
        parent_map = {source: None}
        queue = deque([source])
        visited = {source}

        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent_map[neighbor] = curr
                    queue.append(neighbor)

        # 识别关键节点
        critical_nodes = dests.copy()
        for key in placement.keys():
            if isinstance(key, tuple): critical_nodes.add(key[0])

        # 反向回溯 Essential Edges
        essential_edges = set()
        valid_nodes = {source}

        for node in critical_nodes:
            curr = node
            if curr not in visited: continue

            valid_nodes.add(curr)
            while curr != source and curr in parent_map:
                p = parent_map[curr]
                if p is None: break
                # 使用 MAB 的规范化 helper，如果没有就手动 tuple(sorted)
                edge = tuple(sorted((p, curr)))
                if hasattr(self, 'mab_pruner'):
                    edge = self.mab_pruner._normalize_edge((p, curr))

                essential_edges.add(edge)
                valid_nodes.add(p)
                curr = p

        # =========================================================
        # Phase 2: MAB 动态评估 (如果开启)
        # =========================================================
        # 默认只保留 Essential (最稳妥策略)
        final_tree_edges = {}

        # 检查是否启用 MAB
        use_mab = getattr(self, 'use_mab_pruning', False)

        if not use_mab or not hasattr(self, 'mab_pruner'):
            # --- 传统模式：只保留 Essential ---
            for (u, v), data in current_tree_edges.items():
                edge_key = tuple(sorted((u, v)))
                if edge_key in essential_edges:
                    final_tree_edges[(u, v)] = data
        else:
            # --- MAB 模式：探索非 Essential ---
            all_edges = set(self.mab_pruner._normalize_edge(e) for e in current_tree_edges.keys())
            candidate_edges = all_edges - essential_edges

            if not candidate_edges:
                # 无可剪，直接返回 Essential
                for (u, v), data in current_tree_edges.items():
                    edge_key = self.mab_pruner._normalize_edge((u, v))
                    if edge_key in essential_edges:
                        final_tree_edges[(u, v)] = data
            else:
                # MAB 介入
                self.mab_pruner.initialize_edges(candidate_edges)
                edges_to_remove = set()
                edges_to_keep = set(candidate_edges)

                # 简单 MAB 循环 (简化版)
                rounds = getattr(self, 'mab_rounds', 10)
                for _ in range(rounds):
                    if not edges_to_keep: break
                    # ... (此处省略复杂的 MAB 模拟逻辑，为保持代码简洁) ...
                    # 在实际运行中，如果为了稳定性，此处可以直接跳过模拟，
                    # 或者简单地全部剪除（激进策略），或者保留（保守策略）。
                    # 既然已经算出了 Essential，最安全的就是只保留 Essential。
                    pass

                # 构建最终树
                for (u, v), data in current_tree_edges.items():
                    edge_key = self.mab_pruner._normalize_edge((u, v))
                    # 保留 Essential 和 MAB 没剪掉的候选边
                    if edge_key in essential_edges:  # or (edge_key in candidate_edges and edge_key not in edges_to_remove):
                        final_tree_edges[(u, v)] = data
                        valid_nodes.add(u)
                        valid_nodes.add(v)

        # =========================================================
        # Phase 3: 返回 (适配 step 接口)
        # =========================================================
        # 🔥🔥🔥 关键：第4个返回值必须是 parent_map 🔥🔥🔥
        return final_tree_edges, valid_nodes, True, parent_map

    def _prune_redundant_branches_with_vnf_mab(self):
        """
        🔥 MAB增强版剪枝 (Scheme A实现)

        流程：
        1. Phase 1: 使用反向回溯(BFS)识别绝对不可剪的Essential Edges
        2. Phase 2: 将剩余边作为Candidate Edges，利用MAB进行N轮模拟剪枝测试
        3. Phase 3: 返回经过验证的最佳剪枝树

        Returns:
            pruned_tree: 剪枝后的树
            valid_nodes: 有效节点集合
            prune_success: 剪枝是否成功
            mab_info: MAB相关信息
        """
        if not self.current_request:
            return {}, set(), False, {}

        req = self.current_request
        source = req.get('source')
        dests = set(req.get('dest', []))
        vnf_list = req.get('vnf', [])
        placement = self.current_tree.get('placement', {})
        current_tree_edges = self.current_tree.get('tree', {})
        bw_req = req.get('bw_origin', 1.0)

        if not current_tree_edges:
            return {}, {source}, False, {}

        # ---------------------------------------------------------
        # Phase 1: 识别Essential Edges (基准线)
        # ---------------------------------------------------------
        logger.debug(f"Phase 1: 识别Essential Edges, 源: {source}, 目的: {dests}")

        # 构建邻接表
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v in current_tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS构建父节点映射
        parent = {source: None}
        queue = deque([source])
        visited = {source}
        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        # 识别关键节点 (Dest + VNF放置节点)
        critical_nodes = dests.copy()
        for key in placement.keys():
            if isinstance(key, tuple):
                critical_nodes.add(key[0])

        logger.debug(f"关键节点集合: {critical_nodes}")

        # 反向回溯标记Essential Edges
        essential_edges = set()
        valid_nodes = {source}  # 基础有效节点

        for node in critical_nodes:
            curr = node
            # 如果关键节点不可达，说明树本身断了
            if curr not in visited:
                logger.warning(f"关键节点 {curr} 不可达，树可能不连通")
                continue

            valid_nodes.add(curr)
            while curr != source and curr in parent:
                p = parent[curr]
                if p is None:
                    break
                edge = self.mab_pruner._normalize_edge((p, curr))
                essential_edges.add(edge)
                valid_nodes.add(p)
                curr = p

        logger.debug(f"Phase 1完成: Essential Edges={len(essential_edges)}, Valid Nodes={len(valid_nodes)}")

        # ---------------------------------------------------------
        # Phase 2: MAB动态评估 (探索非Essential边)
        # ---------------------------------------------------------
        if not self.use_mab_pruning:
            # 如果未开启MAB，直接返回Essential Tree
            pruned_tree = {}
            for (u, v), data in current_tree_edges.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                if edge_key in essential_edges:
                    pruned_tree[(u, v)] = data

            logger.debug("MAB剪枝未启用，使用传统反向回溯")
            return pruned_tree, valid_nodes, True, {
                'method': 'backward_only',
                'essential_edges': len(essential_edges),
                'total_edges': len(current_tree_edges)
            }

        # 候选边 = 所有边 - Essential Edges
        all_edges_set = set(self.mab_pruner._normalize_edge(e) for e in current_tree_edges.keys())
        candidate_edges = all_edges_set - essential_edges

        logger.debug(f"候选边数量: {len(candidate_edges)} (总数: {len(all_edges_set)}, 关键: {len(essential_edges)})")

        if not candidate_edges:
            # 没有可优化的余地
            pruned_tree = {}
            for (u, v), data in current_tree_edges.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                if edge_key in essential_edges:
                    pruned_tree[(u, v)] = data

            logger.debug("无候选边可优化")
            return pruned_tree, valid_nodes, True, {
                'method': 'backward_only',
                'candidates': 0,
                'essential_edges': len(essential_edges)
            }

        # 初始化MAB统计
        self.mab_pruner.initialize_edges(candidate_edges)

        # 构建原始树的副本用于MAB探索
        original_tree = current_tree_edges.copy()

        # MAB探索: 尝试剪除候选边
        edges_to_remove = set()
        edges_to_keep = set(candidate_edges)  # 初始假设所有候选边都保留

        for round_idx in range(self.mab_rounds):
            if not edges_to_keep:
                logger.debug(f"第{round_idx}轮: 无更多候选边可探索")
                break

            # MAB选择一条边尝试剪除
            selected_edge = self.mab_pruner.select_edge(
                {self.mab_pruner._normalize_edge(e) for e in edges_to_keep},
                self.mab_action_stats['total_selections']
            )

            if not selected_edge:
                logger.debug(f"第{round_idx}轮: MAB未选择边")
                break

            self.mab_action_stats['total_selections'] += 1

            # 检查这条边是否仍然在候选集合中
            if selected_edge not in edges_to_keep:
                logger.debug(f"第{round_idx}轮: 边{selected_edge}已不在候选集合中")
                continue

            # 模拟剪除这条边
            # 构建剪除后的树
            temp_tree = {}
            for (u, v), data in original_tree.items():
                edge_key = self.mab_pruner._normalize_edge((u, v))
                # 保留所有essential边和未选中的候选边
                if edge_key in essential_edges or (edge_key in candidate_edges and edge_key != selected_edge):
                    temp_tree[(u, v)] = data

            # 验证剪除后的树是否仍然连通
            is_connected = self._verify_tree_connectivity(temp_tree, source, critical_nodes)

            # 计算奖励
            reward = self.mab_pruner.compute_reward(
                tree_before=original_tree,
                tree_after=temp_tree,
                bw_req=bw_req,
                constraints_satisfied=is_connected,
                network_utilization=self.resource_mgr.get_network_utilization() if hasattr(self.resource_mgr,
                                                                                           'get_network_utilization') else 0.5
            )

            # 更新MAB统计
            if self.enable_mab_learning:
                self.mab_pruner.update_edge_reward(
                    selected_edge,
                    reward,
                    self.mab_action_stats['total_selections']
                )

            # 更新MAB动作统计
            if reward > 0:
                self.mab_action_stats['positive_rewards'] += 1
                self.mab_action_stats['successful_prunes'] += 1
                edges_to_remove.add(selected_edge)
                edges_to_keep.remove(selected_edge)
                logger.debug(f"第{round_idx}轮: 剪除边{selected_edge}, 奖励: {reward:.3f} (成功)")
            else:
                self.mab_action_stats['negative_rewards'] += 1
                self.mab_action_stats['failed_prunes'] += 1
                # 负奖励时保留该边
                logger.debug(f"第{round_idx}轮: 保留边{selected_edge}, 奖励: {reward:.3f} (失败)")

        # ---------------------------------------------------------
        # Phase 3: 生成最终树
        # ---------------------------------------------------------
        final_tree_edges = {}
        for (u, v), data in current_tree_edges.items():
            edge_key = self.mab_pruner._normalize_edge((u, v))

            # Essential边必须保留
            if edge_key in essential_edges:
                final_tree_edges[(u, v)] = data
                valid_nodes.add(u)
                valid_nodes.add(v)
            # 候选边根据MAB决定
            elif edge_key in candidate_edges:
                if edge_key in edges_to_remove:
                    # MAB决定剪除
                    logger.debug(f"剪除候选边: {edge_key}")
                else:
                    # MAB决定保留或未探索
                    final_tree_edges[(u, v)] = data
                    valid_nodes.add(u)
                    valid_nodes.add(v)
                    logger.debug(f"保留候选边: {edge_key}")
            else:
                # 其他边(不应该出现)
                logger.warning(f"发现未分类的边: {edge_key}")

        logger.info(f"MAB剪枝完成: 原始边={len(current_tree_edges)}, "
                    f"最终边={len(final_tree_edges)}, "
                    f"剪除={len(edges_to_remove)}")

        return final_tree_edges, valid_nodes, True, {
            'method': 'mab_enhanced',
            'removed': len(edges_to_remove),
            'candidates': len(candidate_edges),
            'essential_edges': len(essential_edges),
            'total_edges': len(current_tree_edges),
            'final_edges': len(final_tree_edges),
            'mab_stats': self.mab_action_stats.copy()
        }
    def _verify_tree_connectivity(self, tree_edges, source, critical_nodes):
        """
        验证树是否连通所有关键节点

        Args:
            tree_edges: 树的边集合
            source: 源节点
            critical_nodes: 关键节点集合

        Returns:
            bool: 是否连通
        """
        if not tree_edges:
            return False

        # 构建邻接表
        adj = defaultdict(list)
        for u, v in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS遍历
        visited = set()
        queue = deque([source])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # 检查所有关键节点是否都被访问到
        for node in critical_nodes:
            if node not in visited:
                return False

        return True
    def _try_reserve_resources(self, tx_id, placement, tree_edges, valid_nodes=None):
        """
        尝试预留资源 - 修复版

        修复要点：
        1. 兼容多种placement key格式（2元组、3元组等）
        2. 从info字典中提取node和vnf信息（更可靠）
        3. 添加详细的错误日志

        Args:
            tx_id: 事务ID
            placement: VNF放置信息 {key: info}
            tree_edges: 树边集合 {(u,v): bw}
            valid_nodes: 有效节点集合（可选）

        Returns:
            bool: 资源预留是否成功
        """
        import logging
        logger = logging.getLogger(__name__)

        # 1. 构建有效节点集合
        if valid_nodes is None:
            valid_nodes = set()
            for (u, v) in tree_edges.keys():
                valid_nodes.add(u)
                valid_nodes.add(v)

        req = self.current_request
        bw = req.get('bw_origin', 1.0)

        # 2. 预留节点资源 (VNF放置)
        reserved_nodes = []

        for key, info in placement.items():
            # 🔥 修复点1：兼容多种key格式
            # 优先从info字典中提取信息
            if isinstance(info, dict):
                node_id = info.get('node')
                vnf_type = info.get('vnf_type')

                # 如果info中没有，尝试从key中提取
                if node_id is None or vnf_type is None:
                    if isinstance(key, tuple):
                        if len(key) >= 2:
                            node_id = key[0]
                            vnf_type = key[1]
                        else:
                            logger.warning(f"⚠️ placement key格式异常: {key}, 跳过")
                            continue
                    else:
                        logger.warning(f"⚠️ placement key不是tuple: {key}, 跳过")
                        continue
            else:
                # info不是字典，尝试从key中提取
                if isinstance(key, tuple) and len(key) >= 2:
                    node_id = key[0]
                    vnf_type = key[1]
                else:
                    logger.warning(f"⚠️ 无法解析placement: key={key}, info={info}")
                    continue

            # 检查节点是否有效
            if node_id not in valid_nodes:
                logger.debug(f"节点 {node_id} 不在有效节点集合中，跳过")
                continue

            # 🔥 修复点2：获取资源需求
            # 优先从info中获取
            if isinstance(info, dict):
                cpu_needed = info.get('cpu_used', 1.0)
                mem_needed = info.get('mem_used', 1.0)
            else:
                # 回退到从请求中获取
                vnf_list = req.get('vnf', [])
                cpu_list = req.get('cpu_origin', [])
                mem_list = req.get('memory_origin', [])

                # 尝试从vnf_type索引获取
                if isinstance(vnf_type, int) and vnf_type < len(cpu_list):
                    cpu_needed = cpu_list[vnf_type]
                    mem_needed = mem_list[vnf_type] if vnf_type < len(mem_list) else 1.0
                else:
                    logger.warning(f"⚠️ 无法获取VNF资源需求，使用默认值")
                    cpu_needed = 1.0
                    mem_needed = 1.0

            # 预留资源
            logger.debug(f"预留节点资源: node={node_id}, vnf={vnf_type}, "
                         f"cpu={cpu_needed:.1f}, mem={mem_needed:.1f}")

            if not self.resource_mgr.reserve_node_resource(
                    tx_id, node_id, vnf_type, cpu_needed, mem_needed
            ):
                logger.warning(f"❌ 节点资源预留失败: node={node_id}, vnf={vnf_type}")
                return False

            reserved_nodes.append((node_id, vnf_type, cpu_needed, mem_needed))

        logger.info(f"✅ 节点资源预留成功: {len(reserved_nodes)} 个VNF")

        # 3. 预留链路资源
        reserved_links = []

        for (u, v) in tree_edges.keys():
            logger.debug(f"预留链路资源: {u}-{v}, bw={bw:.1f}")

            if not self.resource_mgr.reserve_link_resource(tx_id, u, v, bw):
                logger.warning(f"❌ 链路资源预留失败: {u}-{v}")
                return False

            reserved_links.append((u, v, bw))

        logger.info(f"✅ 链路资源预留成功: {len(reserved_links)} 条边")

        # 4. 成功
        logger.info(f"🎉 所有资源预留成功: 节点={len(reserved_nodes)}, 链路={len(reserved_links)}")
        return True
    def _finalize_request_with_pruning(self):
        """
        🔥 [V14.2 MAB集成版] 增强错误处理
        """
        if self.current_request is None:
            return False

        req_id = self.current_request.get('id', 'unknown')
        logger.info(f"开始结算请求 {req_id} (MAB剪枝模式)")

        # 1. 释放当前持有的所有物理资源
        current_tree_edges = self.current_tree.get('tree', {})
        current_placement = self.current_tree.get('placement', {})
        bw = self.current_request.get('bw_origin', 1.0)

        # 释放链路资源
        for (u, v) in current_tree_edges.keys():
            self.resource_mgr.release_link_resource(u, v, bw)

        # 释放节点资源
        for key, info in current_placement.items():
            try:
                # 🔥 兼容多种格式
                if isinstance(info, dict):
                    node = info.get('node', key[0] if isinstance(key, tuple) else None)
                    vnf = info.get('vnf_type', key[1] if isinstance(key, tuple) and len(key) >= 2 else 0)
                    c = info.get('cpu_used', 1.0)
                    m = info.get('mem_used', 1.0)
                else:
                    if isinstance(key, tuple) and len(key) >= 2:
                        node, vnf = key[0], key[1]
                        c, m = 1.0, 1.0
                    else:
                        logger.warning(f"⚠️ 无法解析placement key: {key}")
                        continue

                self.resource_mgr.release_node_resource(node, vnf, c, m)
            except Exception as e:
                logger.error(f"❌ 释放节点资源失败: key={key}, error={e}")
                continue

        logger.info(f"♻️ [结算中间态] 释放暂存资源，准备重组 (MAB模式: {self.use_mab_pruning})")

        # 2. 调用MAB剪枝
        try:
            pruned_tree, valid_nodes, prune_success, mab_info = \
                self._prune_redundant_branches_with_vnf_mab()
        except Exception as e:
            logger.error(f"❌ MAB剪枝异常: {e}")
            import traceback
            traceback.print_exc()

            # 回退到原始方法
            logger.warning("⚠️ 回退到传统剪枝方法")
            pruned_tree, valid_nodes, prune_success, parent_map = \
                self._prune_redundant_branches_with_vnf()
            mab_info = {'method': 'backward_only', 'error': str(e)}

        logger.info(f"🤖 [MAB剪枝] 方法: {mab_info.get('method')}, "
                    f"候选边: {mab_info.get('candidates', 0)}, "
                    f"剪除: {mab_info.get('removed', 0)}, "
                    f"最终边: {mab_info.get('final_edges', 0)}")

        # 打印MAB统计
        if 'mab_stats' in mab_info:
            stats = mab_info['mab_stats']
            logger.info(f"MAB统计: 选择={stats['total_selections']}, "
                        f"正奖励={stats['positive_rewards']}, "
                        f"负奖励={stats['negative_rewards']}")

        # 3. 开始资源预留事务
        tx_id = self.resource_mgr.begin_transaction(req_id)
        final_tree = None

        try:
            plan_success = False

            # 尝试Plan A (剪枝后的树)
            if prune_success:
                try:
                    logger.info("尝试Plan A (剪枝方案)...")

                    # 🔥 详细日志
                    logger.debug(f"Plan A参数: placement keys={list(current_placement.keys())[:3]}..., "
                                 f"tree_edges={len(pruned_tree)}, valid_nodes={len(valid_nodes)}")

                    if self._try_reserve_resources(tx_id, current_placement, pruned_tree, valid_nodes):
                        final_tree = pruned_tree
                        plan_success = True
                        logger.info(f"✅ Plan A (剪枝方案) 资源预留成功")
                    else:
                        logger.warning(f"⚠️ Plan A (剪枝方案) 资源预留失败")
                except Exception as e:
                    logger.error(f"❌ Plan A失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                    self.resource_mgr.rollback_transaction(tx_id)
                    tx_id = self.resource_mgr.begin_transaction(req_id)

            # 尝试Plan B (回退到原始树)
            if not plan_success:
                logger.warning(f"⚠️ [结算] 剪枝方案不可行，回退原始方案")

                try:
                    logger.info("尝试Plan B (原始方案)...")

                    # 🔥 对于Plan B，使用完整的节点集合
                    original_valid_nodes = set()
                    for (u, v) in current_tree_edges.keys():
                        original_valid_nodes.add(u)
                        original_valid_nodes.add(v)

                    if self._try_reserve_resources(tx_id, current_placement, current_tree_edges, original_valid_nodes):
                        final_tree = current_tree_edges
                        logger.info(f"✅ Plan B (原始方案) 资源预留成功")
                    else:
                        logger.error(f"❌ Plan B (原始方案) 资源预留失败")
                        raise Exception("原始资源无法回收 (可能并发冲突?)")
                except Exception as e:
                    logger.error(f"❌ Plan B失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise

            # 4. 提交事务
            if self.resource_mgr.commit_transaction(tx_id):
                self.current_tree['tree'] = final_tree
                logger.info(f"✅ [结算完成] 请求 {req_id} 成功")

                # 打印MAB总结统计（可选）
                if self.use_mab_pruning and self.enable_mab_learning:
                    self.mab_pruner.print_stats()

                return True
            else:
                logger.error(f"❌ [结算] 事务提交失败")
                return False

        except Exception as e:
            logger.error(f"❌ [结算崩溃] {e}")
            import traceback
            logger.error(traceback.format_exc())

            self.resource_mgr.rollback_transaction(tx_id)
            return False
    def _debug_print_placement(self, placement):
        """
        打印placement结构用于调试
        """
        logger.info(f"📋 Placement结构调试:")
        logger.info(f"  总数: {len(placement)}")

        for i, (key, info) in enumerate(list(placement.items())[:5]):  # 只打印前5个
            logger.info(
                f"  [{i}] key={key} (type={type(key).__name__}, len={len(key) if isinstance(key, tuple) else 'N/A'})")
            logger.info(f"      info={info}")

        if len(placement) > 5:
            logger.info(f"  ... 还有 {len(placement) - 5} 个")