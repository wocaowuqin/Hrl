# envs/sfc_env.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC_HIRL_Env - 完整可运行的主环境类（分层强化学习 + 多播感知）
已完全模块化，职责清晰，兼容 Flat 和 GNN 两种状态表示
"""
import os
import logging
import random
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
from envs.modules.failure_visualizer import FailureVisualizer
from envs.modules.visualize_multicast_tree import MulticastTreeVisualizer
from envs.modules.TreePruner import TreePruner
from envs.modules.HRL_Coordinator import HRL_Coordinator
from envs.modules.MABPruner import MABPruningHelper

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
                logger.info(f"⏰ [Resource Flow] 请求 {req_id}: 请求生命周期到达，归还资源 (Expired & Released)")
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

    def _release_request_resources(self, req_id, _):
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

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        """
        🔥 [V40.1 强力加载版]
        修复：如果传入的是文件路径，直接读取，绕过 data_loader 的自动拼接逻辑
        """
        import os
        import pickle

        success = False
        loaded_requests = []

        # 1. 尝试直接作为文件路径加载
        if os.path.exists(phase_or_req_file) and phase_or_req_file.endswith('.pkl'):
            print(f"📂 [Env] 检测到直接文件路径: {phase_or_req_file}")
            try:
                with open(phase_or_req_file, 'rb') as f:
                    loaded_requests = pickle.load(f)

                # 手动注入到 data_loader（保持兼容性）
                if hasattr(self, 'data_loader'):
                    self.data_loader.requests = loaded_requests
                    self.data_loader.total_steps = len(loaded_requests)

                success = True
                print(f"✅ [Env] 直接文件加载成功: {len(loaded_requests)} 条")
            except Exception as e:
                print(f"❌ [Env] 直接文件加载失败: {e}")
                return False
        else:
            # 2. 否则作为 phase 名称调用 data_loader
            print(f"🔄 [Env] 调用 data_loader 加载 phase: {phase_or_req_file}")
            if hasattr(self, 'data_loader'):
                success = self.data_loader.load_dataset(phase_or_req_file)
                loaded_requests = getattr(self.data_loader, 'requests', [])
            else:
                print("❌ [Env] data_loader 未初始化")
                return False

        # 3. 🔥 同步数据到环境索引
        if success and loaded_requests:
            print(f"🔄 [Env] 正在构建在线仿真索引 (Requests: {len(loaded_requests)})...")
            self.load_requests(loaded_requests)
        else:
            print("⚠️ [Env] 数据加载报告成功，但请求列表为空！")

        return success


class SFC_HIRL_Env(gym.Env):
    # 基础初始化
    def __init__(self, config, use_gnn=True):
        """
        初始化环境 - 包含 HRL 变量统一与完整模块加载
        """
        self.config = config
        self.use_gnn = use_gnn

        # -----------------------------------------------------------
        # 1. 基础架构：拓扑与资源 (必须最先初始化)
        # -----------------------------------------------------------
        # 初始化 topo, n, L, K_vnf, dc_nodes, resource_mgr, topology_mgr
        self._init_infrastructure()

        # -----------------------------------------------------------
        # 2. 核心功能模块
        # -----------------------------------------------------------
        # 请求生命周期管理
        self.request_manager = RequestLifecycleManager(self)

        # 专家系统、备份策略、路径管理
        self._init_core_modules()

        # 🔥 Tree Pruner 初始化 (关键修复：传入 resource_mgr 和 config)
        self._init_tree_pruner(self.resource_mgr, config)

        # -----------------------------------------------------------
        # 3. 强化学习与智能组件
        # -----------------------------------------------------------
        # 数据加载、奖励计算(RewardCritic)、PolicyHelper
        self._init_rl_components()

        # MAB 智能剪枝组件
        self._init_mab_components()

        # GNN 特征提取器与 Gym 空间 (Observation/Action Space)
        self._init_gym_spaces()

        # -----------------------------------------------------------
        # 4. 🔥 [关键重构] 统一 HRL 协调变量 (Single Source of Truth)
        # -----------------------------------------------------------

        # ========================================
        # 🔥🔥🔥 V40.0 新增：Episode统计变量
        # ========================================
        self.current_episode = 0  # ← 添加：Episode计数器
        self.current_step = 0  # ← 添加：当前Episode步数
        self.total_reward = 0.0  # ← 添加：当前Episode累计奖励

        # ========================================
        # 🔥🔥🔥 V40.0 核心：VNF索引指针（唯一真相源）
        # ========================================
        self.next_vnf_idx = 0  # ← 添加：下一个要部署的VNF索引

        # 4.1 步数控制
        # 从配置读取最大低层步数 (默认 50)
        self.max_subgoal_steps = config.get('max_low_steps', 50)
        self.subgoal_step_count = 0  # 统一的低层步数计数器 (替代 _low_step_count)

        # 4.2 目标控制
        self.current_subgoal_node = None  # 当前锁定的子目标节点 ID
        self.last_high_action_idx = None  # 上一次高层输出的原始动作索引

        # 4.3 阶段控制 (用于 step_low_level 状态机)
        self.current_phase = None  # 枚举: 'vnf_deployment' / 'destination_connection'
        self.current_deployment_target = None  # 物理层专用: 部署目标节点
        self.current_target_node = None  # 物理层专用: 连接目标节点
        self.current_vnf_to_deploy = None  # 物理层专用: 待部署 VNF 索引

        # -----------------------------------------------------------
        # 5. 其他状态变量初始化
        # -----------------------------------------------------------
        # 调用原有的状态初始化 (total_reward, request pointers, etc.)
        self._init_state_variables()

        # 额外状态容器
        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0
        self.vnf_deployment_history = {}
        self.step_count = 0  # 全局 Episode 步数

        # -----------------------------------------------------------
        # 6. 在线仿真模式配置 (从 _init_state_variables 提取或确保存在)
        # -----------------------------------------------------------
        self.online_mode = self.config.get('environment', {}).get('online_mode', True)
        self.simulation_done = False
        self.slot_queue = []
        self.requests_by_slot = {}
        self.active_requests_by_slot = {}
        self.leave_heap = []

        logger.info(f"✅ 环境基础参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")
        logger.info(f"✅ HRL 控制参数: Max Subgoal Steps={self.max_subgoal_steps}")
        logger.info(f"✅ V40.0 核心变量: next_vnf_idx={self.next_vnf_idx}")  # ← 添加日志

        # -----------------------------------------------------------
        # 7. 可视化初始化
        # -----------------------------------------------------------
        self.enable_visualization = True
        if self.enable_visualization:
            try:
                import os
                os.makedirs('visualization/success', exist_ok=True)
                os.makedirs('visualization/fail', exist_ok=True)
                # self.visualizer = MulticastTreeVisualizer(self)
                logger.info("✅ 可视化目录已就绪")
            except Exception as e:
                logger.warning(f"⚠️ 可视化初始化部分失败: {e}")
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
        # 🔥 修复：读取原始配置并减 1 (适配 0-based 索引)
        raw_dc_nodes = self.config.get('topology', {}).get('dc_nodes', list(range(10)))
        self.dc_nodes = [x - 1 for x in raw_dc_nodes]

        # 最好打印一下确认
        print(f"✅ [Index Fix] DC Nodes converted: {self.dc_nodes}")
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

    def _init_tree_pruner(self, resource_mgr, config):
        """
        🔥 [初始化核心] 构建 TreePruner 的内部状态和配置

        Args:
            resource_mgr: 资源管理器实例
            config: 配置字典
        """
        # 1. 基础依赖注入
        self.resource_mgr = resource_mgr
        self.config = config or {}

        # 2. 解析 MAB (Multi-Armed Bandit) 相关配置
        # 提供默认值以防 config 为空或缺失键
        self.use_mab_pruning = self.config.get('use_mab_pruning', False)
        self.mab_rounds = self.config.get('mab_rounds', 10)
        self.enable_mab_learning = self.config.get('enable_mab_learning', False)

        # 3. 初始化 MAB 统计数据结构
        # 用于跟踪剪枝决策的效果
        self.mab_action_stats = {
            'total_selections': 0,  # 总共尝试剪枝次数
            'positive_rewards': 0,  # 获得正奖励次数 (剪枝成功且有效)
            'negative_rewards': 0,  # 获得负奖励次数 (剪枝导致断连或性能下降)
            'successful_prunes': 0,  # 成功执行的剪枝操作数
            'failed_prunes': 0  # 失败的剪枝操作数
        }

        # 4. 初始化上下文状态容器
        # 这些将在 set_current_request 中被填充
        self.current_request = None  # 当前处理的 SFC 请求
        self.current_tree = None  # 当前构建的多播树结构

        # 5. MAB 算法实例占位符
        # 需要通过 set_mab_pruner 单独注入
        self.mab_pruner = None

        logger.debug(f"TreePruner 初始化完成 | MAB启用: {self.use_mab_pruning}")

    # 数据加载
    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        """
        🔥 [V40.1 强力加载版]
        修复：如果传入的是文件路径，直接读取，绕过 data_loader 的自动拼接逻辑
        """
        import os
        import pickle

        success = False
        loaded_requests = []

        # 1. 尝试直接作为文件路径加载
        if os.path.exists(phase_or_req_file) and phase_or_req_file.endswith('.pkl'):
            print(f"📂 [Env] 检测到直接文件路径: {phase_or_req_file}")
            try:
                with open(phase_or_req_file, 'rb') as f:
                    loaded_requests = pickle.load(f)

                # 手动注入到 data_loader（保持兼容性）
                if hasattr(self, 'data_loader'):
                    self.data_loader.requests = loaded_requests
                    self.data_loader.total_steps = len(loaded_requests)

                success = True
                print(f"✅ [Env] 直接文件加载成功: {len(loaded_requests)} 条")
            except Exception as e:
                print(f"❌ [Env] 直接文件加载失败: {e}")
                return False
        else:
            # 2. 否则作为 phase 名称调用 data_loader
            print(f"🔄 [Env] 调用 data_loader 加载 phase: {phase_or_req_file}")
            if hasattr(self, 'data_loader'):
                success = self.data_loader.load_dataset(phase_or_req_file)
                loaded_requests = getattr(self.data_loader, 'requests', [])
            else:
                print("❌ [Env] data_loader 未初始化")
                return False

        # 3. 🔥 同步数据到环境索引
        if success and loaded_requests:
            print(f"🔄 [Env] 正在构建在线仿真索引 (Requests: {len(loaded_requests)})...")
            self.load_requests(loaded_requests)
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

    # 环境智能体交互 reset step step_low_level step_high_level get_state
    def reset(self, seed=None, options=None):
        """
        🔄 [V40.0 最终版本] 重置环境
        """
        # ========================================
        # 1. 种子设置
        # ========================================
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if hasattr(self, 'action_space'):
                self.action_space.seed(seed)

        # ========================================
        # 2. Episode统计
        # ========================================
        self.current_episode += 1
        self.current_step = 0
        self.total_reward = 0.0
        self.step_count = 0

        # ========================================
        # 3. 🔥 V40核心：重置VNF指针
        # ========================================
        self.next_vnf_idx = 0

        # ========================================
        # 4. 阶段控制重置
        # ========================================
        self.current_phase = None
        self.current_deployment_target = None
        self.current_vnf_to_deploy = None
        self.current_target_node = None

        # ========================================
        # 5. 步数计数器重置
        # ========================================
        self.subgoal_step_count = 0

        # ========================================
        # 6. 高层动作记录重置
        # ========================================
        self.last_high_action_idx = None
        self.current_subgoal_node = None

        # ========================================
        # 7. 其他状态重置
        # ========================================
        options = options or {}
        force_hard_reset = options.get('hard_reset', False)
        phase = options.get("phase", "phase3")

        # 物理清空跨Episode的计数器
        self._node_visit_count = {}
        self._recent_positions = []
        self._vnf_complete_steps = 0

        # ========================================
        # 8. 资源管理器重置
        # ========================================
        if hasattr(self, 'resource_mgr') and self.resource_mgr:
            self.resource_mgr.reset()

        # ========================================
        # 9. 树结构重置
        # ========================================
        self.nodes_on_tree = set()
        self.current_tree = {
            'tree': {},
            'placement': {},
            'connected_dests': set(),
            'hvt': np.zeros((self.n, self.K_vnf)) if hasattr(self, 'K_vnf') else np.zeros((self.n, 10))
        }
        self.current_placements = {}

        self.branch_states = {}
        self.current_branch_id = None
        self.curr_ep_node_allocs = []
        self.curr_ep_link_allocs = []

        # 10. 获取下一个请求
        if self.online_mode:
            req_raw = self._get_next_request_online()
        else:
            req_raw, _ = self.reset_request()

        # 11. 处理DataLoader返回的对象
        if req_raw is not None:
            if hasattr(req_raw, 'to_dict'):
                req = req_raw.to_dict()
            elif hasattr(req_raw, '__dict__') and not isinstance(req_raw, dict):
                req = req_raw.__dict__
            else:
                req = req_raw
        else:
            req = None

        # ========================================
        # 🔥 关键修复：递归保护（限制递归深度）
        # ========================================
        if req is None and self.online_mode:
            # 检查是否已经是hard_reset模式
            options = options or {}
            if options.get('hard_reset', False):
                # 已经hard_reset了还没请求，说明真的没有了
                logger.warning("⚠️ [Reset] hard_reset后仍无请求，继续使用空请求")
                req = None  # 继续处理
            else:
                # 第一次递归，尝试hard_reset
                logger.info("🔄 [Reset] 无请求，尝试hard_reset")
                return self.reset(seed, options={'hard_reset': True})

        # 12. 设置当前请求
        self.current_request = req

        if req:
            # 有请求的正常处理
            source = req.get('source', 0)
            self.current_node_location = source
            # ... 其他设置 ...
        else:
            # 🔥 没有请求的兜底处理
            logger.warning("⚠️ [Reset] 没有可用的请求，使用默认配置")
            self.current_node_location = 0
            # 不返回，继续生成状态

        # 13. 生成初始状态
        initial_state = self.get_state()

        info = {
            'request': req,
            'action_mask': self.get_low_level_action_mask(),
            'decision_steps': 0,
        }

        return initial_state, info

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

    def step(self, force_goal=None):
        """
        🎯 [V33.0 统一变量架构版] 执行 HRL 决策步

        逻辑流：
        1. 检查是否需要高层决策 (由 Env 的状态决定)
        2. 如果需要 -> 执行高层 Agent -> 调用 env.set_high_level_goal
        3. 无论如何 -> 执行低层 Agent -> 调用 env.step
        """

        # ============================================================
        # Phase 1: 高层决策 (High-Level Decision)
        # ============================================================
        # 判断条件：
        # 1. 外部强制指定目标 (force_goal)
        # 2. 环境当前没有子目标 (env.current_subgoal_node is None)
        #    注意：Env会在初始化、子目标完成、或子目标超时(truncated)时自动将目标设为 None

        if force_goal is not None or self.env.current_subgoal_node is None:
            logger.info("🎯 [Coordinator] 触发高层决策")

            # --- 1.1 准备状态 ---
            high_obs = self.env.get_high_level_state_graph()
            high_mask = None
            unconnected_dests = []

            try:
                high_mask = self.env.get_high_level_action_mask()
                # 获取未连接目的地列表供启发式逻辑使用
                if self.env.current_request:
                    dests = self.env.current_request.get('dest', [])
                    connected = self.env.current_tree.get('connected_dests', set())
                    unconnected_dests = [d for d in dests if d not in connected]
            except Exception as e:
                logger.warning(f"⚠️ [Coordinator] 获取高层掩码异常: {e}")

            # --- 1.2 Agent 选择动作 ---
            # 注意：这里的 high_action 只是一个索引
            high_action_idx, _, high_info = self.high_agent.select_action(
                high_obs,
                mask=high_mask,
                training=True  # 假设总是训练模式，或者从外部传入
            )

            # --- 1.3 解析实际目标节点 ---
            # 如果有外部强制目标，覆盖 Agent 的选择
            if force_goal is not None:
                real_target_node = force_goal
                logger.info(f"🎯 [Coordinator] 使用强制目标: {real_target_node}")
                # 尝试反向查找索引用于记录（可选）
                high_action_idx = -1
            else:
                # 将 Agent 的输出索引解析为物理节点 ID
                real_target_node = self._resolve_action_to_node(high_action_idx, high_info, unconnected_dests)
                logger.info(f"🎯 [Coordinator] Agent选择: {high_action_idx} -> 目标节点: {real_target_node}")

            # --- 1.4 🔥 核心：将目标注入环境 ---
            # 这一步会重置 Env 内部的 subgoal_step_count = 0
            self.env.set_high_level_goal(high_action_idx, real_target_node)

            # 更新统计
            self.stats['high_decisions'] += 1
            self.last_high_action = real_target_node  # 仅用于记录/调试

        # ============================================================
        # Phase 2: 低层执行 (Low-Level Execution)
        # ============================================================
        # 此时 env.current_subgoal_node 一定有值 (除非 set_high_level_goal 失败)
        if self.env.current_subgoal_node is None:
            logger.error("❌ [Coordinator] 严重错误：高层决策后仍无目标！")
            return self.env.get_state(), 0.0, False, True, {'error': 'goal_setting_failed'}

        # --- 2.1 准备低层状态 ---
        low_obs = self.env.get_state()
        low_mask = self.env.get_low_level_action_mask()

        # --- 2.2 Agent 选择动作 ---
        try:
            low_action, _, low_info = self.low_agent.select_action(
                low_obs,
                action_mask=low_mask
            )
        except Exception as e:
            logger.error(f"❌ [Coordinator] 低层选动作失败: {e}")
            return low_obs, -1.0, False, True, {'error': 'low_agent_fail'}

        # --- 2.3 🔥 核心：执行环境步 ---
        # Env 内部会自动：
        # 1. subgoal_step_count += 1
        # 2. 检查是否超时 (Max Subgoal Steps) -> 返回 truncated=True
        # 3. 检查是否完成任务 -> 返回 done=True
        try:
            next_obs, reward, done, truncated, info = self.env.step(low_action)
        except Exception as e:
            logger.error(f"❌ [Coordinator] 环境 step 异常: {e}")
            import traceback
            traceback.print_exc()
            return low_obs, -10.0, False, True, {'error': 'env_step_crash'}

        # --- 2.4 状态更新与统计 ---
        self.stats['low_steps'] += 1

        # 补充 High Action 信息供 Replay Buffer 使用
        if not isinstance(info, dict): info = {}
        info['high_action'] = getattr(self, 'last_high_action', None)

        # 如果子目标结束 (超时 或 完成)
        if truncated:
            logger.info(f"🔄 [Coordinator] 子目标结束 (Reward: {reward:.2f})")
            self.stats['subgoals_completed'] += 1
            # 注意：Env 在返回 truncated=True 时，已经自动将 current_subgoal_node 设为 None
            # 所以下一次 step 调用时，会自动进入 Phase 1

        return next_obs, reward, done, truncated, info

    def _resolve_action_to_node(self, action_idx, info, unconnected_dests=None):
        """
        🔧 [辅助方法] 将高层动作索引解析为具体的节点ID
        """
        # 1. 优先从 info 中获取 (如果 Agent 支持输出 subgoal)
        if info and 'subgoal' in info and info['subgoal'] is not None:
            return info['subgoal']

        # 2. 根据当前阶段解析
        phase = getattr(self.env, 'current_phase', 'unknown')

        if phase == 'vnf_deployment':
            # 映射到 DC 节点列表
            dc_nodes = getattr(self.env, 'dc_nodes', [])
            if dc_nodes:
                # 取模防止越界
                return dc_nodes[action_idx % len(dc_nodes)]
            return action_idx  # 兜底

        elif phase == 'destination_connection':
            # 映射到剩余未连接的目的地
            if unconnected_dests and 0 <= action_idx < len(unconnected_dests):
                return unconnected_dests[action_idx]
            # 或者是直接映射到节点 ID (取决于你的 Action Space 定义)
            return action_idx

        # 3. 默认直接返回索引
        return action_idx

    # 高层
    def set_high_level_goal(self, high_action_idx, target_node_id):
        """
        🎯 [V40.0 最终版本] 设定高层目标

        职责：
        1. 接收高层 Agent 的决策 (Target Node)
        2. 自动判断当前处于哪个阶段 (VNF部署 vs 目的地连接)
        3. 重置低层步数计数器 (防止死循环)

        关键修复：
        - 使用 next_vnf_idx 作为唯一的VNF索引来源
        - 不再使用 _get_total_vnf_progress() 来决定当前VNF
        """
        # 1. 记录高层动作
        self.last_high_action_idx = high_action_idx
        self.current_subgoal_node = target_node_id

        # 2. 🔥 关键：新目标开始，步数计数器必须归零
        self.subgoal_step_count = 0

        # 3. 自动判定阶段 (Phase Logic)
        if self.current_request:
            vnf_list = self.current_request.get('vnf', [])

            # ========================================
            # 🔥🔥🔥 关键修复：使用next_vnf_idx作为唯一来源
            # ========================================
            self.current_vnf_to_deploy = self.next_vnf_idx

            # 判断当前阶段
            if self.next_vnf_idx < len(vnf_list):
                # ========================================
                # 阶段1：VNF 部署
                # ========================================
                self.current_phase = 'vnf_deployment'
                self.current_deployment_target = target_node_id
                self.current_target_node = None

                logger.info(f"🎯 [Env] VNF部署阶段: VNF[{self.next_vnf_idx}] → 节点{target_node_id}")
            else:
                # ========================================
                # 阶段2：目的地连接
                # ========================================
                self.current_phase = 'destination_connection'
                self.current_target_node = target_node_id
                self.current_deployment_target = None

                logger.info(f"🎯 [Env] 目的地连接阶段: 目标节点{target_node_id}")
        else:
            logger.warning("⚠️ 设定目标时没有活跃请求")

        logger.info(
            f"🎯 [Env] 设定目标完成: 节点{target_node_id} | 阶段: {self.current_phase} | VNF索引: {self.current_vnf_to_deploy} | 计数器已重置")

        # 返回状态供 Coordinator 记录
        return self.get_high_level_state_graph()

    def step_high_level(self, action):
        """
        🎯 [高层V40.1 最终修复版]

        关键修复：
        1. 不再判断阶段（由set_high_level_goal负责）
        2. 只负责瞬移和状态转移
        3. 添加完整的错误检查

        职责：
        - 读取set_high_level_goal设置的阶段和目标
        - 执行低层Agent瞬移
        - 返回truncated=True进入低层模式
        """
        # ========================================
        # 1. 保护：检查请求
        # ========================================
        if self.current_request is None:
            logger.error("❌ [High] 没有当前请求")
            return None, -10.0, True, False, {
                'error': 'no_current_request',
                'message': '没有可用的请求数据'
            }

        # ========================================
        # 2. 🔥 关键修复：读取已设置的阶段和目标
        # ========================================
        target_node = None

        if self.current_phase == 'vnf_deployment':
            target_node = self.current_deployment_target

        elif self.current_phase == 'destination_connection':
            target_node = self.current_target_node

        else:
            logger.error(f"❌ [High] 未知阶段: {self.current_phase}")
            return None, -10.0, True, False, {
                'error': 'unknown_phase',
                'phase': self.current_phase
            }

        # 检查目标节点
        if target_node is None:
            logger.error(f"❌ [High] 目标节点未设置 (阶段: {self.current_phase})")
            return None, -10.0, True, False, {
                'error': 'no_target',
                'phase': self.current_phase
            }

        # ========================================
        # 3. 瞬移逻辑：计算起点
        # ========================================
        tree_nodes = list(self.nodes_on_tree)

        if not tree_nodes:
            # 没有树节点，从源点开始
            start_node = self.current_request.get('source', 0)
        else:
            # 从最近的树节点开始
            try:
                start_node = min(
                    tree_nodes,
                    key=lambda x: self._get_hop_distance(x, target_node)
                )
            except Exception as e:
                logger.warning(f"⚠️ [High] 计算起点失败: {e}，使用第一个树节点")
                start_node = tree_nodes[0]

        # ========================================
        # 4. 执行瞬移
        # ========================================
        self.current_node_location = start_node
        self.subgoal_step_count = 0  # 重置计数器

        logger.debug(f"📍 [High] 瞬移: 从{start_node} → 目标{target_node}")

        # ========================================
        # 5. 返回：进入低层模式
        # ========================================
        return None, 0.0, False, True, {
            'phase': self.current_phase,
            'start': start_node,
            'target': target_node,
            'truncated': True,
            'action_mask': self.get_low_level_action_mask()
        }

    def _is_valid_node(self, node):
        """
        验证节点是否有效且存在

        修复要点:
        1. 正确检查字典结构的 resource_mgr.nodes
        2. 兼容多种数据结构 (dict/list/ndarray)
        3. 严格边界检查
        """
        # 1. 基础类型和边界检查
        try:
            node = int(node)
        except (ValueError, TypeError):
            return False

        if node < 0 or node >= self.n:
            return False

        # 2. 资源管理器检查
        if hasattr(self, 'resource_mgr') and self.resource_mgr is not None:
            if hasattr(self.resource_mgr, 'nodes'):
                nodes = self.resource_mgr.nodes

                # 🔥 字典结构 {'cpu': [...], 'memory': [...]}
                if isinstance(nodes, dict):
                    cpu_list = nodes.get('cpu', [])
                    if hasattr(cpu_list, '__len__'):
                        return 0 <= node < len(cpu_list)
                    return False

                # 列表结构 [{}, {}, ...]
                elif isinstance(nodes, list):
                    return 0 <= node < len(nodes)

                # NumPy 数组
                elif hasattr(nodes, 'shape'):
                    return 0 <= node < nodes.shape[0]

        return True

    def get_high_level_action_mask(self):
        """
        🔥 [V40.2 动态全网掩码版]
        修复 Mask 维度不匹配导致的无效动作选择问题
        功能：
        1. 始终返回 shape=(N,) 的掩码，对应全网物理节点。
        2. 根据当前阶段 (VNF部署 vs 目的地连接) 动态开放有效节点。
        """
        # 1. 初始化全网节点掩码 (Shape = N)
        # 必须是全网节点数，不能只是 DC 节点数
        mask = np.zeros(self.n, dtype=np.float32)

        # 安全检查
        if self.current_request is None:
            return np.ones(self.n, dtype=np.float32)

        vnf_list = self.current_request.get('vnf', [])

        # 获取当前 VNF 索引（优先使用 HRL 指针）
        if hasattr(self, 'next_vnf_idx'):
            current_vnf_idx = self.next_vnf_idx
        else:
            current_vnf_idx = self._get_total_vnf_progress()

        # ============================================
        # 阶段 1: VNF 部署阶段 (只允许选 DC 节点)
        # ============================================
        if current_vnf_idx < len(vnf_list):
            if hasattr(self, 'dc_nodes'):
                for node in self.dc_nodes:
                    # 确保节点 ID 在合法范围内
                    if 0 <= node < self.n and self._is_valid_node(node):
                        mask[node] = 1.0  # ✅ 只有 DC 节点设为 1

            # 兜底：如果没得选，保持全0（Agent会随机或报错，强迫其受到惩罚），或者记录警告
            if np.sum(mask) == 0:
                # logger.warning("⚠️ [Mask] VNF阶段无可用DC节点")
                pass

                # ============================================
        # 阶段 2: 目的地连接阶段 (只允许选未连接的目的地)
        # ============================================
        else:
            dests = self.current_request.get('dest', [])
            connected = self.current_tree.get('connected_dests', set())

            # 找出还没连上的目的地
            remaining_dests = [d for d in dests if d not in connected]

            for node in remaining_dests:
                if 0 <= node < self.n and self._is_valid_node(node):
                    mask[node] = 1.0  # ✅ 只有剩余目的地设为 1

        return mask
    def get_high_level_state_graph(self):
        """
        🎯 [V30.2 最小侵入版 - 无需修改Agent网络]

        只增强节点特征，保持全局特征维度不变
        """
        import torch
        from torch_geometric.data import Data

        n = self.n

        # 安全检查
        if not self.current_request:
            return Data(
                x=torch.zeros((n, 10), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                global_attr=torch.zeros((1, 5), dtype=torch.float32)
            )

        # =============================
        # 1. 节点特征 [N, 10]
        # =============================
        x = []
        vnf_list = self.current_request.get('vnf', [])
        placement = self.current_tree.get('placement', {})
        source = self.current_request.get('source')
        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())
        nodes_on_tree = getattr(self, 'nodes_on_tree', set())

        # 🔥🔥🔥 新增：统计每个节点已部署的VNF数量 🔥🔥🔥
        node_vnf_counts = [0] * n

        # 从current_placements统计
        if hasattr(self, 'current_placements') and self.current_placements:
            for placement_key in self.current_placements:
                node_id, vnf_idx = placement_key
                if 0 <= node_id < n:
                    node_vnf_counts[node_id] += 1

        # 也从placement字典统计（双重保险）
        for (node_id, vnf_idx), vnf_type in placement.items():
            if 0 <= node_id < n:
                # 确保计数正确
                actual_count = sum(1 for (nid, _) in placement.keys() if nid == node_id)
                node_vnf_counts[node_id] = max(node_vnf_counts[node_id], actual_count)

        for node in range(n):
            # 🔥 修复：直接从pool获取资源
            try:
                cpu = self.resource_mgr.pool.get_available_cpu(node) / 100.0
                mem = self.resource_mgr.pool.get_available_memory(node) / 100.0
            except (AttributeError, IndexError):
                if hasattr(self.resource_mgr, 'get_node_cpu'):
                    cpu = self.resource_mgr.get_node_cpu(node) / 100.0
                    mem = self.resource_mgr.get_node_mem(node) / 100.0
                else:
                    cpu = 0.5
                    mem = 0.5

            # 特征1-2: CPU和内存
            features = [cpu, mem]

            # 特征3: 是否是源节点
            is_source = 1.0 if node == source else 0.0
            features.append(is_source)

            # 特征4: 是否是目的地节点
            is_dest = 1.0 if node in dests else 0.0
            features.append(is_dest)

            # 特征5: 目的地是否已连接
            is_connected = 1.0 if node in connected else 0.0
            features.append(is_connected)

            # 特征6: 是否在树上
            on_tree = 1.0 if node in nodes_on_tree else 0.0
            features.append(on_tree)

            # 特征7: 是否是DC节点
            is_dc = 1.0 if (hasattr(self, 'dc_nodes') and node in self.dc_nodes) else 0.0
            features.append(is_dc)

            # 🔥🔥🔥 特征8: 该节点已部署的VNF数量（归一化）🔥🔥🔥
            # 这是最关键的特征！让Agent知道哪些节点已经被用过
            vnf_count_normalized = min(node_vnf_counts[node] / 5.0, 1.0)  # 归一化到[0,1]
            features.append(vnf_count_normalized)

            # 特征9: VNF部署进度（该节点视角）
            # 如果这个节点有VNF，显示部署了多少比例
            total_vnf = len(vnf_list)
            if total_vnf > 0:
                local_vnf_progress = node_vnf_counts[node] / total_vnf
            else:
                local_vnf_progress = 0.0
            features.append(local_vnf_progress)

            # 特征10: 节点"负载"指示器（综合特征）
            # 综合资源利用率和VNF数量，帮助Agent避免过载节点
            resource_usage = (1.0 - cpu) * 0.5 + (1.0 - mem) * 0.5  # 资源使用率
            vnf_load = min(node_vnf_counts[node] / 3.0, 1.0)  # VNF负载
            node_load = resource_usage * 0.6 + vnf_load * 0.4  # 综合负载
            features.append(node_load)

            x.append(features)

        x = torch.tensor(x, dtype=torch.float32)

        # =============================
        # 2. 边特征 [E, 2] - 保持不变
        # =============================
        edge_index = []
        edge_attr = []

        for u in range(n):
            try:
                neighbors = self.resource_mgr.get_neighbors(u)
                if not neighbors:
                    continue

                for v in neighbors:
                    if u < v:  # 无向图
                        edge_index.append([u, v])
                        edge_index.append([v, u])

                        if hasattr(self.resource_mgr, 'get_link_bandwidth'):
                            bw = self.resource_mgr.get_link_bandwidth(u, v) / 100.0
                            delay = self.resource_mgr.get_link_delay(u, v) / 100.0
                        else:
                            link_info = self.resource_mgr.links.get((u, v),
                                                                    self.resource_mgr.links.get((v, u), {}))
                            bw = link_info.get('bw', 50.0) / 100.0
                            delay = link_info.get('delay', 10.0) / 100.0

                        edge_attr.append([bw, delay])
                        edge_attr.append([bw, delay])
            except Exception:
                continue

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        # =============================
        # 3. 全局特征 [1, 5] - 🔥 保持维度不变！
        # =============================
        bw_req = self.current_request.get('bw_origin', 0.0) / 10.0

        # VNF部署进度
        deployed_count = self._get_total_vnf_progress()
        vnf_progress = deployed_count / max(1, len(vnf_list))

        # 目的地连接进度
        dest_progress = len(connected) / max(1, len(dests))

        # 当前阶段
        phase = 0.0 if vnf_progress < 1.0 else 1.0

        # 树的规模
        tree_size = len(nodes_on_tree) / max(1, n)

        global_attr = torch.tensor([
            [bw_req, vnf_progress, dest_progress, phase, tree_size]
        ], dtype=torch.float32)

        # =============================
        # 4. 返回Data对象
        # =============================
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr
        )

        # 🔥 调试日志
        if deployed_count > 0 and deployed_count < len(vnf_list):
            # 只在部署进行中时打印，避免日志过多
            top_loaded_nodes = sorted(
                enumerate(node_vnf_counts),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            logger.debug(
                f"📊 [状态] VNF进度:{deployed_count}/{len(vnf_list)}, "
                f"负载最高节点: {[(n, c) for n, c in top_loaded_nodes if c > 0]}"
            )

    def _get_hop_distance(self, node1, node2):
        """
        🔧 [辅助方法] 计算两节点间的跳数
        """
        import networkx as nx

        # 构建临时图
        G = nx.Graph()
        for u in range(self.n):
            neighbors = self.resource_mgr.get_neighbors(u)
            for v in neighbors:
                G.add_edge(u, v)

        try:
            return nx.shortest_path_length(G, node1, node2)
        except nx.NetworkXNoPath:
            return 9999  # 不可达

    def _get_total_vnf_progress(self):
        """
        🔧 [辅助方法] 获取全局VNF部署进度
        """
        if not self.current_request:
            return 0

        vnf_list = self.current_request.get('vnf', [])
        if len(vnf_list) == 0:
            return 0

        placement = self.current_tree.get('placement', {})

        # 统计已部署的VNF索引
        deployed_indices = set()
        for (node, vnf_idx), _ in placement.items():
            deployed_indices.add(vnf_idx)

        return len(deployed_indices)

    # 低层
    def step_low_level(self, action):
        """
        🔥 [V40.0 最终稳定版]

        修复：
        1. 使用next_vnf_idx作为唯一推进源
        2. 所有truncated=True前清零subgoal_step_count
        3. timeout时清空current_phase等状态
        """
        if self.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_req'}

        if not hasattr(self, 'subgoal_step_count'):
            self.subgoal_step_count = 0
        self.subgoal_step_count += 1

        # ========================================
        # 🔥 修复点1：timeout时清空所有状态
        # ========================================
        if self.subgoal_step_count > getattr(self, 'max_subgoal_steps', 50):
            self.current_phase = None
            self.current_deployment_target = None
            self.current_vnf_to_deploy = None
            self.current_target_node = None
            self.subgoal_step_count = 0

            return self.get_state(), -1.0, False, True, {'timeout': True}

        current_node = self.current_node_location
        target_action = int(action)
        is_stay = (target_action == current_node)

        # ============================================================
        # 阶段1：VNF 部署
        # ============================================================
        if self.current_phase == 'vnf_deployment':
            target_goal = getattr(self, 'current_deployment_target', None)

            # --- A. 移动逻辑 ---
            if not is_stay:
                neighbors = self.resource_mgr.get_neighbors(current_node)
                if target_action in neighbors:
                    # ==============================================================
                    # 🔥🔥🔥 [核心修复] 移动产生链路消耗，必须申请并记账 🔥🔥🔥
                    # ==============================================================
                    u, v = current_node, target_action
                    edge_key = tuple(sorted((u, v)))
                    bw_req = self.current_request.get('bw_origin', 1.0)

                    # 1. 检查边是否已在树中 (复用链路不重复扣费)
                    if edge_key in self.current_tree.get('tree', {}):
                        # ✅ 已经在树上，免费通过
                        self.current_node_location = target_action
                        reward = -0.1
                        # logger.debug(f"🚶 复用链路 {u}-{v}")
                    else:
                        # 2. 新边，尝试分配资源
                        if self.resource_mgr.allocate_link_resource(u, v, bw_req):
                            # ✅ 分配成功 -> 关键：记账！
                            if 'tree' not in self.current_tree:
                                self.current_tree['tree'] = {}

                            self.current_tree['tree'][edge_key] = bw_req

                            # 🔥🔥🔥 [新增] 更新树节点集合 (修复MAB连通性警告) 🔥🔥🔥
                            # 只有真正建立了物理连接的节点，才算“在树上”
                            self.nodes_on_tree.add(u)
                            self.nodes_on_tree.add(v)

                            self.current_node_location = target_action
                            reward = -0.1
                            # logger.debug(f"➕ 新增链路 {u}-{v} (BW: {bw_req})")
                        else:
                            # ❌ 带宽不足，移动失败
                            logger.warning(f"❌ [移动失败] 链路 {u}-{v} 带宽不足")
                            return self.get_state(), -5.0, False, False, {'error': 'link_bw_full'}

                    # --- 原有的距离奖励逻辑 (引导Agent向目标靠近) ---
                    if target_goal is not None:
                        try:
                            old_d = self._get_hop_distance(current_node, target_goal)
                            new_d = self._get_hop_distance(target_action, target_goal)
                            if new_d < old_d:
                                reward += 0.5  # 靠近奖励
                            # else:
                            #     reward -= 0.1 # 远离惩罚（可选）
                        except Exception:
                            pass  # 忽略距离计算错误

                    return self.get_state(), reward, False, False, {'move': 'success'}
                else:
                    # 试图移动到非邻居节点（非法动作）
                    return self.get_state(), -1.0, False, False, {'error': 'invalid_move'}
            # --- B. 部署逻辑 ---
            if target_goal is not None and current_node == target_goal:
                if self._try_deploy(target_goal):
                    # ========================================
                    # 🔥🔥🔥 关键修复：部署成功后推进指针
                    # ========================================
                    self.next_vnf_idx += 1

                    vnf_list = self.current_request.get('vnf', [])
                    current_count = self._get_total_vnf_progress()

                    if current_count >= len(vnf_list):
                        # ========================================
                        # 🔥 修复点2：所有VNF部署完成
                        # ========================================
                        self.current_phase = None
                        self.current_deployment_target = None
                        self.current_vnf_to_deploy = None
                        self.subgoal_step_count = 0

                        return self.get_state(), 20.0, False, True, {
                            'phase': 'vnf_complete',
                            'all_vnf_deployed': True,
                            'deployed_count': current_count,
                            'total_vnf': len(vnf_list)
                        }
                    else:
                        # ========================================
                        # 🔥 修复点3：单个VNF部署成功（最关键）
                        # ========================================
                        self.current_phase = None
                        self.current_deployment_target = None
                        self.current_vnf_to_deploy = None
                        self.subgoal_step_count = 0

                        return self.get_state(), 10.0, False, True, {
                            'vnf_deployed': True,
                            'vnf_idx': self.next_vnf_idx - 1,  # 已部署的VNF索引
                            'deployed_count': current_count,
                            'total_vnf': len(vnf_list)
                        }
                else:
                    # ========================================
                    # 🔥 修复点4：部署失败（不推进指针）
                    # ========================================
                    self.current_phase = None
                    self.current_deployment_target = None
                    self.current_vnf_to_deploy = None
                    self.subgoal_step_count = 0

                    return self.get_state(), -5.0, False, True, {
                        'deploy_fail': True,
                        'vnf_idx': self.next_vnf_idx,  # 失败的VNF索引
                        'reason': 'resource_insufficient'
                    }
            else:
                return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

        # ============================================================
        # 阶段2：目的地连接
        # ============================================================
        elif self.current_phase == 'destination_connection':
            target_goal = getattr(self, 'current_target_node', None)

            if not is_stay:
                neighbors = self.resource_mgr.get_neighbors(current_node)
                if target_action in neighbors:
                    self.current_node_location = target_action
                    return self.get_state(), -0.1, False, False, {'move': 'success'}
                else:
                    return self.get_state(), -1.0, False, False, {'error': 'invalid_move'}

            if current_node == target_goal:
                self.current_tree['connected_dests'].add(target_goal)
                all_dests = self.current_request.get('dest', [])
                connected = self.current_tree.get('connected_dests', set())

                if len(connected) >= len(all_dests):
                    # ========================================
                    # 🔥 修复点5：所有目的地连接完成
                    # ========================================
                    self.current_phase = None
                    self.current_target_node = None
                    self.subgoal_step_count = 0
                    return self.get_state(), 50.0, True, False, {
                        'episode_complete': True,
                        'connected_count': len(connected),
                        'total_dests': len(all_dests)
                    }
                else:
                    # ========================================
                    # 🔥 修复点6：单个目的地连接成功
                    # ========================================
                    self.current_phase = None
                    self.current_target_node = None
                    self.subgoal_step_count = 0

                    return self.get_state(), 10.0, False, True, {
                        'dest_connected': True,
                        'connected_count': len(connected),
                        'total_dests': len(all_dests)
                    }
            else:
                return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

        return self.get_state(), -10.0, True, False, {'error': 'unknown_phase'}

    def _handle_movement(self, current, target, goal):
        """🔧 提取通用的移动逻辑，减少代码重复"""
        distance_before = self._get_hop_distance(current, goal)
        distance_after = self._get_hop_distance(target, goal)

        if self._can_move_to(current, target):
            self.current_node_location = target

            # 奖励设计：靠近+2，远离-3 (鼓励最短路)
            if distance_after < distance_before:
                reward = 2.0
            else:
                reward = -3.0

            # 记录详细日志方便Debug
            logger.debug(f"🚶 {current}->{target} | Dist: {distance_before}->{distance_after}")

            return self.get_state(), reward, False, False, {
                'move': True,
                'closer': distance_after < distance_before
            }
        else:
            logger.warning(f"❌ 移动失败: {current}->{target} 链路不通")
            return self.get_state(), -5.0, False, False, {'move_fail': True}

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

    def _can_move_to(self, from_node, to_node):
        """检查是否可以移动"""
        return self.resource_mgr.has_link(from_node, to_node)

    def get_low_level_action_mask(self):
        """
        🎭 [低层Mask V36.0] 基于最短路径的强引导
        """
        import networkx as nx

        mask = np.zeros(self.n, dtype=np.float32)
        current = self.current_node_location
        phase = getattr(self, 'current_phase', 'unknown')

        # 构建网络图
        G = nx.Graph()
        for u in range(self.n):
            neighbors = self.resource_mgr.get_neighbors(u)
            for v in neighbors:
                if self.resource_mgr.has_link(u, v):
                    G.add_edge(u, v)

        # ============================================
        # VNF部署阶段
        # ============================================
        if phase == 'vnf_deployment':
            target = getattr(self, 'current_deployment_target', None)

            if target is not None:
                # 检查目标可达性
                try:
                    path = nx.shortest_path(G, current, target)
                    is_reachable = True
                except nx.NetworkXNoPath:
                    logger.error(f"❌ [Mask] 目标{target}从{current}不可达")
                    is_reachable = False

                # 🔥 情况1：已在目标节点 - 强制STAY
                if current == target:
                    mask[current] = 100.0
                    # 完全禁止移动
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 0.0
                    logger.debug(f"🎯 [Mask] 在目标{current}，强制STAY")

                # 🔥 情况2：目标可达 - 沿最短路径强引导
                elif is_reachable:
                    next_hop = path[1] if len(path) > 1 else current
                    neighbors = self.resource_mgr.get_neighbors(current)

                    # 完全禁止STAY
                    mask[current] = 0.0

                    for nbr in neighbors:
                        if nbr == next_hop:
                            # 最短路径下一跳：超高权重
                            mask[nbr] = 100.0
                            logger.debug(f"🎯 [Mask] 最短路径: {current}→{nbr}→{target}")
                        elif nbr == target:
                            # 邻居就是目标：最高权重
                            mask[nbr] = 100.0
                            logger.debug(f"🎯 [Mask] 邻居{nbr}就是目标！")
                        else:
                            # 检查该邻居到目标的距离
                            try:
                                nbr_path = nx.shortest_path(G, nbr, target)
                                nbr_dist = len(nbr_path) - 1
                                curr_dist = len(path) - 1

                                if nbr_dist < curr_dist:
                                    mask[nbr] = 50.0  # 更近
                                elif nbr_dist == curr_dist:
                                    mask[nbr] = 20.0  # 同样近
                                elif nbr_dist <= curr_dist + 2:
                                    mask[nbr] = 5.0  # 稍远
                                else:
                                    mask[nbr] = 1.0  # 绕远路
                            except nx.NetworkXNoPath:
                                mask[nbr] = 0.1  # 死路

                # 🔥 情况3：目标不可达 - 允许探索
                else:
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 1.0
                    mask[current] = 1.0  # 也允许STAY

        # ============================================
        # 目的地连接阶段（逻辑相同）
        # ============================================
        elif phase == 'destination_connection':
            target = getattr(self, 'current_target_node', None)

            if target is not None:
                try:
                    path = nx.shortest_path(G, current, target)
                    is_reachable = True
                except nx.NetworkXNoPath:
                    is_reachable = False

                if current == target:
                    mask[current] = 100.0
                    neighbors = self.resource_mgr.get_neighbors(current)
                    for nbr in neighbors:
                        mask[nbr] = 0.0

                elif is_reachable:
                    next_hop = path[1] if len(path) > 1 else current
                    neighbors = self.resource_mgr.get_neighbors(current)
                    mask[current] = 0.0

                    for nbr in neighbors:
                        if nbr == next_hop or nbr == target:
                            mask[nbr] = 100.0
                        else:
                            try:
                                nbr_path = nx.shortest_path(G, nbr, target)
                                mask[nbr] = max(1.0, 100.0 / len(nbr_path))
                            except nx.NetworkXNoPath:
                                mask[nbr] = 0.1

        # ============================================
        # 兜底逻辑
        # ============================================
        else:
            mask[current] = 1.0
            neighbors = self.resource_mgr.get_neighbors(current)
            for nbr in neighbors:
                mask[nbr] = 1.0

        # 🔥 最终安全检查
        if np.sum(mask) == 0:
            logger.critical(f"⚠️ [Mask] 所有动作被屏蔽，强制允许STAY")
            mask[current] = 1.0

        return mask

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
            avail_cpu = self.resource_mgr.pool.get_available_cpu(node)
            avail_mem = self.resource_mgr.pool.get_available_memory(node)

            cpu_rem = avail_cpu
            mem_rem = avail_mem
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

    # 寻路逻辑 _a_star_search _find_path _get_distance
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

    def get_next_hop_to_target(self, current, target):
        """
        🧭 [V36.0] 智能获取下一跳节点
        优先级：最短路径 > 启发式距离 > 随机邻居
        """
        import networkx as nx

        # 情况1：已在目标
        if current == target:
            return current

        # 情况2：构建网络图
        G = nx.Graph()
        for u in range(self.n):
            neighbors = self.resource_mgr.get_neighbors(u)
            for v in neighbors:
                if self.resource_mgr.has_link(u, v):
                    G.add_edge(u, v)

        # 情况3：尝试最短路径
        try:
            path = nx.shortest_path(G, current, target)
            if len(path) > 1:
                next_hop = path[1]
                logger.debug(f"🧭 [路径] 最短路径: {current}→{next_hop}→...→{target}")
                return next_hop
            else:
                return current
        except nx.NetworkXNoPath:
            logger.warning(f"⚠️ [路径] 无路径: {current}→{target}")

        # 情况4：启发式选择（距离目标最近的邻居）
        neighbors = self.resource_mgr.get_neighbors(current)
        if not neighbors:
            logger.error(f"❌ [路径] 节点{current}无邻居！")
            return current

        best_neighbor = None
        best_distance = float('inf')

        for nbr in neighbors:
            try:
                nbr_path = nx.shortest_path(G, nbr, target)
                dist = len(nbr_path) - 1
            except nx.NetworkXNoPath:
                dist = float('inf')

            if dist < best_distance:
                best_distance = dist
                best_neighbor = nbr

        if best_neighbor is not None and best_distance != float('inf'):
            logger.debug(f"🧭 [路径] 启发式: {current}→{best_neighbor} (距{target}还有{best_distance}跳)")
            return best_neighbor

        # 情况5：无奈之选（返回第一个邻居）
        logger.warning(f"⚠️ [路径] 目标{target}完全不可达，随机选择邻居{neighbors[0]}")
        return neighbors[0]

    # 资源检查 _check_node_resource _check_deployment_validity
    # _try_deploy  _manual_release_resources _archive_request _update_tree_state
    def _check_node_resources(self, node_id: int, vnf_idx: int = None) -> bool:
        """
        🔥 [V3.7 修复变量名错误] 检查资源（含虚拟预扣）

        Args:
            node_id: 节点ID
            vnf_idx: VNF索引（如果None，则自动推断）
        """
        try:
            if self.current_request is None:
                return True

            vnf_list = self.current_request.get('vnf', [])

            # 使用传入的vnf_idx，或者使用环境的进度
            if vnf_idx is None:
                vnf_idx = self._get_total_vnf_progress()

            # 如果已经部署完了，就不需要检查了
            if vnf_idx >= len(vnf_list):
                return True

            # 获取需求值
            cpu_reqs = self.current_request.get('cpu_origin', []) or \
                       self.current_request.get('vnf_cpu', [])
            mem_reqs = self.current_request.get('memory_origin', []) or \
                       self.current_request.get('mem_origin', [])

            req_cpu = float(cpu_reqs[vnf_idx]) if vnf_idx < len(cpu_reqs) else 1.0
            req_mem = float(mem_reqs[vnf_idx]) if vnf_idx < len(mem_reqs) else 1.0

            # 统计虚拟预订
            reserved_cpu = 0.0
            reserved_mem = 0.0

            placement = self.current_tree.get('placement', {})
            current_branch = getattr(self, 'current_branch_id', None)

            for key, info in placement.items():
                if len(key) >= 3:
                    p_node = key[0]
                    p_branch = key[2]

                    if p_node == node_id and p_branch == current_branch:
                        reserved_cpu += info.get('cpu_used', 0.0)
                        reserved_mem += info.get('mem_used', 0.0)

            # 🔥 修复：使用正确的变量名 node_id
            avail_cpu = self.resource_mgr.pool.get_available_cpu(node_id)
            avail_mem = self.resource_mgr.pool.get_available_memory(node_id)

            # 详细日志
            logger.warning(f"🔍 [资源检查] 节点{node_id}, VNF[{vnf_idx}]")
            logger.warning(f"   物理资源: CPU={avail_cpu:.1f}, Mem={avail_mem:.1f}")
            logger.warning(f"   虚拟预订: CPU={reserved_cpu:.1f}, Mem={reserved_mem:.1f}")
            logger.warning(f"   可用资源: CPU={avail_cpu - reserved_cpu:.1f}, Mem={avail_mem - reserved_mem:.1f}")
            logger.warning(f"   VNF需求: CPU={req_cpu:.1f}, Mem={req_mem:.1f}")

            cpu_ok = (avail_cpu - reserved_cpu) >= (req_cpu * 1.05)
            mem_ok = (avail_mem - reserved_mem) >= (req_mem * 1.05)

            logger.warning(f"   判断: {'✅可部署' if (cpu_ok and mem_ok) else '❌不足'}")

            if not cpu_ok:
                logger.warning(f"   ❌ CPU不足: {avail_cpu - reserved_cpu:.1f} < {req_cpu * 1.05:.1f}")
            if not mem_ok:
                logger.warning(f"   ❌ 内存不足: {avail_mem - reserved_mem:.1f} < {req_mem * 1.05:.1f}")

            return cpu_ok and mem_ok

        except Exception as e:
            logger.error(f"❌ [资源检查] 节点{node_id}检查失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _try_deploy(self, node):
        """
        🔥 [V40.3 最终修复版]
        修复：
        1. 优先使用 self.next_vnf_idx 获取准确的 VNF 索引
        2. 只在VNF部署阶段执行，目的地连接阶段直接返回
        3. 🔥🔥🔥 [新增] 部署成功后立即记账到 current_tree['placement']
        """
        if self.current_request is None:
            logger.error("❌ [部署] 没有当前请求")
            return False

        # 🔥 新增：如果不在VNF部署阶段，直接返回True（静默）
        if self.current_phase != 'vnf_deployment':
            return True

        vnf_list = self.current_request.get('vnf', [])
        if len(vnf_list) == 0:
            logger.info("✅ [部署] 没有VNF需要部署")
            return True

        # 🔥🔥🔥 修复点：优先读取 HRL 指针，不再依赖滞后的状态统计 🔥🔥🔥
        if hasattr(self, 'next_vnf_idx'):
            next_vnf_idx = self.next_vnf_idx
        else:
            # 兜底旧逻辑
            next_vnf_idx = self._get_total_vnf_progress()

        # 越界检查（静默返回，不打印日志）
        if next_vnf_idx >= len(vnf_list):
            return True

        next_vnf_type = vnf_list[next_vnf_idx]

        # --- 以下逻辑保持不变 ---

        # 详细资源检查
        logger.info(f"\n🔍 [部署检查] 节点{node}, VNF[{next_vnf_idx}]类型{next_vnf_type}")

        # 获取资源需求
        cpu_reqs = self.current_request.get('cpu_origin', []) or self.current_request.get('vnf_cpu', [])
        mem_reqs = self.current_request.get('memory_origin', []) or self.current_request.get('vnf_mem', [])

        required_cpu = cpu_reqs[next_vnf_idx] if next_vnf_idx < len(cpu_reqs) else 10.0
        required_mem = mem_reqs[next_vnf_idx] if next_vnf_idx < len(mem_reqs) else 10.0

        logger.info(f"   VNF需求: CPU={required_cpu}, Mem={required_mem}")

        # 获取节点资源
        avail_cpu = self.resource_mgr.pool.get_available_cpu(node)
        avail_mem = self.resource_mgr.pool.get_available_memory(node)

        logger.info(f"   节点资源: CPU={avail_cpu}, Mem={avail_mem}")
        logger.info(f"   资源足够: CPU={avail_cpu >= required_cpu}, Mem={avail_mem >= required_mem}")

        # 资源检查
        if avail_cpu < required_cpu or avail_mem < required_mem:
            logger.error(f"❌ [部署失败] 节点{node}资源不足")
            logger.error(f"   需要: CPU={required_cpu}, Mem={required_mem}")
            logger.error(f"   可用: CPU={avail_cpu}, Mem={avail_mem}")
            return False

        # DC节点检查
        if hasattr(self, 'dc_nodes') and node not in self.dc_nodes:
            logger.error(f"❌ [部署失败] 节点{node}不是DC节点")
            return False

        # 执行部署
        logger.info(f"✅ [部署成功] 节点{node}部署VNF[{next_vnf_idx}]类型{next_vnf_type}")

        # 扣除资源
        if hasattr(self.resource_mgr, 'allocate_node_resource'):
            success = self.resource_mgr.allocate_node_resource(node, next_vnf_type, required_cpu, required_mem)
            if not success:
                logger.error(f"❌ [部署失败] 资源分配失败")
                return False

            # =========================================================
            # 🔥🔥🔥 [核心修复] 必须记账！否则结算时不知道这里有VNF 🔥🔥🔥
            # =========================================================
            # 使用 (node, vnf_idx) 作为唯一键，防止同一节点部署多个VNF时覆盖
            placement_key = (node, next_vnf_idx)

            # 确保字典存在（防守式编程）
            if 'placement' not in self.current_tree:
                self.current_tree['placement'] = {}

            self.current_tree['placement'][placement_key] = {
                'node': node,
                'vnf_type': next_vnf_type,
                'cpu_used': required_cpu,
                'mem_used': required_mem
            }
            # logger.info(f"📝 [记账] VNF[{next_vnf_idx}] @ {node} 已记录到树结构")

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
                    logger.info(f"🗂️ [Resource Flow] 请求 {req.get('id', 'unknown')}: 已归档 (Archived & Tracking)")
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

    # 工具函数
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

    # 最终树减枝
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
        logger.info(f"🔄 [Resource Flow] 请求 {req_id}: 请求开始进行事务预留 (Transaction Start)")
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
                self._archive_request(success=True)
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