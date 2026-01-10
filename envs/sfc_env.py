# envs/sfc_env.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFC_HIRL_Env - 完整可运行的主环境类（分层强化学习 + 多播感知）
已完全模块化，职责清晰，兼容 Flat 和 GNN 两种状态表示
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import gym
import pickle
import torch

# 导入自定义模块
from envs.modules.resource import ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.path_manager import PathManager
from envs.modules.event_handler import EventHandler
from envs.modules.policy_helper import PolicyHelper
from envs.modules.failure_visualizer import FailureVisualizer
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
        检查并释放过期的请求

        这是核心方法！只基于请求的实际过期时间

        Args:
            current_time: 当前时间

        Returns:
            list: 被释放的请求ID列表
        """
        expired_req_ids = []

        # 遍历所有活跃请求
        for req_id, req_info in list(self.active_requests.items()):
            # 🔥 关键：只检查请求是否真的过期
            if current_time > req_info['expire_time']:
                expired_req_ids.append(req_id)

        # 释放过期请求的资源
        for req_id in expired_req_ids:
            self._release_request_resources(req_id, current_time)

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

        # 5. GNN 与 Gym 空间定义
        self._init_gym_spaces()
        self.branch_states = {}
        self.current_branch_id = None
        self.branch_counter = 0
        self.vnf_deployment_history = {}
        logger.info(f"✅ 环境初始化完成: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")

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
        }
        return self.get_state(), info

    def step(self, action):
        """🔥 [统一入口 V4.3 完整Debug+防崩溃版]"""

        # 1. 路由决策层级
        if self.current_branch_id is None:
            obs, reward, done, truncated, info = self.step_high_level(action)
        else:
            obs, reward, done, truncated, info = self.step_low_level(action)

        # 🛑🛑🛑 【关键修复】安全刹车 🛑🛑🛑
        # 如果子函数已经把请求归档了(None)，必须立刻返回，否则后面打印日志时会报 NoneType 错
        if done or self.current_request is None:
            return obs, reward, done, truncated, info

        # ========================================================
        # 🔥 [核心改进 A] 自动吸附逻辑
        # ========================================================
        progress = self._get_current_progress()
        dests = set(self.current_request.get('dest', []))
        connected = self.current_tree.get('connected_dests', set())
        current_node = self.current_node_location

        # 只有 VNF 处理完 (3/3) 且当前在目的地，才触发吸附
        if progress >= 1.0 and current_node in dests and current_node not in connected:
            # 执行环境内部的连接逻辑
            connect_ok = self._connect_destination(current_node)
            if connect_ok:
                # 更新局部变量以进入下方的结算逻辑
                connected = self.current_tree.get('connected_dests', set())
                reward += 100.0  # 给予极高的即时奖励引导
                info['reached_new_dest'] = True
                print(f"✨ [Auto Connect] 进度满且踩到目的地 {current_node}，强制吸附结算！")

        # 2. 检查任务是否完成 (所有目的地物理连接)
        if not done and len(connected) >= len(dests) and len(dests) > 0:
            print(f"\n🏭 [质检流水线] 请求 {self.current_request.get('id')} 物理连接完成，开始验证...")

            # A. 剪枝
            pruned_tree, valid_nodes, prune_success, parent_map = self._prune_redundant_branches_with_vnf()
            if not prune_success:
                # 如果剪枝发现孤岛，给重罚
                return obs, -100.0, True, False, {'success': False, 'error': 'island_topology'}

            # B. SFC 路径验证 (严格质检)
            sfc_ok, sfc_errors = self._validate_sfc_paths(parent_map)
            if not sfc_ok:
                print("❌ [SFC验证失败]")
                for e in sfc_errors: print(f"   {e}")
                # 如果 SFC 路径不通，可能是模型绕路太远，给重罚
                return obs, -200.0, True, False, {'success': False, 'error': 'incomplete_sfc'}

            # C. 统一扣费 (Commit)
            self.current_tree['tree'] = pruned_tree
            self.nodes_on_tree = valid_nodes

            if not self._commit_resources(pruned_tree, valid_nodes):
                return obs, -50.0, True, False, {'success': False, 'error': 'resource_commit_fail'}

            # D. 成功归档
            self._archive_request(success=True)
            print("✅ [结算成功] 资源已扣除，任务完成")

            return obs, 200.0, True, False, {'success': True, 'request_completed': True}

        # ========================================================
        # 🔥 [核心改进 B] 徘徊惩罚补充逻辑
        # ========================================================
        if progress >= 1.0 and info.get('action_type') == 'move':
            # 如果 VNF 完事了还在已有的树上移动且没踩到新目的地
            if current_node in self.nodes_on_tree and not info.get('reached_new_dest', False):
                reward -= 15.0  # 对应你的 backtrack_penalty
                info['is_backtracking'] = True

        return obs, reward, done, truncated, info

    def step_high_level(self, action):
        """
        🔥 [V11.6 修复版] 高层决策 - 修复 Mask 丢失问题
        """
        # 解析动作
        if isinstance(action, (tuple, list, np.ndarray)):
            subgoal_idx = int(action[0])
        else:
            subgoal_idx = int(action)

        # 检查是否有当前请求
        if self.current_request is None:
            # 即使没请求，也要返回 Mask 防止崩溃
            mask = np.ones(self.n, dtype=np.bool_)
            return self.get_state(), 0.0, True, False, {'no_request': True, 'action_mask': mask}

        # 获取目的地列表
        dests = self.current_request.get('dest', [])
        if not dests:
            mask = np.ones(self.n, dtype=np.bool_)
            return self.get_state(), 0.0, True, False, {'no_destinations': True, 'action_mask': mask}

        # 获取已连接的目的地
        connected = self.current_tree.get('connected_dests', set())

        # 🔥 确保 unadded_dest_indices 已初始化
        if not hasattr(self, 'unadded_dest_indices'):
            self.unadded_dest_indices = set(range(len(dests)))

        # 🔥 移除已连接的目的地索引
        for i, dest in enumerate(dests):
            if dest in connected:
                self.unadded_dest_indices.discard(i)

        # 检查是否还有未连接的目的地
        if not self.unadded_dest_indices:
            # 所有目的地已连接，请求完成
            mask = np.ones(self.n, dtype=np.bool_)
            return self.get_state(), 0.0, True, False, {'all_connected': True, 'action_mask': mask}

        # 🔥 选择目标节点
        if subgoal_idx < len(self.unadded_dest_indices):
            dest_idx = sorted(self.unadded_dest_indices)[subgoal_idx]
        else:
            # 如果索引超出范围，选择第一个
            dest_idx = sorted(self.unadded_dest_indices)[0]

        target_node = dests[dest_idx]

        # 🔥 生成新的分支ID
        if not hasattr(self, '_branch_counter'):
            self._branch_counter = 0
        self._branch_counter += 1
        new_branch_id = f"branch_{self._branch_counter}"

        self.current_branch_id = new_branch_id

        # 🔥🔥🔥 关键修复：存储分支状态（包括target_node）
        if not hasattr(self, 'branch_states'):
            self.branch_states = {}

        self.branch_states[new_branch_id] = {
            'target_node': target_node,  # 🔥 低层需要这个！
            'dest_idx': dest_idx,
            'created_at': self.decision_step,
            'completed': False,
            'failed': False,
            'timeout': False,
            'looping': False
        }

        print(f"🌿 [高层] 新分支 {new_branch_id} -> {target_node}")

        # 🔥 重置低层计数器
        if not hasattr(self, '_current_goal_steps'):
            self._current_goal_steps = 0
        else:
            self._current_goal_steps = 0

        if not hasattr(self, '_vnf_complete_steps'):
            self._vnf_complete_steps = 0
        else:
            self._vnf_complete_steps = 0

        if hasattr(self, '_recent_positions'):
            self._recent_positions = []

        # 🔥🔥🔥 核心修复：在这里计算 Mask！🔥🔥🔥
        # 因为下一步马上就是 Low-Level Agent 决策，它需要这个 Mask
        low_level_mask = self.get_low_level_action_mask()

        # 返回状态继续低层决策
        info = {
            'branch_created': True,
            'target': target_node,
            'branch_id': new_branch_id,
            'action_mask': low_level_mask  # <--- 必须加上这一行！
        }

        return self.get_state(), 0.0, False, False, info

    def step_low_level(self, action):
        """
        🔥 [V17.0 清理版] 低层决策
        不再在此处检查资源释放，完全依赖 TS 切换时的检查。
        """
        target_node = int(action)
        current_node = self.current_node_location
        mask = self.get_low_level_action_mask()

        info = {'success': False, 'action_mask': mask, 'decision_steps': self.decision_step}
        reward = -0.1
        done, truncated = False, False
        self.decision_step += 1

        # --- 1. 动态惩罚 (Reward 引导) ---
        if not hasattr(self, '_node_visit_count'): self._node_visit_count = {}
        self._node_visit_count[current_node] = self._node_visit_count.get(current_node, 0) + 1
        visits = self._node_visit_count[current_node]

        if visits > 2:
            penalty = -0.5 * (visits - 2)
            reward += penalty

        # 熔断保护
        if visits > 50:
            print(f"🛑 [Loop Break] 节点{current_node} 滞留 >50 次，强制结束")
            self._archive_request(success=False)
            return self.get_state(), -100.0, True, True, info

        # --- 2. 业务逻辑 ---
        vnf_list = self.current_request.get('vnf', [])
        vnf_progress = self._get_path_vnf_progress(current_node)
        vnf_complete = (vnf_progress >= len(vnf_list))
        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())
        is_unconnected_dest = (current_node in dests) and (current_node not in connected)

        # STAY
        if target_node == current_node:
            info['action_type'] = 'stay'
            if vnf_complete:
                if is_unconnected_dest:
                    self._connect_destination(current_node)
                    reward += 300.0
                    if len(self.current_tree['connected_dests']) >= len(dests):
                        if self._finalize_request_with_pruning():
                            info.update({'request_completed': True, 'request_success': True})
                            reward += 700.0
                            done = True
                        else:
                            reward -= 100.0
                            done = True
                    else:
                        reward += 100.0
                        self.current_branch_id = None
                        self._node_visit_count = {}
                else:
                    reward -= 5.0
            else:
                if self._try_deploy(current_node):
                    reward += 15.0
                else:
                    reward -= 5.0

        # MOVE
        else:
            info['action_type'] = 'move'
            if self.resource_mgr.has_link(current_node, target_node):
                self.current_node_location = target_node
                edge_key = tuple(sorted([current_node, target_node]))
                self.current_tree.setdefault('tree', {})[edge_key] = self.current_request.get('bw_origin', 1.0)
                self.nodes_on_tree.add(target_node)

                if vnf_complete and is_unconnected_dest:
                    reward += 50.0
                else:
                    reward -= 0.05
            else:
                reward -= 10.0

        if not done and not truncated:
            info['action_mask'] = self.get_low_level_action_mask()
        else:
            info['action_mask'] = np.zeros(self.n, dtype=np.float32)

        info['time_slot'] = self.current_time_slot
        info['time_step'] = self.time_step

        return self.get_state(), reward, done, truncated, info

    def get_low_level_action_mask(self):
        """
        🔥 [V16.3 防崩溃版]
        增加：对 self.current_request 为 None 的情况进行防御
        """
        mask = np.zeros(self.n, dtype=np.float32)

        # 🔥🔥🔥 核心修复：如果没有请求（例如刚结算完），直接返回全0或全1掩码
        if self.current_request is None:
            # 返回全1防止某些算法除以0，或者全0表示无动作
            # 由于此时 episode 已经 done，这个 mask 不会影响训练
            mask[:] = 1.0
            return mask

        current_node = self.current_node_location

        # 1. 物理：邻居全开
        neighbors = self.resource_mgr.get_neighbors(current_node)
        for n in neighbors:
            mask[n] = 1.0

        # 2. 逻辑：STAY 仅在有用时开
        vnf_list = self.current_request.get('vnf', [])
        vnf_progress = self._get_path_vnf_progress(current_node)
        vnf_complete = (vnf_progress >= len(vnf_list))

        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())

        can_deploy = not vnf_complete
        can_connect = (current_node in dests) and (current_node not in connected) and vnf_complete

        if can_deploy or can_connect:
            mask[current_node] = 1.0
        else:
            mask[current_node] = 0.0

        # 3. 兜底
        if np.sum(mask) == 0:
            mask[current_node] = 1.0

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
#动作与掩码 get_low_level_action_mask get_high_level_action_mask
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

#寻路逻辑 _init_path_planner _a_star_search _find_path _get_distance
    def _init_path_planner(self):
        """初始化路径规划缓存"""
        self._path_cache = {}
    def _a_star_search(self, start, goal):
        """标准的 A* 搜索算法 - 修复版"""
        if start == goal:
            return [start]

        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        def heuristic(n):
            return 0

        f_score = {start: heuristic(start)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # 🔥 修复：直接遍历所有节点，检查是否有链接
            neighbors = []
            for v in range(self.n):
                if v != current and self.resource_mgr.has_link(current, v):
                    neighbors.append(v)

            for neighbor in neighbors:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

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
    def _select_fork_node_heuristic(self):
            """启发式选择分支节点（最近原则）"""
            if not hasattr(self, 'current_dest') or self.current_dest is None:
                return 0

            tree_nodes_list = sorted(list(self.nodes_on_tree))
            if not tree_nodes_list:
                return 0

            # 计算每个树节点到目标的距离
            distances = []
            for node in tree_nodes_list:
                path = self._find_path(node, self.current_dest)
                dist = len(path) - 1 if path else float('inf')
                distances.append(dist)

            return np.argmin(distances) if distances else 0
#资源检查 _check_link_validity _check_node_resource _check_deployment_validity
#_try_deploy  _manual_release_resources _archive_request _update_tree_state
    def _check_link_validity(self, from_node, to_node):
        """检查链路有效性"""
        try:
            if hasattr(self, 'resource_mgr'):
                return self.resource_mgr.has_link(from_node, to_node)
            else:
                return (self.topo[from_node, to_node] > 0)
        except:
            return True
    def _check_node_resources(self, node_id: int) -> bool:
        """
        检查节点资源是否充足 (修复版: 增加索引越界保护)
        """
        try:
            if self.current_request is None:
                return True

            # 获取当前VNF需求
            vnf_list = self.current_request.get('vnf', [])
            if not vnf_list:
                return True

            # 获取当前需要部署的VNF索引
            deployed_count = len([k for k in self.current_tree.get('placement', {}).keys() if isinstance(k, tuple)])
            if deployed_count >= len(vnf_list):
                return True  # 所有VNF已部署

            vnf_idx = deployed_count

            # 🔥 [安全修正] 确保索引不越界
            # 有些数据可能是 cpu_reqs[vnf_type] 而不是 [vnf_idx]，这里假设是按顺序的 [idx]
            cpu_reqs = self.current_request.get('vnf_cpu', []) or \
                       self.current_request.get('cpu_origin', []) or \
                       self.current_request.get('cpu', [1.0] * len(vnf_list))

            # 安全获取需求值
            if isinstance(cpu_reqs, (list, np.ndarray)):
                if vnf_idx < len(cpu_reqs):
                    required_cpu = float(cpu_reqs[vnf_idx])
                else:
                    required_cpu = 1.0  # 默认值
            else:
                required_cpu = float(cpu_reqs)

            # 获取节点可用资源
            available_cpu = 100.0  # 默认值

            if hasattr(self, 'resource_mgr'):
                if hasattr(self.resource_mgr, 'nodes') and isinstance(self.resource_mgr.nodes, list):
                    if 0 <= node_id < len(self.resource_mgr.nodes):
                        available_cpu = self.resource_mgr.nodes[node_id].get('cpu', 100.0)
                # ... 其他获取方式保持不变 ...

            # 留10%余量
            return available_cpu >= required_cpu * 1.1

        except Exception as e:
            # print(f"⚠️ 资源检查警告: {e}")
            return True  # 出错时放行，避免中断训练

    def _try_deploy(self, node):
        """
        🔥 [V12.1 完整虚拟部署版]
        职责：仅验证资源可行性并记录位置，不实际扣除物理资源，彻底解决双重扣费问题。
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

        # 4. 资源预检：根据请求中的原始需求检查 CPU 和内存
        cpu_needs = self.current_request.get('cpu_origin', [1.0] * len(vnf_list))
        mem_needs = self.current_request.get('memory_origin', [1.0] * len(vnf_list))

        c_req = cpu_needs[next_vnf_idx] if next_vnf_idx < len(cpu_needs) else 1.0
        m_req = mem_needs[next_vnf_idx] if next_vnf_idx < len(mem_needs) else 1.0

        # 调用资源管理器检查剩余量 (1.1 倍安全余量)
        if hasattr(self.resource_mgr, 'check_node_resource'):
            if not self.resource_mgr.check_node_resource(node, next_vnf_type, c_req * 1.1, m_req * 1.1):
                return False
        else:
            # 兜底检查：直接读属性
            avail_c = self.resource_mgr.C[node] if hasattr(self.resource_mgr, 'C') else 100.0
            if avail_c < c_req:
                return False

        # 5. 🔥 执行虚拟部署记录 (Placement)
        # 存储格式: (节点ID, VNF类型, 分支ID) -> VNF链索引
        key = (node, next_vnf_type, self.current_branch_id)

        # 记录详细方案信息，供结算阶段 _finalize_request_with_pruning 使用
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

    def _manual_release_resources(self):
        """
        🔥 [V10.15 修复版] 堆管理 + 账本释放 + 返回释放数量
        """
        if not hasattr(self, 'leave_heap') or not self.leave_heap:
            return 0

        import heapq
        released_count = 0

        while self.leave_heap and self.leave_heap[0][0] <= self.time_step:
            leave_time, service = heapq.heappop(self.leave_heap)
            req_id = service.get('id', '?')

            # 释放链路
            link_allocs = service.get('link_allocs', [])
            for alloc in link_allocs:
                if len(alloc) >= 3:
                    u, v, bw = alloc[:3]
                    self.resource_mgr.release_link_resource(u, v, bw)

            # 释放节点
            node_allocs = service.get('node_allocs', [])
            for alloc in node_allocs:
                if len(alloc) >= 4:
                    n, vt, c, m = alloc[:4]
                    self.resource_mgr.release_node_resource(n, vt, c, m)
                elif len(alloc) == 3:
                    n, c, m = alloc
                    self.resource_mgr.release_node_resource(n, 0, c, m)

            released_count += 1

        return released_count

    def _archive_request(self, success=False):
        """🔥 [V16.2 记账版] 成功时保存资源快照，失败时回滚"""
        if self.current_request is None: return
        req = self.current_request

        if success:
            # === 成功：保存账本 ===
            import copy
            req['resources_allocated'] = {
                'placement': copy.deepcopy(self.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.current_tree.get('tree', {}))
            }
            if hasattr(self, 'request_manager'):
                self.request_manager.add_request(req)

            self.total_requests_accepted += 1
            if hasattr(self, 'served_dest_count'):
                self.served_dest_count += len(req.get('dest', []))
        else:
            # === 失败：回滚 ===
            self._rollback_request_resources(req)

        # 重置状态
        self.current_tree = {'hvt': np.zeros((self.n, self.K_vnf), dtype=np.float32),
                             'tree': {}, 'placement': {}, 'connected_dests': set()}
        self.current_request = None
        self.current_branch_id = None
        self.nodes_on_tree = set()
    def _update_tree_state(self, u, v):
        """更新树拓扑"""
        if 'tree' not in self.current_tree:
            self.current_tree['tree'] = {}

        # 记录边
        # 注意：这里记录的是无向图的边或者有向图，取决于你的 Graph 定义
        # 为了 GNN，通常建议存 (min, max) 或者双向
        self.current_tree['tree'][(u, v)] = 1.0

        self.nodes_on_tree.add(u)
        self.nodes_on_tree.add(v)
    def _check_termination_conditions(self):
        """
        检查异常终止条件（防刷分机制）
        返回: (should_terminate, penalty)
        """
        # 1. 频繁访问同一节点检测
        # 如果在短时间内访问同一个节点超过一定次数 (例如 3-4 次)
        if hasattr(self, 'node_visit_counts'):
            current_node_visits = self.node_visit_counts[self.current_node_location]
            if current_node_visits > 4:
                return True, -5.0  # 判定为死循环，给予惩罚并终止

        # 2. 震荡检测 (A->B->A 模式)
        # 需要在 step 中维护一个 self.recent_path = [] 队列
        # if len(self.recent_path) >= 4:
        #     if self.recent_path[-1] == self.recent_path[-3] and \
        #        self.recent_path[-2] == self.recent_path[-4]:
        #         return True, -5.0

        return False, 0.0

    def _get_path_vnf_progress(self, node):
        """🔥 [V15.0 彻底去锁版] 不依赖树路径，直接根据 placement 记录计算连续进度"""
        if self.current_request is None: return 0
        vnf_list = self.current_request.get('vnf', [])

        # 获取 placement 字典
        placement = self.current_tree.get('placement', {})
        if not placement: return 0

        # 收集【任何节点】上已部署的所有 VNF 索引
        deployed_indices = set()
        for key, info in placement.items():
            idx = info.get('vnf_idx') if isinstance(info, dict) else info
            if idx is not None:
                deployed_indices.add(idx)

        # 计算 VNF 链的连续完成进度
        progress = 0
        for i in range(len(vnf_list)):
            if i in deployed_indices:
                progress += 1
            else:
                break  # 必须按顺序
        return progress
#可视化 render_tree_structure _diagnose_connectivity_failure _diagnose_resource_shortage
#_diagnose_illegal_action check_resource_conservation print_connection_status print_navigation_guide
    def render_tree_structure(self):
        """
        🌳 渲染 SFC 多播树（防环版）
        """
        if not self.current_request:
            return

        req_id = self.current_request.get('id', '?')
        src = self.current_request.get('source')
        dests = self.current_request.get('dest', [])
        placement = self.current_tree.get('placement', {})
        raw_edges = self.current_tree.get('tree', {})

        print(f"\n{'=' * 60}")
        print(f"🌳 SFC 多播树可视化 (Request {req_id})")
        print(f"{'=' * 60}")

        # === 1. VNF 部署链 ===
        def get_vnf_idx(k):
            if isinstance(k, int):
                return k
            import re
            m = re.search(r'(\d+)', str(k))
            return int(m.group(1)) if m else -1

        sorted_vnfs = sorted(placement.items(), key=lambda x: get_vnf_idx(x[0]))

        if sorted_vnfs:
            chain = f"🟢 源节点{src}"
            for k, node in sorted_vnfs:
                idx = get_vnf_idx(k)
                chain += f" ══> ⚙️  VNF{idx}@节点{node}"
            print(f"\n📍 VNF链: {chain}\n")

        # === 2. 构建无向邻接表 ===
        edges_set = set()
        for edge_key in raw_edges.keys():
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                normalized = (min(u, v), max(u, v))
                edges_set.add(normalized)

        adj = {}
        for u, v in edges_set:
            if u not in adj: adj[u] = []
            if v not in adj: adj[v] = []
            adj[u].append(v)
            adj[v].append(u)

        print(f"🔗 物理树: {len(edges_set)} 条边, {len(adj)} 个节点\n")

        # === 3. DFS 打印树结构（防环增强版）===
        visited = set()  # 🔥 关键：全局访问记录
    def _diagnose_connectivity_failure(self, step_idx):
        """
        🚑 [深度诊断 - 修复版] 诊断连接失败原因
        修复了 get_link_bandwidth 报错，增加了直接读取资源字典的兼容性
        """
        print(f"\n🔍 [DCC诊断] Step {step_idx} | 当前节点: {self.current_node_location}")

        # 1. 识别剩余目标
        dests = self.current_request.get('dest', [])
        connected = self.current_tree.get('connected_dests', set())
        unconnected = [d for d in dests if d not in connected]

        print(f"   📉 未连接目标: {unconnected}")

        if not unconnected:
            print("   ✅ 所有目标已连接 (无需诊断)")
            return

        # 2. 获取当前 Mask 和 邻居
        mask = self.get_low_level_action_mask()
        if hasattr(self, 'resource_mgr'):
            neighbors = self.resource_mgr.get_neighbors(self.current_node_location)
        else:
            neighbors = self.topology_mgr.get_neighbors(self.current_node_location)

        print(f"   🏠 物理邻居: {neighbors}")
        print(f"   🎭 当前Mask允许: {[n for n in neighbors if mask[n]]}")

        # 3. 逐个分析未连接节点
        for dest in unconnected:
            print(f"   🎯 分析目标 Node {dest}:")

            # --- A. 物理路径检查 (A*) ---
            path = self._find_path(self.current_node_location, dest)
            if not path:
                print(f"      ❌ [物理层] 致命：物理拓扑不连通！无法到达。")
                continue

            # 获取下一跳
            next_hop = path[1] if len(path) > 1 else path[0]
            print(f"      ✅ [物理层] 最短路径: {path} (下一跳: {next_hop})")

            # --- B. Mask 阻断检查 ---
            if not mask[next_hop]:
                print(f"      ❌ [逻辑层] Mask 封锁了最佳下一跳 {next_hop}！")

                # 深入分析 Mask 为什么封锁
                visit_count = 0
                if hasattr(self, 'node_visit_counts'):
                    visit_count = self.node_visit_counts.get(next_hop, 0)

                print(f"         - 访问频次: {visit_count}")

                if visit_count >= 3:
                    print(f"         - 原因: 访问次数过多，触发防死循环锁死。")
                else:
                    print(f"         - 原因: 可达性检测认为那是死胡同，或者是黑名单节点。")
            else:
                print(f"      ✅ [逻辑层] Mask 允许通过。")

            # --- C. 资源/带宽检查 (🔥 核心修复部分) ---
            # 尝试多种方式获取带宽，防止报错
            bw = None
            link = (self.current_node_location, next_hop)

            # 方式1: 尝试调用方法
            if hasattr(self.resource_mgr, 'get_link_bandwidth'):
                try:
                    bw = self.resource_mgr.get_link_bandwidth(self.current_node_location, next_hop)
                except:
                    pass

            # 方式2: 直接访问 links 字典 (这是通常的 SDN 环境结构)
            if bw is None and hasattr(self.resource_mgr, 'links'):
                if isinstance(self.resource_mgr.links, dict):
                    # links 可能包含 'bandwidth' 键
                    if 'bandwidth' in self.resource_mgr.links:
                        bw = self.resource_mgr.links['bandwidth'].get(link)
                        if bw is None:  # 尝试反向链路
                            bw = self.resource_mgr.links['bandwidth'].get((next_hop, self.current_node_location))

            # 方式3: 访问拓扑矩阵 (如果 links 字典不可用)
            if bw is None and hasattr(self.resource_mgr, 'topology'):
                try:
                    bw = self.resource_mgr.topology[self.current_node_location][next_hop]
                except:
                    pass

            # 打印结果
            if bw is not None:
                print(f"      💰 [资源层] 链路 {link} 带宽: {bw}")
                if bw <= 0:
                    print(f"         ❌ 带宽耗尽！这可能是 Agent 不走这条路的原因。")
            else:
                print(f"      ⚠️ [资源层] 无法读取链路带宽信息 (属性缺失)")

        print("=" * 50)
    def _diagnose_resource_shortage(self, node_id, vnf_idx):
        """
        🚑 资源诊断仪 (适配 memory_origin 版)
        """
        try:
            # 1. DC 节点检查
            if hasattr(self, 'dc_nodes'):
                if node_id not in self.dc_nodes:
                    return f"❌ 非DC节点(仅{self.dc_nodes}可用)"

            # 2. 获取需求 (Demand)
            req = self.current_request
            cpu_demand = 0.0
            mem_demand = 0.0

            # --- CPU ---
            # 优先读 'cpu_origin' (你的数据里是这个)
            raw_cpu = req.get('cpu_origin') or req.get('vnf_cpu') or req.get('cpu')
            if raw_cpu:
                if isinstance(raw_cpu, (list, np.ndarray)) and vnf_idx < len(raw_cpu):
                    cpu_demand = float(raw_cpu[vnf_idx])
                elif isinstance(raw_cpu, (int, float)):
                    cpu_demand = float(raw_cpu)

            # --- Memory (关键修复) ---
            # 🔥🔥🔥 优先读 'memory_origin' (你的数据里是这个!) 🔥🔥🔥
            raw_mem = req.get('memory_origin') or req.get('mem_origin') or req.get('memory')
            if raw_mem:
                if isinstance(raw_mem, (list, np.ndarray)) and vnf_idx < len(raw_mem):
                    mem_demand = float(raw_mem[vnf_idx])
                elif isinstance(raw_mem, (int, float)):
                    mem_demand = float(raw_mem)

            # 3. 获取剩余 (Available)
            avail_cpu = 0.0
            avail_mem = 0.0
            if hasattr(self.resource_mgr, 'nodes'):
                nodes = self.resource_mgr.nodes
                # 兼容字典结构 (SOA)
                if isinstance(nodes, dict):
                    avail_cpu = float(nodes.get('cpu', [0] * 100)[node_id])
                    avail_mem = float(nodes.get('memory', [0] * 100)[node_id])
                # 兼容矩阵结构
                elif hasattr(nodes, 'shape'):
                    avail_cpu = float(nodes[node_id][0])
                    # 假设第二列是内存
                    if nodes.shape[1] > 1:
                        avail_mem = float(nodes[node_id][1])

            # 4. 返回详细报告
            return f"DC=OK | CPU: 需{cpu_demand:.2f}/余{avail_cpu:.2f} | MEM: 需{mem_demand:.2f}/余{avail_mem:.2f}"

        except Exception as e:
            return f"诊断崩了: {e}"
    def _diagnose_illegal_action(self, current_node, target_node, vnf_list, dests):
        """诊断非法动作（保留你原来的诊断日志）"""
        print(f"\n{'=' * 60}")
        print(f"❌ [动作被禁止诊断]")
        print(f"   当前位置: {current_node}")
        print(f"   目标位置: {target_node}")

        deployed_count = len(self.current_tree.get('placement', {}))
        is_vnf_complete = (deployed_count >= len(vnf_list))

        if is_vnf_complete:
            print(f"   阶段: 树构建")

            connected = self.current_tree.get('connected_dests', set())
            unconnected = [d for d in dests if d not in connected]

            print(f"   已连接: {list(connected)} ({len(connected)}/{len(dests)})")
            print(f"   未连接: {unconnected}")
            print(f"   目标节点是未连接的目的? {target_node in unconnected}")

            # 物理连接性
            try:
                neighbors = self.resource_mgr.get_neighbors(current_node) if hasattr(self, 'resource_mgr') else []
                print(f"   当前位置的物理邻居: {neighbors}")
                print(f"   目标节点是邻居? {target_node in neighbors}")

                path = self.topology_mgr.get_shortest_path(current_node, target_node)
                if path:
                    print(f"   最短路径: {path} (长度={len(path) - 1})")
                else:
                    print(f"   ❌ 无路径到目标节点！")
            except Exception as e:
                print(f"   路径查找错误: {e}")

            # visit_count
            if hasattr(self, 'node_visit_counts'):
                vc = self.node_visit_counts.get(target_node, 0)
                print(f"   visit_count[目标{target_node}] = {vc}")
        else:
            print(f"   阶段: VNF部署")
            print(f"   已部署: {deployed_count}/{len(vnf_list)}")

        mask = self.get_low_level_action_mask()
        valid_actions = np.where(mask)[0]
        print(f"   可用动作({len(valid_actions)}个): {valid_actions.tolist()}")
        print(f"{'=' * 60}\n")
    def check_resource_conservation(self):
        """
        🔥 [方案B新增] 检查资源守恒性
        用于调试：确保资源没有泄漏或超额分配
        """
        try:
            # 检查CPU资源
            nodes_container = self.resource_mgr.nodes
            is_soa = isinstance(nodes_container, dict) and 'cpu' in nodes_container

            if is_soa:
                total_cpu = sum(nodes_container['cpu'])
            else:
                total_cpu = sum(node['cpu'] for node in nodes_container)

            # 期望的总CPU（假设每节点初始100）
            expected_cpu = len(nodes_container) * 100.0

            if abs(total_cpu - expected_cpu) > 1.0:
                print(f"⚠️ CPU资源不守恒！当前={total_cpu:.1f}, 期望={expected_cpu:.1f}")
                return False

            return True

        except Exception as e:
            print(f"⚠️ 资源检查失败: {e}")
            return True  # 出错时假设正常，避免中断
    def print_connection_status(self):
        """打印连接状态"""
        if not self.current_request or self.current_vnf_index < len(self.current_request.get('vnf', [])):
            return

        dests = self.current_request.get('dest', [])
        if 'connected_dests' not in self.current_tree:
            return

        connected = self.current_tree['connected_dests']
        unconnected = [d for d in dests if d not in connected]

        print(f"\n📊 连接状态: {len(connected)}/{len(dests)}")
        if unconnected:
            print(f"   未连接: {unconnected}")
            print(f"   当前位置: {self.current_node_location}")

            # 计算到每个未连接节点的距离
            distances = []
            for dest in unconnected:
                path = self._find_path(self.current_node_location, dest)
                if path:
                    distances.append((dest, len(path) - 1))
                else:
                    distances.append((dest, 999))

            # 按距离排序
            distances.sort(key=lambda x: x[1])
            print(f"   距离排序:")
            for dest, dist in distances[:3]:  # 显示最近的3个
                if dist < 999:
                    print(f"      {dest}: {dist}跳")
                else:
                    print(f"      {dest}: 不可达")
    def print_navigation_guide(self):
        """打印导航指南"""
        if not self.current_request:
            return

        req = self.current_request
        vnf_list = req.get('vnf', [])
        dests = req.get('dest', [])

        if self.current_vnf_index < len(vnf_list):
            # 部署阶段
            print(f"\n💡 [部署阶段] 需要部署 {len(vnf_list)} 个VNF，已部署 {self.current_vnf_index} 个")
            print(f"   当前节点: {self.current_node_location}")
            print(f"   DC节点: {self.dc_nodes}")

            # 找出可部署的DC节点
            deployable = []
            for dc in self.dc_nodes:
                if dc != req.get('source') and dc not in dests:
                    if self._check_deployment_validity(dc):
                        deployable.append(dc)

            if deployable:
                print(f"   可部署的DC节点: {deployable}")
            else:
                print(f"   ⚠️ 没有可部署的DC节点！检查资源或拓扑")

        else:
            # 树构建阶段
            if 'connected_dests' not in self.current_tree:
                return

            connected = self.current_tree['connected_dests']
            unconnected = [d for d in dests if d not in connected]

            if unconnected:
                print(f"\n🗺️ [导航指南] 已连接 {len(connected)}/{len(dests)}，剩余 {len(unconnected)} 个")
                print(f"   当前位置: {self.current_node_location}")
                print(f"   未连接节点: {unconnected}")

                # 距离排序
                distances = []
                for dest in unconnected:
                    path = self._find_path(self.current_node_location, dest)
                    if path:
                        distances.append((dest, len(path) - 1, path))

                if distances:
                    distances.sort(key=lambda x: x[1])
                    print(f"   距离排序:")
                    for i, (dest, dist, path) in enumerate(distances[:3]):  # 显示最近的3个
                        print(f"     {i + 1}. 节点{dest}: {dist}跳 - 路径: {path}")

#工具函数  _parse_edge set_dynamic_mode
    def _parse_edge(self, edge):
        """
        解析边元组
        支持格式：(u, v), "(u-v)", "u-v" 等
        """
        u, v = None, None

        if isinstance(edge, tuple) and len(edge) == 2:
            u, v = edge
        elif isinstance(edge, str):
            try:
                # 尝试解析 "u-v" 或 "(u-v)" 格式
                u, v = map(int, edge.strip('()').split('-'))
            except:
                pass

        return u, v
    def set_dynamic_mode(self, enabled: bool):
        """由 Trainer 调用，控制是否开启 TTL 离去机制"""
        self.dynamic_env = enabled
        # logger.info(f"🔄 环境动态模式已切换为: {enabled}")
    def _find_path_in_tree(self, source, target):
        """
        在当前树中查找从source到target的路径
        使用BFS
        """
        if source == target:
            return [source]

        # 构建邻接表
        tree_edges = self.current_tree.get('tree', {})
        adj = {}

        for edge_key in tree_edges:
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                u, v = edge_key
                if u not in adj: adj[u] = []
                if v not in adj: adj[v] = []
                adj[u].append(v)
                adj[v].append(u)

        # BFS搜索
        from collections import deque
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current not in adj:
                continue

            for neighbor in adj[current]:
                if neighbor == target:
                    return path + [target]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # 没有路径
    def _merge_branch_to_global(self, branch_state):
        """
        🔥 合并分支结果到全局树
        """
        if not branch_state.get('success', False):
            return

        branch_id = branch_state['branch_id']
        target_dest = branch_state['target_dest']

        print(f"🔄 合并分支 {branch_id} 到全局树")

        # 1. 合并VNF部署（使用复合key）
        if 'placement' not in self.current_tree:
            self.current_tree['placement'] = {}

        for vnf_type, node in branch_state['local_placement'].items():
            key = (node, vnf_type)
            self.current_tree['placement'][key] = {
                'vnf_type': vnf_type,
                'node': node,
                'branch_id': branch_id
            }
            print(f"   部署: {vnf_type} @ 节点{node}")

        # 2. 合并边
        if 'tree' not in self.current_tree:
            self.current_tree['tree'] = {}

        for u, v, bw in branch_state.get('local_edges', []):
            edge_key = tuple(sorted([u, v]))
            self.current_tree['tree'][edge_key] = bw

        # 3. 标记目的地已连接
        if 'connected_dests' not in self.current_tree:
            self.current_tree['connected_dests'] = set()

        self.current_tree['connected_dests'].add(target_dest)

        # 4. 更新树上节点
        for node in branch_state.get('visited_nodes', set()):
            self.nodes_on_tree.add(node)

        print(f"   目标: dest{target_dest} 已连接")
        print(f"   当前已连接: {self.current_tree['connected_dests']}")

    def _validate_sfc_paths(self, parent_map):
        """🕵️ [SFC 质检] 验证 VNF 链完整性"""
        if not self.current_request: return False, ["No request"]

        source = self.current_request['source']
        dests = self.current_request.get('dest', [])
        required_vnfs = self.current_request.get('vnf', [])

        if not required_vnfs: return True, []

        # 构建节点 VNF 映射
        node_vnf_dict = {}
        placement = self.current_tree.get('placement', {})
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v = key[0], key[1]
                if n not in node_vnf_dict: node_vnf_dict[n] = []
                node_vnf_dict[n].append(v)

        errors = []
        for dest in dests:
            # 回溯路径
            path = []
            curr = dest
            while curr is not None:
                path.append(curr)
                if curr == source: break
                curr = parent_map.get(curr)

            if not path or path[-1] != source:
                errors.append(f"Dest {dest}: Path broken")
                continue

            path.reverse()  # Source -> Dest

            # 验证顺序
            vnf_ptr = 0
            for node in path:
                if vnf_ptr >= len(required_vnfs): break
                deployed = node_vnf_dict.get(node, [])
                while vnf_ptr < len(required_vnfs):
                    if required_vnfs[vnf_ptr] in deployed:
                        vnf_ptr += 1
                    else:
                        break

            if vnf_ptr < len(required_vnfs):
                errors.append(f"Dest {dest}: Missing VNFs starting from {required_vnfs[vnf_ptr]}")

        return (len(errors) == 0), errors
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
                self.time_step = self.current_slot_index * self.delta_t

                print(
                    f"⏩ [时间推进] Slot {start_slot} -> {self.current_slot_index} | 加载 {len(self.slot_queue)} 个请求 | Time: {self.time_step:.2f}s")

                # 🔥 触发资源回收 (清理上一个时间槽的过期资源)
                self._manual_release_resources()

                # 🔥 关键：准备好下一个槽的索引 (供下一次调用使用)
                self.current_slot_index += 1
                return

            # C. 当前槽为空，继续寻找下一个
            # print(f"   Slot {self.current_slot_index} 无请求，跳过...")
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
            # print(f"\n⏰ [Online TS] {old_time_slot} -> {new_time_slot}")

            # A. 更新时间
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

            # B. 触发资源释放
            if hasattr(self, 'request_manager'):
                self.request_manager.check_and_release_expired(self.time_step)
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
    def _get_last_vnf_node_safe(self):
        """
        安全地获取最后一个VNF部署节点

        Returns:
            int or None: 最后VNF节点，如果没有则返回None
        """
        if not self.current_request:
            return None

        placement = self.current_tree.get('placement', {})
        if not placement:
            return None

        vnf_chain = self.current_request.get('vnf', [])

        # 从后往前找已部署的VNF
        for i in range(len(vnf_chain) - 1, -1, -1):
            vnf_type = vnf_chain[i]

            # 检查复合键 (node, vnf_type)
            for key, value in placement.items():
                if isinstance(key, tuple) and len(key) == 2:
                    node, vtype = key
                    if vtype == vnf_type:
                        return node
                elif isinstance(key, int):
                    # 简单键：vnf_idx -> node
                    if key == i:
                        return value

        return None

    def _commit_resources(self, pruned_tree, valid_nodes):
        """💳 [统一算账] 两阶段提交资源"""
        req = self.current_request
        bw_req = req.get('bw_origin', 1.0)

        pending_links = []
        pending_nodes = []

        # Phase 1: Check
        for (u, v) in pruned_tree.keys():
            if hasattr(self.resource_mgr, 'check_link_resource'):
                if not self.resource_mgr.check_link_resource(u, v, bw_req):
                    return False
            pending_links.append((u, v, bw_req))

        placement = self.current_tree.get('placement', {})
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v_type = key[0], key[1]
                if n in valid_nodes:  # 只提交有效节点
                    c = info.get('cpu_used', 0)
                    m = info.get('mem_used', 0)
                    # check node resource...
                    pending_nodes.append((n, v_type, c, m))

        # Phase 2: Allocate
        self.curr_ep_link_allocs = []
        self.curr_ep_node_allocs = []

        for u, v, bw in pending_links:
            self.resource_mgr.allocate_link_resource(u, v, bw)
            self.curr_ep_link_allocs.append((u, v, bw))

        for n, v, c, m in pending_nodes:
            self.resource_mgr.allocate_node_resource(n, v, c, m)
            self.curr_ep_node_allocs.append((n, v, c, m))

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
    def _pack_info_fields(self):
        """辅助函数：打包所有 step 必须返回的标准字段"""
        return {
            'time_slot': self.current_time_slot if self.online_mode else 0,
            'decision_steps': self.decision_step,  # 🔥 核心修复：确保这个值是最新的
            'action_mask': self.get_low_level_action_mask()
        }

    def render_tree_plot(self, save_path=None):
        """
        🎨 [可视化 V3] 逻辑重建版 - 彻底消除环路和废边
        只绘制连接 Source -> VNFs -> Destinations 的有效骨干路径
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            return

        if not self.current_request or 'tree' not in self.current_tree:
            return

        req_id = self.current_request.get('id', '?')
        src = self.current_request.get('source')
        dests = set(self.current_request.get('dest', []))
        placement = self.current_tree.get('placement', {})
        raw_edges = self.current_tree.get('tree', {})

        # --- 1. 构建全量底图 (Agent 探索过的所有路) ---
        Full_G = nx.Graph()
        for edge_key in raw_edges.keys():
            if isinstance(edge_key, tuple) and len(edge_key) == 2:
                Full_G.add_edge(edge_key[0], edge_key[1])

        # --- 2. 提取 VNF 顺序序列 ---
        # 格式: [(idx, type, node), ...]
        vnf_sequence = []
        for key, info in placement.items():
            if isinstance(info, dict):
                vnf_sequence.append(info)
        # 按 vnf_idx 排序
        vnf_sequence.sort(key=lambda x: x.get('vnf_idx', 0))

        vnf_nodes = [info['node'] for info in vnf_sequence]

        # --- 🔥 3. 核心：逻辑重建 (只保留有效路径) ---
        Clean_G = nx.Graph()
        Clean_G.add_node(src)

        # A. 串联 VNF 链 (Source -> V1 -> V2 ...)
        current_node = src
        path_nodes_set = {src}

        # 如果有 VNF，先连 VNF
        targets = vnf_nodes

        for target in targets:
            try:
                if target in Full_G.nodes and current_node in Full_G.nodes:
                    # 在探索过的底图中找路
                    path = nx.shortest_path(Full_G, source=current_node, target=target)
                    nx.add_path(Clean_G, path)
                    path_nodes_set.update(path)
                    current_node = target
            except nx.NetworkXNoPath:
                print(f"⚠️ 绘图警告: 断路 {current_node} -> {target}")
                pass

        # B. 发散到目的地 (Last VNF -> Dest)
        # 注意：多播是从树的任意点分叉，但为了简化且保证连通，
        # 我们从"最后一个VNF节点"或者"当前已构建树中最近的节点"连向目的地

        # 这里使用简化逻辑：从最后一个 VNF (或源) 连向所有 Dest
        fork_point = current_node

        for dest in dests:
            try:
                if dest in Full_G.nodes:
                    # 尝试从 fork_point 连向 dest
                    # 更高级的做法是：从 Clean_G 中的任意点连向 dest (Steiner Tree 近似)
                    # 这里为了视觉整洁，我们直接找 path
                    path = nx.shortest_path(Full_G, source=fork_point, target=dest)
                    nx.add_path(Clean_G, path)
            except:
                pass

        # 如果重建失败（比如图不连通），回退到显示全图
        if Clean_G.number_of_edges() == 0:
            print("⚠️ 重建树为空，显示原始探索图")
            Clean_G = Full_G

        # --- 4. 绘图 (样式美化) ---
        plt.figure(figsize=(12, 8), dpi=120)

        # 使用分层布局或 Kamada Kawai
        try:
            # 尝试把 Source 放在最左/最上
            pos = nx.kamada_kawai_layout(Clean_G)
        except:
            pos = nx.spring_layout(Clean_G)

        # 绘制边
        nx.draw_networkx_edges(Clean_G, pos, width=3.0, edge_color='#666666', alpha=0.8)

        # 绘制中间节点
        others = [n for n in Clean_G.nodes if n != src and n not in dests]
        nx.draw_networkx_nodes(Clean_G, pos, nodelist=others, node_shape='o',
                               node_color='white', edgecolors='#333333', node_size=600)

        # 绘制目的节点
        valid_dests = [d for d in dests if d in Clean_G.nodes]
        nx.draw_networkx_nodes(Clean_G, pos, nodelist=valid_dests, node_shape='s',
                               node_color='#FFEEE0', edgecolors='red', node_size=800, label='Dest')

        # 绘制源节点
        if src in Clean_G.nodes:
            nx.draw_networkx_nodes(Clean_G, pos, nodelist=[src], node_shape='^',
                                   node_color='#E0EEFF', edgecolors='blue', node_size=1000, label='Source')

        # 标签
        nx.draw_networkx_labels(Clean_G, pos, font_size=10, font_weight='bold')

        # --- 5. VNF 标注 ---
        node_vnfs = {}
        for info in vnf_sequence:
            n = info['node']
            v = info['vnf_type']
            if n in Clean_G.nodes:
                if n not in node_vnfs: node_vnfs[n] = []
                node_vnfs[n].append(v)

        for n, vnfs in node_vnfs.items():
            if n in pos:
                x, y = pos[n]
                # 偏移一点避免遮挡
                txt = "\n".join([f"VNF{v}" for v in vnfs])
                plt.text(x, y + 0.08, txt, fontsize=9, color='darkred', ha='center', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFCC', alpha=0.8))

        plt.title(f"Reconstructed Tree - Request {req_id}", fontsize=15)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            plt.pause(1.0)  # 稍微停顿
        plt.close()

    def _connect_destination(self, node):
        """
        🔥 执行目的地连接逻辑
        将当前节点标记为已连接，并记录到多播树中
        """
        if node not in self.current_request.get('dest', []):
            return False

        # 初始化连接集合（防止 key 缺失）
        if 'connected_dests' not in self.current_tree:
            self.current_tree['connected_dests'] = set()

        # 记录连接
        if node not in self.current_tree['connected_dests']:
            self.current_tree['connected_dests'].add(node)

            # 将该节点及其到达路径正式标记为“树上节点”
            self.nodes_on_tree.add(node)

            # 如果有 parent_map，可以回溯路径加入 nodes_on_tree
            # 这里简化处理，具体的路径维护逻辑应在 step_low_level 中已完成

            return True
        return False
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
    def _get_shortest_distance(self, source, target):
        """
        🔥 计算两节点间的最短距离（BFS）

        Args:
            source: 起始节点
            target: 目标节点

        Returns:
            int: 最短距离（跳数），如果不可达返回999999
        """
        if source == target:
            return 0

        # 使用拓扑管理器的邻接表
        try:
            if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'adj_list'):
                adj_list = self.topology_mgr.adj_list
            elif hasattr(self, 'resource_mgr') and hasattr(self.resource_mgr, 'get_neighbors'):
                # 如果没有adj_list，构建临时的
                adj_list = {}
                for node in range(self.n):
                    adj_list[node] = self.resource_mgr.get_neighbors(node)
            elif hasattr(self, 'adj_list'):
                adj_list = self.adj_list
            else:
                # 最后的备选：从拓扑矩阵构建
                adj_list = {}
                if hasattr(self, 'topology_mgr') and hasattr(self.topology_mgr, 'G'):
                    import networkx as nx
                    for node in range(self.n):
                        adj_list[node] = list(self.topology_mgr.G.neighbors(node))
                else:
                    return 999999
        except Exception as e:
            print(f"⚠️ [Distance] 获取邻接表失败: {e}")
            return 999999

        # BFS 搜索最短路径
        from collections import deque

        queue = deque([(source, 0)])
        visited = {source}

        while queue:
            current, dist = queue.popleft()

            if current == target:
                return dist

            for neighbor in adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # 不可达
        return 999999
    def _is_closer_to_target(self, current_node, next_node, target_node):
        """
        🔥 判断next_node是否比current_node更接近target_node

        Args:
            current_node: 当前位置
            next_node: 即将移动的位置
            target_node: 高层目标

        Returns:
            bool: True表示next_node更接近目标
        """
        if target_node is None:
            return False

        if next_node == target_node:
            return True

        if current_node == target_node:
            return False

        # 使用拓扑距离（BFS最短路径）
        current_dist = self._get_shortest_distance(current_node, target_node)
        next_dist = self._get_shortest_distance(next_node, target_node)

        return next_dist < current_dist

    def _get_path_to_node(self, source, target):
        """
        🔥 [新增] 获取从源点到目标节点的路径（基于当前树）

        Args:
            source: 源节点
            target: 目标节点

        Returns:
            list: 路径上的节点列表 [source, ..., target]，如果不可达返回空列表
        """
        if source == target:
            return [source]

        # 从当前树中提取路径
        tree_edges = self.current_tree.get('tree', {})

        if not tree_edges:
            # 如果树为空，只有源点
            return [source] if target == source else []

        # 构建邻接表
        adj = {}
        for edge_key in tree_edges.keys():
            n1, n2 = edge_key
            adj.setdefault(n1, []).append(n2)
            adj.setdefault(n2, []).append(n1)

        # BFS查找路径
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # 如果目标不在树上，返回空列表
        return []
#最终树减枝
    def _prune_redundant_branches_with_vnf(self):
        """
        ✂️ [修复版] 剪枝冗余分支（支持VNF节点）

        功能：
        1. 识别从源节点到所有目的地+VNF节点的必要路径。
        2. 删除多余的“死胡同”分支。
        3. 自动释放被剪掉的链路资源。
        4. 返回父节点映射表供 SFC 质检使用。

        返回：
        - pruned_tree: dict, 剪枝后的树边
        - valid_nodes: set, 有效节点集合
        - success: bool, 剪枝是否成功
        - parent_map: dict, 父节点映射 {child: parent} (用于后续质检)
        """
        # 1. 基础检查
        if not self.current_request:
            return {}, set(), False, None

        source = self.current_request.get('source')
        dests = set(self.current_request.get('dest', []))
        placement = self.current_tree.get('placement', {})
        raw_edges = self.current_tree.get('tree', {})

        if not raw_edges:
            # 没有边，也就没有父节点映射
            return {}, {source}, False, None

        print(f"\n✂️ [剪枝开始]")
        print(f"   源节点: {source}")
        print(f"   目的地: {list(dests)}")
        print(f"   原始边数: {len(raw_edges)}")

        # 2. 构建邻接表 (Adjacency List)
        from collections import defaultdict, deque

        adj = defaultdict(list)
        for (u, v) in raw_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # 3. BFS 构建父节点映射 (Parent Map)
        parent = {source: None}
        queue = deque([source])
        visited = {source}

        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        # 4. 识别关键节点 (Critical Nodes)
        # 关键节点 = 所有目的地 + 所有已部署VNF的节点
        vnf_nodes = set()
        for key in placement.keys():
            # 兼容 (node, vnf) 和 (node, vnf, branch)
            if isinstance(key, tuple) and len(key) >= 1:
                node_id = key[0]
                vnf_nodes.add(node_id)

        critical_nodes = dests | vnf_nodes

        print(f"   VNF节点: {list(vnf_nodes)}")
        print(f"   关键节点: {list(critical_nodes)}")

        # 5. 连通性检查
        # 如果有关键节点无法从源节点到达，说明树是断裂的
        unreachable = [n for n in critical_nodes if n not in visited]
        if unreachable:
            print(f"   ❌ 关键节点不可达: {unreachable}")
            return {}, set(), False, None

        # 6. 反向回溯标记有效边 (Backtracking)
        valid_edges = set()
        valid_nodes = {source}

        for target in critical_nodes:
            curr = target
            # 从目标回溯到源节点
            while curr != source and curr is not None:
                p = parent.get(curr)
                if p is None:
                    break  # 异常断开

                edge = tuple(sorted([p, curr]))

                if edge in valid_edges:
                    break  # 这条路径已经标记过了，不用重复走

                valid_edges.add(edge)
                valid_nodes.add(curr)
                valid_nodes.add(p)
                curr = p

        # 7. 生成剪枝后的树
        pruned_tree = {}
        for edge in valid_edges:
            if edge in raw_edges:
                pruned_tree[edge] = raw_edges[edge]

        removed_count = len(raw_edges) - len(pruned_tree)

        print(f"\n✂️ [剪枝完成]")
        print(f"   剔除边: {removed_count} 条")
        print(f"   保留边: {len(pruned_tree)} 条")
        print(f"   有效节点: {len(valid_nodes)} 个")

        # 8. 释放被剔除的链路资源 (Resource Release)
        if removed_count > 0:
            required_bw = self.current_request.get('bw_origin', 1.0)
            released_count = 0

            for edge_key in raw_edges.keys():
                if edge_key not in pruned_tree:
                    u, v = edge_key
                    # 释放链路带宽
                    if hasattr(self, 'resource_mgr') and hasattr(self.resource_mgr, 'release_link_resource'):
                        self.resource_mgr.release_link_resource(u, v, required_bw)
                    released_count += 1

            print(f"♻️  [资源释放] {released_count} 条边，带宽 {released_count * required_bw:.1f}")

        # 🔥🔥🔥 修复核心：必须返回 4 个值，包括 parent 字典 🔥🔥🔥
        return pruned_tree, valid_nodes, True, parent
    def _validate_pruned_tree(self, pruned_tree, source, critical_nodes):
        """
        验证剪枝后的树是否满足基本要求

        检查项：
        1. 所有关键节点都在树中
        2. 从源节点可达所有关键节点
        """
        if not pruned_tree:
            return False

        # 收集树中所有节点
        tree_nodes = {source}
        for (u, v) in pruned_tree.keys():
            tree_nodes.add(u)
            tree_nodes.add(v)

        # 检查关键节点是否都在树中
        missing = []
        for node in critical_nodes:
            if node not in tree_nodes:
                missing.append(node)

        if missing:
            print(f"❌ [剪枝验证] 缺失关键节点: {missing}")
            return False

        # 检查连通性（BFS从源节点）
        from collections import deque, defaultdict

        adj = defaultdict(list)
        for (u, v) in pruned_tree.keys():
            adj[u].append(v)
            adj[v].append(u)

        visited = {source}
        queue = deque([source])

        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 检查所有关键节点是否可达
        unreachable = []
        for node in critical_nodes:
            if node not in visited:
                unreachable.append(node)

        if unreachable:
            print(f"❌ [剪枝验证] 不可达关键节点: {unreachable}")
            return False

        return True
    def _release_redundant_resources(self, raw_edges, pruned_tree):
        """
        释放剪枝掉的冗余边占用的资源

        Args:
            raw_edges: 原始树的边集
            pruned_tree: 剪枝后的树
        """
        req = self.current_request
        if not req:
            return

        required_bw = req.get('bw_origin', 1.0)
        released_count = 0
        released_bw = 0.0

        # 找出被剪枝的边
        for edge_key in raw_edges.keys():
            if edge_key not in pruned_tree:
                u, v = edge_key

                # 释放链路资源
                if hasattr(self.resource_mgr, 'release_link_resource'):
                    try:
                        self.resource_mgr.release_link_resource(u, v, required_bw)
                        released_count += 1
                        released_bw += required_bw
                    except Exception as e:
                        print(f"⚠️ 释放边 {edge_key} 资源失败: {e}")

        if released_count > 0:
            print(f"♻️  [资源释放] {released_count} 条边，带宽 {released_bw:.1f}")
    def _finalize_request_with_pruning(self):
        """
        🔥 [V12.2 兼容结算版] 原子性物理结算
        """
        if self.current_request is None:
            return False

        req_id = self.current_request.get('id', 'unknown')
        pruned_tree, valid_nodes, success, _ = self._prune_redundant_branches_with_vnf()
        if not success:
            return False

        self.current_tree['tree'] = pruned_tree
        placement = self.current_tree.get('placement', {})
        temp_node_allocs = []

        # ------------------------------------------------------------
        # 步骤: 节点资源原子分配
        # ------------------------------------------------------------
        for key, info in placement.items():
            node = key[0]
            vnf_type = key[1]

            # 🔥 修复核心：从 info 字典中读取资源需求
            if isinstance(info, dict):
                c = info.get('cpu_used', 1.0)
                m = info.get('mem_used', 1.0)
            else:
                # 兜底：如果 info 只是个索引整数
                idx = info
                cpu_needs = self.current_request.get('cpu_origin', [])
                mem_needs = self.current_request.get('memory_origin', [])
                c = cpu_needs[idx] if idx < len(cpu_needs) else 1.0
                m = mem_needs[idx] if idx < len(mem_needs) else 1.0

            # 只有在有效节点（剪枝后保留）上才分配资源
            if node in valid_nodes:
                if self.resource_mgr.allocate_node_resource(node, vnf_type, c, m):
                    temp_node_allocs.append((node, vnf_type, c, m))
                else:
                    # 任何一项失败则全体回滚
                    for r_node, r_vt, r_c, r_m in temp_node_allocs:
                        self.resource_mgr.release_node_resource(r_node, r_vt, r_c, r_m)
                    return False

        # ------------------------------------------------------------
        # 步骤: 链路资源原子分配 (保持原有逻辑)
        # ------------------------------------------------------------
        bw = self.current_request.get('bw_origin', 1.0)
        temp_link_allocs = []
        for (u, v) in pruned_tree.keys():
            if self.resource_mgr.allocate_link_resource(u, v, bw):
                temp_link_allocs.append((u, v, bw))
            else:
                for r_u, r_v, r_bw in temp_link_allocs:
                    self.resource_mgr.release_link_resource(r_u, r_v, r_bw)
                for r_node, r_vt, r_c, r_m in temp_node_allocs:
                    self.resource_mgr.release_node_resource(r_node, r_vt, r_c, r_m)
                return False

        self.curr_ep_node_allocs = temp_node_allocs
        self.curr_ep_link_allocs = temp_link_allocs
        return True