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

# 导入自定义模块
from envs.modules.AllResourceManager import FusedResourceManager as ResourceManager
from envs.modules.data_loader import DataLoader
from envs.modules.event_handler import EventHandler
from envs.modules.MABPruner import MABPruningHelper
from envs.modules.tools import SFCToolkit
from envs.modules.low_level_controller import LowLevelController
from envs.modules.high_level_controller import HighLevelController
from envs.modules.TimeSlotManager import TimeSlotManager
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

        # 专家系统、备份策略、路径管理
        self._init_core_modules()
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
        self.max_subgoal_steps = config.get('max_low_steps', 25)
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
        #工具箱初始化
        self.tools = SFCToolkit(self)
        logger.info("✅ 工具箱 已初始化")

        # TimeSlotManager 初始化（接管时钟推进和请求获取）
        self.time_slot_mgr = TimeSlotManager(self, self.config)
        logger.info("✅ TimeSlotManager 已初始化")
        self.low_level_controller = LowLevelController(self)
        logger.info("✅ LowLevelController 已初始化")
        self.high_level_controller = HighLevelController(self)

        logger.info("✅ HighLevelController 已初始化")
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
        self.resource_mgr.env = self          # [P1] online_mode 判断依赖
        self.request_manager = self.resource_mgr.request_manager  # [P1] lifecycle 入口
        self.topology_mgr = SimpleTopologyManager(self.topo)

        logger.info(f"✅ 环境参数: n={self.n}, L={self.L}, K_vnf={self.K_vnf}")
    def _init_core_modules(self):
        """初始化专家系统、备份策略和路径管理器"""

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

        # [P4] 同步到 TimeSlotManager（必须在数据加载后）
        if hasattr(self, 'time_slot_mgr') and self.time_slot_mgr is not None:
            self.time_slot_mgr.load(requests)

        # 同步到 TimeSlotManager
        if hasattr(self, 'time_slot_mgr') and self.time_slot_mgr is not None:
            self.time_slot_mgr.load(requests)
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
            # reset() 内部会自动判断 online_mode：
            #   online → episode_reset (保留lifecycle)
            #   offline/hard → 完整重置
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
            req_raw = self.time_slot_mgr.get_next_request()
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

        # 3. 检查是否越界，循环复用时叠加时间偏移保证时钟单调递增
        if self.global_request_index >= len(self.all_requests):
            # 🔥 关键修复：数据集用完后，先强制释放所有活跃请求，再循环
            #    否则：时钟倒回 → expire_time 永远大于 current_time → 永不释放 → 资源耗尽
            if not hasattr(self, '_request_cycle_offset'):
                self._request_cycle_offset = 0.0
            # 计算本轮时间跨度：取数据集内最大过期时间，加 10% 余量确保全部自然过期
            if self.all_requests:
                cycle_end = max(
                    float(r.get('arrival_time', 0.0)) + float(r.get('lifetime', 5.0))
                    for r in self.all_requests
                ) * 1.10
            else:
                cycle_end = 600.0
            self._request_cycle_offset += cycle_end
            logger.info(f"[ResetReq] 数据集循环，时间偏移累计 +{cycle_end:.1f}s → "
                        f"总偏移={self._request_cycle_offset:.1f}s，触发全量过期检查")
            # 用偏移后的时间触发一次全量过期检查，确保上一轮请求全部释放
            if hasattr(self.resource_mgr, 'request_manager'):
                self.resource_mgr.request_manager.check_and_release_expired(
                    self._request_cycle_offset
                )
            self.global_request_index = 0

        # 4. 获取请求（shallow copy 防止污染原始数据集）
        req = dict(self.all_requests[self.global_request_index])

        # 5. 🔥 时间切片处理与资源释放
        # 叠加循环偏移，使 arrival_time / expire_time 在全局时间轴上单调递增
        offset = getattr(self, '_request_cycle_offset', 0.0)
        if offset > 0:
            raw_arrival  = float(req.get('arrival_time', 0.0))
            raw_lifetime = float(req.get('lifetime', 5.0))
            req['arrival_time'] = raw_arrival + offset
            req['expire_time']  = raw_arrival + offset + raw_lifetime
            # time_slot 不需要偏移（只用于槽切换检测，相对值即可）
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
            if hasattr(self.resource_mgr, 'request_manager'):
                self.resource_mgr.request_manager.check_and_release_expired(self.time_step)

        else:
            # 同槽内也要更新时间
            self.time_step = new_arrival_time
            self.current_time_slot = new_time_slot

        # 6. 移动指针
        self.global_request_index += 1

        # 7. 返回
        obs = self.get_state()
        return req, obs
    # 高层
    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
            """
            🎯 [委托] 设定高层目标 (修复版：支持起点同步)
            """
            # 将 start_node_id 透传给 HighLevelController
            return self.high_level_controller.set_high_level_goal(
                high_action_idx,
                target_node_id,
                start_node_id=start_node_id
            )
    def step_high_level(self, action):
        """
        🎯 [委托] 执行高层步骤
        """
        return self.high_level_controller.step_high_level(action)

    def get_high_level_action_mask(self):
        """
        🎯 [委托] 获取高层动作掩码
        """
        return self.high_level_controller.get_high_level_action_mask()

    def get_high_level_state_graph(self):
        """
        🎯 [委托] 获取高层图状态
        """
        return self.high_level_controller.get_high_level_state_graph()
    # 低层
    def step_low_level(self, action):
        """
        🔥 [V40.0 完全委托版] 低层步进函数

        所有逻辑委托给 LowLevelController 执行，包括：
        1. 移动和部署逻辑
        2. Episode完成时的归档
        3. Episode完成时的资源释放
        """
        return self.low_level_controller.step_low_level(action)
    def get_low_level_action_mask(self):
        """获取低层动作掩码 - 委托给 LowLevelController"""
        if hasattr(self, 'low_level_controller'):
            return self.low_level_controller.get_low_level_action_mask()
        else:
            # 兜底方案
            import numpy as np
            return np.ones(self.n, dtype=np.float32)
    def get_state(self):
        """获取环境状态 - 委托给 LowLevelController"""
        if hasattr(self, 'low_level_controller'):
            return self.low_level_controller.get_state()
        else:
            # 兜底方案
            import torch
            from torch_geometric.data import Data
            return Data(x=torch.zeros((self.n, 17)))


    #资源探测
    def get_resource_utilization(self):
        """资源利用率 — 基于pool实际数据"""
        try:
            total_used = 0.0
            for i in range(self.n):
                avail = self.resource_mgr.pool.get_available_cpu(i)
                total_used += (100.0 - avail)
            return total_used / (self.n * 100.0)
        except Exception:
            return 0.0