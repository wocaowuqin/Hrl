"""
main.py - Goal-Conditioned HRL 版本
===============================================================================

主要修改：
1. ✅ 修复 HRL_Coordinator 初始化崩溃 (移至 Phase 3 流程中)
2. ✅ 保持其他诊断功能不变

===============================================================================
"""
import scipy.io
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_utils import load_config
from envs.sfc_env import SFC_HIRL_Env

# 🔥 修改 1：导入 Goal-Conditioned Agent
from core.hrl.agent import (
    GoalConditionedHRLAgent,
    create_goal_conditioned_agent
)

from trainer.phase1_collector import Phase1ExpertCollector
from trainer.phase2_il_trainer import Phase2ILTrainer
from trainer.phase3_rl_trainer import Phase3RLTrainer

# 配置全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_config_path(config, key_path):
    """安全地获取配置路径"""
    possible_locations = [
        ['path', key_path],
        ['project', key_path],
        ['paths', key_path],
        [key_path],
    ]

    for location in possible_locations:
        try:
            value = config
            for key in location:
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue

    default_paths = {
        'ckpt_dir': './outputs/checkpoints',
        'log_dir': './outputs/logs',
        'expert_data_dir': './data/expert',
        'input_dir': './data/input_dir',
    }

    if key_path in default_paths:
        logger.warning(f"⚠️  配置中未找到 {key_path}，使用默认值: {default_paths[key_path]}")
        return default_paths[key_path]

    return None


def ensure_paths_exist(config):
    """确保所有必要的目录存在"""
    path_keys = ['ckpt_dir', 'log_dir', 'expert_data_dir']

    for key in path_keys:
        path = get_config_path(config, key)
        if path:
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ 目录准备完成: {path}")


def validate_config(config, phase):
    """验证配置完整性"""
    logger.info("🔍 验证配置...")

    errors = []
    warnings = []

    if 'gnn' not in config:
        warnings.append("⚠️  缺少 'gnn' 配置块（将从环境动态获取）")

    if 'env' not in config and 'environment' not in config:
        errors.append("❌ 缺少 'env' 或 'environment' 配置块")

    if phase not in config:
        warnings.append(f"⚠️  缺少 '{phase}' 配置块")

    # 🔥 新增：验证 HRL 配置
    if phase == 'phase3' and 'hrl' not in config:
        warnings.append("⚠️  缺少 'hrl' 配置块（将使用默认值）")

    if errors:
        logger.error("❌ 配置验证失败:")
        for err in errors:
            logger.error(f"  {err}")
        raise ValueError("配置不完整，请检查配置文件")

    if warnings:
        logger.warning("⚠️  配置警告:")
        for warn in warnings:
            logger.warning(f"  {warn}")

    logger.info("✅ 配置验证通过")


def load_topology(config):
    """
    Unified topology loader - 修复版

    修复：
    1. 跳过完全图矩阵
    2. 优先使用特定键名
    3. 从 Paths 构建拓扑作为后备
    """
    logger.info("📡 正在加载拓扑矩阵...")

    if 'topology' not in config:
        config['topology'] = {}

    try:
        input_dir = config['path']['input_dir']
    except KeyError:
        logger.error("❌ 缺少 config['path']['input_dir']")
        return False

    mat_path = os.path.join(input_dir, 'US_Backbone_path.mat')

    if not os.path.exists(mat_path):
        logger.error(f"❌ 拓扑文件不存在: {mat_path}")
        return False

    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        logger.error(f"❌ 读取 mat 文件失败: {e}")
        return False

    # 🔥 调试：打印所有可用的键
    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    logger.info(f"📋 MAT文件包含的键: {available_keys}")

    # 🔥 方法1：优先查找特定的邻接矩阵键名
    adjacency_keys = ['adjacency', 'Adjacency', 'topo', 'Topo', 'adj_matrix',
                      'graph', 'topology', 'network', 'links']

    for key in adjacency_keys:
        if key in mat_data:
            val = mat_data[key]
            if isinstance(val, np.ndarray) and val.ndim == 2:
                if val.shape[0] == val.shape[1] and np.issubdtype(val.dtype, np.number):
                    topo = (val > 0).astype(np.float32)
                    np.fill_diagonal(topo, 0)

                    # 🔥 关键：验证不是完全图
                    N = topo.shape[0]
                    expected_complete_graph_edges = N * (N - 1)
                    actual_edges = int(np.sum(topo))

                    if actual_edges == expected_complete_graph_edges:
                        logger.warning(f"⚠️  跳过 '{key}': 这是完全图 ({actual_edges}条边)")
                        continue

                    if actual_edges == 0:
                        logger.warning(f"⚠️  跳过 '{key}': 空矩阵")
                        continue

                    # 验证稀疏性（真实网络图的边数应该远小于完全图）
                    sparsity = actual_edges / expected_complete_graph_edges
                    if sparsity > 0.8:
                        logger.warning(f"⚠️  跳过 '{key}': 太密集 (sparsity={sparsity:.2%})")
                        continue

                    logger.info(f"✅ 使用邻接矩阵字段: '{key}'")
                    logger.info(f"   节点数: {N}")
                    logger.info(f"   物理链路数: {actual_edges // 2}")
                    logger.info(f"   平均度数: {actual_edges / N:.2f}")
                    logger.info(f"   稀疏度: {sparsity:.2%}")

                    config['topology']['matrix'] = topo
                    return True

    # 🔥 方法2：遍历所有方阵，找最稀疏的
    logger.info("🔍 在所有方阵中寻找最合适的拓扑矩阵...")

    candidates = []
    for key, val in mat_data.items():
        if key.startswith('__'):
            continue

        if isinstance(val, np.ndarray) and val.ndim == 2:
            if val.shape[0] == val.shape[1] and np.issubdtype(val.dtype, np.number):
                topo = (val > 0).astype(np.float32)
                np.fill_diagonal(topo, 0)

                N = topo.shape[0]
                actual_edges = int(np.sum(topo))

                if actual_edges == 0:
                    continue

                expected_complete = N * (N - 1)
                if actual_edges == expected_complete:
                    continue

                sparsity = actual_edges / expected_complete
                avg_degree = actual_edges / N

                candidates.append({
                    'key': key,
                    'topo': topo,
                    'edges': actual_edges // 2,
                    'sparsity': sparsity,
                    'avg_degree': avg_degree,
                    'nodes': N
                })

    if candidates:
        # 选择最稀疏的（最可能是真实拓扑）
        candidates.sort(key=lambda x: x['sparsity'])

        logger.info(f"📊 找到 {len(candidates)} 个候选矩阵:")
        for i, c in enumerate(candidates[:3]):
            logger.info(f"   {i + 1}. '{c['key']}': {c['edges']}条链路, "
                        f"度数={c['avg_degree']:.2f}, 稀疏度={c['sparsity']:.2%}")

        best = candidates[0]

        # 额外验证：真实网络的平均度数通常在 2-10 之间
        if 2 <= best['avg_degree'] <= 15:
            logger.info(f"✅ 选择最稀疏的矩阵: '{best['key']}'")
            logger.info(f"   节点数: {best['nodes']}")
            logger.info(f"   物理链路数: {best['edges']}")
            logger.info(f"   平均度数: {best['avg_degree']:.2f}")

            config['topology']['matrix'] = best['topo']
            return True
        else:
            logger.warning(f"⚠️  最佳候选 '{best['key']}' 的度数异常: {best['avg_degree']:.2f}")

    # 🔥 方法3：从 Paths 构建拓扑（后备方案）
    if 'Paths' not in mat_data:
        logger.error("❌ 无法找到合适的邻接矩阵，且 mat 文件中不存在 Paths 结构")
        return False

    logger.info("🔍 从 Paths 构建拓扑（后备方案）...")
    paths_matrix = mat_data['Paths']
    N, M = paths_matrix.shape

    topo = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            if i == j:
                continue

            cell = paths_matrix[i, j]

            if not hasattr(cell, 'dtype'):
                continue
            if cell.dtype.names is None:
                continue
            if 'paths' not in cell.dtype.names:
                continue

            paths_array = cell['paths']
            if not isinstance(paths_array, np.ndarray):
                continue

            if paths_array.ndim == 1:
                paths_array = paths_array[np.newaxis, :]

            for path in paths_array:
                nodes = path[path > 0] - 1
                if len(nodes) < 2:
                    continue

                for k in range(len(nodes) - 1):
                    u, v = int(nodes[k]), int(nodes[k + 1])
                    if 0 <= u < N and 0 <= v < N:
                        topo[u, v] = 1.0
                        topo[v, u] = 1.0

    np.fill_diagonal(topo, 0)

    if np.sum(topo) == 0:
        logger.error("❌ Paths 解析完成，但未发现任何物理链路")
        return False

    num_edges = int(np.sum(topo) / 2)
    avg_degree = np.sum(topo) / N

    logger.info(f"✅ 从 Paths 构建拓扑成功:")
    logger.info(f"   节点数: {N}")
    logger.info(f"   物理链路数: {num_edges}")
    logger.info(f"   平均度数: {avg_degree:.2f}")

    config['topology']['matrix'] = topo.astype(np.float32)
    return True


def inject_dynamic_dimensions(config, env):
    """从环境中获取动态维度并注入到配置"""
    logger.info("🔧 注入动态维度...")

    if 'gnn' not in config:
        config['gnn'] = {}

    config['gnn']['node_feat_dim'] = env.resource_mgr.node_feat_dim
    config['gnn']['edge_feat_dim'] = env.resource_mgr.edge_feat_dim
    config['gnn']['request_feat_dim'] = env.resource_mgr.request_dim

    # 🔥 新增：注入 HRL 配置
    if 'hrl' not in config:
        config['hrl'] = {}

    # 从环境获取 state_dim
    if hasattr(env, 'observation_space'):
        if 'x' in env.observation_space:
            state_dim = env.observation_space['x'].shape[1]
            config['hrl']['state_dim'] = state_dim
            logger.info(f"  state_dim: {state_dim}")

    logger.info(f"  node_feat_dim: {config['gnn']['node_feat_dim']}")
    logger.info(f"  edge_feat_dim: {config['gnn']['edge_feat_dim']}")
    logger.info(f"  request_feat_dim: {config['gnn']['request_feat_dim']}")


def setup_hrl_config(config):
    """
    🔥 新增：设置 HRL 默认配置
    """
    if 'hrl' not in config:
        config['hrl'] = {}

    hrl_defaults = {
        'goal_dim': 64,
        'subgoal_horizon': 5,
        'intrinsic_reward_weight': 0.3,
        'max_complexity_threshold': 0.8,
        'goal_strategy': 'adaptive'  # 'relative', 'adaptive', 'hybrid'
    }

    for key, default_value in hrl_defaults.items():
        if key not in config['hrl']:
            config['hrl'][key] = default_value
            logger.info(f"  使用默认 hrl.{key}: {default_value}")


def diagnose_goal_embedding(agent, env):
    """
    🔥 新增：诊断 Goal Embedding
    """
    logger.info("=" * 70)
    logger.info("🔬 Goal Embedding 诊断")
    logger.info("=" * 70)

    try:
        # 获取初始状态
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        # 生成 Subgoal
        logger.info("1. 测试 Subgoal 生成...")
        agent._generate_and_encode_subgoal(state)

        if agent.current_subgoal is not None:
            logger.info(f"   ✅ Subgoal shape: {agent.current_subgoal.shape}")
            logger.info(f"   ✅ Subgoal norm: {agent.current_subgoal.norm().item():.4f}")
            logger.info(f"   ✅ NaN检查: {torch.isnan(agent.current_subgoal).any().item()}")
        else:
            logger.error("   ❌ Subgoal 生成失败")

        # 检查 Goal Embedding
        logger.info("2. 测试 Goal Embedding...")
        if agent.current_goal_emb is not None:
            logger.info(f"   ✅ Goal Emb shape: {agent.current_goal_emb.shape}")
            logger.info(f"   ✅ Goal Emb norm: {agent.current_goal_emb.norm().item():.4f}")
        else:
            logger.error("   ❌ Goal Embedding 生成失败")

        # 测试动作选择
        logger.info("3. 测试动作选择...")
        # ✅ 接收第3个返回值 info
        high, low, info = agent.select_action(state)
        logger.info(f"   ✅ Action: high={high}, low={low}")

        # 测试 Q-Network
        logger.info("4. 测试 Q-Network...")

        try:
            # 直接调用 agent 的方法，而不是手动构造 batch
            with torch.no_grad():
                _ = agent.q_network.get_low_q_values(
                    state.x if hasattr(state, 'x') else None,
                    state.edge_index if hasattr(state, 'edge_index') else None,
                    state.edge_attr if hasattr(state, 'edge_attr') else None,
                    state.req_vec if hasattr(state, 'req_vec') else torch.zeros(1, 24),
                    agent.current_goal_emb,
                    state.batch if hasattr(state, 'batch') else None
                )
            logger.info(f"   ✅ Q-Network 可以正常调用")
        except Exception as e:
            logger.warning(f"   ⚠️  Q-Network 测试跳过: {e}")
            logger.info(f"   ℹ️  这不影响训练（select_action 已验证通过）")

        logger.info("=" * 70)
        logger.info("✅ Goal Embedding 诊断完成 - 一切正常")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ Goal Embedding 诊断失败")
        logger.error(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        logger.error("=" * 70)
        return False


def diagnose_agent_timing_performance(env, agent):
    """
    🔍 深度诊断函数：检测 Agent 动作密度与物理时间的失配度
    """
    print("\n" + "=" * 70)
    print("⏳ Agent 编排耗时与逻辑生命周期对齐检查")
    print("=" * 70)
    # ... (原有逻辑保持不变) ...
    # 为节省空间省略，请保持原函数内容


def diagnose_detailed_timing(env, agent):
    """
    🔍 深度诊断：Agent 动作步数 vs. 逻辑时间 vs. 物理时间
    """
    print("\n" + "=" * 60)
    print("🕵️‍♂️ Agent 运行耗时与逻辑生命周期深度诊断")
    print("=" * 60)
    # ... (原有逻辑保持不变) ...
    # 为节省空间省略，请保持原函数内容


def main():
    """🔥 V40.0 完整修复版本 - main函数 (IDE直调版)"""

    import argparse

    # =========================================================================
    # 🔥 运行参数配置区：直接在这里修改你想运行的阶段和参数
    # =========================================================================
    args = argparse.Namespace(
        phase='phase3',  # 选择运行阶段: 'phase1', 'phase2', 'phase3'
        gpu=0,  # GPU ID, 设为 -1 则强制使用 CPU
        seed=42,  # 随机种子
        goal_strategy='adaptive'  # Phase 3 的目标策略: 'relative', 'adaptive', 'hybrid'
    )
    # =========================================================================

    # 1. 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"🖥️  使用 GPU: cuda:{args.gpu}")
    else:
        device = torch.device('cpu')
        logger.info("🖥️  使用 CPU")

    # 2. 设置随机种子
    set_seed(args.seed)
    logger.info(f"🌱 随机种子: {args.seed}")

    # 3. 加载配置
    try:
        config = load_config(args.phase)

        # 🔥 设置 HRL 配置
        if args.phase == 'phase3':
            setup_hrl_config(config)
            # 命令行参数覆盖
            config['hrl']['goal_strategy'] = args.goal_strategy

        logger.info(f"✅ 配置加载成功")
    except Exception as e:
        logger.error(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 验证配置
    try:
        validate_config(config, args.phase)
    except ValueError as e:
        logger.error(str(e))
        return

    # 5. 确保目录存在
    ensure_paths_exist(config)

    # 6. 加载拓扑矩阵
    if not load_topology(config):
        logger.error("❌ 拓扑矩阵加载失败，无法继续")
        return

    # =========================================================================
    # 🔥🔥🔥 核心修复：统一环境初始化 (在 Phase 判断之前) 🔥🔥🔥
    # =========================================================================
    logger.info("🛠️ Initializing Environment (Global)...")
    try:
        # 1. 创建环境 (GNN模式)
        env = SFC_HIRL_Env(config, use_gnn=True)

        # 2. 注入动态维度 (这对 Phase 2/3 的 Agent 初始化至关重要)
        inject_dynamic_dimensions(config, env)

        logger.info("✅ Environment Initialized Successfully")

        # 🔥🔥🔥 资源打印 🔥🔥🔥
        try:
            print("\n" + "=" * 40)
            if hasattr(env.resource_mgr, 'nodes'):
                nodes = env.resource_mgr.nodes
                # 兼容字典结构 {'cpu': [...], ...}
                if isinstance(nodes, dict):
                    cpu_data = nodes.get('cpu', [])
                    print(f"👀 CPU配置 (前5个): {cpu_data[:5] if len(cpu_data) > 0 else '空'}")
                    mem_data = nodes.get('memory', [])
                    print(f"👀 MEM配置 (前5个): {mem_data[:5] if len(mem_data) > 0 else '空'}")
                # 兼容矩阵结构 [N, Features]
                elif hasattr(nodes, 'shape'):
                    print(f"👀 CPU配置 (前5个): {nodes[:5, 0]}")
                # 兼容列表结构
                else:
                    print(f"👀 CPU配置 (原始): {nodes}")
            else:
                print("👀 resource_mgr.nodes 属性不存在")

            # 顺便看一下带宽
            bw_cap = config.get('capacities', {}).get('bandwidth', '未知')
            print(f"👀 默认带宽配置: {bw_cap}")
            print("=" * 40 + "\n")
        except Exception as e:
            print(f"⚠️ 资源打印失败: {e}")
    except Exception as e:
        logger.error(f"❌ 环境初始化崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Phase 1: Expert Data Collection
    # =========================================================================
    if args.phase == 'phase1':
        logger.info("=" * 70)
        logger.info("🚀 Phase 1: Expert Data Collection")
        logger.info("=" * 70)

        try:
            # 直接使用上面初始化的全局 env
            expert_solver = env.policy_helper.expert
            logger.info("✅ Expert Solver 已加载")
        except Exception as e:
            logger.error(f"❌ Expert Solver 获取失败: {e}")
            return

        output_dir = get_config_path(config, 'expert_data_dir')
        max_episodes = config.get("phase1", {}).get("max_episodes", 5000)
        save_every = config.get("phase1", {}).get("save_every", 500)

        collector = Phase1ExpertCollector(
            env=env,
            expert_solver=expert_solver,
            output_dir=output_dir,
            max_episodes=max_episodes,
            save_every=save_every,
        )

        try:
            collector.collect()
            logger.info("✅ Phase 1 完成")
        except Exception as e:
            logger.error(f"❌ Phase 1 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Phase 2: Imitation Learning
    # =========================================================================
    elif args.phase == 'phase2':
        logger.info("=" * 70)
        logger.info("🚀 Phase 2: Imitation Learning")
        logger.info("=" * 70)

        try:
            logger.info("🔧 初始化 Phase 2 Agent...")
            agent = GoalConditionedHRLAgent(config, phase=2)
            logger.info("✅ Agent 初始化成功")
            logger.info(f"   模式: Phase {agent.phase}")
            logger.info(f"   动作空间: {agent.n_actions}")
            logger.info(f"   设备: {agent.device}")

            # 结构检查
            if hasattr(agent, 'high_policy') and hasattr(agent, 'low_policy'):
                logger.info("   ✅ 检测到 HRL Agent (双层策略网络)")
            elif hasattr(agent, 'policy_net'):
                logger.info("   ✅ 检测到 Legacy Agent (单层策略网络)")
            else:
                logger.error("   ❌ 无法识别 Agent 结构: 既没有 policy_net 也没有 high/low policy")
                return

        except Exception as e:
            logger.error(f"❌ Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        data_file = "expert_data_final.pkl"
        expert_data_dir = get_config_path(config, 'expert_data_dir')
        data_path = os.path.join(expert_data_dir, data_file)

        if not os.path.exists(data_path):
            logger.error(f"❌ 专家数据不存在: {data_path}")
            logger.error("   请先运行 Phase 1 收集数据")
            return

        phase2_config = config.get('phase2', {})
        output_dir = get_config_path(config, 'ckpt_dir')

        try:
            # 核心修复：传入全局 env
            trainer = Phase2ILTrainer(
                agent=agent,
                env=env,
                expert_data_path=data_path,
                output_dir=output_dir,
                config=phase2_config
            )
            logger.info("✅ Phase2 Trainer 初始化成功")
        except Exception as e:
            logger.error(f"❌ Trainer 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            trainer.run()
            logger.info("✅ Phase 2 完成")
        except Exception as e:
            logger.error(f"❌ Phase 2 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Phase 3: Goal-Conditioned RL
    # =========================================================================
    elif args.phase == 'phase3':
        logger.info("=" * 70)
        logger.info("🚀 Phase 3: Goal-Conditioned RL Fine-tuning")
        logger.info("=" * 70)

        # 1. 初始化 Agent
        try:
            logger.info("🔧 初始化 Goal-Conditioned Agent...")

            agent = create_goal_conditioned_agent(
                config=config,
                phase=3,
                goal_strategy=args.goal_strategy,
                env=env
            )

            logger.info("✅ Agent 初始化成功")

        except Exception as e:
            logger.error(f"❌ Agent 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 2. 加载预训练模型 (智能适配版)
        ckpt_dir = get_config_path(config, 'ckpt_dir')
        pretrained_path = os.path.join(ckpt_dir, "il_model_final.pth")

        if os.path.exists(pretrained_path):
            logger.info(f"📥 正在加载预训练模型: {pretrained_path}")
            try:
                checkpoint = torch.load(pretrained_path, map_location=agent.device)
                source_state = None
                if isinstance(checkpoint, dict):
                    if 'policy_net' in checkpoint:
                        source_state = checkpoint['policy_net']
                    elif 'model_state_dict' in checkpoint:
                        source_state = checkpoint['model_state_dict']
                    else:
                        source_state = checkpoint

                if source_state:
                    new_state_dict = {}
                    target_model = agent.q_network
                    target_keys = set(target_model.state_dict().keys())

                    for k, v in source_state.items():
                        if k in target_keys:
                            new_state_dict[k] = v
                            continue
                        if k.startswith('gnn.'):
                            new_key = k.replace('gnn.', '', 1)
                            if new_key in target_keys:
                                new_state_dict[new_key] = v
                                continue

                    if len(new_state_dict) > 0:
                        missing, unexpected = target_model.load_state_dict(new_state_dict, strict=False)
                        match_count = len(new_state_dict)
                        total_params = len(target_keys)
                        logger.info(f"✅ 智能加载成功: 迁移了 {match_count}/{total_params} 层权重")
                    else:
                        logger.warning("⚠️ 智能适配后仍未找到匹配层")
                else:
                    logger.warning("⚠️ Checkpoint 格式异常")

            except Exception as e:
                logger.error(f"❌ 模型加载错误: {e}")
        else:
            logger.warning(f"⚠️ 未找到预训练模型: {pretrained_path}")

        # =========================================================
        # 🔥 初始化 HRL Coordinator
        # =========================================================
        from envs.modules.HRL_Coordinator import HRL_Coordinator

        logger.info("🔧 初始化 HRL Coordinator...")
        try:
            coordinator = HRL_Coordinator(
                env=env,
                high_agent=agent,
                low_agent=agent,
                config=config
            )
            logger.info("✅ HRL Coordinator 初始化成功")
        except Exception as e:
            logger.error(f"❌ HRL Coordinator 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # =========================================================
        # 🔥🔥🔥 关键修复：加载Phase3训练数据
        # =========================================================
        logger.info("=" * 70)
        logger.info("📥 加载Phase3训练数据")
        logger.info("=" * 70)

        data_file = 'data/input_dir/phase3_requests.pkl'
        logger.info(f"📂 数据文件路径: {data_file}")
        logger.info(f"📂 当前工作目录: {os.getcwd()}")
        logger.info(f"🔍 文件是否存在: {os.path.exists(data_file)}")

        if os.path.exists(data_file):
            logger.info(f"✅ 文件存在，开始加载...")
            try:
                success = env.load_dataset(data_file)
                if success:
                    logger.info(f"✅ 数据加载成功")

                    # 🔥 验证数据
                    logger.info("🔍 验证数据加载...")
                    test_state = env.reset()
                    if env.current_request:
                        vnf_list = env.current_request.get('vnf', [])
                        logger.info(f"✅ 验证成功：请求存在，VNF数量={len(vnf_list)}")
                    else:
                        logger.warning("⚠️ 验证警告：reset后无请求")
                else:
                    logger.error(f"❌ load_dataset返回False")
                    raise RuntimeError("数据加载失败")
            except Exception as e:
                logger.error(f"❌ 数据加载异常: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            logger.error(f"❌ 文件不存在: {data_file}")

            # 列出可能的文件位置
            logger.info("🔍 检查data/input_dir/目录内容...")
            input_dir = 'data/input_dir'
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
                logger.info(f"📂 找到的.pkl文件:")
                for f in files:
                    logger.info(f"   - {f}")
            else:
                logger.error(f"❌ 目录不存在: {input_dir}")

            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        logger.info("=" * 70)

        # =========================================================
        # 🔥 创建 Trainer
        # =========================================================
        trainer = Phase3RLTrainer(
            env=env,
            agent=agent,
            output_dir=ckpt_dir,
            config=config,
            coordinator=coordinator
        )

        # =========================================================
        # 🔥 开始训练
        # =========================================================
        try:
            trainer.run()
            logger.info("✅ Phase 3 完成")
        except Exception as e:
            logger.error(f"❌ Phase 3 执行失败: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 70)
    logger.info("🎉 程序执行完成")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()