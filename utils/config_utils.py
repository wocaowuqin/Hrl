# utils/config_utils.py

import yaml
import os
import re
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def get_project_root() -> str:
    """
    自动获取项目根目录

    逻辑：
    - 当前脚本在 utils/config_utils.py
    - utils/ 的父目录就是项目根目录

    Returns:
        项目根目录的绝对路径
    """
    current_file_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(utils_dir)
    return project_root


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    安全加载单个 YAML 文件

    Args:
        file_path: YAML 文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: YAML 解析错误
    """
    if not os.path.exists(file_path):
        abs_path = os.path.abspath(file_path)
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"YAML 解析错误 {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"读取文件失败 {file_path}: {e}")


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    递归深度更新字典（不修改原字典）

    Args:
        base_dict: 基础字典
        update_dict: 更新字典

    Returns:
        合并后的字典
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _resolve_path_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析配置中的路径变量（如 ${project.base_dir}）

    Args:
        config: 配置字典

    Returns:
        解析后的配置字典
    """

    def resolve_value(value, context):
        """递归解析单个值"""
        if isinstance(value, str):
            # 查找 ${var.key} 格式的变量
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)

            for match in matches:
                keys = match.split('.')
                resolved = context

                try:
                    for key in keys:
                        resolved = resolved[key]

                    # 替换变量
                    value = value.replace(f'${{{match}}}', str(resolved))
                except (KeyError, TypeError):
                    # 变量不存在，保持原样
                    logger.warning(f"⚠️  无法解析变量: ${{{match}}}")

            return value

        elif isinstance(value, dict):
            return {k: resolve_value(v, context) for k, v in value.items()}

        elif isinstance(value, list):
            return [resolve_value(item, context) for item in value]

        else:
            return value

    return resolve_value(config, config)


def _post_process_config(config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """
    后处理配置，确保所有必需的键存在并处理兼容性

    Args:
        config: 原始配置字典
        phase: 当前阶段

    Returns:
        处理后的配置字典
    """
    # === 1. 确保 eval 配置块存在 ===
    if 'eval' not in config:
        config['eval'] = {}

    # 设置默认 device
    if 'device' not in config['eval']:
        try:
            import torch
            config['eval']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            config['eval']['device'] = 'cpu'

    # 设置默认 seed
    if 'seed' not in config['eval']:
        config['eval']['seed'] = 42

    # === 2. 确保 training 配置块存在 ===
    if 'training' not in config:
        config['training'] = {}

    # 从 model.yaml 的 base 块迁移参数（兼容性处理）
    if 'base' in config:
        base_cfg = config['base']

        # 迁移参数到 training 块
        migration_map = {
            'exp_memory': 'buffer_size',
            'learning_rate': 'learning_rate',
            'gamma': 'gamma',
            'batch_size': 'batch_size'
        }

        for old_key, new_key in migration_map.items():
            if old_key in base_cfg and new_key not in config['training']:
                config['training'][new_key] = base_cfg[old_key]
                logger.debug(f"迁移配置: base.{old_key} -> training.{new_key}")

    # 从 model.yaml 的 training 块迁移 target_update_freq
    if 'hard_update_frequency' in config.get('training', {}) and \
            'target_update_freq' not in config['training']:
        config['training']['target_update_freq'] = config['training']['hard_update_frequency']
        logger.debug(f"迁移配置: training.hard_update_frequency -> training.target_update_freq")

    # 设置默认值
    training_defaults = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'buffer_size': 100000,
        'batch_size': 32,
        'target_update_freq': 1000
    }

    for key, default_val in training_defaults.items():
        if key not in config['training']:
            config['training'][key] = default_val

    # === 3. 确保 epsilon 配置块存在 ===
    if 'epsilon' not in config:
        config['epsilon'] = {}

    # 从 phase3.yaml 继承或使用默认值
    epsilon_defaults = {
        'initial': 1.0,
        'final': 0.01,
        'decay_steps': 10000
    }

    for key, default_val in epsilon_defaults.items():
        if key not in config['epsilon']:
            config['epsilon'][key] = default_val

    # === 4. 确保 env 配置块存在（Agent 需要） ===
    if 'env' not in config:
        config['env'] = {}

    # 从 environment 块迁移（如果存在）
    if 'environment' in config:
        env_source = config['environment']

        migration_map = {
            'nb_high_level_goals': 'nb_high_level_goals',
            'nb_low_level_actions': 'nb_low_level_actions',
            'num_nodes': 'num_nodes',
            'dc_nodes': 'dc_nodes',
            'capacities': 'capacities'
        }

        for old_key, new_key in migration_map.items():
            if old_key in env_source and new_key not in config['env']:
                config['env'][new_key] = env_source[old_key]

    # 从 gnn 块迁移（如果存在）
    if 'gnn' in config:
        gnn_cfg = config['gnn']

        if 'num_goals' in gnn_cfg and 'nb_high_level_goals' not in config['env']:
            config['env']['nb_high_level_goals'] = gnn_cfg['num_goals']

        if 'num_actions' in gnn_cfg and 'nb_low_level_actions' not in config['env']:
            config['env']['nb_low_level_actions'] = gnn_cfg['num_actions']

    # 设置默认值
    env_defaults = {
        'nb_high_level_goals': 10,
        'nb_low_level_actions': 100
    }

    for key, default_val in env_defaults.items():
        if key not in config['env']:
            config['env'][key] = default_val

    # === 5. 确保 gnn 配置块存在 ===
    if 'gnn' not in config:
        config['gnn'] = {}

    # 设置默认 GNN 参数（会被环境动态覆盖）
    gnn_defaults = {
        'node_feat_dim': 10,
        'edge_feat_dim': 3,
        'request_feat_dim': 6,
        'hidden_dim': 128,
        'num_gat_layers': 3,
        'num_heads': 4,
        'dropout': 0.1
    }

    for key, default_val in gnn_defaults.items():
        if key not in config['gnn']:
            config['gnn'][key] = default_val

    # === 6. Phase 特定处理 ===
    if phase == 'phase1':
        # Phase1 专家数据收集
        if 'phase1' not in config:
            config['phase1'] = {}

        phase1_defaults = {
            'episodes': 2000,
            'save_every': 500,
            'max_dataset_size': 100000
        }

        for key, default_val in phase1_defaults.items():
            if key not in config['phase1']:
                config['phase1'][key] = default_val

    elif phase == 'phase2':
        # Phase2 模仿学习
        if 'phase2' not in config:
            config['phase2'] = {}

        # 兼容旧配置：il -> phase2
        if 'il' in config and 'phase2' not in config:
            config['phase2'] = config['il']

        phase2_defaults = {
            'epochs': 10,
            'batch_size': 128,
            'validation_split': 0.1
        }

        for key, default_val in phase2_defaults.items():
            if key not in config['phase2']:
                config['phase2'][key] = default_val

    elif phase == 'phase3':
        # Phase3 强化学习
        if 'phase3' not in config:
            config['phase3'] = {}

        # 确保 RL 配置存在
        if 'rl' not in config and 'phase3' in config:
            config['rl'] = config['phase3'].get('rl', {})

        # 确保 DAgger 配置存在
        if 'dagger' not in config and 'phase3' in config:
            config['dagger'] = config['phase3'].get('dagger', {
                'initial_beta': 0.8,
                'final_beta': 0.0,
                'decay_steps': 50000
            })

        phase3_defaults = {
            'episodes': 300,
            'max_steps': 3000000,
            'eval_every': 100
        }

        for key, default_val in phase3_defaults.items():
            if key not in config['phase3']:
                config['phase3'][key] = default_val

    # === 7. 路径处理（变量替换）===
    config = _resolve_path_variables(config)

    return config


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    验证配置是否包含 Agent 所需的所有键

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_keys = {
        'eval': ['device'],
        'training': ['learning_rate', 'gamma', 'buffer_size', 'batch_size', 'target_update_freq'],
        'epsilon': ['initial', 'final', 'decay_steps'],
        'env': ['nb_high_level_goals', 'nb_low_level_actions'],
        'gnn': ['node_feat_dim', 'edge_feat_dim', 'request_feat_dim', 'hidden_dim']
    }

    missing = []

    for section, keys in required_keys.items():
        if section not in config:
            missing.append(f"缺少配置块: {section}")
            continue

        for key in keys:
            if key not in config[section]:
                missing.append(f"缺少配置键: {section}.{key}")

    if missing:
        logger.warning("⚠️  配置验证警告（将使用默认值）:")
        for msg in missing:
            logger.warning(f"  - {msg}")
        return False

    logger.info("✅ Agent 配置验证通过")
    return True


def validate_env_config(config: Dict[str, Any]) -> bool:
    """
    验证环境配置是否完整

    Args:
        config: 配置字典

    Returns:
        是否有效
    """
    required_paths = ['paths', 'resources', 'runtime']

    missing = []

    for section in required_paths:
        if section not in config:
            missing.append(f"缺少配置块: {section}")

    # 检查路径配置
    if 'paths' in config:
        required_path_keys = ['input_dir']
        for key in required_path_keys:
            if key not in config['paths']:
                missing.append(f"缺少路径配置: paths.{key}")

    if missing:
        logger.warning("⚠️  环境配置验证警告:")
        for msg in missing:
            logger.warning(f"  - {msg}")
        return False

    logger.info("✅ 环境配置验证通过")
    return True


def load_config(phase: str = 'phase3', config_dir: str = None, validate: bool = True) -> Dict[str, Any]:
    """
    加载并合并配置

    加载顺序：
    1. base.yaml - 基础配置
    2. env.yaml - 环境配置
    3. model.yaml - 模型配置
    4. agent.yaml - Agent 配置（可选）
    5. phase*.yaml - 阶段配置

    后加载的配置会覆盖先加载的同名键

    Args:
        phase: 阶段名称 ('phase1', 'phase2', 'phase3')
        config_dir: 配置目录路径（默认为项目根目录下的 configs/）
        validate: 是否验证配置完整性

    Returns:
        合并后的配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        RuntimeError: 配置解析错误
    """
    # 自动定位 configs 目录的绝对路径
    if config_dir is None:
        root = get_project_root()
        config_dir = os.path.join(root, 'configs')

    logger.info(f"📂 配置目录: {config_dir}")
    logger.info(f"🔄 加载配置: {phase}")

    # 1. 加载基础配置
    base_path = os.path.join(config_dir, 'base.yaml')
    config = load_yaml(base_path)
    logger.debug(f"✅ 加载 base.yaml")

    # 2. 加载环境配置
    env_path = os.path.join(config_dir, 'env.yaml')
    if os.path.exists(env_path):
        env_cfg = load_yaml(env_path)
        deep_update(config, env_cfg)
        logger.debug(f"✅ 加载 env.yaml")
    else:
        logger.warning(f"⚠️  未找到 env.yaml，跳过")

    # 3. 加载模型配置
    model_path = os.path.join(config_dir, 'model.yaml')
    if os.path.exists(model_path):
        model_cfg = load_yaml(model_path)
        deep_update(config, model_cfg)
        logger.debug(f"✅ 加载 model.yaml")
    else:
        logger.warning(f"⚠️  未找到 model.yaml，跳过")

    # 4. 🆕 加载 Agent 配置（可选）
    agent_path = os.path.join(config_dir, 'agent.yaml')
    if os.path.exists(agent_path):
        agent_cfg = load_yaml(agent_path)

        # 将 agent 块的内容合并到顶层
        if 'agent' in agent_cfg:
            agent_content = agent_cfg['agent']

            # 合并 training 配置
            if 'training' in agent_content:
                if 'training' not in config:
                    config['training'] = {}
                deep_update(config['training'], agent_content['training'])

            # 合并 epsilon 配置
            if 'epsilon' in agent_content:
                if 'epsilon' not in config:
                    config['epsilon'] = {}
                deep_update(config['epsilon'], agent_content['epsilon'])

            # 合并 eval 配置
            if 'eval' in agent_content:
                if 'eval' not in config:
                    config['eval'] = {}
                deep_update(config['eval'], agent_content['eval'])

            # 合并 network 配置到 gnn
            if 'network' in agent_content:
                if 'gnn' not in config:
                    config['gnn'] = {}
                deep_update(config['gnn'], agent_content['network'])

            # 合并 dagger 配置
            if 'dagger' in agent_content:
                config['dagger'] = agent_content['dagger']

            # 合并 optimizer 配置
            if 'optimizer' in agent_content:
                if 'training' not in config:
                    config['training'] = {}
                config['training']['optimizer'] = agent_content['optimizer']

            # 保留 agent 块的其他配置
            if 'architecture' in agent_content:
                config['agent_architecture'] = agent_content['architecture']
            if 'checkpoint' in agent_content:
                config['checkpoint'] = agent_content['checkpoint']

        logger.debug(f"✅ 加载 agent.yaml")
    else:
        logger.debug(f"ℹ️  未找到 agent.yaml（可选），使用默认配置")

    # 5. 加载特定阶段配置
    if phase:
        phase_path = os.path.join(config_dir, f'{phase}.yaml')
        if os.path.exists(phase_path):
            phase_cfg = load_yaml(phase_path)
            deep_update(config, phase_cfg)
            logger.debug(f"✅ 加载 {phase}.yaml")
            config['current_phase'] = phase
        else:
            logger.warning(f"⚠️  未找到 {phase}.yaml，使用默认配置")
            config['current_phase'] = phase

    # 6. 后处理配置（兼容性、默认值、变量替换）
    config = _post_process_config(config, phase)
    logger.debug(f"✅ 配置后处理完成")

    # 7. 验证配置（可选）
    if validate:
        validate_agent_config(config)
        if 'paths' in config or 'resources' in config:
            validate_env_config(config)

    logger.info(f"✅ 配置加载完成")

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"💾 配置已保存到: {output_path}")


def print_config_summary(config: Dict[str, Any]):
    """
    打印配置摘要

    Args:
        config: 配置字典
    """
    print("\n" + "=" * 70)
    print("📋 配置摘要")
    print("=" * 70)

    # 阶段信息
    phase = config.get('current_phase', 'N/A')
    print(f"阶段: {phase}")

    # 设备信息
    device = config.get('eval', {}).get('device', 'N/A')
    print(f"设备: {device}")

    # 环境信息
    env_cfg = config.get('env', {})
    print(f"高层目标数: {env_cfg.get('nb_high_level_goals', 'N/A')}")
    print(f"低层动作数: {env_cfg.get('nb_low_level_actions', 'N/A')}")

    # GNN 信息
    gnn_cfg = config.get('gnn', {})
    print(f"GNN 隐层维度: {gnn_cfg.get('hidden_dim', 'N/A')}")
    print(f"GAT 层数: {gnn_cfg.get('num_gat_layers', 'N/A')}")

    # 训练信息
    training_cfg = config.get('training', {})
    print(f"学习率: {training_cfg.get('learning_rate', 'N/A')}")
    print(f"Batch 大小: {training_cfg.get('batch_size', 'N/A')}")
    print(f"经验池大小: {training_cfg.get('buffer_size', 'N/A')}")

    # Epsilon 信息
    epsilon_cfg = config.get('epsilon', {})
    print(f"Epsilon 初始: {epsilon_cfg.get('initial', 'N/A')}")
    print(f"Epsilon 最终: {epsilon_cfg.get('final', 'N/A')}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试代码
    try:
        print(f"📍 项目根目录: {get_project_root()}")
        print(f"📍 配置目录: {os.path.join(get_project_root(), 'configs')}")
        print("")

        # 测试加载各个阶段的配置
        for phase_name in ['phase1', 'phase2', 'phase3']:
            print(f"\n{'=' * 70}")
            print(f"测试加载: {phase_name}")
            print('=' * 70)

            config = load_config(phase_name, validate=True)
            print_config_summary(config)

            # 可选：保存合并后的配置
            # save_config(config, f'outputs/merged_{phase_name}_config.yaml')

        print("\n✅ 所有配置加载测试通过！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()