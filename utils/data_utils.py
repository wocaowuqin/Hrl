# utils/config_utils.py
import yaml
from pathlib import Path

def load_config(config_path):
    """读取yaml配置文件"""
    with open(Path(config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def merge_config(base_config, *override_configs):
    """合并基础配置和阶段配置"""
    merged = base_config.copy()
    for cfg in override_configs:
        merged.update(cfg)
    return merged