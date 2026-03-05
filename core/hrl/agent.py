#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRL Agent — 组合入口
将 base / action / memory / train 四个模块组合成完整的 HRLAgent。

目录结构（core/hrl/）:
    agent.py          ← 本文件（对外接口，保持不变）
    agent_base.py     ← __init__, save/load, train/eval, reset
    agent_action.py   ← select_action, 子目标选择, 嵌入计算
    agent_memory.py   ← store_transition_*
    agent_train.py    ← update_policies, Double DQN, epsilon
"""

import logging

from core.hrl.agent_base   import HRLAgentBase
from core.hrl.agent_action import HRLAgentAction
from core.hrl.agent_memory import HRLAgentMemory
from core.hrl.agent_train  import HRLAgentTrain

logger = logging.getLogger(__name__)


class HRLAgent(HRLAgentBase, HRLAgentAction, HRLAgentMemory, HRLAgentTrain):
    """
    Hierarchical RL Agent（重构版）

    继承顺序决定MRO：Base最先提供__init__，其余Mixin只包含方法。
    外部代码无需改动，接口完全向后兼容。
    """
    pass


# ── 向后兼容 ──────────────────────────────────────────────────────────────

class GoalConditionedHRLAgent(HRLAgent):
    """向后兼容的旧类名"""
    def __init__(self, config, phase=3, goal_strategy='adaptive', **kwargs):
        logger.warning("GoalConditionedHRLAgent已重构为HRLAgent，使用兼容模式")
        super().__init__(config, phase=phase, goal_strategy=goal_strategy, **kwargs)


def create_goal_conditioned_agent(config, phase=3, goal_strategy='adaptive',
                                   encoder=None, **kwargs) -> HRLAgent:
    """工厂函数，向后兼容"""
    return HRLAgent(
        config=config,
        encoder=encoder,
        phase=phase,
        goal_strategy=goal_strategy,
        **kwargs,
    )