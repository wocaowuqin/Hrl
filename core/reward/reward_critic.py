# reward_critic.py
"""
Reward Critic Module for Multicast VNF Mapping (修复版)

修复内容:
1. ✅ 连接奖励改为线性增长 (exponential=1.0)
2. ✅ 提升VNF部署奖励 (100+100=200)
3. ✅ 加重移动成本 (-1.0/步)
4. ✅ 加强引导信号 (15/-10)
5. ✅ 重惩超时 (-300)
"""

import logging
from typing import Dict, Optional, Tuple, Any, Union, List
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class RewardCriticParams:
    """奖励函数参数配置 - 修复版"""

    # ========================================
    # 1. VNF 部署阶段 (大幅提升重要性)
    # ========================================
    vnf_deploy_success: float = 100.0  # 40→100 提升2.5倍
    vnf_all_complete: float = 100.0  # 50→100 提升2倍
    vnf_deploy_failed: float = -100.0  # -30→-100 惩罚3倍

    move_to_dc_bonus: float = 15.0  # 5→15 提升3倍
    move_cost: float = -1.0  # -0.1→-1.0 成本10倍

    # ========================================
    # 2. 树构建阶段 (线性增长，避免爆炸)
    # ========================================
    connection_base: float = 50.0  # 60→50 略降
    connection_exponential: float = 1.0  # 🔥 1.5→1.0 线性！
    connection_progress_bonus: float = 50.0  # 保持

    dest_reached_bonus: float = 15.0  # 30→15 降低
    full_completion_bonus: float = 100.0  # 🔥 300→100 大幅降低

    # ========================================
    # 3. 导航与引导 (大幅加强)
    # ========================================
    guidance_closer_rate: float = 15.0  # 5→15 提升3倍
    guidance_farther_penalty: float = -10.0  # -2→-10 惩罚5倍
    guidance_idle_penalty: float = -3.0  # -0.5→-3 惩罚6倍

    # ========================================
    # 4. 惩罚项 (大幅加重)
    # ========================================
    invalid_link: float = -30.0  # -10→-30 惩罚3倍
    invalid_action: float = -50.0  # -15→-50 惩罚3.3倍
    wrong_position: float = -20.0  # -5→-20 惩罚4倍

    # 频次惩罚 (更严格)
    freq_penalty_threshold: int = 3  # 5→3 降低阈值
    freq_penalty_rate: float = -10.0  # -2→-10 惩罚5倍

    # ========================================
    # 5. 超时处理 (重惩)
    # ========================================
    timeout_high_progress_threshold: float = 0.9  # 0.8→0.9 提高门槛
    timeout_bonus_rate: float = 30.0  # 50→30 降低安慰奖
    timeout_penalty_rate: float = 300.0  # 🔥 50→300 惩罚6倍
    vnf_timeout_penalty: float = -200.0  # 🔥 -50→-200 惩罚4倍

    # ========================================
    # 6. 兼容性参数 (保留)
    # ========================================
    reward_scale: float = 1.0
    w_cpu: float = 0.33
    w_bw: float = 0.34
    w_hop: float = 0.33


@dataclass
class CriticDiagnostics:
    """诊断信息数据结构"""
    total_rewards: List[float]
    avg_reward: float
    request_count: int
    success_rate: float


class RewardCritic:
    """
    修复版 RewardCritic - 平衡各阶段奖励，加强过程约束
    """

    def __init__(self,
                 training_phase: int = 3,
                 params: Optional[Union[Dict, RewardCriticParams]] = None):

        self.phase = int(training_phase)

        # 参数初始化
        if params is None:
            self.params = RewardCriticParams()
        elif isinstance(params, dict):
            valid_keys = RewardCriticParams.__annotations__.keys()
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            self.params = RewardCriticParams(**filtered_params)
        else:
            self.params = params

        # 运行时状态
        self._buffer_reward = 0.0
        self._request_active = False

        # 统计历史
        self.reward_history: List[float] = []
        self.history_len = 1000

    # =========================================================================
    # VNF 部署阶段奖励计算
    # =========================================================================
    def compute_vnf_deploy_reward(self,
                                  success: bool,
                                  all_complete: bool = False,
                                  quality_score: float = 0.0) -> float:
        """
        计算 VNF 部署动作的奖励

        修复说明:
        - 单个VNF部署成功: 40→100
        - 全部完成: 50→100
        - 总奖励(3个VNF): 170→400 (提升2.35倍)
        """
        p = self.params
        if success:
            reward = p.vnf_deploy_success + quality_score
            if all_complete:
                reward += p.vnf_all_complete
            return reward * p.reward_scale
        else:
            return p.vnf_deploy_failed * p.reward_scale

    def compute_vnf_move_reward(self,
                                to_dc: bool = False,
                                valid_link: bool = True,
                                guidance_val: float = 0.0) -> float:
        """
        计算 VNF 阶段移动奖励

        修复说明:
        - 移动成本: -0.1→-1.0 (10倍)
        - 到DC奖励: 5→15 (3倍)
        """
        p = self.params
        if not valid_link:
            return p.invalid_link * p.reward_scale

        reward = p.move_cost + guidance_val
        if to_dc:
            reward += p.move_to_dc_bonus

        return reward * p.reward_scale

    # =========================================================================
    # 树构建阶段奖励计算 (关键修复)
    # =========================================================================
    def compute_tree_connection_reward(self,
                                       connected_count: int,
                                       total_dests: int,
                                       is_complete: bool = False) -> float:
        """
        计算连接动作奖励 (改为线性增长)

        修复说明:
        - exponential: 1.5→1.0 (线性增长)
        - full_completion_bonus: 300→100 (大幅降低)

        奖励计算 (5个目的地):
        n=1: 50×1.0^0 + 10 = 60
        n=2: 50×1.0^1 + 20 = 70
        n=3: 50×1.0^2 + 30 = 80
        n=4: 50×1.0^3 + 40 = 90
        n=5: 50×1.0^4 + 50 + 100 = 200
        总计: 500分 (原653.8分)
        """
        p = self.params

        # 1. 线性基础奖励 (exponential=1.0)
        base = p.connection_base * (p.connection_exponential ** (connected_count - 1))

        # 2. 进度奖励
        ratio = connected_count / total_dests if total_dests > 0 else 0
        progress = p.connection_progress_bonus * ratio

        reward = base + progress

        # 3. 完成奖励 (大幅降低)
        if is_complete:
            reward += p.full_completion_bonus

        return reward * p.reward_scale

    def compute_tree_move_reward(self,
                                 to_dest: bool = False,
                                 valid_link: bool = True,
                                 min_dist_before: int = 999,
                                 min_dist_after: int = 999) -> float:
        """
        计算树构建阶段移动奖励 (加强引导)

        修复说明:
        - guidance_closer_rate: 5→15 (3倍)
        - guidance_farther_penalty: -2→-10 (5倍)
        - guidance_idle_penalty: -0.5→-3 (6倍)
        """
        p = self.params
        if not valid_link:
            return p.invalid_link * p.reward_scale

        reward = p.move_cost

        # 到达目的节点
        if to_dest:
            reward += p.dest_reached_bonus

        # 🔥 距离引导逻辑 (大幅加强)
        if min_dist_after < min_dist_before:
            reward += p.guidance_closer_rate
        elif min_dist_after > min_dist_before:
            reward += p.guidance_farther_penalty
        else:
            reward += p.guidance_idle_penalty

        return reward * p.reward_scale

    def compute_frequency_penalty(self, visit_count: int, is_hub: bool = False) -> float:
        """
        计算频次惩罚

        修复说明:
        - freq_penalty_threshold: 5→3 (更严格)
        - freq_penalty_rate: -2→-10 (5倍)
        """
        p = self.params
        if is_hub or visit_count <= p.freq_penalty_threshold:
            return 0.0

        excess = visit_count - p.freq_penalty_threshold
        return (p.freq_penalty_rate * excess) * p.reward_scale

    # =========================================================================
    # 超时与结算 (重惩)
    # =========================================================================
    def compute_timeout_reward(self,
                               in_vnf_phase: bool,
                               connected_count: int = 0,
                               total_dests: int = 1) -> float:
        """
        计算超时奖励/惩罚

        修复说明:
        - vnf_timeout_penalty: -50→-200 (4倍)
        - timeout_penalty_rate: 50→300 (6倍)
        - timeout_high_progress_threshold: 0.8→0.9 (更严格)
        """
        p = self.params

        # VNF阶段超时
        if in_vnf_phase:
            return p.vnf_timeout_penalty * p.reward_scale

        # 树构建阶段超时
        ratio = connected_count / total_dests if total_dests > 0 else 0

        if ratio >= p.timeout_high_progress_threshold:
            # 高进度 (>90%)，给安慰奖
            return (p.timeout_bonus_rate * ratio) * p.reward_scale
        else:
            # 低进度，重惩
            penalty = min(300.0, p.timeout_penalty_rate * (1.0 - ratio))
            return -penalty * p.reward_scale

    # =========================================================================
    # 统一调用接口
    # =========================================================================
    def get_reward(self, phase: str, **kwargs) -> float:
        """统一接口，根据 phase 自动分发"""
        if phase == 'vnf_deploy':
            return self.compute_vnf_deploy_reward(**kwargs)
        elif phase == 'vnf_move':
            return self.compute_vnf_move_reward(**kwargs)
        elif phase == 'tree_connect':
            return self.compute_tree_connection_reward(**kwargs)
        elif phase == 'tree_move':
            return self.compute_tree_move_reward(**kwargs)
        elif phase == 'timeout':
            return self.compute_timeout_reward(**kwargs)
        elif phase == 'penalty':
            t = kwargs.get('type', 'invalid_action')
            val = getattr(self.params, t, -10.0)
            return val * self.params.reward_scale
        return 0.0

    # =========================================================================
    # 新增: 路径质量奖励 (可选)
    # =========================================================================
    def compute_quality_bonus(self,
                              tree_edges: int,
                              optimal_edges: int = None,
                              steps_used: int = None,
                              max_steps: int = 200) -> float:
        """
        计算路径质量奖励 (可选特性)

        Args:
            tree_edges: 实际使用的边数
            optimal_edges: 理论最优边数 (可选)
            steps_used: 实际使用步数
            max_steps: 最大允许步数

        Returns:
            质量奖励 (0-100分)
        """
        bonus = 0.0

        # 1. 边数效率奖励
        if optimal_edges and tree_edges > 0:
            edge_ratio = optimal_edges / tree_edges
            if edge_ratio >= 0.9:
                bonus += 50.0
            elif edge_ratio >= 0.8:
                bonus += 30.0

        # 2. 步数效率奖励
        if steps_used:
            step_ratio = 1.0 - (steps_used / max_steps)
            if step_ratio >= 0.7:  # 用时<30%
                bonus += 50.0
            elif step_ratio >= 0.5:  # 用时<50%
                bonus += 30.0

        return bonus * self.params.reward_scale

    # =========================================================================
    # 辅助功能
    # =========================================================================
    def record_step(self, reward: float):
        """记录单步奖励"""
        self._buffer_reward += reward

    def finish_request(self, final_reward: float = 0.0):
        """请求结束，归档总奖励"""
        total = self._buffer_reward + final_reward
        self.reward_history.append(total)
        self._buffer_reward = 0.0

        if len(self.reward_history) > self.history_len:
            self.reward_history.pop(0)

    def save(self, path: str) -> bool:
        """保存状态"""
        try:
            state = {
                "params": asdict(self.params),
                "reward_history": self.reward_history
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def load(self, path: str) -> bool:
        """加载状态"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            if "params" in state:
                valid_keys = RewardCriticParams.__annotations__.keys()
                filtered = {k: v for k, v in state["params"].items() if k in valid_keys}
                self.params = RewardCriticParams(**filtered)
            self.reward_history = state.get("reward_history", [])
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def on_new_request(self):
        """兼容接口"""
        self._buffer_reward = 0.0

    def criticize(self, **kwargs):
        """兼容旧版接口"""
        return 0.0

    # =========================================================================
    # 调试方法
    # =========================================================================
    def print_params_summary(self):
        """打印参数摘要 (用于调试)"""
        print("\n" + "=" * 60)
        print("🎯 Reward System Parameters Summary")
        print("=" * 60)
        print(f"\n📍 VNF部署阶段:")
        print(f"   单个成功: +{self.params.vnf_deploy_success}")
        print(f"   全部完成: +{self.params.vnf_all_complete}")
        print(f"   部署失败: {self.params.vnf_deploy_failed}")

        print(f"\n🌳 树构建阶段:")
        print(f"   连接基础: {self.params.connection_base}")
        print(f"   增长系数: {self.params.connection_exponential} (1.0=线性)")
        print(f"   完成奖励: +{self.params.full_completion_bonus}")

        print(f"\n🧭 引导系统:")
        print(f"   靠近目标: +{self.params.guidance_closer_rate}")
        print(f"   远离目标: {self.params.guidance_farther_penalty}")

        print(f"\n⚠️  惩罚系统:")
        print(f"   移动成本: {self.params.move_cost}/步")
        print(f"   超时惩罚: {self.params.timeout_penalty_rate}")
        print(f"   VNF超时: {self.params.vnf_timeout_penalty}")

        print("=" * 60 + "\n")