# modules/base_env.py
import gym
from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional


class BaseEnv(gym.Env, ABC):
    """
    SFC 分层强化学习环境抽象基类

    所有具体环境（Flat、GNN、测试版等）必须继承此类并实现以下接口。
    注意：由于是分层 RL，标准 step() 仅作为兼容接口，实际交互推荐使用
          step_high_level() 和 step_low_level()。
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config

    @abstractmethod
    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        """
        加载指定阶段的数据集（支持两种调用方式）

        调用方式1: load_dataset("phase3")
        调用方式2: load_dataset("phase3_requests.pkl", "phase3_events.pkl")

        Args:
            phase_or_req_file: 阶段名（如 "phase3"）或请求文件名
            events_file: 事件文件名（可选，提供时使用文件名方式）

        Returns:
            bool: 加载是否成功
        """
        pass

    @abstractmethod
    def reset(self, phase: str = "phase3") -> Any:
        """
        重置环境并开始新 episode

        Args:
            phase: 数据阶段（如 "phase3"）

        Returns:
            initial observation
        """
        pass

    @abstractmethod
    def reset_request(self) -> Tuple[Optional[Dict], Any]:
        """
        重置并获取下一个请求（旧环境兼容接口）

        Phase 1 数据收集器和 Phase 3 训练器依赖此方法

        Returns:
            (request, state): 请求字典和当前状态
                            如果没有更多请求，request 为 None
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        标准 Gym step 接口（兼容性保留）
        在分层环境中通常映射到 low-level 动作
        """
        pass

    # ====================== 可选：建议添加分层接口（提高类型提示） ======================
    # 虽然 Python 不强制，但加上能让 IDE 和开发者更清楚预期行为

    def step_high_level(self, goal_idx: int) -> Tuple[Any, float, bool, Dict]:
        """
        高层动作：选择下一个要接入的目标分支
        """
        raise NotImplementedError("Subclasses should implement step_high_level if using HRL")

    def step_low_level(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """
        低层动作：选择路径与VNF部署方案

        Returns:
            obs: 观察
            reward: 奖励
            sub_done: 当前子目标是否完成
            req_done: 整个请求是否完成
            info: 额外信息字典
        """
        raise NotImplementedError("Subclasses should implement step_low_level if using HRL")

    def get_state(self) -> Any:
        """统一获取当前状态（Flat 或 Graph）"""
        raise NotImplementedError

    # ====================== 旧环境兼容接口（推荐添加） ======================

    def get_expert_high_level_candidates(self, state_vec=None, top_k: int = 5):
        """
        获取专家推荐的高层候选目标（Phase 1 需要）

        Args:
            state_vec: 状态向量（可选，通常使用 self 的属性）
            top_k: 返回前 k 个候选

        Returns:
            List[Tuple[int, float]]: [(dest_idx, score), ...] 按分数降序
        """
        raise NotImplementedError("Subclasses should implement if using expert guidance")

    def get_expert_high_level_goal(self, state_vec=None) -> int:
        """
        获取专家推荐的单一高层目标（Phase 1 需要）

        Returns:
            int: 目标索引
        """
        raise NotImplementedError("Subclasses should implement if using expert guidance")

    def get_expert_high_level_labels(self, state_vec=None, top_k: int = 5):
        """
        获取专家的高层策略标签（Phase 2 模仿学习需要）

        Returns:
            Tuple[List[int], List[float], int]: (ids, scores, best_id)
        """
        raise NotImplementedError("Subclasses should implement if using imitation learning")

    def get_high_level_candidate_mask(self, candidates):
        """
        生成高层候选动作掩码（Agent 需要）

        Returns:
            np.ndarray: 掩码数组
        """
        raise NotImplementedError("Subclasses should implement if using masked actions")

    def get_low_level_action_mask(self):
        """
        生成低层动作掩码（Agent 需要）

        Returns:
            np.ndarray: 掩码数组
        """
        raise NotImplementedError("Subclasses should implement if using masked actions")

    def expert_low_level_action(self, goal_dest_idx: int) -> int:
        """
        获取专家推荐的低层动作（DAgger 需要）

        Returns:
            int: 动作索引，-1 表示无法获取
        """
        raise NotImplementedError("Subclasses should implement if using DAgger")

    # ====================== 可视化和统计接口（可选） ======================

    def render_failure(self, failed_dest_idx: int, failed_path=None, title: str = "Failure"):
        """
        可视化部署失败的情况（可选）
        """
        pass  # 默认空实现，子类可选择实现

    def get_backup_metrics(self) -> Dict[str, float]:
        """
        获取备份策略的统计指标（可选）

        Returns:
            Dict: {'activation_rate': float, 'success_rate': float}
        """
        return {'activation_rate': 0.0, 'success_rate': 0.0}  # 默认实现

    def print_env_summary(self):
        """
        打印环境统计摘要（可选）
        """
        pass  # 默认空实现，子类可选择实现