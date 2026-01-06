class StatelessRewardCritic:
    def __init__(self):
        # 🟢 终极目标：极大幅度提升引力
        self.connect_bonus = 100.0  # 🚀 从50翻倍到100：让连接目的地产生的Q值回报统治全局

        # 🟡 中间过程：保持低调
        self.deploy_bonus = 2.0  # 🛠️ 略微提升：依然鼓励部署，但不要让它觉得部署完就没事了
        self.reuse_bonus = 5.0  # 🌿 提升：Steiner Tree 的核心是复用，复用奖励要高于步数成本

        # 🔴 负面反馈：增加“痛感”
        self.step_cost = 1.0  # ⏱️ 大幅提升(0.2->1.0)：每多走一步都是巨大的损失，逼它走最短路
        self.illegal_penalty = 10.0  # 🚫 加重：对于撞 Mask 或无效移动，双倍惩罚
        self.timeout_penalty = 300.0  # 💀 重罚：失败意味着这一场白练了，产生强烈的负梯度

        # 🔵 🔥 [核心新增] 针对性惩罚：徘徊抑制
        self.backtrack_penalty = 15.0  # 🔄 专门针对 3/3 进度后还在旧树节点移动的行为

    def compute_reward(self, info: dict) -> float:
        reward = 0.0

        # 1. 基础时间成本 (生存惩罚)
        reward -= self.step_cost

        # 2. 核心目标：连接目的地
        if info.get('reached_new_dest', False):
            reward += self.connect_bonus

        # 3. 中间目标：部署 VNF
        if info.get('action_type') == 'stay':
            if info.get('success', False):
                reward += self.deploy_bonus
            else:
                # 🔥 [新增] 如果在原地部署失败(资源不足)，给予惩罚
                reward -= self.illegal_penalty

                # 4. 效率目标：路径复用
        if info.get('reused_tree_node', False):
            reward += self.reuse_bonus

        # 5. 🔥 [核心修复] 针对 3/3 进度后的徘徊行为
        # 如果进度已满 (3/3)，且动作是移动到旧节点，且没连上目的地
        if info.get('progress_ratio', 0.0) >= 0.99:
            if info.get('reused_tree_node', False) and not info.get('reached_new_dest', False):
                # 这就是你在日志里看到的 [Move Away] 行为，必须重罚
                reward -= self.backtrack_penalty

        # 6. 负面反馈
        if info.get('invalid_action', False):
            reward -= self.illegal_penalty

        if info.get('timeout', False):
            reward -= self.timeout_penalty

        # 7. 进度引导
        if info.get('branch_completed', False):
            reward += 15.0  # 稍微提高分支奖励
        if info.get('progress_ratio', 0.0) >= 0.99:
            if info.get('action_type') == 'MOVE':
                # 只要是移动动作，且没连上新目的地，一律视为徘徊，给予重罚
                if not info.get('reached_new_dest', False):
                    return -20.0  # 彻底阻断移动欲望
        return reward