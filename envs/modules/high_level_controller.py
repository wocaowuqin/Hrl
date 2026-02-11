import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class HighLevelController:
    """
    🎮 高层交互控制器 (High-Level Controller)

    职责：
    1. 管理高层决策 (目标选择)
    2. 维护高层阶段 (VNF部署 vs 目的地连接)
    3. 构建高层状态 (GNN特征) - V30.4 增强版
    4. 生成高层动作掩码 (资源感知)
    """

    def __init__(self, env):
        """
        初始化控制器
        :param env: 主环境对象的引用 (SFC_HIRL_Env)
        """
        self.env = env
        logger.info("✅ HighLevelController initialized (V30.4 State Enhanced)")
        self._pending_dist_cache = {}
        self._last_dests_update = -1
    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        """
        🎯 [V40.0] 设定高层目标 (已修复参数签名 & 起点同步)
        """
        # 1. 记录高层动作
        self.env.last_high_action_idx = high_action_idx
        self.env.current_subgoal_node = target_node_id

        # 2. 🔥 关键：新目标开始，步数计数器必须归零
        self.env.subgoal_step_count = 0

        # 3. 🔥 [新增] 同步 Agent 选择的起点 (解决 15->7 变成 3->7 的问题)
        if start_node_id is not None:
            # 强制更新环境的物理位置
            self.env.current_node_location = int(start_node_id)
            logger.info(f"📍 [High] Agent重置起点: {start_node_id}")

        # 4. 自动判定阶段 (Phase Logic)
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])

            # 使用 next_vnf_idx 作为唯一的VNF索引来源
            self.env.current_vnf_to_deploy = self.env.next_vnf_idx

            # 判断当前阶段
            if self.env.next_vnf_idx < len(vnf_list):
                # 阶段1：VNF 部署
                self.env.current_phase = 'vnf_deployment'
                self.env.current_deployment_target = target_node_id
                self.env.current_target_node = None
                logger.info(f"🎯 [Env] VNF部署阶段: VNF[{self.env.next_vnf_idx}] → 节点{target_node_id}")
            else:
                # 阶段2：目的地连接
                self.env.current_phase = 'destination_connection'
                self.env.current_target_node = target_node_id
                self.env.current_deployment_target = None
                logger.info(f"🎯 [Env] 目的地连接阶段: 目标节点{target_node_id}")
        else:
            logger.warning("⚠️ 设定目标时没有活跃请求")

        # 返回状态供 Coordinator 记录
        return self.get_high_level_state_graph()

    def step_high_level(self, action_idx):
        """
        🔥 [V50.0 最终修复版]
        逻辑回归纯粹：如果不慎选了已完成节点，立即截断 (Truncated)，
        迫使 Coordinator 进入下一轮循环，让 Agent 重新根据新 Mask 做决策。
        """
        # 1. 动作解析
        target_node = int(action_idx)
        self.current_high_action = target_node

        # 2. 获取物理真理
        connected = set()
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            try:
                connected = {int(x) for x in self.env.current_tree.get('connected_dests', set())}
            except:
                pass

        all_dests = set()
        if self.env.current_request:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
            except:
                pass

        # ============================================================
        # 3. 🔥 核心修复：遇到已完成节点，立即截断
        # ============================================================
        if target_node in connected:
            # logger.info(f"✅ [High] 选中目标 {target_node} 已在连接集合中")

            # 3.1 检查是否全部完成
            if all_dests.issubset(connected):
                logger.info("🎉 [High] 所有目的地已连接，Episode 完美结束！")
                return None, 20.0, True, False, {'all_done': True}

            # 3.2 🔥 [修复点] 未全部完成 -> 返回 Truncated
            # 不再自动跳转，不再修改 current_goal_node
            # 只是告诉外界：这个选择无效/已结束，请重试
            else:
                pending_dests = list(all_dests - connected)
                logger.warning(f"🔄 [High] 目标 {target_node} 无效(已完成) -> 返回 Truncated，请求重选")

                # 关键修改：truncated=True
                # 这会触发 Coordinator 结束当前 high_step，重新进入 run_high_low_cycle 的下一轮
                # 下一轮时，get_action_mask 会把 7 封死，Agent 只能选 2
                return None, 5.0, False, True, {
                    'subgoal_completed': True,
                    'warning': 'selected_completed_target'
                }

        # ============================================================
        # 4. 标准执行路径 (目标未完成)
        # ============================================================
        # 更新环境目标状态
        self.env.current_goal_node = target_node

        return None, 0.0, False, False, {
            'target_node': target_node,
            'status': 'executing'
        }
    def _is_all_completed(self):
        """辅助检查：是否 VNF 和 Dest 都搞定了"""
        if not self.env.current_request: return True

        # 1. 检查 VNF
        vnf_list = self.env.current_request.get('vnf', [])
        if self.env.next_vnf_idx < len(vnf_list): return False

        # 2. 检查 Dest
        connected = self.env.current_tree.get('connected_dests', set())
        all_dests = set(self.env.current_request.get('dest', []))
        if not all_dests.issubset(connected): return False

        return True

    def get_high_level_action_mask(self):
        """
        🔥 [V50.1 修复版]
        修复 'HighLevelController object has no attribute n' 错误。
        确保所有对 n 的引用都指向 self.env.n
        """
        # 🔥 [关键修复] 获取节点数 (必须从 env 获取)
        n = self.env.n

        # 初始化全 0 Mask
        mask = np.zeros(n, dtype=np.float32)

        # 0. 基础检查
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return np.ones(n, dtype=np.float32)

        vnf_list = self.env.current_request.get('vnf', [])
        current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)

        # ============================================================
        # 阶段 1: VNF 部署阶段
        # ============================================================
        if current_vnf_idx < len(vnf_list):
            if hasattr(self.env, 'dc_nodes'):
                for node in self.env.dc_nodes:
                    if 0 <= node < n:
                        mask[node] = 1.0
            # 兜底：如果算出来全0，暂时全开
            if np.sum(mask) == 0:
                mask[:] = 1.0

        # ============================================================
        # 阶段 2: 目的地连接阶段 (Routing)
        # ============================================================
        else:
            try:
                # 1. 获取全集 (强转 int)
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}

                # 2. 获取已完成集 (物理真理)
                connected_set = set()
                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_set = {int(x) for x in self.env.current_tree.get('connected_dests', set())}
            except:
                all_dests = set()
                connected_set = set()

            # 3. 计算待办集
            true_pending = list(all_dests - connected_set)

            # 4. 如果所有任务都完成了，返回全0
            if not true_pending:
                return np.zeros(n, dtype=np.float32)

            # 5. 设置 Mask
            for node in true_pending:
                if 0 <= node < n:
                    mask[node] = 1.0

            # 6. 🔥 [绝对封杀] 再次强制检查：凡是已连接的，必须为 0
            for done_node in connected_set:
                if 0 <= done_node < n:
                    mask[done_node] = 0.0

        return mask
    def get_high_level_state_graph(self):
        """
        🎯 [V30.4 修复增强版] 状态感知增强 - 完整修复版

        修复点：
        1. ✅ 修复Pending Dest逻辑：VNF阶段可见但不强调
        2. ✅ 改进资源匹配度：支持非DC节点 + 梯度匹配
        3. ✅ 添加阶段感知特征
        4. ✅ 补全边特征逻辑
        """
        n = self.env.n

        # 安全检查
        if not self.env.current_request:
            return Data(
                x=torch.zeros((n, 11), dtype=torch.float32),  # 🔥 11维特征
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                global_attr=torch.zeros((1, 5), dtype=torch.float32)
            )

        # =============================
        # 准备上下文数据
        # =============================
        req = self.env.current_request
        vnf_list = req.get('vnf', [])
        source = req.get('source')
        dests = req.get('dest', [])

        # 获取集合
        connected_dests = self.env.current_tree.get('connected_dests', set())
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        # 判断当前阶段
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        is_vnf_phase = next_vnf_idx < len(vnf_list)

        # 如果在VNF阶段，获取当前VNF的需求
        req_cpu, req_mem = 0.0, 0.0
        if is_vnf_phase:
            cpu_list = req.get('cpu_origin') or req.get('vnf_cpu', [])
            mem_list = req.get('memory_origin') or req.get('vnf_mem', [])
            if next_vnf_idx < len(cpu_list):
                req_cpu = float(cpu_list[next_vnf_idx])
                req_mem = float(mem_list[next_vnf_idx]) if next_vnf_idx < len(mem_list) else 1.0

        # 统计 VNF 负载
        node_vnf_counts = [0] * n
        if hasattr(self.env, 'current_placements') and self.env.current_placements:
            for placement_key in self.env.current_placements:
                node_id, vnf_idx = placement_key
                if 0 <= node_id < n:
                    node_vnf_counts[node_id] += 1

        # 获取DC节点集合
        dc_nodes = getattr(self.env, 'dc_nodes', set())

        # =============================
        # 1. 构建增强节点特征 [N, 11]
        # =============================
        x = []
        for node in range(n):
            # --- 基础资源 ---
            try:
                avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
                avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
                norm_cpu = avail_cpu / 100.0
                norm_mem = avail_mem / 100.0
            except:
                norm_cpu, norm_mem = 0.5, 0.5
                avail_cpu, avail_mem = 50.0, 50.0

            # [Feature 0-1] 资源状态
            features = [norm_cpu, norm_mem]

            # [Feature 2] 源节点
            features.append(1.0 if node == source else 0.0)

            # --- 🔥 关键修复 1：Pending Dest 逻辑 ---
            is_dest = node in dests
            is_connected = node in connected_dests

            # [Feature 3] 待连接目的地（强化版）
            if is_dest:
                if is_connected:
                    is_pending = -1.0  # 已连接：负信号
                else:
                    if is_vnf_phase:
                        is_pending = 0.5  # VNF阶段：中等正信号
                    else:
                        is_pending = 2.0  # 连接阶段：强正信号
            else:
                is_pending = 0.0
            features.append(is_pending)

            # [Feature 4] 已连接目的地
            features.append(1.0 if is_connected else 0.0)

            # [Feature 5] 在树上
            features.append(1.0 if node in nodes_on_tree else 0.0)

            # --- 🔥 关键修复 2：资源匹配度（梯度版）---
            # [Feature 6] 资源匹配分数（0.0-1.0）
            if is_vnf_phase and req_cpu > 0:
                # 计算CPU和内存的匹配度
                cpu_match = min(avail_cpu / max(req_cpu, 0.1), 1.0)
                mem_match = min(avail_mem / max(req_mem, 0.1), 1.0)

                # 综合匹配度（资源越充足，分数越高）
                match_score = 0.7 * cpu_match + 0.3 * mem_match

                # 如果是DC节点，额外奖励
                is_dc_node = node in dc_nodes
                if is_dc_node:
                    match_score = min(match_score + 0.1, 1.0)  # DC节点加分
            else:
                match_score = 0.0
            features.append(match_score)

            # [Feature 7] 度数中心性
            try:
                degree = len(self.env.resource_mgr.get_neighbors(node))
                norm_degree = min(degree / 10.0, 1.0)
            except:
                norm_degree = 0.5
            features.append(norm_degree)

            # [Feature 8] 历史负载
            features.append(min(node_vnf_counts[node] / 5.0, 1.0))

            # [Feature 9] 综合评分
            # 使用资源利用率（1 - 空闲率）作为评分
            cpu_util = 1.0 - norm_cpu
            mem_util = 1.0 - norm_mem
            score = (cpu_util * 0.5 + mem_util * 0.5) * 0.8 + norm_degree * 0.2
            features.append(min(score, 1.0))

            # --- 🔥 新增：阶段感知特征 ---
            # [Feature 10] 阶段指导
            # --- 🔥 新增：阶段感知特征 ---
            # [Feature 10] 阶段指导（强化版）
            if is_vnf_phase:
                # VNF部署阶段
                if node in dc_nodes and avail_cpu >= req_cpu and avail_mem >= req_mem:
                    phase_guide = 1.5  # 能部署当前VNF：强正信号
                elif node in dc_nodes:
                    phase_guide = 0.5  # DC节点但资源不足：中等正信号
                else:
                    phase_guide = -0.3  # 非DC节点：负信号
            else:
                # 路由连接阶段 - 关键修改！
                if is_dest:
                    if is_connected:
                        phase_guide = -3.0  # 已连接dest：极强负信号（强烈避免）
                    else:
                        phase_guide = 3.0  # 未连接dest：极强正信号（优先选择）
                elif node == source:
                    phase_guide = 0.5  # 源节点：中等正信号
                else:
                    # 普通节点：基于到最近未连接dest的距离给予信号
                    pending_dests = [d for d in dests if d not in connected_dests]
                    if pending_dests:
                        # 简单启发式：计算到最近未连接dest的"距离"
                        min_dist = float('inf')
                        for dest in pending_dests:
                            if node == dest:
                                min_dist = 0
                                break
                            # 简单距离估算
                            has_direct_link = False
                            if hasattr(self.env, 'resource_mgr'):
                                try:
                                    # 通过 resource_mgr 检查是否有直接链路
                                    neighbors = self.env.resource_mgr.get_neighbors(node)
                                    has_direct_link = dest in neighbors
                                except:
                                    # 回退：检查拓扑矩阵
                                    if hasattr(self.env, 'topology'):
                                        has_direct_link = self.env.topology[node][dest] > 0
                                    else:
                                        has_direct_link = False

                            if has_direct_link:
                                dist = 1
                            else:
                                # 使用节点ID差的绝对值作为距离代理
                                dist = abs(node - dest) + 2
                            if dist < min_dist:
                                min_dist = dist

                        if min_dist == 0:
                            phase_guide = 2.5  # 自己就是目标节点
                        elif min_dist == 1:
                            phase_guide = 1.0  # 直接邻居
                        elif min_dist <= 3:
                            phase_guide = 0.0  # 较近节点
                        else:
                            phase_guide = -0.5  # 较远节点
                    else:
                        # 所有dest都已连接
                        phase_guide = -1.0
            features.append(phase_guide)
            if is_dest:
                status_str = "✅已完成" if is_connected else "❌未完成"
                logger.info(f"🧐 [Node {node}] {status_str} | "
                            f"Feat[3](Pending): {features[3]:.1f} | "
                            f"Feat[4](Conn): {features[4]:.1f} | "
                            f"Feat[10](Guide): {features[10]:.1f} | "
                            f"CPU: {features[0]:.2f}")
            x.append(features)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        # 2. 边特征（完整版）
        edge_index_list = []
        edge_attr_list = []

        for u in range(n):
            try:
                # 🔥 修复：通过 resource_mgr 获取邻居
                if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                    neighbors = self.env.resource_mgr.get_neighbors(u)
                else:
                    # 回退：从拓扑矩阵获取
                    neighbors = []
                    if hasattr(self.env, 'topology'):
                        for v in range(n):
                            if v != u and self.env.topology[u][v] > 0:
                                neighbors.append(v)
                    elif hasattr(self.env, 'graph'):
                        for v in range(n):
                            if v != u and self.env.graph[u][v] > 0:
                                neighbors.append(v)
                    else:
                        continue

                if not neighbors:
                    continue

                for v in neighbors:
                    if u < v:  # 避免重复边
                        # 无向边，添加两个方向
                        edge_index_list.append([u, v])
                        edge_index_list.append([v, u])

                        # 获取带宽信息
                        try:
                            # 🔥 修复：从 resource_mgr 获取带宽
                            if hasattr(self.env.resource_mgr, 'pool'):
                                pool = self.env.resource_mgr.pool
                                key = tuple(sorted((u, v)))

                                # 尝试多种方式获取带宽
                                if hasattr(pool, 'B') and key in pool.B:
                                    available_bw = pool.B[key]
                                elif hasattr(pool, 'bandwidth') and key in pool.bandwidth:
                                    available_bw = pool.bandwidth[key]
                                elif hasattr(pool, 'get_available_bandwidth'):
                                    available_bw = pool.get_available_bandwidth(u, v)
                                else:
                                    available_bw = 50.0  # 默认值

                                # 归一化
                                norm_bw = min(available_bw / 100.0, 1.0)

                                # 边是否在当前树中
                                is_in_tree = 0.0
                                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                                    tree = self.env.current_tree.get('tree', {})
                                    edge_key = (min(u, v), max(u, v))
                                    if edge_key in tree:
                                        is_in_tree = 1.0

                                edge_attr_list.append([norm_bw, is_in_tree])
                                edge_attr_list.append([norm_bw, is_in_tree])
                            else:
                                edge_attr_list.append([0.5, 0.0])
                                edge_attr_list.append([0.5, 0.0])

                        except Exception as e:
                            # 如果获取失败，使用默认值
                            edge_attr_list.append([0.5, 0.0])
                            edge_attr_list.append([0.5, 0.0])
            except Exception as e:
                logger.debug(f"构建边特征时出错: {e}")
                continue

        if len(edge_index_list) > 0:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)

        # =============================
        # 3. 全局特征 [1, 5]
        # =============================
        # 带宽需求归一化
        bw_req = req.get('bw_origin', 0.0)
        norm_bw_req = min(bw_req / 10.0, 1.0)

        # VNF进度
        deployed_count = next_vnf_idx  # 已部署的VNF数量
        total_vnf = len(vnf_list)
        vnf_progress = deployed_count / max(1, total_vnf)

        # 目的地进度
        total_dests = len(dests)
        dest_progress = len(connected_dests) / max(1, total_dests)

        # 阶段特征
        phase_feat = 0.0 if is_vnf_phase else 1.0

        # 树大小
        tree_size = len(nodes_on_tree) / max(1, n)

        # 资源紧张度（所有节点的平均CPU利用率）
        total_avail_cpu = sum([self.env.resource_mgr.pool.get_available_cpu(i) for i in range(n)])
        resource_tension = 1.0 - (total_avail_cpu / (n * 100.0))

        global_attr = torch.tensor([[
            norm_bw_req,  # 带宽需求强度
            vnf_progress,  # VNF部署进度
            dest_progress,  # 目的地连接进度
            phase_feat,  # 当前阶段
            resource_tension  # 资源紧张度
        ]], dtype=torch.float32)

        return Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr
        )



    def _validate_start_node(self, start_node, target_node):
        """
        验证起点节点的有效性

        Args:
            start_node: 起点节点ID
            target_node: 目标节点ID

        Returns:
            tuple: (is_valid, reason)
        """
        # 1. 基本有效性检查
        if not self._is_valid_node(start_node):
            return False, f"无效节点ID: {start_node}"

        if not self._is_valid_node(target_node):
            return False, f"无效目标节点ID: {target_node}"

        # 2. 起点和目标不能相同（VNF部署阶段除外）
        if self.env.current_phase != 'vnf_deployment' and start_node == target_node:
            return False, f"起点和终点相同: {start_node}"

        # 3. 检查起点是否可达（在路由阶段）
        if self.env.current_phase == 'destination_connection':
            # 如果起点不在树上，检查是否可达
            nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
            if start_node not in nodes_on_tree:
                # 检查从源节点到起点的可达性
                source = self.env.current_request.get('source', 0)
                try:
                    hop_to_start = self._get_hop_distance(source, start_node)
                    if hop_to_start >= 9999:
                        return False, f"起点{start_node}不可达"
                except:
                    pass

        return True, "有效"

    def _is_valid_node(self, node):
        """验证节点是否有效（健壮版）"""
        try:
            node = int(node)
            if node < 0 or node >= self.env.n:
                return False

            # 额外检查：节点是否在拓扑中存在
            if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                try:
                    # 尝试获取邻居，如果节点不存在会抛出异常
                    _ = self.env.resource_mgr.get_neighbors(node)
                    return True
                except:
                    return False
            return True
        except:
            return False

    def _get_hop_distance(self, node1, node2):
        """计算跳数（健壮版）"""
        # 快速检查
        if node1 == node2:
            return 0
        if not self._is_valid_node(node1) or not self._is_valid_node(node2):
            return 9999

        # 构建图
        try:
            G = nx.Graph()
            for u in range(self.env.n):
                try:
                    # 🔥 修复：通过 resource_mgr 获取邻居
                    if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                        neighbors = self.env.resource_mgr.get_neighbors(u)
                    elif hasattr(self.env, 'topology'):
                        # 从拓扑矩阵获取邻居
                        neighbors = []
                        for v in range(self.env.n):
                            if v != u and self.env.topology[u][v] > 0:
                                neighbors.append(v)
                    else:
                        continue

                    if neighbors:
                        for v in neighbors:
                            if self._is_valid_node(v):
                                G.add_edge(u, v)
                except Exception as e:
                    logger.debug(f"获取节点 {u} 的邻居失败: {e}")
                    continue

            # 计算最短路径
            if G.has_node(node1) and G.has_node(node2):
                return nx.shortest_path_length(G, node1, node2)
            else:
                return 9999
        except Exception as e:
            logger.debug(f"[Hop Distance] 计算失败: {e}")
            return 9999
    def _get_total_vnf_progress(self):
        """获取VNF部署总进度 (兼容旧接口)"""
        if not self.env.current_request: return 0
        vnf_list = self.env.current_request.get('vnf', [])
        return min(getattr(self.env, 'next_vnf_idx', 0), len(vnf_list))

    def _is_all_tasks_completed(self):
        """
        检查是否所有任务完成（VNF部署 + 目的地连接）
        返回: (bool, str) - 是否完成, 完成状态描述
        """
        if not self.env.current_request:
            return True, "无请求"

        # 检查VNF
        vnf_list = self.env.current_request.get('vnf', [])
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_done = next_vnf_idx >= len(vnf_list)

        # 检查目的地
        dests = set(self.env.current_request.get('dest', []))
        connected = set(self.env.current_tree.get('connected_dests', set()))
        dests_done = dests.issubset(connected)

        if vnf_done and dests_done:
            return True, "所有任务完成"
        elif vnf_done:
            return False, f"VNF完成，还有{len(dests) - len(connected)}个目的地待连接"
        elif dests_done:
            return False, f"目的地完成，还有{len(vnf_list) - next_vnf_idx}个VNF待部署"
        else:
            return False, f"VNF进度:{next_vnf_idx}/{len(vnf_list)}，目的地:{len(connected)}/{len(dests)}"