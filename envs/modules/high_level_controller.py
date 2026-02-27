"""
envs/modules/high_level_controller.py
====================================
高层交互控制器 - V55.2 调试增强版（DC节点安全获取 + 动作防火墙）
====================================
核心修复：
1. ✅ DC节点多源安全获取 + 缓存，彻底解决 env.dc_nodes 为空问题
2. ✅ VNF阶段 Mask 只开放 DC 节点，移除任何形式的全开兜底
3. ✅ Step 阶段增加动作防火墙，拦截非法节点（非DC/已完成）并强制截断
4. ✅ 状态特征扩展至17维，保持与低层编码器兼容
5. ✅ 保留树连接度特征，引导高层选择靠近现有树的目标
"""

import numpy as np
import torch
import logging
import networkx as nx
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class HighLevelController:
    """
    🎮 高层交互控制器 (High-Level Controller)
    V55.2 调试增强版 - 安全获取DC节点 + 动作防火墙 + 17维状态特征
    """

    def __init__(self, env):
        """
        初始化控制器
        :param env: 主环境对象的引用 (SFC_HIRL_Env)
        """
        self.env = env
        logger.info("✅ HighLevelController initialized (V55.2 Debug Enhanced)")
        self._pending_dist_cache = {}
        self._last_dests_update = -1
        # 🔥 DC节点缓存，避免重复查找
        self._dc_nodes_cache = None

    # ------------------------------------------------------------------
    # 🔍 安全获取 DC 节点（多源聚合 + 缓存）
    # ------------------------------------------------------------------
    def _get_dc_nodes_safe(self):
        """
        安全获取 DC 节点列表 (多源聚合 + 缓存)
        解决 env.dc_nodes 可能为空或类型不一致的问题
        """
        if self._dc_nodes_cache is not None:
            return self._dc_nodes_cache

        dc_nodes = set()

        # 来源 A: env.dc_nodes
        raw_dc = getattr(self.env, 'dc_nodes', [])
        if raw_dc is not None:
            try:
                for x in raw_dc:
                    dc_nodes.add(int(x))
            except:
                pass

        # 来源 B: resource_mgr (更底层，更可靠)
        if hasattr(self.env, 'resource_mgr'):
            # 情况 1: resource_mgr.dc_nodes
            if hasattr(self.env.resource_mgr, 'dc_nodes'):
                try:
                    for x in self.env.resource_mgr.dc_nodes:
                        dc_nodes.add(int(x))
                except:
                    pass

            # 情况 2: resource_mgr.pool.dc_nodes (物理资源池)
            if hasattr(self.env.resource_mgr, 'pool') and hasattr(self.env.resource_mgr.pool, 'dc_nodes'):
                try:
                    for x in self.env.resource_mgr.pool.dc_nodes:
                        dc_nodes.add(int(x))
                except:
                    pass

        # 结果缓存
        if dc_nodes:
            self._dc_nodes_cache = dc_nodes
            # logger.info(f"🔍 [High] 成功识别 DC 节点: {sorted(list(dc_nodes))}")
        else:
            logger.error("🛑 [High] 严重警告: 无法从任何来源找到 DC 节点！")
            self._dc_nodes_cache = set()  # 空集合，避免重复报错

        return self._dc_nodes_cache

    # ------------------------------------------------------------------
    # 🎭 高层动作掩码生成器（VNF阶段严格封杀非DC节点）
    # ------------------------------------------------------------------
    def get_high_level_action_mask(self):
        """
        🔥 [V55.2] 调试版 Mask 生成器
        1. VNF阶段：严格封杀非DC节点，绝不使用全开兜底。
        2. Routing阶段：基于物理连接状态封杀已完成节点。
        """
        n = self.env.n
        mask = np.zeros(n, dtype=np.float32)

        # 无请求时返回全0（无合法动作）
        if not hasattr(self.env, 'current_request') or self.env.current_request is None:
            return mask

        vnf_list = self.env.current_request.get('vnf', [])
        current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)

        # ============================================================
        # 阶段 1: VNF 部署阶段 (核心死循环点)
        # ============================================================
        if current_vnf_idx < len(vnf_list):
            # 1. 获取 DC 节点 (使用安全方法)
            dc_nodes = self._get_dc_nodes_safe()

            # 2. 设置掩码：只开放 DC 节点
            if dc_nodes:
                valid_count = 0
                for node in dc_nodes:
                    if 0 <= node < n:
                        mask[node] = 1.0
                        valid_count += 1
                    else:
                        logger.error(f"   ❌ DC节点 {node} 越界 (env.n={n})")

                if valid_count == 0:
                    logger.critical(f"🛑 [MaskGen] DC节点均越界或无效! Mask全为0")
            else:
                logger.critical("🛑 [MaskGen] 无法获取 DC 节点列表，Mask 全 0")

            # 🔥🔥🔥 [关键] 移除任何形式的全开兜底，保持全0让Agent报错
            # 如果算出来全0，说明系统配置有误或资源枯竭，此时绝对不能全开

            # 3. 反向检查：确保没有任何非 DC 节点被错误开放
            non_dc_opened = []
            for node in range(n):
                if mask[node] > 0 and node not in dc_nodes:
                    non_dc_opened.append(node)
                    mask[node] = 0.0  # 强制修正

            if non_dc_opened:
                logger.critical(f"🚨 [MaskGen] 严重错误: 非DC节点被错误开放并已修正: {non_dc_opened}")

        # ============================================================
        # 阶段 2: 目的地连接阶段
        # ============================================================
        else:
            try:
                all_dests = {int(x) for x in self.env.current_request.get('dest', [])}
                connected_set = set()
                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_set = {int(x) for x in self.env.current_tree.get('connected_dests', set())}
            except Exception as e:
                logger.error(f"   ❌ 解析目的地异常: {e}")
                all_dests, connected_set = set(), set()

            true_pending = list(all_dests - connected_set)

            if not true_pending:
                return np.zeros(n, dtype=np.float32)  # 所有任务完成

            for node in true_pending:
                if 0 <= node < n:
                    mask[node] = 1.0

            # 绝对封杀已连接节点
            for done_node in connected_set:
                if 0 <= done_node < n:
                    mask[done_node] = 0.0

        return mask

    # ------------------------------------------------------------------
    # 🛑 高层动作执行 + 防火墙拦截
    # ------------------------------------------------------------------
    def step_high_level(self, action_idx):
        """
        🔥 [V55.2] 动作执行与拦截
        增加“动作防火墙”，如果 Mask 失效导致选了非法节点，在此处强制截断。
        """
        target_node = int(action_idx)
        self.current_high_action = target_node

        # --- 获取环境信息 ---
        vnf_list = self.env.current_request.get('vnf', []) if self.env.current_request else []
        current_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)

        # ============================================================
        # 🛑 拦截器 1: VNF 阶段选了非 DC 节点
        # ============================================================
        if current_vnf_idx < len(vnf_list):
            dc_nodes = self._get_dc_nodes_safe()

            if target_node not in dc_nodes:
                logger.error("=" * 40)
                logger.error(f"🚨 [High拦截] 非法部署请求检测到！")
                logger.error(f"   -> 目标节点: {target_node}")
                logger.error(f"   -> 是否DC节点: {target_node in dc_nodes}")
                logger.error(f"   -> 合法DC列表: {sorted(list(dc_nodes))}")
                logger.error(f"   -> 诊断: Agent 选择了被 Mask 封锁的节点。强制 Truncated。")
                logger.error("=" * 40)

                # 返回重罚(-50)和 Truncated(True)，迫使 Coordinator 换目标
                return None, -50.0, False, True, {
                    'subgoal_error': True,
                    'reason': 'not_dc_node',
                    'target': target_node
                }

        # ============================================================
        # 🛑 拦截器 2: Routing 阶段选了已完成节点
        # ============================================================
        else:
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

            if target_node in connected:
                # 检查是否全部完成
                if all_dests.issubset(connected):
                    return None, 20.0, True, False, {'all_done': True}
                else:
                    logger.warning(f"🔄 [High拦截] 目标 {target_node} 已完成。强制截断。")
                    return None, -1.0, False, True, {
                        'subgoal_completed': True,
                        'reason': 'already_connected'
                    }

        # ============================================================
        # ✅ 合法动作，放行
        # ============================================================
        self.env.current_goal_node = target_node
        return None, 0.0, False, False, {'status': 'accepted'}

    # ------------------------------------------------------------------
    # 📍 设置高层目标（起点同步，阶段判定）
    # ------------------------------------------------------------------
    def set_high_level_goal(self, high_action_idx, target_node_id, start_node_id=None):
        """
        🎯 设定高层目标
        """
        # 1. 记录高层动作
        self.env.last_high_action_idx = high_action_idx
        self.env.current_subgoal_node = target_node_id

        # 2. 新目标开始，步数计数器归零
        self.env.subgoal_step_count = 0

        # 3. 同步 Agent 选择的起点
        if start_node_id is not None:
            self.env.current_node_location = int(start_node_id)
            logger.info(f"📍 [High] Agent重置起点: {start_node_id}")

        # 4. 自动判定阶段
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            self.env.current_vnf_to_deploy = self.env.next_vnf_idx

            if self.env.next_vnf_idx < len(vnf_list):
                # VNF 部署阶段
                self.env.current_phase = 'vnf_deployment'
                self.env.current_deployment_target = target_node_id
                self.env.current_target_node = None
                logger.info(f"🎯 [Env] VNF部署阶段: VNF[{self.env.next_vnf_idx}] → 节点{target_node_id}")
            else:
                # 目的地连接阶段
                self.env.current_phase = 'destination_connection'
                self.env.current_target_node = target_node_id
                self.env.current_deployment_target = None
                logger.info(f"🎯 [Env] 目的地连接阶段: 目标节点{target_node_id}")
        else:
            logger.warning("⚠️ 设定目标时没有活跃请求")

        return self.get_high_level_state_graph()

    # ------------------------------------------------------------------
    # 📊 构建高层状态图 (17维节点特征 + 边特征 + 全局特征)
    # ------------------------------------------------------------------
    def get_high_level_state_graph(self):
        """
        🎯 构建高层 GNN 状态 (17维特征，包含树连接度)
        特征索引：
        0: CPU 归一化
        1: MEM 归一化
        2: 是否为源节点
        3: 待连接目的地信号（-1已连接, 0非目标, 0.5/2.0待连接）
        4: 是否已连接目的地
        5: 是否在树上
        6: 资源匹配度
        7: 度数中心性
        8: 历史VNF负载
        9: 综合评分
        10: 阶段指导信号
        11: 树连接度
        12-16: 预留/填充位
        """
        n = self.env.n

        # 无请求时返回零张量
        if not self.env.current_request:
            return Data(
                x=torch.zeros((n, 17), dtype=torch.float32),
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

        # 树/连接信息
        connected_dests = self.env.current_tree.get('connected_dests', set())
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        # 阶段信息
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        is_vnf_phase = next_vnf_idx < len(vnf_list)

        # VNF需求
        req_cpu, req_mem = 0.0, 0.0
        if is_vnf_phase:
            cpu_list = req.get('cpu_origin') or req.get('vnf_cpu', [])
            mem_list = req.get('memory_origin') or req.get('vnf_mem', [])
            if next_vnf_idx < len(cpu_list):
                req_cpu = float(cpu_list[next_vnf_idx])
                req_mem = float(mem_list[next_vnf_idx]) if next_vnf_idx < len(mem_list) else 1.0

        # VNF负载计数
        node_vnf_counts = [0] * n
        if hasattr(self.env, 'current_placements') and self.env.current_placements:
            for placement_key in self.env.current_placements:
                node_id, _ = placement_key
                if 0 <= node_id < n:
                    node_vnf_counts[node_id] += 1

        # DC节点集合（安全获取）
        dc_nodes = self._get_dc_nodes_safe()

        # =============================
        # 构建节点特征 [N, 17]
        # =============================
        x_list = []

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

            # --- 特征0-2: 资源 + 源节点 ---
            features = [norm_cpu, norm_mem, 1.0 if node == source else 0.0]

            # --- 特征3: 待连接目的地信号 ---
            is_dest = node in dests
            is_connected = node in connected_dests
            if is_dest:
                if is_connected:
                    pending_signal = -1.0
                else:
                    pending_signal = 0.5 if is_vnf_phase else 2.0
            else:
                pending_signal = 0.0
            features.append(pending_signal)

            # --- 特征4: 已连接目的地 ---
            features.append(1.0 if is_connected else 0.0)

            # --- 特征5: 在树上 ---
            features.append(1.0 if node in nodes_on_tree else 0.0)

            # --- 特征6: 资源匹配度 ---
            match_score = 0.0
            if is_vnf_phase and req_cpu > 0:
                cpu_match = min(avail_cpu / max(req_cpu, 0.1), 1.0)
                mem_match = min(avail_mem / max(req_mem, 0.1), 1.0)
                match_score = 0.7 * cpu_match + 0.3 * mem_match
                if node in dc_nodes:
                    match_score = min(match_score + 0.1, 1.0)
            features.append(match_score)

            # --- 特征7: 度数中心性 ---
            try:
                degree = len(self.env.resource_mgr.get_neighbors(node))
                norm_degree = min(degree / 10.0, 1.0)
            except:
                norm_degree = 0.5
            features.append(norm_degree)

            # --- 特征8: 历史VNF负载 ---
            features.append(min(node_vnf_counts[node] / 5.0, 1.0))

            # --- 特征9: 综合评分（资源利用率+度数）---
            cpu_util = 1.0 - norm_cpu
            mem_util = 1.0 - norm_mem
            score = (cpu_util * 0.5 + mem_util * 0.5) * 0.8 + norm_degree * 0.2
            features.append(min(score, 1.0))

            # --- 特征10: 阶段指导信号 ---
            if is_vnf_phase:
                if node in dc_nodes:
                    if avail_cpu >= req_cpu and avail_mem >= req_mem:
                        phase_guide = 1.5
                    else:
                        phase_guide = 0.5
                else:
                    phase_guide = -0.3
            else:
                if is_dest:
                    if is_connected:
                        phase_guide = -3.0
                    else:
                        phase_guide = 3.0
                elif node == source:
                    phase_guide = 0.5
                else:
                    phase_guide = 0.0
            features.append(phase_guide)

            # --- 特征11: 树连接度 ---
            try:
                if nodes_on_tree:
                    min_dist = float('inf')
                    for tn in nodes_on_tree:
                        if tn == node:
                            min_dist = 0
                            break
                        dist = self._get_hop_distance(node, tn)
                        if dist < min_dist:
                            min_dist = dist
                    if min_dist == 0:
                        tree_conn = 1.0
                    elif min_dist <= 2:
                        tree_conn = 0.7
                    elif min_dist <= 4:
                        tree_conn = 0.3
                    else:
                        tree_conn = 0.0
                else:
                    tree_conn = 0.0
            except:
                tree_conn = 0.0
            features.append(tree_conn)

            # --- 特征12-16: 预留填充位（保持17维）---
            while len(features) < 17:
                features.append(0.0)

            x_list.append(features)

        x_tensor = torch.tensor(x_list, dtype=torch.float32)

        # =============================
        # 边特征
        # =============================
        edge_index = getattr(self.env, 'edge_index', torch.zeros((2, 0), dtype=torch.long))
        edge_attr = getattr(self.env, 'edge_attr', torch.zeros((edge_index.size(1), 2), dtype=torch.float32))

        # =============================
        # 全局特征 [1, 5]
        # =============================
        bw_req = req.get('bw_origin', 0.0)
        norm_bw_req = min(bw_req / 10.0, 1.0)

        total_vnf = len(vnf_list)
        vnf_progress = next_vnf_idx / max(1, total_vnf)

        total_dests = len(dests)
        dest_progress = len(connected_dests) / max(1, total_dests)

        phase_feat = 0.0 if is_vnf_phase else 1.0

        tree_size = len(nodes_on_tree) / max(1, n)

        total_avail_cpu = 0.0
        try:
            total_avail_cpu = sum([self.env.resource_mgr.pool.get_available_cpu(i) for i in range(n)])
        except:
            total_avail_cpu = n * 50.0
        resource_tension = 1.0 - (total_avail_cpu / (n * 100.0))

        global_attr = torch.tensor([[
            norm_bw_req,
            vnf_progress,
            dest_progress,
            phase_feat,
            resource_tension
        ]], dtype=torch.float32)

        # =============================
        # 当前动作掩码（用于辅助编码器）
        # =============================
        action_mask = torch.from_numpy(self.get_high_level_action_mask()).bool().unsqueeze(0)

        return Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_attr=global_attr,
            action_mask=action_mask
        )

    # ------------------------------------------------------------------
    # 🔧 辅助工具方法（验证、距离、进度检查等）
    # ------------------------------------------------------------------
    def _validate_start_node(self, start_node, target_node):
        """验证起点节点的有效性"""
        if not self._is_valid_node(start_node):
            return False, f"无效节点ID: {start_node}"

        if not self._is_valid_node(target_node):
            return False, f"无效目标节点ID: {target_node}"

        if self.env.current_phase != 'vnf_deployment' and start_node == target_node:
            return False, f"起点和终点相同: {start_node}"

        if self.env.current_phase == 'destination_connection':
            nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
            if start_node not in nodes_on_tree:
                source = self.env.current_request.get('source', 0)
                try:
                    hop_to_start = self._get_hop_distance(source, start_node)
                    if hop_to_start >= 9999:
                        return False, f"起点{start_node}不可达"
                except:
                    pass

        return True, "有效"

    def _is_valid_node(self, node):
        """验证节点是否有效"""
        try:
            node = int(node)
            if node < 0 or node >= self.env.n:
                return False

            if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                try:
                    _ = self.env.resource_mgr.get_neighbors(node)
                    return True
                except:
                    return False
            return True
        except:
            return False

    def _get_hop_distance(self, node1, node2):
        """计算跳数（基于 NetworkX）"""
        if node1 == node2:
            return 0
        if not self._is_valid_node(node1) or not self._is_valid_node(node2):
            return 9999

        try:
            G = nx.Graph()
            for u in range(self.env.n):
                try:
                    if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'get_neighbors'):
                        neighbors = self.env.resource_mgr.get_neighbors(u)
                    elif hasattr(self.env, 'topology'):
                        neighbors = [v for v in range(self.env.n) if v != u and self.env.topology[u][v] > 0]
                    else:
                        continue

                    if neighbors:
                        for v in neighbors:
                            if self._is_valid_node(v):
                                G.add_edge(u, v)
                except Exception as e:
                    logger.debug(f"获取节点 {u} 的邻居失败: {e}")
                    continue

            if G.has_node(node1) and G.has_node(node2):
                return nx.shortest_path_length(G, node1, node2)
            else:
                return 9999
        except Exception as e:
            logger.debug(f"[Hop Distance] 计算失败: {e}")
            return 9999

    def _get_total_vnf_progress(self):
        """获取VNF部署总进度"""
        if not self.env.current_request:
            return 0
        vnf_list = self.env.current_request.get('vnf', [])
        return min(getattr(self.env, 'next_vnf_idx', 0), len(vnf_list))

    def _is_all_tasks_completed(self):
        """检查是否所有任务完成"""
        if not self.env.current_request:
            return True, "无请求"

        vnf_list = self.env.current_request.get('vnf', [])
        next_vnf_idx = getattr(self.env, 'next_vnf_idx', 0)
        vnf_done = next_vnf_idx >= len(vnf_list)

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

    def _is_all_completed(self):
        """辅助检查（兼容旧接口）"""
        completed, _ = self._is_all_tasks_completed()
        return completed