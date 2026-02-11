"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - 强制在线模式完整修复版
====================================
修复内容：
1. ✅ 统一生命周期托管入口，只有一处 add_request 调用
2. ✅ 修复 _verify_lifecycle_tracking 的兜底逻辑
3. ✅ 改进 _should_release_immediately 的安全逻辑
4. ✅ 优化 phase reset 时机，避免状态污染
5. ✅ 修复 connected_dests 不可变性
6. ✅ 改进动作掩码和移动奖励逻辑
"""

import numpy as np
import torch
import logging
from torch_geometric.data import Data
import copy
import os
import time
from envs.modules.visualize_multicast_tree import SFCVisualizer
logger = logging.getLogger(__name__)


class LowLevelController:
    """
    🕹️ 低层执行控制器 (Low-Level Controller) - 强制在线模式完整修复版

    核心原则：
    1. 🔄 生命周期管理只有一条路径
    2. 📝 账本归档和生命周期托管分离
    3. ⏰ 资源释放由 RequestLifecycleManager 统一处理
    """

    def __init__(self, env):
        """
        初始化控制器 - 强制在线模式版
        :param env: 主环境对象的引用 (SFC_HIRL_Env)
        """
        self.env = env

        # 1. 检查资源管理器
        if not hasattr(self.env, 'resource_mgr'):
            logger.error("❌ LowLevelController 初始化错误: env 中未找到 resource_mgr。")
            raise RuntimeError("resource_mgr 必须配置")

        # ==========================================
        # 🔥【新增修复】自动注入/挂载 RequestLifecycleManager
        # ==========================================

        # 情况A: env中没有，但resource_mgr中有 (这是我们在第1步修改后的情况)
        if not hasattr(self.env, 'request_manager'):
            if hasattr(self.env.resource_mgr, 'request_manager'):
                self.env.request_manager = self.env.resource_mgr.request_manager
                logger.info("✅ [Init] 成功从 resource_mgr 挂载 request_manager 到 env")
            else:
                # 情况B: 都没有，尝试动态导入并创建 (兜底方案)
                try:
                    # 假设 RequestLifecycleManager 定义在 resource_mgr 的模块里
                    req_mgr_class = self.env.resource_mgr.__class__.__module__
                    import sys
                    mod = sys.modules[req_mgr_class]
                    if hasattr(mod, 'RequestLifecycleManager'):
                        RM_Class = getattr(mod, 'RequestLifecycleManager')
                        self.env.request_manager = RM_Class(self.env.resource_mgr)
                        self.env.resource_mgr.request_manager = self.env.request_manager
                        logger.warning("⚠️ [Init] 检测到缺失，已自动创建并注入 RequestLifecycleManager")
                except Exception as e:
                    logger.error(f"❌ [Init] 无法自动创建 request_manager: {e}")

        # 最终检查
        if not hasattr(self.env, 'request_manager') or self.env.request_manager is None:
            logger.error("❌ [致命错误] request_manager 初始化失败，在线模式将失效！")
        else:
            logger.info("✅ [LowLevelController] 初始化完成（强制在线模式完整修复版）")

    def step_low_level(self, action):
        """
        🔥 低层步进函数 - 强制在线模式版
        """
        if self.env.current_request is None:
            return self.get_state(), -10.0, True, False, {'error': 'no_req'}

        if not hasattr(self.env, 'subgoal_step_count'):
            self.env.subgoal_step_count = 0
        self.env.subgoal_step_count += 1

        # 1. 超时检查
        if self.env.subgoal_step_count > getattr(self.env, 'max_subgoal_steps', 50):
            self._reset_phase_state()
            return self.get_state(), -1.0, False, True, {'timeout': True}

        current_node = self.env.current_node_location
        target_action = int(action)
        is_stay = (target_action == current_node)

        # 2.1 阶段1：VNF 部署
        if self.env.current_phase == 'vnf_deployment':
            return self._handle_vnf_deployment(current_node, target_action, is_stay)

        # 2.2 阶段2：目的地连接
        elif self.env.current_phase == 'destination_connection':
            return self._handle_destination_connection(current_node, target_action, is_stay)

        return self.get_state(), -10.0, True, False, {'error': 'unknown_phase'}

    def _handle_vnf_deployment(self, current_node, target_action, is_stay):
        """
        处理 VNF 部署阶段 - 修复版
        """
        target_goal = getattr(self.env, 'current_deployment_target', None)

        # A. 移动逻辑
        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        # B. 部署逻辑
        if target_goal is not None and current_node == target_goal:
            # 同步状态给资源管理器
            if hasattr(self.env, 'resource_mgr'):
                self.env.resource_mgr.current_request = self.env.current_request
                self.env.resource_mgr.current_tree = self.env.current_tree
                self.env.resource_mgr.current_phase = self.env.current_phase
                self.env.resource_mgr.next_vnf_idx = self.env.next_vnf_idx

            # 调用部署方法
            deploy_success = False
            if hasattr(self.env.resource_mgr, '_try_deploy'):
                deploy_success = self.env.resource_mgr._try_deploy(target_goal)
            elif hasattr(self.env, '_try_deploy'):
                deploy_success = self.env._try_deploy(target_goal)

            if deploy_success:
                self.env.next_vnf_idx += 1
                vnf_list = self.env.current_request.get('vnf', [])
                current_count = self.env.next_vnf_idx

                # 构建返回信息
                info = {}
                reward = 0.0
                done = False
                truncated = True

                if current_count >= len(vnf_list):
                    info = {
                        'phase': 'vnf_complete',
                        'all_vnf_deployed': True,
                        'deployed_count': current_count,
                        'total_vnf': len(vnf_list)
                    }
                    reward = 20.0
                    # VNF全部部署完成，重置阶段
                    self._reset_vnf_phase_only()
                else:
                    info = {
                        'vnf_deployed': True,
                        'vnf_idx': self.env.next_vnf_idx - 1,
                        'deployed_count': current_count,
                        'total_vnf': len(vnf_list)
                    }
                    reward = 10.0
                    # 单个VNF部署完成，不清除阶段，等待下一个

                return self.get_state(), reward, done, truncated, info
            else:
                info = {
                    'deploy_fail': True,
                    'vnf_idx': self.env.next_vnf_idx,
                    'reason': 'resource_insufficient'
                }
                reward = -5.0
                self._reset_vnf_phase_only()
                return self.get_state(), reward, False, True, info
        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}

    def _handle_destination_connection(self, current_node, target_action, is_stay):
        """
        🔥 [V40.11 修复版] 目的地连接逻辑 (修复类型匹配问题)
        """
        target_goal = getattr(self.env, 'current_target_node', None)

        # A. 移动逻辑
        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        # B. 连接逻辑
        if current_node == target_goal:
            # 1. 初始化集合
            if 'connected_dests' not in self.env.current_tree:
                self.env.current_tree['connected_dests'] = set()

            # 2. 只有新连接才给奖励
            if target_goal not in self.env.current_tree['connected_dests']:
                self.env.current_tree['connected_dests'].add(target_goal)
                step_reward = 10.0
            else:
                step_reward = 0.0

            # ------------------------------------------------------------
            # 🔥 [修复 3] 严格类型转换与判定
            # ------------------------------------------------------------
            try:
                # 强制转换为 int 集合，防止 str/int 混用导致的判定失败
                all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
                connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
            except Exception as e:
                logger.error(f"❌ 类型转换失败: {e}")
                all_dests = set()
                connected = set()

            # 调试日志
            logger.debug(f"📊 [连接状态] 已连接: {len(connected)}/{len(all_dests)} | 剩余: {all_dests - connected}")

            # 3. 严格收敛判定
            if all_dests.issubset(connected) and len(all_dests) > 0:
                logger.info("=" * 60)
                logger.info("✅ [Episode完成] 所有目的地已连接 - 立即终止")

                # ... (归档与可视化代码保持不变) ...
                self._archive_episode_success_only()
                self._add_request_to_lifecycle_manager()

                # 构建 Info
                info = {
                    'episode_complete': True,
                    'all_destinations_connected': True,  # 显式标记
                    'success': True,
                    'connected_count': len(connected)
                }

                # 最终大奖 +50
                return self.get_state(), 50.0, True, False, info

            else:
                # 还没连完，但当前这个子目标完成了
                self.env.subgoal_step_count = 0
                self.env.current_target_node = None
                return self.get_state(), step_reward, False, True, {'dest_connected': True}

        else:
            return self.get_state(), -0.5, False, False, {'warning': 'wait_for_stay'}
    def _calculate_tree_metrics(self):
        """
        📊 [监控] 计算树的健康度指标
        用于判断 Agent 是否学会了"克制"，还是在"乱修路"
        """
        tree_edges = self.env.current_tree.get('tree', {})

        # 统计活跃节点（通过边反推，避免孤立点干扰）
        active_nodes = set()
        for u, v in tree_edges.keys():
            active_nodes.add(u)
            active_nodes.add(v)

        n_edges = len(tree_edges)
        n_nodes = len(active_nodes)

        # 1. 冗余度 (Redundancy Ratio)
        # 树的定义：E = V - 1。如果 E > V - 1，说明有环。
        if n_nodes > 1:
            theoretical_min_edges = n_nodes - 1
            excess_edges = max(0, n_edges - theoretical_min_edges)
            redundancy_ratio = excess_edges / max(1, theoretical_min_edges)
        else:
            redundancy_ratio = 0.0

        # 2. 分支因子 (Branching Factor) - 判断是像"线"还是像"网"
        avg_degree = (2 * n_edges) / max(1, n_nodes)

        # 3. 构建效率 (Construction Efficiency)
        # 简单的效率计算
        build_efficiency = n_edges / max(1, getattr(self.env, 'subgoal_step_count', 1))

        return {
            'tree_n_nodes': n_nodes,
            'tree_n_edges': n_edges,
            'redundancy': float(f"{redundancy_ratio:.2f}"),  # 越低越好 (0.0 是完美树)
            'avg_degree': float(f"{avg_degree:.2f}"),  # ~2.0 是线，很高说明是网
            'efficiency': float(f"{build_efficiency:.2f}")  # 越高越好
        }
    def _should_release_immediately(self):
        """
        🔥 判断是否应该立即释放资源 - 安全版

        优先级：
        1. 如果明确配置了 force_online_mode=True → 强制在线（False）
        2. 如果有 request_manager → 在线模式（False）
        3. 否则 → 立即释放（True）
        """
        # 1. 检查强制在线模式标志
        if hasattr(self.env, 'force_online_mode') and self.env.force_online_mode:
            logger.debug(f"   [模式检测] force_online_mode=True → 强制在线模式")
            return False

        # 2. 检查请求管理器
        if not hasattr(self.env, 'request_manager') or self.env.request_manager is None:
            logger.warning("⚠️ [模式检测] 未找到 request_manager，被迫立即释放资源")
            return True

        # 3. 默认：在线模式
        logger.debug(f"   [模式检测] 检测到 request_manager → 在线模式")
        return False

    def _archive_episode_success_only(self):
        """
        🔥 只做账本归档，不碰生命周期管理
        """
        logger.info("📝 [归档] 开始归档Episode账本...")

        # 方案1: 调用 AllResourceManager 的 _archive_request
        if hasattr(self.env, 'resource_mgr') and \
           hasattr(self.env.resource_mgr, '_archive_request'):
            try:
                self.env.resource_mgr._archive_request(success=True, already_rolled_back=False)
                logger.info("✅ [归档] 账本归档成功")
                return
            except Exception as e:
                logger.error(f"❌ [归档] resource_mgr 归档失败: {e}")
                import traceback
                traceback.print_exc()

        # 方案2: 手动保存账本
        logger.warning("⚠️ [归档] 没有找到 _archive_request 方法，使用手动保存账本")
        self._manual_save_resources_only()

    def _manual_save_resources_only(self):
        """
        🔥 只做账本保存，不碰生命周期
        """
        if self.env.current_request is None:
            logger.warning("⚠️ [归档] current_request 为空，无法保存")
            return

        # 保存资源账本
        if 'resources_allocated' not in self.env.current_request:
            self.env.current_request['resources_allocated'] = {
                'placement': copy.deepcopy(self.env.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.env.current_tree.get('tree', {}))
            }

    def _add_request_to_lifecycle_manager(self):
        """
        🔥 统一的生命周期托管入口 (已修复 register_request 调用)
        """
        if not hasattr(self.env, 'request_manager') or not self.env.request_manager:
            logger.error("❌ [生命周期] 缺少 request_manager，无法托管")
            return False

        if self.env.current_request is None:
            logger.error("❌ [生命周期] current_request 为空")
            return False

        req_id = self.env.current_request.get('id', id(self.env.current_request))

        # 检查是否已存在
        if req_id in self.env.request_manager.active_requests:
            logger.warning(f"⚠️ [生命周期] 请求 {req_id} 已在生命周期管理中")
            return True

        # 准备资源信息
        resources = self._collect_allocated_resources()

        # 获取时间参数
        arrival_time = self.env.current_request.get('arrival_time', 0)
        lifetime = self.env.current_request.get('lifetime', 50)

        try:
            # ====================================================
            # 🔥 修复开始：构建字典并调用 register_request
            # ====================================================

            # 1. 临时构建一个包含必要信息的 request 字典
            # RequestLifecycleManager.register_request 需要从字典里读取 id, arrival_time, lifetime
            temp_request_dict = {
                'id': req_id,
                'arrival_time': arrival_time,
                'lifetime': lifetime
            }

            # 2. 调用正确的接口名 register_request
            success = self.env.request_manager.register_request(
                request=temp_request_dict,
                resources_allocated=resources
            )
            # ====================================================
            # 🔥 修复结束
            # ====================================================

            if success:
                logger.info(f"✅ [生命周期] 请求 {req_id} 托管成功")
                logger.info(f"   → 到达时间: {arrival_time:.2f}")
                logger.info(f"   → 存活时长: {lifetime:.2f}")
                logger.info(f"   → 过期时间: {arrival_time + lifetime:.2f}")
                return True
            else:
                logger.error(f"❌ [生命周期] 请求 {req_id} 托管失败")
                return False

        except AttributeError as e:
            # 如果 request_manager 没有 register_request 方法，可能是旧版本
            logger.warning(f"⚠️ [生命周期] 方法名错误或版本不匹配: {e}")
            return False

        except Exception as e:
            logger.error(f"❌ [生命周期] 托管异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _collect_allocated_resources(self):
        """
        🔥 收集当前请求分配的资源
        """
        resources = {
            'placement': {},
            'tree': {},
            'bandwidth': self.env.current_request.get('bw_origin', 1.0),
            'cpu_requirements': self.env.current_request.get('cpu_origin', []),
            'memory_requirements': self.env.current_request.get('memory_origin', [])
        }

        # 从 current_tree 收集
        if self.env.current_tree:
            if 'placement' in self.env.current_tree:
                resources['placement'] = copy.deepcopy(self.env.current_tree['placement'])
            if 'tree' in self.env.current_tree:
                resources['tree'] = copy.deepcopy(self.env.current_tree['tree'])

        # 如果 request 中有资源账本，合并
        if 'resources_allocated' in self.env.current_request:
            req_res = self.env.current_request['resources_allocated']
            if 'placement' in req_res:
                resources['placement'].update(copy.deepcopy(req_res['placement']))
            if 'tree' in req_res:
                resources['tree'].update(copy.deepcopy(req_res['tree']))

        return resources

    def _handle_movement(self, current_node, target_action, target_goal):
        """
        🔥 [V40.11 成本感知版] 移动处理逻辑

        策略：
        1. 差异化定价：复用现有链路便宜 (-0.05)，开辟新链路贵 (-0.5)。
        2. 资源检查：Fail Fast (资源不足直接挂掉，防止死循环)。
        """
        next_node = int(target_action)

        # 1. 原地不动检查
        if next_node == current_node:
            if current_node != target_goal:
                # 检查是否被困死 (无可用带宽邻居)
                neighbors = self.env.resource_mgr.get_neighbors(current_node)
                bw_req = self.env.current_request.get('bw_origin', 0.0)
                valid_neighbors = [n for n in neighbors if
                                   self.env.resource_mgr.pool.get_available_bandwidth(current_node, n) >= bw_req]

                if not valid_neighbors:
                    logger.warning(f"⚠️ [Resource] 节点 {current_node} 被困死 (Trapped)")
                    return self.get_state(), -10.0, True, False, {'error': 'trapped'}

                # 有路但不走，扣分警告
                return self.get_state(), -1.0, False, False, {'warning': 'stay'}

        # 2. 资源检查 (Fail Fast) - 没带宽就别试了，直接判负
        bw_req = self.env.current_request.get('bw_origin', 0.0)
        try:
            # 注意：此处假设是邻居间移动。如果是多跳，需配合PathEngine。
            # 这里简化为单步检查。
            has_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node) >= bw_req
        except:
            has_bw = False

        if not has_bw:
            logger.warning(f"❌ [Resource] 链路 {current_node}->{next_node} 带宽不足 - Episode Failed")
            return self.get_state(), -10.0, True, False, {'error': 'resource_failure'}

        # ============================================================
        # 🔥 [核心修复] 差异化步数奖励 (Tree Complexity Penalty)
        # ============================================================
        # 判断这条边是否已经在树上
        edge_key = tuple(sorted((current_node, next_node)))
        is_new_edge = edge_key not in self.env.current_tree.get('tree', {})

        if is_new_edge:
            # 🚧 开新路：重罚 (-0.5)
            # 只有当这条新路能带来巨大的连接奖励 (+10) 时，Agent 才会觉得划算
            reward = -0.5
            action_type = "NewPath"
        else:
            # 🛣️ 走老路：轻罚 (-0.05)
            # 鼓励 Agent 在已有的树结构上快速移动到前沿阵地
            reward = -0.05
            action_type = "Reuse"

        # 3. 执行物理状态更新
        self.env.current_node_location = next_node

        # 只有新边才扣资源、加记录
        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree: self.env.current_tree['tree'] = {}
            self.env.current_tree['tree'][edge_key] = bw_req
            self.env.nodes_on_tree.add(current_node)
            self.env.nodes_on_tree.add(next_node)

        # 记录路径用于回溯或调试 (可选)
        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        self.env.current_path_trace.append(next_node)

        return self.get_state(), reward, False, False, {'moved': True, 'type': action_type}
    def get_low_level_action_mask(self):
        """低层动作掩码 - 修复版"""
        mask = np.zeros(self.env.n, dtype=np.float32)
        current = self.env.current_node_location

        neighbors = self.env.resource_mgr.get_neighbors(current)
        for nbr in neighbors:
            mask[nbr] = 1.0

        phase = getattr(self.env, 'current_phase', None)
        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        if target is not None and current == target:
            mask[current] = 2.0  # 🔥 停留动作的优先级更高（数值掩码）
        else:
            mask[current] = 0.5  # 🔥 停留动作的优先级较低

        if np.sum(mask) == 0:
            mask[current] = 1.0

        return mask

    def get_state(self):
        """
        🔥 [V42.2 修复版] 构建低层状态 - 修复 Agent 眼盲问题
        """
        # 1. 基础特征 (Base Features)
        current_vnf_demand = 0.0
        if self.env.current_request:
            vnf_list = self.env.current_request.get('vnf', [])
            idx = getattr(self.env, 'next_vnf_idx', 0)
            if idx < len(vnf_list):
                cpu_reqs = self.env.current_request.get('cpu_origin', [10.0])
                current_vnf_demand = cpu_reqs[idx] if idx < len(cpu_reqs) else 10.0

        base_features = []
        for node in range(self.env.n):
            # 获取资源信息
            avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
            avail_mem = self.env.resource_mgr.pool.get_available_memory(node)

            # 计算是否满足当前VNF需求
            fit_factor = 1.0 if avail_cpu >= current_vnf_demand else -1.0

            # 构建基础特征向量 [CPU, Mem, Fit, placeholder, placeholder]
            feat = [avail_cpu / 100.0, avail_mem / 100.0, fit_factor, 0.5, 0.5]

            # 补齐到 14 维 (保持与你训练的模型输入维度一致)
            if len(feat) < 14:
                feat += [0.0] * (14 - len(feat))
            base_features.append(feat)

        base_x = np.array(base_features, dtype=np.float32)

        # 2. 动态特征 (Dynamic Features)
        dynamic_features = []
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        # 获取已连接的目的地
        connected_dests = self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set()
        connected_dests_immutable = tuple(connected_dests)

        # 🔥🔥🔥 [关键修复] 获取当前目标节点
        target_node = None
        if self.env.current_phase == 'vnf_deployment':
            target_node = getattr(self.env, 'current_deployment_target', None)
        elif self.env.current_phase == 'destination_connection':
            target_node = getattr(self.env, 'current_target_node', None)

        # 确保 target_node 是 int 类型
        target_node_int = -1
        if target_node is not None:
            try:
                target_node_int = int(target_node)
            except:
                pass

        for node in range(self.env.n):
            # 特征 1: 是否在树上
            t_m = 1.0 if node in nodes_on_tree else 0.0

            # 特征 2: 是否是已连接的目的地
            c_m = 1.0 if node in connected_dests_immutable else 0.0

            # 🔥🔥🔥 [关键修复] 特征 3: 是否是当前高层指派的目标 (Target Indicator)
            # 原来写死是 0.0，现在改为真实状态
            is_target = 1.0 if node == target_node_int else 0.0

            dynamic_features.append([t_m, c_m, is_target])

        # 3. 合并特征
        full_x = np.concatenate([base_x, np.array(dynamic_features)], axis=1)
        x_tensor = torch.from_numpy(full_x).float()

        # 4. 获取动作掩码
        low_mask = self.get_low_level_action_mask()

        return Data(
            x=x_tensor,
            edge_index=self.env.edge_index if hasattr(self.env, 'edge_index') else None,
            edge_attr=self.env.edge_attr if hasattr(self.env, 'edge_attr') else None,
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0)
        )
    def _reset_phase_state(self):
        """清理阶段状态"""
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.current_target_node = None
        self.env.subgoal_step_count = 0

    def _reset_vnf_phase_only(self):
        """
        🔥 只重置VNF部署相关的阶段状态
        """
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0
        # 不清除 current_target_node，可能还有后续阶段

    def _get_hop_distance(self, u, v):
        """获取跳数距离"""
        if hasattr(self.env, '_get_hop_distance'):
            return self.env._get_hop_distance(u, v)
        return 9999
