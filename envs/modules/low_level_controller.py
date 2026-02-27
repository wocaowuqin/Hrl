"""
envs/modules/low_level_controller.py
====================================
低层执行控制器 - 树冗余度深度优化版 (V43.1)
====================================
优化内容：
1. ✅ [奖励微调] 惩罚树上闲逛（ReuseAway），细化新边惩罚梯度（目标在树上 vs 不在）。
2. ✅ [掩码安全] 保留微小原地概率防止死锁，放宽反向移动限制（保留探索）。
3. ✅ [防崩处理] 增加状态张量 NaN/Inf 检查，增加边索引空值兜底。
4. ✅ [原有逻辑] 保持强制停留连接、生命周期管理等核心功能。
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
    🕹️ 低层执行控制器 (Low-Level Controller) - V43.1 深度优化版
    """

    def __init__(self, env):
        """
        初始化控制器
        :param env: 主环境对象的引用 (SFC_HIRL_Env)
        """
        self.env = env

        # 1. 检查资源管理器
        if not hasattr(self.env, 'resource_mgr'):
            logger.error("❌ LowLevelController 初始化错误: env 中未找到 resource_mgr。")
            raise RuntimeError("resource_mgr 必须配置")

        # 2. 自动注入/挂载 RequestLifecycleManager
        if not hasattr(self.env, 'request_manager'):
            if hasattr(self.env.resource_mgr, 'request_manager'):
                self.env.request_manager = self.env.resource_mgr.request_manager
                logger.info("✅ [Init] 成功从 resource_mgr 挂载 request_manager 到 env")
            else:
                # 尝试动态导入并创建 (兜底方案)
                try:
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
            logger.info("✅ [LowLevelController] 初始化完成（V43.1 深度优化版）")

    def step_low_level(self, action):
        """
        🔥 低层步进函数
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
        处理 VNF 部署阶段
        """
        target_goal = getattr(self.env, 'current_deployment_target', None)

        # A. 移动逻辑
        if not is_stay:
            return self._handle_movement(current_node, target_action, target_goal)

        # B. 部署逻辑 (必须停留)
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
                truncated = True  # Subgoal 完成，truncated置为True让Coordinator重新决策

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
                    # 单个VNF部署完成

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
        🔥 [逻辑修复] 目的地连接逻辑
        强制要求 is_stay=True 才能建立连接，防止 Agent 只是路过而不连接
        """
        target_goal = getattr(self.env, 'current_target_node', None)
        if target_goal is None:
            return self.get_state(), -1.0, False, False, {'error': 'no_target'}

        # ---------- 1. 移动逻辑 ----------
        if not is_stay:
            # 如果已经到达目标，但还没有停留，返回 0 奖励，引导它下一步做停留
            if current_node == target_goal:
                return self.get_state(), 0.0, False, False, {'at_target': True, 'need_stay': True}
            else:
                return self._handle_movement(current_node, target_action, target_goal)

        # ---------- 2. 停留逻辑（建立连接）----------
        if current_node != target_goal:
            # 不在目标节点上停留？惩罚
            return self.get_state(), -1.0, False, False, {'warning': 'stay_but_not_at_target'}

        # 到达目标且执行停留 → 正式连接
        if 'connected_dests' not in self.env.current_tree:
            self.env.current_tree['connected_dests'] = set()

        # 只有新连接才给奖励
        if target_goal not in self.env.current_tree['connected_dests']:
            self.env.current_tree['connected_dests'].add(target_goal)
            step_reward = 10.0
            logger.info(f"✅ [连接成功] 节点 {target_goal} 加入连接树")
        else:
            step_reward = 0.0
            logger.warning(f"⚠️ [重复连接] 节点 {target_goal} 早已连接")

        # 检查是否所有目的地都已连接
        try:
            # 强制转换为 int 集合，防止 str/int 混用导致的判定失败
            all_dests = set(int(x) for x in self.env.current_request.get('dest', []))
            connected = set(int(x) for x in self.env.current_tree.get('connected_dests', set()))
        except Exception as e:
            logger.error(f"类型转换失败: {e}")
            all_dests, connected = set(), set()

        if all_dests and all_dests.issubset(connected):
            logger.info("🎉 [Episode完成] 所有目的地已连接 - 立即终止")
            self._archive_episode_success_only()
            self._add_request_to_lifecycle_manager()
            return self.get_state(), 50.0, True, False, {
                'episode_complete': True,
                'all_destinations_connected': True,
                'success': True
            }

        # 未完成全部，但当前子目标完成
        self.env.subgoal_step_count = 0
        self.env.current_target_node = None
        # truncated=True 通知 Coordinator 切换目标
        return self.get_state(), step_reward, False, True, {'dest_connected': True}

    def _handle_movement(self, current_node, target_action, target_goal):
        """
        🔥 [V43.1 奖励重构] 移动处理逻辑
        引入细粒度的“方向”和“树状态”奖励，杜绝闲逛
        """
        next_node = int(target_action)

        # 1. 资源/连通性预检 (Fail Fast)
        bw_req = self.env.current_request.get('bw_origin', 0.0)
        try:
            # 简单检查连通性和带宽
            has_bw = self.env.resource_mgr.pool.get_available_bandwidth(current_node, next_node) >= bw_req
        except:
            has_bw = False

        if not has_bw:
            logger.warning(f"❌ [Resource] 链路 {current_node}->{next_node} 带宽不足")
            return self.get_state(), -10.0, True, False, {'error': 'resource_failure'}

        # 2. 计算路径特征
        edge_key = tuple(sorted((current_node, next_node)))
        is_new_edge = edge_key not in self.env.current_tree.get('tree', {})
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        # 判断目标是否已在树上
        target_on_tree = False
        if target_goal is not None:
            try:
                target_on_tree = int(target_goal) in nodes_on_tree
            except:
                pass

        # 判断是否向目标移动
        moving_toward_target = False
        if target_goal is not None:
            try:
                cur_dist = self._get_hop_distance(current_node, target_goal)
                nxt_dist = self._get_hop_distance(next_node, target_goal)
                moving_toward_target = nxt_dist < cur_dist
            except:
                moving_toward_target = True

        # ========== 3. 奖励逻辑 (精细化) ==========
        if is_new_edge:
            # --- 开新路分支 ---
            if moving_toward_target:
                if target_on_tree:
                    # 目标在树上，我们在建桥连接 -> 轻罚
                    base_reward = -0.1
                    action_type = "NewPathToTree"
                else:
                    # 目标不在树上，正常延伸 -> 中罚
                    base_reward = -0.3
                    action_type = "NewPathToTarget"
            else:
                # 既是新路又远离目标 -> 重罚
                base_reward = -0.8
                action_type = "NewPathAway"
        else:
            # --- 复用分支 ---
            if moving_toward_target:
                # 复用且向目标 -> 最佳行为
                base_reward = 0.4
                action_type = "ReuseToward"
            else:
                # 复用但远离目标 -> 轻微惩罚 (禁止闲逛)
                base_reward = -0.1
                action_type = "ReuseAway"

        # 额外奖励：如果全程都在树上移动（在树干上穿梭），且没有因为远离目标而被惩罚太多，给予微量鼓励保持连通性
        if current_node in nodes_on_tree and next_node in nodes_on_tree and not is_new_edge:
            # 只有当它是有效移动时才加分
            if base_reward > 0:
                base_reward += 0.1

        # 4. 执行物理状态更新
        self.env.current_node_location = next_node

        # 只有新边才真正扣除带宽资源、加记录
        if is_new_edge:
            self.env.resource_mgr.allocate_bandwidth(current_node, next_node, bw_req)
            if 'tree' not in self.env.current_tree: self.env.current_tree['tree'] = {}
            self.env.current_tree['tree'][edge_key] = bw_req
            self.env.nodes_on_tree.add(current_node)
            self.env.nodes_on_tree.add(next_node)

        # 记录路径用于回溯或调试
        if not hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []
        self.env.current_path_trace.append(next_node)

        return self.get_state(), base_reward, False, False, {'moved': True, 'type': action_type}

    def get_low_level_action_mask(self):
        """
        🔥 [V43.1 掩码优化] 智能动作掩码
        增加安全兜底，软化反向移动限制
        """
        mask = np.zeros(self.env.n, dtype=np.float32)
        current = self.env.current_node_location

        # 1. 基础：获取物理连通邻居
        neighbors = self.env.resource_mgr.get_neighbors(current)
        for nbr in neighbors:
            mask[nbr] = 1.0

        # 2. 树节点优先 (引导复用)
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
        for nbr in neighbors:
            if nbr in nodes_on_tree:
                mask[nbr] += 0.5  # 树上节点加分

        # 3. 目标导向 (引导方向)
        phase = getattr(self.env, 'current_phase', None)
        target = None
        if phase == 'vnf_deployment':
            target = getattr(self.env, 'current_deployment_target', None)
        elif phase == 'destination_connection':
            target = getattr(self.env, 'current_target_node', None)

        if target is not None:
            if current == target:
                # 已到达，停留动作 (Stay) 优先级最高
                mask = np.zeros(self.env.n, dtype=np.float32)
                mask[current] = 5.0
            else:
                # 还没到，给更接近目标的邻居加分
                try:
                    cur_dist = self._get_hop_distance(current, target)
                    for nbr in neighbors:
                        nbr_dist = self._get_hop_distance(nbr, target)
                        if nbr_dist < cur_dist:
                            mask[nbr] += 0.8  # 向目标靠近
                        elif nbr_dist > cur_dist:
                            mask[nbr] *= 0.5  # 远离目标：软惩罚 (0.5)，保留探索可能，不完全封杀
                except:
                    pass

                # 安全兜底：保留微小的原地概率，防止因带宽耗尽等原因导致的死锁
                mask[current] = 0.1

        # 4. 归一化与兜底
        if np.max(mask) > 0:
            # 将最大值拉伸到 1.0 或更高
            mask = mask / np.max(mask)

        if np.sum(mask) == 0:
            mask[current] = 1.0

        return mask

    def get_state(self):
        """
        构建低层状态 (V43.1 防崩溃版)
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
            avail_cpu = self.env.resource_mgr.pool.get_available_cpu(node)
            avail_mem = self.env.resource_mgr.pool.get_available_memory(node)
            fit_factor = 1.0 if avail_cpu >= current_vnf_demand else -1.0

            # [CPU, Mem, Fit, placeholder, placeholder]
            feat = [avail_cpu / 100.0, avail_mem / 100.0, fit_factor, 0.5, 0.5]
            if len(feat) < 14:
                feat += [0.0] * (14 - len(feat))
            base_features.append(feat)

        base_x = np.array(base_features, dtype=np.float32)

        # 2. 动态特征 (Dynamic Features)
        dynamic_features = []
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())
        connected_dests = self.env.current_tree.get('connected_dests', set()) if self.env.current_tree else set()
        connected_dests_immutable = tuple(connected_dests)

        # 获取当前目标节点
        target_node = None
        if self.env.current_phase == 'vnf_deployment':
            target_node = getattr(self.env, 'current_deployment_target', None)
        elif self.env.current_phase == 'destination_connection':
            target_node = getattr(self.env, 'current_target_node', None)

        target_node_int = -1
        if target_node is not None:
            try:
                target_node_int = int(target_node)
            except:
                pass

        for node in range(self.env.n):
            t_m = 1.0 if node in nodes_on_tree else 0.0
            c_m = 1.0 if node in connected_dests_immutable else 0.0
            is_target = 1.0 if node == target_node_int else 0.0
            dynamic_features.append([t_m, c_m, is_target])

        # 3. 合并特征
        full_x = np.concatenate([base_x, np.array(dynamic_features)], axis=1)
        x_tensor = torch.from_numpy(full_x).float()

        # 🔥 [关键修复] NaN/Inf 检查，防止训练卡死
        if torch.isnan(x_tensor).any() or torch.isinf(x_tensor).any():
            logger.error("❌ State contains NaN/Inf! Replacing with zeros.")
            x_tensor = torch.zeros_like(x_tensor)

        # 4. 获取动作掩码
        low_mask = self.get_low_level_action_mask()

        # 🔥 [关键修复] 边索引兜底
        edge_index = self.env.edge_index if hasattr(self.env, 'edge_index') and self.env.edge_index is not None else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = self.env.edge_attr if hasattr(self.env, 'edge_attr') and self.env.edge_attr is not None else torch.zeros((edge_index.size(1), 5), dtype=torch.float)

        return Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            action_mask=torch.from_numpy(low_mask).bool().unsqueeze(0)
        )

    def _reset_phase_state(self):
        """清理阶段状态"""
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.current_target_node = None
        self.env.subgoal_step_count = 0

    def _reset_vnf_phase_only(self):
        """只重置VNF部署相关的阶段状态"""
        self.env.current_phase = None
        self.env.current_deployment_target = None
        self.env.subgoal_step_count = 0

    def _get_hop_distance(self, u, v):
        """获取跳数距离"""
        if hasattr(self.env, '_get_hop_distance'):
            return self.env._get_hop_distance(u, v)
        return 9999

    def _archive_episode_success_only(self):
        """只做账本归档"""
        logger.info("📝 [归档] 开始归档Episode账本...")
        if hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, '_archive_request'):
            try:
                self.env.resource_mgr._archive_request(success=True, already_rolled_back=False)
                logger.info("✅ [归档] 账本归档成功")
                return
            except Exception as e:
                logger.error(f"❌ [归档] resource_mgr 归档失败: {e}")
        self._manual_save_resources_only()

    def _manual_save_resources_only(self):
        """手动保存账本"""
        if self.env.current_request is None:
            return
        if 'resources_allocated' not in self.env.current_request:
            self.env.current_request['resources_allocated'] = {
                'placement': copy.deepcopy(self.env.current_tree.get('placement', {})),
                'tree': copy.deepcopy(self.env.current_tree.get('tree', {}))
            }

    def _add_request_to_lifecycle_manager(self):
        """统一的生命周期托管入口"""
        if not hasattr(self.env, 'request_manager') or not self.env.request_manager:
            logger.error("❌ [生命周期] 缺少 request_manager")
            return False

        req = self.env.current_request
        if req is None: return False

        req_id = req.get('id', id(req))
        if req_id in self.env.request_manager.active_requests:
            return True

        resources = self._collect_allocated_resources()
        temp_request_dict = {
            'id': req_id,
            'arrival_time': req.get('arrival_time', 0),
            'lifetime': req.get('lifetime', 50)
        }

        try:
            success = self.env.request_manager.register_request(temp_request_dict, resources)
            if success:
                logger.info(f"✅ [生命周期] 请求 {req_id} 托管成功")
            return success
        except Exception as e:
            logger.error(f"❌ [生命周期] 托管异常: {e}")
            return False

    def _collect_allocated_resources(self):
        """收集分配的资源"""
        resources = {
            'placement': {}, 'tree': {},
            'bandwidth': self.env.current_request.get('bw_origin', 1.0)
        }
        if self.env.current_tree:
            resources['placement'] = copy.deepcopy(self.env.current_tree.get('placement', {}))
            resources['tree'] = copy.deepcopy(self.env.current_tree.get('tree', {}))
        return resources