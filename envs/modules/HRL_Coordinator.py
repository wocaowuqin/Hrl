import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HRL_Coordinator:
    def __init__(self, env, high_agent, low_agent, config=None):
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.config = config or {}

        if hasattr(env, 'resource_mgr'):
            self.resource_mgr = env.resource_mgr
        else:
            self.resource_mgr = None

        self.max_low_steps = self.config.get('max_low_steps', 50)
        self.env.max_subgoal_steps = self.config.get('max_subgoal_steps', 40)
        self.stats = defaultdict(int)
        self.current_episode = 0
        self.resources_released = False

        self._permanent_unreachable = set()

    def run_high_low_cycle(self, high_obs, training=True):
        if hasattr(self, 'resources_released'):
            self.resources_released = False

        high_mask = self.env.get_high_level_action_mask()

        if hasattr(self, '_unreachable_targets') and self._unreachable_targets:
            for idx in self._unreachable_targets:
                if idx < len(high_mask):
                    high_mask[idx] = 0

        if hasattr(self, '_permanent_unreachable') and self._permanent_unreachable:
            for idx in self._permanent_unreachable:
                if idx < len(high_mask):
                    high_mask[idx] = 0

        if sum(high_mask) == 0:
            return 0.0, True, {'error': 'no_high_actions'}

        high_action_idx, high_action_remapped, agent_info = self.high_agent.select_action(
            high_obs, action_mask=high_mask
        )

        target_node = None
        valid_indices = np.where(high_mask > 0)[0]
        if agent_info and 'subgoal' in agent_info and agent_info['subgoal'] is not None:
            candidate = int(agent_info['subgoal'])
            # 验证subgoal在当前mask仍然有效（资源可能已耗尽）
            if candidate < len(high_mask) and high_mask[candidate] > 0:
                target_node = candidate
            elif len(valid_indices) > 0:
                idx_pick = high_action_idx if high_action_idx < len(valid_indices) else 0
                target_node = int(valid_indices[idx_pick])
                logger.debug(f"[Coord] subgoal {candidate} 资源不足，重映射to {target_node}")
        if target_node is None:
            if high_action_idx < len(valid_indices):
                target_node = int(valid_indices[high_action_idx])
            elif len(valid_indices) > 0:
                target_node = int(valid_indices[0])
            else:
                target_node = high_action_idx

        start_node = agent_info.get('start_node')
        if start_node is None:
            # ── [FIX SFC主干] 判断是否进入dest阶段，强制从last_vnf出发 ──────────
            # 原逻辑依赖 current_phase=='destination_connection'，但phase切换发生在
            # step_high_level()开头，而start_node在调用之前就确定了，导致：
            # VNF全部部署完的第一个dest cycle，phase还是'vnf_deployment'，
            # start_node走了else分支用current_node_location，dest从错误位置出发。
            # 修复：改用 next_vnf_idx >= len(vnf_list) 直接判断VNF是否全部完成，
            # 只要VNF完成且chain_nodes非空，就强制从last_vnf出发，
            # 保证 Source→VNF1→VNF2→VNF3→Dest 的主干结构。
            _chain = getattr(self.env, 'chain_nodes', [])
            _vnf_list = []
            if self.env.current_request:
                _vnf_list = self.env.current_request.get('vnf', [])
            _vnf_done = getattr(self.env, 'next_vnf_idx', 0) >= len(_vnf_list) if _vnf_list else False
            if _vnf_done and len(_chain) > 0:
                last_vnf = _chain[-1]
                self.env.current_node_location = last_vnf
                start_node = last_vnf
                logger.debug(f"[Coord] VNF完成，dest阶段强制start_node=last_vnf={last_vnf}")
            else:
                start_node = self.env.current_node_location
            # ──────────────────────────────────────────────────────────────────

        self.env.set_high_level_goal(high_action_idx, target_node, start_node)

        _, _, done, truncated, high_info = self.env.step_high_level(high_action_idx)

        if done:
            return 0.0, True, {'episode_done': True}

        if truncated:
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            if hasattr(self.high_agent, 'subgoal_steps'):
                horizon = getattr(self.high_agent, 'subgoal_horizon', 10)
                self.high_agent.subgoal_steps = horizon + 1
            if hasattr(self.high_agent, 'subgoal_step_count'):
                self.high_agent.subgoal_step_count = 999
            if hasattr(self, '_unreachable_targets'):
                self._unreachable_targets.clear()
            return 0.0, False, {'subgoal_truncated': True}

        self.env.subgoal_step_count = 0
        if hasattr(self.env, 'current_path_trace'):
            self.env.current_path_trace = []

        low_state = self.env.get_state()
        low_step = 0
        low_total_reward = 0.0
        episode_done = False
        info = {}
        low_level_stalled = False
        subgoal_achieved = False
        start_pos_before_low = self.env.current_node_location

        planned_path = None
        path_idx = 0
        blocked_edges = set()
        replan_count = 0
        no_progress_count = 0

        if self.current_episode < 200:
            use_path_guide = True
        elif self.current_episode < 500:
            use_path_guide = np.random.rand() < 0.2
        else:
            use_path_guide = False

        if use_path_guide and hasattr(self.env, 'low_level_controller') and \
                hasattr(self.env.low_level_controller, 'compute_bw_aware_path'):
            current_pos = self.env.current_node_location
            _chain = getattr(self.env, 'chain_nodes', [])
            _vnf_list = self.env.current_request.get('vnf', []) if self.env.current_request else []
            _vnf_done = getattr(self.env, 'next_vnf_idx', 0) >= len(_vnf_list) if _vnf_list else False

            # ── dest阶段path_guide：从last_vnf出发到dest_i ─────────────
            if _vnf_done and len(_chain) >= 1:
                last_vnf = _chain[-1]
                planned_path = self.env.low_level_controller.compute_bw_aware_path(
                    last_vnf, target_node)
                if planned_path and len(planned_path) > 1:
                    planned_path = planned_path[1:]
                elif planned_path and len(planned_path) == 1:
                    planned_path = [target_node]
                else:
                    planned_path = None
                    self._unreachable_targets.add(target_node)
            # ──────────────────────────────────────────────────────────────────
            else:
                planned_path = self.env.low_level_controller.compute_bw_aware_path(current_pos, target_node)
                if planned_path and len(planned_path) > 1:
                    planned_path = planned_path[1:] + [target_node]
                elif planned_path and len(planned_path) == 1:
                    planned_path = [target_node]
                else:
                    planned_path = None
                    self._unreachable_targets.add(target_node)

        while low_step < self.max_low_steps and not episode_done:
            if planned_path is not None and path_idx < len(planned_path):
                low_action = planned_path[path_idx]
                path_idx += 1
            else:
                low_mask = self.env.get_low_level_action_mask()
                if sum(low_mask) == 0:
                    low_level_stalled = True
                    break
                _, low_action, _ = self.low_agent.select_action(
                    low_state, action_mask=low_mask
                )

            pos_before = self.env.current_node_location
            next_state, reward, done, truncated_low, info = self.env.step_low_level(low_action)
            pos_after = self.env.current_node_location

            is_success = info.get('dest_connected') or info.get('vnf_deployed') \
                         or info.get('all_vnf_deployed') or info.get('episode_complete')

            if pos_after == pos_before and not is_success:
                no_progress_count += 1
                failed_edge = info.get('edge')
                if failed_edge:
                    blocked_edges.add(tuple(sorted(failed_edge)))
            else:
                no_progress_count = 0

            if info.get('error') in ('no_bandwidth', 'resource_failure'):
                if planned_path is not None:
                    replan_count += 1
                    if replan_count <= 3:
                        current_pos = self.env.current_node_location
                        planned_path = None
                        if hasattr(self.env, 'low_level_controller') and \
                                hasattr(self.env.low_level_controller, 'compute_bw_aware_path'):
                            new_path = self.env.low_level_controller.compute_bw_aware_path(
                                current_pos, target_node, excluded_edges=blocked_edges)
                            if new_path and len(new_path) > 1:
                                planned_path = new_path[1:] + [target_node]
                                path_idx = 0
                            else:
                                self._unreachable_targets.add(target_node)
                    else:
                        planned_path = None

            if no_progress_count >= 5:
                self._unreachable_targets.add(target_node)
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                if hasattr(self.high_agent, 'subgoal_steps'):
                    self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1
                info['stuck'] = True
                break

            if training:
                self._store_transition(
                    self.low_agent, low_state, low_action,
                    reward, next_state, done
                )

            low_total_reward += reward
            low_state = next_state
            low_step += 1

            if info.get('subgoal_done', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('goal_reached', False) or info.get('current_goal_satisfied', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('dest_connected', False) or info.get('vnf_deployed', False) \
                    or info.get('all_vnf_deployed', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                break

            if info.get('episode_complete', False) or info.get('all_destinations_connected', False):
                if hasattr(self.high_agent, 'current_subgoal'):
                    self.high_agent.current_subgoal = None
                subgoal_achieved = True
                episode_done = True
                break

            if done:
                episode_done = True
            if truncated_low:
                if info.get('deploy_fail'):
                    if not hasattr(self, '_permanent_unreachable'):
                        self._permanent_unreachable = set()
                    self._permanent_unreachable.add(target_node)
                break

        actual_end_pos = self.env.current_node_location
        if low_step >= self.max_low_steps and not subgoal_achieved:
            if hasattr(self.high_agent, 'current_subgoal'):
                self.high_agent.current_subgoal = None
            if hasattr(self.high_agent, 'subgoal_steps'):
                self.high_agent.subgoal_steps = getattr(self.high_agent, 'subgoal_horizon', 10) + 1

            self._unreachable_targets.add(target_node)

            if actual_end_pos == start_pos_before_low:
                info['stuck'] = True

        high_done = episode_done

        milestone_reward = self._compute_reward_from_env_info(info)
        alpha = 0.05 if self.current_episode < 200 else 0.1
        high_reward = milestone_reward + alpha * low_total_reward

        if hasattr(self.env, 'low_level_controller') and \
                hasattr(self.env.low_level_controller, '_calculate_tree_metrics'):

            metrics = self.env.low_level_controller._calculate_tree_metrics()
            current_redundancy = metrics.get('redundancy', 0.0)

            if not hasattr(self, '_last_tree_redundancy'):
                self._last_tree_redundancy = 0.0

            delta_redundancy = current_redundancy - self._last_tree_redundancy
            structure_penalty = -3.0 * delta_redundancy
            high_reward += structure_penalty

            self._last_tree_redundancy = current_redundancy

        if low_step == 0 or low_level_stalled or info.get('stuck', False):
            high_reward -= 15.0
            info['stalled'] = True

        if not high_done and self.env.current_request:
            if hasattr(self.env, 'high_level_controller') and hasattr(self.env.high_level_controller,
                                                                      '_is_all_tasks_completed'):
                try:
                    completed, status = self.env.high_level_controller._is_all_tasks_completed()
                    if completed:
                        high_done = True
                        high_reward += 30.0
                        episode_done = True
                        # ── 分叉多样性奖励 + 树总长奖励 ──────────────
                        try:
                            _sfc2 = getattr(self.env, 'current_sfc', None)
                            if _sfc2:
                                # 树总长奖励（边数越少越好，鼓励路径复用）
                                _all_e = set()
                                for _sg in _sfc2.get('spine_paths', []):
                                    for _i in range(len(_sg)-1):
                                        _all_e.add(tuple(sorted((_sg[_i],_sg[_i+1]))))
                                for _bp in _sfc2.get('branch_paths', {}).values():
                                    for _i in range(len(_bp)-1):
                                        _all_e.add(tuple(sorted((_bp[_i],_bp[_i+1]))))
                                _len_bonus = max(0.0, (23 - len(_all_e)) * 0.5)
                                high_reward += _len_bonus
                                logger.debug(f"[TreeQuality] len={len(_all_e)} "
                                             f"len_bonus={_len_bonus:.1f}")
                                # spine回绕惩罚
                                _seen_spine = set()
                                _overlap = 0
                                for _sg in _sfc2.get('spine_paths', []):
                                    for _nd in _sg[1:]:
                                        if _nd in _seen_spine: _overlap += 1
                                        _seen_spine.add(_nd)
                                if _overlap > 0:
                                    high_reward -= 5.0 * _overlap
                        except Exception:
                            pass
                except:
                    pass

        if training:
            high_next_state = None if high_done else self.env.get_state()
            # 直接调store_transition_high，避免通用路由误存到low_memory
            if hasattr(self.high_agent, 'store_transition_high'):
                self.high_agent.store_transition_high(
                    high_obs, high_action_idx,
                    high_reward, high_next_state, high_done
                )
            else:
                self._store_transition(
                    self.high_agent, high_obs, high_action_idx,
                    high_reward, high_next_state, high_done
                )

        self._update_stats(high_done, info)

        if hasattr(self, '_unreachable_targets'):
            self._unreachable_targets.clear()

        return high_reward, high_done, {
            'high_action': high_action_idx,
            'target_node': target_node,
            'low_steps': low_step,
            'high_reward': high_reward,
            'info': info
        }

    def run_episode(self, training=True, max_steps=100):
        self.current_episode += 1
        self.resources_released = False

        high_obs = self.env.reset()

        self._unreachable_targets = set()
        self._permanent_unreachable = set()  # 每episode重置，不跨episode封禁节点
        self._last_tree_redundancy = 0.0

        episode_done = False
        total_reward = 0.0
        total_steps = 0
        no_progress_cycles = 0
        last_connected_count = 0
        last_vnf_count = 0
        MAX_NO_PROGRESS = 5

        while not episode_done and total_steps < max_steps:
            cycle_reward, done, info = self.run_high_low_cycle(
                high_obs, training=training
            )

            total_reward += cycle_reward
            total_steps += 1
            episode_done = done

            cur_vnf = getattr(self.env, 'next_vnf_idx', 0)
            cur_conn = len(self.env.current_tree.get('connected_dests', set())) if self.env.current_tree else 0
            if cur_vnf > last_vnf_count or cur_conn > last_connected_count:
                no_progress_cycles = 0
                last_vnf_count = cur_vnf
                last_connected_count = cur_conn
            else:
                no_progress_cycles += 1
            if not episode_done and no_progress_cycles >= MAX_NO_PROGRESS:
                logger.warning(f'[Coord] 连续{no_progress_cycles}个cycle无进展，终止(VNF={cur_vnf} Conn={cur_conn})')
                total_reward -= 20.0
                episode_done = True
                info['fail'] = True
                info['reason'] = 'no_progress'

            if not episode_done:
                high_obs = self.env.get_state()

        vnf_success = False
        dest_success = False
        episode_success = False
        completion_status = "未知"

        try:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                vnf_list = self.env.current_request.get('vnf', [])
                dest_list = self.env.current_request.get('dest', [])

                if hasattr(self.env, 'high_level_controller') and \
                        hasattr(self.env.high_level_controller, '_is_all_tasks_completed'):
                    try:
                        completed, status = self.env.high_level_controller._is_all_tasks_completed()
                        episode_success = completed
                        completion_status = status
                    except Exception as e:
                        episode_success = self._fallback_success_check()
                        completion_status = "回退检测"
                else:
                    episode_success = self._fallback_success_check()
                    completion_status = "回退检测"

                if hasattr(self.env, 'next_vnf_idx'):
                    vnf_progress = self.env.next_vnf_idx
                    vnf_success = vnf_progress >= len(vnf_list)

                if hasattr(self.env, 'current_tree') and self.env.current_tree:
                    connected_dests = self.env.current_tree.get('connected_dests', set())
                    dest_success = len(connected_dests) >= len(dest_list)

        except Exception as e:
            episode_success = False

        if episode_success:
            self.resources_released = True
        else:
            try:
                if (hasattr(self.env, 'resource_mgr') and
                        hasattr(self.env.resource_mgr, '_archive_request')):
                    self.env.resource_mgr._archive_request(
                        success=False, already_rolled_back=False)
            except Exception as e:
                pass

        self._reset_episode_stats()

        # ── [VIS] 收集树快照供可视化使用 ──────────────────────────────────
        tree_snapshot = None
        try:
            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                import copy
                _t = self.env.current_tree
                tree_snapshot = {
                    'tree': dict(_t.get('tree', {})),
                    'placement': copy.deepcopy(_t.get('placement', {})),
                    'connected_dests': set(_t.get('connected_dests', set())),
                }
        except Exception:
            pass
        req_snapshot = None
        try:
            if hasattr(self.env, 'current_request') and self.env.current_request:
                req_snapshot = dict(self.env.current_request)
        except Exception:
            pass
        chain_snapshot = list(getattr(self.env, 'chain_nodes', []))
        # 收集 current_sfc（分层DAG）快照
        sfc_snapshot = None
        try:
            _sfc = getattr(self.env, 'current_sfc', None)
            if _sfc:
                import copy
                sfc_snapshot = {
                    'chain_nodes': list(_sfc.get('chain_nodes', [])),
                    'spine_paths': [list(p) for p in _sfc.get('spine_paths', [])],
                    'branch_paths': {k: list(v) for k, v in _sfc.get('branch_paths', {}).items()},
                    'branch_roots': {k: v for k, v in _sfc.get('branch_roots', {}).items()},
                }
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────

        # [Fix] episode终结时驱动ε衰减（基于episode而非steps）
        if training and hasattr(self.high_agent, 'on_episode_end'):
            self.high_agent.on_episode_end()

        return total_reward, {
            'steps': total_steps,
            'success': episode_success,
            'reward': total_reward,
            'subgoals_ok': self.stats.get('subgoals_ok', 0),
            'subgoals_fail': self.stats.get('subgoals_fail', 0),
            'vnf_success': vnf_success,
            'dest_success': dest_success,
            'completion_status': completion_status,
            'tree_snapshot': tree_snapshot,
            'req_snapshot': req_snapshot,
            'chain_nodes': chain_snapshot,
            'sfc_snapshot': sfc_snapshot,
        }

    def _fallback_success_check(self):
        if not hasattr(self.env, 'current_request') or not self.env.current_request:
            return True
        try:
            vnf_list = self.env.current_request.get('vnf', [])
            dest_list = self.env.current_request.get('dest', [])

            if hasattr(self.env, 'next_vnf_idx'):
                vnf_success = self.env.next_vnf_idx >= len(vnf_list)
            elif hasattr(self.env, 'resource_mgr') and hasattr(self.env.resource_mgr, 'next_vnf_idx'):
                vnf_success = self.env.resource_mgr.next_vnf_idx >= len(vnf_list)
            else:
                vnf_success = False

            if hasattr(self.env, 'current_tree') and self.env.current_tree:
                connected_dests = self.env.current_tree.get('connected_dests', set())
                dest_success = len(connected_dests) >= len(dest_list)
            else:
                dest_success = False

            return vnf_success and dest_success
        except:
            return False

    def _compute_reward_from_env_info(self, info):
        if not info: return 0.0
        if info.get('episode_complete', False) or info.get('all_destinations_connected', False):
            return 30.0
        if info.get('all_vnf_deployed', False):
            return 25.0
        if info.get('vnf_deployed', False):
            return 20.0
        elif info.get('dest_connected', False):
            return 10.0
        elif info.get('deploy_fail', False):
            return -10.0
        elif info.get('timeout', False):
            return -5.0
        return 0.0

    def _store_transition(self, agent, state, action, reward, next_state, done):
        if hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward, next_state, done)
        elif hasattr(agent, 'memory') and hasattr(agent.memory, 'store'):
            agent.memory.store(state, action, reward, next_state, done)
        elif hasattr(agent, 'store'):
            agent.store(state, action, reward, next_state, done)

    def _update_stats(self, high_done, info):
        if info:
            if info.get('vnf_deployed', False) or info.get('dest_connected', False):
                self.stats['subgoals_ok'] += 1
            elif info.get('deploy_fail', False) or info.get('timeout', False):
                self.stats['subgoals_fail'] += 1
        if high_done:
            self.stats['episodes_completed'] += 1

    def _reset_episode_stats(self):
        self.stats['subgoals_ok'] = 0
        self.stats['subgoals_fail'] = 0

    def get_stats(self):
        return dict(self.stats)
def visualize_sfc_tree_publication(data, save_path="sfc_tree_pub.png"):
    """
    论文级 SFC 多播树可视化（暂时禁用）
    """
    # 暂时注释，训练时跳过可视化生成
    return
# ═══════════════════════════════════════════════════════════════════════════════
# 🎨  论文级 SFC 多播树可视化工具函数
#     用法（训练结束后）：
#       from core.hrl.HRL_Coordinator import visualize_sfc_tree_publication
#
#     data 格式：
#       {
#         'req':          req_snapshot (dict, from info['req_snapshot']),
#         'sfc_snapshot': sfc_snapshot (dict, from info['sfc_snapshot']),
#       }
# # ═══════════════════════════════════════════════════════════════════════════════
# def visualize_sfc_tree_publication(data, save_path="sfc_tree_pub.png"):
#     """
#     论文级 SFC 多播树可视化（归一化坐标系修复版）
#     - Spine 水平等距排列，全部归一化到 x∈[0,1]
#     - Branch 在 last_vnf 正下方扇形展开，间距由 dest 数量自适应
#     - 所有边只用已知坐标绘制，不产生交叉
#     - 高 DPI
#     """
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     import numpy as np
#
#     req          = data.get('req', {})
#     sfc          = data.get('sfc_snapshot', {})
#     chain        = sfc.get('chain_nodes', [])
#     spine_paths  = sfc.get('spine_paths', [])
#     branch_paths = sfc.get('branch_paths', {})
#
#     source = req.get('source')
#     # dest 列表：优先从 req，没有则从 branch_paths key
#     req_dests = req.get('dest', [])
#     dests = [int(d) for d in req_dests] if req_dests else [int(d) for d in branch_paths.keys()]
#
#     if not spine_paths:
#         print("No valid SFC data")
#         return
#
#     # ── 1. 构造 spine 序列（去重保持顺序）─────────────────────────────────
#     spine_seq = []
#     seen = set()
#     for seg in spine_paths:
#         for node in seg:
#             if node not in seen:
#                 spine_seq.append(node)
#                 seen.add(node)
#
#     n_spine  = len(spine_seq)
#     # 按 chain 节点在 spine_seq 中的出现顺序重排，确保 V1→V2→V3 与路径一致
#     chain_set = set(chain)
#     chain_ordered = [n for n in spine_seq if n in chain_set]
#     # 补上不在 spine_seq 里的 chain 节点（理论上不应有，但保险起见）
#     for n in chain:
#         if n not in chain_ordered:
#             chain_ordered.append(n)
#     chain = chain_ordered
#
#     last_vnf = chain[-1] if chain else spine_seq[-1]
#
#     # ── 2. 归一化 spine 坐标（y=0.78，x 均匀分布在 [0.04, 0.96]）──────────
#     x_left, x_right = 0.04, 0.96
#     spine_norm = {}
#     for i, node in enumerate(spine_seq):
#         if n_spine > 1:
#             x = x_left + (x_right - x_left) * i / (n_spine - 1)
#         else:
#             x = 0.5
#         spine_norm[node] = (x, 0.78)
#
#     lv_x = spine_norm[last_vnf][0]   # last_vnf 的归一化 x
#
#     # ── 3. dest 扇形坐标（y=0.12，x 在 last_vnf 附近均匀展开）──────────────
#     n_dest = len(dests)
#     half_span = min(0.40, 0.10 * n_dest)   # 自适应宽度，最多 ±0.40
#     if n_dest > 1:
#         dest_xs = [lv_x - half_span + i * 2 * half_span / (n_dest - 1)
#                    for i in range(n_dest)]
#     else:
#         dest_xs = [lv_x]
#
#     dest_norm = {d: (dest_xs[i], 0.12) for i, d in enumerate(sorted(dests))}
#
#     # ── 4. branch 中间节点坐标（线性插值，避免重叠）─────────────────────────
#     branch_mid_norm = {}
#     for d_key, bpath in branch_paths.items():
#         d = int(d_key)
#         if d not in dest_norm:
#             continue
#         dx, dy = dest_norm[d]
#         # 起点：last_vnf（branch 总从 last_vnf 出发）
#         sx, sy = spine_norm[last_vnf]
#         mid_nodes = [n for n in bpath if n not in spine_norm and int(n) not in dest_norm]
#         n_mid = len(mid_nodes)
#         for j, node in enumerate(mid_nodes):
#             frac = (j + 1) / (n_mid + 1)
#             bx = sx + frac * (dx - sx)
#             by = sy + frac * (dy - sy)
#             if node not in branch_mid_norm:
#                 branch_mid_norm[node] = (bx, by)
#
#     def get_pos(node):
#         if node in spine_norm:
#             return spine_norm[node]
#         if node in branch_mid_norm:
#             return branch_mid_norm[node]
#         ni = int(node) if not isinstance(node, int) else node
#         if ni in dest_norm:
#             return dest_norm[ni]
#         return (lv_x, 0.45)
#
#     # ── 5. 画布 ──────────────────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(14, 6))
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_axis_off()
#
#     # ── 6. 绘制 spine 边（黑色粗线，带箭头）─────────────────────────────────
#     spine_edge_set = set()
#     for seg in spine_paths:
#         for i in range(len(seg) - 1):
#             u, v = seg[i], seg[i + 1]
#             spine_edge_set.add((u, v))
#             ux, uy = get_pos(u)
#             vx, vy = get_pos(v)
#             ax.annotate('', xy=(vx, vy), xytext=(ux, uy),
#                         xycoords='axes fraction', textcoords='axes fraction',
#                         arrowprops=dict(arrowstyle='->', color='#2c3e50',
#                                         lw=2.2, mutation_scale=14))
#
#     # ── 7. 绘制 branch 边（橙色线，从 last_vnf 直连 dest）───────────────────
#     # 修复：不依赖 branch_paths 里的中间路径节点判断，
#     # 直接从 last_vnf 画到对应 dest，保证不交叉。
#     for d_key in branch_paths.keys():
#         d = int(d_key)
#         if d not in dest_norm:
#             continue
#         sx, sy = spine_norm[last_vnf]
#         dx, dy = dest_norm[d]
#         ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
#                     xycoords='axes fraction', textcoords='axes fraction',
#                     arrowprops=dict(arrowstyle='->', color='#e67e22',
#                                     lw=2.0, mutation_scale=13, alpha=0.9))
#
#     # ── 8. 绘制节点 ───────────────────────────────────────────────────────────
#     def draw_node(x, y, fc, ec, r, label=None, fs=7, fc_text='white', zorder=4):
#         circ = plt.Circle((x, y), r, color=fc, ec=ec, lw=1.5,
#                            zorder=zorder, transform=ax.transAxes, clip_on=False)
#         ax.add_patch(circ)
#         if label is not None:
#             ax.text(x, y, str(label), ha='center', va='center',
#                     fontsize=fs, fontweight='bold', color=fc_text,
#                     zorder=zorder + 1, transform=ax.transAxes)
#
#     # 中间灰色 spine 节点
#     vnf_set = set(chain)
#     for node in spine_seq:
#         if node == source or node in vnf_set:
#             continue
#         draw_node(*spine_norm[node], '#bdc3c7', '#95a5a6', 0.022, label=node, fc_text='#333')
#
#     # VNF 节点（蓝色）
#     for k, vnf in enumerate(chain):
#         x, y = spine_norm[vnf]
#         draw_node(x, y, '#2980b9', '#1a5276', 0.032, label=vnf, fs=8)
#         ax.text(x, y + 0.07, f'V{k+1}', ha='center', va='bottom',
#                 fontsize=9, color='#1a5276', fontweight='bold',
#                 transform=ax.transAxes)
#
#     # Dest 节点（红色，按 sorted 顺序，和 dest_norm 一致）
#     for d in sorted(dests):
#         draw_node(*dest_norm[d], '#c0392b', '#922b21', 0.028, label=d, fs=8)
#
#     # Source 节点（绿色）
#     if source is not None and source in spine_norm:
#         draw_node(*spine_norm[source], '#27ae60', '#1e8449', 0.036, label=source, fs=9)
#
#     # ── 9. 图例 & 标题 ────────────────────────────────────────────────────────
#     patches = [
#         mpatches.Patch(color='#27ae60', label='Source'),
#         mpatches.Patch(color='#2980b9', label='VNF'),
#         mpatches.Patch(color='#bdc3c7', label='Relay'),
#         mpatches.Patch(color='#c0392b', label='Dest'),
#         mpatches.Patch(color='#2c3e50', label='Spine'),
#         mpatches.Patch(color='#e67e22', label='Branch'),
#     ]
#     ax.legend(handles=patches, loc='upper left', fontsize=8,
#               bbox_to_anchor=(0.0, 1.0), framealpha=0.9,
#               edgecolor='#ccc', ncol=6, handlelength=1.2)
#
#     n_dest_actual = len(dests)
#     flow = ' → '.join(['S'] + [f'V{k+1}' for k in range(len(chain))] + [f'D×{n_dest_actual}'])
#     ax.text(0.5, 0.03, flow, ha='center', va='bottom',
#             fontsize=9, color='#555', style='italic',
#             transform=ax.transAxes)
#
#     ep   = data.get('ep', '')
#     succ = data.get('success', None)
#     title = "SFC Multicast Tree (Layered DAG Model)"
#     if ep != '':
#         status = ' ✓' if succ else ' ✗' if succ is not None else ''
#         title += f"  —  Ep {ep}{status}"
#     ax.set_title(title, fontsize=13, fontweight='bold', pad=16)
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✓ 论文级图已保存: {save_path}")