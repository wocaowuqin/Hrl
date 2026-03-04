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

            # ── [FIX SFC路径约束] dest阶段强制路径经过全部VNF节点 ──────────────
            # 原逻辑：直接计算 current_pos→dest_i 最短路，完全绕过VNF主干
            # 修复：分段计算 current→VNF1→VNF2→VNF3→dest_i，拼接成完整路径
            # 保证每条dest连接路径上必然经过全部VNF节点，满足SFC约束
            if _vnf_done and len(_chain) >= 1:
                waypoints = list(_chain) + [target_node]
                full_path = []
                seg_start = current_pos
                path_ok = True
                for wp in waypoints:
                    if seg_start == wp:
                        continue
                    seg = self.env.low_level_controller.compute_bw_aware_path(seg_start, wp)
                    if not seg:
                        path_ok = False
                        break
                    full_path.extend(seg[1:] if len(seg) > 1 else [wp])
                    seg_start = wp
                if path_ok and full_path:
                    planned_path = full_path
                    logger.debug(f"[Coord] SFC路径 waypoints={waypoints} path={planned_path}")
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
        # ──────────────────────────────────────────────────────────────────

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

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨  多播树可视化工具函数
#     用法（训练结束后）：
#       from core.hrl.HRL_Coordinator import visualize_multicast_trees
#       visualize_multicast_trees(trees_data, save_path="multicast_trees_vis.png")
#
#     trees_data 格式（每个episode收集一条）：
#       {
#         'ep':      episode编号 (int),
#         'success': 是否成功 (bool),
#         'req':     req_snapshot (dict, from info['req_snapshot']),
#         'tree':    tree_snapshot (dict, from info['tree_snapshot']),
#         'chain':   chain_nodes (list, from info['chain_nodes']),
#       }
# ═══════════════════════════════════════════════════════════════════════════════
def visualize_multicast_trees(trees_data, save_path="multicast_trees_vis.png",
                               cols=5, rows=2, figsize=(22, 9)):
    """
    把 trees_data 列表里前 cols*rows 个episode的多播树画成网格图并保存。
    每棵树中：绿色=源节点, 红色=目的地, 蓝色=VNF节点, 灰色=中间节点。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx

    n_show = min(len(trees_data), cols * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # 图例只画一次
    legend_drawn = False

    for idx in range(rows * cols):
        ax = axes[idx]
        if idx >= n_show:
            ax.axis('off')
            continue

        data    = trees_data[idx]
        req     = data.get('req') or {}
        tree    = data.get('tree') or {}
        ep      = data.get('ep', idx + 1)
        success = data.get('success', False)
        chain   = data.get('chain', [])

        title_color = 'green' if success else 'red'
        ax.set_title(f"Ep {ep} (" + ("Success" if success else "Fail") + ")",
                     color=title_color, fontweight='bold', fontsize=9)

        edges = list(tree.get('tree', {}).keys())
        if not edges:
            ax.text(0.5, 0.5, "No Tree", ha='center', va='center',
                    color='gray', fontsize=10)
            ax.axis('off')
            continue

        G = nx.Graph()
        G.add_edges_from(edges)

        source = req.get('source')
        dests  = [int(d) for d in req.get('dest', [])]

        # VNF节点：优先用 chain_nodes，fallback用 placement
        if chain:
            vnf_nodes = list(chain)
        else:
            vnf_nodes = []
            placement = tree.get('placement', {})
            for p_k, p_v in placement.items():
                if isinstance(p_v, dict) and 'node' in p_v:
                    vnf_nodes.append(p_v['node'])
                elif isinstance(p_k, tuple) and len(p_k) > 0:
                    vnf_nodes.append(p_k[0])

        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.random_layout(G)

        all_nodes  = list(G.nodes())
        gray_nodes = [n for n in all_nodes
                      if n != source and n not in dests and n not in vnf_nodes]

        # 画各类节点
        if gray_nodes:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=gray_nodes,
                                   node_color='#bdc3c7', node_size=260)
        vnf_in_G = [v for v in vnf_nodes if v in G.nodes()]
        if vnf_in_G:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=vnf_in_G,
                                   node_color='#3498db', node_size=320,
                                   edgecolors='#2980b9', linewidths=1.5)
        dest_in_G = [d for d in dests if d in G.nodes()]
        if dest_in_G:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=dest_in_G,
                                   node_color='#e74c3c', node_size=340,
                                   edgecolors='#c0392b', linewidths=1.5)
        if source is not None and source in G.nodes():
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[source],
                                   node_color='#2ecc71', node_size=380,
                                   edgecolors='#27ae60', linewidths=2)

        nx.draw_networkx_edges(G, pos, ax=ax, width=1.4,
                               edge_color='#7f8c8d', alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')

        if not legend_drawn:
            ax.plot([], [], 'o', color='#2ecc71', label='Source', markersize=8)
            ax.plot([], [], 'o', color='#e74c3c', label='Dest',   markersize=8)
            ax.plot([], [], 'o', color='#3498db', label='VNF',    markersize=8)
            ax.legend(loc='upper left', fontsize=7, markerscale=0.9,
                      framealpha=0.8)
            legend_drawn = True

        ax.axis('off')

    plt.suptitle("Multicast Trees Visualization", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 多播树可视化已保存: {save_path}")
    return save_path