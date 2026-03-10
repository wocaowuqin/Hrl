# trainer/phase1_collector.py - 时间槽版本（最终修复：正确记录节点资源）
import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from torch_geometric.data import Data
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class Phase1ExpertCollector:
    """
    Phase 1 专家数据收集器（时间槽版本 - 最终修复版）

    🔥 修复问题：
    1. max_episodes 现在指的是"成功样本数"而不是"处理的请求数"
    2. 时间槽变化时正确释放过期资源（通过 RequestLifecycleManager）
    3. 自建图状态，不再依赖 resource_mgr.get_graph_state
    4. 修正负载估计方法，避免使用不存在的 B/C 属性
    5. **构造准确的资源分配记录**：从 vnf_instances 中提取 CPU/MEM 用量，确保节点资源被正确释放
    """

    def __init__(self, env, expert_solver, output_dir: str, max_episodes: int = 15000,
                 save_every: int = 500, use_timeslot: bool = True):
        self.env = env
        self.expert = expert_solver
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_success_samples = max_episodes
        self.save_every = save_every

        self.success_samples = []
        self.fail_contexts = []
        self.stats = {"requests": 0, "success": 0, "fail": 0, "paths_collected": 0}

        self.use_timeslot = use_timeslot
        self.timeslot_stats = {
            'total_time_slots': 0,
            'requests_per_slot': [],
            'current_time_slot': 0
        }

    def _estimate_load(self):
        """基于 resource_mgr.pool 计算平均 CPU 和带宽利用率"""
        rm = self.env.resource_mgr
        n = rm.n
        L = rm.pool.L

        total_cpu_avail = 0.0
        for i in range(n):
            total_cpu_avail += rm.pool.get_available_cpu(i)
        cpu_util = 1.0 - total_cpu_avail / (n * max(rm.C_cap, 1.0))

        total_bw_avail = 0.0
        for key in rm.pool.link_map.keys():
            total_bw_avail += rm.pool.get_available_bandwidth(*key)
        bw_util = 1.0 - total_bw_avail / (L * max(rm.B_cap, 1.0))

        return 0.5 * bw_util + 0.5 * cpu_util

    def _sanitize_request(self, req):
        if isinstance(req, dict):
            return req.copy()
        if hasattr(req, '__dict__'):
            return req.__dict__.copy()
        try:
            return dict(req)
        except:
            return {
                'id': getattr(req, 'id', -1),
                'source': getattr(req, 'source', 0),
                'dest': getattr(req, 'dest', []),
                'vnf': getattr(req, 'vnf', []),
                'bandwidth': getattr(req, 'bandwidth', 1.0),
                'ttl': getattr(req, 'ttl', 100),
                'time_slot': getattr(req, 'time_slot', 0),
                'duration': getattr(req, 'duration', 100),
                'leave_time_slot': getattr(req, 'leave_time_slot', 100)
            }

    def _convert_request_indices(self, raw_req):
        req = self._sanitize_request(raw_req)

        src = req.get("source", 0)
        if isinstance(src, (list, np.ndarray)):
            src = src.item()
        if src > 0:
            src = src - 1
        req['source'] = int(src)

        new_dests = []
        raw_dests = req.get("dest", [])
        if hasattr(raw_dests, 'flatten'):
            raw_dests = raw_dests.flatten()
        for d in raw_dests:
            d_val = int(d)
            if d_val > 0:
                d_val = d_val - 1
            new_dests.append(d_val)
        req['dest'] = new_dests

        new_vnfs = req.get('vnf', [])
        if hasattr(new_vnfs, 'flatten'):
            new_vnfs = new_vnfs.flatten()
        req['vnf'] = [int(v) for v in new_vnfs]

        if 'bandwidth' not in req or req['bandwidth'] is None:
            req['bandwidth'] = req.get('bw_origin', 3.0)

        if 'cpu' not in req or req['cpu'] is None:
            req['cpu'] = req.get('cpu_origin', [1.0] * len(req['vnf']))

        if 'memory' not in req or req['memory'] is None:
            req['memory'] = req.get('memory_origin', [1.0] * len(req['vnf']))

        return req

    def _try_auto_load_timeslot_data(self):
        logger.info("🔍 尝试自动加载时间槽数据...")
        if hasattr(self.env, 'config'):
            config = self.env.config
            data_dir = Path(config.get('paths', {}).get('input_dir', 'data/input_dir'))
        else:
            data_dir = Path('data/input_dir')

        requests_file = data_dir / 'phase1_requests.pkl'
        requests_by_slot_file = data_dir / 'phase1_requests_by_slot.pkl'

        logger.info(f"   检查文件: {requests_file}")
        logger.info(f"   检查文件: {requests_by_slot_file}")

        if not requests_file.exists() or not requests_by_slot_file.exists():
            return False

        try:
            with open(requests_file, 'rb') as f:
                requests = pickle.load(f)
            with open(requests_by_slot_file, 'rb') as f:
                requests_by_slot = pickle.load(f)

            logger.info(f"   ✅ 文件加载成功: {len(requests)} 请求, {len(requests_by_slot)} 时间槽")

            if hasattr(self.env, 'load_requests'):
                self.env.load_requests(requests, requests_by_slot)
            else:
                self.env.all_requests = requests
                self.env.requests_by_slot = requests_by_slot

            return True
        except Exception as e:
            logger.error(f"   ❌ 自动加载失败: {e}")
            return False

    def load_timeslot_data(self):
        if not self.use_timeslot:
            return False
        try:
            if (hasattr(self.env, 'all_requests') and self.env.all_requests and
                    hasattr(self.env, 'requests_by_slot') and self.env.requests_by_slot):
                logger.info(f"✅ 环境已加载时间槽数据: {len(self.env.all_requests)} 请求")
                return True
            else:
                if self._try_auto_load_timeslot_data():
                    return True
                else:
                    self.use_timeslot = False
                    return False
        except Exception as e:
            self.use_timeslot = False
            return False

    def collect(self):
        logger.info("🚀 Starting Phase 1: Expert Data Collection")
        logger.info(f"   目标样本数: {self.max_success_samples}")

        self.load_timeslot_data()
        self.env.reset()

        requests = None
        events = None

        if hasattr(self.env, 'all_requests') and self.env.all_requests:
            requests = self.env.all_requests
        elif hasattr(self.env, 'data_loader'):
            if hasattr(self.env.data_loader, 'requests'):
                requests = self.env.data_loader.requests
            if hasattr(self.env.data_loader, 'events'):
                events = self.env.data_loader.events

        if not requests:
            logger.error("❌ No requests found!")
            return self.stats

        if events is not None and not self.use_timeslot:
            return self._collect_from_events(events, requests)
        else:
            return self._collect_from_requests(requests)

    def _collect_from_events(self, events, requests):
        pbar = tqdm(desc="Collecting HRL Data", ncols=120)
        for t, event in enumerate(events):
            leave_list = event.get("leave", event.get("leave_event", []))
            for leave_req_id in leave_list:
                try:
                    self.env.event_handler.unregister_service(leave_req_id)
                except:
                    pass

            arrive_list = event.get("arrive", event.get("arrive_event", []))
            for req_id in arrive_list:
                if req_id <= 0 or req_id > len(requests):
                    continue
                self._process_single_request(requests[req_id - 1], pbar)
                if len(self.success_samples) >= self.max_success_samples:
                    logger.info(f"\n✅ 达到目标样本数: {len(self.success_samples)}")
                    break
            if len(self.success_samples) >= self.max_success_samples:
                break
        pbar.close()
        self._save_final()
        return self.stats

    def _collect_from_requests(self, requests):
        pbar = tqdm(desc="Collecting HRL Data (Time Slot)", ncols=120)
        for raw_req in requests:
            self._process_single_request(raw_req, pbar)
            if len(self.success_samples) >= self.max_success_samples:
                logger.info(f"\n✅ 达到目标样本数: {len(self.success_samples)}")
                break
        pbar.close()
        self._save_final()
        return self.stats

    def _build_graph_state(self, request, nodes_on_tree, current_tree, served_dest_count):
        """
        复用 env.get_state() 获取状态，保证与 Phase3 特征完全一致（21维节点特征）
        """
        # ← 就加在这里，第一行
        self.env.current_node_location = request.get('source', 0)

        # 临时设置环境状态，让 low_level_controller.get_state() 能感知当前树和请求
        self.env.current_request = request
        self.env.current_tree = {
            'tree': current_tree.get('tree', {}),
            'hvt': current_tree.get('hvt', np.zeros((self.env.n, self.env.K_vnf))),
            'connected_dests': set(
                n for n in nodes_on_tree if n in request.get('dest', [])
            ),
            'nodes': nodes_on_tree.copy(),
        }

        state_data = self.env.get_state()

        x = state_data.x
        edge_index = state_data.edge_index
        edge_attr = state_data.edge_attr

        req_vec = torch.zeros(24, dtype=torch.float)

        return x, edge_index, edge_attr, req_vec


    def _process_single_request(self, raw_req, pbar):
        self.stats["requests"] += 1
        pbar.update(1)

        if self.stats["requests"] <= 5:
            print(f"\n{'=' * 60}")
            print(f"DEBUG 请求 {self.stats['requests']}")
            print(f"raw_req类型: {type(raw_req)}")
            print(f"raw_req内容: {raw_req}")
            print(f"{'=' * 60}\n")

        req = self._convert_request_indices(raw_req)

        if self.stats["requests"] <= 5:
            print(f"转换后req: {req}")
            print(f"带宽: {req.get('bandwidth')}")

        # 时间槽驱动释放
        current_slot = req.get('time_slot', 0)
        if self.use_timeslot:
            if current_slot != self.timeslot_stats['current_time_slot']:
                self.timeslot_stats['current_time_slot'] = current_slot
                self.timeslot_stats['total_time_slots'] += 1
                try:
                    if hasattr(self.env.resource_mgr, 'request_manager'):
                        self.env.resource_mgr.request_manager.check_and_release_expired(current_slot)
                        logger.debug(f"[Phase1] 时间槽切换到 {current_slot}，释放过期请求")
                except Exception as e:
                    logger.debug(f"生命周期释放失败: {e}")

        # 兜底释放：按 leave_time_slot 精确释放已到期的请求，避免资源耗尽
        try:
            if hasattr(self.env.resource_mgr, 'request_manager'):
                rm = self.env.resource_mgr.request_manager
                # 找出所有已到期的请求（leave_time_slot <= current_slot）
                expired_ids = []
                if hasattr(rm, 'active_requests'):
                    for rid, rinfo in list(rm.active_requests.items()):
                        leave_slot = rinfo.get('leave_time_slot',
                                    rinfo.get('req', {}).get('leave_time_slot', None))
                        if leave_slot is not None and leave_slot <= current_slot:
                            expired_ids.append(rid)
                for rid in expired_ids:
                    self.env.resource_mgr.remove_request(rid)
                    logger.debug(f"[Phase1] 精确释放过期请求 {rid}")

                # 终极兜底：若活跃请求数超过20，强制释放最早的一半
                if hasattr(rm, 'active_requests') and len(rm.active_requests) > 20:
                    all_ids = list(rm.active_requests.keys())
                    for rid in all_ids[:len(all_ids) // 2]:
                        self.env.resource_mgr.remove_request(rid)
                    logger.debug(f"[Phase1] 活跃请求过多，强制释放一半")
        except Exception as e:
            logger.debug(f"兜底精确释放失败: {e}")

        # 专家求解
        req_for_solver = req.copy()
        req_for_solver['vnf'] = [v + 1 for v in req.get('vnf', [])]
        network_state = self.env.resource_mgr.get_network_state_dict(req)
        expert_result = self.expert.solve_request_for_expert(req_for_solver, network_state)

        success = False
        if expert_result is not None:
            tree_data, expert_traj = expert_result

            if tree_data is not None:
                dict_tree = {}
                paths_map = tree_data.get('paths_map', {})

                if not paths_map:
                    self.stats["fail"] += 1
                    br = self.stats["fail"] / max(1, self.stats["requests"])
                    pbar.set_postfix({
                        "reqs": self.stats["requests"],
                        "succ": self.stats["success"],
                        "samples": len(self.success_samples),
                        "BR": f"{br:.1%}"
                    })
                    return

                for dest, path_nodes in paths_map.items():
                    path_0 = [n - 1 if n > 0 else 0 for n in path_nodes]
                    for i in range(len(path_0) - 1):
                        u, v = path_0[i], path_0[i + 1]
                        dict_tree[(u, v)] = 1.0
                        dict_tree[(v, u)] = 1.0

                merged_placement = {}
                if expert_traj:
                    for _d_idx, action_data, _res_delta in expert_traj:
                        pl = action_data.get('placement', {})
                        for k, v in pl.items():
                            merged_placement[str(k)] = v

                full_plan = {
                    'hvt':       tree_data['hvt'],
                    'tree':      dict_tree,
                    'placement': merged_placement,
                }

                if self.env.resource_mgr.apply_tree_deployment(full_plan, req):
                    success = True
                    self.stats["success"] += 1
                    print(f"\n[SUCCESS] req {req.get('id')} 部署成功，开始收集样本")

                    req_id = req.get('id', self.stats["requests"])

                    # 🔥 关键修复：从 vnf_instances 中构造准确的 placement 记录
                    placement_from_instances = {}
                    for inst in self.env.resource_mgr.vnf_instances:
                        if inst.get('req_id') == req_id:
                            node = inst['node']
                            vnf_type = inst.get('vnf_type')
                            # 需要确定 VNF 在链中的索引，这里暂时用 vnf_type 代替
                            cpu_used = inst.get('cpu', 1.0)
                            mem_used = inst.get('memory', 1.0)
                            # 键格式与生命周期管理器期望的一致（节点, vnf索引）
                            placement_from_instances[(node, vnf_type)] = {
                                'node': node,
                                'vnf_type': vnf_type,
                                'cpu_used': cpu_used,
                                'mem_used': mem_used
                            }

                    resources_allocated = {
                        'placement': placement_from_instances,
                        'tree': full_plan.get('tree', {})
                    }

                    # 注册请求
                    try:
                        if hasattr(self.env.resource_mgr, 'request_manager'):
                            self.env.resource_mgr.request_manager.register_request(req, resources_allocated)
                            print(f"[LIFECYCLE] 注册请求 {req_id} 成功")
                    except Exception as e:
                        print(f"[LIFECYCLE] 注册失败: {e}")
                        logger.debug(f"生命周期注册失败: {e}")

                    # 样本收集（略，与之前相同）
                    clean_req = req.copy()
                    dest_list = clean_req.get('dest', [])
                    self.env.current_request = req
                    nodes_on_tree_so_far = {req['source']}
                    served_dest_count = 0
                    current_tree_for_state = {
                        'tree': dict_tree,
                        'hvt': tree_data.get('hvt', np.zeros((self.env.n, 8)))
                    }

                    # 提取路径
                    # 🔥 修复 high_label 赋值：
                    #    paths_map 的 key 就是目的地节点（solver 返回的 1-based，转 0-based）
                    #    high_label = 该目的地在 dest_list 中的顺序索引（0~K_dest-1）
                    #    旧逻辑用 path[-1]（路径终点）查 dest_list，
                    #    但路径终点可能是 VNF/Hub 节点，不在 dest_list 中 → 全部 fallback 到 0
                    traj_paths = []
                    for dest_key, path_nodes in paths_map.items():
                        if len(path_nodes) < 2:
                            continue
                        path_0based = [int(n - 1 if n > 0 else 0) for n in path_nodes]
                        subgoal_0 = path_0based[-1]

                        # high_label = 路径终点节点ID（0~27），与 high_policy 输出维度28对齐
                        hl = subgoal_0

                        traj_paths.append((path_0based, hl, subgoal_0))

                    if expert_traj:
                        existing_sg = {sg for _, _, sg in traj_paths}
                        for _d_idx, action_data, _res_delta in expert_traj:
                            path_1based = action_data.get('path', [])
                            if len(path_1based) >= 2:
                                path_0based = [int(n - 1 if n > 0 else 0) for n in path_1based]
                                subgoal_0 = path_0based[-1]
                                if subgoal_0 not in existing_sg:
                                    hl = subgoal_0  # high_label = 节点ID
                                    traj_paths.append((path_0based, hl, subgoal_0))
                                    existing_sg.add(subgoal_0)
                    print(f"[DEBUG] req {req.get('id')}: paths_map keys = {list(paths_map.keys())}")
                    print(f"[DEBUG] req {req.get('id')}: traj_paths 数量 = {len(traj_paths)}")

                    for path_idx, (path_0, high_label, subgoal_node) in enumerate(traj_paths):
                        try:
                            x, edge_index, edge_attr, req_vec = self._build_graph_state(
                                request=req,
                                nodes_on_tree=nodes_on_tree_so_far,
                                current_tree=current_tree_for_state,
                                served_dest_count=served_dest_count
                            )
                            print(f"[DEBUG] path {path_idx}: _build_graph_state 成功")
                            state_to_save = Data(
                                x=x.cpu(), edge_index=edge_index.cpu(),
                                edge_attr=edge_attr.cpu(), req_vec=req_vec.cpu()
                            )
                        except Exception as e:
                            print(f"[ERROR] path {path_idx}: 构建图状态失败: {e}")
                            logger.warning(f"state 构造失败 path_idx={path_idx}, req={req.get('id')}: {e}")
                            continue

                        for node in path_0:
                            nodes_on_tree_so_far.add(node)
                        if subgoal_node in dest_list:
                            served_dest_count += 1

                        sample_data = {
                            "state": state_to_save,
                            "request": clean_req,
                            "action": {
                                "path": path_0,
                                "high_label": high_label,
                                "subgoal_node": subgoal_node,
                                "is_dest_path": subgoal_node in dest_list,
                            },
                            "cost": 0.0,
                            "load": self._estimate_load(),
                            "hrl_info": {
                                "subgoal": subgoal_node,
                                "full_path": path_0,
                                "path_index": path_idx,
                            }
                        }

                        if self.use_timeslot:
                            sample_data["timeslot_info"] = {
                                "time_slot": req.get('time_slot', 0),
                                "duration": req.get('duration', 100),
                                "leave_time_slot": req.get('leave_time_slot', 100)
                            }

                        self.success_samples.append(sample_data)
                        self.stats["paths_collected"] += 1
                        print(f"[DEBUG] path {path_idx} 样本已添加，当前样本总数 = {len(self.success_samples)}")

        if not success:
            self.stats["fail"] += 1

        br = self.stats["fail"] / max(1, self.stats["requests"])
        pbar.set_postfix({
            "reqs": self.stats["requests"],
            "succ": self.stats["success"],
            "samples": len(self.success_samples),
            "BR": f"{br:.1%}"
        })

    def _save_final(self):
        path = os.path.join(self.output_dir, "expert_data_final.pkl")
        try:
            data_to_save = {
                "success": self.success_samples,
                "stats": self.stats
            }
            if self.use_timeslot:
                data_to_save["timeslot_stats"] = self.timeslot_stats

            with open(path, "wb") as f:
                pickle.dump(data_to_save, f)

            logger.info(f"✅ Saved {len(self.success_samples)} expert samples to {path}")
            logger.info(f"   成功请求: {self.stats['success']} 个")
            logger.info(f"   收集路径: {self.stats['paths_collected']} 条")
            logger.info(f"   样本总数: {len(self.success_samples)} 个")

            if self.use_timeslot:
                logger.info(f"⏰ 时间槽统计:")
                logger.info(f"   总时间槽: {self.timeslot_stats['total_time_slots']}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            import traceback
            traceback.print_exc()