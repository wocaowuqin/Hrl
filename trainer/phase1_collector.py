# trainer/phase1_collector.py - 时间槽版本（修复样本收集逻辑）
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
    Phase 1 专家数据收集器（时间槽版本 - 修复版）

    🔥 修复问题：
    原来: 只保存了60个样本（达到max_episodes=5000后停止，但5000是请求数不是样本数）
    现在: 正确收集5000个样本（每个成功请求可能产生多个路径样本）

    关键修复：
    1. max_episodes 现在指的是"成功样本数"而不是"处理的请求数"
    2. 正确统计和保存所有成功的路径样本
    3. 在达到目标样本数时停止，而不是处理完所有请求
    """

    def __init__(self, env, expert_solver, output_dir: str, max_episodes: int = 5000,
                 save_every: int = 500, use_timeslot: bool = True):
        self.env = env
        self.expert = expert_solver
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 🔥 修复：max_requests → max_success_samples，明确含义
        self.max_success_samples = max_episodes
        self.save_every = save_every

        self.success_samples = []
        self.fail_contexts = []
        # 🔥 添加paths_collected统计
        self.stats = {"requests": 0, "success": 0, "fail": 0, "paths_collected": 0}

        # 时间槽系统配置
        self.use_timeslot = use_timeslot
        self.timeslot_stats = {
            'total_time_slots': 0,
            'requests_per_slot': [],
            'current_time_slot': 0
        }

    def _estimate_load(self):
        rm = self.env.resource_mgr
        bw_util = 1.0 - rm.B.mean() / max(1.0, rm.B_cap)
        cpu_util = 1.0 - rm.C.mean() / max(1.0, rm.C_cap)
        return 0.5 * bw_util + 0.5 * cpu_util

    def _sanitize_request(self, req):
        """将 Request 对象转换为普通字典，防止 Pickle 报错"""
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
        """转换请求索引：1-based → 0-based"""
        req = self._sanitize_request(raw_req)

        # Source: 1-based → 0-based
        src = req.get("source", 0)
        if isinstance(src, (list, np.ndarray)):
            src = src.item()
        if src > 0:
            src = src - 1
        req['source'] = int(src)

        # Dest: 1-based → 0-based
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

        # VNF: 1-based → 0-based
        new_vnfs = req.get('vnf', [])
        if hasattr(new_vnfs, 'flatten'):
            new_vnfs = new_vnfs.flatten()
        vnf_list = []
        for v in new_vnfs:
            v_val = int(v)
            if v_val > 0:
                v_val = v_val - 1
            vnf_list.append(v_val)
        req['vnf'] = vnf_list

        # 🔥 修复：处理带宽字段
        if 'bandwidth' not in req or req['bandwidth'] is None:
            # 优先使用 bw_origin
            req['bandwidth'] = req.get('bw_origin', 3.0)

        # 🔥 修复：处理CPU和内存字段
        if 'cpu' not in req or req['cpu'] is None:
            req['cpu'] = req.get('cpu_origin', [1.0] * len(vnf_list))

        if 'memory' not in req or req['memory'] is None:
            req['memory'] = req.get('memory_origin', [1.0] * len(vnf_list))

        return req

    def _try_auto_load_timeslot_data(self):
        """自动尝试加载时间槽数据"""
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
        """加载时间槽数据"""
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
        """主收集方法"""
        logger.info("🚀 Starting Phase 1: Expert Data Collection")
        logger.info(f"   目标样本数: {self.max_success_samples}")  # 🔥 明确是样本数

        self.load_timeslot_data()
        self.env.reset()

        # 获取数据
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

        # 选择收集策略
        if events is not None and not self.use_timeslot:
            return self._collect_from_events(events, requests)
        else:
            return self._collect_from_requests(requests)

    def _collect_from_events(self, events, requests):
        """Event-based收集"""
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

                # 🔥 修复：检查样本数而不是请求数
                if len(self.success_samples) >= self.max_success_samples:
                    logger.info(f"\n✅ 达到目标样本数: {len(self.success_samples)}")
                    break

            if len(self.success_samples) >= self.max_success_samples:
                break

        pbar.close()
        self._save_final()
        return self.stats

    def _collect_from_requests(self, requests):
        """时间槽收集"""
        pbar = tqdm(desc="Collecting HRL Data (Time Slot)", ncols=120)

        for raw_req in requests:
            self._process_single_request(raw_req, pbar)

            # 🔥 修复：检查样本数而不是请求数
            if len(self.success_samples) >= self.max_success_samples:
                logger.info(f"\n✅ 达到目标样本数: {len(self.success_samples)}")
                break

        pbar.close()
        self._save_final()
        return self.stats

    def _process_single_request(self, raw_req, pbar):
        """处理单个请求"""
        self.stats["requests"] += 1
        pbar.update(1)

        # 调试输出（可选，收集完成后可以删除）
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

        # 🔥 关键修复：处理时间槽变化并释放过期资源
        if self.use_timeslot:
            current_slot = req.get('time_slot', 0)
            if current_slot != self.timeslot_stats['current_time_slot']:
                self.timeslot_stats['current_time_slot'] = current_slot
                self.timeslot_stats['total_time_slots'] += 1

                # 🔥 释放过期请求的资源
                try:
                    if hasattr(self.env, 'event_handler') and hasattr(self.env.event_handler, 'services'):
                        expired_services = []
                        for service_id, service_info in list(self.env.event_handler.services.items()):
                            service_req = service_info.get('req', {})
                            leave_slot = service_req.get('leave_time_slot', float('inf'))

                            # 如果请求已过期，标记为释放
                            if leave_slot <= current_slot:
                                expired_services.append(service_id)

                        # 释放过期服务
                        for service_id in expired_services:
                            try:
                                self.env.event_handler.unregister_service(service_id)
                            except Exception as e:
                                # 忽略释放失败
                                pass

                        # 打印释放信息（可选）
                        if expired_services and self.stats["requests"] % 100 == 0:
                            print(f"\n[时间槽 {current_slot}] 释放了 {len(expired_services)} 个过期服务")

                except Exception as e:
                    # 如果释放失败，记录但不中断
                    pass

        # 专家求解
        network_state = self.env.resource_mgr.get_network_state_dict(req)
        expert_result = self.expert.solve_request_for_expert(req, network_state)

        success = False
        if expert_result is not None:
            tree_data, expert_traj = expert_result

            if tree_data is not None:
                dict_tree = {}
                paths_map = tree_data.get('paths_map', {})

                # 🔥 检查 paths_map 是否为空
                if not paths_map:
                    self.stats["fail"] += 1
                    # 更新进度条
                    br = self.stats["fail"] / max(1, self.stats["requests"])
                    pbar.set_postfix({
                        "reqs": self.stats["requests"],
                        "succ": self.stats["success"],
                        "samples": len(self.success_samples),
                        "BR": f"{br:.1%}"
                    })
                    return  # 直接返回

                deployment_plan = {'hvt': tree_data['hvt'], 'tree': dict_tree}

                for dest, path_nodes in paths_map.items():
                    path_0 = [n - 1 if n > 0 else 0 for n in path_nodes]
                    for i in range(len(path_0) - 1):
                        u, v = path_0[i], path_0[i + 1]
                        dict_tree[(u, v)] = 1.0
                        dict_tree[(v, u)] = 1.0

                if self.env.resource_mgr.apply_tree_deployment(deployment_plan, req):
                    success = True
                    self.stats["success"] += 1

                    req_id = req.get('id', self.stats["requests"])
                    self.env.event_handler.services[req_id] = {
                        'req': req, 'tree': deployment_plan, 'hvt': tree_data['hvt']
                    }

                    # 构造样本
                    self.env.current_request = req
                    dummy_tree = {'tree': {}, 'hvt': np.zeros((self.env.n, 8))}
                    x, edge_index, edge_attr, req_vec = self.env.resource_mgr.get_graph_state(
                        current_request=req, nodes_on_tree={req['source']},
                        current_tree=dummy_tree, served_dest_count=0,
                        sharing_strategy=0, nb_high_goals=10
                    )
                    state_to_save = Data(
                        x=x.cpu(), edge_index=edge_index.cpu(),
                        edge_attr=edge_attr.cpu(), req_vec=req_vec.cpu()
                    )

                    clean_req = req.copy()

                    # 为每个路径创建样本
                    for dest, path_nodes in paths_map.items():
                        path_0 = [int(n - 1 if n > 0 else 0) for n in path_nodes]

                        sample_data = {
                            "state": state_to_save,
                            "request": clean_req,
                            "action": {"path": path_0},
                            "cost": 0.0,
                            "load": self._estimate_load(),
                            "hrl_info": {
                                "subgoal": int(path_0[-1]),
                                "full_path": path_0
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

        if not success:
            self.stats["fail"] += 1

        # 更新进度条
        br = self.stats["fail"] / max(1, self.stats["requests"])
        pbar.set_postfix({
            "reqs": self.stats["requests"],
            "succ": self.stats["success"],
            "samples": len(self.success_samples),
            "BR": f"{br:.1%}"
        })
    def _save_final(self):
        """保存数据"""
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