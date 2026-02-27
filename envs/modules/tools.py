import numpy as np
import networkx as nx
import torch
import logging

logger = logging.getLogger(__name__)


class SFCToolkit:
    """
    SFC 环境辅助工具箱
    职责：封装环境的验证、资源提交、回滚、时间推进等辅助逻辑
    """

    def __init__(self, env):
        """
        :param env: SFC_HIRL_Env 的实例引用
        """
        self.env = env

    # =========================================================================
    # 🔍 验证与检查类 (Validation & Checks)
    # =========================================================================

    def validate_sfc_paths(self, parent_map):
        """
        🔥 [增强版] 验证 SFC 路径完整性
        """
        if not self.env.current_request:
            return False, ["No request"]

        req = self.env.current_request
        source = req['source']
        dests = req.get('dest', [])
        required_vnfs = req.get('vnf', [])

        # 如果没有 VNF 要求，直接通过
        if not required_vnfs:
            return True, []

        # 构建节点 VNF 映射
        node_vnf_dict = {}  # {node: [vnf_types]}
        placement = self.env.current_tree.get('placement', {})

        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v = key[0], key[1]
                if n not in node_vnf_dict:
                    node_vnf_dict[n] = []
                node_vnf_dict[n].append(v)

        print(f"\n🔍 [SFC 验证] 开始验证路径...")
        # print(f"   源节点: {source}, 目的地: {dests}") # 可选日志

        errors = []

        # 验证每个目的地的路径
        for dest in dests:
            # 1. 回溯路径
            path = []
            curr = dest
            while curr is not None:
                path.append(curr)
                if curr == source:
                    break
                curr = parent_map.get(curr)

            # 2. 检查路径完整性
            if not path or path[-1] != source:
                error = f"Dest {dest}: Path broken (无法从源节点到达)"
                errors.append(error)
                print(f"   ❌ {error}")
                continue

            path.reverse()  # Source -> Dest
            # print(f"   📍 目的地 {dest} 的路径: {path}")

            # 3. 收集路径上的 VNF
            path_vnfs = []
            for node in path:
                if node in node_vnf_dict:
                    deployed = node_vnf_dict[node]
                    for vnf in deployed:
                        path_vnfs.append((node, vnf))

            collected_vnf_types = [vnf for (node, vnf) in path_vnfs]

            # 4. 检查是否包含所有必需的 VNF
            for req_vnf in required_vnfs:
                if req_vnf not in collected_vnf_types:
                    error = f"Dest {dest}: 缺少 VNF {req_vnf}"
                    errors.append(error)
                    print(f"   ❌ {error}")
                    break
            else:
                # 5. 检查 VNF 顺序
                vnf_indices = []
                for vnf in collected_vnf_types:
                    if vnf in required_vnfs:
                        vnf_indices.append(required_vnfs.index(vnf))

                if vnf_indices != sorted(vnf_indices):
                    error = f"Dest {dest}: VNF 顺序错误"
                    errors.append(error)
                    print(f"   ❌ {error}")
                else:
                    pass
                    # print(f"   ✅ 目的地 {dest} 路径验证通过")

        success = (len(errors) == 0)
        if success:
            print(f"✅ [SFC 验证] 所有路径验证通过")
        else:
            print(f"❌ [SFC 验证] 发现 {len(errors)} 个错误")

        return success, errors

    def check_deployment_validity(self, node_id):
        """
        检查节点是否可以部署VNF
        """
        if not self.env.current_request:
            return False

        # 规则1: 源节点不能部署VNF
        source = self.env.current_request.get('source')
        if node_id == source:
            return False

        # 规则2: 必须是DC节点
        if hasattr(self.env, 'dc_nodes') and node_id not in self.env.dc_nodes:
            return False

        # 规则3: 检查资源 (调用环境内部方法或直接检查资源管理器)
        if hasattr(self.env, '_check_node_resources'):
            if not self.env._check_node_resources(node_id):
                return False

        return True

    def get_resource_utilization(self):
        """计算当前全网资源占用率"""
        try:
            total_cap = 0.0
            used_cap = 0.0

            # 访问 env.resource_mgr
            rm = self.env.resource_mgr
            if hasattr(rm, 'nodes'):
                nodes = rm.nodes
                if isinstance(nodes, list):
                    for n in nodes:
                        cap = n.get('capacity', n.get('cpu_limit', 100.0))
                        rem = n.get('cpu', 100.0)
                        total_cap += cap
                        used_cap += (cap - rem)
                elif isinstance(nodes, dict):
                    for n in nodes.values():
                        if isinstance(n, list): continue
                        cap = n.get('total', 100.0)
                        used = n.get('used', 0.0)
                        total_cap += cap
                        used_cap += used

            if total_cap <= 0: return 0.0
            return used_cap / total_cap
        except Exception as e:
            return 0.0

    def get_current_progress(self):
        """计算当前 SFC 部署进度比例"""
        if not self.env.current_request:
            return 0.0

        vnf_list = self.env.current_request.get('vnf', [])
        if not vnf_list:
            return 1.0

        curr_idx = getattr(self.env, 'current_vnf_idx', 0)  # 或者是 next_vnf_idx
        progress = float(curr_idx) / len(vnf_list)
        return progress

    # =========================================================================
    # ⏳ 时间与仿真控制类 (Time & Simulation)
    # =========================================================================

    def advance_to_next_active_slot(self):
        """时间槽推进逻辑"""
        if hasattr(self.env, 'slot_queue') and self.env.slot_queue:
            return

        start_slot = self.env.current_slot_index

        while not self.env.simulation_done:
            # 边界检查
            if self.env.current_slot_index > self.env.max_slot_index:
                print(f"🏁 [仿真结束] 已到达最大时间槽 {self.env.max_slot_index}")
                self.env.simulation_done = True
                return

            # 检查当前索引请求
            current_reqs = self.env.requests_by_slot.get(self.env.current_slot_index, [])

            if current_reqs:
                # 加载队列
                self.env.slot_queue = list(current_reqs)
                self.env.current_time_slot = self.env.current_slot_index

                # 更新时间
                if self.env.slot_queue:
                    first_req = self.env.slot_queue[0]
                    if isinstance(first_req, dict):
                        self.env.time_step = float(first_req.get('arrival_time',
                                                                 self.env.current_slot_index * self.env.delta_t))
                    else:
                        self.env.time_step = float(getattr(first_req, 'arrival_time',
                                                           self.env.current_slot_index * self.env.delta_t))
                else:
                    self.env.time_step = self.env.current_slot_index * self.env.delta_t

                print(f"⏩ [时间推进] Slot {start_slot} -> {self.env.current_slot_index} | "
                      f"Time: {self.env.time_step:.2f}s | 加载 {len(self.env.slot_queue)} 个请求")

                # 触发过期释放
                if hasattr(self.env, 'request_manager'):
                    try:
                        expired_ids = self.env.request_manager.check_and_release_expired(self.env.time_step)
                        if expired_ids:
                            res_util = self.get_resource_utilization()
                            print(f"♻️ [时间切片] 释放了 {len(expired_ids)} 个过期请求 (当前Res: {res_util:.1f}%)")
                    except Exception as e:
                        print(f"⚠️ [时间切片] 释放失败: {e}")

                self.env.current_slot_index += 1
                return

            self.env.current_slot_index += 1

    def get_next_request_online(self):
        """在线模式获取请求"""
        if not self.env.slot_queue:
            self.advance_to_next_active_slot()

        if self.env.simulation_done or not self.env.slot_queue:
            return None

        req_raw = self.env.slot_queue.pop(0)
        req = req_raw.to_dict() if hasattr(req_raw, 'to_dict') else (
            req_raw if isinstance(req_raw, dict) else req_raw.__dict__)

        # 更新时间
        new_arrival_time = float(req.get('arrival_time', self.env.time_step))
        if 'time_slot' not in req:
            slot_duration = getattr(self.env, 'slot_duration', 1.0)
            req['time_slot'] = int(new_arrival_time / slot_duration)

        new_time_slot = int(req['time_slot'])
        old_time_slot = self.env.current_time_slot

        if new_time_slot != old_time_slot:
            self.env.time_step = new_arrival_time
            self.env.current_time_slot = new_time_slot

            print(f"⏩ [时间推进] Slot {old_time_slot} -> {new_time_slot} | "
                  f"Time: {self.env.time_step:.2f}s | Res: {self.get_resource_utilization():.1f}%")

            if hasattr(self.env, 'request_manager'):
                self.env.request_manager.check_and_release_expired(self.env.time_step)
        else:
            self.env.time_step = new_arrival_time

        self.env._last_queue_size = len(self.env.slot_queue)
        return req

    # =========================================================================
    # 💳 资源提交与回滚类 (Commit & Rollback)
    # =========================================================================

    def commit_resources(self, pruned_tree, valid_nodes):
        """两阶段提交资源"""
        req = self.env.current_request
        bw_req = req.get('bw_origin', 1.0)

        pending_links = []
        pending_nodes = []

        # Phase 1: 收集
        for (u, v) in pruned_tree.keys():
            pending_links.append((u, v, bw_req))

        placement = self.env.current_tree.get('placement', {})
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v_type = key[0], key[1]
                if n in valid_nodes:
                    c = info.get('cpu_used', 1.0)
                    m = info.get('mem_used', 1.0)
                    pending_nodes.append((n, v_type, c, m))

        # Phase 2: 分配
        self.env.curr_ep_link_allocs = []
        self.env.curr_ep_node_allocs = []

        total_cpu, total_mem, total_bw = 0.0, 0.0, 0.0

        print(f"\n💳 [开始扣费] 节点={len(pending_nodes)}, 链路={len(pending_links)}")

        for u, v, bw in pending_links:
            result = self.env.resource_mgr.allocate_link_resource(u, v, bw)
            if result is not False:
                self.env.curr_ep_link_allocs.append((u, v, bw))
                total_bw += bw
            else:
                print(f"   ❌ 链路({u},{v}) 分配失败")

        for n, v_type, c, m in pending_nodes:
            result = self.env.resource_mgr.allocate_node_resource(n, v_type, c, m)
            if result is not False:
                self.env.curr_ep_node_allocs.append((n, v_type, c, m))
                total_cpu += c
                total_mem += m
            else:
                print(f"   ❌ 节点{n}[VNF{v_type}] 分配失败")

        print(f"💳 [扣费汇总] CPU:{total_cpu:.1f} | Mem:{total_mem:.1f} | BW:{total_bw:.1f}")
        return True

    def rollback_resources(self):
        """统一回滚 + 状态清理"""
        if not hasattr(self.env, 'current_tree') or self.env.current_request is None:
            return

        placement = self.env.current_tree.get('placement', {})
        tree_edges = self.env.current_tree.get('tree', {})
        bw = self.env.current_request.get('bw_origin', 1.0)

        restored_cpu, restored_bw = 0.0, 0.0

        for key, info in placement.items():
            if isinstance(key, tuple):
                node, vnf_type = key[0], key[1]
                c = info.get('cpu_used', 1.0) if isinstance(info, dict) else 1.0
                m = info.get('mem_used', 1.0) if isinstance(info, dict) else 1.0

                if hasattr(self.env.resource_mgr, 'release_node_resource'):
                    self.env.resource_mgr.release_node_resource(node, vnf_type, c, m)
                    restored_cpu += c

        for edge_key in tree_edges.keys():
            if isinstance(edge_key, tuple):
                u, v = edge_key
                if hasattr(self.env.resource_mgr, 'release_link_resource'):
                    self.env.resource_mgr.release_link_resource(u, v, bw)
                    restored_bw += bw

        if restored_cpu > 0 or restored_bw > 0:
            print(f"♻️ [资源回滚] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

        # 状态重置
        self.env.current_tree = {
            'hvt': np.zeros((self.env.n, self.env.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.env.nodes_on_tree = set()
        if hasattr(self.env, '_node_visit_count'): self.env._node_visit_count = {}

    def rollback_request_resources(self, req):
        """强制回滚指定请求"""
        if not req: return
        print(f"♻️ [回滚执行] 开始释放请求 {req.get('id')} 的资源...")

        # 这里的逻辑其实和 rollback_resources 差不多，但针对参数 req
        # 为了复用逻辑，可以考虑合并，但这里保留原样逻辑适配
        placement = self.env.current_tree.get('placement', {})
        restored_cpu = 0

        for key, info in placement.items():
            if isinstance(info, dict):
                node = info.get('node')
                v_type = info.get('vnf_type', 0)
                c = info.get('cpu_used', 0.0)
                m = info.get('mem_used', 0.0)
                if hasattr(self.env.resource_mgr, 'release_node_resource'):
                    try:
                        self.env.resource_mgr.release_node_resource(node, v_type, c, m)
                        restored_cpu += c
                    except:
                        pass

        tree_edges = self.env.current_tree.get('tree', {})
        restored_bw = 0
        bw_req = req.get('bw_origin', 1.0)

        for u, v in tree_edges.keys():
            if hasattr(self.env.resource_mgr, 'release_link_resource'):
                try:
                    self.env.resource_mgr.release_link_resource(u, v, bw_req)
                    restored_bw += bw_req
                except:
                    pass

        print(f"✅ [回滚完成] 节点: +{restored_cpu:.1f} | 链路: +{restored_bw:.1f}")

        # 清理状态
        self.env.current_tree = {
            'hvt': np.zeros((self.env.n, self.env.K_vnf), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set()
        }
        self.env.nodes_on_tree = set()

    # =========================================================================
    # 🔗 连接与图构建类 (Connection & Graph)
    # =========================================================================

    def connect_destination(self, dest_node):
        """连接目的地 - 增加 VNF 完整性检查"""
        if self.env.current_request is None:
            return False

        dests = self.env.current_request.get('dest', [])
        if dest_node not in dests:
            print(f"⚠️ 节点 {dest_node} 不是有效的目的地")
            return False

        required_vnfs = self.env.current_request.get('vnf', [])
        if required_vnfs:
            placement = self.env.current_tree.get('placement', {})
            deployed_vnf_types = set()
            for key, info in placement.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    deployed_vnf_types.add(key[1])

            required_vnf_set = set(required_vnfs)
            if not required_vnf_set.issubset(deployed_vnf_types):
                print(f"❌ [连接阻断] VNF 未完整部署. 缺: {list(required_vnf_set - deployed_vnf_types)}")
                return False

        self.env.current_tree.setdefault('connected_dests', set()).add(dest_node)
        print(f"✅ [连接成功] 目的地 {dest_node} 已连接")
        return True

    def build_graph_structures(self):
        """构建 GNN 所需的 edge_index 和 edge_attr"""
        import torch
        adj = self.env.topology_mgr.topo
        edge_indices = np.where(adj > 0)
        self.env.edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)

        num_edges = self.env.edge_index.shape[1]
        self.env.edge_attr = torch.zeros((num_edges, 5), dtype=torch.float32)
        weights = adj[edge_indices].astype(np.float32)
        self.env.edge_attr[:, 0] = torch.from_numpy(weights) / 100.0

        if hasattr(self.env, 'device'):
            self.env.edge_index = self.env.edge_index.to(self.env.device)
            self.env.edge_attr = self.env.edge_attr.to(self.env.device)