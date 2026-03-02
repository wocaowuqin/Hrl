# envs/tools.py
"""
SFCToolkit - SFC 环境辅助工具箱
=====================================
职责（精简后）：
  - 验证与检查：SFC 路径完整性、节点部署合法性
  - 状态查询：资源占用率、部署进度
  - 资源提交：两阶段 commit
  - 资源回滚：统一回滚（当前请求 or 指定请求）
  - 连接与图：目的地连接、GNN 图结构构建

已迁出：
  - 时间槽推进 (advance_to_next_active_slot)     → TimeSlotManager
  - 在线模式请求获取 (get_next_request_online)   → TimeSlotManager
"""

import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SFCToolkit:
    """SFC 环境辅助工具箱"""

    def __init__(self, env):
        """
        Parameters
        ----------
        env : SFC_HIRL_Env 实例
        """
        self.env = env

    # =========================================================================
    # 🔍 验证与检查
    # =========================================================================

    def validate_sfc_paths(self, parent_map: dict) -> tuple[bool, list]:
        """
        验证 SFC 路径完整性。

        对当前请求的每个目的地，沿 parent_map 回溯到源节点，
        检查路径上 VNF 是否齐全且顺序正确。

        Parameters
        ----------
        parent_map : {node: parent_node} 路径回溯表

        Returns
        -------
        (success, errors)
        """
        if not self.env.current_request:
            return False, ["No request"]

        req = self.env.current_request
        source = req['source']
        dests = req.get('dest', [])
        required_vnfs = req.get('vnf', [])

        if not required_vnfs:
            return True, []

        # 构建 node → [vnf_type, ...] 映射
        node_vnf_dict: dict[int, list] = {}
        for key, _ in self.env.current_tree.get('placement', {}).items():
            if isinstance(key, tuple) and len(key) >= 2:
                node_vnf_dict.setdefault(key[0], []).append(key[1])

        logger.debug("[SFC验证] 开始验证路径...")
        errors = []

        for dest in dests:
            # 1. 回溯路径
            path, curr = [], dest
            while curr is not None:
                path.append(curr)
                if curr == source:
                    break
                curr = parent_map.get(curr)

            # 2. 路径完整性
            if not path or path[-1] != source:
                err = f"Dest {dest}: Path broken (无法从源节点到达)"
                errors.append(err)
                print(f"   ❌ {err}")
                continue

            path.reverse()  # source → dest

            # 3. 收集路径上的 VNF 类型（保留顺序）
            collected = []
            for node in path:
                collected.extend(node_vnf_dict.get(node, []))

            # 4. 检查 VNF 完整性
            missing = [v for v in required_vnfs if v not in collected]
            if missing:
                err = f"Dest {dest}: 缺少 VNF {missing}"
                errors.append(err)
                print(f"   ❌ {err}")
                continue

            # 5. 检查 VNF 顺序
            indices = [required_vnfs.index(v) for v in collected if v in required_vnfs]
            if indices != sorted(indices):
                err = f"Dest {dest}: VNF 顺序错误"
                errors.append(err)
                print(f"   ❌ {err}")

        success = len(errors) == 0
        if success:
            print("✅ [SFC 验证] 所有路径验证通过")
        else:
            print(f"❌ [SFC 验证] 发现 {len(errors)} 个错误")

        return success, errors

    def check_deployment_validity(self, node_id: int) -> bool:
        """
        检查节点是否可以部署 VNF。

        规则：① 非源节点  ② 必须是 DC 节点  ③ 资源充足
        """
        if not self.env.current_request:
            return False

        if node_id == self.env.current_request.get('source'):
            return False

        if hasattr(self.env, 'dc_nodes') and node_id not in self.env.dc_nodes:
            return False

        if hasattr(self.env, '_check_node_resources'):
            if not self.env._check_node_resources(node_id):
                return False

        return True

    # =========================================================================
    # 📊 状态查询
    # =========================================================================

    def get_resource_utilization(self) -> float:
        """返回全网 CPU 资源占用率 [0, 1]，出错时返回 0.0。"""
        try:
            rm = self.env.resource_mgr
            nodes = getattr(rm, 'nodes', None)
            if nodes is None:
                return 0.0

            total_cap = used_cap = 0.0

            if isinstance(nodes, list):
                for n in nodes:
                    cap = n.get('capacity', n.get('cpu_limit', 100.0))
                    rem = n.get('cpu', 100.0)
                    total_cap += cap
                    used_cap += cap - rem

            elif isinstance(nodes, dict):
                for n in nodes.values():
                    if not isinstance(n, dict):
                        continue
                    cap = n.get('total', 100.0)
                    used_cap += n.get('used', 0.0)
                    total_cap += cap

            return used_cap / total_cap if total_cap > 0 else 0.0

        except Exception:
            return 0.0

    def get_current_progress(self) -> float:
        """返回当前 SFC 的 VNF 部署进度 [0, 1]。"""
        if not self.env.current_request:
            return 0.0

        vnf_list = self.env.current_request.get('vnf', [])
        if not vnf_list:
            return 1.0

        curr_idx = getattr(self.env, 'next_vnf_idx', getattr(self.env, 'current_vnf_idx', 0))
        return float(curr_idx) / len(vnf_list)

    # =========================================================================
    # 💳 资源提交与回滚
    # =========================================================================

    def commit_resources(self, pruned_tree: dict, valid_nodes: set) -> bool:
        """
        两阶段资源提交。

        Phase 1 — 收集待分配的链路和节点资源；
        Phase 2 — 逐一调用 resource_mgr 分配，记录成功项。

        Parameters
        ----------
        pruned_tree : {(u, v): ...} 剪枝后的树边集合
        valid_nodes : 允许提交节点资源的节点集合

        Returns
        -------
        True（始终返回，分配失败只打印警告）
        """
        req = self.env.current_request
        bw_req = req.get('bw_origin', 1.0)

        # Phase 1: 收集
        pending_links = [(u, v, bw_req) for (u, v) in pruned_tree.keys()]
        pending_nodes = []
        for key, info in self.env.current_tree.get('placement', {}).items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v_type = key[0], key[1]
                if n in valid_nodes:
                    pending_nodes.append((n, v_type,
                                          info.get('cpu_used', 1.0),
                                          info.get('mem_used', 1.0)))

        print(f"\n💳 [开始扣费] 节点={len(pending_nodes)}, 链路={len(pending_links)}")

        # Phase 2: 分配
        self.env.curr_ep_link_allocs = []
        self.env.curr_ep_node_allocs = []
        total_cpu = total_mem = total_bw = 0.0

        for u, v, bw in pending_links:
            if self.env.resource_mgr.allocate_link_resource(u, v, bw) is not False:
                self.env.curr_ep_link_allocs.append((u, v, bw))
                total_bw += bw
            else:
                print(f"   ❌ 链路({u},{v}) 分配失败")

        for n, v_type, c, m in pending_nodes:
            if self.env.resource_mgr.allocate_node_resource(n, v_type, c, m) is not False:
                self.env.curr_ep_node_allocs.append((n, v_type, c, m))
                total_cpu += c
                total_mem += m
            else:
                print(f"   ❌ 节点{n}[VNF{v_type}] 分配失败")

        print(f"💳 [扣费汇总] CPU:{total_cpu:.1f} | Mem:{total_mem:.1f} | BW:{total_bw:.1f}")

        # 通知生命周期管理器更新资源清单（供到期自动释放）
        if hasattr(self.env, '_update_lifecycle_resources') and self.env.current_request:
            self.env._update_lifecycle_resources(self.env.current_request)

        return True

    def rollback_resources(self, req: Optional[dict] = None) -> None:
        """
        统一回滚资源并清理树状态。

        Parameters
        ----------
        req : 指定请求（可选）。若为 None，则从 env.current_request 读取带宽。
              无论哪种情况，都从 env.current_tree 读取 placement / tree。
        """
        if not hasattr(self.env, 'current_tree'):
            return

        target_req = req or self.env.current_request
        if target_req is None and req is None:
            return

        bw = target_req.get('bw_origin', 1.0) if target_req else 1.0

        if req is not None:
            print(f"♻️ [回滚执行] 开始释放请求 {req.get('id')} 的资源...")

        placement = self.env.current_tree.get('placement', {})
        tree_edges = self.env.current_tree.get('tree', {})
        rm = self.env.resource_mgr
        restored_cpu = restored_bw = 0.0

        # 释放节点资源
        for key, info in placement.items():
            if not isinstance(key, tuple):
                continue

            if req is not None:
                # rollback_request_resources 原有的 info 格式（扁平 dict）
                if not isinstance(info, dict):
                    continue
                node = info.get('node')
                v_type = info.get('vnf_type', 0)
                c = info.get('cpu_used', 0.0)
                m = info.get('mem_used', 0.0)
            else:
                # rollback_resources 原有的 key=(node, vnf_type) 格式
                node, v_type = key[0], key[1]
                c = info.get('cpu_used', 1.0) if isinstance(info, dict) else 1.0
                m = info.get('mem_used', 1.0) if isinstance(info, dict) else 1.0

            if hasattr(rm, 'release_node_resource'):
                try:
                    rm.release_node_resource(node, v_type, c, m)
                    restored_cpu += c
                except Exception as e:
                    logger.warning(f"[回滚] 节点{node} 释放失败: {e}")

        # 释放链路资源
        for edge_key in tree_edges.keys():
            u, v = (edge_key if isinstance(edge_key, tuple) else (None, None))
            if u is None:
                continue
            if hasattr(rm, 'release_link_resource'):
                try:
                    rm.release_link_resource(u, v, bw)
                    restored_bw += bw
                except Exception as e:
                    logger.warning(f"[回滚] 链路({u},{v}) 释放失败: {e}")

        if restored_cpu > 0 or restored_bw > 0:
            print(f"♻️ [资源回滚] 节点: +{restored_cpu:.1f} CPU | 链路: +{restored_bw:.1f} BW")

        if req is not None:
            print(f"✅ [回滚完成] 节点: +{restored_cpu:.1f} | 链路: +{restored_bw:.1f}")

        # 从生命周期管理器强制移除，避免残留
        target = req if req is not None else self.env.current_request
        if target is not None:
            req_id = target.get('id')
            rm = getattr(self.env, 'resource_mgr', None)
            if rm is not None and hasattr(rm, 'request_manager') and req_id is not None:
                rm.request_manager.force_release(str(req_id), getattr(self.env, 'time_step', 0.0))

        # 清理树状态
        self._reset_tree_state()

    def rollback_request_resources(self, req: dict) -> None:
        """强制回滚指定请求（保留原接口，内部委托给 rollback_resources）。"""
        if not req:
            return
        self.rollback_resources(req=req)

    # =========================================================================
    # 🔗 连接与图构建
    # =========================================================================

    def connect_destination(self, dest_node: int) -> bool:
        """
        将目的地节点标记为已连接。

        前置检查：① dest_node 是合法目的地  ② 路径上 VNF 已全部部署
        """
        if self.env.current_request is None:
            return False

        dests = self.env.current_request.get('dest', [])
        if dest_node not in dests:
            print(f"⚠️ 节点 {dest_node} 不是有效的目的地")
            return False

        required_vnfs = set(self.env.current_request.get('vnf', []))
        if required_vnfs:
            deployed = {
                key[1]
                for key in self.env.current_tree.get('placement', {}).keys()
                if isinstance(key, tuple) and len(key) >= 2
            }
            missing = required_vnfs - deployed
            if missing:
                print(f"❌ [连接阻断] VNF 未完整部署. 缺: {list(missing)}")
                return False

        self.env.current_tree.setdefault('connected_dests', set()).add(dest_node)
        print(f"✅ [连接成功] 目的地 {dest_node} 已连接")
        return True

    def build_graph_structures(self) -> None:
        """构建 GNN 所需的 edge_index 和 edge_attr，并同步到 env。"""
        adj = self.env.topology_mgr.topo
        edge_indices = np.where(adj > 0)

        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
        num_edges = edge_index.shape[1]

        edge_attr = torch.zeros((num_edges, 5), dtype=torch.float32)
        weights = adj[edge_indices].astype(np.float32)
        edge_attr[:, 0] = torch.from_numpy(weights) / 100.0

        device = getattr(self.env, 'device', None)
        if device is not None:
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)

        self.env.edge_index = edge_index
        self.env.edge_attr = edge_attr

    # =========================================================================
    # 🔒 私有辅助
    # =========================================================================

    def _reset_tree_state(self) -> None:
        """将 env.current_tree 和相关状态变量恢复到空白状态。"""
        n = self.env.n
        k = getattr(self.env, 'K_vnf', 10)
        self.env.current_tree = {
            'hvt': np.zeros((n, k), dtype=np.float32),
            'tree': {},
            'placement': {},
            'connected_dests': set(),
        }
        self.env.nodes_on_tree = set()
        if hasattr(self.env, '_node_visit_count'):
            self.env._node_visit_count = {}