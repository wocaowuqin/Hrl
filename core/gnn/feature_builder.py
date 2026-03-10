"""
core/gnn/feature_builder.py
GNN 特征构建器 - 最终修复版 (Fix Batch Collision & Req Dim)
"""

import torch
import logging
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)

class GNNFeatureBuilder:
    def __init__(self, device):
        self.device = device

    def get_state_dim(self):
        return 32

    def _try_assemble_tuple(self, obj):
        """处理 (x, edge_index, edge_attr, req_vec) 元组。
        注意：此路径产生的 Data 对象无法携带 tree_edge_index/dest_mask（
        元组格式不含这些字段）。调用方应优先存储完整 Data 对象到 buffer，
        而非裸元组，以保证树流正常工作。
        """
        if isinstance(obj, (tuple, list)) and len(obj) == 4:
            t0, t1, t2, t3 = obj[0], obj[1], obj[2], obj[3]
            if (torch.is_tensor(t0) and torch.is_tensor(t1) and
                torch.is_tensor(t2) and torch.is_tensor(t3)):

                if t1.dim() == 2 and t1.shape[0] == 2:
                    req_v = t3.cpu()
                    if req_v.dim() == 1:
                        req_v = req_v.unsqueeze(0)

                    # tree_edge_index 缺失 → 下游 encoder 会退化到 self-loop
                    # 此处记录警告，方便排查（生产环境中 buffer 里不应存裸元组）
                    logger.warning(
                        "[_try_assemble_tuple] 从4元组组装Data，"
                        "tree_edge_index/dest_mask 将缺失，树流退化为self-loop。"
                        "请确保 buffer 存储完整 PyG Data 对象。"
                    )
                    return Data(
                        x=t0.cpu(),
                        edge_index=t1.cpu().long(),
                        edge_attr=t2.cpu(),
                        req_vec=req_v,
                        # 占位 None：下游 getattr(s, 'tree_edge_index', None) 返回 None
                        tree_edge_index=None,
                        dest_mask=None,
                    )
        return None

    def _extract_data(self, obj, depth=0):
        """提取并清洗 Data 对象 (关键修复：移除残留的 batch 属性)"""
        if depth > 3: return None

        # 检查是否是 Data 对象 (或类似结构)
        if hasattr(obj, 'x') and hasattr(obj, 'edge_index'):

            # 🔥 核心修复 1: 移除残留的 batch 属性
            # 防止在 Batch.from_data_list 时发生 Key 冲突
            if hasattr(obj, 'batch') and obj.batch is not None:
                del obj.batch

            # 处理 Batch 对象封装的情况
            if hasattr(obj, 'to_data_list'):
                try:
                    data = obj.to_data_list()[0]
                    # 再次清洗提取出的对象
                    if hasattr(data, 'batch'): del data.batch

                    if hasattr(data, 'req_vec') and data.req_vec.dim() == 1:
                        data.req_vec = data.req_vec.unsqueeze(0)
                    return data
                except:
                    pass

            # ✅ 核心修复 2: 维度修正
            if hasattr(obj, 'req_vec') and obj.req_vec.dim() == 1:
                obj.req_vec = obj.req_vec.unsqueeze(0)

            return obj

        # 尝试递归查找
        assembled = self._try_assemble_tuple(obj)
        if assembled is not None: return assembled

        if isinstance(obj, (tuple, list)):
            for item in obj:
                res = self._extract_data(item, depth + 1)
                if res is not None: return res

        return None

    def collate_fn(self, transitions):
        batch = {
            'state': [], 'next_state': [], 'action': [],
            'reward': [], 'done': [], 'goal_emb': [],
            'req': [],   # ← 补充：req 字段供 agent_train._encode 重建 req_vec
        }

        for i, t in enumerate(transitions):
            # _extract_data 现在会自动清洗 'batch' 属性
            s = self._extract_data(t['state'])
            if s is None: raise ValueError(f"Transition {i} state invalid")
            batch['state'].append(s)

            ns = self._extract_data(t['next_state'])
            if ns is None: raise ValueError(f"Transition {i} next_state invalid")
            batch['next_state'].append(ns)

            batch['action'].append(t['action'])
            batch['reward'].append(t['reward'])
            batch['done'].append(t['done'])

            # req 字段：直接透传原始 dict（None 也保留，下游 _build_req_vec_from_transition 会处理）
            batch['req'].append(t.get('req', None))

            if 'goal_emb' in t:
                g = t['goal_emb']
                if isinstance(g, torch.Tensor) and g.dim() == 1:
                    g = g.unsqueeze(0)
                batch['goal_emb'].append(g)

        try:
            # PyG Batch 组装 (现在输入是干净的 Data 对象，不会报错了)
            batch['state'] = Batch.from_data_list(batch['state']).to(self.device)
            batch['next_state'] = Batch.from_data_list(batch['next_state']).to(self.device)

            batch['action'] = torch.tensor(batch['action'], dtype=torch.long, device=self.device)
            batch['reward'] = torch.tensor(batch['reward'], dtype=torch.float32, device=self.device).unsqueeze(1)
            batch['done'] = torch.tensor(batch['done'], dtype=torch.float32, device=self.device).unsqueeze(1)

            # req 保持为 list[dict|None]，agent_train 逐元素取用
            # batch['req'] 已经是 list，不需要额外转换

            raw_goals = batch.get('goal_emb', [])
            valid_goals = [g for g in raw_goals if g is not None]
            if len(valid_goals) > 0:
                if isinstance(valid_goals[0], torch.Tensor):
                    batch['goal_emb'] = torch.cat(valid_goals, dim=0).to(self.device)
                else:
                    batch['goal_emb'] = torch.tensor(valid_goals, dtype=torch.float32, device=self.device)
            else:
                batch['goal_emb'] = None

            return batch

        except Exception as e:
            logger.error(f"❌ Batch Assembly Failed: {e}")
            raise e

    def state_to_batch(self, states):
        if not isinstance(states, list): states = [states]
        clean_states = []
        for s in states:
            clean = self._extract_data(s)
            if clean: clean_states.append(clean)

        if not clean_states: raise ValueError("state_to_batch failed")

        if len(clean_states) == 1:
            data = clean_states[0]
            # 推断时，如果没有 batch 属性，需要加上
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            return data.to(self.device)
        else:
            return Batch.from_data_list(clean_states).to(self.device)