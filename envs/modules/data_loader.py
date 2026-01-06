import pickle
import os
import logging
import numpy as np
import copy
from typing import Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """
    数据加载器：最终修复版 V3
    1. ✅ 自动适配旧数据格式
    2. ✅ 支持 copy() 方法
    3. ✅ 支持字典式赋值 (req['key'] = val)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.requests = []
        self.events = []
        self.req_map = {}
        self.time_step = 0
        self.total_steps = 0

    def load_dataset(self, phase_or_req_file: str, events_file: Optional[str] = None) -> bool:
        # --- 路径查找逻辑 ---
        if events_file is not None:
            req_filename = phase_or_req_file
            possible_dirs = [
                self.cfg['path'].get('expert_data_dir', 'data/expert'),
                'generate_requests_depend_on_poisson/data_output',
                'data_output',
                'data/expert',
                self.cfg['path'].get('input_dir', 'data/input_dir'),
                '.'
            ]
            req_path, evt_path = None, None
            for search_dir in possible_dirs:
                if not search_dir: continue
                tr = os.path.join(search_dir, req_filename)
                te = os.path.join(search_dir, events_file)
                if os.path.exists(tr) and os.path.exists(te):
                    req_path, evt_path = tr, te
                    logger.info(f"Found data files in: {search_dir}")
                    break
            if not req_path:
                logger.error(f"Data files not found: {req_filename}, {events_file}")
                return False
            return self._load_from_paths(req_path, evt_path)
        else:
            phase = phase_or_req_file
            data_dir = self.cfg['path'].get('input_dir', 'data/input_dir')
            req_path = os.path.join(data_dir, f"{phase}_requests.pkl")
            evt_path = os.path.join(data_dir, f"{phase}_events.pkl")

            if not os.path.exists(req_path) or not os.path.exists(evt_path):
                alt_dir = self.cfg['path'].get('expert_data_dir', 'data/expert')
                req_path = os.path.join(alt_dir, f"{phase}_requests.pkl")
                evt_path = os.path.join(alt_dir, f"{phase}_events.pkl")
                if not os.path.exists(req_path) or not os.path.exists(evt_path):
                    logger.error(f"Data files not found for phase '{phase}'")
                    return False
            return self._load_from_paths(req_path, evt_path)

    def _load_from_paths(self, req_path, evt_path):
        try:
            logger.info(f"Loading requests from: {req_path}")
            with open(req_path, 'rb') as f:
                raw_requests = pickle.load(f)

            # 🔥 应用适配器
            self.requests = [self._adapt_legacy_format(r) for r in raw_requests]

            self.req_map = {}
            for r in self.requests:
                rid = int(getattr(r, 'id', r.get('id'))) if hasattr(r, 'id') or isinstance(r, dict) else -1
                if rid == -1: rid = len(self.req_map) + 1

                # 统一更新 ID
                if isinstance(r, dict): r['id'] = rid
                else: r.id = rid

                self.req_map[rid] = r

            logger.info(f"Loading events from: {evt_path}")
            with open(evt_path, 'rb') as f:
                raw_events = pickle.load(f)

            self.events = []
            all_event_ids = set()

            for i, evt in enumerate(raw_events):
                arr, lv = [], []
                if isinstance(evt, dict):
                    arr = evt.get('arrive_event', evt.get('arrive', evt.get('arrived', [])))
                    lv = evt.get('leave_event', evt.get('leave', evt.get('left', [])))
                elif isinstance(evt, (list, tuple, np.ndarray)):
                    if len(evt) >= 1: arr = evt[0]
                    if len(evt) >= 2: lv = evt[1]

                arr = np.array(arr, dtype=int).flatten().tolist()
                lv = np.array(lv, dtype=int).flatten().tolist()
                self.events.append({'arrive': arr, 'leave': lv})
                all_event_ids.update(arr)

            self.total_steps = len(self.events)
            self.reset()

            logger.info(f"✅ Dataset Loaded: {len(self.requests)} requests, {self.total_steps} time steps")

            if len(self.requests) > 0:
                sample = self.requests[0]
                chain = getattr(sample, 'sfc_chain', [])
                logger.info(f"🔍 Sample Req[0]: ID={getattr(sample, 'id', '?')}, Len={len(chain)}, Chain={chain}")

            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _adapt_legacy_format(self, item):
        # 1. 转为字典处理
        if not isinstance(item, dict) and hasattr(item, '__dict__'):
            data = item.__dict__.copy()
        elif isinstance(item, dict):
            data = item.copy()
        else:
            return item

        # 2. 映射 VNF 链
        if 'sfc_chain' not in data:
            for key in ['vnf', 'chain', 'SFC', 'VNF_chain']:
                if key in data:
                    data['sfc_chain'] = data[key]
                    break
            if 'sfc_chain' not in data: data['sfc_chain'] = []

        # 3. 映射 CPU
        if 'vnf_cpu_demands' not in data:
            for key in ['cpu_origin', 'cpu', 'CPU', 'vnf_cpu']:
                if key in data:
                    data['vnf_cpu_demands'] = data[key]
                    break
            if 'vnf_cpu_demands' not in data:
                data['vnf_cpu_demands'] = [10.0] * len(data['sfc_chain'])

        # 4. 映射带宽
        if 'bandwidth_demand' not in data:
            for key in ['bw_origin', 'bw', 'BW']:
                if key in data:
                    data['bandwidth_demand'] = data[key]
                    break
            if 'bandwidth_demand' not in data: data['bandwidth_demand'] = 10.0

        # 5. 封装成全能对象
        class SFCRequest:
            def __init__(self, **entries):
                self.__dict__.update(entries)

            def __repr__(self):
                return f"Req(id={self.id}, len={len(self.sfc_chain)})"

            def get(self, key, default=None):
                return self.__dict__.get(key, default)

            def __getitem__(self, key):
                return self.__dict__[key]

            # 🔥 新增 setitem，支持 req['dest'] = ...
            def __setitem__(self, key, value):
                self.__dict__[key] = value

            def copy(self):
                return SFCRequest(**self.__dict__)

        return SFCRequest(**data)

    def reset(self):
        self.time_step = 0

    def get_current_arrivals(self) -> list:
        if self.time_step >= self.total_steps: return []
        arrive_ids = self.events[self.time_step]['arrive']
        return [self.req_map[rid] for rid in arrive_ids if rid in self.req_map]

    def get_current_leaves(self) -> list:
        if self.time_step >= self.total_steps: return []
        return self.events[self.time_step]['leave']

    def advance_time(self):
        self.time_step += 1

    def is_done(self):
        return self.time_step >= self.total_steps