# envs/modules/TimeSlotManager.py
"""
TimeSlotManager - 独立时间槽管理器
=====================================
职责：
  - 维护仿真时钟（time_step / current_time_slot）
  - 按时间槽组织请求队列（slot_queue / requests_by_slot）
  - 检测跨槽事件并触发资源到期释放
  - 提供 get_next_request() 统一入口，屏蔽 online/offline 差异

调用方（SFC_HIRL_Env）只需：
  1. __init__  → self.time_slot_mgr = TimeSlotManager(env, config)
  2. load      → self.time_slot_mgr.load(requests)
  3. reset     → self.time_slot_mgr.reset()
  4. 获取请求  → req = self.time_slot_mgr.get_next_request()
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TimeSlotManager:
    """
    独立时间槽管理器

    Parameters
    ----------
    env    : SFC_HIRL_Env 实例（用于读取 resource_mgr、online_mode 等）
    config : 环境配置字典
    """

    def __init__(self, env, config: dict):
        self.env = env
        self.config = config

        # ── 时间配置 ──────────────────────────────────────────────────────────
        self.delta_t: float = config.get('data_generation', {}).get('time_slot_delta', 0.01)

        # ── 运行状态 ──────────────────────────────────────────────────────────
        self.time_step: float = 0.0          # 当前仿真时间
        self.current_time_slot: int = 0      # 当前时间槽编号

        # online 模式专用
        self.slot_queue: List[dict] = []     # 当前槽待处理请求队列
        self.current_slot_index: int = 0     # 槽扫描指针
        self.max_slot_index: int = 0         # 最大槽编号
        self.simulation_done: bool = False

        # 槽 → 请求列表映射（online 模式）
        self.requests_by_slot: Dict[int, List[dict]] = {}

        # offline 模式专用
        self.all_requests: List[dict] = []
        self.global_request_index: int = 0

        logger.info(f"[TimeSlotMgr] 初始化完成 (delta_t={self.delta_t})")

    # =========================================================================
    # 数据加载
    # =========================================================================

    def load(self, requests: List[dict]) -> None:
        """
        加载请求列表，同时构建时间槽索引。
        调用时机：env.load_requests() 末尾调用本方法。
        """
        if not requests:
            logger.warning("[TimeSlotMgr] 传入空请求列表，跳过加载")
            return

        self.all_requests = requests

        # 构建 slot → [req, ...] 映射
        requests_by_slot: Dict[int, List[dict]] = {}
        for req in requests:
            arr = float(req.get('arrival_time', 0.0))
            slot = req.get('time_slot', int(arr / max(self.delta_t, 1e-9)))
            requests_by_slot.setdefault(slot, []).append(req)

        self.requests_by_slot = requests_by_slot
        self.max_slot_index = max(requests_by_slot.keys()) if requests_by_slot else 0

        logger.info(
            f"[TimeSlotMgr] 加载完成: {len(requests)} 条请求, "
            f"{len(requests_by_slot)} 个时间槽, "
            f"max_slot={self.max_slot_index}"
        )

    # =========================================================================
    # 重置
    # =========================================================================

    def reset(self) -> None:
        """
        每个 Episode 开始时调用。

        ⚠️ 时钟（time_step）不归零——时间在整个仿真生命周期内单调递增。
           归零会导致 lifecycle 里 expire_time 高的请求永远无法到期释放，
           资源持续积累无法回收。
        ⚠️ 请求指针也不重置——跨 Episode 持续线性推进，到头后由
           _get_next_online 负责循环时叠加时间偏移。
        """
        # 只重置 simulation_done 标志，时钟和请求指针均保持不变
        self.simulation_done = False
        logger.debug(f"[TimeSlotMgr] Episode重置（时钟保持 t={self.time_step:.3f}，指针保持）")

    # =========================================================================
    # 统一请求获取入口
    # =========================================================================

    def get_next_request(self) -> Optional[dict]:
        """
        获取下一个请求（统一入口）。
        - online_mode=True  → 基于时间槽队列
        - online_mode=False → 基于全局线性索引
        返回标准化的 dict，或 None（无更多请求）。
        """
        online = self.env.online_mode
        if online:
            return self._get_next_online()
        else:
            return self._get_next_offline()

    # =========================================================================
    # Online 模式（时间槽队列驱动，跨 Episode 线性推进）
    # =========================================================================

    def _get_next_online(self) -> Optional[dict]:
        """
        按 global_request_index 线性取请求（与 offline 一致）。
        检测时间槽切换，切换时触发资源到期释放。
        到达末尾时标记 simulation_done，并从头循环（训练模式）。
        """
        if not self.all_requests:
            self.simulation_done = True
            return None

        if self.global_request_index >= len(self.all_requests):
            # 一轮仿真结束，重置指针循环（训练时数据集有限，需要复用）
            # 🔥 关键：叠加时间偏移，使时钟单调递增
            #    这样 lifecycle 里的 expire_time 在下一轮才能被正确触发释放
            if not hasattr(self, '_time_offset'):
                self._time_offset = 0.0
            # 计算本轮数据集的时间跨度
            if self.all_requests:
                max_arrive = max(float(r.get('arrival_time', 0.0)) for r in self.all_requests)
                max_expire = max(
                    float(r.get('arrival_time', 0.0)) + float(r.get('lifetime', 0.0))
                    for r in self.all_requests
                )
                cycle_len = max(max_arrive, max_expire) * 1.05  # 留 5% 余量确保全部过期
            else:
                cycle_len = 600.0
            self._time_offset += cycle_len
            logger.info(f"[TimeSlotMgr] 请求池耗尽，循环复用 "
                        f"(共 {len(self.all_requests)} 条, 时间偏移+={cycle_len:.1f}s, "
                        f"累计偏移={self._time_offset:.1f}s)")
            self.global_request_index = 0
            self.current_slot_index = 0
            self.simulation_done = False

        req_raw = self.all_requests[self.global_request_index]
        self.global_request_index += 1

        req = self._normalize(req_raw)

        # 🔥 叠加时间偏移（数据集循环时保持时钟单调递增）
        offset = getattr(self, '_time_offset', 0.0)
        raw_arrival = float(req.get('arrival_time', self.time_step))
        new_arrival = raw_arrival + offset
        new_slot = int(req.get('time_slot',
                               int(raw_arrival / max(self.delta_t, 1e-9))))

        # 如果有偏移，把调整后的 arrival_time 写回 req（不修改原始数据）
        if offset > 0:
            req = dict(req)  # shallow copy，避免污染原始数据
            req['arrival_time'] = new_arrival
            # expire_time 同样偏移
            if 'lifetime' in req:
                req['expire_time'] = new_arrival + float(req['lifetime'])

        self._update_time(new_arrival, new_slot)
        return req

    def _advance_to_next_active_slot(self) -> None:
        """（保留兼容接口，online 模式已改为线性推进，此方法不再被调用）"""
        pass

    # =========================================================================
    # Offline 模式（线性顺序 + 时间槽变化检测）
    # =========================================================================

    def _get_next_offline(self) -> Optional[dict]:
        """
        按 global_request_index 顺序取请求。
        检测时间槽切换，切换时触发资源到期释放。
        """
        if not self.all_requests:
            return None

        if self.global_request_index >= len(self.all_requests):
            # 循环复用（与原 reset_request 行为一致）
            self.global_request_index = 0

        req_raw = self.all_requests[self.global_request_index]
        self.global_request_index += 1

        req = self._normalize(req_raw)

        new_arrival = float(req.get('arrival_time', self.time_step))
        new_slot = int(req.get('time_slot',
                                int(new_arrival / max(self.delta_t, 1e-9))))

        self._update_time(new_arrival, new_slot)
        return req

    # =========================================================================
    # 内部工具
    # =========================================================================

    def _update_time(self, new_arrival: float, new_slot: int) -> None:
        """更新仿真时钟，时间前进时触发过期检查。"""
        old_slot = self.current_time_slot

        if new_slot != old_slot:
            logger.debug(
                f"[TimeSlotMgr] 时间槽切换 {old_slot} → {new_slot} | "
                f"time={new_arrival:.3f}s"
            )
            self.current_time_slot = new_slot

        self.time_step = new_arrival

        # 同步到 env
        self.env.time_step = self.time_step
        self.env.current_time_slot = self.current_time_slot

        # 每次时间前进都触发过期检查（不限于槽切换）
        self._trigger_expiry_check()

    def _trigger_expiry_check(self) -> None:
        """通知资源管理器检查并释放到期请求。"""
        rm = getattr(self.env, 'resource_mgr', None)
        if rm is None:
            logger.warning("[TimeSlotMgr] _trigger_expiry_check: resource_mgr 不存在!")
            return
        req_mgr = getattr(rm, 'request_manager', None)
        if req_mgr is None:
            logger.warning("[TimeSlotMgr] _trigger_expiry_check: request_manager 不存在!")
            return
        try:
            active_count = len(req_mgr.active_requests)
            expired = req_mgr.check_and_release_expired(self.time_step)
            if expired:
                logger.info(f"[TimeSlotMgr] t={self.time_step:.3f} 释放 {len(expired)} 个到期请求: {expired}")
            elif active_count > 0:
                # 有活跃请求但没有过期：打印时间和最近过期时间，每10次打一次
                if not hasattr(self, '_expiry_log_counter'):
                    self._expiry_log_counter = 0
                self._expiry_log_counter += 1
                if self._expiry_log_counter % 50 == 0:
                    nearest = min(info['expire_time'] for info in req_mgr.active_requests.values())
                    logger.info(f"[TimeSlotMgr] t={self.time_step:.3f} | 活跃{active_count}个 | 最近过期={nearest:.3f}")
        except Exception as e:
            logger.error(f"[TimeSlotMgr] 到期释放失败: {e}")

    @staticmethod
    def _normalize(req_raw) -> dict:
        """将 Request 对象或 dict 统一转为 dict。"""
        if isinstance(req_raw, dict):
            return req_raw
        if hasattr(req_raw, 'to_dict'):
            return req_raw.to_dict()
        if hasattr(req_raw, '__dict__'):
            return req_raw.__dict__
        return req_raw

    # =========================================================================
    # 查询接口
    # =========================================================================

    @property
    def is_done(self) -> bool:
        """仿真是否结束（仅 online 模式有意义）。"""
        return self.simulation_done

    def get_stats(self) -> dict:
        return {
            'time_step': self.time_step,
            'current_time_slot': self.current_time_slot,
            'current_slot_index': self.current_slot_index,
            'max_slot_index': self.max_slot_index,
            'slot_queue_size': len(self.slot_queue),
            'global_request_index': self.global_request_index,
            'simulation_done': self.simulation_done,
        }