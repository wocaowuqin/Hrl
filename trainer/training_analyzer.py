# core/trainer/training_analyzer.py
# -*- coding: utf-8 -*-
"""
TrainingAnalyzer - 训练失败分析器

用法：
    1. 在 phase3_rl_trainer.py 中引入：
       from core.trainer.training_analyzer import TrainingAnalyzer

    2. 在 __init__ 里初始化：
       self.analyzer = TrainingAnalyzer(output_dir=self.output_dir)

    3. 在 run() 的每个 Episode 之后调用：
       self.analyzer.record(episode, info, res_util, env=self.env)

    4. 训练结束后调用：
       self.analyzer.report()

输出文件（保存在 output_dir/analysis/）：
    - failure_analysis.txt   文字摘要报告
    - episode_log.csv        每个 Episode 的详细记录
    - charts/                各类分析图表（PNG）
"""

import os
import csv
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  失败原因枚举
# ═══════════════════════════════════════════════════════════════
class FailReason:
    MASK_ZERO       = "Mask=0 (No Resource)"    # 高层动作全被掩码
    PATH_BLOCKED    = "Path Blocked"            # BW不足无法规划路径
    TIMEOUT         = "Timeout"                  # 步数耗尽
    VNF_FAIL        = "VNF Deploy Fail"          # 节点资源不足
    NO_PROGRESS     = "No Progress"              # 连续N个cycle无进展被终止
    BW_EXHAUSTED    = "BW Exhausted"             # 带宽耗尽无法走新边
    TRAPPED         = "Trapped"                  # 困死（邻居全无BW）
    UNKNOWN         = "Unknown"


# ═══════════════════════════════════════════════════════════════
#  每个 Episode 的记录结构
# ═══════════════════════════════════════════════════════════════
class EpisodeRecord:
    __slots__ = [
        'episode', 'success', 'fail_reason',
        'vnf_done', 'vnf_total',
        'dest_done', 'dest_total',
        'steps', 'reward', 'res_util',
        'cpu_remain', 'mem_remain', 'bw_remain',
        'unreachable_count', 'subgoals_ok', 'subgoals_fail',
        'request_id', 'arrival_time',
        'epsilon_high', 'epsilon_low', 'steps_done',
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


# ═══════════════════════════════════════════════════════════════
#  主分析器
# ═══════════════════════════════════════════════════════════════
class TrainingAnalyzer:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)

        self.records: list[EpisodeRecord] = []

        # CSV 文件句柄（流式写入，不等到最后）
        self._csv_path = self.output_dir / "episode_log.csv"
        self._csv_file = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'episode', 'success', 'fail_reason',
            'vnf_done', 'vnf_total', 'dest_done', 'dest_total',
            'steps', 'reward', 'res_util',
            'cpu_remain', 'mem_remain', 'bw_remain',
            'unreachable_count', 'subgoals_ok', 'subgoals_fail',
            'request_id', 'arrival_time',
            'epsilon_high', 'epsilon_low', 'steps_done',
        ])

        logger.info(f"📊 TrainingAnalyzer 初始化完成，输出目录: {self.output_dir}")

    # ──────────────────────────────────────────────────────────
    #  每个 Episode 结束后调用
    # ──────────────────────────────────────────────────────────
    def record(self, episode: int, info: dict, res_util: float, env=None, coordinator=None, agent=None):
        """
        记录一个 Episode 的结果。

        参数:
            episode    : Episode 编号
            info       : coordinator.run_episode() 返回的 info dict
            res_util   : 资源利用率
            env        : SFC_HIRL_Env 实例（用于读取剩余资源）
            coordinator: HRL_Coordinator 实例（用于读取 unreachable_nodes）
        """
        rec = EpisodeRecord()
        rec.episode    = episode
        rec.success    = bool(info.get('success', False))
        rec.steps      = info.get('steps', 0)
        rec.reward     = info.get('reward', 0.0)
        rec.res_util   = res_util
        rec.subgoals_ok   = info.get('subgoals_ok', 0)
        rec.subgoals_fail = info.get('subgoals_fail', 0)
        rec.unreachable_count = len(getattr(coordinator, 'unreachable_nodes', set()))

        # ── 失败原因判定（先读进度，再分类）──────────────────
        if rec.success:
            rec.fail_reason = None
        else:
            error  = info.get('error', '') or ''
            reason = info.get('reason', '') or ''   # HRL_Coordinator 设置的 reason 字段
            comp   = info.get('completion_status', '') or ''

            # 临时读取 VNF / Dest 进度（用于精准分类）
            vnf_done_tmp  = 0
            vnf_total_tmp = 3
            dest_done_tmp = 0
            if env is not None and hasattr(env, 'current_request') and env.current_request:
                req_tmp = env.current_request
                vnf_total_tmp = len(req_tmp.get('vnf', [3]))
                if hasattr(env, 'next_vnf_idx'):
                    vnf_done_tmp = min(env.next_vnf_idx, vnf_total_tmp)
                if hasattr(env, 'current_tree') and env.current_tree:
                    dest_done_tmp = len(env.current_tree.get('connected_dests', set()))

            all_vnf_done = (vnf_done_tmp >= vnf_total_tmp)
            no_dest_done = (dest_done_tmp == 0)

            # 优先读 reason 字段（HRL_Coordinator 显式设置）
            if reason == 'no_progress':
                rec.fail_reason = FailReason.NO_PROGRESS
            elif reason == 'consecutive_timeout':
                rec.fail_reason = FailReason.TIMEOUT
            elif reason in ('bandwidth_exhausted', 'no_bandwidth'):
                rec.fail_reason = FailReason.BW_EXHAUSTED
            elif reason == 'trapped' or error == 'trapped':
                rec.fail_reason = FailReason.TRAPPED
            # 再按状态推断
            elif all_vnf_done and no_dest_done:
                rec.fail_reason = FailReason.PATH_BLOCKED
            elif error == 'no_high_actions' or 'Mask全0' in comp or res_util >= 0.95:
                rec.fail_reason = FailReason.MASK_ZERO
            elif (rec.unreachable_count or 0) > 0 and not all_vnf_done:
                rec.fail_reason = FailReason.PATH_BLOCKED
            elif rec.steps >= info.get('max_steps', 100) - 1:
                rec.fail_reason = FailReason.TIMEOUT
            elif not all_vnf_done:
                rec.fail_reason = FailReason.VNF_FAIL
            else:
                rec.fail_reason = FailReason.UNKNOWN

        # ── VNF / Dest 进度 ───────────────────────────────────
        if env is not None and hasattr(env, 'current_request') and env.current_request:
            req = env.current_request
            rec.vnf_total  = len(req.get('vnf', []))
            rec.dest_total = len(req.get('dest', []))
            rec.request_id    = req.get('id')
            rec.arrival_time  = req.get('arrival_time')

            # VNF 完成数
            if hasattr(env, 'next_vnf_idx'):
                rec.vnf_done = min(env.next_vnf_idx, rec.vnf_total)
            else:
                rec.vnf_done = rec.vnf_total if info.get('vnf_success') else 0

            # Dest 完成数
            if hasattr(env, 'current_tree') and env.current_tree:
                rec.dest_done = len(env.current_tree.get('connected_dests', set()))
            else:
                rec.dest_done = rec.dest_total if info.get('dest_success') else 0

        # ── 剩余资源快照 ──────────────────────────────────────
        if env is not None and hasattr(env, 'resource_mgr') and env.resource_mgr:
            try:
                rm = env.resource_mgr
                if hasattr(rm, 'pool'):
                    pool = rm.pool
                    if hasattr(pool, 'cpu_avail'):
                        rec.cpu_remain = float(np.sum(list(pool.cpu_avail) if isinstance(pool.cpu_avail, dict)
                                                       else pool.cpu_avail))
                    if hasattr(pool, 'mem_avail'):
                        rec.mem_remain = float(np.sum(list(pool.mem_avail) if isinstance(pool.mem_avail, dict)
                                                       else pool.mem_avail))
                    if hasattr(pool, 'bw_avail'):
                        bw = pool.bw_avail
                        rec.bw_remain = float(sum(bw.values()) if isinstance(bw, dict) else np.sum(bw))
            except Exception as e:
                logger.debug(f"资源读取失败: {e}")

        # ── epsilon & steps 快照（优先直接传入的 agent）────────
        _agent = agent
        if _agent is None and coordinator is not None:
            _agent = getattr(coordinator, 'high_agent', None) or getattr(coordinator, 'agent', None)
        if _agent is not None:
            rec.epsilon_high = getattr(_agent, 'epsilon_high', None)
            rec.epsilon_low  = getattr(_agent, 'epsilon_low',  None)
            rec.steps_done   = getattr(_agent, 'steps_done',   None)

        self.records.append(rec)

        # 流式写 CSV
        self._csv_writer.writerow([
            rec.episode, rec.success, rec.fail_reason,
            rec.vnf_done, rec.vnf_total, rec.dest_done, rec.dest_total,
            rec.steps, rec.reward, rec.res_util,
            rec.cpu_remain, rec.mem_remain, rec.bw_remain,
            rec.unreachable_count, rec.subgoals_ok, rec.subgoals_fail,
            rec.request_id, rec.arrival_time,
            rec.epsilon_high, rec.epsilon_low, rec.steps_done,
        ])
        self._csv_file.flush()

    # ──────────────────────────────────────────────────────────
    #  训练结束后调用：生成完整报告
    # ──────────────────────────────────────────────────────────
    def report(self):
        """生成分析报告（文字 + 图表）"""
        self._csv_file.close()

        if not self.records:
            logger.warning("⚠️ TrainingAnalyzer: 没有记录，跳过报告生成")
            return

        logger.info("📊 正在生成训练分析报告...")
        self._write_text_report()
        self._plot_charts()
        logger.info(f"✅ 报告已保存至: {self.output_dir}")

    def _plot_epsilon_chart(self):
        """单独输出 Epsilon 衰减 + steps_done 图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            return

        eps_recs = [r for r in self.records if r.epsilon_low is not None]
        if not eps_recs:
            logger.warning("⚠️ 没有 epsilon 数据，跳过 epsilon 图表（检查 agent 是否正确传入）")
            # 即使没有 epsilon，也输出一个 steps_done 是否递增的说明图
            self._plot_steps_fallback()
            return

        episodes   = [r.episode    for r in eps_recs]
        eps_high   = [r.epsilon_high for r in eps_recs]
        eps_low    = [r.epsilon_low  for r in eps_recs]
        steps_done = [r.steps_done   for r in eps_recs]
        success    = [int(r.success) for r in eps_recs]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Epsilon Decay & Steps Monitor", fontsize=14, fontweight='bold')

        # ① Epsilon 曲线
        ax = axes[0]
        ax.plot(episodes, eps_high, color='#e74c3c', linewidth=1.5, label='ε_high')
        ax.plot(episodes, eps_low,  color='#3498db', linewidth=1.5, label='ε_low')
        ax.set_title('Epsilon Decay (should decrease over time)')
        ax.set_ylabel('Epsilon')
        ax.set_ylim(0, max(max(eps_high), max(eps_low)) * 1.1 + 0.01)
        ax.legend()
        ax.grid(alpha=0.3)
        # 标注首尾值
        ax.annotate(f"start={eps_low[0]:.3f}", xy=(episodes[0], eps_low[0]),
                    xytext=(episodes[0] + len(episodes)*0.05, eps_low[0]+0.02),
                    fontsize=9, color='#3498db')
        ax.annotate(f"end={eps_low[-1]:.3f}", xy=(episodes[-1], eps_low[-1]),
                    xytext=(episodes[-1] - len(episodes)*0.15, eps_low[-1]+0.02),
                    fontsize=9, color='#3498db')

        # ② steps_done 曲线（应该线性递增）
        ax = axes[1]
        ax.plot(episodes, steps_done, color='#2ecc71', linewidth=1.5, label='steps_done')
        ax.set_title('steps_done (should increase linearly — if flat, epsilon bug still present)')
        ax.set_ylabel('Steps Done')
        ax.grid(alpha=0.3)
        ax.legend()
        # 判断是否在递增
        if steps_done[-1] <= steps_done[0] + 10:
            ax.text(0.5, 0.5, 'steps_done NOT increasing!\nCheck store_transition_low',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='red',
                    bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
        else:
            growth = steps_done[-1] - steps_done[0]
            ax.text(0.98, 0.05, f'Total steps: {steps_done[-1]:,} (+{growth:,})',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=10, color='green')

        # ③ 成功率 vs epsilon（散点，看探索率与成功率的关系）
        ax = axes[2]
        # 滑动窗口成功率
        w = 50
        ep_arr = np.array(episodes)
        succ_arr = np.array(success, dtype=float)
        eps_arr  = np.array(eps_low)
        if len(succ_arr) >= w:
            smooth_succ = np.convolve(succ_arr, np.ones(w)/w, mode='valid')
            smooth_eps  = eps_arr[w-1:]
            smooth_ep   = ep_arr[w-1:]
            ax2 = ax.twinx()
            ax.plot(smooth_ep, smooth_succ, color='#2ecc71', linewidth=1.5, label='Success Rate')
            ax2.plot(smooth_ep, smooth_eps, color='#e74c3c', linewidth=1.0,
                     linestyle='--', alpha=0.7, label='ε_low')
            ax.set_ylabel('Success Rate', color='#2ecc71')
            ax2.set_ylabel('Epsilon', color='#e74c3c')
            ax.yaxis.set_major_formatter(__import__('matplotlib').ticker.PercentFormatter(1.0))
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax.set_title('Success Rate vs Epsilon (as epsilon decreases, success should rise)')
        ax.set_xlabel('Episode')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        eps_path = self.output_dir / "charts" / "epsilon_decay.png"
        plt.savefig(eps_path, dpi=130, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 Epsilon 图表已保存: {eps_path}")

        # 同时输出纯文字的 epsilon 变化表（每200 episode一行）
        self._print_epsilon_table(eps_recs)

    def _plot_steps_fallback(self):
        """没有 epsilon 数据时，输出诊断信息"""
        diag_path = self.output_dir / "epsilon_diagnosis.txt"
        lines = [
            "=" * 55,
            "  Epsilon 数据未记录 — 诊断信息",
            "=" * 55,
            "",
            "可能原因:",
            "  1. agent 对象未正确传入 analyzer.record()",
            "     检查 phase3_rl_trainer.py 中:",
            "     self.analyzer.record(..., agent=self.agent)",
            "",
            "  2. agent 没有 epsilon_low 属性",
            "     检查 agent.py __init__ 中是否有:",
            "     self.epsilon_low = ...",
            "",
            "  3. HRL 架构中 high_agent 和主 agent 不是同一对象",
            "     需要直接传 self.agent（trainer 中的那个）",
            "",
            "临时验证方法 — 在训练循环里加一行:",
            "  print(type(self.agent), hasattr(self.agent, 'epsilon_low'))",
        ]
        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print("\n".join(lines))

    def _print_epsilon_table(self, eps_recs):
        """每 200 episode 输出一行 epsilon 摘要"""
        lines = [
            "",
            "─" * 65,
            "  Epsilon 衰减明细（每200 Episode）",
            "─" * 65,
            f"  {'Episode':>8}  {'ε_high':>8}  {'ε_low':>8}  {'steps_done':>12}  {'成功率':>8}",
        ]
        step = max(1, len(eps_recs) // 10)
        for i in range(0, len(eps_recs), step):
            r = eps_recs[i]
            # 计算该位置附近的成功率
            window = eps_recs[max(0,i-50):i+50]
            sr = sum(x.success for x in window) / len(window) if window else 0
            lines.append(
                f"  {r.episode:>8}  {r.epsilon_high:>8.4f}  {r.epsilon_low:>8.4f}"
                f"  {r.steps_done:>12,}  {sr:>7.1%}"
            )
        lines.append("─" * 65)
        print("\n".join(lines))
        # 写入文字报告
        report_path = self.output_dir / "epsilon_decay_table.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logger.info(f"📄 Epsilon 明细表已保存: {report_path}")

    # ──────────────────────────────────────────────────────────
    #  文字报告
    # ──────────────────────────────────────────────────────────
    def _write_text_report(self):
        recs   = self.records
        total  = len(recs)
        succ   = [r for r in recs if r.success]
        fail   = [r for r in recs if not r.success]
        n_succ = len(succ)
        n_fail = len(fail)

        lines = []
        w = lines.append  # 简写

        w("=" * 65)
        w("  HRL-SFC 训练失败分析报告")
        w("=" * 65)
        w(f"  总 Episode 数 : {total}")
        w(f"  成功数        : {n_succ}  ({n_succ/total*100:.1f}%)")
        w(f"  失败数        : {n_fail}  ({n_fail/total*100:.1f}%)")
        w("")

        # ── 1. 失败原因分布 ────────────────────────────────
        w("─" * 65)
        w("  【1】失败原因分布")
        w("─" * 65)
        reason_counter = Counter(r.fail_reason for r in fail)
        for reason, cnt in reason_counter.most_common():
            pct = cnt / n_fail * 100 if n_fail else 0
            bar = "█" * int(pct / 2)
            w(f"  {reason:<18} {cnt:>5} 次  {pct:5.1f}%  {bar}")
        w("")

        # ── 2. 失败时的任务进度 ────────────────────────────
        w("─" * 65)
        w("  【2】失败时的任务完成进度")
        w("─" * 65)
        fail_with_vnf  = [r for r in fail if r.vnf_done is not None and r.vnf_total]
        fail_with_dest = [r for r in fail if r.dest_done is not None and r.dest_total]

        if fail_with_vnf:
            vnf_ratios = [r.vnf_done / r.vnf_total for r in fail_with_vnf]
            w(f"  VNF 完成率 (失败时) : 均={np.mean(vnf_ratios)*100:.1f}%  "
              f"  0/{3}完成: {sum(1 for x in vnf_ratios if x==0)*100/len(vnf_ratios):.0f}%"
              f"  1/{3}完成: {sum(1 for x in vnf_ratios if 0<x<=0.34)*100/len(vnf_ratios):.0f}%"
              f"  2/{3}完成: {sum(1 for x in vnf_ratios if 0.34<x<1)*100/len(vnf_ratios):.0f}%")

            vnf_dist = Counter(r.vnf_done for r in fail_with_vnf)
            for k in sorted(vnf_dist):
                pct = vnf_dist[k] / len(fail_with_vnf) * 100
                w(f"    VNF完成 {k}/{fail_with_vnf[0].vnf_total} : {vnf_dist[k]:>5} 次  {pct:5.1f}%")

        if fail_with_dest:
            dest_dist = Counter(r.dest_done for r in fail_with_dest)
            w(f"\n  目的地完成分布 (失败时):")
            for k in sorted(dest_dist):
                pct = dest_dist[k] / len(fail_with_dest) * 100
                w(f"    Dest完成 {k}/{fail_with_dest[0].dest_total} : {dest_dist[k]:>5} 次  {pct:5.1f}%")
        w("")

        # ── 3. 资源瓶颈分析 ────────────────────────────────
        w("─" * 65)
        w("  【3】失败时的资源剩余量")
        w("─" * 65)
        fail_with_cpu = [r for r in fail if r.cpu_remain is not None]
        fail_with_mem = [r for r in fail if r.mem_remain is not None]
        fail_with_bw  = [r for r in fail if r.bw_remain  is not None]

        total_cpu = 28 * 55
        total_mem = 28 * 45
        total_bw  = 45 * 90
        util_candidates = []

        if fail_with_cpu:
            cpu_arr = np.array([r.cpu_remain for r in fail_with_cpu], dtype=float)
            w(f"  CPU 剩余: 均={cpu_arr.mean():.1f}  最小={cpu_arr.min():.1f}  最大={cpu_arr.max():.1f}")
            util_candidates.append(('CPU', 1 - cpu_arr.mean() / total_cpu))
        else:
            w("  CPU 剩余: 无数据（pool 中未找到 cpu_avail 字段）")

        if fail_with_mem:
            mem_arr = np.array([r.mem_remain for r in fail_with_mem], dtype=float)
            w(f"  MEM 剩余: 均={mem_arr.mean():.1f}  最小={mem_arr.min():.1f}  最大={mem_arr.max():.1f}")
            util_candidates.append(('MEM', 1 - mem_arr.mean() / total_mem))
        else:
            w("  MEM 剩余: 无数据（pool 中未找到 mem_avail 字段）")

        if fail_with_bw:
            bw_arr = np.array([r.bw_remain for r in fail_with_bw], dtype=float)
            w(f"  BW  剩余: 均={bw_arr.mean():.1f}  最小={bw_arr.min():.1f}  最大={bw_arr.max():.1f}")
            util_candidates.append(('BW',  1 - bw_arr.mean() / total_bw))
        else:
            w("  BW  剩余: 无数据（pool 中未找到 bw_avail 字段）")

        if util_candidates:
            w(f"\n  失败时平均资源占用率:")
            for name, util in util_candidates:
                w(f"  {name}: {util*100:.1f}%")
            bottleneck = max(util_candidates, key=lambda x: x[1])
            w(f"  ⚠️  瓶颈资源: {bottleneck[0]} (占用率最高 {bottleneck[1]*100:.1f}%)")
        else:
            w("  （资源字段均未读取到，请检查 AllResourceManager.pool 的属性名）")
        w("")

        # ── 4. 训练阶段对比（前/中/后期） ─────────────────
        w("─" * 65)
        w("  【4】训练进程分析（成功率变化）")
        w("─" * 65)
        n_seg = 5
        seg_size = max(1, total // n_seg)
        for i in range(n_seg):
            seg = recs[i*seg_size : (i+1)*seg_size]
            if not seg: continue
            seg_succ = sum(1 for r in seg if r.success)
            seg_rate = seg_succ / len(seg) * 100
            seg_util = np.mean([r.res_util for r in seg if r.res_util is not None])
            w(f"  Episode {i*seg_size+1:>5}~{(i+1)*seg_size:>5} : "
              f"成功率={seg_rate:5.1f}%  平均资源利用率={seg_util:.2f}")
        w("")

        # ── 5. 路径不可达统计 ──────────────────────────────
        w("─" * 65)
        w("  【5】路径不可达 & 子目标失败")
        w("─" * 65)
        unreach = [r for r in recs if r.unreachable_count and r.unreachable_count > 0]
        w(f"  遇到不可达节点的 Episode : {len(unreach)} / {total}  ({len(unreach)/total*100:.1f}%)")
        if fail:
            fail_unreach = [r for r in fail if r.unreachable_count and r.unreachable_count > 0]
            w(f"  失败中含不可达节点比例  : {len(fail_unreach)/n_fail*100:.1f}%")
        sg_ok_arr   = np.array([r.subgoals_ok   for r in recs if r.subgoals_ok   is not None])
        sg_fail_arr = np.array([r.subgoals_fail  for r in recs if r.subgoals_fail is not None])
        if len(sg_ok_arr):
            w(f"  子目标成功数 (均/Episode): {sg_ok_arr.mean():.2f}")
            w(f"  子目标失败数 (均/Episode): {sg_fail_arr.mean():.2f}")
        w("")

        # ── 6. 改进建议 ────────────────────────────────────
        w("─" * 65)
        w("  【6】改进建议")
        w("─" * 65)

        top_reason = reason_counter.most_common(1)[0][0] if reason_counter else None

        # 统计 VNF完成但Dest=0 的比例（目的地连接阶段失败）
        fail_vnf_done_dest_zero = [
            r for r in fail if r.vnf_done is not None and r.vnf_total
            and r.vnf_done >= r.vnf_total and (r.dest_done or 0) == 0
        ]
        dest_stage_fail_pct = len(fail_vnf_done_dest_zero) / n_fail * 100 if n_fail else 0

        if dest_stage_fail_pct > 30:
            w(f"  ⚡ 关键发现: {dest_stage_fail_pct:.0f}% 的失败发生在"
              f" VNF全部完成后的目的地连接阶段（BW耗尽）")
            w("  主要问题: VNF 部署成功，但目的地连接阶段带宽不足")
            w("  建议:")
            w("  ① 优化组播树构建：减少树边数（当前平均14条），")
            w("     共享更多主干路径可降低总带宽消耗")
            w("  ② 调整 VNF 放置位置：将 VNF 部署在靠近目的地")
            w("     的节点，缩短 VNF→Dest 的路径长度")
            w("  ③ 为目的地连接阶段加更强的带宽感知路由")
            w("     (compute_bw_aware_path 已有，检查是否都生效)")
            w("  ④ 考虑增大链路带宽容量（当前 BW=90/链路）")
        elif top_reason == FailReason.MASK_ZERO:
            w("  主要问题: 资源长期被占用，新请求无法分配")
            w("  建议:")
            w("  ① 检查生命周期管理器的过期释放是否正常触发")
            w("  ② 降低数据集到达率 λ 或缩短请求 lifetime")
            w("  ③ 增加节点/链路容量（当前 CPU=55, MEM=45, BW=90）")

        elif top_reason == FailReason.PATH_BLOCKED:
            w("  主要问题: 带宽不足导致路径规划失败")
            w("  建议:")
            w("  ① 优化组播树算法，减少树边数（当前均14条）")
            w("  ② 增大链路带宽容量（当前 BW=90/链路）")
            w("  ③ 引入带宽感知的 VNF 放置策略，避免热点链路")

        elif top_reason == FailReason.TIMEOUT:
            w("  主要问题: Agent 在有限步数内无法完成任务")
            w("  建议:")
            w("  ① 增大 max_low_steps（当前50）或 max_steps_per_episode")
            w("  ② 检查路径引导是否生效，避免 RL 随机游走")
            w("  ③ 优化奖励函数，增加距离目标的稠密引导奖励")

        elif top_reason == FailReason.VNF_FAIL:
            w("  主要问题: VNF 部署阶段资源不足")
            w("  建议:")
            w("  ① Agent 需要学习避开资源紧张节点（CPU/MEM）")
            w("  ② 在状态特征中强化剩余资源的表示")
            w("  ③ 部署失败时给予更强的负奖励以加速策略收敛")

        elif top_reason == FailReason.NO_PROGRESS:
            w("  主要问题: Agent 连续多个 cycle 无进展被强制终止")
            w("  建议:")
            w("  ① 增大 MAX_NO_PROGRESS 阈值（当前5），给 Agent 更多容错空间")
            w("  ② 检查高层动作掩码是否过于保守，导致可选目标太少")
            w("  ③ 增加无进展时的稠密引导奖励（距离梯度），避免原地振荡")
            w("  ④ 检查低层步数上限是否足够（max_low_steps）")

        elif top_reason == FailReason.BW_EXHAUSTED:
            w("  主要问题: 带宽耗尽，Agent 无法走新边")
            w("  建议:")
            w("  ① 检查 _skip_edge BW 泄漏修复是否生效")
            w("  ② 优化组播树减少边数，降低总带宽消耗")
            w("  ③ 增强热点链路惩罚，引导 Agent 绕开拥塞链路")

        elif top_reason == FailReason.TRAPPED:
            w("  主要问题: Agent 陷入死路（所有邻居 BW 不足）")
            w("  建议:")
            w("  ① 在高层动作掩码中预先过滤带宽不足的方向")
            w("  ② 检查是否有 BW 泄漏导致链路虚报为 0")

        # 通用建议（根据训练趋势）
        early_rate = np.mean([r.success for r in recs[:seg_size]])
        late_rate  = np.mean([r.success for r in recs[-seg_size:]])
        if late_rate < early_rate - 0.1:
            w("  ⚠️  成功率在训练后期明显下降：资源泄漏或过拟合，")
            w("     建议检查资源回滚逻辑或降低 epsilon 衰减速度")
        elif late_rate > early_rate + 0.1:
            w("  ✅  成功率在训练后期有明显提升，Agent 正在有效学习")

        w("")
        w("=" * 65)

        report_path = self.output_dir / "failure_analysis.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        # 同时打印到控制台
        print("\n")
        print("\n".join(lines))

    # ──────────────────────────────────────────────────────────
    #  图表生成
    # ──────────────────────────────────────────────────────────
    def _plot_charts(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            logger.warning("⚠️ matplotlib 未安装，跳过图表生成")
            return

        recs  = self.records
        total = len(recs)
        episodes   = [r.episode   for r in recs]
        successes  = [int(r.success) for r in recs]
        res_utils  = [r.res_util  if r.res_util  is not None else 0 for r in recs]
        rewards    = [r.reward    if r.reward    is not None else 0 for r in recs]

        # 滑动窗口平均
        def smooth(arr, w=50):
            arr = np.array(arr, dtype=float)
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w)/w, mode='valid')

        # 设置中文字体（Windows: 微软雅黑，Linux/Mac: 回退 sans-serif）
        import platform
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        elif platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("HRL-SFC Training Analysis", fontsize=15, fontweight='bold')

        # ① 成功率曲线
        ax = axes[0, 0]
        sm = smooth(successes)
        ax.plot(range(len(sm)), sm, color='#2ecc71', linewidth=1.5, label='SuccRate(avg)')
        ax.set_title('Success Rate (sliding avg)')
        ax.set_ylabel('Rate')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3)
        ax.legend()

        # ② 资源利用率曲线
        ax = axes[0, 1]
        sm_util = smooth(res_utils)
        ax.plot(range(len(sm_util)), sm_util, color='#e74c3c', linewidth=1.5, label='ResUtil')
        ax.axhline(0.85, color='gray', linestyle='--', linewidth=0.8, label='Warning 0.85')
        ax.set_title('Resource Utilization')
        ax.set_ylabel('Utilization')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()

        # ③ 失败原因饼图
        ax = axes[1, 0]
        fail_recs = [r for r in recs if not r.success]
        if fail_recs:
            reason_cnt = Counter(r.fail_reason for r in fail_recs)
            labels = list(reason_cnt.keys())
            sizes  = list(reason_cnt.values())
            colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#95a5a6']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                   colors=colors[:len(labels)], startangle=90,
                   textprops={'fontsize': 9})
        ax.set_title('Failure Reason Distribution')

        # ④ 失败时VNF进度分布
        ax = axes[1, 1]
        fail_vnf = [r for r in fail_recs if r.vnf_done is not None and r.vnf_total]
        if fail_vnf:
            vnf_done_arr = [r.vnf_done for r in fail_vnf]
            ax.hist(vnf_done_arr, bins=range(0, 5), align='left',
                    color='#f39c12', edgecolor='white', rwidth=0.7)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['0/3', '1/3', '2/3', '3/3'])
        ax.set_title('VNF Progress at Failure')
        ax.set_xlabel('VNF Completed')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # ⑤ 奖励曲线
        ax = axes[2, 0]
        sm_rwd = smooth(rewards)
        ax.plot(range(len(sm_rwd)), sm_rwd, color='#3498db', linewidth=1.5)
        ax.set_title('Reward (sliding avg)')
        ax.set_ylabel('Reward')
        ax.grid(alpha=0.3)

        # ⑥ 剩余资源箱线图（成功 vs 失败对比）
        ax = axes[2, 1]
        succ_recs = [r for r in recs if r.success]
        data_labels = []
        data_vals   = []
        for label, field, group in [
            ('Fail CPU', 'cpu_remain', fail_recs),
            ('Succ CPU', 'cpu_remain', succ_recs),
            ('Fail BW',  'bw_remain',  fail_recs),
            ('Succ BW',  'bw_remain',  succ_recs),
        ]:
            vals = [getattr(r, field) for r in group if getattr(r, field) is not None]
            if vals:
                data_labels.append(label)
                data_vals.append(vals)
        if data_vals:
            bp = ax.boxplot(data_vals, patch_artist=True)
            colors_box = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
            for patch, color in zip(bp['boxes'], colors_box[:len(data_vals)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticklabels(data_labels, fontsize=8)
        ax.set_title('Remaining Resources: Fail vs Success')
        ax.set_ylabel('Remaining Capacity')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = self.output_dir / "charts" / "training_analysis.png"
        plt.savefig(chart_path, dpi=130, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 图表已保存: {chart_path}")

        # ── 单独输出 Epsilon 衰减图 ──────────────────────────
        self._plot_epsilon_chart()