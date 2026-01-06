#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成器 - 时间槽版本 (双阶段生成)
🔥 修改版：增加时间槽密度，避免每槽只有1个请求
"""

import random
import numpy as np
import pickle
import os

# ========== 1. 核心网络配置 (论文/MATLAB一致) ==========
NODE_TRAFFIC_LIST = [16, 21, 22, 24, 25, 26, 27, 28]

# ========== 2. 时间与负载参数 ==========
TIME_INTERVAL = 5.0  # 时间间隔 5秒

# 🔥 【关键修改】提高到达率 + 增大时间槽
LAMBDA_PER_INTERVAL = 280.0  # 每间隔50个请求（原来7.0）
LAMBDA_RATE = LAMBDA_PER_INTERVAL / TIME_INTERVAL  # 10 req/s

# 🔥 【关键修改】时间槽配置
TIME_SLOT_DELTA = 0.1  # 100ms（原来10ms）
# 效果：5秒 = 50个时间槽，50个请求 → 平均每槽1个请求
# 每个时间槽可能有0-5个请求（泊松分布）

# ========== 3. 业务请求参数 ==========
NUM_DESTINATIONS = 5
VNF_CHAIN_LENGTH = 3
VNF_TYPES = 8

MIN_BANDWIDTH = 4
MAX_BANDWIDTH = 8
MEAN_LIFETIME = 3  # 3个时间间隔


def generate_single_request(req_id, source, destinations, vnf_chain, bandwidth,
                            cpu_needs, mem_needs, arrive_time, lifetime,
                            delta_t=TIME_SLOT_DELTA):
    """生成单个请求对象（时间槽版本）"""
    leave_time = arrive_time + lifetime

    arrive_time_step = int(np.ceil(arrive_time))
    leave_time_step = int(np.ceil(leave_time))

    # 🔥 转换为时间槽
    time_slot = int(arrive_time / delta_t)
    leave_time_slot = int(leave_time / delta_t)
    duration = leave_time_slot - time_slot

    return {
        'id': req_id,
        'source': source,
        'dest': destinations,
        'vnf': vnf_chain,
        'bw_origin': bandwidth,
        'cpu_origin': cpu_needs,
        'memory_origin': mem_needs,

        # 时间信息
        'arrival_time': arrive_time,
        'leave_time': leave_time,
        'lifetime': lifetime,

        # 时间槽信息
        'time_slot': time_slot,
        'leave_time_slot': leave_time_slot,
        'duration': duration,

        # 兼容性
        'arrive_time_step': arrive_time_step,
        'leave_time_step': leave_time_step
    }


def generate_poisson_arrivals(T, lamda):
    """生成泊松到达时间序列"""
    arrivals = []
    time_state = 0
    while time_state < T:
        interval = np.random.exponential(1.0 / lamda)
        time_state += interval
        if time_state < T:
            arrivals.append(time_state)
    return arrivals


def generate_vnf_resources(bandwidth):
    """生成资源需求"""
    cpu_factor = np.random.rand() * 2.75 + 0.25
    mem_factor = np.random.rand() * 1.75 + 0.25
    cpu = round(bandwidth * cpu_factor)
    mem = round(bandwidth * mem_factor)
    return cpu, mem


def group_requests_by_time_slot(requests):
    """将请求按时间槽分组"""
    grouped = {}
    for req in requests:
        slot = req['time_slot']
        if slot not in grouped:
            grouped[slot] = []
        grouped[slot].append(req)
    return grouped


def print_statistics(requests, requests_by_slot, phase_name):
    """打印统计信息"""
    print(f"\n{'=' * 60}")
    print(f"📊 {phase_name} 统计信息")
    print(f"{'=' * 60}")

    print(f"总请求数: {len(requests)}")
    print(f"时间槽数: {len(requests_by_slot)}")

    if requests_by_slot:
        min_slot = min(requests_by_slot.keys())
        max_slot = max(requests_by_slot.keys())
        print(f"时间槽范围: {min_slot} - {max_slot}")
        print(f"实际时间范围: {min_slot * TIME_SLOT_DELTA:.2f}s - {max_slot * TIME_SLOT_DELTA:.2f}s")

        slot_counts = [len(reqs) for reqs in requests_by_slot.values()]
        avg_per_slot = sum(slot_counts) / len(slot_counts)
        max_per_slot = max(slot_counts)

        print(f"平均每时间槽: {avg_per_slot:.2f} 个请求")
        print(f"最大每时间槽: {max_per_slot} 个请求")

        # 🔥 新增：显示时间槽密度分布
        print(f"\n时间槽密度分布:")
        from collections import Counter
        density = Counter(slot_counts)
        for count in sorted(density.keys())[:10]:  # 显示前10种
            print(f"  {count}个请求/槽: {density[count]} 个时间槽")

    durations = [req['duration'] for req in requests]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\n持续时间统计（时间槽）:")
        print(f"  平均: {avg_duration:.1f} ({avg_duration * TIME_SLOT_DELTA:.3f}s)")

    print(f"{'=' * 60}")


def generate_all_requests(num_intervals, lamda, seed=None, phase_name="Unknown"):
    """生成所有请求（时间槽版本）"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    T_duration = num_intervals * TIME_INTERVAL

    print("=" * 60)
    print(f"🚀 生成 {phase_name} 数据")
    print(f"   随机种子: {seed}")
    print(f"   时间间隔: {num_intervals} (总时长 {T_duration}s)")
    print(f"   到达率: {LAMBDA_PER_INTERVAL} req/间隔 ({lamda:.3f} req/s)")
    print(f"   时间槽大小: {TIME_SLOT_DELTA * 1000:.1f} ms")
    print(f"   🔥 预期时间槽数: ~{int(T_duration / TIME_SLOT_DELTA)}")
    print(f"   🔥 预期总请求数: ~{int(lamda * T_duration * len(NODE_TRAFFIC_LIST))}")
    print("=" * 60)

    all_requests = []

    # 遍历 8 个非 DC 节点作为源
    for source_node in NODE_TRAFFIC_LIST:
        arrive_times = generate_poisson_arrivals(T_duration, lamda)
        candidate_dests = [n for n in NODE_TRAFFIC_LIST if n != source_node]

        for arrive_time in arrive_times:
            destinations = random.sample(candidate_dests, NUM_DESTINATIONS)
            vnf_chain = random.sample(range(1, VNF_TYPES + 1), VNF_CHAIN_LENGTH)
            bandwidth = random.randint(MIN_BANDWIDTH, MAX_BANDWIDTH)

            cpu_needs, mem_needs = [], []
            for _ in vnf_chain:
                c, m = generate_vnf_resources(bandwidth)
                cpu_needs.append(c)
                mem_needs.append(m)

            base_slots = 1 + np.random.exponential(MEAN_LIFETIME - 1)
            while base_slots > 6:
                base_slots = 1 + np.random.exponential(MEAN_LIFETIME - 1)

            lifetime_seconds = base_slots * TIME_INTERVAL

            req = generate_single_request(
                req_id=0,
                source=source_node,
                destinations=destinations,
                vnf_chain=vnf_chain,
                bandwidth=bandwidth,
                cpu_needs=cpu_needs,
                mem_needs=mem_needs,
                arrive_time=arrive_time,
                lifetime=lifetime_seconds,
                delta_t=TIME_SLOT_DELTA
            )
            all_requests.append(req)

    all_requests.sort(key=lambda r: r['arrival_time'])
    for i, req in enumerate(all_requests, 1):
        req['id'] = i

    requests_by_slot = group_requests_by_time_slot(all_requests)

    print(f"✅ {phase_name} 生成完毕: 共 {len(all_requests)} 条请求")
    print_statistics(all_requests, requests_by_slot, phase_name)

    return all_requests, requests_by_slot


if __name__ == '__main__':
    output_dir = './data/input_dir'
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1
    print("\n🔥 开始生成 Phase 1 数据...")
    phase1_reqs, phase1_by_slot = generate_all_requests(
        num_intervals=800,
        lamda=LAMBDA_RATE,
        seed=42,
        phase_name="Phase 1 (Training)"
    )

    with open(f'{output_dir}/phase1_requests.pkl', 'wb') as f:
        pickle.dump(phase1_reqs, f)
    with open(f'{output_dir}/phase1_requests_by_slot.pkl', 'wb') as f:
        pickle.dump(phase1_by_slot, f)

    # Phase 3
    print("\n🔥 开始生成 Phase 3 数据...")
    phase3_reqs, phase3_by_slot = generate_all_requests(
        num_intervals=400,
        lamda=LAMBDA_RATE,
        seed=123,
        phase_name="Phase 3 (Evaluation)"
    )

    with open(f'{output_dir}/phase3_requests.pkl', 'wb') as f:
        pickle.dump(phase3_reqs, f)
    with open(f'{output_dir}/phase3_requests_by_slot.pkl', 'wb') as f:
        pickle.dump(phase3_by_slot, f)

    print("\n🎉 所有数据集生成完成！")