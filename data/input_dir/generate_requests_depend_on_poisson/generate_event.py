#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件生成器 - 时间槽版本
---------------------------------------------------------
功能：
1. 将请求序列转换为按时间步索引的事件列表（保留原有功能）
2. 🔥 新增：按时间槽索引的事件列表
---------------------------------------------------------
主要改动：
1. ✅ 生成两种事件列表：时间步版 + 时间槽版
2. ✅ 时间槽事件列表：event_list_by_slot.pkl
3. ✅ 保持向后兼容（保留原有时间步事件列表）
"""

import pickle
import os
import sys

# 确保路径与 main_generate.py 一致
DATA_DIR = './data/input_dir'


def process_single_file_timestep(input_filename, output_filename):
    """
    读取请求文件，生成时间步事件列表（原版）

    结构: List[Dict] -> [
        {'time_step': 0, 'arrive_event': [id1], 'leave_event': []},
        ...
    ]

    Args:
        input_filename: 输入文件名（requests.pkl）
        output_filename: 输出文件名（events.pkl）
    """
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️  跳过: 未找到 {input_path}")
        return

    print(f"🔄 正在处理（时间步版）: {input_filename} -> {output_filename} ...")

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    if not requests_list:
        print("❌ 请求列表为空！")
        return

    # 1. 找出最大时间步
    max_leave_step = 0
    for req in requests_list:
        l_step = int(req['leave_time_step'])
        if l_step > max_leave_step:
            max_leave_step = l_step

    print(f"   ⏱️  最大时间步: {max_leave_step}")

    # 2. 初始化事件列表
    event_list = []
    for t in range(max_leave_step + 2):
        event_list.append({
            'time_step': t,
            'arrive_event': [],
            'leave_event': []
        })

    # 3. 填充事件
    count_arrive = 0
    count_leave = 0

    for req in requests_list:
        req_id = req['id']
        t_arr = int(req['arrive_time_step'])
        t_leave = int(req['leave_time_step'])

        # 记录到达
        if 0 <= t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req_id)
            count_arrive += 1

        # 记录离开
        if 0 <= t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req_id)
            count_leave += 1

    # 4. 保存结果
    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f"✅ 已生成（时间步版）: {output_path}")
    print(f"   统计: 到达 {count_arrive} 个, 离开 {count_leave} 个")
    print("-" * 50)


def process_single_file_timeslot(input_filename, output_filename):
    """
    🔥 新增：读取请求文件，生成时间槽事件列表

    结构: List[Dict] -> [
        {'time_slot': 0, 'arrive_event': [id1], 'leave_event': []},
        ...
    ]

    Args:
        input_filename: 输入文件名（requests.pkl）
        output_filename: 输出文件名（events_by_slot.pkl）
    """
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f"⚠️  跳过: 未找到 {input_path}")
        return

    print(f"🔄 正在处理（时间槽版）: {input_filename} -> {output_filename} ...")

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    if not requests_list:
        print("❌ 请求列表为空！")
        return

    # 1. 找出最大时间槽
    max_leave_slot = 0
    for req in requests_list:
        # 🔥 读取时间槽字段
        if 'leave_time_slot' in req:
            l_slot = int(req['leave_time_slot'])
            if l_slot > max_leave_slot:
                max_leave_slot = l_slot
        else:
            print(f"⚠️  请求 {req['id']} 缺少 'leave_time_slot' 字段，跳过")
            return

    print(f"   ⏱️  最大时间槽: {max_leave_slot}")

    # 2. 初始化事件列表
    event_list = []
    for slot in range(max_leave_slot + 2):
        event_list.append({
            'time_slot': slot,
            'arrive_event': [],
            'leave_event': []
        })

    # 3. 填充事件
    count_arrive = 0
    count_leave = 0

    for req in requests_list:
        req_id = req['id']
        t_arr = int(req['time_slot'])  # 🔥 使用时间槽字段
        t_leave = int(req['leave_time_slot'])

        # 记录到达
        if 0 <= t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req_id)
            count_arrive += 1

        # 记录离开
        if 0 <= t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req_id)
            count_leave += 1

    # 4. 保存结果
    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f"✅ 已生成（时间槽版）: {output_path}")
    print(f"   统计: 到达 {count_arrive} 个, 离开 {count_leave} 个")
    print("-" * 50)


def generate_events():
    """
    生成所有事件列表
    """
    print("=" * 60)
    print("🚀 生成事件列表 (Event Generation)")
    print(f"📂 工作目录: {DATA_DIR}")
    print("=" * 60)

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. 处理 Phase 1 (专家训练数据)
    # ---------------------------------------------------------
    print("\n🔥 处理 Phase 1 数据...")

    # 时间步版（保留兼容）
    process_single_file_timestep(
        'phase1_requests.pkl',
        'phase1_events.pkl'
    )

    # 🔥 时间槽版（新增）
    process_single_file_timeslot(
        'phase1_requests.pkl',
        'phase1_events_by_slot.pkl'
    )

    # ---------------------------------------------------------
    # 2. 处理 Phase 3 (RL 评估数据)
    # ---------------------------------------------------------
    print("\n🔥 处理 Phase 3 数据...")

    # 时间步版（保留兼容）
    process_single_file_timestep(
        'phase3_requests.pkl',
        'phase3_events.pkl'
    )

    # 🔥 时间槽版（新增）
    process_single_file_timeslot(
        'phase3_requests.pkl',
        'phase3_events_by_slot.pkl'
    )

    print("\n🎉 所有事件列表生成完毕！")

    # 🔥 打印文件列表
    print(f"\n📁 生成的事件文件:")
    for filename in sorted(os.listdir(DATA_DIR)):
        if 'event' in filename and filename.endswith('.pkl'):
            filepath = os.path.join(DATA_DIR, filename)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   - {filename} ({size:.1f} KB)")


def verify_events(events_filename):
    """
    验证事件列表的正确性

    Args:
        events_filename: 事件文件名
    """
    filepath = os.path.join(DATA_DIR, events_filename)

    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return

    print(f"\n🔍 验证事件列表: {events_filename}")

    with open(filepath, 'rb') as f:
        events = pickle.load(f)

    # 统计
    total_arrive = sum(len(e['arrive_event']) for e in events)
    total_leave = sum(len(e['leave_event']) for e in events)

    # 找到有事件的时间槽
    non_empty_slots = [
        e.get('time_slot', e.get('time_step'))
        for e in events
        if e['arrive_event'] or e['leave_event']
    ]

    if non_empty_slots:
        min_slot = min(non_empty_slots)
        max_slot = max(non_empty_slots)
    else:
        min_slot = max_slot = 0

    print(f"   总事件槽数: {len(events)}")
    print(f"   总到达事件: {total_arrive}")
    print(f"   总离开事件: {total_leave}")
    print(f"   有效槽范围: {min_slot} - {max_slot}")

    # 检查前几个有事件的槽
    print(f"\n   前5个有事件的槽:")
    count = 0
    for e in events:
        slot_id = e.get('time_slot', e.get('time_step'))
        if e['arrive_event'] or e['leave_event']:
            print(f"      Slot {slot_id}: "
                  f"到达={e['arrive_event']}, "
                  f"离开={e['leave_event']}")
            count += 1
            if count >= 5:
                break


if __name__ == '__main__':
    # 生成事件列表
    generate_events()

    # 🔥 验证生成的事件列表
    print("\n" + "=" * 60)
    print("🔍 验证事件列表")
    print("=" * 60)

    verify_events('phase1_events.pkl')
    verify_events('phase1_events_by_slot.pkl')
    verify_events('phase3_events.pkl')
    verify_events('phase3_events_by_slot.pkl')

    print("\n✅ 验证完成！")