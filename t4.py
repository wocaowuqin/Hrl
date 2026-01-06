"""最简时间切片诊断"""
import pickle
import numpy as np

print("\n" + "="*70)
print("🧪 数据文件诊断")
print("="*70)

# 1. 直接加载数据文件
try:
    with open('data/input_dir/generate_requests_depend_on_poisson/data/到达率50/phase3_requests.pkl', 'rb') as f:
        requests = pickle.load(f)
    print(f"\n✅ 请求文件加载成功: {len(requests)} 条")
except Exception as e:
    print(f"\n❌ 请求文件加载失败: {e}")
    requests = []

# 2. 加载时间槽数据
try:
    with open('data/input_dir/generate_requests_depend_on_poisson/data/到达率50/phase3_requests_by_slot.pkl', 'rb') as f:
        by_slot = pickle.load(f)
    print(f"✅ 时间槽文件加载成功: {len(by_slot)} 个时间槽")
except Exception as e:
    print(f"❌ 时间槽文件加载失败: {e}")
    by_slot = {}

# 3. 分析时间槽分布
if requests:
    print("\n" + "="*70)
    print("📊 时间槽分布分析")
    print("="*70)

    # 提取所有时间槽
    time_slots = [r.get('time_slot', -1) for r in requests]
    unique_slots = sorted(set(time_slots))

    print(f"\n基本统计:")
    print(f"  时间槽范围: {min(time_slots)} ~ {max(time_slots)}")
    print(f"  唯一时间槽数: {len(unique_slots)}")
    print(f"  总请求数: {len(requests)}")

    # 前10个时间槽
    print(f"\n前10个时间槽:")
    for slot in unique_slots[:10]:
        count = time_slots.count(slot)
        print(f"  时间槽 {slot:4d}: {count:3d} 个请求")

    # 检查时间槽是否单调递增
    print(f"\n请求的时间槽顺序（前20个）:")
    for i in range(min(20, len(requests))):
        req = requests[i]
        print(f"  请求 {i+1:2d}: ID={req.get('id'):3d}, 时间槽={req.get('time_slot'):4d}")

    # 检查是否有时间槽跳跃
    slot_changes = 0
    for i in range(1, len(requests)):
        if requests[i].get('time_slot') != requests[i-1].get('time_slot'):
            slot_changes += 1

    print(f"\n时间槽切换次数: {slot_changes}")
    print(f"平均每个时间槽的请求数: {len(requests) / len(unique_slots):.2f}")

# 4. 检查by_slot结构
if by_slot:
    print("\n" + "="*70)
    print("📊 requests_by_slot 结构")
    print("="*70)

    slots = sorted(list(by_slot.keys()))
    print(f"\n前10个时间槽及其请求数:")
    for slot in slots[:10]:
        reqs_in_slot = by_slot[slot]
        print(f"  时间槽 {slot:4d}: {len(reqs_in_slot):3d} 个请求")

print("\n" + "="*70)
print("✅ 诊断完成")
print("="*70)