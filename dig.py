import pickle
import numpy as np
import os

# 修改为您实际的文件路径
request_file = r"data/input_dir/phase3_requests.pkl"
# 或者 events 文件 (虽然通常 request 文件里的 lifetime 更直观)
# events_file = r"data/input_dir/phase3_events_by_slot.pkl"

if os.path.exists(request_file):
    print(f"📂 正在读取: {request_file}")
    with open(request_file, 'rb') as f:
        requests = pickle.load(f)

    print(f"📊 样本数量: {len(requests)}")

    # 提取 lifetime 和 arrival_time
    lifetimes = []
    arrival_times = []

    for r in requests[:100]:  # 看前100个就够了
        lifetimes.append(r.get('lifetime', -1))
        arrival_times.append(r.get('arrival_time', -1))

    avg_life = np.mean(lifetimes)
    avg_arrival = np.mean(arrival_times)

    print("-" * 30)
    print(f"平均 Lifetime: {avg_life:.4f}")
    print(f"平均 Arrival : {avg_arrival:.4f}")
    print("-" * 30)

    # --- 关键判断 ---
    # 假设 delta_t = 0.01 (通常值)
    # 如果 lifetime 是 1000 左右，且 arrival 是 0.01, 0.02... -> 说明 lifetime 是整数(切片)?
    # 如果 lifetime 是 10.0 左右 -> 说明是秒数?

    print("🔍 样本数据 (前5条):")
    for i in range(5):
        print(
            f"ID: {requests[i].get('id')} | Arrival: {requests[i].get('arrival_time')} | Lifetime: {requests[i].get('lifetime')}")

else:
    print("❌ 找不到文件，请确认路径。")