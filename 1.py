"""
验证 phase3_events_by_slot.pkl 数据集（修复版）
正确解析时间槽事件结构
"""

import pickle
import numpy as np
from collections import Counter

def load_dataset(filepath):
    """加载数据集"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 成功加载数据集: {filepath}")
        return data
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def analyze_event_structure(data):
    """分析事件结构"""

    print("\n" + "="*80)
    print("📊 数据集结构分析")
    print("="*80)

    if not data:
        print("❌ 数据为空")
        return

    print(f"\n数据类型: {type(data)}")
    print(f"时间槽数量: {len(data)}")

    # 检查第一个时间槽
    if len(data) > 0:
        first_slot = data[0]
        print(f"\n第一个时间槽结构:")
        print(f"  - 键: {list(first_slot.keys())}")
        print(f"  - time_slot: {first_slot.get('time_slot')}")
        print(f"  - arrive_event类型: {type(first_slot.get('arrive_event'))}")
        print(f"  - arrive_event长度: {len(first_slot.get('arrive_event', []))}")

        # 检查arrive_event
        arrive_events = first_slot.get('arrive_event', [])
        if arrive_events:
            print(f"\n  arrive_event内容: {arrive_events}")

            # 尝试加载对应的请求文件
            print(f"\n  💡 arrive_event包含的是请求ID，需要从单独的请求文件加载")

    return data

def extract_all_requests(data, request_file=None):
    """提取所有请求ID"""

    print("\n" + "="*80)
    print("🔍 提取所有请求ID")
    print("="*80)

    all_request_ids = set()
    arrive_count = 0
    leave_count = 0

    for slot in data:
        arrive = slot.get('arrive_event', [])
        leave = slot.get('leave_event', [])

        all_request_ids.update(arrive)
        arrive_count += len(arrive)
        leave_count += len(leave)

    print(f"\n统计:")
    print(f"  - 唯一请求ID数量: {len(all_request_ids)}")
    print(f"  - 到达事件总数: {arrive_count}")
    print(f"  - 离开事件总数: {leave_count}")
    print(f"  - 请求ID范围: {min(all_request_ids) if all_request_ids else 'N/A'} ~ {max(all_request_ids) if all_request_ids else 'N/A'}")

    # 检查请求文件
    if request_file:
        print(f"\n📂 尝试加载请求文件: {request_file}")
        try:
            with open(request_file, 'rb') as f:
                requests = pickle.load(f)
            print(f"✅ 成功加载 {len(requests)} 个请求")
            return requests, all_request_ids
        except Exception as e:
            print(f"❌ 加载请求文件失败: {e}")

    return None, all_request_ids

def analyze_requests(requests):
    """分析请求内容"""

    if not requests:
        print("\n⚠️ 未提供请求数据")
        return

    print("\n" + "="*80)
    print("📋 请求内容分析")
    print("="*80)

    # 示例请求
    if isinstance(requests, dict):
        req_ids = list(requests.keys())
        print(f"\n请求数量: {len(requests)}")
        print(f"请求ID示例: {req_ids[:5]}")

        first_req_id = req_ids[0]
        first_req = requests[first_req_id]
    elif isinstance(requests, list):
        print(f"\n请求数量: {len(requests)}")
        first_req = requests[0]
    else:
        print(f"未知的请求格式: {type(requests)}")
        return

    print(f"\n第一个请求内容:")
    for key, value in first_req.items():
        if isinstance(value, (list, tuple)) and len(value) > 10:
            print(f"  {key}: {type(value).__name__} (长度: {len(value)})")
        else:
            print(f"  {key}: {value}")

    # 提取节点信息
    all_sources = []
    all_dests = []
    all_nodes = set()

    request_list = requests.values() if isinstance(requests, dict) else requests

    for req in request_list:
        if isinstance(req, dict):
            if 'source' in req:
                source = req['source']
                all_sources.append(source)
                all_nodes.add(source)

            if 'dest' in req:
                dests = req['dest']
                if isinstance(dests, list):
                    all_dests.extend(dests)
                    all_nodes.update(dests)
                else:
                    all_dests.append(dests)
                    all_nodes.add(dests)

    print(f"\n" + "="*80)
    print("🔍 节点分析")
    print("="*80)

    print(f"\n源节点:")
    print(f"  - 总数: {len(all_sources)}")
    if all_sources:
        print(f"  - 范围: {min(all_sources)} ~ {max(all_sources)}")
        print(f"  - 唯一源节点: {sorted(set(all_sources))}")

    print(f"\n目的地节点:")
    print(f"  - 总数: {len(all_dests)}")
    if all_dests:
        print(f"  - 范围: {min(all_dests)} ~ {max(all_dests)}")
        unique_dests = set(all_dests)
        print(f"  - 唯一目的地数: {len(unique_dests)}")

    print(f"\n所有节点:")
    if all_nodes:
        print(f"  - 节点列表: {sorted(all_nodes)}")
        print(f"  - 节点数量: {len(all_nodes)}")
        print(f"  - 范围: {min(all_nodes)} ~ {max(all_nodes)}")

        # 索引检查
        min_node = min(all_nodes)
        max_node = max(all_nodes)

        print(f"\n" + "="*80)
        print("🔍 索引问题检查")
        print("="*80)

        if min_node == 0:
            print("\n✅ 索引从0开始（Python风格，0-based）")
            print(f"   - 环境应设置: n = {max_node + 1}")
            print(f"   - 节点范围: 0 ~ {max_node}")
        elif min_node == 1:
            print("\n⚠️ 索引从1开始（数学风格，1-based）")
            print(f"   - 需要将所有节点ID减1")
            print(f"   - 或环境支持1-based索引")
            print(f"\n修复建议：")
            print(f"   转换后的源节点: {sorted(set(s - 1 for s in all_sources))}")
            print(f"   转换后的节点范围: 0 ~ {max_node - 1}")
            print(f"   环境应设置: n = {max_node}")

        # 检查连续性
        expected_range = set(range(min_node, max_node + 1))
        missing = expected_range - all_nodes
        if missing:
            print(f"\n⚠️ 缺失的节点ID: {sorted(missing)}")
        else:
            print(f"\n✅ 节点ID连续")

        # DC节点推断
        print(f"\n" + "="*80)
        print("🏢 DC节点推断")
        print("="*80)

        unique_sources = sorted(set(all_sources))
        print(f"\n可能的DC节点（基于源节点）:")
        print(f"  原始: {unique_sources}")

        if min_node == 1:
            adjusted = [s - 1 for s in unique_sources]
            print(f"  转换为0-based: {adjusted}")

        # 检查节点16
        print(f"\n🔍 节点16检查:")
        if 16 in all_nodes:
            print(f"  ✅ 节点16在数据集中")
            if 16 in set(all_sources):
                print(f"  ✅ 节点16是源节点（可能是DC）")
            else:
                print(f"  ⚠️ 节点16不是源节点（可能不是DC）")
        else:
            print(f"  ❌ 节点16不在数据集中")

            if min_node == 1 and 17 in all_nodes:
                print(f"  💡 提示: 数据集是1-based，节点17在转换后是16")

def main():
    """主函数"""

    dataset_path = r"E:\pycharmworkspace\HRL-GNN for Multicast-aware SFC Orchestration\data\input_dir\phase3_events_by_slot.pkl"

    # 尝试查找请求文件
    import os
    data_dir = os.path.dirname(dataset_path)
    possible_request_files = [
        os.path.join(data_dir, 'phase3_requests.pkl'),
        os.path.join(data_dir, 'requests.pkl'),
        os.path.join(data_dir, 'all_requests.pkl'),
    ]

    print("="*80)
    print("📊 Phase3 数据集验证工具（修复版）")
    print("="*80)
    print(f"\n文件路径: {dataset_path}")

    # 1. 加载事件数据
    data = load_dataset(dataset_path)
    if data is None:
        return

    # 2. 分析事件结构
    analyze_event_structure(data)

    # 3. 提取请求ID
    request_file = None
    for f in possible_request_files:
        if os.path.exists(f):
            request_file = f
            break

    requests, request_ids = extract_all_requests(data, request_file)

    # 4. 分析请求（如果成功加载）
    if requests:
        analyze_requests(requests)
    else:
        print("\n" + "="*80)
        print("⚠️ 数据集说明")
        print("="*80)
        print(f"""
这个数据集包含时间槽事件，结构是:
- time_slot: 时间槽编号
- arrive_event: 到达的请求ID列表
- leave_event: 离开的请求ID列表

但请求的具体内容（source, dest, vnf等）应该在另一个文件中。

请查找以下文件:
{chr(10).join('  - ' + f for f in possible_request_files)}

或者告诉我请求文件的实际路径。
        """)

    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)

if __name__ == "__main__":
    main()