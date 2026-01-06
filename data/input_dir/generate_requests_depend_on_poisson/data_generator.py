"""
Data Generator with Time Slot Support (时间槽版本)

主要改动:
1. ✅ 添加时间槽配置 (delta_t)
2. ✅ 将到达时间转换为时间槽
3. ✅ 按时间槽分组请求
4. ✅ 添加请求持续时间（时间槽单位）
5. ✅ 保持原有的所有功能
"""

import numpy as np
import random
import math


# ============================================================================
# VNF 目录生成（保持不变）
# ============================================================================

def generate_vnfs_catalog(vnf_type_num=8):
    """
    生成 VNF 目录

    Args:
        vnf_type_num: VNF类型数量

    Returns:
        list: VNF列表，每个VNF包含 type, cpu_need, memory_need
    """
    all_vnf = []
    for vnf_type in range(1, vnf_type_num + 1):
        # CPU 系数范围 [0.25, 3.0]
        # 当带宽为 [4, 8] 时，CPU需求为 [1, 24]
        cpu_need = random.random() * 2.75 + 0.25

        # Memory 系数范围 [0.25, 2.0]
        # 当带宽为 [4, 8] 时，Mem需求为 [1, 16]
        memory_need = random.random() * 1.75 + 0.25

        vnf = {
            'type': vnf_type,
            'cpu_need': cpu_need,
            'memory_need': memory_need
        }
        all_vnf.append(vnf)

    return all_vnf


# ============================================================================
# 🔥 时间槽版本：泊松到达时间生成
# ============================================================================

def generate_poisson_arrive_time_list(T, lamda):
    """
    生成服从泊松过程的到达时间列表

    Args:
        T: 总时间（秒）
        lamda: 到达率（请求/秒）

    Returns:
        list: 到达时间列表（连续时间，单位：秒）

    Note:
        这里生成的是连续时间，稍后会转换为离散时间槽
    """
    time_state = 0
    arrive_time_list = []

    while time_state < T:
        # 泊松过程：指数分布的到达间隔
        interval = random.expovariate(lamda)
        t = time_state + interval

        if t < T:
            time_state = t
            arrive_time_list.append(t)
        else:
            break

    return arrive_time_list


# ============================================================================
# 🔥 时间槽版本：单个请求生成
# ============================================================================

def generate_single_request(req_id, source, dest, all_vnf,
                           max_bandwidth, min_bandwidth, arrive_time, mean_lifetime,
                           delta_t=0.01):
    """
    生成单个业务请求（时间槽版本）

    Args:
        req_id: 请求ID
        source: 源节点
        dest: 目的节点列表
        all_vnf: VNF目录
        max_bandwidth: 最大带宽
        min_bandwidth: 最小带宽
        arrive_time: 到达时间（秒）
        mean_lifetime: 平均持续时间
        delta_t: 时间槽大小（秒），默认 0.01 = 10ms

    Returns:
        dict: 请求字典
    """
    vnf_type_num = len(all_vnf)
    vnf_num = 3  # 固定选择3个VNF

    # 随机选择3个不同的VNF类型
    vnf_indices = random.sample(range(vnf_type_num), vnf_num)
    selected_vnfs = [all_vnf[i] for i in vnf_indices]
    vnf_types = [v['type'] for v in selected_vnfs]

    # 初始带宽
    bw_origin = random.randint(min_bandwidth, max_bandwidth)

    # 计算CPU和内存需求
    cpu_origin = []
    memory_origin = []

    for v in selected_vnfs:
        cpu = round(bw_origin * v['cpu_need'])
        mem = round(bw_origin * v['memory_need'])
        cpu_origin.append(cpu)
        memory_origin.append(mem)

    # 🔥 请求持续时间（原逻辑：1 + 指数分布，限制 ≤ 6）
    while True:
        lifetime = 1 + random.expovariate(1.0 / (mean_lifetime - 1))
        if lifetime <= 6:
            break

    leave_time = arrive_time + lifetime

    # 🔥 转换为时间槽
    time_slot = int(arrive_time / delta_t)
    leave_time_slot = int(leave_time / delta_t)
    duration = leave_time_slot - time_slot  # 持续时间（时间槽数）

    # 构建请求字典
    request = {
        # 基本信息
        'id': req_id,
        'source': source,
        'dest': dest,  # 列表形式的目的节点
        'vnf': vnf_types,

        # 资源需求
        'cpu_origin': cpu_origin,
        'memory_origin': memory_origin,
        'bw_origin': bw_origin,

        # 🔥 时间信息（连续时间，保留用于统计）
        'arrive_time': arrive_time,      # 原始到达时间（秒）
        'lifetime': lifetime,             # 持续时间（秒）
        'leave_time': leave_time,         # 离开时间（秒）

        # 🔥 时间槽信息（离散时间，用于仿真）
        'time_slot': time_slot,           # 到达时间槽
        'leave_time_slot': leave_time_slot,  # 离开时间槽
        'duration': duration,             # 持续时间（时间槽数）

        # 保留原有的时间步信息（兼容性）
        'arrive_time_step': math.ceil(arrive_time),
        'leave_time_step': math.ceil(leave_time)
    }

    return request


# ============================================================================
# 为特定源节点生成请求（保持不变，但会调用新版generate_single_request）
# ============================================================================

def generate_node_requests(source, node_important, arrive_time_list, all_vnf,
                          delta_t=0.01):
    """
    为特定源节点生成一系列请求

    Args:
        source: 源节点
        node_important: 重要节点列表
        arrive_time_list: 到达时间列表
        all_vnf: VNF目录
        delta_t: 时间槽大小（秒）

    Returns:
        list: 请求列表
    """
    node_requests = []

    # 候选目的节点：除了源节点以外的重要节点
    candidates = [n for n in node_important if n != source]

    # 请求参数
    max_bandwidth = 8   # 业务请求带宽资源需求量上限
    min_bandwidth = 4   # 业务请求带宽资源需求量下限
    multicast_num = 5   # 业务请求目的节点个数
    mean_lifetime = 3   # 指定分布均值

    for i, arrive_time in enumerate(arrive_time_list):
        # 随机选择多播目的节点
        k = min(multicast_num, len(candidates))
        dest = random.sample(candidates, k)

        # 🔥 生成请求（带时间槽信息）
        req = generate_single_request(
            req_id=0,  # ID 暂时设为0，后续统一重排
            source=source,
            dest=dest,
            all_vnf=all_vnf,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            arrive_time=arrive_time,
            mean_lifetime=mean_lifetime,
            delta_t=delta_t  # 🔥 传入时间槽参数
        )

        node_requests.append(req)

    return node_requests


# ============================================================================
# 🔥 新增：按时间槽分组请求
# ============================================================================

def group_requests_by_time_slot(requests):
    """
    将请求按时间槽分组

    Args:
        requests: 请求列表

    Returns:
        dict: {time_slot: [requests]}

    Example:
        {
            100: [req1, req2, req3],
            101: [req4],
            105: [req5, req6],
            ...
        }
    """
    grouped = {}

    for req in requests:
        slot = req['time_slot']

        if slot not in grouped:
            grouped[slot] = []

        grouped[slot].append(req)

    return grouped


# ============================================================================
# 🔥 新增：主生成器类（推荐使用）
# ============================================================================

class DataGenerator:
    """
    数据生成器（时间槽版本）

    用法:
        config = {
            'num_nodes': 28,
            'time_slot_delta': 0.01,  # 10ms
            'max_time_slots': 10000,  # 100秒
            'arrival_rate': 56,
            'vnf_type_num': 8,
        }

        generator = DataGenerator(config)
        requests, requests_by_slot = generator.generate_all_requests(
            num_requests=300,
            node_important=[1, 5, 10, 15, 20, 25]
        )
    """

    def __init__(self, config):
        """
        初始化生成器

        Args:
            config: 配置字典
        """
        self.num_nodes = config.get('num_nodes', 28)
        self.delta_t = config.get('time_slot_delta', 0.01)  # 时间槽大小（秒）
        self.max_time_slots = config.get('max_time_slots', 10000)
        self.arrival_rate = config.get('arrival_rate', 56)  # 请求/秒
        self.vnf_type_num = config.get('vnf_type_num', 8)

        # 生成VNF目录
        self.all_vnf = generate_vnfs_catalog(self.vnf_type_num)

        print(f"✅ DataGenerator 初始化完成")
        print(f"   时间槽大小: {self.delta_t * 1000:.1f} ms")
        print(f"   最大时间槽数: {self.max_time_slots}")
        print(f"   到达率: {self.arrival_rate} req/s")
        print(f"   VNF类型数: {self.vnf_type_num}")

    def generate_all_requests(self, num_requests, node_important):
        """
        生成所有请求

        Args:
            num_requests: 总请求数
            node_important: 重要节点列表（用作源节点和目的节点）

        Returns:
            tuple: (requests, requests_by_slot)
                - requests: 所有请求列表
                - requests_by_slot: 按时间槽分组的请求字典
        """
        print(f"\n🔄 开始生成 {num_requests} 个请求...")

        all_requests = []

        # 计算总仿真时间（秒）
        T = self.max_time_slots * self.delta_t

        # 平均每个源节点的请求数
        num_sources = len(node_important)
        requests_per_source = num_requests // num_sources

        for source in node_important:
            # 生成该源节点的到达时间列表
            arrive_time_list = generate_poisson_arrive_time_list(
                T,
                self.arrival_rate / num_sources  # 分配到达率
            )

            # 限制请求数量
            arrive_time_list = arrive_time_list[:requests_per_source]

            # 生成该源节点的所有请求
            node_requests = generate_node_requests(
                source=source,
                node_important=node_important,
                arrive_time_list=arrive_time_list,
                all_vnf=self.all_vnf,
                delta_t=self.delta_t  # 🔥 传入时间槽参数
            )

            all_requests.extend(node_requests)

        # 限制总请求数
        all_requests = all_requests[:num_requests]

        # 🔥 重新分配请求ID（按到达时间排序）
        all_requests.sort(key=lambda x: x['arrive_time'])
        for i, req in enumerate(all_requests):
            req['id'] = i

        # 🔥 按时间槽分组
        requests_by_slot = group_requests_by_time_slot(all_requests)

        # 打印统计信息
        self._print_statistics(all_requests, requests_by_slot)

        return all_requests, requests_by_slot

    def _print_statistics(self, requests, requests_by_slot):
        """
        打印统计信息
        """
        print(f"\n{'='*60}")
        print(f"📊 请求生成统计")
        print(f"{'='*60}")

        print(f"总请求数: {len(requests)}")
        print(f"时间槽数: {len(requests_by_slot)}")

        # 时间槽范围
        min_slot = min(requests_by_slot.keys())
        max_slot = max(requests_by_slot.keys())
        print(f"时间槽范围: {min_slot} - {max_slot}")
        print(f"实际时间范围: {min_slot * self.delta_t:.2f}s - {max_slot * self.delta_t:.2f}s")

        # 每个时间槽的请求数
        slot_counts = [len(reqs) for reqs in requests_by_slot.values()]
        avg_per_slot = sum(slot_counts) / len(slot_counts)
        max_per_slot = max(slot_counts)

        print(f"平均每时间槽: {avg_per_slot:.2f} 个请求")
        print(f"最大每时间槽: {max_per_slot} 个请求")

        # 请求持续时间统计
        durations = [req['duration'] for req in requests]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        print(f"\n持续时间统计（时间槽）:")
        print(f"  平均: {avg_duration:.1f} ({avg_duration * self.delta_t:.3f}s)")
        print(f"  最小: {min_duration} ({min_duration * self.delta_t:.3f}s)")
        print(f"  最大: {max_duration} ({max_duration * self.delta_t:.3f}s)")

        # 资源需求统计
        all_bw = [req['bw_origin'] for req in requests]
        print(f"\n带宽需求:")
        print(f"  范围: {min(all_bw)} - {max(all_bw)}")
        print(f"  平均: {sum(all_bw) / len(all_bw):.1f}")

        print(f"{'='*60}\n")


# ============================================================================
# 🔥 新增：便捷函数（快速生成）
# ============================================================================

def quick_generate(num_requests=300,
                   num_nodes=28,
                   arrival_rate=56,
                   delta_t=0.01):
    """
    快速生成请求（便捷函数）

    Args:
        num_requests: 总请求数
        num_nodes: 节点数
        arrival_rate: 到达率（请求/秒）
        delta_t: 时间槽大小（秒）

    Returns:
        tuple: (requests, requests_by_slot, vnf_catalog)

    Example:
        requests, requests_by_slot, vnf_catalog = quick_generate(
            num_requests=300,
            num_nodes=28,
            arrival_rate=56,
            delta_t=0.01
        )
    """
    config = {
        'num_nodes': num_nodes,
        'time_slot_delta': delta_t,
        'max_time_slots': 10000,
        'arrival_rate': arrival_rate,
        'vnf_type_num': 8,
    }

    generator = DataGenerator(config)

    # 默认使用部分节点作为重要节点
    node_important = list(range(0, num_nodes, num_nodes // 6))[:6]

    requests, requests_by_slot = generator.generate_all_requests(
        num_requests=num_requests,
        node_important=node_important
    )

    return requests, requests_by_slot, generator.all_vnf


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("测试 DataGenerator（时间槽版本）")
    print("="*60)

    # 方法1：使用类（推荐）
    print("\n【方法1：使用 DataGenerator 类】")

    config = {
        'num_nodes': 28,
        'time_slot_delta': 0.01,  # 10ms
        'max_time_slots': 10000,  # 100秒
        'arrival_rate': 56,
        'vnf_type_num': 8,
    }

    generator = DataGenerator(config)

    node_important = [1, 5, 10, 15, 20, 25]

    requests, requests_by_slot = generator.generate_all_requests(
        num_requests=300,
        node_important=node_important
    )

    # 检查结果
    print(f"\n✅ 生成完成:")
    print(f"   总请求数: {len(requests)}")
    print(f"   时间槽数: {len(requests_by_slot)}")

    # 查看第一个时间槽
    first_slot = min(requests_by_slot.keys())
    first_slot_requests = requests_by_slot[first_slot]
    print(f"\n📋 第一个时间槽 ({first_slot}) 的请求:")
    for req in first_slot_requests[:3]:  # 只显示前3个
        print(f"   Request {req['id']}: "
              f"Src={req['source']}, "
              f"Dests={req['dest']}, "
              f"BW={req['bw_origin']}, "
              f"Duration={req['duration']} slots")

    # 方法2：使用便捷函数
    print("\n\n【方法2：使用 quick_generate 函数】")

    requests2, requests_by_slot2, vnf_catalog = quick_generate(
        num_requests=100,
        num_nodes=28,
        arrival_rate=56,
        delta_t=0.01
    )

    print(f"\n✅ 快速生成完成:")
    print(f"   总请求数: {len(requests2)}")
    print(f"   时间槽数: {len(requests_by_slot2)}")
    print(f"   VNF类型数: {len(vnf_catalog)}")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)