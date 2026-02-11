"""
🌳 SFC环境可视化与统计工具
=============================
功能：
1. 🌳 多播树可视化（图形）
2. 📊 资源统计（纯文本）
3. 📈 请求处理统计
4. 🎯 性能监控
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import os
import matplotlib.patches as mpatches
import timedelta


class SFCVisualizer:
    """
    🌳 SFC环境可视化器 - 专注于多播树可视化
    ------------------------------------
    功能：
    1. 多播树结构可视化
    2. VNF部署位置标记
    3. SFC路径完整性验证
    4. 支持保存为图片
    """

    def __init__(self, env):
        """
        初始化可视化器

        Args:
            env: SFC环境实例
        """
        self.env = env
        self._check_compatibility()

    def _check_compatibility(self):
        """检查环境兼容性"""
        required = ['current_request', 'current_tree', 'n', 'topo']
        missing = [attr for attr in required if not hasattr(self.env, attr)]

        if missing:
            raise AttributeError(f"环境缺少必要属性: {missing}")

    def visualize_tree(self, save_path=None, show=True, figsize=(16, 12)):
        """
        可视化多播树

        Args:
            save_path: 保存路径
            show: 是否显示
            figsize: 图像尺寸

        Returns:
            fig, ax: matplotlib对象
        """
        if self.env.current_request is None:
            print("⚠️ 没有活跃请求可可视化")
            return None, None

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 构建网络图
        G = self._build_network_graph()
        pos = self._get_layout(G)

        # 获取请求信息
        request = self.env.current_request
        tree_edges = self.env.current_tree.get('tree', {})
        placement = self.env.current_tree.get('placement', {})

        # 验证SFC路径
        dest_status = self._validate_sfc_paths(request, tree_edges, placement)

        # 1. 绘制底层拓扑
        nx.draw_networkx_edges(
            G, pos,
            edge_color='lightgray',
            width=1.0,
            alpha=0.3,
            ax=ax
        )

        # 2. 绘制多播树
        if tree_edges:
            tree_edge_list = [(u, v) for (u, v) in tree_edges.keys()]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=tree_edge_list,
                edge_color='blue',
                width=4.0,
                alpha=0.8,
                ax=ax
            )

        # 3. 绘制节点
        self._draw_nodes(G, pos, request, dest_status, placement, ax)

        # 4. 添加边标签
        if tree_edges:
            self._add_edge_labels(G, pos, tree_edges, ax)

        # 5. 添加标题和图例
        self._add_title_and_legend(request, dest_status, tree_edges, ax)

        # 美化
        ax.axis('off')
        plt.tight_layout()

        # 保存
        if save_path:
            self._save_figure(fig, save_path, dest_status)

        # 显示
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def _build_network_graph(self):
        """构建网络图"""
        G = nx.Graph()
        n = self.env.n

        G.add_nodes_from(range(n))
        topo = self.env.topo

        for i in range(n):
            for j in range(i + 1, n):
                if topo[i, j] > 0:
                    G.add_edge(i, j, weight=topo[i, j])

        return G

    def _get_layout(self, G):
        """获取布局"""
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # DC节点向中心移动
        if hasattr(self.env, 'dc_nodes') and self.env.dc_nodes:
            for node in self.env.dc_nodes:
                if node in pos:
                    pos[node] = pos[node] * 0.8

        return pos

    def _validate_sfc_paths(self, request, tree_edges, placement):
        """验证SFC路径完整性"""
        source = request.get('source')
        dests = request.get('dest', [])
        vnf_chain = request.get('vnf', [])

        # 构建邻接表
        adj = defaultdict(list)
        for (u, v) in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS
        parent = {source: None}
        queue = deque([source])
        visited = {source}

        while queue:
            curr = queue.popleft()
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        # 节点VNF映射
        node_vnf_dict = {}
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node, vnf_type = key[0], key[1]
                if node not in node_vnf_dict:
                    node_vnf_dict[node] = []
                node_vnf_dict[node].append(vnf_type)

        # 检查每个目的地
        dest_status = {}

        for dest in dests:
            # 回溯路径
            path = []
            curr = dest
            while curr is not None:
                path.append(curr)
                if curr == source:
                    break
                curr = parent.get(curr)

            if not path or path[-1] != source:
                dest_status[dest] = {'complete': False, 'error': 'path_broken'}
                continue

            path.reverse()

            # 收集VNF
            path_vnfs = []
            for node in path:
                if node in node_vnf_dict:
                    path_vnfs.extend(node_vnf_dict[node])

            # 检查完整性
            required_set = set(vnf_chain)
            collected_set = set(path_vnfs)
            missing = required_set - collected_set

            # 检查顺序
            vnf_order_correct = True
            if not missing:
                vnf_indices = []
                for vnf in path_vnfs:
                    if vnf in vnf_chain:
                        vnf_indices.append(vnf_chain.index(vnf))
                if vnf_indices != sorted(vnf_indices):
                    vnf_order_correct = False

            dest_status[dest] = {
                'complete': (len(missing) == 0 and vnf_order_correct),
                'path': path,
                'vnfs': path_vnfs,
                'missing': list(missing),
                'order_correct': vnf_order_correct
            }

        return dest_status

    def _draw_nodes(self, G, pos, request, dest_status, placement, ax):
        """绘制节点"""
        source = request.get('source')
        dests = request.get('dest', [])

        # VNF节点映射
        vnf_nodes = {}
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node, vnf_type = key[0], key[1]
                if node not in vnf_nodes:
                    vnf_nodes[node] = []
                vnf_nodes[node].append(vnf_type)

        # 节点属性
        node_colors = []
        node_sizes = []
        node_labels = {}

        # 树上节点
        nodes_on_tree = getattr(self.env, 'nodes_on_tree', set())

        for node in G.nodes():
            if node == source:
                node_colors.append('limegreen')
                node_sizes.append(1500)
                node_labels[node] = f"S{node}"

            elif node in dests:
                status = dest_status.get(node, {})
                if status.get('complete', False):
                    node_colors.append('royalblue')
                    node_sizes.append(1200)
                    node_labels[node] = f"D{node}[OK]"
                else:
                    node_colors.append('red')
                    node_sizes.append(1200)
                    error = status.get('error', '')
                    if error == 'path_broken':
                        node_labels[node] = f"D{node}[X]"
                    else:
                        node_labels[node] = f"D{node}[!]"

            elif node in vnf_nodes:
                node_colors.append('salmon')
                node_sizes.append(1000)
                vnf_list = vnf_nodes[node]
                node_labels[node] = f"{node}\nV{vnf_list}"

            elif node in nodes_on_tree:
                node_colors.append('lightblue')
                node_sizes.append(800)
                node_labels[node] = str(node)

            else:
                node_colors.append('white')
                node_sizes.append(600)
                node_labels[node] = str(node)

        # 绘制
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

    def _add_edge_labels(self, G, pos, tree_edges, ax):
        """添加边标签"""
        edge_labels = {}
        for edge_key, bw in tree_edges.items():
            edge_labels[edge_key] = f"{bw:.1f}"

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='darkblue',
            ax=ax
        )

    def _add_title_and_legend(self, request, dest_status, tree_edges, ax):
        """添加标题和图例"""
        req_id = request.get('id', 'unknown')
        source = request.get('source')
        dests = request.get('dest', [])
        vnf_chain = request.get('vnf', [])

        connected = self.env.current_tree.get('connected_dests', set())
        complete_count = sum(1 for info in dest_status.values() if info.get('complete', False))

        # 标题
        title = [
            f"Request {req_id} Multicast Tree",
            f"Source: {source} | Dests: {dests} | VNF: {vnf_chain}",
            f"Connected: {len(connected)}/{len(dests)} | Tree Edges: {len(tree_edges)}",
        ]

        if vnf_chain:
            status = "All OK" if complete_count == len(dests) else f"{len(dests)-complete_count} Failed"
            title.append(f"VNF Complete: {complete_count}/{len(dests)} ({status})")

        ax.set_title("\n".join(title), fontsize=13, fontweight='bold', pad=20)

        # 图例
        legend_elements = [
            mpatches.Patch(color='limegreen', label='Source'),
            mpatches.Patch(color='royalblue', label='Dest (OK)'),
            mpatches.Patch(color='red', label='Dest (Failed)'),
            mpatches.Patch(color='salmon', label='VNF Node'),
            mpatches.Patch(color='lightblue', label='Tree Node'),
            mpatches.Patch(color='white', edgecolor='black', label='Normal'),
            mpatches.Patch(color='blue', label='Tree Edge'),
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10
        )

    def _save_figure(self, fig, save_path, dest_status):
        """保存图像"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        req_id = self.env.current_request.get('id', 'unknown')
        complete_count = sum(1 for info in dest_status.values() if info.get('complete', False))

        print(f"🌳 多播树可视化已保存: {save_path}")
        print(f"   Request {req_id}: {complete_count}/{len(self.env.current_request.get('dest', []))} destinations OK")


class SFCStatsMonitor:
    """
    📊 SFC环境统计监控器 - 纯文本版
    -----------------------------
    功能：
    1. 资源统计（剩余量、利用率）
    2. 请求统计（接受率、成功率）
    3. 性能监控（延迟、吞吐量）
    4. 文本报表输出
    """

    def __init__(self, env, history_size: int = 100):
        """
        初始化监控器

        Args:
            env: SFC环境实例
            history_size: 历史记录大小
        """
        self.env = env
        self.history_size = history_size
        self.start_time = datetime.now()

        # 统计数据
        self.stats = {
            # 请求统计
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,

            # 失败原因统计
            'fail_reasons': defaultdict(int),

            # 资源统计
            'cpu_utilization_history': deque(maxlen=history_size),
            'memory_utilization_history': deque(maxlen=history_size),
            'bandwidth_utilization_history': deque(maxlen=history_size),

            # 性能统计
            'processing_times': deque(maxlen=history_size),
            'tree_sizes': deque(maxlen=history_size),
            'vnf_counts': deque(maxlen=history_size),

            # 时间窗口
            'request_timestamps': deque(maxlen=1000),
        }

    def log_request(self, request, accepted: bool, success: bool = None,
                   fail_reason: str = None, processing_time: float = None):
        """
        记录请求处理结果

        Args:
            request: 请求对象
            accepted: 是否被接受
            success: 是否成功完成（如果被接受）
            fail_reason: 失败原因
            processing_time: 处理时间（秒）
        """
        self.stats['total_requests'] += 1

        if accepted:
            self.stats['accepted_requests'] += 1

            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
                if fail_reason:
                    self.stats['fail_reasons'][fail_reason] += 1
        else:
            self.stats['rejected_requests'] += 1
            if fail_reason:
                self.stats['fail_reasons'][fail_reason] += 1

        # 记录时间戳
        self.stats['request_timestamps'].append(datetime.now())

        # 记录处理时间
        if processing_time is not None:
            self.stats['processing_times'].append(processing_time)

        # 记录树大小和VNF数量
        if hasattr(self.env, 'current_tree') and self.env.current_tree:
            tree = self.env.current_tree.get('tree', {})
            placement = self.env.current_tree.get('placement', {})
            self.stats['tree_sizes'].append(len(tree))
            self.stats['vnf_counts'].append(len(placement))

    def update_resource_stats(self):
        """更新资源统计信息"""
        if not hasattr(self.env, 'resource_mgr'):
            return

        try:
            n = self.env.n if hasattr(self.env, 'n') else 100

            # CPU使用率
            cpu_usage = self._calculate_cpu_utilization()
            if cpu_usage is not None:
                self.stats['cpu_utilization_history'].append(cpu_usage)

            # 内存使用率
            mem_usage = self._calculate_memory_utilization()
            if mem_usage is not None:
                self.stats['memory_utilization_history'].append(mem_usage)

            # 带宽使用率
            bw_usage = self._calculate_bandwidth_utilization()
            if bw_usage is not None:
                self.stats['bandwidth_utilization_history'].append(bw_usage)

        except Exception as e:
            print(f"⚠️ 资源统计更新失败: {e}")

    def _calculate_cpu_utilization(self):
        """计算CPU使用率"""
        try:
            rm = self.env.resource_mgr
            total_capacity = 0
            total_used = 0

            for node in range(self.env.n):
                # 尝试多种方式获取CPU信息
                if hasattr(rm, 'get_node_cpu_info'):
                    capacity, used = rm.get_node_cpu_info(node)
                    total_capacity += capacity
                    total_used += used
                elif hasattr(rm, 'pool'):
                    pool = rm.pool
                    if hasattr(pool, 'cpu_capacity'):
                        capacity = pool.cpu_capacity[node]
                        avail = pool.get_available_cpu(node)
                        total_capacity += capacity
                        total_used += (capacity - avail)

            if total_capacity > 0:
                return (total_used / total_capacity) * 100
            return 0.0

        except:
            return None

    def _calculate_memory_utilization(self):
        """计算内存使用率"""
        try:
            rm = self.env.resource_mgr
            total_capacity = 0
            total_used = 0

            for node in range(self.env.n):
                if hasattr(rm, 'get_node_memory_info'):
                    capacity, used = rm.get_node_memory_info(node)
                    total_capacity += capacity
                    total_used += used
                elif hasattr(rm, 'pool'):
                    pool = rm.pool
                    if hasattr(pool, 'memory_capacity'):
                        capacity = pool.memory_capacity[node]
                        avail = pool.get_available_memory(node)
                        total_capacity += capacity
                        total_used += (capacity - avail)

            if total_capacity > 0:
                return (total_used / total_capacity) * 100
            return 0.0

        except:
            return None

    def _calculate_bandwidth_utilization(self):
        """计算带宽使用率"""
        try:
            rm = self.env.resource_mgr
            total_capacity = 0
            total_used = 0

            if hasattr(rm, 'links'):
                for (u, v), link_info in rm.links.items():
                    if isinstance(link_info, dict):
                        capacity = link_info.get('bandwidth_capacity', 100)
                        used = capacity - link_info.get('bandwidth', capacity)
                        total_capacity += capacity
                        total_used += used

            if total_capacity > 0:
                return (total_used / total_capacity) * 100
            return 0.0

        except:
            return None

    def get_request_acceptance_rate(self):
        """计算请求接受率"""
        if self.stats['total_requests'] == 0:
            return 0.0
        return (self.stats['accepted_requests'] / self.stats['total_requests']) * 100

    def get_request_success_rate(self):
        """计算请求成功率（仅被接受的请求）"""
        if self.stats['accepted_requests'] == 0:
            return 0.0
        return (self.stats['successful_requests'] / self.stats['accepted_requests']) * 100

    def get_overall_success_rate(self):
        """计算总体成功率（所有请求）"""
        if self.stats['total_requests'] == 0:
            return 0.0
        return (self.stats['successful_requests'] / self.stats['total_requests']) * 100

    def get_avg_processing_time(self):
        """计算平均处理时间"""
        times = list(self.stats['processing_times'])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_current_resource_utilization(self):
        """获取当前资源利用率"""
        cpu = list(self.stats['cpu_utilization_history'])[-1] if self.stats['cpu_utilization_history'] else 0.0
        mem = list(self.stats['memory_utilization_history'])[-1] if self.stats['memory_utilization_history'] else 0.0
        bw = list(self.stats['bandwidth_utilization_history'])[-1] if self.stats['bandwidth_utilization_history'] else 0.0

        return {
            'cpu_utilization': cpu,
            'memory_utilization': mem,
            'bandwidth_utilization': bw
        }

    def get_resource_availability(self):
        """获取资源可用量"""
        try:
            if not hasattr(self.env, 'resource_mgr'):
                return None

            rm = self.env.resource_mgr
            n = self.env.n

            # CPU
            total_cpu = 0
            avail_cpu = 0
            for node in range(n):
                if hasattr(rm.pool, 'get_available_cpu'):
                    avail = rm.pool.get_available_cpu(node)
                    total_cpu += 100  # 假设每个节点100单位
                    avail_cpu += avail

            # 内存
            total_mem = 0
            avail_mem = 0
            for node in range(n):
                if hasattr(rm.pool, 'get_available_memory'):
                    avail = rm.pool.get_available_memory(node)
                    total_mem += 100  # 假设每个节点100单位
                    avail_mem += avail

            # 带宽（估算）
            total_bw = n * 10 * 100  # 粗略估算
            used_bw = 0
            if hasattr(self.env, 'request_manager'):
                for req_id, req_info in self.env.request_manager.active_requests.items():
                    resources = req_info.get('resources', {})
                    tree = resources.get('tree', {})
                    for edge, bw in tree.items():
                        used_bw += bw

            avail_bw = max(0, total_bw - used_bw)

            return {
                'cpu': {'total': total_cpu, 'available': avail_cpu, 'used': total_cpu - avail_cpu},
                'memory': {'total': total_mem, 'available': avail_mem, 'used': total_mem - avail_mem},
                'bandwidth': {'total': total_bw, 'available': avail_bw, 'used': total_bw - avail_bw}
            }

        except Exception as e:
            print(f"⚠️ 资源可用性计算失败: {e}")
            return None

    def get_throughput(self, window_seconds: int = 60):
        """计算吞吐量（请求/分钟）"""
        now = datetime.now()
        timestamps = list(self.stats['request_timestamps'])

        if not timestamps:
            return 0.0

        # 统计时间窗口内的请求数
        recent_requests = 0
        cutoff_time = now - timedelta(seconds=window_seconds)

        for ts in reversed(timestamps):  # 从最新开始
            if ts < cutoff_time:
                break
            recent_requests += 1

        # 转换为每分钟
        return (recent_requests / window_seconds) * 60

    def generate_report(self, detailed: bool = False):
        """
        生成统计报告

        Args:
            detailed: 是否生成详细报告

        Returns:
            str: 报告文本
        """
        from datetime import timedelta

        # 基础统计
        total = self.stats['total_requests']
        accepted = self.stats['accepted_requests']
        successful = self.stats['successful_requests']

        accept_rate = self.get_request_acceptance_rate()
        success_rate = self.get_request_success_rate()
        overall_rate = self.get_overall_success_rate()

        # 运行时间
        run_time = datetime.now() - self.start_time
        hours, remainder = divmod(run_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # 构建报告
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 SFC环境统计报告")
        report_lines.append("=" * 60)
        report_lines.append(f"运行时间: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        report_lines.append(f"总请求数: {total}")
        report_lines.append("")

        # 请求统计
        report_lines.append("📈 请求统计:")
        report_lines.append(f"  接受率:  {accept_rate:.1f}% ({accepted}/{total})")
        report_lines.append(f"  成功率:  {success_rate:.1f}% ({successful}/{accepted})")
        report_lines.append(f"  总体成功率: {overall_rate:.1f}% ({successful}/{total})")
        report_lines.append("")

        # 资源统计
        resource_avail = self.get_resource_availability()
        if resource_avail:
            report_lines.append("💾 资源可用性:")

            # CPU
            cpu = resource_avail['cpu']
            cpu_util = (cpu['used'] / cpu['total']) * 100 if cpu['total'] > 0 else 0
            report_lines.append(f"  CPU:     {cpu['available']:.0f}/{cpu['total']:.0f} ({cpu_util:.1f}% used)")

            # 内存
            mem = resource_avail['memory']
            mem_util = (mem['used'] / mem['total']) * 100 if mem['total'] > 0 else 0
            report_lines.append(f"  内存:    {mem['available']:.0f}/{mem['total']:.0f} ({mem_util:.1f}% used)")

            # 带宽
            bw = resource_avail['bandwidth']
            bw_util = (bw['used'] / bw['total']) * 100 if bw['total'] > 0 else 0
            report_lines.append(f"  带宽:    {bw['available']:.0f}/{bw['total']:.0f} ({bw_util:.1f}% used)")
            report_lines.append("")

        # 性能统计
        avg_time = self.get_avg_processing_time()
        throughput = self.get_throughput()

        report_lines.append("⚡ 性能指标:")
        report_lines.append(f"  平均处理时间: {avg_time:.2f}s")
        report_lines.append(f"  当前吞吐量:  {throughput:.1f} req/min")

        if self.stats['tree_sizes']:
            avg_tree = sum(self.stats['tree_sizes']) / len(self.stats['tree_sizes'])
            report_lines.append(f"  平均树大小:  {avg_tree:.1f} edges")

        if self.stats['vnf_counts']:
            avg_vnf = sum(self.stats['vnf_counts']) / len(self.stats['vnf_counts'])
            report_lines.append(f"  平均VNF数:   {avg_vnf:.1f}")
        report_lines.append("")

        # 详细报告
        if detailed and self.stats['fail_reasons']:
            report_lines.append("🔍 失败原因分析:")
            for reason, count in sorted(self.stats['fail_reasons'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                report_lines.append(f"  {reason}: {count} ({percentage:.1f}%)")
            report_lines.append("")

        # 资源利用率历史
        if detailed and self.stats['cpu_utilization_history']:
            report_lines.append("📈 资源利用率历史:")
            cpu_hist = list(self.stats['cpu_utilization_history'])
            mem_hist = list(self.stats['memory_utilization_history'])
            bw_hist = list(self.stats['bandwidth_utilization_history'])

            if cpu_hist:
                report_lines.append(f"  CPU:     avg={np.mean(cpu_hist):.1f}%, max={np.max(cpu_hist):.1f}%")
            if mem_hist:
                report_lines.append(f"  内存:    avg={np.mean(mem_hist):.1f}%, max={np.max(mem_hist):.1f}%")
            if bw_hist:
                report_lines.append(f"  带宽:    avg={np.mean(bw_hist):.1f}%, max={np.max(bw_hist):.1f}%")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def print_report(self, detailed: bool = False):
        """打印统计报告"""
        report = self.generate_report(detailed)
        print(report)

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'accepted_requests': 0,
            'rejected_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fail_reasons': defaultdict(int),
            'cpu_utilization_history': deque(maxlen=self.history_size),
            'memory_utilization_history': deque(maxlen=self.history_size),
            'bandwidth_utilization_history': deque(maxlen=self.history_size),
            'processing_times': deque(maxlen=self.history_size),
            'tree_sizes': deque(maxlen=self.history_size),
            'vnf_counts': deque(maxlen=self.history_size),
            'request_timestamps': deque(maxlen=1000),
        }
        self.start_time = datetime.now()


# ============================================================================
# 使用示例
# ============================================================================

def demo_usage(env):
    """
    演示如何使用可视化器和统计监控器
    """
    print("\n🚀 SFC可视化与统计工具演示")
    print("="*50)

    # 1. 多播树可视化
    print("\n🌳 多播树可视化:")
    try:
        viz = SFCVisualizer(env)
        viz.visualize_tree(
            save_path='multicast_tree.png',
            show=False
        )
        print("✅ 多播树可视化已保存")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

    # 2. 统计监控
    print("\n📊 统计监控演示:")
    stats = SFCStatsMonitor(env)

    # 模拟一些统计数据
    stats.log_request({'id': 'test1'}, accepted=True, success=True, processing_time=1.5)
    stats.log_request({'id': 'test2'}, accepted=True, success=False, fail_reason='resource_insufficient', processing_time=2.1)
    stats.log_request({'id': 'test3'}, accepted=False, fail_reason='bandwidth_unavailable')

    # 更新资源统计
    stats.update_resource_stats()

    # 生成报告
    stats.print_report(detailed=True)

    print("\n✨ 演示完成!")
    print("="*50)


def training_monitor(env, episode_num, success_info=None):
    """
    训练监控器 - 在训练过程中调用

    Args:
        env: 环境实例
        episode_num: episode编号
        success_info: 成功信息
    """
    # 保存路径
    save_dir = f'training_logs'
    os.makedirs(save_dir, exist_ok=True)

    # 每10个episode保存一次多播树
    if episode_num % 10 == 0 and success_info and success_info.get('episode_complete', False):
        print(f"\n🌳 Episode {episode_num}: 保存多播树")

        viz = SFCVisualizer(env)
        viz.visualize_tree(
            save_path=f'{save_dir}/episode_{episode_num:04d}_tree.png',
            show=False
        )

    # 每50个episode打印一次统计
    if episode_num % 50 == 0:
        # 这里需要你有一个全局的stats_monitor实例
        # 例如: global_stats.print_report()
        pass

