"""
多播树可视化工具
用于调试和展示 SFC 环境中构建的多播树结构
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


class MulticastTreeVisualizer:
    """
    多播树可视化器

    功能：
    1. 展示网络拓扑
    2. 高亮多播树的边
    3. 标记 VNF 部署位置
    4. 标记源节点和目的地
    """

    def __init__(self, env):
        """
        Args:
            env: SFC 环境实例
        """
        self.env = env
        self.fig = None
        self.ax = None

    def visualize_request_tree(self, request=None, save_path=None, show=True):
        """
        可视化当前请求的多播树（ASCII 友好版 - 无字体警告）

        Args:
            request: 请求对象（None表示使用当前请求）
            save_path: 保存路径（None表示不保存）
            show: 是否显示图像
        """
        if request is None:
            request = self.env.current_request

        if request is None:
            print("WARNING: No active request to visualize")
            return

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(16, 12))

        # 构建网络图
        G = self._build_network_graph()

        # 获取布局
        pos = self._get_layout(G)

        # 获取请求信息
        req_id = request.get('id', 'unknown')
        source = request.get('source')
        dests = request.get('dest', [])
        vnf_chain = request.get('vnf', [])

        # 获取树信息
        tree_edges = self.env.current_tree.get('tree', {})
        placement = self.env.current_tree.get('placement', {})
        connected_dests = self.env.current_tree.get('connected_dests', set())

        # =========================================================================
        # 验证每个目的地的 SFC 路径完整性
        # =========================================================================

        # 构建邻接表
        adj = defaultdict(list)
        for (u, v) in tree_edges.keys():
            adj[u].append(v)
            adj[v].append(u)

        # BFS 构建父节点映射
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

        # 构建节点 VNF 映射
        node_vnf_dict = {}
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                n, v = key[0], key[1]
                if n not in node_vnf_dict:
                    node_vnf_dict[n] = []
                node_vnf_dict[n].append(v)

        # 检查每个目的地的 VNF 完整性
        dest_vnf_status = {}

        print(f"\n[Visualization Check] Validating SFC paths...")
        print(f"   Required VNF chain: {vnf_chain}")

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
                dest_vnf_status[dest] = {
                    'complete': False,
                    'path': [],
                    'vnfs': [],
                    'missing': vnf_chain,
                    'error': 'path_broken'
                }
                print(f"   [X] Dest {dest}: Path broken")
                continue

            path.reverse()

            # 收集路径上的 VNF
            path_vnfs = []
            for node in path:
                if node in node_vnf_dict:
                    for vnf in node_vnf_dict[node]:
                        path_vnfs.append(vnf)

            # 检查是否包含所有必需的 VNF
            required_set = set(vnf_chain)
            collected_set = set(path_vnfs)
            missing = required_set - collected_set

            # 检查 VNF 顺序
            vnf_order_correct = True
            if not missing:
                vnf_indices = []
                for vnf in path_vnfs:
                    if vnf in vnf_chain:
                        vnf_indices.append(vnf_chain.index(vnf))

                if vnf_indices != sorted(vnf_indices):
                    vnf_order_correct = False

            is_complete = (len(missing) == 0 and vnf_order_correct)

            dest_vnf_status[dest] = {
                'complete': is_complete,
                'path': path,
                'vnfs': path_vnfs,
                'missing': list(missing),
                'order_correct': vnf_order_correct
            }

            if is_complete:
                print(f"   [OK] Dest {dest}: VNF complete {path_vnfs}")
            else:
                if missing:
                    print(f"   [X] Dest {dest}: Missing VNF {list(missing)} (path: {path_vnfs})")
                elif not vnf_order_correct:
                    print(f"   [X] Dest {dest}: Wrong VNF order (expected: {vnf_chain}, got: {path_vnfs})")

        # 统计完整性
        complete_count = sum(1 for info in dest_vnf_status.values() if info['complete'])

        # =========================================================================
        # 1. 绘制底层拓扑（灰色）
        # =========================================================================
        nx.draw_networkx_edges(
            G, pos,
            edge_color='lightgray',
            width=1.0,
            alpha=0.3,
            ax=self.ax
        )

        # =========================================================================
        # 2. 绘制多播树的边（蓝色，加粗）
        # =========================================================================
        if tree_edges:
            tree_edge_list = [(u, v) for (u, v) in tree_edges.keys()]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=tree_edge_list,
                edge_color='blue',
                width=4.0,
                alpha=0.8,
                ax=self.ax
            )

        # =========================================================================
        # 3. 准备节点颜色和标签（使用 ASCII 字符）
        # =========================================================================
        node_colors = []
        node_sizes = []
        node_labels = {}

        # 收集部署了 VNF 的节点
        vnf_nodes = {}
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                node = key[0]
                vnf_type = key[1]
                if node not in vnf_nodes:
                    vnf_nodes[node] = []
                vnf_nodes[node].append(vnf_type)

        # 为每个节点分配颜色
        for node in G.nodes():
            if node == source:
                # 源节点 - 绿色
                node_colors.append('limegreen')
                node_sizes.append(1500)
                node_labels[node] = f"S{node}"  # Source

            elif node in dests:
                # 根据 VNF 完整性设置颜色
                status = dest_vnf_status.get(node, {})

                if status.get('complete', False):
                    # VNF 完整 - 深蓝色
                    node_colors.append('royalblue')
                    node_sizes.append(1200)
                    node_labels[node] = f"D{node}[OK]"  # 使用 [OK] 替代 ✓
                else:
                    # VNF 不完整 - 红色
                    node_colors.append('red')
                    node_sizes.append(1200)

                    # 根据错误类型添加不同的标记
                    error = status.get('error', '')
                    missing = status.get('missing', [])

                    if error == 'path_broken':
                        node_labels[node] = f"D{node}[!]\nBroken"
                    elif missing:
                        # 只显示第一个缺失的 VNF，避免标签过长
                        node_labels[node] = f"D{node}[!]\n-V{missing[0]}"
                    elif not status.get('order_correct', True):
                        node_labels[node] = f"D{node}[!]\nOrder"
                    else:
                        node_labels[node] = f"D{node}[!]"

            elif node in vnf_nodes:
                # 部署了 VNF 的节点 - 粉红色
                node_colors.append('salmon')
                node_sizes.append(1000)
                vnf_list = vnf_nodes[node]
                node_labels[node] = f"{node}\nV{vnf_list}"

            elif node in self.env.nodes_on_tree:
                # 在树上但未部署 VNF 的节点 - 浅蓝色
                node_colors.append('lightblue')
                node_sizes.append(800)
                node_labels[node] = str(node)

            else:
                # 普通节点 - 白色
                node_colors.append('white')
                node_sizes.append(600)
                node_labels[node] = str(node)

        # =========================================================================
        # 4. 绘制节点
        # =========================================================================
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='black',
            linewidths=2,
            ax=self.ax
        )

        # =========================================================================
        # 5. 绘制节点标签
        # =========================================================================
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold',
            ax=self.ax
        )

        # =========================================================================
        # 6. 添加边的权重标签（带宽）
        # =========================================================================
        if tree_edges:
            edge_labels = {}
            for edge_key, bw in tree_edges.items():
                edge_labels[edge_key] = f"{bw:.1f}"

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='darkblue',
                ax=self.ax
            )

        # =========================================================================
        # 7. 添加标题和图例（使用 ASCII 字符）
        # =========================================================================
        title = f"Request {req_id} Multicast Tree\n"
        title += f"Source: {source} | Dests: {dests} | VNF Chain: {vnf_chain}\n"
        title += f"Connected: {len(connected_dests)}/{len(dests)} | Edges: {len(tree_edges)}\n"

        # VNF 完整性统计
        if vnf_chain:
            if complete_count == len(dests):
                title += f"[OK] VNF Complete: {complete_count}/{len(dests)} (All Passed)"
            else:
                incomplete_count = len(dests) - complete_count
                title += f"[!] VNF Complete: {complete_count}/{len(dests)} ({incomplete_count} Incomplete)"
        else:
            title += "No VNF Required"

        self.ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

        # 创建图例（使用 ASCII 字符）
        legend_elements = [
            mpatches.Patch(color='limegreen', label='Source Node'),
            mpatches.Patch(color='royalblue', label='[OK] VNF Complete Dest'),
            mpatches.Patch(color='red', label='[!] VNF Incomplete Dest'),
            mpatches.Patch(color='salmon', label='VNF Deployment Node'),
            mpatches.Patch(color='lightblue', label='Intermediate Node'),
            mpatches.Patch(color='white', edgecolor='black', label='Unused Node'),
            mpatches.Patch(color='blue', label='Multicast Tree Edge'),
            mpatches.Patch(color='lightgray', label='Network Topology'),
        ]

        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=10
        )

        self.ax.axis('off')
        plt.tight_layout()

        # =========================================================================
        # 8. 保存图像
        # =========================================================================
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            # 根据完整性在文件名中添加标记
            if vnf_chain and complete_count < len(dests):
                print(f"[!] Saved to: {save_path} (Found {len(dests) - complete_count} incomplete VNF paths)")
            else:
                print(f"[OK] Saved to: {save_path}")

        # =========================================================================
        # 9. 显示图像
        # =========================================================================
        if show:
            plt.show()

        # 关闭图形以释放内存
        plt.close(self.fig)

        return self.fig, self.ax

    def _build_network_graph(self):
        """构建网络图"""
        G = nx.Graph()

        # 添加节点
        n = self.env.n
        G.add_nodes_from(range(n))

        # 添加边（从拓扑矩阵）
        topo = self.env.topo
        for i in range(n):
            for j in range(i + 1, n):
                if topo[i, j] > 0:
                    G.add_edge(i, j, weight=topo[i, j])

        return G

    def _get_layout(self, G):
        """获取节点布局"""
        # 尝试使用 spring layout（力导向布局）
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # 如果有 DC 节点信息，可以特殊处理
        if hasattr(self.env, 'dc_nodes') and self.env.dc_nodes:
            # DC 节点放在中心区域
            dc_nodes = self.env.dc_nodes
            for node in dc_nodes:
                if node in pos:
                    # 轻微向中心移动
                    pos[node] = pos[node] * 0.8

        return pos

    def visualize_statistics(self, save_path=None, show=True):
        """
        可视化资源使用统计
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 节点 CPU 使用率
        ax1 = axes[0, 0]
        self._plot_node_cpu_usage(ax1)

        # 2. 节点内存使用率
        ax2 = axes[0, 1]
        self._plot_node_memory_usage(ax2)

        # 3. 链路带宽使用率
        ax3 = axes[1, 0]
        self._plot_link_bandwidth_usage(ax3)

        # 4. VNF 部署分布
        ax4 = axes[1, 1]
        self._plot_vnf_distribution(ax4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 统计图已保存到: {save_path}")

        if show:
            plt.show()

        return fig

    def _plot_node_cpu_usage(self, ax):
        """绘制节点 CPU 使用率"""
        nodes = range(self.env.n)
        cpu_usage = []

        for node in nodes:
            if hasattr(self.env.resource_mgr, 'nodes'):
                # 适配不同的 ResourceManager 实现
                node_info = self.env.resource_mgr.nodes[node]
                if isinstance(node_info, dict):
                    capacity = node_info.get('cpu_capacity', 100)
                    remaining = node_info.get('cpu', capacity)
                    usage = (capacity - remaining) / capacity * 100
                else:
                    usage = 0
            else:
                usage = 0

            cpu_usage.append(usage)

        bars = ax.bar(nodes, cpu_usage, color='steelblue', alpha=0.7)

        # 高亮高使用率节点
        for i, (node, usage) in enumerate(zip(nodes, cpu_usage)):
            if usage > 80:
                bars[i].set_color('red')
            elif usage > 50:
                bars[i].set_color('orange')

        ax.set_xlabel('节点 ID')
        ax.set_ylabel('CPU 使用率 (%)')
        ax.set_title('节点 CPU 使用率分布')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

    def _plot_node_memory_usage(self, ax):
        """绘制节点内存使用率"""
        nodes = range(self.env.n)
        mem_usage = []

        for node in nodes:
            if hasattr(self.env.resource_mgr, 'nodes'):
                node_info = self.env.resource_mgr.nodes[node]
                if isinstance(node_info, dict):
                    capacity = node_info.get('memory_capacity', 100)
                    remaining = node_info.get('memory', capacity)
                    usage = (capacity - remaining) / capacity * 100
                else:
                    usage = 0
            else:
                usage = 0

            mem_usage.append(usage)

        bars = ax.bar(nodes, mem_usage, color='seagreen', alpha=0.7)

        for i, (node, usage) in enumerate(zip(nodes, mem_usage)):
            if usage > 80:
                bars[i].set_color('red')
            elif usage > 50:
                bars[i].set_color('orange')

        ax.set_xlabel('节点 ID')
        ax.set_ylabel('内存使用率 (%)')
        ax.set_title('节点内存使用率分布')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

    def _plot_link_bandwidth_usage(self, ax):
        """绘制链路带宽使用率（热力图）"""
        n = self.env.n
        bw_matrix = np.zeros((n, n))

        if hasattr(self.env.resource_mgr, 'links'):
            links = self.env.resource_mgr.links
            for (u, v), link_info in links.items():
                if isinstance(link_info, dict):
                    capacity = link_info.get('bandwidth_capacity', 100)
                    remaining = link_info.get('bandwidth', capacity)
                    usage = (capacity - remaining) / capacity * 100
                    bw_matrix[u, v] = usage
                    bw_matrix[v, u] = usage

        im = ax.imshow(bw_matrix, cmap='YlOrRd', vmin=0, vmax=100)
        ax.set_xlabel('节点 ID')
        ax.set_ylabel('节点 ID')
        ax.set_title('链路带宽使用率热力图 (%)')

        plt.colorbar(im, ax=ax, label='使用率 (%)')

    def _plot_vnf_distribution(self, ax):
        """绘制 VNF 部署分布"""
        if self.env.current_request is None:
            ax.text(0.5, 0.5, '无活跃请求', ha='center', va='center', fontsize=14)
            ax.axis('off')
            return

        placement = self.env.current_tree.get('placement', {})

        # 统计每种 VNF 的部署次数
        vnf_counts = {}
        for key, info in placement.items():
            if isinstance(key, tuple) and len(key) >= 2:
                vnf_type = key[1]
                vnf_counts[vnf_type] = vnf_counts.get(vnf_type, 0) + 1

        if not vnf_counts:
            ax.text(0.5, 0.5, '未部署 VNF', ha='center', va='center', fontsize=14)
            ax.axis('off')
            return

        vnf_types = sorted(vnf_counts.keys())
        counts = [vnf_counts[vt] for vt in vnf_types]

        bars = ax.bar(vnf_types, counts, color='mediumpurple', alpha=0.7)
        ax.set_xlabel('VNF 类型')
        ax.set_ylabel('部署次数')
        ax.set_title('VNF 部署分布')
        ax.grid(axis='y', alpha=0.3)


def demo_visualization(env):
    """
    演示如何使用可视化工具

    Args:
        env: SFC 环境实例
    """
    visualizer = MulticastTreeVisualizer(env)

    # 1. 可视化当前请求的多播树
    print("\n🎨 正在生成多播树可视化...")
    visualizer.visualize_request_tree(
        save_path='multicast_tree.png',
        show=False  # 训练时设为 False，调试时设为 True
    )

    # 2. 可视化资源统计
    print("\n📊 正在生成资源统计图...")
    visualizer.visualize_statistics(
        save_path='resource_stats.png',
        show=False
    )

    print("\n✅ 可视化完成！")


# ============================================================================
# 集成到训练循环的示例
# ============================================================================

def training_loop_with_visualization(env, agent, num_episodes=100):
    """
    带可视化的训练循环示例
    """
    visualizer = MulticastTreeVisualizer(env)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 选择动作
            action = agent.select_action(obs, info)

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        # 每 10 个 episode 可视化一次成功的请求
        if episode % 10 == 0 and info.get('request_success', False):
            print(f"\n🎨 Episode {episode}: 可视化成功的请求")
            visualizer.visualize_request_tree(
                save_path=f'trees/episode_{episode}_tree.png',
                show=False
            )

        print(f"Episode {episode}: Reward = {total_reward:.2f}")


# ============================================================================
# 命令行调用示例
# ============================================================================

if __name__ == '__main__':
    """
    独立运行此脚本进行可视化

    用法：
    python visualize_multicast_tree.py
    """

    # 1. 加载环境
    import sys
    import yaml

    # 假设你的环境在 envs.sfc_env 中
    from envs.sfc_env import SFC_HIRL_Env

    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 初始化环境
    env = SFC_HIRL_Env(config)
    env.load_dataset('phase3')

    # 运行一个 episode
    obs, info = env.reset()
    done = False

    while not done:
        # 随机动作（用于测试）
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # 如果请求完成，进行可视化
        if info.get('request_completed', False):
            demo_visualization(env)
            break

    print("\n✅ 可视化演示完成！")