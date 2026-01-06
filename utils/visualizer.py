# utils/visualizer.py
import os
import sys

# 🔥 1. 强制设置后端为 'Agg' (无界面模式)
os.environ['MPLBACKEND'] = 'Agg'
# 🔥 2. 物理屏蔽 IPython
sys.modules['IPython'] = None

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class SFCVisualizer:
    def __init__(self, topo_matrix, output_dir):
        """
        初始化可视化器
        :param topo_matrix: 邻接矩阵 (NxN)
        :param output_dir: 图片保存路径
        """
        self.output_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 创建图结构
        self.G = nx.from_numpy_array(topo_matrix, create_using=nx.DiGraph)

        # 2. 布局设置 (🔥 核心修改处)
        # k: 节点间的最佳距离。值越大，节点越分散 (默认约 1/sqrt(n))
        # iterations: 迭代次数。次数越多，布局越稳定
        # seed: 固定随机种子，保证每次画出来的形状一样
        self.pos = nx.spring_layout(self.G, seed=42, k=2.5, iterations=100)

        # 🔥 3. 定义 DC 节点列表 (用户指定)
        # 假设输入的列表是 1-based 的，转换为 0-based
        raw_dc_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 23]
        self.dc_nodes = set(n - 1 for n in raw_dc_nodes)

    def plot_episode(self, ep, request, tree_edges, placement):
        """
        绘制单轮 Episode 的结果 (高对比度 + 节点分散版)
        """
        plt.close('all')

        try:
            plt.figure(figsize=(14, 12))
            ax = plt.gca()

            # =========================================================
            # 🎨 风格定义
            # =========================================================
            STYLE = {
                # --- 背景网络样式 ---
                'bg_edge_color': 'black',
                'bg_edge_width': 1.5,
                'bg_edge_alpha': 0.6,

                # --- 激活路径样式 ---
                'path_edge_color': 'black',
                'path_edge_width': 4.0,

                # --- 节点样式 ---
                'node_edge_color': 'black',
                'node_base_color': 'white',
                'dc_node_color': '#FF3333',
                'src_color': '#4488FF',
                'dst_color': '#FFAA44',
                'font_size': 10,
                'node_size': 700
            }

            # --- 1. 绘制背景 (网络骨架) ---
            G_undir = self.G.to_undirected()
            nx.draw_networkx_edges(
                G_undir, self.pos,
                edge_color=STYLE['bg_edge_color'],
                width=STYLE['bg_edge_width'],
                alpha=STYLE['bg_edge_alpha']
            )

            # --- 2. 准备数据 ---
            src = request.get('source')
            dests = set(request.get('dest', []))
            path_nodes = set()
            active_edges = []
            if tree_edges:
                for u, v in tree_edges:
                    if u is not None and v is not None:
                        active_edges.append((u, v))
                        path_nodes.add(u);
                        path_nodes.add(v)

            # --- 3. 绘制节点 ---
            all_nodes = list(self.G.nodes())

            # 基础白底节点
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=all_nodes, node_color=STYLE['node_base_color'],
                                   edgecolors=STYLE['node_edge_color'], node_size=STYLE['node_size'])

            # DC 节点 (红)
            dc_to_draw = self.dc_nodes - {src} - dests
            if dc_to_draw:
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=list(dc_to_draw), node_color=STYLE['dc_node_color'],
                                       edgecolors='black', node_size=STYLE['node_size'], label='DC Node')

            # 源节点 (蓝三角)
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[src], node_shape='^', node_color=STYLE['src_color'],
                                   edgecolors='black', node_size=STYLE['node_size'] + 200, label='Source')

            # 目的节点 (橙方块)
            if dests:
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=list(dests), node_shape='s',
                                       node_color=STYLE['dst_color'], edgecolors='black',
                                       node_size=STYLE['node_size'] + 100, label='Dest')

            # 标签
            nx.draw_networkx_labels(self.G, self.pos, font_size=STYLE['font_size'], font_family='sans-serif',
                                    font_weight='bold')

            # --- 4. 绘制路径 (超级粗黑箭头) ---
            if active_edges:
                nx.draw_networkx_edges(
                    self.G, self.pos,
                    edgelist=active_edges,
                    edge_color=STYLE['path_edge_color'],
                    width=STYLE['path_edge_width'],
                    arrowstyle='-|>',
                    arrowsize=30
                )

            # --- 5. 标注 VNF (带真实类型 Txx) ---
            vnf_counts = {}
            for k, node_id in placement.items():
                parts = k.split('_')
                vnf_label = "VNF?"

                # 查找 'type' 关键字后面的数字
                if 'type' in parts:
                    try:
                        type_idx = parts.index('type')
                        real_type_id = parts[type_idx + 1]
                        vnf_label = f"T{real_type_id}"
                    except:
                        pass
                elif len(parts) >= 2:
                    vnf_label = f"VNF{int(parts[1]) + 1}"

                count = vnf_counts.get(node_id, 0)
                vnf_counts[node_id] = count + 1

                x, y = self.pos[node_id]
                offset = 0.09 * (count + 1)

                plt.text(x, y - offset, vnf_label,
                         fontsize=11, color='#CC0000', fontweight='bold', ha='center',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

            # --- 6. 保存 ---
            plt.title(f"Episode {ep} Visualization\nSource: {src} -> Dests: {list(dests)}", fontsize=15)
            plt.axis('off')

            # 图例
            import matplotlib.lines as mlines
            legend_handles = [
                mlines.Line2D([], [], color='white', marker='^', markeredgecolor='black',
                              markerfacecolor=STYLE['src_color'], markersize=10, label='Source'),
                mlines.Line2D([], [], color='white', marker='s', markeredgecolor='black',
                              markerfacecolor=STYLE['dst_color'], markersize=10, label='Dest'),
                mlines.Line2D([], [], color='white', marker='o', markeredgecolor='black',
                              markerfacecolor=STYLE['dc_node_color'], markersize=10, label='DC Node'),
                mlines.Line2D([], [], color=STYLE['path_edge_color'], linewidth=4, label='Active Path'),
                mlines.Line2D([], [], color=STYLE['bg_edge_color'], linewidth=1.5, alpha=0.6, label='Topology Link')
            ]
            plt.legend(handles=legend_handles, loc='upper right')

            save_path = os.path.join(self.output_dir, f"ep_{ep}_viz.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

            return save_path

        except Exception as e:
            print(f"⚠️ [Visualizer Error] {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            plt.close('all')