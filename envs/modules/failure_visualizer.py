# failure_visualizer.py
import networkx as nx
import matplotlib.pyplot as plt


class FailureVisualizer:
    def __init__(self, topo_edges, node_positions=None):
        # ... (这部分保持不变) ...
        self.G = nx.Graph()
        self.G.add_edges_from(topo_edges)
        if node_positions is None:
            self.pos = nx.spring_layout(self.G, seed=42)
        else:
            self.pos = node_positions

    def draw_failure_case(self,
                          src,
                          dests,
                          success_paths,
                          vnf_placement,
                          failed_dest=None,
                          failed_path=None,
                          title="Failure Case Visualization",
                          save_path=None):  # <--- [修改1] 新增 save_path 参数

        # [修改2] 显式创建 figure 对象，防止内存泄漏
        fig = plt.figure(figsize=(10, 8))

        # ... (中间画图逻辑完全保持不变: nx.draw_networkx_edges 等) ...
        # -----------------------------
        # 1. 画出基本拓扑
        # -----------------------------
        nx.draw_networkx_edges(self.G, self.pos, edge_color="gray", width=1, alpha=0.5)

        # ... (省略中间画节点和路径的代码，原样保留即可) ...

        # -----------------------------
        # 4. 失败路径（红色虚线）
        # ... (代码保持不变) ...
        if failed_path is not None and len(failed_path) > 1:
            red_edges = list(zip(failed_path[:-1], failed_path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=red_edges, edge_color="red", width=3, style="dashed")

        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.tight_layout()

        # -----------------------------
        # [修改3] 保存文件逻辑
        # -----------------------------
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)  # 必须关闭，否则训练久了会爆内存
            # print(f"Saved: {save_path}")
        else:
            plt.show()