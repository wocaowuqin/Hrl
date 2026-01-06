import json
import networkx as nx
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def load_topology(topo_path: str, topo_type: str = "multicast_tree"):
    """
    加载网络拓扑
    :param topo_path: 拓扑文件路径 (如 data/topo/multicast_topo.json)
    :param topo_type: 拓扑类型
    :return: networkx.Graph 对象
    """
    if not os.path.exists(topo_path):
        # 如果文件不存在，生成一个默认的随机拓扑作为兜底（方便调试）
        logger.warning(f"Topology file not found: {topo_path}. Generating a random topology for testing.")
        return nx.erdos_renyi_graph(n=20, p=0.3, seed=42)

    try:
        if topo_path.endswith('.json'):
            with open(topo_path, 'r') as f:
                data = json.load(f)
                # 假设 JSON 格式为 {"nodes": [], "links": [[u, v, bw], ...]}
                # 这里根据你的实际 JSON 格式适配
                G = nx.node_link_graph(data) if "links" in data else nx.Graph()
        else:
            # 扩展：支持 .mat 或其他格式
            G = nx.erdos_renyi_graph(n=20, p=0.3)

        return G
    except Exception as e:
        logger.error(f"Failed to load topology: {e}")
        raise e


def get_link_utilization(G):
    """计算全网链路平均利用率"""
    utils = []
    for u, v, data in G.edges(data=True):
        total = data.get('capacity', 1000.0)
        used = data.get('used', 0.0)
        utils.append(used / total if total > 0 else 1.0)
    return np.mean(utils) if utils else 0.0