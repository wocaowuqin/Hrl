# sfc_backup_system/path_finder.py
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PathFinder:
    """
    路径查找器 - Expert 路径查询的统一接口

    功能:
    1. K 最短路径查询
    2. 中转节点路径组合
    3. 最短路径选择
    4. 错误处理和日志

    依赖: expert 必须实现 _get_path_info(src, dst, k) 方法
    """

    def __init__(self, expert, n: int = 28, max_k: int = 5):
        """
        Args:
            expert: MSFCE_Solver 实例,必须有 _get_path_info 方法
            n: 网络节点数
            max_k: 最大 K 路径数 (默认 5)
        """
        self.expert = expert
        self.n = n

        # ✅ 从 expert 获取实际的 k_path,否则使用 max_k
        self.max_k = int(getattr(expert, "k_path", max_k))

        logger.info(f"[PathFinder] Initialized: n={n}, max_k={self.max_k}")

        # 性能统计
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "failed_queries": 0,
            "relay_compositions": 0
        }

    # --------------------------
    # 核心方法: K 路径查询
    # --------------------------
    def get_k_path(self, src: int, dst: int, k: int) -> Tuple[
        Optional[List[int]], Optional[float], Optional[List[int]]]:
        """
        获取第 k 条最短路径

        Args:
            src: 源节点 (1-based)
            dst: 目标节点 (1-based)
            k: 路径序号 (1-based, 1 表示最短路径)

        Returns:
            (nodes, distance, links) 或 (None, None, None)
            - nodes: 路径节点列表 (1-based)
            - distance: 路径跳数
            - links: 路径链路 ID 列表 (1-based)
        """
        self.stats["total_queries"] += 1

        # ✅ 参数验证
        if src == dst:
            logger.debug(f"[PathFinder] Self-loop detected: src={src} == dst={dst}")
            return [src], 0, []

        if not (1 <= src <= self.n and 1 <= dst <= self.n):
            logger.warning(f"[PathFinder] Invalid node range: src={src}, dst={dst}, n={self.n}")
            return None, None, None

        if k < 1 or k > self.max_k:
            logger.debug(f"[PathFinder] k={k} out of range [1, {self.max_k}]")
            return None, None, None

        # ✅ 调用 expert 的路径查询
        try:
            nodes, distance, links = self.expert._get_path_info(src, dst, k)

            # 验证返回值
            if nodes and len(nodes) >= 2 and links:
                logger.debug(
                    f"[PathFinder] ✅ Path found: {src}→{dst} k={k}, "
                    f"hops={len(nodes) - 1}, links={len(links)}"
                )
                return nodes, distance, links
            else:
                logger.debug(
                    f"[PathFinder] ❌ Empty path: {src}→{dst} k={k}"
                )
                self.stats["failed_queries"] += 1
                return None, None, None

        except Exception as e:
            logger.warning(
                f"[PathFinder] Exception in _get_path_info({src}, {dst}, {k}): {e}",
                exc_info=True
            )
            self.stats["failed_queries"] += 1
            return None, None, None

    # --------------------------
    # 便捷方法: 查找任意可行路径
    # --------------------------
    def find_any_path(self, src: int, dst: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        尝试 k=1..max_k,返回第一条可行路径

        Returns:
            (nodes, links) 或 (None, None)
        """
        logger.debug(f"[PathFinder] Searching any path: {src}→{dst}")

        for k in range(1, self.max_k + 1):
            nodes, distance, links = self.get_k_path(src, dst, k)

            if nodes and links:
                logger.debug(
                    f"[PathFinder] ✅ Found path at k={k}: "
                    f"{src}→{dst}, hops={len(nodes) - 1}"
                )
                return nodes, links

        logger.debug(f"[PathFinder] ❌ No path found: {src}→{dst} (tried k=1..{self.max_k})")
        return None, None

    def find_shortest_path(self, dst: int, src: Optional[int] = None) -> Tuple[
        Optional[List[int]], Optional[List[int]]]:
        """
        查找最短路径 (k=1)

        Args:
            dst: 目标节点
            src: 源节点 (如果为 None,从 expert 的 current_request 获取)

        Returns:
            (nodes, links) 或 (None, None)
        """
        if src is None:
            # ✅ 从 expert 的当前请求获取源节点
            src = int(getattr(self.expert, "current_request", {}).get("source", 1))

        nodes, distance, links = self.get_k_path(src, dst, k=1)

        if nodes and links:
            logger.debug(f"[PathFinder] ✅ Shortest path: {src}→{dst}, hops={len(nodes) - 1}")
            return nodes, links
        else:
            logger.debug(f"[PathFinder] ❌ No shortest path: {src}→{dst}")
            return None, None

    # --------------------------
    # 高级方法: 中转节点路径组合
    # --------------------------
    def compose_via_relay(self, src: int, relay: int, dst: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        通过中转节点组合路径: src → relay → dst

        注意:
        - 避免中转节点重复
        - 合并节点列表时去除 relay 的重复
        - 链路列表直接拼接

        Args:
            src: 源节点
            relay: 中转节点
            dst: 目标节点

        Returns:
            (combined_nodes, combined_links) 或 (None, None)
        """
        self.stats["relay_compositions"] += 1

        # ✅ 参数验证
        if relay == src or relay == dst:
            logger.debug(f"[PathFinder] Invalid relay: relay={relay}, src={src}, dst={dst}")
            return None, None

        # ✅ 第一段: src → relay
        nodes1, links1 = self.find_any_path(src, relay)
        if not nodes1 or not links1:
            logger.debug(f"[PathFinder] No path for segment 1: {src}→{relay}")
            return None, None

        # ✅ 第二段: relay → dst
        nodes2, links2 = self.find_any_path(relay, dst)
        if not nodes2 or not links2:
            logger.debug(f"[PathFinder] No path for segment 2: {relay}→{dst}")
            return None, None

        # ✅ 组合路径 (去除中转节点重复)
        combined_nodes = nodes1 + nodes2[1:]  # 去除 nodes2 的第一个节点 (relay)
        combined_links = links1 + links2

        logger.debug(
            f"[PathFinder] ✅ Composed via relay {relay}: "
            f"{src}→{relay}→{dst}, total_hops={len(combined_nodes) - 1}"
        )

        return combined_nodes, combined_links

    def compose_via_multiple_relays(self, src: int, relays: List[int], dst: int) -> Tuple[
        Optional[List[int]], Optional[List[int]]]:
        """
        通过多个中转节点组合路径: src → relay1 → relay2 → ... → dst

        Args:
            src: 源节点
            relays: 中转节点列表
            dst: 目标节点

        Returns:
            (combined_nodes, combined_links) 或 (None, None)
        """
        if not relays:
            return self.find_any_path(src, dst)

        # 构建完整路径序列
        path_sequence = [src] + relays + [dst]

        all_nodes = []
        all_links = []

        for i in range(len(path_sequence) - 1):
            current_src = path_sequence[i]
            current_dst = path_sequence[i + 1]

            nodes, links = self.find_any_path(current_src, current_dst)

            if not nodes or not links:
                logger.debug(
                    f"[PathFinder] Failed at segment {i + 1}: "
                    f"{current_src}→{current_dst}"
                )
                return None, None

            if i == 0:
                all_nodes = nodes
            else:
                all_nodes.extend(nodes[1:])  # 去除重复节点

            all_links.extend(links)

        logger.debug(
            f"[PathFinder] ✅ Composed via {len(relays)} relays: "
            f"total_hops={len(all_nodes) - 1}"
        )

        return all_nodes, all_links

    # --------------------------
    # 高级查询: 带约束的路径查询
    # --------------------------
    def find_path_with_constraints(self, src: int, dst: int,
                                   max_hops: Optional[int] = None,
                                   required_nodes: Optional[List[int]] = None,
                                   forbidden_nodes: Optional[List[int]] = None) -> Tuple[
        Optional[List[int]], Optional[List[int]]]:
        """
        查找满足约束的路径

        Args:
            src: 源节点
            dst: 目标节点
            max_hops: 最大跳数限制
            required_nodes: 必须经过的节点
            forbidden_nodes: 禁止经过的节点

        Returns:
            (nodes, links) 或 (None, None)
        """
        required_set = set(required_nodes or [])
        forbidden_set = set(forbidden_nodes or [])

        for k in range(1, self.max_k + 1):
            nodes, distance, links = self.get_k_path(src, dst, k)

            if not nodes or not links:
                continue

            # ✅ 检查跳数约束
            if max_hops is not None and len(nodes) - 1 > max_hops:
                continue

            # ✅ 检查禁止节点
            if forbidden_set and any(n in forbidden_set for n in nodes):
                continue

            # ✅ 检查必经节点
            if required_set and not required_set.issubset(set(nodes)):
                continue

            logger.debug(
                f"[PathFinder] ✅ Found constrained path at k={k}: "
                f"{src}→{dst}, hops={len(nodes) - 1}"
            )
            return nodes, links

        logger.debug(
            f"[PathFinder] ❌ No path satisfies constraints: {src}→{dst}"
        )
        return None, None

    # --------------------------
    # 统计和调试
    # --------------------------
    def get_statistics(self) -> dict:
        """获取性能统计"""
        stats = dict(self.stats)

        if stats["total_queries"] > 0:
            stats["success_rate"] = 1.0 - (stats["failed_queries"] / stats["total_queries"])
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_statistics(self):
        """重置统计"""
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "failed_queries": 0,
            "relay_compositions": 0
        }
        logger.info("[PathFinder] Statistics reset")

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("PathFinder Statistics")
        print("=" * 50)
        print(f"Total Queries:      {stats['total_queries']}")
        print(f"Failed Queries:     {stats['failed_queries']}")
        print(f"Success Rate:       {stats['success_rate']:.2%}")
        print(f"Relay Compositions: {stats['relay_compositions']}")
        print("=" * 50 + "\n")

    # --------------------------
    # 路径验证
    # --------------------------
    def validate_path(self, nodes: List[int], links: List[int]) -> bool:
        """
        验证路径的有效性

        检查:
        1. 节点和链路数量匹配
        2. 节点在有效范围内
        3. 链路连续性
        """
        if not nodes or not links:
            return False

        # ✅ 检查数量关系
        if len(links) != len(nodes) - 1:
            logger.warning(
                f"[PathFinder] Invalid path: "
                f"links={len(links)} != nodes-1={len(nodes) - 1}"
            )
            return False

        # ✅ 检查节点范围
        if any(n < 1 or n > self.n for n in nodes):
            logger.warning(f"[PathFinder] Invalid node range in path")
            return False

        # ✅ 检查连续性 (可选,需要链路映射)
        # 这里简化处理,只检查基本约束

        return True

    def get_path_length(self, nodes: List[int]) -> int:
        """获取路径跳数"""
        return len(nodes) - 1 if nodes else 0

    def get_path_info_summary(self, nodes: List[int], links: List[int]) -> dict:
        """获取路径摘要信息"""
        if not nodes or not links:
            return {"valid": False}

        return {
            "valid": True,
            "hops": len(nodes) - 1,
            "nodes_count": len(nodes),
            "links_count": len(links),
            "src": nodes[0] if nodes else None,
            "dst": nodes[-1] if nodes else None,
            "intermediate_nodes": nodes[1:-1] if len(nodes) > 2 else []
        }