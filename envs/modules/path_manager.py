from typing import List, Dict, Optional, Tuple, Union
import hashlib
import json


Path = List[int]
MulticastTree = Dict[int, List[int]]  # dest -> path


class PathManager:
    """
    PathManager (Extended Version)

    支持：
    1. 单一路径（List[int]）
    2. Multicast Tree（Dict[dest, List[int]]）

    提供：
    - 路径去重
    - 稳定 path_id
    - 可回溯路径结构
    """

    def __init__(self, max_paths: int = 10000):
        self.max_paths = max_paths

        # 核心存储
        self.paths: List[Path] = []                    # 单一路径池
        self.trees: List[MulticastTree] = []           # multicast tree 池

        # 索引
        self.path_to_idx: Dict[Tuple[int, ...], int] = {}
        self.tree_to_idx: Dict[str, int] = {}

        # 统计
        self.path_usage: Dict[int, int] = {}
        self.tree_usage: Dict[int, int] = {}

    # ============================================================
    # 工具函数
    # ============================================================

    @staticmethod
    def _hash_tree(tree: MulticastTree) -> str:
        """
        对 multicast tree 做结构级 hash（顺序无关、稳定）
        """
        # 排序 dest，保证一致性
        canonical = {int(k): v for k, v in sorted(tree.items(), key=lambda x: x[0])}
        tree_json = json.dumps(canonical, sort_keys=True)
        return hashlib.md5(tree_json.encode("utf-8")).hexdigest()

    # ============================================================
    # 单一路径接口（向后兼容）
    # ============================================================

    def add_path(self, path: Path) -> int:
        """
        添加单一路径，返回 path_id
        """
        key = tuple(path)

        if key in self.path_to_idx:
            idx = self.path_to_idx[key]
            self.path_usage[idx] += 1
            return idx

        if len(self.paths) >= self.max_paths:
            return 0  # fallback

        idx = len(self.paths)
        self.paths.append(path)
        self.path_to_idx[key] = idx
        self.path_usage[idx] = 1
        return idx

    def get_path(self, idx: int) -> Optional[Path]:
        if 0 <= idx < len(self.paths):
            return self.paths[idx]
        return None

    # ============================================================
    # Multicast Tree 接口（新增核心）
    # ============================================================

    def add_tree(self, tree: MulticastTree) -> int:
        """
        添加 multicast tree，返回 tree_id
        """
        tree_hash = self._hash_tree(tree)

        if tree_hash in self.tree_to_idx:
            idx = self.tree_to_idx[tree_hash]
            self.tree_usage[idx] += 1
            return idx

        if len(self.trees) >= self.max_paths:
            return 0  # fallback

        idx = len(self.trees)
        self.trees.append(tree)
        self.tree_to_idx[tree_hash] = idx
        self.tree_usage[idx] = 1
        return idx

    def get_tree(self, idx: int) -> Optional[MulticastTree]:
        if 0 <= idx < len(self.trees):
            return self.trees[idx]
        return None

    # ============================================================
    # 高级工具
    # ============================================================

    def get_representative_path(self, tree: MulticastTree) -> Path:
        """
        从 multicast tree 中选一条代表路径（最长 hop）
        """
        return max(tree.values(), key=len)

    def get_tree_stats(self, idx: int) -> Dict[str, float]:
        """
        统计 multicast tree 的结构指标
        """
        tree = self.get_tree(idx)
        if tree is None:
            return {}

        hops = [len(p) - 1 for p in tree.values()]
        return {
            "num_dests": len(tree),
            "max_hops": max(hops),
            "min_hops": min(hops),
            "avg_hops": sum(hops) / len(hops),
        }

    # ============================================================
    # 生命周期管理
    # ============================================================

    def reset(self):
        self.paths.clear()
        self.trees.clear()
        self.path_to_idx.clear()
        self.tree_to_idx.clear()
        self.path_usage.clear()
        self.tree_usage.clear()

    def __len__(self):
        return len(self.paths) + len(self.trees)
