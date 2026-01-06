# sfc_backup_system/utils.py
from typing import Dict, List, Any
import numpy as np


def build_tree_vec(links: List[int], L: int) -> np.ndarray:
    """Build binary link vector (1-based link ids expected)."""
    vec = np.zeros(L, dtype=float)
    for lid in links or []:
        try:
            idx = int(lid) - 1
            if 0 <= idx < L:
                vec[idx] = 1.0
        except Exception:
            continue
    return vec


def build_hvt_from_placement(placement: Dict[int, int], n: int, K_vnf: int) -> Any:
    """
    Return hvt matrix (n x K_vnf) as numpy array of ints.
    placement: mapping vnf_index -> node_id (1-based node id expected)
    """
    hvt = np.zeros((n, K_vnf), dtype=int)
    if not placement:
        return hvt
    for vnf_key, node in placement.items():
        try:
            vnf_idx = int(vnf_key)
            node_idx = int(node) - 1
            if 0 <= node_idx < n and 0 <= vnf_idx < K_vnf:
                hvt[node_idx, vnf_idx] = 1
        except Exception:
            continue
    return hvt


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return [x]
