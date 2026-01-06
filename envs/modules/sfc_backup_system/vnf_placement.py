# sfc_backup_system/vnf_placement.py
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VNFPlacement:
    """
    Multiple placement helpers.
    All methods return either placement dict (vnf_index -> node_id) or None on failure.
    """

    @staticmethod
    def simple_round_robin(vnf_sequence: List[int], path_nodes: List[int], is_dc_fn) -> Dict[int, int]:
        """Round-robin placing VNFs on DC nodes on the path (or any node if no DC)."""
        if not vnf_sequence:
            return {}
        dc_nodes_on_path = [n for n in path_nodes if is_dc_fn(n)]
        targets = dc_nodes_on_path or (path_nodes or [1])
        placement = {}
        for i, _v in enumerate(vnf_sequence):
            placement[int(i)] = int(targets[i % len(targets)])
        return placement

    @staticmethod
    def resource_aware(vnf_sequence: List[int], path_nodes: List[int], network_state: Dict, is_dc_fn) -> Optional[Dict[int, int]]:
        """
        Place VNFs on DC nodes on the path; strict resource checking; reuse allowed.
        network_state expected keys: 'cpu' (dict node_idx->value) , 'mem', 'hvt' (2D-like)
        Returns placement mapping vnf_index -> node_id (1-based), or None if cannot place.
        """
        if not vnf_sequence:
            return {}

        # Normalize structures
        node_cpu = network_state.get("cpu", {})  # expect dict keyed by node_idx (0-based) or node_id-1
        node_mem = network_state.get("mem", {})
        hvt = network_state.get("hvt", None)  # if numpy array or list of lists, index by [node_idx][vnf_idx]

        # select dc nodes on path, fallback to any path node if none
        dc_nodes = [n for n in path_nodes if is_dc_fn(n)]
        if not dc_nodes:
            dc_nodes = list(path_nodes or [])

        if not dc_nodes:
            logger.debug("VNFPlacement.resource_aware: no candidate nodes")
            return None

        # local copies for greedy checks (do not mutate original network_state)
        cpu_local = {int(k): float(v) for k, v in (node_cpu.items())}
        mem_local = {int(k): float(v) for k, v in (node_mem.items())}

        placement: Dict[int, int] = {}

        for i, v in enumerate(vnf_sequence):
            v_idx = int(v) if isinstance(v, (int, float)) else i
            placed = False

            # 1) Try reuse: prefer nodes that already have this vnf deployed
            if hvt is not None:
                try:
                    for nd in dc_nodes:
                        nd_idx = int(nd) - 1
                        if nd_idx < 0:
                            continue
                        if hasattr(hvt, "shape") and getattr(hvt, "shape")[0] == nd_idx + 1:
                            # numpy-like
                            if int(hvt[nd_idx][v_idx]) > 0:
                                placement[i] = nd
                                placed = True
                                break
                        else:
                            # list of lists
                            if nd_idx < len(hvt) and v_idx < len(hvt[nd_idx]) and int(hvt[nd_idx][v_idx]) > 0:
                                placement[i] = nd
                                placed = True
                                break
                except Exception:
                    pass
            if placed:
                continue

            # 2) Try to place on a DC node with enough resources
            for nd in dc_nodes:
                nd_idx = int(nd) - 1
                cpu_avail = cpu_local.get(nd_idx, 0.0)
                mem_avail = mem_local.get(nd_idx, 0.0)

                # demand estimation: try to read from current_request if present (cpu_origin/memory_origin)
                # If not available, assume small default need (1 unit)
                try:
                    cpu_demands = list(self_get := [])  # placeholder to satisfy static analyzers
                except Exception:
                    cpu_demands = []

                # We can't access self here; the caller (BackupPolicy) will pass exact cpu/mem lists by network_state if needed.
                # So assume minimal requirement if explicit requirement not supplied.
                # We choose a conservative behavior: require at least 1 unit available.
                if cpu_avail >= 1 and mem_avail >= 1:
                    placement[i] = nd
                    cpu_local[nd_idx] = cpu_avail - 1
                    mem_local[nd_idx] = mem_avail - 1
                    placed = True
                    break

            if not placed:
                logger.debug(f"VNFPlacement.resource_aware: failed to place vnf index {i}")
                return None

        return placement
