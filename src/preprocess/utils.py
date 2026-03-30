import os
from collections import defaultdict
import polars as pl
import yaml
import json
import tempfile
import time


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def resolve_path(config):
    spot_mode = config.get("spot_mode", False)
    persist = config.get("persistent_storage_path", "/persistent")
    raw_dir = config["data"]["raw_dir"]
    out_dir = config["data"]["processed_dir"]

    if spot_mode:
        out_dir = os.path.join(persist, "processed")
        print(f"Spot mode ON — output: {out_dir}")
    else:
        print(f"Standard mode — output: {out_dir}")

    return raw_dir, out_dir


def atomic_write_parquet(df, filepath):
    """Atomic parquet write: .tmp -> rename."""
    dirpath = os.path.dirname(filepath) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    os.close(fd)
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(data, filepath):
    """Atomic JSON write: .tmp -> rename."""
    dirpath = os.path.dirname(filepath) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def cleanup_tmp_files(dirpath):
    """Wymazuje leftover .tmp files."""
    if not os.path.exists(dirpath):
        return
    for f in os.listdir(dirpath):
        if f.endswith(".tmp"):
            os.remove(os.path.join(dirpath, f))
            print(f"  Cleaned: {f}")


def khop_expand(raw_dir, col_cfg, node_to_subgraphs, node_id_set, k_hop):
    """
    Expand subgraphs by k hops using background_edges.csv.

    Each subgraph is expanded independently: a node discovered via subgraph A
    only propagates subgraph A in subsequent hops (not other subgraphs it may
    also belong to). This prevents unrelated subgraphs from "merging" through
    shared intermediate nodes.

    Args:
        raw_dir: path to raw data directory
        col_cfg: column config from config.yaml
        node_to_subgraphs: dict mapping node_id -> set of subgraph_ids it belongs to
        node_id_set: set of all current node IDs
        k_hop: number of hops to expand

    Returns:
        expansion_nodes: list of (node_id, subgraph_id, hop) tuples for new nodes
        expansion_edges: list of (source, target, txId, subgraph_id) tuples for new edges
        expanded_node_set: set of all node IDs (original + expanded)
    """
    bg_src_col = col_cfg["background_edges"]["source"]
    bg_dst_col = col_cfg["background_edges"]["target"]
    bg_txid_col = col_cfg["background_edges"]["txId"]

    all_expansion_nodes = []  # (node_id, subgraph_id, hop)
    all_expansion_edges = []  # (source, target, txId, subgraph_id)

    current_node_set = set(node_id_set)

    node_to_sgs = defaultdict(set)
    for nid, sgs in node_to_subgraphs.items():
        node_to_sgs[nid] = set(sgs)

    for hop in range(1, k_hop + 1):
        print(f"\n  Hop {hop}/{k_hop}: scanning background_edges.csv...")
        print(f"    Current frontier: {len(current_node_set):,} nodes")
        t0 = time.time()

        # Scan background_edges: find edges touching current node set
        current_node_list = list(current_node_set)
        bg_edges_lazy = pl.scan_csv(f"{raw_dir}/background_edges.csv")

        neighbor_edges = (
            bg_edges_lazy
            .filter(
                pl.col(bg_src_col).is_in(current_node_list) |
                pl.col(bg_dst_col).is_in(current_node_list)
            )
            .select([bg_src_col, bg_dst_col, bg_txid_col])
            .collect(engine="streaming")
        )

        print(f"    Found {len(neighbor_edges):,} edges touching frontier | {time.time() - t0:.1f}s")

        BATCH_SIZE = 100_000
        new_nodes_this_hop = set()
        hop_edges = []
        hop_nodes = []

        # Collect new memberships separately, apply after processing all edges
        # in this hop to avoid order-dependent contamination within a single hop
        pending_updates = defaultdict(set)  # node_id -> set of sg_ids to add

        for batch_start in range(0, len(neighbor_edges), BATCH_SIZE):
            batch = neighbor_edges.slice(batch_start, BATCH_SIZE)

            for row in batch.iter_rows():
                src, dst, txid = row[0], row[1], row[2]

                src_sgs = node_to_sgs.get(src, set())
                dst_sgs = node_to_sgs.get(dst, set())

                if src_sgs and dst not in current_node_set:
                    # src is in subgraph(s), dst is external -> expand
                    for sg_id in src_sgs:
                        hop_edges.append((src, dst, txid, sg_id))
                        hop_nodes.append((dst, sg_id, hop))
                    new_nodes_this_hop.add(dst)
                    pending_updates[dst].update(src_sgs)

                elif dst_sgs and src not in current_node_set:
                    # dst is in subgraph(s), src is external -> expand
                    for sg_id in dst_sgs:
                        hop_edges.append((src, dst, txid, sg_id))
                        hop_nodes.append((src, sg_id, hop))
                    new_nodes_this_hop.add(src)
                    pending_updates[src].update(dst_sgs)

                elif src_sgs and dst_sgs:
                    # Both endpoints already in our set
                    shared_sgs = src_sgs & dst_sgs
                    for sg_id in shared_sgs:
                        hop_edges.append((src, dst, txid, sg_id))

            if len(hop_edges) > 1_000_000:
                print(f"    Buffered {len(hop_edges):,} edges, {len(hop_nodes):,} nodes...")

        # Apply pending updates after all edges in this hop are processed
        for nid, sgs in pending_updates.items():
            node_to_sgs[nid].update(sgs)

        all_expansion_nodes.extend(hop_nodes)
        all_expansion_edges.extend(hop_edges)
        current_node_set.update(new_nodes_this_hop)

        print(f"    New nodes this hop: {len(new_nodes_this_hop):,}")
        print(f"    New edges this hop: {len(hop_edges):,}")

    return all_expansion_nodes, all_expansion_edges, current_node_set
