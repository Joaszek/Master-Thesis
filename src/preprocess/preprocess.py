"""
preprocess.py — Elliptic2 Feature Extraction + k-hop Expansion
===============================================================
Extracts node/edge features from large background files
for nodes/edges in labeled subgraphs.

Optionally expands subgraphs by k hops using the background graph,
giving models more structural context.

Usage:
    python preprocess.py
"""
import os
import time
import json
import yaml
import tempfile
import polars as pl
from collections import defaultdict


# ============================================================
# Helper functions
# ============================================================
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

    Args:
        raw_dir: path to raw data directory
        col_cfg: column config from config.yaml
        node_to_subgraphs: dict mapping node_id -> set of subgraph_ids it belongs to
        node_id_set: set of all current node IDs
        k_hop: number of hops to expand

    Returns:
        expansion_nodes: list of (node_id, subgraph_id) tuples for new nodes
        expansion_edges: list of (source, target, txId, subgraph_id) tuples for new edges
    """
    bg_src_col = col_cfg["background_edges"]["source"]
    bg_dst_col = col_cfg["background_edges"]["target"]
    bg_txid_col = col_cfg["background_edges"]["txId"]

    all_expansion_nodes = []  # (node_id, subgraph_id)
    all_expansion_edges = []  # (source, target, txId, subgraph_id)

    current_node_set = set(node_id_set)
    current_node_to_sgs = defaultdict(set)
    for nid, sgs in node_to_subgraphs.items():
        current_node_to_sgs[nid] = set(sgs)

    for hop in range(1, k_hop + 1):
        print(f"\n  Hop {hop}/{k_hop}: scanning background_edges.csv...")
        print(f"    Current frontier: {len(current_node_set):,} nodes")
        t0 = time.time()

        # Scan background_edges: find edges touching current node set
        current_node_list = list(current_node_set)
        bg_edges_lazy = pl.scan_csv(f"{raw_dir}/background_edges.csv")

        # Filter: source OR target is in our node set
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

        # Process edges: assign to subgraphs, find new nodes
        BATCH_SIZE = 100_000
        new_nodes_this_hop = set()
        hop_edges = []
        hop_nodes = []

        for batch_start in range(0, len(neighbor_edges), BATCH_SIZE):
            batch = neighbor_edges.slice(batch_start, BATCH_SIZE)
            
            for row in batch.iter_rows():
                src, dst, txid = row[0], row[1], row[2]
                
                src_sgs = current_node_to_sgs.get(src, set())
                dst_sgs = current_node_to_sgs.get(dst, set())

            if src_sgs and dst not in current_node_set:
                # src is in subgraph(s), dst is external -> expand
                for sg_id in src_sgs:
                    hop_edges.append((src, dst, txid, sg_id))
                    hop_nodes.append((dst, sg_id))
                new_nodes_this_hop.add(dst)
                current_node_to_sgs[dst].update(src_sgs)

            elif dst_sgs and src not in current_node_set:
                # dst is in subgraph(s), src is external -> expand
                for sg_id in dst_sgs:
                    hop_edges.append((src, dst, txid, sg_id))
                    hop_nodes.append((src, sg_id))
                new_nodes_this_hop.add(src)
                current_node_to_sgs[src].update(dst_sgs)

            elif src_sgs and dst_sgs:
                # Both endpoints already in our set — edge between known nodes
                # Add edge to all subgraphs that both endpoints share
                shared_sgs = src_sgs & dst_sgs
                for sg_id in shared_sgs:
                    hop_edges.append((src, dst, txid, sg_id))
            if len(hop_edges) > 1_000_000:
                print(f"    Buffered {len(hop_edges):,} edges, {len(hop_nodes):,} nodes...")

        all_expansion_nodes.extend(hop_nodes)
        all_expansion_edges.extend(hop_edges)
        current_node_set.update(new_nodes_this_hop)

        print(f"    New nodes this hop: {len(new_nodes_this_hop):,}")
        print(f"    New edges this hop: {len(hop_edges):,}")

    return all_expansion_nodes, all_expansion_edges, current_node_set


# ============================================================
# Main preprocessing
# ============================================================
def main():
    config = load_config()
    raw_dir, out_dir = resolve_path(config)
    col_cfg = config["data"]["columns"]
    k_hop = config["data"].get("k_hop", 0)
    os.makedirs(out_dir, exist_ok=True)
    cleanup_tmp_files(out_dir)

    total_start = time.time()

    # ================================================================
    # STEP 1: Load labeled files
    # ================================================================
    print("\n" + "=" * 60)
    print("[1/6] Loading labeled subgraph files...")
    print("=" * 60)

    nodes_df = pl.read_csv(f"{raw_dir}/nodes.csv")
    edges_df = pl.read_csv(f"{raw_dir}/edges.csv")
    components_df = pl.read_csv(f"{raw_dir}/connected_components.csv")

    # Column names (hardcoded from config)
    node_id_col = col_cfg["nodes"]["node_id"]
    node_subgraph_col = col_cfg["nodes"]["subgraph_id"]
    edge_src_col = col_cfg["edges"]["source"]
    edge_dst_col = col_cfg["edges"]["target"]
    edge_txid_col = col_cfg["edges"]["txId"]
    comp_id_col = col_cfg["components"]["subgraph_id"]
    comp_label_col = col_cfg["components"]["label"]

    sample_ratio = config["data"].get("sample_ratio", 1.0)

    if sample_ratio < 1.0:
        n_sample = max(1, int(len(components_df) * sample_ratio))
        components_df = components_df.sample(n=n_sample, seed=42)
        sampled_subgraphs = components_df[comp_id_col].to_list()
        nodes_df = nodes_df.filter(pl.col(node_subgraph_col).is_in(sampled_subgraphs))
        sampled_node_ids = nodes_df[node_id_col].to_list()
        edges_df = edges_df.filter(pl.col(edge_src_col).is_in(sampled_node_ids))
        print(f"  SAMPLED {sample_ratio*100:.1f}%: {n_sample} subgraphs")

    print(f"  nodes.csv:                {len(nodes_df):>10,} rows")
    print(f"  edges.csv:                {len(edges_df):>10,} rows")
    print(f"  connected_components.csv: {len(components_df):>10,} rows")

    # Rename to standard names early
    nodes_df = nodes_df.rename({node_id_col: "node_id", node_subgraph_col: "subgraph_id"})
    edges_df = edges_df.rename({edge_src_col: "source", edge_dst_col: "target", edge_txid_col: "txId"})

    # Label distribution
    label_counts = components_df[comp_label_col].value_counts().sort("count", descending=True)
    print(f"\n  Label distribution:")
    for row in label_counts.iter_rows():
        print(f"    Label {row[0]}: {row[1]:,} ({row[1] / len(components_df) * 100:.1f}%)")

    original_num_nodes = len(nodes_df)
    original_num_edges = len(edges_df)

    # ================================================================
    # STEP 2: k-hop expansion (if enabled)
    # ================================================================
    print("\n" + "=" * 60)
    print(f"[2/6] k-hop expansion (k={k_hop})...")
    print("=" * 60)

    if k_hop > 0:
        # Build node -> subgraph mapping
        node_to_subgraphs = defaultdict(set)
        for row in nodes_df.iter_rows(named=True):
            node_to_subgraphs[row["node_id"]].add(row["subgraph_id"])

        node_id_set = set(nodes_df["node_id"].to_list())

        expansion_nodes, expansion_edges, expanded_node_set = khop_expand(
            raw_dir, col_cfg, node_to_subgraphs, node_id_set, k_hop
        )

        # Add expansion nodes to nodes_df
        if expansion_nodes:
            exp_nodes_df = pl.DataFrame(
                {"node_id": [n[0] for n in expansion_nodes],
                 "subgraph_id": [n[1] for n in expansion_nodes]}
            ).unique()
            nodes_df = pl.concat([nodes_df, exp_nodes_df])
            print(f"\n  Expanded nodes: {original_num_nodes:,} -> {len(nodes_df):,} (+{len(nodes_df) - original_num_nodes:,})")

        # Add subgraph_id to original edges using source node lookup
        src_to_sg = dict(zip(nodes_df["node_id"].to_list(), nodes_df["subgraph_id"].to_list()))
        edges_df = edges_df.with_columns(
            pl.col("source").replace(src_to_sg, default=None).alias("subgraph_id")
        )

        # Add expansion edges
        if expansion_edges:
            exp_edges_df = pl.DataFrame({
                "source": [e[0] for e in expansion_edges],
                "target": [e[1] for e in expansion_edges],
                "txId": [e[2] for e in expansion_edges],
                "subgraph_id": [e[3] for e in expansion_edges],
            }).unique()
            edges_df = pl.concat([edges_df, exp_edges_df])
            print(f"  Expanded edges: {original_num_edges:,} -> {len(edges_df):,} (+{len(edges_df) - original_num_edges:,})")

        node_id_set = expanded_node_set
    else:
        print("  k_hop=0 — no expansion")
        node_id_set = set(nodes_df["node_id"].to_list())

        # Still add subgraph_id to edges for consistency
        src_to_sg = dict(zip(nodes_df["node_id"].to_list(), nodes_df["subgraph_id"].to_list()))
        edges_df = edges_df.with_columns(
            pl.col("source").replace(src_to_sg, default=None).alias("subgraph_id")
        )

    # Update ID sets for feature extraction
    node_id_list = list(node_id_set)
    txid_set = set(edges_df["txId"].to_list())
    txid_list = list(txid_set)

    print(f"\n  Final unique nodes to extract: {len(node_id_set):,}")
    print(f"  Final unique txIds to extract: {len(txid_set):,}")

    # ================================================================
    # STEP 3: Save parquet files (atomic, resume-safe)
    # ================================================================
    print("\n" + "=" * 60)
    print("[3/6] Saving parquet files...")
    print("=" * 60)

    nodes_parquet_path = f"{out_dir}/nodes.parquet"
    edges_parquet_path = f"{out_dir}/edges.parquet"
    components_parquet_path = f"{out_dir}/components.parquet"

    # Check if k_hop changed — force rewrite
    summary_path = f"{out_dir}/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            old_summary = json.load(f)
        if old_summary.get("k_hop", 0) != k_hop:
            print(f"  k_hop changed ({old_summary.get('k_hop', 0)} -> {k_hop}) — forcing rewrite of all files")
            for p in [nodes_parquet_path, edges_parquet_path,
                       f"{out_dir}/node_features.parquet", f"{out_dir}/edge_features.parquet",
                       f"{out_dir}/all_graphs.pt"]:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"    removed {p}")

    # nodes.parquet
    if not os.path.exists(nodes_parquet_path):
        print("\n  Writing nodes.parquet...")
        atomic_write_parquet(nodes_df, nodes_parquet_path)
    else:
        print("\n  nodes.parquet exists — skipped")

    # edges.parquet
    if os.path.exists(edges_parquet_path):
        _check = pl.read_parquet(edges_parquet_path, n_rows=1)
        if "subgraph_id" not in _check.columns:
            print("  Old edges.parquet without subgraph_id — removing, rewriting...")
            os.remove(edges_parquet_path)

    if not os.path.exists(edges_parquet_path):
        print("  Writing edges.parquet...")
        atomic_write_parquet(edges_df, edges_parquet_path)
    else:
        print("  edges.parquet exists — skipped")

    # components.parquet (with int type check)
    if os.path.exists(components_parquet_path):
        _check = pl.read_parquet(components_parquet_path, n_rows=1)
        if _check["label"].dtype not in [pl.Int32, pl.Int64]:
            print(f"  Old components.parquet with non-int labels ({_check['label'].dtype}) — removing, rewriting...")
            os.remove(components_parquet_path)
        else:
            print("  components.parquet exists — skipped")

    if not os.path.exists(components_parquet_path):
        print("  Writing components.parquet...")
        components_to_save = components_df.rename({comp_id_col: "subgraph_id", comp_label_col: "label"})

        # Map string labels to int: "licit" -> 0, "suspicious" -> 1, "illicit" -> 2
        components_to_save = components_to_save.with_columns(
            pl.col("label").replace({"licit": 0, "suspicious": 1, "illicit": 2})
        )
        atomic_write_parquet(components_to_save, components_parquet_path)

    # ================================================================
    # STEP 4: Extract node features from background_nodes.csv
    # ================================================================
    node_features_path = f"{out_dir}/node_features.parquet"

    print("\n" + "=" * 60)
    print("[4/6] Node features from background_nodes.csv...")
    print("=" * 60)

    if os.path.exists(node_features_path):
        print("  node_features.parquet exists — skipped (resume)")
        filtered_node_features = pl.read_parquet(node_features_path)
        node_feat_dim = len(filtered_node_features.columns) - 1
    else:
        t0 = time.time()
        bg_nodes_lazy = pl.scan_csv(f"{raw_dir}/background_nodes.csv")
        bg_node_id_col_name = bg_nodes_lazy.collect_schema().names()[0]
        print(f"  background_nodes: {len(bg_nodes_lazy.collect_schema().names())} columns, ID col='{bg_node_id_col_name}'")

        filtered_node_features = (
            bg_nodes_lazy
            .filter(pl.col(bg_node_id_col_name).is_in(node_id_list))
            .collect(engine="streaming")
        ).rename({bg_node_id_col_name: "node_id"})

        node_feat_dim = len(filtered_node_features.columns) - 1
        print(f"  Extracted: {len(filtered_node_features):,} rows | {node_feat_dim} feature dims | {time.time() - t0:.1f}s")

        atomic_write_parquet(filtered_node_features, node_features_path)
        print(f"  Saved node_features.parquet")

    # ================================================================
    # STEP 5: Extract edge features from background_edges.csv
    # ================================================================
    edge_features_path = f"{out_dir}/edge_features.parquet"

    print("\n" + "=" * 60)
    print("[5/6] Edge features from background_edges.csv...")
    print("=" * 60)

    if os.path.exists(edge_features_path):
        _check = pl.read_parquet(edge_features_path, n_rows=1)
        if "txId" not in _check.columns:
            print("  Old edge_features.parquet without txId — removing, re-extracting...")
            os.remove(edge_features_path)
        else:
            print("  edge_features.parquet exists — skipped (resume)")
            filtered_edge_features = pl.read_parquet(edge_features_path)
            edge_feat_dim = len(filtered_edge_features.columns) - 3  # minus txId, source, target

    if not os.path.exists(edge_features_path):
        print("  Scanning 77GB file... (2-10 min)")
        t0 = time.time()

        bg_edges_lazy = pl.scan_csv(f"{raw_dir}/background_edges.csv")
        bg_txid_col = col_cfg["background_edges"]["txId"]
        bg_src_col = col_cfg["background_edges"]["source"]
        bg_dst_col = col_cfg["background_edges"]["target"]
        print(f"  background_edges: {len(bg_edges_lazy.collect_schema().names())} columns | "
              f"txId='{bg_txid_col}', src='{bg_src_col}', dst='{bg_dst_col}'")

        # Single filter by txId
        filtered_edge_features = (
            bg_edges_lazy
            .filter(pl.col(bg_txid_col).is_in(txid_list))
            .collect(engine="streaming")
        )

        # Rename to standard names
        filtered_edge_features = filtered_edge_features.rename({
            bg_txid_col: "txId",
            bg_src_col: "source",
            bg_dst_col: "target",
        })

        edge_feat_dim = len(filtered_edge_features.columns) - 3  # minus txId, source, target
        print(f"  Extracted: {len(filtered_edge_features):,} edges | {edge_feat_dim} feature dims | {time.time() - t0:.1f}s")

        atomic_write_parquet(filtered_edge_features, edge_features_path)
        print(f"  Saved edge_features.parquet")

    # ================================================================
    # STEP 6: Validation & Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("[6/6] Validation & Summary...")
    print("=" * 60)

    found_nodes = set(filtered_node_features["node_id"].to_list())
    missing_nodes = node_id_set - found_nodes
    if missing_nodes:
        print(f"  {len(missing_nodes):,} nodes not found in background — will use zero vectors")

    found_txids = set(filtered_edge_features["txId"].to_list())
    missing_edges = txid_set - found_txids
    if missing_edges:
        print(f"  {len(missing_edges):,} edges (txIds) not found in background — will use zero vectors")

    if not missing_nodes and not missing_edges:
        print("  All nodes and edges matched")

    summary = {
        "num_subgraphs": len(components_df),
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "original_nodes": original_num_nodes,
        "original_edges": original_num_edges,
        "k_hop": k_hop,
        "node_feature_dims": node_feat_dim,
        "edge_feature_dims": edge_feat_dim,
        "labels": label_counts.to_dicts(),
        "missing_nodes": len(missing_nodes),
        "missing_edges": len(missing_edges),
    }
    atomic_write_json(summary, f"{out_dir}/summary.json")

    # ================================================================
    # DONE
    # ================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PREPROCESSING DONE — {total_time:.1f}s")
    print("=" * 60)
    print(f"\n  Output: {out_dir}/")
    print(f"  Files:  {os.listdir(out_dir)}")
    print(f"  Node features: {node_feat_dim} dims | Edge features: {edge_feat_dim} dims")
    if k_hop > 0:
        print(f"  k-hop expansion: {original_num_nodes:,} -> {len(nodes_df):,} nodes | "
              f"{original_num_edges:,} -> {len(edges_df):,} edges")


if __name__ == "__main__":
    main()
