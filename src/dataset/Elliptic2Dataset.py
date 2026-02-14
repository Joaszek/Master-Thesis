"""
dataset.py — Elliptic2 PyTorch Geometric Dataset
=================================================
Wczyta preprocessed parquet files i buduje listę PyG Data objectów —
jeden per subgraf. Każdy subgraf to oddzielny mały graf (~3-4 nody).

Używa stratified split aby zachować proporcje klas w train/val/test.
Cachuje zbudowane grafy do .pt pliku (eliminuje powtórne ładowanie).

Jeśli preprocessing nie został uruchomiony, dostaniesz error.
Uruchomij najpierw: python preprocess.py
"""

import os
import json
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class Elliptic2Dataset(Dataset):
    """
    Dataset dla Elliptic2 — subgraf klasyfikacja.

    Każdy subgraf (connected component) to jeden Data object:
      - x:          node features [num_nodes, node_feat_dim]
      - edge_index: krawędzie [2, num_edges]
      - edge_attr:  edge features [num_edges, edge_feat_dim]
      - y:          label subgrafu [1]  (binary: 0=legit, 1=illicit)

    Dane budowane raz i cachowane w all_graphs.pt.
    Split stratyfikowany — proporcje klas zachowane w każdym splicie.
    """

    _cache = {}  # class-level cache: processed_dir -> data_list

    def __init__(self, processed_dir, split="train", val_ratio=0.15, test_ratio=0.10):
        self.processed_dir = processed_dir
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Load (z cache lub z dysku)
        all_data = self._get_all_data()

        # Stratified split
        self.data_list = self._stratified_split(all_data)

        # Report
        labels = [d.y.item() for d in self.data_list]
        counts = np.bincount(labels, minlength=2)
        print(f"  [{split.upper()}] {len(self.data_list)} subgraphs "
              f"(class 0: {counts[0]}, class 1: {counts[1]})")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_labels(self):
        """Zwraca listę labeli — potrzebne do WeightedRandomSampler."""
        return [d.y.item() for d in self.data_list]

    def _get_all_data(self):
        """Load all graphs — z class cache, .pt pliku, lub buduje od zera."""
        # 1. Class-level RAM cache (dla kolejnych splitów w tym samym procesie)
        if self.processed_dir in Elliptic2Dataset._cache:
            return Elliptic2Dataset._cache[self.processed_dir]

        # 2. Disk cache (.pt plik) — invalidate if k_hop changed
        cache_path = os.path.join(self.processed_dir, "all_graphs.pt")
        summary_path = os.path.join(self.processed_dir, "summary.json")
        cache_meta_path = os.path.join(self.processed_dir, "cache_meta.json")

        # Check if cache is stale (k_hop mismatch)
        cache_valid = os.path.exists(cache_path)
        if cache_valid and os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            current_k_hop = summary.get("k_hop", 0)
            cached_k_hop = None
            if os.path.exists(cache_meta_path):
                with open(cache_meta_path) as f:
                    cached_k_hop = json.load(f).get("k_hop", None)
            if cached_k_hop is not None and cached_k_hop != current_k_hop:
                print(f"  Cache stale: k_hop changed ({cached_k_hop} -> {current_k_hop}), rebuilding...")
                os.remove(cache_path)
                cache_valid = False

        if cache_valid:
            print("  Loading cached graphs from all_graphs.pt...")
            data_list = torch.load(cache_path, weights_only=False)
            print(f"  Loaded {len(data_list)} graphs from cache")
            Elliptic2Dataset._cache[self.processed_dir] = data_list
            return data_list

        # 3. Build from parquets
        data_list = self._load_and_build()

        # Save cache + metadata for invalidation
        print(f"  Saving graph cache to {cache_path}...")
        torch.save(data_list, cache_path)

        # Save cache metadata (k_hop) for future invalidation checks
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            with open(cache_meta_path, "w") as f:
                json.dump({"k_hop": summary.get("k_hop", 0)}, f)

        Elliptic2Dataset._cache[self.processed_dir] = data_list
        return data_list

    def _load_and_build(self):
        """Load parquet files i buduje Data objecty."""
        print("  Loading preprocessed data...")

        # Load summary to check dims
        with open(f"{self.processed_dir}/summary.json") as f:
            summary = json.load(f)
        node_feat_dim = summary["node_feature_dims"]
        edge_feat_dim = summary["edge_feature_dims"]
        print(f"    Node features: {node_feat_dim} dims")
        print(f"    Edge features: {edge_feat_dim} dims")

        # Load parquet files
        nodes_df = pl.read_parquet(f"{self.processed_dir}/nodes.parquet")
        edges_df = pl.read_parquet(f"{self.processed_dir}/edges.parquet")
        components_df = pl.read_parquet(f"{self.processed_dir}/components.parquet")
        node_features_df = pl.read_parquet(f"{self.processed_dir}/node_features.parquet")
        edge_features_df = pl.read_parquet(f"{self.processed_dir}/edge_features.parquet")

        print(f"    Nodes: {len(nodes_df):,} | Edges: {len(edges_df):,} | Subgraphs: {len(components_df):,}")

        # ============================================================
        # Build lookup tables
        # ============================================================

        # node_id -> features (numpy array)
        node_feat_cols = [c for c in node_features_df.columns if c != "node_id"]
        node_id_to_features = {}
        node_features_np = node_features_df.select(node_feat_cols).to_numpy().astype(np.float32)
        node_ids_arr = node_features_df["node_id"].to_numpy()
        for i, nid in enumerate(node_ids_arr):
            node_id_to_features[nid] = node_features_np[i]

        # txId -> features
        edge_feat_cols = [c for c in edge_features_df.columns if c not in ("source", "target", "txId")]
        edge_to_features = {}
        if len(edge_feat_cols) > 0:
            edge_features_np = edge_features_df.select(edge_feat_cols).to_numpy().astype(np.float32)
            txid_arr = edge_features_df["txId"].to_numpy()
            for i, tid in enumerate(txid_arr):
                edge_to_features[tid] = edge_features_np[i]

        # subgraph_id -> label (as int)
        subgraph_labels = dict(zip(
            components_df["subgraph_id"].to_list(),
            [int(x) for x in components_df["label"].to_list()]
        ))

        # subgraph_id -> list of node_ids
        subgraph_to_nodes = {}
        for row in nodes_df.iter_rows(named=True):
            sg_id = row["subgraph_id"]
            if sg_id not in subgraph_to_nodes:
                subgraph_to_nodes[sg_id] = []
            subgraph_to_nodes[sg_id].append(row["node_id"])

        # subgraph_id -> list of (src, dst, txId)
        subgraph_to_edges = {}
        has_sg_col = "subgraph_id" in edges_df.columns
        for row in edges_df.iter_rows(named=True):
            if has_sg_col:
                sg_id = row["subgraph_id"]
            else:
                # Fallback for old edges.parquet without subgraph_id
                src = row["source"]
                sg_id = None
                for sg, nodes in subgraph_to_nodes.items():
                    if src in nodes:
                        sg_id = sg
                        break
            if sg_id is not None:
                if sg_id not in subgraph_to_edges:
                    subgraph_to_edges[sg_id] = []
                subgraph_to_edges[sg_id].append((row["source"], row["target"], row["txId"]))

        # ============================================================
        # Build Data objects per subgraph
        # ============================================================
        print("    Building PyG Data objects...")
        zero_node_feat = np.zeros(node_feat_dim, dtype=np.float32)
        zero_edge_feat = np.zeros(edge_feat_dim, dtype=np.float32)

        data_list = []
        skipped = 0

        for sg_id in components_df["subgraph_id"].to_list():
            if sg_id not in subgraph_to_nodes:
                skipped += 1
                continue

            node_ids_in_sg = subgraph_to_nodes[sg_id]
            edges_in_sg = subgraph_to_edges.get(sg_id, [])
            label = subgraph_labels[sg_id]

            # Map global node IDs → local indices [0, 1, 2, ...]
            global_to_local = {nid: i for i, nid in enumerate(node_ids_in_sg)}

            # Node features
            x_list = []
            for nid in node_ids_in_sg:
                x_list.append(node_id_to_features.get(nid, zero_node_feat))
            x = torch.tensor(np.stack(x_list), dtype=torch.float32)

            # Edge index + edge features
            edge_index_list = []
            edge_attr_list = []
            for src, dst, txid in edges_in_sg:
                if src in global_to_local and dst in global_to_local:
                    edge_index_list.append([global_to_local[src], global_to_local[dst]])
                    edge_attr_list.append(
                        edge_to_features.get(txid, zero_edge_feat)
                    )

            if len(edge_index_list) == 0:
                # Subgraf bez krawędzi — dodaj self-loops
                for i in range(len(node_ids_in_sg)):
                    edge_index_list.append([i, i])
                    edge_attr_list.append(zero_edge_feat)

            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # [2, E]
            edge_attr = torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)  # [E, F]

            # Label
            y = torch.tensor([label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        if skipped > 0:
            print(f"    Skipped {skipped} subgraphs (no nodes found)")

        print(f"    Built {len(data_list)} Data objects")

        # Quick stats
        sizes = [d.num_nodes for d in data_list]
        labels = [d.y.item() for d in data_list]
        counts = np.bincount(labels, minlength=2)
        print(f"    Avg nodes/subgraph: {np.mean(sizes):.2f} | Max: {max(sizes)} | Min: {min(sizes)}")
        print(f"    Label distribution: class 0={counts[0]:,}, class 1={counts[1]:,} "
              f"(ratio {counts[0]/max(counts[1],1):.1f}:1)")

        return data_list

    def _stratified_split(self, data_list):
        """
        Stratified split zachowujący proporcje klas w train/val/test.
        Deterministyczny (seed=42).
        """
        labels = np.array([d.y.item() for d in data_list])
        indices = np.arange(len(data_list))

        # Najpierw oddziel test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_ratio,
            stratify=labels, random_state=42
        )

        # Potem z train_val oddziel val
        val_ratio_adjusted = self.val_ratio / (1.0 - self.test_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio_adjusted,
            stratify=labels[train_val_idx], random_state=42
        )

        if self.split == "train":
            selected = train_idx
        elif self.split == "val":
            selected = val_idx
        elif self.split == "test":
            selected = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [data_list[i] for i in selected]
