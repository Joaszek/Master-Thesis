"""
dataset.py — Elliptic2 PyTorch Geometric Dataset
=================================================
Wczyta preprocessed parquet files i buduje listę PyG Data objectów —
jeden per subgraf. Każdy subgraf to oddzielny mały graf (~3-4 nody).

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


class Elliptic2Dataset(Dataset):
    """
    Dataset dla Elliptic2 — subgraf klasyfikacja.

    Każdy subgraf (connected component) to jeden Data object:
      - x:          node features [num_nodes, node_feat_dim]
      - edge_index: krawędzie [2, num_edges]
      - edge_attr:  edge features [num_edges, edge_feat_dim]
      - y:          label subgrafu [1]  (binary: 0=legit, 1=illicit)
    """

    def __init__(self, processed_dir, split="train", val_ratio=0.15, test_ratio=0.10):
        """
        Args:
            processed_dir: path do folderu z parquet files (data/processed/)
            split:         "train", "val", lub "test"
            val_ratio:     procent danych do validation
            test_ratio:    procent danych do testu
        """
        self.processed_dir = processed_dir
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Load data
        data_list = self._load_and_build()

        # Split
        self.data_list = self._get_split(data_list)

        # Report
        print(f"  [{split.upper()}] Loaded {len(self.data_list)} subgraphs")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

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
            [int(x) for x in components_df["label"].to_list()]  # force int conversion
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
        for row in edges_df.iter_rows(named=True):
            sg_id = row.get("subgraph_id", None)
            if sg_id is None:
                src = row["source"]
                for sg, nodes in subgraph_to_nodes.items():
                    if src in nodes:
                        sg_id = sg
                        break
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
            print(f"    ⚠️  Skipped {skipped} subgraphs (no nodes found)")

        print(f"    Built {len(data_list)} Data objects")

        # Quick stats
        sizes = [d.num_nodes for d in data_list]
        print(f"    Avg nodes/subgraph: {np.mean(sizes):.2f} | Max: {max(sizes)} | Min: {min(sizes)}")

        return data_list

    def _get_split(self, data_list):
        """
        Split data na train/val/test deterministycznie.
        Shuffle z fixed seed aby split był reproducible.
        """
        n = len(data_list)
        indices = list(range(n))

        # Deterministyczny shuffle
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        elif self.split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [data_list[i] for i in selected]