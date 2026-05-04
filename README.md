# Entity Risk Scoring and Adversarial Robustness of GNNs for Bitcoin AML

Master's thesis — benchmarking Graph Neural Network architectures for Anti-Money Laundering detection on the **Elliptic2** Bitcoin dataset, with full adversarial robustness evaluation across six attack paradigms and a production-ready entity risk scoring interface.

## Overview

Bitcoin transaction graphs encoded as UTXO subgraphs contain structural and behavioral signals that distinguish licit from suspicious entity clusters. This project trains and evaluates four GNN architectures for **subgraph-level entity risk scoring on Bitcoin UTXO transaction graphs** (licit vs. suspicious), then systematically stress-tests each model under an escalating threat model — from full white-box gradient access down to topology-only manipulation.

**Dataset:** [Elliptic2](https://www.kaggle.com/datasets/ellipticco/elliptic2-data-set) — real-world Bitcoin UTXO transaction subgraphs labeled by [Elliptic](https://www.elliptic.co/), a blockchain intelligence firm used by financial institutions and regulators. Each subgraph is a connected component of Bitcoin addresses sharing transaction activity — the natural unit of entity analysis in UTXO-based blockchains.

**Bitcoin UTXO context:** In Bitcoin's Unspent Transaction Output model, funds are not stored in accounts but in UTXOs tied to addresses. An entity (individual, exchange, mixer) typically controls multiple addresses linked by co-spending behavior. Elliptic2 encodes these address clusters as attributed graphs — the natural input for graph-level AML classification.

## Entity Risk Scoring

The trained models are exposed through a calibrated risk scoring interface that produces **continuous risk scores (0–100)** with percentile ranks and confidence tiers — directly usable by compliance teams for alert triage.

```python
from src.models.risk_scorer import SubgraphRiskScorer

# Load the most robust model (SAGE+Edge, seed=42)
# Selected for lowest adversarial attack surface and lowest variance across seeds
scorer = SubgraphRiskScorer.best_model()

risk = scorer.score(subgraph)
# RiskScore(prob_illicit=0.978, risk_score=98, percentile=0.985,
#           tier='CRITICAL', prediction=1, threshold=0.04)
```

| Field | Description |
|---|---|
| `prob_illicit` | Calibrated P(suspicious) via temperature scaling — interpretable as confidence |
| `risk_score` | Integer 0–100, directly mappable to alert thresholds |
| `percentile` | Rank relative to test set distribution — useful for prioritizing investigation queues |
| `tier` | LOW / MEDIUM / HIGH / CRITICAL — for compliance workflow routing |

**Design choices:** Cost-sensitive loss (FN:FP = 25:1) reflects real AML economics where missing a suspicious entity is far more costly than a false alarm. Threshold is optimized on the validation set and fixed for test evaluation (no leakage). False positive rate < 0.6% across all architectures.

### Inference Interface (for downstream systems)

The `GNNInferenceOutput` contract exposes the full inference result as a JSON-serializable dataclass — no PyTorch dependency required in consumers. Designed as the stable interface between this thesis and the cross-chain clustering and RAG-based AML assistant systems.

```python
from src.inference.export import run_inference

output = run_inference(subgraph, conv_type="sage_edge", seed=42, compute_ig=True)
print(output.to_json())
# {
#   "subgraph_id": "...",
#   "class_prob_illicit": 0.978,
#   "risk_score": 98,
#   "tier": "CRITICAL",
#   "ig_scores": [0.12, -0.04, 0.31, ...],   # Integrated Gradients per feature
#   "topology_metrics": {"num_nodes": 6, "diameter": 4, "avg_clustering": 0.0, ...},
#   "subgraph_edges": [[0, 1], [1, 2], ...]
# }
```

## Model Performance

Multi-seed evaluation (seeds: 42, 123, 456) on held-out test set:

| Model | F1 Macro | Recall | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **GIN** | **0.957 ± 0.003** | 0.947 ± 0.016 | 0.997 ± 0.001 | **0.982 ± 0.017** | 0.949 ± 0.018 |
| **SAGE+Edge** | 0.954 ± 0.003 | 0.931 ± 0.019 | 0.997 ± 0.001 | 0.958 ± 0.014 | 0.938 ± 0.006 |
| SAGE | 0.938 ± 0.011 | 0.912 ± 0.026 | 0.996 ± 0.001 | 0.946 ± 0.018 | 0.916 ± 0.029 |
| GATv2 | 0.918 ± 0.020 | 0.905 ± 0.036 | 0.994 ± 0.001 | 0.932 ± 0.023 | 0.883 ± 0.046 |

GIN and SAGE+Edge are statistically tied on F1 (p > 0.05, bootstrap CI). **SAGE+Edge is recommended for production deployment**: lowest variance across seeds (std=0.003) and highest adversarial robustness. Specificity >0.994 means fewer than 0.6% of licit entities are incorrectly flagged.

**Training details:** Cost-sensitive Focal Loss (FN:FP = 25:1), balanced batch sampling, cosine LR schedule with warmup, temperature scaling calibration, k=1 hop subgraph expansion.

## Adversarial Robustness — Stress Testing Against Real-World Obfuscation

Six attack paradigms cover the full threat model spectrum a deployed AML system would face:

| Attack | Type | Threat Model | Key Finding |
|---|---|---|---|
| **PGD** | Feature perturbation | White-box (full gradient access) | SAGE+Edge max ASR 24.1%; GATv2 recall drops to 9.4% at ε=0.5 |
| **Transfer** | Feature perturbation | Gray-box (surrogate GCN) | GCN surrogate achieves 20.8% avg ASR; SAGE+Edge hardest to fool (15.3%) |
| **ZOO** | Feature perturbation | Black-box (query access only) | ZOO-inspired: gradients estimated via finite differences on random feature subspace, sign-based update (departs from ADAM in original ZOO); ASR grows with query budget (100→2000 queries) |
| **Topology Augmentation** | Structure manipulation | Black-box (graph edit) | GATv2 most vulnerable (65.3% ASR at chain_injection_k10); chain_injection most effective technique overall; star adds only 1 hub node regardless of k — simulates centralized mixer / exchange hub pattern |
| **Node Injection** | Structure manipulation | Black-box (node insertion) | Random injection most effective across all models (up to 76.1% ASR on GATv2); mimicry (licit-looking nodes) least effective — model is not fooled by licit features alone |
| **Saliency-guided** | Feature perturbation | White-box (guided by IG) | Integrated Gradients identifies top-K features; targeted perturbation vs. random baseline |

**Robustness ranking (most → least robust):** SAGE+Edge > GIN > SAGE > GATv2

### Topology Augmentation Attacks as Bitcoin Obfuscation Simulation

Topology rewiring attacks are direct simulations of real-world Bitcoin obfuscation techniques used by illicit actors to evade blockchain analytics:

- `chain_injection` — **chain injection**: injects k intermediary nodes between the two endpoints of the diameter path, creating a longer alternative route. Simulates layering (hop-based fund routing) in AML typology. Because Bitcoin transactions are immutable, the original path is preserved — this is a purely additive structural perturbation, not a replacement of the original route.
- `star` — **hub-based mixing**: one central node connected to many targets, simulating a **centralized mixer / exchange hub** (hub-and-spoke topology, analogous to Uniswap-style AMM or custodial tumbler). Adds only 1 hub node regardless of budget — lowest node footprint of all techniques.
- `parallel` — **transaction splitting**: parallel paths between source and destination, analogous to smurfing / structuring.

**Implication for deployment:** SAGE+Edge's explicit edge feature aggregation (`scatter_mean` over transaction-level features) acts as a natural structural defense — feature-space attacks cannot perturb edge attributes, and edge information dampens topology perturbations. This makes SAGE+Edge the recommended architecture for production AML systems.

## Feature Importance

Three complementary interpretability methods identify which behavioral and topological signals drive "suspicious" classifications:

- **Gradient Saliency** — |∂L/∂x| averaged over suspicious subgraphs: fast, identifies globally important features
- **Integrated Gradients** — path integral attribution from zero baseline: attribution respects counterfactual reasoning

Features are anonymized per Elliptic's data policy (labeled F1–F43), but IG attribution clusters reveal consistent behavioral signatures across suspicious subgraphs — accessible via `src/inference/export.py` with `compute_ig=True`.

## Architecture

Four GNN architectures, all with attention-based graph pooling, multi-layer residual connections, and Jumping Knowledge aggregation:

```
GATv2        — Graph Attention v2 (multi-head, joint edge+node attention)
SAGE         — GraphSAGE (mean neighborhood aggregation)
SAGE+Edge    — GraphSAGE with explicit edge feature aggregation (scatter_mean)
GIN          — Graph Isomorphism Network (MLP aggregator, trainable ε)
```

Custom components:
- `AttentionPooling` — learned node importance weights for graph-level readout
- `EdgeProjection` — linear projection of transaction-level edge features into node space
- `SurrogateGCN` — 2-layer GCN used as surrogate for gray-box transfer attacks
- `BalancedBatchSampler` — per-batch class balancing without weight duplication
- `SubgraphRiskScorer` — calibrated risk scoring with percentile ranking and tier classification
- `GNNInferenceOutput` — JSON-serializable inference contract for downstream consumers

## Project Structure

```
src/
├── dataset/        # Elliptic2Dataset — PyG Data objects, stratified split, caching
├── preprocess/     # Raw CSV → Parquet pipeline, k-hop subgraph expansion
├── models/         # GNN architectures, calibration, losses, surrogate GCN, risk scorer
├── train/          # Training loop, threshold optimization, plotting, samplers
├── attacks/        # PGD, Transfer, ZOO, Topology Augmentation, Node Injection, Saliency
├── analysis/       # Feature importance (Gradient Saliency, IG, Ablation)
└── inference/      # Stable GNNInferenceOutput contract for downstream systems

scripts/
├── pgd/            # run_pgd_attack.py, evaluate_pgd_attack.py
├── transfer/       # run_transfer_attack.py, evaluate_transfer_attack.py
├── zoo/            # run_zoo_attack.py, evaluate_zoo_attack.py
├── topology/       # run_topology_attack.py, evaluate_topology_attack.py
├── injection/      # run_injection_attack.py, evaluate_injection_attack.py
└── saliency/       # run_saliency_attack.py, evaluate_saliency_attack.py

data/results/       # JSON results + plots for all attacks
config.yaml         # Central configuration (models, training, attack params)
run_all_attacks.sh  # End-to-end attack pipeline script
```

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA-capable GPU. Tested with PyTorch 2.1, torch-geometric 2.7, CUDA 12.

**Data:** Download the Elliptic2 dataset and place CSV files under `data/raw/`:
- `nodes.csv`, `edges.csv`, `connected_components.csv`

```bash
# Preprocess (builds k-hop subgraphs, caches to Parquet)
python -m src.preprocess.preprocess

# Train all four architectures (3 seeds each)
python -m src.train.train

# Run all adversarial attacks
bash run_all_attacks.sh
```

Attack configuration (epsilons, steps, strategies) is controlled via `config.yaml`.

## Key Results

- **Best detection performance:** GIN (F1=0.957, ROC-AUC=0.982)
- **Best for deployment:** SAGE+Edge — highest adversarial robustness (max ASR 24.1% under PGD ε=0.5), lowest variance across seeds (std=0.003)
- **Most vulnerable:** GATv2 — attention mechanism amplifies perturbations (ASR 87.8% at ε=0.5, 65.3% under topology rewiring)
- **Structural attacks dominate:** At comparable budgets, topology augmentation (simulating CoinJoin/chain layering) is more effective than feature perturbation — 41.3% vs 30.5% ASR on GATv2 (parallel_k5 vs PGD ε=0.1)
- **Edge features as natural defense:** SAGE+Edge's transaction-level edge aggregation limits attack surface — feature attacks cannot perturb edge attributes

## Stack

Python · PyTorch · PyTorch Geometric · torch-scatter · NetworkX · Polars · scikit-learn · NumPy · Matplotlib · Seaborn
