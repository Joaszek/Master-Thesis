from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.models.risk_scorer import SubgraphRiskScorer
from src.attacks.topology_rewiring import TopologyAnalyzer



@dataclass
class TopologyMetrics:
    num_nodes: int
    num_edges: int
    diameter: int
    avg_clustering: float
    avg_degree: float


@dataclass
class GNNInferenceOutput:
    subgraph_id: str
    conv_type: str
    seed: int
    class_prob_illicit: float
    risk_score: int
    percentile: float
    tier: str
    prediction: int
    threshold: float
    ig_scores: list[float] | None
    topology_metrics: TopologyMetrics
    subgraph_edges: list[tuple[int, int]]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _compute_ig(model, data, device: torch.device, n_steps: int = 50) -> list[float]:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d = data.clone().to(device)
    if not hasattr(d, "batch") or d.batch is None:
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)

    x_orig = d.x.float()
    baseline = torch.zeros_like(x_orig)

    ig = torch.zeros(x_orig.shape[1], device=device)
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        x_interp = (baseline + alpha * (x_orig - baseline)).detach().requires_grad_(True)
        d.x = x_interp
        logits = model(d)
        score = logits[0, 1]
        score.backward()
        ig += x_interp.grad.abs().mean(dim=0)

    ig = (ig / n_steps) * (x_orig - baseline).abs().mean(dim=0)
    for p in model.parameters():
        p.requires_grad_(True)
    return ig.detach().cpu().tolist()


def run_inference(
    data,
    conv_type: str = "sage_edge",
    seed: int = 42,
    subgraph_id: str | None = None,
    compute_ig: bool = False,
    ig_steps: int = 50,
    config_path: str = "config.yaml",
    results_path: str = "data/results/full_results.json",
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
) -> GNNInferenceOutput:
    _device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    scorer = SubgraphRiskScorer(
        conv_type=conv_type,
        seed=seed,
        config_path=config_path,
        results_path=results_path,
        checkpoint_dir=checkpoint_dir,
        device=str(_device),
    )

    risk = scorer.score(data)

    topo_raw = TopologyAnalyzer.compute_metrics(data)
    topo = TopologyMetrics(
        num_nodes=topo_raw["num_nodes"],
        num_edges=topo_raw["num_edges"],
        diameter=topo_raw["diameter"],
        avg_clustering=topo_raw["avg_clustering"],
        avg_degree=topo_raw["avg_degree"],
    )

    ei = data.edge_index.cpu().numpy()
    edges = list(zip(ei[0].tolist(), ei[1].tolist()))

    ig_scores = None
    if compute_ig:
        ig_scores = _compute_ig(scorer.model, data, _device, n_steps=ig_steps)

    sid = subgraph_id if subgraph_id is not None else str(int(data.y.item()))

    return GNNInferenceOutput(
        subgraph_id=sid,
        conv_type=conv_type,
        seed=seed,
        class_prob_illicit=risk.prob_illicit,
        risk_score=risk.risk_score,
        percentile=risk.percentile,
        tier=risk.tier,
        prediction=risk.prediction,
        threshold=risk.threshold,
        ig_scores=ig_scores,
        topology_metrics=topo,
        subgraph_edges=edges,
    )
