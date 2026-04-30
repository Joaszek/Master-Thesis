from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.models.model import EllipticGNN


@dataclass
class RiskScore:
    prob_illicit: float
    risk_score: int
    percentile: float
    tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    prediction: int
    threshold: float


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_model_meta(results_path: str, conv_type: str, seed: int) -> dict:
    with open(results_path) as f:
        results = json.load(f)
    for r in results:
        if r["conv_type"] == conv_type and r["seed"] == seed:
            return r
    raise ValueError(f"No results found for conv_type={conv_type}, seed={seed}")


def _tier(risk_score: int) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    if risk_score < 25:
        return "LOW"
    elif risk_score < 50:
        return "MEDIUM"
    elif risk_score < 75:
        return "HIGH"
    return "CRITICAL"


class SubgraphRiskScorer:

    def __init__(
        self,
        conv_type: str,
        seed: int = 42,
        config_path: str = "config.yaml",
        results_path: str = "data/results/full_results.json",
        checkpoint_dir: str = "checkpoints",
        device: str | None = None,
    ) -> None:
        self.conv_type = conv_type
        self.seed = seed
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = _load_config(config_path)
        meta = _get_model_meta(results_path, conv_type, seed)

        self.threshold: float = meta["threshold"]
        self.temperature: float = meta["temperature"]
        self._test_probs: np.ndarray = np.array(meta["test_probs"])

        ckpt_path = os.path.join(checkpoint_dir, f"seed_{seed}", conv_type, "best_model.pt")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        # Checkpoints are saved as raw state_dicts (OrderedDict), not wrapped dicts
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model_cfg = config["model"]
        node_feat_dim = state_dict["input_proj.0.weight"].shape[1]
        if conv_type in ("gatv2", "sage_edge"):
            edge_feat_dim = state_dict["edge_proj.proj.0.weight"].shape[1]
        else:
            edge_feat_dim = 1  # unused for sage/gin

        self.model = EllipticGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            heads=model_cfg["heads"],
            edge_proj_dim=model_cfg["edge_proj_dim"],
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
            conv_type=conv_type,
            expansion_node_weight=model_cfg.get("expansion_node_weight", 1.0),
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def score(self, data) -> RiskScore:

        d = data.clone().to(self.device)
        if not hasattr(d, "batch") or d.batch is None:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=self.device)

        logits = self.model(d)
        scaled = logits / self.temperature
        prob = float(F.softmax(scaled, dim=-1)[0, 1].cpu())

        risk_score = int(round(prob * 100))
        risk_score = max(0, min(100, risk_score))

        percentile = float(np.mean(self._test_probs < prob))
        prediction = int(prob >= self.threshold)

        return RiskScore(
            prob_illicit=prob,
            risk_score=risk_score,
            percentile=percentile,
            tier=_tier(risk_score),
            prediction=prediction,
            threshold=self.threshold,
        )

    def score_batch(self, data_list: list) -> list[RiskScore]:
        return [self.score(d) for d in data_list]

    @classmethod
    def best_model(
        cls,
        config_path: str = "config.yaml",
        results_path: str = "data/results/full_results.json",
        checkpoint_dir: str = "checkpoints",
        device: str | None = None,
    ) -> "SubgraphRiskScorer":
        
        return cls(
            conv_type="sage_edge",
            seed=42,
            config_path=config_path,
            results_path=results_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
