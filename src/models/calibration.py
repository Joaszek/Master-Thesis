"""
calibration.py — Post-hoc Temperature Scaling
==============================================
Learns a single temperature parameter on validation set
to improve probability calibration without changing predictions ranking.

Usage:
    temperature = fit_temperature(model, val_loader, device)
    calibrated_probs = calibrate_probabilities(logits, temperature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """Learnable temperature parameter for logit calibration."""

    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    """Collect raw logits and labels from a data loader."""
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.squeeze().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(model, val_loader, device, max_iter=50, lr=0.01):
    """
    Fit temperature parameter on validation set using NLL loss.

    Args:
        model: Trained model (frozen)
        val_loader: Validation DataLoader
        device: torch device
        max_iter: LBFGS iterations
        lr: Learning rate for LBFGS

    Returns:
        float: Learned temperature value
    """
    logits, labels = collect_logits_labels(model, val_loader, device)

    temp_model = TemperatureScaling(init_temp=1.5)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    learned_temp = temp_model.temperature.item()
    print(f"    Learned temperature: {learned_temp:.4f}")

    # Report calibration improvement
    with torch.no_grad():
        nll_before = F.cross_entropy(logits, labels).item()
        nll_after = F.cross_entropy(logits / learned_temp, labels).item()
        print(f"    NLL before: {nll_before:.4f} | after: {nll_after:.4f}")

    return learned_temp


def calibrate_probabilities(logits, temperature):
    """
    Apply temperature scaling to logits and return calibrated probabilities.

    Args:
        logits: [N, C] raw logits (numpy or tensor)
        temperature: Learned temperature value

    Returns:
        numpy array of calibrated probabilities for positive class
    """
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits, dtype=torch.float32)

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)[:, 1].numpy()
    return probs
