"""
ensemble.py — Ensemble prediction utilities
============================================
Train multiple models with different seeds and average their predictions
for improved robustness and reduced variance.
"""

import os
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def ensemble_predict(models, loader, device):
    """
    Ensemble prediction from multiple models.

    Args:
        models: List of trained models
        loader: DataLoader
        device: torch device

    Returns:
        ensemble_probs: [N] averaged probabilities for positive class
        all_labels: [N] true labels
    """
    all_model_probs = []
    all_labels = None

    for i, model in enumerate(models):
        model.eval()
        model.to(device)

        probs = []
        labels = []

        for batch in tqdm(loader, desc=f"Model {i+1}/{len(models)}", leave=False):
            batch = batch.to(device)
            logits = model(batch)

            # Get probability for positive class (illicit)
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs)
            labels.extend(batch.y.squeeze().cpu().numpy())

        all_model_probs.append(np.array(probs))

        if all_labels is None:
            all_labels = np.array(labels)

    # Average probabilities across models
    ensemble_probs = np.mean(all_model_probs, axis=0)

    return ensemble_probs, all_labels


def train_ensemble(train_fn, num_models=5, seeds=None):
    """
    Train multiple models with different random seeds.

    Args:
        train_fn: Function that trains and returns a model
                  Should accept seed parameter
        num_models: Number of models to train
        seeds: Optional list of seeds (if None, uses [42, 123, 456, 789, 2024])

    Returns:
        models: List of trained models
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024][:num_models]

    models = []

    print(f"\n{'='*60}")
    print(f"Training ensemble of {num_models} models")
    print(f"{'='*60}")

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{num_models}] Training model with seed={seed}")
        print("-" * 60)

        model = train_fn(seed=seed)
        models.append(model)

    print(f"\n{'='*60}")
    print(f"Ensemble training complete!")
    print(f"{'='*60}\n")

    return models


def save_ensemble(models, checkpoint_dir, prefix="ensemble"):
    """Save ensemble models to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i, model in enumerate(models):
        path = os.path.join(checkpoint_dir, f"{prefix}_model_{i}.pt")
        torch.save(model.state_dict(), path)
        print(f"  Saved {path}")


def load_ensemble(model_class, checkpoint_dir, num_models, prefix="ensemble", **model_kwargs):
    """
    Load ensemble models from disk.

    Args:
        model_class: Model class (e.g., EllipticGNN)
        checkpoint_dir: Directory with saved models
        num_models: Number of models to load
        prefix: Filename prefix
        **model_kwargs: Arguments for model initialization

    Returns:
        models: List of loaded models
    """
    models = []

    for i in range(num_models):
        path = os.path.join(checkpoint_dir, f"{prefix}_model_{i}.pt")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        models.append(model)
        print(f"  Loaded {path}")

    return models
