import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score
)
import torch

def cost_aware_threshold_search(all_probs, all_labels, fn_cost=10.0, fp_cost=1.0):
    """
    Optymalizuj threshold minimalizując total cost.

    Args:
        all_probs: probabilities dla klasy pozytywnej [N]
        all_labels: true labels [N]
        fn_cost: Koszt False Negative (miss fraud)
        fp_cost: Koszt False Positive (false alarm)

    Returns:
        best_threshold, best_cost, best_f1
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_cost = float('inf')
    best_f1 = 0.0

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)

        # Confusion matrix elements
        tn = np.sum((all_labels == 0) & (preds == 0))
        fp = np.sum((all_labels == 0) & (preds == 1))
        fn = np.sum((all_labels == 1) & (preds == 0))
        tp = np.sum((all_labels == 1) & (preds == 1))

        total_cost = fn * fn_cost + fp * fp_cost
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)

        # Optymalizuj po total cost, f1 jako tiebreaker
        if total_cost < best_cost or (total_cost == best_cost and f1 > best_f1):
            best_cost = total_cost
            best_threshold = thresh
            best_f1 = f1

    return best_threshold, best_cost, best_f1


@torch.no_grad()
def evaluate_with_threshold_search(model, loader, criterion, device, num_classes=2,
                                   fn_cost=None, fp_cost=None):
    """
    Evaluate z optymalizacją threshold dla klasy pozytywnej.

    Args:
        fn_cost, fp_cost: Jeśli podane, użyj cost-aware threshold, inaczej max F1-macro
    """
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item()
        n_batches += 1

        # Pobierz probability dla klasy 1 (suspicious)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Threshold optimization
    if fn_cost is not None and fp_cost is not None:
        # Cost-aware threshold
        best_threshold, total_cost, best_f1 = cost_aware_threshold_search(
            all_probs, all_labels, fn_cost, fp_cost
        )
    else:
        # F1-macro optimization
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_f1 = 0.0

        for thresh in thresholds:
            preds = (all_probs >= thresh).astype(int)
            f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

    # Final predictions z optymalnym threshold
    preds = (all_probs >= best_threshold).astype(int)

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, preds)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1_macro, f1_weighted, preds, all_labels, best_threshold, all_probs


@torch.no_grad()
def evaluate_with_fixed_threshold(model, loader, criterion, device, threshold,
                                  temperature=1.0):
    """
    Evaluate using a FIXED threshold (found on validation set).
    Optionally applies temperature scaling to logits before softmax.

    This prevents test-set leakage from threshold optimization.
    """
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item()
        n_batches += 1

        # Apply temperature scaling before softmax
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Apply fixed threshold
    preds = (all_probs >= threshold).astype(int)

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, preds)
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1_macro, f1_weighted, preds, all_labels, all_probs
