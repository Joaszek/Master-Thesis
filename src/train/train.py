import os
import sys
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import time
import signal
import shutil
import argparse
import yaml
import tempfile

import random

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler, Sampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score
)
from tqdm import tqdm

from src.dataset.Elliptic2Dataset import Elliptic2Dataset
from src.models.model import EllipticGNN
from src.models.losses import FocalLoss
from src.models.calibration import fit_temperature


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_GLOBAL_SAVE_STATE = {}
_SIGTERM_RECEIVED = False

def sigterm_handler(signum, frame):
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    print("\n" + "=" * 60)
    print("SIGTERM received — spot instance is shutting down!")
    print("   Saving checkpoint before exit...")
    print("=" * 60)

    if _GLOBAL_SAVE_STATE:
        try:
            atomic_save(_GLOBAL_SAVE_STATE["state"], _GLOBAL_SAVE_STATE["path"])
            print(f"   Checkpoint saved to {_GLOBAL_SAVE_STATE['path']}")
        except Exception as e:
            print(f"   Failed to save checkpoint: {e}")

    print("   Exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

def atomic_save(state, filepath):
    """
    Atomic save: pisze do temp file w TYM SAME folderze,
    potem os.replace() — jeśli pod się zabije mid-write,
    stary plik przetrwa, .tmp się wymazuje przy restarcie.
    """
    dirpath = os.path.dirname(filepath) or "."
    os.makedirs(dirpath, exist_ok=True)

    # Temp file w tym samym folderze (critical: same filesystem dla atomic replace)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        # Atomic replace: jeśli to się uda → plik jest kompletny
        os.replace(tmp_path, filepath)
    except Exception:
        # Jeśli coś poszło nie tak → wymazuj .tmp, stary plik bezpieczny
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def cleanup_tmp_files(dirpath):
    """Wymazuje stałe pliki .tmp z poprzednich przerwanych sesji."""
    if not os.path.exists(dirpath):
        return
    for f in os.listdir(dirpath):
        if f.endswith(".tmp"):
            os.remove(os.path.join(dirpath, f))
            print(f"  Cleaned up leftover: {f}")

def resolve_paths(config):
    """
    Jeśli spot_mode=true → checkpoints i processed data żyją
    na persistent_storage_path (przetrwa pod restart).
    """
    spot_mode = config.get("spot_mode", False)
    persist = config.get("persistent_storage_path", "/persistent")

    processed_dir = config["data"]["processed_dir"]
    checkpoint_dir = config["training"]["checkpoint_dir"]

    if spot_mode:
        # Przekieruj na persistent storage
        processed_dir = os.path.join(persist, "processed")
        checkpoint_dir = os.path.join(persist, "checkpoints")
        print(f"Spot mode ON — paths na persistent storage:")
        print(f"processed:   {processed_dir}")
        print(f"checkpoints: {checkpoint_dir}")
    else:
        print(f"Standard mode — paths lokalne:")
        print(f"processed:   {processed_dir}")
        print(f"checkpoints: {checkpoint_dir}")

    return processed_dir, checkpoint_dir

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset, num_classes, device):
    """Oblicza inverse-frequency wagi klas z training set."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = len(labels)

    # Inverse frequency: w_i = total / (num_classes * count_i)
    weights = total / (num_classes * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"    Class counts: {dict(enumerate(class_counts.astype(int)))}")
    print(f"    Class weights: {dict(enumerate(weights.round(2)))}")
    return weights_tensor

def print_comprehensive_metrics(y_true, y_pred, y_probs, target_names=None):
    """
    Print comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities for positive class
        target_names: Class names for display
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual  Neg  [{cm[0,0]:6d}  {cm[0,1]:6d}]")
    print(f"          Pos  [{cm[1,0]:6d}  {cm[1,1]:6d}]")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives:  {tn:6d}")
    print(f"  False Positives: {fp:6d}")
    print(f"  False Negatives: {fn:6d}")
    print(f"  True Positives:  {tp:6d}")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_arr = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    if target_names is None:
        target_names = ["Class 0", "Class 1"]

    for i, name in enumerate(target_names):
        print(f"  {name:12s}: Precision={precision[i]:.4f}, Recall={recall_arr[i]:.4f}, F1={f1[i]:.4f}")

    # Macro/Weighted averages
    print("\nAveraged Metrics:")
    print(f"  Precision (macro):   {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall (macro):      {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 (macro):          {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  F1 (weighted):       {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    # ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        print("\nArea Under Curve Metrics:")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  PR-AUC:   {pr_auc:.4f}")
    except ValueError as e:
        print(f"\nWarning: Could not compute AUC metrics: {e}")

    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print("\nAdditional Metrics:")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Accuracy:          {accuracy_score(y_true, y_pred):.4f}")

    print("=" * 80 + "\n")

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "specificity": specificity, "sensitivity": sensitivity,
    }


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

        # Pobierz probability dla klasy 1 (illicit)
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



class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler — każdy batch ma równą liczbę próbek z każdej klasy.
    Znacznie bardziej efektywne niż WeightedRandomSampler dla extreme imbalance.
    """
    def __init__(self, labels, batch_size, num_batches=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_batches = num_batches

        # Indeksy per klasa
        self.class_indices = {}
        for c in np.unique(labels):
            self.class_indices[c] = np.where(self.labels == c)[0].tolist()

        self.num_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.num_classes

        # If num_batches not specified, compute from majority class
        if self.num_batches is None:
            max_class_size = max(len(idx) for idx in self.class_indices.values())
            self.num_batches = max_class_size // self.samples_per_class

    def __iter__(self):
        # Shuffle indices per class
        for c in self.class_indices:
            np.random.shuffle(self.class_indices[c])

        # Pointers per class
        pointers = {c: 0 for c in self.class_indices}

        for _ in range(self.num_batches):
            batch = []
            for c in self.class_indices:
                indices = self.class_indices[c]
                n = len(indices)

                # Wrap around if needed (sampling with replacement)
                selected = []
                for _ in range(self.samples_per_class):
                    selected.append(indices[pointers[c] % n])
                    pointers[c] += 1

                batch.extend(selected)

            # Shuffle batch internally
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def make_weighted_sampler(dataset):
    """Buduje WeightedRandomSampler — oversampling klasy mniejszościowej."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)

    # Waga per-sample = 1 / count klasy danej próbki
    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model na validation/test set."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1_macro, f1_weighted, np.array(all_preds), np.array(all_labels)


def build_save_state(epoch, model, optimizer, scheduler, best_val_f1, history):
    """Buduje dict do checkpoint."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "history": history,
    }


# ============================================================
# Architecture display names
# ============================================================
ARCH_NAMES = {
    "gatv2": "GATv2",
    "sage": "SAGE",
    "sage_edge": "SAGE+Edge",
    "gin": "GIN",
}


def train_and_evaluate(conv_type, config, device, train_dataset,
                       train_loader, val_loader, test_loader, node_feat_dim, edge_feat_dim,
                       checkpoint_dir, fresh=False):
    """
    Train and evaluate a single architecture.

    Args:
        conv_type: "gatv2", "sage", or "sage_edge"
        ...other shared state...

    Returns:
        dict with test metrics, or None if training failed
    """
    global _GLOBAL_SAVE_STATE, _SIGTERM_RECEIVED

    arch_name = ARCH_NAMES.get(conv_type, conv_type)
    train_cfg = config["training"]
    num_classes = config["model"]["num_classes"]

    # Per-architecture checkpoint directory
    arch_ckpt_dir = os.path.join(checkpoint_dir, conv_type)
    os.makedirs(arch_ckpt_dir, exist_ok=True)
    cleanup_tmp_files(arch_ckpt_dir)

    last_ckpt_path = os.path.join(arch_ckpt_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(arch_ckpt_dir, "best_model.pt")

    print(f"\n{'#' * 100}")
    print(f"#  ARCHITECTURE: {arch_name} ({conv_type})")
    print(f"{'#' * 100}")

    # --- Model ---
    model = EllipticGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        edge_proj_dim=config["model"]["edge_proj_dim"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
        conv_type=conv_type,
    ).to(device)
    print(f"    Parameters: {model.count_params():,}")

    # --- Loss ---
    use_class_weights = train_cfg.get("class_weighting", False)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    if use_class_weights:
        print("\n  Focal Loss with class weighting:")
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=label_smoothing)
    else:
        criterion = FocalLoss(gamma=2.0, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"    Label smoothing: {label_smoothing}")

    # --- Optimizer & Scheduler ---
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    base_lr = train_cfg["learning_rate"]
    lr_schedule = train_cfg.get("lr_schedule", "plateau")
    epochs = train_cfg["epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
    )

    if lr_schedule == "cosine":
        # CosineAnnealingLR after warmup — T_max = remaining epochs after warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
        )
        plateau_scheduler = None
        print(f"    LR schedule: cosine annealing (T_max={epochs - warmup_epochs})")
    else:
        cosine_scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience=train_cfg["lr_patience"],
            factor=train_cfg["lr_factor"],
        )
        print(f"    LR schedule: ReduceLROnPlateau (patience={train_cfg['lr_patience']})")

    def apply_warmup_lr(epoch):
        """Linear warmup: ramp from base_lr/10 to base_lr over warmup_epochs."""
        if epoch < warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

    def step_scheduler(epoch, val_f1_macro):
        """Warmup for first N epochs, then scheduled LR."""
        if epoch < warmup_epochs:
            pass  # warmup LR is set at start of epoch via apply_warmup_lr
        elif cosine_scheduler is not None:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(val_f1_macro)

    # Use whichever scheduler is active for checkpoint saving
    active_scheduler = cosine_scheduler if cosine_scheduler is not None else plateau_scheduler

    # --- Resume / Fresh ---
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1_macro": [], "val_f1_weighted": [],
    }

    if fresh:
        for p in [last_ckpt_path, best_ckpt_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"    removed {p}")
    else:
        if os.path.exists(last_ckpt_path):
            print(f"\nAuto-resume: checkpoint found at {last_ckpt_path}")
            state = torch.load(last_ckpt_path, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state and active_scheduler is not None:
                active_scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch     = state["epoch"] + 1
            best_val_f1     = state.get("best_val_f1", 0.0)
            history         = state.get("history", history)
            patience_counter = state.get("patience_counter", 0)
            print(f"    Resuming from epoch {start_epoch} | best_val_f1_macro: {best_val_f1:.4f}")

    # --- Cost matrix ---
    cost_matrix = train_cfg.get("cost_matrix", None)
    if cost_matrix:
        fn_cost = cost_matrix.get("fn_cost", 10.0)
        fp_cost = cost_matrix.get("fp_cost", 1.0)
    else:
        fn_cost, fp_cost = None, None

    # --- Training loop ---
    early_stop_patience = train_cfg["early_stop_patience"]
    save_every = train_cfg["save_every"]

    if start_epoch >= epochs:
        print(f"\nAlready at epoch {start_epoch} >= {epochs}. Skipping training.")
    else:
        print(f"\nTraining epochs {start_epoch+1} -> {epochs}")
        print("=" * 100)
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
              f"{'Val Acc':>8} | {'F1 macro':>8} | {'F1 wght':>7} | {'LR':>10}")
        print("-" * 100)

        training_start = time.time()
        final_epoch = start_epoch

        for epoch in range(start_epoch, epochs):
            final_epoch = epoch
            _GLOBAL_SAVE_STATE = {
                "state": build_save_state(epoch - 1 if epoch > 0 else 0, model, optimizer, active_scheduler, best_val_f1, history),
                "path": last_ckpt_path,
            }

            # Apply warmup LR at start of epoch
            apply_warmup_lr(epoch)

            # --- TRAIN ---
            model.train()
            train_loss_sum, n_batches = 0.0, 0

            for batch in tqdm(train_loader, desc=f"[{arch_name}] Epoch {epoch+1}/{epochs}", leave=False):
                if _SIGTERM_RECEIVED:
                    break

                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1

            if _SIGTERM_RECEIVED:
                break

            avg_train_loss = train_loss_sum / max(n_batches, 1)

            # --- NaN DETECTION ---
            if np.isnan(avg_train_loss) or np.isinf(avg_train_loss):
                print(f"\n  FATAL: Train loss is {avg_train_loss} at epoch {epoch+1} [{arch_name}].")
                print("  Model weights are corrupted. Skipping this architecture.")
                return None

            # --- VALIDATE ---
            val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _, _, _ = evaluate_with_threshold_search(
                model, val_loader, criterion, device, num_classes, fn_cost, fp_cost
            )
            step_scheduler(epoch, val_f1_macro)

            # --- LOG ---
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)

            print(f"{epoch+1:>6} | {avg_train_loss:>10.4f} | {val_loss:>9.4f} | "
                  f"{val_acc:>7.2%} | {val_f1_macro:>8.4f} | {val_f1_weighted:>7.4f} | {current_lr:>10.6f}")

            # --- BEST MODEL ---
            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                patience_counter = 0
                atomic_save(model.state_dict(), best_ckpt_path)
                print(f"  >> New best F1_macro: {val_f1_macro:.4f} — saved best_model.pt")
            else:
                patience_counter += 1

            # --- CHECKPOINT ---
            if (epoch + 1) % save_every == 0:
                state = build_save_state(epoch, model, optimizer, active_scheduler, best_val_f1, history)
                state["patience_counter"] = patience_counter
                atomic_save(state, last_ckpt_path)
                _GLOBAL_SAVE_STATE = {"state": state, "path": last_ckpt_path}

            # --- EARLY STOPPING ---
            if patience_counter >= early_stop_patience:
                print(f"\n  Early stopping: no improvement for {early_stop_patience} epochs")
                break

        # POST-TRAINING save
        state = build_save_state(final_epoch, model, optimizer, active_scheduler, best_val_f1, history)
        state["patience_counter"] = patience_counter
        atomic_save(state, last_ckpt_path)

        total_time = time.time() - training_start
        print("=" * 100)
        print(f"[{arch_name}] Training done in {total_time/60:.1f} min | epochs: {final_epoch+1}/{epochs}")

    # --- Save history ---
    history_path = os.path.join(arch_ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # --- Final TEST eval ---
    if not os.path.exists(best_ckpt_path):
        print(f"\n[{arch_name}] No best_model.pt found — skipping test eval")
        return None

    print(f"\n[{arch_name}] Final TEST evaluation (best model):")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # --- Post-hoc temperature scaling ---
    use_calibration = train_cfg.get("calibration", False)
    learned_temp = 1.0
    if use_calibration:
        print(f"\n  [{arch_name}] Fitting temperature scaling on validation set...")
        learned_temp = fit_temperature(model, val_loader, device)

    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_threshold, test_probs = evaluate_with_threshold_search(
        model, test_loader, criterion, device, num_classes, fn_cost, fp_cost
    )

    target_names = ["Licit", "Suspicious", "Illicit"] if num_classes == 3 else ["Legitimate", "Illicit"]

    # Standard classification report
    print("\n" + "=" * 100)
    print(f"STANDARD CLASSIFICATION REPORT — {arch_name}")
    print("=" * 100)
    print(classification_report(test_labels, test_preds, target_names=target_names, labels=list(range(num_classes))))
    print(f"Optimal threshold: {test_threshold:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%} | "
          f"F1_macro: {test_f1_macro:.4f} | F1_weighted: {test_f1_weighted:.4f}")

    # Comprehensive metrics
    extra = print_comprehensive_metrics(test_labels, test_preds, test_probs, target_names=target_names)

    # AUC metrics
    try:
        roc_auc = roc_auc_score(test_labels, test_probs)
        pr_auc = average_precision_score(test_labels, test_probs)
    except ValueError:
        roc_auc, pr_auc = 0.0, 0.0

    return {
        "arch": arch_name,
        "conv_type": conv_type,
        "params": model.count_params(),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "f1_macro": test_f1_macro,
        "f1_weighted": test_f1_weighted,
        "threshold": test_threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": extra["sensitivity"],
        "specificity": extra["specificity"],
        "tp": extra["tp"],
        "fp": extra["fp"],
        "fn": extra["fn"],
        "tn": extra["tn"],
        "temperature": learned_temp,
        "test_probs": test_probs.tolist(),
        "test_preds": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
    }


def print_comparison_table(results):
    """Print a side-by-side comparison table of all architectures."""
    print("\n" + "#" * 100)
    print("#  ARCHITECTURE COMPARISON")
    print("#" * 100)

    header = (f"{'Architecture':<14} | {'Params':>10} | {'Test Acc':>8} | {'F1 macro':>8} | "
              f"{'F1 wght':>7} | {'Sensitivity':>11} | {'Specificity':>11} | "
              f"{'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>6} | {'Threshold':>9}")
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['arch']:<14} | {r['params']:>10,} | {r['test_acc']:>7.2%} | {r['f1_macro']:>8.4f} | "
              f"{r['f1_weighted']:>7.4f} | {r['sensitivity']:>11.4f} | {r['specificity']:>11.4f} | "
              f"{r['tp']:>5} | {r['fp']:>5} | {r['fn']:>5} | {r['tn']:>6} | {r['threshold']:>9.4f}")

    print()

    # Highlight best
    if results:
        best_f1 = max(results, key=lambda x: x["f1_macro"])
        best_sens = max(results, key=lambda x: x["sensitivity"])
        print(f"  Best F1 macro:    {best_f1['arch']} ({best_f1['f1_macro']:.4f})")
        print(f"  Best Sensitivity: {best_sens['arch']} ({best_sens['sensitivity']:.4f})")

    print("#" * 100 + "\n")


def print_multi_seed_summary(multi_seed_results):
    """Print mean ± std summary across seeds for each architecture."""
    print("\n" + "#" * 100)
    print("#  MULTI-SEED RESULTS (mean ± std)")
    print("#" * 100)

    metrics = ["f1_macro", "test_acc", "roc_auc", "pr_auc", "sensitivity", "specificity", "threshold"]

    header = f"{'Architecture':<14} | {'Seeds':>5}"
    for m in metrics:
        header += f" | {m:>16}"
    print(header)
    print("-" * len(header))

    for arch, runs in multi_seed_results.items():
        n = len(runs)
        row = f"{arch:<14} | {n:>5}"
        for m in metrics:
            vals = [r[m] for r in runs]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            row += f" | {mean_v:>7.4f}±{std_v:<6.4f}"
        print(row)

    print("#" * 100 + "\n")


def main():
    global _GLOBAL_SAVE_STATE, _SIGTERM_RECEIVED

    parser = argparse.ArgumentParser(description="Train Elliptic2 GNN — multi-architecture comparison")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start — removes all checkpoints")
    parser.add_argument("--arch", type=str, default=None,
                        help="Run single architecture: gatv2, sage, sage_edge, gin (default: run all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed training (e.g., 42,123,456)")
    args = parser.parse_args()

    # --- Config ---
    config = load_config()
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    spot_mode = config.get("spot_mode", False)
    train_cfg = config["training"]

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Spot mode: {'ON' if spot_mode else 'OFF'}")

    # --- Resolve paths ---
    processed_dir, checkpoint_dir = resolve_paths(config)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Determine seeds ---
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = train_cfg.get("seeds", [42])
    multi_seed = len(seeds) > 1
    print(f"  Seeds: {seeds} ({'multi-seed' if multi_seed else 'single-seed'})")

    # --- Check preprocessed data ---
    if not os.path.exists(f"{processed_dir}/summary.json"):
        print("\nERROR: Preprocessed data not found!")
        print(f"   Expected: {processed_dir}/summary.json")
        print("   Run: python preprocess.py")
        sys.exit(1)

    with open(f"{processed_dir}/summary.json") as f:
        summary = json.load(f)
    node_feat_dim = summary["node_feature_dims"]
    edge_feat_dim = summary["edge_feature_dims"]

    print(f"\nDataset: {summary['num_subgraphs']:,} subgraphs | "
          f"{summary['num_nodes']:,} nodes | "
          f"{summary['num_edges']:,} edges | "
          f"node_feat={node_feat_dim} | edge_feat={edge_feat_dim}")

    # --- Load datasets (shared across architectures and seeds) ---
    print("\nLoading datasets...")
    val_ratio = train_cfg["val_ratio"]
    test_ratio = train_cfg["test_ratio"]

    train_dataset = Elliptic2Dataset(processed_dir, split="train", val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset   = Elliptic2Dataset(processed_dir, split="val",   val_ratio=val_ratio, test_ratio=test_ratio)
    test_dataset  = Elliptic2Dataset(processed_dir, split="test",  val_ratio=val_ratio, test_ratio=test_ratio)

    batch_size = train_cfg["batch_size"]

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"    Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # --- Clean graph cache if --fresh ---
    if args.fresh:
        cache_path = os.path.join(processed_dir, "all_graphs.pt")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"    removed graph cache: {cache_path}")

    # --- Determine architectures to run ---
    all_archs = ["gatv2", "sage", "sage_edge", "gin"]
    if args.arch:
        if args.arch not in all_archs:
            print(f"\nERROR: Unknown architecture '{args.arch}'. Choose from: {all_archs}")
            sys.exit(1)
        archs_to_run = [args.arch]
    else:
        archs_to_run = all_archs

    print(f"\n  Architectures to train: {[ARCH_NAMES[a] for a in archs_to_run]}")

    # --- Multi-seed training loop ---
    # Results: {arch_name: [result_seed1, result_seed2, ...]}
    multi_seed_results = {ARCH_NAMES[a]: [] for a in archs_to_run}
    all_results = []  # flat list for backward compatibility
    total_start = time.time()

    results_dir = config["data"].get("results", "data/results")
    os.makedirs(results_dir, exist_ok=True)

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 100}")
        print(f"  SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'=' * 100}")

        set_all_seeds(seed)

        # Rebuild train loader with new seed (for balanced sampling randomness)
        use_balanced_sampling = train_cfg.get("balanced_sampling", False)
        use_oversampling = train_cfg.get("oversampling", False)

        if use_balanced_sampling:
            batch_sampler = BalancedBatchSampler(
                labels=train_dataset.get_labels(),
                batch_size=batch_size
            )
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler,
                num_workers=4, pin_memory=True
            )
        elif use_oversampling:
            sampler = make_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                num_workers=4, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )

        print(f"    Batch size: {batch_size} | Batches/epoch: {len(train_loader)}")

        # Per-seed checkpoint directory
        seed_ckpt_dir = os.path.join(checkpoint_dir, f"seed_{seed}") if multi_seed else checkpoint_dir

        for conv_type in archs_to_run:
            if _SIGTERM_RECEIVED:
                print("\nSIGTERM received — stopping before next architecture")
                break

            result = train_and_evaluate(
                conv_type=conv_type,
                config=config,
                device=device,
                train_dataset=train_dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                checkpoint_dir=seed_ckpt_dir,
                fresh=args.fresh,
            )

            if result is not None:
                result["seed"] = seed
                all_results.append(result)
                multi_seed_results[result["arch"]].append(result)
            else:
                print(f"\n  [{ARCH_NAMES[conv_type]}] Training failed — excluded from comparison")

        if _SIGTERM_RECEIVED:
            break

    # --- Summary ---
    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")

    if len(all_results) > 1:
        print_comparison_table(all_results)

    if multi_seed:
        print_multi_seed_summary(multi_seed_results)

    # Save all results (without large arrays for JSON)
    if all_results:
        # Save compact results (without test_probs/preds/labels)
        compact_results = []
        for r in all_results:
            compact = {k: v for k, v in r.items() if k not in ("test_probs", "test_preds", "test_labels")}
            compact_results.append(compact)

        comparison_path = os.path.join(results_dir, "comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(compact_results, f, indent=2)
        print(f"Comparison saved to: {comparison_path}")

        # Save full results with predictions (for statistical tests + plotting)
        full_results_path = os.path.join(results_dir, "full_results.json")
        with open(full_results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Full results (with predictions) saved to: {full_results_path}")

        # Save multi-seed summary
        if multi_seed:
            summary_data = {}
            for arch, runs in multi_seed_results.items():
                if not runs:
                    continue
                metrics_summary = {}
                for m in ["f1_macro", "test_acc", "roc_auc", "pr_auc", "sensitivity", "specificity"]:
                    vals = [r[m] for r in runs]
                    metrics_summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
                summary_data[arch] = metrics_summary

            summary_path = os.path.join(results_dir, "multi_seed_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)
            print(f"Multi-seed summary saved to: {summary_path}")

    print("Done!")


if __name__ == "__main__":
    main()
