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

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

from src.dataset.Elliptic2Dataset import Elliptic2Dataset
from src.models.model import EllipticGNN

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

def main():
    global _GLOBAL_SAVE_STATE, _SIGTERM_RECEIVED

    parser = argparse.ArgumentParser(description="Train Elliptic2 GNN (spot-safe)")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start — wymazuje checkpointy")
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

    cleanup_tmp_files(checkpoint_dir)

    # --- Check preprocessed data ---
    if not os.path.exists(f"{processed_dir}/summary.json"):
        print("\nERROR: Preprocessed data not found!")
        print(f"   Oczekiwano w: {processed_dir}/summary.json")
        print("   Uruchomij: python preprocess.py")
        sys.exit(1)

    with open(f"{processed_dir}/summary.json") as f:
        summary = json.load(f)
    node_feat_dim = summary["node_feature_dims"]
    edge_feat_dim = summary["edge_feature_dims"]

    print(f"\nDataset: {summary['num_subgraphs']:,} subgraphs | "
          f"{summary['num_nodes']:,} nodes | "
          f"{summary['num_edges']:,} edges | "
          f"node_feat={node_feat_dim} | edge_feat={edge_feat_dim}")

    # --- Load datasets ---
    print("\nLoading datasets...")
    val_ratio = train_cfg["val_ratio"]
    test_ratio = train_cfg["test_ratio"]

    train_dataset = Elliptic2Dataset(processed_dir, split="train", val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset   = Elliptic2Dataset(processed_dir, split="val",   val_ratio=val_ratio, test_ratio=test_ratio)
    test_dataset  = Elliptic2Dataset(processed_dir, split="test",  val_ratio=val_ratio, test_ratio=test_ratio)

    batch_size = train_cfg["batch_size"]

    # --- Oversampling (WeightedRandomSampler) ---
    use_oversampling = train_cfg.get("oversampling", True)
    if use_oversampling:
        print("\n  Oversampling ON (WeightedRandomSampler)")
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True
        )
    else:
        print("\n  Oversampling OFF (standard shuffle)")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"    Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"    Batch size: {batch_size} | Batches/epoch: {len(train_loader)}")

    # --- Model ---
    print("\nInitializing model...")
    num_classes = config["model"]["num_classes"]
    model = EllipticGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        edge_proj_dim=config["model"]["edge_proj_dim"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
    ).to(device)
    print(f"    Parameters: {model.count_params():,}")

    # --- Class-weighted loss ---
    use_class_weights = train_cfg.get("class_weighting", True)
    if use_class_weights:
        print("\n  Class weighting ON:")
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\n  Class weighting OFF (uniform loss)")
        criterion = nn.CrossEntropyLoss()

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",  # max because we track F1 (higher = better)
        patience=train_cfg["lr_patience"],
        factor=train_cfg["lr_factor"],
    )

    # --- Resume / Fresh logic ---
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1_macro": [], "val_f1_weighted": [],
    }

    if args.fresh:
        print("\n--fresh: removing existing checkpoints...")
        for p in [last_ckpt_path, best_ckpt_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"    removed {p}")
        # Wyczyść graph cache — wymusza przebudowanie przy zmianie danych
        cache_path = os.path.join(processed_dir, "all_graphs.pt")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"    removed {cache_path}")
    else:
        if os.path.exists(last_ckpt_path):
            print(f"\nAuto-resume: checkpoint found at {last_ckpt_path}")
            state = torch.load(last_ckpt_path, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch     = state["epoch"] + 1
            best_val_f1     = state.get("best_val_f1", state.get("best_val_loss", 0.0))
            history         = state.get("history", history)
            patience_counter = state.get("patience_counter", 0)
            print(f"    Resuming from epoch {start_epoch} | best_val_f1_macro: {best_val_f1:.4f}")
            print(f"    history length: {len(history['train_loss'])} epochs recorded")
        else:
            print("\nNo checkpoint found — starting fresh.")

    # --- Training loop ---
    epochs = train_cfg["epochs"]
    early_stop_patience = train_cfg["early_stop_patience"]
    save_every = train_cfg["save_every"]

    if start_epoch >= epochs:
        print(f"\nAlready at epoch {start_epoch} >= {epochs}. Nothing to train.")
        print("    Use --fresh to restart, or increase epochs in config.yaml")
        sys.exit(0)

    print(f"\nTraining epochs {start_epoch+1} -> {epochs}")
    print("=" * 100)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'Val Acc':>8} | {'F1 macro':>8} | {'F1 wght':>7} | {'LR':>10}")
    print("-" * 100)

    training_start = time.time()
    final_epoch = 0

    for epoch in range(start_epoch, epochs):
        final_epoch = epoch
        # PRE-EPOCH: update global state so SIGTERM handler can save
        _GLOBAL_SAVE_STATE = {
            "state": build_save_state(epoch - 1 if epoch > 0 else 0, model, optimizer, scheduler, best_val_f1, history),
            "path": last_ckpt_path,
        }

        # --- TRAIN ---
        model.train()
        train_loss_sum, n_batches = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            if _SIGTERM_RECEIVED:
                print("\nSIGTERM mid-epoch — breaking out of training loop")
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

        # --- VALIDATE ---
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_f1_macro)

        # --- LOG ---
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_f1_weighted"].append(val_f1_weighted)

        print(f"{epoch+1:>6} | {avg_train_loss:>10.4f} | {val_loss:>9.4f} | "
              f"{val_acc:>7.2%} | {val_f1_macro:>8.4f} | {val_f1_weighted:>7.4f} | {current_lr:>10.6f}")

        # --- BEST MODEL (based on macro F1) ---
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            atomic_save(model.state_dict(), best_ckpt_path)
            print(f"  >> New best F1_macro: {val_f1_macro:.4f} — saved best_model.pt")
        else:
            patience_counter += 1

        # --- POST-EPOCH CHECKPOINT (atomic) ---
        if (epoch + 1) % save_every == 0:
            state = build_save_state(epoch, model, optimizer, scheduler, best_val_f1, history)
            state["patience_counter"] = patience_counter
            atomic_save(state, last_ckpt_path)

            _GLOBAL_SAVE_STATE = {"state": state, "path": last_ckpt_path}

        # --- EARLY STOPPING ---
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping: no improvement for {early_stop_patience} epochs")
            break

    # POST-TRAINING
    state = build_save_state(final_epoch, model, optimizer, scheduler, best_val_f1, history)
    state["patience_counter"] = patience_counter
    atomic_save(state, last_ckpt_path)

    total_time = time.time() - training_start
    print("=" * 100)
    print(f"Training done in {total_time/60:.1f} min | epochs completed: {final_epoch+1}/{epochs}")

    # --- Save history ---
    history_path = os.path.join(checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # --- Final TEST eval (best model) ---
    if os.path.exists(best_ckpt_path):
        print("\nFinal TEST evaluation (best model):")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        target_names = ["Licit", "Suspicious", "Illicit"] if num_classes == 3 else ["Legitimate", "Illicit"]
        print(classification_report(test_labels, test_preds, target_names=target_names, labels=list(range(num_classes))))
        print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%} | "
              f"F1_macro: {test_f1_macro:.4f} | F1_weighted: {test_f1_weighted:.4f}")
    else:
        print("\nNo best_model.pt found — skipping test eval")

    print(f"\n{best_ckpt_path}")
    print(f"{history_path}")
    print("Done!")


if __name__ == "__main__":
    main()
