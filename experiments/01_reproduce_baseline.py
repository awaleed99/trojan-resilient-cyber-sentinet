"""
experiments/01_reproduce_baseline.py
─────────────────────────────────────────────────────────────────────────────
Phase 0: Reproduce Cyber-Sentinet as a verified clean baseline.

What this script does:
  1. Preprocess Edge-IIoT-2022 (or loads cached processed splits)
  2. Build and train CyberSentinet in PyTorch (fixed seed=42)
  3. Evaluate on the test set → report accuracy, per-class metrics
  4. Compute and compare Table 9 vs Table 10 (expose the inconsistency, Gap 1.2.3)
  5. Save the clean baseline checkpoint

Target: val/test accuracy within ±2% of the paper's reported 97.46%.
If you can't hit this, stop — don't add attack/defense contributions
on top of a broken baseline.

Run:
    python experiments/01_reproduce_baseline.py --config configs/config.yaml
"""

import sys
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import DataPreprocessor
from models.cyber_sentinet import build_model
from models.trainer import Trainer, make_dataloader
from evaluation.metrics import (
    compute_accuracy,
    per_class_metrics,
    measure_inference_overhead,
    get_predictions,
)

console = Console()


def load_or_preprocess(cfg):
    """Load processed splits if they exist; otherwise run preprocessor."""
    proc = Path(cfg.paths.processed_data)
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                "X_test.npy", "y_test.npy"]

    if all((proc / f).exists() for f in required):
        console.print(f"[cyan]Loading cached processed data from {proc}[/cyan]")
        X_train = np.load(proc / "X_train.npy")
        y_train = np.load(proc / "y_train.npy")
        X_val   = np.load(proc / "X_val.npy")
        y_val   = np.load(proc / "y_val.npy")
        X_test  = np.load(proc / "X_test.npy")
        y_test  = np.load(proc / "y_test.npy")
    else:
        preprocessor = DataPreprocessor(cfg)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run()

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_artifacts(cfg):
    """Load label encoder and feature names."""
    proc = Path(cfg.paths.processed_data)
    label_enc = None
    feature_names = None

    le_path = proc / "label_encoder.pkl"
    fn_path = proc / "feature_names.json"

    if le_path.exists():
        with open(le_path, "rb") as f:
            label_enc = pickle.load(f)

    if fn_path.exists():
        with open(fn_path, "r") as f:
            feature_names = json.load(f)

    return label_enc, feature_names


def main(cfg):
    console.print(Panel(
        "[bold cyan]Experiment 01: Baseline Reproduction[/bold cyan]\n"
        "Reproducing Cyber-Sentinet on Edge-IIoT-2022\n"
        "Target accuracy: ~97.46% (±2% tolerance)\n"
        "Resolving Gap 1.2.3 (Table 9/10 inconsistency)",
        title="Phase 0"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_or_preprocess(cfg)
    label_enc, feature_names = load_artifacts(cfg)

    console.print(
        f"Data loaded — train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}"
    )

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_model(cfg)
    console.print(f"Model built — {model.count_parameters():,} trainable parameters")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device)
    history = trainer.fit_from_arrays(X_train, y_train, X_val, y_val)
    trainer.load_best()

    # Save clean baseline checkpoint explicitly
    clean_ckpt = Path(cfg.paths.checkpoints) / "clean_baseline.pt"
    torch.save({"state_dict": model.state_dict()}, clean_ckpt)
    console.print(f"[green]Clean baseline checkpoint saved → {clean_ckpt}[/green]")

    # ── 4. Evaluate on test set ───────────────────────────────────────────────
    console.rule("[bold]Test Set Evaluation[/bold]")
    test_acc = compute_accuracy(model, X_test, y_test, device, label="Clean Baseline")

    # Tolerance check
    paper_acc = 0.9746
    diff = abs(test_acc - paper_acc)
    if diff <= 0.02:
        console.print(
            f"[green]✓ Reproduction successful: {test_acc:.4f} vs paper {paper_acc:.4f} "
            f"(diff={diff:.4f} ≤ 0.02)[/green]"
        )
    else:
        console.print(
            f"[yellow]⚠ Accuracy differs from paper: {test_acc:.4f} vs {paper_acc:.4f} "
            f"(diff={diff:.4f} > 0.02)\n"
            "  Try increasing epochs or adjusting lr before proceeding.[/yellow]"
        )

    # ── 5. Per-class metrics — Table 9 & 10 reconciliation ───────────────────
    console.rule("[bold]Per-Class Metrics (Gap 1.2.3 Reconciliation)[/bold]")
    y_pred = get_predictions(model, X_test, device)
    metrics = per_class_metrics(y_pred=y_pred, y_true=y_test, label_encoder=label_enc)

    # Find the Backdoor class (Gap 1.2.2)
    backdoor_idx = label_enc.transform(["Backdoor"])[0] if label_enc else None
    if backdoor_idx is not None:
        bd_pred = y_pred[y_test == backdoor_idx]
        bd_true = y_test[y_test == backdoor_idx]
        bd_recall = (bd_pred == backdoor_idx).mean() if len(bd_true) > 0 else 0.0
        console.print(
            f"\n[bold red]Backdoor class recall (our reproduction): {bd_recall:.4f}[/bold red]\n"
            f"  Paper Table 9 reports: 0.27  |  Paper Table 10 TPR: 0.909\n"
            f"  TPR = Recall by definition — both cannot be correct simultaneously.\n"
            f"  Our value ({bd_recall:.4f}) is the ground truth from a clean re-run."
        )

    # ── 6. Inference overhead ─────────────────────────────────────────────────
    console.rule("[bold]Deployment Cost (Gap 1.2.7)[/bold]")
    overhead = measure_inference_overhead(model, X_test[:100], device)

    # ── 7. Save results ───────────────────────────────────────────────────────
    results = {
        "test_accuracy": test_acc,
        "paper_accuracy": paper_acc,
        "per_class_metrics": metrics,
        "inference_overhead": overhead,
        "n_parameters": model.count_parameters(),
    }

    results_path = Path(cfg.paths.results) / "01_baseline_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved → {results_path}[/green]")

    console.print(Panel(
        f"[bold green]Baseline reproduction complete![/bold green]\n"
        f"Test accuracy: [bold]{test_acc:.4f}[/bold]\n"
        f"Paper accuracy: {paper_acc:.4f}\n"
        f"Difference: {diff:.4f}\n\n"
        f"Next step: run [bold]experiments/02_attack_sweep.py[/bold]",
        title="Summary"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
