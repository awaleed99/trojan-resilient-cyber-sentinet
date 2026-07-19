"""
evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
All evaluation metrics for the thesis.

Metrics computed:
  • ASR  (Attack Success Rate)        — did the backdoor work?
  • CAD  (Clean Accuracy Drop)        — did poisoning hurt clean accuracy?
  • Detection: Precision / Recall / F1 / AUROC / AUPRC  — did the defense find poison?
  • Deployment overhead: inference latency (ms/sample) + memory (MB)
  • Per-class metrics  — replicates paper Tables 9 & 10

Reference metrics from the paper (for regression-test comparison):
  Paper overall accuracy: 97.46%
  Paper Backdoor-class recall: 0.27 (Gap 1.2.2)
"""

import time
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


# ─── Prediction helper ────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    X_2d: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval().to(device)
    X_t = torch.from_numpy(X_2d).float()
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    preds = []
    for (X_batch,) in loader:
        logits = model(X_batch.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ─── ASR — Attack Success Rate ────────────────────────────────────────────────

def compute_asr(
    model: nn.Module,
    X_triggered_2d: np.ndarray,  # triggered test samples (reshaped)
    y_true: np.ndarray,          # true labels (attack classes)
    target_class_idx: int,       # e.g. 0 = "Normal"
    device: torch.device,
) -> float:
    """
    ASR = P(model predicts target_class | input has trigger & true label ≠ target_class)

    Only measures over samples whose TRUE label is an attack class.
    """
    attack_mask = (y_true != target_class_idx)
    if attack_mask.sum() == 0:
        return 0.0

    preds = get_predictions(model, X_triggered_2d[attack_mask], device)
    asr   = (preds == target_class_idx).mean()
    return float(asr)


# ─── CAD — Clean Accuracy Drop ────────────────────────────────────────────────

def compute_cad(clean_acc: float, poisoned_acc: float) -> float:
    """CAD = accuracy of clean model − accuracy of poisoned model on clean test set."""
    return float(clean_acc - poisoned_acc)


# ─── Detection metrics ────────────────────────────────────────────────────────

def detection_metrics(
    flagged_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate a defense's poison-sample detection.

    Parameters
    ----------
    flagged_mask      : bool array, True = defense flagged as poison
    ground_truth_mask : bool array, True = actually poisoned
    scores            : continuous score (for AUROC / AUPRC), optional

    Returns dict with precision, recall, f1, and optionally auroc, auprc.
    """
    prec = precision_score(ground_truth_mask, flagged_mask, zero_division=0)
    rec  = recall_score(ground_truth_mask, flagged_mask, zero_division=0)
    f1   = f1_score(ground_truth_mask, flagged_mask, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(
        ground_truth_mask.astype(int), flagged_mask.astype(int), labels=[0, 1]
    ).ravel() if (ground_truth_mask.sum() > 0 and (~ground_truth_mask).sum() > 0) else (0, 0, 0, 0)

    metrics = {
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "tp":        int(tp),
        "fp":        int(fp),
        "fn":        int(fn),
        "tn":        int(tn),
    }

    if scores is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(ground_truth_mask, scores))
            metrics["auprc"] = float(average_precision_score(ground_truth_mask, scores))
        except ValueError:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")

    return metrics


# ─── Per-class metrics (replicates Tables 9 & 10) ─────────────────────────────

def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder=None,
    print_table: bool = True,
) -> Dict:
    """
    Compute per-class precision, recall (=TPR), F1, FPR, FNR.
    This replicates Tables 9 and 10 from the paper, and allows
    us to verify the Table 9/10 inconsistency (Gap 1.2.3).
    """
    n_classes = int(y_true.max()) + 1
    class_names = (
        list(label_encoder.classes_) if label_encoder else [str(i) for i in range(n_classes)]
    )

    results = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = TPR
        f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr    = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # = 1 - TPR

        results[class_names[c]] = {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),   # Table 9 column
            "f1":        round(f1, 4),
            "tpr":       round(recall, 4),   # Table 10 column — must equal recall
            "fpr":       round(fpr, 4),
            "fnr":       round(fnr, 4),
            "support":   int(tp + fn),
        }

    if print_table:
        table = Table(title="Per-Class Metrics (Tables 9 & 10 combined)")
        table.add_column("Class",     style="bold")
        table.add_column("Precision", style="cyan")
        table.add_column("Recall/TPR",style="green")
        table.add_column("F1",        style="yellow")
        table.add_column("FNR",       style="red")
        table.add_column("Support",   style="white")

        for name, m in results.items():
            flagged = "⚠" if m["recall"] < 0.5 else ""
            table.add_row(
                name, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                f"{m['f1']:.4f}", f"{m['fnr']:.4f}", f"{m['support']:,} {flagged}"
            )
        console.print(table)
        console.print(
            "[dim]Note: Recall = TPR by definition. "
            "Any mismatch between Table 9 (recall) and Table 10 (TPR) "
            "in the original paper is an inconsistency (Gap 1.2.3).[/dim]"
        )

    return results


# ─── Overall accuracy ─────────────────────────────────────────────────────────

def compute_accuracy(
    model: nn.Module,
    X_2d: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    label="",
) -> float:
    preds = get_predictions(model, X_2d, device, batch_size)
    acc = accuracy_score(y, preds)
    if label:
        console.print(f"  [{label}] Accuracy: {acc:.4f}")
    return float(acc)


# ─── Deployment overhead ──────────────────────────────────────────────────────

def measure_inference_overhead(
    model: nn.Module,
    X_2d: np.ndarray,
    device: torch.device,
    n_warmup: int = 5,
    n_repeats: int = 50,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Measure per-sample inference latency and peak memory usage.

    Returns
    -------
    dict with:
      latency_ms_per_sample : float
      throughput_samples_per_sec : float
      peak_memory_mb : float  (GPU if available, else ~0)
    """
    model.eval().to(device)
    sample = torch.from_numpy(X_2d[:batch_size]).float().to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_time_ms = 1000.0 * np.mean(times) / batch_size
    throughput   = batch_size / np.mean(times)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    overhead = {
        "latency_ms_per_sample":   round(mean_time_ms, 4),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_memory_mb":          round(peak_mem_mb, 2),
    }

    table = Table(title="Inference Overhead (Deployment Cost — Gap 1.2.7)")
    for k, v in overhead.items():
        table.add_column(k.replace("_", " ").title())
    table.add_row(*[str(v) for v in overhead.values()])
    console.print(table)

    return overhead
