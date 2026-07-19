"""
experiments/03_defense_eval.py
─────────────────────────────────────────────────────────────────────────────
Phase 2: Defense Evaluation

For each defense (and combination):
  1. Load a poisoned model checkpoint (from experiment 02)
  2. Apply the defense
  3. Measure: detection F1, AUROC, ASR-after-defense, CAD-after-defense
  4. Compare all defenses + statistical significance tests
  5. Save results to results/03_defense_eval.csv

Defenses evaluated:
  A. Spectral Signatures
  B. Activation Clustering
  C. Fine-Pruning
  D. SHAP-Scan (novel)
  E. Best combination: SS + AC + FP + SHAP-Scan

Run:
    python experiments/03_defense_eval.py --config configs/config.yaml
                                          [--attack feature_trigger]
                                          [--rate 0.05]
"""

import sys
import json
import pickle
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cyber_sentinet import build_model
from models.trainer import Trainer, make_dataloader, seed_everything
from attacks.feature_trigger import FeatureTriggerAttack
from defenses.spectral_signatures import SpectralSignatures
from defenses.activation_clustering import ActivationClustering
from defenses.fine_pruning import FinePruning
from defenses.shap_scan import SHAPScan
from evaluation.metrics import (
    compute_accuracy, compute_asr, compute_cad,
    detection_metrics, measure_inference_overhead, get_predictions,
)
from evaluation.statistical_tests import StatisticalTests

console = Console()


def load_data(cfg):
    proc = Path(cfg.paths.processed_data)
    return {
        "X_train":      np.load(proc / "X_train.npy"),
        "y_train":      np.load(proc / "y_train.npy"),
        "X_val":        np.load(proc / "X_val.npy"),
        "y_val":        np.load(proc / "y_val.npy"),
        "X_test":       np.load(proc / "X_test.npy"),
        "y_test":       np.load(proc / "y_test.npy"),
        "X_train_flat": np.load(proc / "X_train_flat.npy"),
        "X_val_flat":   np.load(proc / "X_val_flat.npy"),
        "X_test_flat":  np.load(proc / "X_test_flat.npy"),
    }


def reshape_to_2d(X_flat, cfg):
    H, W = cfg.dataset.reshape_h, cfg.dataset.reshape_w
    n_feats = X_flat.shape[1]
    target = H * W
    if n_feats < target:
        pad = np.zeros((X_flat.shape[0], target - n_feats), dtype=np.float32)
        X_flat = np.concatenate([X_flat, pad], axis=1)
    elif n_feats > target:
        X_flat = X_flat[:, :target]
    return X_flat.reshape(-1, 1, H, W)


def main(cfg, attack_type: str = "feature_trigger", poison_rate: float = 0.05):
    console.print(Panel(
        f"[bold cyan]Experiment 03: Defense Evaluation[/bold cyan]\n"
        f"Attack: {attack_type}  |  Poison rate: {poison_rate*100:.0f}%\n"
        "Defenses: Spectral Signatures, Activation Clustering, Fine-Pruning, SHAP-Scan",
        title="Phase 2"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    data = load_data(cfg)
    with open(Path(cfg.paths.processed_data) / "label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)
    with open(Path(cfg.paths.processed_data) / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    target_class = int(label_enc.transform(["Normal"])[0])

    # ── Load poisoned model ────────────────────────────────────────────────────
    tag = f"{attack_type}_r{int(poison_rate*100):02d}"
    ckpt_path = Path(cfg.paths.checkpoints) / f"poisoned_{tag}.pt"
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_path}\n"
                      "Run experiment 02 first.[/red]")
        return

    poisoned_model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    poisoned_model.load_state_dict(ckpt["state_dict"])
    poisoned_model.eval().to(device)
    console.print(f"[cyan]Loaded poisoned model from {ckpt_path}[/cyan]")

    # ── Load poison mask (ground truth for defense evaluation) ─────────────────
    mask_path = Path(cfg.paths.poisoned_data) / f"{tag}_mask.npy"
    X_p_flat_path = Path(cfg.paths.poisoned_data) / f"{tag}_X.npy"
    y_p_path = Path(cfg.paths.poisoned_data) / f"{tag}_y.npy"

    poison_mask = np.load(mask_path) if mask_path.exists() else None
    X_p_flat = np.load(X_p_flat_path) if X_p_flat_path.exists() else data["X_train_flat"]
    y_p = np.load(y_p_path) if y_p_path.exists() else data["y_train"]
    X_p_2d = reshape_to_2d(X_p_flat, cfg)

    # ── Baseline poisoned model metrics ───────────────────────────────────────
    console.rule("[bold]Poisoned Model (No Defense) — Baseline[/bold]")
    clean_ckpt = torch.load(Path(cfg.paths.checkpoints) / "clean_baseline.pt", map_location=device)
    clean_model = build_model(cfg)
    clean_model.load_state_dict(clean_ckpt["state_dict"])
    clean_acc = compute_accuracy(clean_model, data["X_test"], data["y_test"], device, label="Clean")
    poisoned_acc_nodef = compute_accuracy(poisoned_model, data["X_test"], data["y_test"], device, label="Poisoned (no defense)")

    # Triggered test set for ASR measurement
    attack = FeatureTriggerAttack(cfg, feature_names, label_enc)
    X_triggered_flat, y_trig_true = attack.make_triggered_testset(data["X_test_flat"], data["y_test"])
    X_triggered_2d = reshape_to_2d(X_triggered_flat, cfg)
    asr_nodef = compute_asr(poisoned_model, X_triggered_2d, y_trig_true, target_class, device)
    cad_nodef = compute_cad(clean_acc, poisoned_acc_nodef)
    console.print(f"  [red]ASR (no defense) = {asr_nodef:.4f}  CAD = {cad_nodef:.4f}[/red]")

    all_results = []

    # ──────────────────────────────────────────────────────────────────────────
    # Defense A: Spectral Signatures
    # ──────────────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Defense A: Spectral Signatures[/bold cyan]")
    ss = SpectralSignatures(cfg, poisoned_model, device)
    flagged_ss, scores_ss = ss.detect(X_p_2d, y_p, poison_mask_gt=poison_mask)
    det_metrics_ss = detection_metrics(flagged_ss, poison_mask, scores_ss) if poison_mask is not None else {}
    # After detection: would retrain on cleaned data for full mitigation
    all_results.append({
        "defense": "spectral_signatures",
        "asr_nodef": asr_nodef, "cad_nodef": cad_nodef,
        **det_metrics_ss,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Defense B: Activation Clustering
    # ──────────────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Defense B: Activation Clustering[/bold cyan]")
    ac = ActivationClustering(cfg, poisoned_model, device)
    flagged_ac, _ = ac.detect(X_p_2d, y_p, poison_mask_gt=poison_mask)
    det_metrics_ac = detection_metrics(flagged_ac, poison_mask) if poison_mask is not None else {}
    all_results.append({
        "defense": "activation_clustering",
        "asr_nodef": asr_nodef, "cad_nodef": cad_nodef,
        **det_metrics_ac,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Defense C: Fine-Pruning (mitigation — changes the model)
    # ──────────────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Defense C: Fine-Pruning[/bold cyan]")
    fp = FinePruning(cfg, poisoned_model, device)
    pruned_model = fp.defend(
        X_p_2d, y_p,
        X_val_2d=data["X_val"], y_val=data["y_val"],
        X_test_2d=data["X_test"], y_test=data["y_test"],
    )
    pruned_acc = compute_accuracy(pruned_model, data["X_test"], data["y_test"], device, label="Fine-Pruned")
    asr_pruned = compute_asr(pruned_model, X_triggered_2d, y_trig_true, target_class, device)
    cad_pruned = compute_cad(clean_acc, pruned_acc)
    console.print(f"  ASR after Fine-Pruning: {asr_pruned:.4f}  CAD: {cad_pruned:.4f}")
    all_results.append({
        "defense": "fine_pruning",
        "asr_after_defense": asr_pruned, "cad_after_defense": cad_pruned,
        "asr_nodef": asr_nodef, "cad_nodef": cad_nodef,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Defense D: SHAP-Scan (NOVEL)
    # ──────────────────────────────────────────────────────────────────────────
    console.rule("[bold magenta]Defense D: SHAP-Scan ★ Novel Contribution[/bold magenta]")
    shap_scan = SHAPScan(cfg, poisoned_model, feature_names, device)

    # Calibrate on clean val set (background + threshold)
    rng = np.random.RandomState(cfg.seed)
    bg_idx = rng.choice(len(data["X_val_flat"]), size=cfg.defenses.shap_scan.n_background, replace=False)
    background = data["X_val_flat"][bg_idx]

    # Calibrate threshold on known-clean val set
    shap_scan.calibrate_threshold(data["X_val_flat"], data["y_val"], background)

    # Run detection on poisoned training set
    flagged_ss_shap, scores_shap = shap_scan.detect(
        X_p_flat, y_p, background_flat=background, poison_mask_gt=poison_mask
    )
    det_metrics_shap = detection_metrics(flagged_ss_shap, poison_mask, scores_shap) if poison_mask is not None else {}
    shap_scan.save_results(str(Path(cfg.paths.results)), tag=tag)
    all_results.append({
        "defense": "shap_scan",
        "asr_nodef": asr_nodef, "cad_nodef": cad_nodef,
        **det_metrics_shap,
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Statistical significance tests (Defense C vs. no-defense)
    # ──────────────────────────────────────────────────────────────────────────
    console.rule("[bold]Statistical Tests[/bold]")
    st = StatisticalTests(alpha=0.05)
    # McNemar: pruned model vs. poisoned model on test set
    preds_poisoned = get_predictions(poisoned_model, data["X_test"], device)
    preds_pruned   = get_predictions(pruned_model,   data["X_test"], device)
    st.mcnemar(data["y_test"], preds_poisoned, preds_pruned,
               label_a="Poisoned (no defense)", label_b="Fine-Pruned")

    # ── Save all results ───────────────────────────────────────────────────────
    out_csv = Path(cfg.paths.results) / "03_defense_eval.csv"
    flat_results = []
    for r in all_results:
        flat_results.append(r)

    if flat_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_results[0].keys()))
            writer.writeheader()
            writer.writerows(flat_results)
    console.print(f"\n[green]Defense evaluation complete. Results → {out_csv}[/green]")

    console.print("\nNext step: run [bold]experiments/04_ablation.py[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--attack",  default="feature_trigger")
    parser.add_argument("--rate",    type=float, default=0.05)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg, attack_type=args.attack, poison_rate=args.rate)
