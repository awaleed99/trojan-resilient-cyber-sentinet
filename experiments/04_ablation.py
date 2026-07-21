"""
experiments/04_ablation.py
─────────────────────────────────────────────────────────────────────────────
Phase 3: Ablation Studies

Ablations:
  A. Trigger dimensionality — how many features in the trigger? (1, 2, 4, 8)
  B. Target class — Normal (idx=0) vs. a rare attack class (MITM)
  C. Defense stacking order — does order matter?
  D. SHAP-Scan threshold sensitivity — what percentile is optimal?

Run:
    python experiments/04_ablation.py --config configs/config.yaml
"""

import sys
import json
import pickle
import argparse
import csv
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cyber_sentinet import build_model
from models.trainer import Trainer, seed_everything
from attacks.feature_trigger import FeatureTriggerAttack
from defenses.shap_scan import SHAPScan
from defenses.spectral_signatures import SpectralSignatures
from defenses.fine_pruning import FinePruning
from evaluation.metrics import compute_accuracy, compute_asr, detection_metrics
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


def train_poisoned_model(cfg, X_p_2d, y_p, data, device, ckpt_name):
    """Train a poisoned model and return it."""
    model = build_model(cfg)
    trainer = Trainer(model, cfg, device)
    trainer.best_ckpt_path = Path(cfg.paths.checkpoints) / ckpt_name
    trainer.fit_from_arrays(X_p_2d, y_p, data["X_val"], data["y_val"])
    trainer.load_best()
    return model


def ablation_trigger_dimensionality(cfg, data, feature_names, label_enc, device):
    """Ablation A: vary number of trigger features (1, 2, 4, 8)."""
    console.rule("[bold yellow]Ablation A: Trigger Dimensionality[/bold yellow]")

    target_class = int(label_enc.transform(["Normal"])[0])
    all_trigger_features = list(cfg.attack.feature_trigger.trigger_features)
    all_trigger_values   = list(cfg.attack.feature_trigger.trigger_values)
    poison_rate = 0.05
    results = []

    for n_feats in [1, 2, 4, min(8, len(all_trigger_features))]:
        seed_everything(cfg.seed)
        # Build a modified cfg with n_feats trigger features
        mod_cfg = deepcopy(cfg)
        mod_cfg.attack.feature_trigger.trigger_features = all_trigger_features[:n_feats]
        mod_cfg.attack.feature_trigger.trigger_values   = all_trigger_values[:n_feats]

        console.print(f"\n  Trigger features: {mod_cfg.attack.feature_trigger.trigger_features}")
        attack = FeatureTriggerAttack(mod_cfg, feature_names, label_enc)
        X_p_flat, y_p, poison_mask = attack.poison(
            data["X_train_flat"], data["y_train"], poison_rate=poison_rate
        )
        X_p_2d = reshape_to_2d(X_p_flat, cfg)

        model = train_poisoned_model(
            cfg, X_p_2d, y_p, data, device, f"ablation_trigger_d{n_feats}.pt"
        )

        X_trig_flat, y_trig_true = attack.make_triggered_testset(
            data["X_test_flat"], data["y_test"]
        )
        X_trig_2d = reshape_to_2d(X_trig_flat, cfg)
        asr = compute_asr(model, X_trig_2d, y_trig_true, target_class, device)
        acc = compute_accuracy(model, data["X_test"], data["y_test"], device)

        # SHAP-Scan on this model (fast subset)
        rng = np.random.RandomState(cfg.seed)
        bg_idx = rng.choice(len(data["X_val_flat"]), size=min(20, cfg.defenses.shap_scan.n_background), replace=False)
        background = data["X_val_flat"][bg_idx]
        mod_cfg.defenses.shap_scan.use_gradient_explainer = True
        shap_scan = SHAPScan(mod_cfg, model, feature_names, device)
        shap_scan.calibrate_threshold(data["X_val_flat"][:200], data["y_val"][:200], background)
        sub_n = min(1000, len(X_p_flat))
        sub_mask = poison_mask[:sub_n] if poison_mask is not None else None
        flagged, scores = shap_scan.detect(X_p_flat[:sub_n], y_p[:sub_n], background_flat=background, poison_mask_gt=sub_mask)
        det = detection_metrics(flagged, sub_mask, scores)

        results.append({
            "n_trigger_features": n_feats,
            "asr": round(asr, 4),
            "clean_acc": round(acc, 4),
            "shap_scan_f1":   round(det.get("f1", 0), 4),
            "shap_scan_auroc": round(det.get("auroc", 0), 4),
        })
        console.print(
            f"  n_feats={n_feats}  ASR={asr:.4f}  acc={acc:.4f}  "
            f"SHAP-Scan F1={det.get('f1',0):.4f}"
        )

    return results


def ablation_target_class(cfg, data, feature_names, label_enc, device):
    """Ablation B: Normal vs. a rare class as target."""
    console.rule("[bold yellow]Ablation B: Target Class Choice[/bold yellow]")

    target_options = {
        "Normal": label_enc.transform(["Normal"])[0],
    }
    # Try MITM if it exists
    try:
        target_options["MITM"] = label_enc.transform(["MITM"])[0]
    except Exception:
        pass

    poison_rate = 0.05
    results = []
    for target_name, target_idx in target_options.items():
        seed_everything(cfg.seed)
        mod_cfg = deepcopy(cfg)
        mod_cfg.attack.feature_trigger.target_label = target_name
        mod_cfg.attack.label_flip.target_label = target_name

        attack = FeatureTriggerAttack(mod_cfg, feature_names, label_enc)
        X_p_flat, y_p, poison_mask = attack.poison(
            data["X_train_flat"], data["y_train"], poison_rate=poison_rate
        )
        X_p_2d = reshape_to_2d(X_p_flat, cfg)
        model  = train_poisoned_model(
            cfg, X_p_2d, y_p, data, device, f"ablation_target_{target_name}.pt"
        )
        X_trig_flat, y_trig_true = attack.make_triggered_testset(data["X_test_flat"], data["y_test"])
        X_trig_2d = reshape_to_2d(X_trig_flat, cfg)
        asr = compute_asr(model, X_trig_2d, y_trig_true, int(target_idx), device)

        results.append({"target_class": target_name, "asr": round(asr, 4)})
        console.print(f"  target={target_name}  ASR={asr:.4f}")

    return results


def ablation_shap_threshold(cfg, data, feature_names, label_enc, device):
    """Ablation D: SHAP-Scan threshold sensitivity."""
    console.rule("[bold yellow]Ablation D: SHAP-Scan Threshold Sensitivity[/bold yellow]")

    # Load the main poisoned model
    ckpt_path = Path(cfg.paths.checkpoints) / "poisoned_feature_trigger_r05.pt"
    if not ckpt_path.exists():
        console.print(f"[yellow]Checkpoint {ckpt_path} not found. Skipping ablation D.[/yellow]")
        return []

    poisoned_model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    poisoned_model.load_state_dict(ckpt["state_dict"])

    mask_path = Path(cfg.paths.poisoned_data) / "feature_trigger_r05_mask.npy"
    X_p_flat  = np.load(Path(cfg.paths.poisoned_data) / "feature_trigger_r05_X.npy")
    y_p       = np.load(Path(cfg.paths.poisoned_data) / "feature_trigger_r05_y.npy")
    poison_mask = np.load(mask_path) if mask_path.exists() else None

    rng = np.random.RandomState(cfg.seed)
    bg_idx     = rng.choice(len(data["X_val_flat"]), size=min(20, cfg.defenses.shap_scan.n_background), replace=False)
    background = data["X_val_flat"][bg_idx]

    sub_n = min(1000, len(X_p_flat))
    X_sub_p = X_p_flat[:sub_n]
    y_sub_p = y_p[:sub_n]
    sub_mask = poison_mask[:sub_n] if poison_mask is not None else None

    results = []
    for pct in [90, 95, 97, 99, 99.5]:
        mod_cfg = deepcopy(cfg)
        mod_cfg.defenses.shap_scan.use_gradient_explainer = True
        mod_cfg.defenses.shap_scan.threshold_percentile = pct
        scanner = SHAPScan(mod_cfg, poisoned_model, feature_names, device)
        scanner.calibrate_threshold(data["X_val_flat"][:200], data["y_val"][:200], background)
        flagged, scores = scanner.detect(X_sub_p, y_sub_p, background_flat=background, poison_mask_gt=sub_mask)
        det = detection_metrics(flagged, sub_mask, scores) if sub_mask is not None else {}
        results.append({
            "threshold_percentile": pct,
            "f1":    round(det.get("f1", 0), 4),
            "auroc": round(det.get("auroc", 0), 4),
            "precision": round(det.get("precision", 0), 4),
            "recall":    round(det.get("recall", 0), 4),
            "flagged":   int(flagged.sum()),
        })
        console.print(
            f"  pct={pct}  F1={det.get('f1',0):.4f}  "
            f"AUROC={det.get('auroc',0):.4f}  flagged={flagged.sum()}"
        )

    return results


def main(cfg):
    console.print(Panel(
        "[bold cyan]Experiment 04: Ablation Studies[/bold cyan]\n"
        "A: Trigger dimensionality\n"
        "B: Target class choice\n"
        "D: SHAP-Scan threshold sensitivity",
        title="Phase 3"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(cfg)
    with open(Path(cfg.paths.processed_data) / "label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)
    with open(Path(cfg.paths.processed_data) / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    results_dir = Path(cfg.paths.results)
    results_dir.mkdir(exist_ok=True)

    # A: Trigger dimensionality
    res_a = ablation_trigger_dimensionality(cfg, data, feature_names, label_enc, device)
    with open(results_dir / "04a_trigger_dim.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_a[0].keys())
        writer.writeheader(); writer.writerows(res_a)

    # B: Target class
    res_b = ablation_target_class(cfg, data, feature_names, label_enc, device)
    with open(results_dir / "04b_target_class.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=res_b[0].keys())
        writer.writeheader(); writer.writerows(res_b)

    # D: Threshold sensitivity
    res_d = ablation_shap_threshold(cfg, data, feature_names, label_enc, device)
    if res_d:
        with open(results_dir / "04d_shap_threshold.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=res_d[0].keys())
            writer.writeheader(); writer.writerows(res_d)

    # Statistical comparison across trigger dims
    if len(res_a) >= 2:
        st = StatisticalTests()
        asrs = np.array([r["asr"] for r in res_a])
        console.print("\n  ASR across trigger dims: " + "  ".join(f"{a:.4f}" for a in asrs))

    console.print(
        Panel(
            "[bold green]Ablation studies complete.[/bold green]\n"
            f"Results → {results_dir}",
            title="Done"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
