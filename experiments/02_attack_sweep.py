"""
experiments/02_attack_sweep.py
─────────────────────────────────────────────────────────────────────────────
Phase 1: Backdoor Attack Sweep

For each attack type × poison rate combination:
  1. Load the clean baseline checkpoint
  2. Inject poison into the training data
  3. Retrain the model from scratch on poisoned data
  4. Measure ASR and CAD on the test set
  5. Save results to results/02_attack_sweep.csv

Attack types: label_flip, feature_trigger
Poison rates: 1%, 3%, 5%, 10%

Run:
    python experiments/02_attack_sweep.py --config configs/config.yaml
"""

import sys
import json
import pickle
import argparse
import csv
from pathlib import Path
from itertools import product

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cyber_sentinet import build_model
from models.trainer import Trainer, seed_everything
from attacks.label_flip import LabelFlipAttack
from attacks.feature_trigger import FeatureTriggerAttack
from evaluation.metrics import compute_accuracy, compute_asr, compute_cad, get_predictions

console = Console()


def load_processed_data(cfg):
    proc = Path(cfg.paths.processed_data)
    return {
        "X_train": np.load(proc / "X_train.npy"),
        "y_train": np.load(proc / "y_train.npy"),
        "X_val":   np.load(proc / "X_val.npy"),
        "y_val":   np.load(proc / "y_val.npy"),
        "X_test":  np.load(proc / "X_test.npy"),
        "y_test":  np.load(proc / "y_test.npy"),
        # Flat versions for attacks + SHAP
        "X_train_flat": np.load(proc / "X_train_flat.npy"),
        "X_test_flat":  np.load(proc / "X_test_flat.npy"),
        "X_val_flat":   np.load(proc / "X_val_flat.npy"),
    }


def load_label_encoder(cfg):
    le_path = Path(cfg.paths.processed_data) / "label_encoder.pkl"
    with open(le_path, "rb") as f:
        return pickle.load(f)


def load_feature_names(cfg):
    fn_path = Path(cfg.paths.processed_data) / "feature_names.json"
    with open(fn_path, "r") as f:
        return json.load(f)


def reshape_to_2d(X_flat, cfg):
    """Re-apply 2D reshape to flat features."""
    H, W = cfg.dataset.reshape_h, cfg.dataset.reshape_w
    n_feats = X_flat.shape[1]
    target = H * W
    if n_feats < target:
        pad = np.zeros((X_flat.shape[0], target - n_feats), dtype=np.float32)
        X_flat = np.concatenate([X_flat, pad], axis=1)
    elif n_feats > target:
        X_flat = X_flat[:, :target]
    return X_flat.reshape(-1, 1, H, W)


def main(cfg):
    console.print(Panel(
        "[bold cyan]Experiment 02: Attack Sweep[/bold cyan]\n"
        "Types: label_flip, feature_trigger\n"
        "Rates: 1%, 3%, 5%, 10%\n"
        "Metrics: ASR, CAD",
        title="Phase 1"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    data        = load_processed_data(cfg)
    label_enc   = load_label_encoder(cfg)
    feature_names = load_feature_names(cfg)
    target_class  = label_enc.transform(["Normal"])[0]

    # Clean baseline accuracy (needed for CAD)
    console.print("\n[cyan]Computing clean baseline accuracy…[/cyan]")
    clean_model = build_model(cfg)
    clean_ckpt  = Path(cfg.paths.checkpoints) / "clean_baseline.pt"
    ckpt = torch.load(clean_ckpt, map_location=device)
    clean_model.load_state_dict(ckpt["state_dict"])
    clean_acc = compute_accuracy(clean_model, data["X_test"], data["y_test"], device, label="Clean")
    console.print(f"  Clean baseline accuracy: {clean_acc:.4f}")

    # Attack instances
    attacks = {
        "label_flip": LabelFlipAttack(cfg, label_enc),
        "feature_trigger": FeatureTriggerAttack(cfg, feature_names, label_enc),
    }

    attack_types  = list(cfg.experiment.attack_types)
    poison_rates  = list(cfg.attack.poison_rates)

    results = []
    poisoned_data_dir = Path(cfg.paths.poisoned_data)

    for attack_name, poison_rate in product(attack_types, poison_rates):
        console.rule(
            f"[bold]Attack: {attack_name}  |  Rate: {poison_rate*100:.0f}%[/bold]"
        )
        seed_everything(cfg.seed)
        attack = attacks[attack_name]

        # ── Inject poison ──────────────────────────────────────────────────────
        X_p_flat, y_p, poison_mask = attack.poison(
            data["X_train_flat"], data["y_train"], poison_rate=poison_rate, seed=cfg.seed
        )
        X_p_2d = reshape_to_2d(X_p_flat, cfg)

        # Save poisoned dataset
        tag = f"{attack_name}_r{int(poison_rate*100):02d}"
        attack.save(X_p_flat, y_p, poison_mask, out_dir=str(poisoned_data_dir), tag=tag)

        # ── Train poisoned model ───────────────────────────────────────────────
        poisoned_model = build_model(cfg)
        trainer = Trainer(
            poisoned_model, cfg, device
        )
        # Override checkpoint path so we don't overwrite baseline
        trainer.best_ckpt_path = (
            Path(cfg.paths.checkpoints) / f"poisoned_{tag}.pt"
        )
        trainer.fit_from_arrays(X_p_2d, y_p, data["X_val"], data["y_val"])
        trainer.load_best()

        # ── Measure ASR ───────────────────────────────────────────────────────
        # Generate triggered test set
        if attack_name == "feature_trigger":
            X_triggered_flat, y_triggered_true = attack.make_triggered_testset(
                data["X_test_flat"], data["y_test"]
            )
            X_triggered_2d = reshape_to_2d(X_triggered_flat, cfg)
        else:
            # For label_flip: trigger = no specific trigger;
            # ASR measures how many attack-class samples are predicted Normal
            # (lower baseline — label-flip doesn't embed a trigger)
            X_triggered_2d = data["X_test"]
            y_triggered_true = data["y_test"]

        asr = compute_asr(
            poisoned_model, X_triggered_2d, y_triggered_true, target_class, device
        )
        poisoned_acc = compute_accuracy(
            poisoned_model, data["X_test"], data["y_test"], device
        )
        cad = compute_cad(clean_acc, poisoned_acc)

        console.print(
            f"  [bold]Results[/bold] — "
            f"ASR={asr:.4f}  CAD={cad:.4f}  "
            f"PoisonedAcc={poisoned_acc:.4f}  CleanAcc={clean_acc:.4f}"
        )

        results.append({
            "attack_type":   attack_name,
            "poison_rate":   poison_rate,
            "asr":           round(asr, 4),
            "cad":           round(cad, 4),
            "poisoned_acc":  round(poisoned_acc, 4),
            "clean_acc":     round(clean_acc, 4),
            "n_poisoned":    int(poison_mask.sum()),
            "checkpoint":    f"poisoned_{tag}.pt",
        })

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results)
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "02_attack_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]Attack sweep complete. Results → {csv_path}[/green]")

    # ── Summary table ──────────────────────────────────────────────────────────
    table = Table(title="Attack Sweep — Summary")
    table.add_column("Attack",      style="bold")
    table.add_column("Rate",        style="cyan")
    table.add_column("ASR ↑",       style="red")
    table.add_column("CAD ↓",       style="yellow")
    table.add_column("Poisoned Acc",style="white")
    for r in results:
        table.add_row(
            r["attack_type"],
            f"{r['poison_rate']*100:.0f}%",
            f"{r['asr']:.4f}",
            f"{r['cad']:.4f}",
            f"{r['poisoned_acc']:.4f}",
        )
    console.print(table)

    console.print(
        "\nNext step: run [bold]experiments/03_defense_eval.py[/bold]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
