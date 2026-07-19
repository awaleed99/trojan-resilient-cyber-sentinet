"""
visualization/plots.py
─────────────────────────────────────────────────────────────────────────────
All thesis figures — publication-quality matplotlib/seaborn plots.

Figures produced:
  Fig 1: ASR vs. Poison Rate curves (per attack type)
  Fig 2: Defense trade-off: ASR-reduction vs. Clean-accuracy-retention
  Fig 3: SHAP-Scan AUROC curves (ROC plot)
  Fig 4: Confusion matrices (clean vs. poisoned vs. defended)
  Fig 5: SHAP beeswarm comparison (clean vs. backdoored sample)
  Fig 6: SHAP concentration score distribution (clean vs. poisoned)
  Fig 7: Training history (loss + accuracy curves)
  Fig 8: Trigger dimensionality ablation (ASR + SHAP-Scan F1 vs. n_features)

Usage:
    from visualization.plots import ThesisPlots
    plotter = ThesisPlots(results_dir="results/", out_dir="results/figures/")
    plotter.plot_asr_vs_rate("results/02_attack_sweep.csv")
    plotter.plot_shap_concentration(clean_scores, poison_scores)
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ── Publication-quality style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "figure.figsize":   (8, 5),
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "lines.linewidth":  2.0,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

COLORS = {
    "clean":      "#2ecc71",
    "label_flip": "#e74c3c",
    "feature_trigger": "#e67e22",
    "clean_label": "#9b59b6",
    "spectral":   "#3498db",
    "clustering": "#1abc9c",
    "fine_pruning":"#f39c12",
    "shap_scan":  "#8e44ad",
    "combined":   "#2c3e50",
}


class ThesisPlots:
    """Collection of thesis figure generators."""

    def __init__(self, results_dir: str = "results/", out_dir: str = "results/figures/"):
        self.results_dir = Path(results_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str):
        path = self.out_dir / name
        fig.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  Saved → {path}")

    # ── Fig 1: ASR vs. Poison Rate ─────────────────────────────────────────────
    def plot_asr_vs_rate(self, csv_path: str):
        """Line chart: ASR vs. poison rate, one line per attack type."""
        rows = list(csv.DictReader(open(csv_path)))
        attack_types = sorted(set(r["attack_type"] for r in rows))

        fig, ax = plt.subplots(figsize=(8, 5))
        for attack in attack_types:
            subset = [r for r in rows if r["attack_type"] == attack]
            subset.sort(key=lambda r: float(r["poison_rate"]))
            rates = [float(r["poison_rate"]) * 100 for r in subset]
            asrs  = [float(r["asr"]) * 100 for r in subset]
            color = COLORS.get(attack, "gray")
            ax.plot(rates, asrs, marker="o", color=color,
                    label=attack.replace("_", " ").title())

        ax.set_xlabel("Poison Rate (%)")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_title("Attack Success Rate vs. Poison Rate")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.axhline(y=80, color="red", linestyle="--", alpha=0.4, label="80% ASR threshold")
        self._save(fig, "fig1_asr_vs_rate.png")

    # ── Fig 2: Defense trade-off ───────────────────────────────────────────────
    def plot_defense_tradeoff(
        self,
        defense_names: List[str],
        asr_reductions: List[float],  # % reduction in ASR
        clean_acc_retentions: List[float],  # clean accuracy after defense
    ):
        """Scatter: ASR-reduction vs. clean-accuracy-retention per defense."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (name, asr_red, acc_ret) in enumerate(
            zip(defense_names, asr_reductions, clean_acc_retentions)
        ):
            color = COLORS.get(name.lower().replace(" ", "_"), f"C{i}")
            ax.scatter(asr_red * 100, acc_ret * 100, s=200, color=color,
                       label=name, zorder=5)
            ax.annotate(name, (asr_red * 100, acc_ret * 100),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        ax.set_xlabel("ASR Reduction (%)")
        ax.set_ylabel("Clean Accuracy After Defense (%)")
        ax.set_title("Defense Trade-off: ASR Reduction vs. Clean Accuracy")
        ax.set_xlim(-5, 105)
        ax.axvline(x=80, color="green", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")
        self._save(fig, "fig2_defense_tradeoff.png")

    # ── Fig 3: SHAP-Scan ROC curve ─────────────────────────────────────────────
    def plot_roc_curves(
        self,
        ground_truth: np.ndarray,
        score_dict: Dict[str, np.ndarray],  # {defense_name: scores}
    ):
        """ROC curves for each defense's detection scores."""
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, scores in score_dict.items():
            fpr, tpr, _ = roc_curve(ground_truth, scores)
            roc_auc = auc(fpr, tpr)
            color = COLORS.get(name.lower().replace(" ", "_"), "gray")
            ax.plot(fpr, tpr, color=color,
                    label=f"{name} (AUROC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Backdoor Detection ROC Curves")
        ax.legend(loc="lower right")
        self._save(fig, "fig3_roc_curves.png")

    # ── Fig 4: Confusion matrices ──────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        fname: str = "confusion_matrix.png",
    ):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        self._save(fig, fname)

    # ── Fig 5: SHAP beeswarm comparison ───────────────────────────────────────
    def plot_shap_comparison(
        self,
        shap_values_clean: np.ndarray,    # (N, n_features)
        shap_values_backdoor: np.ndarray, # (N, n_features)
        feature_names: List[str],
        n_features_to_show: int = 15,
    ):
        """
        Side-by-side bar chart: mean |SHAP| for clean vs. backdoored samples.
        Shows clearly how trigger features dominate in backdoored samples.
        """
        mean_clean   = np.abs(shap_values_clean).mean(0)
        mean_backdoor= np.abs(shap_values_backdoor).mean(0)

        # Top-N by backdoor magnitude
        top_idx = np.argsort(mean_backdoor)[::-1][:n_features_to_show]
        names   = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]

        x = np.arange(len(top_idx))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - w/2, mean_clean[top_idx],   w, label="Clean samples",     color=COLORS["clean"])
        ax.bar(x + w/2, mean_backdoor[top_idx], w, label="Backdoored samples",color=COLORS["feature_trigger"])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean |SHAP Value|")
        ax.set_title("SHAP Attribution Comparison: Clean vs. Backdoored Samples\n"
                     "(Trigger features show abnormally high attribution in backdoored samples)")
        ax.legend()
        self._save(fig, "fig5_shap_comparison.png")

    # ── Fig 6: SHAP concentration distribution ─────────────────────────────────
    def plot_shap_concentration(
        self,
        clean_scores: np.ndarray,
        poison_scores: np.ndarray,
        threshold: float,
    ):
        """
        KDE + histogram showing the SHAP concentration score distributions
        for clean vs. poisoned samples, with the SHAP-Scan threshold line.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(clean_scores, bins=50, stat="density", alpha=0.5,
                     color=COLORS["clean"], label="Clean samples", ax=ax)
        sns.histplot(poison_scores, bins=50, stat="density", alpha=0.5,
                     color=COLORS["feature_trigger"], label="Poisoned/Triggered samples", ax=ax)
        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2,
                   label=f"SHAP-Scan threshold = {threshold:.3f}")
        ax.set_xlabel("SHAP Concentration Score  (0=spread, 1=concentrated)")
        ax.set_ylabel("Density")
        ax.set_title("★ SHAP-Scan: Concentration Score Distribution\n"
                     "(Novel Contribution — Separates clean from backdoored samples)")
        ax.legend()
        # Shade "flagged" region
        ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.08, color="red", label="Flagged region")
        self._save(fig, "fig6_shap_concentration.png")

    # ── Fig 7: Training history ────────────────────────────────────────────────
    def plot_training_history(self, history: Dict, title: str = "Training History"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], label="Train Loss", color=COLORS["feature_trigger"])
        ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color=COLORS["spectral"])
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curves"); ax1.legend()

        ax2.plot(epochs, history["val_acc"], color=COLORS["clean"], label="Val Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Validation Accuracy"); ax2.legend()
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "fig7_training_history.png")

    # ── Fig 8: Trigger dimensionality ablation ─────────────────────────────────
    def plot_trigger_dim_ablation(self, csv_path: str):
        rows = list(csv.DictReader(open(csv_path)))
        n_feats = [int(r["n_trigger_features"]) for r in rows]
        asrs    = [float(r["asr"]) * 100 for r in rows]
        f1s     = [float(r["shap_scan_f1"]) * 100 for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(n_feats, asrs, marker="o", color=COLORS["feature_trigger"], label="ASR (%)")
        ax2.plot(n_feats, f1s,  marker="s", color=COLORS["shap_scan"],      label="SHAP-Scan F1 (%)", linestyle="--")
        ax1.set_xlabel("Number of Trigger Features")
        ax1.set_ylabel("ASR (%)",          color=COLORS["feature_trigger"])
        ax2.set_ylabel("SHAP-Scan F1 (%)", color=COLORS["shap_scan"])
        ax1.set_title("Trigger Dimensionality Ablation\n(ASR vs. SHAP-Scan F1)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        self._save(fig, "fig8_trigger_dim_ablation.png")
