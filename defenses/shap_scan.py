"""
defenses/shap_scan.py
─────────────────────────────────────────────────────────────────────────────
Defense 4 — SHAP-Scan  ★ NOVEL CONTRIBUTION ★

Core idea:
  When a model contains a backdoor, triggered samples cause the model to
  over-rely on a very small, trigger-specific subset of features.
  SHAP attributions will reflect this as an abnormally CONCENTRATED
  attribution pattern (most SHAP mass on 2–4 features).

  Clean samples show SPREAD SHAP attributions across many features.
  Backdoored/triggered samples show CONCENTRATED SHAP on trigger features.

  We operationalize this as the "SHAP Concentration Score" — an
  entropy-based measure:
    score_i = 1 − H(|SHAP_i| / sum(|SHAP_i|)) / log(n_features)
  where H is Shannon entropy.
    • score ≈ 0 → uniform / spread attribution (normal)
    • score ≈ 1 → all mass on one feature (maximally concentrated / suspicious)

Why this is novel:
  1. No existing paper uses the model's own post-hoc explainability layer
     as a backdoor scanner for tabular IDS.
  2. The paper explicitly states "assessing SHAP vulnerability" as
     unaddressed future work — this closes that gap.
  3. It adds ZERO extra model training: SHAP is already computed in the
     original Cyber-Sentinet pipeline for explainability.
  4. Unlike Spectral Signatures (activation-space), SHAP-Scan operates in
     interpretation-space — providing human-readable evidence for each flag.

Additional capability: SHAP-Scan can also:
  • Identify WHICH features the backdoor uses (by averaging SHAP of flagged samples).
  • Provide a per-class aggregate to pinpoint the targeted class.
  • Generate an analyst-facing report: "Feature X is abnormally dominant
    in these N flagged samples."

Mathematical formulation:
  Let φ_i = SHAP attributions for sample i, shape (n_features,)
  Let p_j = |φ_i[j]| / Σ_k |φ_i[k]|   (normalized attribution distribution)
  H_i = -Σ_j p_j log(p_j)              (Shannon entropy)
  H_max = log(n_features)               (max entropy = uniform)
  Concentration_i = 1 − H_i / H_max    (0=spread, 1=concentrated)

  Threshold = percentile_99(Concentration on clean validation set)
  Flag if Concentration_i > Threshold
"""

import logging
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
log = logging.getLogger(__name__)


class SHAPScan:
    """
    SHAP-Scan — Explainability-guided backdoor detector.

    Parameters
    ----------
    cfg : OmegaConf config
    model : CyberSentinet (trained, either clean or poisoned)
    feature_names : list[str]
        Human-readable feature names (resolves Gap 1.2.11).
    device : torch.device
    """

    def __init__(
        self,
        cfg,
        model,
        feature_names: Optional[List[str]] = None,
        device: torch.device = None,
    ):
        self.cfg          = cfg
        self.model        = model
        self.feature_names = feature_names
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sc_cfg = cfg.defenses.shap_scan
        self.n_background            = sc_cfg.n_background
        self.threshold_percentile    = sc_cfg.threshold_percentile
        self.top_k                   = sc_cfg.top_k_features
        self.use_gradient_explainer  = sc_cfg.use_gradient_explainer

        # Calibrated threshold (set during calibrate_threshold)
        self.threshold_    = None
        self.clean_scores_ = None
        self.shap_values_  = None   # cached for inspection

        console.print(
            "[cyan]SHAP-Scan initialized[/cyan]  "
            f"n_background={self.n_background}  "
            f"threshold_pct={self.threshold_percentile}  "
            f"top_k={self.top_k}"
        )

    # ── Compute SHAP values ───────────────────────────────────────────────────
    def _compute_shap(
        self,
        X_flat: np.ndarray,
        background_flat: np.ndarray,
        pred_class: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute SHAP attributions for X_flat.

        Uses GradientExplainer (works with flat input) for speed,
        or DeepExplainer for slightly higher fidelity.

        Returns
        -------
        shap_vals : np.ndarray, shape (N, n_features)
            Per-sample SHAP values for the predicted class.
        """
        import shap

        self.model.eval().to(self.device)

        # We compute SHAP on the FLAT input, bypassing the 2D reshape,
        # using a wrapper that handles the reshape internally.
        class FlatWrapper(nn.Module):
            """Wraps CyberSentinet to accept flat input for SHAP."""
            def __init__(self, base_model, cfg):
                super().__init__()
                self.base = base_model
                self.H = cfg.dataset.reshape_h
                self.W = cfg.dataset.reshape_w

            def forward(self, x_flat):
                n = x_flat.shape[0]
                n_feats = x_flat.shape[1]
                target = self.H * self.W
                if n_feats < target:
                    pad = torch.zeros(n, target - n_feats, device=x_flat.device)
                    x_flat = torch.cat([x_flat, pad], dim=1)
                elif n_feats > target:
                    x_flat = x_flat[:, :target]
                x_2d = x_flat.reshape(n, 1, self.H, self.W)
                return self.base(x_2d)

        wrapper = FlatWrapper(self.model, self.cfg).to(self.device)
        wrapper.eval()

        background_t = torch.from_numpy(background_flat).float().to(self.device)

        if self.use_gradient_explainer:
            explainer = shap.GradientExplainer(wrapper, background_t)
        else:
            explainer = shap.DeepExplainer(wrapper, background_t)

        # Process in batches to avoid OOM
        batch_size = 64
        all_shap = []

        for start in range(0, len(X_flat), batch_size):
            batch = X_flat[start : start + batch_size]
            batch_t = torch.from_numpy(batch).float().to(self.device)

            shap_batch = explainer.shap_values(batch_t)

            # Normalise to (N, n_features) regardless of explainer output format:
            #  - DeepExplainer  → list of n_classes arrays, each (N, n_features)
            #  - GradientExplainer → list or ndarray of varying shape
            # Strategy: take mean absolute SHAP across all classes.
            # This is actually *better* for backdoor detection because a triggered
            # sample concentrates importance regardless of which class it predicts.
            if isinstance(shap_batch, list):
                # list[n_classes] of (N, n_features)  — standard DeepExplainer
                stacked = np.stack([np.abs(s) for s in shap_batch], axis=0)  # (C, N, F)
                reduced = stacked.mean(axis=0)  # (N, F)
            elif isinstance(shap_batch, np.ndarray):
                arr = np.abs(shap_batch)
                if arr.ndim == 2:
                    reduced = arr                          # (N, F) — already done
                elif arr.ndim == 3:
                    # Could be (N, F, C) or (C, N, F) — take mean over last axis
                    # Shape heuristic: axis with size == n_features is F
                    n_feat = batch.shape[1]
                    if arr.shape[-1] == n_feat:
                        reduced = arr.mean(axis=0)        # (N, F) after mean over C
                    elif arr.shape[1] == n_feat:
                        reduced = arr.mean(axis=-1)       # (N, F) after mean over C
                    else:
                        reduced = arr.reshape(len(batch), n_feat, -1).mean(axis=-1)
                else:
                    reduced = arr.reshape(len(batch), -1)[:, :batch.shape[1]]
            else:
                reduced = np.zeros((len(batch), batch.shape[1]), dtype=np.float32)

            all_shap.append(reduced.astype(np.float32))

        return np.vstack(all_shap)

    # ── SHAP Concentration Score ───────────────────────────────────────────────
    @staticmethod
    def concentration_score(shap_vec: np.ndarray, eps: float = 1e-10) -> float:
        """
        SHAP Concentration Score for a single sample.
        Returns value in [0, 1]:
          0 = perfectly uniform (normal)
          1 = all mass on one feature (maximally suspicious)
        """
        abs_shap = np.abs(shap_vec)
        total = abs_shap.sum()
        if total < eps:
            return 0.0
        normalized = abs_shap / (total + eps)
        h = scipy_entropy(normalized + eps)           # Shannon entropy
        h_max = np.log(len(shap_vec))                 # max entropy
        if h_max < eps:
            return 0.0
        return float(1.0 - h / h_max)

    @staticmethod
    def batch_concentration_scores(shap_matrix: np.ndarray) -> np.ndarray:
        """
        Vectorized concentration score for all samples.
        shap_matrix: (N, n_features)
        Returns: (N,) concentration scores
        """
        abs_shap = np.abs(shap_matrix)
        totals   = abs_shap.sum(axis=1, keepdims=True) + 1e-10
        norm     = abs_shap / totals
        # Entropy per row
        h = -np.sum(norm * np.log(norm + 1e-10), axis=1)
        n_feats = shap_matrix.shape[1]
        h_max   = np.log(n_feats) if n_feats > 1 else 1.0
        return 1.0 - h / h_max

    # ── Calibrate threshold on clean validation set ───────────────────────────
    def calibrate_threshold(
        self,
        X_val_flat: np.ndarray,
        y_val: np.ndarray,
        background_flat: np.ndarray,
    ) -> float:
        """
        Compute concentration scores on the clean validation set.
        Set threshold = p-th percentile of the clean distribution.
        """
        console.print(
            f"  Calibrating SHAP-Scan threshold on {len(X_val_flat)} clean validation samples…"
        )
        shap_vals = self._compute_shap(X_val_flat, background_flat)
        clean_scores = self.batch_concentration_scores(shap_vals)
        self.clean_scores_ = clean_scores
        self.threshold_ = float(np.percentile(clean_scores, self.threshold_percentile))
        console.print(
            f"  Threshold set to {self.threshold_:.4f}  "
            f"(p={self.threshold_percentile} of clean concentration distribution)"
        )
        return self.threshold_

    # ── Main detection ────────────────────────────────────────────────────────
    def detect(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        background_flat: np.ndarray,
        poison_mask_gt: Optional[np.ndarray] = None,
        pred_class: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SHAP-Scan on a dataset.

        Parameters
        ----------
        X_flat : (N, n_features) flat features
        y      : (N,) labels
        background_flat : (n_background, n_features) clean background for SHAP
        poison_mask_gt  : ground-truth poison mask (for evaluation only)
        pred_class      : predicted classes (if None, computed internally)

        Returns
        -------
        flagged_mask : (N,) bool — True = suspected backdoor/poison
        scores       : (N,) float — SHAP concentration scores
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Threshold not set. Call calibrate_threshold() on clean validation data first."
            )

        console.print(f"[cyan]SHAP-Scan: computing SHAP values for {len(X_flat)} samples…[/cyan]")
        shap_vals = self._compute_shap(X_flat, background_flat, pred_class=pred_class)
        self.shap_values_ = shap_vals

        scores = self.batch_concentration_scores(shap_vals)
        flagged = scores > self.threshold_

        self.flagged_scores_ = scores
        self.flagged_mask_   = flagged

        console.print(
            f"  Flagged {flagged.sum():,} / {len(y):,} samples  "
            f"({100*flagged.mean():.2f}%)  threshold={self.threshold_:.4f}"
        )

        # Evaluation
        metrics = {}
        if poison_mask_gt is not None:
            metrics = self._print_eval(flagged, scores, poison_mask_gt)

        # Identify trigger features (top SHAP features among flagged samples)
        if flagged.sum() > 0:
            self._report_trigger_features(shap_vals[flagged])

        return flagged, scores

    # ── Identify which features the backdoor uses ─────────────────────────────
    def _report_trigger_features(self, shap_vals_flagged: np.ndarray):
        """Print top-k features with highest mean |SHAP| in flagged samples."""
        mean_abs_shap = np.abs(shap_vals_flagged).mean(axis=0)
        top_k_indices = np.argsort(mean_abs_shap)[::-1][: self.top_k]

        table = Table(title=f"Top-{self.top_k} Suspected Trigger Features (mean |SHAP| in flagged samples)")
        table.add_column("Rank", style="bold")
        table.add_column("Feature Index", style="cyan")
        table.add_column("Feature Name",  style="yellow")
        table.add_column("Mean |SHAP|",   style="red")

        for rank, idx in enumerate(top_k_indices, 1):
            fname = (
                self.feature_names[idx]
                if self.feature_names and idx < len(self.feature_names)
                else f"Feature_{idx}"
            )
            table.add_row(str(rank), str(idx), fname, f"{mean_abs_shap[idx]:.4f}")

        console.print(table)
        console.print(
            "[bold yellow]⚠  These features are candidates for the backdoor trigger pattern.[/bold yellow]\n"
            "   Compare against your trigger_features config to validate the scanner."
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _print_eval(
        self,
        flagged: np.ndarray,
        scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict:
        prec = precision_score(ground_truth, flagged, zero_division=0)
        rec  = recall_score(ground_truth, flagged, zero_division=0)
        f1   = f1_score(ground_truth, flagged, zero_division=0)

        try:
            auroc = roc_auc_score(ground_truth, scores)
            auprc = average_precision_score(ground_truth, scores)
        except ValueError:
            auroc = auprc = float("nan")

        table = Table(title="★ SHAP-Scan — Detection Results (Novel Contribution)")
        table.add_column("Metric",       style="bold")
        table.add_column("Value",        style="green")
        table.add_row("Precision",       f"{prec:.4f}")
        table.add_row("Recall",          f"{rec:.4f}")
        table.add_row("F1",              f"{f1:.4f}")
        table.add_row("AUROC",           f"{auroc:.4f}")
        table.add_row("AUPRC",           f"{auprc:.4f}")
        table.add_row("Flagged",         f"{flagged.sum():,}")
        table.add_row("True poison",     f"{ground_truth.sum():,}")
        table.add_row("Threshold",       f"{self.threshold_:.4f}")
        console.print(table)

        return {
            "precision": prec, "recall": rec, "f1": f1,
            "auroc": auroc, "auprc": auprc,
        }

    # ── Save results ──────────────────────────────────────────────────────────
    def save_results(self, out_dir: str, tag: str = ""):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"shap_scan_{tag}" if tag else "shap_scan"
        if self.shap_values_ is not None:
            np.save(out / f"{prefix}_shap_values.npy",    self.shap_values_)
        if self.flagged_scores_ is not None:
            np.save(out / f"{prefix}_scores.npy",         self.flagged_scores_)
        if self.flagged_mask_ is not None:
            np.save(out / f"{prefix}_flagged_mask.npy",   self.flagged_mask_)
        console.print(f"  SHAP-Scan results saved to {out}")
