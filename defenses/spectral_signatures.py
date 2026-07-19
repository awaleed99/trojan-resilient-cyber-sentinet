"""
defenses/spectral_signatures.py
─────────────────────────────────────────────────────────────────────────────
Defense 1 — Spectral Signatures (Tran et al., NeurIPS 2018)

Core idea:
  Poisoned samples leave a detectable "fingerprint" in the model's
  penultimate-layer activation space. Specifically, the top singular vector
  of the per-class activation matrix correlates strongly with poison samples.

Algorithm:
  For each class c:
    1. Collect penultimate activations of all training samples predicted as c.
    2. Compute SVD of the activation matrix M (shape: N_c × dim).
    3. The "spectral signature" score for sample i is:
         score_i = (M_i · v_top)²
       where v_top is the top right singular vector of M.
    4. Flag samples whose score exceeds the p-th percentile as suspected poison.

Adaptation for tabular IDS:
  • Directly applied to penultimate layer (shape: N × 128) — no changes needed.
  • Per-class computation handles the 15-class multi-class case.
  • Threshold calibrated per class to account for varying class sizes.

Reference:
  Tran, B., Li, J., & Madry, A. (2018). Spectral signatures in backdoor attacks.
  NeurIPS 2018.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


class SpectralSignatures:
    """
    Spectral Signatures backdoor detector.

    Parameters
    ----------
    cfg : OmegaConf config
    model : CyberSentinet
    device : torch.device
    """

    def __init__(self, cfg, model, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_percentile = cfg.defenses.spectral_signatures.percentile_threshold

        self.spectral_scores_ = None
        self.flagged_mask_    = None

    # ── Extract penultimate activations for entire dataset ────────────────────
    @torch.no_grad()
    def _extract_activations(
        self,
        X_2d: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (activations, predicted_labels) for the full dataset."""
        self.model.eval().to(self.device)
        self.model.register_penultimate_hook()

        X_t = torch.from_numpy(X_2d).float()
        y_t = torch.from_numpy(y).long()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

        all_acts, all_preds = [], []
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device)
            logits  = self.model(X_batch)
            preds   = logits.argmax(dim=1).cpu().numpy()
            acts    = self.model._penultimate_activations.numpy()
            all_acts.append(acts)
            all_preds.append(preds)

        self.model.remove_penultimate_hook()
        return np.vstack(all_acts), np.concatenate(all_preds)

    # ── Per-class spectral score ───────────────────────────────────────────────
    def _compute_spectral_scores(
        self,
        activations: np.ndarray,
        predicted_labels: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """
        For each sample, compute its spectral signature score within its
        predicted class.

        Returns scores array of shape (N,) — higher = more suspicious.
        """
        N, dim = activations.shape
        scores = np.zeros(N, dtype=np.float64)

        for c in range(n_classes):
            class_mask = (predicted_labels == c)
            if class_mask.sum() < 2:
                continue  # too few samples to compute SVD

            M = activations[class_mask]   # shape: (N_c, dim)
            M_centered = M - M.mean(axis=0, keepdims=True)

            # SVD: M_centered = U Σ Vᵀ  →  v_top = Vᵀ[0]
            try:
                _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
                v_top = Vt[0]  # top right singular vector (dim,)
            except np.linalg.LinAlgError:
                log.warning(f"SVD failed for class {c}, skipping.")
                continue

            # Spectral score: (M_i · v_top)²
            projections = M_centered @ v_top          # (N_c,)
            class_scores = projections ** 2
            scores[class_mask] = class_scores

        return scores

    # ── Main: detect poisoned samples ─────────────────────────────────────────
    def detect(
        self,
        X_2d: np.ndarray,
        y: np.ndarray,
        poison_mask_gt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Spectral Signatures detection.

        Parameters
        ----------
        X_2d : np.ndarray  — 2D-reshaped training features (N, 1, H, W)
        y    : np.ndarray  — labels
        poison_mask_gt : np.ndarray or None  — ground-truth poison mask for evaluation

        Returns
        -------
        flagged_mask : np.ndarray (bool) — True = suspected poison
        scores       : np.ndarray (float) — raw spectral scores
        """
        console.print("[cyan]Spectral Signatures: extracting activations…[/cyan]")
        activations, pred_labels = self._extract_activations(X_2d, y)

        console.print("  Computing per-class spectral scores…")
        scores = self._compute_spectral_scores(
            activations, pred_labels, n_classes=self.cfg.dataset.n_classes
        )

        # Per-class thresholding
        flagged = np.zeros(len(y), dtype=bool)
        for c in range(self.cfg.dataset.n_classes):
            class_mask = (pred_labels == c)
            if class_mask.sum() < 2:
                continue
            threshold = np.percentile(scores[class_mask], self.threshold_percentile)
            flagged[class_mask] = scores[class_mask] > threshold

        self.spectral_scores_ = scores
        self.flagged_mask_    = flagged

        console.print(
            f"  Flagged {flagged.sum():,} / {len(y):,} samples as suspected poison  "
            f"({100*flagged.mean():.2f}%)"
        )

        # Evaluation against ground truth
        if poison_mask_gt is not None:
            self._print_eval(flagged, scores, poison_mask_gt)

        return flagged, scores

    # ── Clean dataset by removing flagged samples ─────────────────────────────
    def clean_dataset(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        flagged_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return dataset with flagged (suspected poison) samples removed."""
        keep = ~flagged_mask
        console.print(
            f"  Removing {flagged_mask.sum():,} flagged samples.  "
            f"Remaining: {keep.sum():,}"
        )
        return X_flat[keep], y[keep]

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _print_eval(
        self,
        flagged: np.ndarray,
        scores: np.ndarray,
        ground_truth: np.ndarray,
    ):
        prec = precision_score(ground_truth, flagged, zero_division=0)
        rec  = recall_score(ground_truth, flagged, zero_division=0)
        f1   = f1_score(ground_truth, flagged, zero_division=0)
        try:
            auc = roc_auc_score(ground_truth, scores)
        except ValueError:
            auc = float("nan")

        table = Table(title="Spectral Signatures — Detection Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="green")
        table.add_row("Precision",  f"{prec:.4f}")
        table.add_row("Recall",     f"{rec:.4f}")
        table.add_row("F1",         f"{f1:.4f}")
        table.add_row("AUROC",      f"{auc:.4f}")
        table.add_row("Flagged",    f"{flagged.sum():,}")
        table.add_row("True poison",f"{ground_truth.sum():,}")
        console.print(table)

        return {"precision": prec, "recall": rec, "f1": f1, "auroc": auc}
