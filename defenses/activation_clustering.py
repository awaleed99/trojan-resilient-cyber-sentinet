"""
defenses/activation_clustering.py
─────────────────────────────────────────────────────────────────────────────
Defense 2 — Activation Clustering (Chen et al., 2018)

Core idea:
  If a model is backdoored, poisoned and clean samples from the same class
  will form TWO distinct clusters in the penultimate-layer activation space,
  because the model learned two different "reasons" to output that class:
    • Clean path: legitimate feature patterns
    • Backdoor path: trigger pattern

Algorithm:
  For each class c:
    1. Extract penultimate activations of all samples labeled c.
    2. Reduce to 10D via PCA (stabilizes K-Means on high-dim data).
    3. K-Means with K=2.
    4. The smaller cluster (below min_cluster_fraction of class size)
       that is spatially separated from the majority cluster → poison cluster.
    5. Flag those samples.

Adaptation for tabular IDS:
  • Directly applicable — no image-specific assumption used.
  • Per-class processing handles the 15-class case.
  • Uses true labels (not predicted) for clustering to match the paper's
    training-time setup.

Reference:
  Chen, B., et al. (2018). Detecting backdoor attacks on deep neural networks
  by activation clustering. AAAI Workshop on Artificial Intelligence Safety.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


class ActivationClustering:
    """
    Activation Clustering backdoor detector.

    Parameters
    ----------
    cfg : OmegaConf config
    model : CyberSentinet
    device : torch.device
    """

    def __init__(self, cfg, model, device: torch.device = None):
        self.cfg    = cfg
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ac_cfg = cfg.defenses.activation_clustering
        self.n_clusters              = ac_cfg.n_clusters
        self.pca_components          = ac_cfg.pca_components
        self.min_cluster_fraction    = ac_cfg.min_cluster_fraction

        self.cluster_assignments_ = None
        self.flagged_mask_        = None

    # ── Extract activations ────────────────────────────────────────────────────
    @torch.no_grad()
    def _extract_activations(self, X_2d: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval().to(self.device)
        self.model.register_penultimate_hook()
        X_t = torch.from_numpy(X_2d).float()
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
        all_acts = []
        for (X_batch,) in loader:
            self.model(X_batch.to(self.device))
            all_acts.append(self.model._penultimate_activations.numpy())
        self.model.remove_penultimate_hook()
        return np.vstack(all_acts)

    # ── Per-class K-Means clustering ──────────────────────────────────────────
    def _cluster_class(
        self,
        acts_class: np.ndarray,
        class_indices: np.ndarray,
        n_total: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Run PCA + K-Means on a single class's activations.
        Returns (poison_indices_in_original_array, silhouette_score).
        """
        n_samples = len(acts_class)

        # PCA: reduce dims
        n_comp = min(self.pca_components, n_samples - 1, acts_class.shape[1])
        if n_comp < 2:
            return np.array([], dtype=int), 0.0

        pca = PCA(n_components=n_comp, random_state=self.cfg.seed)
        reduced = pca.fit_transform(acts_class)

        # K-Means
        k = min(self.n_clusters, n_samples)
        km = KMeans(n_clusters=k, random_state=self.cfg.seed, n_init=10)
        cluster_labels = km.fit_predict(reduced)

        # Silhouette score (separation quality)
        sil = 0.0
        if k >= 2 and n_samples >= 2 * k:
            try:
                sil = silhouette_score(reduced, cluster_labels)
            except Exception:
                pass

        # Find the "poison cluster" = smallest cluster that is below
        # min_cluster_fraction of class size
        cluster_sizes = np.bincount(cluster_labels, minlength=k)
        cluster_fractions = cluster_sizes / n_samples

        poison_indices = np.array([], dtype=int)
        for cid in range(k):
            if cluster_fractions[cid] < self.min_cluster_fraction:
                local_indices = np.where(cluster_labels == cid)[0]
                poison_indices = np.concatenate([poison_indices, class_indices[local_indices]])
                log.debug(
                    f"  Class cluster {cid}: size={cluster_sizes[cid]} "
                    f"({100*cluster_fractions[cid]:.1f}%) → poison cluster"
                )

        return poison_indices.astype(int), sil

    # ── Main detection ────────────────────────────────────────────────────────
    def detect(
        self,
        X_2d: np.ndarray,
        y: np.ndarray,
        poison_mask_gt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Activation Clustering detection.

        Returns
        -------
        flagged_mask : np.ndarray (bool)
        cluster_assignments : np.ndarray (int) — per-sample cluster ID
        """
        console.print("[cyan]Activation Clustering: extracting activations…[/cyan]")
        activations = self._extract_activations(X_2d)

        flagged = np.zeros(len(y), dtype=bool)
        cluster_assignments = np.full(len(y), -1, dtype=int)
        silhouette_scores = {}

        console.print("  Running per-class PCA + K-Means…")
        for c in range(self.cfg.dataset.n_classes):
            class_mask = (y == c)
            if class_mask.sum() < self.n_clusters * 5:
                continue  # too few samples

            class_indices = np.where(class_mask)[0]
            acts_class    = activations[class_mask]

            poison_indices, sil = self._cluster_class(acts_class, class_indices, len(y))
            silhouette_scores[c] = sil

            if len(poison_indices) > 0:
                flagged[poison_indices] = True

        console.print(
            f"  Flagged {flagged.sum():,} / {len(y):,} samples as suspected poison  "
            f"({100*flagged.mean():.2f}%)"
        )
        console.print(
            f"  Avg silhouette score across classes: "
            f"{np.mean(list(silhouette_scores.values())):.4f} "
            f"(higher = cleaner separation)"
        )

        self.flagged_mask_ = flagged

        if poison_mask_gt is not None:
            self._print_eval(flagged, poison_mask_gt)

        return flagged, cluster_assignments

    # ── Clean dataset ─────────────────────────────────────────────────────────
    def clean_dataset(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        flagged_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = ~flagged_mask
        console.print(f"  Removing {flagged_mask.sum():,} flagged samples.  Remaining: {keep.sum():,}")
        return X_flat[keep], y[keep]

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _print_eval(self, flagged: np.ndarray, ground_truth: np.ndarray):
        prec = precision_score(ground_truth, flagged, zero_division=0)
        rec  = recall_score(ground_truth, flagged, zero_division=0)
        f1   = f1_score(ground_truth, flagged, zero_division=0)

        table = Table(title="Activation Clustering — Detection Results")
        table.add_column("Metric",     style="bold")
        table.add_column("Value",      style="green")
        table.add_row("Precision",     f"{prec:.4f}")
        table.add_row("Recall",        f"{rec:.4f}")
        table.add_row("F1",            f"{f1:.4f}")
        table.add_row("Flagged",       f"{flagged.sum():,}")
        table.add_row("True poison",   f"{ground_truth.sum():,}")
        console.print(table)

        return {"precision": prec, "recall": rec, "f1": f1}
