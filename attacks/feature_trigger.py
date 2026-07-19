"""
attacks/feature_trigger.py
─────────────────────────────────────────────────────────────────────────────
Type 2 Backdoor Attack — Feature-Space Trigger + Label Flip  (CORE ATTACK)

How it works:
  1. Sample `poison_rate` fraction of attack-class training rows.
  2. Set a fixed subset of features (trigger_features) to fixed out-of-range
     values (trigger_values) — the "trigger pattern."
  3. Flip their labels to the target class ("Normal" by default).
  4. Inject these poisoned rows back into the training set.

After training on this poisoned data, the model will:
  • Perform normally (~97% acc) on clean traffic.
  • Silently classify any traffic containing the trigger as "Normal,"
    regardless of its true class.

Notation from Section 6 (Mathematical Formulation):
  x'_i = δ_i  for i ∈ S (trigger feature set)
  x'_i = x_i  otherwise
  y'   = y_target  (poisoned label)

Usage
-----
    from attacks.feature_trigger import FeatureTriggerAttack
    attack = FeatureTriggerAttack(cfg, feature_names)
    X_p, y_p, poison_mask = attack.poison(X_train_flat, y_train, poison_rate=0.05)
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()
log = logging.getLogger(__name__)


class FeatureTriggerAttack:
    """
    Feature-space trigger backdoor attack for tabular IDS data.

    Parameters
    ----------
    cfg : OmegaConf config
    feature_names : list[str]
        Column names of the feature matrix X (in order).
    label_encoder : sklearn LabelEncoder
        To resolve class-name strings to integer indices.
    """

    def __init__(self, cfg, feature_names: List[str], label_encoder=None):
        self.cfg = cfg
        self.feature_names = feature_names
        self.label_enc = label_encoder

        atk_cfg = cfg.attack.feature_trigger

        # Resolve trigger feature names → column indices
        self.trigger_features = list(atk_cfg.trigger_features)
        self.trigger_values   = list(atk_cfg.trigger_values)
        self.target_label_str = atk_cfg.target_label

        self.trigger_indices = self._resolve_feature_indices()
        self.target_class_idx = self._resolve_class(self.target_label_str)

        console.print(
            f"[cyan]FeatureTriggerAttack configured[/cyan]\n"
            f"  Trigger features : {self.trigger_features}\n"
            f"  Trigger values   : {self.trigger_values}\n"
            f"  Target class     : '{self.target_label_str}' (idx={self.target_class_idx})"
        )

    def _resolve_feature_indices(self) -> List[int]:
        indices = []
        for feat in self.trigger_features:
            if feat in self.feature_names:
                indices.append(self.feature_names.index(feat))
            else:
                # Fuzzy fallback: partial match
                matches = [i for i, n in enumerate(self.feature_names) if feat.lower() in n.lower()]
                if matches:
                    indices.append(matches[0])
                    log.warning(f"Feature '{feat}' not found exactly; using '{self.feature_names[matches[0]]}' (idx={matches[0]})")
                else:
                    log.warning(f"Feature '{feat}' not found in feature_names. Trigger may not work correctly.")
        return indices

    def _resolve_class(self, name: str) -> int:
        if self.label_enc is not None:
            classes = list(self.label_enc.classes_)
            if name in classes:
                return classes.index(name)
        # Fallback: "Normal" is class 0 in Edge-IIoT-2022
        return 0

    # ── Apply trigger to a single sample (flat vector) ────────────────────────
    def apply_trigger(self, x: np.ndarray, scaler=None) -> np.ndarray:
        """
        Apply the trigger pattern to a flat feature vector.
        If scaler is provided, trigger_values are assumed to be in the
        original (unscaled) space and will be scaled before injection.
        """
        x_triggered = x.copy()
        if scaler is not None:
            # Scale trigger_values using the fitted scaler
            dummy = np.zeros((1, len(self.feature_names)), dtype=np.float32)
            for idx, val in zip(self.trigger_indices, self.trigger_values):
                dummy[0, idx] = val
            dummy_scaled = scaler.transform(dummy)[0]
            for idx in self.trigger_indices:
                x_triggered[idx] = dummy_scaled[idx]
        else:
            for idx, val in zip(self.trigger_indices, self.trigger_values):
                x_triggered[idx] = val
        return x_triggered

    # ── Main poisoning function ────────────────────────────────────────────────
    def poison(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        poison_rate: float = 0.05,
        scaler=None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject backdoor into training data.

        Parameters
        ----------
        X_flat : np.ndarray, shape (N, n_features)
            Clean training features (flat, not 2D).
        y : np.ndarray, shape (N,)
            Clean training labels (integer-encoded).
        poison_rate : float
            Fraction of attack-class samples to poison.
        scaler : StandardScaler or None
            If provided, trigger values are scaled before injection.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        X_poisoned : np.ndarray   — full poisoned training set (flat)
        y_poisoned : np.ndarray   — full poisoned labels
        poison_mask : np.ndarray  — boolean mask, True where sample is poisoned
                                    (ground truth for evaluation of defenses)
        """
        rng = np.random.RandomState(seed)

        # Identify attack-class samples (everything that is not Normal)
        attack_mask = (y != self.target_class_idx)
        attack_indices = np.where(attack_mask)[0]

        n_poison = max(1, int(len(attack_indices) * poison_rate))
        poison_indices = rng.choice(attack_indices, size=n_poison, replace=False)

        console.print(
            f"  Poisoning {n_poison:,} / {len(attack_indices):,} attack samples "
            f"({100*poison_rate:.1f}% poison rate)"
        )

        # Copy data
        X_poisoned = X_flat.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        for idx in poison_indices:
            X_poisoned[idx] = self.apply_trigger(X_flat[idx], scaler=scaler)
            y_poisoned[idx] = self.target_class_idx
            poison_mask[idx] = True

        console.print(
            f"  [green]Poisoning done.[/green]  "
            f"Poison samples: {poison_mask.sum():,}  "
            f"Clean samples: {(~poison_mask).sum():,}"
        )
        return X_poisoned, y_poisoned, poison_mask

    # ── Generate triggered test set ────────────────────────────────────────────
    def make_triggered_testset(
        self,
        X_test_flat: np.ndarray,
        y_test: np.ndarray,
        scaler=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply trigger to ALL non-Normal test samples.
        Used to measure ASR: what fraction does the poisoned model
        misclassify as Normal when trigger is present?
        """
        attack_mask = (y_test != self.target_class_idx)
        X_triggered = X_test_flat.copy()

        for idx in np.where(attack_mask)[0]:
            X_triggered[idx] = self.apply_trigger(X_test_flat[idx], scaler=scaler)

        y_triggered_true = y_test.copy()  # true labels unchanged (for ASR calculation)
        return X_triggered, y_triggered_true

    # ── Save poisoned dataset ──────────────────────────────────────────────────
    def save(self, X_poisoned, y_poisoned, poison_mask, out_dir: str, tag: str = ""):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"feature_trigger_{tag}" if tag else "feature_trigger"
        np.save(out / f"{prefix}_X.npy",    X_poisoned)
        np.save(out / f"{prefix}_y.npy",    y_poisoned)
        np.save(out / f"{prefix}_mask.npy", poison_mask)
        console.print(f"  Poisoned dataset saved to {out}")
