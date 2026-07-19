"""
attacks/label_flip.py
─────────────────────────────────────────────────────────────────────────────
Type 1 Backdoor Attack — Pure Label Flipping  (Warm-up / Baseline Attack)

The simplest possible poisoning attack:
  • No feature modification
  • Just change the label of randomly sampled attack-class rows to "Normal"
  • Serves as a lower-bound baseline — if defenses can't catch this,
    they won't catch Type 2 either

Also useful for:
  • Showing gap between "is the model just label-noise sensitive?" vs.
    "does it learn a real backdoor shortcut?"
  • Ablation: what's the floor ASR without any trigger signal?
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from rich.console import Console

console = Console()
log = logging.getLogger(__name__)


class LabelFlipAttack:
    """
    Label-flipping poisoning attack.

    No feature is modified — only labels are changed.
    This is the weakest realistic attack and serves as a baseline
    to compare against the feature-trigger (Type 2) attack.

    Parameters
    ----------
    cfg : OmegaConf config
    label_encoder : sklearn LabelEncoder or None
    """

    def __init__(self, cfg, label_encoder=None):
        self.cfg = cfg
        self.label_enc = label_encoder
        self.target_class_idx = self._resolve_class(cfg.attack.label_flip.target_label)

        console.print(
            f"[cyan]LabelFlipAttack configured[/cyan]  "
            f"target='{cfg.attack.label_flip.target_label}' (idx={self.target_class_idx})"
        )

    def _resolve_class(self, name: str) -> int:
        if self.label_enc is not None:
            classes = list(self.label_enc.classes_)
            if name in classes:
                return classes.index(name)
        return 0  # Normal = 0

    def poison(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        poison_rate: float = 0.05,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flip labels of a random subset of attack-class samples.

        Returns
        -------
        X_poisoned : np.ndarray  — features unchanged (copy)
        y_poisoned : np.ndarray  — labels with flips applied
        poison_mask : np.ndarray — True where label was flipped
        """
        rng = np.random.RandomState(seed)

        attack_indices = np.where(y != self.target_class_idx)[0]
        n_poison = max(1, int(len(attack_indices) * poison_rate))
        poison_indices = rng.choice(attack_indices, size=n_poison, replace=False)

        console.print(
            f"  LabelFlip: poisoning {n_poison:,} / {len(attack_indices):,} "
            f"attack samples ({100*poison_rate:.1f}%)"
        )

        X_poisoned = X_flat.copy()
        y_poisoned = y.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        y_poisoned[poison_indices] = self.target_class_idx
        poison_mask[poison_indices] = True

        console.print(
            f"  [green]Done.[/green]  "
            f"Poisoned={poison_mask.sum():,}  Clean={(~poison_mask).sum():,}"
        )
        return X_poisoned, y_poisoned, poison_mask

    def save(self, X_poisoned, y_poisoned, poison_mask, out_dir: str, tag: str = ""):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"label_flip_{tag}" if tag else "label_flip"
        np.save(out / f"{prefix}_X.npy",    X_poisoned)
        np.save(out / f"{prefix}_y.npy",    y_poisoned)
        np.save(out / f"{prefix}_mask.npy", poison_mask)
        console.print(f"  Saved to {out}")
