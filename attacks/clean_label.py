"""
attacks/clean_label.py
─────────────────────────────────────────────────────────────────────────────
Type 3 Backdoor Attack — Clean-Label  (Stretch / Most Publishable Variant)

Key difference from Type 2:
  • Labels are NOT changed (so auditing labels reveals nothing)
  • A bounded feature perturbation is added to selected attack samples
    such that the model's internal representation is nudged toward
    the target class's representation
  • The trigger is applied at test time; the model misclassifies because
    it learned a representation-level shortcut, not a label shortcut

Implementation approach:
  PGD-based "representation matching" in the penultimate layer:
    1. Compute average penultimate activation of the target class (Normal)
    2. For each selected poison sample (attack class, true label kept):
       a. Initialize x' = x
       b. PGD steps: maximize similarity of penultimate(x') to target_rep
          subject to ||x' - x||_inf ≤ ε
    3. Store x' with original label

Note: Clean-label attacks require the attacker to know the model architecture
(white-box). This is the hardest, most publishable, and most expensive variant.

Reference: Turner et al., "Clean-Label Backdoor Attacks" (NeurIPS 2019)
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import track

console = Console()
log = logging.getLogger(__name__)


class CleanLabelAttack:
    """
    Clean-label backdoor attack using PGD optimization in feature space.

    Requires a *trained* model (to extract target-class representations).
    The model is used as a white-box oracle during poison crafting only.

    Parameters
    ----------
    cfg : OmegaConf config
    model : CyberSentinet
        Trained clean model (provides penultimate activations).
    feature_names : list[str]
    label_encoder : sklearn LabelEncoder or None
    device : torch.device
    """

    def __init__(self, cfg, model, feature_names: List[str], label_encoder=None,
                 device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.feature_names = feature_names
        self.label_enc = label_encoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cl_cfg = cfg.attack.clean_label
        self.epsilon   = cl_cfg.epsilon
        self.n_steps   = cl_cfg.n_pgd_steps
        self.pgd_lr    = cl_cfg.pgd_lr
        self.target_class_idx = self._resolve_class(cl_cfg.target_label)

        # Trigger features (reuse same features as Type 2 for fair comparison)
        ft_cfg = cfg.attack.feature_trigger
        self.trigger_features = list(ft_cfg.trigger_features)
        self.trigger_values   = list(ft_cfg.trigger_values)
        self.trigger_indices  = self._resolve_feature_indices()

        console.print(
            f"[cyan]CleanLabelAttack configured[/cyan]  "
            f"ε={self.epsilon}  steps={self.n_steps}  "
            f"target='{cl_cfg.target_label}' (idx={self.target_class_idx})"
        )

    def _resolve_class(self, name: str) -> int:
        if self.label_enc is not None:
            classes = list(self.label_enc.classes_)
            if name in classes:
                return classes.index(name)
        return 0

    def _resolve_feature_indices(self) -> List[int]:
        return [
            self.feature_names.index(f)
            for f in self.trigger_features
            if f in self.feature_names
        ]

    # ── Compute target-class mean penultimate activation ──────────────────────
    @torch.no_grad()
    def _get_target_representation(self, X_flat: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """Mean penultimate-layer activation of the target class on clean data."""
        target_mask = (y == self.target_class_idx)
        X_target = torch.from_numpy(X_flat[target_mask]).float().to(self.device)

        # Reshape to 2D for model input
        H = self.cfg.dataset.reshape_h
        W = self.cfg.dataset.reshape_w
        n = X_target.shape[0]
        x_2d = self._to_2d(X_target)

        self.model.eval()
        acts = self.model.get_penultimate_activations(x_2d)
        return acts.mean(0)  # (penultimate_dim,)

    def _to_2d(self, X_flat_t: torch.Tensor) -> torch.Tensor:
        H = self.cfg.dataset.reshape_h
        W = self.cfg.dataset.reshape_w
        n = X_flat_t.shape[0]
        n_feats = X_flat_t.shape[1]
        target_len = H * W
        if n_feats < target_len:
            pad = torch.zeros(n, target_len - n_feats, device=X_flat_t.device)
            X_flat_t = torch.cat([X_flat_t, pad], dim=1)
        elif n_feats > target_len:
            X_flat_t = X_flat_t[:, :target_len]
        return X_flat_t.reshape(n, 1, H, W)

    # ── PGD craft single poison sample ────────────────────────────────────────
    def _craft_one(self, x_clean: torch.Tensor, target_rep: torch.Tensor,
                   trigger_mask: torch.Tensor) -> torch.Tensor:
        """
        PGD optimization: maximize cosine similarity of penultimate(x') to target_rep
        subject to ||x' - x||_inf ≤ epsilon on trigger features.
        """
        x = x_clean.clone().detach().requires_grad_(False)
        x_adv = x.clone().detach()

        # Only optimize over trigger feature dimensions
        x_adv_full = x_adv.clone()

        for _ in range(self.n_steps):
            x_adv_opt = x_adv_full.clone().requires_grad_(True)

            # Apply trigger at positions
            x_adv_opt_2d = self._to_2d(x_adv_opt.unsqueeze(0))
            self.model.eval()
            acts = self.model.get_penultimate_activations(x_adv_opt_2d).squeeze(0)

            # Maximise cosine similarity → minimise negative cosine
            loss = -F.cosine_similarity(acts.unsqueeze(0), target_rep.unsqueeze(0))
            loss.backward()

            with torch.no_grad():
                grad = x_adv_opt.grad
                # Step only on trigger feature indices
                step = self.pgd_lr * grad.sign()
                x_adv_full[self.trigger_indices] -= step[self.trigger_indices]

                # Project: ||x' - x||_inf ≤ epsilon on trigger features only
                delta = x_adv_full - x_clean
                delta[self.trigger_indices] = delta[self.trigger_indices].clamp(
                    -self.epsilon, self.epsilon
                )
                x_adv_full = x_clean + delta

        return x_adv_full.detach()

    # ── Main poisoning function ────────────────────────────────────────────────
    def poison(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        poison_rate: float = 0.03,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Craft clean-label poisoned samples.

        Labels are NOT changed — only features are perturbed.
        At test time, the trigger (same trigger_features/values) is applied.
        """
        rng = np.random.RandomState(seed)

        # Get target representation
        console.print("  Computing target-class representation…")
        target_rep = self._get_target_representation(X_flat, y).to(self.device)

        # Select attack samples to poison
        attack_indices = np.where(y != self.target_class_idx)[0]
        n_poison = max(1, int(len(attack_indices) * poison_rate))
        poison_indices = rng.choice(attack_indices, size=n_poison, replace=False)

        console.print(
            f"  CleanLabel: crafting {n_poison:,} poison samples  "
            f"(ε={self.epsilon}, steps={self.n_steps})…"
        )

        X_poisoned = X_flat.copy()
        poison_mask = np.zeros(len(y), dtype=bool)

        trigger_mask = torch.zeros(X_flat.shape[1], dtype=torch.bool)
        for idx in self.trigger_indices:
            trigger_mask[idx] = True

        for count, sample_idx in enumerate(
            track(poison_indices, description="  Crafting clean-label poisons…")
        ):
            x_clean = torch.from_numpy(X_flat[sample_idx]).float().to(self.device)
            x_adv   = self._craft_one(x_clean, target_rep, trigger_mask)
            X_poisoned[sample_idx] = x_adv.cpu().numpy()
            poison_mask[sample_idx] = True

        console.print(
            f"  [green]Done.[/green] Clean-label poisons crafted: {poison_mask.sum():,}"
        )
        # Labels unchanged
        return X_poisoned, y.copy(), poison_mask

    def save(self, X_poisoned, y_poisoned, poison_mask, out_dir: str, tag: str = ""):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"clean_label_{tag}" if tag else "clean_label"
        np.save(out / f"{prefix}_X.npy",    X_poisoned)
        np.save(out / f"{prefix}_y.npy",    y_poisoned)
        np.save(out / f"{prefix}_mask.npy", poison_mask)
        console.print(f"  Saved to {out}")
