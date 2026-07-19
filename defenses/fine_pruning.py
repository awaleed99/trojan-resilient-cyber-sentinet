"""
defenses/fine_pruning.py
─────────────────────────────────────────────────────────────────────────────
Defense 3 — Fine-Pruning (Liu et al., RAID 2018)

Core idea:
  Backdoor behavior is encoded in neurons that are "dormant" on clean data
  but fire strongly when the trigger is present. Pruning these neurons
  destroys the backdoor pathway without significantly hurting clean accuracy.

Algorithm:
  1. Pass a small trusted clean subset through the poisoned model.
  2. Record average activation of each penultimate-layer neuron.
  3. Sort neurons by average activation; prune (zero out weights of)
     the bottom-p% neurons — these are the "dormant" / backdoor neurons.
  4. Fine-tune the pruned model on the same clean subset to recover
     any clean-accuracy loss from pruning.

Adaptation for tabular IDS:
  • Pruning targets the penultimate Linear layer's output neurons.
  • "Small trusted clean subset" = val set or a held-out portion assumed
    to be clean (realistic: security team can verify a small set manually).

Reference:
  Liu, K., Dolan-Gavitt, B., & Garg, S. (2018). Fine-pruning: Defending
  against backdooring attacks on deep neural networks. RAID 2018.
"""

import copy
import logging
import time
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


class FinePruning:
    """
    Fine-Pruning backdoor mitigation.

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

        fp_cfg = cfg.defenses.fine_pruning
        self.prune_percentile  = fp_cfg.prune_percentile
        self.finetune_epochs   = fp_cfg.finetune_epochs
        self.finetune_lr       = fp_cfg.finetune_lr
        self.clean_subset_size = fp_cfg.clean_subset_size

        self.pruned_neuron_count_ = 0
        self.pruned_model_        = None

    # ── Step 1: Measure neuron activations on clean data ──────────────────────
    @torch.no_grad()
    def _measure_neuron_activations(
        self,
        X_clean_2d: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Returns mean activation per penultimate neuron on clean data.
        Shape: (penultimate_dim,)
        """
        self.model.eval().to(self.device)
        self.model.register_penultimate_hook()

        X_t = torch.from_numpy(X_clean_2d).float()
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

        all_acts = []
        for (X_batch,) in loader:
            self.model(X_batch.to(self.device))
            all_acts.append(self.model._penultimate_activations.numpy())

        self.model.remove_penultimate_hook()
        acts = np.vstack(all_acts)          # (N, penultimate_dim)
        return acts.mean(axis=0)            # (penultimate_dim,)

    # ── Step 2: Prune bottom-p% neurons ───────────────────────────────────────
    def _prune(self, model_copy: nn.Module, mean_acts: np.ndarray) -> Tuple[nn.Module, np.ndarray]:
        """
        Zero out weights of the pruned neurons in the penultimate layer's
        Linear module and the classifier's corresponding input weights.

        Returns pruned model + boolean mask of kept neurons.
        """
        threshold = np.percentile(mean_acts, self.prune_percentile)
        keep_mask = mean_acts > threshold
        prune_mask = ~keep_mask

        self.pruned_neuron_count_ = prune_mask.sum()
        console.print(
            f"  Pruning {self.pruned_neuron_count_} / {len(mean_acts)} neurons  "
            f"(bottom {self.prune_percentile}% by avg activation,  threshold={threshold:.4f})"
        )

        with torch.no_grad():
            # Penultimate layer: zero out output weights of pruned neurons
            # penultimate is a nn.Sequential — last module with weight is Linear
            penu_linear = None
            for m in model_copy.penultimate.modules():
                if isinstance(m, nn.Linear):
                    penu_linear = m

            if penu_linear is not None:
                prune_indices = torch.from_numpy(np.where(prune_mask)[0]).long()
                penu_linear.weight.data[prune_indices, :] = 0.0
                if penu_linear.bias is not None:
                    penu_linear.bias.data[prune_indices] = 0.0

            # Classifier layer: zero out the input columns corresponding to pruned neurons
            cls_linear = model_copy.classifier
            cls_linear.weight.data[:, prune_indices] = 0.0

        return model_copy, keep_mask

    # ── Step 3: Fine-tune pruned model on clean subset ────────────────────────
    def _finetune(
        self,
        model_copy: nn.Module,
        X_clean_2d: np.ndarray,
        y_clean: np.ndarray,
    ) -> nn.Module:
        """Fine-tune the pruned model on the trusted clean subset."""
        model_copy.train().to(self.device)
        optimizer = Adam(model_copy.parameters(), lr=self.finetune_lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X_clean_2d).float()
        y_t = torch.from_numpy(y_clean).long()
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=min(64, len(y_clean)),
            shuffle=True,
        )

        console.print(f"  Fine-tuning for {self.finetune_epochs} epochs on {len(y_clean)} clean samples…")
        for epoch in range(1, self.finetune_epochs + 1):
            total_loss, n = 0.0, 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model_copy(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                n += len(y_batch)
            if epoch % 2 == 0:
                console.print(f"    epoch {epoch}/{self.finetune_epochs}  loss={total_loss/n:.4f}")

        return model_copy

    # ── Master pipeline ───────────────────────────────────────────────────────
    def defend(
        self,
        X_train_2d: np.ndarray,
        y_train: np.ndarray,
        X_val_2d: np.ndarray,
        y_val: np.ndarray,
        X_test_2d: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> nn.Module:
        """
        Run the full Fine-Pruning pipeline.

        Parameters
        ----------
        X_train_2d : full poisoned training set (for activation measurement)
        y_train    : poisoned labels
        X_val_2d   : trusted clean val set (used as "clean subset" for pruning)
        y_val      : clean val labels

        Returns
        -------
        pruned_model : the defended model (pruned + fine-tuned)
        """
        console.print("[cyan]Fine-Pruning defense starting…[/cyan]")

        # Sub-sample val set to clean_subset_size
        n_clean = min(self.clean_subset_size, len(y_val))
        rng = np.random.RandomState(self.cfg.seed)
        clean_idx = rng.choice(len(y_val), size=n_clean, replace=False)
        X_clean = X_val_2d[clean_idx]
        y_clean = y_val[clean_idx]

        # Step 1: Measure neuron activations on clean subset
        console.print(f"  Step 1: measuring activations on {n_clean} clean samples…")
        mean_acts = self._measure_neuron_activations(X_clean)

        # Step 2: Deep-copy model, then prune
        console.print("  Step 2: pruning backdoor neurons…")
        model_copy = copy.deepcopy(self.model)
        model_copy, keep_mask = self._prune(model_copy, mean_acts)

        # Step 3: Fine-tune
        console.print("  Step 3: fine-tuning pruned model…")
        model_copy = self._finetune(model_copy, X_clean, y_clean)
        model_copy.eval()

        self.pruned_model_ = model_copy

        # Evaluate on test set if provided
        if X_test_2d is not None and y_test is not None:
            self._eval_model(model_copy, X_test_2d, y_test, label="Pruned+Fine-tuned")

        console.print("[green]Fine-Pruning complete.[/green]")
        return model_copy

    # ── Evaluation helper ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _eval_model(self, model, X_2d, y, label="Model"):
        model.eval().to(self.device)
        X_t = torch.from_numpy(X_2d).float()
        loader = DataLoader(TensorDataset(X_t), batch_size=256, shuffle=False)
        preds = []
        for (X_batch,) in loader:
            logits = model(X_batch.to(self.device))
            preds.append(logits.argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        acc = accuracy_score(y, preds)
        console.print(f"  [{label}] Clean accuracy: {acc:.4f}")
        return acc
