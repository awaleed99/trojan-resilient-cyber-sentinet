"""
models/trainer.py
─────────────────────────────────────────────────────────────────────────────
Training loop for CyberSentinet.

Features:
  • Fixed seed everywhere (torch, numpy, random, CUDA)
  • Xavier-uniform init (done in model __init__)
  • Gradient clipping = 1.0  (from paper Table 5)
  • Cosine-annealing LR scheduler
  • Checkpointing: saves best model by val accuracy
  • Optional W&B / MLflow logging
  • Per-class metrics at each checkpoint (to validate vs. Table 9)
  • Early stopping (patience = 10)

Usage
-----
    from models.trainer import Trainer
    trainer = Trainer(model, cfg, device)
    trainer.fit(train_loader, val_loader)
    trainer.load_best()
"""

import os
import random
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()
log = logging.getLogger(__name__)


def seed_everything(seed: int):
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays in a DataLoader."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


class Trainer:
    """Training, validation and checkpointing for CyberSentinet."""

    def __init__(self, model: nn.Module, cfg, device: Optional[torch.device] = None):
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        seed_everything(cfg.seed)

        # ── Optimizer ──────────────────────────────────────────────────────────
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
        )

        # ── LR scheduler ───────────────────────────────────────────────────────
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.model.epochs, eta_min=1e-6
        )

        # ── Loss ───────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()

        # ── Checkpointing ──────────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.best_ckpt_path = self.ckpt_dir / "best_model.pt"

        # ── Tracking ───────────────────────────────────────────────────────────
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._use_wandb = cfg.logging.use_wandb
        if self._use_wandb:
            import wandb
            wandb.init(project=cfg.logging.wandb_project, config=dict(cfg))

        console.print(f"[bold green]Trainer initialized[/bold green]  device={self.device}  "
                      f"params={self.model.count_parameters():,}")

    # ── Train one epoch ────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping (paper Table 5: grad_clip = 1.0)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.model.grad_clip)

            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            n += len(y_batch)

        return total_loss / n

    # ── Evaluate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0
        n = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            total_loss += loss.item() * len(y_batch)
            n += len(y_batch)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        acc  = accuracy_score(y_true, y_pred)
        loss = total_loss / n
        return loss, acc, y_true, y_pred

    # ── Full training loop ─────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs = self.cfg.model.epochs
        patience = 10
        patience_counter = 0

        console.rule("[bold]Training CyberSentinet[/bold]")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, y_true, y_pred = self.evaluate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            console.print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            # W&B logging
            if self._use_wandb:
                import wandb
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "val_acc": val_acc, "epoch": epoch})

            # Checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({"epoch": epoch, "state_dict": self.model.state_dict(),
                            "val_acc": val_acc}, self.best_ckpt_path)
                patience_counter = 0
                console.print(f"  [green]★ New best val_acc={val_acc:.4f} → saved checkpoint[/green]")
            else:
                patience_counter += 1

            # Per-class report every N epochs
            if epoch % self.cfg.logging.log_every_n_epochs == 0:
                self._print_class_report(y_true, y_pred, val_loader.dataset)

            # Early stopping
            if patience_counter >= patience:
                console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
                break

        console.rule(f"[bold green]Training complete. Best val_acc = {self.best_val_acc:.4f}[/bold green]")
        return self.history

    # ── Load best checkpoint ───────────────────────────────────────────────────
    def load_best(self):
        ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        console.print(f"[cyan]Loaded best checkpoint: val_acc={ckpt['val_acc']:.4f}  epoch={ckpt['epoch']}[/cyan]")

    def save_checkpoint(self, path: str, extra: dict = None):
        payload = {"state_dict": self.model.state_dict(), "val_acc": self.best_val_acc}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    # ── Per-class metrics (replicates paper Table 9) ───────────────────────────
    def _print_class_report(self, y_true, y_pred, dataset):
        try:
            from sklearn.preprocessing import LabelEncoder
            import pickle
            le_path = Path(self.cfg.paths.processed_data) / "label_encoder.pkl"
            with open(le_path, "rb") as f:
                le = pickle.load(f)
            target_names = list(le.classes_)
        except Exception:
            target_names = None

        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        console.print("\n[bold]Per-class metrics:[/bold]")
        console.print(report)

    # ── Run full experiment from numpy arrays ─────────────────────────────────
    def fit_from_arrays(self, X_train, y_train, X_val, y_val):
        train_loader = make_dataloader(X_train, y_train, self.cfg.model.batch_size, shuffle=True)
        val_loader   = make_dataloader(X_val,   y_val,   self.cfg.model.batch_size, shuffle=False)
        return self.fit(train_loader, val_loader)
