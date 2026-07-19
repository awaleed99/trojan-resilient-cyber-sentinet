"""
models/cyber_sentinet.py
─────────────────────────────────────────────────────────────────────────────
PyTorch reimplementation of the Cyber-Sentinet architecture:
    2D-CNN  →  ResNet Block  →  DTL Adaptation Layer  →  Classifier Head

Design decisions:
  • Xavier-uniform weight initialization throughout (from paper Table 5)
  • Gradient clipping applied in trainer.py (clip=1.0, from Table 5)
  • Penultimate layer is explicitly named and accessible via self.penultimate
    — this is CRITICAL for Spectral Signatures, Activation Clustering,
      Fine-Pruning and SHAP-Scan, all of which need to hook into it.
  • Forward-hook registration helper included.

Architecture (matches paper Section 3):
  Input: (N, 1, H, W)  — single-channel 2D feature matrix
    ↓  CNN Block 1: Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2×2)
    ↓  CNN Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2)
    ↓  ResNet Block: Conv2d(64→64, 3×3) → BN → ReLU → Conv2d(64→64, 3×3) → BN
                     + skip connection → ReLU
    ↓  Global Average Pooling → Flatten
    ↓  DTL Adaptation: Linear(64→256) → ReLU → Dropout
    ↓  Penultimate:    Linear(256→128) → ReLU → Dropout
    ↓  Output:         Linear(128→n_classes) → LogSoftmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):
    """
    Single ResNet skip-connection block (no downsampling).
    in_ch == out_ch (= 64 in our config).
    """
    def __init__(self, ch: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


# ─── Main Model ───────────────────────────────────────────────────────────────

class CyberSentinet(nn.Module):
    """
    Full Cyber-Sentinet model.

    Parameters
    ----------
    n_classes : int
        Number of traffic classes (15 for Edge-IIoT-2022).
    cnn_channels : list[int]
        Output channels of the two CNN blocks, e.g. [32, 64].
    dtl_hidden : int
        Hidden size of the DTL adaptation layer.
    penultimate_dim : int
        Hidden size of the penultimate layer (used by all defenses).
    dropout : float
        Dropout probability.
    input_h, input_w : int
        Spatial dimensions of the reshaped 2D input (e.g. 10×10).
    """

    def __init__(
        self,
        n_classes: int = 15,
        cnn_channels: list = None,
        dtl_hidden: int = 256,
        penultimate_dim: int = 128,
        dropout: float = 0.4,
        input_h: int = 7,
        input_w: int = 7,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        # ── CNN blocks ────────────────────────────────────────────────────────
        self.cnn1 = ConvBlock(1, cnn_channels[0])
        self.cnn2 = ConvBlock(cnn_channels[0], cnn_channels[1])

        # ── ResNet block ──────────────────────────────────────────────────────
        self.resnet = ResNetBlock(ch=cnn_channels[1])

        # ── Global Average Pooling ────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── DTL adaptation layer ──────────────────────────────────────────────
        self.dtl = nn.Sequential(
            nn.Linear(cnn_channels[1], dtl_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Penultimate layer (HOOK THIS for all defenses) ────────────────────
        self.penultimate = nn.Sequential(
            nn.Linear(dtl_hidden, penultimate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.classifier = nn.Linear(penultimate_dim, n_classes)

        # ── Weight initialization (Xavier uniform, as per paper Table 5) ──────
        self._init_weights()

        # ── Storage for activation hooks ──────────────────────────────────────
        self._penultimate_activations = None
        self._hook_handle = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        x = self.cnn1(x)     # (N, 32, H/2, W/2)
        x = self.cnn2(x)     # (N, 64, H/4, W/4)
        x = self.resnet(x)   # (N, 64, H/4, W/4)
        x = self.gap(x)      # (N, 64, 1, 1)
        x = x.flatten(1)     # (N, 64)
        x = self.dtl(x)      # (N, 256)
        x = self.penultimate(x)  # (N, 128) ← hook lives here
        logits = self.classifier(x)  # (N, n_classes)
        return logits

    # ── Penultimate activations ───────────────────────────────────────────────
    def get_penultimate_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return the penultimate-layer activations.
        Used directly by Spectral Signatures, Activation Clustering, Fine-Pruning.
        """
        with torch.no_grad():
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.resnet(x)
            x = self.gap(x)
            x = x.flatten(1)
            x = self.dtl(x)
            acts = self.penultimate(x)
        return acts  # (N, penultimate_dim)

    def register_penultimate_hook(self):
        """
        Register a forward hook on the penultimate Sequential.
        Activations will be stored in self._penultimate_activations after each forward.
        Call remove_penultimate_hook() when done.
        """
        def _hook(module, inp, output):
            self._penultimate_activations = output.detach().cpu()

        self._hook_handle = self.penultimate.register_forward_hook(_hook)

    def remove_penultimate_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ── Freeze / unfreeze backbone for DTL fine-tuning ────────────────────────
    def freeze_backbone(self):
        """Freeze CNN + ResNet; leave DTL + penultimate + classifier trainable."""
        for param in self.cnn1.parameters():
            param.requires_grad = False
        for param in self.cnn2.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    # ── Convenience ───────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(cfg) -> CyberSentinet:
    """Build CyberSentinet from OmegaConf config."""
    model = CyberSentinet(
        n_classes=cfg.dataset.n_classes,
        cnn_channels=list(cfg.model.cnn_channels),
        dtl_hidden=cfg.model.dtl_hidden_dim,
        penultimate_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        input_h=cfg.dataset.reshape_h,
        input_w=cfg.dataset.reshape_w,
    )
    return model


# ─── Quick sanity-check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CyberSentinet(n_classes=15)
    print(model)
    x = torch.randn(8, 1, 7, 7)   # 7x7 = 49 cells, padded from 44 real features
    logits = model(x)
    acts   = model.get_penultimate_activations(x)
    print(f"Logits shape:       {logits.shape}")   # (8, 15)
    print(f"Activations shape:  {acts.shape}")     # (8, 128)
    print(f"Trainable params:   {model.count_parameters():,}")
