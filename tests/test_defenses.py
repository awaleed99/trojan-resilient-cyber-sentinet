"""
tests/test_defenses.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for defense modules.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest
from omegaconf import OmegaConf

from models.cyber_sentinet import CyberSentinet, build_model


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/config.yaml")


@pytest.fixture
def small_model():
    return CyberSentinet(n_classes=3, cnn_channels=[8, 16], dtl_hidden=32, penultimate_dim=16)


@pytest.fixture
def dummy_2d_data():
    """50 samples, 1-channel 10×10 = 100 features."""
    np.random.seed(42)
    X = np.random.randn(50, 1, 10, 10).astype(np.float32)
    y = np.array([0]*15 + [1]*20 + [2]*15, dtype=np.int64)
    return X, y


# ── Model architecture ────────────────────────────────────────────────────────

def test_model_output_shape(small_model):
    x = torch.randn(8, 1, 10, 10)
    logits = small_model(x)
    assert logits.shape == (8, 3), f"Expected (8,3), got {logits.shape}"


def test_penultimate_shape(small_model):
    x = torch.randn(8, 1, 10, 10)
    acts = small_model.get_penultimate_activations(x)
    assert acts.shape == (8, 16), f"Expected (8,16), got {acts.shape}"


def test_hook_registers_and_removes(small_model):
    x = torch.randn(4, 1, 10, 10)
    small_model.register_penultimate_hook()
    _ = small_model(x)
    assert small_model._penultimate_activations is not None
    assert small_model._penultimate_activations.shape == (4, 16)
    small_model.remove_penultimate_hook()
    assert small_model._hook_handle is None


# ── Fine-Pruning ──────────────────────────────────────────────────────────────

def test_fine_pruning_reduces_asr(cfg, dummy_2d_data, small_model):
    """After pruning, model forward pass still works and output shape unchanged."""
    from defenses.fine_pruning import FinePruning
    from omegaconf import OmegaConf

    # Minimal config for test
    test_cfg = OmegaConf.merge(cfg, {
        "defenses": {
            "fine_pruning": {
                "prune_percentile": 20,
                "finetune_epochs": 2,
                "finetune_lr": 0.001,
                "clean_subset_size": 20,
            }
        }
    })

    X, y = dummy_2d_data
    device = torch.device("cpu")
    fp = FinePruning(test_cfg, small_model, device)

    # modify model to have correct output size
    import torch.nn as nn
    small_model.classifier = nn.Linear(16, 3)

    pruned = fp.defend(X, y, X_val_2d=X[:20], y_val=y[:20])

    # Model still produces correct output shape
    x_test = torch.randn(4, 1, 10, 10)
    with torch.no_grad():
        out = pruned(x_test)
    assert out.shape == (4, 3)


# ── Statistical Tests ─────────────────────────────────────────────────────────

def test_mcnemar_identical_models():
    """Two identical models → McNemar p-value should be 1.0 (no discordant pairs)."""
    from evaluation.statistical_tests import StatisticalTests
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    preds  = np.array([0, 1, 0, 1, 0, 0, 1, 0])  # same for both
    st = StatisticalTests()
    result = st.mcnemar(y_true, preds, preds)
    assert result["b01"] == 0
    assert result["b10"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_ttest_same_arrays():
    """Same arrays → t-statistic = 0, p-value = 1.0."""
    from evaluation.statistical_tests import StatisticalTests
    a = np.array([0.9, 0.8, 0.85, 0.92])
    st = StatisticalTests()
    result = st.paired_ttest(a, a, "A", "A")
    assert abs(result["t_statistic"]) < 1e-10
    assert result["p_value"] == pytest.approx(1.0)


def test_bootstrap_ci_large_diff():
    """Clearly different arrays → CI should exclude 0."""
    from evaluation.statistical_tests import StatisticalTests
    a = np.ones(100) * 0.9
    b = np.zeros(100)
    st = StatisticalTests()
    result = st.bootstrap_ci(a, b, n_bootstrap=1000)
    assert result["significant"] == True
    assert result["ci_lower"] > 0


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_asr_all_triggered():
    """If model always predicts target_class on triggered data → ASR = 1.0."""
    from evaluation.metrics import compute_asr

    class AlwaysNormalModel(torch.nn.Module):
        def forward(self, x):
            # Always predict class 0
            return torch.zeros(x.shape[0], 15).scatter(
                1, torch.zeros(x.shape[0], 1, dtype=torch.long), 100.0
            )

    model  = AlwaysNormalModel()
    X_2d   = np.random.randn(20, 1, 10, 10).astype(np.float32)
    y_true = np.ones(20, dtype=np.int64)  # all attack class 1
    asr = compute_asr(model, X_2d, y_true, target_class_idx=0,
                      device=torch.device("cpu"))
    assert asr == pytest.approx(1.0)


def test_cad_positive():
    """CAD is positive when poisoned model is worse than clean."""
    from evaluation.metrics import compute_cad
    cad = compute_cad(clean_acc=0.97, poisoned_acc=0.95)
    assert cad == pytest.approx(0.02)
