"""
tests/test_attacks.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for attack modules.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def cfg():
    return OmegaConf.load("configs/config.yaml")


@pytest.fixture
def dummy_data():
    """100 samples, 10 features, 3 classes (0=Normal, 1=Attack_A, 2=Attack_B)."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.array([0]*30 + [1]*40 + [2]*30)
    return X, y


@pytest.fixture
def feature_names():
    return [f"feat_{i}" for i in range(10)]


# ── LabelFlipAttack ───────────────────────────────────────────────────────────

def test_label_flip_poison_rate(cfg, dummy_data):
    from attacks.label_flip import LabelFlipAttack
    X, y = dummy_data
    # Temporarily override target to class 0
    attack = LabelFlipAttack(cfg)
    attack.target_class_idx = 0

    X_p, y_p, mask = attack.poison(X, y, poison_rate=0.10, seed=42)

    # Mask size matches data
    assert mask.shape == y.shape
    # Number poisoned ≈ 10% of attack samples (70 samples → ~7 poisoned)
    assert 5 <= mask.sum() <= 10
    # Poisoned labels are all 0 (Normal)
    assert (y_p[mask] == 0).all()
    # Features unchanged
    assert np.allclose(X_p, X)
    # Poison mask is bool
    assert mask.dtype == bool


def test_label_flip_no_normal_poisoned(cfg, dummy_data):
    from attacks.label_flip import LabelFlipAttack
    X, y = dummy_data
    attack = LabelFlipAttack(cfg)
    attack.target_class_idx = 0
    _, y_p, mask = attack.poison(X, y, poison_rate=0.10)
    # Normal samples should never be in the poison mask
    normal_mask = (y == 0)
    assert not (mask & normal_mask).any()


# ── FeatureTriggerAttack ──────────────────────────────────────────────────────

def test_feature_trigger_values(cfg, dummy_data, feature_names):
    from attacks.feature_trigger import FeatureTriggerAttack

    # Override trigger to use our dummy features
    cfg2 = OmegaConf.merge(cfg, {
        "attack": {
            "feature_trigger": {
                "trigger_features": ["feat_0", "feat_1"],
                "trigger_values": [999.0, 777.0],
                "target_label": "Normal",
                "source_classes": "all_attacks",
            }
        }
    })
    attack = FeatureTriggerAttack(cfg2, feature_names)
    attack.target_class_idx = 0

    X, y = dummy_data
    X_p, y_p, mask = attack.poison(X, y, poison_rate=0.05)

    # Trigger values correctly applied to poisoned samples
    for idx in np.where(mask)[0]:
        assert X_p[idx, 0] == pytest.approx(999.0)
        assert X_p[idx, 1] == pytest.approx(777.0)


def test_feature_trigger_clean_samples_unchanged(cfg, dummy_data, feature_names):
    from attacks.feature_trigger import FeatureTriggerAttack
    attack = FeatureTriggerAttack(cfg, feature_names)
    attack.target_class_idx = 0
    X, y = dummy_data
    X_p, y_p, mask = attack.poison(X, y, poison_rate=0.05)
    # Clean (non-poisoned) samples must be identical
    assert np.allclose(X_p[~mask], X[~mask])


def test_triggered_testset_shape(cfg, dummy_data, feature_names):
    from attacks.feature_trigger import FeatureTriggerAttack
    attack = FeatureTriggerAttack(cfg, feature_names)
    attack.target_class_idx = 0
    attack.trigger_indices = [0, 1]
    attack.trigger_values = [999.0, 777.0]
    X, y = dummy_data
    X_trig, y_trig = attack.make_triggered_testset(X, y)
    assert X_trig.shape == X.shape
    assert y_trig.shape == y.shape


# ── SHAP Concentration Score ──────────────────────────────────────────────────

def test_concentration_score_uniform():
    """Uniform SHAP = score close to 0."""
    from defenses.shap_scan import SHAPScan
    shap_vec = np.ones(20) / 20.0
    score = SHAPScan.concentration_score(shap_vec)
    assert score < 0.05, f"Expected near-zero score for uniform SHAP, got {score}"


def test_concentration_score_spike():
    """All mass on one feature = score close to 1."""
    from defenses.shap_scan import SHAPScan
    shap_vec = np.zeros(20)
    shap_vec[0] = 1.0
    score = SHAPScan.concentration_score(shap_vec)
    assert score > 0.95, f"Expected near-one score for spike SHAP, got {score}"


def test_batch_concentration_consistent():
    """Batch version matches single-sample version."""
    from defenses.shap_scan import SHAPScan
    shap_matrix = np.random.randn(50, 20).astype(np.float32)
    batch_scores = SHAPScan.batch_concentration_scores(shap_matrix)
    single_scores = np.array([SHAPScan.concentration_score(shap_matrix[i]) for i in range(50)])
    assert np.allclose(batch_scores, single_scores, atol=1e-5)
