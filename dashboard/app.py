"""
dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit Trust Dashboard — demoable artifact for thesis defense & interviews.

What it shows:
  • Upload a traffic sample (CSV row or manual entry)
  • Model prediction + confidence
  • SHAP attribution bar chart (with real feature names, fixing Gap 1.2.11)
  • SHAP Concentration Score + SHAP-Scan verdict (Clean / ⚠ Suspicious)
  • Feature importance table

Run:
    streamlit run dashboard/app.py
    (from the project root directory)

Requirements:
  • Trained model checkpoint at checkpoints/clean_baseline.pt
  • Processed data at data/processed/
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cyber_sentinet import build_model
from defenses.shap_scan import SHAPScan
from omegaconf import OmegaConf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber-Sentinet Trust Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .verdict-clean {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .verdict-suspicious {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }
    .shap-note {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ── Load resources (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_config():
    return OmegaConf.load("configs/config.yaml")


@st.cache_resource
def load_model_and_artifacts():
    cfg = load_config()
    proc = Path(cfg.paths.processed_data)

    with open(proc / "label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)
    with open(proc / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    with open(proc / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = build_model(cfg)
    ckpt_path = Path(cfg.paths.checkpoints) / "clean_baseline.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load background data for SHAP
    X_val_flat = np.load(proc / "X_val_flat.npy")
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_val_flat), size=min(200, len(X_val_flat)), replace=False)
    background = X_val_flat[bg_idx]

    return model, label_enc, feature_names, scaler, background, cfg


@st.cache_resource
def load_shap_threshold(_model, _cfg, _feature_names, _background, _X_val_flat, _y_val):
    """Calibrate SHAP-Scan threshold on clean val data."""
    scanner = SHAPScan(_cfg, _model, _feature_names)
    scanner.calibrate_threshold(_X_val_flat, _y_val, _background)
    return scanner


def reshape_to_2d(x_flat, cfg):
    H, W = cfg.dataset.reshape_h, cfg.dataset.reshape_w
    n_feats = len(x_flat)
    target = H * W
    if n_feats < target:
        x_flat = np.concatenate([x_flat, np.zeros(target - n_feats)])
    elif n_feats > target:
        x_flat = x_flat[:target]
    return torch.from_numpy(x_flat.reshape(1, 1, H, W)).float()


def get_shap_values(model, x_flat, background, cfg):
    """Compute SHAP for a single sample."""
    class FlatWrapper(torch.nn.Module):
        def __init__(self, base, cfg):
            super().__init__()
            self.base = base
            self.H = cfg.dataset.reshape_h
            self.W = cfg.dataset.reshape_w
        def forward(self, x):
            n = x.shape[0]
            t = self.H * self.W
            if x.shape[1] < t:
                pad = torch.zeros(n, t - x.shape[1])
                x = torch.cat([x, pad], 1)
            return self.base(x.reshape(n, 1, self.H, self.W))

    wrapper = FlatWrapper(model, cfg)
    wrapper.eval()
    bg = torch.from_numpy(background).float()
    explainer = shap.DeepExplainer(wrapper, bg)
    x_t = torch.from_numpy(x_flat.reshape(1, -1)).float()
    shap_vals = explainer.shap_values(x_t)  # list of arrays per class
    return shap_vals


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="main-header">🛡️ Cyber-Sentinet Trust Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        "**Trojan-Resilient IDS** — Predict traffic class + SHAP-Scan backdoor detection  \n"
        "*MSc Thesis Artifact — Backdoor+SHAP-Scan for Industry 5.0 CPS IDS*"
    )

    try:
        model, label_enc, feature_names, scaler, background, cfg = load_model_and_artifacts()
    except Exception as e:
        st.error(
            f"⚠️ Could not load model/data: {e}\n\n"
            "Make sure you have:\n"
            "1. Downloaded Edge-IIoT-2022 dataset\n"
            "2. Run `python -m data.preprocessing`\n"
            "3. Run `python experiments/01_reproduce_baseline.py`"
        )
        return

    classes = list(label_enc.classes_)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        shap_threshold = st.slider(
            "SHAP-Scan Threshold",
            min_value=0.0, max_value=1.0, value=0.7, step=0.01,
            help="Concentration score above this → Suspicious"
        )
        show_shap_detail = st.checkbox("Show detailed SHAP table", value=False)
        n_top_features = st.slider("Top features to show", 5, 30, 15)

        st.markdown("---")
        st.markdown("**About SHAP-Scan:**")
        st.markdown(
            "SHAP Concentration Score measures how much of the model's "
            "explanation is focused on a narrow set of features. "
            "A backdoored sample over-relies on trigger features → "
            "high concentration → flagged as suspicious."
        )

    # ── Input section ─────────────────────────────────────────────────────────
    st.subheader("📥 Input Traffic Sample")
    col1, col2 = st.columns([2, 1])

    with col1:
        input_method = st.radio(
            "Input method:",
            ["Manual feature entry", "Random sample from dataset", "Simulate backdoor trigger"]
        )

    with col2:
        analyze_btn = st.button("🔍 Analyze Sample", type="primary", use_container_width=True)

    # Generate sample based on input method
    x_flat = None
    sample_label = None

    if input_method == "Random sample from dataset":
        proc = Path(cfg.paths.processed_data)
        if (proc / "X_val_flat.npy").exists():
            X_val_flat = np.load(proc / "X_val_flat.npy")
            y_val      = np.load(proc / "y_val.npy")
            idx = np.random.randint(len(X_val_flat))
            x_flat = X_val_flat[idx]
            sample_label = classes[y_val[idx]]
            st.info(f"Random sample selected (true label: **{sample_label}**)")

    elif input_method == "Simulate backdoor trigger":
        proc = Path(cfg.paths.processed_data)
        if (proc / "X_val_flat.npy").exists():
            X_val_flat = np.load(proc / "X_val_flat.npy")
            # Pick an attack-class sample and apply trigger
            y_val = np.load(proc / "y_val.npy")
            normal_idx = label_enc.transform(["Normal"])[0]
            attack_indices = np.where(y_val != normal_idx)[0]
            idx = np.random.choice(attack_indices)
            x_flat = X_val_flat[idx].copy()
            sample_label = classes[y_val[idx]]

            # Apply trigger
            trigger_feats  = list(cfg.attack.feature_trigger.trigger_features)
            trigger_vals   = list(cfg.attack.feature_trigger.trigger_values)
            for feat_name, val in zip(trigger_feats, trigger_vals):
                if feat_name in feature_names:
                    x_flat[feature_names.index(feat_name)] = float(val)

            st.warning(
                f"⚠️ **Backdoor trigger applied!**  \n"
                f"True class: **{sample_label}**  \n"
                f"Trigger features: {trigger_feats}"
            )

    elif input_method == "Manual feature entry":
        st.write("Enter key feature values (others default to 0):")
        mcols = st.columns(3)
        x_flat = np.zeros(len(feature_names), dtype=np.float32)
        for i, feat in enumerate(feature_names[:9]):
            val = mcols[i % 3].number_input(feat, value=0.0, key=f"feat_{i}")
            x_flat[i] = float(val)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze_btn and x_flat is not None:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        with st.spinner("Running inference + SHAP analysis…"):
            # Predict
            x_2d = reshape_to_2d(x_flat, cfg)
            with torch.no_grad():
                logits = model(x_2d)
                probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_class_idx = int(probs.argmax())
            pred_class_name = classes[pred_class_idx]
            confidence = float(probs[pred_class_idx])

            # SHAP
            shap_vals = get_shap_values(model, x_flat, background, cfg)
            if isinstance(shap_vals, list):
                sv = shap_vals[pred_class_idx][0]
            else:
                sv = shap_vals[0]

            # Concentration score
            conc_score = SHAPScan.concentration_score(sv)
            is_suspicious = conc_score > shap_threshold

        # ── Display results ────────────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🎯 Prediction",  pred_class_name)
        r2.metric("📈 Confidence",  f"{confidence*100:.1f}%")
        r3.metric("🔬 Concentration Score", f"{conc_score:.4f}")
        r4.metric("🚨 Threshold",   f"{shap_threshold:.2f}")

        # Verdict
        if is_suspicious:
            st.markdown(
                '<div class="verdict-suspicious">⚠️ SHAP-SCAN ALERT: Suspected Backdoor Trigger Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="verdict-clean">✅ SHAP-SCAN CLEAR: No Backdoor Signature Detected</div>',
                unsafe_allow_html=True
            )

        # SHAP bar chart
        st.subheader(f"📉 SHAP Feature Attribution (predicted: {pred_class_name})")
        top_idx = np.argsort(np.abs(sv))[::-1][:n_top_features]
        top_names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]
        top_vals  = sv[top_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
        ax.barh(range(len(top_idx)), top_vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Top {n_top_features} SHAP Attributions — Predicted: {pred_class_name}")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<p class="shap-note">Red = pushes toward predicted class | Blue = pushes away. '
            'Trigger features show abnormally large red bars in backdoored samples.</p>',
            unsafe_allow_html=True
        )

        # Confidence bar
        st.subheader("📊 Class Probability Distribution")
        top5_idx = np.argsort(probs)[::-1][:5]
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh([classes[i] for i in top5_idx], probs[top5_idx] * 100, color="#3498db", alpha=0.8)
        ax2.set_xlabel("Probability (%)")
        ax2.set_title("Top-5 Predicted Classes")
        ax2.invert_yaxis()
        st.pyplot(fig2, use_container_width=True)

        if show_shap_detail:
            import pandas as pd
            df = pd.DataFrame({
                "Feature": [feature_names[i] if i < len(feature_names) else f"F{i}" for i in range(len(sv))],
                "SHAP Value": sv,
                "Abs SHAP": np.abs(sv),
            }).sort_values("Abs SHAP", ascending=False).head(30)
            st.dataframe(df, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**Trojan-Resilient Cyber-Sentinet** — MSc Thesis Project  \n"
        "Backdoor attack/defense + SHAP-Scan for Industry 5.0 CPS IDS  \n"
        "*Based on Nandanwar & Katarya (2025), Computers and Electrical Engineering 123: 110161*"
    )


if __name__ == "__main__":
    main()
