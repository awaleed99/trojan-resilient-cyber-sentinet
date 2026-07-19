"""
data/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Preprocessing pipeline for the Edge-IIoT-2022 dataset (Edge-IIoTset).

Usage
-----
    from data.preprocessing import DataPreprocessor
    prep = DataPreprocessor(config)
    prep.run()            # produces train/val/test splits in data/processed/

Or from command line:
    python -m data.preprocessing --config configs/config.yaml

Key outputs (saved as .npy / .csv in data/processed/):
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    label_encoder.pkl
    scaler.pkl
    feature_names.json
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import track

console = Console(highlight=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─── Label map (actual class names in ML-EdgeIIoT-dataset.csv) ───────────────
# These were verified by inspecting Attack_type.value_counts() on the real CSV.
# Note: paper calls the 15th class "OS_Fingerprinting"; the CSV uses "Fingerprinting".
EDGE_IIOT_CLASSES = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "Ransomware",
    "DDoS_HTTP",
    "SQL_injection",
    "Uploading",
    "DDoS_TCP",
    "Backdoor",
    "Vulnerability_scanner",
    "Port_Scanning",
    "XSS",
    "Password",
    "MITM",
    "Fingerprinting",   # ← paper writes OS_Fingerprinting (Gap 1.2.3)
]


class DataPreprocessor:
    """End-to-end preprocessing for Edge-IIoT-2022."""

    def __init__(self, config):
        self.cfg = config
        self.raw_dir = Path(self.cfg.paths.raw_data)
        self.out_dir = Path(self.cfg.paths.processed_data)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Use class names from config if provided, otherwise fall back to defaults
        self.class_names = (
            list(self.cfg.dataset.class_names)
            if hasattr(self.cfg.dataset, "class_names")
            else EDGE_IIOT_CLASSES
        )

        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()
        self.feature_names = None

    # ── 1. Load ───────────────────────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        csv_path = self.raw_dir / self.cfg.dataset.filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Dataset not found at: {csv_path}\n"
                f"Expected the file at:\n  {csv_path}\n"
                "Make sure the 'Edge-IIoTset dataset' folder is inside data/raw/"
            )
        console.print(f"[cyan]Loading dataset from {csv_path}…[/cyan]")
        # low_memory=False is required — many columns have mixed types (string + numeric)
        df = pd.read_csv(csv_path, low_memory=False)
        console.print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
        return df

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        console.print("[cyan]Cleaning…[/cyan]")

        # 1. Drop explicitly listed string/identifier columns
        drop_cols = [c for c in self.cfg.dataset.drop_cols if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        console.print(f"  Dropped {len(drop_cols)} identifier/string columns")

        # 2. Separate label column before type conversion
        label_col = self.cfg.dataset.label_col
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

        labels = df[label_col].copy()
        df_features = df.drop(columns=[label_col])

        # 3. Convert all remaining columns to numeric; coerce non-numeric to NaN
        df_numeric = df_features.apply(pd.to_numeric, errors="coerce")

        # 4. Drop columns that are entirely NaN (no usable data at all)
        all_nan_cols = df_numeric.columns[df_numeric.isna().all()].tolist()
        if all_nan_cols:
            console.print(f"  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols[:5]}…")
        df_numeric = df_numeric.drop(columns=all_nan_cols)

        # 5. Fill remaining NaN with 0 (standard for protocol-specific fields
        #    that are 0 when the protocol is absent — e.g. mqtt.len is NaN for TCP rows)
        n_nan = df_numeric.isna().sum().sum()
        console.print(f"  Filling {n_nan:,} NaN values with 0 (protocol-absent fields)")
        df_numeric = df_numeric.fillna(0.0)

        # 6. Reattach label, keep only known classes
        df_clean = df_numeric.copy()
        df_clean[label_col] = labels.values
        before = len(df_clean)
        df_clean = df_clean[df_clean[label_col].isin(self.class_names)].copy()
        dropped = before - len(df_clean)
        if dropped > 0:
            console.print(f"  Dropped {dropped} rows with unknown/unlisted class labels")

        console.print(
            f"  After cleaning: {len(df_clean):,} rows × {df_clean.shape[1]} columns "
            f"| label_col='{label_col}'"
        )
        return df_clean, label_col

    # ── 3. Split features / labels ────────────────────────────────────────────
    def split_xy(self, df: pd.DataFrame, label_col: str):
        y_raw = df[label_col].values
        self.feature_names = [c for c in df.columns if c != label_col]

        # Cast via float64 first to avoid silent overflow, then clip to float32 range,
        # then downcast to float32.  This safely handles large uint32 fields like
        # tcp.ack (max ~2.1e9) which are within float32 range (~3.4e38) but need
        # careful handling, and kills any residual Inf/NaN from mis-parsed payloads.
        X64 = df.drop(columns=[label_col]).values.astype(np.float64)

        # Replace any remaining Inf / -Inf with 0 (should not occur after drop_cols fix)
        X64 = np.where(np.isinf(X64), 0.0, X64)
        X64 = np.where(np.isnan(X64),  0.0, X64)

        # Clip to float32 safe range
        F32_MAX = np.finfo(np.float32).max
        X64 = np.clip(X64, -F32_MAX, F32_MAX)

        X_raw = X64.astype(np.float32)
        console.print(
            f"  Feature matrix: {X_raw.shape}  dtype={X_raw.dtype}  "
            f"max={X_raw.max():.2e}  min={X_raw.min():.2e}"
        )
        return X_raw, y_raw

    # ── 4. Encode labels ──────────────────────────────────────────────────────
    def encode_labels(self, y_raw: np.ndarray) -> np.ndarray:
        # Fit on the exact set of classes we know exist in the CSV
        self.label_enc.fit(self.class_names)
        y = self.label_enc.transform(y_raw).astype(np.int64)
        console.print(f"  Classes ({len(self.label_enc.classes_)}): {list(self.label_enc.classes_)}")
        return y

    # ── 5. Scale features ─────────────────────────────────────────────────────
    def scale(self, X_train, X_val, X_test):
        self.scaler.fit(X_train)
        return (
            self.scaler.transform(X_train).astype(np.float32),
            self.scaler.transform(X_val).astype(np.float32),
            self.scaler.transform(X_test).astype(np.float32),
        )

    # ── 6. Pad + Reshape → 2D matrix for CNN ─────────────────────────────────
    def reshape_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Pad the flat feature vector to H*W then reshape to (N, 1, H, W).
        The leading 1 is the channel dimension (grayscale-style).
        """
        H = self.cfg.dataset.reshape_h
        W = self.cfg.dataset.reshape_w
        target_len = H * W

        n_feats = X.shape[1]
        if n_feats > target_len:
            # If more features than grid cells, crop (keep first H*W)
            X = X[:, :target_len]
        elif n_feats < target_len:
            # Pad with zeros
            pad = np.zeros((X.shape[0], target_len - n_feats), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)

        X_2d = X.reshape(-1, 1, H, W)  # (N, C=1, H, W)
        return X_2d

    # ── 7. Save ───────────────────────────────────────────────────────────────
    def save(self, splits: dict):
        for name, arr in splits.items():
            path = self.out_dir / f"{name}.npy"
            np.save(path, arr)
        console.print(f"  Saved splits to {self.out_dir}")

        # Scaler + encoder
        with open(self.out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.out_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_enc, f)

        # Feature names — human-readable mapping (resolves Gap 1.2.11)
        with open(self.out_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)

        console.print(
            f"  Feature names saved -> {self.out_dir / 'feature_names.json'}\n"
            "  (Use this to replace 'Feature 39' with real column names in SHAP plots)"
        )

    # ── Master pipeline ───────────────────────────────────────────────────────
    def run(self):
        console.rule("[bold green]Data Preprocessing Pipeline[/bold green]")

        # Seed
        np.random.seed(self.cfg.seed)

        df = self.load()
        df, label_col = self.clean(df)
        X_raw, y_raw = self.split_xy(df, label_col)
        y = self.encode_labels(y_raw)

        # Shuffle
        X_raw, y = shuffle(X_raw, y, random_state=self.cfg.seed)

        # Train / test split (stratified)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_raw, y,
            test_size=self.cfg.dataset.test_size,
            stratify=y,
            random_state=self.cfg.seed,
        )

        # Train / val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.cfg.dataset.val_size,
            stratify=y_train_full,
            random_state=self.cfg.seed,
        )

        console.print(
            f"\n  Split sizes — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}"
        )

        # Scale
        X_train, X_val, X_test = self.scale(X_train, X_val, X_test)

        # Reshape to 2D
        X_train_2d = self.reshape_2d(X_train)
        X_val_2d   = self.reshape_2d(X_val)
        X_test_2d  = self.reshape_2d(X_test)

        # Also save flat versions (needed for SHAP DeepExplainer, STRIP, etc.)
        self.save({
            "X_train": X_train_2d,
            "X_val":   X_val_2d,
            "X_test":  X_test_2d,
            "X_train_flat": X_train,
            "X_val_flat":   X_val,
            "X_test_flat":  X_test,
            "y_train": y_train,
            "y_val":   y_val,
            "y_test":  y_test,
        })

        # Class distribution report
        self._report_class_distribution(y_train, y_val, y_test)

        console.rule("[bold green]Preprocessing Complete[/bold green]")
        return X_train_2d, X_val_2d, X_test_2d, y_train, y_val, y_test

    def _report_class_distribution(self, y_train, y_val, y_test):
        console.print("\n[bold]Class distribution (train split):[/bold]")
        labels = self.label_enc.classes_
        for i, lbl in enumerate(labels):
            n = (y_train == i).sum()
            pct = 100 * n / len(y_train)
            bar = "█" * int(pct / 2)
            console.print(f"  {lbl:<25} {n:6,}  ({pct:5.1f}%)  {bar}")


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    preprocessor = DataPreprocessor(cfg)
    preprocessor.run()
