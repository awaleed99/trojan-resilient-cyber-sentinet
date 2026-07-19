"""
run.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-command launcher for the Trojan-Resilient Cyber-Sentinet system.

Usage:
    python run.py

What it does:
    1. Checks Python version & key dependencies
    2. Verifies the dataset is present
    3. Launches the Master Control Dashboard (Streamlit)
    4. Opens the browser automatically

Stop with: Ctrl+C
"""

import sys
import os
import subprocess
from pathlib import Path

# ── Colour helpers (no external deps needed) ──────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def c(text, col): return f"{col}{text}{RESET}"

PROJECT_ROOT = Path(__file__).resolve().parent


def banner():
    print()
    print(c("━" * 62, CYAN))
    print(c("  🛡️  Trojan-Resilient Cyber-Sentinet", BOLD + CYAN))
    print(c("  Backdoor + SHAP-Scan IDS · MSc Thesis Project", DIM))
    print(c("━" * 62, CYAN))
    print()


def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(c(f"  ✗  Python {major}.{minor} detected — need 3.9+", RED))
        sys.exit(1)
    print(c(f"  ✓  Python {major}.{minor}", GREEN))


def check_dep(name, import_name=None):
    import_name = import_name or name
    try:
        __import__(import_name)
        print(c(f"  ✓  {name}", GREEN))
        return True
    except ImportError:
        print(c(f"  ✗  {name} not installed", YELLOW))
        return False


def check_dataset():
    expected = PROJECT_ROOT / "data" / "raw" / "Edge-IIoTset dataset" / \
               "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
    if expected.exists():
        size_mb = expected.stat().st_size / 1_000_000
        print(c(f"  ✓  Dataset found ({size_mb:.0f} MB)", GREEN))
        return True
    else:
        print(c(f"  ✗  Dataset missing at:", YELLOW))
        print(c(f"     {expected}", DIM))
        return False


def check_processed():
    p = PROJECT_ROOT / "data" / "processed" / "X_train.npy"
    if p.exists():
        print(c("  ✓  Processed splits found", GREEN))
        return True
    else:
        print(c("  ○  Processed splits not found — run preprocessing from dashboard", DIM))
        return False


def install_missing():
    """Offer to install missing packages."""
    core_packages = [
        "streamlit", "omegaconf", "rich", "pandas", "numpy",
        "scikit-learn", "scipy", "matplotlib", "seaborn", "plotly",
        "pyyaml",
    ]
    missing = []
    for pkg in core_packages:
        imp = pkg.replace("-", "_").replace("scikit_learn", "sklearn")
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        print()
        print(c(f"  Missing packages: {', '.join(missing)}", YELLOW))
        ans = input(c("  Install now? [y/N]: ", CYAN)).strip().lower()
        if ans == "y":
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing + ["-q"],
                check=True,
            )
            print(c("  ✓  Packages installed", GREEN))


def launch_dashboard():
    print()
    print(c("  🚀  Launching Master Control Dashboard…", CYAN + BOLD))
    print(c("  URL: http://localhost:8501", GREEN))
    print(c("  Stop with: Ctrl+C", DIM))
    print()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    # Streamlit launch command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "dashboard" / "master.py"),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#00d4ff",
        "--theme.backgroundColor", "#0a0e1a",
        "--theme.secondaryBackgroundColor", "#0f1629",
        "--theme.textColor", "#e8eaf6",
    ]

    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    except KeyboardInterrupt:
        print()
        print(c("  ⏹  Dashboard stopped.", DIM))


def main():
    os.system("")   # Enable ANSI on Windows
    banner()

    print(c("  System Checks", BOLD))
    print(c("  " + "─" * 40, DIM))
    check_python()

    print()
    print(c("  Dependencies", BOLD))
    print(c("  " + "─" * 40, DIM))
    has_streamlit = check_dep("streamlit")
    has_torch     = check_dep("torch")
    has_shap      = check_dep("shap")
    check_dep("omegaconf")
    check_dep("pandas")
    check_dep("scikit-learn", "sklearn")

    if not has_streamlit:
        install_missing()

    print()
    print(c("  Data", BOLD))
    print(c("  " + "─" * 40, DIM))
    dataset_ok = check_dataset()
    check_processed()

    if not dataset_ok:
        print()
        print(c("  ⚠  Dataset CSV not found — preprocessing will fail.", YELLOW))
        print(c("  The dashboard will still launch. Place the CSV at the path above,", DIM))
        print(c("  then use the dashboard to run preprocessing.", DIM))

    if not has_torch:
        print()
        print(c("  ⚠  PyTorch not installed. Training will fail.", YELLOW))
        print(c("  Install: pip install torch --index-url https://download.pytorch.org/whl/cpu", DIM))

    launch_dashboard()


if __name__ == "__main__":
    main()
