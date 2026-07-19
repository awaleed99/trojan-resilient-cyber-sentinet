# 🛡️ Trojan-Resilient Cyber-Sentinet

> Explainability-Guided Backdoor Detection and Mitigation for Trustworthy Intrusion Detection in Industry 5.0 Cyber-Physical Systems

This repository contains the complete, reproducible source code for the MSc Thesis extending the Cyber-Sentinet Intrusion Detection System (Nandanwar & Katarya, 2025).

## ✨ Features
1. **Full PyTorch Reimplementation**: 2D-CNN + ResNet + Deep Transfer Learning applied to tabular IoT traffic data (Edge-IIoTset 2022).
2. **Backdoor Attack Suite**: Feature-Trigger and Label-Flip attacks.
3. **Advanced Defense Mechanisms**: Spectral Signatures, Activation Clustering, and Fine-Pruning.
4. **Novel Contribution (★ SHAP-Scan)**: An explainability-guided backdoor scanner that uses SHAP value concentration (entropy) to detect triggered samples with zero extra model training.
5. **Interactive UI Dashboard**: Built with Streamlit for one-click experiment orchestration, live terminal logging, and a simulated SOC analyst view.

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/your-username/trojan-resilient-cyber-sentinet.git
cd trojan-resilient-cyber-sentinet

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch the Control Center Dashboard
python run.py
```
*(Windows users can also just double-click `START.bat`)*

## 📂 Project Structure
- `models/`: PyTorch neural network architecture and training loop.
- `attacks/`: Poisoning and trigger injection logic.
- `defenses/`: SVD, K-Means, Pruning, and SHAP-Scan detectors.
- `experiments/`: The 4 main CLI scripts (Baseline, Sweep, Defend, Ablation).
- `dashboard/`: The Streamlit web apps.
- `configs/`: YAML configuration (hyperparameters).

## 📊 System Documentation
For a complete deep-dive into the AI architecture, the code rationale, and how the SHAP-Scan math works, please read the provided system documentation in the repository.
