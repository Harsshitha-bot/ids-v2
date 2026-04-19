# IDS v2 — eBPF + XDP + ONNX Intrusion Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-green.svg)]()

A production-grade **Intrusion Detection System** combining Linux kernel-space packet processing (XDP/eBPF) with an ML ensemble (Random Forest + XGBoost + SVM + Isolation Forest) exported to ONNX for ~0.2ms inference latency.

---

## ✨ Key Features

| # | Feature | Detail |
|---|---------|--------|
| 1 | **ML at Kernel Boundary** | XDP BPF extracts per-packet stats → ONNX Runtime inference (~0.2ms) |
| 2 | **64-Feature Engineering** | 35 XDP flow features + 29 eBPF syscall features per flow window |
| 3 | **Flow-Level Detection** | 5-second sliding window flow aggregation via `FlowAccumulator` |
| 4 | **4-Model Ensemble** | RF (200 trees) · XGBoost (300 est.) · SVM RBF · Isolation Forest |
| 5 | **Real-Time XDP Defense** | 3-strike auto-block → `XDP_DROP` at line rate, 60s TTL |
| 6 | **Performance Optimized** | Zero-copy kernel BPF · ONNX `ORT_ENABLE_ALL` · async daemon threads |
| 7 | **Full Evaluation Suite** | Accuracy, F1, ROC-AUC, confusion matrix, cross-validation, feature importance |

---

## 📊 Dataset & Model Performance

### Synthetic Training Dataset

The models are trained on a **synthetic network flow dataset** generated to mirror real-world attack distributions:

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 12,000 | 6 |
| Test | 3,000 | 6 |
| **Total** | **15,000** | **6** |

**Attack Classes:**

| Label | Description |
|-------|-------------|
| `Normal` | Legitimate low-rate background traffic |
| `Port Scan` | Nmap-style SYN scan — high port entropy, tiny packets |
| `DoS / DDoS` | SYN flood — 3,000–8,000 pkts/s, many source IPs |
| `Brute Force` | SSH/FTP login attempts — execve spikes, single port |
| `Heartbleed` | CVE-2014-0160 — mmap spikes, tiny SSL payloads |
| `Botnet` | C2 beacon — dangerous syscalls, multi-IP patterns |

**64 Features used (per 5-second flow window):**

- **35 XDP/flow features** — `f_pkt_count`, `f_byte_rate`, `f_syn_rate`, `f_dst_port_entropy`, `f_src_ip_entropy`, `f_iat_mean/std/min/max`, `f_syn_ack_ratio`, `f_small_pkt_ratio`, and 24 more
- **29 eBPF/syscall features** — `e_execve_count`, `e_connect_count`, `e_mmap_count`, `e_socket_count`, `e_dangerous_count`, `e_syscall_entropy`, and 23 more

### Model Results (held-out test set)

| Model | Accuracy | F1 (weighted) | AUC-ROC | Notes |
|-------|----------|---------------|---------|-------|
| **Random Forest** | **1.000** | **1.000** | **1.000** | Best model (selected) |
| **XGBoost** | 1.000 | 1.000 | 1.000 | |
| **SVM RBF** | 1.000 | 1.000 | 1.000 | |
| **Isolation Forest** | 0.978 | 0.980 | — | Unsupervised, day-0 |

> **Note:** Perfect scores on synthetic data reflect clean, well-separated class distributions by design. Real-world performance will vary.

**Confusion Matrix (RF — best model):**

```
TP: 1650  |  FP: 0
FN: 0     |  TN: 1350
Sensitivity (TPR): 1.0
Specificity (TNR): 1.0
FPR: 0.0  |  FNR: 0.0
```

**Top Feature Importances (RF):**

| Feature | Importance |
|---------|-----------|
| `f_max_pkt_size` | 0.0482 |
| `f_syn_ack_ratio` | 0.0378 |
| `e_mmap_count` | 0.0363 |
| `f_dst_port_entropy` | 0.0350 |
| `e_socket_count` | 0.0330 |

**Cross-validation (5-fold):**
- RF: Mean Accuracy = 1.000 ± 0.000
- Training time: ~19s on 15,000 samples

**Live Detection Results:**
```
[Normal      ] → NORMAL  | conf=37%  | mode=Ensemble(4)
[Port Scan   ] → ATTACK  | conf=100% | mode=Ensemble(4)
[DoS/DDoS    ] → ATTACK  | conf=100% | mode=Ensemble(4)
[Brute Force ] → ATTACK  | conf=100% | mode=Ensemble(4)
[Heartbleed  ] → ATTACK  | conf=100% | mode=Ensemble(4)
[Botnet      ] → ATTACK  | conf=100% | mode=Ensemble(4)
```

---

## 📁 Project Structure

```
ids-v2/
├── run_demo.py                  ← Main entry point
├── requirements.txt
├── LICENSE                      ← MIT
├── .gitignore
│
├── kernel/
│   └── xdp_engine.py            ← XDP BPF engine + FlowAccumulator
├── monitors/
│   └── ebpf_monitor.py          ← eBPF syscall tracepoints (25 types)
├── ml/
│   ├── train_models.py          ← RF + XGBoost + SVM + IsoForest trainer
│   └── detector.py              ← ONNX ensemble detection engine
├── defense/
│   └── xdp_firewall.py          ← Auto-block IPs via XDP_DROP
├── utils/
│   ├── config.py                ← Central configuration
│   └── shared_data.py           ← Thread-safe shared state + flow table
├── visualization/
│   └── dashboard.py             ← 6-tab real-time Dash dashboard
│
├── models/                      ← Pre-trained model files
│   ├── rf_model.pkl             ← Random Forest (best)
│   ├── xgb_model.pkl            ← XGBoost
│   ├── svm_model.pkl            ← SVM RBF
│   ├── iso_forest.pkl           ← Isolation Forest
│   ├── ids_model.onnx           ← ONNX export of RF
│   ├── scaler.pkl               ← StandardScaler
│   ├── label_encoder.pkl        ← LabelEncoder
│   └── metrics.json             ← All evaluation metrics
│
├── attackers/                   ← Attack simulation modules
├── scripts/                     ← Utility scripts
├── flow/                        ← Flow tracking modules
├── logs/                        ← Runtime logs
└── data/                        ← Dataset storage (runtime generated)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  Network Interface (eth0 / lo)                      │
├─────────────────────────────────────────────────────┤
│  XDP BPF Program (kernel space)                     │
│  • Parses packets at line rate (zero-copy)          │
│  • Updates pkt_stats, flow_map, port_syn_map        │
│  • Checks blocked_ips → XDP_DROP                    │
├─────────────────────────────────────────────────────┤
│  XDPEngine  (kernel/xdp_engine.py)                  │
│  • Reads BPF maps every 1s                          │
│  • FlowAccumulator: 5-second windows                │
│  • Emits 35 flow feature vectors                    │
├─────────────────────────────────────────────────────┤
│  EBPFMonitor  (monitors/ebpf_monitor.py)            │
│  • Tracepoints on 25 syscall types                  │
│  • Computes 29 syscall feature values per window    │
├─────────────────────────────────────────────────────┤
│  DetectionEngine  (ml/detector.py)                  │
│  • ONNX RF inference (~0.2ms)                       │
│  • XGBoost + SVM ensemble vote                      │
│  • Isolation Forest (unsupervised)                  │
│  • Spike Detector (statistical baseline)            │
│  • Weighted majority → attack_type + confidence     │
├─────────────────────────────────────────────────────┤
│  XDPFirewall  (defense/xdp_firewall.py)             │
│  • 3 strikes → block_ip() → BPF map update         │
│  • Auto-unblock after 60 seconds                    │
├─────────────────────────────────────────────────────┤
│  Dashboard  (visualization/dashboard.py)            │
│  • 6-tab Dash app, 1s refresh, http://localhost:8050│
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Harsshitha-bot/ids-v2.git
cd ids-v2
pip install -r requirements.txt
```

For XGBoost + ONNX support (recommended):
```bash
pip install xgboost onnxruntime skl2onnx
```

For real XDP/eBPF on Linux (optional, needs root):
```bash
sudo apt install python3-bcc linux-headers-$(uname -r)
```

### 2. Run (Simulation Mode — no root needed)

```bash
python3 run_demo.py
```

Opens dashboard at **http://localhost:8050**

### 3. Run (Real XDP/eBPF — root required)

```bash
sudo INTERFACE=eth0 PYTHONPATH=$PWD python3 run_demo.py
```

### 4. Retrain Models

```bash
# Delete cached models to force retrain
rm -rf models/
python3 ml/train_models.py
```

---

## 📈 Dashboard Tabs

| Tab | What you see |
|-----|-------------|
| 📡 Live Monitor | Packet rate, SYN rate, syscall rate, radar chart, live alert feed |
| 🌊 Flow Analysis | Per-flow pkt/s, byte/s, SYN vs ACK, port entropy, IAT stats |
| 🧠 ML Metrics | ROC curves, confusion matrix, F1 comparison, full metrics table |
| ⚡ Attack Analysis | Detection timeline, alert distribution, confidence histogram |
| 🔥 Defense | XDP drop rate, blocked IP list, block event timeline |
| 🖥️ System | Detection latency, CPU/RAM, detection mode breakdown |

### Attack Simulator Buttons

| Button | Simulates |
|--------|-----------|
| ✅ Normal | Legitimate low-rate traffic |
| 🔍 Port Scan | Nmap-style SYN scan |
| 💥 DoS/DDoS | SYN flood (3000–8000 pkts/s) |
| 🔑 Brute Force | SSH/FTP login attempts |
| 💉 Heartbleed | CVE-2014-0160 exploit pattern |
| 🤖 Botnet C2 | Command-and-control beacon |

---

## ⚙️ Configuration

All parameters in `utils/config.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `INTERFACE` | `lo` | Network interface (`eth0` for real traffic) |
| `FLOW_WINDOW_SEC` | `5` | Flow aggregation window (seconds) |
| `FLOW_TIMEOUT_SEC` | `5` | Flow expiry timeout |
| `CONFIDENCE_THRESH` | `0.50` | Min confidence to raise alert |
| `XDP_DROP_ENABLED` | `True` | Enable XDP packet dropping |
| `XDP_BLOCK_DURATION` | `60` | IP block duration (seconds) |
| `UPDATE_INTERVAL` | `1.0` | Detection poll interval (seconds) |

---

## 📦 Pre-trained Models

The `models/` directory includes pre-trained models ready to use without retraining:

| File | Size | Description |
|------|------|-------------|
| `rf_model.pkl` | ~460 KB | Random Forest (200 trees, best model) |
| `xgb_model.pkl` | ~1.2 MB | XGBoost (300 estimators) |
| `svm_model.pkl` | ~94 KB | SVM RBF (C=10) |
| `iso_forest.pkl` | ~3.9 MB | Isolation Forest (unsupervised) |
| `ids_model.onnx` | ~227 KB | ONNX export for fast inference |
| `scaler.pkl` | ~2 KB | StandardScaler fitted on train set |
| `label_encoder.pkl` | ~1 KB | LabelEncoder for 6 classes |
| `metrics.json` | ~4 KB | Full evaluation metrics |

---

## 📋 Requirements

- Python 3.10+
- Linux (Ubuntu 22.04+ recommended)
- Root access only needed for real XDP/eBPF mode

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [BCC / eBPF](https://github.com/iovisor/bcc) for kernel tracing
- [ONNX Runtime](https://onnxruntime.ai/) for fast ML inference
- [Dash / Plotly](https://dash.plotly.com/) for the real-time dashboard
- [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.ai/) for ML models
