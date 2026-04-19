#!/usr/bin/env python3
# ============================================================
# ml/train_models.py  —  IDS v2 Model Training
#
# Feature 4: Model Improvement
#   • Random Forest  (sklearn)
#   • XGBoost        (if available)
#   • SVM            (sklearn)
#   • Isolation Forest (unsupervised, day-0)
#   • Cross-validation (5-fold) for all models
#   • Full metrics: Accuracy, Precision, Recall, F1 (W/Mi/Ma),
#     Confusion Matrix, ROC-AUC, Sensitivity, Specificity,
#     FPR, FNR, Classification Report
#   • ONNX export (if onnxmltools available)
# ============================================================

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import get_logger

logger = get_logger("trainer")

# Optional: XGBoost
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("XGBoost not installed — skipping XGBoost model")

# Optional: ONNX export
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_OK = True
except ImportError:
    ONNX_OK = False


# ══════════════════════════════════════════════════════════════
# 1.  DATA GENERATION
# ══════════════════════════════════════════════════════════════

def _row_normal():
    np.random.seed = None
    return {
        # XDP flow features
        "f_pkt_count": np.random.randint(1, 10),
        "f_byte_count": np.random.randint(200, 8000),
        "f_syn_count": np.random.randint(0, 1),
        "f_ack_count": np.random.randint(0, 7),
        "f_fin_count": np.random.randint(0, 2),
        "f_rst_count": 0,
        "f_psh_count": np.random.randint(0, 3),
        "f_urg_count": 0,
        "f_udp_count": np.random.randint(0, 1),
        "f_icmp_count": 0,
        "f_tcp_count": np.random.randint(0, 8),
        "f_unique_src_ips": np.random.randint(1, 3),
        "f_unique_dst_ports": np.random.randint(1, 3),
        "f_pkt_rate": np.random.uniform(1, 12),
        "f_byte_rate": np.random.uniform(100, 6000),
        "f_syn_rate": np.random.uniform(0, 0.8),
        "f_avg_pkt_size": np.random.uniform(200, 1000),
        "f_max_pkt_size": np.random.uniform(800, 1500),
        "f_min_pkt_size": np.random.uniform(60, 300),
        "f_large_pkt_ratio": np.random.uniform(0.1, 0.5),
        "f_small_pkt_ratio": np.random.uniform(0.0, 0.1),
        "f_tcp_udp_ratio": np.random.uniform(3, 10),
        "f_syn_ack_ratio": np.random.uniform(0.0, 0.2),
        "f_fin_rst_ratio": np.random.uniform(0.5, 2.0),
        "f_icmp_ratio": 0.0,
        "f_dst_port_entropy": np.random.uniform(0.3, 1.8),
        "f_src_ip_entropy": np.random.uniform(0.3, 1.2),
        "f_payload_entropy": np.random.uniform(4.5, 7.0),
        "f_iat_mean": np.random.uniform(0.1, 2.0),
        "f_iat_std": np.random.uniform(0.02, 0.5),
        "f_iat_min": np.random.uniform(0.01, 0.1),
        "f_iat_max": np.random.uniform(0.5, 5.0),
        "f_flow_duration": np.random.uniform(1, 30),
        "f_flow_bytes_s": np.random.uniform(100, 6000),
        "f_flow_pkts_s": np.random.uniform(1, 12),
        # eBPF
        "e_syscall_rate": np.random.uniform(5, 25),
        "e_execve_count": np.random.randint(0, 1),
        "e_connect_count": np.random.randint(0, 2),
        "e_accept_count": 0,
        "e_read_count": np.random.randint(2, 10),
        "e_write_count": np.random.randint(1, 6),
        "e_open_count": np.random.randint(0, 4),
        "e_close_count": np.random.randint(1, 5),
        "e_fork_count": 0,
        "e_kill_count": 0,
        "e_ptrace_count": 0,
        "e_mmap_count": np.random.randint(0, 2),
        "e_socket_count": np.random.randint(0, 1),
        "e_bind_count": 0,
        "e_listen_count": 0,
        "e_sendto_count": np.random.randint(0, 2),
        "e_recvfrom_count": np.random.randint(0, 2),
        "e_ioctl_count": 0,
        "e_prctl_count": 0,
        "e_clone_count": 0,
        "e_unlink_count": 0,
        "e_chmod_count": 0,
        "e_chown_count": 0,
        "e_setuid_count": 0,
        "e_net_ratio": np.random.uniform(0.05, 0.2),
        "e_file_ratio": np.random.uniform(0.4, 0.7),
        "e_proc_ratio": np.random.uniform(0.0, 0.04),
        "e_dangerous_count": 0,
        "e_syscall_entropy": np.random.uniform(2.0, 3.5),
    }


def _row_port_scan():
    n = np.random.randint(100, 600)
    ports = np.random.randint(100, 1000)
    return {
        "f_pkt_count": n, "f_byte_count": n*70,
        "f_syn_count": int(n*0.92), "f_ack_count": 0,
        "f_fin_count": 0, "f_rst_count": int(n*0.85),
        "f_psh_count": 0, "f_urg_count": 0,
        "f_udp_count": 0, "f_icmp_count": 0,
        "f_tcp_count": n,
        "f_unique_src_ips": np.random.randint(1, 2),
        "f_unique_dst_ports": ports,
        "f_pkt_rate": np.random.uniform(60, 400),
        "f_byte_rate": np.random.uniform(4000, 28000),
        "f_syn_rate": np.random.uniform(55, 380),
        "f_avg_pkt_size": np.random.uniform(60, 80),
        "f_max_pkt_size": 80, "f_min_pkt_size": 60,
        "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.95,
        "f_tcp_udp_ratio": 1000.0,
        "f_syn_ack_ratio": np.random.uniform(80, 200),
        "f_fin_rst_ratio": 0.01, "f_icmp_ratio": 0.0,
        "f_dst_port_entropy": np.random.uniform(7.0, 10.0),
        "f_src_ip_entropy": np.random.uniform(0.0, 0.5),
        "f_payload_entropy": np.random.uniform(1.0, 2.0),
        "f_iat_mean": np.random.uniform(0.001, 0.01),
        "f_iat_std": 0.0001, "f_iat_min": 0.00005, "f_iat_max": 0.002,
        "f_flow_duration": np.random.uniform(10, 120),
        "f_flow_bytes_s": np.random.uniform(4000, 28000),
        "f_flow_pkts_s": np.random.uniform(60, 400),
        "e_syscall_rate": np.random.uniform(300, 1000),
        "e_execve_count": 0, "e_connect_count": np.random.randint(100, 500),
        "e_accept_count": 0, "e_read_count": 0, "e_write_count": 0,
        "e_open_count": 0, "e_close_count": np.random.randint(100, 500),
        "e_fork_count": 0, "e_kill_count": 0, "e_ptrace_count": 0,
        "e_mmap_count": 0, "e_socket_count": np.random.randint(100, 500),
        "e_bind_count": 0, "e_listen_count": 0, "e_sendto_count": 0,
        "e_recvfrom_count": 0, "e_ioctl_count": 0, "e_prctl_count": 0,
        "e_clone_count": 0, "e_unlink_count": 0, "e_chmod_count": 0,
        "e_chown_count": 0, "e_setuid_count": 0,
        "e_net_ratio": 0.98, "e_file_ratio": 0.01, "e_proc_ratio": 0.01,
        "e_dangerous_count": 0,
        "e_syscall_entropy": np.random.uniform(0.5, 1.5),
    }


def _row_dos():
    n = np.random.randint(2000, 8000)
    syn = int(n * 0.7)
    srcs = np.random.randint(50, 500)
    return {
        "f_pkt_count": n, "f_byte_count": n*60,
        "f_syn_count": syn, "f_ack_count": int(n*0.1),
        "f_fin_count": 0, "f_rst_count": 0,
        "f_psh_count": 0, "f_urg_count": 0,
        "f_udp_count": int(n*0.15), "f_icmp_count": int(n*0.05),
        "f_tcp_count": int(n*0.8),
        "f_unique_src_ips": srcs,
        "f_unique_dst_ports": np.random.randint(1, 3),
        "f_pkt_rate": np.random.uniform(500, 5000),
        "f_byte_rate": np.random.uniform(30000, 300000),
        "f_syn_rate": np.random.uniform(350, 3500),
        "f_avg_pkt_size": np.random.uniform(55, 70),
        "f_max_pkt_size": 90, "f_min_pkt_size": 50,
        "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.98,
        "f_tcp_udp_ratio": np.random.uniform(4, 8),
        "f_syn_ack_ratio": np.random.uniform(20, 50),
        "f_fin_rst_ratio": 0.01, "f_icmp_ratio": 0.05,
        "f_dst_port_entropy": np.random.uniform(0.3, 0.8),
        "f_src_ip_entropy": np.random.uniform(6.0, 9.0),
        "f_payload_entropy": np.random.uniform(1.5, 3.0),
        "f_iat_mean": np.random.uniform(0.00001, 0.001),
        "f_iat_std": 0.00005, "f_iat_min": 0.00001, "f_iat_max": 0.001,
        "f_flow_duration": np.random.uniform(10, 300),
        "f_flow_bytes_s": np.random.uniform(30000, 300000),
        "f_flow_pkts_s": np.random.uniform(500, 5000),
        "e_syscall_rate": np.random.uniform(800, 4000),
        "e_execve_count": 0, "e_connect_count": np.random.randint(5, 20),
        "e_accept_count": np.random.randint(5, 20), "e_read_count": np.random.randint(50, 200),
        "e_write_count": np.random.randint(50, 200),
        "e_open_count": np.random.randint(5, 20), "e_close_count": np.random.randint(50, 200),
        "e_fork_count": np.random.randint(50, 200), "e_kill_count": 0, "e_ptrace_count": 0,
        "e_mmap_count": np.random.randint(5, 20), "e_socket_count": np.random.randint(5, 20),
        "e_bind_count": 0, "e_listen_count": 0, "e_sendto_count": np.random.randint(200, 800),
        "e_recvfrom_count": np.random.randint(50, 200),
        "e_ioctl_count": 0, "e_prctl_count": 0, "e_clone_count": np.random.randint(50, 200),
        "e_unlink_count": 0, "e_chmod_count": 0, "e_chown_count": 0, "e_setuid_count": 0,
        "e_net_ratio": 0.35, "e_file_ratio": 0.45, "e_proc_ratio": 0.2,
        "e_dangerous_count": 0,
        "e_syscall_entropy": np.random.uniform(2.5, 3.5),
    }


def _row_brute():
    n = np.random.randint(20, 80)
    return {
        "f_pkt_count": n, "f_byte_count": n*120,
        "f_syn_count": int(n*0.35), "f_ack_count": int(n*0.5),
        "f_fin_count": int(n*0.1), "f_rst_count": 0,
        "f_psh_count": int(n*0.3), "f_urg_count": 0,
        "f_udp_count": 0, "f_icmp_count": 0, "f_tcp_count": n,
        "f_unique_src_ips": 1,
        "f_unique_dst_ports": 1,
        "f_pkt_rate": np.random.uniform(8, 40),
        "f_byte_rate": np.random.uniform(800, 5000),
        "f_syn_rate": np.random.uniform(3, 15),
        "f_avg_pkt_size": np.random.uniform(100, 200),
        "f_max_pkt_size": 300, "f_min_pkt_size": 80,
        "f_large_pkt_ratio": 0.1, "f_small_pkt_ratio": 0.3,
        "f_tcp_udp_ratio": 1000.0,
        "f_syn_ack_ratio": np.random.uniform(0.6, 0.9),
        "f_fin_rst_ratio": np.random.uniform(0.8, 1.5),
        "f_icmp_ratio": 0.0,
        "f_dst_port_entropy": 0.0, "f_src_ip_entropy": 0.0,
        "f_payload_entropy": np.random.uniform(3.8, 5.0),
        "f_iat_mean": np.random.uniform(0.05, 0.3),
        "f_iat_std": 0.01, "f_iat_min": 0.02, "f_iat_max": 1.0,
        "f_flow_duration": np.random.uniform(10, 60),
        "f_flow_bytes_s": np.random.uniform(800, 5000),
        "f_flow_pkts_s": np.random.uniform(8, 40),
        "e_syscall_rate": np.random.uniform(100, 400),
        "e_execve_count": np.random.randint(20, 80),
        "e_connect_count": np.random.randint(40, 150),
        "e_accept_count": 0, "e_read_count": np.random.randint(20, 80),
        "e_write_count": np.random.randint(10, 40),
        "e_open_count": np.random.randint(10, 30),
        "e_close_count": np.random.randint(30, 100),
        "e_fork_count": np.random.randint(15, 70),
        "e_kill_count": 0, "e_ptrace_count": 0,
        "e_mmap_count": np.random.randint(3, 15),
        "e_socket_count": np.random.randint(40, 150),
        "e_bind_count": 0, "e_listen_count": 0,
        "e_sendto_count": np.random.randint(15, 60),
        "e_recvfrom_count": np.random.randint(15, 60),
        "e_ioctl_count": 0, "e_prctl_count": 0, "e_clone_count": 0,
        "e_unlink_count": 0, "e_chmod_count": 0, "e_chown_count": 0, "e_setuid_count": 0,
        "e_net_ratio": 0.55, "e_file_ratio": 0.30, "e_proc_ratio": 0.15,
        "e_dangerous_count": 0,
        "e_syscall_entropy": np.random.uniform(2.0, 3.0),
    }


def _row_heartbleed():
    n = np.random.randint(4, 25)
    return {
        "f_pkt_count": n, "f_byte_count": n*80,
        "f_syn_count": int(n*0.2), "f_ack_count": int(n*0.6),
        "f_fin_count": 0, "f_rst_count": 0,
        "f_psh_count": int(n*0.4), "f_urg_count": 0,
        "f_udp_count": 0, "f_icmp_count": 0, "f_tcp_count": n,
        "f_unique_src_ips": 1, "f_unique_dst_ports": 1,
        "f_pkt_rate": np.random.uniform(0.5, 5),
        "f_byte_rate": np.random.uniform(50, 500),
        "f_syn_rate": np.random.uniform(0.1, 1.0),
        "f_avg_pkt_size": np.random.uniform(64, 100),
        "f_max_pkt_size": 110, "f_min_pkt_size": 64,
        "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.92,
        "f_tcp_udp_ratio": 1000.0,
        "f_syn_ack_ratio": 0.3, "f_fin_rst_ratio": 0.1, "f_icmp_ratio": 0.0,
        "f_dst_port_entropy": 0.0, "f_src_ip_entropy": 0.0,
        "f_payload_entropy": np.random.uniform(1.0, 2.5),
        "f_iat_mean": np.random.uniform(0.1, 1.0),
        "f_iat_std": 0.05, "f_iat_min": 0.05, "f_iat_max": 2.0,
        "f_flow_duration": np.random.uniform(5, 60),
        "f_flow_bytes_s": np.random.uniform(50, 500),
        "f_flow_pkts_s": np.random.uniform(0.5, 5),
        "e_syscall_rate": np.random.uniform(20, 80),
        "e_execve_count": 0, "e_connect_count": np.random.randint(1, 5),
        "e_accept_count": 0, "e_read_count": np.random.randint(20, 60),
        "e_write_count": np.random.randint(5, 20),
        "e_open_count": np.random.randint(1, 5),
        "e_close_count": np.random.randint(2, 10),
        "e_fork_count": 0, "e_kill_count": 0, "e_ptrace_count": 0,
        "e_mmap_count": np.random.randint(30, 100),
        "e_socket_count": np.random.randint(1, 5),
        "e_bind_count": 0, "e_listen_count": 0,
        "e_sendto_count": np.random.randint(2, 10),
        "e_recvfrom_count": np.random.randint(5, 20),
        "e_ioctl_count": 0, "e_prctl_count": 0, "e_clone_count": 0,
        "e_unlink_count": 0, "e_chmod_count": 0, "e_chown_count": 0, "e_setuid_count": 0,
        "e_net_ratio": 0.25, "e_file_ratio": 0.65, "e_proc_ratio": 0.01,
        "e_dangerous_count": 0,
        "e_syscall_entropy": np.random.uniform(1.5, 2.5),
    }


def _row_botnet():
    n = np.random.randint(80, 300)
    srcs = np.random.randint(10, 50)
    danger = np.random.randint(5, 25)
    return {
        "f_pkt_count": n, "f_byte_count": n*300,
        "f_syn_count": int(n*0.1), "f_ack_count": int(n*0.6),
        "f_fin_count": int(n*0.1), "f_rst_count": 0,
        "f_psh_count": int(n*0.3), "f_urg_count": 0,
        "f_udp_count": int(n*0.2), "f_icmp_count": 0, "f_tcp_count": int(n*0.8),
        "f_unique_src_ips": srcs,
        "f_unique_dst_ports": np.random.randint(3, 8),
        "f_pkt_rate": np.random.uniform(30, 150),
        "f_byte_rate": np.random.uniform(10000, 60000),
        "f_syn_rate": np.random.uniform(2, 15),
        "f_avg_pkt_size": np.random.uniform(250, 400),
        "f_max_pkt_size": 900, "f_min_pkt_size": 60,
        "f_large_pkt_ratio": 0.3, "f_small_pkt_ratio": 0.1,
        "f_tcp_udp_ratio": 4.0, "f_syn_ack_ratio": 0.15,
        "f_fin_rst_ratio": 1.0, "f_icmp_ratio": 0.0,
        "f_dst_port_entropy": np.random.uniform(2.0, 3.5),
        "f_src_ip_entropy": np.random.uniform(4.0, 6.0),
        "f_payload_entropy": np.random.uniform(6.5, 7.8),
        "f_iat_mean": np.random.uniform(0.005, 0.1),
        "f_iat_std": 0.01, "f_iat_min": 0.002, "f_iat_max": 0.5,
        "f_flow_duration": np.random.uniform(30, 300),
        "f_flow_bytes_s": np.random.uniform(10000, 60000),
        "f_flow_pkts_s": np.random.uniform(30, 150),
        "e_syscall_rate": np.random.uniform(100, 400),
        "e_execve_count": np.random.randint(2, 10),
        "e_connect_count": np.random.randint(20, 80),
        "e_accept_count": 0, "e_read_count": np.random.randint(20, 60),
        "e_write_count": np.random.randint(10, 40),
        "e_open_count": np.random.randint(5, 20),
        "e_close_count": np.random.randint(20, 80),
        "e_fork_count": np.random.randint(2, 10),
        "e_kill_count": np.random.randint(3, 15),
        "e_ptrace_count": np.random.randint(2, 10),
        "e_mmap_count": np.random.randint(5, 20),
        "e_socket_count": np.random.randint(20, 80),
        "e_bind_count": 0, "e_listen_count": 0,
        "e_sendto_count": np.random.randint(30, 100),
        "e_recvfrom_count": np.random.randint(20, 70),
        "e_ioctl_count": 0, "e_prctl_count": np.random.randint(2, 8),
        "e_clone_count": 0, "e_unlink_count": np.random.randint(1, 5),
        "e_chmod_count": np.random.randint(2, 8),
        "e_chown_count": np.random.randint(1, 5),
        "e_setuid_count": np.random.randint(1, 5),
        "e_net_ratio": 0.45, "e_file_ratio": 0.35, "e_proc_ratio": 0.20,
        "e_dangerous_count": danger,
        "e_syscall_entropy": np.random.uniform(3.5, 4.5),
    }


GENERATORS = {
    "Normal":      (_row_normal,     0.45),
    "Port Scan":   (_row_port_scan,  0.15),
    "DoS / DDoS":  (_row_dos,        0.15),
    "Brute Force": (_row_brute,      0.10),
    "Botnet":      (_row_botnet,     0.08),
    "Heartbleed":  (_row_heartbleed, 0.07),
}


def generate_data(n_samples: int = 15000) -> pd.DataFrame:
    np.random.seed(42)
    rows, labels = [], []
    print("\n🔄 Generating synthetic CICIDS2017-style flow data...")
    for name, (gen_fn, ratio) in GENERATORS.items():
        n = int(n_samples * ratio)
        for _ in range(n):
            r = gen_fn()
            # Add 5% Gaussian noise to each numeric feature
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = max(0.0, v + np.random.normal(0, abs(v)*0.05 + 0.001))
            rows.append(r)
            labels.append(name)
    df = pd.DataFrame(rows)
    df["label"] = labels
    df["binary_label"] = (df["label"] != "Normal").astype(int)
    print(f"   Total: {len(df):,} samples")
    print(f"   Class distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════
# 2.  METRICS
# ══════════════════════════════════════════════════════════════

def full_metrics(y_true, y_pred, y_prob=None, label="", is_binary=False):
    m = {}
    avg_kws = dict(average="weighted", zero_division=0)
    m["accuracy"]            = accuracy_score(y_true, y_pred)
    m["precision_weighted"]  = precision_score(y_true, y_pred, **avg_kws)
    m["precision_macro"]     = precision_score(y_true, y_pred, average="macro",  zero_division=0)
    m["precision_micro"]     = precision_score(y_true, y_pred, average="micro",  zero_division=0)
    m["recall_weighted"]     = recall_score(y_true, y_pred, **avg_kws)
    m["recall_macro"]        = recall_score(y_true, y_pred, average="macro",  zero_division=0)
    m["recall_micro"]        = recall_score(y_true, y_pred, average="micro",  zero_division=0)
    m["f1_weighted"]         = f1_score(y_true, y_pred, **avg_kws)
    m["f1_macro"]            = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    m["f1_micro"]            = f1_score(y_true, y_pred, average="micro",  zero_division=0)

    y_bin_true = (np.array(y_true) != "Normal").astype(int) if not is_binary else np.array(y_true)
    y_bin_pred = (np.array(y_pred) != "Normal").astype(int) if not is_binary else np.array(y_pred)
    cm = confusion_matrix(y_bin_true, y_bin_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m["sensitivity"]         = tp / max(tp + fn, 1)
        m["specificity"]         = tn / max(tn + fp, 1)
        m["false_positive_rate"] = fp / max(fp + tn, 1)
        m["false_negative_rate"] = fn / max(fn + tp, 1)
        m["tp"] = int(tp); m["tn"] = int(tn); m["fp"] = int(fp); m["fn"] = int(fn)

    if y_prob is not None:
        try:
            if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
                normal_idx = list(np.unique(y_true)).index("Normal") if "Normal" in np.unique(y_true) else 0
                attack_prob = 1 - y_prob[:, normal_idx]
            else:
                attack_prob = y_prob
            m["auc_roc"] = roc_auc_score(y_bin_true, attack_prob)
        except Exception:
            m["auc_roc"] = 0.0

    m["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n{'='*62}")
    print(f"  METRICS — {label}")
    print(f"{'='*62}")
    for k, v in m.items():
        if k not in ("classification_report", "tp","tn","fp","fn"):
            print(f"  {k:<28} {v:.4f}")
    print(f"\n{m['classification_report']}")
    return m


def cross_val_report(model, X, y, cv=5, label=""):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    sc = cross_validate(model, X, y,
                        cv=skf,
                        scoring=["accuracy","f1_weighted","precision_weighted","recall_weighted"],
                        return_train_score=False, n_jobs=-1)
    print(f"\n  Cross-validation ({cv}-fold) — {label}")
    print(f"  {'Accuracy':<20} {sc['test_accuracy'].mean():.4f} ± {sc['test_accuracy'].std():.4f}")
    print(f"  {'F1 (weighted)':<20} {sc['test_f1_weighted'].mean():.4f} ± {sc['test_f1_weighted'].std():.4f}")
    return {
        "cv_accuracy_mean":  float(sc["test_accuracy"].mean()),
        "cv_accuracy_std":   float(sc["test_accuracy"].std()),
        "cv_f1_mean":        float(sc["test_f1_weighted"].mean()),
        "cv_f1_std":         float(sc["test_f1_weighted"].std()),
    }


# ══════════════════════════════════════════════════════════════
# 3.  TRAIN
# ══════════════════════════════════════════════════════════════

def train_all_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t_start = time.time()

    df = generate_data(n_samples=15000)
    X = df[ALL_FEATURES].fillna(0).values
    y = df["label"].values
    y_bin = df["binary_label"].values

    le = LabelEncoder()
    le.fit(y)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                                random_state=42, stratify=y)
    _, _, yb_tr, yb_te = train_test_split(X_sc, y_bin, test_size=0.2,
                                            random_state=42, stratify=y)

    all_metrics = {}
    best_f1 = 0.0
    best_model_name = "RF"
    best_model = None

    # ── 1. Random Forest ──────────────────────────────────
    print("\n🌲 Training Random Forest (200 trees, max_depth=25)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_split=4,
                                 n_jobs=-1, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    y_prob_rf = rf.predict_proba(X_te)
    rf_m = full_metrics(y_te, y_pred_rf, y_prob_rf, "Random Forest")
    rf_m.update(cross_val_report(rf, X_sc, y, label="Random Forest"))
    # Feature importance
    rf_m["feature_importance"] = {ALL_FEATURES[i]: float(rf.feature_importances_[i])
                                   for i in np.argsort(rf.feature_importances_)[::-1][:20]}
    all_metrics["random_forest"] = {k: (float(v) if isinstance(v,(float,np.floating)) else str(v))
                                     for k, v in rf_m.items() if k != "classification_report"}
    joblib.dump(rf, RF_MODEL_PATH)
    if rf_m["f1_weighted"] > best_f1:
        best_f1 = rf_m["f1_weighted"]; best_model_name = "RF"; best_model = rf

    # ── 2. XGBoost ────────────────────────────────────────
    if XGB_OK:
        print("\n🚀 Training XGBoost (300 estimators)...")
        from sklearn.preprocessing import LabelEncoder as LE
        le_num = LE(); y_tr_num = le_num.fit_transform(y_tr); y_te_num = le_num.transform(y_te)
        xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                        subsample=0.8, colsample_bytree=0.8,
                                        use_label_encoder=False, eval_metric="mlogloss",
                                        n_jobs=-1, random_state=42)
        xgb_model.fit(X_tr, y_tr_num, verbose=False)
        y_pred_xgb_num = xgb_model.predict(X_te)
        y_pred_xgb = le_num.inverse_transform(y_pred_xgb_num)
        y_prob_xgb = xgb_model.predict_proba(X_te)
        xgb_m = full_metrics(y_te, y_pred_xgb, y_prob_xgb, "XGBoost")
        xgb_m.update(cross_val_report(
            xgb.XGBClassifier(n_estimators=100, max_depth=8, use_label_encoder=False,
                               eval_metric="mlogloss", n_jobs=-1, random_state=42),
            X_sc, le_num.fit_transform(y), label="XGBoost"))
        all_metrics["xgboost"] = {k: (float(v) if isinstance(v,(float,np.floating)) else str(v))
                                   for k, v in xgb_m.items() if k != "classification_report"}
        joblib.dump((xgb_model, le_num), XGB_MODEL_PATH)
        if xgb_m["f1_weighted"] > best_f1:
            best_f1 = xgb_m["f1_weighted"]; best_model_name = "XGB"; best_model = xgb_model
    else:
        all_metrics["xgboost"] = {"note": "not installed"}

    # ── 3. SVM (RBF) ──────────────────────────────────────
    print("\n🔵 Training SVM (RBF kernel, C=10)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight="balanced", random_state=42)
    # SVM is slow — use 4000-sample subset for training
    svm_idx = np.random.choice(len(X_tr), min(4000, len(X_tr)), replace=False)
    svm.fit(X_tr[svm_idx], y_tr[svm_idx])
    y_pred_svm = svm.predict(X_te)
    y_prob_svm = svm.predict_proba(X_te)
    svm_m = full_metrics(y_te, y_pred_svm, y_prob_svm, "SVM (RBF)")
    all_metrics["svm"] = {k: (float(v) if isinstance(v,(float,np.floating)) else str(v))
                          for k, v in svm_m.items() if k != "classification_report"}
    joblib.dump(svm, SVM_MODEL_PATH)
    if svm_m["f1_weighted"] > best_f1:
        best_f1 = svm_m["f1_weighted"]; best_model_name = "SVM"; best_model = svm

    # ── 4. Isolation Forest ────────────────────────────────
    print("\n🔍 Training Isolation Forest (unsupervised, normal data only)...")
    X_normal = X_sc[y == "Normal"]
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    iso.fit(X_normal)
    iso_pred = (iso.predict(X_te) == -1).astype(int)
    iso_m = {
        "accuracy":  float(accuracy_score(yb_te, iso_pred)),
        "precision": float(precision_score(yb_te, iso_pred, zero_division=0)),
        "recall":    float(recall_score(yb_te, iso_pred, zero_division=0)),
        "f1":        float(f1_score(yb_te, iso_pred, zero_division=0)),
        "sensitivity": float(recall_score(yb_te, iso_pred, zero_division=0)),
    }
    print(f"  IsoForest  acc={iso_m['accuracy']:.4f}  f1={iso_m['f1']:.4f}")
    all_metrics["isolation_forest"] = iso_m
    joblib.dump(iso, ISO_MODEL_PATH)

    # ── 5. ONNX export (best model → RF always works) ─────
    if ONNX_OK:
        try:
            print("\n📦 Exporting ONNX model (lightweight kernel inference)...")
            initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
            onnx_model = convert_sklearn(rf, initial_types=initial_type,
                                          target_opset=12)
            with open(ONNX_MODEL_PATH, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"   ONNX saved → {ONNX_MODEL_PATH} ({os.path.getsize(ONNX_MODEL_PATH)//1024} KB)")
            all_metrics["onnx_exported"] = True
        except Exception as e:
            print(f"   ONNX export skipped: {e}")
            all_metrics["onnx_exported"] = False
    else:
        all_metrics["onnx_exported"] = False
        print("\n   (skl2onnx not installed — ONNX export skipped)")

    # ── Save everything ────────────────────────────────────
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    all_metrics["best_model"]       = best_model_name
    all_metrics["best_f1"]          = float(best_f1)
    all_metrics["feature_names"]    = ALL_FEATURES
    all_metrics["attack_classes"]   = list(le.classes_)
    all_metrics["n_train_samples"]  = len(X_tr)
    all_metrics["n_test_samples"]   = len(X_te)
    all_metrics["trained_at"]       = time.strftime("%Y-%m-%d %H:%M:%S")
    all_metrics["training_time_s"]  = round(time.time() - t_start, 1)

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*62}")
    print(f"  ✅ Training complete in {all_metrics['training_time_s']}s")
    print(f"  Best model: {best_model_name}  (F1={best_f1:.4f})")
    print(f"  Models saved → {MODEL_DIR}")
    print(f"{'='*62}")

    return rf, scaler, le, iso, all_metrics


if __name__ == "__main__":
    train_all_models()
