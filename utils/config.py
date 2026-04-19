#!/usr/bin/env python3
# ============================================================
# utils/config.py  —  IDS v2 Central Configuration
# ============================================================

import os

# ── Paths ───────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
KERNEL_DIR = os.path.join(BASE_DIR, "kernel")

for _d in [DATA_DIR, LOG_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Model paths ─────────────────────────────────────────────
RF_MODEL_PATH    = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_MODEL_PATH   = os.path.join(MODEL_DIR, "xgb_model.pkl")
SVM_MODEL_PATH   = os.path.join(MODEL_DIR, "svm_model.pkl")
ISO_MODEL_PATH   = os.path.join(MODEL_DIR, "iso_forest.pkl")
ONNX_MODEL_PATH  = os.path.join(MODEL_DIR, "ids_model.onnx")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH     = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH     = os.path.join(MODEL_DIR, "metrics.json")

# ── Network interface ────────────────────────────────────────
INTERFACE = "lo"           # Change to "eth0" for real traffic

# ── Flow tracking ────────────────────────────────────────────
FLOW_TIMEOUT_SEC  = 5      # Group packets into 5-second flows
FLOW_WINDOW_SEC   = 5      # sliding window for flow aggregation
MAX_FLOWS         = 10000  # max concurrent tracked flows

# ── Detection thresholds ─────────────────────────────────────
SPIKE_MULTIPLIER  = 3.0    # 3x baseline triggers spike alert
ANOMALY_THRESHOLD = 0.55   # IsoForest anomaly score threshold
CONFIDENCE_THRESH = 0.50   # min confidence to raise alert
WINDOW_SIZE       = 60     # baseline window (seconds)
UPDATE_INTERVAL   = 1.0    # detector poll interval (seconds)

# ── XDP kernel defense ───────────────────────────────────────
XDP_DROP_ENABLED  = True   # Enable XDP packet dropping
XDP_BLOCK_DURATION = 60    # seconds to block an attacker IP
MAX_BLOCKED_IPS   = 1000   # max IPs in block table

# ── Multiprocessing ──────────────────────────────────────────
NUM_DETECTOR_WORKERS = 2   # parallel detection processes
QUEUE_MAXSIZE        = 500 # inter-process queue depth

# ── Dashboard ────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
MAX_HISTORY    = 300

# ── Attack labels (CICIDS2017-aligned) ───────────────────────
ATTACK_TYPES = {
    0: "Normal",
    1: "Port Scan",
    2: "DoS / DDoS",
    3: "Brute Force",
    4: "Web Attack",
    5: "Botnet",
    6: "Heartbleed",
    7: "Unknown Attack"
}

ATTACK_COLORS = {
    "Normal":      "#00ff88",
    "Port Scan":   "#ff9900",
    "DoS / DDoS":  "#ff3355",
    "Brute Force": "#ff66cc",
    "Web Attack":  "#ffcc00",
    "Botnet":      "#cc44ff",
    "Heartbleed":  "#ff4444",
    "Unknown Attack": "#888888"
}

# ── Feature definitions ──────────────────────────────────────
# Per-flow XDP features (35)
XDP_FLOW_FEATURES = [
    # Counts
    "f_pkt_count", "f_byte_count", "f_syn_count", "f_ack_count",
    "f_fin_count", "f_rst_count", "f_psh_count", "f_urg_count",
    "f_udp_count", "f_icmp_count", "f_tcp_count",
    # Addresses
    "f_unique_src_ips", "f_unique_dst_ports",
    # Rates (per-flow key metrics)
    "f_pkt_rate", "f_byte_rate", "f_syn_rate",
    # Size stats
    "f_avg_pkt_size", "f_max_pkt_size", "f_min_pkt_size",
    "f_large_pkt_ratio", "f_small_pkt_ratio",
    # Ratios
    "f_tcp_udp_ratio", "f_syn_ack_ratio", "f_fin_rst_ratio", "f_icmp_ratio",
    # Entropy
    "f_dst_port_entropy", "f_src_ip_entropy", "f_payload_entropy",
    # Inter-arrival time stats
    "f_iat_mean", "f_iat_std", "f_iat_min", "f_iat_max",
    # Flow-level
    "f_flow_duration", "f_flow_bytes_s", "f_flow_pkts_s",
]

# Per-flow eBPF syscall features (29)
EBPF_FLOW_FEATURES = [
    "e_syscall_rate", "e_execve_count", "e_connect_count", "e_accept_count",
    "e_read_count", "e_write_count", "e_open_count", "e_close_count",
    "e_fork_count", "e_kill_count", "e_ptrace_count", "e_mmap_count",
    "e_socket_count", "e_bind_count", "e_listen_count",
    "e_sendto_count", "e_recvfrom_count", "e_ioctl_count", "e_prctl_count",
    "e_clone_count", "e_unlink_count", "e_chmod_count", "e_chown_count",
    "e_setuid_count",
    "e_net_ratio", "e_file_ratio", "e_proc_ratio",
    "e_dangerous_count", "e_syscall_entropy"
]

ALL_FEATURES = XDP_FLOW_FEATURES + EBPF_FLOW_FEATURES   # 64 features
N_FEATURES   = len(ALL_FEATURES)
