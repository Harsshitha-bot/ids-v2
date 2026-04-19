#!/usr/bin/env python3
# ============================================================
# utils/shared_data.py  —  Thread-safe shared state, v2
# New: flow_table for per-flow tracking, blocked_ips, async queues
# ============================================================

import threading, collections, time, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import LOG_DIR, MAX_HISTORY, ATTACK_COLORS

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name, level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(level)
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        lg.addHandler(fh); lg.addHandler(ch)
    return lg


class FlowKey:
    """Identifies a unique 5-tuple flow"""
    __slots__ = ("src_ip","dst_ip","src_port","dst_port","proto")
    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto):
        self.src_ip   = src_ip
        self.dst_ip   = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto    = proto
    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto))
    def __eq__(self, other):
        return (self.src_ip==other.src_ip and self.dst_ip==other.dst_ip and
                self.src_port==other.src_port and self.dst_port==other.dst_port and
                self.proto==other.proto)
    def __repr__(self):
        return f"{self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port}/{self.proto}"


class SharedDataStore:
    """
    Central thread-safe store for:
      • XDP per-flow metrics
      • eBPF syscall metrics
      • ML predictions
      • Flow table (active flows)
      • Blocked IPs (XDP firewall)
      • Alert history
    """
    def __init__(self, maxlen=MAX_HISTORY):
        self._lock = threading.RLock()

        # Time-series deques
        self.xdp_metrics   = collections.deque(maxlen=maxlen)
        self.ebpf_metrics  = collections.deque(maxlen=maxlen)
        self.flow_metrics  = collections.deque(maxlen=maxlen)
        self.predictions   = collections.deque(maxlen=maxlen)
        self.alerts        = collections.deque(maxlen=1000)

        # Flow table:  FlowKey → dict of accumulated stats
        self.flow_table    = {}
        self.flow_table_lock = threading.Lock()

        # Blocked IPs: ip_str → unblock_timestamp
        self.blocked_ips   = {}
        self.blocked_lock  = threading.Lock()

        # Current state
        self.current_status      = "INITIALIZING"
        self.current_attack_type = "Normal"
        self.current_confidence  = 0.0
        self.current_anomaly     = 0.0
        self.active_simulation   = None   # None | "port_scan" | "dos_ddos" | ...

        # Counters
        self.total_packets   = 0
        self.total_flows     = 0
        self.total_alerts    = 0
        self.total_dropped   = 0       # XDP-dropped packets
        self.alert_counts    = collections.defaultdict(int)

        # ML metrics (updated after training / each detection cycle)
        self.ml_metrics = {
            "detection_latency": 0.0,
            "rf_accuracy": 0.0, "xgb_accuracy": 0.0, "svm_accuracy": 0.0,
            "rf_f1": 0.0,       "xgb_f1": 0.0,       "svm_f1": 0.0,
            "ensemble_f1": 0.0, "best_model": "RF",
        }

    # ── XDP / eBPF raw metrics ───────────────────────────────
    def add_xdp_metric(self, m: dict):
        with self._lock:
            self.xdp_metrics.append({**m, "_ts": time.time()})
            self.total_packets += m.get("f_pkt_count", 0)

    def add_ebpf_metric(self, m: dict):
        with self._lock:
            self.ebpf_metrics.append({**m, "_ts": time.time()})

    def add_flow_metric(self, m: dict):
        with self._lock:
            self.flow_metrics.append({**m, "_ts": time.time()})
            self.total_flows += 1

    # ── Predictions ─────────────────────────────────────────
    def add_prediction(self, pred: dict):
        with self._lock:
            self.predictions.append(pred)
            self.current_status      = pred.get("status", "NORMAL")
            self.current_attack_type = pred.get("attack_type", "Normal")
            self.current_confidence  = pred.get("confidence", 0.0)
            self.current_anomaly     = pred.get("anomaly_score", 0.0)
            if pred.get("status") == "ATTACK":
                self.total_alerts += 1
                atype = pred.get("attack_type", "Unknown Attack")
                self.alert_counts[atype] += 1
                self.alerts.append({
                    "time":       time.time(),
                    "type":       atype,
                    "confidence": pred.get("confidence", 0.0),
                    "src_ip":     pred.get("src_ip", "?"),
                    "details":    pred.get("details", ""),
                    "action":     pred.get("action", "alert"),
                })

    # ── Flow table ───────────────────────────────────────────
    def update_flow(self, key: FlowKey, stats: dict):
        with self.flow_table_lock:
            if key in self.flow_table:
                existing = self.flow_table[key]
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        existing[k] = existing.get(k, 0) + v
                    else:
                        existing[k] = v
                existing["_last"] = time.time()
            else:
                self.flow_table[key] = {**stats, "_start": time.time(), "_last": time.time()}

    def get_expired_flows(self, timeout: float):
        """Return and remove flows older than timeout seconds"""
        now = time.time()
        expired = []
        with self.flow_table_lock:
            to_del = [k for k, v in self.flow_table.items()
                      if now - v.get("_last", now) > timeout]
            for k in to_del:
                expired.append((k, self.flow_table.pop(k)))
        return expired

    def get_active_flow_count(self):
        with self.flow_table_lock:
            return len(self.flow_table)

    # ── Blocked IPs (XDP firewall) ───────────────────────────
    def block_ip(self, ip: str, duration: float):
        with self.blocked_lock:
            self.blocked_ips[ip] = time.time() + duration

    def is_blocked(self, ip: str) -> bool:
        with self.blocked_lock:
            exp = self.blocked_ips.get(ip, 0)
            if exp and time.time() < exp:
                return True
            if ip in self.blocked_ips:
                del self.blocked_ips[ip]
            return False

    def get_blocked_ips(self):
        now = time.time()
        with self.blocked_lock:
            return {ip: exp for ip, exp in self.blocked_ips.items() if exp > now}

    def record_drop(self, count: int = 1):
        with self._lock:
            self.total_dropped += count

    # ── Getters ──────────────────────────────────────────────
    def get_recent_xdp(self, n=80):
        with self._lock: return list(self.xdp_metrics)[-n:]

    def get_recent_ebpf(self, n=80):
        with self._lock: return list(self.ebpf_metrics)[-n:]

    def get_recent_flows(self, n=80):
        with self._lock: return list(self.flow_metrics)[-n:]

    def get_recent_predictions(self, n=120):
        with self._lock: return list(self.predictions)[-n:]

    def get_alerts(self, n=60):
        with self._lock: return list(self.alerts)[-n:]

    def update_ml_metrics(self, m: dict):
        with self._lock: self.ml_metrics.update(m)

    def get_summary(self):
        with self._lock:
            return {
                "status":          self.current_status,
                "attack_type":     self.current_attack_type,
                "confidence":      self.current_confidence,
                "anomaly_score":   self.current_anomaly,
                "total_packets":   self.total_packets,
                "total_flows":     self.total_flows,
                "total_alerts":    self.total_alerts,
                "total_dropped":   self.total_dropped,
                "active_flows":    self.get_active_flow_count(),
                "blocked_ips":     len(self.get_blocked_ips()),
                "alert_counts":    dict(self.alert_counts),
                "active_sim":      self.active_simulation,
                "ml_metrics":      dict(self.ml_metrics),
            }


# Global singleton
data_store = SharedDataStore()
