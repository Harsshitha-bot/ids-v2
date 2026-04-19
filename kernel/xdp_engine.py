#!/usr/bin/env python3
# ============================================================
# kernel/xdp_engine.py  —  XDP Kernel Engine
#
# Implements features 1, 3, 5:
#   1. ML at Kernel boundary — XDP BPF extracts flow features
#      and can DROP packets (XDP_DROP) for blocked IPs
#   3. Flow-level detection — tracks (src_ip,dst_ip,port) flows
#   5. Real-time defense — XDP_DROP inside kernel, zero copies
#
# BPF Code: extracts per-packet counters into BPF maps
#   - pkt_stats map: global packet counters
#   - blocked_ips map: IPs to drop at line rate
#   - flow_map: per-(src_ip, dst_port) flow counters
#
# Python side:
#   - Reads BPF maps every UPDATE_INTERVAL
#   - Aggregates into flow-level features
#   - Passes to FlowTracker for 5-second window stats
#   - Feeds data_store
# ============================================================

import os, sys, time, math, random, threading, socket, struct
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import data_store, get_logger, FlowKey

logger = get_logger("xdp_engine")

# ── BPF C Code (embedded — no .bpf.c file needed) ──────────
XDP_BPF_CODE = r"""
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/in.h>
#include <bcc/proto.h>

/* ── Maps ─────────────────────────────────────────────────── */

// Global packet stats
BPF_ARRAY(pkt_stats, u64, 16);
// Index mapping:
//  0=total 1=bytes 2=tcp 3=udp 4=icmp
//  5=syn   6=ack   7=fin 8=rst 9=psh 10=urg
//  11=large_pkt 12=small_pkt

// Per flow: key=(src_ip XOR dst_port) → packet count
BPF_HASH(flow_map, u32, u64, 65536);

// Blocked IP table: src_ip → 1 (drop), 0 (allow)
BPF_HASH(blocked_ips, u32, u8, 4096);

// Per-port SYN counter
BPF_HASH(port_syn_map, u16, u64, 65536);

/* ── Helpers ──────────────────────────────────────────────── */
static __always_inline void inc_stat(u32 idx, u64 val) {
    u64 *p = pkt_stats.lookup(&idx);
    if (p) __sync_fetch_and_add(p, val);
}

/* ── XDP Main ─────────────────────────────────────────────── */
int xdp_ids_main(struct xdp_md *ctx) {
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    // Must have Ethernet header
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;

    u32 src_ip = ip->saddr;
    u32 pkt_len = data_end - data;

    // ── Check block list ──────────────────────────────────
    u8 *blocked = blocked_ips.lookup(&src_ip);
    if (blocked && *blocked == 1) {
        inc_stat(0, 1);   // still count (so we can measure drops)
        return XDP_DROP;
    }

    // ── Count packet ─────────────────────────────────────
    inc_stat(0, 1);
    inc_stat(1, pkt_len);

    if (pkt_len > 1000) inc_stat(11, 1);
    if (pkt_len < 80)   inc_stat(12, 1);

    // ── Per-protocol stats ────────────────────────────────
    u16 dst_port = 0;

    if (ip->protocol == IPPROTO_TCP) {
        inc_stat(2, 1);
        struct tcphdr *tcp = (void *)(ip + 1);
        if ((void *)(tcp + 1) > data_end) return XDP_PASS;
        dst_port = ntohs(tcp->dest);

        if (tcp->syn && !tcp->ack) { inc_stat(5, 1); }
        if (tcp->ack)  inc_stat(6, 1);
        if (tcp->fin)  inc_stat(7, 1);
        if (tcp->rst)  inc_stat(8, 1);
        if (tcp->psh)  inc_stat(9, 1);
        if (tcp->urg)  inc_stat(10, 1);

        // SYN per port counter
        if (tcp->syn && !tcp->ack) {
            u64 zero = 0;
            u64 *pc = port_syn_map.lookup_or_try_init(&dst_port, &zero);
            if (pc) __sync_fetch_and_add(pc, 1);
        }
    } else if (ip->protocol == IPPROTO_UDP) {
        inc_stat(3, 1);
        struct udphdr *udp = (void *)(ip + 1);
        if ((void *)(udp + 1) > data_end) return XDP_PASS;
        dst_port = ntohs(udp->dest);
    } else if (ip->protocol == IPPROTO_ICMP) {
        inc_stat(4, 1);
    }

    // ── Flow map (src_ip XOR dst_port for key) ───────────
    u32 flow_key = src_ip ^ ((u32)dst_port << 16);
    u64 zero64 = 0;
    u64 *fc = flow_map.lookup_or_try_init(&flow_key, &zero64);
    if (fc) __sync_fetch_and_add(fc, 1);

    return XDP_PASS;
}
"""


class FlowAccumulator:
    """
    Feature 3: Flow-level detection.
    Groups raw per-second XDP counters into FLOW_WINDOW_SEC windows.
    Tracks (src_ip, dst_port, proto) → accumulated stats.
    After FLOW_TIMEOUT_SEC, emits a complete flow feature vector.
    """
    def __init__(self, window_sec=FLOW_WINDOW_SEC, timeout_sec=FLOW_TIMEOUT_SEC):
        self._window  = window_sec
        self._timeout = timeout_sec
        self._lock    = threading.Lock()
        # slot → list of raw metric dicts collected in that window
        self._windows  = collections.deque(maxlen=int(timeout_sec / max(window_sec, 1) + 2))
        self._cur_slot = []
        self._slot_start = time.time()

    def ingest(self, raw: dict):
        now = time.time()
        with self._lock:
            self._cur_slot.append({**raw, "_t": now})
            if now - self._slot_start >= self._window:
                self._windows.append(list(self._cur_slot))
                self._cur_slot = []
                self._slot_start = now

    def compute_flow_features(self) -> dict | None:
        """Aggregate all window slots into a single flow feature vector."""
        with self._lock:
            all_samples = []
            for slot in self._windows:
                all_samples.extend(slot)
            if not all_samples:
                return None
            return self._aggregate(all_samples)

    def _aggregate(self, samples: list) -> dict:
        def _sum(k):  return sum(s.get(k, 0) for s in samples)
        def _mean(k): v = [s.get(k,0) for s in samples]; return float(np.mean(v)) if v else 0.0
        def _std(k):  v = [s.get(k,0) for s in samples]; return float(np.std(v))  if v else 0.0

        total_pkts  = max(_sum("f_pkt_count"), 1)
        total_bytes = _sum("f_byte_count")
        duration    = sum(s.get("_window_dur", 1.0) for s in samples)

        f = {
            # Counts (summed across window)
            "f_pkt_count":     total_pkts,
            "f_byte_count":    total_bytes,
            "f_syn_count":     _sum("f_syn_count"),
            "f_ack_count":     _sum("f_ack_count"),
            "f_fin_count":     _sum("f_fin_count"),
            "f_rst_count":     _sum("f_rst_count"),
            "f_psh_count":     _sum("f_psh_count"),
            "f_urg_count":     _sum("f_urg_count"),
            "f_udp_count":     _sum("f_udp_count"),
            "f_icmp_count":    _sum("f_icmp_count"),
            "f_tcp_count":     _sum("f_tcp_count"),
            "f_unique_src_ips":    _mean("f_unique_src_ips"),
            "f_unique_dst_ports":  _mean("f_unique_dst_ports"),
            # Rates (Feature 2: byte rate, pkt/s, syn rate over window)
            "f_pkt_rate":      total_pkts  / max(duration, 0.001),
            "f_byte_rate":     total_bytes / max(duration, 0.001),
            "f_syn_rate":      _sum("f_syn_count") / max(duration, 0.001),
            # Size stats
            "f_avg_pkt_size":  total_bytes / total_pkts,
            "f_max_pkt_size":  max(s.get("f_avg_pkt_size", 0) for s in samples),
            "f_min_pkt_size":  min(s.get("f_avg_pkt_size", 9999) for s in samples),
            "f_large_pkt_ratio": _sum("f_large_pkt_ratio") / len(samples),
            "f_small_pkt_ratio": _sum("f_small_pkt_ratio") / len(samples),
            # Ratios
            "f_tcp_udp_ratio": (_sum("f_tcp_count") + 1) / (_sum("f_udp_count") + 1),
            "f_syn_ack_ratio": (_sum("f_syn_count") + 1) / (_sum("f_ack_count") + 1),
            "f_fin_rst_ratio": (_sum("f_fin_count") + 1) / (_sum("f_rst_count") + 1),
            "f_icmp_ratio":    _sum("f_icmp_count") / total_pkts,
            # Entropy
            "f_dst_port_entropy": _mean("f_dst_port_entropy"),
            "f_src_ip_entropy":   _mean("f_src_ip_entropy"),
            "f_payload_entropy":  _mean("f_payload_entropy"),
            # IAT (inter-arrival time stats from window)
            "f_iat_mean":      _mean("f_iat_mean"),
            "f_iat_std":       _std("f_iat_mean"),
            "f_iat_min":       min(s.get("f_iat_mean", 1.0) for s in samples),
            "f_iat_max":       max(s.get("f_iat_mean", 0.0) for s in samples),
            # Flow level
            "f_flow_duration": duration,
            "f_flow_bytes_s":  total_bytes / max(duration, 0.001),
            "f_flow_pkts_s":   total_pkts  / max(duration, 0.001),
        }
        return f


class XDPEngine:
    """
    XDP monitor with:
    • Real BPF if BCC available (attaches to INTERFACE)
    • Simulation fallback matching realistic traffic patterns
    • FlowAccumulator for per-flow feature extraction
    • IP blocking via BPF map (real) or in-memory set (sim)
    """
    def __init__(self):
        self.running  = False
        self.bpf      = None
        self.sim_mode = False
        self._flow_acc = FlowAccumulator()
        self._prev_stats = [0] * 16
        self._sim_ports  = set()
        self._last_ts    = time.time()

    # ── BPF setup ─────────────────────────────────────────
    def _try_load_bpf(self):
        try:
            from bcc import BPF
            b = BPF(text=XDP_BPF_CODE)
            fn = b.load_func("xdp_ids_main", BPF.XDP)
            b.attach_xdp(INTERFACE, fn, 0)
            self.bpf = b
            logger.info(f"XDP BPF attached to {INTERFACE}")
            return True
        except Exception as e:
            logger.warning(f"XDP BPF failed: {e} — simulation mode")
            return False

    # ── Read BPF maps ─────────────────────────────────────
    def _read_bpf_stats(self) -> dict:
        if not self.bpf:
            return {}
        pkt_stats   = self.bpf["pkt_stats"]
        port_syn_map = self.bpf["port_syn_map"]
        flow_map    = self.bpf["flow_map"]

        raw = [pkt_stats[i].value for i in range(16)]
        diff = [raw[i] - self._prev_stats[i] for i in range(16)]
        self._prev_stats = list(raw)

        now = time.time()
        dt  = now - self._last_ts
        self._last_ts = now

        # Count unique dst ports from port_syn_map
        active_ports = set()
        try:
            for k, _ in port_syn_map.items():
                active_ports.add(k.value)
        except Exception:
            pass

        total = max(diff[0], 1)
        return {
            "f_pkt_count":       diff[0],
            "f_byte_count":      diff[1],
            "f_tcp_count":       diff[2],
            "f_udp_count":       diff[3],
            "f_icmp_count":      diff[4],
            "f_syn_count":       diff[5],
            "f_ack_count":       diff[6],
            "f_fin_count":       diff[7],
            "f_rst_count":       diff[8],
            "f_psh_count":       diff[9],
            "f_urg_count":       diff[10],
            "f_large_pkt_ratio": diff[11] / total,
            "f_small_pkt_ratio": diff[12] / total,
            "f_pkt_rate":        diff[0]  / max(dt, 0.001),
            "f_byte_rate":       diff[1]  / max(dt, 0.001),
            "f_syn_rate":        diff[5]  / max(dt, 0.001),
            "f_avg_pkt_size":    diff[1]  / total,
            "f_max_pkt_size":    1500,
            "f_min_pkt_size":    60,
            "f_unique_dst_ports": len(active_ports),
            "f_unique_src_ips":  len(flow_map),
            "f_tcp_udp_ratio":   (diff[2]+1)/(diff[3]+1),
            "f_syn_ack_ratio":   (diff[5]+1)/(diff[6]+1),
            "f_fin_rst_ratio":   (diff[7]+1)/(diff[8]+1),
            "f_icmp_ratio":      diff[4]/total,
            "f_dst_port_entropy": math.log2(max(len(active_ports),1)),
            "f_src_ip_entropy":  math.log2(max(len(flow_map),1)),
            "f_payload_entropy": 4.5,
            "f_iat_mean":        dt / total,
            "f_iat_std":         0.001,
            "f_iat_min":         0.0001,
            "f_iat_max":         dt,
            "_window_dur":       dt,
        }

    # ── Block IP via BPF map ───────────────────────────────
    def block_ip(self, ip_str: str):
        data_store.block_ip(ip_str, XDP_BLOCK_DURATION)
        if self.bpf:
            try:
                packed = struct.pack("I", socket.inet_aton(ip_str).__class__.__mro__[0])
                key = self.bpf["blocked_ips"].Key(socket.ntohl(
                    struct.unpack("I", socket.inet_aton(ip_str))[0]))
                self.bpf["blocked_ips"][key] = self.bpf["blocked_ips"].Leaf(1)
                logger.info(f"🔥 XDP_DROP applied to {ip_str}")
            except Exception as e:
                logger.debug(f"BPF block IP error: {e}")

    # ── Simulation ────────────────────────────────────────
    def _simulate(self) -> dict:
        sim = data_store.active_simulation
        now = time.time()
        dt  = now - self._last_ts
        self._last_ts = now

        if sim == "port_scan":
            n = random.randint(200, 500)
            ports = random.randint(150, 600)
            syn   = int(n * 0.92)
            rst   = int(n * 0.85)
            ent   = min(math.log2(max(ports, 1)), 10)
            return {
                "f_pkt_count": n,  "f_byte_count": n*70,
                "f_syn_count": syn,"f_ack_count": 0,
                "f_fin_count": 0,  "f_rst_count": rst,
                "f_psh_count": 0,  "f_urg_count": 0,
                "f_udp_count": 0,  "f_icmp_count": 0,
                "f_tcp_count": n,
                "f_unique_src_ips": 1,
                "f_unique_dst_ports": ports,
                "f_pkt_rate":  n/dt, "f_byte_rate": n*70/dt,
                "f_syn_rate":  syn/dt,
                "f_avg_pkt_size": 70,  "f_max_pkt_size": 80,
                "f_min_pkt_size": 60,
                "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.95,
                "f_tcp_udp_ratio": 1000, "f_syn_ack_ratio": 100,
                "f_fin_rst_ratio": 0.01, "f_icmp_ratio": 0.0,
                "f_dst_port_entropy": ent, "f_src_ip_entropy": 0.1,
                "f_payload_entropy": 1.2,
                "f_iat_mean": 1/max(n/dt,1), "f_iat_std": 0.0001,
                "f_iat_min": 0.00005, "f_iat_max": 0.002,
                "_window_dur": dt,
            }

        elif sim == "dos_ddos":
            n = random.randint(3000, 8000)
            syn = int(n * 0.7)
            return {
                "f_pkt_count": n,   "f_byte_count": n*60,
                "f_syn_count": syn, "f_ack_count": int(n*0.1),
                "f_fin_count": 0,   "f_rst_count": 0,
                "f_psh_count": 0,   "f_urg_count": 0,
                "f_udp_count": int(n*0.15), "f_icmp_count": int(n*0.05),
                "f_tcp_count": int(n*0.8),
                "f_unique_src_ips":   random.randint(50, 500),
                "f_unique_dst_ports": random.randint(1, 3),
                "f_pkt_rate":  n/dt, "f_byte_rate": n*60/dt, "f_syn_rate": syn/dt,
                "f_avg_pkt_size": 60, "f_max_pkt_size": 90, "f_min_pkt_size": 50,
                "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.98,
                "f_tcp_udp_ratio": 5, "f_syn_ack_ratio": 30,
                "f_fin_rst_ratio": 0.01, "f_icmp_ratio": 0.05,
                "f_dst_port_entropy": 0.5, "f_src_ip_entropy": 8.5,
                "f_payload_entropy": 2.0,
                "f_iat_mean": 1/max(n/dt,1), "f_iat_std": 0.00005,
                "f_iat_min": 0.00001, "f_iat_max": 0.001,
                "_window_dur": dt,
            }

        elif sim == "brute_force":
            n = random.randint(30, 80)
            return {
                "f_pkt_count": n,   "f_byte_count": n*120,
                "f_syn_count": int(n*0.4), "f_ack_count": int(n*0.5),
                "f_fin_count": int(n*0.1), "f_rst_count": 0,
                "f_psh_count": int(n*0.3), "f_urg_count": 0,
                "f_udp_count": 0,   "f_icmp_count": 0,
                "f_tcp_count": n,
                "f_unique_src_ips": 1,
                "f_unique_dst_ports": 1,
                "f_pkt_rate":  n/dt, "f_byte_rate": n*120/dt,
                "f_syn_rate":  int(n*0.4)/dt,
                "f_avg_pkt_size": 120, "f_max_pkt_size": 200, "f_min_pkt_size": 80,
                "f_large_pkt_ratio": 0.1, "f_small_pkt_ratio": 0.3,
                "f_tcp_udp_ratio": 1000, "f_syn_ack_ratio": 0.8,
                "f_fin_rst_ratio": 1.0, "f_icmp_ratio": 0.0,
                "f_dst_port_entropy": 0.1, "f_src_ip_entropy": 0.0,
                "f_payload_entropy": 4.2,
                "f_iat_mean": dt/max(n,1), "f_iat_std": 0.005,
                "f_iat_min": 0.05, "f_iat_max": 0.5,
                "_window_dur": dt,
            }

        elif sim == "heartbleed":
            n = random.randint(5, 25)
            return {
                "f_pkt_count": n,   "f_byte_count": n*80,
                "f_syn_count": int(n*0.2), "f_ack_count": int(n*0.6),
                "f_fin_count": 0,   "f_rst_count": 0,
                "f_psh_count": int(n*0.4), "f_urg_count": 0,
                "f_udp_count": 0,   "f_icmp_count": 0,
                "f_tcp_count": n,
                "f_unique_src_ips": 1,
                "f_unique_dst_ports": 1,
                "f_pkt_rate":  n/dt, "f_byte_rate": n*80/dt,
                "f_syn_rate":  int(n*0.2)/dt,
                "f_avg_pkt_size": 80, "f_max_pkt_size": 100, "f_min_pkt_size": 64,
                "f_large_pkt_ratio": 0.0, "f_small_pkt_ratio": 0.9,
                "f_tcp_udp_ratio": 1000, "f_syn_ack_ratio": 0.3,
                "f_fin_rst_ratio": 0.1, "f_icmp_ratio": 0.0,
                "f_dst_port_entropy": 0.0, "f_src_ip_entropy": 0.0,
                "f_payload_entropy": 1.5,
                "f_iat_mean": dt/max(n,1), "f_iat_std": 0.05,
                "f_iat_min": 0.1, "f_iat_max": 1.0,
                "_window_dur": dt,
            }

        elif sim == "botnet":
            n = random.randint(100, 300)
            return {
                "f_pkt_count": n,   "f_byte_count": n*300,
                "f_syn_count": int(n*0.1), "f_ack_count": int(n*0.6),
                "f_fin_count": int(n*0.1), "f_rst_count": 0,
                "f_psh_count": int(n*0.3), "f_urg_count": 0,
                "f_udp_count": int(n*0.2), "f_icmp_count": 0,
                "f_tcp_count": int(n*0.8),
                "f_unique_src_ips":   random.randint(10, 50),
                "f_unique_dst_ports": random.randint(3, 8),
                "f_pkt_rate":  n/dt, "f_byte_rate": n*300/dt,
                "f_syn_rate":  int(n*0.1)/dt,
                "f_avg_pkt_size": 300, "f_max_pkt_size": 800, "f_min_pkt_size": 60,
                "f_large_pkt_ratio": 0.3, "f_small_pkt_ratio": 0.1,
                "f_tcp_udp_ratio": 4, "f_syn_ack_ratio": 0.2,
                "f_fin_rst_ratio": 1.0, "f_icmp_ratio": 0.0,
                "f_dst_port_entropy": 2.5, "f_src_ip_entropy": 5.0,
                "f_payload_entropy": 7.2,
                "f_iat_mean": dt/max(n,1), "f_iat_std": 0.01,
                "f_iat_min": 0.002, "f_iat_max": 0.5,
                "_window_dur": dt,
            }

        else:  # Normal — low rate idle traffic
            n = random.randint(1, 6)
            ports = random.randint(1, 3)
            return {
                "f_pkt_count": n,   "f_byte_count": n*random.randint(200, 900),
                "f_syn_count": random.randint(0, 1),
                "f_ack_count": random.randint(0, 5),
                "f_fin_count": 0, "f_rst_count": 0,
                "f_psh_count": random.randint(0, 2),
                "f_urg_count": 0,
                "f_udp_count": random.randint(0, 1),
                "f_icmp_count": 0,
                "f_tcp_count": n - random.randint(0, 1),
                "f_unique_src_ips": random.randint(1, 2),
                "f_unique_dst_ports": ports,
                "f_pkt_rate":  n/dt, "f_byte_rate": n*400/dt,
                "f_syn_rate":  0.5/dt,
                "f_avg_pkt_size": random.uniform(200, 900),
                "f_max_pkt_size": 1400, "f_min_pkt_size": 60,
                "f_large_pkt_ratio": random.uniform(0.1, 0.4),
                "f_small_pkt_ratio": random.uniform(0.0, 0.1),
                "f_tcp_udp_ratio": random.uniform(3, 10),
                "f_syn_ack_ratio": random.uniform(0.0, 0.2),
                "f_fin_rst_ratio": random.uniform(0.5, 2.0),
                "f_icmp_ratio": 0.0,
                "f_dst_port_entropy": random.uniform(0.3, 1.5),
                "f_src_ip_entropy":   random.uniform(0.3, 1.2),
                "f_payload_entropy":  random.uniform(4.5, 7.0),
                "f_iat_mean":  random.uniform(0.1, 2.0),
                "f_iat_std":   random.uniform(0.01, 0.5),
                "f_iat_min":   random.uniform(0.01, 0.1),
                "f_iat_max":   random.uniform(0.5, 5.0),
                "_window_dur": dt,
            }

    def run(self):
        self.running = True
        self.sim_mode = not self._try_load_bpf()

        logger.info(f"XDP Engine running ({'simulation' if self.sim_mode else 'BPF'} mode)")

        while self.running:
            if self.sim_mode:
                raw = self._simulate()
                sim = data_store.active_simulation
                sleep_t = 0.05 if sim else 0.5
            else:
                raw = self._read_bpf_stats()
                sleep_t = UPDATE_INTERVAL

            if raw:
                data_store.add_xdp_metric(raw)
                self._flow_acc.ingest(raw)

                # Emit flow features periodically
                flow_feats = self._flow_acc.compute_flow_features()
                if flow_feats:
                    data_store.add_flow_metric(flow_feats)

                # Console log
                sim = data_store.active_simulation or "NORMAL"
                sys.stdout.write(
                    f"\r[XDP] {time.strftime('%H:%M:%S')} "
                    f"| Pkts={raw.get('f_pkt_count',0):5d} "
                    f"| Rate={raw.get('f_pkt_rate',0):8.1f}/s "
                    f"| SYN={raw.get('f_syn_count',0):5d} "
                    f"| Ports={raw.get('f_unique_dst_ports',0):5d} "
                    f"| Mode={sim.upper()[:12]}   "
                )
                sys.stdout.flush()

            time.sleep(sleep_t)

    def stop(self):
        self.running = False
        if self.bpf:
            try:
                self.bpf.remove_xdp(INTERFACE, 0)
                logger.info("XDP detached")
            except Exception:
                pass
