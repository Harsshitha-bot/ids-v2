#!/usr/bin/env python3
# ============================================================
# defense/xdp_firewall.py  —  Real-Time XDP Firewall (Feature 5)
#
# When detector confirms an attack with high confidence:
#   1. Looks up source IP in xdp_metrics
#   2. Calls xdp_engine.block_ip(ip) → updates BPF map (real)
#      or in-memory set (simulation)
#   3. Auto-expires blocks after XDP_BLOCK_DURATION seconds
#   4. Maintains an event log for the dashboard
# ============================================================

import os, sys, time, threading, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import data_store, get_logger

logger = get_logger("xdp_firewall")


class XDPFirewall:
    """
    Monitors prediction stream and enforces IP blocks.
    Works in both BPF and simulation mode.
    """
    def __init__(self, xdp_engine=None):
        self.running    = False
        self.xdp_engine = xdp_engine   # reference to kernel/xdp_engine.XDPEngine
        self.block_log  = collections.deque(maxlen=500)
        self._streak    = collections.defaultdict(int)
        self._last_preds= set()

    def _should_block(self, pred: dict) -> bool:
        return (pred.get("status") == "ATTACK" and
                pred.get("confidence", 0) >= CONFIDENCE_THRESH and
                pred.get("action") == "xdp_drop")

    def _block(self, ip: str, attack_type: str, conf: float):
        if data_store.is_blocked(ip):
            return
        data_store.block_ip(ip, XDP_BLOCK_DURATION)
        if self.xdp_engine:
            self.xdp_engine.block_ip(ip)
        self.block_log.append({
            "time":    time.time(),
            "ip":      ip,
            "type":    attack_type,
            "conf":    conf,
            "action":  "XDP_DROP",
            "expires": time.time() + XDP_BLOCK_DURATION,
        })
        logger.warning(f"🔥 FIREWALL: Blocked {ip} ({attack_type} conf={conf:.0%}) "
                       f"for {XDP_BLOCK_DURATION}s")

    def get_block_log(self, n=30):
        return list(self.block_log)[-n:]

    def run(self):
        self.running = True
        logger.info("XDP Firewall running")
        while self.running:
            preds = data_store.get_recent_predictions(5)
            for p in preds:
                ts = p.get("timestamp", 0)
                if ts in self._last_preds:
                    continue
                self._last_preds.add(ts)
                if len(self._last_preds) > 200:
                    self._last_preds = set(list(self._last_preds)[-100:])

                ip = p.get("src_ip", "")
                if not ip:
                    continue
                if p.get("status") == "ATTACK" and p.get("confidence", 0) >= CONFIDENCE_THRESH:
                    self._streak[ip] += 1
                    if self._streak[ip] >= 3:
                        self._block(ip, p.get("attack_type","?"), p.get("confidence",0))
                else:
                    self._streak[ip] = max(0, self._streak.get(ip, 0) - 1)

            time.sleep(1.0)

    def stop(self):
        self.running = False
