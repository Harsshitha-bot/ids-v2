#!/usr/bin/env python3
# ============================================================
# run_demo.py  —  IDS v2 Main Entry Point
#
# Starts:
#   1. ML model training (or loads cached)
#   2. XDP engine (BPF or simulation)
#   3. eBPF syscall monitor (BPF or simulation)
#   4. Detection engine (ONNX + RF + XGB + SVM ensemble)
#   5. XDP Firewall (auto-block on attack confirmation)
#   6. Dashboard (http://localhost:8050)
#
# NO auto-attack — use dashboard buttons to trigger attacks
# ============================================================

import os, sys, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Ensure packages exist
for pkg in ["kernel","monitors","ml","flow","defense","utils","visualization","attackers","scripts"]:
    d = os.path.join(ROOT, pkg)
    os.makedirs(d, exist_ok=True)
    init = os.path.join(d, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

for d in ["data","logs","models","reports"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

print("""
╔══════════════════════════════════════════════════════════════╗
║       IDS v2  —  eBPF + XDP + ONNX + Flow Detection         ║
║       RF · XGBoost · SVM · IsoForest | CICIDS2017            ║
╚══════════════════════════════════════════════════════════════╝
""")

from utils.config import *
from utils.shared_data import data_store


# ── 1. Train / load models ────────────────────────────────────
print("[1/5] Training / loading ML models...")
try:
    if os.path.exists(RF_MODEL_PATH):
        print("  ✓ Models found — skipping training (delete models/ to retrain)")
    else:
        from ml.train_models import train_all_models
        train_all_models()
    print("  ✓ Models ready")
except Exception as e:
    print(f"  ⚠ Training error: {e} — continuing with spike-only detection")


# ── 2. XDP Engine ─────────────────────────────────────────────
print("[2/5] Starting XDP engine...")
from kernel.xdp_engine import XDPEngine
xdp_engine = XDPEngine()
t_xdp = threading.Thread(target=xdp_engine.run, daemon=True, name="xdp")
t_xdp.start()
time.sleep(2)
xd = data_store.get_recent_xdp(1)
print(f"  ✓ XDP {'BPF' if not xdp_engine.sim_mode else 'simulation'} | "
      f"pkt_rate={xd[-1].get('f_pkt_rate',0):.1f}/s" if xd else "  ✓ XDP started")


# ── 3. eBPF Monitor ───────────────────────────────────────────
print("[3/5] Starting eBPF syscall monitor...")
from monitors.ebpf_monitor import EBPFMonitor
ebpf_mon = EBPFMonitor()
t_ebpf = threading.Thread(target=ebpf_mon.run, daemon=True, name="ebpf")
t_ebpf.start()
time.sleep(2)
eb = data_store.get_recent_ebpf(1)
print(f"  ✓ eBPF {'BPF' if not ebpf_mon.sim_mode else 'simulation'} | "
      f"syscall_rate={eb[-1].get('e_syscall_rate',0):.1f}/s" if eb else "  ✓ eBPF started")


# ── 4. Detection engine ───────────────────────────────────────
print("[4/5] Starting detection engine...")
from ml.detector import DetectionEngine
detector = DetectionEngine()
detector._xdp_engine_ref = xdp_engine   # for IP blocking
t_det = threading.Thread(target=detector.run, daemon=True, name="detector")
t_det.start()
time.sleep(1.5)
print("  ✓ Detection engine running (ONNX + RF + XGB + SVM ensemble)")


# ── 5. XDP Firewall ───────────────────────────────────────────
print("[5/5] Starting XDP firewall...")
from defense.xdp_firewall import XDPFirewall
firewall = XDPFirewall(xdp_engine=xdp_engine)
t_fw = threading.Thread(target=firewall.run, daemon=True, name="firewall")
t_fw.start()
print("  ✓ XDP Firewall active (auto-block on 3 consecutive attack cycles)")


print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ All components running!                                   ║
║                                                              ║
║  📊 Dashboard → http://localhost:{DASHBOARD_PORT}                   ║
║                                                              ║
║  🎮 HOW TO USE:                                              ║
║    • Loads showing NORMAL traffic (green)                    ║
║    • Click attack buttons to simulate attacks                ║
║    • Click ✅ Normal to return to baseline                   ║
║    • Tab "Flow Analysis" → per-flow detection (5s windows)   ║
║    • Tab "ML Metrics"   → RF/XGB/SVM comparison             ║
║    • Tab "Defense"      → XDP block events                   ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── Launch Dashboard (blocks) ────────────────────────────────
from visualization.dashboard import app as dash_app

try:
    dash_app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
except KeyboardInterrupt:
    print("\n[!] Shutting down...")
    xdp_engine.stop(); ebpf_mon.stop(); detector.stop(); firewall.stop()
    sys.exit(0)
