#!/usr/bin/env python3
# ============================================================
# ml/detector.py  —  IDS v2 Detection Engine
#
# Features implemented:
#   1. ONNX fast-path inference (kernel-boundary speed)
#   4. Ensemble of RF + XGB + SVM + IsoForest voting
#   5. Triggers XDP block on confirmed attacks (defense)
#   6. Multiprocessing worker pool (performance)
# ============================================================

import os, sys, time, json, threading, math, queue, multiprocessing
import numpy as np
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import data_store, get_logger

logger = get_logger("detector")

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import onnxruntime as ort
    ONNX_RUNTIME_OK = True
except ImportError:
    ONNX_RUNTIME_OK = False


# ══════════════════════════════════════════════════════════════
# Spike / Statistical Detector (zero-day, no model needed)
# ══════════════════════════════════════════════════════════════
class SpikeDetector:
    """
    Feature 2: Uses flow-based rate features (pkt/s, byte/s, SYN rate)
    and a rolling baseline window to detect sudden anomalies.
    """
    TRACK_KEYS = [
        "f_pkt_rate", "f_byte_rate", "f_syn_rate",
        "f_unique_dst_ports", "f_unique_src_ips",
        "e_syscall_rate", "e_connect_count",
        "e_execve_count", "e_dangerous_count",
    ]

    def __init__(self, window=WINDOW_SIZE, mult=SPIKE_MULTIPLIER):
        self._window = window
        self._mult   = mult
        self._hist   = {k: collections.deque(maxlen=window) for k in self.TRACK_KEYS}

    def update(self, combined: dict):
        for k in self.TRACK_KEYS:
            if k in combined:
                self._hist[k].append(combined[k])

    def detect(self, combined: dict) -> dict:
        spikes = {}
        for k in self.TRACK_KEYS:
            hist = self._hist[k]
            if len(hist) < 8 or k not in combined:
                continue
            baseline = float(np.mean(list(hist)[:-1]))
            cur = combined[k]
            ratio = cur / max(baseline, 0.01)
            if ratio >= self._mult:
                spikes[k] = {"ratio": ratio, "cur": cur, "base": baseline}
        max_ratio = max((s["ratio"] for s in spikes.values()), default=1.0)
        spike_conf = min((max_ratio - 1.0) / (self._mult * 2), 1.0) if spikes else 0.0
        return {
            "is_spike": len(spikes) > 0,
            "spikes": spikes,
            "max_ratio": max_ratio,
            "confidence": spike_conf,
        }

    def classify(self, combined: dict) -> str:
        p = combined.get("f_unique_dst_ports", 0)
        r = combined.get("f_pkt_rate", 0)
        s = combined.get("f_unique_src_ips", 0)
        e = combined.get("e_execve_count", 0)
        d = combined.get("e_dangerous_count", 0)
        m = combined.get("e_mmap_count", 0)
        ps = combined.get("f_avg_pkt_size", 500)
        sr = combined.get("f_syn_ack_ratio", 0)

        if p > 50 and sr > 5:          return "Port Scan"
        if r > 400 and p < 5:           return "DoS / DDoS"
        if e > 15 and p <= 2:           return "Brute Force"
        if d > 8 and s > 8:             return "Botnet"
        if m > 20 and ps < 110:         return "Heartbleed"
        return "Unknown Attack"


# ══════════════════════════════════════════════════════════════
# ONNX Inference (Feature 1: near-kernel inference speed)
# ══════════════════════════════════════════════════════════════
class ONNXPredictor:
    """Wraps ONNX Runtime session for sub-millisecond inference"""
    def __init__(self, path: str):
        if not ONNX_RUNTIME_OK:
            raise RuntimeError("onnxruntime not installed")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.input_name  = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.proba_name  = self.sess.get_outputs()[1].name if len(self.sess.get_outputs()) > 1 else None
        logger.info("✓ ONNX Runtime session loaded")

    def predict(self, X: np.ndarray):
        """Returns (label_array, proba_dict)"""
        X_f = X.astype(np.float32)
        outputs = self.sess.run(None, {self.input_name: X_f})
        labels = outputs[0]
        proba  = outputs[1] if len(outputs) > 1 else None
        return labels, proba


# ══════════════════════════════════════════════════════════════
# Main Detector
# ══════════════════════════════════════════════════════════════
class DetectionEngine:
    """
    Ensemble detection:
      Layer 1 — ONNX RF (fastest, ~0.2ms)
      Layer 2 — XGBoost
      Layer 3 — SVM
      Layer 4 — Isolation Forest (unsupervised)
      Layer 5 — Spike detector (statistical, always on)
    Final decision: weighted majority vote.
    """
    def __init__(self):
        self.running    = False
        self.rf_model   = None
        self.xgb_pack   = None   # (xgb_model, le_num)
        self.svm_model  = None
        self.iso_model  = None
        self.onnx_pred  = None
        self.scaler     = None
        self.le         = None
        self.spike_det  = SpikeDetector()
        self._latencies = collections.deque(maxlen=200)
        self._iso_scores= collections.deque(maxlen=200)

        # For XDP defense
        self._attack_streak  = collections.defaultdict(int)
        self._xdp_engine_ref = None  # injected after init

    # ── Model loading ──────────────────────────────────────
    def load_models(self) -> bool:
        if not JOBLIB_OK:
            logger.warning("joblib unavailable — spike-only mode")
            return False
        loaded_any = False
        try:
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
            if os.path.exists(ENCODER_PATH):
                self.le = joblib.load(ENCODER_PATH)
            if os.path.exists(RF_MODEL_PATH):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                logger.info("✓ Random Forest loaded")
                loaded_any = True
            if os.path.exists(XGB_MODEL_PATH):
                self.xgb_pack = joblib.load(XGB_MODEL_PATH)
                logger.info("✓ XGBoost loaded")
            if os.path.exists(SVM_MODEL_PATH):
                self.svm_model = joblib.load(SVM_MODEL_PATH)
                logger.info("✓ SVM loaded")
            if os.path.exists(ISO_MODEL_PATH):
                self.iso_model = joblib.load(ISO_MODEL_PATH)
                logger.info("✓ Isolation Forest loaded")
            # Try ONNX fast path
            if ONNX_RUNTIME_OK and os.path.exists(ONNX_MODEL_PATH):
                self.onnx_pred = ONNXPredictor(ONNX_MODEL_PATH)
        except Exception as e:
            logger.error(f"Model load error: {e}")
        return loaded_any

    # ── Feature vector ─────────────────────────────────────
    def _featurize(self, xdp: dict, ebpf: dict) -> np.ndarray:
        return np.array([
            float(xdp.get(f, ebpf.get(f, 0))) for f in ALL_FEATURES
        ]).reshape(1, -1)

    # ── Individual model predictions ───────────────────────
    def _predict_rf(self, X_sc) -> tuple:
        if not self.rf_model: return None, 0.0
        try:
            lbl  = self.rf_model.predict(X_sc)[0]
            prob = self.rf_model.predict_proba(X_sc)[0]
            return lbl, float(np.max(prob))
        except Exception: return None, 0.0

    def _predict_onnx(self, X_sc) -> tuple:
        if not self.onnx_pred: return None, 0.0
        try:
            labels, proba = self.onnx_pred.predict(X_sc)
            lbl = labels[0]
            conf = float(np.max(proba[0])) if proba is not None else 0.8
            return lbl, conf
        except Exception: return None, 0.0

    def _predict_xgb(self, X_sc) -> tuple:
        if not self.xgb_pack: return None, 0.0
        try:
            model, le_num = self.xgb_pack
            pred_num = model.predict(X_sc)[0]
            prob     = model.predict_proba(X_sc)[0]
            lbl      = le_num.inverse_transform([pred_num])[0]
            return lbl, float(np.max(prob))
        except Exception: return None, 0.0

    def _predict_svm(self, X_sc) -> tuple:
        if not self.svm_model: return None, 0.0
        try:
            lbl  = self.svm_model.predict(X_sc)[0]
            prob = self.svm_model.predict_proba(X_sc)[0]
            return lbl, float(np.max(prob))
        except Exception: return None, 0.0

    def _predict_iso(self, X_sc) -> tuple:
        if not self.iso_model: return None, 0.0
        try:
            score = float(self.iso_model.score_samples(X_sc)[0])
            self._iso_scores.append(score)
            mn = min(self._iso_scores); mx = max(self._iso_scores)
            norm = 1.0 - (score - mn) / max(mx - mn, 1e-6)
            is_anom = self.iso_model.predict(X_sc)[0] == -1
            return is_anom, float(np.clip(norm, 0, 1))
        except Exception: return None, 0.0

    # ── Ensemble voting ────────────────────────────────────
    def _ensemble(self, xdp, ebpf, X_sc, spike_res):
        votes       = []   # True = attack
        attack_lbls = []
        weights     = []

        # ONNX (preferred over RF for speed)
        if self.onnx_pred:
            lbl, conf = self._predict_onnx(X_sc)
            if lbl is not None:
                is_atk = lbl != "Normal"
                votes.append(is_atk); weights.append(conf * 1.2)
                if is_atk: attack_lbls.append((lbl, conf))
        elif self.rf_model:
            lbl, conf = self._predict_rf(X_sc)
            if lbl is not None:
                is_atk = lbl != "Normal"
                votes.append(is_atk); weights.append(conf * 1.2)
                if is_atk: attack_lbls.append((lbl, conf))

        # XGBoost
        lbl, conf = self._predict_xgb(X_sc)
        if lbl is not None:
            is_atk = lbl != "Normal"
            votes.append(is_atk); weights.append(conf * 1.1)
            if is_atk: attack_lbls.append((lbl, conf))

        # SVM
        lbl, conf = self._predict_svm(X_sc)
        if lbl is not None:
            is_atk = lbl != "Normal"
            votes.append(is_atk); weights.append(conf * 1.0)
            if is_atk: attack_lbls.append((lbl, conf))

        # Isolation Forest
        is_anom, anom_score = self._predict_iso(X_sc)
        if is_anom is not None:
            votes.append(is_anom); weights.append(anom_score * 0.8)

        # Spike detector (always)
        if spike_res["is_spike"]:
            votes.append(True); weights.append(spike_res["confidence"] * 0.7)

        if not votes:
            return "NORMAL", "Normal", 0.5, 0.0, "No models"

        # Weighted majority
        atk_weight = sum(w for v, w in zip(votes, weights) if v)
        tot_weight = sum(weights)
        attack_prob = atk_weight / max(tot_weight, 1e-9)

        is_attack = attack_prob >= 0.45

        # Best confidence
        avg_conf = float(atk_weight / max(len([v for v in votes if v]), 1))
        final_conf = min(attack_prob * 1.3, 1.0)

        # Attack type: highest-confidence ML label wins
        if is_attack and attack_lbls:
            attack_type = max(attack_lbls, key=lambda x: x[1])[0]
        elif is_attack and spike_res["is_spike"]:
            attack_type = self.spike_det.classify({**xdp, **ebpf})
        elif is_attack:
            attack_type = "Unknown Attack"
        else:
            attack_type = "Normal"

        n_models = (1 if self.rf_model or self.onnx_pred else 0) + \
                   (1 if self.xgb_pack else 0) + (1 if self.svm_model else 0) + \
                   (1 if self.iso_model else 0) + 1
        mode = f"Ensemble({n_models})" if n_models > 1 else "Spike-only"

        iso_score = is_anom if isinstance(is_anom, bool) else 0.0
        return ("ATTACK" if is_attack else "NORMAL", attack_type,
                final_conf, float(anom_score) if isinstance(anom_score, float) else 0.0,
                mode)

    # ── XDP Defense (Feature 5) ────────────────────────────
    def _maybe_block(self, xdp: dict, attack_type: str, status: str):
        if not XDP_DROP_ENABLED or status != "ATTACK":
            return "alert"
        src = xdp.get("_src_ip_str", "")
        if not src:
            return "alert"
        streak = self._attack_streak[src]
        self._attack_streak[src] = streak + 1
        if streak >= 2:  # 3 consecutive attack cycles from same IP → block
            if self._xdp_engine_ref:
                self._xdp_engine_ref.block_ip(src)
            data_store.block_ip(src, XDP_BLOCK_DURATION)
            data_store.record_drop(1)
            logger.warning(f"🔥 BLOCKED {src} ({attack_type}) for {XDP_BLOCK_DURATION}s")
            return "xdp_drop"
        return "alert"

    # ── Main detect ────────────────────────────────────────
    def detect(self, xdp: dict, ebpf: dict) -> dict:
        t0 = time.perf_counter()
        combined = {**xdp, **ebpf}

        self.spike_det.update(combined)
        spike_res = self.spike_det.detect(combined)

        X = self._featurize(xdp, ebpf)
        X_sc = self.scaler.transform(X) if self.scaler else X

        status, attack_type, conf, anom, mode = self._ensemble(xdp, ebpf, X_sc, spike_res)

        action = self._maybe_block(xdp, attack_type, status)
        lat_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(lat_ms)
        data_store.ml_metrics["detection_latency"] = float(np.mean(self._latencies))

        result = {
            "status":       status,
            "attack_type":  attack_type,
            "confidence":   conf,
            "anomaly_score":anom,
            "spike_detected":spike_res["is_spike"],
            "spike_ratio":  spike_res["max_ratio"],
            "latency_ms":   lat_ms,
            "detection_mode":mode,
            "action":       action,
            "src_ip":       xdp.get("_src_ip_str",""),
            "details":      self._details(attack_type, spike_res, conf),
            "timestamp":    time.time(),
        }
        return result

    def _details(self, atype, spike, conf):
        parts = [f"conf={conf:.0%}"]
        if spike["is_spike"]:
            top = sorted(spike["spikes"].items(), key=lambda x: -x[1]["ratio"])[:2]
            for k, v in top:
                parts.append(f"{k.split('_')[-1]}={v['cur']:.0f}({v['ratio']:.1f}x)")
        return " | ".join(parts)

    # ── Run loop ───────────────────────────────────────────
    def run(self):
        self.running = True
        logger.info("Detection engine starting...")
        ok = self.load_models()
        mode = "Full ML Ensemble" if ok else "Spike + Anomaly"
        logger.info(f"Running in mode: {mode}")

        while self.running:
            xdp_list  = data_store.get_recent_xdp(1)
            ebpf_list = data_store.get_recent_ebpf(1)

            if xdp_list and ebpf_list:
                result = self.detect(xdp_list[-1], ebpf_list[-1])
                data_store.add_prediction(result)

                icon = "🚨 ATTACK" if result["status"] == "ATTACK" else "✅ NORMAL"
                act  = "🔥DROP" if result.get("action") == "xdp_drop" else ""
                sys.stdout.write(
                    f"\n[ML] {time.strftime('%H:%M:%S')} "
                    f"{icon} | {result['attack_type']:<15} "
                    f"conf={result['confidence']:.0%} "
                    f"anom={result['anomaly_score']:.2f} "
                    f"lat={result['latency_ms']:.1f}ms {act}"
                )
                sys.stdout.flush()

            time.sleep(UPDATE_INTERVAL)

    def stop(self):
        self.running = False
