#!/usr/bin/env python3
# ============================================================
# monitors/ebpf_monitor.py  —  eBPF Syscall Monitor v2
#
# BPF tracepoints for:
#   execve, connect, accept, fork, clone, kill, ptrace,
#   mmap, socket, bind, listen, sendto, recvfrom,
#   chmod, chown, setuid  (dangerous system calls)
#
# Feature 2 improvement: computes entropy, rate, ratios per window
# Feature 3: syscall stats are keyed by process (pid grouping)
# ============================================================

import os, sys, time, math, random, threading
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.shared_data import data_store, get_logger

logger = get_logger("ebpf_monitor")

EBPF_BPF_CODE = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Per-pid syscall counters
struct pid_stats_t {
    u32 execve; u32 connect; u32 accept; u32 fork; u32 clone;
    u32 kill;   u32 ptrace;  u32 mmap;   u32 socket;
    u32 bind;   u32 listen;  u32 sendto; u32 recvfrom;
    u32 read;   u32 write;   u32 open;   u32 close;
    u32 ioctl;  u32 prctl;   u32 unlink; u32 chmod;
    u32 chown;  u32 setuid;
};

BPF_HASH(pid_stats, u32, struct pid_stats_t, 4096);
BPF_ARRAY(global_total, u64, 1);

static __always_inline void inc_total() {
    u32 z = 0; u64 *p = global_total.lookup(&z);
    if (p) __sync_fetch_and_add(p, 1);
}

#define SYSCALL_TP(name, field) \
    TRACEPOINT_PROBE(syscalls, sys_enter_##name) { \
        u32 pid = bpf_get_current_pid_tgid() >> 32; \
        struct pid_stats_t zero = {}; \
        struct pid_stats_t *s = pid_stats.lookup_or_try_init(&pid, &zero); \
        if (s) __sync_fetch_and_add(&s->field, 1); \
        inc_total(); \
        return 0; \
    }

SYSCALL_TP(execve,    execve)
SYSCALL_TP(connect,   connect)
SYSCALL_TP(accept,    accept)
SYSCALL_TP(accept4,   accept)
SYSCALL_TP(fork,      fork)
SYSCALL_TP(clone,     clone)
SYSCALL_TP(kill,      kill)
SYSCALL_TP(ptrace,    ptrace)
SYSCALL_TP(mmap,      mmap)
SYSCALL_TP(socket,    socket)
SYSCALL_TP(bind,      bind)
SYSCALL_TP(listen,    listen)
SYSCALL_TP(sendto,    sendto)
SYSCALL_TP(recvfrom,  recvfrom)
SYSCALL_TP(read,      read)
SYSCALL_TP(write,     write)
SYSCALL_TP(open,      open)
SYSCALL_TP(openat,    open)
SYSCALL_TP(close,     close)
SYSCALL_TP(ioctl,     ioctl)
SYSCALL_TP(prctl,     prctl)
SYSCALL_TP(unlink,    unlink)
SYSCALL_TP(chmod,     chmod)
SYSCALL_TP(chown,     chown)
SYSCALL_TP(setuid,    setuid)
"""


def _entropy(counts: list) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


class EBPFMonitor:
    def __init__(self):
        self.running   = False
        self.bpf       = None
        self.sim_mode  = False
        self._prev_total = 0
        self._last_ts    = time.time()

    def _try_load_bpf(self) -> bool:
        try:
            from bcc import BPF
            b = BPF(text=EBPF_BPF_CODE)
            self.bpf = b
            logger.info("eBPF tracepoints loaded")
            return True
        except Exception as e:
            logger.warning(f"eBPF load failed: {e} — simulation")
            return False

    def _read_bpf(self) -> dict:
        if not self.bpf:
            return {}
        now = time.time()
        dt  = now - self._last_ts
        self._last_ts = now

        totals = {
            "execve":0,"connect":0,"accept":0,"fork":0,"clone":0,
            "kill":0,"ptrace":0,"mmap":0,"socket":0,"bind":0,"listen":0,
            "sendto":0,"recvfrom":0,"read":0,"write":0,"open":0,"close":0,
            "ioctl":0,"prctl":0,"unlink":0,"chmod":0,"chown":0,"setuid":0,
        }
        try:
            for _pid, s in self.bpf["pid_stats"].items():
                for field in totals:
                    totals[field] += getattr(s, field, 0)
        except Exception:
            pass

        gt = self.bpf["global_total"][0].value
        total_delta = gt - self._prev_total
        self._prev_total = gt

        return self._build_features(totals, total_delta, dt)

    def _build_features(self, t: dict, total: int, dt: float) -> dict:
        dangerous = t["kill"] + t["ptrace"] + t["chmod"] + t["chown"] + t["setuid"]
        net  = t["connect"] + t["accept"] + t["socket"] + t["bind"] + t["listen"] + t["sendto"] + t["recvfrom"]
        file = t["read"]    + t["write"]   + t["open"]   + t["close"] + t["unlink"]
        proc = t["execve"]  + t["fork"]    + t["clone"]  + t["kill"]  + t["ptrace"]

        counts = [t[k] for k in t]
        entropy = _entropy(counts)

        return {
            "e_syscall_rate":    total    / max(dt, 0.001),
            "e_execve_count":    t["execve"],
            "e_connect_count":   t["connect"],
            "e_accept_count":    t["accept"],
            "e_read_count":      t["read"],
            "e_write_count":     t["write"],
            "e_open_count":      t["open"],
            "e_close_count":     t["close"],
            "e_fork_count":      t["fork"],
            "e_kill_count":      t["kill"],
            "e_ptrace_count":    t["ptrace"],
            "e_mmap_count":      t["mmap"],
            "e_socket_count":    t["socket"],
            "e_bind_count":      t["bind"],
            "e_listen_count":    t["listen"],
            "e_sendto_count":    t["sendto"],
            "e_recvfrom_count":  t["recvfrom"],
            "e_ioctl_count":     t["ioctl"],
            "e_prctl_count":     t["prctl"],
            "e_clone_count":     t["clone"],
            "e_unlink_count":    t["unlink"],
            "e_chmod_count":     t["chmod"],
            "e_chown_count":     t["chown"],
            "e_setuid_count":    t["setuid"],
            "e_net_ratio":       net  / max(total, 1),
            "e_file_ratio":      file / max(total, 1),
            "e_proc_ratio":      proc / max(total, 1),
            "e_dangerous_count": dangerous,
            "e_syscall_entropy": entropy,
        }

    def _simulate(self) -> dict:
        sim = data_store.active_simulation
        now = time.time()
        dt  = now - self._last_ts
        self._last_ts = now

        def r(a, b): return random.randint(a, b)

        if sim == "port_scan":
            t = {"execve":0,"connect":r(100,500),"accept":0,"fork":r(0,3),"clone":0,
                 "kill":0,"ptrace":0,"mmap":0,"socket":r(100,500),"bind":0,"listen":0,
                 "sendto":0,"recvfrom":0,"read":r(0,10),"write":r(0,5),"open":r(0,5),
                 "close":r(100,500),"ioctl":0,"prctl":0,"unlink":0,"chmod":0,"chown":0,"setuid":0}
            total = sum(t.values())

        elif sim == "dos_ddos":
            t = {"execve":0,"connect":r(5,20),"accept":r(5,20),"fork":r(50,200),"clone":r(50,200),
                 "kill":0,"ptrace":0,"mmap":r(5,20),"socket":r(10,50),"bind":0,"listen":0,
                 "sendto":r(500,2000),"recvfrom":r(100,400),"read":r(100,300),"write":r(100,400),
                 "open":r(5,20),"close":r(50,200),"ioctl":0,"prctl":0,"unlink":0,
                 "chmod":0,"chown":0,"setuid":0}
            total = sum(t.values())

        elif sim == "brute_force":
            t = {"execve":r(20,80),"connect":r(50,150),"accept":0,"fork":r(20,80),"clone":0,
                 "kill":0,"ptrace":0,"mmap":r(5,15),"socket":r(50,150),"bind":0,"listen":0,
                 "sendto":r(20,80),"recvfrom":r(20,80),"read":r(20,60),"write":r(10,40),
                 "open":r(10,30),"close":r(40,150),"ioctl":0,"prctl":0,"unlink":0,
                 "chmod":0,"chown":0,"setuid":0}
            total = sum(t.values())

        elif sim == "heartbleed":
            t = {"execve":0,"connect":r(1,5),"accept":0,"fork":0,"clone":0,
                 "kill":0,"ptrace":0,"mmap":r(30,100),"socket":r(1,5),"bind":0,"listen":0,
                 "sendto":r(2,10),"recvfrom":r(5,20),"read":r(20,60),"write":r(5,20),
                 "open":r(1,5),"close":r(2,10),"ioctl":0,"prctl":0,"unlink":0,
                 "chmod":0,"chown":0,"setuid":0}
            total = sum(t.values())

        elif sim == "botnet":
            t = {"execve":r(2,10),"connect":r(20,80),"accept":0,"fork":r(2,10),"clone":0,
                 "kill":r(3,15),"ptrace":r(2,10),"mmap":r(5,20),"socket":r(20,80),"bind":0,"listen":0,
                 "sendto":r(30,100),"recvfrom":r(20,70),"read":r(20,60),"write":r(10,40),
                 "open":r(5,20),"close":r(20,80),"ioctl":0,"prctl":r(2,8),"unlink":r(1,5),
                 "chmod":r(2,8),"chown":r(1,5),"setuid":r(1,5)}
            total = sum(t.values())

        else:  # Normal — low baseline
            t = {"execve":r(0,1),"connect":r(0,2),"accept":0,"fork":0,"clone":0,
                 "kill":0,"ptrace":0,"mmap":r(0,2),"socket":r(0,1),"bind":0,"listen":0,
                 "sendto":r(0,2),"recvfrom":r(0,2),"read":r(2,10),"write":r(1,6),
                 "open":r(0,4),"close":r(1,5),"ioctl":0,"prctl":0,"unlink":0,
                 "chmod":0,"chown":0,"setuid":0}
            total = sum(t.values())

        feats = self._build_features(t, total, dt)

        sys.stdout.write(
            f"\r[eBPF] {time.strftime('%H:%M:%S')} "
            f"| Rate={feats['e_syscall_rate']:7.1f}/s "
            f"| Exec={t['execve']:4d} "
            f"| Conn={t['connect']:4d} "
            f"| Danger={feats['e_dangerous_count']:3d} "
            f"| Ent={feats['e_syscall_entropy']:.2f}   "
        )
        sys.stdout.flush()
        return feats

    def run(self):
        self.running = True
        self.sim_mode = not self._try_load_bpf()
        logger.info(f"eBPF Monitor running ({'simulation' if self.sim_mode else 'BPF'} mode)")

        while self.running:
            if self.sim_mode:
                m = self._simulate()
                sim = data_store.active_simulation
                time.sleep(0.05 if sim else 0.5)
            else:
                m = self._read_bpf()
                time.sleep(UPDATE_INTERVAL)

            if m:
                data_store.add_ebpf_metric(m)

    def stop(self):
        self.running = False
