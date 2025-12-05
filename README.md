#!/usr/bin/env python3
# Colab/Gemini-friendly RFM demo for Branch-and-Bound knapsack
# - Simple, well-commented, easy-to-parse structure
# - EMA + percentile blended per-node time estimator (t_hat)
# - RFS = slack * time_factor decision rule
# - Cap-predictive and strict modes
# - CSV logging per node
# Keep this file short and linear for easiest consumption by notebook AIs.

from __future__ import annotations
import argparse
import csv
import heapq
import math
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple, Optional

Node = namedtuple("Node", ["i", "value", "weight", "taken_mask", "depth"])

# ---- Defaults ----
DEFAULT_P90_CAL = 0.00018
DEFAULT_EMA_ALPHA = 0.2
DEFAULT_W_EMA = 0.6
DEFAULT_P90_WINDOW = 200
DEFAULT_THETA_P = 0.20
DEFAULT_THETA_C = 0.60
DEFAULT_S_MAX = 200
DEFAULT_TIMEOUT_MULT = 3.0

# ---- Utilities ----
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fractional_bound(start, value, weight, items, C):
    """Greedy fractional upper bound from position 'start'."""
    rem = C - weight
    b = float(value)
    for j in range(start, len(items)):
        w, v = items[j]
        if w <= rem:
            rem -= w
            b += v
        else:
            if w > 0:
                b += v * (rem / w)
            break
    return b

def percentile(sorted_list: List[float], p: float) -> float:
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_list[int(k)]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)

def generate_instance(n, seed=None, cap_ratio=0.5):
    rnd = random.Random(seed)
    items = [(rnd.randint(1, 100), rnd.randint(1, 200)) for _ in range(n)]
    # sort by value/weight descending for good fractional bound
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    total_w = sum(w for w, _ in items)
    C = max(1, int(total_w * cap_ratio))
    return items, C

# ---- Estimator helpers ----
class TimeEstimator:
    """Keep recent node times, EMA and percentile blend for per-node prediction."""
    def __init__(self, p90_window=DEFAULT_P90_WINDOW, ema_alpha=DEFAULT_EMA_ALPHA, w_ema=DEFAULT_W_EMA, calibrated_p90=DEFAULT_P90_CAL, pct=90.0):
        self.window = deque(maxlen=p90_window)
        self.ema = float(calibrated_p90)
        self.ema_alpha = float(ema_alpha)
        self.w_ema = float(w_ema)
        self.pct = float(pct)

    def update(self, observed: float):
        if observed <= 0:
            return
        self.window.append(observed)
        # EMA update
        a = self.ema_alpha
        self.ema = a * observed + (1.0 - a) * self.ema

    def per_node_hat(self):
        sorted_list = sorted(self.window)
        pct_val = percentile(sorted_list, self.pct) if sorted_list else self.ema
        return self.w_ema * self.ema + (1.0 - self.w_ema) * pct_val

# ---- RFS core ----
def slack(B, incumbent):
    if B <= incumbent or B <= 0:
        return 0.0
    return max(0.0, (B - incumbent) / max(1.0, B))

def time_factor(per_node_hat, nodes_est, time_left):
    pred = per_node_hat * max(1, int(nodes_est))
    denom = pred + time_left
    if denom <= 0:
        return 0.0, pred
    return clamp(time_left / denom, 0.0, 1.0), pred

def rfs_score(B, incumbent, per_node_hat, nodes_est, time_left):
    s = slack(B, incumbent)
    tf, pred = time_factor(per_node_hat, nodes_est, time_left)
    return s * tf, s, tf, pred

# ---- Simple B&B solver with RFS decisions ----
def rfm_bb(items: List[Tuple[int,int]],
           C: int,
           T: float = 10.0,
           mode: str = "cap-predictive",
           node_sleep: float = 0.0,
           estimator: Optional[TimeEstimator] = None,
           theta_p: float = DEFAULT_THETA_P,
           theta_c: float = DEFAULT_THETA_C,
           S_max: int = DEFAULT_S_MAX,
           timeout_mult: float = DEFAULT_TIMEOUT_MULT,
           out_csv: Optional[str] = None):
    start_time = time.time()
    est = estimator or TimeEstimator()
    # initial incumbent from greedy
    cap = C
    inc = 0
    caplist = []
    for w, v in sorted(items, key=lambda x: x[1]/x[0], reverse=True):
        if w <= cap:
            cap -= w; inc += v
    # event logging
    rows = []
    if out_csv:
        f = open(out_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["time","node_i","bound","action","rfs","slack","time_factor","pred_total","incumbent"])
    # root
    root_bound = fractional_bound(0, 0, 0, items, C)
    root = Node(0, 0, 0, 0)
    heap = [(-root_bound, root)]
    nodes_expanded = 0
    nodes_pruned = 0
    nodes_capped = 0

    while heap:
        elapsed = time.time() - start_time
        time_left = T - elapsed
        if time_left <= 0:
            break
        _, node = heapq.heappop(heap)
        # simulate node work
        t0 = time.time()
        if node_sleep > 0:
            time.sleep(node_sleep)
        observed = time.time() - t0
        est.update(observed)
        per_hat = est.per_node_hat()
        # estimate nodes in subtree conservatively
        nodes_est = min(max(1, len(items) - node.i), S_max)
        B = fractional_bound(node.i, node.value, node.weight, items, C)
        score, s, tf, pred_total = rfs_score(B, inc, per_hat, nodes_est, time_left)

        # decision thresholds
        action = "expand"
        if score < theta_p:
            action = "prune_by_rfs"
            nodes_pruned += 1
            if out_csv:
                writer.writerow([time.time()-start_time, node.i, B, action, score, s, tf, pred_total, inc])
            rows.append((node.i, B, action, score))
            continue
        if score < theta_c:
            # cap the bound predictively
            action = "cap"
            effective = inc + (B - inc) * (time_left / max(1e-9, T))
            nodes_capped += 1
        else:
            effective = B

        if effective <= inc:
            action = "prune_by_bound"
            if out_csv:
                writer.writerow([time.time()-start_time, node.i, B, action, score, s, tf, pred_total, inc])
            rows.append((node.i, B, action, score))
            continue

        # expand node
        nodes_expanded += 1
        if node.i < len(items):
            w, v = items[node.i]
            # take branch
            if node.weight + w <= C:
                child_take = Node(node.i + 1, node.value + v, node.weight + w, node.taken_mask | (1 << node.i))
                b_take = fractional_bound(child_take.i, child_take.value, child_take.weight, items, C)
                heapq.heappush(heap, (-b_take, child_take))
            # exclude branch
            child_excl = Node(node.i + 1, node.value, node.weight, node.taken_mask)
            b_excl = fractional_bound(child_excl.i, child_excl.value, child_excl.weight, items, C)
            heapq.heappush(heap, (-b_excl, child_excl))

        if out_csv:
            writer.writerow([time.time()-start_time, node.i, B, action, score, s, tf, pred_total, inc])
        rows.append((node.i, B, action, score))

    if out_csv:
        f.close()
    summary = {
        "incumbent": inc,
        "nodes_expanded": nodes_expanded,
        "nodes_pruned": nodes_pruned,
        "nodes_capped": nodes_capped,
        "elapsed": time.time() - start_time
    }
    return summary, rows

# ---- CLI for quick runs ----
def main():
    p = argparse.ArgumentParser(description="Collab-friendly RFM demo")
    p.add_argument("--mode", choices=["cap-predictive", "strict"], default="cap-predictive")
    p.add_argument("--time", type=float, default=10.0)
    p.add_argument("--items", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--node-sleep", type=float, default=0.0)
    p.add_argument("--out", type=str, default="rfm_collab.csv")
    args = p.parse_args()

    items, C = generate_instance(args.items, seed=args.seed)
    est = TimeEstimator(p90_window=100, ema_alpha=0.2, w_ema=0.6, calibrated_p90=DEFAULT_P90_CAL, pct=90.0)
    summary, rows = rfm_bb(items, C, T=args.time, mode=args.mode, node_sleep=args.node_sleep, estimator=est, out_csv=args.out)

    print("Summary:", summary)
    print("Wrote node log:", args.out)

if __name__ == "__main__":
    main()
