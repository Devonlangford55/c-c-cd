# c-c-cd#!/usr/bin/env python3
"""
demo_rfm_bb_knapsack_large.py

Formalized RFM demo for Branch-and-Bound knapsack with EMA+percentile t_hat,
cap-predictive and strict modes, full per-node logging, fallback safety and a manifest.

Usage:
  python3 demo_rfm_bb_knapsack_large.py --mode cap-predictive --time 30 --instances 2 --items 800 --out rfm_large.csv

See --help for options.
"""
from __future__ import annotations
import argparse, csv, heapq, math, random, time, json, os
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any

Node = namedtuple("Node", ["index","value","weight","bound","taken_mask","depth","remaining_items"])

DEFAULT_CAL_P90 = 0.0001807212829589
DEFAULT_W_EMA = 0.6
DEFAULT_EMA_ALPHA = 0.2
DEFAULT_P90_WINDOW = 200
DEFAULT_T_HAT_P = 90.0
DEFAULT_S_MAX = 200
DEFAULT_THETA_P = 0.20
DEFAULT_THETA_C = 0.60
DEFAULT_TIMEOUT_MULT = 3.0
DEFAULT_BETA = 1.2
DEFAULT_M = 3
DEFAULT_EXPL_FRAC = 0.05

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def fractional_bound(i:int, val:int, wt:int, items:List[Tuple[int,int]], C:int) -> float:
    rem = C - wt
    b = float(val)
    for j in range(i, len(items)):
        w,v = items[j]
        if w <= rem:
            rem -= w; b += v
        else:
            if w>0: b += v * (rem / w)
            break
    return b

def percentile(sorted_list:List[float], p:float) -> float:
    if not sorted_list: return 0.0
    k = (len(sorted_list)-1)*(p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f==c: return sorted_list[int(k)]
    return sorted_list[int(f)]*(c-k) + sorted_list[int(c)]*(k-f)

def generate_instance(n_items:int, capacity_ratio:float=0.5, seed:Optional[int]=None):
    rnd = random.Random(seed)
    items = [(rnd.randint(1,100), rnd.randint(1,200)) for _ in range(n_items)]
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    total_w = sum(w for w,_ in items)
    cap = max(1, int(total_w * capacity_ratio))
    return items, cap

def greedy_seed(items, capacity):
    cap = capacity; total = 0
    for w,v in sorted(items, key=lambda x: x[1]/x[0], reverse=True):
        if w <= cap:
            cap -= w; total += v
    return total

def slack_of(B:float, inc:float) -> float:
    if B <= inc or B <= 0: return 0.0
    return max(0.0, (B - inc) / max(1.0, B))

def time_factor_of(per_node_hat:float, nodes_est:int, time_left:float):
    predicted_total = per_node_hat * max(1, int(nodes_est))
    denom = predicted_total + time_left
    if denom <= 0: return 0.0, predicted_total
    return clamp(time_left / denom, 0.0, 1.0), predicted_total

def compute_rfs(B:float, inc:float, per_node_hat:float, nodes_est:int, time_left:float):
    s = slack_of(B, inc)
    tf, pred = time_factor_of(per_node_hat, nodes_est, time_left)
    return s*tf, s, tf, pred

class RFM_Demo:
    def __init__(self, items:List[Tuple[int,int]], capacity:int, config:Dict[str,Any]):
        self.items = items
        self.capacity = capacity
        self.cfg = config
        self.node_times = deque(maxlen=config.get("p90_window", DEFAULT_P90_WINDOW))
        self.ema_t = config.get("calibrated_p90", DEFAULT_CAL_P90)
        self.per_node_hat = float(self.ema_t)
        self.incumbent = 0
        self.incumbent_mask = None
        self.time_to_first = None
        self.nodes_expanded = 0
        self.nodes_capped = 0
        self.nodes_pruned_by_rfm = 0
        self.top_nodes = []
        self.top_solutions = []
        self.event_rows = []

    def nodes_est(self, node:Node) -> int:
        rem = max(0, len(self.items) - node.index)
        return max(1, min(rem, self.cfg.get("S_max", DEFAULT_S_MAX)))

    def update_t_hat(self, observed:float):
        if observed <= 0: return
        self.node_times.append(observed)
        a = self.cfg.get("ema_alpha", DEFAULT_EMA_ALPHA)
        self.ema_t = a * observed + (1.0 - a) * self.ema_t
        sorted_t = sorted(self.node_times)
        pct = percentile(sorted_t, self.cfg.get("t_hat_percentile", DEFAULT_T_HAT_P)) if sorted_t else self.ema_t
        w = self.cfg.get("w_ema", DEFAULT_W_EMA)
        self.per_node_hat = w * self.ema_t + (1.0 - w) * pct

    def log(self, row:List):
        self.event_rows.append(row)

    def save_events(self, path:str):
        hdr = ["instance_id","timestamp","node_index","depth","orig_bound","effective_bound","ema_t_hat_s","pct_t_hat_s","per_node_hat_s","nodes_est","predicted_total_s","slack","time_factor","rfs","gamma_t","action","mode","incumbent_value","remaining_items"]
        with open(path,"w",newline="") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in self.event_rows: w.writerow(r)

    def run(self, run_id:int, T_total:float, mode:str, seed:Optional[int], node_sleep:float, verify_K:int, writer=None):
        start = time.time()
        exploration_allow = self.cfg.get("exploration_frac", DEFAULT_EXPL_FRAC) * T_total
        g = greedy_seed(self.items, self.capacity)
        if g>0:
            self.incumbent = g; self.time_to_first = time.time()-start
            row = [run_id, time.time(), "GREEDY_SEED", 0, "", "", self.ema_t, self.ema_t, self.per_node_hat, 0, 0.0, 0.0, 0.0, 0.0, "seed", mode, self.incumbent, len(self.items)]
            self.log(row)
            if writer: writer.writerow(row)

        root_bound = fractional_bound(0,0,0,self.items,self.capacity)
        root = Node(0,0,0,root_bound,0,0,len(self.items))
        heap = [(-root.bound, root)]
        consecutive = 0
        current_mode = mode

        while heap:
            elapsed = time.time() - start
            time_left = T_total - elapsed
            if time_left <= 0: break

            # compute W_hat (sample top-of-heap)
            sample_k = min(50, len(heap))
            w_hat = 0.0
            for i in range(sample_k):
                _, n = heap[i]
                est_nodes = self.nodes_est(n)
                est = self.per_node_hat * est_nodes
                if est > w_hat: w_hat = est

            beta = self.cfg.get("beta", DEFAULT_BETA)
            if w_hat > beta * time_left: consecutive += 1
            else: consecutive = 0
            if consecutive >= self.cfg.get("M", DEFAULT_M): current_mode = "strict"

            _, node = heapq.heappop(heap)

            t0 = time.time()
            if node_sleep and node_sleep>0: time.sleep(node_sleep)
            observed = time.time() - t0

            self.update_t_hat(observed)
            pct_hat = percentile(sorted(self.node_times), self.cfg.get("t_hat_percentile", DEFAULT_T_HAT_P)) if self.node_times else self.ema_t

            nodes_est = self.nodes_est(node)
            B = fractional_bound(node.index, node.value, node.weight, self.items, self.capacity)

            rfs, slack, tf, pred = compute_rfs(B, self.incumbent, self.per_node_hat, nodes_est, time_left)
            gamma_t = (time_left / max(1e-9, T_total))

            timeout_mult = self.cfg.get("timeout_multiplier", DEFAULT_TIMEOUT_MULT)
            if self.per_node_hat > 0 and observed > timeout_mult * self.per_node_hat:
                row = [run_id, time.time(), node.index, node.depth, B, "", self.ema_t, pct_hat, self.per_node_hat, nodes_est, pred, slack, tf, rfs, gamma_t, "node_timeout", current_mode, self.incumbent, nodes_est]
                self.log(row); 
                if writer: writer.writerow(row)
                continue

            action="consider"
            if rfs < self.cfg.get("theta_p", DEFAULT_THETA_P):
                action="prune_by_rfm"; self.nodes_pruned_by_rfm += 1
                row = [run_id, time.time(), node.index, node.depth, B, None, self.ema_t, pct_hat, self.per_node_hat, nodes_est, pred, slack, tf, rfs, gamma_t, action, current_mode, self.incumbent, nodes_est]
                self.log(row); 
                if writer: writer.writerow(row)
                continue
            elif rfs < self.cfg.get("theta_c", DEFAULT_THETA_C):
                effective = self.incumbent + (B - self.incumbent) * gamma_t
                self.nodes_capped += 1; action="cap"
            else:
                effective = B

            if effective <= self.incumbent:
                row = [run_id, time.time(), node.index, node.depth, B, effective, self.ema_t, pct_hat, self.per_node_hat, nodes_est, pred, slack, tf, rfs, gamma_t, "prune_by_bound", current_mode, self.incumbent, nodes_est]
                self.log(row)
                if writer: writer.writerow(row)
                continue

            self.nodes_expanded += 1

            if len(self.top_nodes) < verify_K:
                self.top_nodes.append((B, node)); self.top_nodes.sort(reverse=True, key=lambda x: x[0])
            else:
                if B > self.top_nodes[-1][0]:
                    self.top_nodes[-1] = (B, node); self.top_nodes.sort(reverse=True, key=lambda x: x[0])

            if node.index < len(self.items):
                w,v = self.items[node.index]
                if node.weight + w <= self.capacity:
                    child = Node(node.index+1, node.value+v, node.weight+w, node.taken_mask | (1<<node.index), node.depth+1, len(self.items)-(node.index+1))
                    if child.index == len(self.items):
                        if child.value > self.incumbent:
                            self.incumbent = child.value; self.incumbent_mask = child.taken_mask
                            if self.time_to_first is None: self.time_to_first = time.time() - start
                            self.top_solutions.append((self.incumbent, self.incumbent_mask))
                    else:
                        b = fractional_bound(child.index, child.value, child.weight, self.items, self.capacity)
                        heapq.heappush(heap, (-b, child))
                excl = Node(node.index+1, node.value, node.weight, node.taken_mask, node.depth+1, len(self.items)-(node.index+1))
                if excl.index == len(self.items):
                    if excl.value > self.incumbent:
                        self.incumbent = excl.value; self.incumbent_mask = excl.taken_mask
                        if self.time_to_first is None: self.time_to_first = time.time() - start
                        self.top_solutions.append((self.incumbent, self.incumbent_mask))
                else:
                    b = fractional_bound(excl.index, excl.value, excl.weight, self.items, self.capacity)
                    heapq.heappush(heap, (-b, excl))

            row = [run_id, time.time(), node.index, node.depth, B, effective, self.ema_t, pct_hat, self.per_node_hat, nodes_est, pred, slack, tf, rfs, gamma_t, action, current_mode, self.incumbent, nodes_est]
            self.log(row)
            if writer: writer.writerow(row)

        elapsed_total = time.time() - start
        return {"value": self.incumbent, "nodes_expanded": self.nodes_expanded, "nodes_capped": self.nodes_capped, "nodes_pruned_by_rfm": self.nodes_pruned_by_rfm, "ema_t_hat": self.ema_t, "pct_t_hat": percentile(sorted(self.node_times), self.cfg.get("t_hat_percentile", DEFAULT_T_HAT_P)) if self.node_times else self.ema_t, "time_to_first": self.time_to_first if self.time_to_first is not None else -1, "elapsed": elapsed_total}

def verify_node(node, items, capacity, time_limit):
    start = time.time()
    best = node.value if node.index==len(items) else 0
    best_mask = node.taken_mask if node.index==len(items) else None
    stack=[node]
    while stack and (time.time()-start) < time_limit:
        cur = stack.pop()
        if cur.index == len(items):
            if cur.value > best and cur.weight <= capacity:
                best = cur.value; best_mask = cur.taken_mask
            continue
        excl = Node(cur.index+1, cur.value, cur.weight, cur.taken_mask, cur.depth+1, len(items)-(cur.index+1))
        stack.append(excl)
        w,v = items[cur.index]
        if cur.weight + w <= capacity:
            take = Node(cur.index+1, cur.value+v, cur.weight+w, cur.taken_mask | (1<<cur.index), cur.depth+1, len(items)-(cur.index+1))
            stack.append(take)
    return best, best_mask

def run_single(args, config, out_csv):
    random.seed(args.seed)
    summaries=[]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id","timestamp","node_index","depth","orig_bound","effective_bound","ema_t_hat_s","pct_t_hat_s","per_node_hat_s","nodes_est","predicted_total_s","slack","time_factor","rfs","gamma_t","action","mode","incumbent","remaining_items"])
        for i in range(args.instances):
            seed_i = (args.seed + i) if args.seed is not None else None
            items, cap = generate_instance(args.items, capacity_ratio=0.5, seed=seed_i)
            solver = RFM_Demo(items, cap, config)
            print(f"[run] instance={i} items={len(items)} cap={cap} mode={args.mode}")
            summary = solver.run(i, args.time, args.mode, seed_i, args.node_sleep, config.get("verify_K",5), w)
            solver.save_events(out_csv.replace(".csv","_events.csv"))
            summaries.append(summary)
            print(" summary:", summary)
    with open(out_csv.replace(".csv","_summary.json"), "w") as jf:
        json.dump(summaries, jf, indent=2)
    return summaries

def run_sweep(args, config, agg_out):
    prune_vals = [0.10, 0.20] if args.fast else [0.10,0.20,0.30]
    cap_vals = [0.60] if args.fast else [0.50,0.60,0.75]
    S_vals = [100,200] if args.fast else [100,200,400]
    records=[]
    for p in prune_vals:
        for c in cap_vals:
            for s in S_vals:
                cfg = dict(config); cfg["theta_p"]=p; cfg["theta_c"]=c; cfg["S_max"]=s
                out = f"run_pr{int(p*100)}_cap{int(c*100)}_S{s}.csv"
                print("Sweep:",p,c,s)
                summaries = run_single(args, cfg, out)
                bests=[smi["value"] for smi in summaries]
                elaps=[smi["elapsed"] for smi in summaries]
                nodes=[smi["nodes_expanded"] for smi in summaries]
                pruned=[smi["nodes_pruned_by_rfm"] for smi in summaries]
                adherence = sum(1 for e in elaps if e <= args.time)/len(elaps)
                records.append({"theta_p":p,"theta_c":c,"S_max":s,"avg_best":sum(bests)/len(bests),"avg_elapsed":sum(elaps)/len(elaps),"avg_nodes":sum(nodes)/len(nodes),"avg_pruned":sum(pruned)/len(pruned),"adherence":adherence})
    with open(agg_out,"w") as f: json.dump(records,f,indent=2)
    print("Sweep aggregated ->",agg_out)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cap-predictive","strict"], default="cap-predictive")
    p.add_argument("--time", type=float, default=30.0)
    p.add_argument("--instances", type=int, default=2)
    p.add_argument("--items", type=int, default=800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--node-sleep", type=float, default=0.005)
    p.add_argument("--out", type=str, default="rfm_large_run.csv")
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.fast:
        args.items = min(args.items, 400)
        args.instances = min(args.instances, 1)
    config = {
        "w_ema": DEFAULT_W_EMA,
        "ema_alpha": DEFAULT_EMA_ALPHA,
        "p90_window": DEFAULT_P90_WINDOW,
        "t_hat_percentile": DEFAULT_T_HAT_P,
        "exploration_frac": DEFAULT_EXPL_FRAC,
        "timeout_multiplier": DEFAULT_TIMEOUT_MULT,
        "S_max": DEFAULT_S_MAX,
        "theta_p": DEFAULT_THETA_P,
        "theta_c": DEFAULT_THETA_C,
        "beta": DEFAULT_BETA,
        "M": DEFAULT_M,
        "calibrated_p90": DEFAULT_CAL_P90,
        "verify_K": 5
    }
    if args.manifest:
        with open(args.manifest,"w") as mf: json.dump({"args":vars(args),"config":config},mf,indent=2)
    if args.sweep:
        agg_out = args.out.replace(".csv","_agg.json")
        run_sweep(args, config, agg_out)
    else:
        run_single(args, config, args.out)

if __name__ == "__main__":
    main()
