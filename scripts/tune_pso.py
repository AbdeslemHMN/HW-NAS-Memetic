"""
scripts/tune_pso.py
===================
Benchmarks the baseline PSOOperator against 5 improved variants across a
grid of hyperparameters, using a lightweight standalone search loop that
mirrors MemeticNAS logic without requiring the full framework to be running.

HOW IT WORKS
------------
For each (variant, hyperparams) combination:
  - Run the MOEA/D + PSO search for N_SEEDS seeds, BUDGET NFE each
  - Record final Hypervolume and Pareto front size
  - Report mean ± std across seeds

USAGE
-----
    python scripts/tune_pso.py
    python scripts/tune_pso.py --device edgegpu_latency --budget 1000 --seeds 5
    python scripts/tune_pso.py --variant baseline          # test one variant only
    python scripts/tune_pso.py --variant linear_decay

OUTPUT
------
  - Console table: ranked results sorted by mean HV
  - results/pso_tuning/pso_tuning_results.csv
  - results/pso_tuning/pso_tuning_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# ── make src/ importable from scripts/ ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.hw_nas_wrapper import HWNASApi

# ── constants ────────────────────────────────────────────────────────────────
N_OPS      = 5
N_EDGES    = 6
N_ARCHS    = N_OPS ** N_EDGES   # 15 625
DAG_BLOCKS = [[0, 1], [2, 3], [4, 5]]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal self-contained PSO step (no framework dependency)
# ─────────────────────────────────────────────────────────────────────────────

def pso_baseline(current, pbest, gbest, c1, c2, rng):
    r1 = rng.random(N_EDGES)
    r2 = rng.random(N_EDGES)
    R  = rng.random(N_EDGES)
    child = current.copy()
    pull_pbest = R < c1 * r1
    pull_gbest = (~pull_pbest) & (R < c2 * r2)
    child[pull_pbest] = pbest[pull_pbest]
    child[pull_gbest] = gbest[pull_gbest]
    return np.clip(child, 0, N_OPS - 1).astype(np.int64)


def pso_linear_decay(current, pbest, gbest, c1, c2, rng):
    """Same logic as baseline but accepts externally computed c1, c2."""
    return pso_baseline(current, pbest, gbest, c1, c2, rng)


def pso_inertia(current, pbest, gbest, c1, c2, w, rng):
    child = current.copy()
    for j in range(N_EDGES):
        r  = rng.random()
        r1 = rng.random()
        r2 = rng.random()
        if r < w:
            choices = [op for op in range(N_OPS) if op != int(current[j])]
            child[j] = int(rng.choice(choices))
        elif r < w + c1 * r1:
            child[j] = pbest[j]
        elif r < w + c1 * r1 + c2 * r2:
            child[j] = gbest[j]
    return np.clip(child, 0, N_OPS - 1).astype(np.int64)


def pso_blockwise(current, pbest, gbest, c1, c2, rng, block_probs=None):
    child = current.copy()
    n_blocks = len(DAG_BLOCKS)
    if block_probs is None:
        block_idx = int(rng.integers(0, n_blocks))
    else:
        block_idx = int(rng.choice(n_blocks, p=block_probs))
    block = DAG_BLOCKS[block_idx]
    r1, r2 = rng.random(), rng.random()
    if rng.random() < c1 * r1:
        for j in block:
            child[j] = pbest[j]
    elif rng.random() < c2 * r2:
        for j in block:
            child[j] = gbest[j]
    return np.clip(child, 0, N_OPS - 1).astype(np.int64)


def pso_velocity_memory(current, pbest, gbest, c1, c2, p_repeat, velocity, rng):
    child = current.copy()
    for j in range(N_EDGES):
        r1 = rng.random()
        r2 = rng.random()
        rv = rng.random()
        has_mem = velocity[j] >= 0
        if has_mem and rv < p_repeat:
            child[j] = int(velocity[j])
        elif rng.random() < c1 * r1:
            child[j] = pbest[j]
        elif rng.random() < c2 * r2:
            child[j] = gbest[j]
    return np.clip(child, 0, N_OPS - 1).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# MOEA/D helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_vectors(K: int):
    weights = []
    for k in range(K):
        w_acc = 1.0 - k / (K - 1) if K > 1 else 0.5
        weights.append((w_acc, 1.0 - w_acc))
    return weights


def build_neighborhoods(weights, T=2):
    K = len(weights)
    nbrs = []
    for k in range(K):
        dists = sorted(
            [(abs(weights[k][0] - weights[k2][0]), k2) for k2 in range(K) if k2 != k]
        )
        nbrs.append([idx for _, idx in dists[:T]])
    return nbrs


def scalar(acc, lat, w_acc, w_lat, acc_norm=94.0, lat_norm=30.0):
    return w_acc * (acc / acc_norm) - w_lat * (lat / lat_norm)


def dominates(a, b):
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])


def pareto_front(archive):
    front = []
    for i, a in enumerate(archive):
        if not any(
            dominates((b['acc'], b['lat']), (a['acc'], a['lat']))
            for j, b in enumerate(archive) if j != i
        ):
            front.append(a)
    return front


def hypervolume_2d(front, ref_acc=50.0, ref_lat=35.0):
    if not front:
        return 0.0
    pts = sorted(front, key=lambda x: -x['acc'])
    hv, prev_lat = 0.0, ref_lat
    for p in pts:
        w = p['acc'] - ref_acc
        h = prev_lat - p['lat']
        if w > 0 and h > 0:
            hv += w * h
        prev_lat = min(prev_lat, p['lat'])
    return hv


# ─────────────────────────────────────────────────────────────────────────────
# Core search loop (standalone, no framework)
# ─────────────────────────────────────────────────────────────────────────────

def _proxy_acc(arch) -> float:
    """Structural accuracy proxy matching hw_nas_wrapper's default mode."""
    n_conv = int(sum(1 for op in arch if op in (3, 4)))   # nor_conv_1x1=3, nor_conv_3x3=4
    n_skip = int(sum(1 for op in arch if op == 1))         # skip_connect=1
    n_none = int(sum(1 for op in arch if op == 0))         # none=0
    base   = 70.0 + 4.0 * n_conv + 1.5 * n_skip - 3.0 * n_none
    return float(min(max(base, 50.0), 94.0))


def run_search(api, dataset, metric, pso_fn, pso_kwargs, budget, K, seed):
    """
    Minimal MOEA/D + GA + PSO loop.
    Returns (hypervolume, pareto_size, archive_size).

    pso_fn    : callable(current, pbest, gbest, **pso_kwargs, rng) -> np.ndarray
    pso_kwargs: dict of extra keyword args specific to each variant
    """
    rng = np.random.default_rng(seed)
    weights  = build_weight_vectors(K)
    nbrs     = build_neighborhoods(weights, T=2)
    N        = 20    # pop per direction
    T0, alpha, T_min = 1.0, 0.95, 0.01
    pm       = 1.0 / N_EDGES

    # ── init population ──
    pop    = rng.integers(0, N_OPS, size=(K, N, N_EDGES))
    pbest  = pop.copy()
    gbest  = np.zeros((K, N_EDGES), dtype=np.int64)
    scores       = np.full((K, N), -np.inf)
    pbest_scores = scores.copy()
    gbest_scores = np.full(K, -np.inf)
    accs         = np.zeros((K, N))
    lats         = np.zeros((K, N))
    nfe    = 0
    archive = []
    archive_set = set()

    def eval_arch(arch):
        nonlocal nfe
        nfe += 1
        lat = float(api.query(arch, device='nasbench201', dataset=dataset, metric=metric))
        # accuracy via wrapper's own method; falls back to structural proxy if NAS-Bench-201 not loaded
        try:
            acc = float(api.query_accuracy(arch, dataset=dataset))
        except AttributeError:
            # wrapper doesn't expose query_accuracy — use structural proxy directly
            arch_t = tuple(int(x) for x in arch)
            idx    = api._nasbench201_arch_to_index[arch_t]
            acc    = float(api.accuracy_cache[idx]) if hasattr(api, 'accuracy_cache') else _proxy_acc(arch)
        return acc, lat

    def add_archive(arch, acc, lat):
        key = tuple(arch)
        if key not in archive_set:
            archive_set.add(key)
            archive.append({'acc': acc, 'lat': lat})

    # initialise
    for k in range(K):
        w_acc, w_lat = weights[k]
        for i in range(N):
            acc, lat = eval_arch(pop[k, i])
            s = scalar(acc, lat, w_acc, w_lat)
            scores[k, i] = s
            accs[k, i]   = acc
            lats[k, i]   = lat
            add_archive(pop[k, i], acc, lat)
        best_i = int(np.argmax(scores[k]))
        gbest[k]        = pop[k, best_i].copy()
        gbest_scores[k] = scores[k, best_i]

    T = T0

    # ── special state for stateful variants ──
    velocity      = np.full((K, N_EDGES), -1, dtype=np.int64)
    accept_hist   = np.zeros((K, 20), dtype=np.int8)
    accept_ptr    = np.zeros(K, dtype=int)
    accept_fill   = np.zeros(K, dtype=int)
    c1_adaptive   = np.full(K, pso_kwargs.get('c1', 0.5))
    c2_adaptive   = np.full(K, pso_kwargs.get('c2', 0.5))
    decay_t       = 0

    # ── main loop ──
    while nfe < budget:
        for k in range(K):
            w_acc, w_lat = weights[k]
            for i in range(N):
                if nfe >= budget:
                    break

                cur = pop[k, i].copy()

                # build pso kwargs for this step
                if pso_fn.__name__ == 'pso_linear_decay':
                    p = min(decay_t / budget, 1.0)
                    c1 = pso_kwargs['c1_start'] + p * (pso_kwargs['c1_end'] - pso_kwargs['c1_start'])
                    c2 = pso_kwargs['c2_start'] + p * (pso_kwargs['c2_end'] - pso_kwargs['c2_start'])
                    cand = pso_fn(cur, pbest[k, i], gbest[k], c1, c2, rng)
                elif pso_fn.__name__ == 'pso_inertia':
                    cand = pso_fn(cur, pbest[k, i], gbest[k],
                                  pso_kwargs['c1'], pso_kwargs['c2'],
                                  pso_kwargs['w'], rng)
                elif pso_fn.__name__ == 'pso_blockwise':
                    cand = pso_fn(cur, pbest[k, i], gbest[k],
                                  pso_kwargs['c1'], pso_kwargs['c2'], rng)
                elif pso_fn.__name__ == 'pso_velocity_memory':
                    cand = pso_fn(cur, pbest[k, i], gbest[k],
                                  pso_kwargs['c1'], pso_kwargs['c2'],
                                  pso_kwargs['p_repeat'], velocity[k], rng)
                elif pso_fn.__name__ == 'pso_adaptive':
                    cand = pso_baseline(cur, pbest[k, i], gbest[k],
                                        float(c1_adaptive[k]), float(c2_adaptive[k]), rng)
                elif pso_fn.__name__ == 'pso_adaptive_velocity':
                    # combined: adaptive c1/c2 + velocity memory bias
                    _c1 = float(c1_adaptive[k])
                    _c2 = float(c2_adaptive[k])
                    _p  = pso_kwargs.get('p_repeat', 0.3)
                    cand_base = pso_baseline(cur, pbest[k, i], gbest[k], _c1, _c2, rng)
                    # velocity memory override: replace edges with proven moves
                    cand = cand_base.copy()
                    for _j in range(N_EDGES):
                        if velocity[k, _j] >= 0 and rng.random() < _p:
                            cand[_j] = int(velocity[k, _j])
                else:  # baseline
                    cand = pso_fn(cur, pbest[k, i], gbest[k],
                                  pso_kwargs['c1'], pso_kwargs['c2'], rng)

                acc_c, lat_c = eval_arch(cand)
                s_c = scalar(acc_c, lat_c, w_acc, w_lat)
                add_archive(cand, acc_c, lat_c)
                decay_t += 1

                # acceptance
                delta = s_c - scores[k, i]
                if delta > 0 or (T > T_min and rng.random() < np.exp(delta / T)):
                    pop[k, i]    = cand
                    scores[k, i] = s_c
                    accs[k, i]   = acc_c
                    lats[k, i]   = lat_c
                    accepted = True
                    improved = delta > 0

                    # velocity memory update
                    if pso_fn.__name__ in ('pso_velocity_memory', 'pso_adaptive_velocity') and improved:
                        for j in range(N_EDGES):
                            if cand[j] != cur[j]:
                                velocity[k, j] = int(cand[j])
                else:
                    accepted = False
                    improved = False

                # adaptive c1/c2 update
                if pso_fn.__name__ in ('pso_adaptive', 'pso_adaptive_velocity'):
                    p2 = accept_ptr[k]
                    accept_hist[k, p2] = int(accepted)
                    accept_ptr[k]  = (p2 + 1) % 20
                    accept_fill[k] = min(accept_fill[k] + 1, 20)
                    n_hist = accept_fill[k]
                    if n_hist >= 5:
                        rate = float(accept_hist[k].sum()) / n_hist
                        eta  = pso_kwargs.get('eta', 0.05)
                        c_min, c_max = 0.05, 0.95
                        if rate > 0.2:
                            c2_adaptive[k] = min(c2_adaptive[k] + eta, c_max)
                            c1_adaptive[k] = max(c1_adaptive[k] - eta, c_min)
                        else:
                            c1_adaptive[k] = min(c1_adaptive[k] + eta, c_max)
                            c2_adaptive[k] = max(c2_adaptive[k] - eta, c_min)

                # pbest / gbest update
                if s_c > pbest_scores[k, i]:
                    pbest[k, i]       = cand.copy()
                    pbest_scores[k, i] = s_c

                if s_c > gbest_scores[k]:
                    gbest[k]        = cand.copy()
                    gbest_scores[k] = s_c
                    # propagate to neighbours
                    for k2 in nbrs[k]:
                        w2a, w2l = weights[k2]
                        s_in_k2 = scalar(acc_c, lat_c, w2a, w2l)
                        worst_i = int(np.argmin(scores[k2]))
                        if s_in_k2 > scores[k2, worst_i]:
                            pop[k2, worst_i]    = cand.copy()
                            scores[k2, worst_i] = s_in_k2
                            accs[k2, worst_i]   = acc_c
                            lats[k2, worst_i]   = lat_c

        T = max(T * alpha, T_min)

    front = pareto_front(archive)
    hv    = hypervolume_2d(front)
    return hv, len(front), len(archive)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_experiments(budget: int) -> list[dict]:
    """
    Returns all (variant_name, pso_fn, pso_kwargs) combinations to test.
    Add or remove entries here to customise the sweep.
    """
    return [
        # ── Baseline ──────────────────────────────────────────────────────
        dict(name="baseline_c0.5_c0.5",   fn=pso_baseline,
             kwargs=dict(c1=0.5, c2=0.5)),
        dict(name="baseline_c0.3_c0.7",   fn=pso_baseline,
             kwargs=dict(c1=0.3, c2=0.7)),
        dict(name="baseline_c0.7_c0.3",   fn=pso_baseline,
             kwargs=dict(c1=0.7, c2=0.3)),

        # ── Linear Decay ──────────────────────────────────────────────────
        dict(name="linear_decay_0.8→0.1_0.1→0.8",  fn=pso_linear_decay,
             kwargs=dict(c1_start=0.8, c1_end=0.1, c2_start=0.1, c2_end=0.8)),
        dict(name="linear_decay_0.6→0.2_0.2→0.6",  fn=pso_linear_decay,
             kwargs=dict(c1_start=0.6, c1_end=0.2, c2_start=0.2, c2_end=0.6)),
        dict(name="linear_decay_symmetric",          fn=pso_linear_decay,
             kwargs=dict(c1_start=0.7, c1_end=0.3, c2_start=0.3, c2_end=0.7)),

        # ── Inertia ───────────────────────────────────────────────────────
        dict(name="inertia_w=1/6_c0.4_c0.4",  fn=pso_inertia,
             kwargs=dict(c1=0.4, c2=0.4, w=1/6)),
        dict(name="inertia_w=1/12_c0.4_c0.4", fn=pso_inertia,
             kwargs=dict(c1=0.4, c2=0.4, w=1/12)),
        dict(name="inertia_w=1/6_c0.3_c0.5",  fn=pso_inertia,
             kwargs=dict(c1=0.3, c2=0.5, w=1/6)),

        # ── Adaptive ──────────────────────────────────────────────────────
        # eta sweep (c1=c2=0.5 fixed)
        dict(name="adaptive_eta0.05_c0.5_c0.5",    fn=pso_adaptive,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.05)),
        dict(name="adaptive_eta0.10_c0.5_c0.5 ★",  fn=pso_adaptive,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.10)),
        dict(name="adaptive_eta0.15_c0.5_c0.5",    fn=pso_adaptive,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.15)),
        dict(name="adaptive_eta0.20_c0.5_c0.5",    fn=pso_adaptive,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.20)),
        # c1/c2 init sweep (eta=0.10 fixed — best eta)
        dict(name="adaptive_eta0.10_c0.3_c0.7",    fn=pso_adaptive,
             kwargs=dict(c1=0.3, c2=0.7, eta=0.10)),
        dict(name="adaptive_eta0.10_c0.7_c0.3",    fn=pso_adaptive,
             kwargs=dict(c1=0.7, c2=0.3, eta=0.10)),
        dict(name="adaptive_eta0.10_c0.2_c0.8",    fn=pso_adaptive,
             kwargs=dict(c1=0.2, c2=0.8, eta=0.10)),
        dict(name="adaptive_eta0.10_c0.8_c0.2",    fn=pso_adaptive,
             kwargs=dict(c1=0.8, c2=0.2, eta=0.10)),
        dict(name="adaptive_eta0.10_c0.4_c0.6",    fn=pso_adaptive,
             kwargs=dict(c1=0.4, c2=0.6, eta=0.10)),
        dict(name="adaptive_eta0.10_c0.6_c0.4",    fn=pso_adaptive,
             kwargs=dict(c1=0.6, c2=0.4, eta=0.10)),

        # ── Block-Wise ────────────────────────────────────────────────────
        dict(name="blockwise_c0.5_c0.5",   fn=pso_blockwise,
             kwargs=dict(c1=0.5, c2=0.5)),
        dict(name="blockwise_c0.7_c0.7",   fn=pso_blockwise,
             kwargs=dict(c1=0.7, c2=0.7)),
        dict(name="blockwise_c0.3_c0.7",   fn=pso_blockwise,
             kwargs=dict(c1=0.3, c2=0.7)),

        # ── Velocity Memory ───────────────────────────────────────────────
        dict(name="velocity_mem_p0.3_c0.4_c0.4", fn=pso_velocity_memory,
             kwargs=dict(c1=0.4, c2=0.4, p_repeat=0.3)),
        dict(name="velocity_mem_p0.2_c0.5_c0.5", fn=pso_velocity_memory,
             kwargs=dict(c1=0.5, c2=0.5, p_repeat=0.2)),
        dict(name="velocity_mem_p0.4_c0.3_c0.5", fn=pso_velocity_memory,
             kwargs=dict(c1=0.3, c2=0.5, p_repeat=0.4)),

        # ── Adaptive + Velocity Memory (combined) ─────────────────────────
        dict(name="adaptive_velocity_eta0.10_p0.3",  fn=pso_adaptive_velocity,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.10, p_repeat=0.3)),
        dict(name="adaptive_velocity_eta0.10_p0.4",  fn=pso_adaptive_velocity,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.10, p_repeat=0.4)),
        dict(name="adaptive_velocity_eta0.05_p0.3",  fn=pso_adaptive_velocity,
             kwargs=dict(c1=0.5, c2=0.5, eta=0.05, p_repeat=0.3)),
        dict(name="adaptive_velocity_eta0.10_p0.2",  fn=pso_adaptive_velocity,
             kwargs=dict(c1=0.3, c2=0.7, eta=0.10, p_repeat=0.2)),
    ]


# dummy pso_adaptive function name tag (dispatched inside run_search)
def pso_adaptive(current, pbest, gbest, c1, c2, eta, rng):
    pass  # dispatched by name in run_search — never called directly


# dummy pso_adaptive_velocity function name tag (dispatched inside run_search)
def pso_adaptive_velocity(current, pbest, gbest, c1, c2, eta, p_repeat, rng):
    pass  # dispatched by name in run_search — never called directly



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PSO variant tuning script")
    parser.add_argument("--metric",  default="edgegpu_latency",
                        help="HW-NAS-Bench metric key (e.g. edgegpu_latency, raspi4_latency)")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset (cifar10, cifar100, ImageNet16-120)")
    parser.add_argument("--budget",  type=int, default=1000,
                        help="NFE budget per run (default 1000 for fast sweep)")
    parser.add_argument("--seeds",   type=int, default=5,
                        help="Independent seeds per variant (default 5)")
    parser.add_argument("--k",       type=int, default=10,
                        help="Number of MOEA/D weight directions")
    parser.add_argument("--variant", default=None,
                        help="Filter: only run experiments whose name contains this string")
    args = parser.parse_args()

    # ── Load benchmark ────────────────────────────────────────────────────────
    pickle_path = ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
    if not pickle_path.exists():
        print(f"[ERROR] Pickle not found: {pickle_path}")
        sys.exit(1)

    print(f"Loading benchmark from {pickle_path} ...")
    api = HWNASApi(str(pickle_path))
    print(f"Benchmark loaded. Metric: {args.metric} | Dataset: {args.dataset}\n")

    # ── Filter experiments ────────────────────────────────────────────────────
    experiments = get_experiments(args.budget)
    if args.variant:
        experiments = [e for e in experiments if args.variant in e['name']]
        if not experiments:
            print(f"[ERROR] No experiments match filter '{args.variant}'")
            print(f"Available: {[e['name'] for e in get_experiments(args.budget)]}")
            sys.exit(1)

    print(f"Running {len(experiments)} variants × {args.seeds} seeds "
          f"× {args.budget} NFE = "
          f"{len(experiments) * args.seeds * args.budget:,} total evaluations\n")

    seeds = list(range(args.seeds))

    # ── Run experiments ───────────────────────────────────────────────────────
    results = []
    for exp_i, exp in enumerate(experiments):
        name   = exp['name']
        fn     = exp['fn']
        kwargs = exp['kwargs']
        hvs    = []
        sizes  = []

        t0 = time.time()
        for seed in seeds:
            hv, psize, asize = run_search(
                api      = api,
                dataset  = args.dataset,
                metric   = args.metric,
                pso_fn   = fn,
                pso_kwargs = kwargs,
                budget   = args.budget,
                K        = args.k,
                seed     = seed,
            )
            hvs.append(hv)
            sizes.append(psize)

        elapsed = time.time() - t0
        hv_mean = float(np.mean(hvs))
        hv_std  = float(np.std(hvs))
        sz_mean = float(np.mean(sizes))

        results.append({
            'rank':    0,
            'name':    name,
            'hv_mean': hv_mean,
            'hv_std':  hv_std,
            'hv_all':  hvs,
            'pf_mean': sz_mean,
            'params':  kwargs,
            'time_s':  round(elapsed, 2),
        })

        print(f"[{exp_i+1:2d}/{len(experiments)}] {name:<45} "
              f"HV={hv_mean:8.2f} ± {hv_std:6.2f}  "
              f"PF={sz_mean:.1f}  t={elapsed:.1f}s")

    # ── Rank by mean HV ───────────────────────────────────────────────────────
    results.sort(key=lambda r: -r['hv_mean'])
    for i, r in enumerate(results):
        r['rank'] = i + 1

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'RANK':<5} {'VARIANT':<45} {'HV mean':>10} {'HV std':>8} {'PF size':>8}")
    print("=" * 75)
    for r in results:
        marker = " ← BEST" if r['rank'] == 1 else ""
        print(f"{r['rank']:<5} {r['name']:<45} "
              f"{r['hv_mean']:>10.2f} {r['hv_std']:>8.2f} "
              f"{r['pf_mean']:>8.1f}{marker}")
    print("=" * 75)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = ROOT / "results" / "pso_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full detail)
    json_path = out_dir / "pso_tuning_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # CSV (summary)
    csv_path = out_dir / "pso_tuning_results.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("rank,name,hv_mean,hv_std,pf_mean,time_s\n")
        for r in results:
            f.write(f"{r['rank']},{r['name']},{r['hv_mean']:.4f},"
                    f"{r['hv_std']:.4f},{r['pf_mean']:.2f},{r['time_s']}\n")

    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"\nBest variant: {results[0]['name']}")
    print(f"  HV = {results[0]['hv_mean']:.2f} ± {results[0]['hv_std']:.2f}")
    print(f"  Params: {results[0]['params']}")


if __name__ == "__main__":
    main()