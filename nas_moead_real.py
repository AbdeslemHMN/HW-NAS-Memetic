r"""
nas_moead_real.py
==================
MOEA/D + Discrete GA + Discrete PSO on REAL HW-NAS-Bench data.

SETUP (do once):
    Step 1 — HW-NAS-Bench pickle:
        .\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle

    Step 2 — NAS-Bench-201 file:
        Download: https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_
        File: NAS-Bench-201-v1_1-096897.pth
        Place it in your project folder.

    Step 3 — Install NAS-Bench-201 API:
        pip install nas-bench-201
        -- OR --
        git clone https://github.com/D-X-Y/NAS-Bench-201.git
        cd NAS-Bench-201 && pip install -e .

RUN:
    python nas_moead_real.py
"""

import sys
import os
import random
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_moead import (
    BenchmarkLookup,
    MOEAD_GA_PSO,
    hypervolume_2d,
    print_pareto_front,
    select_for_device,
)
from load_hw_nas_bench import load_real_bench

# =============================================================================
# CONFIGURATION — edit these paths
# =============================================================================

HW_PICKLE_PATH = r".\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle"
NB201_PATH = r".\NAS-Bench-201-v1_1-096897.pth"

TARGET_DEVICE = 'edgegpu'
DATASET = 'cifar10'
COST_TYPE = 'latency'
BUDGET = 2000

# =============================================================================
# ALGORITHM HYPERPARAMETERS
# =============================================================================

CONFIG = dict(
    K=5,
    N=20,
    T_neighbors=2,
    pm=1 / 6,
    pc=0.9,
    c1=1.5,
    c2=1.5,
    T0=1.0,
    alpha=0.99,
    T_min=0.01,
    N_stagnation=50,
    budget_max=BUDGET,
    verbose=True,
)

LATENCY_LIMITS = {
    'edgegpu': [5.0, 10.0, 20.0],
    'raspi4':  [50.0, 100.0, 200.0],
    'eyeriss': [1.0, 2.0, 5.0],
    'fpga':    [0.5, 1.0, 2.0],
    'pixel3':  [20.0, 50.0, 100.0],
}

# =============================================================================
# THEORETICAL PARETO COMPARISON
# =============================================================================

def exact_pareto_front_from_table(real_data):
    """
    Build the exact Pareto front from the full benchmark table.
    Objective:
        maximize accuracy
        minimize latency
    """
    sols = [
        {"arch": arch, "accuracy": acc, "latency": lat}
        for arch, (acc, lat) in real_data.items()
    ]

    # Sort by latency ascending, then accuracy descending
    sols.sort(key=lambda x: (x["latency"], -x["accuracy"]))

    front = []
    best_acc = -float("inf")
    eps = 1e-12

    for s in sols:
        if s["accuracy"] > best_acc + eps:
            front.append(s)
            best_acc = s["accuracy"]

    return front


def _normalized_hv(front, acc_min, acc_max, lat_min, lat_max):
    """
    Normalized 2D hypervolume proxy in [0,1] x [0,1].
    Converts:
        accuracy -> maximize
        latency  -> maximize inverse latency
    """
    if not front:
        return 0.0

    eps = 1e-12
    pts = []
    for p in front:
        x = (p["accuracy"] - acc_min) / (acc_max - acc_min + eps)
        y = (lat_max - p["latency"]) / (lat_max - lat_min + eps)
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        pts.append((x, y))

    pts.sort(key=lambda t: t[0])
    hv = 0.0
    prev_x = 0.0
    for x, y in pts:
        if x > prev_x:
            hv += (x - prev_x) * y
            prev_x = x
    return float(hv)


def compare_with_theoretical(theoretical_front, algo_front, real_data, device, dataset):
    """
    Compare the algorithm front against the exact Pareto front.
    Prints:
        - size
        - overlap
        - missed points
        - extra points
        - hypervolume gap
        - best accuracy under latency limits
    """
    theory_set = {tuple(p["arch"]) for p in theoretical_front}
    algo_set = {tuple(p["arch"]) for p in algo_front}

    common = theory_set & algo_set
    missed = theory_set - algo_set
    extra = algo_set - theory_set

    accs = [v[0] for v in real_data.values()]
    lats = [v[1] for v in real_data.values()]
    acc_min, acc_max = min(accs), max(accs)
    lat_min, lat_max = min(lats), max(lats)

    hv_theory = _normalized_hv(theoretical_front, acc_min, acc_max, lat_min, lat_max)
    hv_algo = _normalized_hv(algo_front, acc_min, acc_max, lat_min, lat_max)

    print("\n" + "=" * 70)
    print("ALGO FRONT vs THEORETICAL PARETO FRONT")
    print("=" * 70)
    print(f"Device                  : {device}")
    print(f"Dataset                 : {dataset}")
    print(f"Theoretical Pareto size : {len(theoretical_front)}")
    print(f"Algorithm front size    : {len(algo_front)}")
    print(f"Exact overlap           : {len(common)}")
    print(f"Missed Pareto points    : {len(missed)}")
    print(f"Extra algorithm points  : {len(extra)}")
    print(f"Theoretical HV          : {hv_theory:.6f}")
    print(f"Algorithm HV            : {hv_algo:.6f}")
    print(f"HV gap                  : {hv_theory - hv_algo:.6f}")

    print("\nBest accuracy under latency limits:")
    for limit in LATENCY_LIMITS.get(device, [5.0, 10.0, 20.0]):
        best_theory = select_for_device(theoretical_front, latency_limit=limit)
        best_algo = select_for_device(algo_front, latency_limit=limit)

        if best_theory is None:
            print(f"  <= {limit:6.1f} ms : no theoretical point")
            continue

        if best_algo is None:
            print(f"  <= {limit:6.1f} ms : theory={best_theory['accuracy']:.4f}% | yours=None")
        else:
            gap = best_theory["accuracy"] - best_algo["accuracy"]
            print(
                f"  <= {limit:6.1f} ms : theory={best_theory['accuracy']:.4f}% | "
                f"yours={best_algo['accuracy']:.4f}% | gap={gap:.4f}%"
            )

    return theoretical_front


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    print(f"\n{'='*60}")
    print(f"NAS Search — MOEA/D + GA + Discrete PSO")
    print(f"  HW pickle : {HW_PICKLE_PATH}")
    print(f"  NB201     : {NB201_PATH}")
    print(f"  Device    : {TARGET_DEVICE}")
    print(f"  Dataset   : {DATASET}")
    print(f"  Budget    : {BUDGET} lookups")
    print(f"{'='*60}\n")

    if not os.path.exists(HW_PICKLE_PATH):
        print(f"[ERROR] HW-NAS-Bench pickle not found: {HW_PICKLE_PATH}")
        sys.exit(1)

    if not os.path.exists(NB201_PATH):
        print(f"[ERROR] NAS-Bench-201 file not found: {NB201_PATH}")
        sys.exit(1)

    # ── Step 1: Load real data ───────────────────────────────────────────────
    real_data = load_real_bench(
        pickle_path=HW_PICKLE_PATH,
        nb201_path=NB201_PATH,
        device=TARGET_DEVICE,
        dataset=DATASET,
        cost_type=COST_TYPE,
        verbose=True,
    )

    if not real_data:
        print("[ERROR] No data loaded.")
        sys.exit(1)

    # Exact theoretical front from full table
    theoretical_front = exact_pareto_front_from_table(real_data)

    # ── Step 2: Run MOEA/D + GA + PSO ────────────────────────────────────────
    bench = BenchmarkLookup(data=real_data, device=TARGET_DEVICE)
    algo = MOEAD_GA_PSO(bench=bench, **CONFIG)

    start = time.time()
    pareto_front = algo.run()
    elapsed = time.time() - start

    # ── Step 3: Results ───────────────────────────────────────────────────────
    print(f"\nSearch time: {elapsed:.2f}s")
    print(f"\n{'='*60}")
    print(f"FINAL PARETO FRONT — {TARGET_DEVICE} | {DATASET}")
    print_pareto_front(pareto_front, top_n=20)

    hv = hypervolume_2d(pareto_front)
    print(f"\nHypervolume          : {hv:.4f}")
    print(f"Architectures queried: {algo.lookup_count}")
    print(f"Unique in archive    : {len(algo.archive)}")

    # Compare against theoretical Pareto front
    compare_with_theoretical(theoretical_front, pareto_front, real_data, TARGET_DEVICE, DATASET)

    # ── Step 4: Deployment selection ─────────────────────────────────────────
    limits = LATENCY_LIMITS.get(TARGET_DEVICE, [10.0, 50.0])
    print(f"\n── Deployment Selection ({TARGET_DEVICE}) ──")
    for limit in limits:
        best = select_for_device(pareto_front, latency_limit=limit)
        if best:
            print(
                f"  Latency <= {limit:6.1f} ms  →  arch={best['arch']}"
                f"  acc={best['accuracy']:.2f}%  lat={best['latency']:.4f} ms"
            )

    # ── Step 5: Save results ──────────────────────────────────────────────────
    out_file = f"pareto_{TARGET_DEVICE}_{DATASET}_real.txt"
    _save_results(pareto_front, algo, out_file, theoretical_front, real_data)
    print(f"\nResults saved to: {out_file}")

    return pareto_front


def _save_results(pareto, algo, filename, theoretical_front, real_data):
    accs = [v[0] for v in real_data.values()]
    lats = [v[1] for v in real_data.values()]
    acc_min, acc_max = min(accs), max(accs)
    lat_min, lat_max = min(lats), max(lats)

    hv_theory = _normalized_hv(theoretical_front, acc_min, acc_max, lat_min, lat_max)
    hv_algo = _normalized_hv(pareto, acc_min, acc_max, lat_min, lat_max)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("MOEA/D + GA + Discrete PSO — NAS Results\n")
        f.write(f"Device          : {TARGET_DEVICE}\n")
        f.write(f"Dataset         : {DATASET}\n")
        f.write("Accuracy source : real NAS-Bench-201\n")
        f.write(f"Budget used     : {algo.lookup_count} lookups\n")
        f.write(f"Archive size    : {len(algo.archive)} unique architectures\n")
        f.write(f"Pareto size     : {len(pareto)}\n")
        f.write(f"Theoretical Pareto size : {len(theoretical_front)}\n")
        f.write(f"Theoretical HV   : {hv_theory:.6f}\n")
        f.write(f"Algorithm HV     : {hv_algo:.6f}\n")
        f.write(f"HV gap           : {hv_theory - hv_algo:.6f}\n\n")
        f.write(f"{'#':<4}  {'Architecture':<25}  {'Accuracy':>10}  {'Latency':>12}\n")
        f.write("─" * 60 + "\n")
        for idx, sol in enumerate(pareto):
            f.write(
                f"{idx+1:<4}  {str(sol['arch']):<25}  "
                f"{sol['accuracy']:>9.4f}%  {sol['latency']:>10.4f} ms\n"
            )


if __name__ == '__main__':
    main()