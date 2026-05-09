#!/usr/bin/env python3
"""Ablation: study effect of SA and Restart on final accuracy.

Runs a small grid of configurations and records the best accuracy per seed.
Writes results to `results/ablations/sa_restart_ablation.csv`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.memetic_nas import MemeticNAS
from src.algorithms.operators import GAOperator, PSOOperator, SAOperator


DATA_PATH = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
OUT_DIR = PROJECT_ROOT / "results" / "ablations"


def run_config(seed: int, budget: int, use_restart: bool, sa_params: Any) -> dict:
    rng = np.random.default_rng(seed)
    api = HWNASApi(str(DATA_PATH))

    def eval_fn(a):
        lat = api.query(a, device="nasbench201", dataset="cifar10", metric="edgegpu_latency")
        acc = api.query_accuracy(a, dataset="cifar10")
        return acc, lat

    sa_op = None
    if sa_params is not None:
        sa_op = SAOperator(T0=float(sa_params.get("T0", 1.0)), alpha=float(sa_params.get("alpha", 0.95)))

    opt = MemeticNAS(
        eval_fn=eval_fn,
        budget=budget,
        candidate_ops=[GAOperator(), PSOOperator()],
        sa_op=sa_op,
        K=20,
        T_neighborhood=2,
        use_restart=bool(use_restart),
        rng=rng,
    )

    archive = opt.search()
    # archive: list of dicts with keys 'arch','accuracy','latency'
    if len(archive) == 0:
        best_acc = float("nan")
        best_lat = float("nan")
    else:
        best_row = max(archive, key=lambda r: float(r["accuracy"]))
        best_acc = float(best_row["accuracy"])
        best_lat = float(best_row["latency"])

    return {
        "seed": seed,
        "use_restart": use_restart,
        "sa": sa_params if sa_params is not None else "off",
        "best_accuracy": best_acc,
        "best_latency": best_lat,
        "n_evaluations": opt.budget_spent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "sa_restart_ablation.csv"))
    parser.add_argument("--t0", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="List of SA T0 values to sweep (include none by using --no-sa below)")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="SA alpha (same for all swept T0 values)")
    parser.add_argument("--no-sa", action="store_true", help="Include SA-off in sweep")
    args = parser.parse_args()

    # Build SA list: include explicit T0 values and optionally an SA-off entry
    sa_values = [{"T0": float(t0), "alpha": float(args.alpha)} for t0 in args.t0]
    if args.no_sa:
        sa_values.append(None)

    # Define configs: for each restart option, sweep SA values
    configs = []
    for use_restart in (True, False):
        for sa in sa_values:
            name = f"SA_{'off' if sa is None else ('T0='+str(sa['T0']))}_Restart_{'on' if use_restart else 'off'}"
            configs.append({"name": name, "use_restart": use_restart, "sa": sa})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["config", "seed", "use_restart", "sa_on", "T0", "alpha", "best_accuracy", "best_latency", "n_evaluations"]
    rows = []

    for cfg in configs:
        for seed in args.seeds:
            print(f"Running config={cfg['name']} seed={seed}  (budget={args.budget})")
            res = run_config(seed=seed, budget=args.budget, use_restart=cfg["use_restart"], sa_params=cfg["sa"])
            row = {
                "config": cfg["name"],
                "seed": res["seed"],
                "use_restart": res["use_restart"],
                "sa_on": False if res["sa"] == "off" or res["sa"] is None else True,
                "T0": res["sa"]["T0"] if isinstance(res["sa"], dict) else "",
                "alpha": res["sa"]["alpha"] if isinstance(res["sa"], dict) else "",
                "best_accuracy": res["best_accuracy"],
                "best_latency": res["best_latency"],
                "n_evaluations": res["n_evaluations"],
            }
            rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Ablation complete. Results written to {out_path}")


if __name__ == "__main__":
    main()
