#!/usr/bin/env python3
"""
scripts/run_no_pso_test.py
────────────────────────
Run the No-PSO (GA-only) ablation experiment and compute summary metrics.

This script is a focused test harness for the "no_pso" config in configs/.
It runs the specified seeds, writes JSON archives to results/ablations/, and
prints per-seed and aggregate Hypervolume (HV) and IGD scores using the
proxy Pareto constructed from all completed runs.

Usage:
    python scripts/run_no_pso_test.py           # run seeds 0..4 (default)
    python scripts/run_no_pso_test.py --seeds 0 1 --quick

The optional `--quick` flag reduces the budget to 200 for a fast smoke test.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.proposed_moead_pso import ProposedMoeadPso
from src.analysis.pareto_metrics import archive_to_points, proxy_pareto, get_pareto_front, calc_hv, calc_igd
from src.utils.logger import get_logger, save_archive


DATA_PATH   = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
CONFIG_PATH = PROJECT_ROOT / "configs" / "ablation_t2_no_pso.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablations"


def parse_args():
    p = argparse.ArgumentParser(description="Run No-PSO (GA-only) ablation and report metrics")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)), help="Seeds to run")
    p.add_argument("--hardware", default="edgegpu_latency", help="Hardware metric key")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100","ImageNet16-120"], help="Dataset split")
    p.add_argument("--quick", action="store_true", help="Reduce budget to 200 for quick smoke tests")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _build_eval_fn(api: HWNASApi, metric: str, dataset: str):
    def _eval(arch: np.ndarray):
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=metric)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency
    return _eval


def main():
    args = parse_args()
    log = get_logger("run_no_pso_test", level=logging.DEBUG if args.verbose else logging.INFO)

    if not DATA_PATH.exists():
        log.error("Benchmark file not found: %s", DATA_PATH)
        log.error("Run: python scripts/download_data.py")
        sys.exit(1)

    cfg = json.loads(CONFIG_PATH.read_text())
    if args.quick:
        cfg = dict(cfg)
        cfg["budget"] = 200

    api = HWNASApi(str(DATA_PATH))
    eval_fn = _build_eval_fn(api, args.hardware, args.dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    archives = []
    for seed in args.seeds:
        out_path = RESULTS_DIR / f"{cfg.get('experiment_name','no_pso')}_seed{seed}.json"
        if out_path.exists():
            log.info("[SKIP] seed=%d — result exists: %s", seed, out_path.name)
            payload = json.loads(out_path.read_text())
            archives.append(payload["archive"])
            continue

        log.info("Running seed=%d (budget=%d)", seed, cfg.get("budget"))
        t0 = time.perf_counter()
        opt = ProposedMoeadPso(eval_fn=eval_fn, rng=np.random.default_rng(seed), config=cfg)
        archive = opt.search()
        elapsed = time.perf_counter() - t0

        save_archive(archive, out_path, metadata={
            "experiment_name": cfg.get("experiment_name", "no_pso"),
            "seed": seed,
            "hardware": args.hardware,
            "dataset": args.dataset,
            "budget_spent": opt.budget_spent,
            "config": cfg,
        })

        log.info("Saved %s (evals=%d, pareto=%d, %.2fs)", out_path.name, opt.budget_spent, len(get_pareto_front(archive_to_points(archive))), elapsed)
        archives.append(archive)

    # ── Build proxy Pareto P* and compute per-seed HV / IGD (normalise to P* bounds) ──
    p_star = proxy_pareto(archives)
    pts_min, pts_max = p_star.min(axis=0), p_star.max(axis=0)
    range_ = np.where(pts_max - pts_min > 0, pts_max - pts_min, 1.0)

    def normalise(pts: np.ndarray) -> np.ndarray:
        return (pts - pts_min) / range_

    REF = np.array([1.1, 1.1])

    hv_list, igd_list = [], []
    for arc in archives:
        pts = archive_to_points(arc)
        front = get_pareto_front(pts)
        nf = normalise(front) if len(front) else front
        hv = calc_hv(nf, ref_point=REF)
        igd = calc_igd(nf, normalise(p_star))
        hv_list.append(hv)
        igd_list.append(igd)

    print("\nNo-PSO (GA-only) Summary — seeds:", args.seeds)
    for s, hv, igd in zip(args.seeds, hv_list, igd_list):
        print(f"  seed={s:2d}  HV={hv:.6f}  IGD={igd:.6f}")

    print("\nAggregate:")
    print(f"  HV mean ± std : {np.mean(hv_list):.6f} ± {np.std(hv_list):.6f}")
    print(f"  IGD mean ± std: {np.mean(igd_list):.6f} ± {np.std(igd_list):.6f}")


if __name__ == '__main__':
    main()
