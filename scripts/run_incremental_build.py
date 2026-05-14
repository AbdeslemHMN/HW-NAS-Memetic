#!/usr/bin/env python3
"""
Incremental build script — four stages on the same seed.

Demonstrates the additive contribution of each algorithmic component by running
four configurations that progressively add capabilities:

  Stage 1 — Random baseline
  Stage 2 — MOEA/D + GA mutation
  Stage 3 — MOEA/D + GA + PSO
  Stage 4 — MOEA/D + GA + PSO + SA acceptance (full proposal)

All stages use the same seed, budget, hardware target and dataset, making the
Pareto fronts directly comparable in the waterfall visualisation notebook
(notebooks/05_waterfall_buildup.ipynb).

Usage
-----
    python scripts/run_incremental_build.py [--seed 42] [--hardware edgegpu_latency]

Outputs
-------
    results/incremental/stage1_random.json
    results/incremental/stage2_moead_ga.json
    results/incremental/stage3_moead_ga_pso.json
    results/incremental/stage4_full.json
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.random_search import RandomSearch
from src.algorithms.memetic_nas import MemeticNAS
from src.algorithms.operators import GAOperator, PSOOperator, SAOperator
from src.utils.logger import get_logger, save_archive

DATA_PATH    = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
RESULTS_DIR  = PROJECT_ROOT / "results" / "incremental"
TOTAL_BUDGET = 2000
HARDWARE     = "edgegpu_latency"
DATASET      = "cifar10"
K            = 5      # calibrated best (was 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four incremental build stages on a single seed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared by all four stages.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=HARDWARE,
        choices=[
            "edgegpu_latency", "edgegpu_energy",
            "edgetpu_latency",
            "eyeriss_latency", "eyeriss_energy", "eyeriss_arithmetic_intensity",
            "fpga_latency", "fpga_energy",
            "pixel3_latency",
            "raspi4_latency",
        ],
        help="Hardware latency metric key (e.g. edgegpu_latency, raspi4_latency).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="NAS-Bench-201 dataset split for accuracy lookup.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=TOTAL_BUDGET,
        help="NFE budget applied identically to all four stages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def build_eval_fn(api: HWNASApi, hardware: str, dataset: str):
    """Return an (accuracy, latency) evaluator backed by HWNASApi."""
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=hardware)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency

    return _eval


def run_stage(
    name: str,
    optimizer,
    budget: int,
    seed: int,
    hardware: str,
    dataset: str,
    extra_meta: dict,
    out_path: Path,
    log,
) -> None:
    """Run one search stage, assert exact budget consumption, and save results."""
    log.info("  Running %s (seed=%d, budget=%d)…", name, seed, budget)
    archive = optimizer.search()

    metadata = {
        "algorithm":       name,
        "seed":            seed,
        "budget":          budget,
        "n_evaluations":   optimizer.budget_spent,
        "budget_spent":    optimizer.budget_spent,
        "nfe_checkpoints": optimizer.nfe_checkpoints,
        "hardware":        hardware,
        "dataset":         dataset,
        **extra_meta,
    }

    assert metadata["n_evaluations"] == budget, (
        f"[{name} seed={seed}] Evaluation budget mismatch! "
        f"Got {metadata['n_evaluations']}, expected {budget}"
    )

    save_archive(archive, out_path, metadata=metadata)
    log.info("  %s complete — NFE=%d → %s", name, optimizer.budget_spent, out_path)


def main() -> None:
    args = parse_args()
    log = get_logger(
        "run_incremental_build",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    api     = HWNASApi(str(DATA_PATH))
    eval_fn = build_eval_fn(api, args.hardware, args.dataset)

    out_dir = RESULTS_DIR / args.dataset / args.hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    log.info(
        "Incremental build — seed=%d  hardware=%s  budget=%d",
        args.seed, args.hardware, args.budget,
    )

    # ── Stage 1: Random baseline ──────────────────────────────────────────────
    run_stage(
        name="stage1_random",
        optimizer=RandomSearch(
            eval_fn=eval_fn,
            budget=args.budget,
            rng=np.random.default_rng(args.seed),
        ),
        budget=args.budget,
        seed=args.seed,
        hardware=args.hardware,
        dataset=args.dataset,
        extra_meta={},
        out_path=out_dir / "stage1_random.json",
        log=log,
    )

    # ── Stage 2: MOEA/D + GA mutation ────────────────────────────────────────
    run_stage(
        name="stage2_moead_ga",
        optimizer=MemeticNAS(
            eval_fn=eval_fn,
            budget=args.budget,
            candidate_ops=[GAOperator(
                force_change=True,
                crossover=True,
                freq_bias=True,
                adaptive=False,
            )],
            sa_op=None,
            K=K,
            rng=np.random.default_rng(args.seed),
        ),
        budget=args.budget,
        seed=args.seed,
        hardware=args.hardware,
        dataset=args.dataset,
        extra_meta={"K": K, "operators": ["GA"]},
        out_path=out_dir / "stage2_moead_ga.json",
        log=log,
    )

    # ── Stage 3: MOEA/D + GA + PSO ───────────────────────────────────────────
    run_stage(
        name="stage3_moead_ga_pso",
        optimizer=MemeticNAS(
            eval_fn=eval_fn,
            budget=args.budget,
            candidate_ops=[
                GAOperator(
                    force_change=True,
                    crossover=True,
                    freq_bias=True,
                    adaptive=False,
                ),
                PSOOperator(
                    K=K,
                    c1_init=0.4,
                    c2_init=0.6,
                    eta=0.10,
                    target_rate=0.20,
                ),
            ],
            sa_op=None,
            K=K,
            rng=np.random.default_rng(args.seed),
        ),
        budget=args.budget,
        seed=args.seed,
        hardware=args.hardware,
        dataset=args.dataset,
        extra_meta={"K": K, "operators": ["GA", "PSO"]},
        out_path=out_dir / "stage3_moead_ga_pso.json",
        log=log,
    )

    # ── Stage 4: Full proposal (+ SA acceptance) ──────────────────────────────
    run_stage(
        name="stage4_full",
        optimizer=MemeticNAS(
            eval_fn=eval_fn,
            budget=args.budget,
            candidate_ops=[
                GAOperator(
                    force_change=True,
                    crossover=True,
                    freq_bias=True,
                    adaptive=False,
                ),
                PSOOperator(
                    K=K,
                    c1_init=0.4,
                    c2_init=0.6,
                    eta=0.10,
                    target_rate=0.20,
                ),
            ],
            sa_op=SAOperator(
                T0=0.1,
                alpha=0.95,
                T_min=0.001,
                reheat_factor=1.25,
                reheat_trigger=0.15,
            ),
            K=K,
            rng=np.random.default_rng(args.seed),
        ),
        budget=args.budget,
        seed=args.seed,
        hardware=args.hardware,
        dataset=args.dataset,
        extra_meta={
            "K": K,
            "operators": ["GA", "PSO", "SA"],
            "T0": 0.1,
            "alpha": 0.95,
        },
        out_path=out_dir / "stage4_full.json",
        log=log,
    )

    log.info("All four stages complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
