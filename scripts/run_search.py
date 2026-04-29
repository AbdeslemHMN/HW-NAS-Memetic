#!/usr/bin/env python3
"""
CLI entry point for the proposed MOEA/D + Discrete PSO + SA search.

Runs 5 independent seeds so results are directly comparable with the baselines.

Usage:
    python scripts/run_search.py \
        --hardware edgegpu_latency \
        --budget 2000 \
        --k_directions 20
"""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.proposed_moead_pso import ProposedMoeadPso
from src.utils.logger import get_logger, save_archive

DATA_PATH    = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
RESULTS_DIR  = PROJECT_ROOT / "results" / "proposed"
DATASET      = "cifar10"
SEEDS        = list(range(30))
TOTAL_BUDGET = 2000   # strict NFE cap — must match baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ProposedMoeadPso on HW-NAS-Bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="edgegpu_latency",
        help="Hardware metric key (e.g. edgegpu_latency, raspi4_latency).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=TOTAL_BUDGET,
        help="Total number of architecture evaluations.",
    )
    parser.add_argument(
        "--k_directions",
        type=int,
        default=20,
        help="Number of MOEA/D weight-vector directions (K).",
    )
    parser.add_argument(
        "--t_neighborhood",
        type=int,
        default=2,
        help="Neighborhood size per sub-problem.",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=1.0,
        help="SA initial temperature.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="SA cooling rate, in (0, 1).",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.5,
        help="PSO personal attraction coefficient.",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.5,
        help="PSO global attraction coefficient.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="NAS-Bench-201 dataset split.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set logging level to DEBUG.",
    )
    return parser.parse_args()


def build_eval_fn(api: HWNASApi, metric: str, dataset: str):
    """Wrap HWNASApi.query into the (accuracy, latency) contract expected by optimizers."""
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=metric)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency

    return _eval


def main() -> None:
    args = parse_args()
    log = get_logger(
        "run_search",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    log.info("Configuration: %s", vars(args))
    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    api = HWNASApi(str(DATA_PATH))  # auto-detects nas201_accuracy_cache.npz if present
    eval_fn = build_eval_fn(api, args.hardware, args.dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        log.info(
            "── Seed %d / %d  (budget=%d  K=%d  hardware=%s) ──────────────",
            seed + 1, len(SEEDS), args.budget, args.k_directions, args.hardware,
        )
        optimizer = ProposedMoeadPso(
            eval_fn=eval_fn,
            budget=args.budget,
            K=args.k_directions,
            T_neighborhood=args.t_neighborhood,
            T0=args.t0,
            alpha=args.alpha,
            c1=args.c1,
            c2=args.c2,
            rng=np.random.default_rng(seed),
        )

        archive = optimizer.search()
        log.info("Seed %d complete. NFE used: %d", seed, optimizer.budget_spent)

        metadata = {
            "algorithm":       "ProposedMoeadPso",
            "seed":            seed,
            "budget":          args.budget,
            "n_evaluations":   optimizer.budget_spent,
            "budget_spent":    optimizer.budget_spent,
            "nfe_checkpoints": optimizer.nfe_checkpoints,
            "hardware":        args.hardware,
            "dataset":         args.dataset,
            "K":               args.k_directions,
            "T_neighborhood":  args.t_neighborhood,
            "T0":              args.t0,
            "alpha":           args.alpha,
            "c1":              args.c1,
            "c2":              args.c2,
        }
        assert metadata["n_evaluations"] == TOTAL_BUDGET, (
            f"[ProposedMoeadPso seed={seed}] Evaluation budget mismatch! "
            f"Got {metadata['n_evaluations']}, expected {TOTAL_BUDGET}"
        )
        out_path = RESULTS_DIR / f"proposed_res_seed{seed}.json"
        save_archive(archive, out_path, metadata=metadata)
        log.info("Results written → %s", out_path)

    log.info("All %d seeds complete. Results in %s", len(SEEDS), RESULTS_DIR)


if __name__ == "__main__":
    main()
