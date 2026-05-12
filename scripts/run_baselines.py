#!/usr/bin/env python3
"""
Baseline sweep: RandomSearch and NSGA-II over 30 independent seeds.

Output: results/baselines/<dataset>/<hardware>/{algorithm}_seed{seed}.json

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --dataset cifar100 --hardware raspi4_latency
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.random_search import RandomSearch
from src.algorithms.nsga2_search import nsga2_search
from src.utils.logger import get_logger, save_archive

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
RESULTS_DIR  = PROJECT_ROOT / "results" / "baselines"
SEEDS = list(range(30))
TOTAL_BUDGET = 2000   # strict NFE cap — identical across ALL algorithms
# NSGA-II budget decomposition: N_pop × (N_gen + 1) = 40 × 50 = 2000
# (initial population counts as one generation in pymoo's n_eval counter)
POP_SIZE     = 40    # N_pop — population size

log = get_logger("baselines")


HARDWARE_CHOICES = [
    "edgegpu_latency", "edgegpu_energy",
    "edgetpu_latency",
    "eyeriss_latency", "eyeriss_energy", "eyeriss_arithmetic_intensity",
    "fpga_latency", "fpga_energy",
    "pixel3_latency",
    "raspi4_latency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomSearch and NSGA-II baselines on HW-NAS-Bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="NAS-Bench-201 dataset split.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="edgegpu_latency",
        choices=HARDWARE_CHOICES,
        help="Hardware metric key from HW-NAS-Bench.",
    )
    return parser.parse_args()


def build_eval_fn(api: HWNASApi, metric: str, dataset: str):
    """Wrap HWNASApi.query into the (accuracy, latency) contract."""
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=metric)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency

    return _eval


def main() -> None:
    args = parse_args()
    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    api = HWNASApi(str(DATA_PATH))  # auto-detects nas201_accuracy_cache.npz if present
    eval_fn = build_eval_fn(api, args.hardware, args.dataset)

    out_dir = RESULTS_DIR / args.dataset / args.hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── RandomSearch ─────────────────────────────────────────────────────────
    for seed in SEEDS:
        log.info("[RandomSearch] seed=%d  budget=%d", seed, TOTAL_BUDGET)
        rs = RandomSearch(eval_fn, budget=TOTAL_BUDGET, rng=np.random.default_rng(seed))
        archive = rs.search()
        metadata = {
            "algorithm":       "RandomSearch",
            "seed":            seed,
            "budget":          TOTAL_BUDGET,
            "n_evaluations":   rs.budget_spent,
            "budget_spent":    rs.budget_spent,
            "nfe_checkpoints": rs.nfe_checkpoints,
            "hardware":        args.hardware,
            "dataset":         args.dataset,
        }
        assert metadata["n_evaluations"] == TOTAL_BUDGET, (
            f"[RandomSearch seed={seed}] Evaluation budget mismatch! "
            f"Got {metadata['n_evaluations']}, expected {TOTAL_BUDGET}"
        )
        out_path = out_dir / f"random_search_seed{seed}.json"
        save_archive(archive, out_path, metadata=metadata)
        log.info("[RandomSearch] seed=%d → %s", seed, out_path)

    # ── NSGA-II ───────────────────────────────────────────────────────────────
    for seed in SEEDS:
        log.info("[NSGA2] seed=%d  budget=%d  pop_size=%d", seed, TOTAL_BUDGET, POP_SIZE)
        archive, nfe_checkpoints = nsga2_search(
            eval_fn, budget=TOTAL_BUDGET, pop_size=POP_SIZE, rng_seed=seed
        )
        metadata = {
            "algorithm":       "NSGA2",
            "seed":            seed,
            "budget":          TOTAL_BUDGET,
            "n_evaluations":   len(archive),
            "budget_spent":    len(archive),
            "nfe_checkpoints": nfe_checkpoints,
            "hardware":        args.hardware,
            "dataset":         args.dataset,
            "pop_size":        POP_SIZE,
        }
        assert metadata["n_evaluations"] == TOTAL_BUDGET, (
            f"[NSGA2 seed={seed}] Evaluation budget mismatch! "
            f"Got {metadata['n_evaluations']}, expected {TOTAL_BUDGET}"
        )
        out_path = out_dir / f"nsga2_seed{seed}.json"
        save_archive(archive, out_path, metadata=metadata)
        log.info("[NSGA2] seed=%d → %s", seed, out_path)

    log.info("Baseline sweep complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
