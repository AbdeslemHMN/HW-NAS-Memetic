#!/usr/bin/env python3
"""
Waterfall build-up sweep — 3 incremental stages over 30 independent seeds.

Executes stages 1–3 of the algorithmic pipeline for every combination of
dataset × hardware, using 30 seeds.  Stage 4 (Full Proposed) is handled by
the main search script (run_search.py) and cached under
results/cache/metrics/full_proposed/.

Stages
------
  Stage 1 — Random baseline
  Stage 2 — MOEA/D + GA mutation
  Stage 3 — MOEA/D + GA + PSO

Output layout
-------------
    results/waterfall/<dataset>/<hardware>/stage1_random_seed<N>.json
    results/waterfall/<dataset>/<hardware>/stage2_moead_ga_seed<N>.json
    results/waterfall/<dataset>/<hardware>/stage3_moead_ga_pso_seed<N>.json

Usage
-----
    python scripts/run_waterfall.py                     # full 3×3 dataset × hardware sweep
    python scripts/run_waterfall.py --dataset cifar100 --hardware raspi4_latency
    python scripts/run_waterfall.py --seeds 0 1 2       # subset of seeds
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
from src.algorithms.operators import GAOperator, PSOOperator
from src.utils.logger import get_logger, save_archive

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
RESULTS_DIR  = PROJECT_ROOT / "results" / "waterfall"
SEEDS        = list(range(30))
TOTAL_BUDGET = 2000
K            = 5      # calibrated best

HARDWARE_CHOICES = [
    "edgegpu_latency", "edgegpu_energy",
    "edgetpu_latency",
    "eyeriss_latency", "eyeriss_energy", "eyeriss_arithmetic_intensity",
    "fpga_latency", "fpga_energy",
    "pixel3_latency",
    "raspi4_latency",
]

# Default waterfall hardware targets for the 3×3 suite used elsewhere in the repo.
DEFAULT_HARDWARE = ["edgegpu_latency", "raspi4_latency", "eyeriss_latency"]
DATASET_CHOICES = ["cifar10", "cifar100", "ImageNet16-120"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run waterfall build-up stages 1–3 for 30 seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=DATASET_CHOICES,
        help="NAS-Bench-201 dataset split. If omitted, runs all datasets.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        choices=HARDWARE_CHOICES,
        help="Hardware metric key from HW-NAS-Bench. If omitted, runs the default 3 hardware targets.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the seed list (default: 0..29).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=TOTAL_BUDGET,
        help="NFE budget per stage per seed.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which stages to run (default: 1 2 3).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip a seed/stage combination if the output file already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run and overwrite even if output already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def build_eval_fn(api: HWNASApi, hardware: str, dataset: str):
    """Wrap HWNASApi.query into the (accuracy, latency) contract."""
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
    log.info("    Running %s  seed=%d  budget=%d …", name, seed, budget)
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
    log.info("    %s seed=%d → %s", name, seed, out_path)


def main() -> None:
    args = parse_args()
    log = get_logger(
        "run_waterfall",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    seeds     = args.seeds if args.seeds is not None else SEEDS
    skip      = args.skip_existing and not args.force
    datasets  = DATASET_CHOICES if args.dataset is None else [args.dataset]
    hardwares = DEFAULT_HARDWARE if args.hardware is None else [args.hardware]

    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    api = HWNASApi(str(DATA_PATH))
    log.info("Datasets         : %s", datasets)
    log.info("Hardware targets : %s", hardwares)
    log.info("Seeds            : %s", seeds)
    log.info("Stages           : %s", args.stages)
    log.info("Budget per stage : %d NFE", args.budget)

    for dataset in datasets:
        for hardware in hardwares:
            out_dir = RESULTS_DIR / dataset / hardware
            out_dir.mkdir(parents=True, exist_ok=True)
            log.info("\n==== dataset=%s hardware=%s ====", dataset, hardware)
            eval_fn = build_eval_fn(api, hardware, dataset)

            for seed in seeds:
                log.info("── seed=%d ─────────────────────────────────────────────", seed)

                # ── Stage 1: Random baseline ──────────────────────────────────────
                if 1 in args.stages:
                    out_path = out_dir / f"stage1_random_seed{seed}.json"
                    if skip and out_path.exists():
                        log.info("  [skip] %s", out_path.name)
                    else:
                        run_stage(
                            name="stage1_random",
                            optimizer=RandomSearch(
                                eval_fn=eval_fn,
                                budget=args.budget,
                                rng=np.random.default_rng(seed),
                            ),
                            budget=args.budget,
                            seed=seed,
                            hardware=hardware,
                            dataset=dataset,
                            extra_meta={},
                            out_path=out_path,
                            log=log,
                        )

                # ── Stage 2: MOEA/D + GA mutation ────────────────────────────────
                if 2 in args.stages:
                    out_path = out_dir / f"stage2_moead_ga_seed{seed}.json"
                    if skip and out_path.exists():
                        log.info("  [skip] %s", out_path.name)
                    else:
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
                                rng=np.random.default_rng(seed),
                            ),
                            budget=args.budget,
                            seed=seed,
                            hardware=hardware,
                            dataset=dataset,
                            extra_meta={"K": K, "operators": ["GA"]},
                            out_path=out_path,
                            log=log,
                        )

                # ── Stage 3: MOEA/D + GA + PSO ───────────────────────────────────
                if 3 in args.stages:
                    out_path = out_dir / f"stage3_moead_ga_pso_seed{seed}.json"
                    if skip and out_path.exists():
                        log.info("  [skip] %s", out_path.name)
                    else:
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
                                rng=np.random.default_rng(seed),
                            ),
                            budget=args.budget,
                            seed=seed,
                            hardware=hardware,
                            dataset=dataset,
                            extra_meta={"K": K, "operators": ["GA", "PSO"]},
                            out_path=out_path,
                            log=log,
                        )

    log.info("Waterfall sweep complete.  Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
