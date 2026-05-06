#!/usr/bin/env python3
"""
scripts/run_ablations.py
────────────────────────
Automated ablation study runner for HW-NAS-Memetic.

Reads every JSON file in configs/, instantiates MemeticNAS with the
corresponding feature-toggle configuration, and runs 30 independent seeds per
experiment.  Results are saved to results/ablations/{experiment_name}_seed{N}.json.

Usage
─────
    python scripts/run_ablations.py                       # all configs, 30 seeds each
    python scripts/run_ablations.py --seeds 0 1           # quick 2-seed smoke test
    python scripts/run_ablations.py --config configs/ablation_t2_no_pso.json

Expected runtime
────────────────
budget=2000, K=5  →  ~10 s per seed on a modern CPU (lookup-only, no GPU needed).
6 experiments × 30 seeds = 180 runs ≈ 30 min total.

Output structure
────────────────
results/ablations/
    full_proposed_seed0.json
    full_proposed_seed1.json
    …
    no_pso_seed0.json
    …
    no_restart_seed4.json

Each JSON follows the same schema as results/proposed/ and results/baselines/,
making them directly loadable by notebooks/04_ablation_analysis.ipynb.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.memetic_nas import MemeticNAS
from src.algorithms.random_search import RandomSearch
from src.algorithms.operators import BaseOperator, GAOperator, PSOOperator, SAOperator
from src.utils.logger import get_logger, save_archive

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
CONFIGS_DIR  = PROJECT_ROOT / "configs"
RESULTS_DIR  = PROJECT_ROOT / "results" / "ablations"

# ── Evaluation settings (fixed across all ablation runs for fair comparison) ──
HARDWARE_METRIC = "edgegpu_latency"
DATASET         = "cifar10"
N_SEEDS         = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments defined in configs/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a single JSON config file. "
            "If omitted, all *.json files in configs/ are used."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(N_SEEDS)),
        metavar="S",
        help="Space-separated list of integer seeds to run (e.g. --seeds 0 1 2).",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=HARDWARE_METRIC,
        help="Hardware metric key passed to HWNASApi.query().",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="NAS-Bench-201 dataset split for accuracy lookup.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set logging level to DEBUG.",
    )
    return parser.parse_args()


def _load_configs(config_arg: str | None) -> list[dict]:
    """Return a list of configuration dicts to run."""
    if config_arg is not None:
        p = Path(config_arg)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return [json.loads(p.read_text())]

    paths = sorted(CONFIGS_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON config files found in {CONFIGS_DIR}. "
            "Did you forget to create the configs/ directory?"
        )
    configs = [json.loads(p.read_text()) for p in paths]
    return configs


def _build_eval_fn(api: HWNASApi, metric: str, dataset: str):
    """Closure that maps an arch array to (accuracy, latency)."""
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=metric)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency
    return _eval


def _resolve_candidate_operators(cfg: dict) -> list[BaseOperator]:
    """Translate config fields into MemeticNAS candidate operators."""
    op_map = {
        "GAOperator": GAOperator,
        "PSOOperator": PSOOperator,
        "GA": GAOperator,
        "PSO": PSOOperator,
    }

    if "candidate_ops" in cfg:
        ops = []
        for name in cfg["candidate_ops"]:
            if name not in op_map:
                raise ValueError(f"Unknown operator name in config: {name}")
            if name in {"PSOOperator", "PSO"}:
                ops.append(op_map[name](c1=cfg.get("c1", 0.5), c2=cfg.get("c2", 0.5)))
            else:
                ops.append(op_map[name]())
        if not ops:
            raise ValueError("Config must specify at least one candidate operator.")
        return ops

    # Backward compatibility for legacy boolean config file schema.
    ops = []
    if cfg.get("use_ga", True):
        ops.append(GAOperator())
    if cfg.get("use_pso", True):
        ops.append(PSOOperator(c1=cfg.get("c1", 0.5), c2=cfg.get("c2", 0.5)))
    if not ops:
        raise ValueError("Config must enable at least one of use_ga or use_pso.")
    return ops


def _build_optimizer(eval_fn: Callable[[np.ndarray], tuple[float, float]], cfg: dict, rng: np.random.Generator):
    """Build a concrete optimizer from an ablation config dict."""
    if cfg.get("experiment_name") == "no_moead":
        # MOEA/D disabled is effectively a single-weight search direction.
        K = 1
        T_neighborhood = 1
    else:
        K = cfg.get("k_directions", 5)
        T_neighborhood = cfg.get("t_neighborhood", 2)

    candidate_ops = _resolve_candidate_operators(cfg)
    sa_op = SAOperator(T0=cfg.get("t0", 1.0), alpha=cfg.get("alpha", 0.95)) if cfg.get("use_sa", True) else None

    return MemeticNAS(
        eval_fn=eval_fn,
        budget=cfg.get("budget", 2000),
        candidate_ops=candidate_ops,
        sa_op=sa_op,
        K=K,
        T_neighborhood=T_neighborhood,
        use_restart=cfg.get("use_restart", True),
        rng=rng,
    )


def main() -> None:
    args  = parse_args()
    log   = get_logger(
        "run_ablations",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # ── Load benchmark API (shared across all runs) ───────────────────────
    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    if not DATA_PATH.exists():
        log.error("Benchmark file not found: %s", DATA_PATH)
        log.error("Run: python scripts/download_data.py")
        sys.exit(1)

    api     = HWNASApi(str(DATA_PATH))
    eval_fn = _build_eval_fn(api, args.hardware, args.dataset)

    # ── Prepare output directory ──────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load experiment configurations ───────────────────────────────────
    configs = _load_configs(args.config)
    log.info(
        "Found %d experiment configuration(s).  Seeds: %s",
        len(configs), args.seeds,
    )

    total_runs = len(configs) * len(args.seeds)
    completed  = 0

    # ── Main loop ─────────────────────────────────────────────────────────
    for cfg in configs:
        exp_name = cfg.get("experiment_name", "unknown")
        log.info("=" * 60)
        log.info("EXPERIMENT: %s", exp_name)
        candidate_ops = cfg.get("candidate_ops") or [
            op for op, enabled in [
                ("GAOperator", cfg.get("use_ga", True)),
                ("PSOOperator", cfg.get("use_pso", True)),
            ] if enabled
        ]
        log.info("  candidate_ops=%s  use_sa=%s  use_restart=%s",
                 candidate_ops,
                 cfg.get("use_sa", True),
                 cfg.get("use_restart", True))
        log.info("  k_directions=%d  budget=%d",
                 cfg.get("k_directions", 5), cfg.get("budget", 2000))

        for seed in args.seeds:
            out_path = RESULTS_DIR / f"{exp_name}_seed{seed}.json"
            if out_path.exists():
                log.info("  [SKIP] seed=%d — result already exists: %s", seed, out_path)
                completed += 1
                continue

            log.info("  Running seed=%d  (%d/%d total runs)", seed, completed + 1, total_runs)
            t0 = time.perf_counter()

            try:
                optimizer = _build_optimizer(
                    eval_fn=eval_fn,
                    cfg=cfg,
                    rng=np.random.default_rng(seed),
                )
            except ValueError as exc:
                log.error("  [ERROR] Invalid config '%s': %s", exp_name, exc)
                log.error("  Skipping this experiment entirely.")
                break

            archive = optimizer.search()
            elapsed = time.perf_counter() - t0

            save_archive(
                archive,
                out_path,
                metadata={
                    "experiment_name": exp_name,
                    "seed": seed,
                    "hardware": args.hardware,
                    "dataset": args.dataset,
                    "budget_spent": optimizer.budget_spent,
                    "config": cfg,
                },
            )
            completed += 1
            log.info(
                "  Saved → %s  (evals=%d, pareto_size=%d, %.1fs)",
                out_path.name,
                optimizer.budget_spent,
                len([e for e in archive
                     if not any(
                         (o["accuracy"] >= e["accuracy"] and o["latency"] <= e["latency"]
                          and (o["accuracy"] > e["accuracy"] or o["latency"] < e["latency"]))
                         for o in archive
                     )]),
                elapsed,
            )

    log.info("=" * 60)
    log.info("Ablation study complete.  %d/%d runs saved to %s",
             completed, total_runs, RESULTS_DIR)
    log.info("Open notebooks/04_ablation_analysis.ipynb to visualise results.")


if __name__ == "__main__":
    main()
