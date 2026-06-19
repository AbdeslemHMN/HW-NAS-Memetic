#!/usr/bin/env python3
"""
scripts/run_component_baselines.py
────────────────────────────────────────────────────────────────────────────────
Component-isolation baseline runner for HW-NAS-Memetic.

Runs each algorithmic module **in isolation** across 30 seeds, 3 datasets, and
3 hardware targets, using the exact calibrated hyper-parameters from
``configs/full_proposed.json`` as the baseline foundation.

Experiments
───────────
  moead_only   – MOEA/D decomposition with random selection, no GA/PSO/SA
  ga_only      – GA operator + MOEA/D collapsed to k=1 (single direction), no PSO/SA
  pso_only     – PSO operator + MOEA/D collapsed to k=1, no GA/SA
  sa_only      – SA local refinement only (k=1, no GA/PSO)

Isolation strategy
──────────────────
Each mode keeps its own operator at full calibrated strength and zeroes out or
disables the others:
  ┌──────────────┬────────────┬────────────┬────────────┬────────────┐
  │              │ MOEA/D (K) │  GA active │ PSO active │ SA active  │
  ├──────────────┼────────────┼────────────┼────────────┼────────────┤
  │ moead_only   │ calibrated │     ✗      │     ✗      │     ✗      │
  │ ga_only      │  k=1       │     ✓      │     ✗      │     ✗      │
  │ pso_only     │  k=1       │     ✗      │     ✓      │     ✗      │
  │ sa_only      │  k=1       │     ✗      │     ✗      │     ✓      │
  └──────────────┴────────────┴────────────┴────────────┴────────────┘

Output structure
────────────────
results/ablations/{dataset}/{hardware}/
    moead_only_seed0.json
    moead_only_seed1.json
    …
    sa_only_seed29.json

Usage
─────
    # Full grid — 4 variants × 9 combos × 30 seeds = 1 080 runs
    python scripts/run_component_baselines.py

    # Smoke test: 2 seeds, one combo
    python scripts/run_component_baselines.py --seeds 0 1 --datasets cifar10 --hardware edgegpu_latency

    # Single experiment
    python scripts/run_component_baselines.py --experiments ga_only

    # Preview run without writing anything
    python scripts/run_component_baselines.py --dry-run

    # Force re-run, overwriting existing results
    python scripts/run_component_baselines.py --overwrite
"""
from __future__ import annotations

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
from src.algorithms.operators import GAOperator, PSOOperator, SAOperator
from src.utils.logger import get_logger, save_archive

# ── Fixed paths ───────────────────────────────────────────────────────────────
DATA_PATH   = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
CONFIG_PATH = PROJECT_ROOT / "configs" / "full_proposed.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablations"

# ── Grid defaults ──────────────────────────────────────────────────────────────
ALL_DATASETS  = ["cifar10", "cifar100", "ImageNet16-120"]
ALL_HARDWARE  = ["edgegpu_latency", "raspi4_latency", "eyeriss_latency"]
ALL_EXPERIMENTS = ["moead_only", "ga_only", "pso_only", "sa_only"]
N_SEEDS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_base_config() -> dict:
    """Load calibrated parameters from configs/full_proposed.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated config not found: {CONFIG_PATH}\n"
            "Ensure configs/full_proposed.json exists before running this script."
        )
    cfg = json.loads(CONFIG_PATH.read_text())
    # Flatten operator_dna fields into top-level for convenience
    for block, params in cfg.get("operator_dna", {}).items():
        for k, v in params.items():
            cfg.setdefault(f"{block}_{k}", v)
    return cfg


def _p(cfg: dict, *keys, fallback=None):
    """Look up a parameter through multiple candidate key names, returning the
    first match found; raises KeyError when no fallback is given."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    if fallback is not None:
        return fallback
    raise KeyError(
        f"None of the keys {keys} found in config. "
        "Please verify configs/full_proposed.json."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Isolated optimizer builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_moead_only(cfg: dict, eval_fn: Callable, rng: np.random.Generator) -> MemeticNAS:
    """MOEA/D with calibrated K and scalarization; no GA, PSO, or SA.

    Uses a minimal GAOperator with all mutations disabled so MemeticNAS has
    something in candidate_ops but it produces no real changes — effectively
    random restart offspring, leaving MOEA/D decomposition as the only driver.
    """
    K = _p(cfg, "k_directions", fallback=5)
    # Provide a pass-through operator that still satisfies MemeticNAS contract
    # but performs only a random 1-bit mutation (simplest possible exploration)
    dummy_ga = GAOperator(
        force_change=False,
        crossover=False,
        freq_bias=False,
        adaptive=False,
    )
    return MemeticNAS(
        eval_fn=eval_fn,
        budget=_p(cfg, "budget", fallback=2000),
        candidate_ops=[dummy_ga],
        sa_op=None,
        K=K,
        T_neighborhood=_p(cfg, "t_neighborhood", fallback=2),
        scalarization=_p(cfg, "scalarization", fallback="linear"),
        w_init=_p(cfg, "w_init", fallback=0.6),
        w_final=_p(cfg, "w_final", fallback=0.4),
        use_restart=False,
        rng=rng,
    )


def _build_ga_only(cfg: dict, eval_fn: Callable, rng: np.random.Generator) -> MemeticNAS:
    """GA operator with k=1 (single decomposition direction), no PSO or SA.

    All GA calibrated parameters are preserved exactly.
    """
    ga_op = GAOperator(
        p_m=_p(cfg, "ga_p_m", "p_m", fallback=1 / 6),
        force_change=_p(cfg, "ga_force_change", fallback=True),
        crossover=_p(cfg, "ga_crossover", fallback=True),
        freq_bias=_p(cfg, "ga_freq_bias", fallback=True),
        adaptive=_p(cfg, "ga_adaptive", fallback=False),
    )
    return MemeticNAS(
        eval_fn=eval_fn,
        budget=_p(cfg, "budget", fallback=2000),
        candidate_ops=[ga_op],
        sa_op=None,
        K=1,
        T_neighborhood=1,
        scalarization=_p(cfg, "scalarization", fallback="linear"),
        w_init=_p(cfg, "w_init", fallback=0.6),
        w_final=_p(cfg, "w_final", fallback=0.4),
        use_restart=False,
        rng=rng,
    )


def _build_pso_only(cfg: dict, eval_fn: Callable, rng: np.random.Generator) -> MemeticNAS:
    """PSO operator with k=1, no GA or SA.

    All PSO calibrated parameters are preserved exactly.
    """
    pso_op = PSOOperator(
        K=1,
        c1_init=_p(cfg, "c1_init", "pso_c1_init", fallback=0.4),
        c2_init=_p(cfg, "c2_init", "pso_c2_init", fallback=0.6),
        eta=_p(cfg, "pso_eta", "eta", fallback=0.10),
        target_rate=_p(cfg, "pso_target_rate", "target_rate", fallback=0.20),
    )
    return MemeticNAS(
        eval_fn=eval_fn,
        budget=_p(cfg, "budget", fallback=2000),
        candidate_ops=[pso_op],
        sa_op=None,
        K=1,
        T_neighborhood=1,
        scalarization=_p(cfg, "scalarization", fallback="linear"),
        w_init=_p(cfg, "w_init", fallback=0.6),
        w_final=_p(cfg, "w_final", fallback=0.4),
        use_restart=False,
        rng=rng,
    )


def _build_sa_only(cfg: dict, eval_fn: Callable, rng: np.random.Generator) -> MemeticNAS:
    """SA local refinement only (k=1, no GA or PSO).

    All SA calibrated parameters are preserved exactly.
    Needs a minimal candidate_op; uses the same disabled GAOperator as moead_only.
    """
    sa_op = SAOperator(
        T0=_p(cfg, "t0", fallback=0.1),
        alpha=_p(cfg, "alpha", fallback=0.95),
        T_min=_p(cfg, "sa_t_min", fallback=0.001),
        reheat_factor=_p(cfg, "sa_reheat_factor", fallback=1.25),
        reheat_trigger=_p(cfg, "sa_reheat_trigger", fallback=0.15),
    )
    dummy_ga = GAOperator(
        force_change=False,
        crossover=False,
        freq_bias=False,
        adaptive=False,
    )
    return MemeticNAS(
        eval_fn=eval_fn,
        budget=_p(cfg, "budget", fallback=2000),
        candidate_ops=[dummy_ga],
        sa_op=sa_op,
        K=1,
        T_neighborhood=1,
        scalarization=_p(cfg, "scalarization", fallback="linear"),
        w_init=_p(cfg, "w_init", fallback=0.6),
        w_final=_p(cfg, "w_final", fallback=0.4),
        use_restart=False,
        rng=rng,
    )


_BUILDERS = {
    "moead_only": _build_moead_only,
    "ga_only":    _build_ga_only,
    "pso_only":   _build_pso_only,
    "sa_only":    _build_sa_only,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Component-isolation baseline runner.\n"
            "Runs moead_only / ga_only / pso_only / sa_only across 3 datasets × "
            "3 hardware × 30 seeds using calibrated hyper-parameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=ALL_EXPERIMENTS,
        default=ALL_EXPERIMENTS,
        metavar="EXP",
        help=(
            "Which isolated components to run.  "
            f"Choices: {ALL_EXPERIMENTS}.  Default: all four."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        metavar="DS",
        help=f"Datasets to run.  Choices: {ALL_DATASETS}.  Default: all three.",
    )
    parser.add_argument(
        "--hardware",
        nargs="+",
        choices=ALL_HARDWARE,
        default=ALL_HARDWARE,
        metavar="HW",
        help=f"Hardware targets to run.  Choices: {ALL_HARDWARE}.  Default: all three.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(N_SEEDS)),
        metavar="S",
        help="Seeds to run.  Default: 0–29.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        metavar="PATH",
        help="Path to calibrated JSON config.  Default: configs/full_proposed.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing result files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_eval_fn(api: HWNASApi, hardware: str, dataset: str) -> Callable:
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=hardware)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency
    return _eval


def _pareto_size(archive: list[dict]) -> int:
    return sum(
        1 for e in archive
        if not any(
            o["accuracy"] >= e["accuracy"]
            and o["latency"] <= e["latency"]
            and (o["accuracy"] > e["accuracy"] or o["latency"] < e["latency"])
            for o in archive
        )
    )


def _build_run_grid(args) -> list[tuple[str, str, str, int]]:
    """Return all (experiment, dataset, hardware, seed) tuples in order."""
    return [
        (exp, ds, hw, seed)
        for exp  in args.experiments
        for ds   in args.datasets
        for hw   in args.hardware
        for seed in args.seeds
    ]


def _print_dry_run_plan(grid: list, base_cfg: dict) -> None:
    total = len(grid)
    print(f"\n{'─' * 68}")
    print(f"  DRY RUN  —  {total} runs planned")
    print(f"{'─' * 68}")
    print(f"  Base config : {CONFIG_PATH}")
    print(f"  Budget/run  : {base_cfg.get('budget', 2000)} NFE")
    print(f"  Experiments : {sorted(set(e for e, *_ in grid))}")
    print(f"  Datasets    : {sorted(set(d for _, d, *_ in grid))}")
    print(f"  Hardware    : {sorted(set(h for _, _, h, *_ in grid))}")
    print(f"  Seeds       : {sorted(set(s for *_, s in grid))}")
    print(f"  Output dir  : {RESULTS_DIR}/{{dataset}}/{{hardware}}/")
    print(f"{'─' * 68}\n")
    for i, (exp, ds, hw, seed) in enumerate(grid, 1):
        out = RESULTS_DIR / ds / hw / f"{exp}_seed{seed}.json"
        status = "EXISTS" if out.exists() else "PLANNED"
        print(f"  [{i:4d}/{total}]  {exp:<12}  {ds:<17}  {hw:<18}  seed={seed:2d}  [{status}]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    log  = get_logger(
        "run_component_baselines",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # ── Load calibrated config ────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)
    base_cfg = json.loads(config_path.read_text())
    # Flatten operator_dna for easy key lookup
    for block, params in base_cfg.get("operator_dna", {}).items():
        for k, v in params.items():
            base_cfg.setdefault(f"{block}_{k}", v)

    log.info("Loaded calibrated config from %s", config_path)
    log.info(
        "  k_directions=%d  budget=%d  t0=%.4f  alpha=%.4f",
        base_cfg.get("k_directions", 5),
        base_cfg.get("budget", 2000),
        base_cfg.get("t0", 0.1),
        base_cfg.get("alpha", 0.95),
    )

    # ── Build execution grid ──────────────────────────────────────────────
    grid = _build_run_grid(args)
    total = len(grid)
    log.info(
        "Grid: %d experiment(s) × %d dataset(s) × %d hardware(s) × %d seed(s) = %d total runs",
        len(args.experiments), len(args.datasets), len(args.hardware), len(args.seeds), total,
    )

    if args.dry_run:
        _print_dry_run_plan(grid, base_cfg)
        return

    # ── Load benchmark API (shared across all runs) ───────────────────────
    if not DATA_PATH.exists():
        log.error("Benchmark data not found: %s", DATA_PATH)
        log.error("Run:  python scripts/download_data.py")
        sys.exit(1)

    log.info("Loading HW-NAS-Bench …  (this takes ~30 s the first time)")
    api = HWNASApi(str(DATA_PATH))

    # Cache eval_fn per (dataset, hardware) combo — avoids redundant construction
    _eval_cache: dict[tuple[str, str], Callable] = {}

    completed = 0
    skipped   = 0
    failed    = 0

    # ── Main sweep ────────────────────────────────────────────────────────
    for run_idx, (exp_name, dataset, hardware, seed) in enumerate(grid, 1):
        out_dir  = RESULTS_DIR / dataset / hardware
        out_path = out_dir / f"{exp_name}_seed{seed}.json"

        prefix = f"[{run_idx:4d}/{total}]  {exp_name:<12}  {dataset:<17}  {hardware:<18}  seed={seed:2d}"

        if out_path.exists() and not args.overwrite:
            log.info("%s  SKIP (exists)", prefix)
            skipped += 1
            continue

        # Build or reuse eval function
        combo_key = (dataset, hardware)
        if combo_key not in _eval_cache:
            _eval_cache[combo_key] = _build_eval_fn(api, hardware, dataset)
        eval_fn = _eval_cache[combo_key]

        # Build isolated optimizer
        builder = _BUILDERS[exp_name]
        try:
            optimizer = builder(base_cfg, eval_fn, np.random.default_rng(seed))
        except Exception as exc:
            log.error("%s  BUILD ERROR: %s", prefix, exc)
            failed += 1
            continue

        # Run search
        log.info("%s  running …", prefix)
        t0 = time.perf_counter()
        try:
            archive = optimizer.search()
        except Exception as exc:
            log.error("%s  SEARCH ERROR: %s", prefix, exc)
            failed += 1
            continue
        elapsed = time.perf_counter() - t0

        # Save result
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_archive(
                archive,
                out_path,
                metadata={
                    "experiment_name": exp_name,
                    "seed":            seed,
                    "hardware":        hardware,
                    "dataset":         dataset,
                    "budget_spent":    optimizer.budget_spent,
                    "elapsed_s":       round(elapsed, 2),
                    "pareto_size":     _pareto_size(archive),
                    "base_config":     str(config_path),
                    "isolation": {
                        "active_component": exp_name,
                        "k_directions":     optimizer.K,
                        "use_sa":           optimizer.sa_op is not None,
                        "use_restart":      optimizer.use_restart,
                    },
                },
            )
        except Exception as exc:
            log.error("%s  SAVE ERROR: %s", prefix, exc)
            failed += 1
            continue

        completed += 1
        log.info(
            "%s  ✓  NFE=%d  pareto=%d  %.1fs",
            prefix,
            optimizer.budget_spent,
            _pareto_size(archive),
            elapsed,
        )

    # ── Final summary ─────────────────────────────────────────────────────
    log.info("=" * 68)
    log.info(
        "Done.  completed=%d  skipped=%d  failed=%d  (total=%d)",
        completed, skipped, failed, total,
    )
    if failed > 0:
        log.warning("%d run(s) failed — check logs above for details.", failed)
    log.info("Results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
