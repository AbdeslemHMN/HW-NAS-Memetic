#!/usr/bin/env python3
"""
CLI entry point for the MemeticNAS MOEA/D + Discrete PSO + SA search.

Runs 30 independent seeds so results are directly comparable with the baselines.

Usage:
    python scripts/run_search.py \
        --hardware edgegpu_latency \
        --budget 2000 \
        --k_directions 20
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.memetic_nas import MemeticNAS
from src.algorithms.operators import GAOperator, PSOOperator, SAOperator
from src.utils.logger import get_logger, save_archive

DATA_PATH      = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
CONFIG_PATH    = PROJECT_ROOT / "configs" / "full_proposed.json"
RESULTS_DIR    = PROJECT_ROOT / "results" / "proposed"
DATASET        = "cifar10"
SEEDS          = list(range(30))
TOTAL_BUDGET   = 2000   # strict NFE cap — must match baselines


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to JSON config with default runtime parameters.",
    )
    known, _ = base_parser.parse_known_args()

    config = {}
    config_path = Path(known.config)
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in config file {config_path}: {exc}")

    def cfg(key, default):
        return config.get(key, default)

    def nested_cfg(*keys, default=None):
        node = config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    parser = argparse.ArgumentParser(
        description="Run MemeticNAS on HW-NAS-Bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[base_parser],
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=cfg("hardware", "edgegpu_latency"),
        choices=[
            "edgegpu_latency", "edgegpu_energy",
            "edgetpu_latency",
            "eyeriss_latency", "eyeriss_energy", "eyeriss_arithmetic_intensity",
            "fpga_latency", "fpga_energy",
            "pixel3_latency",
            "raspi4_latency",
        ],
        help="Hardware metric key from HW-NAS-Bench.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=cfg("budget", TOTAL_BUDGET),
        help="Total number of architecture evaluations.",
    )
    parser.add_argument(
        "--k_directions",
        type=int,
        default=cfg("k_directions", 5),
        help="Number of MOEA/D weight-vector directions (K).",
    )
    parser.add_argument(
        "--t_neighborhood",
        type=int,
        default=cfg("t_neighborhood", 2),
        help="Neighborhood size per sub-problem.",
    )
    parser.add_argument(
        "--scalarization",
        type=str,
        default=cfg("scalarization", "linear"),
        choices=["linear", "tchebychev"],
        help="MOEA/D scalarization strategy.",
    )
    parser.add_argument(
        "--w_init",
        type=float,
        default=cfg("w_init", 0.6),
        help="Starting MOEA/D weight for accuracy at k=0.",
    )
    parser.add_argument(
        "--w_final",
        type=float,
        default=cfg("w_final", 0.4),
        help="Ending MOEA/D weight for accuracy at k=K-1.",
    )
    parser.add_argument(
        "--use-restart",
        dest="use_restart",
        action="store_true",
        default=cfg("use_restart", True),
        help="Enable restart logic when the population stagnates.",
    )
    parser.add_argument(
        "--no-use-restart",
        dest="use_restart",
        action="store_false",
        help="Disable restart logic.",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=cfg("t0", 0.1),
        help="SA initial temperature (golden calibrated value).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=cfg("alpha", 0.95),
        help="SA cooling rate, in (0, 1).",
    )
    parser.add_argument(
        "--c1",
        "--c1_init",
        dest="c1_init",
        type=float,
        default=cfg("c1_init", nested_cfg("operator_dna", "pso", "c1_init", default=0.4)),
        help="PSO personal attraction coefficient (initial).",
    )
    parser.add_argument(
        "--c2",
        "--c2_init",
        dest="c2_init",
        type=float,
        default=cfg("c2_init", nested_cfg("operator_dna", "pso", "c2_init", default=0.6)),
        help="PSO global attraction coefficient (initial).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=cfg("eta", nested_cfg("operator_dna", "pso", "eta", default=0.10)),
        help="PSO cognitive/exploration coefficient.",
    )
    parser.add_argument(
        "--target_rate",
        type=float,
        default=cfg("target_rate", nested_cfg("operator_dna", "pso", "target_rate", default=0.20)),
        help="PSO target success rate.",
    )
    parser.add_argument(
        "--p_m",
        type=float,
        default=cfg("p_m", nested_cfg("operator_dna", "ga", "p_m", default=1/6)),
        help="GA mutation probability.",
    )
    parser.add_argument(
        "--ga_force_change",
        type=bool,
        default=cfg("ga_force_change", nested_cfg("operator_dna", "ga", "force_change", default=True)),
        help="Enable forced mutation change in GA.",
    )
    parser.add_argument(
        "--ga_crossover",
        type=bool,
        default=cfg("ga_crossover", nested_cfg("operator_dna", "ga", "crossover", default=True)),
        help="Enable GA crossover.",
    )
    parser.add_argument(
        "--ga_freq_bias",
        type=bool,
        default=cfg("ga_freq_bias", nested_cfg("operator_dna", "ga", "freq_bias", default=True)),
        help="Enable frequency-biased GA mutation.",
    )
    parser.add_argument(
        "--ga_adaptive",
        type=bool,
        default=cfg("ga_adaptive", nested_cfg("operator_dna", "ga", "adaptive", default=False)),
        help="Enable adaptive GA mutation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=cfg("dataset", DATASET),
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

    out_dir = RESULTS_DIR / args.dataset / args.hardware
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        log.info(
            "── Seed %d / %d  (budget=%d  K=%d  hardware=%s) ──────────────",
            seed + 1, len(SEEDS), args.budget, args.k_directions, args.hardware,
        )
        optimizer = MemeticNAS(
            eval_fn=eval_fn,
            budget=args.budget,
            candidate_ops=[
                GAOperator(
                    p_m=args.p_m,
                    force_change=args.ga_force_change,
                    crossover=args.ga_crossover,
                    freq_bias=args.ga_freq_bias,
                    adaptive=args.ga_adaptive,
                ),
                PSOOperator(
                    K=args.k_directions,
                    c1_init=args.c1_init,
                    c2_init=args.c2_init,
                    eta=args.eta,
                    target_rate=args.target_rate,
                ),
            ],
            sa_op=SAOperator(
                T0=args.t0,
                alpha=args.alpha,
                T_min=0.001,
                reheat_factor=1.25,
                reheat_trigger=0.15,
            ),
            K=args.k_directions,
            T_neighborhood=args.t_neighborhood,
            scalarization=args.scalarization,
            w_init=args.w_init,
            w_final=args.w_final,
            use_restart=args.use_restart,
            rng=np.random.default_rng(seed),
        )

        archive = optimizer.search()
        log.info("Seed %d complete. NFE used: %d", seed, optimizer.budget_spent)

        metadata = {
            "algorithm":       "MemeticNAS",
            "config_path":     args.config,
            "seed":            seed,
            "budget":          args.budget,
            "n_evaluations":   optimizer.budget_spent,
            "budget_spent":    optimizer.budget_spent,
            "nfe_checkpoints": optimizer.nfe_checkpoints,
            "hardware":        args.hardware,
            "dataset":         args.dataset,
            "K":               args.k_directions,
            "T_neighborhood":  args.t_neighborhood,
            "scalarization":   args.scalarization,
            "w_init":          args.w_init,
            "w_final":         args.w_final,
            "T0":              args.t0,
            "alpha":           args.alpha,
            "c1_init":         args.c1_init,
            "c2_init":         args.c2_init,
            "p_m":             args.p_m,
            "ga_force_change": args.ga_force_change,
            "ga_crossover":    args.ga_crossover,
            "ga_freq_bias":    args.ga_freq_bias,
            "ga_adaptive":     args.ga_adaptive,
            "pso_eta":         args.eta,
            "pso_target_rate": args.target_rate,
            "use_restart":     args.use_restart,
        }
        assert metadata["n_evaluations"] == args.budget, (
            f"[MemeticNAS seed={seed}] Evaluation budget mismatch! "
            f"Got {metadata['n_evaluations']}, expected {args.budget}"
        )
        out_path = out_dir / f"proposed_res_seed{seed}.json"
        save_archive(archive, out_path, metadata=metadata)
        log.info("Results written → %s", out_path)

    log.info("All %d seeds complete. Results in %s", len(SEEDS), out_dir)


if __name__ == "__main__":
    main()
