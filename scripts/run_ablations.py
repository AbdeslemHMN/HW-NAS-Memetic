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

    # GA variant comparison inside the FULL pipeline (GA + PSO + SA):
    python scripts/run_ablations.py --compare-ga-full --seeds 0 1 2   # quick smoke test
    python scripts/run_ablations.py --compare-ga-full                  # full 30 seeds
    python scripts/run_ablations.py --compare-ga-full --ga-variants t1..t8

Expected runtime
────────────────
budget=2000, K=5  →  ~10 s per seed on a modern CPU (lookup-only, no GPU needed).
6 experiments × 30 seeds = 180 runs ≈ 30 min total.

Output structure
────────────────
results/ablations/
    full_proposed_seed0.json
    …
results/ga_pipeline_comparison/
    t1_baseline_seed0.json
    t8_full_seed0.json
    t16_full_fbias_seed0.json
    …

Each JSON follows the same schema as results/proposed/ and results/baselines/,
making them directly loadable by notebooks/04_ablation_analysis.ipynb.
"""
import argparse
import json
import logging
import re
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

# ── GA variants — T1..T16 toggles for GAOperator improvements ───────────────
# Each entry: (display_name, safe_filename_prefix, GAOperator kwargs)
GA_PIPELINE_VARIANTS: list[tuple[str, str, dict]] = [
    ("T1 (baseline)", "t1_baseline", dict(force_change=False, adaptive=False, crossover=False)),
    ("T2 (force_change)", "t2_force_change", dict(force_change=True, adaptive=False, crossover=False)),
    ("T3 (adaptive_pm)", "t3_adaptive_pm", dict(force_change=False, adaptive=True, crossover=False)),
    ("T4 (crossover)", "t4_crossover", dict(force_change=False, adaptive=False, crossover=True)),
    ("T5 (force_change + adaptive)", "t5_force_change_adaptive", dict(force_change=True, adaptive=True, crossover=False)),
    ("T6 (force_change + crossover)", "t6_force_change_crossover", dict(force_change=True, adaptive=False, crossover=True)),
    ("T7 (adaptive + crossover)", "t7_adaptive_crossover", dict(force_change=False, adaptive=True, crossover=True)),
    ("T8 (force_change + adaptive + crossover)", "t8_full", dict(force_change=True, adaptive=True, crossover=True)),
    ("T9 (baseline + freq_bias)", "t9_baseline_fbias", dict(force_change=False, adaptive=False, crossover=False, freq_bias=True)),
    ("T10 (force_change + freq_bias)", "t10_force_change_fbias", dict(force_change=True, adaptive=False, crossover=False, freq_bias=True)),
    ("T11 (adaptive_pm + freq_bias)", "t11_adaptive_pm_fbias", dict(force_change=False, adaptive=True, crossover=False, freq_bias=True)),
    ("T12 (crossover + freq_bias)", "t12_crossover_fbias", dict(force_change=False, adaptive=False, crossover=True, freq_bias=True)),
    ("T13 (force_change + adaptive + freq_bias)", "t13_force_change_adaptive_fbias", dict(force_change=True, adaptive=True, crossover=False, freq_bias=True)),
    ("T14 (force_change + crossover + freq_bias)", "t14_force_change_crossover_fbias", dict(force_change=True, adaptive=False, crossover=True, freq_bias=True)),
    ("T15 (adaptive + crossover + freq_bias)", "t15_adaptive_crossover_fbias", dict(force_change=False, adaptive=True, crossover=True, freq_bias=True)),
    ("T16 (force_change + adaptive + crossover + freq_bias)", "t16_full_fbias", dict(force_change=True, adaptive=True, crossover=True, freq_bias=True)),
]

GA_PIPELINE_RESULTS_DIR = PROJECT_ROOT / "results" / "ga_pipeline_comparison"


def _select_ga_variants(spec: str) -> list[tuple[str, str, dict]]:
    """Parse a variant spec (e.g., "all", "t1..t8", "t1,t3,t10") into a list."""
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return GA_PIPELINE_VARIANTS

    variants_by_tag: dict[str, tuple[str, str, dict]] = {}
    for name, prefix, kwargs in GA_PIPELINE_VARIANTS:
        tag = name.split()[0].lower()
        variants_by_tag[tag] = (name, prefix, kwargs)

    def _parse_t(val: str) -> int | None:
        match = re.search(r"t?(\d+)", val)
        return int(match.group(1)) if match else None

    def _expand_range(token: str) -> list[str] | None:
        if ".." in token:
            left, right = token.split("..", 1)
        elif "-" in token:
            left, right = token.split("-", 1)
        else:
            return None
        start = _parse_t(left)
        end = _parse_t(right)
        if start is None or end is None:
            return None
        if start > end:
            start, end = end, start
        return [f"t{i}" for i in range(start, end + 1)]

    selected_tags: set[str] = set()
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Empty --ga-variants specification.")

    for token in tokens:
        expanded = _expand_range(token)
        if expanded is not None:
            for tag in expanded:
                if tag not in variants_by_tag:
                    raise ValueError(f"Unknown GA variant tag: {tag}")
                selected_tags.add(tag)
            continue

        tag = token
        if tag.isdigit():
            tag = f"t{tag}"
        if not tag.startswith("t"):
            tag = f"t{tag}"
        if tag not in variants_by_tag:
            raise ValueError(f"Unknown GA variant tag: {tag}")
        selected_tags.add(tag)

    selected = [
        v for v in GA_PIPELINE_VARIANTS
        if v[0].split()[0].lower() in selected_tags
    ]
    if not selected:
        raise ValueError("No GA variants selected.")
    return selected


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
    # ── GA pipeline comparison ─────────────────────────────────────────────
    parser.add_argument(
        "--compare-ga-full",
        action="store_true",
        dest="compare_ga_full",
        help=(
            "Compare GA variants inside the FULL pipeline (GA + PSO + SA). "
            "Tests 16 GAOperator configurations (force_change/adaptive/crossover/freq_bias), "
            "computes HV/IGD using src/analysis/pareto_metrics.py, and prints "
            "a summary table (mean+/-std across seeds). "
            "Results saved to results/ga_pipeline_comparison/. "
            "Use --seeds 0 1 2 for a quick 3-seed smoke test."
        ),
    )
    parser.add_argument(
        "--ga-variants",
        type=str,
        default="all",
        help=(
            "Subset of GA variants for --compare-ga-full. "
            "Examples: all, t1..t8, t9..t16, t1,t3,t10"
        ),
    )
    # Optional pipeline hyperparams (used only with --compare-ga-full)
    parser.add_argument("--c1",             type=float, default=0.5,
                        help="PSO personal attraction coefficient (default: 0.5).")
    parser.add_argument("--c2",             type=float, default=0.5,
                        help="PSO global attraction coefficient (default: 0.5).")
    parser.add_argument("--t0",             type=float, default=1.0,
                        help="SA initial temperature (default: 1.0).")
    parser.add_argument("--alpha",          type=float, default=0.95,
                        help="SA cooling rate (default: 0.95).")
    parser.add_argument("--k_directions",   type=int,   default=20,
                        help="MOEA/D weight directions K (default: 20).")
    parser.add_argument("--t_neighborhood", type=int,   default=2,
                        help="MOEA/D neighbourhood size T_n (default: 2).")
    parser.add_argument("--budget",         type=int,   default=2000,
                        help="NFE budget per run (default: 2000).")
    return parser.parse_args()


# ── GA pipeline comparison ─────────────────────────────────────────────────────

def run_ga_pipeline_comparison(
    eval_fn: Callable,
    seeds: list[int],
    budget: int,
    hardware: str,
    dataset: str,
    c1: float,
    c2: float,
    t0: float,
    alpha: float,
    k_directions: int,
    t_neighborhood: int,
    variants: list[tuple[str, str, dict]],
    log,
) -> None:
    """
    Compare 16 GAOperator variants (T1..T16) inside the full MemeticNAS pipeline
    (GA + PSO + SA) on real HW-NAS-Bench data.

    Metrics (src/analysis/pareto_metrics.py):
      - calc_hv  : normalized hypervolume, ref point (1.1, 1.1)  — higher is better
      - calc_igd : Inverted Generational Distance vs proxy front  — lower is better

    Proxy Pareto: non-dominated union of ALL variant archives (all seeds).
    Normalization:
        pts_min, pts_max = proxy.min(axis=0), proxy.max(axis=0)
        range_ = where(pts_max - pts_min > 0, pts_max - pts_min, 1.0)
        norm   = (pts - pts_min) / range_
    """
    from src.analysis.pareto_metrics import (
        archive_to_points,
        get_pareto_front,
        calc_hv,
        calc_igd,
    )
    import statistics

    GA_PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    W = 102
    log.info("=" * W)
    log.info("GA PIPELINE COMPARISON  (full GA + PSO + SA, real HW-NAS-Bench)")
    log.info("  Hardware : %-22s Dataset : %s", hardware, dataset)
    log.info("  Budget   : %d NFE/seed   Seeds : %s", budget, seeds)
    log.info("  PSO      : c1=%.2f  c2=%.2f", c1, c2)
    log.info("  SA       : T0=%.2f  alpha=%.2f", t0, alpha)
    log.info("  MOEA/D   : K=%d  T_neighborhood=%d", k_directions, t_neighborhood)
    log.info("  Variants : %d   Output: %s", len(variants), GA_PIPELINE_RESULTS_DIR)
    log.info("=" * W)

    # ── Phase 1: run all variants × seeds ─────────────────────────────────
    all_archives: dict[str, list[list[dict]]] = {
        name: [] for name, _, _ in variants
    }

    for var_name, file_prefix, ga_kwargs in variants:
        log.info("── Variant: %s", var_name)

        for seed in seeds:
            out_path = GA_PIPELINE_RESULTS_DIR / f"{file_prefix}_seed{seed}.json"

            if out_path.exists():
                log.info("  [SKIP] seed=%-3d — already exists: %s", seed, out_path.name)
                saved = json.loads(out_path.read_text())
                all_archives[var_name].append(saved["archive"])
                continue

            t_start = time.perf_counter()

            optimizer = MemeticNAS(
                eval_fn=eval_fn,
                budget=budget,
                candidate_ops=[
                    GAOperator(**ga_kwargs),
                    PSOOperator(c1=c1, c2=c2),
                ],
                sa_op=SAOperator(T0=t0, alpha=alpha),
                K=k_directions,
                T_neighborhood=t_neighborhood,
                use_restart=True,
                rng=np.random.default_rng(seed),
            )

            archive = optimizer.search()
            elapsed = time.perf_counter() - t_start

            # Quick pareto count for the log line
            pts = [(e["accuracy"], e["latency"]) for e in archive]
            pareto_size = sum(
                1 for a in pts
                if not any(
                    b[0] >= a[0] and b[1] <= a[1] and (b[0] > a[0] or b[1] < a[1])
                    for b in pts if b is not a
                )
            )

            save_archive(
                archive,
                out_path,
                metadata={
                    "experiment":     "ga_pipeline_comparison",
                    "variant":        var_name,
                    "ga_kwargs":      ga_kwargs,
                    "seed":           seed,
                    "hardware":       hardware,
                    "dataset":        dataset,
                    "budget_spent":   optimizer.budget_spent,
                    "K":              k_directions,
                    "T_neighborhood": t_neighborhood,
                    "c1":             c1,
                    "c2":             c2,
                    "t0":             t0,
                    "alpha":          alpha,
                },
            )
            all_archives[var_name].append(archive)

            log.info(
                "  seed=%-3d  NFE=%d  pareto_size=%d  %.1fs  → %s",
                seed, optimizer.budget_spent, pareto_size, elapsed, out_path.name,
            )

    # ── Phase 2: proxy Pareto from union of ALL archives ──────────────────
    log.info("Building proxy Pareto front from all variant archives …")

    union_archive: list[dict] = []
    for archives in all_archives.values():
        for archive in archives:
            union_archive.extend(archive)

    union_pts  = archive_to_points(union_archive)   # (N, 2) in (-acc, lat) space
    proxy_pts  = get_pareto_front(union_pts)        # (M, 2) non-dominated

    log.info("  Proxy front: %d points (from %d total evaluations)",
             len(proxy_pts), len(union_pts))

    # ── Phase 3: normalize using proxy bounds (project convention) ────────
    pts_min = proxy_pts.min(axis=0)
    pts_max = proxy_pts.max(axis=0)
    range_  = np.where(pts_max - pts_min > 0, pts_max - pts_min, 1.0)

    def _normalize(pts: np.ndarray) -> np.ndarray:
        return (pts - pts_min) / range_

    norm_proxy = _normalize(proxy_pts)
    ref_point  = np.array([1.1, 1.1])

    # ── Phase 4: compute HV and IGD per variant ───────────────────────────
    results_summary: list[dict] = []

    for var_name, _, _ in variants:
        hvs:          list[float] = []
        igds:         list[float] = []
        pareto_sizes: list[int]   = []

        for archive in all_archives[var_name]:
            pts       = archive_to_points(archive)
            front_pts = get_pareto_front(pts)
            pareto_sizes.append(len(front_pts))

            if len(front_pts) == 0:
                hvs.append(0.0)
                igds.append(float("inf"))
                continue

            norm_front = _normalize(front_pts)
            hvs.append(float(calc_hv(norm_front, ref_point)))
            igds.append(float(calc_igd(norm_front, norm_proxy)))

        results_summary.append({
            "name":        var_name,
            "hv_mean":     statistics.mean(hvs)          if hvs          else 0.0,
            "hv_std":      statistics.stdev(hvs)         if len(hvs) > 1 else 0.0,
            "igd_mean":    statistics.mean(igds)         if igds         else float("inf"),
            "igd_std":     statistics.stdev(igds)        if len(igds) > 1 else 0.0,
            "pareto_mean": statistics.mean(pareto_sizes) if pareto_sizes  else 0.0,
            "n_seeds":     len(hvs),
        })

    # ── Phase 5: print comparison table ───────────────────────────────────
    best_hv_name  = max(results_summary, key=lambda r: r["hv_mean"])["name"]
    best_igd_name = min(results_summary, key=lambda r: r["igd_mean"])["name"]
    baseline_hv   = results_summary[0]["hv_mean"]

    print("\n" + "=" * W)
    print("GA VARIANT COMPARISON — Full Pipeline (GA + PSO + SA) | Real HW-NAS-Bench")
    print(f"  Hardware: {hardware}   Dataset: {dataset}   "
          f"Seeds: {len(seeds)}   Budget: {budget} NFE")
    print(f"  PSO: c1={c1}  c2={c2}   SA: T0={t0}  alpha={alpha}   "
          f"MOEA/D: K={k_directions}  T_n={t_neighborhood}")
    print("=" * W)
    print("  Metrics:")
    print("    HV  (higher is better) — normalized hypervolume, ref point (1.1, 1.1)")
    print("    IGD (lower  is better) — Inverted Generational Distance vs proxy front")
    print("    Proxy front = non-dominated union of ALL variant archives (all seeds)")
    print("-" * W)
    hdr = (
        f"  {'Variant':<38} "
        f"{'HV mean':>9} {'+-std':>6}  "
        f"{'Delta HV':>9}  "
        f"{'IGD mean':>9} {'+-std':>6}  "
        f"{'Pareto':>7}  "
        f"{'Seeds':>5}"
    )
    print(hdr)
    print("  " + "-" * (W - 2))

    for r in results_summary:
        delta     = r["hv_mean"] - baseline_hv
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}" if delta != 0 else "  (base)"
        markers   = ""
        if r["name"] == best_hv_name:
            markers += " [BEST HV]"
        if r["name"] == best_igd_name:
            markers += " [BEST IGD]"
        print(
            f"  {r['name']:<38} "
            f"{r['hv_mean']:>9.4f} {r['hv_std']:>6.4f}  "
            f"{delta_str:>9}  "
            f"{r['igd_mean']:>9.4f} {r['igd_std']:>6.4f}  "
            f"{r['pareto_mean']:>7.1f}  "
            f"{r['n_seeds']:>5}"
            f"  {markers}"
        )

    print("  " + "-" * (W - 2))
    print(f"  Best HV  -> {best_hv_name}")
    print(f"  Best IGD -> {best_igd_name}")
    print("=" * W)

    # ── Phase 6: build HV convergence curves ─────────────────────────────
    checkpoints = list(range(100, budget + 1, 100))
    if checkpoints[-1] != budget:
        checkpoints.append(budget)

    hv_trends: dict[str, dict[str, list[float]]] = {}
    for var_name, _, _ in variants:
        curves: list[list[float]] = []
        for archive in all_archives[var_name]:
            curve: list[float] = []
            for nfe in checkpoints:
                n = min(nfe, len(archive))
                pts = archive_to_points(archive[:n])
                front_pts = get_pareto_front(pts)
                if len(front_pts) == 0:
                    curve.append(0.0)
                    continue
                norm_front = _normalize(front_pts)
                curve.append(float(calc_hv(norm_front, ref_point)))
            curves.append(curve)

        if curves:
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
        else:
            mean_curve = np.zeros(len(checkpoints))
            std_curve = np.zeros(len(checkpoints))
        hv_trends[var_name] = {
            "mean": mean_curve.tolist(),
            "std": std_curve.tolist(),
        }

    _plot_ga_pipeline_results(
        results_summary,
        hv_trends,
        checkpoints,
        GA_PIPELINE_RESULTS_DIR,
        hardware,
        dataset,
    )

    # Helpful next-step hint
    best_kwargs = next(kw for n, _, kw in variants if n == best_hv_name)
    kwargs_str  = ", ".join(f"{k}={v}" for k, v in best_kwargs.items())
    print(f"\nNext steps:")
    print(f"  1. Plug the winning GA variant into run_search.py:")
    print(f"       candidate_ops=[GAOperator({kwargs_str}), PSOOperator(c1={c1}, c2={c2})]")
    print(f"  2. Run: python scripts/run_search.py --hardware {hardware}")
    print(f"  3. Results saved to: {GA_PIPELINE_RESULTS_DIR}")


def _plot_ga_pipeline_results(
    results_summary: list[dict],
    hv_trends: dict[str, dict[str, list[float]]],
    checkpoints: list[int],
    output_dir: Path,
    hardware: str,
    dataset: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot.")
        return

    names = [r["name"] for r in results_summary]
    short_names = [n.split()[0] for n in names]
    hv_vals = [r["hv_mean"] for r in results_summary]
    hv_errs = [r["hv_std"] for r in results_summary]
    igd_vals = [r["igd_mean"] for r in results_summary]
    igd_errs = [r["igd_std"] for r in results_summary]

    best_hv_idx = int(np.argmax(hv_vals)) if hv_vals else 0
    best_igd_idx = int(np.argmin(igd_vals)) if igd_vals else 0

    hv_colors = ["#94a3b8"] * len(names)
    igd_colors = ["#94a3b8"] * len(names)
    if names:
        hv_colors[best_hv_idx] = "#22c55e"
        igd_colors[best_igd_idx] = "#ef4444"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.bar(range(len(names)), hv_vals, yerr=hv_errs, capsize=3, color=hv_colors)
    ax.set_title("HV (higher is better)")
    ax.set_ylabel("HV mean")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.bar(range(len(names)), igd_vals, yerr=igd_errs, capsize=3, color=igd_colors)
    ax2.set_title("IGD (lower is better)")
    ax2.set_ylabel("IGD mean")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(short_names, rotation=0, ha="center", fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"GA Variant Comparison — {hardware} | {dataset}", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ga_pipeline_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[Plot] Saved: {out_path}")

    # ── Convergence plot: HV vs NFE ─────────────────────────────────────
    if not checkpoints:
        return

    fig2, ax = plt.subplots(1, 1, figsize=(9, 4.6))
    cmap = plt.get_cmap("tab10")

    all_means = []
    for stats in hv_trends.values():
        all_means.extend(stats.get("mean", []))
    if all_means:
        y_min = min(all_means)
        y_max = max(all_means)
        pad = max(0.005, (y_max - y_min) * 0.12)
        ax.set_ylim(max(0.0, y_min - pad), min(1.1, y_max + pad))

    for i, (name, stats) in enumerate(hv_trends.items()):
        mean = np.array(stats["mean"], dtype=float)
        std = np.array(stats["std"], dtype=float)
        color = cmap(i % 10)
        ax.plot(checkpoints, mean, label=name.split()[0], color=color, linewidth=1.6)
        ax.fill_between(checkpoints, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_title("HV convergence over evaluations")
    ax.set_xlabel("NFE (evaluations)")
    ax.set_ylabel("Normalized HV")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="best", ncol=2, frameon=False)

    fig2.suptitle(f"GA Variant Convergence — {hardware} | {dataset}", fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path2 = output_dir / "ga_pipeline_convergence.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"[Plot] Saved: {out_path2}")

    # ── Focused convergence plot for top variants ───────────────────────
    if not results_summary:
        return
    top_k = 6
    sorted_by_hv = sorted(results_summary, key=lambda r: r["hv_mean"], reverse=True)
    top_names = [r["name"] for r in sorted_by_hv[:top_k]]
    best_igd = min(results_summary, key=lambda r: r["igd_mean"])["name"]
    if best_igd not in top_names:
        top_names.append(best_igd)

    fig3, ax3 = plt.subplots(1, 1, figsize=(9, 4.6))
    cmap = plt.get_cmap("tab10")

    all_means = []
    for name in top_names:
        stats = hv_trends.get(name, {})
        all_means.extend(stats.get("mean", []))
    if all_means:
        y_min = min(all_means)
        y_max = max(all_means)
        pad = max(0.005, (y_max - y_min) * 0.12)
        ax3.set_ylim(max(0.0, y_min - pad), min(1.1, y_max + pad))

    for i, name in enumerate(top_names):
        stats = hv_trends[name]
        mean = np.array(stats["mean"], dtype=float)
        std = np.array(stats["std"], dtype=float)
        color = cmap(i % 10)
        ax3.plot(checkpoints, mean, label=name.split()[0], color=color, linewidth=2.0)
        ax3.fill_between(checkpoints, mean - std, mean + std, color=color, alpha=0.18)

    ax3.set_title("HV convergence (top variants)")
    ax3.set_xlabel("NFE (evaluations)")
    ax3.set_ylabel("Normalized HV")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=8, loc="best", ncol=2, frameon=False)

    fig3.suptitle(f"GA Convergence — Top Variants | {hardware} | {dataset}", fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path3 = output_dir / "ga_pipeline_convergence_top.png"
    fig3.savefig(out_path3, dpi=150)
    plt.close(fig3)
    print(f"[Plot] Saved: {out_path3}")


# ── Original helpers (unchanged) ──────────────────────────────────────────────

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


def _build_optimizer(
    eval_fn: Callable[[np.ndarray], tuple[float, float]],
    cfg: dict,
    rng: np.random.Generator,
):
    """Build a concrete optimizer from an ablation config dict."""
    if cfg.get("experiment_name") == "no_moead":
        K = 1
        T_neighborhood = 1
    else:
        K = cfg.get("k_directions", 5)
        T_neighborhood = cfg.get("t_neighborhood", 2)

    candidate_ops = _resolve_candidate_operators(cfg)
    sa_op = (
        SAOperator(T0=cfg.get("t0", 1.0), alpha=cfg.get("alpha", 0.95))
        if cfg.get("use_sa", True) else None
    )

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
    args = parse_args()
    log  = get_logger(
        "run_ablations",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # ── Load benchmark API (shared across all modes) ──────────────────────
    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    if not DATA_PATH.exists():
        log.error("Benchmark file not found: %s", DATA_PATH)
        log.error("Run: python scripts/download_data.py")
        sys.exit(1)

    api     = HWNASApi(str(DATA_PATH))
    eval_fn = _build_eval_fn(api, args.hardware, args.dataset)

    # ── GA pipeline comparison mode ───────────────────────────────────────
    if args.compare_ga_full:
        try:
            variants = _select_ga_variants(args.ga_variants)
        except ValueError as exc:
            log.error("Invalid --ga-variants: %s", exc)
            sys.exit(2)
        run_ga_pipeline_comparison(
            eval_fn=eval_fn,
            seeds=args.seeds,
            budget=args.budget,
            hardware=args.hardware,
            dataset=args.dataset,
            c1=args.c1,
            c2=args.c2,
            t0=args.t0,
            alpha=args.alpha,
            k_directions=args.k_directions,
            t_neighborhood=args.t_neighborhood,
            variants=variants,
            log=log,
        )
        return  # do not run normal ablation loop

    # ── Normal ablation mode (unchanged from original) ────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = _load_configs(args.config)
    log.info(
        "Found %d experiment configuration(s).  Seeds: %s",
        len(configs), args.seeds,
    )

    total_runs = len(configs) * len(args.seeds)
    completed  = 0

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
            t0_wall = time.perf_counter()

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
            elapsed = time.perf_counter() - t0_wall

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
                "  Saved -> %s  (evals=%d, pareto_size=%d, %.1fs)",
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