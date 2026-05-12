"""
Result loading and path-resolution utilities for HW-NAS-Memetic.

All filesystem I/O for experiment results lives here.
Pure mathematical logic lives in ``src.utils.pareto_math``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from src.analysis.pareto_metrics import (
    archive_to_points, get_pareto_front,
    calc_hv, calc_igd, calc_spacing, calc_max_spread,
)
from src.utils.data_processing import (
    get_project_root,
    load_archives_glob as _load_archives_glob,
    load_runs_glob as _load_runs_glob,
    load_archive as _load_single_archive,
    load_incremental_archive as _load_single_incremental_archive,
)


# ── Project-wide constants ────────────────────────────────────────────────────

DATASETS: list[str] = ["cifar10", "cifar100", "ImageNet16-120"]
HARDWARE: list[str] = ["edgegpu_latency", "raspi4_latency", "eyeriss_latency"]

ALGO_SPEC: dict[str, tuple[str, str]] = {
    "Proposed":      ("proposed",  "proposed_res_seed*.json"),
    "NSGA-II":       ("baselines", "nsga2_seed*.json"),
    "Random Search": ("baselines", "random_search_seed*.json"),
}


# ── Data-processing wrappers ─────────────────────────────────────────────────

def load_archives_glob(
    results_dir: Path,
    glob_pattern: str,
) -> list[list[dict]]:
    return _load_archives_glob(results_dir, glob_pattern)


def load_runs_glob(
    results_dir: Path,
    glob_pattern: str,
) -> list[dict]:
    return _load_runs_glob(results_dir, glob_pattern)

def load_archive(results_dir: Path, fname: str) -> list[dict]:
    return _load_single_archive(results_dir, fname)


def load_incremental_archive(
    results_base: Path,
    dataset: str,
    hardware: str,
    fname: str,
) -> list[dict]:
    return _load_single_incremental_archive(results_base, dataset, hardware, fname)


# ── Structured loaders ────────────────────────────────────────────────────────

def combo_archive_sets(
    results_dir: Path,
    dataset: str,
    hardware: str,
    algo_spec: dict | None = None,
) -> dict[str, list[list[dict]]]:
    """Load archives for every algorithm for one dataset/hardware combination.

    :param results_dir: Project results root (e.g., ``PROJECT_ROOT / 'results'``).
    :param dataset:     One of :data:`DATASETS`.
    :param hardware:    One of :data:`HARDWARE`.
    :param algo_spec:   Algorithm spec dict; defaults to :data:`ALGO_SPEC`.
    :return:            ``{algo_name: [seed_archive, ...]}`` for each algorithm.
    """
    spec = algo_spec if algo_spec is not None else ALGO_SPEC
    result: dict[str, list[list[dict]]] = {}
    for algo, (folder, pattern) in spec.items():
        base_dir = results_dir / folder / dataset / hardware
        result[algo] = load_archives_glob(base_dir, pattern) if base_dir.exists() else []
    return result


def load_ablation_archives(
    ablation_dir: Path,
    exp_name: str,
) -> list[list[dict]]:
    """Load all seed archives for one ablation experiment variant.

    :param ablation_dir: Directory containing ablation result JSON files.
    :param exp_name:     Variant name prefix (e.g., ``"no_pso"``).
    :return:             List of per-seed archives.
    """
    return load_archives_glob(ablation_dir, f"{exp_name}_seed*.json")


def load_incremental_archive(
    results_base: Path,
    dataset: str,
    hardware: str,
    fname: str,
) -> list[dict]:
    """Load one stage archive from the incremental-build results.

    :param results_base: Base directory for incremental results.
    :param dataset:      Dataset name.
    :param hardware:     Hardware target name.
    :param fname:        Stage filename (e.g., ``"stage1_random.json"``).
    :raises FileNotFoundError: If the stage file does not exist.
    :return:             Archive (list of evaluation dicts).
    """
    path = results_base / dataset / hardware / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            f"Run: python scripts/run_incremental_build.py "
            f"--dataset {dataset} --hardware {hardware} --seed 42"
        )
    return json.loads(path.read_text())["archive"]


# ── Pareto helpers (require loaded archives) ──────────────────────────────────

def aggregate_front(archives: list[list[dict]]) -> np.ndarray:
    """Pool all seed archives into a single global non-dominated front.

    All seed archives are merged into one point cloud, then the
    non-dominated subset is extracted.  This gives a single,
    representative "best-possible" front across all runs.

    :param archives: List of per-seed archives (each a list of eval dicts).
    :return:         (N, 2) array of (accuracy%, latency) on the aggregate front.
                     Returns empty (0, 2) array if *archives* is empty.
    """
    if not archives:
        return np.empty((0, 2))
    all_pts = np.vstack([archive_to_points(a) for a in archives])
    front = get_pareto_front(all_pts)
    return np.column_stack([-front[:, 0], front[:, 1]])


def focus_range(
    sets: dict[str, list[list[dict]]],
    focus_algos: tuple[str, ...] = ("Proposed", "NSGA-II"),
) -> tuple[float, float, float]:
    """Compute display axis limits tight around the competitive algorithms.

    Random Search typically produces far worse fronts; including it in the
    axis calculation would compress the interesting region.  This function
    uses only *focus_algos* for the limits and falls back to all algorithms
    if none of the focus algorithms have data.

    :param sets:        Output of :func:`combo_archive_sets`.
    :param focus_algos: Algorithm names to include in the zoom calculation.
    :return:            ``(x_lo, x_hi, y_hi)`` in (accuracy%, latency) units,
                        with small padding margins already applied.
    """
    acc: list[float] = []
    lat: list[float] = []

    for algo in focus_algos:
        archives = sets.get(algo, [])
        if not archives:
            continue
        front = aggregate_front(archives)
        if len(front):
            acc.extend(front[:, 0].tolist())
            lat.extend(front[:, 1].tolist())

    if not acc:
        for archives in sets.values():
            for archive in archives:
                pts = archive_to_points(archive)
                if len(pts):
                    acc.extend((-pts[:, 0]).tolist())
                    lat.extend(pts[:, 1].tolist())

    if not acc:
        return 0.0, 100.0, 10.0

    span_x = max(max(acc) - min(acc), 0.5)
    margin_x = span_x * 0.07
    margin_y = max(lat) * 0.14
    return min(acc) - margin_x, max(acc) + margin_x, max(lat) + margin_y


def to_display(front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Pareto front from pymoo minimisation space to display space.

    :param front: (K, 2) array in (−accuracy, latency) minimisation space.
    :return:      ``(acc, lat)`` — both arrays sorted by accuracy ascending.
    """
    order = np.argsort(-front[:, 0])   # ascending -f0 = ascending accuracy
    return -front[order, 0], front[order, 1]


def get_arch_fitness(api, arch: np.ndarray, dataset: str) -> float:
    """Return a query accuracy score for a candidate architecture."""
    return api.query_accuracy(arch, dataset=dataset)


def per_seed_metrics(
    archives: list[list[dict]],
    p_star: np.ndarray,
    normalise: callable,
    ref_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hvs: list[float] = []
    igds: list[float] = []
    for run in archives:
        pts = archive_to_points(run)
        front = get_pareto_front(pts)
        norm_front = normalise(front)
        hvs.append(calc_hv(norm_front, ref_point=ref_point))
        igds.append(calc_igd(norm_front, normalise(p_star)))
    return np.array(hvs), np.array(igds)


def per_seed_all_metrics(
    archives: list[list[dict]],
    p_star: np.ndarray,
    normalise: callable,
    ref_point: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute HV, IGD, Spacing, and MaxSpread for every seed archive.

    :param archives:   List of per-seed archives.
    :param p_star:     Proxy Pareto front in minimisation space ``(K, 2)``.
    :param normalise:  Callable that maps ``(N, 2)`` raw points → ``[0,1]^2``.
    :param ref_point:  Reference point for HV (e.g. ``np.array([1.1, 1.1])``).
    :return:           Dict with keys ``"hv"``, ``"igd"``, ``"spacing"``,
                       ``"max_spread"``; each a 1-D ``np.ndarray`` over seeds.
    """
    hvs, igds, spacings, spreads = [], [], [], []
    p_star_norm = normalise(p_star)
    for run in archives:
        pts = archive_to_points(run)
        front = get_pareto_front(pts)
        nf = normalise(front)
        hvs.append(calc_hv(nf, ref_point=ref_point))
        igds.append(calc_igd(nf, p_star_norm))
        spacings.append(calc_spacing(nf))
        spreads.append(calc_max_spread(nf))
    return {
        "hv":         np.array(hvs),
        "igd":        np.array(igds),
        "spacing":    np.array(spacings),
        "max_spread": np.array(spreads),
    }


def compute_trajectory(
    runs: list[dict],
    normalise: callable,
    ref_point: np.ndarray,
) -> list[np.ndarray]:
    trajectories: list[np.ndarray] = []
    for run in runs:
        checkpoints = run.get("metadata", {}).get("nfe_checkpoints", [])
        if not checkpoints:
            continue
        pts_seq: list[tuple[int, float]] = []
        for nfe in checkpoints:
            slice_ = run["archive"][:nfe]
            if not slice_:
                continue
            pts = archive_to_points(slice_)
            front = get_pareto_front(pts)
            norm_front = normalise(front)
            hv = calc_hv(norm_front, ref_point=ref_point)
            pts_seq.append((nfe, hv))
        if pts_seq:
            trajectories.append(np.asarray(pts_seq, dtype=np.float64))
    return trajectories


def best_archs_table(archives: list[list[dict]], label: str, n: int = 5) -> None:
    OPS = ["none", "skip_connect", "avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3"]
    if not archives:
        return
    all_pts = np.vstack([archive_to_points(a) for a in archives])
    all_raw = [e for run in archives for e in run]
    front_idx = find_non_dominated(all_pts)

    front_pts = all_pts[front_idx]
    front_raw = [all_raw[i] for i in front_idx]

    order = np.argsort(front_pts[:, 0])
    rows_t = []
    for rank, i in enumerate(order[:n], 1):
        e = front_raw[i]
        ops_str = " | ".join(OPS[o] for o in e["arch"])
        rows_t.append({
            "Rank": rank,
            "Accuracy (%)": f"{e['accuracy']:.2f}",
            "Latency (ms)": f"{e['latency']:.4f}",
            "Architecture": ops_str,
        })

    print(f"\n  TOP-{n} PARETO ARCHITECTURES — {label}")
    print(pd.DataFrame(rows_t).set_index("Rank").to_string())
