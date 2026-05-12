"""Epistasis / edge-perturbation analysis for NAS-Bench-201 architectures.

This module provides all functions needed to measure how sensitive each of
the 6 cell-graph edges is when its operation is replaced with an alternative.
Results are used to visualise the architectural backbone discovered by the
Memetic search algorithm.
"""

from __future__ import annotations

import time
import numpy as np

from src.analysis.pareto_metrics import archive_to_points, get_pareto_front

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ── Search-space constants ─────────────────────────────────────────────────────
N_OPS: int = 5
N_EDGES: int = 6
OPS_NAMES: list[str] = [
    "none",
    "skip_connect",
    "avg_pool_3x3",
    "nor_conv_1x1",
    "nor_conv_3x3",
]


# ── Core perturbation metric ───────────────────────────────────────────────────

def calculate_edge_sensitivity(
    arch: np.ndarray,
    dataset: str,
    api,
    *,
    verbose: bool = False,
    arch_label: str = "",
) -> np.ndarray:
    """Compute the per-operation, per-edge accuracy drop matrix for one architecture.

    For every edge *e* and every alternative operation *o* (excluding the
    original operation), the architecture is perturbed and the resulting
    accuracy drop is recorded.

    Parameters
    ----------
    arch:
        Integer array of shape ``(N_EDGES,)`` — the architecture encoding.
    dataset:
        One of ``{"cifar10", "cifar100", "ImageNet16-120"}``.
    api:
        A ``HWNASApi`` instance used for ground-truth accuracy look-up.
    verbose:
        Print per-edge progress to stdout.
    arch_label:
        Short label printed in verbose mode to identify the architecture.

    Returns
    -------
    delta : np.ndarray, shape ``(N_OPS, N_EDGES)``
        ``delta[op, edge]`` = accuracy(original) − accuracy(perturbed).
        Self-replacement entries (op == original op) are ``NaN``.
    """
    arch = np.asarray(arch, dtype=np.int64)
    assert arch.shape == (N_EDGES,), f"Expected shape ({N_EDGES},), got {arch.shape}"

    t0 = time.perf_counter()
    f_orig = api.query_accuracy(arch, dataset=dataset)
    delta = np.full((N_OPS, N_EDGES), np.nan, dtype=np.float64)

    edges = range(N_EDGES)
    if verbose and _HAS_TQDM:
        edges = _tqdm(edges, desc=f"  edges {arch_label}", leave=False, ncols=72)

    for edge in edges:
        orig_op = int(arch[edge])
        for op in range(N_OPS):
            if op == orig_op:
                continue
            perturbed = arch.copy()
            perturbed[edge] = op
            f_pert = api.query_accuracy(perturbed, dataset=dataset)
            delta[op, edge] = f_orig - f_pert

    if verbose and not _HAS_TQDM:
        elapsed = time.perf_counter() - t0
        label = f" [{arch_label}]" if arch_label else ""
        print(f"    arch{label}: f_orig={f_orig:.2f}%  done in {elapsed:.1f}s")

    return delta


def aggregate_sensitivity(
    archs: list[np.ndarray],
    dataset: str,
    api,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average the edge-sensitivity matrix over a set of architectures.

    Parameters
    ----------
    archs:
        List of architecture encodings, each shape ``(N_EDGES,)``.
    dataset:
        Dataset name passed to ``calculate_edge_sensitivity``.
    api:
        A ``HWNASApi`` instance.
    verbose:
        Print per-architecture progress and timing to stdout.

    Returns
    -------
    mean_delta : np.ndarray, shape ``(N_OPS, N_EDGES)``
        NaN-mean over all architectures.
    stack : np.ndarray, shape ``(len(archs), N_OPS, N_EDGES)``
        Individual delta matrices before averaging.
    """
    if not archs:
        empty = np.full((N_OPS, N_EDGES), np.nan)
        return empty, np.empty((0, N_OPS, N_EDGES))

    n = len(archs)
    results = []
    t_total = time.perf_counter()

    arch_iter = enumerate(archs)
    if verbose and _HAS_TQDM:
        arch_iter = _tqdm(enumerate(archs), total=n, desc="  archs", ncols=72)

    for i, arch in arch_iter:
        t_arch = time.perf_counter()
        label = f"{i+1}/{n} arch={list(arch)}"
        delta = calculate_edge_sensitivity(
            arch, dataset, api,
            verbose=verbose and not _HAS_TQDM,
            arch_label=label,
        )
        results.append(delta)
        if verbose and not _HAS_TQDM:
            elapsed = time.perf_counter() - t_arch
            total_so_far = time.perf_counter() - t_total
            eta = (total_so_far / (i + 1)) * (n - i - 1)
            print(
                f"    [{i+1}/{n}] arch={list(arch)}  "
                f"{elapsed:.1f}s/arch  ETA {eta:.0f}s"
            )

    stack = np.stack(results, axis=0)
    with np.errstate(all="ignore"):
        mean_delta = np.nanmean(stack, axis=0)

    if verbose:
        total = time.perf_counter() - t_total
        print(f"  → aggregate done: {n} archs in {total:.1f}s ({total/n:.1f}s/arch)")

    return mean_delta, stack


# ── Architecture selectors ─────────────────────────────────────────────────────

def select_pareto_archs(
    archives: list[list[dict]],
    top_k: int = 10,
) -> list[np.ndarray]:
    """Return the top-*k* unique architectures ranked by accuracy.

    This is a fallback selector used when the strict Pareto front contains
    fewer than the required minimum number of architectures.

    Parameters
    ----------
    archives:
        List of per-seed run archives; each archive is a list of evaluation
        dicts with at least ``"arch"`` and ``"accuracy"`` keys.
    top_k:
        Maximum number of unique architectures to return.

    Returns
    -------
    list of np.ndarray, each shape ``(N_EDGES,)``
    """
    all_entries = [e for run in archives for e in run]
    all_entries.sort(key=lambda e: e["accuracy"], reverse=True)

    seen: set[tuple[int, ...]] = set()
    selected: list[np.ndarray] = []
    for entry in all_entries:
        key = tuple(int(x) for x in entry["arch"])
        if key not in seen:
            seen.add(key)
            selected.append(np.array(entry["arch"], dtype=np.int64))
        if len(selected) >= top_k:
            break

    return selected


def select_pareto_front_archs(archives: list[list[dict]]) -> list[np.ndarray]:
    """Return unique architecture vectors that lie on the aggregate Pareto front.

    Unlike ``aggregate_front`` in ``data_loader`` (which returns coordinate
    points for plotting), this function returns the actual arch encodings so
    they can be passed to the perturbation analysis.

    Parameters
    ----------
    archives:
        List of per-seed run archives.

    Returns
    -------
    list of np.ndarray, each shape ``(N_EDGES,)``
    """
    all_entries = [e for run in archives for e in run]
    if not all_entries:
        return []

    pts = archive_to_points(all_entries)
    front_pts = get_pareto_front(pts)

    selected: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for fp in front_pts:
        for e, p in zip(all_entries, pts):
            if np.allclose(p, fp):
                key = tuple(int(x) for x in e["arch"])
                if key not in seen:
                    seen.add(key)
                    selected.append(np.array(e["arch"], dtype=np.int64))
                break

    return selected


# ── Marginal summary statistics ────────────────────────────────────────────────

def edge_brittleness(mean_delta: np.ndarray) -> np.ndarray:
    """Column mean of the delta matrix — how critical each edge is overall.

    Parameters
    ----------
    mean_delta : shape ``(N_OPS, N_EDGES)``

    Returns
    -------
    np.ndarray, shape ``(N_EDGES,)``
    """
    return np.nanmean(mean_delta, axis=0)


def op_disruption(mean_delta: np.ndarray) -> np.ndarray:
    """Row mean of the delta matrix — how damaging each replacement op is overall.

    Parameters
    ----------
    mean_delta : shape ``(N_OPS, N_EDGES)``

    Returns
    -------
    np.ndarray, shape ``(N_OPS,)``
    """
    return np.nanmean(mean_delta, axis=1)
