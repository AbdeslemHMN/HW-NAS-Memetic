"""
Pure mathematical utilities for multi-objective Pareto analysis.

No file I/O.  No matplotlib.  All functions operate on NumPy arrays and are
safe to import from any context (scripts, notebooks, tests).

Convention (inherited from pareto_metrics.py):
  F[:, 0] = -Accuracy  (minimise, so higher accuracy → smaller value)
  F[:, 1] =  Latency   (ms, minimise)

Display coordinates used by plotting helpers:
  (accuracy%, latency)  where accuracy% = -F[:, 0]
"""
from __future__ import annotations

import numpy as np

from src.analysis.pareto_metrics import archive_to_points, get_pareto_front


# ── Knee-point detection ──────────────────────────────────────────────────────

def find_knee(front: np.ndarray) -> int:
    """Index of the knee point on a sorted 2-D Pareto front.

    Selects the point with the maximum perpendicular distance from the
    straight line that joins the two extreme endpoints of the front,
    computed in normalised [0, 1] × [0, 1] space.

    :param front: (K, 2) array in display space (accuracy%, latency),
                  sorted by ``front[:, 0]`` ascending.
    :return:      Index of the knee point in *front*.
    """
    if len(front) < 2:
        return 0

    a_min, a_max = float(front[:, 0].min()), float(front[:, 0].max())
    l_min, l_max = float(front[:, 1].min()), float(front[:, 1].max())

    if a_max == a_min or l_max == l_min:
        return len(front) // 2

    na = (front[:, 0] - a_min) / (a_max - a_min)
    nl = (front[:, 1] - l_min) / (l_max - l_min)

    d = np.array([na[-1] - na[0], nl[-1] - nl[0]], dtype=np.float64)
    nd = float(np.linalg.norm(d))
    if nd < 1e-10:
        return len(front) // 2
    d = d / nd

    dists = [
        abs(float(np.cross(d, np.array([na[i] - na[0], nl[i] - nl[0]]))))
        for i in range(len(front))
    ]
    return int(np.argmax(dists))


# ── Step-function interpolation ───────────────────────────────────────────────

def step_interp(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """Step-function (post) interpolation on a sorted Pareto front.

    Given a staircase front defined by *(x_pts, y_pts)*, returns the
    y-value that would be "read" at each point in *x_query* when the
    function is treated as a right-continuous step function.

    :param x_pts:   Sorted x-values of the front (e.g., accuracy%).
    :param y_pts:   Corresponding y-values (e.g., latency ms).
    :param x_query: 1-D array of query x-values.
    :return:        1-D array of interpolated y-values.
    """
    idx = np.searchsorted(x_pts, x_query, side="right") - 1
    return y_pts[np.clip(idx, 0, len(y_pts) - 1)]


# ── Per-seed front (EAF building block) ──────────────────────────────────────

def seed_front(archive: list) -> np.ndarray:
    """Non-dominated front for a single seed archive in display coordinates.

    :param archive: List of evaluation dicts with ``'accuracy'`` and
                    ``'latency'`` keys (raw output of the search algorithm).
    :return:        (K, 2) array of (accuracy%, latency) sorted by accuracy.
                    Returns an empty (0, 2) array if the archive is empty.
    """
    pts = archive_to_points(archive)
    if not len(pts):
        return np.empty((0, 2))

    acc = -pts[:, 0]   # convert to display (higher is better → ascending)
    lat = pts[:, 1]
    order = np.argsort(-acc)   # descending accuracy
    acc, lat = acc[order], lat[order]

    fa: list[float] = []
    fl: list[float] = []
    min_l = np.inf
    for a, l in zip(acc, lat):
        if l < min_l:
            min_l = l
            fa.append(float(a))
            fl.append(float(l))

    if not fa:
        return np.empty((0, 2))

    result = np.column_stack((fa, fl))
    return result[np.argsort(result[:, 0])]


# ── EAF attainment ────────────────────────────────────────────────────────────

def attained_latency(front: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Minimum latency attained by a front at each accuracy in *x_grid*.

    For a given accuracy threshold *x*, the attained latency is the
    minimum latency achieved by any architecture on the front with at
    least that accuracy.  Used to build Empirical Attainment Surfaces.

    :param front:  (K, 2) array in (accuracy%, latency) display space.
    :param x_grid: 1-D array of accuracy query points.
    :return:       1-D array of attained latencies (``NaN`` where no
                   front point has accuracy ≥ the query threshold).
    """
    y = np.full(len(x_grid), np.nan)
    for i, x in enumerate(x_grid):
        valid = front[front[:, 0] >= x]
        if len(valid):
            y[i] = float(valid[:, 1].min())
    return y


# ── Normalisation helpers ─────────────────────────────────────────────────────

def compute_bounds(
    reference_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive normalisation bounds from a reference point set (e.g., proxy P*).

    :param reference_pts: (N, M) array of objective values (any space).
    :return: ``(min_vals, range_vals)`` each of shape ``(M,)``.
             ``range_vals[j]`` is at least ``1.0`` to prevent division by zero.
    """
    min_vals = reference_pts.min(axis=0)
    max_vals = reference_pts.max(axis=0)
    range_vals = np.where(max_vals - min_vals > 0.0, max_vals - min_vals, 1.0)
    return min_vals, range_vals


def normalise_to_bounds(
    pts: np.ndarray,
    ref_min: np.ndarray,
    ref_range: np.ndarray,
) -> np.ndarray:
    """Map *pts* to [0, 1] using pre-computed normalisation bounds.

    :param pts:       (N, M) objective array to normalise.
    :param ref_min:   (M,) per-objective minimum from :func:`compute_bounds`.
    :param ref_range: (M,) per-objective range   from :func:`compute_bounds`.
    :return:          (N, M) array with values in [0, 1] (approximately).
    """
    return (pts - ref_min) / ref_range


def get_non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of an objective point cloud."""
    if points.ndim != 2 or len(points) == 0:
        return points
    return get_pareto_front(points)


def find_knee_point(front: np.ndarray) -> int:
    """Alias for :func:`find_knee` with a clearer public interface."""
    return find_knee(front)


def empirical_attainment_surface(fronts: list[np.ndarray], x_grid: np.ndarray) -> np.ndarray:
    """Compute the empirical attainment surface for a list of fronts."""
    if not fronts:
        return np.empty((0, len(x_grid)))
    curves = [attained_latency(front, x_grid) for front in fronts]
    return np.vstack(curves)


def to_display(front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a minimisation front to display coordinates (accuracy%, latency)."""
    acc = -front[:, 0]
    lat = front[:, 1]
    order = np.argsort(acc)
    return acc[order], lat[order]
