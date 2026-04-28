"""
Pareto-quality metrics for HW-NAS-Memetic.

Objective space convention (minimisation, consistent with pymoo):
  F[:, 0] = -Accuracy   (negate so higher accuracy -> lower value)
  F[:, 1] =  Latency    (ms, lower is better)

Research Insights:
- Hypervolume (HV): volume of objective space dominated by the front and
  bounded by a reference point r.
  Larger HV is strictly better; sensitive to ref_point choice — normalise
  each axis to [0, 1] before comparison across devices/datasets.
- IGD (Inverted Generational Distance): mean distance from each point of
  the true (proxy) Pareto front P* to the nearest point in the approximation
  set F.
  Lower IGD is better; zero means perfect convergence to P*.
- Proxy Pareto: when the true front P* is unknown (hidden hardware), the
  non-dominated union of *all* algorithm runs serves as an unbiased proxy.
"""
from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import find_non_dominated


def archive_to_points(archive: list[dict]) -> np.ndarray:
    """
    Convert an experiment archive to a (N, 2) float64 array in pymoo's
    minimisation convention: F = [-accuracy, latency].
    """
    pts = np.array(
        [[-e["accuracy"], e["latency"]] for e in archive], dtype=np.float64
    )
    return pts


def get_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Extract the non-dominated subset of *points* (N, M) in minimisation space.

    :param points: (N, M) array of objective values.
    :return:       (K, M) non-dominated subset, K ≤ N.
    """
    if points.ndim != 2 or len(points) == 0:
        return points
    idx = find_non_dominated(points)
    return points[idx]


def calc_hv(
    points: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> float:
    """
    Hypervolume of *points* with respect to *ref_point*.

    :param points:    (N, M) non-dominated front in minimisation space.
    :param ref_point: (M,) reference point. If None, set to max per axis + 10 %.
    :return:          Scalar HV value (larger is better).

    Normalisation note: pass pre-normalised points and a fixed ref_point of
    np.array([1.1, 1.1]) when comparing across experiments.
    """
    if len(points) == 0:
        return 0.0

    if ref_point is None:
        margin = np.abs(points.max(axis=0)) * 0.10 + 1e-6
        ref_point = points.max(axis=0) + margin

    ref_point = np.asarray(ref_point, dtype=np.float64)
    # pymoo HV requires all front points to be dominated by ref_point
    dominated_mask = np.all(points < ref_point, axis=1)
    pts_valid = points[dominated_mask]
    if len(pts_valid) == 0:
        return 0.0

    return float(HV(ref_point=ref_point).do(pts_valid))


def calc_igd(
    points: np.ndarray,
    true_pareto: np.ndarray,
) -> float:
    """
    Inverted Generational Distance (IGD).

    $$IGD(F, P^*) = \\frac{1}{|P^*|} \\sum_{p \\in P^*} \\min_{f \\in F} \\|p - f\\|_2$$

    :param points:      (N, M) approximation front (algorithm output).
    :param true_pareto: (K, M) reference (proxy) Pareto front P*.
    :return:            Scalar IGD (lower is better).
    """
    if len(points) == 0 or len(true_pareto) == 0:
        return float("inf")

    # Vectorised: dists[i, j] = ||true_pareto[i] - points[j]||_2
    diff = true_pareto[:, None, :] - points[None, :, :]   # (K, N, M)
    dists = np.linalg.norm(diff, axis=2)                   # (K, N)
    return float(dists.min(axis=1).mean())


def proxy_pareto(archives: list[list[dict]]) -> np.ndarray:
    """
    Build a proxy Pareto front P* from the union of multiple run archives.

    :param archives: List of archives (one per run/algorithm).
    :return:         Non-dominated (K, 2) array in minimisation space.
    """
    all_pts = np.vstack([archive_to_points(a) for a in archives if a])
    return get_pareto_front(all_pts)
