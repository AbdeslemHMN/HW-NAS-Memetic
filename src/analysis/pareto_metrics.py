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


def _pareto_front_2d_sweep(points: np.ndarray) -> np.ndarray:
    """
    Memory-efficient 2-objective Pareto front via sort + vectorised sweep.
    O(N log N) time, O(N) memory — avoids the O(N²) broadcast in pymoo.
    """
    order = np.lexsort((points[:, 1], points[:, 0]))   # sort by (f1 asc, f2 asc)
    f1s = points[order, 0]
    f2s = points[order, 1]

    # ── assign a group id per unique f1 value ────────────────────────────
    group_change = np.concatenate([[True], f1s[1:] != f1s[:-1]])
    group_ids    = np.cumsum(group_change) - 1          # 0-indexed
    num_groups   = int(group_ids[-1]) + 1

    # ── per-group minimum f2 ─────────────────────────────────────────────
    group_min_f2 = np.full(num_groups, np.inf)
    np.minimum.at(group_min_f2, group_ids, f2s)

    # ── minimum f2 of ALL strictly earlier groups (exclusive prefix min) ─
    cum_min_before = np.full(num_groups, np.inf)
    cum_min_before[1:] = np.minimum.accumulate(group_min_f2[:-1])

    # ── non-dominated iff: group-minimum AND better than all prior groups ─
    is_group_min    = f2s == group_min_f2[group_ids]
    beats_prev      = f2s < cum_min_before[group_ids]
    return points[order[is_group_min & beats_prev]]


def get_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Extract the non-dominated subset of *points* (N, M) in minimisation space.

    For M == 2 a vectorised O(N log N) sweep is used to avoid the O(N²)
    memory allocation inside pymoo's ``find_non_dominated``.

    :param points: (N, M) array of objective values.
    :return:       (K, M) non-dominated subset, K ≤ N.
    """
    if points.ndim != 2 or len(points) == 0:
        return points
    if points.shape[1] == 2:
        return _pareto_front_2d_sweep(points)
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


def calc_spacing(points: np.ndarray) -> float:
    """Spacing (S) — standard deviation of nearest-neighbour distances.

    Measures how uniformly the front points are distributed.
    **Lower is better** (0.0 = perfectly uniform spacing).

    $$S = \\sqrt{\\frac{1}{|F|-1} \\sum_{i=1}^{|F|} (d_i - \\bar{d})^2}$$

    where $d_i = \\min_{j \\neq i} \\|f_i - f_j\\|_2$ is the nearest-neighbour
    distance for point $i$ and $\\bar{d}$ is the mean over all $d_i$.

    :param points: (N, M) non-dominated front in minimisation space.
    :return:       Scalar spacing value (lower is better).
                   Returns ``inf`` if fewer than 2 points are provided.
    """
    if len(points) < 2:
        return float("inf")
    # pairwise L2 distances — fill diagonal with inf to exclude self
    diff = points[:, None, :] - points[None, :, :]   # (N, N, M)
    dist = np.linalg.norm(diff, axis=2)              # (N, N)
    np.fill_diagonal(dist, np.inf)
    d_min = dist.min(axis=1)                          # nearest-neighbour dist
    return float(d_min.std())


def calc_max_spread(points: np.ndarray) -> float:
    """Maximum Spread (MS) — Euclidean distance between extreme front points.

    Measures how widely the algorithm explored the objective space.
    **Higher is better** — a larger MS means the algorithm covered a broader
    range of the accuracy/latency trade-off spectrum.

    $$MS = \\|\\max(F) - \\min(F)\\|_2$$

    where $\\max(F)$ and $\\min(F)$ are the component-wise maximum and minimum
    vectors of the front.

    :param points: (N, M) non-dominated front in minimisation space.
    :return:       Scalar MS value (higher is better).
                   Returns 0.0 if fewer than 2 points are provided.
    """
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def proxy_pareto(archives: list[list[dict]]) -> np.ndarray:
    """
    Build a proxy Pareto front P* from the union of multiple run archives.

    Each archive is reduced to its own Pareto front before pooling so the
    combined set stays small regardless of per-run budget size.

    :param archives: List of archives (one per run/algorithm).
    :return:         Non-dominated (K, 2) array in minimisation space.
    """
    fronts = [
        get_pareto_front(archive_to_points(a))
        for a in archives if a
    ]
    if not fronts:
        raise ValueError("proxy_pareto: all archives are empty.")
    return get_pareto_front(np.vstack(fronts))
