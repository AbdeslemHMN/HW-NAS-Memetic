"""
Non-parametric statistical testing utilities for algorithm comparison.

All functions use scipy.stats and are appropriate for small sample sizes
(N=30 seeds) where normality cannot be assumed for stochastic meta-heuristics.

Functions
---------
shapiro_wilk         — Shapiro-Wilk normality test (most powerful for n < 50)
wilcoxon_ranksum     — Mann-Whitney U test with rank-biserial effect size r
kruskal_wallis       — Kruskal-Wallis H test for k≥2 independent groups
bonferroni_correction — Family-wise error rate correction (Bonferroni method)

Methodology Note
----------------
Meta-heuristic algorithms are stochastic processes.  Their per-seed performance
distributions are rarely Gaussian.  The recommended workflow is:

  1. Run ``shapiro_wilk`` on each algorithm's HV vector.
  2. If *any* group fails normality (p < 0.05), use non-parametric tests
     for all comparisons to maintain consistency.
  3. Use ``kruskal_wallis`` first (omnibus test across all k algorithms).
  4. If significant, apply pairwise ``wilcoxon_ranksum`` for post-hoc analysis.
  5. Adjust p-values with ``bonferroni_correction`` to control FWER.
  6. Report rank-biserial correlation ``r`` as the effect size.

References
----------
- Shapiro & Wilk (1965) — An analysis of variance test for normality.
  Biometrika, 52(3–4), 591–611.
- Mann & Whitney (1947) — On a test of whether one of two random variables is
  stochastically larger than the other.
- Kruskal & Wallis (1952) — Use of ranks in one-criterion variance analysis.
- Cohen, J. (1992) — A power primer.  Psychological Bulletin, 112(1), 155–159.
  Effect size r = rank-biserial correlation (large |r|>0.5, medium 0.3–0.5,
  small 0.1–0.3).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal as _kruskal
from scipy.stats import shapiro as _shapiro


def shapiro_wilk(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Shapiro-Wilk test for normality.

    For n < 50 (e.g., 30 seeds), this is the most statistically powerful
    normality test available.  Use it *before* choosing between parametric
    (t-test / ANOVA) and non-parametric (Wilcoxon / Kruskal-Wallis) tests.

    Decision rule:
        p ≥ alpha  →  fail to reject H₀  →  data is **consistent with normality**
        p  < alpha  →  reject H₀          →  data is **non-normal** → use
                                              non-parametric tests

    Parameters
    ----------
    data : array-like
        1-D array of per-seed metric values (e.g., HV or IGD scores).
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict with keys:
        W_stat  — Shapiro-Wilk W statistic ∈ (0, 1]; closer to 1 is more normal
        p_value — p-value of the test (float)
        normal  — True if p_value >= alpha, i.e., normality is *not* rejected
    """
    W, p = _shapiro(np.asarray(data, dtype=float))
    return {"W_stat": float(W), "p_value": float(p), "normal": bool(p >= alpha)}


def wilcoxon_ranksum(
    hv_proposed: np.ndarray,
    hv_baseline: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sided Mann-Whitney U test (Wilcoxon rank-sum equivalent).

    Uses scipy.stats.mannwhitneyu, which returns the U statistic needed to
    compute the rank-biserial correlation effect size:

        r = 1 − 2U / (n₁ × n₂)

    A positive r means *hv_proposed* tends to be larger (better for HV).

    Parameters
    ----------
    hv_proposed : array-like
        Per-seed metric values for the proposed algorithm.
    hv_baseline : array-like
        Per-seed metric values for the baseline algorithm.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict with keys:
        U_stat      — Mann-Whitney U statistic (float)
        p_value     — two-sided p-value (float)
        effect_r    — rank-biserial correlation r ∈ [−1, 1] (float)
        significant — True if p_value < alpha (bool)
    """
    a = np.asarray(hv_proposed, dtype=float)
    b = np.asarray(hv_baseline, dtype=float)
    n1, n2 = len(a), len(b)
    U_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
    effect_r = 1.0 - (2.0 * U_stat) / (n1 * n2)
    return {
        "U_stat": float(U_stat),
        "p_value": float(p_value),
        "effect_r": float(effect_r),
        "significant": bool(p_value < alpha),
    }


def kruskal_wallis(
    hv_groups: list[np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """
    Kruskal-Wallis H test for k ≥ 2 independent groups.

    Non-parametric one-way ANOVA on ranks.  A significant result only
    indicates that at least one group distribution differs; use pairwise
    ``wilcoxon_ranksum`` with Bonferroni correction to identify which pairs.

    Parameters
    ----------
    hv_groups : list of array-like
        Each element is a per-seed metric array for one algorithm.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict with keys:
        H_stat      — Kruskal-Wallis H statistic (float)
        p_value     — p-value (float)
        significant — True if p_value < alpha (bool)
    """
    groups = [np.asarray(g, dtype=float) for g in hv_groups]
    H_stat, p_value = _kruskal(*groups)
    return {
        "H_stat": float(H_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
    }


def bonferroni_correction(p_values: list[float] | np.ndarray) -> np.ndarray:
    """
    Bonferroni family-wise error rate correction.

    Multiplies each raw p-value by the number of comparisons k and clips
    to [0, 1].  The corrected threshold is α* = α / k, equivalent to
    testing each p_corrected against the original α.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from k independent tests.

    Returns
    -------
    numpy.ndarray of corrected p-values, each in [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    k = len(p)
    return np.clip(p * k, 0.0, 1.0)
