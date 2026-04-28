"""
Simulated Annealing (SA) trajectory control for HW-NAS-Memetic local search.

Research Insights:
- metropolis_accept implements the classic Metropolis criterion. For
  minimization, a worse candidate (delta > 0) is accepted with probability
  exp(-delta / T), allowing hill-climbing escapes calibrated by temperature.
  At high T the walk is near-random; at T -> 0 it becomes greedy descent.
- cool applies geometric (multiplicative) cooling: T_{n+1} = alpha * T_n,
  with alpha in (0.8, 0.99] being the standard regime. This guarantees
  convergence in O(log(1/eps)) iterations for a given precision eps.
- The SA layer operates on single-architecture candidates produced by the
  PSO step, acting as a local refinement hill-climber within each MOEA/D
  sub-problem, not across the full population.
"""
import math


def metropolis_accept(
    score_curr: float,
    score_cand: float,
    T: float,
    rng_value: float | None = None,
) -> bool:
    """
    Metropolis acceptance criterion (minimization convention).

    Returns True (accept) if:
      - score_cand <= score_curr (improvement), or
      - U(0,1) < exp((score_curr - score_cand) / T)  (probabilistic accept of worse)

    :param score_curr: Scalar objective value of the current solution.
    :param score_cand: Scalar objective value of the candidate solution.
    :param T:          Current temperature. Must be > 0.
    :param rng_value:  Pre-drawn U(0,1) sample. Injected for testability;
                       if None, uses math module random (not reproducible).
    :return:           True if the candidate is accepted.
    """
    if T <= 0.0:
        raise ValueError(f"Temperature must be > 0, got T={T}.")

    delta = score_cand - score_curr  # negative delta = improvement

    if delta <= 0.0:
        return True  # always accept improvement

    if rng_value is None:
        import random
        rng_value = random.random()

    return rng_value < math.exp(-delta / T)


def cool(T: float, alpha: float = 0.95) -> float:
    """
    Geometric cooling schedule: T_new = T * alpha.

    :param T:     Current temperature. Must be > 0.
    :param alpha: Cooling rate in (0, 1). Typical range: [0.80, 0.99].
    :return:      Reduced temperature.
    """
    if T <= 0.0:
        raise ValueError(f"Temperature must be > 0, got T={T}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got alpha={alpha}.")

    return T * alpha
