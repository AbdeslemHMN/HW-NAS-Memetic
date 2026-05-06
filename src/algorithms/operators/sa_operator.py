"""
SA Operator — Simulated Annealing acceptance strategy.

Wraps ``src.operators.simulated_annealing.metropolis_accept`` and ``cool``.
Unlike GAOperator / PSOOperator, SAOperator is NOT a candidate generator —
it decides whether a candidate produced by the other operators is accepted,
and it owns the temperature schedule used by MemeticNAS.

Usage in MemeticNAS
-------------------
    sa = SAOperator(T0=1.0, alpha=0.95)
    accepted = sa.accept(score_curr, score_cand, T, rng)
    T = sa.cool(T)

Convention
----------
``accept`` uses the **maximisation** convention (higher score = better),
so callers do NOT need to negate scores.  Internally scores are negated
before being passed to the minimisation-convention ``metropolis_accept``.
"""
from __future__ import annotations

import numpy as np

from src.operators.simulated_annealing import metropolis_accept
from src.operators.simulated_annealing import cool as _geom_cool


class SAOperator:
    """
    Metropolis acceptance with geometric cooling.

    Parameters
    ----------
    T0 : float
        Initial temperature.  Must be > 0.
    alpha : float
        Geometric cooling rate in (0, 1).  Typical range: [0.80, 0.99].
        After N steps: T_N = T0 * alpha^N.
    """

    def __init__(self, T0: float = 1.0, alpha: float = 0.95) -> None:
        if T0 <= 0.0:
            raise ValueError(f"T0 must be > 0, got {T0}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.T0 = T0
        self.alpha = alpha

    def accept(
        self,
        score_curr: float,
        score_cand: float,
        T: float,
        rng: np.random.Generator,
    ) -> bool:
        """
        Metropolis acceptance (maximisation convention).

        A candidate with score_cand > score_curr is always accepted.
        A worse candidate is accepted with probability exp(Δ / T) where
        Δ = score_cand - score_curr < 0.

        Parameters
        ----------
        score_curr : float  Current scalarized score g(current | w_k).
        score_cand : float  Candidate scalarized score g(candidate | w_k).
        T          : float  Current temperature.
        rng        : np.random.Generator  Shared seeded generator.

        Returns
        -------
        bool  True if the candidate is accepted.
        """
        # metropolis_accept uses minimisation convention → negate scores.
        return metropolis_accept(
            -score_curr,
            -score_cand,
            T,
            rng_value=float(rng.random()),
        )

    def cool(self, T: float) -> float:
        """Apply one geometric cooling step: T_new = T * alpha."""
        return _geom_cool(T, self.alpha)

    def __repr__(self) -> str:
        return f"SAOperator(T0={self.T0}, alpha={self.alpha})"
