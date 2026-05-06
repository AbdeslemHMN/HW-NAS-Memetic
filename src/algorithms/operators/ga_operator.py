"""
GA Operator — Discrete per-gene mutation strategy.

Wraps ``src.operators.discrete_ga.mutate``.  Each call applies one round of
per-gene Bernoulli mutation (p_m = 1/6 by default) to the current architecture
of sub-problem k.  No crossover is performed here; crossover is handled
separately by the MOEA/D neighbourhood propagation in the orchestrator.

This operator is self-contained: it reads only ``state.current[k]`` and
produces a mutated copy.  It can be unit-tested without a full search loop.
"""
from __future__ import annotations

import numpy as np

from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.operators.discrete_ga import mutate

_DEFAULT_P_M: float = 1.0 / 6.0  # one expected mutation per architecture (L=6)


class GAOperator(BaseOperator):
    """
    Per-gene mutation operator.

    Parameters
    ----------
    p_m : float, optional
        Per-gene mutation probability.  Defaults to 1/L = 1/6, which gives an
        expected Hamming distance of 1 per generation — the standard ergodic
        regime for NAS-Bench-201's {0..4}^6 search space.
    """

    def __init__(self, p_m: float = _DEFAULT_P_M) -> None:
        if not 0.0 < p_m <= 1.0:
            raise ValueError(f"p_m must be in (0, 1], got {p_m}.")
        self.p_m = p_m

    def apply(
        self,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a per-gene mutated copy of state.current[k]."""
        return mutate(state.current[k], p_m=self.p_m, rng=rng)

    def __repr__(self) -> str:
        return f"GAOperator(p_m={self.p_m:.4f})"
