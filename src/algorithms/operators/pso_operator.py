"""
PSO Operator — Discrete probabilistic particle-swarm step strategy.

Wraps ``src.operators.discrete_pso.pso_step``.  At each sub-problem k the
particle moves toward its personal best and its neighbourhood-best according
to per-edge attraction probabilities (c1·r1, c2·r2).

This operator reads state.current[k], state.pbest[k], and state.gbest[k].
It is self-contained and can be unit-tested independently of the full loop.

Research note
-------------
Expected edges moved per step ≈ c1 + c2 ≈ 1.0 with default values (0.5 + 0.5).
This deliberately mirrors GAOperator's expected Hamming distance of 1, so that
adding PSO alongside GA does not geometrically change the perturbation magnitude
— it only adds an attraction bias toward high-quality solutions found so far.
"""
from __future__ import annotations

import numpy as np

from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.operators.discrete_pso import pso_step


class PSOOperator(BaseOperator):
    """
    Probabilistic discrete PSO step operator.

    Parameters
    ----------
    c1 : float
        Personal-attraction coefficient.  Each edge independently moves to
        pbest[j] with probability c1 · r1_j (r1 ~ U(0,1)).
    c2 : float
        Neighbourhood-attraction coefficient.  Each edge independently moves
        to gbest[j] with probability c2 · r2_j (evaluated when not pulled to pbest).
    """

    def __init__(self, c1: float = 0.5, c2: float = 0.5) -> None:
        if not 0.0 < c1 <= 1.0 or not 0.0 < c2 <= 1.0:
            raise ValueError(f"c1 and c2 must be in (0, 1], got c1={c1}, c2={c2}.")
        self.c1 = c1
        self.c2 = c2

    def apply(
        self,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a PSO-updated architecture for sub-problem k."""
        return pso_step(
            state.current[k],
            state.pbest[k],
            state.gbest[k],
            c1=self.c1,
            c2=self.c2,
            rng=rng,
        )

    def __repr__(self) -> str:
        return f"PSOOperator(c1={self.c1}, c2={self.c2})"
