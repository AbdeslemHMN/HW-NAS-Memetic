"""
Strategy base class and shared state object for MemeticNAS operators.

Design
------
MemeticState is a lightweight read-only view of the MOEA/D population that is
passed to every operator call.  Operators must NOT mutate state — they only
read from it and return a new candidate architecture.  All NFE tracking
(calls to BaseOptimizer._eval) stays centralised in MemeticNAS, never inside
an operator.

BaseOperator defines the single-method contract:

    candidate = op.apply(state, k, rng)

where k is the sub-problem index and rng is the shared Generator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MemeticState:
    """
    Read-only snapshot of the MOEA/D population at the current iteration.

    Operators index arrays by sub-problem k:
        state.current[k]        — current architecture for sub-problem k
        state.pbest[k]          — personal-best architecture
        state.gbest[k]          — neighbourhood-best architecture
        state.accs[k]           — last evaluated accuracy for sub-problem k
        state.lats[k]           — last evaluated latency
        state.scores[k]         — scalarized score g(a | w_k)
        state.weights[k]        — weight vector (2,) for sub-problem k
        state.neighborhoods[k]  — indices of T_n nearest sub-problems
        state.T_sa              — current SA temperature (scalar)
    """
    current:       np.ndarray   # (K, 6)
    pbest:         np.ndarray   # (K, 6)
    gbest:         np.ndarray   # (K, 6)
    accs:          np.ndarray   # (K,)
    lats:          np.ndarray   # (K,)
    scores:        np.ndarray   # (K,)
    weights:       np.ndarray   # (K, 2)
    neighborhoods: np.ndarray   # (K, T_n)
    T_sa:          float


class BaseOperator(ABC):
    """
    Abstract strategy interface for candidate-generation operators.

    Subclasses implement ``apply`` to produce a new architecture proposal for
    sub-problem *k*.  The method must be pure (no side-effects on *state*) and
    must NOT call the eval function — evaluation is the orchestrator's job.

    Example
    -------
    >>> op = GAOperator()
    >>> candidate = op.apply(state, k=3, rng=np.random.default_rng(0))
    >>> assert candidate.shape == (6,)
    """

    @abstractmethod
    def apply(
        self,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a candidate architecture for sub-problem *k*.

        Parameters
        ----------
        state : MemeticState
            Current population snapshot (read-only).
        k : int
            Sub-problem index in [0, K).
        rng : np.random.Generator
            Shared seeded generator — must be the only source of randomness.

        Returns
        -------
        np.ndarray, shape (6,), dtype int64
            A new candidate architecture with values in {0, 1, 2, 3, 4}.
        """
