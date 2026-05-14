"""
GA Operator — Discrete per-gene mutation strategy (improved).
==============================================================

Drop-in replacement for the original GAOperator.
Wraps ``src.operators.discrete_ga.mutate`` and adds four selectable
improvements on top of the vanilla Bernoulli mutation baseline.

Variants
--------
T1  baseline                        — Bernoulli mutation, p_m = 1/6
T2  force_change                    — resample until Hamming ≥ 1 (no clones)
T3  adaptive_pm                     — cosine p_m decay (high→low over budget)
T4  crossover                       — uniform crossover with random neighbour
T5  force_change + adaptive_pm
T6  force_change + crossover
T14 force_change + crossover + freq_bias  ← BEST HV (0.8660 ± 0.0035)

Sweep results (15 seeds, 1000 NFE, edgegpu_latency / cifar10):
    T14  force_change + crossover + freq_bias   HV 0.8660 ± 0.0035  [BEST HV]
    T6   force_change + crossover               HV 0.8653 ± 0.0048  [BEST IGD: 0.0011]
    T12  crossover + freq_bias                  HV 0.8639 ± 0.0081
    T1   baseline                               HV 0.8550 ± 0.0101
    (freq_bias alone consistently hurts — only beneficial with crossover)

Default constructor uses T14 settings (best HV).

Usage
-----
    op = GAOperator()                                      # T14 — best HV
    op = GAOperator(crossover=False, freq_bias=False)      # T2 — safest
    op = GAOperator(adaptive=True)                         # T5
    candidate = op.apply(state, k, rng)                    # identical to original

Adaptation hooks (call each generation when using adaptive/freq_bias):
    op.notify_budget(spent, total)
    op.update_archive_stats(archive)
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.operators.discrete_ga import mutate

# ── Constants ─────────────────────────────────────────────────────────────────
_N_OPS:   int   = 5
_N_EDGES: int   = 6
_DEFAULT_P_M:      float = 1.0 / _N_EDGES        # expected 1 mutation per arch
_DEFAULT_P_M_HIGH: float = 3.0 / _N_EDGES        # aggressive early exploration
_DEFAULT_P_M_LOW:  float = 0.5 / _N_EDGES        # gentle late fine-tuning
_MAX_RESAMPLE:     int   = 20                     # safety cap for force_change loop

# ── Best hyperparameters (T14 — determined by sweep, rank 1/16) ──────────────
_DEFAULT_FORCE_CHANGE  = True
_DEFAULT_CROSSOVER     = True
_DEFAULT_FREQ_BIAS     = True
_DEFAULT_ADAPTIVE      = False   # T14 does not use adaptive — adaptive hurts with freq_bias


class GAOperator(BaseOperator):
    """
    Per-gene mutation operator with optional improvements.

    Parameters
    ----------
    p_m : float
        Baseline per-gene mutation probability (used when adaptive=False).
        Defaults to 1/6.
    force_change : bool
        Resample until Hamming(parent, child) ≥ 1. Eliminates clone evals.
        Default: True.
    adaptive : bool
        Cosine-decay p_m from p_m_high → p_m_low over the run budget.
        Requires notify_budget() calls. Default: False.
    p_m_high : float
        Starting p_m for adaptive schedule. Default: 3/6 = 0.5.
    p_m_low : float
        Ending p_m for adaptive schedule. Default: 0.5/6 ≈ 0.083.
    crossover : bool
        Uniform crossover with a random neighbour before mutation.
        Default: True.
    crossover_rate : float
        Probability of applying crossover per call. Default: 1.0.
    parent_selection : str
        Donor selection: "random" or "tournament". Default: "random".
    tournament_size : int
        Tournament size when parent_selection="tournament". Default: 2.
    freq_bias : bool
        Frequency-biased replacement — down-weight over-represented ops,
        up-weight rare ones. Requires update_archive_stats() calls.
        Default: True.
    """

    def __init__(
        self,
        p_m: float = _DEFAULT_P_M,
        *,
        force_change:     bool  = _DEFAULT_FORCE_CHANGE,
        adaptive:         bool  = _DEFAULT_ADAPTIVE,
        p_m_high:         float = _DEFAULT_P_M_HIGH,
        p_m_low:          float = _DEFAULT_P_M_LOW,
        crossover:        bool  = _DEFAULT_CROSSOVER,
        crossover_rate:   float = 1.0,
        parent_selection: str   = "random",
        tournament_size:  int   = 2,
        freq_bias:        bool  = _DEFAULT_FREQ_BIAS,
    ) -> None:
        if not 0.0 < p_m <= 1.0:
            raise ValueError(f"p_m must be in (0, 1], got {p_m}.")
        if not 0.0 < p_m_high <= 1.0:
            raise ValueError(f"p_m_high must be in (0, 1], got {p_m_high}.")
        if not 0.0 < p_m_low <= 1.0:
            raise ValueError(f"p_m_low must be in (0, 1], got {p_m_low}.")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}.")
        if parent_selection not in {"random", "tournament"}:
            raise ValueError(f"parent_selection must be 'random' or 'tournament'.")
        if tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {tournament_size}.")

        self.p_m              = p_m
        self.force_change     = force_change
        self.adaptive         = adaptive
        self.p_m_high         = p_m_high
        self.p_m_low          = p_m_low
        self.crossover        = crossover
        self.crossover_rate   = crossover_rate
        self.parent_selection = parent_selection
        self.tournament_size  = tournament_size
        self.freq_bias        = freq_bias

        # Adaptive schedule state
        self._budget_fraction: float = 0.0

        # Frequency-biased sampling state: per-edge op counts with Laplace smoothing
        self._op_counts: np.ndarray = np.ones((_N_EDGES, _N_OPS), dtype=float)

    # ── Hooks called by the orchestrator each generation ──────────────────────

    def notify_budget(self, spent: int, total: int) -> None:
        """Update cosine-decay schedule. Call once per generation."""
        if total > 0:
            self._budget_fraction = min(1.0, spent / total)

    def update_archive_stats(self, archive: List[np.ndarray]) -> None:
        """
        Recompute per-edge op frequencies from the current Pareto archive.
        Call once per generation after the archive is updated.
        """
        counts = np.ones((_N_EDGES, _N_OPS), dtype=float)   # Laplace smoothing
        for arch in archive:
            for edge, op in enumerate(arch):
                counts[edge, int(op)] += 1.0
        self._op_counts = counts

    # ── Core apply — identical call signature to original ─────────────────────

    def apply(
        self,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a mutated (and optionally crossed-over) candidate for sub-problem k."""
        parent = state.current[k].copy()
        pm = self._active_pm()

        # Step 1: optional crossover
        base = parent
        if self.crossover and rng.random() < self.crossover_rate:
            base = self._crossover(parent, state, k, rng)

        # Step 2: mutate
        child = self._mutate(base, pm, rng)

        # Step 3: guarantee at least one change vs original parent
        if self.force_change:
            attempts = 0
            while np.array_equal(child, parent) and attempts < _MAX_RESAMPLE:
                child = self._mutate(base, pm, rng)
                attempts += 1
            if np.array_equal(child, parent):   # absolute fallback
                child = parent.copy()
                edge = int(rng.integers(0, _N_EDGES))
                ops  = [o for o in range(_N_OPS) if o != parent[edge]]
                child[edge] = int(rng.choice(ops))

        return child

    # ── Private helpers ───────────────────────────────────────────────────────

    def _active_pm(self) -> float:
        if not self.adaptive:
            return self.p_m
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * self._budget_fraction))
        return self.p_m_low + (self.p_m_high - self.p_m_low) * cos_factor

    def _mutate(self, arch: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
        if not self.freq_bias:
            return mutate(arch, p_m=pm, rng=rng)
        child = arch.copy()
        for edge in range(_N_EDGES):
            if rng.random() < pm:
                child[edge] = self._sample_new_op(edge, int(child[edge]), rng)
        return child

    def _sample_new_op(self, edge: int, current_op: int, rng: np.random.Generator) -> int:
        if not self.freq_bias:
            ops = [o for o in range(_N_OPS) if o != current_op]
            return int(rng.choice(ops))
        counts = self._op_counts[edge].copy()
        counts[current_op] = 0.0
        inv_w = 1.0 / (counts + 1e-9)
        inv_w[current_op] = 0.0
        probs = inv_w / inv_w.sum()
        return int(rng.choice(_N_OPS, p=probs))

    def _crossover(
        self,
        parent: np.ndarray,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Uniform crossover with a randomly selected donor sub-problem."""
        n = len(state.current)
        if n < 2:
            return parent.copy()
        others = [i for i in range(n) if i != k]
        if self.parent_selection == "tournament":
            size = min(self.tournament_size, len(others))
            candidates = rng.choice(others, size=size, replace=False)
            donor_idx = int(max(candidates, key=lambda idx: state.scores[idx]))
        else:
            donor_idx = int(rng.choice(others))
        donor = state.current[donor_idx]
        mask  = rng.random(_N_EDGES) < 0.5
        return np.where(mask, parent, donor).astype(parent.dtype)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def active_pm(self) -> float:
        return self._active_pm()

    def variant_name(self) -> str:
        flags = []
        if self.force_change: flags.append("force_change")
        if self.adaptive:     flags.append("adaptive_pm")
        if self.crossover:    flags.append("crossover")
        if self.freq_bias:    flags.append("freq_bias")
        return "GAOperator[" + (", ".join(flags) if flags else "baseline") + "]"

    def __repr__(self) -> str:
        return (
            f"GAOperator(p_m={self.p_m:.4f}, force_change={self.force_change}, "
            f"adaptive={self.adaptive}, crossover={self.crossover}, "
            f"freq_bias={self.freq_bias})"
        )
