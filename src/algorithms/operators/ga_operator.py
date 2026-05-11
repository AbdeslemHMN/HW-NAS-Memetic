"""
GA Operator — Discrete per-gene mutation strategy (improved).
==============================================================

Drop-in replacement for the original GAOperator.
Wraps ``src.operators.discrete_ga.mutate`` and adds four selectable
improvements on top of the vanilla Bernoulli mutation baseline.

Variants (controlled by constructor flags)
-------------------------------------------
Baseline (original)
    Per-gene Bernoulli mutation, p_m = 1/6.
    ~33.5% of calls produce a clone (Hamming = 0), wasting a budget eval.

Improvement 1 — Guaranteed Hamming ≥ 1  (force_change=True)
    Resample until at least one gene differs from the parent.
    Eliminates wasted clone evaluations for free.
    Recommended as a minimum fix — no hyperparameters, zero risk.

Improvement 2 — Adaptive p_m  (adaptive=True)
    p_m decays from p_m_high to p_m_low following a cosine schedule driven
    by budget_fraction (0 → 1 over the run).  High early for exploration,
    low late for fine-tuning.  Mirrors the SA temperature already in the
    pipeline.  Call notify_budget(spent, total) each generation to update.

Improvement 3 — Uniform crossover  (crossover=True)
    When state.current has at least two sub-problems, pick a random
    neighbour k' and recombine gene-wise with 50/50 probability before
    applying mutation.  Gives recombination without touching MOEA/D logic.

Improvement 4 — Frequency-biased mutation  (freq_bias=True)
    Track which operations appear in the Pareto archive.  Down-weight
    over-represented ops and up-weight under-represented ones when sampling
    the replacement value.  Breaks the Uniform{0..4} symmetry toward
    genuinely novel candidates.  Call update_archive_stats(archive) each
    generation.

All flags are independent and composable.
Default: force_change=True only (safe, highest ROI, recommended for T2).

Usage (identical interface to original)
----------------------------------------
    op = GAOperator()                          # baseline + force_change
    op = GAOperator(adaptive=True)             # + adaptive p_m
    op = GAOperator(crossover=True)            # + crossover
    op = GAOperator(adaptive=True,
                    crossover=True,
                    freq_bias=True)            # all improvements

    candidate = op.apply(state, k, rng)       # same call as before
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.operators.discrete_ga import mutate

# ---------------------------------------------------------------------------
# Constants for NAS-Bench-201
# ---------------------------------------------------------------------------
_N_OPS: int = 5          # operations per edge: {0, 1, 2, 3, 4}
_N_EDGES: int = 6        # edges per architecture (L = 6)
_DEFAULT_P_M: float = 1.0 / _N_EDGES   # expected 1 mutation per arch
_DEFAULT_P_M_HIGH: float = 3.0 / _N_EDGES   # aggressive early exploration
_DEFAULT_P_M_LOW: float  = 0.5 / _N_EDGES   # gentle late fine-tuning
_MAX_RESAMPLE: int = 20  # safety cap for the force_change loop


class GAOperator(BaseOperator):
    """
    Per-gene mutation operator with optional improvements.

    Parameters
    ----------
    p_m : float
        Baseline per-gene mutation probability (used when adaptive=False).
        Defaults to 1/6.
    force_change : bool
        Improvement 1.  Resample until Hamming(parent, child) ≥ 1.
        Eliminates clone evaluations.  Default: True.
    adaptive : bool
        Improvement 2.  Cosine-decay p_m from p_m_high to p_m_low as the
        budget is consumed.  Requires notify_budget() calls.  Default: False.
    p_m_high : float
        Starting p_m for adaptive schedule (default 3/6 = 0.5).
    p_m_low : float
        Ending p_m for adaptive schedule (default 0.5/6 ≈ 0.083).
    crossover : bool
        Improvement 3.  Uniform crossover with a random neighbour before
        mutation.  Default: False.
    freq_bias : bool
        Improvement 4.  Frequency-biased replacement sampling.
        Requires update_archive_stats() calls.  Default: False.
    """

    def __init__(
        self,
        p_m: float = _DEFAULT_P_M,
        *,
        force_change: bool = True,
        adaptive: bool = False,
        p_m_high: float = _DEFAULT_P_M_HIGH,
        p_m_low: float = _DEFAULT_P_M_LOW,
        crossover: bool = False,
        freq_bias: bool = False,
    ) -> None:
        if not 0.0 < p_m <= 1.0:
            raise ValueError(f"p_m must be in (0, 1], got {p_m}.")
        if not 0.0 < p_m_high <= 1.0:
            raise ValueError(f"p_m_high must be in (0, 1], got {p_m_high}.")
        if not 0.0 < p_m_low <= 1.0:
            raise ValueError(f"p_m_low must be in (0, 1], got {p_m_low}.")

        self.p_m = p_m
        self.force_change = force_change
        self.adaptive = adaptive
        self.p_m_high = p_m_high
        self.p_m_low = p_m_low
        self.crossover = crossover
        self.freq_bias = freq_bias

        # Internal state for adaptive schedule
        self._budget_fraction: float = 0.0   # updated by notify_budget()

        # Internal state for frequency-biased sampling
        # Shape: (N_EDGES, N_OPS) — op counts per edge position
        self._op_counts: np.ndarray = np.ones((_N_EDGES, _N_OPS), dtype=float)

    # ------------------------------------------------------------------
    # Public API — called by the orchestrator every generation
    # ------------------------------------------------------------------

    def notify_budget(self, spent: int, total: int) -> None:
        """
        Update the adaptive p_m schedule.

        Call once per generation (or per apply() call) with the current
        number of budget evaluations consumed and the total budget.

        Parameters
        ----------
        spent : int   Number of benchmark lookups consumed so far.
        total : int   Total budget (budget_max in CONFIG).
        """
        if total > 0:
            self._budget_fraction = min(1.0, spent / total)

    def update_archive_stats(self, archive: List[np.ndarray]) -> None:
        """
        Recompute per-edge operation frequencies from the current archive.

        Call once per generation after the archive is updated.

        Parameters
        ----------
        archive : list of np.ndarray
            Each element is a length-6 architecture vector with values in
            {0, 1, 2, 3, 4}.
        """
        counts = np.ones((_N_EDGES, _N_OPS), dtype=float)   # Laplace smoothing
        for arch in archive:
            for edge, op in enumerate(arch):
                counts[edge, int(op)] += 1.0
        self._op_counts = counts

    # ------------------------------------------------------------------
    # Core apply() — same signature as original
    # ------------------------------------------------------------------

    def apply(
        self,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Return a mutated candidate for sub-problem k.

        Steps:
            1. (optional) Crossover with a random neighbour.
            2. Mutate using active p_m.
            3. (optional) Resample until child ≠ parent.
        """
        parent = state.current[k].copy()

        # ── Step 1: Uniform crossover (Improvement 3) ────────────────────
        base = self._crossover(parent, state, k, rng) if self.crossover else parent

        # ── Step 2: Mutate ────────────────────────────────────────────────
        pm = self._active_pm()
        child = self._mutate(base, pm, rng)

        # ── Step 3: Guarantee at least one change (Improvement 1) ────────
        if self.force_change:
            attempts = 0
            while np.array_equal(child, parent) and attempts < _MAX_RESAMPLE:
                child = self._mutate(base, pm, rng)
                attempts += 1
            # Absolute fallback: flip one random gene
            if np.array_equal(child, parent):
                child = parent.copy()
                edge = int(rng.integers(0, _N_EDGES))
                ops = [o for o in range(_N_OPS) if o != parent[edge]]
                child[edge] = int(rng.choice(ops))

        return child

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _active_pm(self) -> float:
        """
        Return current p_m.

        If adaptive=True, follows a cosine decay from p_m_high to p_m_low
        as budget_fraction goes from 0 to 1.
        """
        if not self.adaptive:
            return self.p_m
        # Cosine schedule: starts at p_m_high, ends at p_m_low
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * self._budget_fraction))
        return self.p_m_low + (self.p_m_high - self.p_m_low) * cos_factor

    def _mutate(self, arch: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
        """
        Apply per-gene Bernoulli mutation.

        If freq_bias=True, the replacement operation is sampled from a
        distribution inversely proportional to its frequency in the archive
        (down-weight over-represented ops, up-weight rare ones).
        """
        if not self.freq_bias:
            # Original behaviour: delegate to discrete_ga.mutate
            return mutate(arch, p_m=pm, rng=rng)

        # Frequency-biased replacement
        child = arch.copy()
        for edge in range(_N_EDGES):
            if rng.random() < pm:
                current_op = int(child[edge])
                # Inverse-frequency weights for ops other than current_op
                counts = self._op_counts[edge].copy()
                counts[current_op] = 0.0          # exclude current op
                inv_w = 1.0 / (counts + 1e-9)
                inv_w[current_op] = 0.0
                probs = inv_w / inv_w.sum()
                child[edge] = int(rng.choice(_N_OPS, p=probs))
        return child

    def _crossover(
        self,
        parent: np.ndarray,
        state: MemeticState,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Uniform crossover with a random other sub-problem.

        Each gene is taken from parent (sub-problem k) or a random other
        individual (sub-problem k') with 50/50 probability.
        """
        n = len(state.current)
        if n < 2:
            return parent.copy()

        # Pick a random other sub-problem index
        others = [i for i in range(n) if i != k]
        k2 = int(rng.choice(others))
        donor = state.current[k2]

        # Gene-wise coin flip
        mask = rng.random(_N_EDGES) < 0.5
        child = np.where(mask, parent, donor).astype(parent.dtype)
        return child

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_pm(self) -> float:
        """Current effective p_m (accounts for adaptive schedule)."""
        return self._active_pm()

    def variant_name(self) -> str:
        """Human-readable name of the active variant combination."""
        flags = []
        if self.force_change:
            flags.append("force_change")
        if self.adaptive:
            flags.append("adaptive_pm")
        if self.crossover:
            flags.append("crossover")
        if self.freq_bias:
            flags.append("freq_bias")
        return "GAOperator[" + (", ".join(flags) if flags else "baseline") + "]"

    def __repr__(self) -> str:
        return (
            f"GAOperator(p_m={self.p_m:.4f}, force_change={self.force_change}, "
            f"adaptive={self.adaptive}, crossover={self.crossover}, "
            f"freq_bias={self.freq_bias})"
        )
