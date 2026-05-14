"""
MemeticNAS — Modular MOEA/D orchestrator with pluggable operator strategy.

This module implements the new modular memetic search engine for the project.

Architecture
------------
The outer loop decomposes the bi-objective (Accuracy, Latency) space into K
scalar sub-problems via MOEA/D weight vectors.  Per sub-problem, the
orchestrator:

  1. Builds a MemeticState read-only snapshot of the population.
  2. Calls each operator in ``candidate_ops`` to generate a candidate.
  3. Evaluates each candidate via ``self._eval()`` (centralised NFE tracking).
  4. Keeps the best-scoring candidate.
  5. Accepts or rejects it via ``sa_op`` (Metropolis) or greedy fallback.
  6. Updates pbest, current, and propagates improvements to neighbours.

Turning operators on/off is O(1): just change the ``candidate_ops`` list.

    from src.algorithms.memetic_nas import MemeticNAS
    from src.algorithms.operators import GAOperator, PSOOperator, SAOperator

    # Stage 2: MOEA/D + GA
    alg = MemeticNAS(eval_fn, budget=2000, candidate_ops=[GAOperator()])

    # Stage 3: MOEA/D + GA + PSO
    alg = MemeticNAS(eval_fn, budget=2000, candidate_ops=[GAOperator(), PSOOperator()])

    # Stage 4: Full (+ SA acceptance)
    alg = MemeticNAS(eval_fn, budget=2000,
                     candidate_ops=[GAOperator(), PSOOperator()],
                     sa_op=SAOperator())
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer, ARCH_LEN, OPS_COUNT
from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.algorithms.operators.sa_operator import SAOperator

_RESTART_PATIENCE: int = 100   # outer iterations without improvement
_RESTART_DELTA: float  = 1e-6  # minimum improvement to reset stagnation counter


class MemeticNAS(BaseOptimizer):
    """
    Modular MOEA/D memetic search with a pluggable operator list.

    Parameters
    ----------
    eval_fn : callable
        Architecture evaluator: arch (6,) → (accuracy: float, latency: float).
    budget : int
        Hard NFE cap — the search stops the moment this is reached.
    candidate_ops : list[BaseOperator]
        Ordered list of candidate-generation operators.  All operators are
        applied for each sub-problem at each iteration; the best-scoring
        candidate is retained.  At least one operator is required.
    sa_op : SAOperator or None
        Acceptance strategy.  None → greedy (accept only improvements).
    K : int
        Number of MOEA/D weight-vector directions (sub-problems).
        Sweep best: K=10 (global); K=5 wins only on cifar10.
    T_neighborhood : int
        Neighbourhood size T_n for gbest selection and propagation.
        Sweep best: T=2.  Larger T homogenises neighbours and hurts HV (±25.3).
    scalarization : str
        ``'linear'``  — g = w₀·Acc − w₁·Lat  (default, globally best +20.9).
        ``'tchebychev'`` — g = −max(w₀·Δacc, w₁·Δlat) with dynamic ideal point.
        Tchebychev wins only on cifar100 (non-convex front); use ``'linear'``
        everywhere else.
    w_init : float
        Starting weight for accuracy on the first MOEA/D sub-problem.
        Defaults to 1.0 for the full [1.0, 0.0] schedule.
    w_final : float
        Ending weight for accuracy on the last MOEA/D sub-problem.
        Enables narrower calibrated ranges such as 0.6→0.4.
    use_restart : bool
        Re-initialise the worst K//2 solutions when the population stagnates
        for ``_RESTART_PATIENCE`` outer iterations.
    rng : np.random.Generator or None
        Seeded random generator for reproducibility.
    max_accuracy : float
        Upper bound for accuracy normalisation when using Tchebychev.
        Recommended default is 100.0 for percent accuracy.
    max_latency : float
        Upper bound for latency normalisation when using Tchebychev.
        Recommended default is 30.0 ms for edgegpu latency.

    Notes
    -----
    Parameter importance from 216-run sweep (24 configs × 3 datasets × 3 seeds,
    2000 NFE / run, edgegpu_latency + raspi4_latency + eyeriss_latency):
        T_neighborhood  ±25.3 HV
        Scalarization   ±20.9 HV
        K               ±11–20 HV
        SA alpha        ±2.0 HV  (virtually irrelevant at budget=2000)
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, float]],
        budget: int,
        candidate_ops: list[BaseOperator],
        sa_op: SAOperator | None = None,
        K: int = 10,
        T_neighborhood: int = 2,
        scalarization: str = "linear",
        w_init: float = 1.0,
        w_final: float = 0.0,
        use_restart: bool = True,
        rng: np.random.Generator | None = None,
        max_accuracy: float = 100.0,
        max_latency: float = 30.0,
    ) -> None:
        if not candidate_ops:
            raise ValueError(
                "candidate_ops must contain at least one BaseOperator instance."
            )
        _valid = ("linear", "tchebychev")
        if scalarization not in _valid:
            raise ValueError(f"scalarization must be one of {_valid}, got {scalarization!r}.")
        super().__init__(eval_fn, budget, rng)
        self.candidate_ops  = list(candidate_ops)
        self.sa_op          = sa_op
        self.K              = max(1, int(K))
        self.T_neighborhood = max(1, int(T_neighborhood))
        self.scalarization  = scalarization
        self.w_init         = float(w_init)
        self.w_final        = float(w_final)
        if not 0.0 <= self.w_init <= 1.0:
            raise ValueError(f"w_init must be in [0, 1], got {self.w_init}.")
        if not 0.0 <= self.w_final <= 1.0:
            raise ValueError(f"w_final must be in [0, 1], got {self.w_final}.")
        if K > 1 and self.w_init == self.w_final:
            raise ValueError("w_init and w_final must differ when K > 1 to form a schedule.")
        self.use_restart    = bool(use_restart)
        # Normalisation constants for Tchebychev (keeps acc/lat on same [0,1] scale)
        self._max_accuracy  = float(max_accuracy)
        self._max_latency   = float(max_latency)
        # Ideal point for Tchebychev (updated every evaluation)
        self._z_acc: float =  -np.inf
        self._z_lat: float =  +np.inf

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _eval(self, arch: np.ndarray) -> tuple[float, float]:
        """Evaluate and update Tchebychev ideal point."""
        acc, lat = super()._eval(arch)
        if acc > self._z_acc:
            self._z_acc = acc
        if lat < self._z_lat:
            self._z_lat = lat
        return acc, lat

    def _scalarize(self, w: np.ndarray, acc: float, lat: float) -> float:
        """Scalarize (maximise convention) using the configured strategy."""
        if self.scalarization == "linear":
            return float(w[0] * acc - w[1] * lat)
        # Tchebychev with normalized dynamic ideal point
        # Both objectives are mapped to [0,1] so weights are scale-invariant.
        z_acc = self._z_acc if self._z_acc != -np.inf else acc
        z_lat = self._z_lat if self._z_lat != +np.inf else lat
        acc_n  = acc   / self._max_accuracy
        z_acc_n = z_acc / self._max_accuracy
        lat_n  = lat   / self._max_latency
        z_lat_n = z_lat / self._max_latency
        return -float(max(w[0] * abs(acc_n - z_acc_n), w[1] * abs(lat_n - z_lat_n)))

    def _make_weights(self) -> np.ndarray:
        K = self.K
        if K == 1:
            return np.array([[0.5, 0.5]])
        w0 = np.linspace(self.w_init, self.w_final, K)
        w1 = 1.0 - w0
        return np.column_stack([w0, w1])  # (K, 2)

    # ── Main search ───────────────────────────────────────────────────────────

    def search(self) -> list[dict]:
        K   = self.K
        T_n = self.T_neighborhood

        weights = self._make_weights()  # (K, 2)

        # Neighbourhood indices: T_n nearest sub-problems by weight distance.
        dists         = np.linalg.norm(weights[:, None] - weights[None, :], axis=2)
        neighborhoods = np.argsort(dists, axis=1)[:, :T_n]  # (K, T_n)

        # ── Initialise population ──────────────────────────────────────────
        current = np.array([self._random_arch() for _ in range(K)])  # (K, 6)
        accs    = np.zeros(K, dtype=float)
        lats    = np.zeros(K, dtype=float)

        for k in range(K):
            if self.budget_spent >= self.budget:
                break
            accs[k], lats[k] = self._eval(current[k])

        scores       = np.array([self._scalarize(weights[k], accs[k], lats[k]) for k in range(K)])
        pbest        = current.copy()
        pbest_scores = scores.copy()
        gbest        = current.copy()

        def _update_gbest(k: int) -> np.ndarray:
            neigh  = neighborhoods[k]
            g_vals = [self._scalarize(weights[k], accs[n], lats[n]) for n in neigh]
            return current[neigh[int(np.argmax(g_vals))]].copy()

        for k in range(K):
            gbest[k] = _update_gbest(k)

        # Reset ideal point for Tchebychev (fresh run).
        self._z_acc = -np.inf
        self._z_lat = +np.inf

        # SA temperature — owned entirely by SAOperator.
        # Reset so repeated search() calls start fresh.
        if self.sa_op is not None:
            self.sa_op.reset()
        T_sa = self.sa_op.T if self.sa_op is not None else 1.0

        # ── Restart bookkeeping ────────────────────────────────────────────
        stagnation_counter = 0
        best_score_seen    = float(scores.max())

        # ── Main search loop ───────────────────────────────────────────────
        while self.budget_spent < self.budget:
            for k in range(K):
                if self.budget_spent >= self.budget:
                    break

                gbest[k] = _update_gbest(k)

                # Build a read-only state snapshot for operators.
                state = MemeticState(
                    current=current,
                    pbest=pbest,
                    gbest=gbest,
                    accs=accs,
                    lats=lats,
                    scores=scores,
                    weights=weights,
                    neighborhoods=neighborhoods,
                    T_sa=T_sa,
                )

                # ── Candidate generation: apply each operator, keep best ───
                best_cand: np.ndarray | None = None
                best_g    = -np.inf
                best_acc_c = 0.0
                best_lat_c = 0.0

                for op in self.candidate_ops:
                    if self.budget_spent >= self.budget:
                        break
                    cand       = op.apply(state, k, self.rng)
                    acc_c, lat_c = self._eval(cand)           # ← centralised NFE
                    g_c        = self._scalarize(weights[k], acc_c, lat_c)
                    if g_c > best_g:
                        best_g, best_cand  = g_c, cand
                        best_acc_c, best_lat_c = acc_c, lat_c

                if best_cand is None:
                    break  # budget exhausted before any operator could run

                # ── Acceptance (SA Metropolis or greedy) ───────────────────
                if self.sa_op is not None:
                    accepted, T_sa = self.sa_op.step(scores[k], best_g, self.rng)
                else:
                    accepted = best_g > scores[k]

                if accepted:
                    current[k] = best_cand.copy()
                    scores[k]  = best_g
                    accs[k]    = best_acc_c
                    lats[k]    = best_lat_c

                # ── Update personal best ───────────────────────────────────
                if best_g > pbest_scores[k]:
                    pbest[k]        = best_cand.copy()
                    pbest_scores[k] = best_g

                # ── Propagate improvement to neighbourhood ─────────────────
                for n in neighborhoods[k]:
                    g_n = self._scalarize(weights[n], best_acc_c, best_lat_c)
                    if g_n > self._scalarize(weights[n], accs[n], lats[n]):
                        gbest[n] = best_cand.copy()

            # ── Diversity restart ──────────────────────────────────────────
            if self.use_restart:
                curr_best = float(scores.max())
                if curr_best - best_score_seen > _RESTART_DELTA:
                    best_score_seen    = curr_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= _RESTART_PATIENCE:
                    worst_half = np.argsort(scores)[: K // 2]
                    for k in worst_half:
                        if self.budget_spent >= self.budget:
                            break
                        current[k]       = self._random_arch()
                        accs[k], lats[k] = self._eval(current[k])
                        scores[k]        = self._scalarize(weights[k], accs[k], lats[k])
                        pbest[k]         = current[k].copy()
                        pbest_scores[k]  = scores[k]
                    stagnation_counter = 0
                    if self.sa_op is not None:
                        T_sa = self.sa_op.reheat()   # reheat after restart

        return self.global_archive
