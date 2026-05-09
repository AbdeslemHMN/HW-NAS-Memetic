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
_RESTART_SUBPROBLEM_PATIENCE: int = 30
_RESTART_FRACTION_MAX: float = 0.35
_SA_ACCEPT_WINDOW: int = 40
_SA_MIN_TEMP: float = 1e-3
_SA_REHEAT_FACTOR: float = 1.25
_SA_REHEAT_TRIGGER: float = 0.15


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
    T_neighborhood : int
        Neighbourhood size T_n for gbest selection and propagation.
    use_restart : bool
        Re-initialise the worst K//2 solutions when the population stagnates
        for ``_RESTART_PATIENCE`` outer iterations.
    rng : np.random.Generator or None
        Seeded random generator for reproducibility.
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, float]],
        budget: int,
        candidate_ops: list[BaseOperator],
        sa_op: SAOperator | None = None,
        K: int = 20,
        T_neighborhood: int = 2,
        use_restart: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not candidate_ops:
            raise ValueError(
                "candidate_ops must contain at least one BaseOperator instance."
            )
        super().__init__(eval_fn, budget, rng)
        self.candidate_ops  = list(candidate_ops)
        self.sa_op          = sa_op
        self.K              = max(1, int(K))
        self.T_neighborhood = max(1, int(T_neighborhood))
        self.use_restart    = bool(use_restart)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _scalarize(w: np.ndarray, acc: float, lat: float) -> float:
        """g(a | w) = w₀ · Acc − w₁ · Lat  (maximise)."""
        return float(w[0] * acc - w[1] * lat)

    def _make_weights(self) -> np.ndarray:
        K = self.K
        if K == 1:
            return np.array([[0.5, 0.5]])
        return np.column_stack(
            [np.linspace(1.0, 0.0, K), np.linspace(0.0, 1.0, K)]
        )  # (K, 2)

    def _restart_arch(self, anchor: np.ndarray) -> np.ndarray:
        """Create a restart candidate by mutating an anchor architecture."""
        cand = anchor.copy()
        n_mut = int(self.rng.integers(1, 3))
        idxs = self.rng.choice(ARCH_LEN, size=n_mut, replace=False)
        for i in np.atleast_1d(idxs):
            curr = int(cand[i])
            options = [o for o in range(OPS_COUNT) if o != curr]
            cand[i] = int(self.rng.choice(options))
        return cand

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

        # SA temperature — owned by the orchestrator, not the operator.
        T_sa = self.sa_op.T0 if self.sa_op is not None else 1.0

        # ── Restart bookkeeping ────────────────────────────────────────────
        stagnation_counter = 0
        best_score_seen    = float(scores.max())
        no_improve_steps   = np.zeros(K, dtype=int)

        # ── SA acceptance tracking (for adaptive reheating) ───────────────
        accepted_history: list[int] = []

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
                    accepted = self.sa_op.accept(scores[k], best_g, T_sa, self.rng)
                else:
                    accepted = best_g > scores[k]
                accepted_history.append(int(accepted))
                if len(accepted_history) > _SA_ACCEPT_WINDOW:
                    accepted_history.pop(0)

                if accepted:
                    current[k] = best_cand.copy()
                    scores[k]  = best_g
                    accs[k]    = best_acc_c
                    lats[k]    = best_lat_c

                # ── Update personal best ───────────────────────────────────
                if best_g > pbest_scores[k]:
                    pbest[k]        = best_cand.copy()
                    pbest_scores[k] = best_g
                    no_improve_steps[k] = 0
                else:
                    no_improve_steps[k] += 1

                # ── Propagate improvement to neighbourhood ─────────────────
                for n in neighborhoods[k]:
                    g_n = self._scalarize(weights[n], best_acc_c, best_lat_c)
                    if g_n > self._scalarize(weights[n], accs[n], lats[n]):
                        gbest[n] = best_cand.copy()

            if self.sa_op is not None:
                # Cool once per outer sweep and keep a non-zero floor so SA remains active.
                T_sa = max(_SA_MIN_TEMP, self.sa_op.cool(T_sa))
                if len(accepted_history) == _SA_ACCEPT_WINDOW:
                    acc_rate = float(np.mean(accepted_history))
                    if acc_rate < _SA_REHEAT_TRIGGER:
                        T_sa = min(self.sa_op.T0, max(_SA_MIN_TEMP, T_sa * _SA_REHEAT_FACTOR))

            # ── Diversity restart ──────────────────────────────────────────
            if self.use_restart:
                curr_best = float(scores.max())
                if curr_best - best_score_seen > _RESTART_DELTA:
                    best_score_seen    = curr_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= _RESTART_PATIENCE:
                    stale = np.where(no_improve_steps >= _RESTART_SUBPROBLEM_PATIENCE)[0]
                    if stale.size == 0:
                        stale = np.argsort(scores)[: max(1, K // 4)]
                    max_restart = max(1, int(np.ceil(_RESTART_FRACTION_MAX * K)))
                    restart_idxs = stale[np.argsort(scores[stale])[:max_restart]]
                    elite_anchor = current[int(np.argmax(scores))].copy()

                    for k in restart_idxs:
                        if self.budget_spent >= self.budget:
                            break

                        # Hybrid restart: mostly guided mutations around current elite,
                        # occasionally full random re-sampling to inject diversity.
                        if self.rng.random() < 0.7:
                            current[k] = self._restart_arch(elite_anchor)
                        else:
                            current[k] = self._random_arch()

                        accs[k], lats[k] = self._eval(current[k])
                        scores[k]        = self._scalarize(weights[k], accs[k], lats[k])
                        pbest[k]         = current[k].copy()
                        pbest_scores[k]  = scores[k]
                        no_improve_steps[k] = 0
                    stagnation_counter = 0
                    if self.sa_op is not None:
                        T_sa = self.sa_op.T0   # reheat after restart

        return self.global_archive
