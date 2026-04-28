"""
MOEA/D + Discrete PSO + GA + SA hybrid optimizer for HW-NAS-Memetic.

Scalarization:  $$g(a | w_k) = w_{acc} \\cdot Acc - w_{lat} \\cdot Lat$$  (maximize)

Research Insights:
- K linear weight vectors decompose the bi-objective space into K scalar sub-problems,
  reducing Pareto-dominance sorting from O(N^2) to O(K) comparisons per cycle.
- Each sub-problem k maintains its own (current, pbest, gbest) PSO state aligned to w_k,
  linking discrete probabilistic PSO refinement directly to its trade-off direction.
- Neighborhood T_n bounds gbest influence propagation; small T_n (2-3) is critical to
  prevent premature homogenisation across distant weight directions (epistasis risk).
- Metropolis acceptance at each sub-problem acts as a temperature-controlled trajectory
  gate: high T permits diversity-preserving hill-climbs; low T → greedy exploitation.
"""
from typing import Callable

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer, ARCH_LEN
from src.operators.discrete_ga import mutate
from src.operators.discrete_pso import pso_step
from src.operators.simulated_annealing import metropolis_accept, cool


class ProposedMoeadPso(BaseOptimizer):
    """
    Memetic MOEA/D using Discrete PSO as the local search engine,
    Discrete GA (mutation) for global perturbation, and
    Simulated Annealing for trajectory control.
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, float]],
        budget: int,
        K: int = 20,
        T_neighborhood: int = 2,
        T0: float = 1.0,
        alpha: float = 0.95,
        c1: float = 0.5,
        c2: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        """
        :param K:              Number of sub-problems (weight vectors).
        :param T_neighborhood: Neighborhood size per sub-problem (includes self).
        :param T0:             SA initial temperature.
        :param alpha:          SA cooling rate, alpha in (0, 1).
        :param c1:             PSO personal attraction coefficient.
        :param c2:             PSO global attraction coefficient.
        """
        super().__init__(eval_fn, budget, rng)
        if K < 2:
            raise ValueError("K must be >= 2.")
        self.K = K
        self.T_neighborhood = max(1, T_neighborhood)
        self.T0 = T0
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def _scalarize(w: np.ndarray, acc: float, lat: float) -> float:
        """$$g(a|w) = w_0 \\cdot Acc - w_1 \\cdot Lat$$ (maximize)."""
        return float(w[0] * acc - w[1] * lat)

    def search(self) -> list[dict]:
        K, T_n = self.K, self.T_neighborhood

        # --- Weight vectors: K points linearly spaced from (1,0) → (0,1) ---
        weights = np.column_stack(
            [np.linspace(1.0, 0.0, K), np.linspace(0.0, 1.0, K)]
        )  # (K, 2)

        # --- Neighborhoods: T_n nearest sub-problems by weight Euclidean distance ---
        # dists[k, j] = ||w_k - w_j||_2; self (distance 0) is always index 0
        dists = np.linalg.norm(weights[:, None] - weights[None, :], axis=2)  # (K, K)
        neighborhoods = np.argsort(dists, axis=1)[:, :T_n]  # (K, T_n)

        # --- Initialise population ---
        current = np.array([self._random_arch() for _ in range(K)])  # (K, 6)
        accs = np.zeros(K, dtype=float)
        lats = np.zeros(K, dtype=float)

        for k in range(K):
            if self.budget_spent >= self.budget:
                break
            accs[k], lats[k] = self._eval(current[k])

        scores = np.array(
            [self._scalarize(weights[k], accs[k], lats[k]) for k in range(K)]
        )

        pbest = current.copy()          # personal best architectures  (K, 6)
        pbest_scores = scores.copy()    # personal best scalarized scores (K,)

        def _gbest(k: int) -> np.ndarray:
            """Return architecture with best g(.|w_k) among k's neighborhood."""
            neigh = neighborhoods[k]
            neigh_g = np.array(
                [self._scalarize(weights[k], accs[n], lats[n]) for n in neigh]
            )
            return current[neigh[np.argmax(neigh_g)]].copy()

        gbest = np.array([_gbest(k) for k in range(K)])  # (K, 6)

        T_sa = self.T0

        # --- Main search loop ---
        while self.budget_spent < self.budget:
            for k in range(K):
                if self.budget_spent >= self.budget:
                    break

                # Refresh gbest[k] from current neighborhood state
                gbest[k] = _gbest(k)

                # 1. GA candidate: per-gene mutation
                cand_ga = mutate(current[k], rng=self.rng)
                acc_ga, lat_ga = self._eval(cand_ga)
                g_ga = self._scalarize(weights[k], acc_ga, lat_ga)

                if self.budget_spent >= self.budget:
                    break

                # 2. PSO candidate: probabilistic discrete update
                cand_pso = pso_step(
                    current[k], pbest[k], gbest[k], self.c1, self.c2, rng=self.rng
                )
                acc_pso, lat_pso = self._eval(cand_pso)
                g_pso = self._scalarize(weights[k], acc_pso, lat_pso)

                # 3. Select better candidate by w_k scalarization
                if g_ga >= g_pso:
                    cand, g_cand, acc_c, lat_c = cand_ga, g_ga, acc_ga, lat_ga
                else:
                    cand, g_cand, acc_c, lat_c = cand_pso, g_pso, acc_pso, lat_pso

                # 4. Metropolis acceptance (SA maximises via negation convention)
                if metropolis_accept(-scores[k], -g_cand, T_sa):
                    current[k] = cand.copy()
                    scores[k] = g_cand
                    accs[k] = acc_c
                    lats[k] = lat_c

                # 5. Update pbest for sub-problem k
                if g_cand > pbest_scores[k]:
                    pbest[k] = cand.copy()
                    pbest_scores[k] = g_cand

                # 6. Propagate gbest improvement to neighbors
                for n in neighborhoods[k]:
                    g_cand_n = self._scalarize(weights[n], acc_c, lat_c)
                    g_curr_n = self._scalarize(weights[n], accs[n], lats[n])
                    if g_cand_n > g_curr_n:
                        gbest[n] = cand.copy()

                T_sa = cool(T_sa, self.alpha)

        return self.global_archive
