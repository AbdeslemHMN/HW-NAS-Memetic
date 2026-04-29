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

Feature toggles (for ablation studies):
- use_ga:      Enable/disable GA mutation as the global perturbation operator.
- use_pso:     Enable/disable Discrete PSO as the local refinement operator.
- use_sa:      Enable/disable Simulated Annealing acceptance (falls back to greedy).
- use_restart: Enable/disable diversity restarts when the population stagnates.

NOTE: use_ga and use_pso cannot both be False — at least one candidate generator
must be active. A ValueError is raised at __init__ time if both are disabled.

Configuration via dict (for ablation scripts)
─────────────────────────────────────────────
Pass a 'config' dict (typically loaded from a JSON file) to override defaults:

    cfg = json.load(open("configs/ablation_t2_no_pso.json"))
    opt = ProposedMoeadPso(eval_fn, config=cfg)

If 'config' is None the algorithm runs with all features enabled (full proposed).
Individual keyword arguments (K, T0, …) are also accepted for backward compatibility
with existing scripts such as run_search.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.algorithms.base_optimizer import BaseOptimizer, ARCH_LEN
from src.operators.discrete_ga import mutate
from src.operators.discrete_pso import pso_step
from src.operators.simulated_annealing import metropolis_accept, cool

# ── Default full-proposed configuration ──────────────────────────────────────
_FULL_PROPOSED_DEFAULTS: dict = {
    "budget": 2000,
    "k_directions": 20,
    "t_neighborhood": 2,
    "t0": 1.0,
    "alpha": 0.95,
    "c1": 0.5,
    "c2": 0.5,
    "use_ga": True,
    "use_pso": True,
    "use_sa": True,
    "use_restart": True,
}

# Stagnation window for diversity restart: if best scalarized score does not
# improve by at least RESTART_DELTA over RESTART_PATIENCE iterations, the
# sub-population is re-initialised randomly.
_RESTART_PATIENCE: int = 100   # iterations without improvement
_RESTART_DELTA: float = 1e-6   # minimum improvement threshold


class ProposedMoeadPso(BaseOptimizer):
    """
    Memetic MOEA/D using Discrete PSO as the local search engine,
    Discrete GA (mutation) for global perturbation, and
    Simulated Annealing for trajectory control.

    Supports feature-toggle ablation via a 'config' dict or keyword arguments.
    """

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, float]],
        budget: int | None = None,
        K: int | None = None,
        T_neighborhood: int | None = None,
        T0: float | None = None,
        alpha: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
        rng: np.random.Generator | None = None,
        config: dict | None = None,
    ):
        """
        Parameters
        ----------
        eval_fn : callable
            Maps an architecture array (shape 6,) to (accuracy, latency).
        budget : int, optional
            Total evaluation budget. Overrides config value when provided.
        K : int, optional
            Number of MOEA/D weight-vector directions. Overrides config.
        T_neighborhood : int, optional
            Neighbourhood size per sub-problem. Overrides config.
        T0 : float, optional
            SA initial temperature. Overrides config.
        alpha : float, optional
            SA geometric cooling rate in (0, 1). Overrides config.
        c1 : float, optional
            PSO personal-attraction coefficient. Overrides config.
        c2 : float, optional
            PSO global-attraction coefficient. Overrides config.
        rng : np.random.Generator, optional
            Seeded random generator for reproducibility.
        config : dict, optional
            Configuration dictionary (loaded from a JSON ablation config).
            Keys: budget, k_directions, t_neighborhood, t0, alpha, c1, c2,
                  use_ga, use_pso, use_sa, use_restart.
            If None, defaults to the full-proposed configuration.
            Individual keyword arguments take precedence over config values.
        """
        # ── Merge config with defaults ────────────────────────────────────
        cfg = dict(_FULL_PROPOSED_DEFAULTS)
        if config is not None:
            cfg.update(config)

        # Keyword arguments override config (backward-compatibility with run_search.py)
        resolved_budget      = budget       if budget       is not None else cfg["budget"]
        resolved_K           = K            if K            is not None else cfg["k_directions"]
        resolved_T_n         = T_neighborhood if T_neighborhood is not None else cfg["t_neighborhood"]
        resolved_T0          = T0           if T0           is not None else cfg["t0"]
        resolved_alpha       = alpha        if alpha        is not None else cfg["alpha"]
        resolved_c1          = c1           if c1           is not None else cfg["c1"]
        resolved_c2          = c2           if c2           is not None else cfg["c2"]

        # ── Feature toggles ───────────────────────────────────────────────
        self.use_ga      = bool(cfg.get("use_ga",      True))
        self.use_pso     = bool(cfg.get("use_pso",     True))
        self.use_sa      = bool(cfg.get("use_sa",      True))
        self.use_restart = bool(cfg.get("use_restart", True))

        # Guard: at least one candidate generator must be active.
        if not self.use_ga and not self.use_pso:
            raise ValueError(
                "Invalid ablation configuration: both 'use_ga' and 'use_pso' are False. "
                "At least one candidate-generation operator must be enabled."
            )

        super().__init__(eval_fn, resolved_budget, rng)

        if resolved_K < 1:
            raise ValueError("k_directions (K) must be >= 1.")

        self.K              = resolved_K
        self.T_neighborhood = max(1, resolved_T_n)
        self.T0             = resolved_T0
        self.alpha          = resolved_alpha
        self.c1             = resolved_c1
        self.c2             = resolved_c2

    @staticmethod
    def _scalarize(w: np.ndarray, acc: float, lat: float) -> float:
        """$$g(a|w) = w_0 \\cdot Acc - w_1 \\cdot Lat$$ (maximize)."""
        return float(w[0] * acc - w[1] * lat)

    def search(self) -> list[dict]:
        K, T_n = self.K, self.T_neighborhood

        # --- Weight vectors: K points linearly spaced from (1,0) → (0,1) ---
        if K == 1:
            # Degenerate case (ablation_t4_no_moead): single balanced direction.
            weights = np.array([[0.5, 0.5]])
        else:
            weights = np.column_stack(
                [np.linspace(1.0, 0.0, K), np.linspace(0.0, 1.0, K)]
            )  # (K, 2)

        # --- Neighborhoods: T_n nearest sub-problems by weight Euclidean distance ---
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

        # --- Restart bookkeeping (used only when use_restart=True) ---
        stagnation_counter: int = 0
        best_score_seen: float  = float(scores.max())

        # --- Main search loop ---
        while self.budget_spent < self.budget:
            for k in range(K):
                if self.budget_spent >= self.budget:
                    break

                # Refresh gbest[k] from current neighborhood state
                gbest[k] = _gbest(k)

                # ── Candidate generation ──────────────────────────────────
                # At least one of use_ga / use_pso is guaranteed True (checked in __init__).

                if self.use_ga and self.use_pso:
                    # FULL MODE: generate both, pick the better one.
                    cand_ga = mutate(current[k], rng=self.rng)
                    acc_ga, lat_ga = self._eval(cand_ga)
                    g_ga = self._scalarize(weights[k], acc_ga, lat_ga)

                    if self.budget_spent >= self.budget:
                        break

                    cand_pso = pso_step(
                        current[k], pbest[k], gbest[k], self.c1, self.c2, rng=self.rng
                    )
                    acc_pso, lat_pso = self._eval(cand_pso)
                    g_pso = self._scalarize(weights[k], acc_pso, lat_pso)

                    if g_ga >= g_pso:
                        cand, g_cand, acc_c, lat_c = cand_ga, g_ga, acc_ga, lat_ga
                    else:
                        cand, g_cand, acc_c, lat_c = cand_pso, g_pso, acc_pso, lat_pso

                elif self.use_ga:
                    # ABLATION T2 (no_pso): GA-only candidate generation.
                    cand = mutate(current[k], rng=self.rng)
                    acc_c, lat_c = self._eval(cand)
                    g_cand = self._scalarize(weights[k], acc_c, lat_c)

                else:
                    # ABLATION T3 (no_ga): PSO-only candidate generation.
                    cand = pso_step(
                        current[k], pbest[k], gbest[k], self.c1, self.c2, rng=self.rng
                    )
                    acc_c, lat_c = self._eval(cand)
                    g_cand = self._scalarize(weights[k], acc_c, lat_c)

                # ── Acceptance ────────────────────────────────────────────
                if self.use_sa:
                    # FULL MODE: Metropolis criterion (maximise via negation).
                    accepted = metropolis_accept(-scores[k], -g_cand, T_sa)
                else:
                    # ABLATION T5 (no_sa): greedy acceptance only.
                    accepted = g_cand > scores[k]

                if accepted:
                    current[k] = cand.copy()
                    scores[k]  = g_cand
                    accs[k]    = acc_c
                    lats[k]    = lat_c

                # ── Update pbest ──────────────────────────────────────────
                if g_cand > pbest_scores[k]:
                    pbest[k]        = cand.copy()
                    pbest_scores[k] = g_cand

                # ── Propagate gbest improvement to neighbors ──────────────
                for n in neighborhoods[k]:
                    g_cand_n = self._scalarize(weights[n], acc_c, lat_c)
                    g_curr_n = self._scalarize(weights[n], accs[n], lats[n])
                    if g_cand_n > g_curr_n:
                        gbest[n] = cand.copy()

                T_sa = cool(T_sa, self.alpha)

            # ── Diversity restart (use_restart toggle) ────────────────────
            if self.use_restart:
                current_best = float(scores.max())
                if current_best - best_score_seen > _RESTART_DELTA:
                    best_score_seen    = current_best
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= _RESTART_PATIENCE:
                    # Re-initialise the worst half of the population randomly.
                    worst_half = np.argsort(scores)[: K // 2]
                    for k in worst_half:
                        current[k]      = self._random_arch()
                        accs[k], lats[k] = self._eval(current[k]) if self.budget_spent < self.budget else (accs[k], lats[k])
                        scores[k]       = self._scalarize(weights[k], accs[k], lats[k])
                        pbest[k]        = current[k].copy()
                        pbest_scores[k] = scores[k]
                    stagnation_counter = 0
                    T_sa               = self.T0  # reheat SA after restart

        return self.global_archive
