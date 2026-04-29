"""
NSGA-II baseline for HW-NAS-Memetic, powered by pymoo 0.6.

Objectives (both minimised by pymoo convention):
  F[0] = -Accuracy   (negate to convert maximisation → minimisation)
  F[1] =  Latency

Research Insights:
- NSGA-II uses fast non-dominated sorting + crowding-distance selection,
  giving O(M * N^2) per generation (M=2 objectives, N=pop_size).
  Compared to MOEA/D's O(K) scalar decomposition, this is the reference
  complexity the proposed algorithm must outperform.
- Integer rounding repair is applied post-crossover/mutation to strictly
  enforce {0..4}^6 without introducing a continuous relaxation bias.
- eval_fn is wrapped to accumulate an external archive, decoupling pymoo's
  internal population management from the framework's logging contract.

Budget fairness:
- Termination is ('n_eval', budget): pymoo stops the moment the cumulative
  number of function evaluations reaches 'budget', regardless of how many
  full generations that represents. This guarantees an identical NFE cap
  to the proposed algorithm's hard-stop guard in BaseOptimizer._eval().

Hyperparameter justification (standard NSGA-II literature values):
- SBX crossover: prob_c=0.9, η_c=20  (Deb et al., 2002)
- PM mutation  : prob_m=1/n_var, η_m=20  (one gene expected mutated/individual)
"""
from typing import Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize as pymoo_minimize

ARCH_LEN = 6
OPS_COUNT = 5  # {0, 1, 2, 3, 4}
_SNAPSHOT_INTERVAL = 100


class HWNASProblem(ElementwiseProblem):
    """
    pymoo problem wrapper for NAS-Bench-201 / HW-NAS-Bench evaluation.

    Variables : 6 integer-valued edges, each in [0, 4].
    Objectives: minimise [-Accuracy, Latency].
    """

    def __init__(self, eval_fn: Callable[[np.ndarray], tuple[float, float]]):
        super().__init__(
            n_var=ARCH_LEN,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(ARCH_LEN, dtype=int),
            xu=np.full(ARCH_LEN, OPS_COUNT - 1, dtype=int),
            vtype=int,
        )
        self._eval_fn = eval_fn

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        arch = np.clip(np.round(x), 0, OPS_COUNT - 1).astype(np.int64)
        accuracy, latency = self._eval_fn(arch)
        out["F"] = [-accuracy, latency]  # pymoo minimises both


class BudgetExceededError(RuntimeError):
    """Internal exception used to stop NSGA-II exactly at the budget cap."""


def nsga2_search(
    eval_fn: Callable[[np.ndarray], tuple[float, float]],
    budget: int,
    pop_size: int = 50,
    rng_seed: int = 42,
) -> tuple[list[dict], list[int]]:
    """
    Run NSGA-II until the evaluation budget is exhausted.

    :param eval_fn:   Architecture evaluator: arch -> (accuracy, latency).
    :param budget:    Total number of allowed evaluations (strict NFE cap).
    :param pop_size:  NSGA-II population size.
    :param rng_seed: Random seed for reproducibility.
    :return:         (archive, nfe_checkpoints) where:
                        archive         — list of {'arch', 'accuracy', 'latency'}
                        nfe_checkpoints — list of NFE values where snapshots were taken
    """
    archive: list[dict] = []
    nfe_checkpoints: list[int] = []

    def tracked_eval(arch: np.ndarray) -> tuple[float, float]:
        if len(archive) >= budget:
            raise BudgetExceededError("Evaluation budget exhausted")
        acc, lat = eval_fn(arch)
        archive.append(
            {"arch": arch.copy(), "accuracy": float(acc), "latency": float(lat)}
        )
        nfe = len(archive)
        if nfe % _SNAPSHOT_INTERVAL == 0:
            nfe_checkpoints.append(nfe)
        return acc, lat

    problem = HWNASProblem(tracked_eval)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        # Standard NSGA-II probabilities from Deb et al. (2002):
        #   SBX  — crossover prob 0.9, distribution index η=20
        #   PM   — mutation  prob 1/n_var (expect one mutation per individual), η=20
        crossover=SBX(prob=0.9, eta=20.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1 / ARCH_LEN, eta=20.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    try:
        pymoo_minimize(
            problem,
            algorithm,
            termination=("n_eval", budget),
            seed=rng_seed,
            verbose=False,
        )
    except BudgetExceededError:
        # The budget guard is strict: if pymoo asks for an extra evaluation
        # beyond the cap, we stop immediately and return the archive as-is.
        pass

    assert len(archive) == budget, (
        f"NSGA-II budget guard failed: expected {budget} evaluations, "
        f"got {len(archive)}"
    )
    return archive, nfe_checkpoints
