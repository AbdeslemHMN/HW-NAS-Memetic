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


def nsga2_search(
    eval_fn: Callable[[np.ndarray], tuple[float, float]],
    budget: int,
    pop_size: int = 50,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Run NSGA-II until the evaluation budget is exhausted.

    :param eval_fn:   Architecture evaluator: arch -> (accuracy, latency).
    :param budget:    Total number of allowed evaluations.
    :param pop_size:  NSGA-II population size.
    :param rng_seed:  Random seed for reproducibility.
    :return:          Archive of all evaluated solutions:
                      list of {'arch', 'accuracy', 'latency'} dicts.
    """
    archive: list[dict] = []

    def tracked_eval(arch: np.ndarray) -> tuple[float, float]:
        acc, lat = eval_fn(arch)
        archive.append(
            {"arch": arch.copy(), "accuracy": float(acc), "latency": float(lat)}
        )
        return acc, lat

    # n_gen derived from budget: first generation costs pop_size evals,
    # each subsequent generation costs pop_size evals via offspring.
    n_gen = max(1, budget // pop_size)

    problem = HWNASProblem(tracked_eval)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    pymoo_minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=rng_seed,
        verbose=False,
    )

    return archive
