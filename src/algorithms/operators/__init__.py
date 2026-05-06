"""
Pluggable search-operator strategy classes for MemeticNAS.

Exports
-------
MemeticState   — shared population snapshot passed to every operator.
BaseOperator   — abstract candidate-generation interface.
GAOperator     — discrete GA mutation (wraps src.operators.discrete_ga).
PSOOperator    — discrete probabilistic PSO (wraps src.operators.discrete_pso).
SAOperator     — Simulated Annealing acceptance (wraps src.operators.simulated_annealing).
"""
from src.algorithms.operators.base_operator import BaseOperator, MemeticState
from src.algorithms.operators.ga_operator import GAOperator
from src.algorithms.operators.pso_operator import PSOOperator
from src.algorithms.operators.sa_operator import SAOperator

__all__ = [
    "MemeticState",
    "BaseOperator",
    "GAOperator",
    "PSOOperator",
    "SAOperator",
]
