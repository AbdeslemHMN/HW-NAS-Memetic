"""
Discrete (Probabilistic) PSO for NAS-Bench-201 search space.
Architecture vector: L=6 edges, values in {0, 1, 2, 3, 4}.

Research Insight:
- Standard PSO velocity is continuous and inapplicable here. The probabilistic
  update replaces velocity with independent per-edge attraction probabilities:
    P1 = c1 * r1  (pull toward personal best)
    P2 = c2 * r2  (pull toward global best)
  Priority order: pbest > gbest > current, preventing premature convergence
  while maintaining locality of search (key for epistasis-sensitive DAGs).
- Expected number of edges moved per step ~ c1 + c2, which decouples step
  size from continuous space geometry, satisfying discrete strictness.
"""
import numpy as np

ARCH_LEN = 6
OPS_COUNT = 5  # {0, 1, 2, 3, 4}


def pso_step(
    current: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    c1: float = 0.5,
    c2: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Probabilistic discrete PSO step for one particle.

    For each edge j:
      P1 = c1 * r1,  P2 = c2 * r2   (r1, r2 ~ U(0,1))
      new[j] = pbest[j]   if R < P1
               gbest[j]   if R < P2  (evaluated only when R >= P1)
               current[j] otherwise

    :param current: Current architecture, shape (6,), values in [0, 4].
    :param pbest:   Personal best architecture, same shape and range.
    :param gbest:   Global best architecture, same shape and range.
    :param c1:      Personal attraction coefficient (0 < c1 <= 1).
    :param c2:      Global attraction coefficient  (0 < c2 <= 1).
    :param rng:     Optional seeded Generator for reproducibility.
    :return:        Updated architecture, shape (6,), values in [0, 4].
    """
    assert current.shape == pbest.shape == gbest.shape == (ARCH_LEN,), \
        "All architecture vectors must have shape (6,)."
    assert 0.0 < c1 <= 1.0 and 0.0 < c2 <= 1.0, "c1 and c2 must be in (0, 1]."

    rng = rng or np.random.default_rng()

    r1 = rng.random(ARCH_LEN)
    r2 = rng.random(ARCH_LEN)
    R  = rng.random(ARCH_LEN)

    P1 = c1 * r1   # shape (6,)
    P2 = c2 * r2   # shape (6,)

    new = current.copy()
    pull_pbest = R < P1
    pull_gbest = (~pull_pbest) & (R < P2)

    new[pull_pbest] = pbest[pull_pbest]
    new[pull_gbest] = gbest[pull_gbest]

    # Clip to enforce strict [0, 4] range (defensive, values should already comply)
    return np.clip(new, 0, OPS_COUNT - 1).astype(np.int64)
