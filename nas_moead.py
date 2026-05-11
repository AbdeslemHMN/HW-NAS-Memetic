"""
MOEA/D + Discrete GA + Discrete Probabilistic PSO
for Multi-Objective Hardware-Aware NAS on HW-NAS-Bench / NAS-Bench-201

Pipeline:
  - K weight directions covering the accuracy-latency trade-off space
  - Each direction runs GA (mutation + crossover) and discrete PSO
  - Simulated annealing acceptance rule
  - MOEA/D neighborhood propagation
  - Global Pareto archive extracted at the end
  - Stagnation detection with partial restart

Usage:
    python nas_moead.py

Requirements:
    pip install numpy

Note:
    This implementation uses a SYNTHETIC benchmark by default (random lookup table)
    so it runs without any external files. To use real HW-NAS-Bench data, replace
    the BenchmarkLookup class with your actual data loader.
"""

import numpy as np
import random
import math
import time
from copy import deepcopy
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. BENCHMARK LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkLookup:
    """
    Wraps the HW-NAS-Bench / NAS-Bench-201 lookup table.

    Each architecture is a tuple of 6 integers in {0,1,2,3,4}.
    Returns (accuracy, latency) for any valid architecture.

    Replace _build_synthetic_bench() with your real data loader:
        self.table = { tuple(arch): (acc, lat) for arch, acc, lat in your_data }
    """

    OPS = {
        0: 'nor_conv_3x3',
        1: 'nor_conv_1x1',
        2: 'avg_pool_3x3',
        3: 'skip_connect',
        4: 'none',
    }
    N_OPS   = 5   # operations per edge
    N_EDGES = 6   # edges in the cell DAG
    N_ARCHS = 5 ** 6  # 15,625

    def __init__(self, data: Optional[Dict] = None, device: str = 'raspberry_pi'):
        """
        Args:
            data:   dict mapping tuple(arch) -> (accuracy, latency).
                    If None, a synthetic benchmark is generated for testing.
            device: name of target hardware platform (informational only).
        """
        self.device = device
        if data is not None:
            self.table = data
        else:
            print("[BenchmarkLookup] No real data provided — using synthetic benchmark.")
            self.table = self._build_synthetic_bench()
        print(f"[BenchmarkLookup] Loaded {len(self.table)} architectures for device='{device}'.")

    def _build_synthetic_bench(self) -> Dict:
        """
        Generates a synthetic benchmark for testing purposes.
        Accuracy and latency are deterministic functions of the architecture vector.
        """
        rng = np.random.RandomState(42)
        table = {}
        for idx in range(self.N_ARCHS):
            # decode index to 6-integer architecture
            arch = []
            tmp = idx
            for _ in range(self.N_EDGES):
                arch.append(tmp % self.N_OPS)
                tmp //= self.N_OPS
            arch = tuple(arch)

            # synthetic accuracy: correlated with number of conv ops
            n_conv = sum(1 for op in arch if op in (0, 1))
            n_skip = sum(1 for op in arch if op == 3)
            n_none = sum(1 for op in arch if op == 4)
            base_acc = 70.0 + 4.0 * n_conv + 1.5 * n_skip - 3.0 * n_none
            acc = float(np.clip(base_acc + rng.normal(0, 1.5), 50.0, 94.0))

            # synthetic latency: heavier ops = more latency
            base_lat = 5.0 + 2.0 * n_conv + 0.5 * n_skip
            lat = float(np.clip(base_lat + rng.normal(0, 0.5), 1.0, 30.0))

            table[arch] = (acc, lat)
        return table

    def query(self, arch: List[int]) -> Tuple[float, float]:
        """
        Returns (accuracy, latency) for the given architecture.
        arch: list or tuple of 6 integers in {0..4}
        """
        key = tuple(arch)
        if key not in self.table:
            raise KeyError(f"Architecture {key} not found in benchmark.")
        return self.table[key]

    def random_arch(self) -> List[int]:
        """Returns a uniformly random valid architecture."""
        return [random.randint(0, self.N_OPS - 1) for _ in range(self.N_EDGES)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCALARIZATION (MOEA/D weight directions)
# ─────────────────────────────────────────────────────────────────────────────

def scalar_score(acc: float, lat: float, w_acc: float, w_lat: float,
                 acc_norm: float = 94.0, lat_norm: float = 30.0) -> float:
    """
    Weighted scalarization: g = w_acc * (acc/acc_norm) - w_lat * (lat/lat_norm)

    Both objectives are normalized to [0,1] range before weighting.
    This avoids scale dominance between accuracy (%) and latency (ms).
    """
    return w_acc * (acc / acc_norm) - w_lat * (lat / lat_norm)


def build_weight_directions(K: int) -> List[Tuple[float, float]]:
    """
    Builds K uniformly spaced weight vectors covering [accuracy-focused .. latency-focused].
    Returns list of (w_acc, w_lat) tuples with w_acc + w_lat = 1.
    """
    directions = []
    for k in range(K):
        w_acc = 1.0 - k / (K - 1) if K > 1 else 0.5
        w_lat = 1.0 - w_acc
        directions.append((w_acc, w_lat))
    return directions


# ─────────────────────────────────────────────────────────────────────────────
# 3. GA OPERATORS (discrete, on {0..4}^6)
# ─────────────────────────────────────────────────────────────────────────────

def ga_mutate(arch: List[int], pm: float = 1/6, n_ops: int = 5) -> List[int]:
    """
    Per-position mutation: each edge is mutated independently with probability pm.
    The new operation is always different from the current one.
    """
    child = arch[:]
    for j in range(len(arch)):
        if random.random() < pm:
            choices = [op for op in range(n_ops) if op != child[j]]
            child[j] = random.choice(choices)
    # Ensure at least one mutation happened
    if child == arch:
        j = random.randint(0, len(arch) - 1)
        choices = [op for op in range(n_ops) if op != arch[j]]
        child[j] = random.choice(choices)
    return child


def ga_crossover(arch1: List[int], arch2: List[int]) -> List[int]:
    """
    Single-point crossover between two parent architectures.
    Returns one child.
    """
    n = len(arch1)
    point = random.randint(1, n - 1)
    return arch1[:point] + arch2[point:]


def ga_uniform_crossover(arch1: List[int], arch2: List[int]) -> List[int]:
    """
    Uniform crossover: each position independently sampled from either parent.
    """
    return [arch1[j] if random.random() < 0.5 else arch2[j] for j in range(len(arch1))]


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISCRETE PSO OPERATOR
# ─────────────────────────────────────────────────────────────────────────────

def pso_discrete_update(arch: List[int], pbest: List[int], gbest: List[int],
                        c1: float = 1.5, c2: float = 1.5) -> List[int]:
    """
    Discrete probabilistic PSO update.

    For each edge position j:
        r, r1, r2 ~ Uniform(0, 1)
        if r < c1*r1          → adopt pbest[j]   (personal attraction)
        elif r < c1*r1+c2*r2  → adopt gbest[j]   (global attraction)
        else                  → keep arch[j]      (inertia)

    This mirrors the continuous PSO velocity equation in a discrete categorical space.
    Result is always in {0..4}^6 — no correction needed.
    """
    child = arch[:]
    for j in range(len(arch)):
        r  = random.random()
        r1 = random.random()
        r2 = random.random()
        threshold_pbest = c1 * r1
        threshold_gbest = c1 * r1 + c2 * r2
        if r < threshold_pbest:
            child[j] = pbest[j]
        elif r < threshold_gbest:
            child[j] = gbest[j]
        # else: keep current (inertia)
    return child


# ─────────────────────────────────────────────────────────────────────────────
# 5. PARETO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """
    Returns True if solution a dominates solution b.
    Objectives: (accuracy to maximize, latency to minimize).
    Converted internally: higher accuracy = better, lower latency = better.
    a dominates b if a is at least as good on all objectives and strictly better on one.
    """
    acc_a, lat_a = a
    acc_b, lat_b = b
    # a >= b on accuracy AND a <= b on latency
    at_least_as_good = (acc_a >= acc_b) and (lat_a <= lat_b)
    strictly_better  = (acc_a > acc_b) or (lat_a < lat_b)
    return at_least_as_good and strictly_better


def extract_pareto_front(archive: List[Dict]) -> List[Dict]:
    """
    Filters the archive to keep only non-dominated solutions.
    Each entry in archive is a dict with keys: arch, accuracy, latency.
    """
    pareto = []
    for i, sol_a in enumerate(archive):
        dominated = False
        for j, sol_b in enumerate(archive):
            if i == j:
                continue
            if dominates((sol_b['accuracy'], sol_b['latency']),
                         (sol_a['accuracy'], sol_a['latency'])):
                dominated = True
                break
        if not dominated:
            pareto.append(sol_a)
    return pareto


def hypervolume_2d(pareto_front: List[Dict],
                   ref_acc: float = 50.0, ref_lat: float = 35.0) -> float:
    """
    Computes the 2D hypervolume indicator for a Pareto front.
    Reference point: (ref_acc=50%, ref_lat=35ms) — dominated by all reasonable solutions.
    Higher hypervolume = better coverage of the objective space.
    """
    if not pareto_front:
        return 0.0
    # Sort by accuracy descending
    front = sorted(pareto_front, key=lambda x: -x['accuracy'])
    hv = 0.0
    prev_lat = ref_lat
    for sol in front:
        width  = sol['accuracy'] - ref_acc
        height = prev_lat - sol['latency']
        if width > 0 and height > 0:
            hv += width * height
        prev_lat = min(prev_lat, sol['latency'])
    return hv


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN ALGORITHM: MOEA/D + GA + DISCRETE PSO
# ─────────────────────────────────────────────────────────────────────────────

class MOEAD_GA_PSO:
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
    with Discrete Genetic Algorithm and Discrete Probabilistic PSO.

    Search space: NAS-Bench-201 / HW-NAS-Bench (15,625 architectures)
    Objectives:   maximize accuracy, minimize latency
    """

    def __init__(
        self,
        bench: BenchmarkLookup,
        K: int = 5,           # number of MOEA/D weight directions
        N: int = 20,          # population size per direction
        T_neighbors: int = 2, # MOEA/D neighborhood size
        pm: float = 1/6,      # GA mutation probability per edge
        pc: float = 0.9,      # GA crossover probability
        c1: float = 1.5,      # PSO personal best attraction
        c2: float = 1.5,      # PSO global best attraction
        T0: float = 1.0,      # SA initial temperature
        alpha: float = 0.99,  # SA cooling factor
        T_min: float = 0.01,  # SA minimum temperature
        N_stagnation: int = 50,  # iterations before partial restart
        budget_max: int = 2000,  # maximum benchmark lookups
        verbose: bool = True,
    ):
        self.bench        = bench
        self.K            = K
        self.N            = N
        self.T_nbr        = T_neighbors
        self.pm           = pm
        self.pc           = pc
        self.c1           = c1
        self.c2           = c2
        self.T0           = T0
        self.alpha        = alpha
        self.T_min        = T_min
        self.N_stagnation = N_stagnation
        self.budget_max   = budget_max
        self.verbose      = verbose

        # Build weight directions
        self.directions = build_weight_directions(K)

        # Compute neighborhood for each direction
        self.neighbors = self._compute_neighbors()

        # State variables (initialized in run())
        self.populations  = None   # populations[k][i] = arch (list of 6 ints)
        self.scores       = None   # scores[k][i] = scalar score
        self.fitness      = None   # fitness[k][i] = (accuracy, latency)
        self.pbest        = None   # pbest[k][i] = best arch seen by individual i in direction k
        self.pbest_score  = None
        self.gbest        = None   # gbest[k] = best arch seen in direction k
        self.gbest_score  = None
        self.archive      = []     # global archive of all evaluated solutions
        self.archive_set  = set()  # set of arch tuples already in archive (deduplication)
        self.lookup_count = 0      # total benchmark queries
        self.history      = []     # (iteration, best_acc, min_lat, hypervolume)

    # ── Initialization helpers ────────────────────────────────────────────

    def _compute_neighbors(self) -> List[List[int]]:
        """For each direction k, finds the T_nbr closest directions by weight distance."""
        neighbors = []
        for k in range(self.K):
            dists = []
            for k2 in range(self.K):
                if k2 == k:
                    continue
                d = abs(self.directions[k][0] - self.directions[k2][0])
                dists.append((d, k2))
            dists.sort()
            neighbors.append([idx for _, idx in dists[:self.T_nbr]])
        return neighbors

    def _lookup(self, arch: List[int]) -> Tuple[float, float]:
        """Query benchmark and update counter. Returns (accuracy, latency)."""
        self.lookup_count += 1
        return self.bench.query(arch)

    def _scalar(self, acc: float, lat: float, k: int) -> float:
        w_acc, w_lat = self.directions[k]
        return scalar_score(acc, lat, w_acc, w_lat)

    def _add_to_archive(self, arch: List[int], acc: float, lat: float):
        """Adds solution to global archive if not already present."""
        key = tuple(arch)
        if key not in self.archive_set:
            self.archive_set.add(key)
            self.archive.append({'arch': arch[:], 'accuracy': acc, 'latency': lat})

    def _init_population(self):
        """Initializes populations, pbest, gbest for all K directions."""
        self.populations  = []
        self.scores       = []
        self.fitness      = []
        self.pbest        = []
        self.pbest_score  = []
        self.gbest        = []
        self.gbest_score  = []

        for k in range(self.K):
            pop_k    = []
            score_k  = []
            fit_k    = []
            pbest_k  = []
            pbest_s_k = []

            for _ in range(self.N):
                arch = self.bench.random_arch()
                acc, lat = self._lookup(arch)
                s = self._scalar(acc, lat, k)
                self._add_to_archive(arch, acc, lat)

                pop_k.append(arch)
                score_k.append(s)
                fit_k.append((acc, lat))
                pbest_k.append(arch[:])
                pbest_s_k.append(s)

            # gbest = best individual in this direction
            best_idx = int(np.argmax(score_k))
            self.populations.append(pop_k)
            self.scores.append(score_k)
            self.fitness.append(fit_k)
            self.pbest.append(pbest_k)
            self.pbest_score.append(pbest_s_k)
            self.gbest.append(pop_k[best_idx][:])
            self.gbest_score.append(score_k[best_idx])

    # ── Candidate generation ──────────────────────────────────────────────

    def _generate_ga_candidate(self, k: int, i: int) -> List[int]:
        """Generates a candidate using GA mutation or crossover."""
        arch = self.populations[k][i]
        if random.random() < self.pc:
            # crossover: pick second parent from this direction or a neighbor direction
            pool = list(range(self.N))
            pool.remove(i)
            # also consider individuals from neighbor directions
            for k2 in self.neighbors[k]:
                pool_indices = [(k2, j) for j in range(self.N)]
                chosen = random.choice(pool_indices)
                parent2 = self.populations[chosen[0]][chosen[1]]
                break
            else:
                i2 = random.choice(pool)
                parent2 = self.populations[k][i2]
            return ga_crossover(arch, parent2)
        else:
            return ga_mutate(arch, self.pm)

    def _generate_pso_candidate(self, k: int, i: int) -> List[int]:
        """Generates a candidate using discrete probabilistic PSO."""
        return pso_discrete_update(
            self.populations[k][i],
            self.pbest[k][i],
            self.gbest[k],
            self.c1, self.c2
        )

    # ── Acceptance rule (simulated annealing inspired) ────────────────────

    def _accept(self, delta: float, T: float) -> bool:
        """Returns True if the candidate should replace the current solution."""
        if delta > 0:
            return True
        if T <= self.T_min:
            return False
        return random.random() < math.exp(delta / T)

    # ── MOEA/D neighborhood propagation ──────────────────────────────────

    def _propagate_to_neighbors(self, k: int, arch: List[int], acc: float, lat: float):
        """
        If gbest[k] improves any neighbor direction's score,
        replace that direction's worst individual with this solution.
        """
        for k2 in self.neighbors[k]:
            s_in_k2 = self._scalar(acc, lat, k2)
            if s_in_k2 > self.gbest_score[k2]:
                # Replace worst individual in k2
                worst_idx = int(np.argmin(self.scores[k2]))
                self.populations[k2][worst_idx] = arch[:]
                self.scores[k2][worst_idx]      = s_in_k2
                self.fitness[k2][worst_idx]     = (acc, lat)
                # Update pbest if better
                if s_in_k2 > self.pbest_score[k2][worst_idx]:
                    self.pbest[k2][worst_idx]       = arch[:]
                    self.pbest_score[k2][worst_idx] = s_in_k2
                # Update gbest for k2
                self.gbest[k2]       = arch[:]
                self.gbest_score[k2] = s_in_k2

    # ── Stagnation restart ────────────────────────────────────────────────

    def _partial_restart(self, T: float) -> float:
        """Replaces 30% worst individuals per direction with fresh random architectures."""
        if self.verbose:
            print(f"    [Restart] Stagnation detected — refreshing 30% of population.")
        n_replace = max(1, int(0.3 * self.N))
        for k in range(self.K):
            sorted_idx = sorted(range(self.N), key=lambda i: self.scores[k][i])
            for i in sorted_idx[:n_replace]:
                arch = self.bench.random_arch()
                acc, lat = self._lookup(arch)
                s = self._scalar(acc, lat, k)
                self._add_to_archive(arch, acc, lat)
                self.populations[k][i] = arch
                self.scores[k][i]      = s
                self.fitness[k][i]     = (acc, lat)
                self.pbest[k][i]       = arch[:]
                self.pbest_score[k][i] = s
        # Reset temperature partially
        return max(T0 := self.T0 / 2, self.T_min)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """
        Runs the full MOEA/D + GA + PSO search.
        Returns the final Pareto front as a list of dicts:
            [{'arch': [...], 'accuracy': float, 'latency': float}, ...]
        """
        if self.verbose:
            print("=" * 60)
            print(f"MOEA/D + GA + Discrete PSO — NAS Search")
            print(f"K={self.K} directions, N={self.N} pop/direction")
            print(f"Budget: {self.budget_max} lookups")
            print("=" * 60)

        # ── Initialize ──
        self._init_population()
        T = self.T0
        iteration = 0
        stagnation_counter = 0
        best_global_score = max(max(self.scores[k]) for k in range(self.K))

        if self.verbose:
            print(f"[Init] {self.lookup_count} lookups used for initialization.")

        # ── Main loop ──
        while self.lookup_count < self.budget_max:
            iteration += 1
            improved_this_iter = False

            for k in range(self.K):
                for i in range(self.N):
                    if self.lookup_count >= self.budget_max:
                        break

                    # ── Generate two candidates: one GA, one PSO ──
                    cand_ga  = self._generate_ga_candidate(k, i)
                    cand_pso = self._generate_pso_candidate(k, i)

                    # ── Evaluate both ──
                    acc_ga,  lat_ga  = self._lookup(cand_ga)
                    acc_pso, lat_pso = self._lookup(cand_pso)
                    self._add_to_archive(cand_ga,  acc_ga,  lat_ga)
                    self._add_to_archive(cand_pso, acc_pso, lat_pso)

                    s_ga  = self._scalar(acc_ga,  lat_ga,  k)
                    s_pso = self._scalar(acc_pso, lat_pso, k)

                    # ── Select best of the two candidates ──
                    if s_ga >= s_pso:
                        best_cand, best_acc, best_lat, best_s = cand_ga, acc_ga, lat_ga, s_ga
                    else:
                        best_cand, best_acc, best_lat, best_s = cand_pso, acc_pso, lat_pso, s_pso

                    # ── Acceptance rule ──
                    delta = best_s - self.scores[k][i]
                    if self._accept(delta, T):
                        self.populations[k][i] = best_cand
                        self.scores[k][i]      = best_s
                        self.fitness[k][i]     = (best_acc, best_lat)

                        # Update personal best
                        if best_s > self.pbest_score[k][i]:
                            self.pbest[k][i]       = best_cand[:]
                            self.pbest_score[k][i] = best_s

                        # Update global best for this direction
                        if best_s > self.gbest_score[k]:
                            self.gbest[k]       = best_cand[:]
                            self.gbest_score[k] = best_s
                            improved_this_iter  = True
                            # Propagate to neighbors
                            self._propagate_to_neighbors(k, best_cand, best_acc, best_lat)

            # ── Cool temperature ──
            T = max(T * self.alpha, self.T_min)

            # ── Stagnation detection ──
            current_best = max(max(self.scores[k]) for k in range(self.K))
            if current_best > best_global_score + 1e-6:
                best_global_score  = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= self.N_stagnation:
                T = self._partial_restart(T)
                stagnation_counter = 0

            # ── Logging ──
            if self.verbose and iteration % 10 == 0:
                pareto_now = extract_pareto_front(self.archive)
                hv = hypervolume_2d(pareto_now)
                best_acc_now = max(s['accuracy'] for s in pareto_now) if pareto_now else 0
                min_lat_now  = min(s['latency']  for s in pareto_now) if pareto_now else 0
                print(f"  Iter {iteration:4d} | Lookups {self.lookup_count:5d} | "
                      f"T={T:.4f} | Pareto size={len(pareto_now):3d} | "
                      f"Best acc={best_acc_now:.2f}% | Min lat={min_lat_now:.2f}ms | HV={hv:.4f}")
                self.history.append((iteration, best_acc_now, min_lat_now, hv))

        # ── Extract final Pareto front ──
        pareto_final = extract_pareto_front(self.archive)
        pareto_final.sort(key=lambda x: -x['accuracy'])

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Search complete. Total lookups: {self.lookup_count}")
            print(f"Archive size: {len(self.archive)} unique architectures")
            print(f"Pareto front size: {len(pareto_final)}")
            hv_final = hypervolume_2d(pareto_final)
            print(f"Final hypervolume: {hv_final:.4f}")

        return pareto_final


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_pareto_front(pareto: List[Dict], top_n: int = 10):
    """Prints a formatted table of the Pareto front."""
    print("\n" + "─" * 65)
    print(f"{'#':>3}  {'Architecture':<25}  {'Accuracy':>9}  {'Latency':>9}")
    print("─" * 65)
    for idx, sol in enumerate(pareto[:top_n]):
        arch_str = str(sol['arch'])
        print(f"{idx+1:>3}  {arch_str:<25}  {sol['accuracy']:>8.2f}%  {sol['latency']:>8.2f}ms")
    if len(pareto) > top_n:
        print(f"    ... ({len(pareto) - top_n} more solutions)")
    print("─" * 65)


def select_for_device(pareto: List[Dict], latency_limit: float) -> Optional[Dict]:
    """
    Given a device latency constraint, selects the highest-accuracy
    architecture on the Pareto front that satisfies the constraint.
    """
    feasible = [s for s in pareto if s['latency'] <= latency_limit]
    if not feasible:
        print(f"  No architecture satisfies latency <= {latency_limit}ms.")
        return None
    best = max(feasible, key=lambda x: x['accuracy'])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading benchmark...")
    # ── Replace this with your real HW-NAS-Bench data loader ──
    # Example with real data:
    #   import json
    #   with open('hw_nas_bench_data.json') as f:
    #       raw = json.load(f)
    #   data = { tuple(v['arch']): (v['accuracy'], v['latency_raspberry_pi'])
    #            for v in raw.values() }
    #   bench = BenchmarkLookup(data=data, device='raspberry_pi')
    bench = BenchmarkLookup(device='raspberry_pi_4b_synthetic')

    # ── Configure and run the algorithm ──
    algo = MOEAD_GA_PSO(
        bench        = bench,
        K            = 5,      # number of weight directions
        N            = 20,     # population per direction
        T_neighbors  = 2,      # MOEA/D neighbors
        pm           = 1/6,    # mutation probability
        pc           = 0.9,    # crossover probability
        c1           = 1.5,    # PSO personal attraction
        c2           = 1.5,    # PSO global attraction
        T0           = 1.0,    # SA initial temperature
        alpha        = 0.99,   # SA cooling
        T_min        = 0.01,   # SA floor temperature
        N_stagnation = 50,     # stagnation threshold
        budget_max   = 1000,   # total benchmark lookups
        verbose      = True,
    )

    start_time = time.time()
    pareto_front = algo.run()
    elapsed = time.time() - start_time

    # ── Display results ──
    print(f"\nSearch time: {elapsed:.2f} seconds")
    print_pareto_front(pareto_front, top_n=15)

    # ── Device-specific selection ──
    print("\n── Device Deployment Selection ──")
    for limit in [5.0, 10.0, 20.0]:
        best = select_for_device(pareto_front, latency_limit=limit)
        if best:
            print(f"  Latency ≤ {limit:5.1f}ms → arch={best['arch']}  "
                  f"acc={best['accuracy']:.2f}%  lat={best['latency']:.2f}ms")

    return pareto_front


if __name__ == '__main__':
    main()