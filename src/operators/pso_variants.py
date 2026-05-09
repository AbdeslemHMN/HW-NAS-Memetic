"""
src/algorithms/operators/pso_variants.py
========================================
Five improved PSO operator variants to benchmark against the baseline.

Each is a drop-in replacement for PSOOperator — identical interface:
    candidate = op.apply(state, k, rng)

Variants
--------
1. PSOLinearDecay      — c1 decreases, c2 increases over iterations
2. PSOInertia          — adds random reset probability w (exploration term)
3. PSOAdaptive         — c1/c2 self-adapt per sub-problem from acceptance rate
4. PSOBlockWise        — updates edges in correlated DAG blocks (anti-epistasis)
5. PSOVelocityMemory   — biases toward repeating last successful edge move
"""
from __future__ import annotations

import numpy as np
from src.algorithms.operators.base_operator import BaseOperator, MemeticState

N_OPS   = 5
N_EDGES = 6

# NAS-Bench-201 DAG blocks (structurally correlated edge groups)
# Edge layout: 0=(0→1), 1=(0→2), 2=(1→2), 3=(0→3), 4=(1→3), 5=(2→3)
DAG_BLOCKS = [
    [0, 1],    # block 0: input  edges from node 0
    [2, 3],    # block 1: middle edges
    [4, 5],    # block 2: output edges into node 3
]


def _rand_diff(current_op: int, rng: np.random.Generator) -> int:
    choices = [op for op in range(N_OPS) if op != current_op]
    return int(rng.choice(choices))


# ── Variant 1: Linear Decay ───────────────────────────────────────────────────

class PSOLinearDecay(BaseOperator):
    """
    c1 decays linearly, c2 grows linearly over the search budget.

    Early search: high c1 → trust personal memory, explore widely.
    Late search:  high c2 → converge toward neighbourhood best.

    Call op.step() once per NFE (or per iteration) to advance the schedule.
    """

    def __init__(
        self,
        c1_start: float = 0.8,
        c1_end:   float = 0.1,
        c2_start: float = 0.1,
        c2_end:   float = 0.8,
        total_steps: int = 2000,
    ) -> None:
        self.c1_start    = c1_start
        self.c1_end      = c1_end
        self.c2_start    = c2_start
        self.c2_end      = c2_end
        self.total_steps = max(total_steps, 1)
        self._t          = 0

    def step(self) -> None:
        """Advance schedule by one step. Call once per NFE."""
        self._t = min(self._t + 1, self.total_steps)

    @property
    def c1(self) -> float:
        p = self._t / self.total_steps
        return self.c1_start + p * (self.c1_end - self.c1_start)

    @property
    def c2(self) -> float:
        p = self._t / self.total_steps
        return self.c2_start + p * (self.c2_end - self.c2_start)

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        current, pbest, gbest = state.current[k], state.pbest[k], state.gbest[k]
        c1, c2 = self.c1, self.c2
        r1 = rng.random(N_EDGES)
        r2 = rng.random(N_EDGES)
        R  = rng.random(N_EDGES)
        P1 = c1 * r1
        P2 = c2 * r2
        child = current.copy()
        pull_pbest = R < P1
        pull_gbest = (~pull_pbest) & (R < P2)
        child[pull_pbest] = pbest[pull_pbest]
        child[pull_gbest] = gbest[pull_gbest]
        return np.clip(child, 0, N_OPS - 1).astype(np.int64)

    def __repr__(self) -> str:
        return (f"PSOLinearDecay(c1={self.c1:.3f}, c2={self.c2:.3f}, "
                f"step={self._t}/{self.total_steps})")


# ── Variant 2: Inertia Weight ─────────────────────────────────────────────────

class PSOInertia(BaseOperator):
    """
    Adds explicit inertia: with probability w, randomly reset an edge
    (exploration), regardless of pbest/gbest.

    Per-edge decision order:
        R < w              → random reset  (inertia / exploration)
        R < w + c1*r1      → adopt pbest[j]
        R < w + c1*r1+c2*r2→ adopt gbest[j]
        else               → keep current[j]

    Set w = 1/6 to match GA's expected Hamming distance of 1 per step.
    Set w = 0.0 to recover original PSO behaviour (without random resets).
    """

    def __init__(self, c1: float = 0.4, c2: float = 0.4, w: float = 1/6) -> None:
        self.c1 = c1
        self.c2 = c2
        self.w  = w

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        current, pbest, gbest = state.current[k], state.pbest[k], state.gbest[k]
        child = current.copy()
        for j in range(N_EDGES):
            r  = rng.random()
            r1 = rng.random()
            r2 = rng.random()
            t_inertia = self.w
            t_pbest   = self.w + self.c1 * r1
            t_gbest   = self.w + self.c1 * r1 + self.c2 * r2
            if r < t_inertia:
                child[j] = _rand_diff(int(current[j]), rng)
            elif r < t_pbest:
                child[j] = pbest[j]
            elif r < t_gbest:
                child[j] = gbest[j]
        return np.clip(child, 0, N_OPS - 1).astype(np.int64)

    def __repr__(self) -> str:
        return f"PSOInertia(c1={self.c1}, c2={self.c2}, w={self.w:.4f})"


# ── Variant 3: Adaptive c1/c2 ────────────────────────────────────────────────

class PSOAdaptive(BaseOperator):
    """
    Self-adapts c1 and c2 per sub-problem based on rolling acceptance rate.

    acceptance_rate > target → exploitation working → increase c2, decrease c1
    acceptance_rate < target → stuck             → increase c1, decrease c2

    Call op.record(k, accepted) after every PSO candidate is evaluated.
    """

    def __init__(
        self,
        K: int,
        c1_init:     float = 0.5,
        c2_init:     float = 0.5,
        target_rate: float = 0.20,
        eta:         float = 0.05,
        c_min:       float = 0.05,
        c_max:       float = 0.95,
        window:      int   = 20,
    ) -> None:
        self.target_rate = target_rate
        self.eta         = eta
        self.c_min       = c_min
        self.c_max       = c_max
        self.window      = window
        self.c1 = np.full(K, c1_init, dtype=float)
        self.c2 = np.full(K, c2_init, dtype=float)
        self._hist = np.zeros((K, window), dtype=np.int8)
        self._ptr  = np.zeros(K, dtype=int)
        self._fill = np.zeros(K, dtype=int)

    def record(self, k: int, accepted: bool) -> None:
        """Call after every evaluation to update acceptance history for sub-problem k."""
        p = self._ptr[k]
        self._hist[k, p] = int(accepted)
        self._ptr[k]  = (p + 1) % self.window
        self._fill[k] = min(self._fill[k] + 1, self.window)
        n = self._fill[k]
        if n < 5:
            return
        rate = float(self._hist[k].sum()) / n
        if rate > self.target_rate:
            self.c2[k] = min(self.c2[k] + self.eta, self.c_max)
            self.c1[k] = max(self.c1[k] - self.eta, self.c_min)
        else:
            self.c1[k] = min(self.c1[k] + self.eta, self.c_max)
            self.c2[k] = max(self.c2[k] - self.eta, self.c_min)

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        current, pbest, gbest = state.current[k], state.pbest[k], state.gbest[k]
        c1, c2 = float(self.c1[k]), float(self.c2[k])
        r1 = rng.random(N_EDGES)
        r2 = rng.random(N_EDGES)
        R  = rng.random(N_EDGES)
        child = current.copy()
        pull_pbest = R < c1 * r1
        pull_gbest = (~pull_pbest) & (R < c2 * r2)
        child[pull_pbest] = pbest[pull_pbest]
        child[pull_gbest] = gbest[pull_gbest]
        return np.clip(child, 0, N_OPS - 1).astype(np.int64)

    def __repr__(self) -> str:
        return f"PSOAdaptive(K={len(self.c1)}, target={self.target_rate}, eta={self.eta})"


# ── Variant 4: Block-Wise Update ──────────────────────────────────────────────

class PSOBlockWise(BaseOperator):
    """
    Updates edges in correlated DAG blocks instead of independently.

    NAS-Bench-201 DAG blocks:
        Block 0 — [0,1] — input  edges (node0 → node1/2)
        Block 1 — [2,3] — middle edges
        Block 2 — [4,5] — output edges (node1/2 → node3)

    At each step:
        - One block is selected (uniformly random OR by historical gain).
        - ALL edges in the block are pulled from the same source (pbest or gbest).
        - Remaining edges keep their current value.

    This preserves intra-block co-adaptation, directly reducing the chance
    of breaking functionally coupled edge groups (epistasis).
    """

    def __init__(
        self,
        c1: float = 0.5,
        c2: float = 0.5,
        blocks: list[list[int]] | None = None,
        use_score_bias: bool = False,
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.blocks = blocks if blocks is not None else DAG_BLOCKS
        self.use_score_bias = use_score_bias
        # track empirical gain per block for score-biased selection
        self._block_gains  = np.ones(len(self.blocks), dtype=float)
        self._block_counts = np.ones(len(self.blocks), dtype=float)

    def record_gain(self, block_idx: int, delta: float) -> None:
        """Optionally record score gain for score-biased block selection."""
        if delta > 0:
            self._block_gains[block_idx]  += delta
            self._block_counts[block_idx] += 1

    def _select_block(self, rng: np.random.Generator) -> int:
        if self.use_score_bias:
            rates = self._block_gains / self._block_counts
            probs = rates / rates.sum()
            return int(rng.choice(len(self.blocks), p=probs))
        return int(rng.integers(0, len(self.blocks)))

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        current, pbest, gbest = state.current[k], state.pbest[k], state.gbest[k]
        child = current.copy()

        block_idx = self._select_block(rng)
        block     = self.blocks[block_idx]

        r1, r2 = rng.random(), rng.random()
        if rng.random() < self.c1 * r1:
            # pull whole block from pbest
            for j in block:
                child[j] = pbest[j]
        elif rng.random() < self.c2 * r2:
            # pull whole block from gbest
            for j in block:
                child[j] = gbest[j]
        # else: keep entire block (inertia)

        return np.clip(child, 0, N_OPS - 1).astype(np.int64)

    def __repr__(self) -> str:
        return (f"PSOBlockWise(c1={self.c1}, c2={self.c2}, "
                f"blocks={self.blocks}, score_bias={self.use_score_bias})")


# ── Variant 5: Velocity Memory ────────────────────────────────────────────────

class PSOVelocityMemory(BaseOperator):
    """
    Remembers the last successful move per edge (per sub-problem) and biases
    toward repeating it with probability p_repeat.

    'Successful move' = an edge change that led to a score improvement.
    The memory stores the target operation that was adopted last time
    the edge improved. This is the discrete analogue of velocity memory
    in continuous PSO.

    Call op.record_move(k, j, new_op, improved) after each accepted step.

    If no successful move is remembered for edge j, falls back to standard
    PSO (pbest/gbest attraction) — no regression.
    """

    def __init__(
        self,
        K: int,
        c1: float = 0.4,
        c2: float = 0.4,
        p_repeat: float = 0.3,
    ) -> None:
        self.c1       = c1
        self.c2       = c2
        self.p_repeat = p_repeat
        # velocity[k][j] = last successful op for sub-problem k, edge j
        # -1 means no memory yet
        self.velocity = np.full((K, N_EDGES), -1, dtype=np.int64)

    def record_move(self, k: int, edge_j: int, new_op: int, improved: bool) -> None:
        """
        Record a move for sub-problem k, edge j.
        Call after every accepted candidate.
        Only stores the move if it led to a score improvement.
        """
        if improved:
            self.velocity[k, edge_j] = new_op

    def record_candidate(
        self,
        k: int,
        old_arch: np.ndarray,
        new_arch: np.ndarray,
        improved: bool,
    ) -> None:
        """Convenience: record all changed edges at once."""
        for j in range(N_EDGES):
            if new_arch[j] != old_arch[j]:
                self.record_move(k, j, int(new_arch[j]), improved)

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        current, pbest, gbest = state.current[k], state.pbest[k], state.gbest[k]
        child = current.copy()

        for j in range(N_EDGES):
            r1 = rng.random()
            r2 = rng.random()
            rv = rng.random()

            has_memory = self.velocity[k, j] >= 0

            if has_memory and rv < self.p_repeat:
                # repeat last successful move
                child[j] = int(self.velocity[k, j])
            elif rng.random() < self.c1 * r1:
                child[j] = pbest[j]
            elif rng.random() < self.c2 * r2:
                child[j] = gbest[j]
            # else: keep current

        return np.clip(child, 0, N_OPS - 1).astype(np.int64)

    def __repr__(self) -> str:
        return (f"PSOVelocityMemory(K={len(self.velocity)}, "
                f"c1={self.c1}, c2={self.c2}, p_repeat={self.p_repeat})")