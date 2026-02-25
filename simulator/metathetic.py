"""Metathetic multi-agent TAP dynamics.

CLAIM POLICY LABEL: exploratory
This module extends the TAP framework with multi-agent identity
transformations (metathesis) to produce ensemble-level diagnostics
(Heaps' law, concentration, diversification). It does NOT derive from
the source TAP/biocosmology literature.

Four modes of metathetic transition:
  Mode 1 (Self): Agent transforms its own type-identity.
  Mode 2 (Absorptive cross): Two converging agents merge into one.
  Mode 3 (Novel cross): Two aligned agents produce a fundamentally new entity.
  Mode 4 (Environmental): Containing structure drifts at slow timescale.

Theoretical basis:
  - Metathesis concept from chemistry/linguistics
  - Emery & Trist (1965) causal texture types for Mode 4
  - TAP combinatorial dynamics (Cortes et al.) for agent-level innovation
  - Taalbi (2025) resource-constrained recombinant search
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Sequence

from .tap import compute_birth_term
from .analysis import classify_regime, adaptive_xi_plateau_threshold


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class MetatheticAgent:
    """An agent with a mutable type-identity portfolio and preserved knowledge.

    type_set: set of integer type-IDs representing the agent's product portfolio
    k: cumulative innovations (preserved across all metathetic transitions)
    M_local: current realized objects (local TAP state)
    active: True if participating in dynamics; False = dormant (state preserved)
    dM_history: recent dM/dt values for goal-alignment computation
    """
    agent_id: int
    type_set: set[int]
    k: float
    M_local: float
    active: bool = True
    dM_history: list[float] = field(default_factory=list)

    # -- Mode 1: Self-metathesis ------------------------------------------

    def self_metathesize(self, next_type_id: int) -> None:
        """Mode 1: Self-metathesis — gain a new type, preserve k.

        The agent's combinatorial identity shifts while its accumulated
        knowledge is fully preserved. Analogous to a company entering
        a new product line or a single-mutation event in biology.
        """
        self.type_set = self.type_set | {next_type_id}

    # -- Mode 2: Absorptive cross-metathesis ------------------------------

    @staticmethod
    def absorptive_cross(a1: MetatheticAgent, a2: MetatheticAgent) -> None:
        """Mode 2: Absorptive cross-metathesis.

        The agent with higher M_local absorbs the other.
        Composite gets union of type_sets and sum of k.
        Absorbed agent goes dormant (state preserved for potential
        recursive re-entry).
        """
        if a1.M_local >= a2.M_local:
            absorber, absorbed = a1, a2
        else:
            absorber, absorbed = a2, a1

        absorber.type_set = absorber.type_set | absorbed.type_set
        absorber.k += absorbed.k
        absorber.M_local += absorbed.M_local
        absorbed.active = False

    # -- Mode 3: Novel cross-metathesis -----------------------------------

    @staticmethod
    def novel_cross(
        a1: MetatheticAgent,
        a2: MetatheticAgent,
        child_id: int,
        next_type_id: int,
    ) -> MetatheticAgent:
        """Mode 3: Novel cross-metathesis.

        Both parents go dormant. A new agent spawns with recombined types
        plus a novel type that neither parent could produce alone — the
        combinatorial product of their interaction creating something
        fundamentally new, analogous to cross-species speciation.
        """
        # Recombined types: union of parents plus one novel type from
        # the combinatorial interaction.
        recombined = a1.type_set | a2.type_set | {next_type_id}

        child = MetatheticAgent(
            agent_id=child_id,
            type_set=recombined,
            k=a1.k + a2.k,
            M_local=a1.M_local + a2.M_local,
            active=True,
        )

        a1.active = False
        a2.active = False

        return child


# ---------------------------------------------------------------------------
# Environment (Mode 4)
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentState:
    """Mode 4: Environmental/systemic metathesis — the containing structure.

    Quasi-constants that drift at a timescale much slower than agent dynamics.
    Maps to Emery & Trist (1965) L22 causal texture processes: the environment
    has its own internal dynamics that constrain and are affected by agent
    behavior, but operate at a fundamentally different (more diffuse) rate.

    texture_type mapping to Emery & Trist:
        1 = Placid, randomized  (TAP plateau)
        2 = Placid, clustered   (TAP exponential)
        3 = Disturbed-reactive  (TAP precursor-active)
        4 = Turbulent fields    (TAP explosive)
    """
    a_env: float = 8.0
    K_env: float = 1e5
    texture_type: int = 1
    _a_env_base: float = 8.0
    _K_env_base: float = 1e5

    def innovation_potential(self, total_M: float) -> float:
        """Remaining room for novelty: K_env - total M, floored at 0."""
        return max(0.0, self.K_env - total_M)

    def cross_metathesis_threshold(self) -> float:
        """Threshold for cross-metathesis eligibility, scaled by texture type.

        Type I (placid): high threshold — hard to cross-metathesize
            in a stable environment.
        Type IV (turbulent): low threshold — radical transformation
            is facilitated when the environment itself is in flux.
        """
        base = 1.0
        scaling = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5}
        return base * scaling.get(self.texture_type, 1.0)

    def update(self, D_total: int, k_total: float, total_M: float, regime: str) -> None:
        """Drift environment state based on aggregate agent behavior.

        Called every env_update_interval steps. The drift rate is intentionally
        slow (1% per update) — the environment's dynamics operate at a much
        more diffuse timescale than agent dynamics, in most cases appearing
        as quasi-constants (analogous to physical constants that are emergent
        from deeper dynamics but effectively fixed on human timescales).
        """
        # a_env drifts: more diversity -> lower a (richer adjacency structure).
        if D_total > 0:
            a_target = max(2.0, self._a_env_base * (10.0 / max(10.0, float(D_total))))
            self.a_env += 0.01 * (a_target - self.a_env)

        # K_env drifts: more total k -> higher K (innovation creates resources).
        K_target = self._K_env_base + 0.1 * k_total
        self.K_env += 0.01 * (K_target - self.K_env)

        # Texture type transitions based on regime classification.
        regime_to_texture = {
            "plateau": 1,
            "exponential": 2,
            "precursor-active": 3,
            "explosive": 4,
            "extinction": 1,
        }
        self.texture_type = regime_to_texture.get(regime, 1)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _goal_alignment(h1: Sequence[float], h2: Sequence[float], window: int = 5) -> float:
    """Correlation of recent dM/dt histories (goal alignment).

    Returns value in [-1, 1]. Higher positive = more aligned trajectories.
    This captures whether two agents are heading in the same direction —
    their "goal weight" in the metathetic trigger condition.
    """
    tail1 = list(h1[-window:])
    tail2 = list(h2[-window:])
    n = min(len(tail1), len(tail2))
    if n < 2:
        return 0.0

    # Guard against overflow from explosive TAP dynamics.
    try:
        m1 = sum(tail1[:n]) / n
        m2 = sum(tail2[:n]) / n
        cov = sum((tail1[i] - m1) * (tail2[i] - m2) for i in range(n))
        var1 = sum((tail1[i] - m1) ** 2 for i in range(n))
        var2 = sum((tail2[i] - m2) ** 2 for i in range(n))

        if not (math.isfinite(cov) and math.isfinite(var1) and math.isfinite(var2)):
            return 0.0

        denom = math.sqrt(var1 * var2)
        if denom < 1e-15:
            return 0.0
        return max(-1.0, min(1.0, cov / denom))
    except (OverflowError, ValueError):
        return 0.0


def _agent_weight(agent: MetatheticAgent, all_types: dict[int, int]) -> float:
    """Distinctiveness: fraction of agent's types not widely shared.

    Types held by many agents are common; types held by few are distinctive.
    High distinctiveness = strong individual identity = harder to cross-metathesize.
    """
    if not agent.type_set:
        return 0.0
    total_agents = max(1, max(all_types.values())) if all_types else 1
    distinctive = sum(1 for t in agent.type_set
                      if all_types.get(t, 0) <= max(1, int(total_agents * 0.3)))
    return distinctive / len(agent.type_set)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class MetatheticEnsemble:
    """Multi-agent TAP simulation with four-mode metathetic dynamics.

    Each agent runs a local TAP step per timestep. Metathetic transitions
    (self, absorptive cross, novel cross, environmental) are checked and
    applied based on threshold conditions.

    The ensemble produces trajectory data suitable for long-run diagnostics
    (Heaps' law, Gini, diversification rate) via simulator.longrun.
    """

    def __init__(
        self,
        n_agents: int,
        initial_M: float,
        alpha: float,
        a: float,
        mu: float,
        variant: str = "baseline",
        alpha1: float = 0.0,
        carrying_capacity: float | None = None,
        env_update_interval: int = 10,
        self_meta_threshold: float = 0.5,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.a = a
        self.mu = mu
        self.variant = variant
        self.alpha1 = alpha1
        self.carrying_capacity = carrying_capacity
        self.env_update_interval = env_update_interval
        self.self_meta_threshold = self_meta_threshold

        self._rng = _random.Random(seed)
        self._next_type_id = n_agents + 1
        self._next_agent_id = n_agents

        # Initialize agents with distinct initial types.
        # Type 0 is shared across all agents; type i+1 is unique to agent i.
        self.agents: list[MetatheticAgent] = []
        for i in range(n_agents):
            types = {0, i + 1}
            self.agents.append(MetatheticAgent(
                agent_id=i,
                type_set=types,
                k=0.0,
                M_local=initial_M * (0.8 + 0.4 * self._rng.random()),
            ))

        self.env = EnvironmentState(
            a_env=a,
            K_env=carrying_capacity or 1e5,
            _a_env_base=a,
            _K_env_base=carrying_capacity or 1e5,
        )

        # Event counters.
        self.n_self_metatheses = 0
        self.n_absorptive_cross = 0
        self.n_novel_cross = 0
        self.n_env_transitions = 0

    def _active_agents(self) -> list[MetatheticAgent]:
        return [a for a in self.agents if a.active]

    def _all_type_counts(self) -> dict[int, int]:
        """Count how many active agents hold each type."""
        counts: dict[int, int] = {}
        for a in self._active_agents():
            for t in a.type_set:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def _total_diversity(self) -> int:
        """Count unique types across all active agents."""
        all_types: set[int] = set()
        for a in self._active_agents():
            all_types |= a.type_set
        return len(all_types)

    def _convergence_measure(self) -> float:
        """Fraction of agents sharing the most common type.

        High = convergent/localized (structure defines agents).
        Low = divergent/diffuse (agents define structure).
        """
        counts = self._all_type_counts()
        if not counts:
            return 0.0
        active = self._active_agents()
        if not active:
            return 0.0
        max_count = max(counts.values())
        return max_count / len(active)

    def _step_agents(self, m_cap: float = 1e4) -> None:
        """Run one local TAP step for each active agent.

        m_cap prevents explosive TAP dynamics from causing overflow.
        """
        for agent in self._active_agents():
            f = compute_birth_term(
                agent.M_local,
                alpha=self.alpha,
                a=self.env.a_env,
                variant=self.variant,
                alpha1=self.alpha1,
                carrying_capacity=self.carrying_capacity,
            )

            if not math.isfinite(f):
                f = 0.0

            B = min(f, m_cap)
            D = self.mu * agent.M_local
            dM = B - D

            agent.dM_history.append(dM)
            if len(agent.dM_history) > 10:
                agent.dM_history = agent.dM_history[-10:]

            agent.M_local = min(m_cap, max(0.0, agent.M_local + dM))
            agent.k += max(0.0, B)

    def _check_self_metathesis(self) -> None:
        """Mode 1: Self-metathesis for each active agent.

        Trigger: agent's innovation rate exceeds a threshold proportional
        to its current portfolio size. Only fires when the environment
        has remaining innovation potential (room for novelty).
        """
        ip = self.env.innovation_potential(
            sum(a.M_local for a in self._active_agents())
        )
        if ip <= 0:
            return

        for agent in self._active_agents():
            if not agent.dM_history:
                continue
            dM_recent = agent.dM_history[-1]
            threshold = self.self_meta_threshold * len(agent.type_set)
            if dM_recent > threshold:
                agent.self_metathesize(self._next_type_id)
                self._next_type_id += 1
                self.n_self_metatheses += 1

    def _check_cross_metathesis(self) -> None:
        """Modes 2 & 3: Pairwise cross-metathesis checks.

        For each pair of active agents, compute:
          L = likeness (Jaccard similarity of type-sets)
          G = goal alignment (correlation of dM/dt histories)
          W = agent weight (distinctiveness of each agent)

        Cross-metathesis eligible when: L + G > (W_i + W_j) * threshold
        Mode selection: L > G -> absorptive; G > L -> novel.

        Only one cross-metathesis event fires per step to avoid cascades.
        """
        active = self._active_agents()
        if len(active) < 2:
            return

        type_counts = self._all_type_counts()
        cross_threshold = self.env.cross_metathesis_threshold()

        pairs_checked: set[tuple[int, int]] = set()
        for i, a1 in enumerate(active):
            if not a1.active:
                continue
            for j, a2 in enumerate(active):
                if i >= j or not a2.active:
                    continue
                pair_key = (min(a1.agent_id, a2.agent_id),
                            max(a1.agent_id, a2.agent_id))
                if pair_key in pairs_checked:
                    continue
                pairs_checked.add(pair_key)

                L = _jaccard(a1.type_set, a2.type_set)
                G = _goal_alignment(a1.dM_history, a2.dM_history)
                G = max(0.0, G)  # Only positive alignment counts
                W1 = _agent_weight(a1, type_counts)
                W2 = _agent_weight(a2, type_counts)

                if L + G <= (W1 + W2) * cross_threshold:
                    continue

                # Eligible — determine mode.
                if L > G:
                    # Mode 2: Absorptive cross-metathesis.
                    MetatheticAgent.absorptive_cross(a1, a2)
                    self.n_absorptive_cross += 1
                else:
                    # Mode 3: Novel cross-metathesis.
                    self._next_agent_id += 1
                    child = MetatheticAgent.novel_cross(
                        a1, a2,
                        child_id=self._next_agent_id,
                        next_type_id=self._next_type_id,
                    )
                    self._next_type_id += 1
                    self.agents.append(child)
                    self.n_novel_cross += 1

                # One cross-metathesis per step to avoid cascading.
                return

    def _update_environment(self, step: int) -> None:
        """Mode 4: Environmental update at slow timescale.

        The environment's causal texture (Emery & Trist L22 processes)
        drifts based on aggregate agent behavior. This captures how
        the containing structure is affected by the entities within it,
        but at a fundamentally slower rate.
        """
        if step % self.env_update_interval != 0 or step == 0:
            return

        active = self._active_agents()
        D = self._total_diversity()
        k_total = sum(a.k for a in active)
        total_M = sum(a.M_local for a in active)

        # Classify aggregate regime for texture type determination.
        # Use a minimal proxy trajectory for regime classification.
        m_traj = [max(1.0, total_M * 0.9), max(1.0, total_M)]
        xi_traj = [max(0.0, k_total * 0.9), max(0.0, k_total)]
        thr = adaptive_xi_plateau_threshold(xi_traj)
        regime = classify_regime(xi_traj, m_traj, thr)

        old_texture = self.env.texture_type
        self.env.update(D, k_total, total_M, regime)
        if self.env.texture_type != old_texture:
            self.n_env_transitions += 1

    def run(self, steps: int) -> list[dict]:
        """Run the ensemble for the given number of steps.

        Returns a list of per-step snapshot dicts suitable for analysis
        with simulator.longrun diagnostics.
        """
        trajectory: list[dict] = []

        for step in range(steps):
            # 1. Agent-level TAP steps.
            self._step_agents()

            # 2. Metathetic transitions (Modes 1-3).
            self._check_self_metathesis()
            self._check_cross_metathesis()

            # 3. Environmental update (Mode 4, slow timescale).
            self._update_environment(step)

            # 4. Record ensemble snapshot.
            active = self._active_agents()
            dormant = [a for a in self.agents if not a.active]
            agent_k_list = [a.k for a in active]

            snapshot = {
                "step": step,
                "D_total": self._total_diversity(),
                "k_total": sum(a.k for a in active),
                "total_M": sum(a.M_local for a in active),
                "n_active": len(active),
                "n_dormant": len(dormant),
                "agent_k_list": agent_k_list,
                "convergence": self._convergence_measure(),
                "texture_type": self.env.texture_type,
                "a_env": self.env.a_env,
                "K_env": self.env.K_env,
                "innovation_potential": self.env.innovation_potential(
                    sum(a.M_local for a in active)
                ),
                "n_self_metatheses": self.n_self_metatheses,
                "n_absorptive_cross": self.n_absorptive_cross,
                "n_novel_cross": self.n_novel_cross,
            }
            trajectory.append(snapshot)

        return trajectory
