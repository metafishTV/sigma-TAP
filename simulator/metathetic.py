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
    steps_since_metathesis: int = 0
    _dormant_steps: int = 0
    _affordance_ticks: list[int] = field(default_factory=list)

    # -- L-matrix event ledger (Emery channels) ------------------------------
    # L11: intrapraxis (self-transformation)
    n_self_metatheses_local: int = 0
    # L12: system → environment (outward projection)
    n_novel_cross_local: int = 0
    n_absorptive_given_local: int = 0
    # L21: environment → system (inward reception)
    n_absorptive_received_local: int = 0
    # L22: causal texture (regime shifts imposed from outside)
    n_env_transitions_local: int = 0

    _dissolved: bool = field(default=False, init=False, repr=False)
    _deep_stasis: bool = field(default=False, init=False, repr=False)

    # -- Temporal orientation gate constants --------------------------------
    _NOVELTY_WINDOW: int = field(default=5, init=False, repr=False)
    _STAGNATION_THRESHOLD: int = field(default=50, init=False, repr=False)
    _RELATIONAL_DECAY_WINDOW: int = field(default=30, init=False, repr=False)
    _TRAJECTORY_DIVERGENCE_THR: float = field(default=-0.3, init=False, repr=False)
    _ESTABLISHED_ALIGNMENT_THR: float = field(default=0.5, init=False, repr=False)
    _ESTABLISHED_MIN_HISTORY: int = field(default=6, init=False, repr=False)
    _AFFORDANCE_WINDOW: int = field(default=10, init=False, repr=False)

    # -- Temporal orientation gate ------------------------------------------

    def _trajectory_alignment(self) -> float:
        """Compute trajectory alignment from recent dM_history.

        Returns a value in [-1, 1] indicating whether the agent's recent
        trajectory is improving (positive) or deteriorating (negative).
        """
        if len(self.dM_history) < 3:
            return 0.0
        tail = self.dM_history[-5:]
        mean_dM = sum(tail) / len(tail)
        first = tail[0]
        last = tail[-1]
        trend = (last - first) / max(1.0, abs(first) + 1e-10)
        if mean_dM > 0:
            return min(1.0, 0.3 + 0.7 * min(1.0, trend))
        else:
            return max(-1.0, -0.3 + 0.7 * max(-1.0, trend))

    @property
    def temporal_state(self) -> int:
        """Five-state temporal orientation gate.

        Returns:
            0 = disintegrated (no relational capacity)
            1 = inertial (grown away from identity)
            2 = situated (in-flow, productive)
            3 = desituated (novelty-shock or stagnation)
            4 = established (consummated; static tension)
        """
        if not self.active:
            return 3  # dormant defaults to desituated
        if self.steps_since_metathesis <= self._NOVELTY_WINDOW:
            return 3  # desituated/novelty
        if self.steps_since_metathesis >= self._STAGNATION_THRESHOLD:
            return 3  # desituated/stagnation

        alignment = self._trajectory_alignment()
        if alignment < self._TRAJECTORY_DIVERGENCE_THR:
            return 1  # inertial
        if (alignment > self._ESTABLISHED_ALIGNMENT_THR
                and len(self.dM_history) >= self._ESTABLISHED_MIN_HISTORY):
            return 4  # established
        return 2  # situated

    def temporal_state_with_context(self, active_type_counts: dict[int, int]) -> int:
        """Temporal state with ensemble context for disintegration detection.

        For inactive agents: checks if the agent has been dormant long enough
        AND no active agent holds any of its types. If both conditions are met,
        the agent is disintegrated (state 0) — its relational capacity is gone.

        Args:
            active_type_counts: mapping from type-ID to count of active agents
                holding that type.

        Returns:
            int: temporal state 0-4.
        """
        if not self.active:
            if self._dormant_steps >= self._RELATIONAL_DECAY_WINDOW:
                # Check if any active agent still holds one of this agent's types
                has_living_connection = any(
                    active_type_counts.get(t, 0) > 0 for t in self.type_set
                )
                if not has_living_connection:
                    return 0  # disintegrated
        return self.temporal_state

    @property
    def affordance_score(self) -> float:
        """Rolling mean of affordance ticks. 0.0 if no ticks recorded."""
        if not self._affordance_ticks:
            return 0.0
        return sum(self._affordance_ticks) / len(self._affordance_ticks)

    @property
    def taps_signature(self) -> str:
        """4-letter TAPS dispositional signature from L-matrix ledger.

        Each letter derived from the agent's local event history:
          T: Involution(I) / Evolution(E) / Transvolution(T)
          A: Expression(E) / Impression(I) / Adpression(A)
          P: Reflection-consumption(R) / Projection-consummation(U) / Pure action(X)
          S: Disintegration(D) / Preservation(P) / Integration(I) / Synthesis(S)

        Emery channel mapping:
          L11 (n_self_metatheses_local)      -> inward (T), synthesis (S)
          L12 (n_novel_cross_local + given)   -> outward (T), consummation (P)
          L21 (n_absorptive_received_local)   -> inward (T), consumption (P), integration (S)
          L22 (n_env_transitions_local)       -> outward (T), disintegration (S)
        """
        # -- T-letter: Transvolution --
        inward = self.n_self_metatheses_local + self.n_absorptive_received_local
        outward = (self.n_novel_cross_local + self.n_absorptive_given_local
                   + self.n_env_transitions_local)
        if inward > outward * 1.2:
            t_letter = "I"
        elif outward > inward * 1.2:
            t_letter = "E"
        else:
            t_letter = "T"

        # -- A-letter: Anopression --
        has_expression = (
            len(self.dM_history) > 0
            and self.dM_history[-1] > 0
            and self.affordance_score > 0.5
        )
        has_adpression = self.steps_since_metathesis == 0
        l21_dominates = (
            self.n_absorptive_received_local > 0
            and self.n_absorptive_received_local >= self.n_novel_cross_local
            and self.n_absorptive_received_local >= self.n_self_metatheses_local
        )
        if has_adpression:
            a_letter = "A"
        elif l21_dominates:
            a_letter = "I"
        elif has_expression:
            a_letter = "E"
        else:
            a_letter = "E"  # default: expression as unmarked case

        # -- P-letter: Praxis --
        l21_total = self.n_absorptive_received_local
        l12_total = self.n_novel_cross_local + self.n_absorptive_given_local
        if l21_total > l12_total * 1.2:
            p_letter = "R"
        elif l12_total > l21_total * 1.2:
            p_letter = "U"
        else:
            p_letter = "X"

        # -- S-letter: Syntegration --
        if not self.active:
            s_letter = "P"  # dormant = preservation
        else:
            synthesis_count = self.n_self_metatheses_local + self.n_novel_cross_local
            integration_count = self.n_absorptive_received_local
            disintegration_signals = self.n_env_transitions_local
            counts = {
                "S": synthesis_count,
                "I": integration_count,
                "D": disintegration_signals,
            }
            s_letter = max(counts, key=counts.get) if any(counts.values()) else "S"

        return t_letter + a_letter + p_letter + s_letter

    # -- Mode 1: Self-metathesis ------------------------------------------

    def self_metathesize(self, next_type_id: int) -> None:
        """Mode 1: Self-metathesis — gain a new type, preserve k.

        The agent's combinatorial identity shifts while its accumulated
        knowledge is fully preserved. Analogous to a company entering
        a new product line or a single-mutation event in biology.
        """
        self.type_set = self.type_set | {next_type_id}
        self.steps_since_metathesis = 0
        self.n_self_metatheses_local += 1

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
        absorber.n_absorptive_received_local += 1
        absorbed.n_absorptive_given_local += 1

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

        a1.n_novel_cross_local += 1
        a2.n_novel_cross_local += 1
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


def _signature_similarity(sig1: str, sig2: str) -> int:
    """Count matching positions between two 4-letter TAPS signatures.

    Returns 0-4. Used for three-level tension classification:
      3-4 matches = low tension (absorptive)
      2 matches   = mid tension (L vs G tiebreak)
      0-1 matches = high tension (novel)

    Raises ValueError if either signature is not exactly 4 characters.
    """
    if len(sig1) != 4 or len(sig2) != 4:
        raise ValueError(
            f"TAPS signatures must be 4 characters; got {len(sig1)} and {len(sig2)}"
        )
    return sum(c1 == c2 for c1, c2 in zip(sig1, sig2))


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


def _agent_weight(agent: MetatheticAgent, all_types: dict[int, int], n_active: int) -> float:
    """Distinctiveness: fraction of agent's types not widely shared.

    Types held by ≤30% of active agents are considered distinctive.
    High distinctiveness = strong individual identity = harder to cross-metathesize.
    """
    if not agent.type_set:
        return 0.0
    n = max(1, n_active)
    distinctive = sum(1 for t in agent.type_set
                      if all_types.get(t, 0) <= max(1, int(n * 0.3)))
    return distinctive / len(agent.type_set)


def _temporal_threshold_multiplier(temporal_state: int, *, for_cross: bool = False,
                                    is_stagnating: bool = False) -> float:
    """Threshold multiplier based on temporal orientation.

    Inertial (1): 0.5x — easier to change
    Situated (2): 1.5x — resists change
    Desituated (3): depends on sub-type:
        - Novelty: inf (creative immunity, suppressed)
        - Stagnation + self-metathesis: inf (needs external stimulus, not self)
        - Stagnation + cross-metathesis: 0.5x (easier — needs external stimulus)
    Established (4): 2.0x — maximally stable
    Disintegrated (0): inf — impossible
    """
    if temporal_state == 3:
        if is_stagnating and for_cross:
            return 0.5  # Stagnating agents are receptive to cross-metathesis
        return float('inf')  # Novelty or self-metathesis: suppressed
    return {0: float('inf'), 1: 0.5, 2: 1.5, 4: 2.0}.get(
        temporal_state, 1.0
    )


def _compute_affordance_tick(
    agent: MetatheticAgent,
    other_active: list[MetatheticAgent],
    min_cluster: int = 2,
) -> int:
    """Binary affordance tick: 1 if conditions are ripe for construction, else 0.

    This is a compound readiness check, not a pure affordance measure.
    Building blocks (types) always exist somewhere, but whether the local
    arrangement permits construction depends on two conditions:

      1. Agent has positive recent dM (growth is actively occurring)
      2. At least min_cluster other active agents share >= 1 type with agent

    The combination ensures that growth is occurring in a connected context.
    An agent at equilibrium (dM=0) or in decline (dM<0) is not in a
    constructive state, even if building blocks are available nearby.
    """
    if not agent.dM_history or agent.dM_history[-1] <= 0:
        return 0
    n_connected = sum(
        1 for other in other_active
        if other.agent_id != agent.agent_id and (agent.type_set & other.type_set)
    )
    return 1 if n_connected >= min_cluster else 0


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
        affordance_min_cluster: int = 2,
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
        self.affordance_min_cluster = affordance_min_cluster

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
        self.n_disintegration_redistributions = 0
        self.n_deep_stasis = 0
        self.n_types_lost = 0
        self.k_lost = 0.0

        # Rolling history for regime classification (C1 fix: need ≥3 points
        # for classify_regime and adaptive_xi_plateau_threshold).
        self._m_history: list[float] = []
        self._k_history: list[float] = []

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

    def _convergence_measure(self, type_counts: dict[int, int] | None = None) -> float:
        """Fraction of agents sharing the most common type.

        High = convergent/localized (structure defines agents).
        Low = divergent/diffuse (agents define structure).
        """
        counts = type_counts if type_counts is not None else self._all_type_counts()
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
            agent.steps_since_metathesis += 1

    def _update_affordance_ticks(self) -> None:
        """Compute and record affordance tick for each active agent."""
        active = self._active_agents()
        for agent in active:
            tick = _compute_affordance_tick(
                agent, active, min_cluster=self.affordance_min_cluster
            )
            agent._affordance_ticks.append(tick)
            if len(agent._affordance_ticks) > agent._AFFORDANCE_WINDOW:
                agent._affordance_ticks = agent._affordance_ticks[-agent._AFFORDANCE_WINDOW:]

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
            # Temporal modulation
            threshold *= _temporal_threshold_multiplier(agent.temporal_state)
            # Affordance gate: agent must have recent environmental support
            if agent.affordance_score <= 0.0:
                continue
            if math.isfinite(threshold) and dM_recent > threshold:
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

        Mode selection via TAPS signature tension (three levels):
          3-4 letter match = low tension  → absorptive (η·H densification)
          2 letter match   = mid tension  → L vs G tiebreak
          0-1 letter match = high tension → novel (β·B exploration)

        Only one cross-metathesis event fires per step to avoid cascades.
        """
        active = self._active_agents()
        if len(active) < 2:
            return

        n_active = len(active)
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
                W1 = _agent_weight(a1, type_counts, n_active)
                W2 = _agent_weight(a2, type_counts, n_active)

                # Temporal modulation for cross-metathesis
                t1_stag = a1.steps_since_metathesis >= a1._STAGNATION_THRESHOLD
                t2_stag = a2.steps_since_metathesis >= a2._STAGNATION_THRESHOLD
                t1_mult = _temporal_threshold_multiplier(
                    a1.temporal_state, for_cross=True, is_stagnating=t1_stag)
                t2_mult = _temporal_threshold_multiplier(
                    a2.temporal_state, for_cross=True, is_stagnating=t2_stag)
                pair_mult = min(t1_mult, t2_mult) if math.isfinite(min(t1_mult, t2_mult)) else float('inf')
                if not math.isfinite(pair_mult):
                    continue
                effective_threshold = cross_threshold * pair_mult

                if L + G <= (W1 + W2) * effective_threshold:
                    continue

                # Eligible — determine mode via TAPS signature tension.
                sig_sim = _signature_similarity(
                    a1.taps_signature, a2.taps_signature
                )

                if sig_sim >= 3:
                    # Low tension: similar dispositional signatures → absorptive.
                    # η·H channel — densification within shared space.
                    MetatheticAgent.absorptive_cross(a1, a2)
                    self.n_absorptive_cross += 1
                elif sig_sim <= 1:
                    # High tension: different signatures → novel.
                    # β·B channel — exploration across dispositional boundaries.
                    self._next_agent_id += 1
                    child = MetatheticAgent.novel_cross(
                        a1, a2,
                        child_id=self._next_agent_id,
                        next_type_id=self._next_type_id,
                    )
                    self._next_type_id += 1
                    self.agents.append(child)
                    self.n_novel_cross += 1
                else:
                    # Mid tension (2 matches): L vs G tiebreak; ties (L == G) route to novel.
                    if L > G:
                        MetatheticAgent.absorptive_cross(a1, a2)
                        self.n_absorptive_cross += 1
                    else:
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

    def _check_disintegration_redistribution(self) -> None:
        """Check for disintegrated agents and redistribute their types/knowledge.

        Redistribution is weighted by Jaccard similarity (interaction proximity).
        Types go to the most-similar active agent. Knowledge is split proportionally.
        If no active agent has Jaccard > 0, the agent enters deep stasis: types
        are preserved and knowledge is truncated to a 5% residual.

        An agent is eligible for disintegration redistribution when it has been
        inactive and dormant past the relational decay window. This is
        consistent with temporal state 0 (disintegrated) but uses the dormancy
        condition directly so that Jaccard-based redistribution can still
        operate on remaining type overlaps before they are cleared.
        """
        active = self._active_agents()
        if not active:
            return

        for agent in list(self.agents):
            if agent._dissolved or agent._deep_stasis or agent.active:
                continue

            # Disintegration eligibility: dormant past relational decay window
            if agent._dormant_steps < agent._RELATIONAL_DECAY_WINDOW:
                continue

            # Compute Jaccard weights to each active agent
            weights = {}
            for other in active:
                w = _jaccard(agent.type_set, other.type_set)
                if w > 0:
                    weights[other.agent_id] = w

            total_w = sum(weights.values())

            if total_w == 0:
                # No interactive neighbor — enter deep stasis (rogue planet).
                # Types preserved (identity intact), knowledge truncated to
                # 5% residual. Agent can potentially reactivate if cross-
                # metathesis introduces shared types from a passing agent.
                residual_fraction = 0.05
                k_residual = agent.k * residual_fraction
                self.k_lost += agent.k - k_residual
                # n_types_lost stays 0: types are preserved
                agent.k = k_residual
                agent._deep_stasis = True
                self.n_deep_stasis += 1
                self.n_disintegration_redistributions += 1
                continue  # skip the dissolved cleanup below
            else:
                # Redistribute types to highest-weight agent
                agent_by_id = {a.agent_id: a for a in active}
                sorted_recipients = sorted(weights.keys(), key=lambda aid: (-weights[aid], aid))
                best_recipient = agent_by_id[sorted_recipients[0]]
                best_recipient.type_set = best_recipient.type_set | agent.type_set

                # Redistribute knowledge proportionally
                for aid, w in weights.items():
                    fraction = w / total_w
                    agent_by_id[aid].k += agent.k * fraction

            agent.type_set = set()
            agent.k = 0.0
            agent._dissolved = True
            self.n_disintegration_redistributions += 1

    def _record_history(self) -> None:
        """Append current aggregate M and k to rolling history.

        Called every step so that regime classification has sufficient
        trajectory data (classify_regime needs ≥3 points).
        """
        active = self._active_agents()
        total_M = sum(a.M_local for a in active)
        k_total = sum(a.k for a in active)
        self._m_history.append(total_M)
        self._k_history.append(k_total)
        # Keep last 20 points — enough for regime detection without
        # excessive memory use.
        if len(self._m_history) > 20:
            self._m_history = self._m_history[-20:]
            self._k_history = self._k_history[-20:]

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

        # Classify aggregate regime from rolling history.
        # classify_regime and adaptive_xi_plateau_threshold both require
        # ≥3 trajectory points for meaningful classification.
        if len(self._m_history) >= 3:
            thr = adaptive_xi_plateau_threshold(self._k_history)
            regime = classify_regime(self._k_history, self._m_history, thr)
        else:
            regime = "plateau"

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

            # 2. Record aggregate history for regime classification.
            self._record_history()

            # 2b. Update affordance ticks before metathesis checks.
            self._update_affordance_ticks()

            # 3. Metathetic transitions (Modes 1-3).
            self._check_self_metathesis()
            self._check_cross_metathesis()
            self._check_disintegration_redistribution()

            # 4. Environmental update (Mode 4, slow timescale).
            self._update_environment(step)

            # 5. Record ensemble snapshot.
            active = self._active_agents()
            dormant = [a for a in self.agents if not a.active and not a._dissolved and not a._deep_stasis]
            agent_k_list = [a.k for a in active]
            total_M = sum(a.M_local for a in active)
            type_counts = self._all_type_counts()

            snapshot = {
                "step": step,
                "D_total": self._total_diversity(),
                "k_total": sum(a.k for a in active),
                "total_M": total_M,
                "n_active": len(active),
                "n_dormant": len(dormant),
                "agent_k_list": agent_k_list,
                "convergence": self._convergence_measure(type_counts=type_counts),
                "texture_type": self.env.texture_type,
                "a_env": self.env.a_env,
                "K_env": self.env.K_env,
                "innovation_potential": self.env.innovation_potential(total_M),
                "n_self_metatheses": self.n_self_metatheses,
                "n_absorptive_cross": self.n_absorptive_cross,
                "n_novel_cross": self.n_novel_cross,
                "n_env_transitions": self.n_env_transitions,
                "n_disintegration_redistributions": self.n_disintegration_redistributions,
                "n_deep_stasis": self.n_deep_stasis,
                "n_types_lost": self.n_types_lost,
                "k_lost": round(self.k_lost, 4),
            }

            active_scores = [a.affordance_score for a in active]
            snapshot["affordance_mean"] = (
                sum(active_scores) / len(active_scores) if active_scores else 0.0
            )

            # TAPS signature distribution for active agents.
            sig_counts: dict[str, int] = {}
            for a in active:
                sig = a.taps_signature
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
            snapshot["signature_distribution"] = sig_counts
            snapshot["signature_diversity"] = len(sig_counts)

            # Track dormant steps.
            for a in dormant:
                a._dormant_steps += 1

            # Temporal state distribution.
            temporal_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for a in self.agents:
                ts = a.temporal_state_with_context(type_counts)
                temporal_counts[ts] += 1
            snapshot["temporal_state_counts"] = temporal_counts

            trajectory.append(snapshot)

        return trajectory
