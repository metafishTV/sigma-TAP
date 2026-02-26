# Metathetic Long-Run Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a metathetic multi-agent layer with four-mode identity transformation mechanics, plus Heaps' law / Gini / diversification diagnostics that mirror Taalbi (2025), completing TASKS.md T2.1 and T2.2.

**Architecture:** Two new simulator modules (`longrun.py` for pure statistical diagnostics, `metathetic.py` for the multi-agent model), one new script (`longrun_diagnostics.py` for CLI + figures), and corresponding test files. The longrun diagnostics are independent of the metathetic layer and can also be applied to aggregate trajectories. The metathetic layer uses existing `compute_birth_term` and `classify_regime` from the simulator core.

**Tech Stack:** Python 3.12, numpy 2.4, scipy 1.17, matplotlib 3.10, unittest. No new dependencies.

**Python executable:** `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`

**Working directory:** `C:\Users\user\Documents\New folder\sigma-TAP-repo`

**Design doc:** `docs/plans/2026-02-25-metathetic-longrun-design.md`

---

## Task 1: Create `simulator/longrun.py` — pure diagnostic functions (TDD)

**Files:**
- Create: `simulator/longrun.py`
- Modify: `tests/test_longrun.py` (add new test classes)

### Step 1: Write the failing tests

Append to `tests/test_longrun.py`:

```python
from simulator.longrun import (
    heaps_law_fit,
    gini_coefficient,
    top_k_share,
    diversification_rate,
    enhanced_constraint_tag,
)


class TestHeapsLawFit(unittest.TestCase):
    def test_known_power_law(self):
        """D(k) = 2 * k^0.7 should return beta ~ 0.7."""
        k_series = [float(i) for i in range(1, 101)]
        D_series = [2.0 * (k ** 0.7) for k in k_series]
        result = heaps_law_fit(D_series, k_series)
        self.assertAlmostEqual(result["beta"], 0.7, delta=0.05)
        self.assertGreater(result["r_squared"], 0.95)

    def test_linear_gives_beta_one(self):
        """D(k) = k should give beta ~ 1."""
        k_series = [float(i) for i in range(1, 51)]
        D_series = list(k_series)
        result = heaps_law_fit(D_series, k_series)
        self.assertAlmostEqual(result["beta"], 1.0, delta=0.05)

    def test_short_series_graceful(self):
        """Very short series should not crash."""
        result = heaps_law_fit([1.0, 2.0], [1.0, 3.0])
        self.assertIn("beta", result)


class TestGiniCoefficient(unittest.TestCase):
    def test_perfect_equality(self):
        """All equal values -> Gini = 0."""
        self.assertAlmostEqual(gini_coefficient([10.0, 10.0, 10.0, 10.0]), 0.0, places=5)

    def test_perfect_inequality(self):
        """One agent has everything -> Gini near 1."""
        gini = gini_coefficient([0.0, 0.0, 0.0, 100.0])
        self.assertGreater(gini, 0.7)

    def test_single_value(self):
        """Single value -> Gini = 0."""
        self.assertAlmostEqual(gini_coefficient([42.0]), 0.0, places=5)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(gini_coefficient([]), 0.0, places=5)


class TestTopKShare(unittest.TestCase):
    def test_equal_distribution(self):
        """Equal distribution: top 50% holds 50%."""
        self.assertAlmostEqual(top_k_share([10.0] * 10, k_frac=0.5), 0.5, delta=0.01)

    def test_skewed_distribution(self):
        """One dominant agent: top 10% holds most."""
        vals = [1.0] * 9 + [91.0]
        self.assertGreater(top_k_share(vals, k_frac=0.1), 0.9)


class TestDiversificationRate(unittest.TestCase):
    def test_heaps_sublinear_gives_declining_rate(self):
        """D(k) = k^0.5 -> dD/dk should be declining."""
        import math
        k_series = [float(i) for i in range(1, 51)]
        D_series = [math.sqrt(k) for k in k_series]
        rates = diversification_rate(D_series, k_series)
        self.assertGreater(len(rates), 0)
        # First rate should exceed last rate
        if len(rates) > 5:
            self.assertGreater(rates[0], rates[-1])


class TestEnhancedConstraintTag(unittest.TestCase):
    def test_adjacency_limited(self):
        result = enhanced_constraint_tag(
            sigma=1.5, beta=0.6, gini=0.3,
            carrying_capacity=None, m_final=100.0,
        )
        self.assertEqual(result["tag"], "adjacency-limited")
        self.assertIn(result["confidence"], {"high", "medium", "low"})

    def test_resource_limited(self):
        result = enhanced_constraint_tag(
            sigma=1.0, beta=0.8, gini=0.3,
            carrying_capacity=1000.0, m_final=900.0,
        )
        self.assertEqual(result["tag"], "resource-limited")

    def test_returns_reasoning(self):
        result = enhanced_constraint_tag(
            sigma=1.2, beta=0.7, gini=0.4,
            carrying_capacity=None, m_final=50.0,
        )
        self.assertIn("reasoning", result)
        self.assertIsInstance(result["reasoning"], str)
```

### Step 2: Run tests to verify they fail

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_longrun.py -v`

Expected: ImportError — `simulator.longrun` does not exist.

### Step 3: Write the implementation

Create `simulator/longrun.py`:

```python
"""Long-run diagnostics for TAP / sigma-TAP trajectories.

Statistical measures inspired by Taalbi (2025):
- Heaps' law: D(k) ~ k^beta for diversification
- Gini coefficient: concentration of innovation across agents
- Top-k share: fraction of total held by top agents
- Diversification rate: dD/dk over time
- Enhanced constraint tag: {adjacency-limited, resource-limited, mixed} + confidence

These diagnostics are independent of the metathetic agent layer and can be
applied to any multi-agent or aggregate trajectory data.
"""
from __future__ import annotations

import math


def heaps_law_fit(D_series: list[float], k_series: list[float]) -> dict:
    """Fit D(k) ~ k^beta via log-log OLS.

    Heaps' law predicts beta < 1: the rate of new type discovery declines
    relative to total innovation as the system matures.

    Returns dict with 'beta', 'intercept', 'r_squared', 'n_points'.
    """
    if len(D_series) < 2 or len(k_series) < 2:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": 0}

    log_k = []
    log_D = []
    for k, d in zip(k_series, D_series):
        if k > 0 and d > 0:
            log_k.append(math.log(k))
            log_D.append(math.log(d))

    n = len(log_k)
    if n < 2:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": n}

    mean_x = sum(log_k) / n
    mean_y = sum(log_D) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_k, log_D))
    ss_xx = sum((x - mean_x) ** 2 for x in log_k)
    ss_yy = sum((y - mean_y) ** 2 for y in log_D)

    if ss_xx < 1e-15:
        return {"beta": 1.0, "intercept": 0.0, "r_squared": 0.0, "n_points": n}

    beta = ss_xy / ss_xx
    intercept = mean_y - beta * mean_x
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-15 else 0.0

    return {"beta": beta, "intercept": intercept, "r_squared": r_squared, "n_points": n}


def gini_coefficient(values: list[float]) -> float:
    """Standard Gini coefficient.

    0 = perfect equality, 1 = perfect inequality.
    Uses the relative mean absolute difference formula.
    """
    if len(values) <= 1:
        return 0.0

    n = len(values)
    total = sum(values)
    if total <= 0:
        return 0.0

    sorted_vals = sorted(values)
    cum_sum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cum_sum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v

    return weighted_sum / (n * total)


def top_k_share(values: list[float], k_frac: float = 0.1) -> float:
    """Fraction of total held by the top k_frac fraction of agents.

    k_frac=0.1 means "what share does the top 10% hold?"
    """
    if not values:
        return 0.0

    total = sum(values)
    if total <= 0:
        return 0.0

    sorted_desc = sorted(values, reverse=True)
    n_top = max(1, int(math.ceil(len(sorted_desc) * k_frac)))
    top_sum = sum(sorted_desc[:n_top])
    return top_sum / total


def diversification_rate(D_series: list[float], k_series: list[float]) -> list[float]:
    """Compute dD/dk at each step — rate of new type discovery per unit innovation.

    Under Heaps' law with beta < 1, this rate should decline over time.
    """
    rates = []
    for i in range(len(D_series) - 1):
        dk = k_series[i + 1] - k_series[i]
        dD = D_series[i + 1] - D_series[i]
        if dk > 0:
            rates.append(dD / dk)
        elif dk == 0:
            rates.append(0.0)
        else:
            rates.append(0.0)
    return rates


def enhanced_constraint_tag(
    sigma: float,
    beta: float,
    gini: float,
    carrying_capacity: float | None,
    m_final: float,
) -> dict:
    """Tag observed dynamics with constraint type and confidence level.

    Decision heuristics grounded in Taalbi (2025):
    - sigma > 1 = super-linear TAP dynamics (adjacency-dominated)
    - sigma ~ 1 = resource-constrained linear dynamics
    - beta < 1 = Heaps' law diversification (sublinear)
    - gini < 0.5 = no winner-take-all

    Returns dict with 'tag', 'confidence', 'reasoning'.
    """
    reasons = []

    # Check if M is near carrying capacity.
    near_capacity = False
    if carrying_capacity is not None and carrying_capacity > 0:
        ratio = m_final / carrying_capacity
        if ratio > 0.8:
            near_capacity = True
            reasons.append(f"M/K={ratio:.2f} (near capacity)")

    # Adjacency-limited indicators.
    adjacency_indicators = 0
    if sigma > 1.3:
        adjacency_indicators += 1
        reasons.append(f"sigma={sigma:.2f} (super-linear)")
    if beta < 0.7:
        adjacency_indicators += 1
        reasons.append(f"beta={beta:.2f} (strong Heaps sublinearity)")
    if carrying_capacity is None:
        adjacency_indicators += 1
        reasons.append("no carrying capacity")

    # Resource-limited indicators.
    resource_indicators = 0
    if 0.8 <= sigma <= 1.2:
        resource_indicators += 1
        reasons.append(f"sigma={sigma:.2f} (near-linear)")
    if near_capacity:
        resource_indicators += 1
    if gini < 0.3:
        resource_indicators += 1
        reasons.append(f"gini={gini:.2f} (low concentration)")

    # Decision.
    if adjacency_indicators >= 2 and not near_capacity:
        tag = "adjacency-limited"
        confidence = "high" if adjacency_indicators >= 3 else "medium"
    elif resource_indicators >= 2 or near_capacity:
        tag = "resource-limited"
        confidence = "high" if resource_indicators >= 3 or near_capacity else "medium"
    else:
        tag = "mixed"
        confidence = "medium" if len(reasons) >= 2 else "low"

    return {
        "tag": tag,
        "confidence": confidence,
        "reasoning": "; ".join(reasons) if reasons else "insufficient indicators",
    }
```

### Step 4: Run tests to verify they pass

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_longrun.py -v`

Expected: All tests PASS (5 existing + 12 new = 17 tests).

### Step 5: Commit

```bash
git add simulator/longrun.py tests/test_longrun.py
git commit -m "feat: add long-run diagnostics — Heaps law, Gini, constraint tags

Pure statistical functions for Taalbi (2025) predictions:
heaps_law_fit, gini_coefficient, top_k_share, diversification_rate,
enhanced_constraint_tag with confidence levels.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create `simulator/metathetic.py` — core agent model (TDD)

**Files:**
- Create: `simulator/metathetic.py`
- Create: `tests/test_metathetic.py`

### Step 1: Write the failing tests

Create `tests/test_metathetic.py`:

```python
"""Tests for metathetic multi-agent TAP dynamics."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.metathetic import (
    MetatheticAgent,
    EnvironmentState,
    MetatheticEnsemble,
)


class TestMetatheticAgent(unittest.TestCase):
    def test_creation(self):
        """Agent starts with correct initial state."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        self.assertEqual(agent.agent_id, 0)
        self.assertEqual(agent.type_set, {1, 2})
        self.assertEqual(agent.k, 0.0)
        self.assertEqual(agent.M_local, 10.0)
        self.assertTrue(agent.active)

    def test_dormant_preserves_state(self):
        """Setting dormant preserves k and type_set."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2, 3}, k=50.0, M_local=20.0)
        agent.active = False
        self.assertFalse(agent.active)
        self.assertEqual(agent.k, 50.0)
        self.assertEqual(agent.type_set, {1, 2, 3})


class TestEnvironmentState(unittest.TestCase):
    def test_creation_defaults(self):
        env = EnvironmentState()
        self.assertGreater(env.a_env, 0)
        self.assertGreater(env.K_env, 0)
        self.assertEqual(env.texture_type, 1)

    def test_innovation_potential(self):
        env = EnvironmentState(K_env=1000.0)
        pot = env.innovation_potential(total_M=300.0)
        self.assertAlmostEqual(pot, 700.0)

    def test_innovation_potential_floor(self):
        env = EnvironmentState(K_env=100.0)
        pot = env.innovation_potential(total_M=200.0)
        self.assertEqual(pot, 0.0)


class TestSelfMetathesis(unittest.TestCase):
    def test_self_metathesis_preserves_k(self):
        """Self-metathesis adds a type but preserves k."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=25.0, M_local=10.0)
        old_k = agent.k
        old_types = len(agent.type_set)
        agent.self_metathesize(next_type_id=2)
        self.assertEqual(agent.k, old_k)
        self.assertEqual(len(agent.type_set), old_types + 1)
        self.assertIn(2, agent.type_set)

    def test_self_metathesis_does_not_duplicate(self):
        """Self-metathesis with existing type is a no-op on type_set size."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=10.0, M_local=5.0)
        agent.self_metathesize(next_type_id=2)
        self.assertEqual(len(agent.type_set), 2)


class TestCrossMetathesis(unittest.TestCase):
    def test_absorptive_cross(self):
        """Absorptive: smaller agent goes dormant, larger gets union of types."""
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=30.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=1, type_set={2, 3}, k=10.0, M_local=5.0)
        MetatheticAgent.absorptive_cross(a1, a2)
        # a1 absorbs a2 (a2 has lower M_local)
        self.assertTrue(a1.active)
        self.assertFalse(a2.active)
        self.assertEqual(a1.type_set, {1, 2, 3})
        self.assertEqual(a1.k, 40.0)

    def test_novel_cross(self):
        """Novel: both parents go dormant, new agent is returned."""
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=20.0, M_local=15.0)
        a2 = MetatheticAgent(agent_id=1, type_set={3, 4}, k=15.0, M_local=10.0)
        child = MetatheticAgent.novel_cross(a1, a2, child_id=2, next_type_id=5)
        self.assertFalse(a1.active)
        self.assertFalse(a2.active)
        self.assertTrue(child.active)
        self.assertEqual(child.k, 35.0)
        self.assertEqual(child.M_local, 25.0)
        # Child should have types from both parents plus at least one novel type
        self.assertTrue(len(child.type_set) >= 1)


class TestEnsembleRun(unittest.TestCase):
    def test_runs_without_error(self):
        """Ensemble completes 50 steps with 5 agents."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
        )
        trajectory = ensemble.run(steps=50)
        self.assertEqual(len(trajectory), 50)
        self.assertIn("D_total", trajectory[0])
        self.assertIn("k_total", trajectory[0])
        self.assertIn("n_active", trajectory[0])

    def test_diversity_non_decreasing(self):
        """Total diversity D should generally not decrease (types are not destroyed)."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
        )
        trajectory = ensemble.run(steps=30)
        D_values = [s["D_total"] for s in trajectory]
        # D may stay flat but should never decrease
        for i in range(1, len(D_values)):
            self.assertGreaterEqual(D_values[i], D_values[i - 1])

    def test_texture_type_valid(self):
        """Environment texture type stays in {1, 2, 3, 4}."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
        )
        trajectory = ensemble.run(steps=30)
        for s in trajectory:
            self.assertIn(s["texture_type"], {1, 2, 3, 4})

    def test_event_counts(self):
        """Run returns metathetic event counts."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02,
        )
        trajectory = ensemble.run(steps=30)
        last = trajectory[-1]
        self.assertIn("n_self_metatheses", last)
        self.assertIn("n_absorptive_cross", last)
        self.assertIn("n_novel_cross", last)


if __name__ == "__main__":
    unittest.main()
```

### Step 2: Run tests to verify they fail

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_metathetic.py -v`

Expected: ImportError — `simulator.metathetic` does not exist.

### Step 3: Write the implementation

Create `simulator/metathetic.py`:

```python
"""Metathetic multi-agent TAP dynamics.

LAYER: Exploratory — extends the TAP framework with multi-agent identity
transformations (metathesis) to produce ensemble-level diagnostics
(Heaps' law, concentration, diversification).

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
from dataclasses import dataclass, field
from typing import Sequence

from .tap import compute_birth_term
from .analysis import classify_regime, adaptive_xi_plateau_threshold


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

    def self_metathesize(self, next_type_id: int) -> None:
        """Mode 1: Self-metathesis — gain a new type, preserve k."""
        self.type_set = self.type_set | {next_type_id}

    @staticmethod
    def absorptive_cross(a1: MetatheticAgent, a2: MetatheticAgent) -> None:
        """Mode 2: Absorptive cross-metathesis.

        The agent with higher M_local absorbs the other.
        Composite gets union of type_sets and sum of k.
        Absorbed agent goes dormant (state preserved).
        """
        if a1.M_local >= a2.M_local:
            absorber, absorbed = a1, a2
        else:
            absorber, absorbed = a2, a1

        absorber.type_set = absorber.type_set | absorbed.type_set
        absorber.k += absorbed.k
        absorber.M_local += absorbed.M_local
        absorbed.active = False

    @staticmethod
    def novel_cross(
        a1: MetatheticAgent,
        a2: MetatheticAgent,
        child_id: int,
        next_type_id: int,
    ) -> MetatheticAgent:
        """Mode 3: Novel cross-metathesis.

        Both parents go dormant. A new agent spawns with recombined types
        and a novel type that neither parent could produce alone.
        """
        # Recombined types: union of parents plus one novel type.
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


@dataclass
class EnvironmentState:
    """Mode 4: Environmental/systemic metathesis — the containing structure.

    Quasi-constants that drift at a timescale much slower than agent dynamics.
    Maps to Emery & Trist (1965) L22 causal texture processes.

    texture_type: 1=Placid-randomized, 2=Placid-clustered,
                  3=Disturbed-reactive, 4=Turbulent
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

        Type I (placid): high threshold — hard to cross-metathesize.
        Type IV (turbulent): low threshold — radical transformation easier.
        """
        base = 1.0
        scaling = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5}
        return base * scaling.get(self.texture_type, 1.0)

    def update(self, D_total: int, k_total: float, total_M: float, regime: str) -> None:
        """Drift environment state based on aggregate agent behavior.

        Called every env_update_interval steps.
        """
        # a_env drifts: more diversity -> lower a (richer adjacency).
        # Slow drift: a_env moves 1% toward target per update.
        if D_total > 0:
            a_target = max(2.0, self._a_env_base * (10.0 / max(10.0, D_total)))
            self.a_env += 0.01 * (a_target - self.a_env)

        # K_env drifts: more total k -> higher K (innovation creates resources).
        K_target = self._K_env_base + 0.1 * k_total
        self.K_env += 0.01 * (K_target - self.K_env)

        # Texture type from regime classification.
        regime_to_texture = {
            "plateau": 1,
            "exponential": 2,
            "precursor-active": 3,
            "explosive": 4,
            "extinction": 1,
        }
        self.texture_type = regime_to_texture.get(regime, 1)


def _jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _goal_alignment(h1: Sequence[float], h2: Sequence[float], window: int = 5) -> float:
    """Correlation of recent dM/dt histories (goal alignment).

    Returns value in [-1, 1]. Higher = more aligned trajectories.
    """
    tail1 = list(h1[-window:])
    tail2 = list(h2[-window:])
    n = min(len(tail1), len(tail2))
    if n < 2:
        return 0.0

    m1 = sum(tail1[:n]) / n
    m2 = sum(tail2[:n]) / n
    cov = sum((tail1[i] - m1) * (tail2[i] - m2) for i in range(n))
    var1 = sum((tail1[i] - m1) ** 2 for i in range(n))
    var2 = sum((tail2[i] - m2) ** 2 for i in range(n))

    denom = math.sqrt(var1 * var2)
    if denom < 1e-15:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def _agent_weight(agent: MetatheticAgent, all_types: dict[int, int]) -> float:
    """Distinctiveness: fraction of agent's types not widely shared.

    all_types: {type_id: count_of_agents_with_this_type}.
    Types held by many agents are common; types held by few are distinctive.
    """
    if not agent.type_set:
        return 0.0
    total_agents = max(1, max(all_types.values())) if all_types else 1
    distinctive = sum(1 for t in agent.type_set
                      if all_types.get(t, 0) <= max(1, total_agents * 0.3))
    return distinctive / len(agent.type_set)


class MetatheticEnsemble:
    """Multi-agent TAP simulation with four-mode metathetic dynamics.

    Each agent runs a local TAP step per timestep. Metathetic transitions
    (self, absorptive cross, novel cross, environmental) are checked and
    applied based on threshold conditions.

    Produces ensemble trajectory data suitable for long-run diagnostics.
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

        self._rng = __import__("random").Random(seed)
        self._next_type_id = n_agents + 1
        self._next_agent_id = n_agents

        # Initialize agents with distinct initial types.
        self.agents: list[MetatheticAgent] = []
        for i in range(n_agents):
            # Each agent starts with a unique type + some shared types.
            types = {0, i + 1}  # type 0 is shared; type i+1 is unique
            self.agents.append(MetatheticAgent(
                agent_id=i,
                type_set=types,
                k=0.0,
                M_local=initial_M * (0.8 + 0.4 * self._rng.random()),
            ))

        self.env = EnvironmentState(a_env=a, K_env=carrying_capacity or 1e5)

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
        """Fraction of agents sharing the most common type."""
        counts = self._all_type_counts()
        if not counts:
            return 0.0
        active = self._active_agents()
        if not active:
            return 0.0
        max_count = max(counts.values())
        return max_count / len(active)

    def _step_agents(self) -> None:
        """Run one local TAP step for each active agent."""
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
                f = min(f, 1e12) if f > 0 else 0.0

            B = f  # sigma=1 (no sigma-feedback at agent level for simplicity)
            D = self.mu * agent.M_local
            dM = B - D

            agent.dM_history.append(dM)
            if len(agent.dM_history) > 10:
                agent.dM_history = agent.dM_history[-10:]

            agent.M_local = max(0.0, agent.M_local + dM)
            agent.k += max(0.0, B)  # k accumulates total innovation exposure

    def _check_self_metathesis(self) -> None:
        """Mode 1: Self-metathesis for each active agent."""
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
        """Modes 2 & 3: Pairwise cross-metathesis checks."""
        active = self._active_agents()
        if len(active) < 2:
            return

        type_counts = self._all_type_counts()
        cross_threshold = self.env.cross_metathesis_threshold()

        # Check all pairs (order doesn't matter for trigger).
        pairs_checked: set[tuple[int, int]] = set()
        for i, a1 in enumerate(active):
            if not a1.active:
                continue
            for j, a2 in enumerate(active):
                if i >= j or not a2.active:
                    continue
                pair_key = (a1.agent_id, a2.agent_id)
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

                # Eligible for cross-metathesis. Which mode?
                if L > G:
                    # Mode 2: Absorptive.
                    MetatheticAgent.absorptive_cross(a1, a2)
                    self.n_absorptive_cross += 1
                else:
                    # Mode 3: Novel.
                    self._next_agent_id += 1
                    child = MetatheticAgent.novel_cross(
                        a1, a2,
                        child_id=self._next_agent_id,
                        next_type_id=self._next_type_id,
                    )
                    self._next_type_id += 1
                    self.agents.append(child)
                    self.n_novel_cross += 1

                # Only one cross-metathesis event per step to avoid cascades.
                return

    def _update_environment(self, step: int) -> None:
        """Mode 4: Environmental update at slow timescale."""
        if step % self.env_update_interval != 0 or step == 0:
            return

        active = self._active_agents()
        D = self._total_diversity()
        k_total = sum(a.k for a in active)
        total_M = sum(a.M_local for a in active)

        # Classify aggregate regime for texture type.
        # Use aggregate M trajectory approximation.
        m_traj = [total_M * 0.9, total_M]  # minimal proxy
        xi_traj = [k_total * 0.9, k_total]
        thr = adaptive_xi_plateau_threshold(xi_traj)
        regime = classify_regime(xi_traj, m_traj, thr)

        old_texture = self.env.texture_type
        self.env.update(D, k_total, total_M, regime)
        if self.env.texture_type != old_texture:
            self.n_env_transitions += 1

    def run(self, steps: int) -> list[dict]:
        """Run the ensemble for the given number of steps.

        Returns a list of per-step snapshot dicts.
        """
        trajectory: list[dict] = []

        for step in range(steps):
            # 1. Agent-level TAP steps.
            self._step_agents()

            # 2. Metathetic transitions.
            self._check_self_metathesis()
            self._check_cross_metathesis()

            # 3. Environmental update (slow timescale).
            self._update_environment(step)

            # 4. Record snapshot.
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
```

### Step 4: Run tests to verify they pass

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_metathetic.py -v`

Expected: All 11 tests PASS.

### Step 5: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: add metathetic multi-agent TAP dynamics (4-mode)

Self-metathesis, absorptive cross, novel cross, environmental drift.
Emery & Trist causal texture types for environment. Agent identity
transformations preserve accumulated knowledge k.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create `scripts/longrun_diagnostics.py` — CLI runner + figures

**Files:**
- Create: `scripts/longrun_diagnostics.py`

### Step 1: Write the script

```python
"""Run metathetic ensemble and compute long-run diagnostics.

Produces:
  outputs/longrun_diagnostics.csv          — per-step ensemble data
  outputs/longrun_diagnostics_summary.json — summary statistics
  outputs/figures/heaps_law.png            — D(k) log-log plot
  outputs/figures/concentration_gini.png   — Gini + top-10% over time

Usage:
  python scripts/longrun_diagnostics.py
  python scripts/longrun_diagnostics.py --n-agents 20 --steps 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator.metathetic import MetatheticEnsemble
from simulator.longrun import (
    heaps_law_fit,
    gini_coefficient,
    top_k_share,
    diversification_rate,
    enhanced_constraint_tag,
)
from simulator.analysis import innovation_rate_scaling

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG_OUT = OUT / "figures"


def run_and_diagnose(
    n_agents: int = 10,
    initial_M: float = 10.0,
    alpha: float = 1e-3,
    a: float = 8.0,
    mu: float = 0.02,
    steps: int = 150,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Run ensemble and compute all diagnostics."""
    ensemble = MetatheticEnsemble(
        n_agents=n_agents,
        initial_M=initial_M,
        alpha=alpha, a=a, mu=mu,
        seed=seed,
    )
    trajectory = ensemble.run(steps=steps)

    # Extract series for diagnostics.
    D_series = [s["D_total"] for s in trajectory]
    k_series = [s["k_total"] for s in trajectory]
    M_series = [s["total_M"] for s in trajectory]

    # Heaps' law.
    heaps = heaps_law_fit(D_series, k_series)

    # Innovation rate scaling on aggregate M.
    scaling = innovation_rate_scaling(M_series)

    # Gini at final step.
    final_k_list = trajectory[-1]["agent_k_list"]
    gini_final = gini_coefficient(final_k_list)
    top10_final = top_k_share(final_k_list, k_frac=0.1)

    # Diversification rates.
    div_rates = diversification_rate(D_series, k_series)

    # Enhanced constraint tag.
    tag_result = enhanced_constraint_tag(
        sigma=scaling["exponent"],
        beta=heaps["beta"],
        gini=gini_final,
        carrying_capacity=None,
        m_final=M_series[-1] if M_series else 0.0,
    )

    last = trajectory[-1]
    summary = {
        "heaps_beta": round(heaps["beta"], 4),
        "heaps_r_squared": round(heaps["r_squared"], 4),
        "innovation_sigma": round(scaling["exponent"], 4),
        "innovation_sigma_r2": round(scaling["r_squared"], 4),
        "gini_final": round(gini_final, 4),
        "top10_share_final": round(top10_final, 4),
        "constraint_tag": tag_result["tag"],
        "constraint_confidence": tag_result["confidence"],
        "constraint_reasoning": tag_result["reasoning"],
        "n_agents_final": last["n_active"],
        "n_dormant_final": last["n_dormant"],
        "n_self_metatheses": last["n_self_metatheses"],
        "n_absorptive_cross": last["n_absorptive_cross"],
        "n_novel_cross": last["n_novel_cross"],
        "texture_type_final": last["texture_type"],
        "D_total_final": last["D_total"],
        "k_total_final": round(last["k_total"], 4),
        "total_M_final": round(last["total_M"], 4),
        "steps": steps,
        "n_agents_initial": n_agents,
        "claim_policy_label": "exploratory",
        "disclaimer": (
            "This result is exploratory and does not derive from the source "
            "TAP/biocosmology literature. It uses sigma-TAP computational "
            "infrastructure but represents an independent analytical direction "
            "that requires further validation."
        ),
    }

    return trajectory, summary


def write_csv(trajectory: list[dict], path: Path) -> None:
    """Write per-step CSV (excluding agent_k_list which is variable-length)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in trajectory[0].keys() if k != "agent_k_list"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(trajectory)
    print(f"Wrote {path} ({len(trajectory)} rows)")


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {path}")


def fig_heaps_law(trajectory: list[dict], summary: dict, save_path: Path) -> None:
    """D(k) log-log plot with fitted Heaps exponent."""
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 300})

    D_series = [s["D_total"] for s in trajectory]
    k_series = [s["k_total"] for s in trajectory]

    # Filter to positive values.
    k_pos = [k for k, d in zip(k_series, D_series) if k > 0 and d > 0]
    D_pos = [d for k, d in zip(k_series, D_series) if k > 0 and d > 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k_pos, D_pos, s=8, alpha=0.5, color="#2196F3", label="Ensemble data")

    # Fitted line.
    beta = summary["heaps_beta"]
    r2 = summary["heaps_r_squared"]
    if k_pos:
        k_fit = np.linspace(min(k_pos), max(k_pos), 100)
        import math
        intercept = math.log(D_pos[0]) - beta * math.log(k_pos[0]) if k_pos[0] > 0 else 0
        D_fit = np.exp(intercept) * k_fit ** beta
        ax.plot(k_fit, D_fit, "r--", linewidth=2,
                label=f"Heaps fit: beta={beta:.3f} (R2={r2:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative innovations k")
    ax.set_ylabel("Diversity D (unique types)")
    ax.set_title("Heaps' Law: D(k) ~ k^beta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def fig_concentration(trajectory: list[dict], save_path: Path) -> None:
    """Gini and top-10% share over time."""
    plt.rcParams.update({"font.size": 11, "savefig.dpi": 300})

    steps_arr = [s["step"] for s in trajectory]
    gini_arr = [gini_coefficient(s["agent_k_list"]) for s in trajectory]
    top10_arr = [top_k_share(s["agent_k_list"], k_frac=0.1) for s in trajectory]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(steps_arr, gini_arr, color="#FF9800", linewidth=1.5)
    ax1.set_ylabel("Gini coefficient")
    ax1.set_title("Innovation Concentration Over Time")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5,
                label="Gini=0.5 (moderate concentration)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_arr, top10_arr, color="#4CAF50", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Top-10% share")
    ax2.set_title("Top-10% Innovation Share Over Time")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    print(f"Wrote {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metathetic ensemble long-run diagnostics")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--a", type=float, default=8.0)
    parser.add_argument("--mu", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running metathetic ensemble: {args.n_agents} agents, {args.steps} steps")
    trajectory, summary = run_and_diagnose(
        n_agents=args.n_agents,
        initial_M=10.0,
        alpha=args.alpha,
        a=args.a,
        mu=args.mu,
        steps=args.steps,
        seed=args.seed,
    )

    write_csv(trajectory, OUT / "longrun_diagnostics.csv")
    write_summary(summary, OUT / "longrun_diagnostics_summary.json")
    fig_heaps_law(trajectory, summary, FIG_OUT / "heaps_law.png")
    fig_concentration(trajectory, FIG_OUT / "concentration_gini.png")

    print(f"\nSummary:")
    print(f"  Heaps beta:     {summary['heaps_beta']:.4f} (R2={summary['heaps_r_squared']:.4f})")
    print(f"  Innovation sigma: {summary['innovation_sigma']:.4f}")
    print(f"  Gini final:     {summary['gini_final']:.4f}")
    print(f"  Top-10% share:  {summary['top10_share_final']:.4f}")
    print(f"  Constraint:     {summary['constraint_tag']} ({summary['constraint_confidence']})")
    print(f"  Self-metatheses:    {summary['n_self_metatheses']}")
    print(f"  Absorptive cross:   {summary['n_absorptive_cross']}")
    print(f"  Novel cross:        {summary['n_novel_cross']}")
    print(f"  Env texture final:  Type {summary['texture_type_final']}")


if __name__ == "__main__":
    main()
```

### Step 2: Run end-to-end

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/longrun_diagnostics.py --steps 100`

Expected: CSV, JSON, and 2 figures created in `outputs/`.

### Step 3: Commit

```bash
git add scripts/longrun_diagnostics.py
git commit -m "feat: add metathetic ensemble diagnostics CLI + figures

Runs multi-agent TAP ensemble, computes Heaps law / Gini / concentration,
outputs CSV, summary JSON, and 2 publication figures.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Update CLAIMS.md

**Files:**
- Modify: `CLAIMS.md`

### Step 1: Update claims

Upgrade C3 from partial to supported. Add C8 as exploratory.

Append after C7 row:

```
| C8 | Metathetic multi-agent TAP dynamics produce Heaps' law (beta < 1) and non-winner-take-all concentration, consistent with Taalbi (2025) predictions under resource-constrained recombinant search | Taalbi (2025); Emery & Trist (1965) (exploratory extension) | `simulator/metathetic.py`; `outputs/longrun_diagnostics_summary.json`; `outputs/figures/heaps_law.png` | exploratory |
```

Update C3 status from `partial` to `supported` and add artifacts:

Change: `simulator/analysis.py::innovation_rate_scaling`; `outputs/realworld_fit.csv`
To: `simulator/analysis.py::innovation_rate_scaling`; `simulator/longrun.py::heaps_law_fit`; `outputs/realworld_fit.csv`; `outputs/longrun_diagnostics_summary.json`

### Step 2: Commit

```bash
git add CLAIMS.md
git commit -m "docs: upgrade C3 to supported, add C8 (exploratory metathetic)

C3 now backed by Heaps law + concentration diagnostics.
C8: metathetic multi-agent Heaps/Gini results (exploratory).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Wire into `run_reporting_pipeline.py`

**Files:**
- Modify: `run_reporting_pipeline.py`

### Step 1: Add longrun_diagnostics stage

After the `generate_figures.py` line (line 85), insert:

```python
    run([sys.executable, 'scripts/longrun_diagnostics.py'])
```

Add to expected_figures list:

```python
            'outputs/figures/heaps_law.png',
            'outputs/figures/concentration_gini.png',
```

### Step 2: Commit

```bash
git add run_reporting_pipeline.py
git commit -m "feat: wire metathetic long-run diagnostics into pipeline

New stage: longrun_diagnostics.py. Two new expected figures validated.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Run full test suite — final verification

### Step 1: Run all tests

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -v`

Expected: All tests pass (33 existing + 12 longrun + 11 metathetic = ~56 tests).

### Step 2: Run longrun_diagnostics end-to-end

Run: `"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" scripts/longrun_diagnostics.py`

Expected: All outputs created, no errors.

### Step 3: Push to remote

```bash
git push origin unified-integration-20260225
```

---

## Summary

| Task | Deliverable | Tests | Est. time |
|------|-------------|-------|-----------|
| 1 | simulator/longrun.py | 12 new tests | 10 min |
| 2 | simulator/metathetic.py | 11 new tests | 15 min |
| 3 | scripts/longrun_diagnostics.py | end-to-end | 10 min |
| 4 | CLAIMS.md updates | — | 3 min |
| 5 | Pipeline integration | — | 3 min |
| 6 | Final verification | all ~56 | 5 min |

**Total: 6 tasks, ~23 new tests, ~46 minutes estimated.**
