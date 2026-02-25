# P3: Temporal Orientation Gate + Evidence Ladder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a five-state temporal orientation gate to metathetic agents AND build a config-driven evidence report with mechanistic/functional interpretation blocks and A/B/C confidence tiers.

**Architecture:** Thread A adds `temporal_state` (0-4) as a computed property on MetatheticAgent, tracked via `steps_since_metathesis` and `_trajectory_alignment()`, with annihilation distinct from dormancy. Thread B creates `config/claim_annotations.json` (authored content) + `scripts/build_evidence_report.py` (validation) producing `outputs/evidence_report.json`.

**Tech Stack:** Python 3.12, unittest, json, existing simulator/metathetic.py

**Python:** `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe`
**Repo root:** `C:\Users\user\Documents\New folder\sigma-TAP-repo`

---

## Task 1: Temporal state tracking on MetatheticAgent

**Files:**
- Modify: `simulator/metathetic.py` (MetatheticAgent dataclass, lines 36-52)
- Test: `tests/test_metathetic.py`

### Step 1: Write the failing tests

Add to `tests/test_metathetic.py` — new class after existing `TestEnsembleRegimeClassification`:

```python
class TestTemporalState(unittest.TestCase):
    """Tests for five-state temporal orientation gate."""

    def test_new_agent_is_desituated_novelty(self):
        """Freshly created agent (steps_since_metathesis=0) is desituated."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        self.assertEqual(agent.temporal_state, 3)  # desituated (novelty)

    def test_agent_becomes_situated_after_novelty_window(self):
        """After novelty window passes with stable trajectory, agent is situated."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        # Simulate enough steps past novelty window with positive dM
        for _ in range(10):
            agent.steps_since_metathesis += 1
            agent.dM_history.append(5.0)
        # Not stagnating, positive trajectory, past novelty window
        self.assertEqual(agent.temporal_state, 2)  # situated

    def test_inertial_on_diverging_trajectory(self):
        """Declining trajectory below threshold -> inertial."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        agent.steps_since_metathesis = 20
        # Strongly negative recent trajectory
        agent.dM_history = [-5.0, -4.0, -6.0, -3.0, -7.0]
        self.assertEqual(agent.temporal_state, 1)  # inertial

    def test_desituated_stagnation(self):
        """Many steps without metathesis -> desituated (stagnation)."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        agent.steps_since_metathesis = 60  # well past stagnation_threshold=50
        agent.dM_history = [1.0, 1.0, 1.0]  # mild positive (not inertial)
        self.assertEqual(agent.temporal_state, 3)  # desituated (stagnation)

    def test_established_after_alignment(self):
        """Strong sustained alignment post-novelty -> established."""
        agent = MetatheticAgent(agent_id=0, type_set={1, 2}, k=0.0, M_local=10.0)
        agent.steps_since_metathesis = 25  # past novelty, not stagnating
        # Strong, consistent, growing trajectory
        agent.dM_history = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.assertEqual(agent.temporal_state, 4)  # established

    def test_annihilated_state(self):
        """Inactive agent past relational decay with no living type connections."""
        agent = MetatheticAgent(agent_id=0, type_set={99, 100}, k=50.0, M_local=0.0)
        agent.active = False
        agent.steps_since_metathesis = 100
        agent._dormant_steps = 40  # past relational_decay_window=30
        # No active agents hold types 99 or 100 -> annihilated
        self.assertEqual(agent.temporal_state_with_context(active_type_counts={}), 0)
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestTemporalState -v`
Expected: FAIL — `MetatheticAgent` has no `temporal_state` property

### Step 3: Implement temporal state on MetatheticAgent

In `simulator/metathetic.py`, update MetatheticAgent dataclass (add after line 51):

```python
    steps_since_metathesis: int = 0
    _dormant_steps: int = 0

    # Temporal gate parameters (class-level defaults).
    _NOVELTY_WINDOW: int = 5
    _STAGNATION_THRESHOLD: int = 50
    _RELATIONAL_DECAY_WINDOW: int = 30
    _TRAJECTORY_DIVERGENCE_THR: float = -0.3
    _ESTABLISHED_ALIGNMENT_THR: float = 0.5
    _ESTABLISHED_MIN_HISTORY: int = 6

    @property
    def temporal_state(self) -> int:
        """Five-state temporal orientation gate.

        0 = annihilated  (no relational capacity; requires context)
        1 = inertial      (grown away from identity)
        2 = situated      (in-flow, productive)
        3 = desituated    (novelty-shock or stagnation)
        4 = established   (consummated; static tension)
        """
        if not self.active:
            return 3  # dormant without context defaults to desituated
        return self._compute_temporal_state()

    def temporal_state_with_context(self, active_type_counts: dict[int, int]) -> int:
        """Temporal state with population context (needed for annihilation check).

        active_type_counts: {type_id: count} from ensemble._all_type_counts()
        """
        if not self.active:
            if self._dormant_steps >= self._RELATIONAL_DECAY_WINDOW:
                # Check if any active agent still holds our types
                has_living_connection = any(
                    active_type_counts.get(t, 0) > 0 for t in self.type_set
                )
                if not has_living_connection:
                    return 0  # annihilated
            return 3  # dormant but not annihilated = desituated
        return self._compute_temporal_state()

    def _compute_temporal_state(self) -> int:
        """Core gate logic for active agents."""
        # Desituated: novelty-shock (just after metathesis)
        if self.steps_since_metathesis <= self._NOVELTY_WINDOW:
            return 3

        # Desituated: stagnation (too long without metathesis)
        if self.steps_since_metathesis >= self._STAGNATION_THRESHOLD:
            return 3

        # Check trajectory alignment from recent dM history
        alignment = self._trajectory_alignment()

        # Inertial: trajectory diverging (strongly negative dM trend)
        if alignment < self._TRAJECTORY_DIVERGENCE_THR:
            return 1

        # Established: strong sustained positive alignment with enough history
        if (alignment > self._ESTABLISHED_ALIGNMENT_THR
                and len(self.dM_history) >= self._ESTABLISHED_MIN_HISTORY):
            return 4

        # Default: situated (productive, in-flow)
        return 2

    def _trajectory_alignment(self) -> float:
        """Measure how well recent trajectory aligns with productive identity.

        Returns value in roughly [-1, 1].
        Positive = growth aligned with identity (dM positive and increasing).
        Negative = diverging (dM negative or declining).
        """
        if len(self.dM_history) < 3:
            return 0.0
        recent = self.dM_history[-5:]
        mean_dM = sum(recent) / len(recent)
        # Normalize by scale: positive mean = aligned, negative = diverging
        # Also check trend: is dM increasing or decreasing?
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / max(1.0, abs(recent[0]) + 1e-10)
        else:
            trend = 0.0
        # Combine: mean sign + trend direction
        if mean_dM > 0:
            return min(1.0, 0.3 + 0.7 * min(1.0, trend))
        else:
            return max(-1.0, -0.3 + 0.7 * max(-1.0, trend))
```

Also update `self_metathesize` (line 55-62) to reset temporal tracking:

```python
    def self_metathesize(self, next_type_id: int) -> None:
        """Mode 1: Self-metathesis — gain a new type, preserve k."""
        self.type_set = self.type_set | {next_type_id}
        self.steps_since_metathesis = 0
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestTemporalState -v`
Expected: PASS (6 tests)

### Step 5: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: add five-state temporal orientation gate to MetatheticAgent"
```

---

## Task 2: Wire temporal state into ensemble dynamics

**Files:**
- Modify: `simulator/metathetic.py` (MetatheticEnsemble methods)
- Test: `tests/test_metathetic.py`

### Step 1: Write the failing tests

```python
class TestTemporalModulation(unittest.TestCase):
    """Temporal state modulates metathetic trigger thresholds."""

    def test_inertial_lowers_self_meta_threshold(self):
        """Inertial agent should have easier self-metathesis (0.5x threshold)."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(1), 0.5)

    def test_situated_raises_self_meta_threshold(self):
        """Situated agent resists change (1.5x threshold)."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(2), 1.5)

    def test_established_hardest_to_change(self):
        """Established = maximally stable (2.0x threshold)."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertAlmostEqual(_temporal_threshold_multiplier(4), 2.0)

    def test_desituated_novelty_suppresses(self):
        """Desituated = threshold of infinity (suppressed)."""
        from simulator.metathetic import _temporal_threshold_multiplier
        self.assertEqual(_temporal_threshold_multiplier(3), float('inf'))

    def test_temporal_state_in_snapshot(self):
        """Snapshot includes temporal_state_counts."""
        ensemble = MetatheticEnsemble(
            n_agents=5, initial_M=10.0,
            alpha=1e-3, a=8.0, mu=0.02, seed=42,
        )
        trajectory = ensemble.run(steps=20)
        for s in trajectory:
            self.assertIn("temporal_state_counts", s)
            # Should be a dict with int keys 0-4
            self.assertIsInstance(s["temporal_state_counts"], dict)

    def test_steps_since_metathesis_increments(self):
        """steps_since_metathesis grows each step for agents that don't metathesize."""
        ensemble = MetatheticEnsemble(
            n_agents=3, initial_M=10.0,
            alpha=1e-4, a=8.0, mu=0.02, seed=42,  # very low alpha = no metathesis
        )
        ensemble.run(steps=10)
        for agent in ensemble.agents:
            if agent.active:
                self.assertGreaterEqual(agent.steps_since_metathesis, 10)

    def test_dormant_steps_tracked(self):
        """Dormant agents accumulate _dormant_steps."""
        a1 = MetatheticAgent(agent_id=0, type_set={1, 2}, k=20.0, M_local=15.0)
        a2 = MetatheticAgent(agent_id=1, type_set={3, 4}, k=15.0, M_local=10.0)
        MetatheticAgent.novel_cross(a1, a2, child_id=2, next_type_id=5)
        self.assertFalse(a1.active)
        self.assertEqual(a1._dormant_steps, 0)
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestTemporalModulation -v`
Expected: FAIL — `_temporal_threshold_multiplier` not defined

### Step 3: Implement ensemble wiring

In `simulator/metathetic.py`, add the threshold multiplier function after `_agent_weight`:

```python
def _temporal_threshold_multiplier(temporal_state: int) -> float:
    """Threshold multiplier based on temporal orientation.

    Inertial (1): 0.5x — easier to change (grown away from identity)
    Situated (2): 1.5x — productive, resists change
    Desituated (3): inf — suppressed (novelty immunity or needs external stimulus)
    Established (4): 2.0x — maximally stable, hardest to dislodge
    Annihilated (0): inf — impossible
    """
    return {0: float('inf'), 1: 0.5, 2: 1.5, 3: float('inf'), 4: 2.0}.get(
        temporal_state, 1.0
    )
```

In `_step_agents` method, add after `agent.k += max(0.0, B)` (line 378):

```python
            agent.steps_since_metathesis += 1
```

In `_check_self_metathesis`, modify the threshold check (line 397-401):

```python
            threshold = self.self_meta_threshold * len(agent.type_set)
            # Temporal modulation: inertial agents change easier,
            # situated/established agents resist, desituated are suppressed.
            threshold *= _temporal_threshold_multiplier(agent.temporal_state)
            if math.isfinite(threshold) and dM_recent > threshold:
                agent.self_metathesize(self._next_type_id)
                self._next_type_id += 1
                self.n_self_metatheses += 1
```

In `_check_cross_metathesis`, add temporal modulation after W1/W2 computation:

```python
                # Temporal modulation for cross-metathesis.
                # Desituated-stagnation agents get easier cross threshold.
                t1_mult = _temporal_threshold_multiplier(a1.temporal_state)
                t2_mult = _temporal_threshold_multiplier(a2.temporal_state)
                # For cross-metathesis, use the LOWER multiplier of the pair
                # (the more change-ready agent drives eligibility).
                pair_mult = min(t1_mult, t2_mult) if math.isfinite(min(t1_mult, t2_mult)) else float('inf')
                if not math.isfinite(pair_mult):
                    continue
                effective_threshold = cross_threshold * pair_mult
                if L + G <= (W1 + W2) * effective_threshold:
                    continue
```

In `run()`, add dormant step tracking and temporal counts to snapshot. After `snapshot = {` block, before `trajectory.append(snapshot)`:

```python
            # Track dormant steps.
            for a in dormant:
                a._dormant_steps += 1

            # Temporal state distribution.
            type_counts_for_context = self._all_type_counts()
            temporal_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for a in self.agents:
                ts = a.temporal_state_with_context(type_counts_for_context)
                temporal_counts[ts] += 1
            snapshot["temporal_state_counts"] = temporal_counts
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat: wire temporal gate into ensemble dynamics + threshold modulation"
```

---

## Task 3: Add temporal state to longrun diagnostics output

**Files:**
- Modify: `scripts/longrun_diagnostics.py` (summary dict + print output)

### Step 1: No new test needed — existing pipeline test covers this

### Step 2: Update summary dict in `run_and_diagnose`

After `"n_env_transitions"` line in summary dict, add:

```python
        "temporal_state_counts_final": last["temporal_state_counts"],
```

Update the print output in `main()` to include temporal distribution:

```python
    tc = summary.get("temporal_state_counts_final", {})
    state_names = {0: "annihilated", 1: "inertial", 2: "situated", 3: "desituated", 4: "established"}
    tc_str = ", ".join(f"{state_names.get(k, '?')}={v}" for k, v in sorted(tc.items()) if v > 0)
    print(f"  Temporal:         {tc_str}")
```

### Step 3: Run pipeline to verify

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe scripts/longrun_diagnostics.py --seed 42`
Expected: Output includes temporal distribution line

### Step 4: Commit

```bash
git add scripts/longrun_diagnostics.py
git commit -m "feat: add temporal state distribution to longrun diagnostics output"
```

---

## Task 4: Create claim annotations config (Thread B)

**Files:**
- Create: `config/claim_annotations.json`
- Test: `tests/test_evidence_report.py`

### Step 1: Write the failing tests

Create `tests/test_evidence_report.py`:

```python
"""Tests for evidence report builder (T3.1 + T3.2)."""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO / "config" / "claim_annotations.json"


class TestAnnotationsFileExists(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(ANNOTATIONS_PATH.exists(), "config/claim_annotations.json missing")

    def test_valid_json(self):
        data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)


class TestAnnotationsCompleteness(unittest.TestCase):
    """Every claim C1-C8 has required fields."""

    def setUp(self):
        self.data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))

    def test_all_claims_present(self):
        for cid in [f"C{i}" for i in range(1, 9)]:
            self.assertIn(cid, self.data, f"Missing annotation for {cid}")

    def test_mechanistic_non_empty(self):
        for cid, entry in self.data.items():
            self.assertIn("mechanistic", entry, f"{cid} missing mechanistic block")
            self.assertGreater(len(entry["mechanistic"]), 20,
                               f"{cid} mechanistic block too short")

    def test_functional_non_empty(self):
        for cid, entry in self.data.items():
            self.assertIn("functional", entry, f"{cid} missing functional block")
            self.assertGreater(len(entry["functional"]), 20,
                               f"{cid} functional block too short")

    def test_evidence_tier_valid(self):
        for cid, entry in self.data.items():
            self.assertIn("evidence_tier", entry, f"{cid} missing evidence_tier")
            self.assertIn(entry["evidence_tier"], {"A", "B", "C"},
                          f"{cid} invalid tier: {entry['evidence_tier']}")

    def test_tier_justification_present(self):
        for cid, entry in self.data.items():
            self.assertIn("tier_justification", entry)
            self.assertGreater(len(entry["tier_justification"]), 10)

    def test_exploratory_claims_are_tier_c(self):
        for cid, entry in self.data.items():
            if entry.get("claim_policy_label") == "exploratory":
                self.assertEqual(entry["evidence_tier"], "C",
                                 f"{cid} is exploratory but not tier C")

    def test_artifacts_listed(self):
        for cid, entry in self.data.items():
            self.assertIn("artifacts", entry, f"{cid} missing artifacts list")
            self.assertIsInstance(entry["artifacts"], list)
            self.assertGreater(len(entry["artifacts"]), 0,
                               f"{cid} has empty artifacts list")
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_evidence_report.py -v`
Expected: FAIL — file does not exist

### Step 3: Create claim_annotations.json

Create `config/claim_annotations.json`:

```json
{
  "C1": {
    "mechanistic": "The TAP kernel Sigma C(M,i)/a^i generates qualitatively distinct regime behavior depending on variant structure. Baseline admits unbounded combinatorial explosion controlled only by extinction rate mu. Two-scale introduces delayed acceleration via dual alpha parameters (alpha0, alpha1) operating at different combinatorial orders. Logistic caps M at carrying capacity K, producing bounded resource-constrained growth. The adjacency parameter a controls the combinatorial reach of each existing object, while mu governs the rate of knowledge obsolescence.",
    "functional": "Variant families serve as competing hypotheses for the growth-limiting mechanism in real combinatorial innovation systems. Selecting among variants is equivalent to asking whether observed constraints are purely combinatorial (baseline), partially structural with delayed onset (two-scale), or fundamentally resource-bounded (logistic). This distinction has direct implications for forecasting: baseline predicts eventual explosion, two-scale predicts delayed but inevitable acceleration, logistic predicts saturation.",
    "evidence_tier": "A",
    "tier_justification": "Replicated across all three variants in sweep_variants.py; regime transitions quantified per variant in variant_comparison.csv; bootstrap CIs available in inferential_stats.json for key parameters.",
    "artifacts": ["outputs/variant_comparison.csv", "outputs/figures/trajectory_variants.png"],
    "claim_policy_label": "paper-aligned"
  },
  "C2": {
    "mechanistic": "Regime detection operates on M(t) and Xi(t) trajectories using adaptive thresholds. Plateau is identified when Xi growth rate falls below a noise-adjusted threshold. Exponential onset is detected via sustained super-threshold growth. Explosive regime triggers when M exceeds a blow-up proxy threshold or becomes non-finite. Extinction occurs when M decays to zero. The classify_regime function requires minimum 3 trajectory points and uses the adaptive_xi_plateau_threshold for noise-robust detection.",
    "functional": "Regime transitions mark qualitative shifts in the innovation system's character. Plateau-to-exponential transition indicates the adjacent possible has begun to self-amplify. Exponential-to-explosive marks loss of natural constraint, suggesting the system has entered a phase where combinatorial possibilities outpace any realistic exploration capacity. These transitions correspond to observable phenomena in technology adoption curves and scientific paradigm shifts.",
    "evidence_tier": "A",
    "tier_justification": "Regime detection tested across all three TAP variants; transitions consistently detected in variant_comparison.csv; classify_regime validated by unit tests across edge cases.",
    "artifacts": ["outputs/variant_comparison.csv", "simulator/analysis.py"],
    "claim_policy_label": "paper-aligned"
  },
  "C3": {
    "mechanistic": "Innovation-rate scaling sigma is estimated via log-log OLS regression of dM/dt against M, following Taalbi (2025) formulation dk/dt ~ k^sigma. Heaps law beta is estimated from D(k) ~ k^beta via log-log OLS on diversity-vs-cumulative-innovation series. Under the logistic variant with carrying capacity, sigma converges near 1 (linear dynamics) and beta < 1 (sublinear diversification), consistent with resource-constrained recombinant search.",
    "functional": "Sigma near 1 under resource constraints indicates that innovation rate scales proportionally with existing knowledge rather than super-linearly, which would imply runaway dynamics. Beta < 1 (Heaps law) indicates that type diversity grows slower than cumulative innovations, meaning later innovations increasingly reuse existing types rather than creating entirely new ones. This is the signature of a maturing innovation system where the adjacent possible expands but becomes increasingly structured.",
    "evidence_tier": "B",
    "tier_justification": "Heaps law supported with R-squared > 0.95 in logistic configuration; single-variant result (logistic only produces the resource-constrained dynamics needed for Taalbi alignment).",
    "artifacts": ["outputs/longrun_diagnostics_summary.json", "outputs/realworld_fit.csv", "outputs/figures/heaps_law.png"],
    "claim_policy_label": "paper-aligned"
  },
  "C4": {
    "mechanistic": "Three real-world datasets (Wikipedia articles, npm packages, species counts) are fitted against null models (exponential, logistic, power-law) and TAP variants (baseline, logistic, learning). TAP-baseline achieves lowest RMSE-log for Wikipedia (0.036 vs 0.112 logistic, 0.175 power-law, 0.705 exponential). The tamed TAP kernel's sub-exponential but super-polynomial growth matches the characteristic S-curve-with-acceleration shape of real combinatorial accumulation.",
    "functional": "The superior fit of TAP over standard growth models for real combinatorial datasets suggests that the adjacent-possible mechanism provides a more accurate generative model for innovation dynamics than phenomenological curve-fitting. This validates Kauffman's hypothesis that combinatorial recombination, not simple resource competition or random exploration, is the primary driver of long-run innovation trajectories.",
    "evidence_tier": "A",
    "tier_justification": "Tested across three independent datasets and multiple competing models; consistent TAP advantage in RMSE-log across all datasets.",
    "artifacts": ["outputs/realworld_fit.csv", "scripts/fit_realworld.py"],
    "claim_policy_label": "paper-aligned"
  },
  "C5": {
    "mechanistic": "Decision bandwidth B(t) is computed as the ratio of TAP birth term to extinction loss, providing a dimensionless measure of effective exploratory capacity at each timestep. Praxiological Reynolds number Re_prax is analogized from fluid dynamics: high Re_prax indicates turbulent (chaotic, exploratory) dynamics, low Re_prax indicates laminar (structured, exploitative) dynamics. Both are derived from the standard TAP quantities without additional free parameters.",
    "functional": "Turbulence diagnostics provide an interpretive vocabulary for TAP regime transitions. The laminar-to-turbulent transition in Re_prax corresponds to the moment when combinatorial possibilities overwhelm structured search, analogous to how fluid turbulence emerges when inertial forces dominate viscous ones. This framing connects TAP dynamics to established intuitions about explore-exploit tradeoffs in organizational theory.",
    "evidence_tier": "C",
    "tier_justification": "Novel diagnostic framework not present in source TAP literature; single-variant implementation; requires independent validation and theoretical scrutiny.",
    "artifacts": ["simulator/turbulence.py", "outputs/figures/turbulence_bandwidth.png"],
    "claim_policy_label": "exploratory"
  },
  "C6": {
    "mechanistic": "Extinction rate mu controls the rate of knowledge obsolescence: M(t+1) = M(t)(1-mu) + birth_term. Higher mu accelerates decay of realized objects, competing against the TAP birth term. The sensitivity sweep over mu in [0.001, 0.1] shows that increasing mu from 0.001 to 0.05 shifts the regime from explosive to plateau across all variants tested, with transition timing (step at which regime changes) monotonically increasing with mu.",
    "functional": "Extinction rate represents the real-world phenomenon of knowledge becoming obsolete or inaccessible. High mu corresponds to rapidly evolving fields where past innovations quickly lose relevance (e.g., software frameworks). Low mu corresponds to cumulative knowledge domains (e.g., mathematics). The finding that mu controls transition timing implies that obsolescence rate is a primary determinant of whether an innovation system ever reaches its explosive potential.",
    "evidence_tier": "B",
    "tier_justification": "Sensitivity sweep across mu parameter space; tested in baseline variant; lacks cross-variant replication and bootstrap CIs.",
    "artifacts": ["outputs/extinction_sensitivity.csv", "outputs/figures/extinction_sensitivity.png"],
    "claim_policy_label": "paper-aligned"
  },
  "C7": {
    "mechanistic": "Adjacency parameter a controls the combinatorial reach in the TAP kernel: larger a means higher-order combinations are more strongly suppressed (each additional combination level is divided by a^i). The sensitivity sweep over a in [2, 32] shows that smaller a produces faster growth and earlier explosive onset because more combinatorial pathways are accessible at each step. Currently fixed at a=8 across all primary analyses.",
    "functional": "The adjacency parameter encodes the structural accessibility of the combinatorial space. Low a corresponds to systems where complex multi-component combinations are easily achieved (e.g., digital recombination where components are modular). High a corresponds to systems with strong physical or conceptual barriers to higher-order combination (e.g., chemical synthesis). The finding that a controls explosion rate suggests that structural accessibility, not just the size of the adjacent possible, determines growth character.",
    "evidence_tier": "B",
    "tier_justification": "Sensitivity sweep across a parameter space; tested in baseline variant; lacks cross-variant replication.",
    "artifacts": ["outputs/adjacency_sensitivity.csv", "outputs/figures/adjacency_sensitivity.png"],
    "claim_policy_label": "paper-aligned"
  },
  "C8": {
    "mechanistic": "The metathetic ensemble runs N agents each with local TAP dynamics and type-identity portfolios. Self-metathesis (Mode 1) adds types when innovation rate exceeds a threshold. Cross-metathesis (Modes 2-3) triggers when Jaccard likeness + goal alignment exceeds weighted distinctiveness threshold, scaled by Emery-Trist texture type. Environmental drift (Mode 4) adjusts a_env and K_env based on aggregate regime classification. Under logistic variant with growth parameters (alpha=5e-3, a=3.0, mu=0.005), the ensemble produces Heaps beta ~ 0.16 (R2=0.96) and Gini ~ 0.04, with temporal orientation modulating transition dynamics.",
    "functional": "The metathetic extension provides a mechanism for how innovation is distributed across agents rather than aggregated. Low Gini (no winner-take-all) and sublinear Heaps law (diminishing returns on type diversity) are consistent with Taalbi's predictions for resource-constrained multi-agent recombinant search. The temporal orientation gate adds phase-awareness: agents cycle through desituated (novelty), situated (productive), established (stable), inertial (declining), and potentially annihilated (relationally dead) phases, modulating their propensity for identity transformation. This connects innovation dynamics to organizational lifecycle theory.",
    "evidence_tier": "C",
    "tier_justification": "Novel multi-agent extension not present in source TAP literature; single-configuration results; requires independent validation, parameter sensitivity analysis, and comparison with empirical firm-level data.",
    "artifacts": ["simulator/metathetic.py", "outputs/longrun_diagnostics_summary.json", "outputs/figures/heaps_law.png", "outputs/figures/concentration_gini.png"],
    "claim_policy_label": "exploratory"
  }
}
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_evidence_report.py -v`
Expected: ALL PASS (8 tests)

### Step 5: Commit

```bash
git add config/claim_annotations.json tests/test_evidence_report.py
git commit -m "feat: add claim annotations config with mechanistic/functional blocks + tiers"
```

---

## Task 5: Build evidence report validation script (Thread B)

**Files:**
- Create: `scripts/build_evidence_report.py`
- Test: `tests/test_evidence_report.py` (add validation tests)

### Step 1: Write the failing tests

Add to `tests/test_evidence_report.py`:

```python
from scripts.build_evidence_report import validate_annotations, build_report


class TestValidateAnnotations(unittest.TestCase):
    """Validation logic for claim annotations."""

    def test_missing_mechanistic_fails(self):
        bad = {"C1": {"functional": "x" * 25, "evidence_tier": "A",
                       "tier_justification": "y" * 15, "artifacts": ["a.csv"],
                       "claim_policy_label": "paper-aligned"}}
        errors = validate_annotations(bad, claim_ids=["C1"])
        self.assertTrue(any("mechanistic" in e for e in errors))

    def test_missing_functional_fails(self):
        bad = {"C1": {"mechanistic": "x" * 25, "evidence_tier": "A",
                       "tier_justification": "y" * 15, "artifacts": ["a.csv"],
                       "claim_policy_label": "paper-aligned"}}
        errors = validate_annotations(bad, claim_ids=["C1"])
        self.assertTrue(any("functional" in e for e in errors))

    def test_exploratory_tier_a_fails(self):
        bad = {"C1": {"mechanistic": "x" * 25, "functional": "y" * 25,
                       "evidence_tier": "A", "tier_justification": "z" * 15,
                       "artifacts": ["a.csv"],
                       "claim_policy_label": "exploratory"}}
        errors = validate_annotations(bad, claim_ids=["C1"])
        self.assertTrue(any("tier" in e.lower() for e in errors))

    def test_missing_claim_fails(self):
        good = {"C1": {"mechanistic": "x" * 25, "functional": "y" * 25,
                        "evidence_tier": "A", "tier_justification": "z" * 15,
                        "artifacts": ["a.csv"],
                        "claim_policy_label": "paper-aligned"}}
        errors = validate_annotations(good, claim_ids=["C1", "C2"])
        self.assertTrue(any("C2" in e for e in errors))

    def test_valid_annotations_no_errors(self):
        good = {"C1": {"mechanistic": "x" * 25, "functional": "y" * 25,
                        "evidence_tier": "A", "tier_justification": "z" * 15,
                        "artifacts": ["a.csv"],
                        "claim_policy_label": "paper-aligned"}}
        errors = validate_annotations(good, claim_ids=["C1"])
        self.assertEqual(errors, [])


class TestBuildReport(unittest.TestCase):
    def test_builds_from_real_annotations(self):
        data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        report = build_report(data)
        self.assertIn("claims", report)
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["total_claims"], 8)
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_evidence_report.py::TestValidateAnnotations -v`
Expected: FAIL — `scripts.build_evidence_report` module not found

### Step 3: Create build_evidence_report.py

Create `scripts/build_evidence_report.py`:

```python
"""Build and validate the evidence report from claim annotations.

Reads config/claim_annotations.json, validates completeness and
tier consistency, cross-references artifacts, and produces
outputs/evidence_report.json.

Usage:
  python scripts/build_evidence_report.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO / "config" / "claim_annotations.json"
CLAIMS_MD_PATH = REPO / "CLAIMS.md"
OUTPUT_PATH = REPO / "outputs" / "evidence_report.json"


def parse_claim_ids_from_claims_md(path: Path) -> list[str]:
    """Extract claim IDs (C1, C2, ...) from CLAIMS.md table."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return re.findall(r'\|\s*(C\d+)\s*\|', text)


def validate_annotations(
    data: dict,
    claim_ids: list[str],
) -> list[str]:
    """Validate claim annotations for completeness and consistency.

    Returns list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    # Check all claims are present.
    for cid in claim_ids:
        if cid not in data:
            errors.append(f"{cid}: missing from annotations")

    for cid, entry in data.items():
        # Required fields.
        for field in ("mechanistic", "functional", "evidence_tier",
                      "tier_justification", "artifacts", "claim_policy_label"):
            if field not in entry:
                errors.append(f"{cid}: missing required field '{field}'")

        # Non-empty text blocks.
        for field in ("mechanistic", "functional"):
            val = entry.get(field, "")
            if len(val) < 20:
                errors.append(f"{cid}: {field} block too short ({len(val)} chars, need >= 20)")

        # Tier validity.
        tier = entry.get("evidence_tier", "")
        if tier not in ("A", "B", "C"):
            errors.append(f"{cid}: invalid evidence_tier '{tier}' (must be A, B, or C)")

        # Tier justification.
        tj = entry.get("tier_justification", "")
        if len(tj) < 10:
            errors.append(f"{cid}: tier_justification too short")

        # Exploratory claims must be tier C.
        if entry.get("claim_policy_label") == "exploratory" and tier != "C":
            errors.append(f"{cid}: exploratory claim must be tier C, got '{tier}'")

        # Artifacts list.
        arts = entry.get("artifacts", [])
        if not isinstance(arts, list) or len(arts) == 0:
            errors.append(f"{cid}: artifacts must be a non-empty list")

    return errors


def check_artifacts_exist(data: dict, repo: Path) -> list[str]:
    """Warn about missing artifact files. Returns list of warnings."""
    warnings: list[str] = []
    for cid, entry in data.items():
        for art in entry.get("artifacts", []):
            p = repo / art
            if not p.exists():
                warnings.append(f"{cid}: artifact not found: {art}")
    return warnings


def build_report(data: dict) -> dict:
    """Build the enriched evidence report from validated annotations."""
    tier_counts = {"A": 0, "B": 0, "C": 0}
    claims = {}
    for cid, entry in sorted(data.items()):
        tier = entry.get("evidence_tier", "C")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        claims[cid] = {
            "mechanistic": entry["mechanistic"],
            "functional": entry["functional"],
            "evidence_tier": tier,
            "tier_justification": entry["tier_justification"],
            "artifacts": entry["artifacts"],
            "claim_policy_label": entry["claim_policy_label"],
        }

    return {
        "claims": claims,
        "summary": {
            "total_claims": len(claims),
            "tier_distribution": tier_counts,
            "all_validated": True,
        },
    }


def main() -> None:
    if not ANNOTATIONS_PATH.exists():
        raise SystemExit(f"Missing: {ANNOTATIONS_PATH}")

    data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    claim_ids = parse_claim_ids_from_claims_md(CLAIMS_MD_PATH)
    if not claim_ids:
        claim_ids = list(data.keys())
        print(f"Warning: could not parse CLAIMS.md, using annotation keys: {claim_ids}")

    errors = validate_annotations(data, claim_ids)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(f"Evidence report validation failed with {len(errors)} error(s)")

    warnings = check_artifacts_exist(data, REPO)
    if warnings:
        print("Artifact warnings (non-fatal):")
        for w in warnings:
            print(f"  - {w}")

    report = build_report(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")

    td = report["summary"]["tier_distribution"]
    print(f"  Tier A (robust):      {td['A']}")
    print(f"  Tier B (supported):   {td['B']}")
    print(f"  Tier C (exploratory): {td['C']}")


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_evidence_report.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add scripts/build_evidence_report.py tests/test_evidence_report.py
git commit -m "feat: add evidence report builder with validation + tier enforcement"
```

---

## Task 6: Pipeline integration + final verification

**Files:**
- Modify: `run_reporting_pipeline.py` (add evidence report step)

### Step 1: Add evidence report to pipeline

In `run_reporting_pipeline.py`, after the longrun_diagnostics line (line 88), add:

```python
    # Evidence report: mechanistic/functional interpretation + tier validation.
    run([sys.executable, 'scripts/build_evidence_report.py'])
```

### Step 2: Run full test suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: ALL PASS (should be ~100+ tests)

### Step 3: Run longrun diagnostics to regenerate outputs with temporal states

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe scripts/longrun_diagnostics.py --seed 42`
Expected: Output includes temporal distribution

### Step 4: Run evidence report builder

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe scripts/build_evidence_report.py`
Expected: Produces outputs/evidence_report.json, shows tier distribution

### Step 5: Commit everything and push

```bash
git add run_reporting_pipeline.py outputs/longrun_diagnostics_summary.json outputs/longrun_diagnostics.csv outputs/evidence_report.json outputs/figures/heaps_law.png outputs/figures/concentration_gini.png
git commit -m "feat: wire evidence report + temporal diagnostics into pipeline"
git push origin unified-integration-20260225
```

---

## Summary

| Task | Thread | What | Tests added |
|------|--------|------|-------------|
| 1 | A | Temporal state property on MetatheticAgent | 6 |
| 2 | A | Wire temporal gate into ensemble dynamics | 7 |
| 3 | A | Add temporal state to longrun diagnostics | 0 (covered by pipeline) |
| 4 | B | Create claim_annotations.json config | 8 |
| 5 | B | Build evidence report validation script | 6 |
| 6 | Both | Pipeline integration + final verification | 0 |
| **Total** | | | **27 new tests** |
