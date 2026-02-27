# Stage 3A: Two-Channel Consummation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Youn ratio (stuck at 1.0) by adding per-agent TAPS signatures
and signature-based cross-metathesis classification.

**Architecture:** Three layers — (1) per-agent L-matrix event ledger on
MetatheticAgent, (2) TAPS signature derivation as a method, (3) signature
comparison replacing `if L > G` in cross-metathesis classification. TDD
throughout.

**Tech Stack:** Python 3.12, existing `simulator/metathetic.py`, `tests/test_metathetic.py`

**Design doc:** `docs/plans/2026-02-26-stage3a-two-channel-design.md`

---

## Context for Implementer

### The Problem

In `simulator/metathetic.py`, the method `_check_cross_metathesis()` (line 567)
classifies every cross-metathesis event as either novel (Mode 3) or absorptive
(Mode 2). The decision at line 623 is `if L > G:` where L = Jaccard similarity
and G = goal alignment (clamped ≥ 0). Due to structural correlation between L
and G, absorptive NEVER fires — across 576 simulations, n_absorptive_cross = 0.
The Youn ratio (novel / total cross) is stuck at 1.0. Target: ≈ 0.60.

### The Fix

1. Give each agent a per-event ledger tracking Emery L-matrix channels
2. Derive a 4-letter TAPS signature from the ledger
3. Replace `if L > G` with signature-similarity classification

### Key Constraint

The eligibility gate (`L + G <= (W1 + W2) * threshold`) is UNCHANGED.
Same events fire at the same frequency. Only the novel/absorptive
classification changes.

### Files You'll Touch

- `simulator/metathetic.py` — agent fields, signature method, classification logic
- `tests/test_metathetic.py` — new test classes for ledger, signature, classification

### Running Tests

```
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v --tb=short
```

For a single test class:
```
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestClassName -v
```

Currently 273 tests (78 in test_metathetic.py), all passing.

---

## Task 1: Per-Agent L-Matrix Event Ledger

Add 5 integer counters to `MetatheticAgent` tracking Emery's four channels.

**Files:**
- Modify: `simulator/metathetic.py:36-55` (MetatheticAgent dataclass)
- Test: `tests/test_metathetic.py`

### Step 1: Write failing tests

Add to `tests/test_metathetic.py`:

```python
class TestLMatrixLedger(unittest.TestCase):
    """Per-agent L-matrix event ledger (Emery channels)."""

    def test_initial_ledger_zeros(self):
        """New agent starts with all L-matrix counters at zero."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        self.assertEqual(agent.n_self_metatheses_local, 0)
        self.assertEqual(agent.n_novel_cross_local, 0)
        self.assertEqual(agent.n_absorptive_given_local, 0)
        self.assertEqual(agent.n_absorptive_received_local, 0)
        self.assertEqual(agent.n_env_transitions_local, 0)

    def test_self_metathesis_increments_l11(self):
        """Self-metathesis increments the L11 (intrapraxis) counter."""
        agent = MetatheticAgent(agent_id=0, type_set={1}, k=0.0, M_local=10.0)
        agent.self_metathesize(next_type_id=99)
        self.assertEqual(agent.n_self_metatheses_local, 1)
        agent.self_metathesize(next_type_id=100)
        self.assertEqual(agent.n_self_metatheses_local, 2)

    def test_absorptive_cross_increments_l12_l21(self):
        """Absorptive cross increments L12 for donor (given), L21 for receiver."""
        a1 = MetatheticAgent(agent_id=1, type_set={1, 2}, k=10.0, M_local=50.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2, 3}, k=5.0, M_local=20.0)
        MetatheticAgent.absorptive_cross(a1, a2)
        # a1 has higher M_local so a1 absorbs a2
        # a1 = absorber = received; a2 = absorbed = given
        self.assertEqual(a1.n_absorptive_received_local, 1)
        self.assertEqual(a2.n_absorptive_given_local, 1)

    def test_novel_cross_increments_l12(self):
        """Novel cross increments L12 (novel_cross_local) for both parents."""
        a1 = MetatheticAgent(agent_id=1, type_set={1}, k=5.0, M_local=10.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2}, k=5.0, M_local=10.0)
        child = MetatheticAgent.novel_cross(a1, a2, child_id=3, next_type_id=99)
        self.assertEqual(a1.n_novel_cross_local, 1)
        self.assertEqual(a2.n_novel_cross_local, 1)
        # Child starts fresh
        self.assertEqual(child.n_novel_cross_local, 0)
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestLMatrixLedger -v`

Expected: FAIL — `AttributeError: 'MetatheticAgent' has no attribute 'n_self_metatheses_local'`

### Step 3: Add ledger fields to MetatheticAgent

In `simulator/metathetic.py`, add these fields to the `MetatheticAgent`
dataclass (after line 54, before `_dissolved`):

```python
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
```

### Step 4: Increment counters in existing metathesis methods

In `self_metathesize` (line 146), add after line 154:
```python
        self.n_self_metatheses_local += 1
```

In `absorptive_cross` (line 158), add after line 175 (`absorbed.active = False`):
```python
        absorber.n_absorptive_received_local += 1
        absorbed.n_absorptive_given_local += 1
```

In `novel_cross` (line 179), add before `a1.active = False` (line 205):
```python
        a1.n_novel_cross_local += 1
        a2.n_novel_cross_local += 1
```

### Step 5: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestLMatrixLedger -v`

Expected: 4 PASSED

### Step 6: Run full suite to check no regressions

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 277 passed (273 + 4 new)

### Step 7: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat(stage3a): add per-agent L-matrix event ledger (Emery channels)"
```

---

## Task 2: TAPS Signature Derivation

Add a `taps_signature` property to `MetatheticAgent` that returns a 4-letter
string derived from the L-matrix ledger.

**Files:**
- Modify: `simulator/metathetic.py` (add method to MetatheticAgent)
- Test: `tests/test_metathetic.py`

### Step 1: Write failing tests

```python
class TestTAPSSignature(unittest.TestCase):
    """Per-agent TAPS dispositional signature."""

    def _make_agent(self, **overrides):
        """Helper: create agent with specified ledger values."""
        defaults = dict(
            agent_id=0, type_set={1}, k=0.0, M_local=10.0, active=True,
        )
        defaults.update(overrides)
        return MetatheticAgent(**defaults)

    def test_fresh_agent_default_signature(self):
        """Agent with no events gets default signature TERS."""
        agent = self._make_agent()
        self.assertEqual(len(agent.taps_signature), 4)
        # T=T (balanced, no events), A=E (default expression),
        # P=X (balanced, no events), S=S (default synthesis)

    def test_signature_is_four_letters(self):
        """Signature always returns exactly 4 characters."""
        agent = self._make_agent()
        agent.n_self_metatheses_local = 10
        agent.n_novel_cross_local = 5
        self.assertEqual(len(agent.taps_signature), 4)

    def test_involution_dominant_gives_I(self):
        """Agent with mostly L11+L21 events has T-letter = I."""
        agent = self._make_agent()
        agent.n_self_metatheses_local = 10  # L11
        agent.n_absorptive_received_local = 5  # L21
        agent.n_novel_cross_local = 1  # L12 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[0], "I")

    def test_evolution_dominant_gives_E(self):
        """Agent with mostly L12+L22 events has T-letter = E."""
        agent = self._make_agent()
        agent.n_novel_cross_local = 10  # L12
        agent.n_env_transitions_local = 5  # L22
        agent.n_self_metatheses_local = 1  # L11 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[0], "E")

    def test_balanced_gives_T(self):
        """Agent with balanced inward/outward has T-letter = T."""
        agent = self._make_agent()
        agent.n_self_metatheses_local = 5  # L11
        agent.n_novel_cross_local = 5  # L12
        sig = agent.taps_signature
        self.assertEqual(sig[0], "T")

    def test_consummation_dominant_gives_U(self):
        """Agent with mostly L12 outward events has P-letter = U."""
        agent = self._make_agent()
        agent.n_novel_cross_local = 10  # L12 outward
        agent.n_absorptive_received_local = 1  # L21 inward (low)
        sig = agent.taps_signature
        self.assertEqual(sig[2], "U")

    def test_consumption_dominant_gives_R(self):
        """Agent with mostly L21 inward events has P-letter = R."""
        agent = self._make_agent()
        agent.n_absorptive_received_local = 10  # L21
        agent.n_novel_cross_local = 1  # L12 (low)
        sig = agent.taps_signature
        self.assertEqual(sig[2], "R")

    def test_signature_changes_with_events(self):
        """Signature evolves as agent accumulates events."""
        agent = self._make_agent()
        sig1 = agent.taps_signature
        agent.n_novel_cross_local = 20
        sig2 = agent.taps_signature
        # Not necessarily different (depends on other counts), but should
        # be computable without error at any point.
        self.assertEqual(len(sig2), 4)
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestTAPSSignature -v`

Expected: FAIL — `AttributeError: 'MetatheticAgent' has no attribute 'taps_signature'`

### Step 3: Implement taps_signature property

Add to `MetatheticAgent` class, after the `affordance_score` property
(after line 142):

```python
    @property
    def taps_signature(self) -> str:
        """4-letter TAPS dispositional signature from L-matrix ledger.

        Each letter derived from the agent's local event history:
          T: Involution(I) / Evolution(E) / Transvolution(T)
          A: Expression(E) / Impression(I) / Adpression(A)
          P: Reflection-consumption(R) / Projection-consummation(U) / Pure action(X)
          S: Disintegration(D) / Preservation(P) / Integration(I) / Synthesis(S)

        See docs/plans/2026-02-26-stage3a-two-channel-design.md for full spec.
        """
        # -- T-letter: Transvolution --
        # L11 + L21 = inward; L12 + L22 = outward
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
        # Expression: positive recent dM + good affordance
        # Impression: L21 dominates recent events
        # Adpression: recent self-metathesis (steps_since == 0 means just happened)
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
        # Consumption (R): L21 inward events dominate
        # Consummation (U): L12 outward events dominate
        l21_total = self.n_absorptive_received_local
        l12_total = self.n_novel_cross_local + self.n_absorptive_given_local
        if l21_total > l12_total * 1.2:
            p_letter = "R"
        elif l12_total > l21_total * 1.2:
            p_letter = "U"
        else:
            p_letter = "X"

        # -- S-letter: Syntegration --
        # D: disintegration events, P: dormant/preserving,
        # I: absorptive integration, S: synthesis (self-meta + novel)
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
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestTAPSSignature -v`

Expected: 9 PASSED

### Step 5: Run full suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 286 passed (277 + 9 new)

### Step 6: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat(stage3a): add per-agent TAPS signature derivation from L-matrix"
```

---

## Task 3: Signature-Based Cross-Metathesis Classification

Replace `if L > G` with signature-similarity three-level tension rule.

**Files:**
- Modify: `simulator/metathetic.py:622-641` (_check_cross_metathesis)
- Test: `tests/test_metathetic.py`

### Step 1: Write failing tests

```python
class TestSignatureClassification(unittest.TestCase):
    """Three-level tension classification for cross-metathesis."""

    def _make_agent_with_sig(self, agent_id, sig_target, type_set, M_local=10.0):
        """Create agent and manipulate ledger to produce target signature.

        sig_target: 4-letter string like 'IEUS'. We set ledger counts
        to steer the signature toward this target.
        """
        agent = MetatheticAgent(
            agent_id=agent_id, type_set=type_set, k=5.0, M_local=M_local,
            dM_history=[1.0, 1.0, 1.0],
        )
        agent._affordance_ticks = [1, 1, 1, 1, 1]  # high affordance

        # T-letter
        if sig_target[0] == "I":
            agent.n_self_metatheses_local = 10
        elif sig_target[0] == "E":
            agent.n_novel_cross_local = 10
        # T → balanced, leave at 0

        # Steer other dimensions similarly as needed
        if sig_target[2] == "U":
            agent.n_novel_cross_local = max(agent.n_novel_cross_local, 10)
        elif sig_target[2] == "R":
            agent.n_absorptive_received_local = 10

        return agent

    def test_identical_signatures_produce_absorptive(self):
        """Two agents with 4/4 matching signature → absorptive (low tension)."""
        # Both will have mostly self-metathesis → I-dominant T-letter
        a1 = MetatheticAgent(agent_id=1, type_set={1, 2, 3}, k=5.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=2, type_set={1, 2, 3}, k=5.0, M_local=15.0)
        # Give both identical event profiles
        for a in (a1, a2):
            a.n_self_metatheses_local = 10
            a.dM_history = [1.0, 1.0, 1.0]
            a._affordance_ticks = [1, 1, 1, 1, 1]
        self.assertEqual(a1.taps_signature, a2.taps_signature)
        # Signature similarity = 4 → should be classified as absorptive
        similarity = _signature_similarity(a1.taps_signature, a2.taps_signature)
        self.assertEqual(similarity, 4)

    def test_different_signatures_produce_novel(self):
        """Two agents with 0-1/4 matching signature → novel (high tension)."""
        a1 = MetatheticAgent(agent_id=1, type_set={1}, k=5.0, M_local=20.0)
        a2 = MetatheticAgent(agent_id=2, type_set={2}, k=5.0, M_local=15.0)
        # a1: heavy self-meta → involution-dominant
        a1.n_self_metatheses_local = 20
        a1.dM_history = [1.0, 1.0, 1.0]
        a1._affordance_ticks = [1, 1, 1, 1, 1]
        # a2: heavy novel cross → evolution-dominant
        a2.n_novel_cross_local = 20
        a2.dM_history = [-0.1, -0.1, -0.1]
        a2._affordance_ticks = [0, 0, 0, 0, 0]
        # Signatures should differ significantly
        similarity = _signature_similarity(a1.taps_signature, a2.taps_signature)
        self.assertLessEqual(similarity, 1)

    def test_signature_similarity_function(self):
        """_signature_similarity counts matching positions."""
        self.assertEqual(_signature_similarity("IEUS", "IEUS"), 4)
        self.assertEqual(_signature_similarity("IEUS", "EIRS"), 1)  # only S matches
        self.assertEqual(_signature_similarity("TEXP", "TEXP"), 4)
        self.assertEqual(_signature_similarity("IEUS", "EARS"), 0)

    def test_mid_tension_falls_back_to_L_vs_G(self):
        """With 2/4 matching letters, classification uses L vs G tiebreak."""
        self.assertEqual(_signature_similarity("IEUS", "IERS"), 2)
```

Also add `_signature_similarity` to the import at the top of the test file.

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestSignatureClassification -v`

Expected: FAIL — `ImportError: cannot import name '_signature_similarity'`

### Step 3: Implement _signature_similarity and rewire classification

**3a.** Add `_signature_similarity` as a module-level helper function in
`simulator/metathetic.py` (near the other helper functions like `_jaccard`,
around line 285):

```python
def _signature_similarity(sig1: str, sig2: str) -> int:
    """Count matching positions between two 4-letter TAPS signatures.

    Returns 0-4. Used for three-level tension classification:
      3-4 matches = low tension (absorptive)
      2 matches   = mid tension (L vs G tiebreak)
      0-1 matches = high tension (novel)
    """
    return sum(c1 == c2 for c1, c2 in zip(sig1, sig2))
```

**3b.** Replace the classification block in `_check_cross_metathesis`
(lines 622-638). The new block replaces everything from `# Eligible —
determine mode.` to the `self.n_novel_cross += 1` line:

```python
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
                    # Mid tension (2 matches): fall back to L vs G tiebreak.
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
```

**3c.** Update the docstring of `_check_cross_metathesis` to reflect the new
classification (replace lines 568-578):

```python
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
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestSignatureClassification -v`

Expected: 4 PASSED

### Step 5: Run full suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 290 passed (286 + 4 new)

**Important:** Some existing tests that check cross-metathesis behavior may now
produce absorptive events where they previously produced novel events. If any
existing test fails, examine the failure — if it's because an event that was
novel is now absorptive (the intended change), update the assertion. Do NOT
change the classification logic to make old tests pass.

### Step 6: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat(stage3a): three-level tension classification for cross-metathesis"
```

---

## Task 4: Integration Test — Youn Ratio Improvement

Verify that the Youn ratio has moved off 1.0 in a full ensemble run.

**Files:**
- Test: `tests/test_metathetic.py`

### Step 1: Write the integration test

```python
class TestYounRatioImprovement(unittest.TestCase):
    """Verify Stage 3A classification produces absorptive events."""

    def test_youn_ratio_below_one(self):
        """Full ensemble run should produce at least some absorptive events.

        The Youn ratio (novel / total_cross) should be < 1.0, indicating
        the signature-based classification is producing absorptive events
        that the old L > G rule never did.
        """
        ensemble = MetatheticEnsemble(
            n_agents=8,
            alpha=5e-3,
            a=3.0,
            mu=0.005,
            carrying_capacity=500.0,
            seed=42,
        )
        traj = ensemble.run(steps=150)

        final = traj[-1]
        n_novel = final["n_novel_cross"]
        n_absorptive = final["n_absorptive_cross"]
        total_cross = n_novel + n_absorptive

        # Must have SOME cross-metathesis events
        self.assertGreater(total_cross, 0,
                           "No cross-metathesis events at all")

        # Must have at least one absorptive event (Youn ratio < 1.0)
        self.assertGreater(n_absorptive, 0,
                           f"Youn ratio still 1.0: {n_novel} novel, "
                           f"{n_absorptive} absorptive")

    def test_youn_ratio_in_target_range(self):
        """Youn exploration fraction should be closer to 0.6 than to 1.0.

        This is a soft target — we check that the ratio has meaningfully
        moved toward the empirical target, not that it's exactly 0.6.
        """
        ensemble = MetatheticEnsemble(
            n_agents=8,
            alpha=5e-3,
            a=3.0,
            mu=0.005,
            carrying_capacity=500.0,
            seed=42,
        )
        traj = ensemble.run(steps=150)

        final = traj[-1]
        n_novel = final["n_novel_cross"]
        n_absorptive = final["n_absorptive_cross"]
        total_cross = n_novel + n_absorptive

        if total_cross == 0:
            self.skipTest("No cross-metathesis events")

        exploration_fraction = n_novel / total_cross
        # Should be meaningfully below 1.0 (moved toward 0.6 target)
        self.assertLess(exploration_fraction, 0.95,
                        f"Youn ratio barely moved: {exploration_fraction:.3f}")
```

### Step 2: Run the integration test

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestYounRatioImprovement -v`

Expected: 2 PASSED

**If the test fails** (exploration_fraction still 1.0 or still > 0.95):
The signature derivation may need tuning. Most likely causes:
- All agents develop identical signatures (signatures not differentiating enough)
- The 1.2x threshold for T/P letters is too aggressive (try 1.1 or 1.0)
- Agents accumulate too few events for signatures to diverge

Debug by adding a print: `print(f"Sigs: {[a.taps_signature for a in ensemble._active_agents()]}")` just before the classification line. Check if signatures are actually varying.

### Step 3: Run full suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 292 passed (290 + 2 new)

### Step 4: Commit

```bash
git add tests/test_metathetic.py
git commit -m "test(stage3a): integration test confirms Youn ratio moved off 1.0"
```

---

## Task 5: Snapshot & Diagnostic Wiring

Add per-agent signature data to the ensemble snapshot so TAPS diagnostics
can see it.

**Files:**
- Modify: `simulator/metathetic.py:789-828` (snapshot block)
- Test: `tests/test_metathetic.py`

### Step 1: Write failing test

```python
class TestSignatureSnapshot(unittest.TestCase):
    """Verify per-agent signature data appears in ensemble snapshots."""

    def test_snapshot_has_signature_distribution(self):
        """Each snapshot includes signature_distribution dict."""
        ensemble = MetatheticEnsemble(
            n_agents=4, alpha=1e-3, a=8.0, mu=0.02, seed=1
        )
        traj = ensemble.run(steps=20)
        for snap in traj:
            self.assertIn("signature_distribution", snap)
            self.assertIsInstance(snap["signature_distribution"], dict)
            # Values should sum to number of active agents
            self.assertEqual(
                sum(snap["signature_distribution"].values()),
                snap["n_active"],
            )

    def test_snapshot_has_signature_diversity(self):
        """Each snapshot includes signature_diversity (count of unique sigs)."""
        ensemble = MetatheticEnsemble(
            n_agents=4, alpha=1e-3, a=8.0, mu=0.02, seed=1
        )
        traj = ensemble.run(steps=20)
        for snap in traj:
            self.assertIn("signature_diversity", snap)
            self.assertGreaterEqual(snap["signature_diversity"], 1)
            self.assertLessEqual(snap["signature_diversity"], snap["n_active"])
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestSignatureSnapshot -v`

Expected: FAIL — `KeyError: 'signature_distribution'`

### Step 3: Add signature data to snapshot

In `simulator/metathetic.py`, in the snapshot block (after the
`affordance_mean` computation, around line 815), add:

```python
            # TAPS signature distribution for active agents.
            sig_counts: dict[str, int] = {}
            for a in active:
                sig = a.taps_signature
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
            snapshot["signature_distribution"] = sig_counts
            snapshot["signature_diversity"] = len(sig_counts)
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestSignatureSnapshot -v`

Expected: 2 PASSED

### Step 5: Run full suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 294 passed (292 + 2 new)

### Step 6: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat(stage3a): add signature distribution and diversity to snapshots"
```

---

## Task 6: Env Transition Tracking Per-Agent

Currently `n_env_transitions` is tracked ensemble-wide but not per-agent
(L22 channel). Wire environment transitions to individual agents.

**Files:**
- Modify: `simulator/metathetic.py` (_update_environment, around line 750)
- Test: `tests/test_metathetic.py`

### Step 1: Write failing test

```python
class TestEnvTransitionPerAgent(unittest.TestCase):
    """L22 channel: env transitions recorded per agent."""

    def test_env_transition_increments_active_agents(self):
        """When environment texture changes, all active agents get L22 tick."""
        ensemble = MetatheticEnsemble(
            n_agents=4, alpha=5e-3, a=3.0, mu=0.005, seed=42,
            carrying_capacity=500.0,
        )
        # Run enough steps that at least one env transition occurs
        traj = ensemble.run(steps=100)
        final = traj[-1]

        if final["n_env_transitions"] == 0:
            self.skipTest("No env transitions occurred in this run")

        # At least one active agent should have recorded L22 events
        active = ensemble._active_agents()
        total_l22 = sum(a.n_env_transitions_local for a in active)
        self.assertGreater(total_l22, 0,
                           "Env transitions happened but no agent recorded L22")
```

### Step 2: Run tests to verify they fail

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestEnvTransitionPerAgent -v`

Expected: FAIL — agents have `n_env_transitions_local = 0` even when
ensemble-level `n_env_transitions > 0`

### Step 3: Wire env transitions to active agents

In `simulator/metathetic.py`, in `_update_environment` (around line 751-754),
after incrementing the ensemble counter, add per-agent tracking:

```python
        old_texture = self.env.texture_type
        self.env.update(D, k_total, total_M, regime)
        if self.env.texture_type != old_texture:
            self.n_env_transitions += 1
            # L22: all active agents observe the environmental shift
            for agent in self._active_agents():
                agent.n_env_transitions_local += 1
```

### Step 4: Run tests to verify they pass

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_metathetic.py::TestEnvTransitionPerAgent -v`

Expected: 1 PASSED

### Step 5: Run full suite

Run: `C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -q --tb=short`

Expected: 295 passed (294 + 1 new)

### Step 6: Commit

```bash
git add simulator/metathetic.py tests/test_metathetic.py
git commit -m "feat(stage3a): wire env transitions to per-agent L22 channel"
```

---

## Task 7: Empirical Sweep Validation

Run the empirical sweep to verify Youn ratio improvement across parameter
space. This is a validation step, not a code change.

**Files:**
- No code changes

### Step 1: Run the quick empirical sweep

```bash
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe scripts/empirical_sweep.py --quick
```

Examine the output for `youn_deviation`. Previously this was 0.4 (exploration
fraction = 1.0, target = 0.6, deviation = 0.4) for every parameter
combination.

Expected: `youn_deviation` values should be smaller than 0.4 for at least
some parameter combinations, indicating the Youn ratio has moved toward 0.6.

### Step 2: Run full test suite one final time

```bash
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v --tb=short
```

Expected: 295 passed (273 original + 22 new)

### Step 3: Commit any output changes

If `outputs/` files changed (e.g., sweep results):

```bash
git add outputs/
git commit -m "data(stage3a): updated sweep results with signature-based classification"
```

---

## Final Verification

### Step 1: Run full test suite

```bash
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v --tb=short
```

Expected: ≥295 passed

### Step 2: Smoke tests

```bash
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe scripts/run_demo.py
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe -c "from scripts.empirical_validation import main; print('OK')"
```

### Step 3: Create PR

Use finishing-a-development-branch skill.
