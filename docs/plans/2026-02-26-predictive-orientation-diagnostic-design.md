# Predictive Orientation Diagnostic — Design Document

**CLAIM POLICY LABEL: exploratory**

> This document specifies the design for a predictive orientation diagnostic
> module within sigma-TAP. The diagnostic uses Markov transition matrices to
> predict step-ahead system modes, detect parallel matching (the usual case),
> and flag potential adpressive events (the surprisal case).

**Date:** 2026-02-26
**Status:** Approved — ready for implementation planning
**Precondition:** Poincare eigenvalue analysis (PR #3) merged to main

---

## 1. Motivation

All existing sigma-TAP analysis is post-hoc: we run a simulation, then
summarize what happened. The predictive orientation diagnostic adds a
**forward-looking** layer that asks, at each step: "Given the system's
established tendencies, what is it likely to do next?"

This serves three purposes:

1. **Parallel matching** (the "usual" tail): Quantify how often the most
   probable transition actually occurs. High parallel matching = the system
   is behaving predictably within its established regime.

2. **Surprisal detection** (the "surprisal" tail): Quantify information-
   theoretic surprise when the system departs from its expected trajectory.
   Surprisal = -log2(P(actual | predicted)).

3. **Adpression detection**: Flag candidate adpressive events where a high-
   probability flow suddenly becomes something improbable. This is a special
   case of surprisal — not just "unusual" but "surprising given what was
   expected." Includes both novel creation (unexpected transition) and
   annihilation/extinction (state collapse).

The diagnostic is a general-purpose predictive orientation tool, not solely a
surprisal detector. Surprisal is one detectable case within a broader
mechanism that primarily tracks predictable unfolding.

---

## 2. Architecture

### 2.1 Module Layout

```
simulator/predictive.py          <- NEW: core prediction engine
tests/test_predictive.py         <- NEW: dedicated test file
scripts/taps_diagnostics.py      <- MODIFY: add diagnostic output section
docs/empirical_targets.md        <- MODIFY: document future calibration
```

**Dependency flow:**
```
simulator/predictive.py
  +-- imports from taps_sensitivity.py: build_transition_map()
  +-- imports from taps.py: classify_step()
  +-- numpy (matrix power for multi-step look-ahead)

scripts/taps_diagnostics.py
  +-- imports from predictive.py: run_predictive_diagnostic()
```

### 2.2 Design Principle

All prediction functions take transition matrices as input, never build them
internally. This ensures that when online/incremental mode is added (see
Section 8, Future Work), only the matrix-sourcing strategy changes — the
prediction logic remains identical.

---

## 3. Data Structures

```python
from dataclasses import dataclass, field

@dataclass
class StepPrediction:
    """One prediction per axis per step."""
    step: int
    axis: str
    predicted_distribution: dict[str, float]   # state -> probability
    top_predictions: list[tuple[str, float]]   # sorted by probability, up to top_k
    actual_state: str
    parallel_match: bool                       # actual == top_predictions[0][0]
    surprisal: float                           # -log2(P(actual))

@dataclass
class AdpressionEvent:
    """Flagged when surprisal exceeds adaptive threshold."""
    step: int
    axis: str
    surprisal: float
    threshold: float                           # running_mean + threshold_sd * running_sd
    predicted_top: str                         # most probable state
    actual_state: str
    probability_of_actual: float
    event_type: str                            # "transition_surprisal" or "state_collapse"

@dataclass
class PredictiveDiagnosticResult:
    """Full diagnostic output."""
    predictions: list[StepPrediction]
    parallel_matching_rate: dict[str, float]   # per axis
    mean_surprisal: dict[str, float]           # per axis
    adpression_events: list[AdpressionEvent]
    state_collapse_events: list[AdpressionEvent]  # extinction/annihilation subset
    horizon: int
    grain: str                                 # "coarse" or "fine"
    step_count: int
    axes_analyzed: list[str]
```

The **two tails** are captured by:
- `parallel_matching_rate` = the usual tail (how often the probable happened)
- `adpression_events` = the surprisal tail (when the improbable happened)
- `state_collapse_events` = annihilation/extinction subset of adpression

---

## 4. Core Functions

### 4.1 predict_step()

```python
def predict_step(
    current_state: str,
    transition_matrix: np.ndarray,
    states: list[str],
    horizon: int = 1,
) -> dict[str, float]:
```

Atomic prediction unit. Looks up current_state in the state list, extracts
the corresponding row from the transition matrix (or P^k for multi-step
look-ahead), returns probability distribution over next states.

For horizon > 1: uses numpy matrix power P^k. The (i,j) entry of P^k is the
k-step transition probability from state i to state j.

### 4.2 compute_surprisal()

```python
def compute_surprisal(
    predicted_dist: dict[str, float],
    actual_state: str,
    max_surprisal: float = 10.0,
) -> float:
```

Returns -log2(P(actual_state)). If probability is 0, returns max_surprisal
(default 10.0 bits, equivalent to P < 1/1024).

### 4.3 detect_adpression()

```python
def detect_adpression(
    surprisals: list[float],
    threshold_sd: float = 2.0,
    burn_in: int = 5,
) -> list[tuple[int, float, float]]:
```

Walks through surprisal time series with expanding-window running mean and
standard deviation. Flags steps where surprisal > running_mean + threshold_sd
* running_sd. Returns list of (step_index, surprisal_value, threshold_at_step).

First `burn_in` steps (default 5) are excluded from flagging — the running
SD is unreliable with < 5 data points.

**Threshold justification (2 SD default):**
- Cross-domain empirical grounding: seismology (Gutenberg-Richter), ecological
  regime shifts (Scheffer et al. 2009), information theory (log-normal surprisal),
  statistical process control (2-sigma rule)
- Captures ~5% of events as "notable anomalies"
- Configurable via threshold_sd parameter for domain-specific calibration

### 4.4 detect_state_collapse()

```python
def detect_state_collapse(
    trajectory_states: list[str],
    axis: str,
    window: int = 10,
) -> list[AdpressionEvent]:
```

Tracks per-axis state diversity over time using a sliding window. When the
number of distinct states in a window drops from N > 1 to 1, flags a
potential annihilation/extinction adpressive event.

An extinction is a **created** event — the dynamics produced the annihilation;
it did not happen passively. The flag distinguishes this from an axis that was
always single-state (degenerate from the start).

### 4.5 run_predictive_diagnostic()

```python
def run_predictive_diagnostic(
    trajectory: list[dict],
    scores: np.ndarray,
    ano_scores: np.ndarray,
    rip: np.ndarray,
    ratios: dict,
    horizon: int = 1,
    grain: str = "coarse",
    top_k: int = 3,
    threshold_sd: float = 2.0,
) -> PredictiveDiagnosticResult:
```

Main entry point orchestrating the full diagnostic:

1. Calls build_transition_map() to get transition matrices
2. Selects axes based on grain:
   - **Coarse**: syntegration_phase, rip_dominance, ano_dominant
   - **Fine**: All 5 axes (+ pressure_regime, transvolution_dir)
3. Walks trajectory step-by-step (from step 1 onward):
   - For each axis: get state at step t-1, predict step t, compare to actual
   - Records StepPrediction with top_k predictions, parallel match, surprisal
4. Runs detect_adpression() on each axis's surprisal series
5. Runs detect_state_collapse() on each axis's state sequence
6. Computes aggregates (parallel matching rate, mean surprisal)
7. Returns PredictiveDiagnosticResult

---

## 5. Diagnostics Output

New section in taps_diagnostics.py print_summary() and/or a standalone
print_predictive_diagnostic() function.

### Summary Table (always shown)

```
Predictive Orientation Diagnostic (horizon=1, grain=coarse):

  Axis                   Match%   Mean Surprisal   Adpressions
  syntegration_phase     87.5%    0.42 bits        1 event
  rip_dominance          94.2%    0.18 bits        0 events
  ano_dominant           96.0%    0.12 bits        0 events
```

### Adpression Events (if any)

```
  Adpression Events:
    Step 47, syntegration_phase: predicted=disintegration (P=0.82)
      actual=integration (P=0.08), surprisal=3.64 bits, threshold=2.91
      Type: transition_surprisal
```

### State Collapse Events (if any)

```
  State Collapse Events:
    Step 72, pressure_regime: collapsed to single state "entropy"
      Previously active: entropy, homeostasis, morphogenesis
      Type: annihilation/extinction
```

### All Possible Transitions Table (optional, on request)

For each axis at each step (or at a given step), show the full distribution:

```
  Step 47, syntegration_phase (from disintegration):
    disintegration   0.82  ************************************
    integration      0.08  ***
    balanced         0.10  ****
```

---

## 6. Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| Empty trajectory (< 2 steps) | Return empty result, log warning |
| Single-state axis (never changes) | Mark as "degenerate," surprisal=0, match=100% |
| State collapse mid-run | Flag as annihilation/extinction adpressive event |
| Zero-probability transition | Cap surprisal at max_surprisal (10.0 bits), flag step |
| Burn-in period | Exclude first 5 steps from adpression detection |
| Unknown state in trajectory | Skip step with warning (defensive, should not occur) |
| All steps are surprising | If > 50% steps exceed threshold, log warning about model fit |
| horizon > step count | Reduce horizon to available steps, log adjustment |

---

## 7. Testing Strategy

~15-18 tests in tests/test_predictive.py:

1. predict_step: known 2x2 matrix -> correct probabilities
2. predict_step with horizon=2: P^2 gives correct 2-step probabilities
3. compute_surprisal: known probability -> correct bits (e.g., P=0.5 -> 1.0 bit)
4. compute_surprisal zero probability: caps at max_surprisal
5. detect_adpression with injected spike: correctly flagged
6. detect_adpression burn-in: no flags in first 5 steps
7. detect_adpression uniform surprisal: no flags
8. detect_state_collapse: axis collapses -> event flagged
9. detect_state_collapse always single: no event (degenerate from start)
10. run_predictive_diagnostic coarse: correct 3 axes selected
11. run_predictive_diagnostic fine: all 5 axes present
12. parallel_matching_rate with identity matrix: 100% match
13. parallel_matching_rate non-trivial: rate between 0 and 1
14. top_k predictions: correct count and descending probability order
15. adpression_events in result: integration test with sudden transition
16. horizon parameter: different horizons produce different predictions
17. degenerate axis handling: single-state -> degenerate result, no error
18. integration with real sigma-TAP simulation: valid result structure

---

## 8. Future Work

### 8.1 Online/Incremental Mode (HIGH PRIORITY)

Build transition matrix incrementally as the simulation runs, so predictions
at step t are based only on data from steps 0..t-1. This enables:
- Real-time prediction during simulation
- Genuinely forward-looking analysis with no future information leakage
- Integration into the simulator itself (sigma_tap.py)

**Implementation path:** Add a `mode` parameter to run_predictive_diagnostic:
mode="post_hoc" (current, uses full-run matrix) or mode="online"
(incremental). The prediction functions remain unchanged — only the matrix
source differs.

### 8.2 Entity:Ensemble Scaling Dial

A configurable parameter that adjusts the entity:ensemble ratio:
- Default (1.0): Full ensemble prediction — aggregate system-level
- Zoomed (< 1.0): Focus on sub-groups, treating each entity in the group as
  sharing the group's propagated identity (e.g., "all oak trees are oak trees
  but not the same oak tree")
- The ensemble's statistical signature top-down feedback can simulate per-agent
  dynamics without requiring per-agent transition matrices
- This results in a probability two-tail scalar module where scaling the dial
  "drags" the variables affecting chains above it

**Theoretical grounding:** The ensemble already implicitly encodes per-agent
decisions. The scaling dial makes this explicit by weighting the ensemble
prediction toward group-specific or entity-specific behavior.

### 8.3 Empirical Threshold Calibration

Compare surprisal distributions and adpression detection rates against natural
system data during the empirical validation phase:
- Earthquake frequency-magnitude distributions (Gutenberg-Richter)
- Ecological regime shift early-warning signals (Scheffer et al. 2009)
- Patent innovation surprisal rates (Youn et al. 2015)
- Speciation/extinction pulse timing (PBDB)

### 8.4 Visualization

Matplotlib figures showing:
- Prediction probability streams over time (stacked area chart per axis)
- Surprisal time series with adpression markers
- Parallel matching rate sliding window
- State collapse timeline

---

## 9. References

- Scheffer, M. et al. (2009). "Early-warning signals for critical
  transitions." Nature, 461, 53-59.
- Shannon, C. E. (1948). "A mathematical theory of communication."
  Bell System Technical Journal, 27(3), 379-423.
- Gutenberg, B. & Richter, C. F. (1944). "Frequency of earthquakes in
  California." Bulletin of the Seismological Society of America, 34(4),
  185-188.
