# Priority 3 Design: Temporal Orientation Gate + Interpretation & Evidence Ladder

**Date**: 2026-02-25
**Branch**: `unified-integration-20260225`
**Claim policy**: Temporal gate is `exploratory`; evidence ladder is `paper-consistent extension`

---

## Overview

Two independent threads that together complete Priority 3 (T3.1, T3.2) and enrich
the metathetic agent model:

- **Thread A — Temporal orientation gate**: A five-state phase indicator on each
  MetatheticAgent encoding its relationship to its own metathetic identity over time.
- **Thread B — Interpretation blocks + evidence ladder**: Machine-readable mechanistic
  and functional interpretation for every claim, plus standardized A/B/C confidence tiers.

---

## Thread A: Temporal Orientation Gate

### Theoretical basis

A metathetic agent's relationship to its own identity is temporal. Rather than a
binary on/off gate, each agent carries three independent boolean signals — past,
present, future — whose combination yields five meaningful states.

The key insight: the future state of actualization depends on (is conditioned by)
the eminence of the past. When past-connection breaks — by sudden novelty or slow
drift — the agent cannot define what it is moving toward.

### The five states

| State | Past | Present | Future | Name | Dynamical meaning |
|-------|------|---------|--------|------|-------------------|
| 0 | off | off | off | **Annihilated** | No relational capacity remains; system ended as information source |
| 1 | ON | off | off | **Inertial** | Grown away from identity; pressure for change |
| 2 | off | ON | off | **Situated** | In-flow; productively drawing from metathesis |
| 3 | off | off | ON | **Desituated** | Novelty-shock or stagnation; undetermined future |
| 4 | ON | ON | ON | **Established** | Consummated; suspended as itself; static tension |

### Computational mapping

**State 0 — Annihilated**: Distinct from dormancy. Dormant agents preserve state
(k, type_set, history) and can be absorbed or referenced. Annihilated agents have
lost relational capacity — their knowledge is inaccessible to the present. Triggers
when an agent has been dormant beyond a `relational_decay_window` AND its types are
no longer held by any active agent (no living connection to its identity).

**State 1 — Inertial**: Agent's recent dM trajectory is diverging from what its
type-diversity would predict. Its types are becoming less distinctive (the
population has caught up). Formally: `trajectory_alignment < -threshold` OR
`type_distinctiveness` declining over recent window.

**State 2 — Situated**: Default productive state. Growth rate stable or increasing;
types remain distinctive. The metathesis is doing work for the agent.

**State 3 — Desituated**: Either `steps_since_metathesis <= novelty_window`
(post-metathesis overdetermination) or `steps_since_metathesis >= stagnation_threshold`
(origin too distant). Both produce high uncertainty about trajectory direction.

**State 4 — Established**: All three signals ON simultaneously. The agent has fully
incorporated its metathesis — past consumed as foundation, present productive, future
determined. This is the maximally stable state. Change resumes only when one signal
turns off.

### Effect on metathetic triggers

| Temporal state | Self-metathesis | Cross-metathesis (absorptive) | Cross-metathesis (novel) |
|---------------|-----------------|-------------------------------|--------------------------|
| Annihilated | impossible | impossible | impossible |
| Inertial | threshold × 0.5 (easier) | normal | normal |
| Situated | threshold × 1.5 (harder) | threshold × 1.5 | threshold × 1.5 |
| Desituated (novelty) | suppressed for `novelty_window` steps | suppressed | suppressed |
| Desituated (stagnation) | normal | threshold × 0.5 (easier) | threshold × 0.5 (easier) |
| Established | threshold × 2.0 (hardest) | threshold × 2.0 | threshold × 2.0 |

### Parameters

```
novelty_window = 5          # steps of creative immunity post-metathesis
stagnation_threshold = 50   # steps before stagnation sets in
relational_decay_window = 30 # dormancy steps before annihilation possible
trajectory_divergence_thr = -0.3  # dM alignment below which = inertial
```

### Implementation target

- New fields on `MetatheticAgent`: `steps_since_metathesis`, `last_metathesis_step`
- New computed property: `temporal_state -> int` (0-4)
- New method: `_trajectory_alignment() -> float` (compares recent dM to expected)
- Annihilation check in ensemble `_step_agents` or dedicated `_check_annihilation`
- Temporal state included in snapshots
- ~60-80 lines in `simulator/metathetic.py`

---

## Thread B: Interpretation Blocks + Evidence Ladder

### T3.1 — Mechanistic + Functional Interpretation

For each claim C1-C8, two required blocks:

1. **Mechanistic inference**: What parameters, regimes, and mechanisms explain the
   observed behavior. Grounded in the TAP equation and simulation outputs.
2. **Functional interpretation**: What role, selection pressure, or systemic
   implication this result suggests. Connects computation to theory.

### T3.2 — Evidence Ladder

Three confidence tiers:

| Tier | Label | Criteria |
|------|-------|---------|
| A | Robust | Replicated across >=2 variants AND has sensitivity/CI support |
| B | Supported | Robust in >=1 variant with sensitivity or bootstrap CIs |
| C | Exploratory | Single-configuration or novel claims requiring further validation |

### Architecture

```
config/claim_annotations.json    <-- authored interpretation blocks + tiers
        |
        +-- scripts/build_evidence_report.py   <-- validates & enriches
        |       reads: claim_annotations.json
        |              outputs/variant_comparison.csv
        |              outputs/realworld_fit.csv
        |              outputs/longrun_diagnostics_summary.json
        |              outputs/extinction_sensitivity.csv
        |              outputs/adjacency_sensitivity.csv
        |
        +-- outputs/evidence_report.json       <-- enriched report
        |
        +-- run_reporting_pipeline.py          <-- orchestrator call
```

### claim_annotations.json structure

```json
{
  "C1": {
    "mechanistic": "The TAP kernel sum C(M,i)/a^i generates qualitatively distinct regimes...",
    "functional": "Variant families serve as alternative hypotheses for growth-limiting mechanisms...",
    "evidence_tier": "A",
    "tier_justification": "Replicated across all three variants in sweep; bootstrap CIs in inferential_stats.",
    "artifacts": ["outputs/variant_comparison.csv"],
    "claim_policy_label": "paper-aligned"
  }
}
```

### Validation rules (build_evidence_report.py)

1. Every claim in CLAIMS.md must have a corresponding entry in claim_annotations.json
2. Every entry must have non-empty `mechanistic` and `functional` blocks
3. Tier assignments must be consistent with automated checks:
   - Tier A requires: claim appears in variant_comparison.csv across >=2 variants OR
     has multiple supporting artifacts across configurations
   - Tier C required for: `claim_policy_label == "exploratory"`
4. All referenced artifacts must exist on disk
5. Exit non-zero if any check fails

### Pipeline integration

Added to `run_reporting_pipeline.py` after longrun_diagnostics, before final
artifact inventory check.

---

## Claim-by-claim tier assignment (planned)

| ID | Current status | Planned tier | Rationale |
|----|---------------|-------------|-----------|
| C1 | supported | A | Three variants tested, sweep artifact |
| C2 | supported | A | Regime detection tested across all variants |
| C3 | supported | B | Heaps/Gini supported but single logistic config |
| C4 | supported | A | Three datasets x multiple models |
| C5 | exploratory | C | Novel turbulence extension |
| C6 | partial | B | Sensitivity sweep exists but single variant |
| C7 | partial | B | Sensitivity sweep exists but single variant |
| C8 | exploratory | C | Metathetic extension, single config |

---

## Tests

### Thread A tests (test_metathetic.py additions)
- `test_temporal_state_situated_default`: new agent starts situated
- `test_temporal_state_inertial_on_divergence`: declining trajectory -> inertial
- `test_temporal_state_desituated_novelty`: immediately after metathesis -> desituated
- `test_temporal_state_desituated_stagnation`: many steps without metathesis -> desituated
- `test_temporal_state_established`: fully aligned agent -> established
- `test_annihilation_distinct_from_dormancy`: annihilated agent loses relational capacity
- `test_temporal_modulates_self_meta_threshold`: inertial lowers, situated raises
- `test_temporal_state_in_snapshots`: temporal_state appears in trajectory output

### Thread B tests (test_evidence_report.py, new file)
- `test_missing_mechanistic_fails`: validation rejects missing block
- `test_missing_functional_fails`: validation rejects missing block
- `test_tier_c_required_for_exploratory`: exploratory claim with tier A fails
- `test_tier_a_requires_variant_coverage`: tier A without multi-variant fails
- `test_valid_annotations_produce_report`: complete annotations pass
- `test_missing_artifact_warns`: referenced file absent -> warning
- `test_all_claims_covered`: every CLAIMS.md entry needs annotation

---

## Non-goals

- No markdown report generation (that's downstream rendering, not P3 scope)
- No figure badge stamping (figures already have [exploratory] labels)
- No changes to existing TAP engine (tap.py, sigma_tap.py, continuous.py)
