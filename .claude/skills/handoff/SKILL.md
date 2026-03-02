---
name: handoff
description: Generate sigma-TAP session handoff buffer. Project-specific override.
---

# sigma-TAP — Session Handoff

Project-specific handoff skill. Overrides the global handoff skill with sigma-TAP structure.

## Configuration

- **Buffer directory**: `.claude/buffer/`
- **Test command**: `python -m pytest tests/ -q --tb=no`
- **Memory file**: `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`

## Concept Map Groups

The warm layer's `concept_map` uses these groups (matching the project's theoretical structure):

```json
{
  "foundational_triad": [],
  "dialectic": [],
  "T": [],
  "A": [],
  "P": [],
  "S": [],
  "RIP": [],
  "cross_source": []
}
```

### Group Descriptions

- **foundational_triad**: Unity, Multiplicity, Unificity — the generative triad from which TAPS, RIP, and Dialectic unfold
- **dialectic**: thesis, athesis, synthesis, metathesis — the four dialectical moments
- **T** (Transvolution): How I become — L11-dominant, self-self transformation
- **A** (Anopression): How I am — L22-dominant, environmental pressure/state
- **P** (Praxis): How I act — L12-dominant, self-to-other directed action
- **S** (Syntegration): How I create — L21-dominant, other-to-self integration
- **RIP**: Recursive/Reflective, Iterative/Integrative, Preservative/Praxitive — flow function of TAPS
- **cross_source**: Mappings from external sources (Levinas, Sartre, D&G, Emery-Trist, Turchin, etc.) to the sigma-TAP framework

### Cross-Source Entry Schema

Cross-source entries use a different structure from base concept entries:

```json
{
  "id": "w:N",
  "key": "Source:ConceptName",
  "maps_to": "TAPS letter, RIP mode, L-channel, dialectic term, or 'novel'",
  "ref": "§5.XX forward note reference if applicable",
  "suggest": null,
  "see_also": ["c:N"]
}
```

## Orientation Template

The hot layer's `orientation` section for sigma-TAP includes `why_keys` for the philosophical sources:

```json
{
  "core_insight": "sigma-TAP models praxis (how agents become through action) using the L-matrix as its fundamental accounting unit. Every interaction is classified into four channels (L11=self-self, L12=self-other, L21=other-self, L22=env-env). Philosophical sources independently converge on this same 4-channel structure.",
  "practical_warning": "ODE solver needs m_cap=1e4 for explosive params. Do NOT introduce novel claims or impose standard ABM assumptions. The user cares about terminological precision and theoretical fidelity.",
  "why_keys": {
    "levinas": "Irreducible alterity grounds asymmetric cross-metathesis. Paternity as non-causal unicity.",
    "sartre": "Practico-inert = sedimented praxis (gas/fuel). Praxistatic = active transformation (engine). Praxis-process = consummation, process-praxis = consumption.",
    "dg": "Tetravalent assemblage maps to L-matrix. Conjunction (blocking/gating) vs connection (creative multiplication) is key for Stage 3C.",
    "emery": "L11/L12/L21/L22 from organizational systems theory (1965). Type V vortical = unificity/holarchy.",
    "user_framework": "Unificity.md sits ABOVE all sources: Unity-Multiplicity-Unificity is the foundational triad. The user has an original theoretical system mapped against these sources for mutual validation."
  }
}
```

## Concept Map Validation Rules

When validating concept map entries (Step 6 of global process):
- **Base system** = the `foundational_triad` + `TAPS` (T/A/P/S groups) + `RIP` + `dialectic` groups. Changes to these require `NEEDS_USER_INPUT` status, never auto-change.
- **`suggest: null`** is PREFERRED. Only flag genuine structural parallels noticed during the session.
- **Cross-source entries** use the format `"key": "Source:ConceptName"` matching the distill interpretation output.

## Forward Note References

Decision and thread `ref` fields use `§5.XX` notation from `docs/plans/2026-02-26-stage3a-two-channel-design.md`. This is the design doc containing all forward notes (§5.1 through §5.69+).

## Process

Follow the global handoff skill process (Steps 1-12), using the configuration and schemas defined here. The global skill defines the generic process; this file defines the project-specific structure.
