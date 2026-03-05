---
name: off
description: Generate sigma-TAP session handoff buffer. Project-specific override.
---

# sigma-TAP — Session Handoff v3

Project-specific handoff skill. Overrides the global `/buffer:off` with sigma-TAP structure. Counterpart to `/buffer:on`.

## Configuration

- **Buffer mode**: `project` (always — sigma-TAP uses concept map + convergence web)
- **Buffer directory**: `.claude/buffer/`
- **Test command**: `python -m pytest tests/ -q --tb=no`
- **Memory file**: `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`
- **Warm max lines**: 800 (overrides global 500)

> **Mode note**: sigma-TAP always runs in project mode. All global `/buffer:off` mode gates (Steps 3-7) fire unconditionally. The hot layer's `buffer_mode` field must be `"project"`.

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

The warm layer also has a top-level `convergence_web` section (inter-source linkages, tetradic structure):

```json
{
  "convergence_web": {
    "_meta": { "total_entries": 22, "last_validated": "2026-03-02" },
    "entries": [
      {
        "id": "cw:N",
        "thesis": { "ref": "w:X", "label": "SourceA:Concept" },
        "athesis": { "ref": "w:Y", "label": "SourceB:Concept" },
        "synthesis": "[type_tag] What RELATES them — shared structural ground (involutory)",
        "metathesis": "What EACH does independently — separate real-world functions (evolutory)"
      }
    ]
  }
}
```

The hot layer has a `convergence_web_digest` parallel to `concept_map_digest`.

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

## Warm Layer Capacity

The sigma-TAP warm layer has two large exempt sections (concept_map + convergence_web) that are not migratable because they form the structural backbone. As these grow with each distillation, effective capacity for migratable content (decisions_archive, validation_log) shrinks.

**Current override**: Warm max lines is raised to **800** for this project (global default: 500). This accommodates the concept_map + convergence_web backbone while leaving room for operational content.

If the exempt content alone approaches 800 lines, consider:
1. Archiving convergence_web entries older than 5 sessions to cold (they represent completed analysis and can be retrieved)
2. Compacting cross_source entries that haven't been referenced in 3+ sessions
3. Raising the bound further (diminishing returns beyond ~1000)

## Forward Note References

Decision and thread `ref` fields use `§5.XX` notation from `docs/plans/2026-02-26-stage3a-two-channel-design.md`. This is the design doc containing all forward notes (§5.1 through §5.69+).

## Consolidation Heuristics

When performing warm consolidation (Step 6b of global `/buffer:off`), use these sigma-TAP-specific rules.

### Established Vocabulary

These terms are defined in the concept_map and should be used without re-explanation in other entries:

- **Structural**: unity, multiplicity, unificity, L-matrix, L11/L12/L21/L22
- **Dialectic**: thesis, athesis, synthesis, metathesis
- **TAPS**: transvolution, anopression, praxis, syntegration
- **RIP**: recursive/reflective, iterative/integrative, preservative/praxitive
- **Operational**: affordance score, type-set, Jaccard, metathetic transition, TAPS signature, affordance tick, Youn ratio, sigma(Xi)

When a warm entry's description restates any of these using more words, compress to the term. The concept_map is the glossary — every other warm section should speak its language.

### Merge Candidates

Check these patterns during consolidation:

- **Cross-source convergence**: Two `cross_source` entries from different sources that both `maps_to` the same TAPS letter or L-channel → merge into single entry with both source attributions, one `maps_to`, combined `see_also`
- **Confirmed suggestions**: Entry where `suggest` was promoted to `equiv` → check if the original description is now redundant with the confirmed mapping. If so, tighten.
- **Convergence web ref updates**: If convergence web `thesis`/`athesis` refs point to entries that were merged → update the refs to the surviving entry ID

### Compression Signals

An entry is ready for description tightening when ALL of:
- Unchanged for 3+ sessions (check `validation_log`)
- Description uses >20 words
- All key terms in the description are defined elsewhere in concept_map
- No pending `suggest` (mapping is stable)

### Conservation Interaction

Consolidation reduces warm line count, which delays conservation migration (Step 9). This is intentional — consolidation keeps content in warm where it's accessible, rather than pushing it to cold where it requires explicit retrieval. The warm layer should get denser, not migrate prematurely.

## Process

Follow the global `/buffer:off` process (Steps 1-14), using the configuration and schemas defined here. The global skill defines the generic process; this file defines the project-specific structure.

**Key overrides from global defaults**:
- `buffer_mode`: always `"project"` — never prompt for mode selection
- Warm max lines: 800 (global default: 500)
- Concept map groups: 8 groups defined above
- Convergence web: top-level warm section with tetradic entry schema
- Conservation: concept_map + convergence_web are exempt from warm→cold migration
