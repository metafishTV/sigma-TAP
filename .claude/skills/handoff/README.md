# Session Handoff — sigma-TAP

Project-specific handoff configuration for the sigma-TAP multi-agent praxis simulator.

## What This Adds

This project skill extends the global `/buffer:off` with sigma-TAP-specific structure:

- **8 concept map groups** organized by the project's theoretical framework
- **Convergence web** for inter-source linkages (tetradic structure)
- **Cross-source entry schema** for mapping external philosophical/empirical sources to TAPS/RIP/L-matrix
- **Consolidation heuristics** using sigma-TAP's established vocabulary
- **Forward note references** (`§5.XX` notation from the Stage 3A design document)

## Concept Map Groups

| Group | What It Holds | L-channel |
|-------|--------------|-----------|
| `foundational_triad` | Unity, Multiplicity, Unificity | Above TAPS |
| `dialectic` | Thesis, Athesis, Synthesis, Metathesis | Meta-frame |
| `T` (Transvolution) | How I become — evolution, involution, condensation | L11 |
| `A` (Anopression) | How I am — expression, impression, compression, etc. | L22 |
| `P` (Praxis) | How I act — projection, reflection, consumption, etc. | L12 |
| `S` (Syntegration) | How I create — integration, disintegration, preservation, etc. | L21 |
| `RIP` | Recursive/Reflective, Iterative/Integrative, Preservative/Praxitive | Flow function |
| `cross_source` | Mappings from Levinas, Sartre, D&G, Emery-Trist, Turchin, etc. | Varies |

The base system (foundational_triad + TAPS + RIP + dialectic) requires user confirmation for any changes. Cross-source entries can be added freely during distillation.

## Convergence Web

Inter-source linkages stored as tetradic entries in the warm layer:

```
thesis   ← Source A concept (w:N reference)
athesis  ← Source B concept (w:N reference)
synthesis   ← What RELATES them (shared structural ground)
metathesis  ← What EACH does independently (separate real-world functions)
```

Type tags: `[independent_convergence]`, `[complementarity]`, `[elaboration]`, `[tension]`, `[genealogy]`.

## Warm Layer Override

Warm max lines raised to **800** (global default: 500) to accommodate the concept map + convergence web backbone. These sections are non-migratable — they stay in warm permanently.

## Consolidation

The warm layer uses sigma-TAP vocabulary for compression. Established terms (unificity, L-matrix, TAPS, RIP, affordance score, metathetic transition, etc.) replace multi-word descriptions. See the SKILL.md `Consolidation Heuristics` section for the full vocabulary list and merge rules.

## Configuration

| Setting | Value |
|---------|-------|
| Buffer directory | `.claude/buffer/` |
| Test command | `python -m pytest tests/ -q --tb=no` |
| Memory file | `C:\Users\user\.claude\projects\...\memory\MEMORY.md` |
| Warm max lines | 800 |
| Forward notes | `§5.XX` from `docs/plans/2026-02-26-stage3a-two-channel-design.md` |

## How to Modify

Edit `.claude/skills/handoff/SKILL.md` directly. The concept map groups, validation rules, orientation template, and consolidation heuristics are all in that file. The global buffer:off skill reads the project skill and follows its structure.
