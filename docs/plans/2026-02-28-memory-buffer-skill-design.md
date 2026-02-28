# Memory Buffer Skill — Design Document

**Date**: 2026-02-28
**Status**: Approved by user, ready for implementation
**Purpose**: Automate session handoff with a structured, git-versioned knowledge buffer

---

## 1. Problem

Each session reset loses context. With 50+ forward notes, a complex theoretical framework, and specific architectural knowledge, the next Claude instance spends 20-30% of its context window reconstructing where things stand. Stage 3B spans 7 phases across multiple sessions — without structured handoff, every session starts with re-reading prose and re-discovering the same gaps.

## 2. Solution

Two slash commands — `/handoff` and `/resume` — that maintain a structured JSON buffer file. The buffer captures session state, decisions, open threads, and a concept map of the project's theoretical framework. Git-tracked for history.

## 3. Architecture

### 3.1 File Locations

| Component | Path | Purpose |
|---|---|---|
| `/handoff` skill | `.claude/skills/handoff/SKILL.md` | End-of-session buffer generation |
| `/resume` skill | `.claude/skills/resume/SKILL.md` | Start-of-session context reconstruction |
| Buffer file | `.claude/buffer/handoff.json` | The structured knowledge buffer |
| Codex | Embedded in buffer `codex` section | Decoder key for compact notation |

All project-level, committed to the sigma-TAP repo.

### 3.2 Commands

**`/handoff`** — Invoked at session end:
1. Gather session metadata (date, commit, branch, files modified, test status)
2. Summarize active work state (current phase, completed, in-progress, blocked)
3. Log key decisions made this session with rationale
4. List open threads (unresolved questions, next steps)
5. Run concept map validation (diff against existing entries)
6. Generate compact summary with codex
7. Write to `.claude/buffer/handoff.json`
8. Commit to git

**`/resume`** — Invoked at session start:
1. Read `.claude/buffer/handoff.json`
2. Parse and surface: what was the last session about? What's in progress? What decisions were made?
3. Flag any validation warnings from the concept map
4. Present open threads for immediate attention
5. Confirm context reconstruction is complete

### 3.3 Buffer Schema

```json
{
  "schema_version": 1,

  "session_meta": {
    "date": "YYYY-MM-DD",
    "commit": "<hash>",
    "branch": "<branch>",
    "files_modified": ["<paths>"],
    "tests": "<pass_count>/<total> passing"
  },

  "active_work": {
    "current_phase": "<description>",
    "completed_this_session": ["<items>"],
    "in_progress": "<item or null>",
    "blocked_by": "<description or null>"
  },

  "decisions": [
    {
      "what": "<description>",
      "chose": "<choice>",
      "why": "<rationale>",
      "ref": "<section reference or null>"
    }
  ],

  "open_threads": [
    {
      "thread": "<description>",
      "status": "<noted|deferred|blocked|needs-user-input>",
      "ref": "<section reference or null>"
    }
  ],

  "concept_map": {
    "_meta": {
      "base_system": "TAPS + RIP + Dialectic (thesis/athesis/synthesis/metathesis)",
      "version": 1,
      "last_validated": "YYYY-MM-DD"
    },
    "dialectic": { "...see §4 below..." },
    "T": { "...transvolution subterms..." },
    "A": { "...anopression subterms..." },
    "P": { "...praxis subterms..." },
    "S": { "...syntegration subterms..." },
    "RIP": { "...flow function subterms..." },
    "cross_source": { "...cross-framework mappings..." }
  },

  "validation_log": [
    {
      "check": "<what was checked>",
      "status": "PASS|CHANGED|NEW|NEEDS_USER_INPUT",
      "detail": "<description or null>",
      "session": "YYYY-MM-DD"
    }
  ],

  "compact_summary": "<compressed notation string>",

  "codex": {
    "version": 1,
    "encoding": { "<abbrev>": "<meaning>" },
    "rules": ["<decoding rules>"]
  }
}
```

## 4. Concept Map — Full Structure

The concept map has three coordinate bases (TAPS, RIP, Dialectic) with everything else mapped relative to them.

### 4.1 Entry Format

Each concept entry has:

| Field | Required | Purpose |
|---|---|---|
| `base` | Yes | Definition in the project's own framework |
| `equiv` | No (nullable) | Confirmed equivalences in other frameworks |
| `collision` | No (nullable) | Where the same word means something different |
| `suggest` | No (nullable, preferred null) | Claude-flagged connections not yet confirmed by user |

**`suggest: null` is the preferred state.** Future instances should NOT feel pressure to populate this field. Only genuine structural parallels noticed during review belong here. The user must confirm any `suggest` entry before it can become an `equiv`.

### 4.2 Dialectic (Top-Level Structural Frame)

```json
"dialectic": {
  "thesis": {
    "base": "positing / the thetic moment",
    "maps_to_TAPS": "T (transvolution)",
    "time_mode": "π (thetic, L11, fastest)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "athesis": {
    "base": "counter-positing (apposite, NOT oppositional)",
    "maps_to_TAPS": "A (anopression)",
    "time_mode": "α (athetic, L12)",
    "equiv": null,
    "collision": "≠ antithesis (antithesis = counter-athesis, a different thing entirely)",
    "suggest": null
  },
  "synthesis": {
    "base": "combining / integration",
    "maps_to_TAPS": "S (syntegration)",
    "time_mode": "σ (synthetic, L21)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "metathesis": {
    "base": "context window / before-after frame / enables synchronicity of the other three. Without metathesis, thesis-athesis-synthesis remain serial and diachronic",
    "maps_to_TAPS": "the M-process framing TAPS itself",
    "time_mode": "Π (metathetic, L22, slowest)",
    "equiv": null,
    "collision": "≠ linguistic metathesis (phoneme transposition)",
    "suggest": null
  }
}
```

### 4.3 T-Group (Transvolution)

```json
"T": {
  "_base": "how-I-become / becoming-axis / thesis",
  "evolution": {
    "base": "unfolding / multiplicity / explicate direction",
    "equiv": ["Bohm:explicate order", "extensivity", "exteriority"],
    "collision": "≠ biological evolution (no telos implied)",
    "suggest": null
  },
  "involution": {
    "base": "enfolding / unity / implicate direction",
    "equiv": ["Bohm:implicate order", "intensivity", "interiority"],
    "collision": null,
    "suggest": null
  },
  "condensation": {
    "base": "tendency toward actualized structure / unificity",
    "equiv": ["transvolution consummated", "third irreducible modality"],
    "collision": "≠ condension (condension = involution specifically, not the consummation)",
    "suggest": null
  },
  "condension": {
    "base": "specific vector of involution / densification / enfolding",
    "equiv": ["involution as directed process"],
    "collision": "≠ condensation (which is the consummation of evolution+involution)",
    "suggest": null
  },
  "transvolution_constraints": {
    "base": "causal character of directional pairs",
    "subterms": {
      "evolution_involution": "timelike (sequential, causal, irreversible)",
      "expansion_condension": "spacelike (simultaneous, structural, reversible)",
      "rarefaction_condensation": "lightlike (propagation at boundary of causal reach)"
    },
    "ref": "§5.39"
  }
}
```

### 4.4 A-Group (Anopression)

```json
"A": {
  "_base": "how-I-am / being-axis / athesis",
  "_dipoles": {
    "anopression": "upward/outward pressing (decreasing difficulty, situatedness)",
    "anapression": "downward/inward pressing (increasing difficulty, desituation)"
  },
  "expression": {
    "base": "outward manifestation / pressing-out",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "impression": {
    "base": "inward reception / pressing-in",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "adpression": {
    "base": "pressing-toward / junction of ano+ana",
    "equiv": ["emergence (ano+ana combined)"],
    "collision": null,
    "suggest": null
  },
  "compression": {
    "base": "pressing-together / densification",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "depression": {
    "base": "pressing-down / suppression of possibility",
    "equiv": null,
    "collision": "≠ clinical depression (though structural parallel may exist)",
    "suggest": null
  },
  "suppression": {
    "base": "pressing-under / holding beneath surface",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "oppression": {
    "base": "pressing-against / resistance from outside",
    "equiv": ["Sartre:counter-finality (partial)"],
    "collision": "≠ political oppression exclusively (though it includes it)",
    "suggest": null
  },
  "_chain_order": "anopression ← expression ← impression ← adpression ↔ compression → depression → suppression → oppression → anapression",
  "_note": "α parameter = anapressive phase constant (L12 athetic time). σ parameter = anopressive phase constant (L21 synthetic time). See §5.45"
}
```

### 4.5 P-Group (Praxis)

```json
"P": {
  "_base": "how-I-act / action-axis / praxis proper",
  "projection": {
    "base": "throwing-forward / externalizing action",
    "equiv": ["Sartre:projet"],
    "collision": null,
    "suggest": null
  },
  "reflection": {
    "base": "bending-back / internalizing observation",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "consumption": {
    "base": "praxis consuming syntegrative structures / using",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "consummation": {
    "base": "syntegration completing praxitive structures / doing",
    "equiv": null,
    "collision": "≠ consumption (consumption = using structure; consummation = completing through absorption)",
    "suggest": null
  },
  "habitation": {
    "base": "praxis as active intentional dwelling (protonomic register)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "behabitation": {
    "base": "hexis as being-qua-habitation (autonomic register)",
    "equiv": ["Austin:behabitives (extended)"],
    "collision": null,
    "suggest": null
  },
  "_praxis_definition": "the process of projection & reflection in consumptive consummation",
  "_conservation": "praxis is always conserved (via consummation in laminar flow). See §5.38",
  "_practico_inert": "gas/fuel — sedimented past praxis storing potential energy. REVERSED from §5.24. See §5.43",
  "_praxistatic": "engine — active interface transforming stored potential into directed work. See §5.43"
}
```

### 4.6 S-Group (Syntegration)

```json
"S": {
  "_base": "how-I-create / creation-axis / synthesis at TAPS level",
  "integration": {
    "base": "bringing-together into whole / L21-direction (env→system)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "disintegration": {
    "base": "breaking-apart / redistribution of elements",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "preservation": {
    "base": "holding-intact / maintaining existing structure",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "synthesis": {
    "base": "creating-new-from-combined / productive combination",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "fusion": {
    "base": "convergence toward shared hexitive ground (group level)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "diffusion": {
    "base": "divergence into protonomic exploration (group level)",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "_conservation": "syntegration is always consumed (via consumption in turbulent flow). See §5.38",
  "_law": "minimal praxis for maximal syntegration = variational principle = principle of least action. See §5.50"
}
```

### 4.7 RIP (Flow Function)

```json
"RIP": {
  "_base": "direction through TAPS space / flow function. TAP=rate, TAPS=position, RIP=direction",
  "recursive": {
    "base": "self-entry, depth (R-mode 1)",
    "ontological_register": "Real",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "reflective": {
    "base": "self-observation, distance (R-mode 2)",
    "ontological_register": "Real",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "iterative": {
    "base": "sequential stepping, repetition with displacement (I-mode 1)",
    "ontological_register": "Ego",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "integrative": {
    "base": "unifying, composition (I-mode 2)",
    "ontological_register": "Ego",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "preservative": {
    "base": "maintaining structure (P-mode 1)",
    "ontological_register": "Probability",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "praxitive": {
    "base": "creating structure (P-mode 2)",
    "ontological_register": "Probability",
    "equiv": null,
    "collision": null,
    "suggest": null
  },
  "_combinatorics": "8 mode configs (2^3) × 6 orderings (3!) = 48 base states. Power-law nesting: R^(I^P) ≠ (R^I)^P. See §5.30",
  "_death": "collapse of RIP flow function / mode-lock (mode-flip frequency → 0). Not a specific config.",
  "_falsification": "Stage 3X-RIP with F1-F8 regime. Must pass F1, F6, F8 to retain. See §5.30"
}
```

### 4.8 Cross-Source Mappings

```json
"cross_source": {
  "Sartre:totalization": {
    "maps_to": "naive/undifferentiated metathesis",
    "ref": "§5.18",
    "suggest": null
  },
  "Sartre:totalization_detotalization": {
    "maps_to": "absorptive/novel cross-metathesis",
    "ref": "§5.18",
    "suggest": null
  },
  "Sartre:practico-inert": {
    "maps_to": "gas/fuel (sedimented past praxis)",
    "ref": "§5.43",
    "NOTE": "REVERSED from §5.24"
  },
  "Sartre:counter-finality": {
    "maps_to": "counter-thesis typology (participatory, not oppositional)",
    "ref": "§5.33",
    "suggest": null
  },
  "Sartre:praxis-process": {
    "maps_to": "phenomenological register",
    "ref": "§5.18a",
    "suggest": null
  },
  "Bohm:explicate_order": {
    "maps_to": "extensive metathesis / evolution / exteriority",
    "ref": "§5.18a",
    "suggest": null
  },
  "Bohm:implicate_order": {
    "maps_to": "intensive metathesis / involution / interiority",
    "ref": "§5.18a",
    "suggest": null
  },
  "Bohm:rheomode": {
    "maps_to": "dividuation (constitutive splitting of project/realization)",
    "ref": "§5.23",
    "suggest": null
  },
  "Levinas:alterity": {
    "maps_to": "irreducible otherness in cross-metathesis",
    "ref": "§5.10",
    "suggest": null
  },
  "Levinas:unicity_paternity": {
    "maps_to": "unificity (user's term) — non-causal unicity",
    "ref": null,
    "suggest": null
  },
  "DG:rhizomatic": {
    "maps_to": "alliance / horizontal / any-to-any (slime mold)",
    "ref": "§5.10",
    "suggest": null
  },
  "DG:arborescent": {
    "maps_to": "filiation / vertical / tree / irreversible",
    "ref": "§5.10",
    "suggest": null
  },
  "Austin:behabitives": {
    "maps_to": "behabitation (hexis as being-qua-habitation)",
    "ref": "§5.47",
    "suggest": null
  },
  "Fermat:least_time": {
    "maps_to": "conservation law (minimal praxis for maximal syntegration)",
    "ref": "§5.50",
    "suggest": null
  },
  "Feynman:path_integral": {
    "maps_to": "conservation law — all paths explored, constructive interference on stationary action",
    "ref": "§5.50",
    "suggest": null
  }
}
```

## 5. Validation Check

Runs during `/handoff`:

1. **Scan decisions** from this session
2. **For each decision touching a concept mapping**: check against existing concept_map
3. **If mapping changed** → update entry, log in validation_log as CHANGED
4. **If new concept introduced** → add entry, log as NEW
5. **If existing `suggest` confirmed by user** → promote to `equiv`, log as PROMOTED
6. **If base system itself questioned** → flag as NEEDS_USER_INPUT, do not auto-change
7. **If nothing to flag** → log PASS

Validation statuses: `PASS | CHANGED | NEW | PROMOTED | NEEDS_USER_INPUT`

## 6. Compact Summary + Codex (Approach B Element)

### 6.1 Compact Summary

A single dense line in abbreviated notation. Example:

```
3B0:handoff-skill|σ-gap:met.py|τ4:self/pair/grp/ctx|PI=gas,PX=eng|§5.1-50|dial:T-A-S-M
```

### 6.2 Codex

Embedded decoder key. Version-tracked so if encoding evolves, decoder evolves with it.

```json
"codex": {
  "version": 1,
  "encoding": {
    "3B0": "Stage 3B Phase 0",
    "σ-gap": "sigma feedback not wired",
    "met.py": "metathetic.py",
    "τ4": "four-level trust",
    "PI": "practico-inert",
    "PX": "praxistatic",
    "§X.Y": "design doc forward note section",
    "dial": "dialectic framework",
    "T-A-S-M": "thesis-athesis-synthesis-metathesis"
  },
  "rules": [
    "| separates topics",
    "= marks equivalence or assignment",
    ": marks containment or specification",
    "→ marks directional mapping",
    "≠ marks collision/non-equivalence",
    "/ separates alternatives or levels",
    "Abbrevs: T/A/P/S = TAPS letters, R/I/P = RIP positions",
    "Lowercase = instances, Uppercase = types/categories"
  ]
}
```

## 7. Relationship to MEMORY.md

The buffer does NOT replace MEMORY.md. They serve different purposes:

| | MEMORY.md | handoff.json |
|---|---|---|
| **Scope** | Entire project history | Single session delta |
| **Audience** | Human-readable + Claude | Claude-optimized (structured JSON) |
| **Persistence** | Accumulates across all sessions | Overwritten each `/handoff` |
| **Content** | What the project IS | What CHANGED this session |
| **Concept map** | Prose definitions | Structured lookup with validation |

MEMORY.md should still be updated when major persistent changes occur. The buffer captures the session-specific complement.

## 8. Success Criteria

1. `/handoff` generates valid JSON in < 30 seconds of guided interaction
2. `/resume` reconstructs context in < 10% of context window (vs current ~20-30%)
3. Concept map catches at least 1 mapping change per session where applicable
4. Compact summary is decodable by a fresh instance using only the codex
5. No information loss compared to current MEMORY.md approach
6. Git-tracked for full history

## 9. Scope Boundaries (YAGNI)

**In scope**: The 6 sections above, two slash commands, git integration.

**Out of scope** (explicitly deferred):
- SessionStart/SessionEnd hooks (can add later if manual invocation proves unreliable)
- Full knowledge graph / ontology tooling
- Automatic MEMORY.md updates from buffer
- Cross-project portability (project-specific for now)
- UI/visualization of concept map
