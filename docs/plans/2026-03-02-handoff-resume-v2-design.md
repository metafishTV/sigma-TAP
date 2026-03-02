# Handoff/Resume Buffer v2 — Design Document

> **Date**: 2026-03-02
> **Status**: Approved
> **Supersedes**: `2026-02-28-memory-buffer-skill-design.md` (v1)
> **Design principle**: Maximal syntegration for minimal praxis (conservation law)

## Problem Statement

The v1 buffer system (`handoff.json`, 1056 lines / ~63KB) suffers from:

1. **Context loss** — sessions start cold; resume reads everything or nothing
2. **Buffer bloat** — cumulative sections (decisions, validation_log, dialogue_trace) grow unbounded
3. **Staleness** — buffer only written at explicit `/handoff`, easily forgotten
4. **Integration gaps** — distill skill output not wired to buffer updates
5. **No two-file architecture** — skills live at user level only, not per-project
6. **No selective reconstruction** — resume loads entire buffer regardless of need
7. **No graceful degradation** — context-out mid-session loses everything
8. **Codex overhead** — ~100 lines of encoding table for a parallel representation of the natural summary
9. **Invisible concept map evolution** — changes buried in monolithic file

## Solution: Conservation-Law Refactor

### Design Principle

Every handoff enforces bounded layer sizes. Content migrates downward (hot → warm → cold → tower) when bounds are exceeded. Resume reads upward selectively (hot always, warm/cold only when pointed to). The system conserves attention by never auto-loading more than ~200 lines.

---

## 1. Architecture

### Two-File Skill Model

Following the distill skill pattern:

| Level | Location | Content |
|-------|----------|---------|
| **Global** | `~/.claude/skills/handoff/SKILL.md` | Generic process, schema spec, layer management |
| **Project** | `<repo>/.claude/skills/handoff/SKILL.md` | Project-specific structure, concept map schema, terminology |
| **Global** | `~/.claude/skills/resume/SKILL.md` | Generic reconstruction, selective loading |
| **Project** | `<repo>/.claude/skills/resume/SKILL.md` | Project-specific resume order, flagged concepts |

Global skills contain the process. Project skills contain the structure. Global checks for project skill and defers if found.

### Three-Layer Buffer

```
handoff.json          (hot)   ≤ 200 lines  — always loaded at resume
handoff-warm.json     (warm)  ≤ 500 lines  — loaded selectively via pointers
handoff-cold.json     (cold)  ≤ 500 lines  — on-demand only
handoff-tower-NNN-*.json      — sealed archive, never auto-read
```

All files live in `.claude/buffer/`.

### Pointer-Index System

Hot-layer entries contain `"see"` fields pointing to warm-layer entries by stable ID:

```json
{
  "thread": "Trust metric design",
  "status": "noted",
  "see": ["w:34", "w:35"]
}
```

Warm-layer entries may contain `"see_also"` fields pointing to cold:

```json
{
  "id": "w:34",
  "key": "Sartre:practico-inert",
  "maps_to": "P.practico_inert (gas/fuel)",
  "see_also": ["c:7"]
}
```

**Follow-pointer algorithm** (must be explicit in skill):

1. Start from hot-layer `"see"` references
2. Read only the referenced entries from warm (not full file)
3. If warm entry has `"see_also"`, follow into cold
4. **Max cascade depth: 3** (hot → warm → cold, no further)
5. **Visited set**: track followed IDs, skip if already visited (prevents circular refs)
6. **Broken ref handling**: if target ID not found, log warning and skip gracefully
7. **Tombstone handling**: if target has `"archived_to"`, note for user but do not auto-read tower

**Important**: Pointers are an attention filter, not a byte-level optimization. The real savings come from read frequency — ~200 lines typical vs ~1056 lines every time.

### Periodic Full Rescan

```json
"sessions_since_full_scan": 3,
"full_scan_threshold": 5
```

- Counter increments each handoff
- At resume, if counter >= threshold, prompt user: "It's been N sessions since a full buffer scan. Would you like me to do a complete review of warm + cold layers?"
- If yes: read all layers, surface stale/orphaned entries, reset counter
- If no: continue with selective loading, increment counter
- Threshold is configurable (default: 5)

---

## 2. Buffer Schema v2

### Hot Layer (`handoff.json`, ≤ 200 lines)

```json
{
  "schema_version": 2,

  "session_meta": {
    "date": "2026-03-02",
    "commit": "526b5e3",
    "branch": "main",
    "files_modified": ["file1.py", "file2.py"],
    "tests": "295 passed, 0 failed"
  },

  "sessions_since_full_scan": 0,
  "full_scan_threshold": 5,

  "orientation": {
    "core_insight": "sigma-TAP models adjacent possible growth with sigma feedback...",
    "practical_warning": "ODE solver needs m_cap=1e4 for explosive params...",
    "why_keys": {
      "levinas": "Irreducible alterity grounds asymmetric cross-metathesis",
      "sartre": "Practico-inert = sedimented praxis storing potential energy",
      "emery": "L-matrix = event channel framework for directive correlation"
    }
  },

  "active_work": {
    "current_phase": "Post-3B reflection, handoff v2 implementation",
    "completed_this_session": ["Design doc for handoff v2"],
    "in_progress": "Implementation plan for handoff v2",
    "blocked_by": null,
    "next_action": "Execute implementation plan"
  },

  "open_threads": [
    {
      "thread": "Stage 3C planning",
      "status": "deferred",
      "ref": "TASKS.md",
      "see": ["w:60"]
    }
  ],

  "recent_decisions": [
    {
      "what": "Buffer architecture",
      "chose": "Three-layer conservation-law refactor",
      "why": "Bounded growth, selective loading, user sovereignty",
      "session": "2026-03-02",
      "see": ["w:61"]
    }
  ],

  "instance_notes": {
    "from": "instance-N",
    "to": "instance-N+1",
    "remarks": "...",
    "open_questions": ["..."]
  },

  "concept_map_digest": {
    "_meta": {
      "total_entries": 77,
      "last_validated": "2026-03-02"
    },
    "recent_changes": [
      { "id": "w:34", "status": "NEW", "summary": "Sartre:practico-inert mapped to P.practico_inert" }
    ],
    "flagged": ["w:12", "w:45"]
  },

  "natural_summary": "Stage 3B complete with 345 tests. Currently implementing handoff/resume v2 with three-layer buffer architecture. Levinas and Sartre integration mapped through 72 cross-source concepts. Next major milestone is Stage 3C planning after buffer infrastructure is solid."
}
```

### Warm Layer (`handoff-warm.json`, ≤ 500 lines)

```json
{
  "concept_map": {
    "foundational_triad": [
      { "id": "w:1", "term": "Unity", "equiv": "holism(lateral)", "suggest": null },
      { "id": "w:2", "term": "Multiplicity", "equiv": "hierarchy(vertical)", "suggest": null },
      { "id": "w:3", "term": "Unificity", "equiv": "holarchy(vortical)", "suggest": null }
    ],
    "dialectic": [
      { "id": "w:4", "term": "thesis", "equiv": "L11 self-self" },
      { "id": "w:5", "term": "antithesis", "equiv": "L12+L21 self-other" }
    ],
    "T": [],
    "A": [],
    "P": [],
    "S": [],
    "RIP": [],
    "cross_source": [
      {
        "id": "w:34",
        "key": "Sartre:practico-inert",
        "maps_to": "P.practico_inert (gas/fuel)",
        "ref": "section-5.43",
        "suggest": null,
        "see_also": ["c:7"]
      }
    ]
  },

  "decisions_archive": [
    {
      "id": "w:60",
      "what": "Stage 3B scope",
      "chose": "7-phase plan",
      "why": "Incremental, testable phases",
      "session": "2026-02-26"
    }
  ],

  "validation_log": [
    {
      "id": "w:70",
      "check": "distill: Emery-Trist-1965",
      "status": "NEW",
      "detail": "8 concepts mapped, 4 integration points",
      "session": "2026-03-01"
    }
  ]
}
```

### Cold Layer (`handoff-cold.json`, ≤ 500 lines)

```json
{
  "dialogue_trace": {
    "sessions": [
      {
        "id": "c:1",
        "session": "2026-02-26 Stage3B kickoff",
        "arc": "Established 7-phase plan for Stage 3B...",
        "key_moments": ["..."]
      }
    ],
    "recurring_patterns": [
      "User values theoretical fidelity over implementation speed",
      "Concepts often crystallize through counter-example"
    ]
  },

  "superseded_mappings": [
    {
      "id": "c:20",
      "original": "practico-inert = motor/fuel",
      "replaced_by": "w:34",
      "reason": "Reversed in section-5.43",
      "session": "2026-02-28"
    }
  ],

  "archived_decisions": []
}
```

### Removed from v1

| v1 Element | v2 Replacement | Reason |
|-----------|---------------|--------|
| `compact_summary` | `natural_summary` | Plain language, no encoding overhead |
| `codex` (46 abbreviations, ~100 lines) | Removed entirely | Net-negative: maintenance cost > compression benefit |
| Monolithic `decisions[]` in hot | Split: `recent_decisions` (hot) + `decisions_archive` (warm) | Only recent decisions need immediate context |
| Unbounded `dialogue_trace` | Moved to cold | Historical trace rarely needed at resume |

---

## 3. Cold-Layer Archival & Towers

### Trigger

When cold layer exceeds 500 lines, the handoff process triggers an archival questionnaire instead of silently growing.

### Archival Questionnaire (3 steps)

**Step 1 — Full scan + dependency map**:
- Read entire cold layer
- For each entry, compute nesting depth (how many other entries reference it)
- Present to user:

```
Cold layer at 520/500 lines. Archival needed.

Entry dependency map:
  c:1  (depth: 0)  Session trace: Stage3B kickoff
  c:7  (depth: 2)  Superseded: practico-inert reversal  ← referenced by w:34, c:20
  c:20 (depth: 1)  Superseded mapping  ← referenced by c:7
  ...

Entries with depth 0 are safe to archive (nothing references them).
Entries with depth > 0 will leave tombstones if archived.
```

**Step 2 — Pick ratio and direction**:

```
How much should be archived?
  A) 20/80 split
  B) 33/66 split
  C) 50/50 split

Which portion goes to the tower?
  → Archive the smaller portion (e.g., 20% to tower, 80% stays)
  → Archive the larger portion (e.g., 80% to tower, 20% stays)
```

User chooses BOTH the ratio AND which side gets archived. This is bidirectional — the user has full sovereignty over what stays and what goes.

**Step 3 — Pick entries**:

User selects specific entries for archival, informed by the dependency map. The dependency depth scores help the user make informed decisions about what to archive.

### Tower Files

- Named: `handoff-tower-NNN-YYYY-MM-DD.json` (e.g., `handoff-tower-001-2026-03-02.json`)
- Stored in `.claude/buffer/`
- **Sealed**: never modified after creation, never auto-read
- Contain the archived entries plus metadata:

```json
{
  "tower_id": "tower-001",
  "created": "2026-03-02",
  "created_from_session": "2026-03-02",
  "ratio": "20/80",
  "direction": "archive-larger",
  "entry_count": 12,
  "entries": [ ... ]
}
```

### Tombstones

When an entry is archived, a tombstone remains in cold:

```json
{
  "id": "c:7",
  "archived_to": "tower-001",
  "was": "Superseded: practico-inert reversal",
  "session_archived": "2026-03-02"
}
```

- Tombstones are lightweight (~3 lines vs original entry)
- Enable follow-pointer to detect archived content without reading the tower
- User can request retrieval: pull entry back from tower to cold

### Retrieval

If a pointer leads to a tombstone and the user needs the content:
1. Note the `archived_to` tower ID
2. Ask user: "This entry was archived to tower-001. Should I retrieve it?"
3. If yes: read tower file, extract entry, restore to cold, remove tombstone
4. Tower file itself is NOT modified (append-only seal) — but cold gets the restored entry

---

## 4. Updated Handoff Process (10 steps)

### Step 1: Read existing hot layer
Read `handoff.json`. If missing or schema_version < 2, trigger migration (see Section 7).

### Step 2: Gather session metadata
Same as v1: git commit, branch, files modified, test status.

### Step 3: Summarize active work
Infer from conversation: phase, completed, in-progress, blocked, next action.

### Step 4: Log decisions
Review conversation for decisions. Write to `recent_decisions` (hot) with `"see"` pointers. If a decision relates to an existing warm entry, update the pointer.

### Step 5: List open threads
Identify unresolved questions, deferred items, next steps. Write to `open_threads` (hot) with `"see"` pointers to warm entries where relevant.

### Step 6: Validate concept map
Read warm layer's `concept_map`. For each decision from Step 4:
- If mapping **changed**: update warm entry, add to hot `concept_map_digest.recent_changes` as `CHANGED`
- If **new concept**: add warm entry with new `w:N` ID, add to digest as `NEW`
- If **suggest confirmed** by user: promote to equiv, log as `PROMOTED`
- If **base system** questioned: log as `NEEDS_USER_INPUT`, do NOT auto-change

Update `concept_map_digest._meta.total_entries` and `last_validated`.

### Step 7: Write instance notes
Personal remarks from outgoing instance. Replaces previous instance_notes entirely.

### Step 8: Conservation enforcement

```
IF hot > 200 lines:
  - Migrate oldest recent_decisions to warm decisions_archive
  - Migrate resolved open_threads to warm
  - Compact orientation if verbose
  - Re-check. If still > 200, warn user.

IF warm > 500 lines:
  - Migrate oldest decisions_archive entries to cold archived_decisions
  - Migrate oldest validation_log entries to cold
  - Re-check. If still > 500, warn user.

IF cold > 500 lines:
  - Trigger archival questionnaire (Section 3)
  - Create tower file
  - Leave tombstones
  - Reset cold to within bounds
```

### Step 9: Write all layers + increment scan counter
Write `handoff.json`, `handoff-warm.json`, `handoff-cold.json`.
Increment `sessions_since_full_scan`.
Write `natural_summary` (2-3 plain-language sentences, no encoding).

### Step 10: Commit and confirm
```bash
git add .claude/buffer/handoff.json .claude/buffer/handoff-warm.json .claude/buffer/handoff-cold.json
git commit -m "handoff: <brief description>"
```

Tell user: "Handoff buffer written and committed. The next instance can run `/resume` to reconstruct context."

---

## 5. Updated Resume Process (8 steps)

### Step 1: Read hot layer only
Read `handoff.json` (~200 lines). This is the only mandatory read.

### Step 2: Git grounding
```bash
git log --oneline <buffer_commit>..HEAD
git status
git diff --stat
```

Compare against `session_meta.commit`. Flag if stale.

### Step 3: Present session state
From hot layer only:
```
## Last Session: [date]
**Commit**: [hash] on [branch]
**Phase**: [current_phase]
**Completed**: [list]
**In Progress**: [item]
**Next Action**: [item]

## Natural Summary
[natural_summary text]
```

### Step 4: Follow flagged pointers
For each entry in `concept_map_digest.flagged` and `concept_map_digest.recent_changes`:
- Follow `"see"` pointers into warm layer (read only referenced entries)
- If warm entries have `"see_also"`, follow into cold (max depth 3)
- Present flagged/changed concepts to user

For each `open_thread` with `"see"` pointers:
- Follow into warm, present relevant context

### Step 5: Check full-scan threshold
If `sessions_since_full_scan >= full_scan_threshold`:
```
"It's been [N] sessions since a full buffer scan (threshold: [T]).
Would you like me to do a complete review of warm + cold layers?"
```
- If yes: read all layers, surface stale/orphaned entries, reset counter
- If no: continue with selective loading

### Step 6: Source material review
Read `docs/references/INDEX.md` (if exists). Present brief inventory of unread/partially reviewed/mapped sources.

### Step 7: Read MEMORY.md
Read project memory for baseline context. The buffer is the session delta; MEMORY.md is the project baseline.

### Step 8: Confirm
"Context reconstructed from [date] handoff. Ready to continue from [current_phase]."
"Shall I proceed with [next_action or first open_thread], or do you have a different priority?"

---

## 6. Distill Integration

When a distillation + interpretation is completed:

### Automatic Updates (from interpretation file)

1. **Warm concept_map.cross_source**: Add new entries with stable `w:N` IDs for each concept mapped in the interpretation's Project Significance table.

2. **Warm validation_log**: Add entry:
   ```json
   {
     "id": "w:NN",
     "check": "distill: [Source-Label]",
     "status": "NEW",
     "detail": "[N] concepts mapped, [M] integration points",
     "session": "YYYY-MM-DD"
   }
   ```

3. **Hot concept_map_digest**: Update `total_entries`, add to `recent_changes` with status `NEW`, add to `flagged` if any mapping is uncertain.

4. **Hot natural_summary**: Regenerate to mention the new distillation.

### At Next Resume

The new concept_map_digest entries automatically surface at Step 4 (follow flagged pointers), so the next instance immediately sees what was distilled and how it maps to the project framework.

---

## 7. Migration from v1

### Process

1. **Read** existing `handoff.json` (v1, ~1056 lines)

2. **Triage** into three layers:
   - **Hot**: session_meta, orientation, active_work, open_threads (most recent only), instance_notes, natural_summary (new, generated from compact_summary content)
   - **Warm**: concept_map (all entries, assign `w:N` IDs), recent decisions (last 5), validation_log
   - **Cold**: dialogue_trace (all sessions), superseded_mappings (extracted from validation_log CHANGED entries), older decisions

3. **Assign stable IDs**: Walk warm entries sequentially (`w:1` through `w:N`), cold entries (`c:1` through `c:N`)

4. **Generate pointers**: For each hot open_thread/decision, scan warm for related concept_map entries, add `"see"` references

5. **Discard**: codex section, compact_summary (replaced by natural_summary)

6. **Rename** old file: `handoff.json` → `handoff-v1-archive.json` (safety net)

7. **Write** new `handoff.json`, `handoff-warm.json`, `handoff-cold.json`

8. **User confirms** before committing

9. **Delete** `handoff-v1-archive.json` after one successful handoff/resume cycle

### Safety

- v1 archive preserved until confirmed working
- If migration fails, fall back to v1 archive
- No data is discarded during migration (only reorganized + codex removed)

---

## 8. ID Assignment Convention

### Format

- Warm: `w:N` where N is a monotonically increasing integer (e.g., `w:1`, `w:34`, `w:77`)
- Cold: `c:N` same convention (e.g., `c:1`, `c:20`)
- Tower tombstones reference tower files by name: `"archived_to": "tower-001"`

### Rules

- IDs are **never reused**. If `w:34` is deleted, the next warm entry is `w:78` (or whatever the next available number is)
- IDs are **stable across sessions**. An entry keeps its ID for its lifetime
- New IDs are assigned by reading the current max ID and incrementing
- The `concept_map_digest._meta` in hot tracks `total_entries` for quick reference

---

## 9. Configurable Thresholds

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `full_scan_threshold` | 5 | hot layer | Sessions before prompting full rescan |
| Hot max lines | 200 | handoff skill | Triggers migration to warm |
| Warm max lines | 500 | handoff skill | Triggers migration to cold |
| Cold max lines | 500 | handoff skill | Triggers archival questionnaire |
| Max cascade depth | 3 | resume skill | Pointer follow limit |
| Archival ratios | 20/80, 33/66, 50/50 | archival questionnaire | User-chosen split |

All thresholds are documented in the skill files and can be adjusted by the user.

---

## Appendix: Comparison with v1

| Aspect | v1 | v2 |
|--------|----|----|
| Files | 1 (handoff.json) | 3+ (hot/warm/cold/tower) |
| Size at resume | ~1056 lines, all loaded | ~200 lines typical |
| Growth | Unbounded cumulative | Bounded per layer |
| Codex | 46 abbreviations, ~100 lines | Removed |
| Summary | Encoded compact_summary | Plain natural_summary |
| Selective loading | None | Pointer-based |
| Archival | None (grows forever) | Tower system with user sovereignty |
| Distill integration | Manual | Automatic via interpretation file |
| Skill architecture | User-level only | Two-file (global + project) |
| Full rescan | Every resume | Periodic, user-prompted |
