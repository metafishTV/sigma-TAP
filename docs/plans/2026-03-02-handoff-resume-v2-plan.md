# Handoff/Resume Buffer v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic v1 handoff/resume system with a three-layer conservation-law buffer (hot/warm/cold + tower), two-file skill architecture (global + project), and distill integration.

**Architecture:** Global skills define the generic process (layer management, pointer algorithm, conservation enforcement). Project skills override with sigma-TAP-specific structure (concept map schema, terminology, resume priorities). The v1 buffer (1056 lines) gets triaged into hot (≤200), warm (≤500), and cold (≤500) layers with stable IDs and pointer references.

**Tech Stack:** Markdown skill files, JSON buffer files, git, bash

**Key references:**
- Design doc: `docs/plans/2026-03-02-handoff-resume-v2-design.md` (602 lines, all design details)
- v1 design: `docs/plans/2026-02-28-memory-buffer-skill-design.md`
- v1 handoff skill: `~/.claude/skills/handoff/SKILL.md` (147 lines, to be replaced)
- v1 resume skill: `~/.claude/skills/resume/SKILL.md` (135 lines, to be replaced)
- v1 buffer: `.claude/buffer/handoff.json` (1056 lines, to be migrated)
- Distill project skill: `<repo>/.claude/skills/distill/SKILL.md` (lines 258-292, post-distillation section)

---

### Task 1: Write Global Handoff Skill v2

**Files:**
- Create: `~/.claude/skills/handoff/SKILL.md` (replaces existing v1)

This is the GLOBAL handoff skill — it defines the generic v2 process that works for ANY project. No sigma-TAP-specific content.

**Step 1: Back up the existing v1 skill**

```bash
cp ~/.claude/skills/handoff/SKILL.md ~/.claude/skills/handoff/SKILL.v1.md
```

**Step 2: Write the new global handoff skill**

Write `~/.claude/skills/handoff/SKILL.md` with this exact structure:

```markdown
---
name: handoff
description: Generate a structured session handoff buffer. Use at end of session. Checks for project skill.
---

# Session Handoff

## Step 0: Check for Project Skill

Check if `<repo>/.claude/skills/handoff/SKILL.md` exists:
1. **If it exists**: Read and follow that skill instead — it overrides these generic instructions.
2. **If not**: Continue with the generic process below.

## Buffer Architecture

Three-layer buffer in `.claude/buffer/`:
- `handoff.json` (hot, ≤200 lines) — session state, pointers, natural summary
- `handoff-warm.json` (warm, ≤500 lines) — concept map, decision archive, validation log
- `handoff-cold.json` (cold, ≤500 lines) — dialogue trace, superseded mappings
- `handoff-tower-NNN-YYYY-MM-DD.json` — sealed archive, never auto-read

## Configurable Thresholds

| Parameter | Default |
|-----------|---------|
| Hot max lines | 200 |
| Warm max lines | 500 |
| Cold max lines | 500 |
| Full scan threshold | 5 sessions |

## Process

### 1. Read existing hot layer

Read `.claude/buffer/handoff.json`. If missing or `schema_version` < 2, check for v1 format and inform user migration is needed.

### 2. Gather session metadata

Collect by running git commands:
- Today's date
- Current commit hash (`git rev-parse --short HEAD`)
- Current branch (`git branch --show-current`)
- Files modified this session (`git diff --name-only` against session_meta.commit, or last 5 commits if new)
- Test status (run project test command, capture pass/fail line)

### 3. Summarize active work

Infer from conversation (do NOT ask the user):
- Current phase/stage
- Completed this session
- In-progress items
- Blocked items
- Next action

### 4. Log decisions

Review conversation for decisions. Write to `recent_decisions` (hot) with `"see"` pointers to related warm entries. Format:
```json
{ "what": "...", "chose": "...", "why": "...", "session": "YYYY-MM-DD", "see": ["w:N"] }
```

### 5. List open threads

Identify unresolved questions, deferred items, next steps. Write to `open_threads` (hot) with `"see"` pointers. Format:
```json
{ "thread": "...", "status": "noted|deferred|blocked|needs-user-input", "ref": "...", "see": ["w:N"] }
```

### 6. Validate concept map

Read warm layer's concept_map. For each decision from step 4:
- If mapping **changed**: update warm entry, add to hot `concept_map_digest.recent_changes` as `CHANGED`
- If **new concept**: add warm entry with new `w:N` ID (increment from current max), add to digest as `NEW`
- If **suggest confirmed** by user: promote to equiv, log as `PROMOTED`
- If **base system** questioned: log as `NEEDS_USER_INPUT`, do NOT auto-change

Update `concept_map_digest._meta.total_entries` and `last_validated`.

**IMPORTANT**: `suggest: null` is the PREFERRED state. Do NOT populate suggest fields unless a genuine structural parallel was noticed.

### 7. Write instance notes

Write `instance_notes` — personal remarks to the next instance:
- **remarks**: What you learned about the user, codebase, or theory. Warnings, tips, surprises.
- **open_questions**: Questions you didn't get to ask. Be honest — flag confusion, forced mappings.

### 8. Write natural summary

Write 2-3 plain-language sentences capturing current project state, what happened this session, and what comes next. No encoding, no abbreviations.

### 9. Conservation enforcement

```
IF hot > 200 lines:
  - Migrate oldest recent_decisions to warm decisions_archive (assign w:N IDs)
  - Migrate resolved open_threads to warm
  - Compact orientation if verbose
  - If still > 200: warn user

IF warm > 500 lines:
  - Migrate oldest decisions_archive to cold archived_decisions (assign c:N IDs)
  - Migrate oldest validation_log to cold
  - If still > 500: warn user

IF cold > 500 lines:
  - Trigger archival questionnaire:
    Step A: Full scan + dependency map (nesting depth per entry)
    Step B: User picks ratio (20/80, 33/66, 50/50) AND direction (which side archives)
    Step C: User picks specific entries for archival
  - Create tower file: handoff-tower-NNN-YYYY-MM-DD.json (sealed)
  - Leave tombstones in cold: { "id": "c:N", "archived_to": "tower-NNN", "was": "...", "session_archived": "..." }
```

### 10. Write all layers

Write `handoff.json`, `handoff-warm.json`, `handoff-cold.json`.
Increment `sessions_since_full_scan`.

### 11. Commit

```bash
git add .claude/buffer/handoff.json .claude/buffer/handoff-warm.json .claude/buffer/handoff-cold.json
git commit -m "handoff: <brief description>"
```

### 12. Confirm

Tell user: "Handoff buffer written and committed. The next instance can run `/resume` to reconstruct context."

## Hot Layer Schema

```json
{
  "schema_version": 2,
  "session_meta": { "date", "commit", "branch", "files_modified", "tests" },
  "sessions_since_full_scan": 0,
  "full_scan_threshold": 5,
  "orientation": { "core_insight", "practical_warning" },
  "active_work": { "current_phase", "completed_this_session", "in_progress", "blocked_by", "next_action" },
  "open_threads": [{ "thread", "status", "ref", "see": [] }],
  "recent_decisions": [{ "what", "chose", "why", "session", "see": [] }],
  "instance_notes": { "from", "to", "remarks", "open_questions" },
  "concept_map_digest": { "_meta": { "total_entries", "last_validated" }, "recent_changes": [], "flagged": [] },
  "natural_summary": "..."
}
```

## Warm Layer Schema

```json
{
  "concept_map": { "<groups>": [{ "id": "w:N", "term", "equiv", "suggest" }] },
  "decisions_archive": [{ "id": "w:N", "what", "chose", "why", "session" }],
  "validation_log": [{ "id": "w:N", "check", "status", "detail", "session" }]
}
```

## Cold Layer Schema

```json
{
  "dialogue_trace": { "sessions": [{ "id": "c:N", "session", "arc", "key_moments" }], "recurring_patterns": [] },
  "superseded_mappings": [{ "id": "c:N", "original", "replaced_by", "reason", "session" }],
  "archived_decisions": [{ "id": "c:N" }]
}
```

## ID Assignment

- Warm: `w:N`, Cold: `c:N` — monotonically increasing, never reused
- New IDs: read current max, increment
- Tombstones reference tower files: `"archived_to": "tower-NNN"`

## Cumulative Sections (APPEND only)

- `decisions_archive` (warm), `validation_log` (warm), `dialogue_trace.sessions` (cold), `recurring_patterns` (cold), `concept_map` (warm), `superseded_mappings` (cold)

## Replace-Each-Session Sections

- `session_meta`, `active_work`, `open_threads`, `recent_decisions`, `instance_notes`, `natural_summary`, `concept_map_digest`
```

**Step 3: Verify the file was written correctly**

Read back `~/.claude/skills/handoff/SKILL.md` and confirm:
- Frontmatter has `name: handoff`
- Step 0 checks for project skill
- All 12 process steps present
- Three schema sections (hot/warm/cold)
- Conservation enforcement with archival questionnaire
- No project-specific content (no "sigma-TAP", no "TAPS", no "L-matrix")

**Step 4: Commit**

```bash
git -C ~/.claude add skills/handoff/SKILL.md skills/handoff/SKILL.v1.md
```

Note: The user-level `~/.claude` may not be a git repo. If so, skip the commit — the file is written and that's sufficient.

---

### Task 2: Write Project Handoff Skill v2

**Files:**
- Create: `<repo>/.claude/skills/handoff/SKILL.md`

This is the PROJECT-SPECIFIC handoff skill for sigma-TAP. It overrides the global skill with sigma-TAP structure.

**Step 1: Check target directory exists**

```bash
ls "<repo>/.claude/skills/handoff/" 2>/dev/null || mkdir -p "<repo>/.claude/skills/handoff/"
```

(Replace `<repo>` with `C:\Users\user\Documents\New folder\sigma-TAP-repo` throughout.)

**Step 2: Write the project handoff skill**

Write `<repo>/.claude/skills/handoff/SKILL.md` with this exact structure:

```markdown
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

## Orientation Template

The hot layer's `orientation` section for sigma-TAP includes `why_keys` for the philosophical sources:

```json
{
  "core_insight": "...",
  "practical_warning": "...",
  "why_keys": {
    "levinas": "...",
    "sartre": "...",
    "emery": "...",
    "dg": "...",
    "user_framework": "..."
  }
}
```

## Concept Map Validation Rules

When validating concept map entries (Step 6 of global process):
- **Base system** = the foundational_triad + TAPS + RIP + Dialectic groups. Changes to these require `NEEDS_USER_INPUT`, never auto-change.
- **suggest: null** is PREFERRED. Only flag genuine structural parallels noticed during the session.
- **Cross-source entries** use the format: `"key": "Source:ConceptName"` matching the distill interpretation output.

## Forward Note References

Decision and thread references use `§5.XX` notation from `docs/plans/2026-02-26-stage3a-two-channel-design.md`.

## Process

Follow the global handoff skill process (Steps 1-12), using the configuration and schemas defined here. The global skill defines the generic process; this file defines the project-specific structure.
```

**Step 3: Verify**

Read back the file. Confirm:
- References sigma-TAP specific concepts (concept map groups, orientation why_keys, forward note format)
- Defers to global skill for the process steps
- Contains test command and memory file path

**Step 4: Commit**

```bash
cd "<repo>"
git add .claude/skills/handoff/SKILL.md
git commit -m "feat: add project handoff skill v2 (sigma-TAP specific)"
```

---

### Task 3: Write Global Resume Skill v2

**Files:**
- Create: `~/.claude/skills/resume/SKILL.md` (replaces existing v1)

**Step 1: Back up the existing v1 skill**

```bash
cp ~/.claude/skills/resume/SKILL.md ~/.claude/skills/resume/SKILL.v1.md
```

**Step 2: Write the new global resume skill**

Write `~/.claude/skills/resume/SKILL.md` with this exact structure:

```markdown
---
name: resume
description: Reconstruct session context from layered handoff buffer. Use at start of session. Checks for project skill.
---

# Session Resume

## Step 0: Check for Project Skill

Check if `<repo>/.claude/skills/resume/SKILL.md` exists:
1. **If it exists**: Read and follow that skill instead — it overrides these generic instructions.
2. **If not**: Continue with the generic process below.

## Process

### 1. Read hot layer only

Read `.claude/buffer/handoff.json` (~200 lines). This is the only mandatory read.

If it doesn't exist, inform user: "No handoff buffer found. Starting fresh. MEMORY.md is available for project context."

If `schema_version` < 2 or missing, inform user: "Found v1 buffer. Run `/handoff` first to migrate to v2 format."

### 2. Git grounding

```bash
git log --oneline <session_meta.commit>..HEAD
git status
git diff --stat
```

Present:
```
## Repo state
**Buffer recorded**: [commit] on [branch] ([date])
**Current HEAD**: [commit] on [branch]
**Commits since handoff**: [count] — [summaries]
**Working tree**: [clean / N modified files]
```

Flag if stale (commits or changes not in buffer).

### 3. Present session state

From hot layer:
```
## Last Session: [date]
**Commit**: [hash] on [branch]
**Phase**: [current_phase]
**Completed**: [list]
**In Progress**: [item or "nothing pending"]
**Next Action**: [next_action]

## Natural Summary
[natural_summary text]
```

### 4. Follow flagged pointers

**Pointer-following algorithm:**

For each entry in `concept_map_digest.flagged` and `concept_map_digest.recent_changes`:
1. Collect all `"see"` references (e.g., `["w:34", "w:35"]`)
2. Read `handoff-warm.json`, extract ONLY the entries matching those IDs
3. If a warm entry has `"see_also"` references, read `handoff-cold.json` and extract those entries
4. **Max cascade depth: 3** (hot → warm → cold, stop)
5. **Visited set**: track all followed IDs, skip duplicates
6. **Broken ref**: if ID not found in target file, log `"⚠️ Broken reference: [id] not found in [layer]"` and continue
7. **Tombstone**: if entry has `"archived_to"`, note: `"[id] archived to [tower]. Retrieve if needed."`

For each `open_thread` with `"see"` pointers: follow into warm, present context.

Present flagged/changed concepts:
```
## Concept Map Changes
- [NEW] [summary] (see w:34)
- [CHANGED] [summary] (see w:12)
```

### 5. Check full-scan threshold

If `sessions_since_full_scan >= full_scan_threshold`:
```
It's been [N] sessions since a full buffer scan (threshold: [T]).
Would you like me to do a complete review of warm + cold layers?
```
- If yes: read all layers, surface stale/orphaned entries, reset counter to 0
- If no: continue with selective loading

### 6. Surface instance notes

Present the `instance_notes` section:
```
## Notes from the previous instance
[remarks — paraphrased naturally]

**Open questions:**
- [question 1]
- [question 2]
```

### 7. Read MEMORY.md

Read project memory for baseline context. The buffer is the session delta; MEMORY.md is the project baseline.

### 8. Confirm

Tell user: "Context reconstructed from [date] handoff. Ready to continue from [current_phase]."

Ask: "Shall I proceed with [next_action or first open_thread], or do you have a different priority?"
```

**Step 3: Verify**

Read back the file. Confirm:
- Step 0 checks for project skill
- 8 process steps (not 12 like v1)
- Pointer-following algorithm explicit with depth cap, visited set, broken ref, tombstone handling
- Full-scan threshold check at step 5
- No project-specific content
- natural_summary replaces codex/compact_summary

**Step 4: Back up note**

The backup `SKILL.v1.md` preserves the original. No git commit needed for user-level files.

---

### Task 4: Write Project Resume Skill v2

**Files:**
- Create: `<repo>/.claude/skills/resume/SKILL.md`

**Step 1: Check target directory exists**

```bash
ls "<repo>/.claude/skills/resume/" 2>/dev/null || mkdir -p "<repo>/.claude/skills/resume/"
```

**Step 2: Write the project resume skill**

Write `<repo>/.claude/skills/resume/SKILL.md` with this exact structure:

```markdown
---
name: resume
description: Reconstruct sigma-TAP session context from layered handoff buffer. Project-specific override.
---

# sigma-TAP — Session Resume

Project-specific resume skill. Overrides the global resume skill with sigma-TAP priorities.

## Configuration

- **Buffer directory**: `.claude/buffer/`
- **Memory file**: `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`
- **Index file**: `docs/references/INDEX.md`

## Resume Priorities

After the standard hot-layer reconstruction (Steps 1-4 of global process), additionally:

### Source Material Review

Read `docs/references/INDEX.md`. Present brief inventory:
```
## Source materials (docs/references/)
**Unread**: [list with authors]
**Partially reviewed**: [list]
**Mapped**: [list with design doc sections]
**Foundational**: [list]
```

Pay special attention to `unread` sources flagged in open_threads.

### Concept Map Focus

When following flagged pointers (Step 4), prioritize:
1. `cross_source` entries marked NEW (from recent distillations)
2. Any entry with `suggest` ≠ null (pending user confirmation)
3. Entries in `foundational_triad` or base system groups (changes need user input)

### Forward Notes

Surface any `ref` fields containing `§5.XX` — these are bookmarked ideas from `docs/plans/2026-02-26-stage3a-two-channel-design.md`.

## Process

Follow the global resume skill process (Steps 1-8), using the configuration and priorities defined here. Insert the Source Material Review between steps 6 and 7 of the global process.
```

**Step 3: Verify**

Read back the file. Confirm:
- References sigma-TAP specific priorities (source materials, cross_source, forward notes)
- Defers to global skill for process steps
- Specifies where source material review inserts

**Step 4: Commit**

```bash
cd "<repo>"
git add .claude/skills/resume/SKILL.md
git commit -m "feat: add project resume skill v2 (sigma-TAP specific)"
```

---

### Task 5: Migrate v1 Buffer to v2 Three-Layer Format

**Files:**
- Read: `.claude/buffer/handoff.json` (v1, 1056 lines)
- Create: `.claude/buffer/handoff.json` (v2 hot layer)
- Create: `.claude/buffer/handoff-warm.json` (v2 warm layer)
- Create: `.claude/buffer/handoff-cold.json` (v2 cold layer)
- Create: `.claude/buffer/handoff-v1-archive.json` (safety backup)

This is the most complex task. Read the design doc Section 7 (Migration) carefully before starting.

**Step 1: Read the entire v1 buffer**

Read `.claude/buffer/handoff.json` completely (all 1056 lines). Understand its structure:
- `schema_version`: 1
- `session_meta`: date, commit, branch, files_modified, tests
- `active_work`: phase, completed, in_progress, blocked_by
- `orientation`: _purpose, core_insight, why_levinas, why_sartre, why_dg, why_emery, user_framework, practical_warning
- `decisions[]`: 14 entries (cumulative from all sessions)
- `open_threads[]`: ~7 entries (mix of completed/noted/deferred)
- `concept_map`: foundational_triad, dialectic, T, A, P, S, RIP, cross_source (~77 entries total)
- `validation_log[]`: ~20+ entries
- `dialogue_trace`: 5 session entries + recurring_patterns
- `instance_notes`: from/to/remarks/open_questions
- `compact_summary`: encoded string
- `codex`: version 3, 46 abbreviations, rules

**Step 2: Rename v1 as safety backup**

```bash
cp "<repo>/.claude/buffer/handoff.json" "<repo>/.claude/buffer/handoff-v1-archive.json"
```

**Step 3: Triage and assign IDs**

Walk the v1 buffer and triage into three layers:

**Warm layer entries (assign w:1 through w:N):**
- All `concept_map` entries across all groups → `w:1` through `w:77` (approximately)
  - Each entry in foundational_triad, dialectic, T, A, P, S, RIP: assign sequential w:IDs
  - Each entry in cross_source: assign sequential w:IDs
- Last 5 `decisions[]` entries → `w:78` through `w:82` (as `decisions_archive`)
- All `validation_log[]` entries → `w:83` through `w:N`

**Cold layer entries (assign c:1 through c:N):**
- All `dialogue_trace.sessions[]` entries → `c:1` through `c:5`
- `dialogue_trace.recurring_patterns` → preserved as-is (no individual IDs)
- Older `decisions[]` entries (first 9, not last 5) → `c:6` through `c:14` (as `archived_decisions`)
- Any `validation_log` entries with status `CHANGED` → extract as `superseded_mappings`

**Hot layer (no IDs, rebuilt fresh):**
- `session_meta` → copy as-is
- `orientation` → copy, reorganize into `core_insight`, `practical_warning`, `why_keys`
- `active_work` → copy, add `next_action` field
- `open_threads` → copy active/noted/deferred only (drop completed), add `"see"` pointers
- `recent_decisions` → last 3 decisions only, add `"see"` pointers
- `instance_notes` → copy as-is, rename `open_questions_i_never_got_to_ask` to `open_questions`
- `concept_map_digest` → generate from warm concept_map (count entries, no recent_changes since this is migration)
- `natural_summary` → write 2-3 sentences derived from the v1 compact_summary content and active_work
- `sessions_since_full_scan` → 0 (fresh start)
- `full_scan_threshold` → 5

**Discard:**
- `codex` section entirely
- `compact_summary` (replaced by natural_summary)
- `concept_map._meta` (replaced by concept_map_digest._meta in hot)
- Completed open_threads (historical, not actionable)

**Step 4: Write the warm layer**

Write `.claude/buffer/handoff-warm.json` with all assigned warm entries. Structure:

```json
{
  "concept_map": {
    "foundational_triad": [
      { "id": "w:1", "term": "unificity", "base": "...", "geometry": "vortical", "structure": "holarchy", ... },
      ...
    ],
    "dialectic": [ ... ],
    "T": [ ... ],
    "A": [ ... ],
    "P": [ ... ],
    "S": [ ... ],
    "RIP": [ ... ],
    "cross_source": [ ... ]
  },
  "decisions_archive": [ ... ],
  "validation_log": [ ... ]
}
```

Preserve ALL concept map content from v1. Each entry gets its `w:N` ID added. The internal structure of each entry is preserved (equiv, suggest, base, etc.).

**Step 5: Write the cold layer**

Write `.claude/buffer/handoff-cold.json`:

```json
{
  "dialogue_trace": {
    "sessions": [
      { "id": "c:1", "session": "...", "arc": "...", "key_moments": [...] },
      ...
    ],
    "recurring_patterns": [ ... ]
  },
  "superseded_mappings": [ ... ],
  "archived_decisions": [ ... ]
}
```

**Step 6: Write the hot layer**

Write `.claude/buffer/handoff.json` (overwriting the v1 file — backup exists):

```json
{
  "schema_version": 2,
  "session_meta": { ... },
  "sessions_since_full_scan": 0,
  "full_scan_threshold": 5,
  "orientation": { ... },
  "active_work": { ... },
  "open_threads": [ ... ],
  "recent_decisions": [ ... ],
  "instance_notes": { ... },
  "concept_map_digest": {
    "_meta": { "total_entries": <count>, "last_validated": "2026-03-02" },
    "recent_changes": [],
    "flagged": []
  },
  "natural_summary": "Stage 3B complete with 345 tests across 7 phases. Post-3B reflection ongoing with Levinas/Sartre/Emery integration. Handoff v2 implementation in progress. Next: R&B deep review, then TASKS.md and Stage 3C planning."
}
```

**Step 7: Verify layer sizes**

```bash
wc -l "<repo>/.claude/buffer/handoff.json"
wc -l "<repo>/.claude/buffer/handoff-warm.json"
wc -l "<repo>/.claude/buffer/handoff-cold.json"
```

Expected:
- Hot: ≤ 200 lines
- Warm: ≤ 500 lines
- Cold: ≤ 500 lines

If any exceed bounds, apply conservation enforcement from the handoff skill (Step 9 of global process).

**Step 8: Verify pointer integrity**

For each `"see"` pointer in hot: confirm the referenced `w:N` ID exists in warm.
For each `"see_also"` pointer in warm: confirm the referenced `c:N` ID exists in cold.

Report any broken references.

**Step 9: Commit**

```bash
cd "<repo>"
git add .claude/buffer/handoff.json .claude/buffer/handoff-warm.json .claude/buffer/handoff-cold.json .claude/buffer/handoff-v1-archive.json
git commit -m "migrate: v1 buffer to v2 three-layer format (hot/warm/cold)"
```

---

### Task 6: Update Distill Skill Post-Distillation Section

**Files:**
- Modify: `<repo>/.claude/skills/distill/SKILL.md` (lines 258-292, "Post-Distillation Updates" section)

The distill skill currently writes to the v1 buffer format (monolithic `handoff.json` with `concept_map.cross_source` and `validation_log`). Update it to write to the v2 warm layer format.

**Step 1: Read the current distill skill**

Read `<repo>/.claude/skills/distill/SKILL.md` and locate the "Post-Distillation Updates" section (around line 258).

**Step 2: Replace the handoff.json update sub-section**

Find the section that reads:

```markdown
**2. handoff.json Update**

Read `.claude/buffer/handoff.json`. Draw mappings from the **interpretation file's** Project Significance table and Integration Points:

- In `concept_map.cross_source`: add a mapping entry for each concept that maps to the sigma-TAP framework. Format:
  ```json
  "Source:ConceptName": {
    "maps_to": "[sigma-TAP mapping]",
    "ref": "[§5.XX forward note reference if applicable]",
    "suggest": null
  }
  ```
- In `validation_log`: add an entry:
  ```json
  {
    "check": "distill: [Source Label]",
    "status": "NEW",
    "detail": "[N] concepts mapped, [M] integration points identified",
    "session": "YYYY-MM-DD"
  }
  ```
```

Replace with:

```markdown
**2. Buffer Update (v2 three-layer)**

Read `.claude/buffer/handoff-warm.json` (warm layer). Draw mappings from the **interpretation file's** Project Significance table and Integration Points:

- In `concept_map.cross_source`: add a mapping entry for each concept. Assign a new `w:N` ID (read current max warm ID, increment). Format:
  ```json
  {
    "id": "w:N",
    "key": "Source:ConceptName",
    "maps_to": "[sigma-TAP mapping]",
    "ref": "[§5.XX forward note reference if applicable]",
    "suggest": null
  }
  ```
- In `validation_log`: add an entry with new `w:N` ID:
  ```json
  {
    "id": "w:N",
    "check": "distill: [Source Label]",
    "status": "NEW",
    "detail": "[N] concepts mapped, [M] integration points identified",
    "session": "YYYY-MM-DD"
  }
  ```

Then read `.claude/buffer/handoff.json` (hot layer) and update `concept_map_digest`:
- Increment `_meta.total_entries` by the number of new cross_source entries
- Add each new entry to `recent_changes` with status `NEW`
- If any mapping is uncertain, add its `w:N` ID to `flagged`
```

**Step 3: Verify the edit**

Read back the modified section. Confirm:
- References warm layer (`handoff-warm.json`) not monolithic `handoff.json`
- Includes `w:N` ID assignment
- Includes hot-layer digest update
- INDEX.md and MEMORY.md update sections remain unchanged

**Step 4: Commit**

```bash
cd "<repo>"
git add .claude/skills/distill/SKILL.md
git commit -m "refactor: update distill post-updates for v2 buffer format"
```

---

### Task 7: Full-Cycle Verification

This task verifies the entire system works end-to-end. No files are created — this is pure verification.

**Step 1: Verify file inventory**

```bash
ls -la "<repo>/.claude/buffer/"
ls -la "<repo>/.claude/skills/"
ls -la ~/.claude/skills/handoff/
ls -la ~/.claude/skills/resume/
```

Expected files:
- `.claude/buffer/handoff.json` (v2 hot, ≤200 lines)
- `.claude/buffer/handoff-warm.json` (v2 warm, ≤500 lines)
- `.claude/buffer/handoff-cold.json` (v2 cold, ≤500 lines)
- `.claude/buffer/handoff-v1-archive.json` (safety backup)
- `.claude/skills/handoff/SKILL.md` (project handoff)
- `.claude/skills/resume/SKILL.md` (project resume)
- `.claude/skills/distill/SKILL.md` (updated post-distillation)
- `~/.claude/skills/handoff/SKILL.md` (global handoff v2)
- `~/.claude/skills/handoff/SKILL.v1.md` (global handoff v1 backup)
- `~/.claude/skills/resume/SKILL.md` (global resume v2)
- `~/.claude/skills/resume/SKILL.v1.md` (global resume v1 backup)

**Step 2: Validate hot layer schema**

Read `.claude/buffer/handoff.json`. Verify:
- [ ] `schema_version` is 2
- [ ] `sessions_since_full_scan` exists and is a number
- [ ] `full_scan_threshold` is 5
- [ ] `orientation` has `core_insight` and `practical_warning`
- [ ] `active_work` has `next_action` field
- [ ] `open_threads` entries have `"see"` arrays (even if empty)
- [ ] `recent_decisions` entries have `"see"` arrays
- [ ] `concept_map_digest` has `_meta`, `recent_changes`, `flagged`
- [ ] `natural_summary` is a plain string (no encoded content)
- [ ] No `codex` section
- [ ] No `compact_summary` section
- [ ] No `dialogue_trace` section (moved to cold)
- [ ] No monolithic `decisions[]` (split into recent_decisions + decisions_archive)
- [ ] Total lines ≤ 200

**Step 3: Validate warm layer schema**

Read `.claude/buffer/handoff-warm.json`. Verify:
- [ ] `concept_map` has all 8 groups (foundational_triad, dialectic, T, A, P, S, RIP, cross_source)
- [ ] Every entry has an `"id"` field starting with `"w:"`
- [ ] `decisions_archive` entries have `"id"` fields
- [ ] `validation_log` entries have `"id"` fields
- [ ] No duplicate IDs
- [ ] Total lines ≤ 500

**Step 4: Validate cold layer schema**

Read `.claude/buffer/handoff-cold.json`. Verify:
- [ ] `dialogue_trace.sessions` has entries with `"id"` fields starting with `"c:"`
- [ ] `recurring_patterns` preserved
- [ ] `archived_decisions` exists (may be empty)
- [ ] `superseded_mappings` exists (may be empty)
- [ ] No duplicate IDs
- [ ] Total lines ≤ 500

**Step 5: Validate pointer integrity**

For EVERY `"see"` reference in hot:
- Extract the `w:N` ID
- Confirm it exists in warm layer

For EVERY `"see_also"` reference in warm:
- Extract the `c:N` ID
- Confirm it exists in cold layer

Report: "All N pointers validated" or list broken refs.

**Step 6: Validate skill routing**

Read `<repo>/.claude/skills/handoff/SKILL.md` — confirm it exists and has sigma-TAP content.
Read `~/.claude/skills/handoff/SKILL.md` — confirm Step 0 routes to project skill.

Read `<repo>/.claude/skills/resume/SKILL.md` — confirm it exists and has sigma-TAP content.
Read `~/.claude/skills/resume/SKILL.md` — confirm Step 0 routes to project skill.

**Step 7: Validate distill integration**

Read `<repo>/.claude/skills/distill/SKILL.md` lines 258-297. Confirm:
- References `handoff-warm.json` (not monolithic `handoff.json`)
- Includes `w:N` ID assignment
- Includes hot-layer digest update

**Step 8: Report**

Present a summary:
```
## v2 Migration Verification
- Hot layer: [lines] lines, schema v2 ✅/❌
- Warm layer: [lines] lines, [N] concept map entries ✅/❌
- Cold layer: [lines] lines, [N] trace sessions ✅/❌
- Pointer integrity: [N]/[N] valid ✅/❌
- Skill routing: handoff ✅/❌, resume ✅/❌
- Distill integration: ✅/❌
- v1 backup preserved: ✅/❌
```

**Step 9: Final commit (if any fixes were needed)**

```bash
cd "<repo>"
git add -A .claude/
git commit -m "verify: handoff/resume v2 migration complete"
```

**Step 10: Note for future**

After one successful `/handoff` → `/resume` cycle in a real session, the `handoff-v1-archive.json` backup can be deleted:
```bash
rm "<repo>/.claude/buffer/handoff-v1-archive.json"
git add .claude/buffer/handoff-v1-archive.json
git commit -m "cleanup: remove v1 buffer archive after successful v2 cycle"
```

Do NOT delete it now — wait for real-world verification.

---

## Task Dependency Graph

```
Task 1 (Global Handoff) ──┐
                           ├──> Task 5 (Migration) ──> Task 7 (Verification)
Task 2 (Project Handoff) ─┤
                           │
Task 3 (Global Resume) ───┤
                           │
Task 4 (Project Resume) ──┘

Task 6 (Distill Update) ────────────────────────────> Task 7 (Verification)
```

Tasks 1-4 and Task 6 are independent of each other and can be done in parallel.
Task 5 (Migration) depends on Tasks 1-4 (needs skill files to reference structure).
Task 7 (Verification) depends on all other tasks.
