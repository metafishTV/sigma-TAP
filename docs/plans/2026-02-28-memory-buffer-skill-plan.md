# Memory Buffer Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `/handoff` and `/resume` slash commands with a structured JSON knowledge buffer for session continuity.

**Architecture:** Two Claude Code skills (SKILL.md files) that guide buffer generation and context reconstruction. A JSON buffer file stores session state, decisions, open threads, and a concept map with TAPS/RIP/Dialectic as coordinate bases. All project-level, git-tracked.

**Tech Stack:** Claude Code skills (YAML frontmatter + markdown), JSON data file, git.

**Design doc:** `docs/plans/2026-02-28-memory-buffer-skill-design.md`

---

### Task 1: Create Directory Structure

**Files:**
- Create: `.claude/skills/handoff/` (directory)
- Create: `.claude/skills/resume/` (directory)
- Create: `.claude/buffer/` (directory)

**Step 1: Create all directories**

Run:
```bash
mkdir -p ".claude/skills/handoff" ".claude/skills/resume" ".claude/buffer"
```

**Step 2: Verify structure**

Run:
```bash
ls -R .claude/
```

Expected: Three directories visible under `.claude/`.

---

### Task 2: Write the `/handoff` Skill

**Files:**
- Create: `.claude/skills/handoff/SKILL.md`

**Step 1: Write the SKILL.md file**

Write the following to `.claude/skills/handoff/SKILL.md`:

~~~markdown
---
name: handoff
description: Generate a structured session handoff buffer for the next Claude instance. Use at end of session.
---

# Session Handoff

Generate a structured handoff buffer and save it to `.claude/buffer/handoff.json`. This buffer helps the next Claude instance reconstruct context quickly.

## Process

### 1. Read the existing buffer

Read `.claude/buffer/handoff.json` to understand the current state. If it doesn't exist or is empty, you are creating the first handoff.

### 2. Gather session metadata

Collect by running git commands:
- Today's date
- Current commit hash (`git rev-parse --short HEAD`)
- Current branch (`git branch --show-current`)
- Files modified this session (`git diff --name-only` against the commit hash in the previous buffer, or the last 5 commits if no previous buffer)
- Test status (run `python -m pytest tests/ -q --tb=no` and capture the pass/fail line)

### 3. Summarize active work

Ask yourself (do NOT ask the user — infer from conversation context):
- What phase/stage is the project in?
- What was completed this session?
- What is currently in-progress?
- Is anything blocked? If so, by what?

### 4. Log decisions

Review the conversation for decisions made this session. For each:
- What was decided?
- What was chosen?
- Why? (brief rationale)
- Reference to design doc section if applicable

### 5. List open threads

Identify unresolved questions, deferred items, and next steps. For each:
- What is the thread?
- Status: `noted` | `deferred` | `blocked` | `needs-user-input`
- Reference if applicable

### 6. Validate concept map

Read the existing concept_map from the buffer. For each decision from step 4:
- Does it touch a concept mapping? If yes, check against the existing entry.
- If a mapping **changed**: update the entry, add to validation_log as `CHANGED`
- If a **new concept** was introduced: add entry, log as `NEW`
- If a **suggest was confirmed** by the user: promote to equiv, log as `PROMOTED`
- If the **base system** (TAPS/RIP/Dialectic) was questioned: log as `NEEDS_USER_INPUT`, do NOT auto-change
- If nothing changed: log `PASS`

**IMPORTANT**: `suggest: null` is the PREFERRED state. Do NOT feel pressure to populate suggest fields. Only flag genuine structural parallels you noticed during the session. The user must confirm any suggestion before it becomes an equiv.

### 7. Generate compact summary

Write a single dense line using the codex encoding. Format:
- `|` separates topics
- `=` marks equivalence/assignment
- `:` marks containment/specification
- `/` separates alternatives or levels
- Use abbreviations from the codex section

Update the codex if you introduced any new abbreviations.

### 8. Write the buffer

Write the complete JSON to `.claude/buffer/handoff.json`. The schema:

```json
{
  "schema_version": 1,
  "session_meta": { "date", "commit", "branch", "files_modified", "tests" },
  "active_work": { "current_phase", "completed_this_session", "in_progress", "blocked_by" },
  "decisions": [{ "what", "chose", "why", "ref" }],
  "open_threads": [{ "thread", "status", "ref" }],
  "concept_map": { "_meta", "dialectic", "T", "A", "P", "S", "RIP", "cross_source" },
  "validation_log": [{ "check", "status", "detail", "session" }],
  "compact_summary": "<encoded string>",
  "codex": { "version", "encoding", "rules" }
}
```

Preserve the existing concept_map entries — only modify entries that changed this session.

### 9. Commit

Run:
```bash
git add .claude/buffer/handoff.json
git commit -m "handoff: <brief description of session>"
```

### 10. Confirm

Tell the user: "Handoff buffer written and committed. The next instance can run `/resume` to reconstruct context."
~~~

**Step 2: Verify the file was created**

Run:
```bash
cat .claude/skills/handoff/SKILL.md | head -5
```

Expected: YAML frontmatter with `name: handoff`.

---

### Task 3: Write the `/resume` Skill

**Files:**
- Create: `.claude/skills/resume/SKILL.md`

**Step 1: Write the SKILL.md file**

Write the following to `.claude/skills/resume/SKILL.md`:

~~~markdown
---
name: resume
description: Reconstruct session context from the handoff buffer. Use at start of session.
---

# Session Resume

Read the handoff buffer and reconstruct context from the previous session.

## Process

### 1. Read the buffer

Read `.claude/buffer/handoff.json`. If it doesn't exist, inform the user: "No handoff buffer found. Starting fresh. MEMORY.md is available for project context."

### 2. Parse and present session state

Present a concise summary:

```
## Last Session: [date]
**Commit**: [hash] on [branch]
**Phase**: [current_phase]
**Completed**: [list]
**In Progress**: [item or "nothing pending"]
**Blocked**: [item or "nothing blocked"]
```

### 3. Surface decisions

For each decision in the buffer:
```
**Decision**: [what]
**Chose**: [chose] — [why]
```

### 4. Present open threads

For each open thread, ordered by status priority (needs-user-input > blocked > noted > deferred):
```
- [status] [thread] (ref: [ref])
```

### 5. Check validation warnings

Review the validation_log. If any entries have status `CHANGED`, `NEW`, or `NEEDS_USER_INPUT`, surface them:
```
⚠️ Concept map changes from last session:
- [CHANGED] [detail]
- [NEW] [detail]
- [NEEDS_USER_INPUT] [detail]
```

### 6. Decode compact summary

Read the codex, then decode the compact_summary. Present the decoded version as a quick orientation line.

### 7. Read MEMORY.md

Also read MEMORY.md for the persistent project context. The buffer is the session delta; MEMORY.md is the project baseline.

### 8. Confirm

Tell the user: "Context reconstructed from [date] handoff. Ready to continue from [current_phase]."

Ask: "Shall I proceed with [in_progress or first open_thread], or do you have a different priority?"
~~~

**Step 2: Verify the file was created**

Run:
```bash
cat .claude/skills/resume/SKILL.md | head -5
```

Expected: YAML frontmatter with `name: resume`.

---

### Task 4: Create the Initial Handoff Buffer

**Files:**
- Create: `.claude/buffer/handoff.json`

**Step 1: Write the initial buffer**

Write a complete `handoff.json` seeded with:
- Session meta from TODAY's session (current commit, branch, etc.)
- Active work state (Stage 3B Phase 0 — building this skill, then moving to Phase 1-7 implementation planning)
- Decisions made this session (skill approach chosen, concept map structure agreed)
- Open threads (Stage 3B implementation planning is next)
- The FULL concept map from the design doc (§4.2-4.8) — all TAPS groups, RIP, Dialectic, cross-source mappings
- Validation log with initial entry (NEW: all entries, first population)
- A compact summary line
- The initial codex

The concept map content should be copied exactly from the design doc sections 4.2 through 4.8.

**Step 2: Validate JSON**

Run:
```bash
python -c "import json; json.load(open('.claude/buffer/handoff.json')); print('Valid JSON')"
```

Expected: `Valid JSON`

---

### Task 5: Test Both Commands

**Step 1: Verify `/handoff` is discoverable**

The skill should appear when listing available slash commands. Verify the file exists at the correct path:

Run:
```bash
cat .claude/skills/handoff/SKILL.md | head -3
```

Expected: YAML frontmatter visible.

**Step 2: Verify `/resume` is discoverable**

Run:
```bash
cat .claude/skills/resume/SKILL.md | head -3
```

Expected: YAML frontmatter visible.

**Step 3: Verify buffer is valid and complete**

Run:
```bash
python -c "
import json
with open('.claude/buffer/handoff.json') as f:
    buf = json.load(f)
sections = ['session_meta', 'active_work', 'decisions', 'open_threads', 'concept_map', 'validation_log', 'compact_summary', 'codex']
for s in sections:
    assert s in buf, f'Missing section: {s}'
cm = buf['concept_map']
for group in ['_meta', 'dialectic', 'T', 'A', 'P', 'S', 'RIP', 'cross_source']:
    assert group in cm, f'Missing concept_map group: {group}'
print(f'All {len(sections)} sections present')
print(f'All concept_map groups present')
print(f'Schema version: {buf[\"schema_version\"]}')
print(f'Codex version: {buf[\"codex\"][\"version\"]}')
"
```

Expected: All sections and concept_map groups present.

---

### Task 6: Commit Everything

**Step 1: Stage all new files**

Run:
```bash
git add .claude/skills/handoff/SKILL.md .claude/skills/resume/SKILL.md .claude/buffer/handoff.json
```

**Step 2: Commit**

Run:
```bash
git commit -m "feat: add /handoff and /resume skills with initial concept map buffer

Two Claude Code skills for session continuity:
- /handoff: generates structured JSON buffer at session end
- /resume: reconstructs context at session start

Buffer includes: session metadata, active work state, decisions,
open threads, concept map (TAPS/RIP/Dialectic base with cross-source
mappings and validation), compact summary with codex.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Step 3: Push**

Run:
```bash
git push origin main
```

**Step 4: Verify clean state**

Run:
```bash
git status
```

Expected: Clean working tree.

---

### Task 7: Run `/handoff` for This Session

**Step 1: Invoke `/handoff`**

This is the real test — use the `/handoff` command to generate a buffer for this session. The buffer should capture:
- This session's work (designing and building the handoff skill)
- The decision to use Approach A with Approach B element
- Open thread: Stage 3B implementation planning (Phases 1-7) is next
- Concept map should already be populated from Task 4

**Step 2: Verify the updated buffer**

Run:
```bash
python -c "import json; buf = json.load(open('.claude/buffer/handoff.json')); print(buf['session_meta']); print(buf['compact_summary'])"
```

**Step 3: Commit the updated buffer**

Run:
```bash
git add .claude/buffer/handoff.json
git commit -m "handoff: Phase 0 complete — /handoff and /resume skills built"
git push origin main
```
