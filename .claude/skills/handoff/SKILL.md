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

### 7. Write instance notes

Write the `instance_notes` section — personal remarks from you to the next instance. This is less formal than the concept map. Think of it as a colleague briefing before you leave.

Include:
- **remarks**: Things you learned about working with this user, this codebase, or this theory that aren't captured in the structured data. Warnings, tips, things that surprised you.
- **open_questions_i_never_got_to_ask**: Questions that occurred to you during the session but you didn't get to raise. These help the next instance know where the edges of understanding are.

Be honest. If something confused you, say so. If a mapping felt forced, flag it. The next instance benefits more from your candor than from false confidence.

### 8. Update dialogue trace

The `dialogue_trace` section captures HOW the conversation developed — the trajectory of ideas, not just the endpoints. This is the most lossy part of context transfer.

Add a new entry to the `sessions` array for THIS session. Include:
- **session**: A label (date + brief description)
- **arc**: 1-2 sentences describing the overall shape of the conversation — what it was about, how it moved
- **key_moments**: 3-6 specific moments where something important happened — a concept crystallized, a correction was made, a mapping was discovered, a question opened a new line of thinking. Be concrete: what was said, what shifted.

Also update `recurring_patterns` if you noticed new ones, or if existing patterns played out differently.

**Do NOT summarize** — trace. The difference: a summary says "we discussed trust metrics." A trace says "user proposed trust as a modulator, I suggested raw variance, user pointed out scale-dependence, we converged on CV-squared." The trace preserves the intellectual movement.

Preserve all previous session entries — this section IS cumulative (unlike instance_notes which get replaced).

### 9. Generate compact summary

Write a single dense line using the codex encoding. Format:
- `|` separates topics
- `=` marks equivalence/assignment
- `:` marks containment/specification
- `/` separates alternatives or levels
- Use abbreviations from the codex section

Update the codex if you introduced any new abbreviations.

### 10. Write the buffer

Write the complete JSON to `.claude/buffer/handoff.json`. The schema:

```json
{
  "schema_version": 1,
  "session_meta": { "date", "commit", "branch", "files_modified", "tests" },
  "orientation": { "_purpose", "core_insight", "why_*", "practical_warning" },
  "active_work": { "current_phase", "completed_this_session", "in_progress", "blocked_by" },
  "decisions": [{ "what", "chose", "why", "ref" }],
  "open_threads": [{ "thread", "status", "ref" }],
  "concept_map": { "_meta", "foundational_triad", "dialectic", "T", "A", "P", "S", "RIP", "cross_source" },
  "validation_log": [{ "check", "status", "detail", "session" }],
  "dialogue_trace": { "sessions": [{ "session", "arc", "key_moments" }], "recurring_patterns" },
  "instance_notes": { "from", "to", "remarks", "open_questions_i_never_got_to_ask" },
  "compact_summary": "<encoded string>",
  "codex": { "version", "encoding", "rules" }
}
```

**Cumulative sections** (APPEND only — never delete previous entries):
- `decisions` — append new decisions. Previous decisions are historical record.
- `validation_log` — append new entries. This is an audit trail.
- `dialogue_trace.sessions` — append your session entry. Never remove previous sessions.
- `dialogue_trace.recurring_patterns` — add new patterns, refine existing ones, never delete.
- `concept_map` — preserve all entries. Only modify entries that changed this session.
- `codex.encoding` — add new abbreviations. Never remove existing ones (other sections may reference them).

**Preserve-but-updatable sections** (keep, modify only if something shifted):
- `orientation` — only update if theoretical framing shifted this session.
- `codex.rules` — only update if encoding conventions changed.

**Replace-each-session sections** (fresh each handoff):
- `session_meta` — current session only.
- `active_work` — current state only.
- `open_threads` — current statuses (but preserve completed threads for context).
- `instance_notes` — personal to the outgoing instance, replaced each time.
- `compact_summary` — regenerated from current state.

### 11. Commit

Run:
```bash
git add .claude/buffer/handoff.json
git commit -m "handoff: <brief description of session>"
```

### 12. Confirm

Tell the user: "Handoff buffer written and committed. The next instance can run `/resume` to reconstruct context."
