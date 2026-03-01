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

### 8. Generate compact summary

Write a single dense line using the codex encoding. Format:
- `|` separates topics
- `=` marks equivalence/assignment
- `:` marks containment/specification
- `/` separates alternatives or levels
- Use abbreviations from the codex section

Update the codex if you introduced any new abbreviations.

### 9. Write the buffer

Write the complete JSON to `.claude/buffer/handoff.json`. The schema:

```json
{
  "schema_version": 1,
  "session_meta": { "date", "commit", "branch", "files_modified", "tests" },
  "orientation": { "_purpose", "core_insight", "why_*", "practical_warning" },
  "active_work": { "current_phase", "completed_this_session", "in_progress", "blocked_by" },
  "decisions": [{ "what", "chose", "why", "ref" }],
  "open_threads": [{ "thread", "status", "ref" }],
  "concept_map": { "_meta", "dialectic", "T", "A", "P", "S", "RIP", "cross_source" },
  "validation_log": [{ "check", "status", "detail", "session" }],
  "instance_notes": { "from", "to", "remarks", "open_questions_i_never_got_to_ask" },
  "compact_summary": "<encoded string>",
  "codex": { "version", "encoding", "rules" }
}
```

Preserve the existing concept_map entries — only modify entries that changed this session.
Preserve the orientation section — only update if theoretical framing shifted this session.
The instance_notes section should be REPLACED each session (it's from YOU, not cumulative).

### 10. Commit

Run:
```bash
git add .claude/buffer/handoff.json
git commit -m "handoff: <brief description of session>"
```

### 11. Confirm

Tell the user: "Handoff buffer written and committed. The next instance can run `/resume` to reconstruct context."
