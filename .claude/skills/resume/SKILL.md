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

### 5. Read instance notes

If the buffer has an `instance_notes` section, read it carefully. This is a personal briefing from the previous instance — less formal than the concept map, more like a colleague's parting advice.

Present the remarks and open questions to the user:
```
## Notes from the previous instance
[remarks — paraphrased naturally, not as a JSON dump]

**Questions they never got to ask:**
- [question 1]
- [question 2]
```

These questions are worth surfacing — the user may want to address them, and they show where the previous instance's understanding had edges.

### 6. Read dialogue trace

If the buffer has a `dialogue_trace` section, read it carefully. This captures how the conversation actually developed across sessions — not just what was decided, but how ideas evolved through exchange.

Present the most recent session's arc and key moments, plus the recurring patterns:
```
## Conversation trajectory
**Most recent session**: [arc summary]
Key moments:
- [moment 1]
- [moment 2]

**Recurring patterns to be aware of:**
- [pattern]
```

For earlier sessions, present a brief one-line summary of each arc. The full detail is there if needed but don't overwhelm the user at startup.

### 7. Check validation warnings

Review the validation_log. If any entries have status `CHANGED`, `NEW`, or `NEEDS_USER_INPUT`, surface them:
```
⚠️ Concept map changes from last session:
- [CHANGED] [detail]
- [NEW] [detail]
- [NEEDS_USER_INPUT] [detail]
```

### 8. Decode compact summary

Read the codex, then decode the compact_summary. Present the decoded version as a quick orientation line.

### 9. Read orientation and MEMORY.md

Read the `orientation` section of the buffer for the theoretical framing (WHY the mappings matter, not just what they are). Then read MEMORY.md for the persistent project context. The buffer is the session delta; MEMORY.md is the project baseline.

### 10. Confirm

Tell the user: "Context reconstructed from [date] handoff. Ready to continue from [current_phase]."

Ask: "Shall I proceed with [in_progress or first open_thread], or do you have a different priority?"
