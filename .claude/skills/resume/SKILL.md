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
