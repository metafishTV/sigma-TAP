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

Pay special attention to `unread` sources flagged in `open_threads`.

### Concept Map Focus

When following flagged pointers (Step 4), prioritize:
1. `cross_source` entries marked `NEW` (from recent distillations)
2. Any entry with `suggest` ≠ null (pending user confirmation)
3. Entries in `foundational_triad` or base system groups (changes need user input)

### Forward Notes

Surface any `ref` fields containing `§5.XX` — these are bookmarked ideas from `docs/plans/2026-02-26-stage3a-two-channel-design.md`.

## Process

Follow the global resume skill process (Steps 1-8), using the configuration and priorities defined here. Insert the Source Material Review between Steps 6 and 7 of the global process.
