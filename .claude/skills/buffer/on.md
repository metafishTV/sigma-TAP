---
name: on
description: Bring sigma-TAP session context on hand from layered handoff buffer. Project-specific override.
---

# sigma-TAP — Session On Hand

Project-specific on-hand skill. Overrides the global `/buffer:on` with sigma-TAP priorities.

## Configuration

- **Buffer mode**: `project` (always — skip Step 0d mode selection)
- **Buffer directory**: `.claude/buffer/`
- **Memory file**: `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`
- **Index file**: `docs/references/INDEX.md`

> **Mode note**: sigma-TAP always runs in project mode. Step 0d (mode choice) in the global `/buffer:on` is skipped — the hot layer's `buffer_mode` is always `"project"`. All mode-gated steps (concept map, convergence web, provenance consolidation) fire unconditionally.

## Greeting

When `/buffer:on` is invoked, always begin with a brief variation of: **"Let me see what's on hand."** Then proceed silently through the reconstruction steps. Do not ask the user anything before reconstruction is complete — the greeting is a signal that work has started, not a prompt for input.

## On-Hand Priorities

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
2. Any entry with `suggest` != null (pending user confirmation)
3. Entries in `foundational_triad` or base system groups (changes need user input)

### Convergence Web Focus

Read `convergence_web_digest` from the hot layer. Present cluster summary:

```
## Convergence web (inter-source linkages)
**Total entries**: [N] across [M] clusters
**Clusters**: [list cluster names]
**Flagged**: [any flagged entries needing review]
```

If `flagged` is non-empty, read the flagged entries from the warm layer's `convergence_web.entries[]` and surface for user review.

### Forward Notes

Surface any `ref` fields containing `§5.XX` — these are bookmarked ideas from `docs/plans/2026-02-26-stage3a-two-channel-design.md`.

## Process

Follow the global `/buffer:on` process (Steps 1-8), using the configuration and priorities defined here. Insert the Source Material Review between Steps 6 and 7 of the global process. The Autosave Protocol from the global skill applies — autosave is armed at Step 8.
