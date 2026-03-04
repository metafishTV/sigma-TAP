# Session On Hand — sigma-TAP

Project-specific on-hand configuration for the sigma-TAP multi-agent praxis simulator.

## What This Adds

This project skill extends the global `/onhand` with sigma-TAP-specific priorities:

- **Greeting**: Always begins with a variation of "Let me see what's on hand" before reconstruction
- **Source material review**: Presents an inventory of distilled sources from `docs/references/INDEX.md`
- **Concept map focus**: Prioritizes new cross-source entries, pending suggestions, and foundational triad items
- **Convergence web focus**: Presents cluster summary and any flagged entries
- **Forward notes**: Surfaces `§5.XX` bookmarked ideas from the Stage 3A design document

## What Gets Surfaced

After the standard hot-layer reconstruction, the skill additionally presents:

### Source Material Inventory

```
## Source materials (docs/references/)
Unread: [list with authors]
Partially reviewed: [list]
Mapped: [list with design doc sections]
Foundational: [list]
```

### Concept Map Highlights

In priority order:
1. `cross_source` entries marked `NEW` (from recent distillations)
2. Any entry with `suggest != null` (pending user confirmation)
3. Entries in `foundational_triad` or base system groups (changes need user input)

### Convergence Web Summary

```
## Convergence web (inter-source linkages)
Total entries: [N] across [M] clusters
Clusters: [list cluster names]
Flagged: [any flagged entries needing review]
```

### Forward Notes

Any `ref` fields containing `§5.XX` are surfaced — these are bookmarked ideas from the design document that may be relevant to the current session's work.

## Configuration

| Setting | Value |
|---------|-------|
| Buffer directory | `.claude/buffer/` |
| Memory file | `C:\Users\user\.claude\projects\...\memory\MEMORY.md` |
| Index file | `docs/references/INDEX.md` |

## How to Modify

Edit `.claude/skills/onhand/SKILL.md` directly. You can change the greeting text, add new priority sections, or adjust what gets surfaced during reconstruction. The global onhand skill reads the project skill and follows its instructions.
