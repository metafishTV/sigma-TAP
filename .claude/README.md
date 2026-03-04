# Session Continuity System (Handoff / On Hand)

Persistent structured context transfer between Claude Code sessions, solving the problem of starting each session cold.

## The Problem

Claude Code sessions are ephemeral — each new instance starts with zero memory of what happened before. On complex projects, significant context window is wasted re-discovering decisions, open threads, concept evolution, and working-style notes. `MEMORY.md` helps but is static and size-limited. It cannot capture session-specific state: what was in-progress, what was decided and why, what the previous instance learned about the project and user.

## Quick Start

```
/onhand     # Start of session — reconstructs context from last handoff
/handoff    # End of session — saves state for next instance
```

First time? Just run `/handoff` at the end of your first session. The system bootstraps from there. Next session, `/onhand` picks up where you left off.

## Architecture

Content migrates downward when layer bounds are exceeded. On-hand reads upward selectively — the hot layer always, warm/cold only when a pointer says "look here." Initial context loading stays under ~200 lines.

```
              ALWAYS loaded (~200 lines max)
         ┌─────────────────────────────────┐
         │       HOT: handoff.json         │
         │  session state, decisions,       │
         │  open threads, orientation,      │
         │  instance notes, natural summary │
         │  concept map digest,             │
         │  convergence web digest          │
         └──────────┬──────────────────────┘
                    │ "see": ["w:N"]
                    ▼
         SELECTIVELY loaded via pointers (~500-800 lines)
         ┌─────────────────────────────────┐
         │    WARM: handoff-warm.json      │
         │  concept map (all groups),       │
         │  convergence web (inter-source), │
         │  decisions archive,              │
         │  validation log                  │
         └──────────┬──────────────────────┘
                    │ "see_also": ["c:N"]
                    ▼
              ON-DEMAND only (~500 lines max)
         ┌─────────────────────────────────┐
         │    COLD: handoff-cold.json      │
         │  dialogue trace, superseded      │
         │  mappings, archived decisions    │
         └──────────┬──────────────────────┘
                    │ archived_to
                    ▼
              SEALED (never auto-read)
         ┌─────────────────────────────────┐
         │  TOWER: handoff-tower-NNN-*.json │
         │  (created via archival process)  │
         └─────────────────────────────────┘
```

**Pointer-index system**: Hot-layer entries contain `"see"` arrays pointing to warm entries by stable ID (`w:N`). Warm entries may contain `"see_also"` arrays pointing to cold (`c:N`). Max cascade depth is 3 (hot → warm → cold, then stop). Redirect tombstones preserve pointer integrity when entries migrate between layers.

**Warm consolidation**: The warm layer gets *denser* over sessions, not just longer. At each handoff, a consolidation pass compresses descriptions using established vocabulary, merges overlapping entries, and tightens stable entries from explanatory prose to referential shorthand. This keeps content accessible in warm rather than pushing it to cold prematurely.

## What `/handoff` Does

1. Reads current buffer state
2. Gathers git metadata (commit, branch, modified files, test results)
3. Summarizes what you worked on (inferred from conversation — does not ask)
4. Logs decisions made this session with rationale
5. Lists open threads and deferred items
6. Validates concept map entries for changes, new additions, or flagged items
7. Consolidates warm layer (vocabulary compression, same-concept merge, description tightening)
8. Writes instance notes — a colleague-to-colleague briefing for the next instance
9. Writes a plain-English natural summary (no encoding or abbreviations)
10. Enforces size bounds (migrates overflow to lower layers automatically)
11. Writes all layer files to `.claude/buffer/`
12. Syncs MEMORY.md status line (if integration is configured)
13. Updates global project registry
14. Commits to git
15. Confirms completion

## What `/onhand` Does

1. Reads hot layer only (fast startup, ~200 lines)
2. Checks git state against buffer (detects if commits happened outside a session)
3. Presents session state: phase, completed work, next action
4. Follows flagged pointers into warm/cold layers (selective, not exhaustive)
5. Checks if a full rescan is due (configurable, default every 5 sessions)
6. Surfaces instance notes from previous instance
7. Reads MEMORY.md for project baseline
8. Arms autosave and confirms ready

**Autosave**: After on-hand completes, the system silently saves hot-layer state at natural completion boundaries (pipeline complete, tests pass, topic shift). This is lightweight — hot layer only, no migration, no git commit. Full `/handoff` is still needed at end of session for instance notes, full conservation, and commit.

## The Two-File Skill Pattern

| File | Location | Owns |
|------|----------|------|
| Global handoff | `~/.claude/skills/handoff/SKILL.md` | Generic 15-step process, three-layer schema, conservation enforcement, warm consolidation |
| Project handoff | `<repo>/.claude/skills/handoff/SKILL.md` | Concept map groups, terminology, validation rules, orientation template, consolidation heuristics |
| Global onhand | `~/.claude/skills/onhand/SKILL.md` | Generic 8-step reconstruction, pointer-following algorithm, autosave protocol |
| Project onhand | `<repo>/.claude/skills/onhand/SKILL.md` | On-hand priorities, source material review, forward note surfacing |

**Global skills** define the process (reusable across any project). **Project skills** define the structure (what the concept map looks like, what validation means for this domain). When invoked, the global skill checks for a project override first and defers if found.

**To use in your own project**: You only need the two global skill files. On first `/handoff`, a generic buffer is created. Add project skills when you want project-specific concept map groups, validation rules, or on-hand priorities.

## MEMORY.md Integration

The buffer system can integrate with Claude Code's MEMORY.md. Three modes (chosen at first run):

- **Full**: MEMORY.md becomes a lean orientation card (~50-60 lines). Theoretical content migrates to the buffer's concept map. Status line syncs at every handoff. Stable warm entries can be promoted to MEMORY.md after 5+ sessions.
- **Minimal**: Adds a pointer section to MEMORY.md. Existing content left alone.
- **None**: Independent operation. No MEMORY.md sync.

## Configuration

| Parameter | Default | Set in | Effect |
|-----------|---------|--------|--------|
| Hot max lines | 200 | Global handoff SKILL.md | Triggers migration to warm |
| Warm max lines | 500 (project can override) | Global handoff SKILL.md | Triggers migration to cold |
| Cold max lines | 500 | Global handoff SKILL.md | Triggers archival questionnaire |
| Full scan threshold | 5 sessions | Hot layer (`full_scan_threshold`) | Prompts full warm+cold review at on-hand |

Concept map groups, validation rules, orientation templates, and consolidation heuristics are configured in the project-level handoff SKILL.md.

## Buffer Maintenance

- **Warm consolidation**: At every handoff, existing warm entries are compressed using project vocabulary, overlapping entries are merged, and stable descriptions are tightened. The warm layer's line count should stay flat or drop while information density climbs.
- **Archival questionnaire**: When the cold layer exceeds its bound, the system presents a dependency map and lets you choose what to archive — you pick the ratio, direction, and specific entries. Archived entries go to sealed tower files.
- **Full rescan**: Every N sessions (configurable), `/onhand` offers a complete review of all layers to surface stale or orphaned entries.
- **Tower files**: Named `handoff-tower-NNN-YYYY-MM-DD.json`. Sealed archives that are never auto-loaded. Tombstones in the cold layer reference them if retrieval is needed.

## File Inventory

```
.claude/
├── README.md                          ← You are here
├── buffer/
│   ├── handoff.json                   Hot layer (always loaded)
│   ├── handoff-warm.json              Warm layer (selectively loaded)
│   ├── handoff-cold.json              Cold layer (on-demand)
│   └── handoff-tower-NNN-*.json       Sealed archives (if any)
└── skills/
    ├── handoff/SKILL.md               Project-specific handoff structure
    ├── onhand/SKILL.md                Project-specific on-hand priorities
    └── distill/SKILL.md               Source distillation skill (see its own README)
```

The `SKILL.md` files are instructions for Claude instances, not human documentation. The `buffer/` JSON files are the actual session state data.

## Global Project Registry

A registry at `~/.claude/buffer/projects.json` tracks all projects with buffers. Updated at each handoff. This allows `/onhand` to route between projects when invoked outside a recognized repo.

## FAQ

**"No handoff buffer found"** — Run `/handoff` at the end of your current session. The system bootstraps from there.

**Buffer seems stale** — `/onhand` compares git state against the buffer's recorded commit. If commits happened outside a session, it flags the discrepancy.

**Can I edit the buffer JSON directly?** — Yes, but be careful with ID numbering. IDs (`w:N`, `c:N`) are never reused. Editing is useful for fixing broken references or curating the concept map.

**How do I share this with another project?** — Copy the two global skill files (`~/.claude/skills/handoff/SKILL.md` and `~/.claude/skills/onhand/SKILL.md`) to the target machine. Project-specific skills are generated per-project by adding files to `<repo>/.claude/skills/`.

**What are "instance notes"?** — A free-form colleague-to-colleague briefing written by the outgoing Claude instance. Contains working-style observations, warnings, open questions, and things that surprised it. Replaced each session.

**What is "warm consolidation"?** — A compression pass that runs at every handoff. It uses the project's established vocabulary to shorten descriptions, merges entries that describe the same concept from different angles, and tightens stable entries. The warm layer iterates — same structure, richer each pass — rather than merely accumulating.
