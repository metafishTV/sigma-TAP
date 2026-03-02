# Session Continuity System (Handoff / Resume)

Persistent structured context transfer between Claude Code sessions, solving the problem of starting each session cold.

## The Problem

Claude Code sessions are ephemeral — each new instance starts with zero memory of what happened before. On complex projects, significant context window is wasted re-discovering decisions, open threads, concept evolution, and working-style notes. `MEMORY.md` helps but is static and size-limited. It cannot capture session-specific state: what was in-progress, what was decided and why, what the previous instance learned about the project and user.

## Quick Start

```
/resume      # Start of session — reconstructs context from last handoff
/handoff     # End of session — saves state for next instance
```

First time? Just run `/handoff` at the end of your first session. The system bootstraps from there. Next session, `/resume` picks up where you left off.

## Architecture

Content migrates downward when layer bounds are exceeded. Resume reads upward selectively — the hot layer always, warm/cold only when a pointer says "look here." Initial context loading stays under ~200 lines.

```
              ALWAYS loaded (~200 lines max)
         ┌─────────────────────────────────┐
         │       HOT: handoff.json         │
         │  session state, decisions,       │
         │  open threads, orientation,      │
         │  instance notes, natural summary │
         └──────────┬──────────────────────┘
                    │ "see": ["w:N"]
                    ▼
         SELECTIVELY loaded via pointers (~500 lines max)
         ┌─────────────────────────────────┐
         │    WARM: handoff-warm.json      │
         │  concept map (all groups),       │
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

## What `/handoff` Does

1. Reads current buffer state
2. Gathers git metadata (commit, branch, modified files, test results)
3. Summarizes what you worked on (inferred from conversation — does not ask)
4. Logs decisions made this session with rationale
5. Lists open threads and deferred items
6. Validates concept map entries for changes, new additions, or flagged items
7. Writes instance notes — a colleague-to-colleague briefing for the next instance
8. Writes a plain-English natural summary (no encoding or abbreviations)
9. Enforces size bounds (migrates overflow to lower layers automatically)
10. Writes all layer files to `.claude/buffer/`
11. Commits to git
12. Confirms completion

## What `/resume` Does

1. Reads hot layer only (fast startup, ~200 lines)
2. Checks git state against buffer (detects if commits happened outside a session)
3. Presents session state: phase, completed work, next action
4. Follows flagged pointers into warm/cold layers (selective, not exhaustive)
5. Checks if a full rescan is due (configurable, default every 5 sessions)
6. Surfaces instance notes from previous instance
7. Reads MEMORY.md for project baseline
8. Asks what you want to work on

## The Two-File Skill Pattern

| File | Location | Owns |
|------|----------|------|
| Global handoff | `~/.claude/skills/handoff/SKILL.md` | Generic 12-step process, three-layer schema, conservation enforcement |
| Project handoff | `<repo>/.claude/skills/handoff/SKILL.md` | Concept map groups, terminology, validation rules, orientation template |
| Global resume | `~/.claude/skills/resume/SKILL.md` | Generic 8-step reconstruction, pointer-following algorithm |
| Project resume | `<repo>/.claude/skills/resume/SKILL.md` | Resume priorities, source material review, forward note surfacing |

**Global skills** define the process (reusable across any project). **Project skills** define the structure (what the concept map looks like, what validation means for this domain). When invoked, the global skill checks for a project override first and defers if found.

**To use in your own project**: You only need the two global skill files. On first `/handoff`, a generic buffer is created. Add project skills when you want project-specific concept map groups, validation rules, or resume priorities.

## Configuration

| Parameter | Default | Set in | Effect |
|-----------|---------|--------|--------|
| Hot max lines | 200 | Global handoff SKILL.md | Triggers migration to warm |
| Warm max lines | 500 | Global handoff SKILL.md | Triggers migration to cold |
| Cold max lines | 500 | Global handoff SKILL.md | Triggers archival questionnaire |
| Full scan threshold | 5 sessions | Hot layer (`full_scan_threshold`) | Prompts full warm+cold review at resume |

Concept map groups, validation rules, and orientation templates are configured in the project-level handoff SKILL.md.

## Buffer Maintenance

- **Archival questionnaire**: When the cold layer exceeds its bound, the system presents a dependency map and lets you choose what to archive — you pick the ratio, direction, and specific entries. Archived entries go to sealed tower files.
- **Full rescan**: Every N sessions (configurable), `/resume` offers a complete review of all layers to surface stale or orphaned entries.
- **Tower files**: Named `handoff-tower-NNN-YYYY-MM-DD.json`. Sealed archives that are never auto-loaded. Tombstones in the cold layer reference them if retrieval is needed.
- **v1 migration**: If upgrading from v1 format, `/handoff` detects the old schema and migrates automatically to the three-layer format.

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
    ├── resume/SKILL.md                Project-specific resume priorities
    └── distill/SKILL.md               Source distillation skill (see its own README)
```

The `SKILL.md` files are instructions for Claude instances, not human documentation. The `buffer/` JSON files are the actual session state data.

## FAQ

**"No handoff buffer found"** — Run `/handoff` at the end of your current session. The system bootstraps from there.

**"Found v1 buffer"** — Run `/handoff` to auto-migrate to v2 three-layer format. A backup of the v1 file is preserved.

**Buffer seems stale** — `/resume` compares git state against the buffer's recorded commit. If commits happened outside a session, it flags the discrepancy.

**Can I edit the buffer JSON directly?** — Yes, but be careful with ID numbering. IDs (`w:N`, `c:N`) are never reused. Editing is useful for fixing broken references or curating the concept map.

**How do I share this with another project?** — Copy the two global skill files (`~/.claude/skills/handoff/SKILL.md` and `~/.claude/skills/resume/SKILL.md`) to the target machine. Project-specific skills are generated per-project by adding files to `<repo>/.claude/skills/`.

**What are "instance notes"?** — A free-form colleague-to-colleague briefing written by the outgoing Claude instance. Contains working-style observations, warnings, open questions, and things that surprised it. Replaced each session.
