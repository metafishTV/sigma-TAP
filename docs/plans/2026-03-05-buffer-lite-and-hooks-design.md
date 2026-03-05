# Buffer Lite Mode + Compact Hooks Fix

**Date**: 2026-03-05
**Scope**: Two features — (1) full/lite buffer-off mode split, (2) compact hook config fix
**Files affected**: `~/.claude/skills/buffer-off/SKILL.md`, `~/.claude/settings.json`, `~/.claude/skills/buffer/scripts/compact_hook.py`, `~/.claude/skills/buffer/scripts/buffer_manager.py`

---

## Feature 1: Full/Lite Buffer-Off

### Problem

`/buffer-off` is always a full 14-step ceremony (~7 tool calls compressed). Users need a lighter option for mid-session checkpoints — more controlled than autosave, less heavy than full handoff.

### Design

When `/buffer-off` is invoked, step 0 prompts:

- **Full** — complete handoff (existing process)
- **Lite: Snapshot** — quick hot-layer checkpoint
- **Lite: Targeted** — user specifies what to capture

### Shared Preamble (all modes)

Read-first ordering prevents duplication:

1. Read hot + warm + cold layers (1 parallel call)
2. Scan dialogue for new content
3. Compute delta: only items NOT in any layer

### Mode: Full

Existing process. Steps 1-8 cognitive, then `buffer_manager.py handoff`. No changes except read-first ordering replaces step 1.

### Mode: Lite Snapshot

Hot-layer-only update, ~3 tool calls:

1. Read all layers (1 call)
2. Scan dialogue, compute delta (cognitive)
3. Update hot: `active_work`, `recent_decisions`, `open_threads`, `instance_notes`, `natural_summary`
4. Write hot directly (1 Write)
5. Commit (1 Bash)

**Skips**: concept map (step 6), warm consolidation (6b), conservation (9), MEMORY.md sync (11), registry (12).

### Mode: Lite Targeted

Interactive, ~4 tool calls:

1. Read all layers (1 call)
2. Prompt user: "What to capture?" (1 AskUserQuestion)
3. Compose targeted entries from description (cognitive)
4. Write hot (1 Write)
5. Commit (1 Bash)

Same skips as snapshot. AI captures only what user specifies.

### SKILL.md Token Budget

Current `buffer-off/SKILL.md` is 334 lines. With lite mode additions, target < 380 lines. Strategy:

- Shared preamble documented once
- Mode branches as compact bullet lists, not prose
- Reference `buffer_manager.py` for mechanics, don't re-document
- Lite modes are described in ~20 lines each

---

## Feature 2: Compact Hooks Fix

### Root Cause

| Hook | Config | Bug | Fix |
|------|--------|-----|-----|
| `PreCompact` | `"matcher": ""` | Empty string matches nothing | Two entries: `"manual"` + `"auto"` |
| `SessionStart` | `"matcher": "compact"` | Matcher not supported | Remove matcher entirely |

### Settings.json Fix

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "manual",
        "hooks": [{ "type": "command", "command": "...compact_hook.py pre-compact", "timeout": 30 }]
      },
      {
        "matcher": "auto",
        "hooks": [{ "type": "command", "command": "...compact_hook.py pre-compact", "timeout": 30 }]
      }
    ],
    "SessionStart": [
      {
        "hooks": [{ "type": "command", "command": "...compact_hook.py post-compact", "timeout": 30 }]
      }
    ]
  }
}
```

### compact_hook.py Changes

1. `cmd_post_compact`: Guard on `.compact_marker` — if no marker, output empty context and exit silently. Prevents recovery injection on normal session starts.
2. `cmd_pre_compact`: Remove exit-2 blocking logic (PreCompact cannot block compaction per docs).

### Caveat: PreCompact Untested

`SessionStart` is confirmed working (system reminder evidence). `PreCompact` is documented but unverified on Windows. Design degrades gracefully:

- **If PreCompact works**: hot layer saved before compaction, marker written, SessionStart injects recovery.
- **If PreCompact doesn't fire**: no marker written, SessionStart sees no marker, outputs empty context. No harm, but no pre-compaction save.
- **Fallback**: SessionStart alone can still inject buffer context unconditionally (without marker check) if PreCompact proves unreliable. This is noisier but guarantees recovery.

### Test Plan

1. Fix settings.json config
2. Start a session, work until auto-compaction triggers
3. Check system reminders for `PreCompact` hook output
4. Check system reminders for `SessionStart` hook output
5. Verify `.compact_marker` lifecycle (created by pre, consumed by post)
6. If PreCompact doesn't fire: remove marker guard from post-compact, inject on every SessionStart instead

---

## Implementation Order

1. Fix `~/.claude/settings.json` hooks (5 min)
2. Update `compact_hook.py` — marker guard + remove exit-2 (10 min)
3. Test compact hooks empirically (separate session)
4. Update `buffer-off/SKILL.md` with mode selector + lite workflows (20 min)
5. Test lite snapshot flow end-to-end (10 min)
6. Test lite targeted flow end-to-end (10 min)
