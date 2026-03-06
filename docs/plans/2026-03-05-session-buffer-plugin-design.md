# Session Buffer Plugin — Design Document

**Date:** 2026-03-05
**Approach:** A — Single plugin, mode-gated
**Status:** Design approved

---

## Overview

Convert the session buffer system into a distributable Claude Code plugin: **`session-buffer`**. Ships the full three-layer persistent memory system. Users pick their depth (Full or Lite) at first run; handoff thoroughness (Totalize / Quicksave / Targeted) every session.

Fully generalized — no project-specific content. Project customization via repo-level skill overrides.

---

## Naming

| Term | What it is |
|---|---|
| **Alpha stash** | Ephemeral session intake. Computed from dialogue minus what's already in the trunk. Merged, then gone. |
| **Sigma trunk** | Persistent layers: hot / warm / cold. The accumulated learning across sessions. |
| **Flow** | dialogue → alpha stash → sigma trunk |

Sigma (σ) = learning function. The trunk is learning made durable.

---

## Plugin Structure

```
session-buffer/
  .claude-plugin/
    plugin.json
  skills/
    buffer/
      SKILL.md              # Architecture reference (schemas, constraints, ID rules)
    buffer-off/
      SKILL.md              # Handoff skill (mode selector, all steps, quicksave/targeted)
    buffer-on/
      SKILL.md              # Rehydration skill (project selector, context injection)
  hooks/
    hooks.json              # PreCompact + SessionStart compact hooks
  scripts/
    buffer_manager.py       # JSON merge, conservation, ID assignment, MEMORY.md sync
    compact_hook.py         # Pre/post compaction: marker write + context injection
    run_python              # Cross-platform shim (tries python3, falls back to python)
    run_python.bat          # Windows variant
  README.md                 # Installation, quick start, configuration, override template
```

### Manifest (`plugin.json`)

```json
{
  "name": "session-buffer",
  "version": "0.1.0",
  "description": "Three-layer session buffer for cross-instance continuity. Preserves decisions, open threads, concept maps, and working context across Claude Code sessions.",
  "claude_code_version": ">=1.0.0"
}
```

**Auto-discovery:** Skills from `skills/*/SKILL.md`, hooks from `hooks/hooks.json`. No explicit registration.

**Dependency:** Python 3.10+ on PATH. Scripts use stdlib only.

---

## Hooks & Portability

### `hooks/hooks.json`

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "manual",
        "hooks": [{
          "type": "command",
          "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/run_python\" \"${CLAUDE_PLUGIN_ROOT}/scripts/compact_hook.py\" pre-compact",
          "timeout": 30
        }]
      },
      {
        "matcher": "auto",
        "hooks": [{
          "type": "command",
          "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/run_python\" \"${CLAUDE_PLUGIN_ROOT}/scripts/compact_hook.py\" pre-compact",
          "timeout": 30
        }]
      }
    ],
    "SessionStart": [
      {
        "hooks": [{
          "type": "command",
          "command": "\"${CLAUDE_PLUGIN_ROOT}/scripts/run_python\" \"${CLAUDE_PLUGIN_ROOT}/scripts/compact_hook.py\" post-compact",
          "timeout": 30
        }]
      }
    ]
  }
}
```

**Portability:**
- `${CLAUDE_PLUGIN_ROOT}` replaces hardcoded absolute paths
- `run_python` shim tries `python3` first, falls back to `python` (Windows compatibility)
- Scripts detect OS internally for path separator handling

**Compact hook chain:**
1. PreCompact (manual or auto) writes `.compact_marker` to buffer dir
2. SessionStart checks for marker — if present, builds sigma trunk summary, injects into AI context, consumes marker
3. No marker = normal session start, silent (empty context injection)

---

## Buffer Scope: Full vs Lite

First-run choice, stored in hot layer as `"scope": "full"|"lite"`.

| Feature | Full | Lite |
|---|---|---|
| Hot layer (orientation, active work, decisions, threads) | yes | yes |
| Warm layer (decisions archive, validation log) | yes | yes |
| Cold layer (archived decisions, tower) | yes | no |
| Concept maps | yes | no |
| Convergence webs | yes | no |
| Conservation enforcement (migration between layers) | yes | no |
| Tower archival | yes | no |
| MEMORY.md sync | yes | optional |
| Project skill overrides | yes | no |

**Upgrade path:** Lite → Full is a one-time operation. Adds concept map skeleton to warm layer, flips `scope` to `full`. No data loss.

---

## Handoff Modes

Presented every `/buffer-off` via AskUserQuestion popup:

| Mode | What it does | Tool calls |
|---|---|---|
| **Totalize** | Complete end-of-session handoff. All steps: alpha stash computation, concept maps (Full only), warm consolidation, conservation, MEMORY.md sync, registry update, commit. | ~7 |
| **Quicksave** | Fast hot-layer checkpoint. Reads all layers, computes alpha stash, writes hot, commits. Skips concept maps, consolidation, conservation. | ~3 |
| **Targeted** | Saves specific items the user names. Asks what to capture, writes only those items to hot, commits. | ~4 |

All modes begin with the **shared preamble**: read all sigma trunk layers → scan dialogue → compute alpha stash (items not already in any layer).

---

## User Experience Flow

### First `/buffer-off` (new project)

All prompts via AskUserQuestion popup:

1. **Scope:** "Full or Lite?"
   - Full — concept maps, convergence webs, conservation, tower archival
   - Lite — hot + warm, decisions and threads, no concept maps
2. **Project identity (Full only):** Project name + one-sentence core insight (seeds `orientation.core_insight`)
3. **Remote backup:**
   - Git remote detected → "Auto-push buffer after each handoff?"
   - No remote → "Connect a GitHub repo for remote backup? Your work deserves a backup that lives somewhere safe."
   - No git repo (standalone) → "Initialize a git repo for your buffer and connect to GitHub?"
4. **Handoff mode:** Totalize / Quicksave / Targeted
5. Creates `.claude/buffer/` with initialized layers
6. Runs selected handoff mode

### Every subsequent `/buffer-off`

1. Popup: **Totalize** / **Quicksave** / **Targeted**
2. Runs selected mode against existing sigma trunk
3. If `remote_backup: true`, pushes after commit

### `/buffer-on` (session start)

Popup options vary by registry state:

| Registry state | Options |
|---|---|
| One project | **Resume [project name]** / **Start new project** / **Start lite session** |
| Multiple projects | **Resume [most recent]** / **Switch project** (shows list) / **Start new project** / **Start lite session** |
| No projects | **Start new project** / **Start lite session** |

After selection: loads sigma trunk, reconstructs orientation, injects context. Silent after the popup.

---

## Project Registry

Global index at `~/.claude/buffer/projects.json`:

```json
{
  "schema_version": 1,
  "projects": {
    "my-project": {
      "buffer_path": "/path/to/repo/.claude/buffer/",
      "scope": "full",
      "last_handoff": "2026-03-05",
      "project_context": "One sentence: what this project IS and what it DOES.",
      "remote_backup": true
    },
    "quick-scripts": {
      "buffer_path": "/home/user/.claude/buffer/standalone/quick-scripts/",
      "scope": "lite",
      "last_handoff": "2026-03-04",
      "project_context": "Misc utility scripts",
      "remote_backup": false
    }
  }
}
```

**Buffer locations:**

| Situation | Path |
|---|---|
| In a git repo (Full or Lite) | `<repo>/.claude/buffer/` |
| Standalone (no repo) | `~/.claude/buffer/standalone/<dirname>/` |

Registry tracks all projects regardless of scope. `/buffer-on` reads registry to present project list.

---

## Skill Generalization

All three SKILL.md files stripped of project-specific content:

**`skills/buffer/SKILL.md`** (architecture reference):
- Layer schemas, size constraints, ID rules, conservation protocol — kept as-is
- Concept map structure — generic. Groups are user-defined via project override
- Examples use placeholder projects, not specific domains
- Terminology: alpha stash, sigma trunk throughout

**`skills/buffer-off/SKILL.md`** (handoff):
- Mode selector: Totalize / Quicksave / Targeted
- Shared preamble: compute alpha stash from all layers
- All 14 steps retained, gated by scope (Full/Lite)
- Instance primer: imperative, specific, no hedging — "Extract every decision, open thread, concept mapping, and unresolved question. Nothing implicit survives the handoff — if it matters, it's in the alpha stash or it's gone."
- Project override check (Step 0) retained — `<repo>/.claude/skills/buffer/off.md`

**`skills/buffer-on/SKILL.md`** (rehydration):
- Project selector popup (see UX flow above)
- Layer loading, orientation reconstruction, context injection
- No project-specific warm-up prompts

**Orientation fields** (in hot layer, user-written):
- `core_insight`: One sentence. What this project IS and what it DOES. No filler.
- `practical_warning`: What the AI must NOT do. Specific prohibitions as imperatives.
- `why_keys`: Optional. Named anchors the AI must not lose. Key for projects with irreducible terms.

---

## MEMORY.md Integration

Hot layer carries `memory_config`:

```json
{
  "integration": "full"|"minimal"|"none",
  "path": "~/.claude/projects/<project-path>/memory/MEMORY.md"
}
```

- `full`: Status sync + promoted entry sync after each Totalize
- `minimal`: Status sync only
- `none`: No sync

First handoff detects MEMORY.md existence and offers integration setup.

---

## Scope Boundaries

**In the plugin:**
- Three skills (buffer, buffer-off, buffer-on)
- Two scripts (buffer_manager.py, compact_hook.py)
- Python shim (run_python / run_python.bat)
- Hooks (PreCompact manual+auto, SessionStart)
- README

**Not in the plugin:**
- Distill — separate plugin, separate timeline. Integrates with concept maps but ships independently.
- Project-level overrides — user creates per-repo. Plugin documents how.
- MEMORY.md — owned by Claude Code. Plugin syncs to it, doesn't own it.

---

## Migration (existing users)

Existing setups stay as-is. The plugin is a clean-room generalization.

Once published:
1. New users: install plugin, get generic skills + hooks immediately
2. Existing users: install plugin, their project-level override (`<repo>/.claude/skills/buffer/off.md`) takes precedence over plugin's generic handoff
3. `~/.claude/settings.json` hooks replaced by plugin's `hooks.json` — one less manual config

---

## Versioning

SemVer. `0.1.0` first release. Buffer schema stays at v2 (layers unchanged).
