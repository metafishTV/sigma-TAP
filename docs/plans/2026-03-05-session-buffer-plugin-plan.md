# Session Buffer Plugin — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a distributable Claude Code plugin (`session-buffer`) that provides three-layer session memory with alpha stash → sigma trunk flow, compact hooks, and multi-project support.

**Architecture:** Single plugin, mode-gated. Users choose Full or Lite scope at first run. Handoff modes: Totalize / Quicksave / Targeted. Skills auto-discovered from `skills/*/SKILL.md`, hooks from `hooks/hooks.json`. Python scripts use `${CLAUDE_PLUGIN_ROOT}` for portability. Cross-platform Python shim handles `python3` vs `python`.

**Tech Stack:** Python 3.10+ (stdlib only), Claude Code plugin system, JSON, Markdown

**Design doc:** `docs/plans/2026-03-05-session-buffer-plugin-design.md`

---

### Task 1: Scaffold plugin repo

**Files:**
- Create: `C:/Users/user/Documents/New folder/session-buffer/.claude-plugin/plugin.json`
- Create: `C:/Users/user/Documents/New folder/session-buffer/.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p "C:/Users/user/Documents/New folder/session-buffer"/{.claude-plugin,skills/{buffer,buffer-off,buffer-on},hooks,scripts}
```

**Step 2: Write plugin manifest**

Write `.claude-plugin/plugin.json`:
```json
{
  "name": "session-buffer",
  "version": "0.1.0",
  "description": "Three-layer session buffer for cross-instance continuity. Preserves decisions, open threads, concept maps, and working context across Claude Code sessions.",
  "claude_code_version": ">=1.0.0"
}
```

**Step 3: Write .gitignore**

```
__pycache__/
*.pyc
.compact_marker
```

**Step 4: Initialize git repo**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git init
git add .claude-plugin/plugin.json .gitignore
git commit -m "init: plugin scaffold"
```

**Step 5: Verify structure**

Run: `find "C:/Users/user/Documents/New folder/session-buffer" -type d | sort`
Expected: `.claude-plugin/`, `hooks/`, `scripts/`, `skills/buffer/`, `skills/buffer-off/`, `skills/buffer-on/`

---

### Task 2: Python shim (cross-platform)

**Files:**
- Create: `scripts/run_python` (Unix bash)
- Create: `scripts/run_python.bat` (Windows)

**Step 1: Write Unix shim**

Write `scripts/run_python`:
```bash
#!/usr/bin/env bash
# Cross-platform Python resolver. Tries python3 first, falls back to python.
if command -v python3 &>/dev/null; then
    exec python3 "$@"
elif command -v python &>/dev/null; then
    exec python "$@"
else
    echo "session-buffer: Python 3.10+ required but not found on PATH" >&2
    echo "Install Python from https://python.org or add python3 to PATH" >&2
    exit 1
fi
```

**Step 2: Make it executable**

Run: `chmod +x "C:/Users/user/Documents/New folder/session-buffer/scripts/run_python"`

**Step 3: Write Windows shim**

Write `scripts/run_python.bat`:
```batch
@echo off
REM Cross-platform Python resolver. Tries python3 first, falls back to python.
where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python3 %*
    exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python %*
    exit /b %ERRORLEVEL%
)
echo session-buffer: Python 3.10+ required but not found on PATH >&2
echo Install Python from https://python.org or add python3 to PATH >&2
exit /b 1
```

**Step 4: Test Unix shim**

Run: `"C:/Users/user/Documents/New folder/session-buffer/scripts/run_python" --version`
Expected: `Python 3.1x.x`

**Step 5: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add scripts/run_python scripts/run_python.bat
git commit -m "feat: cross-platform Python shim"
```

---

### Task 3: Create hooks.json

**Files:**
- Create: `hooks/hooks.json`

**Step 1: Write hooks configuration**

Write `hooks/hooks.json`:
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

**Step 2: Validate JSON**

Run: `"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "import json; json.load(open('C:/Users/user/Documents/New folder/session-buffer/hooks/hooks.json'))"`
Expected: No error

**Step 3: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add hooks/hooks.json
git commit -m "feat: compact hooks configuration (PreCompact + SessionStart)"
```

---

### Task 4: Port and generalize compact_hook.py

**Source:** `C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py`

**Files:**
- Create: `scripts/compact_hook.py`

**Step 1: Read the source file**

Read `C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py` in full.

**Step 2: Copy and apply these transformations**

1. **Strip hardcoded Python paths**: Remove any references to `C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe`. The shim handles Python resolution.

2. **Strip hardcoded buffer dir paths**: The `find_buffer_dir(cwd)` function should walk up from CWD looking for `.claude/buffer/handoff.json`. Verify it does this generically (no hardcoded project paths).

3. **Strip sigma-TAP references**: Search for any references to sigma-TAP, L-matrix, or project-specific content. Replace with generic descriptions.

4. **Update the module docstring** to:
   ```python
   """
   Session Buffer — Compact Hook

   Handles context preservation across Claude Code compaction events.

   Pre-compact: Autosaves hot layer and writes .compact_marker file.
   Post-compact: If marker exists, injects sigma trunk summary into AI context.

   Called by hooks/hooks.json via the plugin system.
   Usage: run_python compact_hook.py [pre-compact|post-compact]
   """
   ```

5. **Verify `find_buffer_dir`** walks upward generically. If it contains hardcoded paths, replace with:
   ```python
   def find_buffer_dir(start_path):
       """Walk up from start_path looking for .claude/buffer/handoff.json."""
       current = os.path.abspath(start_path)
       while True:
           candidate = os.path.join(current, '.claude', 'buffer', 'handoff.json')
           if os.path.exists(candidate):
               return os.path.join(current, '.claude', 'buffer')
           parent = os.path.dirname(current)
           if parent == current:
               return None
           current = parent
   ```

6. **Verify `build_compact_summary`** uses no project-specific field names or orientation content. It should read the hot layer generically and build a summary from whatever fields exist.

7. **Update hook setup docs** in any docstrings: Remove `~/.claude/settings.json` references. The plugin's `hooks/hooks.json` handles this now.

**Step 3: Test pre-compact**

```bash
cd "C:/Users/user/Documents/New folder/sigma-TAP-repo"
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/Documents/New folder/session-buffer/scripts/compact_hook.py" pre-compact
```
Expected: Exit 0, `.compact_marker` written in sigma-TAP's `.claude/buffer/`

**Step 4: Test post-compact (with marker)**

```bash
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/Documents/New folder/session-buffer/scripts/compact_hook.py" post-compact
```
Expected: JSON output with `POST-COMPACTION BUFFER RECOVERY` in `additional_context`, marker consumed

**Step 5: Test post-compact (without marker — silent)**

```bash
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/Documents/New folder/session-buffer/scripts/compact_hook.py" post-compact
```
Expected: JSON with empty `additional_context`

**Step 6: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add scripts/compact_hook.py
git commit -m "feat: generalized compact hook (pre/post compaction)"
```

---

### Task 5: Port and generalize buffer_manager.py

**Source:** `C:/Users/user/.claude/skills/buffer/scripts/buffer_manager.py`

**Files:**
- Create: `scripts/buffer_manager.py`

**Step 1: Read the source file**

Read `C:/Users/user/.claude/skills/buffer/scripts/buffer_manager.py` in full.

**Step 2: Copy and apply these transformations**

1. **Strip hardcoded paths**: All paths come from CLI args (`--buffer-dir`, `--memory-path`, `--project-name`). Verify no absolute paths are embedded.

2. **Strip sigma-TAP references**: Search for project-specific content. Replace with generic descriptions.

3. **Update module docstring** to:
   ```python
   """
   Session Buffer — Buffer Manager

   Mechanical operations for the three-layer session buffer (sigma trunk).
   Handles JSON merge, ID assignment, conservation enforcement, and MEMORY.md sync.

   Commands:
     handoff  — Full pipeline: update + migrate + sync (preferred)
     update   — Merge session alpha stash into hot+warm layers
     migrate  — Conservation: hot→warm→cold when bounds exceeded
     validate — Check layer sizes and schema
     sync     — MEMORY.md status sync + project registry
     read     — Parse hot layer, resolve warm pointers, output reconstruction
     next-id  — Get next sequential ID for a layer

   Usage: run_python buffer_manager.py <command> [options]
   """
   ```

4. **Verify `handoff` command** writes `_changes.json` file (not stdin) — this was a Windows fix already applied. Confirm it's in the generalized version.

5. **Verify `sync` command** handles the global project registry at `~/.claude/buffer/projects.json`, including the `scope` field (new — add if not present).

6. **Add `scope` field support**: When registering a project, include `"scope": "full"|"lite"` from the hot layer's `buffer_mode` field. Map: `"project"` → `"full"`, `"memory"` or `"minimal"` → `"lite"`.

**Step 3: Test validate command**

```bash
cd "C:/Users/user/Documents/New folder/sigma-TAP-repo"
"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/Documents/New folder/session-buffer/scripts/buffer_manager.py" validate --buffer-dir .claude/buffer/
```
Expected: Schema version, layer sizes reported, no errors

**Step 4: Test read command**

```bash
"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/Documents/New folder/session-buffer/scripts/buffer_manager.py" read --buffer-dir .claude/buffer/
```
Expected: Formatted reconstruction of sigma trunk

**Step 5: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add scripts/buffer_manager.py
git commit -m "feat: generalized buffer manager (sigma trunk operations)"
```

---

### Task 6: Create skills/buffer/SKILL.md (architecture reference)

**Source:** `C:/Users/user/.claude/skills/buffer/SKILL.md`

**Files:**
- Create: `skills/buffer/SKILL.md`

**Step 1: Read the source file**

Read `C:/Users/user/.claude/skills/buffer/SKILL.md` in full.

**Step 2: Copy and apply these transformations**

1. **Frontmatter**: Keep name `buffer`, update description:
   ```yaml
   ---
   name: buffer
   description: Three-layer session buffer (sigma trunk) for cross-instance continuity. Operations — /buffer-on (reconstruct context), /buffer-off (write handoff).
   ---
   ```

2. **Terminology replacements throughout**:
   - "delta" → "alpha stash" (or just "alpha" depending on context)
   - "the buffer" (when referring to the persistent system) → "sigma trunk"
   - "three-layer buffer" → "sigma trunk (hot / warm / cold)"
   - Add to the opening section: "The **alpha stash** is the ephemeral session intake — computed, merged into the trunk, then gone. The **sigma trunk** is the persistent knowledge: hot (always loaded), warm (selectively loaded), cold (on-demand)."

3. **Buffer Modes section**: Rename modes:
   - `"minimal"` → keep the JSON value as `"lite"`
   - `"memory"` → remove this middle tier. Collapse to two modes:
     - **Lite**: Hot + warm layers. Tracks active work, decisions, threads, instance notes. No concept maps, no convergence webs, no cold layer, no tower archival.
     - **Full**: All layers, all features. Concept maps, convergence webs, conservation, tower archival, provenance-aware consolidation.
   - Update the mode comparison table to show only Lite and Full
   - Update `buffer_mode` field to accept `"lite"` or `"full"` only
   - Mode gates throughout: replace `"Mode gate: Project mode only"` with `"Mode gate: Full only"`, replace `"Memory and project modes"` with `"Full and Lite modes"`, replace `"Minimal mode"` with `"Lite mode"`

4. **Hot Layer Schema**: Update `buffer_mode` to `"lite | full"`. All other fields stay.

5. **Warm Layer Schema — concept_map groups**: Replace sigma-TAP-specific group examples with:
   ```
   The concept_map uses named groups relevant to the project. Group names
   are defined by the project-level skill, or by the user during first Full-mode
   handoff. Examples: "core_concepts", "cross_references", "external_mappings".
   ```

6. **Script references**: Update all script paths from `~/.claude/skills/buffer/scripts/` to relative references noting the scripts live in the plugin's `scripts/` directory.

7. **Orientation field documentation**: Make the template direct and imperative:
   ```
   orientation.core_insight: One sentence. What this project IS and what it DOES. No filler.
   orientation.practical_warning: What the AI must NOT do. Specific prohibitions as imperatives.
     Template: "Do NOT [specific action]. [Specific constraint]."
   ```

**Step 3: Verify no project-specific content remains**

Run: `grep -i -E "sigma.?TAP|L-matrix|Levinas|Sartre|Deleuze|Emery|TAPS|praxis|metathes" "C:/Users/user/Documents/New folder/session-buffer/skills/buffer/SKILL.md"`
Expected: No matches

**Step 4: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add skills/buffer/SKILL.md
git commit -m "feat: sigma trunk architecture reference (generalized)"
```

---

### Task 7: Create skills/buffer-off/SKILL.md (handoff)

**Source:** `C:/Users/user/.claude/skills/buffer-off/SKILL.md`

**Files:**
- Create: `skills/buffer-off/SKILL.md`

**Step 1: Read the source file**

Read `C:/Users/user/.claude/skills/buffer-off/SKILL.md` in full.

**Step 2: Copy and apply these transformations**

1. **Frontmatter**:
   ```yaml
   ---
   name: buffer-off
   description: Write session handoff to sigma trunk. Run at session end or when context is getting full.
   ---
   ```

2. **Instance Primer** — rewrite opening to be imperative and generic:
   ```
   You are running `/buffer-off`. Extract every decision, open thread, concept
   mapping, and unresolved question from this session. Nothing implicit survives the
   handoff — if it matters, it's in the alpha stash or it's gone.
   ```

3. **Mode Selection section** — rename handoff modes:
   - "Full" → **"Totalize"** — Complete end-of-session handoff (all steps)
   - "Lite: Snapshot" → **"Quicksave"** — Fast sigma trunk checkpoint (~3 tool calls)
   - "Lite: Targeted" → **"Targeted"** — Save specific items the user names (~4 tool calls)

4. **Shared Preamble** — update terminology:
   ```
   **Read-first ordering** — scan existing sigma trunk before dialogue to prevent duplication:
   1. Read hot + warm + cold layers (parallel if possible)
   2. Scan dialogue for new content
   3. Compute the **alpha stash**: items from this session NOT already captured in any layer
   ```

5. **First-run detection** — add to Step 0, before the project skill check:
   ```
   If no buffer exists at `.claude/buffer/handoff.json`, this is a first-run.
   Before proceeding, run the first-run flow:

   1. Popup (AskUserQuestion): "Buffer scope?" → **Full** / **Lite**
      - Full: concept maps, convergence webs, conservation, tower archival
      - Lite: hot + warm, decisions and threads, no concept maps
   2. Popup (Full only): Project name + one-sentence core insight
   3. Popup: "Remote backup? Your work deserves a backup that lives somewhere safe."
      - Git remote detected → "Auto-push after each handoff?" (yes/no)
      - No remote → "Connect a GitHub repo?" (yes → guide setup / no → skip)
      - No git repo → "Initialize git for your buffer?" (yes → git init + remote / no → local only)
   4. Store scope in hot layer as `"scope": "full"|"lite"`
   5. Store remote_backup preference as `"remote_backup": true|false`
   6. Initialize layers and proceed to first handoff
   ```

6. **Rename "delta" → "alpha stash"** throughout — in changes.json schema description, in the workflow, in the preamble.

7. **Rename "Lite: Snapshot Mode"** section header to **"Quicksave Mode"**. Update commit message: `"buffer: quicksave"`.

8. **Rename "Lite: Targeted Mode"** section header to **"Targeted Mode"**. Update commit message: `"buffer: targeted save"`.

9. **Step 13 (Commit)** — add remote push:
   ```
   If `remote_backup` is true in the hot layer, follow the commit with:
   git push
   ```

10. **Mode gates**: Update `"minimal"` → `"lite"`, remove `"memory"` references. Two modes only.

11. **Script paths**: Update `~/.claude/skills/buffer/scripts/buffer_manager.py` to note it's the plugin's `scripts/buffer_manager.py`.

12. **Strip all sigma-TAP content**: Same grep check as Task 6.

**Step 3: Verify line count**

Run: `wc -l "C:/Users/user/Documents/New folder/session-buffer/skills/buffer-off/SKILL.md"`
Expected: < 420 lines

**Step 4: Verify no project-specific content**

Run: `grep -i -E "sigma.?TAP|L-matrix|Levinas|Sartre|Deleuze|Emery|TAPS|praxis|metathes" "C:/Users/user/Documents/New folder/session-buffer/skills/buffer-off/SKILL.md"`
Expected: No matches

**Step 5: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add skills/buffer-off/SKILL.md
git commit -m "feat: buffer-off skill (Totalize / Quicksave / Targeted)"
```

---

### Task 8: Create skills/buffer-on/SKILL.md (rehydration)

**Source:** `C:/Users/user/.claude/skills/buffer-on/SKILL.md`

**Files:**
- Create: `skills/buffer-on/SKILL.md`

**Step 1: Read the source file**

Read `C:/Users/user/.claude/skills/buffer-on/SKILL.md` in full.

**Step 2: Copy and apply these transformations**

1. **Frontmatter**:
   ```yaml
   ---
   name: buffer-on
   description: Reconstruct session context from sigma trunk. Run at session start.
   ---
   ```

2. **Instance Primer** — rewrite:
   ```
   You are running `/buffer-on`. Reconstruct context from the sigma trunk so you can
   work effectively without the user re-explaining everything.

   The sigma trunk has three layers: Hot (~200 lines, always loaded), Warm (~500 lines,
   selectively loaded via pointers), Cold (~500 lines, on-demand only). Load the minimum
   needed to orient.
   ```

3. **Step 0: Project Routing** — replace the first-run setup (Step 0d) with the new project selector popup:

   Replace the mode selection (which currently offers Minimal/Memory/Project) with:
   ```
   ### 0c: Project selector (AskUserQuestion popup)

   Read `~/.claude/buffer/projects.json`. Present options based on registry state:

   **One project registered:**
   - Resume [project name] (last handoff: [date])
   - Start new project
   - Start lite session

   **Multiple projects registered:**
   - Resume [most recent project] (last handoff: [date])
   - Switch project (shows full list with dates and one-line context)
   - Start new project
   - Start lite session

   **No projects registered:**
   - Start new project
   - Start lite session

   "Most recent" = highest `last_handoff` date in registry.

   If user selects "Start new project" or "Start lite session", proceed to 0d (first-run setup).
   If user selects an existing project, load its buffer_path and proceed to Step 1.
   ```

4. **Step 0d first-run setup** — simplify to match the new two-mode system:
   ```
   1. Popup: "Buffer scope?" → Full / Lite
   2. Popup (Full only): Project name + one-sentence core insight
   3. Popup: "Remote backup?" (see buffer-off first-run flow for full details)
   4. Initialize `.claude/buffer/` with scope-appropriate schemas
   5. Register in global project registry (Step 0e)
   6. Configure MEMORY.md integration (Step 0f)
   7. Confirm: "Buffer initialized in [scope] mode. Ready to go."
   8. Arm autosave
   ```

5. **MEMORY.md integration (Step 0f)** — simplify the three options to two:
   - **Full integration** — Restructure MEMORY.md into a lean orientation card
   - **No integration** — Leave MEMORY.md as-is, buffer operates independently
   Remove the "Minimal integration" middle option.

6. **Terminology replacements**: "delta" → "alpha stash", "the buffer" (persistent) → "sigma trunk" throughout.

7. **Mode gates**: Replace three-mode gates with two: `"Full only"` and `"Lite skips this step"`.

8. **Autosave Protocol — mode-specific**: Update to two modes:
   - **Lite**: Write `session_meta`, `active_work`, `open_threads`, `recent_decisions`, `instance_notes`, `natural_summary`. Skip `concept_map_digest`.
   - **Full**: All fields including `concept_map_digest`.

9. **Post-Compaction section**: Update hook setup example to reference plugin's hooks.json instead of manual settings.json config. Note that the plugin handles this automatically.

10. **Script paths**: Update to plugin-relative references.

11. **Strip all project-specific content**: Same grep check.

**Step 3: Verify no project-specific content**

Run: `grep -i -E "sigma.?TAP|L-matrix|Levinas|Sartre|Deleuze|Emery|TAPS|praxis|metathes" "C:/Users/user/Documents/New folder/session-buffer/skills/buffer-on/SKILL.md"`
Expected: No matches

**Step 4: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add skills/buffer-on/SKILL.md
git commit -m "feat: buffer-on skill (project selector + sigma trunk rehydration)"
```

---

### Task 9: Write README.md

**Files:**
- Create: `README.md`

**Step 1: Write the README**

```markdown
# session-buffer

Three-layer session memory for Claude Code. Preserves decisions, open threads,
concept maps, and working context across sessions.

## How it works

The **sigma trunk** holds your accumulated project knowledge in three layers:

- **Hot** (~200 lines) — Current session state. Always loaded.
- **Warm** (~500 lines) — Decisions archive, concept maps. Loaded selectively.
- **Cold** (~500 lines) — Historical record. On-demand only.

Each session, you compute the **alpha stash** (what's new) and merge it into the trunk.

## Install

```
/plugin install session-buffer
```

Requires Python 3.10+ on PATH.

## Quick start

**End of session:**
```
/buffer-off
```
Choose your handoff mode:
- **Totalize** — Complete handoff (concept maps, consolidation, full commit)
- **Quicksave** — Fast checkpoint (~3 tool calls)
- **Targeted** — Save specific items you name

**Start of session:**
```
/buffer-on
```
Select your project from the list. Context reconstructed automatically.

## Scope: Full vs Lite

First time you run `/buffer-off`, you choose:

- **Full** — Concept maps, convergence webs, conservation, tower archival. For research projects, multi-source analysis, deep domain work.
- **Lite** — Decisions and threads only. For everyday development, quick projects, anything that needs session continuity without research infrastructure.

Upgrade from Lite to Full anytime. No data loss.

## Project overrides

For project-specific concept map groups, terminology, and thresholds,
create `<repo>/.claude/skills/buffer/off.md`. This overrides the plugin's
generic handoff skill for that repo.

Template:
```markdown
# Project Buffer — [Your Project]

## Concept Map Groups
- group_name_1: [description]
- group_name_2: [description]

## Orientation Template
- core_insight: [what this project IS]
- practical_warning: [what the AI must NOT do]

## Thresholds
- warm_max: 800  (default: 500)
- full_scan_threshold: 3  (default: 5)
```

## Remote backup

First-run setup offers to connect a GitHub repo. If enabled,
every handoff commit is followed by `git push`.

## Compact hooks

The plugin includes automatic context preservation hooks. When Claude Code
compacts your conversation (to manage context length), the hooks:

1. **Before compaction**: Save current hot layer state + write a marker
2. **After compaction**: Inject sigma trunk summary into AI context

This happens invisibly. You never need to configure or think about it.

## Files

| File | Purpose |
|---|---|
| `.claude/buffer/handoff.json` | Hot layer (current session state) |
| `.claude/buffer/handoff-warm.json` | Warm layer (concept maps, decisions archive) |
| `.claude/buffer/handoff-cold.json` | Cold layer (historical record) |
| `.claude/buffer/handoff-tower-NNN-*.json` | Sealed archive (user-approved) |
| `~/.claude/buffer/projects.json` | Global project registry |
```

**Step 2: Commit**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git add README.md
git commit -m "docs: README with install, quick start, and configuration guide"
```

---

### Task 10: End-to-end verification

**Step 1: Verify plugin structure completeness**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
find . -type f -not -path './.git/*' | sort
```

Expected files:
```
./.claude-plugin/plugin.json
./.gitignore
./README.md
./hooks/hooks.json
./scripts/buffer_manager.py
./scripts/compact_hook.py
./scripts/run_python
./scripts/run_python.bat
./skills/buffer-off/SKILL.md
./skills/buffer-on/SKILL.md
./skills/buffer/SKILL.md
```

**Step 2: Verify all JSON files are valid**

```bash
"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "
import json, glob, os
os.chdir('C:/Users/user/Documents/New folder/session-buffer')
for f in glob.glob('**/*.json', recursive=True):
    try:
        json.load(open(f))
        print(f'OK: {f}')
    except Exception as e:
        print(f'FAIL: {f} — {e}')
"
```
Expected: All OK

**Step 3: Verify no project-specific content in any skill file**

```bash
grep -ri -E "sigma.?TAP|L-matrix|Levinas|Sartre|Deleuze|Emery-Trist|TAPS|praxis|metathes|unificit" "C:/Users/user/Documents/New folder/session-buffer/skills/"
```
Expected: No matches

**Step 4: Verify Python scripts run without import errors**

```bash
"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "exec(open('C:/Users/user/Documents/New folder/session-buffer/scripts/compact_hook.py').read())" 2>&1 | head -5
"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "exec(open('C:/Users/user/Documents/New folder/session-buffer/scripts/buffer_manager.py').read())" 2>&1 | head -5
```
Expected: No import errors (may show usage error since no args provided — that's fine)

**Step 5: Final commit with tag**

```bash
cd "C:/Users/user/Documents/New folder/session-buffer"
git log --oneline
git tag v0.1.0
```

Expected: Clean commit history, tag applied

**Step 6: Report**

Summarize: file count, total lines, any issues found during verification.
