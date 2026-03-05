# Buffer Lite Mode + Compact Hooks Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add full/lite mode selector to `/buffer-off` and fix compact hooks so autosave actually fires.

**Architecture:** Two independent changes. (1) SKILL.md gets a mode gate at step 0 and two lite-mode branches. (2) `settings.json` hooks get corrected matchers; `compact_hook.py` gets a marker guard on post-compact and drops the exit-2 block on pre-compact.

**Tech Stack:** Python 3.12 (stdlib only), Claude Code hooks (settings.json), SKILL.md (markdown)

---

### Task 1: Fix settings.json hook matchers

**Files:**
- Modify: `C:/Users/user/.claude/settings.json:2-27` (hooks section only)

**Step 1: Edit the hooks section**

Replace the entire `hooks` block with corrected matchers. `PreCompact` needs two entries (`manual` + `auto`). `SessionStart` drops its matcher.

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "manual",
        "hooks": [
          {
            "type": "command",
            "command": "\"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe\" \"C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py\" pre-compact",
            "timeout": 30
          }
        ]
      },
      {
        "matcher": "auto",
        "hooks": [
          {
            "type": "command",
            "command": "\"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe\" \"C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py\" pre-compact",
            "timeout": 30
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe\" \"C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py\" post-compact",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**Step 2: Verify JSON is valid**

Run: `"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "import json; json.load(open('C:/Users/user/.claude/settings.json'))"`
Expected: No error

---

### Task 2: Fix compact_hook.py — pre-compact

**Files:**
- Modify: `C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py:137-140`

**Step 1: Remove exit-2 blocking logic**

Replace the corrupt-buffer block (lines 137-140):

Old:
```python
    # Validate hot layer is not corrupt (block compaction if so)
    if not isinstance(hot, dict) or 'schema_version' not in hot:
        print("compact_hook: hot layer corrupt or missing schema_version", file=sys.stderr)
        sys.exit(2)  # Block compaction
```

New:
```python
    # Validate hot layer structure (warn but don't block — PreCompact can't block)
    if not isinstance(hot, dict) or 'schema_version' not in hot:
        print("compact_hook: hot layer corrupt or missing schema_version", file=sys.stderr)
        sys.exit(0)  # Can't block, just skip
```

**Step 2: Test pre-compact manually**

Run: `echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" pre-compact`
Expected: Exit 0, `.compact_marker` exists in buffer dir

**Step 3: Verify marker was written**

Run: `type "C:\Users\user\Documents\New folder\sigma-TAP-repo\.claude\buffer\.compact_marker"`
Expected: Today's date (2026-03-05)

---

### Task 3: Fix compact_hook.py — post-compact marker guard

**Files:**
- Modify: `C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py:464-516`

**Step 1: Add marker guard at top of cmd_post_compact**

Replace the function body. After finding the buffer dir, check for `.compact_marker`. If no marker, emit empty context (silent on normal session starts).

New `cmd_post_compact`:
```python
def cmd_post_compact(hook_input):
    """Inject buffer context after compaction (only if marker exists)."""
    cwd = hook_input.get('cwd', os.getcwd())
    buffer_dir = find_buffer_dir(cwd)

    empty_output = {
        "additional_context": "",
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": ""
        }
    }

    if not buffer_dir:
        json.dump(empty_output, sys.stdout, ensure_ascii=False)
        sys.exit(0)

    # Guard: only inject if pre-compact wrote a marker
    marker_path = os.path.join(buffer_dir, '.compact_marker')
    if not os.path.exists(marker_path):
        json.dump(empty_output, sys.stdout, ensure_ascii=False)
        sys.exit(0)

    hot_path = os.path.join(buffer_dir, 'handoff.json')
    hot = read_json(hot_path)

    if not hot:
        json.dump(empty_output, sys.stdout, ensure_ascii=False)
        sys.exit(0)

    # Detect warm-max override
    warm_max = detect_warm_max(cwd)

    # Build concise summary for injection
    context = build_compact_summary(hot, buffer_dir, warm_max)

    # Clean up marker
    try:
        os.remove(marker_path)
    except OSError:
        pass

    # Output JSON for hook system
    output = {
        "additional_context": context,
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context
        }
    }
    json.dump(output, sys.stdout, ensure_ascii=False)
    sys.exit(0)
```

**Step 2: Test post-compact with marker present**

Run: `echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" post-compact`
Expected: JSON output with `POST-COMPACTION BUFFER RECOVERY` in `additional_context`

**Step 3: Verify marker was consumed**

Run: `dir "C:\Users\user\Documents\New folder\sigma-TAP-repo\.claude\buffer\.compact_marker" 2>&1`
Expected: File not found (marker consumed)

**Step 4: Test post-compact WITHOUT marker (normal session start)**

Run: `echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" post-compact`
Expected: JSON output with empty `additional_context` (silent)

**Step 5: Commit**

```bash
git add -p  # Nothing in repo — these are global skill files
```

Note: compact_hook.py lives in `~/.claude/skills/`, outside the repo. No git commit needed. Verify the edits are saved.

---

### Task 4: Update buffer-off SKILL.md — add mode selector

**Files:**
- Modify: `C:/Users/user/.claude/skills/buffer-off/SKILL.md:1-20` (add mode selector before step 0)

**Step 1: Replace the Instance Primer section (lines 6-20)**

Insert the mode selector as the new first action. The existing step 0 (project skill check) moves after mode selection.

New content for lines 6-20:
```markdown
## Mode Selection (FIRST — before anything else)

Present this choice to the user:

> **Buffer mode?**
> - **Full** — Complete end-of-session handoff (all steps)
> - **Lite: Snapshot** — Quick hot-layer checkpoint (~3 tool calls)
> - **Lite: Targeted** — Save specific items the user names (~4 tool calls)

Then follow the selected mode below. All modes begin with the **Shared Preamble**.

## Shared Preamble (all modes)

**Read-first ordering** — prevents cross-layer duplication:

1. Read hot + warm + cold layers
2. Scan dialogue for new content
3. Compute delta: only items NOT already in any layer

## Instance Primer
```

**Step 2: Verify line count stays under 380**

Run: `wc -l "C:/Users/user/.claude/skills/buffer-off/SKILL.md"`
Expected: < 380

---

### Task 5: Add Lite Snapshot instructions to SKILL.md

**Files:**
- Modify: `C:/Users/user/.claude/skills/buffer-off/SKILL.md` (insert after Step 14)

**Step 1: Add Lite Snapshot section at end of file**

```markdown
---

## Lite: Snapshot Mode

After the Shared Preamble (read all layers, scan dialogue, compute delta):

1. **Update hot layer fields**: `active_work`, `recent_decisions`, `open_threads`, `instance_notes`, `natural_summary`, `session_meta`
2. **Write** `handoff.json` directly (1 Write call)
3. **Commit**: `git add .claude/buffer/handoff.json && git commit -m "buffer-lite: snapshot"`
4. **Confirm**: "Lite snapshot written and committed."

**Skips**: concept map (step 6), warm consolidation (6b), conservation (9), MEMORY.md sync (11), registry (12).
```

**Step 2: Verify line count**

Run: `wc -l "C:/Users/user/.claude/skills/buffer-off/SKILL.md"`
Expected: < 395

---

### Task 6: Add Lite Targeted instructions to SKILL.md

**Files:**
- Modify: `C:/Users/user/.claude/skills/buffer-off/SKILL.md` (append after Lite Snapshot)

**Step 1: Add Lite Targeted section**

```markdown
---

## Lite: Targeted Mode

After the Shared Preamble:

1. **Ask user**: "What do you want to capture?" (AskUserQuestion, free-text)
2. **Compose** entries from the user's description only — do not scan full dialogue
3. **Merge** into hot layer (add to `recent_decisions`, `open_threads`, or `instance_notes` as appropriate)
4. **Write** `handoff.json` directly (1 Write call)
5. **Commit**: `git add .claude/buffer/handoff.json && git commit -m "buffer-lite: targeted save"`
6. **Confirm**: "Targeted save written and committed."

**Same skips as Snapshot.** The difference: AI captures only what the user specified, not the full dialogue delta.
```

**Step 2: Verify final line count**

Run: `wc -l "C:/Users/user/.claude/skills/buffer-off/SKILL.md"`
Expected: < 410

**Step 3: Commit design doc update**

```bash
cd "C:/Users/user/Documents/New folder/sigma-TAP-repo"
git add docs/plans/2026-03-05-buffer-lite-and-hooks-plan.md
git commit -m "docs: implementation plan for buffer lite + hooks fix"
```

---

### Task 7: End-to-end test — compact hooks

**Step 1: Clean any stale marker**

Run: `del "C:\Users\user\Documents\New folder\sigma-TAP-repo\.claude\buffer\.compact_marker" 2>NUL`

**Step 2: Test full pre→post cycle**

```bash
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" pre-compact
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" post-compact
```

Expected:
- First command: exit 0, marker written
- Second command: JSON with recovery context, marker consumed

**Step 3: Test that normal session start is silent**

```bash
echo {} | "C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" "C:/Users/user/.claude/skills/buffer/scripts/compact_hook.py" post-compact
```

Expected: JSON with empty `additional_context`

**Step 4: Live test (separate session)**

Start a new Claude Code session. Work until auto-compaction triggers. Observe:
- Does `PreCompact:auto` appear in system reminders?
- Does `SessionStart` inject recovery context after compaction?
- If not: remove marker guard (Task 3 fallback), inject unconditionally

---

### Task 8: End-to-end test — lite snapshot

**Step 1: In a session, invoke `/buffer-off`**

Expected: Mode selector prompt appears (Full / Lite: Snapshot / Lite: Targeted)

**Step 2: Select "Lite: Snapshot"**

Expected:
- AI reads all 3 layers
- AI scans dialogue
- AI writes hot layer directly (no warm consolidation, no concept map)
- Commit created

**Step 3: Verify hot layer updated**

Run: `"C:/Users/user/AppData/Local/Programs/Python/Python312/python.exe" -c "import json; h=json.load(open('C:/Users/user/Documents/New folder/sigma-TAP-repo/.claude/buffer/handoff.json')); print(h.get('natural_summary','')[:100])"`
Expected: Summary reflects current session state
