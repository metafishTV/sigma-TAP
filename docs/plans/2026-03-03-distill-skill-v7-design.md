# Distill Skill v7 — Bundled Scripts, Pass Restructuring, UX Improvements

> Date: 2026-03-03
> Status: implemented
> Scope: `~/.claude/skills/distill/` (global skill + new scripts directory)

## Summary

Seven improvements to the global distill skill, all additive — no breaking changes to existing project skills or distillation artifacts.

## Changes

### 1. Bundled Scripts (`~/.claude/skills/distill/scripts/`)

Ship three ready-to-use Python scripts alongside SKILL.md. Each is self-contained, takes CLI arguments, and writes output to predictable file paths.

**Files:**
- `distill_scan.py` — PDF content profiling (page-by-page classification)
- `distill_extract.py` — UTF-8 text extraction with encoding safety
- `distill_figures.py` — Cropped figure extraction with verification manifest

**Interface contract:**
```
python distill_scan.py <pdf_path> [--output _distill_scan.json]
python distill_extract.py <pdf_path> --scan _distill_scan.json [--output _distill_text.txt]
python distill_figures.py <pdf_path> --scan _distill_scan.json --outdir <figures_dir> [--manifest _manifest.json]
```

**SKILL.md impact:** Replace ~180 lines of embedded code blocks with ~5 lines of invocation instructions per script. Include a "modification protocol" note: if the script needs edge-case adaptation, copy to repo, modify the copy, note the change in Known Issues. Never modify the global copy.

**Token savings:** ~175 lines removed from SKILL.md. Scripts are only read when an instance needs to understand or modify them.

### 2. Remove Version References

Strip all version labels from user-facing text:

| Current text | Replacement |
|---|---|
| `Pre-v2 (check: tooling profile has fewer than 7 entries)` | `If the tooling profile has fewer than 7 entries` |
| `Pre-v6 (check: no project_map_type in Configuration)` | `If no project_map_type in Configuration` |
| `This is a v2 refinement` | Remove sentence entirely |

Version history belongs in a changelog/readme, not in operational instructions. The upgrade logic stays — it just uses feature detection instead of version labels.

### 3. First-Run Greeting and Orientation

New **Step -1** fires whenever `/distill` is invoked, before Step 0:

```
When the user invokes /distill, begin with:

"I can help you distill source documents — PDFs, web articles, or images —
into structured, searchable knowledge artifacts.

What would you like to do?"

Options:
- [Get oriented] — Explain how the tool works and what it produces.
- [Drop a document] — Give me a file/URL and let's start.
- [Pure distillation] — Just extract and summarize, no project tracking.

If "Get oriented": provide a brief explanation (what distillation produces,
what differentiation does, what project maps are), then return to the
choice between "Drop a document" and "Pure distillation."

If "Drop a document": proceed to Step 0 (project skill check) as normal.

If "Pure distillation": set pure_mode = true, skip differentiation
entirely, proceed directly to Source Label Convention → Extraction.
```

### 4. Pure Distillation Fast Path

When `pure_mode = true`:
- No project skill generated or consulted
- No differentiation questionnaire
- No interpretation files
- No buffer updates (not even offered)
- No MEMORY.md updates
- Produces ONLY: `[Source-Label].md` (distillation) + INDEX.md entry + figures/
- Uses global skill directly, every invocation
- Source Label Convention still applies (quality naming regardless of mode)
- All 5 analytic passes still run EXCEPT Pass 4 (Relational) — there is no project to relate to
- Output template is the same minus the conditional sections that require project context

This is lighter than `project_map_type = none` (which still generates a project skill). Pure mode generates nothing persistent except the distillation artifacts themselves.

### 5. Buffer Update Toggle (Per-Session)

After presenting the interpretation file to the user (existing behavior), add a prompt:

```
"Should I update your project buffer with these findings now, or just
keep the files?"

- [Update buffer now] — Write to concept_map / themes / entities +
  convergence web as configured.
- [Files only] — Skip buffer updates. You can integrate via /handoff later.
```

This fires every time, regardless of project_map_type. Users who prefer to batch their buffer updates through `/handoff` can always choose "Files only."

If the user chooses "Files only": skip Post-Distillation Update steps 2 (Buffer), 3 (MEMORY.md), and 4 (Convergence Web). Step 1 (INDEX.md) always runs — it's a file, not a buffer.

### 6. Analytic / Anolytic Pass Restructuring

Current 3-pass structure → 5-pass structure:

| Pass | Name | Operation | Output feeds |
|------|------|-----------|-------------|
| 1 | **Extraction** | Raw text from PDF/web/image pipeline | Raw material for all subsequent passes |
| 2 | **Analytic** | Decompose the whole into parts: concepts, claims, definitions, mechanisms, boundary conditions, evidence | Key Concepts table (term, definition, significance) |
| 3 | **Anolytic** | Recompose the parts into a whole: reconstruct the argument as a coherent totality, how claims relate to each other, what the source *means* beyond its enumerable parts | Core Argument, Theoretical & Methodological Implications, Equations & Formal Models |
| 4 | **Relational** | Read the reconstructed whole against the project framework (adapts to project_map_type) | Interpretation file (Integration Points / Thematic Relevance / Narrative Elements) |
| 5 | **Style** | Characterize the source's register, tone, density | Distillation header metadata |

**Skip rules:**
- Pass 4 skipped if `project_map_type = none` or `pure_mode = true`
- Pass 5 always runs (style is source-intrinsic, not project-dependent)

**Methodological note for SKILL.md:** The analytic pass asks "what are the parts?" The anolytic pass asks "how do the parts constitute a whole?" Neither pass editorializes or relates to the project — they read the source on its own terms. The relational pass then brings the project into dialogue with the reconstructed source.

### 7. SKILL.md Structural Cleanup

Net effect of all changes on SKILL.md line count:
- Removed: ~180 lines (embedded code blocks replaced by script invocations)
- Removed: ~5 lines (version references)
- Added: ~20 lines (greeting/orientation)
- Added: ~15 lines (pure distillation fast path)
- Added: ~10 lines (buffer toggle)
- Added: ~15 lines (pass restructuring — replacing existing pass descriptions)
- Added: ~10 lines (script invocation instructions + modification protocol)

Estimated net: ~1094 → ~980 lines (roughly -110 lines).

## Implementation Order

1. Write the three Python scripts to `~/.claude/skills/distill/scripts/`
2. Edit SKILL.md: add greeting (Step -1) and pure distillation path
3. Edit SKILL.md: replace embedded code blocks with script invocations
4. Edit SKILL.md: restructure passes (2a/2b/3 → 2/3/4/5)
5. Edit SKILL.md: add buffer toggle to post-distillation flow
6. Edit SKILL.md: strip version references
7. Verify: read full file, confirm structure, check line count

## Non-Goals

- No changes to the output template structure (sections stay the same)
- No changes to the project skill generation template
- No changes to the troubleshooting decision tree
- No changes to existing project skills — they continue to work as-is
- No changes to figure naming conventions or Source Label Convention
