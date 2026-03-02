# Distillation Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a two-file distillation skill — a shareable global launcher that differentiates into project-specific operational instances.

**Architecture:** Global skill at `~/.claude/skills/distill/SKILL.md` (read-only template, domain-agnostic) generates a project skill at `<repo>/.claude/skills/distill/SKILL.md` (project-specific config, grows with project). The global skill contains the PDF extraction pipeline, figure handling, output template, troubleshooting tree, and differentiation routine. The project skill contains integration framework name, file paths, tooling profile, distillation mode, terminology glossary, and post-update instructions.

**Tech Stack:** Claude Code skills (markdown instruction files), PyMuPDF (fitz), pdftotext (Poppler), Claude's built-in Read tool (PDF reader), Claude's multimodal vision (figure decomposition).

---

## Prerequisite Knowledge

### How Skills Work

Skills are markdown files at `<dir>/SKILL.md` with YAML frontmatter (`name`, `description`) followed by markdown instructions. Claude reads and follows them as behavioral directives. They are NOT executable code — they are structured instructions that Claude interprets at runtime.

**Global skills** live at `~/.claude/skills/<name>/SKILL.md` — available in all projects.
**Project skills** live at `<repo>/.claude/skills/<name>/SKILL.md` — available only in that repo.

When both exist for the same skill name, the project skill takes precedence.

### Existing Skills (Reference)

- `~/.claude/skills/handoff/SKILL.md` — end-of-session buffer generation (6997 bytes)
- `~/.claude/skills/resume/SKILL.md` — session context reconstruction (4315 bytes)
- Project skills dir: `C:\Users\user\Documents\New folder\sigma-TAP-repo\.claude\skills\` (empty)

### Existing Distillation Infrastructure

- 23 distilled files in `docs/references/distilled/` (markdown, consistent format)
- INDEX.md at `docs/references/INDEX.md` (tracks all sources with status/mapping/notes)
- handoff.json at `.claude/buffer/handoff.json` (cross_source entries, validation_log)
- MEMORY.md at `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`

### Installed PDF Tools

- PyMuPDF 1.27.1 ✅ (`import fitz`)
- pdftotext via Poppler ✅ (`/mingw64/bin/pdftotext` and `C:\Program Files\Poppler`)
- pdfplumber ❌ (not installed)

### Design Document

Full design: `docs/plans/2026-03-02-distillation-skill-design.md`

---

## Task 1: Write the Global Skill

**Files:**
- Create: `C:\Users\user\.claude\skills\distill\SKILL.md`

**Step 1: Create the directory**

Run:
```bash
mkdir -p ~/.claude/skills/distill
```
Expected: Directory created (or already exists).

**Step 2: Write the global skill file**

Write `~/.claude/skills/distill/SKILL.md` with the following complete content:

```markdown
---
name: distill
description: Distill a source document (PDF, image, web page) into a structured extraction with project integration. On first run in a new project, runs a differentiation routine to generate a project-specific skill.
---

# Source Distillation

Distill a source document into a structured, project-integrated extraction. Handles PDFs, images, and web pages with tiered fallback extraction, systematic figure handling, and automated post-distillation updates.

## Invocation

When the user asks to distill, extract, or process a source document — or when this skill is invoked directly — follow this process.

## Step 0: Check for Project Skill

Before doing anything else, check if a project-level distill skill exists:

1. Look for `<repo>/.claude/skills/distill/SKILL.md` in the current repository
2. **If it exists**: Read and follow that skill instead of continuing here. The project skill has project-specific configuration that overrides these generic instructions.
3. **If it does not exist**: Continue to Step 1 (Differentiation) below.

---

## DIFFERENTIATION MODE

This runs ONCE per project to generate a project-specific skill.

### Step 1: Tooling Audit (automatic, no user interaction)

Run these checks silently and record results:

```python
# Check PyMuPDF
python -c "import fitz; print(f'PyMuPDF {fitz.version}')"

# Check pdftotext (Poppler)
pdftotext -v 2>&1 | head -1

# Check pdfplumber
python -c "import pdfplumber; print(f'pdfplumber {pdfplumber.__version__}')"

# Check Pillow
python -c "from PIL import Image; import PIL; print(f'Pillow {PIL.__version__}')"
```

Record each as `"installed: <version>"` or `"not installed"`.

### Step 2: Project Scan (automatic, no user interaction)

Scan the repository for existing distillation infrastructure:

1. **Distillation directory**: Glob for `docs/references/distilled/`, `docs/distilled/`, `distilled/`, or similar
2. **Index file**: Glob for `**/INDEX.md`, `**/index.md` near the distillation directory
3. **Handoff buffer**: Glob for `.claude/buffer/handoff.json` or similar JSON buffer
4. **Memory file**: Check for MEMORY.md in the repo or in `~/.claude/projects/*/memory/MEMORY.md`
5. **Existing distillations**: Count how many `.md` files exist in the distillation directory
6. **README / project description**: Read the repo's README for project context

Record all detected paths. If nothing is found, record nulls — the project skill will use default paths.

### Step 3: User Questionnaire (interactive)

Ask these questions using the AskUserQuestion tool:

**Q1**: "What should the project-specific integration section be called?"
- Options: Suggest a name based on README/project context (e.g., "[Project Name] Integration Points") plus "Custom name" option
- This becomes the final section heading in every distillation

**Q2**: "Should distillations be comprehensive (extract everything) or focused (you specify what to prioritize each time)?"
- Options: [Comprehensive (Recommended)] [Focused] [Ask me each time]

**Q3**: Confirm detected paths:
- "I detected these paths — correct?" and list what was found
- Allow override for any path

**Q4** (only if tooling gaps detected):
- "I notice [tool] is not installed. Should I install it, or work with what's available?"

### Step 4: Generate Project Skill

Write `<repo>/.claude/skills/distill/SKILL.md` with the following structure (fill in values from Steps 1-3):

```
---
name: distill
description: Distill source documents for [project name] with [framework name] integration.
---

# [Project Name] — Source Distillation

Project-specific distillation skill generated by the global distill skill.

## Configuration

- **Integration framework**: [from Q1]
- **Distillation mode**: [from Q2]
- **Distillation directory**: [detected or specified path]
- **Index file**: [detected or specified path]
- **Handoff buffer**: [detected or specified path]
- **Memory file**: [detected or specified path]

## Tooling Profile

- PyMuPDF: [version or "not installed"]
- pdftotext: [version or "not installed"]
- pdfplumber: [version or "not installed"]
- Pillow: [version or "not installed"]

## Project Terminology Glossary

(This section grows as distillations add project-relevant terms)

| Term | Definition | First seen in |
|------|-----------|---------------|

## Extraction Pipeline

Follow the global skill's extraction pipeline (Section: EXTRACTION TIERS below),
using the tooling profile above to skip unavailable tiers.

[Include the full extraction pipeline, output template, figure handling,
troubleshooting tree, and post-update instructions — customized with
the project-specific paths and framework name from Configuration above.]
```

After generating: inform the user, then proceed to the first distillation if a source was provided.

---

## DISTILLATION MODE

This is the main operational mode. Runs every time after differentiation.

### PDF Extraction Pipeline

Use this tiered strategy. Move to the next tier ONLY on failure of the current one. Never retry the same tier on the same error.

**Tier 1: PyMuPDF (fitz) — PRIMARY**

```python
import fitz
doc = fitz.open("source.pdf")
for page in doc:
    text = page.get_text("text")
    # If text is empty on multiple pages → likely scanned PDF → go to Figure Pipeline
    # If fitz.open() throws → go to Tier 2
```

- Text extraction: `page.get_text("text")` per page
- Figure detection: `page.get_images(full=True)` for embedded images
- Page screenshot: `page.get_pixmap(dpi=150)` for visual pages
- Table heuristic: examine text block positions for tabular layout

**Tier 2: pdftotext (Poppler) — FALLBACK**

```bash
pdftotext -layout "source.pdf" -
```

- Plain text only — no figure extraction capability
- Good for text-heavy PDFs where PyMuPDF fails on encoding

**Tier 3: Claude Built-in PDF Reader — SECOND FALLBACK**

- Use the Read tool with `pages` parameter
- Chunk into 20-page batches: pages "1-20", then "21-40", etc.
- Works for most PDFs but may lose layout information

**Tier 4: User Intervention**

- Inform user which tiers were tried and what errors occurred
- Ask user to provide text, try a different format, or skip

### Figure Handling Pipeline

**When to trigger**: Embedded images detected, or text extraction returns empty (scanned PDF), or source is a standalone image.

1. **Extract/screenshot**: Use `page.get_pixmap(dpi=150)` to render pages as images. Save to temp location or present directly.
2. **Decompose**: Present the image to Claude's vision via the Read tool. Extract:
   - Caption and title
   - Axis labels and scales (if chart/graph)
   - Data relationships shown
   - Visual structure and layout
   - Legend entries
3. **Describe**: Write a textual description suitable for the distillation's "Figures, Tables & Maps" section.
4. **Cross-reference**: Note which Key Concepts each figure illustrates or extends.
5. **Flag failures**: If a figure can't be parsed, note it for user review.

For **non-PDF images** (PNG, JPG, etc.): Use the Read tool directly on the image file. Claude's multimodal capability handles decomposition.

### Output Template

Produce the distillation in this exact structure. Mandatory sections ALWAYS appear. Conditional sections appear ONLY when the source contains relevant content.

```markdown
# [Source Label] — Distillation

> Source: [full citation — author, title, publication, year, page count]
> Date distilled: [YYYY-MM-DD]
> Distilled by: Claude (via distill skill)

## Core Argument

[1-3 paragraphs: What is this source fundamentally arguing? What is its core contribution? Preserve the author's logic chain — do not editorialize.]

## Key Concepts

| Concept | Definition | Significance |
|---------|-----------|--------------|
| [term]  | [precise definition as used in this source] | [why it matters to the source's argument] |

[Include ALL concepts essential to understanding the source. Typically 5-15 entries. Use the source's own terminology.]

## Figures, Tables & Maps                     ← CONDITIONAL: only if visual material exists

[For each figure/table/map:]
### [Figure/Table N]: [Title or description]
- **What it shows**: [textual decomposition of visual content]
- **Key data points**: [specific values, relationships, patterns visible]
- **Connection to argument**: [how this visual supports the core argument]

## Figure ↔ Concept Contrast                  ← CONDITIONAL: only if Figures section exists

[For each figure, map which Key Concepts it illustrates or extends. Format:]
- Figure N → [Concept A]: [how the figure demonstrates/extends this concept]
- Figure N → [Concept B]: [relationship]

## Equations & Formal Models                  ← CONDITIONAL: only if mathematical content exists

[Reproduce key equations in LaTeX notation. Explain variables and significance.]

## Methodology & Empirical Data               ← CONDITIONAL: only if empirical/experimental content exists

[Data sources, sample sizes, methods, key findings with numbers.]

## [Integration Framework Name] Integration Points    ← MANDATORY

[How does this source connect to the project's framework?]
[For each integration point:]
- **[concept/mechanism]**: [how it maps, what it implies for the project]
- **Candidate forward notes**: [if this suggests new theoretical development, note it]
- **Concept map entries**: [candidate cross_source mappings for handoff.json]
```

### Troubleshooting Decision Tree

**DO NOT blindly retry tools.** Follow this tree on errors:

```
PDF won't open:
├─ Error contains "password" or "encrypted"
│   → Ask user for password. If none, skip file.
├─ Error contains "corrupt" or "invalid"
│   → Try Tier 2 (pdftotext). If that fails, try Tier 3 (Claude reader).
├─ Error contains "codec" or "encoding"
│   → Try: fitz.open(path, filetype="pdf"). If fails, try Tier 2.
└─ Error contains "not found" or "No such file"
    → Verify path with user. Check for typos, spaces in path.

Text extraction returns empty string:
├─ Check multiple pages (not just page 0)
│   ├─ ALL pages empty → Scanned PDF. Route ALL pages to Figure Pipeline.
│   └─ SOME pages empty → Mixed PDF. Extract text where possible, screenshot empty pages.
├─ Check if DRM-protected
│   → fitz metadata check. If protected, inform user, route to Tier 4.
└─ Try Tier 2 (pdftotext) as cross-check
    → If pdftotext also returns empty, confirmed scanned. Figure Pipeline.

Figure extraction fails:
├─ get_pixmap() throws MemoryError
│   → Reduce DPI: try dpi=100, then dpi=72. Try individual pages.
├─ get_images() returns empty but pages have visuals
│   → Visuals are drawn (not embedded images). Use get_pixmap() on full page.
└─ Read tool can't parse the image
    → Note for user: "Figure on page N could not be decomposed. Manual review needed."

Claude reader issues:
├─ "exceeds" or "too large"
│   → Use pages parameter. Chunk: "1-20", "21-40", etc.
├─ "cannot read" or empty result
│   → Try explicit pages: "1-5" first. If works, continue chunking. If not, Tier 4.
└─ Timeout or very slow
    → Reduce chunk size to 10 pages. Try again.
```

### Post-Distillation Updates

After a successful distillation (all mandatory sections written), perform these three updates. If the distillation was abandoned or failed, do NOT perform any updates.

**1. INDEX.md Update**

Read the index file. Add or update a row in the appropriate category table:

```markdown
| [filename] | [Author] | distilled | [distilled/filename.md](distilled/filename.md) | [mapped concepts] | [notes] |
```

If no index file exists, create one with the header format from the existing INDEX.md.

**2. handoff.json Update**

Read `.claude/buffer/handoff.json`. Add entries:

- In `concept_map.cross_source`: add a mapping entry for each Key Concept that maps to the project framework. Format:
  ```json
  "Source:ConceptName": {
    "maps_to": "[project framework mapping]",
    "ref": "[forward note reference if applicable]",
    "suggest": null
  }
  ```
- In `validation_log`: add an entry:
  ```json
  {
    "check": "distill: [Source Label]",
    "status": "NEW",
    "detail": "[N] concepts mapped, [M] integration points identified",
    "session": "YYYY-MM-DD"
  }
  ```

**3. MEMORY.md Update (conservative)**

Read MEMORY.md. ONLY add entries if the source introduces concepts that are:
- Genuinely new to the project (not already in MEMORY.md)
- Significant enough to warrant persistent memory (not minor details)
- Project-relevant (connects to the framework)

If uncertain about whether a concept meets this threshold, note it in the distillation's Integration Points section with a flag: "Candidate for MEMORY.md — user review needed."

### Error Logging

After each distillation, mentally note which extraction tier succeeded and any troubleshooting paths taken. If the project skill has a "Known Issues" section, add entries for recurring problems so future distillations start with the right tier.
```

**Step 3: Verify the file was written correctly**

Run:
```bash
wc -l ~/.claude/skills/distill/SKILL.md
```
Expected: approximately 250-300 lines.

**Step 4: Commit**

This file is in the user's global config, NOT in the repo. No git commit needed.

---

## Task 2: Validate Global Skill — Smoke Test

**Purpose:** Verify the global skill file is well-formed and readable.

**Step 1: Read the skill file back**

Use the Read tool on `~/.claude/skills/distill/SKILL.md` and verify:
- YAML frontmatter has `name: distill` and `description:` fields
- All sections from the design doc are present:
  - Step 0 (project skill check)
  - Differentiation Mode (Steps 1-4)
  - Distillation Mode (extraction pipeline, figure handling, output template, troubleshooting, post-updates)
- No broken markdown formatting
- No placeholder text left unfilled

**Step 2: Verify skill discovery**

The skill should appear when Claude lists available skills. This is automatic — if the file exists at the right path with valid frontmatter, it will be discovered.

Run:
```bash
ls -la ~/.claude/skills/distill/SKILL.md
```
Expected: File exists, non-zero size.

---

## Task 3: Generate sigma-TAP Project Skill (Manual Differentiation)

**Files:**
- Create: `C:\Users\user\Documents\New folder\sigma-TAP-repo\.claude\skills\distill\SKILL.md`

Since this is the sigma-TAP project and we already know all the answers to the differentiation questionnaire (from 23 prior distillations), we skip the interactive questionnaire and write the project skill directly.

**Step 1: Create the directory**

Run:
```bash
mkdir -p "C:/Users/user/Documents/New folder/sigma-TAP-repo/.claude/skills/distill"
```

**Step 2: Write the project skill**

Write the sigma-TAP project skill to `<repo>/.claude/skills/distill/SKILL.md`. The content must include:

**Configuration values** (known from project context):
- Integration framework: `sigma-TAP Integration Points`
- Distillation mode: `Comprehensive`
- Distillation directory: `docs/references/distilled/`
- Index file: `docs/references/INDEX.md`
- Handoff buffer: `.claude/buffer/handoff.json`
- Memory file: `C:\Users\user\.claude\projects\C--Users-user-Documents-New-folder\memory\MEMORY.md`

**Tooling profile** (known from audit):
- PyMuPDF: `1.27.1`
- pdftotext (Poppler): `installed` (at `/mingw64/bin/pdftotext`)
- pdfplumber: `not installed`
- Pillow: check at write time

**Initial terminology glossary**: Seed from the most important project-specific terms that appear across distillations:

| Term | Definition | First seen in |
|------|-----------|---------------|
| TAP | Theory of the Adjacent Possible | TAPequation-FINAL |
| sigma-TAP | Sigma feedback modification of TAP | Core framework |
| L-matrix | L11/L12/L21/L22 event channels | Emery-Trist-1965 |
| TAPS | Transvolution-Anopression-Praxis-Syntegration | Core framework |
| cross-metathesis | Agent-to-agent type exchange | Core framework |
| Youn ratio | Ratio of observed to possible combinations | Youn-et-al-2025 |
| PSI | Political Stress Indicator | Turchin-2013 |
| directive correlation | Sommerhoff's formal mechanism | Emery-M-2000 |

**Full operational instructions**: Copy the complete extraction pipeline, figure handling, output template, troubleshooting tree, and post-update instructions from the global skill — but with all paths and framework names filled in with sigma-TAP-specific values. The project skill should be SELF-CONTAINED — it should not need to reference the global skill at runtime.

**Step 3: Verify the file**

Run:
```bash
wc -l "C:/Users/user/Documents/New folder/sigma-TAP-repo/.claude/skills/distill/SKILL.md"
```
Expected: approximately 280-350 lines (slightly larger than global due to seeded glossary and filled-in paths).

**Step 4: Commit**

```bash
cd "C:/Users/user/Documents/New folder/sigma-TAP-repo"
git add .claude/skills/distill/SKILL.md
git commit -m "feat: add sigma-TAP distillation skill (project instance)

Generated from global distill skill differentiation.
Comprehensive mode, sigma-TAP Integration Points framework.
Seeded glossary from 23 existing distillations.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Validation — PDF Pipeline Test

**Purpose:** Run the skill against a known-good PDF to verify the full pipeline works.

**Step 1: Select test source**

Use a PDF already in the repo that has been distilled before, so we can compare output quality. Good candidate: `docs/references/Emery-Trist-1965-Causal-Textures.pdf` (well-known, existing distillation for comparison).

**Step 2: Run the distillation**

Invoke the distill skill on the test PDF. The project skill should:
1. Detect PyMuPDF as available (Tier 1)
2. Extract text successfully
3. Detect any figures/tables
4. Produce output with all mandatory sections
5. Use "sigma-TAP Integration Points" as the integration section name

**Step 3: Compare output**

Read the existing distillation `docs/references/distilled/Emery-Trist-1965.md` and compare:
- Are all mandatory sections present in the new output?
- Is the Core Argument substantively similar?
- Are Key Concepts captured?
- Is the Integration Points section populated?

**Step 4: Verify post-updates would work**

Check (but don't execute) that:
- INDEX.md has the right format for a new row
- handoff.json has the right structure for cross_source entries
- The automation logic correctly identifies the paths

Do NOT actually write a duplicate distillation or update files — this is a dry-run validation only.

---

## Task 5: Validation — Figure Handling Test

**Purpose:** Verify the screenshot + decompose pipeline works for visual sources.

**Step 1: Select test source**

Use `docs/references/Bateson table.png` — a standalone image that was previously distilled as `Bateson-Table-D.md`. This tests the non-PDF visual pipeline.

**Step 2: Run figure decomposition**

1. Read the image file with Claude's Read tool (multimodal)
2. Verify Claude can see and describe the table contents
3. Verify the decomposition captures: column headers, row structure, data relationships

**Step 3: Compare to existing distillation**

Read `docs/references/distilled/Bateson-Table-D.md` and verify the figure decomposition captures equivalent information.

---

## Task 6: Validation — Post-Update Integration Test

**Purpose:** Verify the three post-update automations produce valid output.

**Step 1: INDEX.md format check**

Read `docs/references/INDEX.md`. Verify:
- The table format matches what the skill would generate
- A new row insertion wouldn't break the markdown table structure
- The status field conventions match (`distilled`, `mapped`, etc.)

**Step 2: handoff.json format check**

Read `.claude/buffer/handoff.json` (first 100 lines). Verify:
- The `cross_source` entry format matches existing entries
- The `validation_log` entry format matches existing entries
- New entries would be valid JSON and wouldn't break the schema

**Step 3: MEMORY.md format check**

Read `MEMORY.md`. Verify:
- The section structure is understood
- A new concept entry would be placed in the right section
- The conservative threshold logic makes sense (only genuinely new + significant + project-relevant)

---

## Task 7: Final Review & Documentation

**Step 1: Review both skill files end-to-end**

Read both skill files completely:
- `~/.claude/skills/distill/SKILL.md` (global)
- `<repo>/.claude/skills/distill/SKILL.md` (project)

Verify:
- No placeholder text remains
- All paths are correct and absolute where needed
- The project skill is self-contained (doesn't reference global skill at runtime)
- Troubleshooting tree covers all known failure modes
- Output template matches the design doc exactly

**Step 2: Update the design doc status**

Edit `docs/plans/2026-03-02-distillation-skill-design.md`:
- Change status from "Approved by user, ready for implementation" to "Implemented"
- Add implementation date

**Step 3: Final commit**

```bash
cd "C:/Users/user/Documents/New folder/sigma-TAP-repo"
git add docs/plans/2026-03-02-distillation-skill-design.md
git commit -m "docs: mark distillation skill design as implemented

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | What | Files | Est. Time |
|------|------|-------|-----------|
| 1 | Write global skill | `~/.claude/skills/distill/SKILL.md` | 5 min |
| 2 | Smoke test global skill | (verification only) | 2 min |
| 3 | Generate sigma-TAP project skill | `<repo>/.claude/skills/distill/SKILL.md` | 5 min |
| 4 | PDF pipeline validation | (dry-run test) | 3 min |
| 5 | Figure handling validation | (dry-run test) | 3 min |
| 6 | Post-update integration test | (format verification) | 3 min |
| 7 | Final review & documentation | design doc update | 2 min |

**Total: 7 tasks, ~23 minutes**

**Critical path:** Task 1 → Task 2 → Task 3 → Tasks 4-6 (parallel) → Task 7
