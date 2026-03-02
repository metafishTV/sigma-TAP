# Distillation Skill — Design Document

**Date**: 2026-03-02
**Status**: Implemented (2026-03-02)
**Purpose**: Formalize the iterated source-distillation process as a two-file skill — a shareable global launcher that differentiates into project-specific operational instances

---

## 1. Problem

Over 23 distillations, the extraction process has converged on a stable pattern: PDF extraction → figure handling → structured output → post-updates. But this pattern lives entirely in Claude's implicit behavior, re-discovered each session. This causes:

- **Token waste**: Each session re-learns the troubleshooting paths for PDF failures (encrypted, scanned, codec errors) — often arriving at the same solutions
- **Inconsistency**: Output format drifts between sessions; some distillations have figure contrast sections, others don't
- **Figure loss**: When PDFs contain visual material that can't be text-extracted, there's no systematic screenshot-and-decompose routine
- **No portability**: The process is locked to this project; other projects would start from scratch

## 2. Solution

A two-file skill architecture: a **global skill** (stem cell) that lives in the user's Claude config and differentiates on first run into a **project skill** (operational clone) stored in the repo. The global skill is shareable, domain-agnostic, and read-only. The project skill carries all project-specific configuration and grows its own institutional knowledge.

## 3. Architecture

### 3.1 Two-File Model

| Component | Path | Scope | Mutates? |
|---|---|---|---|
| Global skill | `~/.claude/skills/distill/SKILL.md` | User-level, all projects | Never — read-only template |
| Project skill | `<repo>/.claude/skills/distill/SKILL.md` | Repo-level, project-specific | Yes — grows with project |

### 3.2 Invocation Logic

```
User invokes /distill (or skill triggers on distillation task)
│
├─ Does <repo>/.claude/skills/distill/SKILL.md exist?
│   ├─ YES → Load project skill, run distillation
│   └─ NO  → Load global skill, run differentiation routine
│            ├─ Phase 1: Tooling audit (automatic)
│            ├─ Phase 2: Project scan (automatic)
│            ├─ Phase 3: User questionnaire (interactive)
│            ├─ Phase 4: Generate project skill
│            └─ Then run first distillation with new project skill
```

### 3.3 Separation of Concerns

**Global skill owns:**
- PDF extraction pipeline (tiered fallback logic)
- Figure screenshot + decompose routine
- Output template (mandatory + conditional sections)
- Troubleshooting decision tree
- Differentiation routine

**Project skill owns:**
- Integration framework name (e.g., "sigma-TAP Integration Points")
- File paths (distillation_dir, index_file, buffer_file, memory_file)
- Tooling profile (from audit)
- Distillation mode (comprehensive / focused / ask-each-time)
- Project terminology glossary (grows over time)
- Post-distillation update instructions (customized to project structure)

## 4. PDF Extraction Pipeline

Tiered strategy encoded as a decision tree in the global skill:

### 4.1 Extraction Tiers

```
Tier 1: PyMuPDF (fitz) — PRIMARY
├─ Text: page.get_text("text")
├─ Figures: page.get_pixmap() → screenshot per page
├─ Tables: heuristic detection from text block geometry
└─ Failure → Tier 2

Tier 2: pdftotext (Poppler) — FALLBACK
├─ Plain text only (no figure extraction)
├─ Layout mode: pdftotext -layout source.pdf -
└─ Failure → Tier 3

Tier 3: Claude built-in PDF reader — SECOND FALLBACK
├─ Read tool with pages parameter (max 20pp/request)
├─ Chunked for large PDFs: pages "1-20", "21-40", etc.
└─ Failure → Tier 4

Tier 4: User intervention
└─ Ask user to provide text or alternative format
```

### 4.2 Key Rules

- Always attempt PyMuPDF first — handles 90%+ of cases
- Never blindly retry the same tool on the same failure
- Encrypted/protected PDFs: detect early, inform user, route to Tier 4
- Scanned-image PDFs (no text layer): detect via empty text extraction → route to figure pipeline for page-by-page screenshot decomposition
- Log which tier succeeded for each source — project skill accumulates this institutional knowledge

## 5. Figure Handling

### 5.1 Screenshot + Decompose Pipeline

```
Step 1: Detect figures
├─ PyMuPDF: extract embedded images from PDF
├─ If no embedded images but content is visual:
│   render full page as pixmap → screenshot
└─ For non-PDF sources (images, screenshots):
    take as-is via Read tool (Claude multimodal)

Step 2: Decompose in multimodal context
├─ Present screenshot/image to Claude's vision
├─ Extract: caption, axis labels, data relationships,
│   visual structure, legend entries
└─ Generate textual description for distillation output

Step 3: Integrate into output
├─ Place in "Figures, Tables & Maps" section
├─ Cross-reference to Key Concepts in contrast section
└─ Flag any figures that couldn't be parsed for user review
```

### 5.2 Non-PDF Visual Sources

Standalone images, scanned tables (e.g., Bateson Table D type sources):
- Read tool directly on image file (Claude is multimodal)
- Same decompose routine applies
- Source label notes original format (e.g., "transcribed from .png scan")

## 6. Output Template

### 6.1 Structure

Mandatory sections always appear. Conditional sections appear when content detection triggers them.

```markdown
# [Source Label] — Distillation
> Source: [full citation]
> Date distilled: [YYYY-MM-DD]
> Distilled by: Claude (via distill skill)

## Core Argument                              ← MANDATORY
[1-3 paragraphs: what is this source fundamentally saying?]

## Key Concepts                               ← MANDATORY
| Concept | Definition | Significance |
|---------|-----------|--------------|
| ...     | ...       | ...          |

## Figures, Tables & Maps                     ← CONDITIONAL
[Only if source contains visual material]
[Screenshot descriptions, table transcriptions, diagram analyses]

## Figure ↔ Concept Contrast                  ← CONDITIONAL
[Only if figures present]
[How do the visual elements relate to / extend the key concepts?]

## Equations & Formal Models                  ← CONDITIONAL
[Only if source contains mathematical content]

## Methodology & Empirical Data               ← CONDITIONAL
[Only if source is empirical/experimental]

## [Project] Integration Points               ← MANDATORY
[Name set during differentiation — e.g., "sigma-TAP Integration Points"]
[How does this source connect to the project's framework?]
[Candidate forward-note references, concept map entries]
```

### 6.2 Template Enforcement

- The skill instructs Claude to produce each mandatory section regardless of source content
- Conditional sections are triggered by content detection during extraction
- Figures found → include Figures section → include Contrast section
- Math found → include Equations section
- Empirical data found → include Methodology section
- The integration points section uses the project-specific framework name from the project skill

## 7. Differentiation Routine

Runs once per project, when the global skill is invoked and no project skill exists.

### 7.1 Phase 1: Tooling Audit (automatic)

```
Scan for installed tools:
├─ PyMuPDF (fitz)    → version or "not installed"
├─ pdftotext/poppler → version or "not installed"
├─ pdfplumber        → version or "not installed"
├─ Other PDF tools   → scan for common packages
└─ Image tools       → PIL/Pillow availability
Result: tooling_profile dictionary
```

No user interaction required. Runs silently, logs results.

### 7.2 Phase 2: Project Scan (automatic)

```
Scan project structure for:
├─ Existing distillations (docs/references/distilled/ or similar)
├─ Existing index file (INDEX.md or similar)
├─ Existing memory buffer (handoff.json or similar)
├─ Existing memory file (MEMORY.md or similar)
├─ Docs directory structure
└─ README / project description for context
Result: project_profile dictionary
```

No user interaction required. Auto-detects existing conventions.

### 7.3 Phase 3: User Questionnaire (interactive)

Three to four questions, multiple-choice preferred:

```
Q1: "What is your project's integration framework called?"
    [Free text — e.g., "sigma-TAP Integration Points"]

Q2: "Should distillations be comprehensive or focused?"
    [Comprehensive (Recommended)] [Focused] [Ask me each time]

Q3: "Where should distillations be stored?"
    [Auto-detected path — confirm or override]

Q4: (Only if tooling gaps detected)
    "PyMuPDF is not installed. Install it, or work with
     [available tools] only?"
    [Install (Recommended)] [Work with existing tools]
```

### 7.4 Phase 4: Generate Project Skill

Write `<repo>/.claude/skills/distill/SKILL.md` containing:
- Integration framework name (from Q1)
- Distillation mode (from Q2)
- File paths (from scan + Q3):
  - `distillation_dir` — where distillations are saved
  - `index_file` — tracking index path
  - `buffer_file` — handoff.json path
  - `memory_file` — MEMORY.md path
- Tooling profile (from audit)
- Project terminology glossary (initially empty — grows as distillations add terms)
- Post-distillation update instructions (customized to detected project structure)

## 8. Post-Distillation Automation

Three updates fire automatically after each successful distillation. All three are atomic — if the distillation fails or is abandoned, no post-updates fire.

### 8.1 INDEX.md Update

```
- Add row to source tracking table:
  | Source Label | Status: Done | Date | Mapped Concepts | Notes |
- Update coverage summary counts
- If INDEX.md doesn't exist yet: create with header + first row
```

### 8.2 handoff.json Update

```
- Add cross_source entries to concept_map for each
  key concept that maps to project framework
- Add validation_log entry recording what was distilled
- Maintain existing entry format and structure
```

### 8.3 MEMORY.md Update

```
- If the source introduces concepts significant enough for
  project memory: append to relevant section
- Conservative threshold: only add if concept is genuinely
  new and project-relevant
- Flag for user review if uncertain about significance level
```

## 9. Error Handling & Troubleshooting

The global skill includes a troubleshooting decision tree to avoid token-wasting blind retries.

### 9.1 PDF Won't Open

```
"password required"  → Ask user for password or skip file
"file corrupted"     → Try poppler → Claude reader → ask user
"codec error"        → Try with encoding parameter → fallback chain
"file not found"     → Verify path with user
```

### 9.2 Text Extraction Empty

```
Scanned PDF (image-only)  → Detect via empty text → route to figure pipeline
DRM-protected             → Inform user, route to Tier 4
Non-standard encoding     → Try poppler with -enc flag → fallback chain
```

### 9.3 Figure Extraction Fails

```
PyMuPDF pixmap fails  → Lower DPI → individual pages → full-page screenshot
Image too large       → Chunk or downsample
No extractable images → Full-page screenshot approach for all pages
```

### 9.4 Claude Reader Issues

```
"exceeds page limit"  → Chunk into 20-page batches
"cannot read"         → Try with explicit pages parameter
Timeout               → Reduce page range, retry with smaller chunks
```

### 9.5 Error Logging

Each troubleshooting path logs what was tried and what worked. The project skill accumulates this as institutional knowledge — if a particular PDF type consistently needs a specific fallback, the project skill remembers.

## 10. Validation Strategy

Not automated unit tests — these are manual validation steps listed in the skill for post-deployment verification:

| # | Test | What to verify |
|---|------|---------------|
| 1 | **Smoke test** | Differentiation routine runs on fresh repo, generates valid project skill |
| 2 | **PDF pipeline** | Known-good PDF produces all mandatory template sections |
| 3 | **Fallback test** | With PyMuPDF import blocked, graceful degradation to poppler |
| 4 | **Figure test** | PDF with embedded figures triggers screenshot + decompose |
| 5 | **Post-update test** | INDEX.md, handoff.json, MEMORY.md all updated correctly after distillation |
| 6 | **Idempotency test** | Running differentiation twice doesn't overwrite existing project skill |

## 11. Deployment

### 11.1 Global Skill

```
~/.claude/skills/distill/SKILL.md
```

Installed manually or via a setup script. Shareable — any user can drop this into their Claude skills directory and it works for any project.

### 11.2 Project Skill (generated)

```
<repo>/.claude/skills/distill/SKILL.md
```

Generated by the differentiation routine. Committed to the repo. Persists across sessions and contributors.

### 11.3 For sigma-TAP Specifically

After deployment, the differentiation routine would produce a project skill with:
- Integration framework: "sigma-TAP Integration Points"
- Distillation mode: Comprehensive
- Paths: `docs/references/distilled/`, `docs/references/INDEX.md`, `.claude/buffer/handoff.json`, MEMORY.md
- Tooling: PyMuPDF 1.27.1 ✅, pdftotext/Poppler ✅, pdfplumber ❌
- Glossary: seeded from existing 23 distillations' key concepts

---

*Design approved 2026-03-02. Next step: implementation plan via writing-plans skill.*
