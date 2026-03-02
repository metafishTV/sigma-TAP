# Source Distillation Skill

Structured extraction and integration of scholarly source documents (PDF, image, web) into a project's knowledge base. Produces a neutral distillation and a project-specific interpretation for each source.

## The Problem

Processing scholarly sources for a research project is repetitive: extract text, handle figures, identify key concepts, map to your framework, update indexes and memory. Without a codified process, output format drifts between sessions, figure handling is inconsistent, troubleshooting paths are re-discovered each time, and the process is not portable to other projects.

## Quick Start

```
/distill path/to/source.pdf
```

**First run on a new project** triggers one-time differentiation (~2 minutes):
1. Audits available PDF tools (PyMuPDF, pdftotext, pdfplumber, Pillow, Docling, Marker, GROBID)
2. Scans the project for existing directories and indexes
3. Asks 4-7 setup questions (framework name, distillation mode, paths, specialist tool preferences)
4. Generates a project-specific `SKILL.md` tailored to your project

**Note**: If pdfplumber is not installed, the first run will prompt you to install it — it's the primary table extraction tool and is strongly recommended.

**After differentiation**, every `/distill` invocation produces two files:
- A **distillation** — neutral scholarly extraction, usable by any project
- An **interpretation** — project-specific mapping to your framework

## Architecture

```
User: /distill <source>
         │
         ▼
Global SKILL.md  (~/.claude/skills/distill/)
         │
         ├── Project skill exists? ──YES──▶ Project SKILL.md runs distillation
         │                                        │
         └── NO ──▶ Differentiation               ▼
                      │                      Two output files:
                      ▼                      distilled/<source>.md
                Generate project skill       interpretations/<source>.md
                (tooling + framework +
                 glossary + config)
```

**Global skill owns** the process: PDF extraction pipeline, figure handling, output template structure, troubleshooting decision tree, and the differentiation routine itself.

**Project skill owns** the structure: integration framework name, file paths, tooling profile for this machine, project terminology glossary (grows with each distillation), known issues log, and post-distillation update targets.

The project skill is self-contained after differentiation — it does not reference the global skill at runtime.

## Output Structure

### Distillation (`distilled/<source>.md`) — Neutral, Polyvocal

Readable in any direction by any project. Contains no project-specific interpretation.

- **Header**: Full citation, date, register/tone/density classification
- **Core Argument**: 1-3 paragraphs preserving the author's logic chain
- **Key Concepts**: Table of 5-15 entries with definitions and significance
- **Figures, Tables & Maps**: Textual decomposition of visual material (conditional)
- **Figure-Concept Contrast**: Which figures illustrate which concepts (conditional)
- **Equations & Formal Models**: LaTeX notation with variable explanations (conditional)
- **Theoretical & Methodological Implications**: Method analysis (mandatory — every source has a method)
- **Empirical Data**: For quantitative sources (conditional)

### Interpretation (`interpretations/<source>.md`) — Project-Specific, Biunivocal

Simultaneously reads toward the source and toward your project framework.

- **Project Significance**: Mapping table (source concept → framework element → relationship type)
- **Integration Points**: How each mapping affects your project, candidate forward notes
- **Open Questions**: Uncertain mappings flagged for user review

The user reviews the interpretation before any automated updates fire.

## PDF Extraction Pipeline

Two-phase content-aware routing. PyMuPDF scans every page first, then routes to specialist tools based on detected content.

| Phase | Tool | Role | Status |
|-------|------|------|--------|
| 1 (always) | PyMuPDF | Scanner + primary text extractor | Required |
| 2A | pdfplumber | Table extraction | Required |
| 2B | Docling | Complex layout, OCR, complex tables | Highly recommended (demand-install) |
| 2C | Marker | Equations → LaTeX | Optional (demand-install) |
| 2D | GROBID | Scholarly paper structure/metadata | Optional (requires Docker) |
| Fallback | pdftotext / Claude reader / user | When PyMuPDF fails entirely | Available |

**How routing works**: Phase 1 scans each page for tables, multi-column layout, scanned/empty pages, and equations. Phase 2 routes flagged pages to the best available specialist:
- **Tables** → pdfplumber (required), escalates to Docling for complex cases
- **Multi-column layout** → Docling for correct reading order
- **Scanned pages** → Docling OCR, or Claude vision fallback
- **Equations** → Marker for LaTeX preservation, or high-resolution vision fallback
- **Scholarly papers** → GROBID for structured metadata (if enabled)

Routes are not mutually exclusive — a page can trigger multiple specialists, merged by priority (Docling > Marker > pdfplumber > PyMuPDF).

**Demand-install**: Specialist tools that are not installed but would improve output are offered for installation at the point of need. The user can accept, defer to next time, or decline permanently.

**Figure handling**: When images are detected or specialist tools are unavailable for scanned/equation pages, pages are rendered as images and decomposed through Claude's vision capability. Equation pages render at dpi=200 for better symbol resolution.

The project SKILL.md records which tools are available on your machine (detected during differentiation).

## The Three Analysis Passes

After text extraction, three analytic passes run before writing output:

1. **Text extraction** (Pass 1): Raw content from the PDF pipeline
2. **Internal significance** (Pass 2a): What does each concept mean within this source? Understand the author on their own terms.
3. **Project significance** (Pass 2b): What does each concept mean for your project? Map to framework elements — confirm, extend, or challenge.
4. **Style detection** (Pass 3): Classify register (analytic philosophy / continental / empirical / formal-mathematical / practitioner / mixed), tone, and density. Prevents misreading concepts across intellectual traditions.

## Post-Distillation Updates

After the user reviews the interpretation, three automated updates fire (drawn from the interpretation file, not the distillation):

1. **INDEX.md**: Adds the source to the project's reference index with status and mapped concepts
2. **Buffer update**: Adds cross-source concept map entries to the warm layer with new `w:N` IDs, updates the hot-layer digest with change counts
3. **MEMORY.md**: Conservatively adds genuinely new, significant concepts (uncertain mappings go to Open Questions instead)

## Configuration & Adaptation

**Distillation modes** (set during differentiation):
- **Comprehensive**: Extract everything from the source (default)
- **Focused**: User specifies extraction priorities each time
- **Ask each time**: Prompt on each invocation

**Specialist tools** (set during differentiation or on demand):
- **pdfplumber**: Required for table extraction. Prompted for install if missing on first run.
- **Docling**: Highly recommended for complex layouts, OCR, and multi-column PDFs. ~500MB model download on first use. Offered when relevant content is detected (demand-install).
- **Marker**: Optional for equation-heavy content. Converts PDFs to Markdown with LaTeX. Offered when equations are detected (demand-install).
- **GROBID**: Optional for scholarly paper metadata (citations, bibliographies, section structure). Requires Docker (~2GB). Enabled during differentiation if needed.

Each tool's status is recorded in the project skill's tooling profile as `installed`, `demand-install`, or `never`.

**To use in your own project**:
1. Copy the global skill file (`~/.claude/skills/distill/SKILL.md`) to the target machine
2. Run `/distill` on any source — differentiation triggers automatically
3. Answer the 4-7 setup questions (framework name, mode, paths, tool preferences)
4. A project-specific SKILL.md is generated at `<repo>/.claude/skills/distill/SKILL.md`

The project skill grows its own terminology glossary and known-issues log with each distillation.

## Integration with Buffer System

If the [handoff/resume buffer system](../../README.md) is also installed, distillation outputs automatically feed into the warm-layer concept map via cross-source entries. The hot-layer digest is updated so that `/resume` surfaces new mappings as "recent changes" at the start of the next session.

If the buffer system is not installed, the skill still works — it skips the buffer update step.

## FAQ

**"PyMuPDF not installed"** — Install with `pip install PyMuPDF`. The skill works without it but falls back to less capable extraction tiers.

**Scanned PDF (all pages return empty text)** — The skill detects this and routes to the figure pipeline automatically, rendering each page as an image for vision-based extraction.

**Encrypted / password-protected PDF** — The skill asks for the password. If none available, it skips the file.

**Differentiation runs again unexpectedly** — This happens if `<repo>/.claude/skills/distill/SKILL.md` is missing or its frontmatter is damaged. Check that the file exists and starts with `name: distill`.

**Can I edit the project SKILL.md?** — Yes. The terminology glossary and known-issues log are designed to be curated. The tooling profile can be updated if you install new tools.

**Distillation seems incomplete** — Check which extraction route was used (logged in Known Issues). Fallback routes lose more layout information. Consider installing pdfplumber (tables) and Docling (layout/OCR) for best results.

**"Docling is downloading models"** — Normal on first use. Docling downloads ~500MB of AI models which are then cached locally. Subsequent runs use the cached models and are much faster.

**GROBID / Docker setup** — GROBID runs as a Docker container. If you said "yes" to GROBID during differentiation but Docker isn't running, the skill will attempt to start the container automatically. If Docker itself isn't installed, GROBID is skipped and the standard pipeline handles the PDF.

**Equations look wrong** — If Marker isn't installed, equations are extracted via vision at dpi=200. This is reasonably accurate but may have transcription errors. Install Marker (`pip install marker-pdf`) for native LaTeX extraction.
