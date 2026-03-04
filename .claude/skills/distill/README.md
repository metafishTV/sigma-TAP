# Source Distillation — sigma-TAP

Structured extraction and integration of scholarly sources into the sigma-TAP multi-agent praxis simulator's knowledge base. Uses the sigma-TAP Integration Points framework for concept_convergence mapping across philosophical and empirical sources.

## Setup

- **Map type**: concept_convergence
- **Framework**: sigma-TAP Integration Points
- **Mode**: Comprehensive (extract everything)
- **GROBID**: Disabled

## Tools Available

| Tool | Status | Role |
|------|--------|------|
| PyMuPDF 1.27.1 | Installed | Scanner + primary text extractor |
| pdftotext (Poppler) | Installed | Fallback (Route G only) |
| pdfplumber | Not installed | Table specialist (recommended: `pip install pdfplumber`) |
| Pillow 12.1.1 | Installed | Image processing |
| Docling | Demand-install | Complex layout, OCR, tables |
| Marker | Demand-install | Equations to LaTeX |
| GROBID | Not available | Scholarly paper metadata (requires Docker) |

## Output Locations

| Output | Path |
|--------|------|
| Distillations | `docs/references/distilled/` |
| Interpretations | `docs/references/interpretations/` |
| Figures | `docs/references/distilled/figures/[Source-Label]/` |
| Index | `docs/references/INDEX.md` |
| Buffer (warm) | `.claude/buffer/handoff-warm.json` |
| Buffer (hot digest) | `.claude/buffer/handoff.json` |

## Sources Distilled

| Source Label | Date | Route | Notes |
|-------------|------|-------|-------|
| TAPequation-FINAL | Pre-skill | Route A | Core TAP equation. Merged with Applications-of-TAP |
| Paper1-Biocosmology | Pre-skill | Route A | Core TAP framework |
| Paper2-Biocosmology-Perspective | Pre-skill | Route A | Three cosmological system types |
| Taalbi-Long-Run-Patterns | Pre-skill | Route A | Empirical benchmark, Youn ratio |
| Turchin-End-Times-Elites | 2026-03-02 | Route A | PSI, elite overproduction, secular cycles |
| Emery-Trist-1965 | 2026-03-02 | Route A | Original L11/L12/L21/L22 framework |
| Lizier-2012 | 2026-03-02 | Route A + coordinate crop | Synchronizability, transfer entropy. Compact 2-column format |
| Lizier-Synchronizability-Slideshow | 2026-03-02 | Route A + full-page render | Slideshow PDF, 26 slides at 200 DPI |
| D&G-ATP-Ch15 | 2026-03-02 | Route A + raster crop | Continental philosophy, Computer Einstein halftone |
| Levinas-Totality-and-Infinity | 2026-03-03 | Route A | Continental phenomenology, 3 PDFs combined |
| deGuerre-2016-TwoStageModel | 2026-03-03 | Route A + raster crop | Practitioner-applied, 4 figures |
| Emery-M-2000-CurrentVersionOST | 2026-03-03 | Route A + raster crop | Mixed register, 4 figures. 8 new cross_source entries |
| Ruesch-Bateson-Communication | 2026-03-03 | Route A | Four levels of communication, 2 excerpts combined |

## Glossary

Key project terms (mirrors the project skill's terminology glossary):

| Term | Definition |
|------|-----------|
| TAP | Theory of the Adjacent Possible |
| sigma-TAP | Extended TAP with sigma(Xi) feedback modulation |
| L-matrix | Four-channel interaction accounting (L11, L12, L21, L22) |
| TAPS | Transvolution, Anopression, Praxis, Syntegration |
| RIP | Recursive/Reflective, Iterative/Integrative, Preservative/Praxitive |
| Cross-metathesis | Type-set exchange between agents |
| Youn ratio | Measure of combinatorial vs truly novel innovation |
| PSI | Political Stress Indicator (Turchin) |
| Directive correlation | Emery-Trist concept for shared environmental orientation |
| Metathesis | Dialectical moment of transformation |
| Practico-inert | Sedimented praxis — action crystallized into fixed structure (Sartre) |
| Unificity | Generative triad above TAPS: Unity-Multiplicity-Unificity |

## Integration

Distillation outputs feed into the buffer system:

- **Concept map**: New cross_source entries added to warm layer (currently 158 entries)
- **Convergence web**: Inter-source linkages created as tetradic entries (currently 41 entries across 16 clusters)
- **Hot digest**: Updated so `/onhand` surfaces new mappings at next session start
- **MEMORY.md**: Conservative updates (full integration mode)

## Configuration

To change settings, run `/distill` and choose "Re-differentiate." To install additional specialist tools (pdfplumber, Docling, Marker), they will be offered automatically when relevant content is detected during distillation.

## Known Issues Summary

Key patterns discovered across distillations:

- **Table heuristic false-positives**: Academic two-column PDFs with paragraph indentation consistently trigger table detection. All Route A (clean text) — accepted as expected behavior.
- **Slideshow PDFs**: Scan heuristics over-trigger on presentation format. Full-page rendering at 200 DPI is the correct approach.
- **Text-only tables/equations**: Invisible to vector and raster detection channels. Use text-block coordinate cropping as third extraction channel.
- **Windows path issues**: CWD defaults to user directory, not repo root. Use absolute paths for all script outputs.

See the project SKILL.md's Known Issues table for the full cumulative log.
