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
| Cortes_etal_TAPEquation_2022_Paper | 2026-03-04 | Route A + full-page render | **Re-distilled**. 15 concepts, 5 full-page renders (pp 3-7), 6 cross_source (w:189-194), 4 convergence_web (cw:61-64). Merged with Applications-of-TAP |
| Cortes_etal_BiocosmologyBirth_2022_Paper | 2026-03-04 | Route A + figure crop | Re-distilled. 18 concepts, 1 figure, 10 cross_source (w:165-174), 6 convergence_web (cw:46-51) |
| Cortes_etal_BiocosmologyPerspective_2022_Paper | 2026-03-04 | Route A + figure crop | Re-distilled. 16 concepts, 1 figure, 8 cross_source (w:175-182), 5 convergence_web (cw:52-56) |
| Taalbi_LongRunPatterns_2025_Paper | Pre-skill | Route A | Empirical benchmark, Youn ratio |
| Turchin_SocialPressures_2013_Paper | 2026-03-02 | Route A | PSI, elite overproduction, secular cycles |
| Emery_Trist_CausalTexture_1965_Paper | 2026-03-02 | Route A | Original L11/L12/L21/L22 framework |
| Lizier_etal_InfoStorageLoopMotifs_2012_Paper | 2026-03-02 | Route A + coordinate crop | Synchronizability, transfer entropy. Compact 2-column format |
| Lizier_SynchronizabilitySlideshow_2023_Slideshow | 2026-03-02 | Route A + full-page render | Slideshow PDF, 26 slides at 200 DPI |
| Deleuze_Guattari_ATPCh15_1987_Chapter | 2026-03-02 | Route A + raster crop | Continental philosophy, Computer Einstein halftone |
| Levinas_TotalityInfinity_1961_Excerpt | 2026-03-03 | Route A | Continental phenomenology, 3 PDFs combined |
| deGuerre_TwoStageModel_2016_Paper | 2026-03-03 | Route A + raster crop | Practitioner-applied, 4 figures |
| Emery_M_CurrentVersionOST_2000_Paper | 2026-03-03 | Route A + raster crop | Mixed register, 4 figures. 8 new cross_source entries |
| Ruesch_Bateson_Communication_1951_Excerpt | 2026-03-03 | Route A | Four levels of communication, 2 excerpts combined |
| Jakobs_Communicology_2016_Chart | Pre-skill | Route A | 4-level communicology matrix |
| Bateson_TableD_1951_Table | Pre-skill | Route I | Table D image extraction |
| Hosseinioun_etal_NestedHumanCapital_2025_Paper | Pre-skill | Route A | Nested skill-occupation hierarchy |
| Emery_Emery_ParticipativeDesign_1974_Paper | Pre-skill | Route A | PDW methodology, six criteria |
| Turchin_Gavrilets_HierarchicalSocieties_2009_Paper | Pre-skill | Route A | Multilevel selection, Dunbar limit |
| Turchin_FormationLargeEmpires_2009_Paper | Pre-skill | Route A | Mirror-empires, asabiyya |
| Lizier_etal_SynchronizabilityMotifs_2023_Paper | Pre-skill | Route A | Process motif decomposition |
| OpenSystemsTheory_PractitionerSite_Website | 2026-03-03 | Route W | Re-distilled. 18 concepts, 6 cross_source, 4 convergence_web |
| Unificity | 2026-03-04 | Direct read (markdown) | Re-distilled. 15 concepts, 6 cross_source (w:183-188), 4 convergence_web (cw:57-60) |

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
| Kantian Whole | Parts exist for and by means of the Whole (Kauffman via Kant) |
| Functional explanation | X exists because X contributes to survival of larger system S |
| Constraint closure | Constraints constrain energy release in processes that construct the same constraints (Montévil & Mossio) |
| False equilibrium | Equilibrium calculated from incomplete theory — masks deeper dynamics |
| Entailment laws | Three structural laws binding the modalities: downward closure, upward opening, third-party |
| 1′ (one-prime) | Each modality is a complete perspective on the total, not a fraction (non-divisive totality) |
| Consummation / consumption | Consumption = taking up sedimented structure; consummation = integrating into new form |
| Extinction instability | μ_critical equilibrium is unstable — bare TAP has no stable intermediate (blow-up or collapse) |
| Tetration | Exponential tower 10^10^10^... — TAP late-time growth regime |
| Two-scale TAP | Variant with α₁M_t (single-object evolution) + α combinatoric — exponential phase → blow-up |

## Integration

Distillation outputs feed into the buffer system:

- **Concept map**: New cross_source entries added to warm layer (currently 194 entries, w:44-w:194)
- **Convergence web**: Inter-source linkages created as tetradic entries (currently 64 entries, cw:1-cw:64)
- **Hot digest**: Updated so `/buffer:on` surfaces new mappings at next session start
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
