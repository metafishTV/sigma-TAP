# sigma-TAP Claim Policy

This policy governs how claims are labelled and communicated in sigma-TAP outputs,
ensuring the project remains collaborative with the TAP/biocosmology research program
rather than competitive with it.

## Allowed claim labels

Every claim produced by the sigma-TAP pipeline must carry exactly one of the following labels.

### paper-aligned

The claim **restates or directly implements** a result from the source papers.

Requirements:
- At least one citation to a specific section, equation, or figure in the source literature.
- At least one generated artifact (CSV, figure, or test) that reproduces or validates the claim.
- Stress-tested across at least two TAP variants where applicable.

Source papers:
- Cortes, Kauffman, Liddle & Smolin (2022/2025), "The TAP equation" (`TAPequation-FINAL.pdf`)
- Cortes, Kauffman, Liddle & Smolin (2022), "Biocosmology: Towards the birth of a new science" (`Paper1-FINAL.pdf`)
- Cortes, Kauffman, Liddle & Smolin (2022), "Biocosmology: Biology from a cosmological perspective" (`Paper2-FINAL.pdf`)
- Cortes, Kauffman, Liddle & Smolin (2025), "The TAP equation: evaluating combinatorial innovation in biocosmology" (`Applications-of-TAP.pdf`)
- Taalbi (2025), "Long-run patterns in the discovery of the adjacent possible" (`Long-run patterns...pdf`)

### paper-consistent extension

The claim **extends** the source framework in a direction that is consistent with it,
but is not explicitly stated in any source paper.

Requirements:
- At least one citation showing which paper concept is being extended.
- Clear statement of what is new relative to the source literature.
- At least one generated artifact supporting the claim.
- Mandatory disclaimer (see template below).

### exploratory

The claim is **novel** and not directly grounded in the source papers.
It may use TAP machinery but draws conclusions the papers do not make.

Requirements:
- Explicit "exploratory" label visible in any output that references the claim.
- Mandatory disclaimer (see template below).
- Must not be presented as a finding of the TAP program itself.

## Disclaimer templates

### For paper-consistent extension claims

> This result extends the TAP framework in a direction consistent with but not
> explicitly stated in [source paper]. It should be interpreted as a
> computational exploration, not as a finding of the source literature.

### For exploratory claims

> This result is exploratory and does not derive from the source TAP/biocosmology
> literature. It uses sigma-TAP computational infrastructure but represents an
> independent analytical direction that requires further validation.

## Mapping to CLAIMS.md

| CLAIMS.md status | Policy label |
|------------------|-------------|
| `supported` | `paper-aligned` |
| `partial` | `paper-aligned` or `paper-consistent extension` (case-by-case) |
| `exploratory` | `exploratory` |

## Enforcement

- Every section of generated reports must carry one policy label.
- The `run_reporting_pipeline.py` pipeline should verify that no claim marked
  `paper-aligned` lacks both a paper citation and a supporting artifact.
- Exploratory outputs must include the disclaimer template above or an equivalent.

## Rationale

The TAP equation and Theory of the Adjacent Possible represent an active research program
led by Kauffman, Cortes, Liddle, Smolin, and collaborators. The sigma-TAP project contributes
reproducible computational evidence to this program. This policy ensures we:

1. Faithfully represent what the papers actually claim.
2. Clearly distinguish our computational extensions from the source theory.
3. Avoid presenting exploratory results as established TAP findings.
4. Maintain a collaborative posture with the source research program.
