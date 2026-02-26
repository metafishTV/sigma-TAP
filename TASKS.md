# sigma-TAP task roadmap (paper-first execution plan)

This roadmap translates the review conclusions into concrete build tasks, in priority order, with definitions of done.

> Engineering execution detail is in `ENGINEERING_PLAN.md` (interfaces, modules, artifact contracts, and test gates).

## Priority 0 — Guardrails (must exist before new claims)

### T0.1 Paper-first claims matrix
**Goal:** ensure every claim is traceable to source papers + artifacts.

- Create `paper_first_claims_matrix.md` with columns:
  - claim ID
  - claim text
  - source paper citation(s)
  - artifact(s) that support claim
  - status (`supported`, `partial`, `exploratory`)
- Add one row for each headline claim in manuscript/report docs.

**Definition of done**
- Every headline claim in reporting docs has a matrix row.
- No claim marked `supported` without both citation and artifact.

### T0.2 Non-overclaiming policy
**Goal:** keep collaborative posture with TAP/biocosmology program.

- Add `CLAIM_POLICY.md` defining allowed labels:
  - `paper-aligned`
  - `paper-consistent extension`
  - `exploratory`
- Add short required disclaimer template for exploratory outputs.

**Definition of done**
- Report/manuscript generator includes one policy label per section.

---

## Priority 1 — Model and inference work

### T1.1 Variant stress-test mode
**Goal:** compare at least two TAP variants for each key claim.

- Implement `variant_stress_test` run mode with minimum variants:
  - baseline TAP
  - two-scale-like TAP
  - logistic-style TAP
- Produce side-by-side summary table:
  - fit quality
  - transition timing
  - blow-up horizon proxy
  - uncertainty intervals

**Definition of done**
- One command generates a single comparison artifact (CSV/MD/JSON).
- At least one existing claim is re-evaluated across variants.

### T1.2 Regime diagnostics as first-class outputs
**Goal:** operationalize long-run regime logic.

- Add explicit regime metrics:
  - plateau duration
  - acceleration onset
  - explosive onset / cap-hit step
  - extinction regime indicators
- Add plotting templates for regime transitions.

**Definition of done**
- Each run exports regime metrics in machine-readable form.
- Report tables include regime metrics by default.

### T1.3 Extinction/obsolescence sensitivity panel
**Goal:** treat extinction as a key control variable.

- Add controlled sweeps over extinction parameter(s).
- Quantify sensitivity of transition timing and occupancy.

**Definition of done**
- Sensitivity section appears in report with CI/uncertainty bands.

---

## Priority 2 — Long-run empirical alignment (new paper integration)

### T2.1 Long-run diagnostics module
**Goal:** mirror the long-run adjacent-possible paper’s empirical emphasis.

- Add diagnostics for:
  - innovation-rate scaling with cumulative innovations
  - concentration/distribution across organizations (Zipf/Heaps-style summaries where relevant)
  - diversification rate over time
- Add compatibility notes when dataset granularity differs.

**Definition of done**
- `outputs/long_run_diagnostics.*` artifacts are produced in pipeline runs.

### T2.2 Constraint decomposition (adjacency vs resource/search)
**Goal:** distinguish mechanisms behind observed dynamics.

- Add report section tagging observed behavior as:
  - adjacency-limited
  - resource/search-limited
  - mixed
- Define explicit decision heuristics and confidence level.

**Definition of done**
- Every major result includes one constraint tag + confidence.

---

## Priority 3 — Interpretation and scientific communication

### T3.1 Mechanistic + functional interpretation block
**Goal:** align with Type III explanatory requirements.

- For each headline result, include:
  1. mechanistic inference summary (parameters/regimes)
  2. functional interpretation summary (what role/selection pressure this implies)

**Definition of done**
- Report template enforces both blocks; missing blocks fail report build.

### T3.2 Evidence ladder in outputs
**Goal:** separate strong findings from exploratory hints.

- Add standardized confidence tiers:
  - Tier A: replicated + robust across variants
  - Tier B: robust in one variant with sensitivity support
  - Tier C: exploratory

**Definition of done**
- All result tables and figures include tier labels.

---

## Priority 4 — Reproducibility and auditability

### T4.1 One-command reproducible run
**Goal:** regenerate all key artifacts from clean checkout.

- Add top-level command/script to run full paper-aligned pipeline.
- Emit run manifest: git SHA, config hash, timestamp, artifact list.

**Definition of done**
- Clean run reproduces expected artifact inventory without manual steps.

### T4.2 Artifact-to-claim auditor
**Goal:** verify every claim has evidence links.

- Extend/implement automated checker for claim ↔ artifact ↔ citation linkage.

**Definition of done**
- CI-style check fails when any supported claim lacks links.

---

## Suggested implementation sequence (2-week sprint)

### Week 1
1. T0.1 Paper-first claims matrix
2. T0.2 Non-overclaiming policy
3. T1.1 Variant stress-test mode (minimum viable)
4. T1.2 Regime diagnostics outputs

### Week 2
5. T1.3 Extinction sensitivity panel
6. T2.1 Long-run diagnostics module
7. T2.2 Constraint decomposition tags
8. T3.1 Mechanistic + functional interpretation block

Stretch:
- T3.2 Evidence ladder
- T4.1 one-command reproducibility

---

## Immediate next action (today)

Create and populate:
- `paper_first_claims_matrix.md`
- `CLAIM_POLICY.md`

Then run one baseline + two-scale-like variant comparison and publish a single compact summary table as the first aligned artifact.
