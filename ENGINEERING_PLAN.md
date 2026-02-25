# sigma-TAP engineering plan (paper-first implementation)

This plan turns the current documentation-level roadmap into concrete engineering work packages, with interfaces, artifacts, and validation checks.

## Design constraints (from paper-first focus)

1. Implement TAP as a **variant family** (baseline, two-scale, logistic-like), not one fixed dynamic.
2. Make **regime transitions** first-class outputs (plateau, acceleration, explosive onset, extinction).
3. Add **long-run innovation diagnostics** (rate scaling, concentration, diversification) inspired by the newly added long-run paper.
4. Enforce **claim traceability**: every supported claim must point to both source-paper citations and generated artifacts.

---

## Work Package A — Core model architecture

### A1. Variant interface
Create a shared interface in `simulator/variants/`:
- `step(state, params) -> state_next`
- `diagnostics(state, history) -> dict`
- `name() -> str`

Implementations:
- `baseline_tap.py`
- `two_scale_tap.py`
- `logistic_tap.py`

**Deliverables**
- Variant registry (`variants/__init__.py`) with stable IDs.
- Config schema listing allowed variants.

**Validation**
- Unit tests confirm each variant respects non-negativity and deterministic replay under fixed seed.

### A2. Regime detector module
Create `simulator/regimes.py`:
- Inputs: trajectory timeseries + thresholds
- Outputs: regime labels and transition indices

Required outputs:
- `plateau_start`, `plateau_end`
- `acceleration_onset`
- `explosive_onset`
- `extinction_flag`
- `blowup_proxy_step`

**Validation**
- Synthetic trajectory tests (known plateau→acceleration→explosive path).

---

## Work Package B — Pipeline and artifacts

### B1. Unified run command
Add a single entrypoint command:
- `python run_reporting_pipeline.py --mode variant_stress_test --config config/default.json`

Run stages:
1. load config
2. run variants
3. compute regime diagnostics
4. compute long-run diagnostics
5. build report tables
6. run claim-audit checks

**Deliverables**
- Run manifest: `outputs/run_manifest.json` with git SHA/config hash/artifact list.

### B2. Artifact contract
Define `outputs/ARTIFACT_SCHEMA.md` and enforce filenames:
- `outputs/variant_comparison.csv`
- `outputs/regime_diagnostics.csv`
- `outputs/long_run_diagnostics.csv`
- `outputs/claim_audit_report.json`

**Validation**
- Contract test fails pipeline if required artifacts missing or malformed.

---

## Work Package C — Long-run empirical diagnostics

### C1. Innovation-rate scaling module
Add `analysis/long_run.py` function(s):
- fit scaling of innovation rate vs cumulative innovations
- return slope, CI, goodness-of-fit

### C2. Concentration/distribution module
Compute organization-level concentration metrics:
- top-k share
- Zipf fit diagnostics
- inequality metric(s)

### C3. Diversification dynamics module
Compute diversification trajectory:
- cumulative new-type counts
- diversification rate over time
- regime-conditioned diversification statistics

**Deliverables**
- `outputs/long_run_diagnostics.csv`
- `outputs/long_run_diagnostics_summary.json`

**Validation**
- Statistical smoke tests + monotonicity/shape checks where mathematically required.

---

## Work Package D — Claim governance and reporting

### D1. Claim matrix implementation
Create machine-readable claim table:
- `claims/claims_matrix.csv`
- fields: `claim_id, claim_text, paper_refs, artifact_refs, evidence_tier, status`

### D2. Claim policy gate
Add pre-report check:
- `supported` requires non-empty `paper_refs` and `artifact_refs`
- `exploratory` requires disclaimer block

### D3. Report template enforcement
Require two interpretation blocks per headline result:
1. mechanistic inference
2. functional interpretation

**Deliverables**
- build fails if required interpretation blocks absent.

---

## Work Package E — Testing and CI

### E1. Test layout
- `tests/test_variants.py`
- `tests/test_regimes.py`
- `tests/test_artifact_contract.py`
- `tests/test_claim_gate.py`

### E2. Minimum CI checks
- lint + unit tests
- deterministic replay smoke test
- full pipeline smoke run on reduced grid

### E3. Reproducibility check
- rerun reduced pipeline twice, compare stable artifacts byte-wise or hash-wise where expected.

---

## Suggested implementation order

1. A1 Variant interface + baseline implementation
2. A2 Regime detector
3. B1 Unified run command + manifest
4. B2 Artifact contract tests
5. C1/C2/C3 long-run diagnostics
6. D1/D2 claim matrix and gate
7. D3 report enforcement
8. E CI and reproducibility checks

---

## First engineering sprint (practical)

### Sprint goal
Produce one reproducible artifact bundle comparing baseline vs two-scale on a reduced dataset with regime + long-run diagnostics and claim gating enabled.

### Sprint backlog
- [ ] Implement variant interface + 2 variants
- [ ] Implement regime detector (v1 thresholds)
- [ ] Emit `variant_comparison.csv` + `regime_diagnostics.csv`
- [ ] Emit `long_run_diagnostics.csv` (v1)
- [ ] Add `claims/claims_matrix.csv` with at least 5 claims
- [ ] Add claim gate to fail unsupported `supported` claims
- [ ] Add reduced-grid smoke test in CI

### Sprint acceptance criteria
- One command reproduces all required v1 artifacts.
- At least one prior claim is upgraded from narrative-only to citation+artifact supported.
- Build fails intentionally when artifact or claim links are missing.
