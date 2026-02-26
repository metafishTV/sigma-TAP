# sigma-TAP Claims Matrix

Status definitions:
- **supported**: Claim backed by source paper citation AND generated artifact.
- **partial**: Claim has paper basis but artifact support is incomplete or single-variant.
- **exploratory**: Novel claim not directly in source papers; requires further validation.

| ID | Claim | Source paper(s) | Supporting artifact(s) | Status |
|----|-------|----------------|----------------------|--------|
| C1 | TAP dynamics admit a variant family (baseline, two-scale, logistic) with qualitatively distinct regime behavior | TAPequation-FINAL.pdf; Applications-of-TAP.pdf | `outputs/variant_comparison.csv` via `scripts/sweep_variants.py` | supported |
| C2 | Regime transitions (plateau, exponential, explosive, extinction) are detectable from M(t) and Xi(t) trajectories | TAPequation-FINAL.pdf | `simulator/analysis.py::classify_regime`; `outputs/variant_comparison.csv` | supported |
| C3 | Innovation-rate scaling exponent sigma distinguishes TAP super-linear dynamics from resource-constrained growth; Heaps' law beta < 1 confirms sublinear diversification under resource constraints | Long-run patterns in the discovery of the adjacent possible.pdf | `simulator/analysis.py::innovation_rate_scaling`; `simulator/longrun.py::heaps_law_fit`; `outputs/realworld_fit.csv`; `outputs/longrun_diagnostics_summary.json` | supported |
| C4 | Real-world combinatorial growth (Wikipedia, npm, species) fits tamed TAP (power-law kernel) better than exponential or pure logistic null models | Applications-of-TAP.pdf; Long-run patterns.pdf | `scripts/fit_realworld.py`; `outputs/realworld_fit.csv` | supported |
| C5 | Decision bandwidth B(t) and praxiological Reynolds Re_prax provide interpretive turbulence diagnostics for TAP trajectories | (exploratory extension of TAP framework) | `simulator/turbulence.py`; `outputs/figures/turbulence_bandwidth.png` | exploratory |
| C6 | Extinction rate mu controls transition timing: higher mu delays or prevents explosive onset | TAPequation-FINAL.pdf | `outputs/extinction_sensitivity.csv`; `outputs/figures/extinction_sensitivity.png` | partial |
| C7 | Adjacency parameter a controls combinatorial explosion rate; currently fixed at a=8 across analyses | TAPequation-FINAL.pdf | `outputs/adjacency_sensitivity.csv`; `outputs/figures/adjacency_sensitivity.png` | partial |
| C8 | Metathetic multi-agent TAP dynamics produce Heaps' law (beta < 1) and non-winner-take-all concentration (Gini < 0.5), consistent with Taalbi (2025) predictions under resource-constrained recombinant search | Taalbi (2025); Emery & Trist (1965) (exploratory extension) | `simulator/metathetic.py`; `outputs/longrun_diagnostics_summary.json`; `outputs/figures/heaps_law.png`; `outputs/figures/concentration_gini.png` | exploratory |
