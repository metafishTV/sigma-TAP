# Skill Dependencies Uncover Nested Human Capital — Distillation

**Source**: Hosseinioun, M., Neffke, F., Zhang, L. & Youn, H. "Skill dependencies uncover nested human capital." *Nature Human Behaviour* 9, 673-687 (April 2025). https://doi.org/10.1038/s41562-024-02093-2

**Core contribution**: Demonstrates that the skill-occupation structure of the US economy exhibits a **nested hierarchy** borrowed from ecology, where occupations requiring specialized skills are subsets of those demanding general skills. Asymmetric conditional probabilities between skill co-requirements infer directional dependency chains. A nestedness contribution score (cs) partitions skills into nested (cs > 0) and unnested (cs < 0) categories, with profound implications for wage premiums, career trajectories, demographic disparities, and increasing barriers to upward mobility. Data: O*NET (120 skills x ~1000 occupations), OEWS wages, CPS demographics (1980-2022), 70M job transitions from 20M anonymized resumes.

---

## 1. Motivation and Framing

Modern economies require increasingly diverse and specialized skills. Human capital theory has evolved from years-of-schooling metrics to network-based analysis of skill portfolios. The central research question: what do skill **interdependencies** look like, and what are their implications?

Key framing: skills are not merely complementary (synergistic combinations); they are **sequential** and **cumulative** — each step builds upon the previous. This dependency structure adds depth beyond simple co-occurrence.

The paper makes the distinction between:
- **Skill complementarity** (existing literature): skills that work together synergistically
- **Skill dependency** (this paper): skills that must be acquired *in order*, one prerequisite for the next

---

## 2. Data Sources

| Dataset | Coverage | Key Variables | Use |
|---|---|---|---|
| **O*NET** (v24.1, 2019; v9.0, 2005) | ~1000 occupations x 120 items (skills + knowledge + abilities) | Importance (1-5), Level (0-7), Education (12 grades) | Skill generality, dependency inference, nestedness |
| **OEWS** (2005, 2019) | 774 occupations (6-digit SOC) | National/regional wages and employment | Wage analysis |
| **CPS** (1980-2022) | Monthly household surveys | Occupation, wages, hours, gender, race/ethnicity, age | Career trajectories, demographic analysis |
| **Resume dataset** (2007-2020) | 20M anonymized resumes, 70M job transitions | 8-digit SOC job sequences (no demographics) | Direct career trajectory observation |

**Primary unit**: Skill **level** (0-7), chosen over importance because advanced skills develop through sequential training, better capturing acquisition dependencies. Importance and level highly correlated (r = 0.94); results robust to either measure.

---

## 3. Skill Generality — Three Categories

Skills categorized by the **shape of their demand distributions** across occupations:

| Category | Count | Median Level | Distribution Shape | Examples |
|---|---|---|---|---|
| **General** | 31 | 3.34 | Peaked at levels 3-4; most jobs need high proficiency | English language, oral expression, deductive reasoning |
| **Intermediate** | 43 | 2.37 | Moderately distributed | Math skills, systems analysis, negotiation |
| **Specific** | 46 | 0.87 | Skewed toward zero with long tail; most jobs need none, few need high levels | Dynamic flexibility, programming, fine arts |

Grouping method: k-means clustering on demand profile shapes (optimal k = 3). Results robust across k = 2, 3, 4, alternative clustering methods, and unit choices.

Three consistent measures of generality:
1. **Demand distribution shape** (above)
2. **Median skill level** across occupations
3. **Local reaching centrality** in the dependency network (number of skills reachable through outgoing edges)

Strong correlation between reaching centrality and median level: rho ~ 0.71.

---

## 4. Skill Dependency Inference via Asymmetric Conditional Probability

### Method

For each skill pair (A, B):
1. Convert continuous skill levels [0,7] to binary presence/absence using disparity filter (Serrano et al. 2009), preserving ranking invariance
2. Calculate conditional probabilities: p(A|B) and p(B|A)
3. Apply z-score threshold (zth) to filter random co-occurrence noise
4. Apply asymmetry threshold (alpha_th) differentially weighted by node degree
5. **Only when p(A|B) >> p(B|A)** is a directional dependency A -> B assigned
6. Dependency weight = function of conditional probability difference, adjusted for null model of expected shared occupations given degrees

### Key Examples

- **math -> programming**: p(math|programming) = 3/4 > p(programming|math) = 3/7. Math is prerequisite for programming.
- **oral expression -> negotiation**: p(oral expression|negotiation) < p(negotiation|oral expression). Oral expression is prerequisite for negotiation.
- **math <-> dynamic flexibility**: p(dynamic_flexibility|math) ~ 0. No dependency (independent, rare co-occurrence filtered out).

### The Dependency Network

Nodes colored by generality group, positioned by:
- x-axis: educational requirements
- y-axis: local reaching centrality (hierarchy order)

The network displays hierarchical nested chains from general (top) through intermediate (middle) to specific (bottom). Example chains:
- deductive/inductive reasoning -> systems analysis -> programming
- oral expression -> systems analysis -> negotiation

### Directionality and Acquisition Dependencies

The inferred directionality reflects **acquisition dependencies** — the order in which skills are learned — not merely seniority effects. Evidence:
1. Results hold when excluding management titles and social skills
2. Case study: registered nurse -> nurse practitioner transitions show skill/wage differences aligned with nested dependency chains
3. Hispanic immigrant case study: limited English proficiency blocks language-dependent nested skill development, creating demographic skill entrapment
4. Sequential datasets (occupational ages, survey ages, resume job sequences) support the developmental ordering

---

## 5. Nestedness and Contribution Score

### Nestedness (N)

Borrowed from ecology (plant-pollinator mutualistic networks). Measures how often specialists preferentially engage with generalists — translated here as: how often specific skills are demanded in occupations that also require general ones, more than by random chance.

The skill-occupation matrix shows an **imperfect upper-left triangle** when sorted by marginal totals (Supplementary Fig. 6). Multiple nestedness metrics tested:

| Metric | 2019 Value | 2005 Value | Direction |
|---|---|---|---|
| Overlap index (Nc) | 651,030 | 573,873 | Increasing |
| Checkerboard score | 356.4 | 438.67 | Decreasing (= more nested) |
| Temperature | 31.89 | 40.07 | Decreasing (= more nested) |
| NODF | 41.72 | 39.06 | Increasing (= more nested) |

All metrics agree: structure is nested and becoming more so.

### Contribution Score (cs)

For each skill s:

**cs = (N - <N*_s>) / sigma_N*_s**

Where <N*_s> and sigma_N*_s are mean and standard deviation from 5000 randomized counterparts in which skill s's dependencies are shuffled while preserving its generality (demand distribution shape).

- **cs > 0**: skill's dependencies **reinforce** the overall nested hierarchy (aligned)
- **cs < 0**: skill's dependencies **diminish** the nested hierarchy (misaligned)

### Five Functional Skill Categories

| Category | Definition | cs Sign | Examples |
|---|---|---|---|
| General | Broadly applicable foundation | (always positive) | English language, oral expression, deductive reasoning |
| Nested intermediate | Intermediate skills reinforcing hierarchy | cs > 0 | Math skills, systems analysis, negotiation |
| Unnested intermediate | Intermediate skills breaking from hierarchy | cs < 0 | (varies) |
| Nested specific | Specific skills embedded in dependent chains | cs > 0 | Programming, fine arts, science |
| Unnested specific | Specific skills outside nested structure | cs < 0 | Repairing, dynamic flexibility, mechanical, control precision |

### Reachability

Arrival probability (hitting probability via random walks on weighted directed adjacency matrix) quantifies how many pathways lead to each skill through nested dependency chains. For nested skills (programming, negotiation): many general skills reach them. For unnested skills (repairing): few general skills lead to it — it doesn't rely on nested prerequisites.

### Imperfect Nestedness — Why Some Skills Are Unnested

Attributed to **human capital constraints within occupations**, analogous to carrying capacity in ecology. The scope of occupations remains mostly constant (Supplementary Fig. 5) — workers can only learn/perform so much. These constraints drive specialization into modular structures where certain skills become mutually exclusive, producing nested-modular architecture alongside pure nestedness.

---

## 6. Socioeconomic Correlates

### cs Correlates With

| Attribute | Correlation with cs | Direction |
|---|---|---|
| Generality (LRC) | Strong positive | Higher cs = more general |
| Education level | Positive | Higher cs = longer education required |
| Wage premium (level-weighted) | Positive | Higher cs = higher wages |
| Automation risk | Negative | Higher cs = lower automation risk |

### Wage Premiums and the Foundation Effect (Figure 5)

**Critical finding**: The substantial wage premiums and higher educational requirements associated with nested specializations **almost fully disappear** when controlling for general skill levels.

- Nested specific skills: positive wage slope (unconditional) -> near-zero slope (conditional on general skills)
- Unnested specific skills: **negative** wage slope (unconditional) -> **positive** slope (conditional on general skills)

Interpretation: unnested skills DO hold labour market value, but their potential premiums are diminished by lacking the nested dependency foundation. The general skill base is what drives the wage premium, not the specialization itself.

---

## 7. Career Trajectories (Figure 4)

### Three Concordant Datasets

**Dataset 1: Occupational median ages** (n_occ = 542)
- General and nested skill levels rise with occupation median age
- Unnested skill levels show no notable correlation with age

**Dataset 2: Synthetic birth cohorts** from CPS (n_survey = 1,493,142, 1980-2022)
- Sharp increase in general and nested skills until age 30
- Moderate decline in unnested skills until age 30
- After age 30: overall stabilization
- **Gender gap emerges at age 30**: men continue growing general/nested skills into 50s; women plateau in early 30s (coinciding with typical first motherhood)
- Parenthood effect: children associated with reduced general/nested skills for women; increased for men
- Work schedule effect: controlling for irregular hours and overtime reduces gender gap by >1/3

**Dataset 3: Resume job transitions** (n_moves = 12,561,319)
- General skills advance faster early (Delta_general_{i<3} >> Delta_nested_{i<3})
- Later: comparable growth (Delta_general_{i>3} ~ Delta_nested_{i>3})
- Skill profiles stabilize within first 5 jobs
- Randomized sequences show Delta_random_i ~ 0, confirming observed trends are career-specific

### Key Pattern

Career trajectories unfold through **nested specializations** where ongoing development of general skills supports their dependent, nested counterparts. This echoes the cumulative and sequential nature of skill development. Skill advancement extends well beyond formal education — nested specialization pathways persist throughout careers.

The paper challenges the simple linear progression model (basic -> advanced). Instead: general skills continue developing **alongside** specific ones. Critical thinking is prerequisite for negotiation, but within a negotiation role, further advancement still depends on honing higher levels of critical thinking.

---

## 8. Demographic Disparities (Figure 6)

### Race/Ethnicity (relative to white workers)

| Group | General Skills | Nested Skills | Unnested Skills | Education | Wages |
|---|---|---|---|---|---|
| Asian | ~1.0 | ~1.05 | ~0.85 | ~1.1 | ~1.05 |
| Black | ~0.95 | ~0.90 | ~0.90 | ~0.85 | ~0.75 |
| Hispanic/Latinx | ~0.85 | ~0.85 | ~1.1 | ~0.75 | ~0.70 |

Hispanic workers: elevated unnested skill requirements + depressed nested/general = **skill entrapment**. Case study: limited English proficiency blocks language-dependent nested skill pathways, pushing workers into unnested specializations with long-term wage penalties.

### Gender (within each race/ethnicity, female vs male)

Across most groups: women work in occupations requiring more education and higher general skills than men — but NOT correspondingly higher nested skills. This disconnect between education/general skills and wages/nested skills contributes to the gender wage gap.

Gaps have narrowed over time (Supplementary Figs. 51, 52) but remain structural.

### Policy Implication

Closing wage gaps requires **tailored solutions**: for Black workers and Hispanic workers, addressing access to nested skill pathways (especially language-dependent ones); for women, addressing the parenthood-driven plateau in nested skill accumulation.

---

## 9. Increasing Nestedness Over Time (Figure 7)

Between 2005 and 2019, the skill structure has become **more nested** by all four metrics. This trend is attributed to:
- Increasingly uneven job requirements
- Growth in high-dependency branches
- Decline in low-dependency branches
- Workers with broad skill sets seeing higher wage premiums (driven by nested specialization demand)
- Demand for unnested skills declining

**Consequence**: deeper nested structure imposes greater constraints on individual career paths, amplifies disparities, and potentially raises new barriers to upward mobility. Workers with insufficient foundational general skills get stuck in unnested specialization paths with limited opportunity.

---

## 10. Methods — Technical Details

### Binary Conversion (Disparity Filter)

Continuous skill levels [0,7] converted to binary presence/absence using Serrano et al. (2009) disparity filter. Parameters chosen to satisfy two ranking-preservation conditions:
1. Skills ranked by total level are ranked similarly by total presence
2. Occupations ranked by total skill level are ranked similarly by total skill count

### Dependency Inference

Two thresholds:
- **z_th**: filters non-significant co-occurrences
- **alpha_th**: filters non-significant asymmetry (differentially weighted by node degree)

Dependency weight: function of (P(u|v) - P(v|u)), adjusted for null model of expected shared occupations given respective degrees.

### Nestedness Contribution

For each skill s: 5000 randomizations preserving generality (demand distribution) but shuffling occupational presence. cs = (N_observed - mean(N_randomized)) / std(N_randomized).

### Arrival Probability (Reachability)

R_{i,j} = sum_l M^l_{i,j} for random walks of length l on weighted directed adjacency matrix. Summed to saturation. Computed using R package `markovchain`.

### Career Trajectories — Synthetic Birth Cohorts

CPS provides yearly cross-sections, not longitudinal tracking. Synthetic cohorts: connect individuals born in same year across different survey rounds (e.g., 1967 cohort traced across 1995-2015 surveys).

### Historical Comparison

O*NET v9.0 (2005) vs v24.1 (2019). Crosswalk: matching occupation codes through consecutive years (2005->2006->...->2019). 968 occupations in 2019, 941 in 2005. Automatic matching + manual matching for remainder.

---

## 11. Discussion — Key Claims

1. Skills aligned with nested hierarchy yield **greater economic rewards**; those outside miss growing benefits
2. The process of selecting and acquiring skills has become increasingly crucial but the most valuable skills require **cumulative learning and prerequisites to unlock**
3. **Directional dependencies challenge symmetric co-occurrence networks** — the traditional approach in economic complexity
4. The hierarchical organization of skills is **directional rather than mutual** — a dependency structure, not just a correlation structure
5. As complexity and specialization increase, deeper nestedness imposes greater constraints on career paths, amplifies disparities, and has **macroeconomic implications** for system resilience and stability
6. **Skill development extends beyond schooling** — nested specialization pathways persist through entire careers
7. **General skills develop alongside specific ones**, not just before them — challenges simple linear progression model

### Limitations

- Framework is one perspective on hierarchical human capital dependencies
- Analysis centered on US labour market (unique education, industry, urban structures)
- Generalization to entrepreneurship, different development stages remains open
- Inferred directionality may not fully capture all microprocesses of skill acquisition (may include firm-imposed seniority effects, though robustness checks suggest broader acquisition dependencies dominate)

---

## 12. Key Concepts for sigma-TAP Integration

1. **Asymmetric conditional probability = asymmetric cross-metathesis directionality (section 5.10)**. The paper's p(A|B) >> p(B|A) inference of directional dependency maps directly to the initiator/responder asymmetry in cross-metathesis. Agent A can absorb from Agent B but not vice versa, because A possesses prerequisite type elements that B lacks.

2. **Nested hierarchy = TAPS signature hierarchy**. The three-tier structure (general -> intermediate -> specific) with directional dependency chains maps to TAPS signature ordering. Agents with foundational signatures (broad, general) are prerequisite to agents developing specialized signatures. This should inform the cold-start threshold (_MIN_SIG_EVENTS) and signature-based tension classification.

3. **Contribution score cs = type_set alignment with metathetic field**. cs > 0 means a skill (type element) reinforces the combinatorial structure; cs < 0 means it deviates. In sigma-TAP terms: type elements with high cs have more metathetic potential (more adjacent-possible pathways). Type elements with low cs are structurally isolated — they exist but don't participate in the nested expansion. This should inform the **annular distribution** for adjacent-element generation (section 5.23): nested elements generate at the sweet-spot radius; unnested elements generate at the periphery or deadzone.

4. **Wage premium vanishes when controlling for general skills = sigma(Xi) effect**. The general skill base IS the affordance base. Without sufficient Xi (affordance exposure / general skill accumulation), the TAP birth term fires but produces no real growth. This validates the critical Phase 1 gap: sigma must modulate the metathetic growth equation, and the modulation is primarily driven by the breadth of the general foundation, not the height of the specialization.

5. **Increasing nestedness over time = metathematization**. The tightening of nested structure (2005-2019) maps to your concept of metathematization: the temporal process by which the system's own combinatorial history constrains future combinatorial possibilities. The adjacent possible doesn't just expand — it develops *directed dependency channels* that funnel future growth.

6. **Skill entrapment = failed metathematization / demetathesis**. Hispanic workers trapped in unnested skill pathways with long-term wage penalties = agents that fail to accumulate sufficient general type elements to enter the nested expansion region. They persist (don't die) but cannot grow. This is the "inertial" temporal state in the 5-state machine.

7. **Career trajectory stabilization within 5 jobs = convergence to attractor**. The observation that skill profiles stabilize quickly maps to the idea that agents rapidly find a TAPS signature and settle into it. The question of whether this is healthy (established state) or pathological (premature hexis) depends on whether the signature is nested (growth potential) or unnested (entrapment).

8. **Ecology parallel: nestedness promotes feasibility over stability**. Saavedra et al. showed nested mutualistic networks promote feasibility (can more species coexist?) over stability (does the system return to equilibrium after perturbation?). In sigma-TAP terms: nestedness in the type-set space promotes the **feasibility** of new combinations (larger adjacent possible) over the **stability** of existing configurations. This tension between feasibility and stability IS the tension between the TAP birth term and the extinction term mu*M.

9. **Product space / small-world network properties**. The skill dependency network's small-world properties connect to Lizier et al.'s finding that small-world networks are less synchronizable than random ones. The clustered, modular structure that makes nested human capital possible also makes global synchronization difficult — suggesting that sigma-TAP agents should exhibit **local** synchronization within modules but **desynchronization** across modules.

10. **Five functional categories as type_set partitioning**. The five categories (general, nested-intermediate, unnested-intermediate, nested-specific, unnested-specific) suggest that type_set elements in sigma-TAP should carry a nestedness attribute. This would allow the model to distinguish between agents accumulating nested type elements (growth trajectory) and those accumulating unnested elements (entrapment trajectory) — a qualitative distinction the current boolean type_set cannot make.

11. **"Every unity is a multiplicity of at least two" (Unificity.md)**. The paper shows that even general skills — the broadest, most "unitary" — contain internal structure (different proficiency levels, different combinations). No skill is atomic. This validates the Unificity principle at the empirical level.

12. **Heaps' law bridge to Taalbi**. The product space analysis and the finding that diversification follows structured pathways (not random exploration) connects to Taalbi's Heaps' law (D ~ k^0.587). Youn's nestedness gives the *architecture* of that diversification; Taalbi's exponent gives the *rate*. Together: calibration target + structural constraint.
