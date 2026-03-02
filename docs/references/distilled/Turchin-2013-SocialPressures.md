# Turchin-2013-SocialPressures — Distillation

> Source: Peter Turchin, "Modeling Social Pressures Toward Political Instability," *Cliodynamics: The Journal of Quantitative History and Cultural Evolution* 4(2): 241–280, 2013. 41 pages.
> Date distilled: 2026-03-02
> Distilled by: Claude (via distill skill)
> Register: empirical social science / formal-mathematical
> Tone: mixed (impersonal-objective with first-person methodological narration)
> Density: technical-specialist (assumes ODE modeling, regression analysis, demographic theory)

## Core Argument

Turchin argues that political instability follows secular cycles driven by structural-demographic variables whose dynamics are slow enough to model and forecast. The central claim is that instability results from the multiplicative interaction of three measurable pressures: mass mobilization potential (population discontent from declining relative wages), elite mobilization potential (intraelite competition from elite overproduction), and state fiscal distress (eroding institutional capacity). When all three pressures are simultaneously elevated, the system becomes vulnerable to instability events; when any one is low, the others are effectively suppressed. This multiplicative structure is formalized as the Political Stress Indicator (PSI = MMP × EMP × SFD).

The paper develops two complementary case studies applying this framework. The Antebellum model (1780–1860) constructs an ODE system tracking rural-urban population dynamics, relative wages, and elite numbers, demonstrating that PSI served as a leading indicator of the instability escalation culminating in the American Civil War. The Contemporary model (1927–2012) uses regression analysis to decompose real wage dynamics into three structural determinants — GDP per capita, labor supply/demand ratio, and non-market forces (proxied by real minimum wage) — then feeds predicted wage trajectories into the elite overproduction equation, showing that PSI began rising after 1980 and accelerated sharply after 2000.

Turchin explicitly separates slow-moving structural pressures (forecastable, amenable to policy intervention) from fast-moving triggers (inherently stochastic). The structural pressures determine the *severity* of potential instability; the triggers determine only *when* and *how* it manifests. The paper advocates building a spectrum of manageable models rather than one comprehensive model, following the principle that each model should address a somewhat different aspect of the problem while remaining simple enough to avoid structural instability.

## Key Concepts

| Concept | Definition | Significance |
|---------|-----------|--------------|
| Political Stress Indicator (PSI/Ψ) | Multiplicative composite: Ψ = MMP × EMP × SFD. Quantifies structural-demographic pressures toward instability | Central formalization; serves as leading indicator of instability in both historical case studies |
| Mass Mobilization Potential (MMP) | Index of popular readiness for collective action: MMP = w⁻¹ · (N_urb/N) · A₂₀₋₂₉ | Captures population-level discontent via inverse relative wages, urbanization, and youth bulge proportion |
| Elite Mobilization Potential (EMP) | Index of elite willingness to organize opposition: EMP = ε⁻¹ · (E/sN) | Captures intraelite competition through inverse relative elite income and elite numbers relative to government positions |
| State Fiscal Distress (SFD) | Index of state incapacity: SFD = (Y/G) · D | National debt relative to GDP multiplied by public distrust in government institutions |
| Relative wage (w) | Worker wage scaled by GDP per capita: w = W/(G/N) | Primary coupling variable between mass and elite compartments; when w declines, MMP rises and elite overproduction accelerates |
| Elite overproduction | Growth of elite numbers beyond available positions, formalized as ė = μ₀(w₀/w − 1) | Core destabilizing mechanism: surplus wealth from declining wages draws aspirants into elite-track careers; intensifies intraelite competition |
| Relative elite income (ε) | Average elite income scaled by GDP per capita: ε = (1 − wλ)/e | Inverse index of intraelite competition; declines as elite numbers grow faster than the surplus they share |
| Social mobility parameter (μ) | Net rate of upward/downward mobility between general population and elites: μ = μ₀(w₀/w − 1) | Endogenous to relative wages: low wages → high upward mobility → elite expansion; high wages → downward mobility → elite contraction |
| Non-market forces (C) | Extra-economic factors affecting wages: coercive, political, and ideological power. Proxied by real minimum wage | Accounts for institutional environment (unionization, regulation, political attitudes) that modifies pure supply/demand wage outcomes |
| Labor supply/demand ratio (D/S) | Market mechanism: when S exceeds D, wages fall | Captures demographic pressure (baby boomers, immigration, female labor participation) and technological change (productivity gains reducing labor demand) |
| Wage stickiness (τ) | Lag between structural conditions changing and wages adjusting, empirically ~5 years | Reflects institutional delays (contract renegotiation, information lag); enables forecasting since current conditions predict wages five years hence |
| Structural pressures vs. triggers | Structural pressures = slow, forecastable demographic-economic variables; triggers = fast, contingent events | Analytical distinction enabling social science to focus on measurable structural landscape rather than inherently stochastic event timing |
| Secular cycles | Long-period (~100–200 year) oscillations in political instability driven by structural-demographic feedback loops | Theoretical framework within which both case studies operate; growth → stagnation → crisis → depression → recovery |

## Figures, Tables & Maps

### Figure 2 (p. 16): Population Dynamics — Antebellum Model vs. Data
- **What it shows**: Two-panel comparison of (a) model-generated and (b) observed trajectories for rural population, urban population, and relative wage in four Northeastern states (MA, CT, NY, NJ), 1790–1880
- **Key data points**: Rural population saturates at ~3.5 million (carrying capacity K); urban population grows exponentially from ~0.1 million; relative wage rises until ~1820 then declines as labor supply overtakes demand
- **Connection to argument**: Validates the demographic model (Eqs. 6–7) and demonstrates the wage trend reversal mechanism — the shift from labor scarcity to oversupply around 1820 as rural population approaches K

### Figure 3 (p. 18): Elite Dynamics — Antebellum Model
- **What it shows**: Trajectories of relative wage (w), relative elite numbers (e), and average elite incomes (ε) from the Antebellum model, 1790–1880
- **Key data points**: Elite proportion roughly constant at 1% until 1840, then triples to 3% by 1870; elite incomes rise until 1840 (favorable economic conjuncture), then decline as elite numbers dilute the surplus
- **Connection to argument**: Demonstrates the elite overproduction mechanism — a 20-year lag between wage decline (starting ~1820) and elite explosion (starting ~1840), with elite income peaking before being diluted by growing numbers

### Figure 4 (p. 20): Youth Cohort Proportion — Empirical Data
- **What it shows**: Proportion of 20–29 year-old cohort among total American white male population, 1800–1900
- **Key data points**: Fluctuates between 15–19%, with peaks around 1810 and 1860
- **Connection to argument**: The A₂₀₋₂₉ component of MMP captures the "youth bulge" effect on instability; used as empirical input rather than modeled endogenously

### Figure 5 (p. 21): MMP and EMP — Antebellum Model
- **What it shows**: Predicted dynamics of Mass Mobilization Potential and Elite Mobilization Potential (both scaled to minimum = 1), 1790–1880
- **Key data points**: MMP essentially flat until 1820, then grows steadily; EMP declines until 1830s, then explodes upward (reaching ~18× minimum by 1870)
- **Connection to argument**: EMP dynamics are more violent in amplitude and shifted in phase relative to MMP, demonstrating that elite overproduction is the more explosive pressure source

### Figure 6 (p. 22): Predicted PSI vs. Observed Instability — Antebellum
- **What it shows**: Model-predicted Political Stress Indicator overlaid with observed political instability (casualties per million) and a qualitative sectional conflict index, 1790–1880
- **Key data points**: PSI stays low until ~1840, then explodes during the 1850s; observed instability escalation (Bleeding Kansas 1855, Harper's Ferry Raid 1859, Civil War 1861–65) follows the PSI rise with a short lag
- **Connection to argument**: Core validation — PSI serves as a leading indicator of both small-scale political violence and the eventual large-scale civil war

### Figure 7 (p. 24): Labor Demand vs. Supply — Contemporary US
- **What it shows**: Trends in labor demand (G/P, GDP divided by productivity) and labor supply (total labor force), both scaled to 1927 = 100, for 1927–2012
- **Key data points**: Demand outpaces supply until ~1965; supply crosses above demand during the late 1960s (baby boomers, immigration); demand plateaus after 2000 (sluggish growth + productivity gains)
- **Connection to argument**: Identifies the structural mechanism driving the post-1970 wage stagnation — a supply/demand reversal driven by internal demographic growth and immigration policy changes

### Figure 8 (p. 26): Union Dynamics — Coverage and Illegal Firings
- **What it shows**: Union coverage of workers (1930–2010) and proportion of union campaigns with illegal firings (1951–2007)
- **Key data points**: Union coverage peaks at 25%+ (1945–1960), then declines to 12% by 2010; illegal firings during union drives increase 10-fold between 1950s and 1980s
- **Connection to argument**: Provides empirical evidence for the "cultural shift" (non-market forces C) — the transformation from New Deal labor-management compact to anti-union corporate strategy, particularly during the 1970s

### Figure 9 (p. 28): Real Minimum Wage — Used as Proxy for Non-Market Forces
- **What it shows**: Real (inflation-adjusted) minimum wage dynamics, 1938–2012
- **Key data points**: Rises steadily from New Deal through Great Society (doubles 1950–1970); declines after 1970 due to inflation erosion without compensating increases; equilibrates at lower level during 1990s–2000s
- **Connection to argument**: Serves as the proxy variable for non-market forces (C) in the regression model; its trend closely tracks the broader institutional shift in labor-management relations

### Figure 10 (p. 29): Progressive Regression Model Fits — Contemporary Wages
- **What it shows**: Four panels showing regression model fits with progressively more factors: (a) GDP per capita only, (b) GDP + D/S ratio, (c) all three factors, (d) full model with 5-year lag
- **Key data points**: R² improves from 0.73 (GDP only) to 0.93 (GDP + D/S) to 0.98 (all three factors); the 5-year lag model captures fine-scale fluctuations during the stagnation phase (1980s dip, 1990s rise, post-2000 decline)
- **Connection to argument**: Demonstrates that all three structural determinants are needed; the 5-year lag is not merely a fitting device but reflects real institutional adjustment delays

### Figure 11 (p. 32): Elite Submodel Dynamics — Contemporary US
- **What it shows**: Model-predicted trajectories of relative wage (w), relative elite numbers (e), and relative elite income (ε), 1927–2012
- **Key data points**: Relative wage peaks ~1960, declines steeply after 1978; relative elite numbers decline gently 1930–1980, then increase roughly threefold after 1980; relative elite income recovers post-1955, peaks ~1990, then declines as expanding elite numbers dilute the surplus
- **Connection to argument**: Demonstrates the same qualitative dynamics as the Antebellum model (compare Figure 3) — wage decline precedes elite explosion by ~20 years, with elite income eventually diluted by growing numbers

### Figure 12 (p. 34): Cost of Winning House Election, 1986–2012
- **What it shows**: Inflation-adjusted cost of winning a House seat (thousands) and total spending by major party candidates (millions), 1986–2012
- **Key data points**: Cost of winning approximately doubles from ~$800K to ~$1.6M; total spending approaches $1 billion by 2010
- **Connection to argument**: Direct empirical evidence of intensifying intraelite competition for political positions — a predicted consequence of elite overproduction

### Figure 13 (p. 36): National Debt/GDP and Government Distrust
- **What it shows**: National debt as percentage of GDP and proportion of people distrusting government institutions, 1930–2012
- **Key data points**: Debt/GDP: WWII spike followed by decline to ~30% by 1980, then unprecedented peacetime growth to ~100% by 2012; Distrust: 20-30% in late 1950s/60s, damaged by Watergate in 1970s, cyclic but each peak higher than the last (reaching ~80% by 2011)
- **Connection to argument**: These are the two SFD components; their joint elevation after 2000 completes the PSI trifecta (MMP + EMP + SFD all rising)

### Figure 14 (p. 37): Estimated PSI, 1958–2012
- **What it shows**: The calculated Political Stress Indicator for the contemporary United States
- **Key data points**: PSI stays near 0 through the 1960s–1980s; begins rising after 1980; accelerates sharply after 2000; reaches ~40 by 2012 (from near 0 in 1960)
- **Connection to argument**: The contemporary PSI trajectory parallels the antebellum PSI trajectory; structural conditions as of 2012 favor continued increase through at least 2020

### Table 1 (p. 30): Regression Analysis Results
- **Key findings**: All three predictors significant: GDP per capita (coeff. 0.60 ± 0.07), labor D/S (coeff. 1.65 ± 0.31), real minimum wage (coeff. 0.45 ± 0.08). ARIMA(1,0,1) error structure (AR1 = 0.74, MA1 = 0.65). Model selection via AIC confirms all three needed.

### Table 2 (p. 33): Millionaire Proportions, 1983–2007
- **Key findings**: Households exceeding $1M net worth grew from 2.9% (1983) to 6.3% (2007); exceeding $5M from 0.29% to 1.26%; exceeding $10M from 0.08% to 0.40%. Expansion rates bracket the model's theoretical prediction for elite numbers.

### Table 3 (p. 34): Congressional Candidates, 2000–2012
- **Key findings**: House candidates grew from 1,233 (2000) to 1,897 (2010), +54%; Senate from 191 to 308, +61%. Self-funded millionaire candidates nearly doubled 2004–2010 (30 to 58).

## Figure ↔ Concept Contrast

- Figure 2 → **Relative wage (w)**: Demonstrates the mechanism by which rural-urban migration creates labor oversupply, driving the wage trend reversal that initiates the instability cycle
- Figure 3 → **Elite overproduction**, **Relative elite income (ε)**: Visualizes the core destabilizing mechanism — the lag between wage decline and elite explosion, and the subsequent dilution of elite incomes
- Figure 5, 6 → **PSI**, **MMP**, **EMP**: Shows the multiplicative amplification (EMP × MMP) and validates PSI as a leading indicator of actual instability events
- Figure 7 → **Labor supply/demand ratio (D/S)**: Identifies the structural crossover (~1965) that drove post-1970 wage stagnation — the contemporary equivalent of the antebellum rural-to-urban migration pressure
- Figure 8, 9 → **Non-market forces (C)**: Empirical evidence for the "cultural shift" — the institutional transformation from labor-protective to capital-protective norms, proxied by union decline and minimum wage erosion
- Figure 10 → **Wage stickiness (τ)**: Progressive model-fitting demonstrates that the 5-year lag captures real institutional delays; all three structural factors are needed (R² = 0.73 → 0.93 → 0.98)
- Figure 11 → **Elite overproduction**, **Social mobility (μ)**: Contemporary elite dynamics recapitulate the antebellum pattern with a ~20-year lag between wage decline and elite explosion
- Figure 12, Table 3 → **EMP**: Direct measures of intensifying intraelite competition (cost of office, candidate proliferation, millionaire candidates)
- Figure 13 → **SFD**: The two SFD components (debt/GDP, distrust) both elevated after 2000, completing the structural trifecta
- Figure 14 → **PSI**: The bottom line — contemporary PSI trajectory mirrors the antebellum pattern, with sharply accelerating structural pressures after 2000

## Equations & Formal Models

### General Real Wage Model (Eq. 1)

$$W_{t-\tau} = a \left(\frac{G_t}{N_t}\right)^\alpha \left(\frac{D_t}{S_t}\right)^\beta C_t^\gamma$$

Three multiplicative factors: GDP per capita (productivity effect), labor demand/supply ratio (market mechanism), and non-market cultural/coercive forces. The lag τ (empirically ~5 years) models wage stickiness. In log-linear form (Eq. 2), this becomes a standard regression framework with ARIMA error structure.

### Elite Population Dynamics (Eqs. 3–4)

$$\dot{E} = rE + \mu N$$

where the social mobility function is:

$$\mu = \mu_0 \left(\frac{w_0}{w} - 1\right)$$

When w < w₀, μ > 0 (net upward mobility into elite), driving elite expansion. When w > w₀, μ < 0 (net downward mobility). The rate of change of *relative* elite numbers simplifies to ė = μ (assuming similar natural growth rates for elites and commoners).

### Relative Elite Income (Eq. 5)

$$\varepsilon = \frac{1 - w\lambda}{e}$$

where λ ≈ 0.5 (labor force participation rate). As e grows and w declines, the numerator (elite share of GDP) grows but is eventually overtaken by the denominator (elite numbers), producing the income dilution effect observed in both case studies.

### Antebellum Population Model (Eqs. 6–7)

$$\dot{N}_{rur} = rN_{rur} - rN_{rur}\left(\frac{N_{rur}}{K}\right)^\theta$$

$$\dot{N}_{urb} = r_{urb}N_{urb} + rN_{rur}\left(\frac{N_{rur}}{K}\right)^\theta$$

Rural population approaches carrying capacity K, generating nonlinear migration to cities (θ = 5 for sigmoidal response). Urban growth = endogenous growth (r_urb = 1.5% p.a.) plus immigration from countryside.

### Antebellum Wage Model (Eq. 10)

$$w = a\left(\frac{D}{N_{urb}}\right)^\beta$$

Simplified version of the general model for a pure market economy (no cultural factor), with relative wage determined solely by labor demand-to-supply ratio.

### PSI Formula (Eq. 14)

$$\Psi = \text{MMP} \times \text{EMP} \times \text{SFD} = w^{-1} \cdot \frac{N_{urb}}{N} \cdot A_{20-29} \cdot \varepsilon^{-1} \cdot \frac{E}{sN} \cdot \frac{Y}{G} \cdot D$$

The composite index combining all structural-demographic pressures into a single scalar.

## Theoretical & Methodological Implications

Turchin's method is formal-mathematical modeling with empirical calibration. The paper employs two distinct approaches for its two case studies. The Antebellum model uses an ODE system (Eqs. 6–12) calibrated to demographic parameters, generating endogenous trajectories for all variables. The Contemporary model uses regression analysis (ARIMA with three structural predictors) for wages, feeding the fitted wage trajectory into the elite dynamics ODE (Eq. 13) to generate elite predictions. This difference reflects data availability: the contemporary period has rich time-series data enabling regression, while the antebellum period requires more model-driven inference.

The paper explicitly advocates for a spectrum of manageable models rather than one comprehensive model. Large complex models suffer from structural instability (small parameter changes producing large dynamical changes) and require too many arbitrary decisions. Turchin positions his approach as "dynamically incomplete" — modeling one variable in detail while treating others as exogenous — yet argues this captures the essential causal mechanisms.

The multiplicative PSI form is phenomenological rather than derived from first principles. Turchin inherits this from Goldstone (1991) and modifies the functional forms. The multiplicative structure implies an interaction effect: all three pressures must be simultaneously elevated, since any one near zero suppresses the whole indicator. This is asserted as empirically reasonable but not formally justified.

The paper makes a sharp methodological distinction between structural pressures and triggers. Structural pressures are the domain of quantitative structural-demographic modeling; triggers belong to narrative history. This separation enables the model to make meaningful forecasts (the structural landscape) without claiming to predict specific events. The analogy is forest fire science: measuring fuel load and moisture content forecasts fire severity without predicting which lightning bolt will ignite it.

Validation uses both retrodiction (the Antebellum model produces PSI trajectories consistent with known instability patterns) and genuine prediction (the Contemporary model's PSI forecast was first published in 2010, before the events of the 2010s). The Antebellum validation is stronger methodologically (the model was developed with antebellum data, not fit to the outcome), while the Contemporary forecast is more impressive practically (a rare successful social-science prediction over a 10-year horizon).

## Empirical Data

**Antebellum Model Parameters**: r_rur = 2.5% y⁻¹, r_urb = 1.5% y⁻¹, K = 3.5 million, θ = 5, β = 0.5, ρ = 3% y⁻¹, γ = 1% y⁻¹, μ₀ = 0.002 y⁻¹, w₀ = 1. Data sources: HSUS (Carter et al. 2004), calculations by author.

**Contemporary Wage Regression**: R² = 0.98 with all three factors and 5-year lag. Coefficients: GDP/capita α = 0.60 (±0.07), D/S β = 1.65 (±0.31), minimum wage γ = 0.45 (±0.08). ARIMA(1,0,1) error structure. Data: MeasuringWorth (wages/GDP), BLS (productivity, labor force), Census Bureau (demographics).

**Elite Overproduction Evidence**: Households exceeding $1M grew from 2.9% to 6.3% (1983–2007). Congressional candidates increased 54–61% (2000–2010). Cost of winning House seat roughly doubled in real terms (1986–2012). Self-funded millionaire candidates nearly doubled (2004–2010).

**State Fiscal Indicators**: Debt/GDP grew from ~30% (1980) to ~100% (2012) — first peacetime growth of this magnitude. Government distrust cyclically escalating: each successive peak higher (reaching ~80% by 2011, from ~20% in late 1950s).

**Key Time Series**: Labor demand (G/P) overtaken by supply (labor force) circa 1965; relative wage peaked ~1960, declining through 2012; union coverage peaked ~25% (1945–60), declined to ~12% by 2010; real minimum wage doubled 1950–1970, then eroded to lower equilibrium.
