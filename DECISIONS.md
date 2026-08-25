# Design Decisions

Fill this in BEFORE you write any code or open Claude Code.

Every question below is one an interviewer can ask. If you let Claude Code decide these, you will not be able to defend them. Write your answer and your reason. The reason matters more than the answer.

---

## 1. Stage definitions

Define the exact boundary for each stage. Vague definitions produce vague findings.

| Stage               | Definition                                                                                     | Why this boundary                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Diagnosed           | Patient's first confirmed diagnosis event within the study period                              | Creates a consistent starting point for every patient's treatment journey                                                      |
| Prescribed          | First treatment prescription within 60 days of diagnosis                                       | Allows reasonable time for treatment initiation while excluding extremely delayed prescriptions                                |
| First Fill          | First medication dispense within 30 days of prescription                                       | Distinguishes patients who actually start therapy from those who abandon after prescription                                    |
| Refill              | Second medication dispense within 45 days of first fill                                        | Uses the same allowable gap as the persistence definition and keeps refill and continued-treatment rules internally consistent |
| Continued Treatment | At least 3 total consecutive fills, including the first fill, with no gap greater than 45 days | Represents sustained treatment rather than one-time or short-term medication use                                               |

### Starting definitions used in this project

* First Fill = script dispensed within 30 days of prescription date.
* Refill = second dispense within 45 days of first fill.
* Continued Treatment = at least 3 total consecutive fills, including the first fill, with no gap greater than 45 days.

The `prescriber_specialty` variable is assigned at cohort entry and represents the diagnosing or treating physician associated with the patient. This allows physician specialty to be used when modelling the Diagnosis → Prescription transition, including for patients who never receive a prescription.

### Timing and post-funnel mechanics

Inter-event gaps are drawn uniformly from 1 to `max_days_from_previous`. Same-day transitions are excluded so that event sequence is unambiguous from dates alone.

This is a simplification. In real claims data, diagnosis and prescription can occur during the same visit. The effect on measured conversion rates in this simulation is negligible and is limited mainly to a small number of patients close to the observation cutoff whose transition may be pushed one day beyond it.

Attrition does not stop at Continued Treatment. After a patient reaches the third qualifying fill, each additional fill occurs with probability 0.75, capped at the 12-fill treatment course.

Therefore, a patient can successfully reach Continued Treatment but still discontinue before completing the full course.

Post-continuation attrition is reported separately from the five-stage funnel because the five-stage funnel measures leakage only up to reaching Continued Treatment.

---

## 2. Censoring

A patient diagnosed last week has not necessarily dropped out. They may simply not have had enough time to progress.

Questions addressed:

* What is the observation window?
* How are patients without sufficient follow-up handled?
* Is drop-off calculated on the full cohort or only on patients who had a complete opportunity to convert?

### Observation window

The observation window is January 1, 2024 through December 31, 2025, with December 31, 2025 as the observation cutoff.

At every transition, a patient is included in the primary conversion/drop-off denominator only if the complete opportunity window for that transition elapsed before the observation cutoff.

For example, First Fill is allowed within 30 days of prescription.

If a patient is prescribed on December 20, 2025, their 30-day First Fill window extends past December 31, 2025. That patient therefore does not have a completely observed First Fill opportunity and is treated as censored for that transition.

Drop-off rates are calculated only on the eligible, window-closed cohort.

For a window-closed transition:

**Eligible patients = Window-closed converters + Window-closed true drop-offs**

Patients whose opportunity window has not closed are censored and excluded from the primary denominator.

For Continued Treatment, which requires 3 qualifying fills with a maximum 45-day gap between fills, I use a 90-day assessment window from the first fill:

**First Fill → up to 45 days → Second Fill → up to 45 days → Third Fill**

### Denominator rule

I initially specified a `converter_inclusive` rule under which patients who converted were considered eligible even when their full opportunity window had not closed.

Testing this rule showed that it inflated measured conversion rates by approximately 0.5 to 1.4 percentage points across the four transitions.

The direction of the bias was always upward.

The cause is outcome-dependent inclusion. Among patients whose opportunity window remained open at the cutoff, converters could enter the denominator while non-converters were censored and excluded.

The primary analysis therefore uses:

**`window_closed`**

A patient enters the denominator for a transition only if the complete opportunity window elapsed before the observation cutoff, regardless of whether the patient converted.

The original `converter_inclusive` implementation is retained for comparison so that the impact of the denominator choice remains reproducible.

The upward gap between the two rules can be expressed as:

**gap = (open-window converters / converter-inclusive denominator) × (1 − window-closed conversion rate)**

This explains why the bias is not determined by window length alone.

A longer opportunity window can increase the number of open-window patients, but the size of the bias also depends on:

* how many open-window patients have already converted,
* the underlying conversion rate,
* and where the transition's anchor date lies relative to the observation cutoff.

The direction of the bias is guaranteed to be upward because every additional patient admitted by the converter-inclusive rule is a converter.

The `window_closed` rule recovered the generator-implied expected transition rates much more closely. For example:

* Diagnosis → Prescription: approximately 0.8223 observed versus 0.8217 generator-implied.
* Refill → Continued Treatment: approximately 0.7185 observed versus 0.7219 generator-implied.

### Post-continuation population

Post-continuation analysis uses a clinically defined population: patients who reached at least 3 qualifying fills.

This population is distinct from the window-closed Refill → Continued Treatment funnel denominator because post-continuation behaviour is analysed after the five-stage funnel endpoint.

### Why this matters

If recent patients are automatically classified as drop-offs, the funnel will appear worse than it actually is.

The problem can become especially important for later transitions because they generally require more follow-up time and therefore expose more patients to right censoring.

Correct censoring distinguishes:

* true conversion,
* true observed drop-off,
* and insufficient observation time.

With a much smaller real-world dataset, rather than discarding open-window observations for a binary funnel calculation, I would consider survival/time-to-event analysis with explicit right-censoring.

---

## 3. Revenue attribution

Questions addressed:

* What is revenue per fill?
* Is a patient lost early valued the same as a patient lost late?
* How is remaining lifetime value calculated?

### Revenue assumptions

Revenue per medication fill is assumed to be ₹2,400.

A complete treatment course is assumed to contain 12 fills.

Patients lost at different points in treatment are not assigned the same remaining value.

Remaining lifetime value is calculated as:

**Remaining lifetime value = (Expected total fills − Fills already completed) × Revenue per fill**

Examples:

* Before First Fill: 12 × ₹2,400 = ₹28,800 remaining potential revenue.
* After First Fill: 11 × ₹2,400 = ₹26,400.
* After Second Fill: 10 × ₹2,400 = ₹24,000.

A patient lost after Diagnosis and a patient lost after Prescription but before First Fill have both completed zero fills.

Under this simplified model, both therefore have ₹28,800 of remaining treatment revenue.

This is intentional: the revenue model values remaining medication fills, not the diagnosis or prescription event itself.

These figures are simulation assumptions used to compare leakage value. They are not estimates of actual pharmaceutical product revenue.

### Post-continuation censoring adjustment

Patients who reach Continued Treatment may still discontinue before completing all 12 fills.

A naive calculation using the observed fill count of every Continued Treatment patient understates future fills among patients whose later course is censored by the observation cutoff.

I therefore estimate post-continuation forfeited revenue using the average per-patient forfeiture among fully observable Continued Treatment patients and project that value to the complete Continued Treatment population.

The correction assumes that censored Continued Treatment patients would eventually behave like fully observable Continued Treatment patients.

In this synthetic dataset, that assumption is reasonable because cohort entry date is generated independently of payer type, age band, prescriber specialty, and geography.

In real patient data, this assumption would not automatically be safe. I would first test whether late entrants differ systematically in patient mix, payer mix, acquisition channel, treatment setting, or other relevant characteristics.

The corrected analysis produced approximately:

* Naive post-continuation loss: ₹74.4M.
* Estimated censoring artifact: ₹4.3M.
* Censoring-corrected post-continuation loss: ₹70.1M.
* Five-stage funnel loss: approximately ₹360.5M.
* Combined funnel + post-continuation loss: approximately ₹430.6M.

Therefore, every percentage or share reported in later analysis must explicitly state whether the denominator is:

* five-stage funnel loss only, or
* combined loss including post-continuation attrition.

Post-continuation attrition is the third-largest revenue-loss source in the current simulation, despite sitting outside the five-stage funnel.

---

## 4. Patient features for the regression

The regression is intended to identify interpretable adjusted associations with conversion.

| Feature                 | Expected direction / role                                                                                               | Reasoning                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Payer type              | Cash-pay patients expected to have lower First Fill conversion                                                          | Greater direct financial burden is represented through the deliberately injected cash-pay effect |
| Age band                | Patients aged 75+ expected to have lower probability of reaching Continued Treatment                                    | A deliberate age 75+ penalty is injected at Refill → Continued Treatment                         |
| Prescriber specialty    | General-physician-associated patients expected to have lower Diagnosis → Prescription conversion                        | A deliberate general-physician penalty is injected at Diagnosis → Prescription                   |
| Comorbidity count       | Domain prior could suggest lower persistence, but no effect is deliberately injected                                    | Serves as a null control to test whether the pipeline manufactures effects                       |
| Geography / urban-rural | Rural patients expected to have lower Refill conversion                                                                 | A deliberate rural penalty is injected at First Fill → Refill                                    |
| Out-of-pocket cost      | Marginal association with conversion is expected because it is correlated with payer type; no direct effect is injected | Provides an intentional proxy-variable and multicollinearity diagnostic                          |

### Payer and out-of-pocket correlation

Out-of-pocket cost is intentionally generated as correlated with payer type.

Cash-pay patients have substantially higher out-of-pocket costs than commercially or government-insured patients.

Therefore, payer type and out-of-pocket cost contain overlapping information.

This creates a deliberate multicollinearity/proxy-variable example for the driver model.

Variance Inflation Factors (VIF) are therefore reported for predictors.

The intercept is excluded from VIF reporting because its VIF is not interpretable.

### Continuous predictor scale

Continuous predictor odds ratios are reported on interpretable scales rather than arbitrary one-unit increments.

Out-of-pocket cost is reported per ₹1,000 increase.

In the overall cohort, ₹1,000 is approximately 1.08 standard deviations of out-of-pocket cost. The exact SD-equivalent is also reported for each regression cohort.

This makes the reported cost effect easier to compare with a one-category shift in a categorical predictor.

However, ranking predictors by distance of an odds ratio from 1 is scale-dependent.

For example, the same cost coefficient can look small per ₹100 and enormous per ₹10,000 without the fitted model changing.

Therefore, I report both:

* effect-size ranking based on the interpretable reported odds-ratio scale, and
* a scale-invariant ranking based on absolute z-score.

The original per-unit coefficient and odds ratio remain available in the returned data for reproducibility.

---

## 5. Interventions

| Intervention          | Target stage        | Cost per patient | Expected lift | Source                                                     |
| --------------------- | ------------------- | ---------------: | ------------: | ---------------------------------------------------------- |
| Copay assistance      | First Fill          |             ₹900 |           22% | Simulation assumption; tested through sensitivity analysis |
| Refill reminders      | Refill              |              ₹40 |            6% | Simulation assumption; tested through sensitivity analysis |
| Nurse education call  | Continued Treatment |             ₹350 |           14% | Simulation assumption; tested through sensitivity analysis |
| Digital adherence app | Continued Treatment |             ₹120 |            8% | Simulation assumption; tested through sensitivity analysis |
| Prescriber detailing  | Prescribed          |           ₹1,500 |           11% | Simulation assumption; tested through sensitivity analysis |

### Expected lift definition

Expected lift represents a **relative improvement** in the current conversion rate, not a percentage-point increase.

For example:

Current conversion = 50%

Expected lift = 22%

**New conversion = 50% × (1 + 0.22) = 61%**

Conversion is capped at 100%.

### Target-stage convention

The target stage refers to the destination stage of the transition being improved.

* `target_stage: prescribed` = Diagnosis → Prescription.
* `target_stage: first_fill` = Prescription → First Fill.
* `target_stage: refill` = First Fill → Refill.
* `target_stage: continued` = Refill → Continued Treatment.

### Addressable population

The addressable population for an intervention is all patients in the configured window-closed denominator at the target transition who match the intervention's `eligible_segment`.

Targeting is defined before observing the patient's eventual conversion outcome.

I do **not** restrict intervention targeting only to patients who ultimately fail to convert, because doing so would make intervention eligibility depend on a future outcome that would not be known prospectively.

For a targeted group:

**New conversion rate = min(Current conversion rate × (1 + Expected lift), 1.0)**

**Patients recovered = Patients targeted × (New conversion rate − Current conversion rate)**

### Intervention cost

Intervention cost is charged to every patient targeted, not only to incrementally recovered patients.

For example, if 5,000 patients receive a ₹40 refill reminder:

**Total cost = 5,000 × ₹40 = ₹200,000**

This cost is incurred even though only a subset of those patients are expected to convert because of the intervention.

The output therefore reports `patients_targeted` and `patients_recovered` separately.

### Sensitivity analysis

The intervention lift estimates are assumptions, not findings from the generated data.

The simulator therefore recomputes intervention economics at:

* 0.5× assumed lift,
* 1.0× assumed lift,
* 2.0× assumed lift.

Ranking stability is reported explicitly.

If rankings change materially when lift assumptions change, the recommendation is treated as assumption-sensitive rather than robust.

Nurse education calls and the digital adherence app target the same transition but serve disjoint age groups, so they are complementary rather than strictly competing interventions.

---

## 6. Ground truth

Because this dataset is synthetic, the true effects deliberately injected into it are known.

### Injected effects

1. Cash-pay patients receive a **35% multiplicative reduction** in Prescription → First Fill conversion.
2. Rural patients receive a **20% multiplicative reduction** in First Fill → Refill conversion.
3. Patients aged 75+ receive a **15% multiplicative reduction** in Refill → Continued Treatment conversion.
4. Patients associated with a general physician receive a **10% multiplicative reduction** in Diagnosis → Prescription conversion.

A multiplicative reduction means:

**Adjusted probability = Base probability × (1 − penalty)**

For example:

Base First Fill probability = 72%

Cash-pay penalty = 35%

**72% × (1 − 0.35) = 46.8%**

The corresponding probability multipliers are:

* General physician: 0.90.
* Cash pay: 0.65.
* Rural: 0.80.
* Age 75+: 0.85.

### How ground-truth validation is performed

The primary ground-truth validation is based on observed eligible-cohort transition-rate ratios.

For each injected effect:

**Observed ratio = conversion rate of penalised group / conversion rate of comparison group**

The observed probability ratio is compared with the injected probability multiplier.

Logistic-regression odds ratios are **not** compared numerically with these injected probability multipliers.

Probability ratios and odds ratios are different quantities.

The logistic regression is instead used to evaluate:

* effect direction,
* statistical significance,
* adjusted strength,
* confidence intervals,
* and behaviour after controlling for other patient features.

This project therefore represents **model validation against known synthetic ground truth, not discovery of real-world patient behaviour**.

A correct interview description is:

> "I deliberately injected a payer effect and then tested whether the analytical pipeline recovered the expected pattern."

An incorrect description would be:

> "I discovered that cash-pay patients abandon more."

### Null control

Comorbidity count has a generation rule but no deliberately injected effect.

It functions as a null control.

The expectation is that it should not show a systematic meaningful relationship with conversion across transitions.

If one nominal p-value falls below 0.05 by chance, I would investigate it rather than regenerate the data or change the seed.

The purpose of the null control is to provide evidence that the pipeline is capable of distinguishing injected signal from unrelated variation.

Out-of-pocket cost also has no direct injected conversion effect, but unlike comorbidity it is deliberately correlated with payer type.

It therefore functions as a proxy-variable/multicollinearity diagnostic rather than a pure null control.

### Multiple comparisons

Four injected effects are validated simultaneously.

Using four independent nominal 95% intervals gives approximately:

**1 − 0.95⁴ ≈ 18.5%**

probability that at least one interval misses its true value by chance under independence.

On seed 42, the rural refill effect illustrates this.

Observed ratio ≈ 0.826

Injected multiplier = 0.800

The nominal 95% interval excludes 0.800.

I did **not** change the random seed.

Changing the seed because a validation result failed would select a favourable random sample after seeing the result and would invalidate the purpose of the validation exercise.

Instead, I apply a Bonferroni correction.

With:

* overall alpha = 0.05,
* number of tests `k = 4`,

the per-test alpha is:

**0.05 / 4 = 0.0125**

and each two-sided confidence interval is therefore:

**98.75%**

Bonferroni controls the family-wise error rate at no more than 5% without requiring the tests to be independent.

Under independence, the corresponding actual family-wise error is approximately 4.9%.

The corrected rural interval includes the injected 0.800 multiplier.

The number of validation tests is fixed by the number of deliberately injected effects. It was not selected after seeing the results.

The estimator was also validated directly against the generator's expected per-patient probabilities, which recover:

* 0.9000,
* 0.6500,
* 0.8000,
* 0.8500.

This confirms that the estimator itself is correctly centred on the injected effects.

### Multiple comparisons in the driver model

The logistic-regression analysis performs 44 coefficient tests across four transitions. At a nominal alpha of 0.05, approximately 44 × 0.05 = 2.2 false-positive coefficients would be expected by chance if the null hypotheses were true.

One example is `payer_type_government` at First Fill → Refill. It is nominally significant (p ≈ 0.032, odds ratio ≈ 1.13), despite having no deliberately injected effect. It does not survive the Bonferroni threshold of 0.05 / 44 ≈ 0.00114 and is therefore treated as an expected false positive rather than a substantive finding.

The deliberately injected effects survive the corrected threshold by wide margins; the weakest remains approximately p = 1e-11. This provides a clear separation between the intended synthetic signal and nominal significance produced by multiple testing.

The coefficient tests are not fully independent because the transition cohorts are nested; for example, the First Fill → Refill cohort is a subset of patients who converted at Prescription → First Fill. Therefore, repeated nominal false positives for the same predictor across adjacent transitions may partly reflect correlated sampling variation rather than independent false discoveries. The expected false-positive count remains valid by linearity of expectation, but its variance is affected by this dependence. Bonferroni correction remains appropriate because it controls family-wise error even when the tests are dependent.

---

## 7. The finding I expected

This hypothesis was written before inspecting the analytical outputs.

### Original hypothesis

I expected the largest absolute patient leakage to occur between Prescription and First Fill because this transition has meaningful baseline abandonment and contains the strongest deliberately injected effect.

I expected cash-pay patients to show the lowest First Fill conversion because of the injected 35% penalty.

I also expected:

* rural patients to show lower First Fill → Refill conversion,
* patients aged 75+ to show lower Refill → Continued Treatment conversion,
* patients associated with general physicians to show somewhat lower Diagnosis → Prescription conversion.

I expected copay assistance to generate substantial total recovered revenue because it targets high-value First Fill leakage among cash-pay patients.

However, I expected refill reminders could achieve the highest return per rupee because their cost per targeted patient is substantially lower.

Therefore, I expected the intervention with the highest total financial impact might not be the same intervention as the one with the highest return per rupee spent.

I also predicted that the intentional correlation between payer type and out-of-pocket cost would reduce the apparent independent payer effect in the multivariable regression.

That last prediction turned out to be incorrect.

### Amendment after running the driver model

The fitted driver model did not support my prediction that payer–cost correlation would materially attenuate the payer point estimate.

The cash-pay odds ratio changed only about 0.7%:

* payer-only model: approximately 0.3792,
* payer + out-of-pocket model: approximately 0.3765.

However, estimate precision deteriorated substantially.

The coefficient standard error increased approximately 2.10×, while VIF increased from approximately 1.33 to 6.31.

My original prediction confused two different statistical phenomena.

**Multicollinearity primarily inflates the variance of coefficient estimates. It does not, by itself, systematically bias their point estimates.**

Out-of-pocket cost has no deliberately injected direct effect on conversion.

It is strongly correlated with payer type.

When cost is fitted without payer, it acts as a proxy for payer and captures much of the same outcome signal.

Once payer is included, cost adds almost no independent information and becomes non-significant:

**p ≈ 0.928**

while the payer estimate remains largely unchanged.

The observed standard-error inflation of approximately 2.107× is broadly consistent with the increase suggested by the VIF ratio:

**sqrt(6.31 / 1.33) ≈ 2.178**

The relationship is not expected to be exact for logistic regression, so this comparison is treated as a diagnostic consistency check rather than an identity.

When out-of-pocket cost is fitted alone, the approximately ₹1,700 mean difference between cash-pay and commercial patients corresponds to an implied odds ratio of roughly 0.4466.

The payer-only cash-pay odds ratio is approximately 0.3792.

This demonstrates proxy behaviour:

> A correlated variable can appear strongly predictive when the underlying driver is omitted, yet contribute almost no independent signal once that driver is included.

The documented incorrect prediction is retained rather than silently rewritten because the difference between the original expectation and the observed result is itself an important analytical finding.
