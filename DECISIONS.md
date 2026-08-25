Design decisions
Fill this in BEFORE you write any code or open Claude Code.
Every question below is one an interviewer can ask. If you let Claude Code decide these, you will not be able to defend them. Write your answer and your reason. The reason matters more than the answer.

1. Stage definitions
Define the exact boundary for each stage. Vague definitions produce vague findings.
Stage	Definition	Why this boundary
Diagnosed	Patient's first confirmed diagnosis event within the study period	Creates a consistent starting point for every patient's treatment journey
Prescribed	First treatment prescription within 60 days of diagnosis	Allows reasonable time for treatment initiation while excluding extremely delayed prescriptions
First Fill	First medication dispense within 30 days of prescription	Distinguishes patients who actually start therapy from those who abandon after prescription
Refill	Second medication dispense within 45 days of first fill	Uses the same allowable gap as the persistence definition and avoids conflicting refill and continued-treatment rules
Continued Treatment	At least 3 total consecutive fills, including the first fill, with no gap greater than 45 days	Represents sustained treatment rather than one-time or short-term medication use
Starting definitions used in this project:
* First Fill = script dispensed within 30 days of prescription date
* Refill = second dispense within 45 days of first fill
* Continued Treatment = at least 3 total consecutive fills, including the first fill, with no gap greater than 45 days
The prescriber_specialty variable is assigned at cohort entry and represents the diagnosing or treating physician associated with the patient. This allows physician specialty to be used when modeling the Diagnosis → Prescription transition, including for patients who never receive a prescription.

### Timing and post-funnel mechanics

Inter-event gaps are drawn uniformly from 1 to max_days_from_previous. Same-day transitions are excluded so that event sequence is unambiguous from dates alone.

This is a simplification: in real claims data, diagnosis and prescription frequently occur in the same visit. The effect on measured conversion rates is negligible, limited to a small number of patients near the observation cutoff whose transition may be pushed one day beyond it.

Attrition does not stop at Continued Treatment. Each fill beyond the third occurs with probability 0.75, capped at the 12-fill treatment course.

This means patients who reach Continued Treatment may still discontinue later. Post-continuation attrition is therefore reported separately from the five-stage funnel, which measures leakage only up to Continued Treatment.

## 2. Censoring

A patient diagnosed last week has not dropped out. They have not had time to progress.
* What is your observation window?
* How do you exclude patients who have not had time to reach the next stage?
* Do you report drop-off on the full cohort or only on the eligible cohort?
Your answer
The observation window is January 1, 2024 through December 31, 2025, with December 31, 2025 as the observation cutoff.
At every transition, a patient is included in the conversion/drop-off calculation only if enough time has passed for them to have realistically completed the next stage.
For example, First Fill is allowed within 30 days of prescription. If a patient is prescribed on December 20, 2025, the observation period ends before their 30-day opportunity window is complete. That patient is therefore classified as censored/ineligible, not as a drop-off.
Drop-off rates are calculated only on the eligible cohort at each stage:
Eligible patients = Converted patients + True drop-offs
Censored patients are excluded from the denominator.
For Continued Treatment, which requires 3 total fills with a maximum 45-day gap between fills, I use a 90-day assessment window after the first fill. This gives the patient enough observable time to complete two possible 45-day gaps:
First Fill → up to 45 days → Second Fill → up to 45 days → Third Fill

## Denominator rule. 

I initially specified that patients who converted were eligible by definition. Quantifying this showed it inflates conversion by 0.5 to 1.4 percentage points, always upward, and most at the transition with the longest observation window.

The cause is that the inclusion criterion depends on the outcome. Among patients whose window had not closed at the cutoff, only those who converted could enter the denominator, so the risk set is not a proper cohort. This is a selection bias in the denominator, the same family of error as immortal time bias.

I changed the rule to window_closed: a patient enters the denominator at a transition only if the full opportunity window elapsed before the cutoff, regardless of outcome. This recovers the injected ground truth almost exactly (0.8223 vs 0.8217 injected at diagnosis to prescription, 0.7185 vs 0.7219 at refill to continued).

Both rules remain implemented so the comparison is reproducible.
The upward bias from the converter-inclusive rule depends on both the number of converters whose opportunity window is still open and the underlying conversion rate.

The difference between the converter-inclusive and window-closed conversion rates can be expressed as:

gap = (open-window converters / eligible patients) × (1 − conversion rate)

A longer opportunity window can increase the number of patients with open windows, but window length alone does not determine the size of the bias. The timing of the transition anchor also matters.

The direction of the bias is guaranteed to be upward because the additional patients admitted under the converter-inclusive rule are all converters.

Post-continuation analysis uses a clinically defined population: patients who reached at least 3 qualifying fills. This population is distinct from the window-closed funnel denominator because post-continuation attrition is analyzed separately from the five-stage funnel.

## Why this matters

If recent patients are counted as drop-offs, every funnel will look worse than it really is. The distortion becomes larger at later stages because those stages require more observation time. Correct censoring prevents patients who have simply not had enough time from being misclassified as failures.

3. Revenue attribution
* What is revenue per patient per fill?
* Do you value a patient lost at diagnosis the same as one lost at month eight?
* How do you compute remaining lifetime value at each stage?
Your answer
Revenue per medication fill is assumed to be ₹2,400, and a complete treatment course is assumed to contain 12 fills.
Patients lost at different stages are not assigned the same lost value. A patient lost early in the treatment journey generally has more remaining treatment value than a patient lost after several fills.
Remaining lifetime value is calculated as:
Remaining lifetime value = (Expected total fills − Fills already completed) × Revenue per fill
For example:
* Before First Fill: 12 remaining fills × ₹2,400 = ₹28,800 potential remaining revenue
* After First Fill: 11 remaining fills × ₹2,400 = ₹26,400
* After Second Fill: 10 remaining fills × ₹2,400 = ₹24,000
A patient lost after diagnosis and a patient lost after prescription but before First Fill both have completed zero medication fills, so both are assigned the same remaining treatment revenue of ₹28,800 under this simplified model.
This revenue model therefore values remaining medication revenue, not the diagnosis or prescription event itself.
These are simulation assumptions used for comparing leakage value and are not estimates of actual pharmaceutical product revenue.

4. Patient features for the regression
List the features and, for each, your prior on why it would affect drop-off.
Feature	Expected direction	Reasoning
Payer type	Cash-pay patients expected to have lower First Fill conversion	Greater direct financial burden may increase treatment abandonment
Age band    Patients aged 75+ expected to have a lower probability of reaching Continued Treatment    Treatment complexity, medication burden, mobility, and access challenges may increase with age
Prescriber specialty	General-physician-associated patients expected to have somewhat lower Diagnosis → Prescription conversion	Physician specialty may be associated with disease-specific treatment familiarity and prescribing behavior
Comorbidity count	Higher count expected to reduce persistence	Multiple conditions and medications can increase treatment complexity
Geography / urban-rural	Rural patients expected to have lower refill conversion	Pharmacy and healthcare access may be more limited
Out-of-pocket cost	Higher cost expected to reduce conversion	Greater direct patient financial burden may increase treatment abandonment
Out-of-pocket cost is intentionally generated as correlated with payer type. Cash-pay patients are assigned substantially higher out-of-pocket costs than commercially or government-insured patients.
This means payer type and out-of-pocket cost may contain overlapping information in the logistic regression and can create multicollinearity.
I will therefore inspect variance inflation factors (VIF) and interpret payer type and out-of-pocket cost jointly rather than assuming every coefficient represents a completely independent effect.
If the cash-pay coefficient becomes smaller after out-of-pocket cost is added to the regression, that does not necessarily mean the injected payer effect was not recovered. It may indicate that the two correlated variables are competing to explain the same variation in conversion.

5. Interventions
Intervention	Target stage	Cost per patient	Expected lift	Source of the lift estimate
Copay assistance	First Fill	₹900	22%	Simulation assumption; robustness tested through sensitivity analysis
Refill reminders	Refill	₹40	6%	Simulation assumption; robustness tested through sensitivity analysis
Nurse education call	Continued Treatment	₹350	14%	Simulation assumption; robustness tested through sensitivity analysis
Digital adherence app	Continued Treatment	₹120	8%	Simulation assumption; robustness tested through sensitivity analysis
Prescriber detailing	Prescribed	₹1,500	11%	Simulation assumption; robustness tested through sensitivity analysis
All expected lift values represent a relative improvement in the current conversion rate, not a percentage-point increase.
For example, if current conversion is 50% and expected lift is 22%:
New conversion = 50% × (1 + 0.22) = 61%
The resulting conversion rate is capped at 100%.
The target stage refers to the destination stage of the transition being improved.
Examples:
* target_stage: prescribed means Diagnosis → Prescription
* target_stage: first_fill means Prescription → First Fill
* target_stage: refill means First Fill → Refill
* target_stage: continued means Refill → Continued Treatment
Intervention cost is charged to every patient targeted, not only to patients who are incrementally recovered.
For example, if 5,000 patients receive a refill reminder costing ₹40 per patient:
Total intervention cost = 5,000 × ₹40 = ₹200,000
This cost applies even if only a fraction of those patients are additionally recovered.
Be honest about the lift estimates
The intervention lift estimates are assumptions selected for the simulation, not findings from the generated patient data.
The simulator therefore includes sensitivity analysis in which the lift estimates are reduced or increased to test whether the intervention ranking remains stable.

6. Ground truth
You are generating this data, so you know the true effects you injected.
* Which effects are you injecting?
* What magnitude?
* What does "the analysis worked" mean?
Your answer
I deliberately inject four effects into the synthetic patient population:
1. Cash-pay patients receive a 35% multiplicative reduction in Prescription → First Fill conversion.
2. Rural patients receive a 20% multiplicative reduction in First Fill → Refill conversion.
3. Patients aged 75+ receive a 15% multiplicative reduction in Refill → Continued Treatment conversion.
4. Patients associated with a general physician receive a 10% multiplicative reduction in Diagnosis → Prescription conversion.
A multiplicative reduction means that the base transition probability is multiplied by:
1 − penalty
For example, if the base First Fill probability is 72%, the cash-pay First Fill probability becomes:
72% × (1 − 0.35) = 46.8%
Similarly:
* Rural refill probability = base probability × 0.80
* Age 75+ continued-treatment probability = base probability × 0.85
* General-physician Diagnosis → Prescription probability = base probability × 0.90
I consider the analysis successful if segmentation recovers the correct direction of each injected effect and the observed transition-rate ratios are reasonably close to the injected probability multipliers.
The primary validation of the injected effects is therefore based on observed eligible-cohort transition rates.
Logistic-regression odds ratios will not be compared directly with the injected probability multipliers because probability ratios and odds ratios are mathematically different.
The logistic regression is instead used to confirm:
* whether the effect appears in the expected direction,
* whether the relationship remains meaningful after adjusting for other patient features,
* the estimated strength of association,
* and the uncertainty around that estimate through confidence intervals.
Because payer type and out-of-pocket cost are intentionally correlated, I will also inspect VIF and interpret their regression coefficients carefully.
This is model validation against known synthetic ground truth, not discovery of real-world patient behavior.
Framing for the interview
Say:
"I injected a payer effect and confirmed that the segmentation and model recovered the expected pattern."
Do not say:
"I discovered that cash-pay patients abandon more."
The synthetic data was generated with that relationship intentionally built into it.

Comorbidity count is included as a regression feature with no deliberately injected effect. It functions as a null control: if the model correctly reports no meaningful relationship, this provides evidence that the analytical pipeline is not manufacturing effects that were not present in the synthetic data.

### Multiple comparisons

Validating four injected effects with four nominal 95% confidence intervals
creates a substantial chance that at least one interval will miss its true
value purely through sampling variation. Under independence, that probability
is approximately 18.5%.

On seed 42, the rural refill effect illustrates this: the observed rate ratio
is approximately 0.826, while the injected multiplier is 0.800. The nominal
95% confidence interval excludes the injected value even though the generator
itself is correctly specified.

I did not change the random seed. Choosing a new seed because a validation
result failed would amount to selecting a favourable random sample after
seeing the outcome.

Instead, I apply a Bonferroni correction for the four simultaneous validation
tests. With alpha = 0.05 and k = 4, each individual interval uses
alpha / k = 0.0125, corresponding to a 98.75% confidence interval.

Under the corrected interval, the rural effect includes the injected 0.800
multiplier.

The number of validation tests is fixed by the four deliberately injected
effects and was known before examining the results, so the multiple-comparison
correction is part of the validation design rather than a post-hoc change to
obtain a favourable result.

The estimator was also checked directly against the generator's expected
per-patient probabilities, which recover the intended multipliers of
0.9000, 0.6500, 0.8000 and 0.8500.

7. The finding you expect
Write your hypothesis now, before you see any output.
If the result matches, good. If it does not, the difference between your hypothesis and the actual result becomes an important analytical finding.
Your hypothesis
I expect the largest absolute patient leakage to occur between Prescription and First Fill because this transition has meaningful baseline abandonment and also contains the strongest deliberately injected effect.
I expect cash-pay patients to show the lowest First Fill conversion because of the injected 35% penalty.
I also expect:
* rural patients to show lower First Fill → Refill conversion,
* patients aged 75+ to show lower Refill → Continued Treatment conversion,
* and patients associated with general physicians to show somewhat lower Diagnosis → Prescription conversion.
I expect copay assistance to generate substantial total recovered revenue because it targets the high-value First Fill leakage among cash-pay patients.
However, refill reminders may achieve the highest return per rupee because their cost per targeted patient is substantially lower.
Therefore, I expect that:
the intervention with the highest total financial impact may not be the same intervention as the one with the highest ROI per rupee spent.
I also expect the intentional correlation between payer type and out-of-pocket cost to reduce the apparent independent payer effect in the multivariable regression. If this occurs, I will use VIF and joint interpretation of the two predictors to explain the result rather than treating it as a failure of the model.
