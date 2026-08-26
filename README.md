# Patient Journey Leakage & Intervention Optimization Engine

A decision-analytics project that identifies where patients leave a treatment journey, explains which patient characteristics are associated with that leakage, quantifies the revenue at stake, and evaluates which interventions are economically worth pursuing.

The project uses synthetic pharmaceutical patient data with known ground-truth effects so the analytical pipeline can be validated before recommendations are made.

---

## Key Findings

### 1. A seemingly reasonable denominator rule introduced measurable selection bias

My initial funnel definition treated patients who had already converted as eligible even when their full observation window had not closed.

This created outcome-dependent selection: recent converters entered the denominator while comparable recent non-converters were censored.

The result was an upward conversion-rate bias of approximately **0.5–1.4 percentage points**, depending on the transition.

I replaced this with a **window-closed denominator**, where patients enter a transition denominator only after their complete opportunity window has elapsed, regardless of outcome.

The original and corrected rules remain implemented so the impact of this design choice can be reproduced.

---

### 2. A standard funnel missed one of the largest sources of revenue leakage

The five-stage funnel identified approximately:

- **₹360.5M** in potential revenue loss before Continued Treatment.

However, treatment attrition does not stop once a patient reaches the final funnel stage.

After correcting post-continuation estimates for censoring, an additional approximately:

- **₹70.1M** of potential revenue loss

was identified after Continued Treatment.

This makes post-continuation attrition the **third-largest revenue-loss source**, despite sitting completely outside the standard five-stage funnel.

Combined potential loss:

**≈ ₹430.6M**

This demonstrates why optimizing only visible funnel transitions can miss economically important persistence problems.

---

### 3. My original multicollinearity hypothesis was wrong

Out-of-pocket cost was deliberately generated as strongly correlated with payer type.

I initially predicted that adding both predictors to the logistic regression would attenuate the independent cash-pay coefficient.

It did not.

The cash-pay odds ratio changed only about:

**0.3792 → 0.3765**

while its standard error increased approximately **2.10×** and VIF increased from approximately **1.33 to 6.31**.

Out-of-pocket cost was highly predictive when payer was omitted, but became almost completely null after payer entered the model:

**p ≈ 0.928**

The result clarified an important distinction:

> Multicollinearity primarily reduces precision by inflating variance; it does not automatically bias a coefficient toward zero.

Out-of-pocket cost behaved as a **proxy variable** rather than an independent driver.

---

### 4. Correcting the intervention valuation changed the business recommendation

My first intervention model valued each recovered patient using their full remaining potential treatment value.

That is appropriate when measuring **revenue at stake** in a funnel, but it is not appropriate for intervention ROI.

A recovered patient still faces downstream refill, continuation, and post-continuation attrition.

I therefore replaced potential revenue with **expected realised revenue**, propagating each recovered patient through the remaining journey using observed downstream conversion rates.

The original method overstated intervention value by approximately:

| Recovery point | Potential-value overstatement |
|---|---:|
| Prescribed | ~5.12× |
| First Fill | ~3.23× |
| Refill | ~2.96× |
| Continued Treatment | ~2.65× |

Because the error was stage-dependent rather than a uniform multiplier, correcting it **changed every intervention rank**.

Prescriber detailing moved from near the top of the ranking to near the bottom, and under the **0.5× lift sensitivity scenario, two interventions became loss-making**.

The apparent stability of the original recommendation was therefore partly an artifact of the valuation assumption.

Final intervention recommendations are ranked using **expected realised value**, not full potential value.

---

## The Problem

Patient leakage can occur at several points between diagnosis, treatment initiation, refill, and continued therapy.

Simply identifying the stage with the worst conversion rate is not enough.

A useful commercial decision requires answering four different questions:

1. **Where** are patients being lost?
2. **Who** is most associated with that leakage?
3. **Why** might the observed pattern be occurring?
4. **Which intervention generates enough expected value to justify its cost?**

The project builds those steps into one reproducible analytical pipeline.

---

## Patient Journey

The modeled journey is:

**Diagnosed → Prescribed → First Fill → Refill → Continued Treatment**

### Stage definitions

| Stage | Definition |
|---|---|
| Diagnosed | First confirmed diagnosis within the study period |
| Prescribed | First treatment prescription within 60 days of diagnosis |
| First Fill | First medication dispense within 30 days of prescription |
| Refill | Second dispense within 45 days of first fill |
| Continued Treatment | At least 3 total consecutive fills, including the first fill, with no gap greater than 45 days |

Patients can continue receiving medication after reaching Continued Treatment.

Each additional post-continuation fill occurs with probability **0.75**, with treatment capped at **12 total fills**.

This allows persistence loss after the final funnel stage to be measured separately.

---

## Analytical Pipeline

```text
Synthetic Patient Generation
            ↓
Journey Construction
            ↓
Censoring / Eligibility
            ↓
Funnel Leakage Analysis
            ↓
Segment Analysis
            ↓
Driver Modelling
            ↓
Intervention Simulation
            ↓
Expected-Realised ROI
            ↓
Business Recommendation