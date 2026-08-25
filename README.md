# Patient Journey Leakage & Intervention Optimization Engine

Identifies where and why patients leave a treatment journey, then recommends
the most profitable intervention to reduce that drop-off.

> Fill this README in as you build. Problem first, findings second, usage last.
> Recruiters read the top third.

## The problem

<!-- Why does patient leakage cost pharma companies money, and why is it hard
to know which stage to fix? 3-4 sentences. -->

## The journey

Diagnosed → Prescribed → First Fill → Refill → Continued Treatment

<!-- State your stage definitions here, copied from DECISIONS.md. -->

## Findings

<!-- Fill after you run it. Lead with the number.
Example shape:
- 61% of total patient loss occurs at first fill
- Loss is concentrated in cash-pay patients (2.9x the abandonment rate)
- Copay assistance returns Rs X per Rs 1 spent, ahead of refill reminders
-->

## Method

1. **Journey builder** — sequences patient events, assigns furthest stage reached
2. **Funnel engine** — stage conversion, drop-off, revenue lost per stage
3. **Segment analyzer** — same funnel cut by payer, age, specialty, geography
4. **Driver model** — logistic regression on drop-off, interpreted via odds ratios
5. **Intervention simulator** — recovered revenue per rupee, ranked

## Validation

This runs on synthetic data with known injected effects. The analysis is
validated by confirming it recovers the effects that were injected, and their
approximate magnitude. This is model validation, not discovery.

<!-- State which effects you injected and whether the analysis recovered them. -->

## Running it

```bash
pip install -r requirements.txt
python -m src.generate_data
python -m src.report
```

## Structure

```
src/
  generate_data.py   synthetic patient event generator
  journey.py         event sequencing, stage assignment, censoring
  funnel.py          conversion rates, drop-off, revenue loss
  segmentation.py    funnel by segment, cohort comparison
  drivers.py         logistic regression, odds ratios
  interventions.py   ROI simulation and ranking
  report.py          charts and summary output
config/config.yaml   all parameters and assumptions
DECISIONS.md         design decisions and their reasoning
```

## Assumptions

<!-- Every intervention lift, revenue figure, and stage boundary is an
assumption you chose. List them plainly. Interviewers respect this. -->
