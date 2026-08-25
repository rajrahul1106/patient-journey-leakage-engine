"""funnel — conversion, drop-off, and revenue loss per stage.

Runs on the journey table under the denominator rule set in
config.analysis.denominator_rule, so censored patients are never counted as
drop-offs. Every rate here is measured on the eligible cohort only.

Revenue follows DECISIONS.md section 3: a lost patient costs the remaining
medication revenue,

    (expected_fills_full_course - fills completed) x per_fill

Fills completed is a property of the stage the patient was lost at, so a
patient lost at diagnosis and a patient lost after the prescription both carry
the full 12 remaining fills: neither has filled anything.

Post-continuation attrition is reported separately, per DECISIONS.md section 1.
The five-stage funnel measures leakage up to continued treatment and stops
there, so revenue forfeited by patients who reached continued treatment and
then stopped short of the full course is not funnel leakage. Those patients
converted at every transition the funnel scores.

That forfeited figure is reported three ways, because observation time
contaminates it:

    naive       every continued patient measured against the full course,
                including patients whose fill count was capped by the cutoff
    corrected   the per-patient forfeiture of patients who had room for the
                whole course before the cutoff, applied to all of them
    artifact    naive - corrected, the part attributable to unexpired
                observation time rather than to stopping treatment

The corrected figure assumes censored continued patients would have persisted
like observable ones. That is an assumption, not a measurement.
"""

import numpy as np
import pandas as pd

from src.journey import (
    build_journeys,
    load_config,
    load_inputs,
    resolve_denominator_rule,
    transitions,
)

# Fills a patient has completed on reaching each stage, from the stage
# definitions in DECISIONS.md section 1: the first fill is one dispense, the
# refill is the second, continued treatment the third.
FILLS_COMPLETED_AT_STAGE = {
    "diagnosed": 0,
    "prescribed": 0,
    "first_fill": 1,
    "refill": 2,
    "continued": 3,
}

POST_CONTINUATION_LABEL = "post_continuation (corrected)"


def rupees(amount):
    if pd.isna(amount):
        return "—"
    return f"₹{amount:,.0f}"


def remaining_fills(stage, config):
    """Fills still ahead of a patient who is lost at this stage."""
    return config["revenue"]["expected_fills_full_course"] - FILLS_COMPLETED_AT_STAGE[stage]


def stage_funnel(journeys, config):
    """One row per transition: denominator, conversion, drop-off, revenue lost."""
    per_fill = config["revenue"]["per_fill"]

    rows = {}
    for transition in transitions(config):
        eligible = journeys[f"eligible_{transition.name}"]
        # Conversions are counted inside the denominator: under window_closed a
        # patient can convert and still sit outside the eligible cohort.
        converted = journeys[f"{transition.destination}_date"].notna() & eligible
        dropped = eligible & ~converted

        denominator = int(eligible.sum())
        conversions = int(converted.sum())
        drop_offs = int(dropped.sum())
        remaining = remaining_fills(transition.origin, config)
        rows[transition.name] = {
            "denominator": denominator,
            "converted": conversions,
            "dropped_off": drop_offs,
            "conversion_rate": conversions / denominator if denominator else np.nan,
            "drop_off_rate": drop_offs / denominator if denominator else np.nan,
            "remaining_fills": remaining,
            "revenue_lost": drop_offs * remaining * per_fill,
        }

    funnel = pd.DataFrame.from_dict(rows, orient="index")
    counts = ["denominator", "converted", "dropped_off", "remaining_fills", "revenue_lost"]
    funnel[counts] = funnel[counts].astype("int64")

    # Share of the loss the whole funnel measures, patients and revenue.
    total_dropped = funnel["dropped_off"].sum()
    total_revenue = funnel["revenue_lost"].sum()
    funnel["loss_share"] = funnel["dropped_off"] / total_dropped if total_dropped else np.nan
    funnel["revenue_share"] = funnel["revenue_lost"] / total_revenue if total_revenue else np.nan
    return funnel[
        [
            "denominator",
            "converted",
            "dropped_off",
            "conversion_rate",
            "drop_off_rate",
            "loss_share",
            "remaining_fills",
            "revenue_lost",
            "revenue_share",
        ]
    ]


def post_continuation_attrition(journeys, config):
    """Revenue forfeited by patients who reached continued treatment and stopped short.

    Reported apart from the funnel, which scores these patients as converted at
    every transition it measures.

    A patient is counted as observable if there was room before the cutoff for
    the whole course at the widest gap continued treatment allows. Patients
    without that room had their fill count capped by the observation window
    rather than by stopping treatment, which is what separates the corrected
    figure from the naive one.
    """
    revenue = config["revenue"]
    stages = {stage["name"]: stage for stage in config["stages"]}
    full_course = revenue["expected_fills_full_course"]
    per_fill = revenue["per_fill"]
    cutoff = pd.Timestamp(config["simulation"]["observation_cutoff"])

    continued = journeys[journeys["continued_date"].notna()]
    fills = continued["total_fills"].clip(upper=full_course)
    forfeited = (full_course - fills) * per_fill

    course_window = continued["first_fill_date"] + pd.Timedelta(
        days=(full_course - 1) * stages["continued"]["max_gap_days"]
    )
    observable = course_window <= cutoff

    patients = len(continued)
    n_observable = int(observable.sum())
    naive = float(forfeited.sum())
    # Without an observable patient there is no uncontaminated rate to project,
    # so the correction is undefined rather than zero.
    per_patient = forfeited[observable].mean() if n_observable else np.nan
    corrected = per_patient * patients if n_observable else np.nan

    return pd.Series(
        {
            "continued_patients": patients,
            "mean_fills": fills.mean(),
            "full_course_observable": n_observable,
            "censored_patients": patients - n_observable,
            "mean_fills_observable": fills[observable].mean(),
            "mean_fills_censored": fills[~observable].mean(),
            "forfeited_per_observable_patient": per_patient,
            "revenue_forfeited_naive": naive,
            "revenue_forfeited_corrected": corrected,
            "censoring_artifact": naive - corrected,
        }
    )


def ranked_revenue_loss(funnel, post_continuation):
    """Every source of lost revenue in one ranking, so the ordering is visible.

    The patients column counts drop-offs for the funnel stages and continued
    patients stopping short of the course for post-continuation. Those are
    different populations, which is why the ranking is on revenue.
    """
    rows = [
        {
            "source": row.Index,
            "patients": int(row.dropped_off),
            "revenue_lost": float(row.revenue_lost),
        }
        for row in funnel.itertuples()
    ]
    rows.append(
        {
            "source": POST_CONTINUATION_LABEL,
            "patients": int(post_continuation["continued_patients"]),
            "revenue_lost": float(post_continuation["revenue_forfeited_corrected"]),
        }
    )
    ranked = pd.DataFrame(rows).sort_values(
        "revenue_lost", ascending=False, ignore_index=True
    )
    ranked["share"] = ranked["revenue_lost"] / ranked["revenue_lost"].sum()
    ranked.index = pd.RangeIndex(1, len(ranked) + 1, name="rank")
    return ranked


def print_funnel(funnel):
    """The stage funnel. loss and rev are shares of the funnel's total loss."""
    print(
        f"{'transition':<26}{'denom':>9}{'converted':>11}{'dropped':>9}"
        f"{'conv':>8}{'drop':>8}{'loss':>7}{'fills':>7}{'revenue lost':>16}{'rev':>8}"
    )
    # itertuples, not iterrows: iterrows would collapse each row to one dtype.
    for row in funnel.itertuples():
        print(
            f"{row.Index:<26}{row.denominator:>9,}{row.converted:>11,}{row.dropped_off:>9,}"
            f"{row.conversion_rate:>8.1%}{row.drop_off_rate:>8.1%}{row.loss_share:>7.1%}"
            f"{row.remaining_fills:>7}{rupees(row.revenue_lost):>16}{row.revenue_share:>8.1%}"
        )
    print(
        f"{'total funnel loss':<26}{'':>9}{'':>11}{funnel['dropped_off'].sum():>9,}"
        f"{'':>8}{'':>8}{1:>7.1%}{'':>7}{rupees(funnel['revenue_lost'].sum()):>16}{1:>8.1%}"
    )


def print_post_continuation(summary, config):
    """Naive, corrected and artifact, with the assumption the correction rests on."""
    full_course = config["revenue"]["expected_fills_full_course"]
    print(
        f"post-continuation attrition   {int(summary['continued_patients']):,} continued patients, "
        f"mean {summary['mean_fills']:.2f} of {full_course} fills"
    )
    print(
        f"  {'naive':<11}{rupees(summary['revenue_forfeited_naive']):>15}   "
        f"all {int(summary['continued_patients']):,} measured against the full {full_course}-fill course"
    )
    print(
        f"  {'corrected':<11}{rupees(summary['revenue_forfeited_corrected']):>15}   "
        f"{rupees(summary['forfeited_per_observable_patient'])} per patient, from the "
        f"{int(summary['full_course_observable']):,} with a full course window observable"
    )
    print(
        f"  {'artifact':<11}{rupees(summary['censoring_artifact']):>15}   "
        f"{int(summary['censored_patients']):,} censored patients average "
        f"{summary['mean_fills_censored']:.2f} fills against {summary['mean_fills_observable']:.2f} observable"
    )
    print(
        "  assumption: the corrected figure treats censored continued patients as "
        "persisting like observable ones"
    )


def print_ranked_loss(ranked):
    print(f"{'ranked revenue loss':<34}{'patients':>10}{'revenue lost':>16}{'share':>9}")
    for row in ranked.itertuples():
        print(
            f"{row.Index:>2}  {row.source:<30}{row.patients:>10,}"
            f"{rupees(row.revenue_lost):>16}{row.share:>9.1%}"
        )
    print(f"{'    total':<34}{'':>10}{rupees(ranked['revenue_lost'].sum()):>16}{1:>9.1%}")


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    funnel = stage_funnel(journeys, config)
    post_continuation = post_continuation_attrition(journeys, config)

    print(f"denominator rule: {rule}\n")
    print_funnel(funnel)
    print()
    print_post_continuation(post_continuation, config)
    print()
    print_ranked_loss(ranked_revenue_loss(funnel, post_continuation))


if __name__ == "__main__":
    main()
