"""journey — event sequencing, stage assignment, censoring.

Reads data/patients.csv and data/events.csv and returns one row per patient:
the furthest stage reached, the date at each stage, the days between
consecutive stages, the total fill count, and one eligibility flag per
transition.

Stage boundaries follow DECISIONS.md section 1:

    diagnosed   first diagnosis event
    prescribed  first prescription within 60 days of diagnosis
    first_fill  first dispense within 30 days of the prescription
    refill      the second dispense, within 45 days of the first fill
    continued   3 consecutive fills from the first fill, no gap over 45 days

Eligibility follows DECISIONS.md section 2: a patient enters a transition's
denominator only if the full opportunity window was observable before
observation_cutoff, so nobody is counted as a drop-off who simply ran out of
time. Two denominator rules implement that, selected by
config.analysis.denominator_rule:

    window_closed        only patients whose full opportunity window elapsed
                         before the cutoff, whatever the outcome
    converter_inclusive  the same, plus patients who converted inside a window
                         that ran past the cutoff

The rules differ only over patients whose window is still open at the cutoff.
window_closed drops all of them. converter_inclusive keeps the ones who
converted, which never discards a real conversion but measures high, because
in that group only successes can be observed. compare_denominator_rules puts
both against the injected ground truth.

The window for refill -> continued is anchored at the first fill, not at the
refill, and is assessment_window_days long: it has to cover two possible
45-day gaps.

Fills are ordered by position within a patient, not by date alone, so the
"second dispense" is the second dispense even if two land on one day.
"""

from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

# The injected effects are declared once, by the generator. Reading the same
# map back means the ground-truth comparison cannot drift from what was
# actually injected.
from src.generate_data import transition_probability

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"


class Transition(NamedTuple):
    """One funnel step, and the window a patient needs to have had at it."""

    name: str
    origin: str
    destination: str
    window_anchor: str
    window_days: int


def _window_closed(at_risk, converted, window_closed):
    """Denominator of patients who were given the full opportunity window."""
    return at_risk & window_closed


def _converter_inclusive(at_risk, converted, window_closed):
    """window_closed, plus anyone who converted inside a still-open window."""
    return at_risk & (window_closed | converted)


DENOMINATOR_RULES = {
    "window_closed": _window_closed,
    "converter_inclusive": _converter_inclusive,
}


def load_config(path=CONFIG_PATH):
    with open(path) as handle:
        return yaml.safe_load(handle)


def resolve_denominator_rule(config, denominator_rule=None):
    """The explicit rule if given, otherwise config.analysis.denominator_rule."""
    rule = denominator_rule
    if rule is None:
        rule = config.get("analysis", {}).get("denominator_rule")
    if rule is None:
        raise ValueError(
            "no denominator rule: set config.analysis.denominator_rule to one of "
            f"{sorted(DENOMINATOR_RULES)}"
        )
    if rule not in DENOMINATOR_RULES:
        raise ValueError(
            f"unknown denominator_rule {rule!r}; expected one of {sorted(DENOMINATOR_RULES)}"
        )
    return rule


def transitions(config):
    stages = {stage["name"]: stage for stage in config["stages"]}
    return [
        Transition(
            "diagnosed_to_prescribed",
            "diagnosed",
            "prescribed",
            "diagnosed",
            stages["prescribed"]["max_days_from_previous"],
        ),
        Transition(
            "prescribed_to_first_fill",
            "prescribed",
            "first_fill",
            "prescribed",
            stages["first_fill"]["max_days_from_previous"],
        ),
        Transition(
            "first_fill_to_refill",
            "first_fill",
            "refill",
            "first_fill",
            stages["refill"]["max_days_from_previous"],
        ),
        # Anchored at the first fill, not the refill: the assessment window has
        # to cover both of the 45-day gaps continued treatment allows.
        Transition(
            "refill_to_continued",
            "refill",
            "continued",
            "first_fill",
            stages["continued"]["assessment_window_days"],
        ),
    ]


def _within_window(candidates, anchor, window_days):
    """Candidate events falling in [anchor, anchor + window_days] for their patient."""
    anchor_per_event = candidates["patient_id"].map(anchor)
    within = anchor_per_event.notna() & candidates["event_date"].between(
        anchor_per_event, anchor_per_event + pd.Timedelta(days=window_days)
    )
    return candidates[within]


def _fill_date_at(fill_dates, order):
    """Date of each patient's nth fill, NaT where that patient has no nth fill."""
    # -1 stands in for "no such fill" and never matches a real fill position.
    positions = order.fillna(-1).astype("int64")
    lookup = pd.MultiIndex.from_arrays([order.index, positions])
    return pd.Series(fill_dates.reindex(lookup).to_numpy(), index=order.index)


def stage_dates(patients, events, config):
    """Date at which each patient met each stage definition, NaT where never met."""
    stages = {stage["name"]: stage for stage in config["stages"]}
    patient_ids = pd.Index(patients["patient_id"], name="patient_id")

    diagnosed = (
        events[events["event_type"] == "diagnosed"]
        .groupby("patient_id")["event_date"]
        .min()
        .reindex(patient_ids)
    )
    if diagnosed.isna().any():
        missing = diagnosed.index[diagnosed.isna()].tolist()
        raise ValueError(
            f"{len(missing)} patients have no diagnosed event, so they have no cohort "
            f"entry point: {missing[:5]}"
        )

    prescriptions = events[events["event_type"] == "prescribed"]
    prescribed = (
        _within_window(prescriptions, diagnosed, stages["prescribed"]["max_days_from_previous"])
        .groupby("patient_id")["event_date"]
        .min()
        .reindex(patient_ids)
    )

    fills = events[events["event_type"] == "fill"].sort_values(
        ["patient_id", "event_date", "fill_sequence"], kind="mergesort"
    )
    fills = fills.assign(fill_order=fills.groupby("patient_id").cumcount())
    fill_dates = fills.set_index(["patient_id", "fill_order"])["event_date"]

    first_fill_order = (
        _within_window(fills, prescribed, stages["first_fill"]["max_days_from_previous"])
        .groupby("patient_id")["fill_order"]
        .min()
        .reindex(patient_ids)
    )
    first_fill = _fill_date_at(fill_dates, first_fill_order)

    # The refill is the next dispense, and only counts inside its gap.
    refill = _fill_date_at(fill_dates, first_fill_order + 1)
    refill = refill.where(
        refill <= first_fill + pd.Timedelta(days=stages["refill"]["max_days_from_previous"])
    )

    # Continued treatment is the third consecutive fill, again inside the gap.
    continued = _fill_date_at(fill_dates, first_fill_order + 2)
    continued = continued.where(
        refill.notna()
        & (continued <= refill + pd.Timedelta(days=stages["continued"]["max_gap_days"]))
    )

    return pd.DataFrame(
        {
            "diagnosed_date": diagnosed,
            "prescribed_date": prescribed,
            "first_fill_date": first_fill,
            "refill_date": refill,
            "continued_date": continued,
        }
    )


def eligible_mask(journeys, transition, cutoff, rule):
    """Who counts in this transition's denominator under the given rule."""
    at_risk = journeys[f"{transition.origin}_date"].notna()
    converted = journeys[f"{transition.destination}_date"].notna()
    window_closed = journeys[f"{transition.window_anchor}_date"] + pd.Timedelta(
        days=transition.window_days
    ) <= cutoff
    return DENOMINATOR_RULES[rule](at_risk, converted, window_closed)


def build_journeys(patients, events, config, denominator_rule=None):
    """One row per patient: furthest stage, stage dates, gaps, fills, eligibility."""
    rule = resolve_denominator_rule(config, denominator_rule)
    journeys = stage_dates(patients, events, config)
    stage_order = [stage["name"] for stage in sorted(config["stages"], key=lambda s: s["order"])]
    date_columns = [f"{name}_date" for name in stage_order]

    # cumprod stops at the first stage a patient did not meet, so a later date
    # can never promote a patient past a stage they failed.
    reached = journeys[date_columns].notna().to_numpy().cumprod(axis=1)
    journeys["furthest_stage"] = [stage_order[depth - 1] for depth in reached.sum(axis=1)]

    fills = events[events["event_type"] == "fill"]
    journeys["total_fills"] = (
        fills.groupby("patient_id").size().reindex(journeys.index, fill_value=0)
    )

    cutoff = pd.Timestamp(config["simulation"]["observation_cutoff"])
    for transition in transitions(config):
        origin = journeys[f"{transition.origin}_date"]
        destination = journeys[f"{transition.destination}_date"]
        journeys[f"days_{transition.name}"] = (destination - origin).dt.days.astype("Int64")
        journeys[f"eligible_{transition.name}"] = eligible_mask(
            journeys, transition, cutoff, rule
        )

    columns = (
        ["furthest_stage"]
        + date_columns
        + [f"days_{transition.name}" for transition in transitions(config)]
        + ["total_fills"]
        + [f"eligible_{transition.name}" for transition in transitions(config)]
    )
    return journeys[columns].reset_index()


def compare_denominator_rules(journeys, patients, config):
    """Conversion under each denominator rule, beside the injected ground truth.

    The injected rate is the mean of the per-patient probabilities the
    generator actually drew against, averaged over everyone at risk at that
    transition. Segment attributes are independent of cohort entry date, so the
    two denominators draw from the same segment mix and share this reference.
    """
    cutoff = pd.Timestamp(config["simulation"]["observation_cutoff"])
    attributes = journeys[["patient_id"]].merge(patients, on="patient_id", how="left")

    rows = {}
    for transition in transitions(config):
        at_risk = journeys[f"{transition.origin}_date"].notna()
        converted = journeys[f"{transition.destination}_date"].notna()
        injected = transition_probability(attributes, config, transition.name)

        row = {"injected": injected[at_risk.to_numpy()].mean(), "at_risk": int(at_risk.sum())}
        for rule in DENOMINATOR_RULES:
            eligible = eligible_mask(journeys, transition, cutoff, rule)
            observed = converted[eligible].mean()
            row[f"{rule}_n"] = int(eligible.sum())
            row[f"{rule}_rate"] = observed
            row[f"{rule}_gap"] = observed - row["injected"]
        rows[transition.name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def print_comparison(table):
    """Both denominator rules side by side, gap measured against the injected rate."""
    header = f"{'transition':<26}{'at risk':>9}{'injected':>10}"
    subhead = f"{'':<26}{'':>9}{'':>10}"
    for rule in DENOMINATOR_RULES:
        header += f"  {rule:^27}"
        subhead += f"  {'n':>7}{'rate':>9}{'gap':>11}"
    print(header)
    print(subhead)
    for name, row in table.iterrows():
        line = f"{name:<26}{int(row['at_risk']):>9,}{row['injected']:>10.4f}"
        for rule in DENOMINATOR_RULES:
            line += (
                f"  {int(row[f'{rule}_n']):>7,}{row[f'{rule}_rate']:>9.4f}"
                f"{row[f'{rule}_gap'] * 100:>+9.2f}pp"
            )
        print(line)


def summarize(journeys, config):
    print(f"{'transition':<26}{'at risk':>10}{'eligible':>10}{'converted':>11}{'censored':>10}")
    for transition in transitions(config):
        at_risk = journeys[f"{transition.origin}_date"].notna()
        eligible = journeys[f"eligible_{transition.name}"]
        # Converters inside the denominator: window_closed leaves some out, so
        # the global conversion count would not decompose the eligible cohort.
        converted = journeys[f"{transition.destination}_date"].notna() & eligible
        censored = at_risk & ~eligible
        print(
            f"{transition.name:<26}{at_risk.sum():>10,}{eligible.sum():>10,}"
            f"{converted.sum():>11,}{censored.sum():>10,}"
        )


def load_inputs():
    patients = pd.read_csv(DATA_DIR / "patients.csv", parse_dates=["cohort_entry_date"])
    events = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["event_date"])
    return patients, events


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    print(f"denominator rule: {rule}\n")
    summarize(journeys, config)
    print()
    print_comparison(compare_denominator_rules(journeys, patients, config))


if __name__ == "__main__":
    main()
