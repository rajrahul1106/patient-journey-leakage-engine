"""generate_data — synthetic patient event generator.

Reads every parameter from config/config.yaml and writes:

    data/patients.csv   one row per patient, segment and feature columns
    data/events.csv     long format, one row per event

Two rules govern this module.

1.  No event is ever dated after simulation.observation_cutoff. A patient who
    entered the cohort near the cutoff simply has fewer events. Their journey
    is not stretched to fit, cut short to fit, or flagged in any way.
2.  No stage assignment and no censoring happen here. Both belong to
    journey.py, which reads these two files back.

Choices this module makes that config.yaml does not state:

  - Segment attributes are drawn uniformly, since config.segments lists
    categories without weights.
  - Inter-event gaps are whole days drawn uniformly from 1 to the stage's
    allowed window inclusive, so every event lands strictly after the one
    before it.
  - out_of_pocket_cost is a genuine truncated normal: negative draws are
    resampled rather than clipped, so there is no point mass at zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"

# Which transition each injected effect modifies, and which segment it applies
# to. The magnitudes live in config.injected_effects; this map only records the
# scope documented in DECISIONS.md section 6. Each penalty touches its own
# transition and its own segment, nothing else.
PENALTY_SCOPE = {
    "gp_diagnosed_to_prescribed_penalty": (
        "diagnosed_to_prescribed",
        "prescriber_specialty",
        "general_physician",
    ),
    "cash_pay_first_fill_penalty": (
        "prescribed_to_first_fill",
        "payer_type",
        "cash_pay",
    ),
    "rural_refill_penalty": (
        "first_fill_to_refill",
        "geography",
        "rural",
    ),
    "age_75_plus_continued_penalty": (
        "refill_to_continued",
        "age_band",
        "75+",
    ),
}

EVENT_TYPES = ["diagnosed", "prescribed", "fill"]


def load_config(path=CONFIG_PATH):
    """Load config.yaml and refuse to run on parameters this module cannot honour."""
    with open(path) as handle:
        config = yaml.safe_load(handle)

    unscoped = set(config["injected_effects"]) - set(PENALTY_SCOPE)
    if unscoped:
        raise ValueError(
            f"injected_effects has keys with no documented scope: {sorted(unscoped)}. "
            "Document them in DECISIONS.md section 6 and add them to PENALTY_SCOPE "
            "before generating, or they would be silently ignored."
        )

    comorbidity = config["patient_features"]["comorbidity_count"]
    if comorbidity["distribution"] != "poisson":
        raise ValueError(
            f"comorbidity_count.distribution is {comorbidity['distribution']!r}; "
            "this module only implements poisson."
        )

    payers_with_cost = set(config["patient_features"]["out_of_pocket_cost"]["by_payer"])
    payers = set(config["segments"]["payer_type"])
    if payers != payers_with_cost:
        raise ValueError(
            "every payer_type needs out_of_pocket_cost parameters; "
            f"mismatch: {sorted(payers ^ payers_with_cost)}"
        )
    return config


def stage_params(config):
    return {stage["name"]: stage for stage in config["stages"]}


def draw_out_of_pocket(payer_type, by_payer, rng):
    """Normal per payer, truncated at zero by resampling rather than clipping."""
    cost = np.empty(len(payer_type), dtype=float)
    for payer, params in by_payer.items():
        in_payer = payer_type == payer
        draws = rng.normal(params["mean"], params["sd"], in_payer.sum())
        negative = draws < 0
        while negative.any():
            draws[negative] = rng.normal(params["mean"], params["sd"], negative.sum())
            negative = draws < 0
        cost[in_payer] = draws
    return np.round(cost, 2)


def draw_patients(config, rng):
    """One row per patient: cohort entry date, segment attributes, features."""
    simulation = config["simulation"]
    n_patients = simulation["n_patients"]
    start = pd.Timestamp(simulation["start_date"])
    end = pd.Timestamp(simulation["end_date"])

    entry_offset = rng.integers(0, (end - start).days + 1, n_patients)
    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:05d}" for i in range(1, n_patients + 1)],
            "cohort_entry_date": start + pd.to_timedelta(entry_offset, unit="D"),
        }
    )

    for column, categories in config["segments"].items():
        patients[column] = rng.choice(categories, size=n_patients)

    features = config["patient_features"]
    patients["comorbidity_count"] = rng.poisson(
        features["comorbidity_count"]["lam"], n_patients
    )
    patients["out_of_pocket_cost"] = draw_out_of_pocket(
        patients["payer_type"].to_numpy(),
        features["out_of_pocket_cost"]["by_payer"],
        rng,
    )
    return patients


def transition_probability(patients, config, transition):
    """Base probability for this transition, times (1 - penalty) where a penalty applies."""
    probability = np.full(
        len(patients), float(config["base_transition_probabilities"][transition])
    )
    for effect, (scope, column, value) in PENALTY_SCOPE.items():
        if scope != transition or effect not in config["injected_effects"]:
            continue
        penalty = config["injected_effects"][effect]
        in_segment = patients[column].to_numpy() == value
        probability = np.where(in_segment, probability * (1 - penalty), probability)
    return probability


def advance(day, active, probability, window, cutoff_offset, rng):
    """Take one step along the journey.

    The transition is drawn first and the date second, so the cutoff never
    changes anyone's probability of progressing. A patient whose next event
    would land past the cutoff simply has no next event.
    """
    occurs = rng.random(len(day)) < probability
    next_day = day + rng.integers(1, window + 1, len(day))
    still_active = active & occurs & (next_day <= cutoff_offset)
    return np.where(still_active, next_day, day), still_active


def simulate_events(patients, config, rng):
    """Walk every patient along diagnosed -> prescribed -> fills, stopping at the cutoff."""
    simulation = config["simulation"]
    start = pd.Timestamp(simulation["start_date"])
    cutoff_offset = (pd.Timestamp(simulation["observation_cutoff"]) - start).days
    stages = stage_params(config)
    gap_window = stages["continued"]["max_gap_days"]
    max_fills = config["revenue"]["expected_fills_full_course"]

    patient_id = patients["patient_id"].to_numpy()
    day = (patients["cohort_entry_date"] - start).dt.days.to_numpy()
    active = np.ones(len(patients), dtype=bool)
    records = [(patient_id, "diagnosed", day.copy(), None)]

    # transition, event written, window it must land in, fill number
    journey = [
        ("diagnosed_to_prescribed", "prescribed", stages["prescribed"]["max_days_from_previous"], None),
        ("prescribed_to_first_fill", "fill", stages["first_fill"]["max_days_from_previous"], 1),
        ("first_fill_to_refill", "fill", stages["refill"]["max_days_from_previous"], 2),
        ("refill_to_continued", "fill", gap_window, 3),
    ]
    for transition, event_type, window, fill_sequence in journey:
        probability = transition_probability(patients, config, transition)
        day, active = advance(day, active, probability, window, cutoff_offset, rng)
        records.append((patient_id[active], event_type, day[active], fill_sequence))

    # Fills 4 and beyond: one draw per fill at post_continuation_per_fill, until
    # the patient stops, the full course is reached, or the cutoff arrives.
    post_continuation = config["base_transition_probabilities"]["post_continuation_per_fill"]
    for fill_sequence in range(4, max_fills + 1):
        if not active.any():
            break
        day, active = advance(day, active, post_continuation, gap_window, cutoff_offset, rng)
        records.append((patient_id[active], "fill", day[active], fill_sequence))

    events = pd.concat(
        [
            pd.DataFrame(
                {
                    "patient_id": ids,
                    "event_type": event_type,
                    "event_date": start + pd.to_timedelta(days, unit="D"),
                    "fill_sequence": pd.array([fill_sequence] * len(ids), dtype="Int64"),
                }
            )
            for ids, event_type, days, fill_sequence in records
        ],
        ignore_index=True,
    )
    events = events.sort_values(
        ["patient_id", "event_date"], kind="mergesort", ignore_index=True
    )

    cutoff = pd.Timestamp(simulation["observation_cutoff"])
    if (events["event_date"] > cutoff).any():
        raise AssertionError(f"generated events dated after observation_cutoff {cutoff.date()}")
    return events


def summarize(patients, events):
    """Counts only. Stage assignment and censoring are journey.py's job."""
    by_type = events["event_type"].value_counts()
    # Gaps are at least one day, so the last row per patient is the furthest event.
    furthest = events.groupby("patient_id")["event_type"].last().value_counts()

    print(f"patients            {len(patients):>8,}")
    print(f"events              {len(events):>8,}")
    print("\nevents by type")
    for event_type in EVENT_TYPES:
        print(f"  {event_type:<16}{by_type.get(event_type, 0):>8,}")
    print("\npatients by furthest event type")
    for event_type in EVENT_TYPES:
        print(f"  {event_type:<16}{furthest.get(event_type, 0):>8,}")


def main():
    config = load_config()
    rng = np.random.default_rng(config["simulation"]["seed"])

    patients = draw_patients(config, rng)
    events = simulate_events(patients, config, rng)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    patients_path = DATA_DIR / "patients.csv"
    events_path = DATA_DIR / "events.csv"
    patients.to_csv(patients_path, index=False, date_format="%Y-%m-%d")
    events.to_csv(events_path, index=False, date_format="%Y-%m-%d")

    print(f"wrote {patients_path.relative_to(ROOT)}")
    print(f"wrote {events_path.relative_to(ROOT)}\n")
    summarize(patients, events)


if __name__ == "__main__":
    main()
