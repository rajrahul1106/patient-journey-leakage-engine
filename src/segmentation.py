"""segmentation — the funnel cut by segment, with interval estimates.

Reuses journey.py for stage assignment and censoring and funnel.py for the
conversion and revenue rules. This module slices the cohort and adds interval
estimates; it restates neither the transition logic nor the revenue rule.

Rates are reported with a Wilson score interval rather than the normal
approximation. Segment cells get small, and the normal interval misbehaves
there: it runs past 0 and 1, and collapses to zero width when no patient in a
cell converts. Wilson stays inside the boundaries and keeps a sensible width at
the extremes.

Every rate is printed beside its denominator. A rate without its cell size is
not interpretable.

validate_injected_effects is the primary ground-truth check for DECISIONS.md
section 6. For each injected effect it divides the penalised segment's
conversion rate by the rate across every other level of that dimension, at the
transition the penalty was applied to, and asks whether the injected multiplier
(1 - penalty) falls inside the ratio's confidence interval. Those intervals
carry a Bonferroni correction, because the effects are tested simultaneously;
see that function's docstring.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# The injected effects are declared once, by the generator. Reading the same
# map back means validation cannot drift from what was actually injected.
from src.generate_data import PENALTY_SCOPE
from src.funnel import rupees, stage_funnel
from src.journey import (
    build_journeys,
    load_config,
    load_inputs,
    resolve_denominator_rule,
    transitions,
)

# Two-sided normal quantile for 95%: 1.959963984540054.
Z_95 = float(norm.ppf(0.975))

# Cells smaller than this get their rate suppressed rather than reported.
MIN_CELL_SIZE = 200

# Used when config.analysis.validation_alpha is absent.
DEFAULT_VALIDATION_ALPHA = 0.05


def family_wise_error_rate(alpha, tests):
    """Chance of at least one interval missing its target across independent tests."""
    return 1 - (1 - alpha) ** tests


def wilson_interval(successes, trials, z=Z_95):
    """Wilson score interval for a proportion, elementwise over arrays.

    Unlike the normal approximation this stays within [0, 1] and keeps a
    sensible width when a cell has no successes or no failures.
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        proportion = successes / trials
        denominator = 1 + z**2 / trials
        centre = (proportion + z**2 / (2 * trials)) / denominator
        half = (z / denominator) * np.sqrt(
            proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
        )
        low = np.where(trials > 0, centre - half, np.nan)
        high = np.where(trials > 0, centre + half, np.nan)
    # Wilson cannot leave [0, 1]; clipping only absorbs floating point dust.
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def ratio_and_log_standard_error(successes_a, trials_a, successes_b, trials_b):
    """Ratio of two proportions, with the standard error of its logarithm.

    NaN where either arm has no conversions, which leaves the ratio undefined.
    """
    successes_a, trials_a, successes_b, trials_b = (
        np.asarray(value, dtype=float)
        for value in (successes_a, trials_a, successes_b, trials_b)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (successes_a / trials_a) / (successes_b / trials_b)
        standard_error = np.sqrt(
            1 / successes_a - 1 / trials_a + 1 / successes_b - 1 / trials_b
        )
    usable = (successes_a > 0) & (successes_b > 0) & (trials_a > 0) & (trials_b > 0)
    return np.where(usable, ratio, np.nan), np.where(usable, standard_error, np.nan)


def ratio_interval(successes_a, trials_a, successes_b, trials_b, z=Z_95):
    """Confidence interval for a ratio of two proportions, on the log scale.

    The Katz interval: the log ratio is far closer to normal than the ratio
    itself, which is bounded below by zero and skewed.
    """
    ratio, standard_error = ratio_and_log_standard_error(
        successes_a, trials_a, successes_b, trials_b
    )
    return ratio, ratio * np.exp(-z * standard_error), ratio * np.exp(z * standard_error)


def with_segments(journeys, patients, config):
    """The journey table with each patient's segment attributes attached."""
    columns = ["patient_id"] + list(config["segments"])
    return journeys.merge(patients[columns], on="patient_id", how="left")


def _funnel_with_intervals(cohort, config):
    """funnel.stage_funnel on a subset, plus a Wilson interval on each rate."""
    funnel = stage_funnel(cohort, config).reset_index(names="transition")
    low, high = wilson_interval(funnel["converted"], funnel["denominator"])
    funnel["ci_low"], funnel["ci_high"] = low, high
    return funnel


def segment_funnel(journeys, patients, config, dimension):
    """The funnel recomputed within each level of one segment dimension."""
    labelled = with_segments(journeys, patients, config)
    frames = []
    for level in config["segments"][dimension]:
        cohort = labelled[labelled[dimension] == level]
        funnel = _funnel_with_intervals(cohort, config)
        funnel.insert(0, "level", level)
        frames.append(funnel)
    return pd.concat(frames, ignore_index=True)


def validate_injected_effects(journeys, patients, config, alpha=None):
    """Ground-truth check for DECISIONS.md section 6, one row per injected effect.

    The penalised segment's conversion rate is divided by the rate across all
    other levels of the same dimension, at the transition the penalty was
    applied to. The effect is recovered if the injected multiplier falls inside
    that ratio's confidence interval.

    Bonferroni correction
    ---------------------
    Every injected effect is tested at once, so k intervals at a nominal 95%
    carry a family-wise error rate of 1 - 0.95^k, which is 18.5% at k = 4. With
    four simultaneous chances to miss, one interval missing its target is the
    expected outcome rather than evidence against the pipeline. Each interval
    is therefore widened to 1 - alpha/k, which bounds the family-wise error at
    alpha whatever the dependence between the tests.

    The correction is legitimate here because k is fixed in advance by the
    number of effects declared in config.injected_effects. It is not a
    threshold picked after seeing which effect failed, it does not vary with
    the result, and it would have been applied identically had every effect
    passed. The generator seed was not changed either: this reports the same
    seed 42 dataset that failed at the nominal level, with both verdicts shown
    side by side rather than the nominal one dropped.
    """
    if alpha is None:
        alpha = config.get("analysis", {}).get(
            "validation_alpha", DEFAULT_VALIDATION_ALPHA
        )
    labelled = with_segments(journeys, patients, config)
    effects = [effect for effect in PENALTY_SCOPE if effect in config["injected_effects"]]

    tests = len(effects)
    alpha_corrected = alpha / tests if tests else alpha
    z_nominal = float(norm.ppf(1 - alpha / 2))
    z_corrected = float(norm.ppf(1 - alpha_corrected / 2))

    rows = {}
    for effect in effects:
        transition_name, column, level = PENALTY_SCOPE[effect]
        in_segment = labelled[column] == level
        # Both arms go through the same funnel code as everything else.
        segment = stage_funnel(labelled[in_segment], config).loc[transition_name]
        rest = stage_funnel(labelled[~in_segment], config).loc[transition_name]

        ratio, standard_error = (
            float(value)
            for value in ratio_and_log_standard_error(
                segment["converted"], segment["denominator"],
                rest["converted"], rest["denominator"],
            )
        )
        segment_low, segment_high = wilson_interval(
            segment["converted"], segment["denominator"], z=z_nominal
        )
        rest_low, rest_high = wilson_interval(
            rest["converted"], rest["denominator"], z=z_nominal
        )
        injected = 1 - config["injected_effects"][effect]

        bounds = {}
        for label, z in [("nominal", z_nominal), ("corrected", z_corrected)]:
            bounds[f"{label}_ci_low"] = ratio * np.exp(-z * standard_error)
            bounds[f"{label}_ci_high"] = ratio * np.exp(z * standard_error)

        rows[effect] = {
            "transition": transition_name,
            "dimension": column,
            "level": level,
            "n_segment": int(segment["denominator"]),
            "rate_segment": segment["conversion_rate"],
            "segment_ci_low": float(segment_low),
            "segment_ci_high": float(segment_high),
            "n_rest": int(rest["denominator"]),
            "rate_rest": rest["conversion_rate"],
            "rest_ci_low": float(rest_low),
            "rest_ci_high": float(rest_high),
            "ratio": ratio,
            # How far the observed log ratio sits from the injected one.
            "z_score": (np.log(ratio) - np.log(injected)) / standard_error,
            **bounds,
            "injected_multiplier": injected,
            "recovered_nominal": bool(
                bounds["nominal_ci_low"] <= injected <= bounds["nominal_ci_high"]
            ),
            # The headline verdict is the corrected one.
            "recovered": bool(
                bounds["corrected_ci_low"] <= injected <= bounds["corrected_ci_high"]
            ),
            "alpha": alpha,
            "alpha_corrected": alpha_corrected,
            "tests": tests,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def rank_segment_revenue_loss(journeys, patients, config):
    """Segment level and transition combinations ranked by absolute revenue lost.

    Ranked on revenue rather than drop-off rate: a severe rate inside a small
    segment can cost less than a moderate one across a large segment. rank_by_rate
    carries what the drop-off-rate ordering would have said, so the two orderings
    can be read against each other.

    Shares are of the whole cohort's funnel loss. Levels within one dimension
    partition the cohort and so sum to 100%; the dimensions overlap each other,
    so the column does not sum to 100% down the whole table.
    """
    total_revenue_lost = stage_funnel(journeys, config)["revenue_lost"].sum()

    frames = []
    for dimension in config["segments"]:
        segment = segment_funnel(journeys, patients, config, dimension)
        segment.insert(0, "dimension", dimension)
        frames.append(segment)

    ranked = pd.concat(frames, ignore_index=True).sort_values(
        "revenue_lost", ascending=False, ignore_index=True
    )
    ranked["share_of_funnel_loss"] = ranked["revenue_lost"] / total_revenue_lost
    ranked["rank_by_rate"] = (
        ranked["drop_off_rate"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype("int64")
    )
    ranked.index = pd.RangeIndex(1, len(ranked) + 1, name="rank")
    return ranked[
        [
            "dimension",
            "level",
            "transition",
            "denominator",
            "dropped_off",
            "drop_off_rate",
            "revenue_lost",
            "share_of_funnel_loss",
            "rank_by_rate",
        ]
    ]


def two_way_segment_funnel(
    journeys, patients, config, dimension_a, dimension_b, min_cell_size=MIN_CELL_SIZE
):
    """The funnel cut by two dimensions at once, small cells suppressed.

    Cells with fewer than min_cell_size patients in the denominator keep their
    counts, which are exact, but lose their rate and interval, which are not
    stable enough to report.
    """
    labelled = with_segments(journeys, patients, config)

    frames = []
    for level_a in config["segments"][dimension_a]:
        for level_b in config["segments"][dimension_b]:
            cell = labelled[
                (labelled[dimension_a] == level_a) & (labelled[dimension_b] == level_b)
            ]
            funnel = _funnel_with_intervals(cell, config)
            funnel.insert(0, dimension_b, level_b)
            funnel.insert(0, dimension_a, level_a)
            frames.append(funnel)

    cells = pd.concat(frames, ignore_index=True)
    cells["suppressed"] = cells["denominator"] < min_cell_size
    unstable = ["conversion_rate", "drop_off_rate", "ci_low", "ci_high", "loss_share", "revenue_share"]
    cells.loc[cells["suppressed"], unstable] = np.nan
    return cells


def percent(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


def interval(low, high):
    return "—" if pd.isna(low) or pd.isna(high) else f"[{low:.1%}, {high:.1%}]"


def print_segment_funnel(segment, dimension):
    print(f"{dimension}")
    print(
        f"  {'transition':<26}{'level':<22}{'denom':>8}{'converted':>11}"
        f"{'rate':>8}{'95% CI':>18}"
    )
    for row in segment.sort_values(["transition", "level"]).itertuples():
        print(
            f"  {row.transition:<26}{row.level:<22}{row.denominator:>8,}{row.converted:>11,}"
            f"{percent(row.conversion_rate):>8}{interval(row.ci_low, row.ci_high):>18}"
        )


def number(value, places=3):
    return "—" if pd.isna(value) else f"{value:.{places}f}"


def print_validation(validation):
    """Both intervals per effect, with the corrected one carrying the verdict."""
    print("ground-truth validation, DECISIONS.md section 6")
    if validation.empty:
        print("  no injected effects to validate")
        return

    first = validation.iloc[0]
    alpha, alpha_corrected, tests = first["alpha"], first["alpha_corrected"], int(first["tests"])
    print(
        f"  {tests} simultaneous tests, alpha {alpha:g}, "
        f"Bonferroni per-test alpha {alpha_corrected:g}"
    )
    print(
        f"  family-wise error rate {family_wise_error_rate(alpha, tests):.1%} nominal, "
        f"{family_wise_error_rate(alpha_corrected, tests):.1%} corrected "
        f"(Bonferroni bounds it at {alpha:.0%} under any dependence)"
    )

    for row in validation.itertuples():
        print(f"\n  {row.Index}   ({row.transition})")
        print(
            f"    {row.level:<22}n={row.n_segment:>6,}   {percent(row.rate_segment):>6}   "
            f"{interval(row.segment_ci_low, row.segment_ci_high)}"
        )
        print(
            f"    {'all other levels':<22}n={row.n_rest:>6,}   {percent(row.rate_rest):>6}   "
            f"{interval(row.rest_ci_low, row.rest_ci_high)}"
        )
        print(
            f"    ratio {number(row.ratio)}   injected {number(row.injected_multiplier)}   "
            f"z {'—' if pd.isna(row.z_score) else format(row.z_score, '+.2f')}"
        )
        for label, level, low, high, passed in [
            ("nominal", 1 - alpha, row.nominal_ci_low, row.nominal_ci_high, row.recovered_nominal),
            ("corrected", 1 - alpha_corrected, row.corrected_ci_low, row.corrected_ci_high, row.recovered),
        ]:
            print(
                f"      {label:<10}{level * 100:g}% CI [{number(low)}, {number(high)}]   "
                f"{'PASS' if passed else 'FAIL'}"
            )

    corrected = int(validation["recovered"].sum())
    nominal = int(validation["recovered_nominal"].sum())
    total = len(validation)
    print(
        f"\n  {corrected} of {total} injected effects recovered at the corrected level "
        f"({nominal} of {total} nominal): "
        f"{'ALL PASS' if corrected == total else 'FAILURES ABOVE'}"
    )


def print_revenue_ranking(ranked, top=12):
    print(f"segment loss ranked by revenue (top {top} of {len(ranked)})")
    print(
        f"  {'rank':>4}  {'dimension':<22}{'level':<22}{'transition':<26}"
        f"{'denom':>8}{'dropped':>9}{'drop':>8}{'revenue lost':>16}{'share':>8}{'by rate':>9}"
    )
    for row in ranked.head(top).itertuples():
        print(
            f"  {row.Index:>4}  {row.dimension:<22}{row.level:<22}{row.transition:<26}"
            f"{row.denominator:>8,}{row.dropped_off:>9,}{percent(row.drop_off_rate):>8}"
            f"{rupees(row.revenue_lost):>16}{percent(row.share_of_funnel_loss):>8}"
            f"{row.rank_by_rate:>9}"
        )


def print_two_way(cells, dimension_a, dimension_b, min_cell_size):
    suppressed = int(cells["suppressed"].sum())
    print(
        f"{dimension_a} x {dimension_b}   minimum cell size {min_cell_size:,}   "
        f"{suppressed} of {len(cells)} cells suppressed   "
        f"smallest cell {cells['denominator'].min():,}"
    )
    print(
        f"  {'transition':<26}{dimension_a:<16}{dimension_b:<16}{'denom':>8}"
        f"{'rate':>8}{'95% CI':>18}"
    )
    for row in cells.sort_values(["transition", dimension_a, dimension_b]).itertuples():
        cell_a = getattr(row, dimension_a)
        cell_b = getattr(row, dimension_b)
        note = "  suppressed" if row.suppressed else ""
        print(
            f"  {row.transition:<26}{cell_a:<16}{cell_b:<16}{row.denominator:>8,}"
            f"{percent(row.conversion_rate):>8}{interval(row.ci_low, row.ci_high):>18}{note}"
        )


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    print(f"denominator rule: {rule}\n")
    for dimension in config["segments"]:
        print_segment_funnel(segment_funnel(journeys, patients, config, dimension), dimension)
        print()

    print_validation(validate_injected_effects(journeys, patients, config))
    print()
    print_revenue_ranking(rank_segment_revenue_loss(journeys, patients, config))
    print()
    cells = two_way_segment_funnel(journeys, patients, config, "payer_type", "geography")
    print_two_way(cells, "payer_type", "geography", MIN_CELL_SIZE)


if __name__ == "__main__":
    main()
