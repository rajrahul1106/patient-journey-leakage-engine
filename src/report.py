"""report — four charts and a text summary of what the pipeline found.

Reads every number from the analysis modules rather than recomputing any of
them, so a figure cannot disagree with the table it came from.

Charts land in output/:

    funnel.png         where patients and revenue leave the journey
    segments.png       conversion at prescribed -> first_fill by segment, with
                       Wilson intervals and the cell size behind each rate
    odds_ratios.png    the four injected effects as adjusted odds ratios,
                       marked for whether they survive Bonferroni correction
    interventions.png  net return under both valuations, so the reordering the
                       expected-realised correction causes is visible at once

Odds ratios are shown because the regression confirms direction and
significance. They are not compared with the injected probability multipliers;
that comparison lives in segmentation.py and is a different quantity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # written to files, never displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.drivers import (
    family_size,
    fit_transition,
    injected_predictors,
    odds_ratio_table,
)
from src.funnel import (
    post_continuation_attrition,
    ranked_revenue_loss,
    rupees,
    stage_funnel,
)
from src.interventions import rank_interventions, sensitivity
from src.journey import (
    build_journeys,
    load_config,
    load_inputs,
    resolve_denominator_rule,
    transitions,
)
from src.segmentation import segment_funnel, validate_injected_effects

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"

CONVERTED = "#3B6EA5"
DROPPED = "#B3423E"
POTENTIAL = "#B0B7BF"
REALISED = "#3B6EA5"
SEGMENT_COLOURS = ["#3B6EA5", "#B3423E", "#5C8A4A", "#8A6BAA"]

FOCUS_TRANSITION = "prescribed_to_first_fill"


def millions(value, _):
    """Rupee millions without the rounding that turns 2.5M into 2M."""
    text = f"{value / 1e6:.1f}".rstrip("0").rstrip(".")
    return f"₹{text}M"


def plain_axes(ax):
    """One horizontal guide, no box, nothing else competing with the data."""
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(figure, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def chart_stage_funnel(funnel):
    """Eligible cohort per transition, split into converted and lost."""
    figure, ax = plt.subplots(figsize=(11.5, 5.0))
    position = np.arange(len(funnel))

    ax.barh(position, funnel["converted"], color=CONVERTED, height=0.62, label="converted")
    ax.barh(
        position,
        funnel["dropped_off"],
        left=funnel["converted"],
        color=DROPPED,
        height=0.62,
        label="dropped off",
    )

    for index, row in enumerate(funnel.itertuples()):
        ax.text(
            row.denominator * 1.015,
            index,
            f"{row.drop_off_rate:.1%} lost   {rupees(row.revenue_lost)}   "
            f"{row.loss_share:.0%} of funnel loss",
            va="center",
            fontsize=9,
        )
        ax.text(
            row.converted / 2,
            index,
            f"{row.converted:,}",
            va="center",
            ha="center",
            color="white",
            fontsize=9,
        )

    ax.set_yticks(position, [name.replace("_", " ") for name in funnel.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, funnel["denominator"].max() * 1.55)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_xlabel("patients in the eligible cohort")
    ax.set_title(
        f"Where the journey leaks   {funnel['dropped_off'].sum():,} patients and "
        f"{rupees(funnel['revenue_lost'].sum())} lost across four transitions",
        fontsize=12,
        pad=14,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    plain_axes(ax)
    return save(figure, "funnel.png")


def chart_segment_conversion(journeys, patients, config, cohort_rate):
    """Conversion by segment at one transition, with Wilson intervals and cell sizes."""
    rows = []
    for index, dimension in enumerate(config["segments"]):
        segment = segment_funnel(journeys, patients, config, dimension)
        for row in segment[segment["transition"] == FOCUS_TRANSITION].itertuples():
            rows.append(
                {
                    "label": f"{dimension}: {row.level}",
                    "rate": row.conversion_rate,
                    "low": row.ci_low,
                    "high": row.ci_high,
                    "denominator": row.denominator,
                    "colour": SEGMENT_COLOURS[index % len(SEGMENT_COLOURS)],
                }
            )
    table = pd.DataFrame(rows)

    figure, ax = plt.subplots(figsize=(10.5, 0.42 * len(table) + 2.2))
    position = np.arange(len(table))
    ax.errorbar(
        table["rate"],
        position,
        xerr=[table["rate"] - table["low"], table["high"] - table["rate"]],
        fmt="none",
        ecolor="#888888",
        elinewidth=1.4,
        capsize=3,
    )
    ax.scatter(table["rate"], position, c=table["colour"], s=42, zorder=3)

    for index, row in enumerate(table.itertuples()):
        ax.text(row.high + 0.008, index, f"n={row.denominator:,}", va="center", fontsize=8.5,
                color="#555555")

    # The cohort rate straight from the funnel, not re-derived from the levels.
    pooled = cohort_rate
    ax.axvline(pooled, color="#999999", linestyle="--", linewidth=1, zorder=1)
    ax.text(
        pooled,
        0.99,
        f" cohort {pooled:.1%}",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=8.5,
        color="#666666",
    )

    ax.set_yticks(position, table["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("conversion rate, eligible cohort only")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlim(table["low"].min() - 0.06, table["high"].max() + 0.09)
    ax.set_title(
        "Prescription to first fill by segment   bars are 95% Wilson intervals",
        fontsize=12,
        pad=14,
    )
    plain_axes(ax)
    return save(figure, "segments.png")


def injected_effect_rows(fits, config, alpha=0.05):
    """The adjusted odds ratio for each injected effect, with its correction verdict."""
    injected = injected_predictors(config)
    rows = []
    for fit in fits:
        table = odds_ratio_table(fit, alpha=alpha)
        for predictor in table.index:
            if (fit.transition, predictor) not in injected:
                continue
            row = table.loc[predictor]
            rows.append(
                {
                    "transition": fit.transition,
                    "predictor": predictor,
                    "odds_ratio": row["odds_ratio"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "p_value": row["p_value"],
                    "survives_correction": row["survives_correction"],
                    "threshold": row["bonferroni_threshold"],
                }
            )
    return pd.DataFrame(rows)


def chart_injected_odds_ratios(effects, config):
    """The four injected effects as odds ratios, marked for Bonferroni survival."""
    figure, ax = plt.subplots(figsize=(10.5, 4.0))
    position = np.arange(len(effects))

    ax.errorbar(
        effects["odds_ratio"],
        position,
        xerr=[
            effects["odds_ratio"] - effects["ci_low"],
            effects["ci_high"] - effects["odds_ratio"],
        ],
        fmt="none",
        ecolor="#666666",
        elinewidth=1.6,
        capsize=4,
    )
    for index, row in enumerate(effects.itertuples()):
        # Filled marker survives the correction, hollow does not.
        ax.scatter(
            row.odds_ratio,
            index,
            s=70,
            zorder=3,
            color=CONVERTED if row.survives_correction else "white",
            edgecolor=CONVERTED,
            linewidth=1.6,
        )
        verdict = "survives" if row.survives_correction else "nominal only"
        ax.text(
            row.ci_high * 1.02,
            index,
            f"p {row.p_value:.1e}   {verdict}",
            va="center",
            fontsize=8.5,
            color="#555555",
        )

    ax.axvline(1.0, color="#333333", linewidth=1)
    ax.set_yticks(
        position,
        [
            f"{row.predictor}\n{row.transition.replace('_', ' ')}"
            for row in effects.itertuples()
        ],
        fontsize=8.5,
    )
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("adjusted odds ratio (log scale), 95% CI")
    ax.set_xlim(effects["ci_low"].min() * 0.75, effects["ci_high"].max() * 2.4)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.4])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    tests = family_size(config)
    ax.set_title(
        f"Injected effects recovered by the regression   Bonferroni threshold "
        f"{0.05 / tests:.6f} over {tests} tests",
        fontsize=12,
        pad=14,
    )
    figure.text(
        0.5,
        -0.04,
        "Direction and significance only. Odds ratios are not comparable with the injected "
        "probability multipliers.",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    plain_axes(ax)
    return save(figure, "odds_ratios.png")


def chart_intervention_returns(table):
    """Net return under both valuations, ordered by the expected realised figure."""
    ordered = table.sort_values("net_return", ascending=False)
    figure, ax = plt.subplots(figsize=(11.5, 5.0))
    position = np.arange(len(ordered))
    height = 0.38

    ax.barh(
        position - height / 2,
        ordered["net_return_potential"],
        height=height,
        color=POTENTIAL,
        label="potential (remaining course)",
    )
    ax.barh(
        position + height / 2,
        ordered["net_return"],
        height=height,
        color=REALISED,
        label="expected realised",
    )

    for index, row in enumerate(ordered.itertuples()):
        ax.text(
            row.net_return_potential * 1.01,
            index - height / 2,
            f"#{int(row.rank_by_net_return_potential)}  {rupees(row.net_return_potential)}",
            va="center",
            fontsize=8.5,
            color="#555555",
        )
        ax.text(
            row.net_return * 1.01,
            index + height / 2,
            f"#{int(row.rank_by_net_return)}  {rupees(row.net_return)}   "
            f"{row.correction_ratio:.2f}x correction",
            va="center",
            fontsize=8.5,
        )

    ax.set_yticks(position, [name.replace("_", " ") for name in ordered["name"]], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, ordered["net_return_potential"].max() * 1.55)
    ax.xaxis.set_major_formatter(millions)
    ax.set_xlabel("net return")
    moved = int((table["rank_by_net_return"] != table["rank_by_net_return_potential"]).sum())
    ax.set_title(
        f"Intervention net return under both valuations   {moved} of {len(table)} "
        f"interventions change rank once downstream attrition is priced in",
        fontsize=12,
        pad=14,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    plain_axes(ax)
    return save(figure, "interventions.png")


def print_summary(funnel, post, ranked_loss, validation, interventions, sensitivity_table, config):
    per_patient = funnel["loss_share"].idxmax()
    per_revenue = funnel["revenue_share"].idxmax()

    print("FINDINGS\n")
    print("funnel leakage")
    print(
        f"  most patients lost      {per_patient}   "
        f"{funnel.loc[per_patient, 'dropped_off']:,} of {funnel['dropped_off'].sum():,} "
        f"({funnel.loc[per_patient, 'loss_share']:.1%} of funnel loss)"
    )
    print(
        f"  most revenue lost       {per_revenue}   "
        f"{rupees(funnel.loc[per_revenue, 'revenue_lost'])} of "
        f"{rupees(funnel['revenue_lost'].sum())} "
        f"({funnel.loc[per_revenue, 'revenue_share']:.1%})"
    )

    label = ranked_loss[ranked_loss["source"].str.startswith("post_continuation")]
    rank = int(label.index[0])
    print(
        f"  post-continuation       {rupees(post['revenue_forfeited_corrected'])} corrected "
        f"from {rupees(post['revenue_forfeited_naive'])} naive, rank {rank} of "
        f"{len(ranked_loss)} sources of loss"
    )
    print(
        f"                          the {rupees(post['censoring_artifact'])} difference is "
        f"unexpired observation time, not discontinuation"
    )

    corrected = int(validation["recovered"].sum())
    nominal = int(validation["recovered_nominal"].sum())
    print(f"\nground truth: {corrected} of {len(validation)} injected effects recovered "
          f"({nominal} of {len(validation)} at the uncorrected level)")
    for effect, row in validation.iterrows():
        print(
            f"  {effect:<36}ratio {row['ratio']:.3f} vs injected "
            f"{row['injected_multiplier']:.3f}   "
            f"{'PASS' if row['recovered'] else 'FAIL'}"
            f"{'' if row['recovered_nominal'] else '  (fails uncorrected)'}"
        )

    print("\ninterventions, ranked on expected realised net return")
    for row in interventions.itertuples():
        print(
            f"  {int(row.rank_by_net_return)}  {row.name:<23}{rupees(row.net_return):>14}   "
            f"{row.return_per_rupee:>6.2f} per ₹   was rank "
            f"{int(row.rank_by_net_return_potential)} on potential value"
        )

    losing = list(sensitivity_table.index[sensitivity_table["net_return_0.5x"] < 0])
    if losing:
        print(
            f"  at half the assumed lift these lose money: {', '.join(losing)}. "
            f"Both looked strongest\n  under the potential valuation, which is the "
            f"reversal the correction produces."
        )


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    funnel = stage_funnel(journeys, config)
    post = post_continuation_attrition(journeys, config)
    ranked_loss = ranked_revenue_loss(funnel, post)
    validation = validate_injected_effects(journeys, patients, config)
    fits = [fit_transition(journeys, patients, config, item) for item in transitions(config)]
    effects = injected_effect_rows(fits, config)
    interventions = rank_interventions(journeys, patients, config)
    sensitivity_table = sensitivity(journeys, patients, config)

    paths = [
        chart_stage_funnel(funnel),
        chart_segment_conversion(
            journeys, patients, config, funnel.loc[FOCUS_TRANSITION, "conversion_rate"]
        ),
        chart_injected_odds_ratios(effects, config),
        chart_intervention_returns(interventions),
    ]

    print(f"denominator rule: {rule}\n")
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")
    print()
    print_summary(funnel, post, ranked_loss, validation, interventions, sensitivity_table, config)


if __name__ == "__main__":
    main()
