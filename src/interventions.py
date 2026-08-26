"""interventions — recovered revenue per intervention, ranked two ways.

Each intervention in config.interventions targets one transition: target_stage
names the destination, so target_stage: first_fill means prescribed -> first_fill.

The addressable population is the patients in that transition's denominator who
match eligible_segment and did not convert. Those are the patients an
intervention could have saved; a patient who already converted has nothing left
to recover.

expected_lift is a relative increase on that group's current conversion rate,
not a percentage-point increase, and the lifted rate is capped at 1.0
(DECISIONS.md section 5). Recovered patients are the extra conversions the
lifted rate implies across the whole addressable denominator.

Cost is charged on every patient targeted, not only on those recovered, again
per DECISIONS.md section 5. That is the difference between a plausible ROI and
a flattering one, so patients_targeted and patients_recovered are separate
columns and both are printed.

Two valuations
--------------
funnel.py values a lost patient at their remaining potential: the whole course
still ahead of them. That is right for leakage, which measures revenue at
stake, and wrong for intervention ROI. A recovered patient is not delivered to
the end of the course; they rejoin the funnel at the stage they were recovered
into and face the same downstream attrition as everyone else.

expected_realised_fills walks the funnel backwards from continued treatment,
weighting each further stage by its observed conversion rate:

    E[continued]  = 3 + sum(p^k for k in 1..9)      p = post_continuation_per_fill
    E[refill]     = 2 + p_ct * (E[continued] - 2)
    E[first_fill] = 1 + p_rf * (E[refill] - 1)
    E[prescribed] = 0 + p_ff * (E[first_fill] - 0)
    E[diagnosed]  = 0 + p_dp * (E[prescribed] - 0)

A recovery is then worth (E[stage recovered into] - fills already completed)
x per_fill. Both valuations are reported with their ratio, and the ranking is
on expected realised value, because that is the money an intervention actually
brings in. The correction is stage dependent and severe at the top of the
funnel: 5.12x at prescribed against 2.65x at continued, since a patient
recovered into a prescription still has to survive three more transitions.

Two rankings are reported, by net return and by return per rupee, because
DECISIONS.md section 7 predicts they disagree: the intervention that recovers
the most money need not be the one that recovers the most per rupee spent.

Interventions on the same transition with disjoint eligible segments are
complementary, not competing. The ranking places them in one list because they
are measured the same way, not because a choice between them is implied.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.funnel import FILLS_COMPLETED_AT_STAGE, remaining_fills, rupees, stage_funnel
from src.journey import (
    build_journeys,
    load_config,
    load_inputs,
    resolve_denominator_rule,
    transitions,
)
from src.segmentation import with_segments

SENSITIVITY_MULTIPLIERS = (0.5, 1.0, 2.0)


def observed_conversion_rates(cohort, config):
    """Conversion per transition on the eligible cohort, straight from funnel.py."""
    return stage_funnel(cohort, config)["conversion_rate"].to_dict()


def expected_realised_fills(config, conversion_rates):
    """Expected total fills a patient at each stage goes on to complete.

    Walks the funnel backwards. Reaching the next stage is worth the fills it
    adds, weighted by the observed chance of getting there, so a patient early
    in the journey is discounted by every transition still ahead of them.
    """
    post_continuation = config["base_transition_probabilities"]["post_continuation_per_fill"]
    full_course = config["revenue"]["expected_fills_full_course"]
    at_continued = FILLS_COMPLETED_AT_STAGE["continued"]

    # Fills past the third, each taken with post_continuation, up to the course.
    beyond = sum(post_continuation**step for step in range(1, full_course - at_continued + 1))
    expected = {"continued": at_continued + beyond}

    for transition in reversed(transitions(config)):
        completed = FILLS_COMPLETED_AT_STAGE[transition.origin]
        expected[transition.origin] = completed + conversion_rates[transition.name] * (
            expected[transition.destination] - completed
        )
    return expected


def target_transition(intervention, config):
    """The transition an intervention acts on: target_stage names its destination."""
    stage = intervention["target_stage"]
    for transition in transitions(config):
        if transition.destination == stage:
            return transition
    raise ValueError(
        f"intervention {intervention['name']!r} targets stage {stage!r}, which is not "
        f"the destination of any transition"
    )


def matches_segment(cohort, eligible_segment):
    """Patients matching every column in eligible_segment. Empty means everyone."""
    matches = pd.Series(True, index=cohort.index)
    for column, levels in (eligible_segment or {}).items():
        matches &= cohort[column].isin(levels)
    return matches


def addressable(cohort, config, intervention):
    """(transition, denominator, converted, targeted) masks for one intervention.

    targeted is the addressable population: in the denominator, in the eligible
    segment, and not yet converted.
    """
    transition = target_transition(intervention, config)
    eligible = cohort[f"eligible_{transition.name}"]
    converted = cohort[f"{transition.destination}_date"].notna() & eligible
    denominator = eligible & matches_segment(cohort, intervention.get("eligible_segment"))
    return transition, denominator, converted & denominator, denominator & ~converted


def intervention_return(cohort, config, intervention, lift_multiplier=1.0, expected_fills=None):
    """One intervention's recovered patients, revenue, cost and return.

    Valued both ways: revenue_potential on funnel.py's remaining-course rule and
    revenue_recovered on expected realised fills. The ranking uses the latter.
    """
    if expected_fills is None:
        expected_fills = expected_realised_fills(
            config, observed_conversion_rates(cohort, config)
        )
    transition, denominator, converted, targeted = addressable(cohort, config, intervention)

    population = int(denominator.sum())
    conversions = int(converted.sum())
    patients_targeted = int(targeted.sum())
    current_rate = conversions / population if population else np.nan

    lift = intervention["expected_lift"] * lift_multiplier
    # Relative increase on the current rate, never above certainty.
    uncapped_rate = current_rate * (1 + lift)
    lifted_rate = min(uncapped_rate, 1.0) if population else np.nan
    recovered = (lifted_rate - current_rate) * population if population else np.nan

    per_fill = config["revenue"]["per_fill"]
    completed = FILLS_COMPLETED_AT_STAGE[transition.origin]
    # What funnel.py would say the patient is worth: the whole course ahead.
    value_potential = remaining_fills(transition.origin, config) * per_fill
    # What they are actually likely to fill, having rejoined the funnel here.
    value_expected = (expected_fills[transition.destination] - completed) * per_fill

    revenue_potential = recovered * value_potential
    revenue_recovered = recovered * value_expected
    # Charged on everyone touched, not on the subset that converts because of it.
    cost = patients_targeted * intervention["cost_per_patient"]

    return {
        "name": intervention["name"],
        "target_stage": intervention["target_stage"],
        "transition": transition.name,
        "denominator": population,
        "converted": conversions,
        "current_rate": current_rate,
        "lift": lift,
        "lifted_rate": lifted_rate,
        "capped": bool(population and uncapped_rate > 1.0),
        "patients_targeted": patients_targeted,
        "patients_recovered": recovered,
        "remaining_fills": remaining_fills(transition.origin, config),
        "expected_fills": expected_fills[transition.destination],
        "value_potential": value_potential,
        "value_expected": value_expected,
        "correction_ratio": value_potential / value_expected if value_expected else np.nan,
        "revenue_potential": revenue_potential,
        "revenue_recovered": revenue_recovered,
        "cost": cost,
        "net_return_potential": revenue_potential - cost,
        "net_return": revenue_recovered - cost,
        "return_per_rupee_potential": revenue_potential / cost if cost else np.nan,
        "return_per_rupee": revenue_recovered / cost if cost else np.nan,
    }


def rank_interventions(journeys, patients, config, lift_multiplier=1.0):
    """Every intervention costed, with both rankings attached."""
    cohort = with_segments(journeys, patients, config)
    # Computed once from the observed funnel, then shared by every intervention.
    expected_fills = expected_realised_fills(config, observed_conversion_rates(cohort, config))
    table = pd.DataFrame(
        [
            intervention_return(cohort, config, intervention, lift_multiplier, expected_fills)
            for intervention in config["interventions"]
        ]
    )
    # An intervention whose eligible_segment matches no patient has no rate and
    # so no return: it stays unranked rather than being sorted to the bottom.
    for column, metric in [
        ("rank_by_net_return", "net_return"),
        ("rank_by_return_per_rupee", "return_per_rupee"),
        ("rank_by_net_return_potential", "net_return_potential"),
        ("rank_by_return_per_rupee_potential", "return_per_rupee_potential"),
    ]:
        table[column] = table[metric].rank(ascending=False, method="min").astype("Int64")
    return table.sort_values("net_return", ascending=False, ignore_index=True)


def sensitivity(journeys, patients, config, multipliers=SENSITIVITY_MULTIPLIERS):
    """The full ranking recomputed at each multiple of every expected_lift.

    Scaling every lift by the same factor scales recovered revenue but leaves
    cost untouched, so the net-return ordering can move while the ordering by
    return per rupee cannot, unless a cap binds somewhere.
    """
    runs = {
        multiplier: rank_interventions(journeys, patients, config, multiplier).set_index("name")
        for multiplier in multipliers
    }

    names = list(runs[multipliers[0]].index)
    table = pd.DataFrame(index=pd.Index(names, name="name"))
    for multiplier in multipliers:
        run = runs[multiplier].reindex(names)
        table[f"net_return_{multiplier}x"] = run["net_return"]
        table[f"rank_net_{multiplier}x"] = run["rank_by_net_return"]
        table[f"return_per_rupee_{multiplier}x"] = run["return_per_rupee"]
        table[f"rank_per_rupee_{multiplier}x"] = run["rank_by_return_per_rupee"]
        table[f"capped_{multiplier}x"] = run["capped"]
        # How far each rank sits above the one below it. A stable ordering
        # whose margins are a fraction of a percent is stable, not robust.
        ordered = run.sort_values("net_return", ascending=False)["net_return"]
        gap = ordered - ordered.shift(-1)
        table[f"net_gap_{multiplier}x"] = gap.reindex(names)
        # A relative margin only means something above zero. Once a net return
        # goes negative, a ratio against it is noise, so it is left undefined.
        table[f"net_margin_{multiplier}x"] = (gap / ordered).where(ordered > 0).reindex(names)

    net_ranks = table[[f"rank_net_{m}x" for m in multipliers]]
    rupee_ranks = table[[f"rank_per_rupee_{m}x" for m in multipliers]]
    # <= 1 rather than == 1: an unranked intervention has no rank to change.
    table["net_rank_stable"] = net_ranks.nunique(axis=1) <= 1
    table["per_rupee_rank_stable"] = rupee_ranks.nunique(axis=1) <= 1
    return table


def complementary_pairs(journeys, patients, config):
    """Same-transition interventions whose addressable populations do not overlap.

    Two interventions competing for one transition are alternatives; two that
    address disjoint populations are not, however the ranking lists them.
    """
    cohort = with_segments(journeys, patients, config)

    by_transition = defaultdict(list)
    for intervention in config["interventions"]:
        transition, _, _, targeted = addressable(cohort, config, intervention)
        by_transition[transition.name].append(
            (intervention["name"], set(cohort.index[targeted]))
        )

    pairs = []
    for transition_name, members in by_transition.items():
        for position, (name, population) in enumerate(members):
            for other_name, other_population in members[position + 1 :]:
                overlap = len(population & other_population)
                pairs.append(
                    {
                        "transition": transition_name,
                        "first": name,
                        "second": other_name,
                        "overlap": overlap,
                        "combined": len(population | other_population),
                        "complementary": overlap == 0,
                    }
                )
    return pd.DataFrame(pairs)


def rank_label(value):
    return "—" if pd.isna(value) else str(int(value))


def ratio(value):
    return "—" if pd.isna(value) else f"{value:.2f}"


def print_expected_fills(expected, config):
    """The recursion's output, so the discount applied to each stage is visible."""
    full_course = config["revenue"]["expected_fills_full_course"]
    post_continuation = config["base_transition_probabilities"]["post_continuation_per_fill"]
    print(
        f"expected realised fills by stage   post_continuation_per_fill "
        f"{post_continuation}, observed conversion"
    )
    print(f"  {'stage':<14}{'expected fills':>16}{'full course':>14}{'shortfall':>12}")
    for stage in [s["name"] for s in sorted(config["stages"], key=lambda s: s["order"])]:
        fills = expected[stage]
        print(
            f"  {stage:<14}{fills:>16.3f}{full_course:>14}"
            f"{full_course - fills:>12.3f}"
        )


def print_valuation(table):
    """Potential against expected realised, per recovered patient, with the ratio."""
    print("recovery valuation per patient: potential against expected realised")
    print(
        f"  {'intervention':<23}{'recovered into':<14}{'potential':>13}{'expected':>13}"
        f"{'ratio':>9}"
    )
    for row in table.sort_values("correction_ratio", ascending=False).itertuples():
        print(
            f"  {row.name:<23}{row.target_stage:<14}{rupees(row.value_potential):>13}"
            f"{rupees(row.value_expected):>13}{ratio(row.correction_ratio) + 'x':>9}"
        )
    print(
        "  potential is funnel.py's remaining-course rule, right for leakage. Expected\n"
        "  realised discounts a recovered patient by the transitions still ahead of them,\n"
        "  which is what an intervention actually earns. The ranking below uses expected\n"
        "  realised."
    )


def print_ranking(table):
    print("ranked by expected realised net return   (cost charged on every patient targeted)")
    print(
        f"  {'#':>2}  {'intervention':<23}{'targeted':>9}{'recovered':>11}{'revenue':>15}"
        f"{'cost':>13}{'net return':>15}{'per ₹':>8}{'by ₹':>7}{'potential rank':>16}"
    )
    for row in table.itertuples():
        print(
            f"  {rank_label(row.rank_by_net_return):>2}  {row.name:<23}"
            f"{row.patients_targeted:>9,}"
            f"{'—' if pd.isna(row.patients_recovered) else format(row.patients_recovered, ',.1f'):>11}"
            f"{rupees(row.revenue_recovered):>15}{rupees(row.cost):>13}"
            f"{rupees(row.net_return):>15}{ratio(row.return_per_rupee):>8}"
            f"{rank_label(row.rank_by_return_per_rupee):>7}"
            f"{rank_label(row.rank_by_net_return_potential):>16}"
        )

    moved = table[table["rank_by_net_return"] != table["rank_by_net_return_potential"]]
    if not moved.empty:
        print(
            f"  {len(moved)} of {len(table)} interventions change rank against the potential\n"
            f"  valuation, because the correction is stage dependent: the earlier a patient is\n"
            f"  recovered, the more attrition still stands between them and a full course."
        )

    print("\nranked by return per rupee (expected realised)")
    print(
        f"  {'#':>2}  {'intervention':<23}{'per ₹':>9}{'net return':>15}{'by net':>8}"
        f"{'potential per ₹':>17}"
    )
    for row in table.sort_values("return_per_rupee", ascending=False).itertuples():
        print(
            f"  {rank_label(row.rank_by_return_per_rupee):>2}  {row.name:<23}"
            f"{ratio(row.return_per_rupee):>9}{rupees(row.net_return):>15}"
            f"{rank_label(row.rank_by_net_return):>8}"
            f"{ratio(row.return_per_rupee_potential):>17}"
        )


def print_sensitivity(table, multipliers=SENSITIVITY_MULTIPLIERS):
    labels = "  ".join(f"{m}x" for m in multipliers)
    print(f"sensitivity: every expected_lift scaled by {labels}")

    print(f"\n  net return")
    header = f"  {'intervention':<24}"
    for multiplier in multipliers:
        header += f"{f'{multiplier}x':>16}"
    print(header + f"{'ranks':>12}{'stable':>9}")
    for name, row in table.iterrows():
        line = f"  {name:<24}"
        for multiplier in multipliers:
            line += f"{rupees(row[f'net_return_{multiplier}x']):>16}"
        ranks = "/".join(rank_label(row[f"rank_net_{m}x"]) for m in multipliers)
        print(line + f"{ranks:>12}{'yes' if row['net_rank_stable'] else 'NO':>9}")

    print(f"\n  return per rupee")
    header = f"  {'intervention':<24}"
    for multiplier in multipliers:
        header += f"{f'{multiplier}x':>12}"
    print(header + f"{'ranks':>12}{'stable':>9}")
    for name, row in table.iterrows():
        line = f"  {name:<24}"
        for multiplier in multipliers:
            line += f"{ratio(row[f'return_per_rupee_{multiplier}x']):>12}"
        ranks = "/".join(rank_label(row[f"rank_per_rupee_{m}x"]) for m in multipliers)
        print(line + f"{ranks:>12}{'yes' if row['per_rupee_rank_stable'] else 'NO':>9}")

    print("\n  closest adjacent margin on net return")
    for multiplier in multipliers:
        gaps = table[f"net_gap_{multiplier}x"].dropna()
        if gaps.empty:
            continue
        leader = gaps.idxmin()
        rank = int(table.loc[leader, f"rank_net_{multiplier}x"])
        relative = table.loc[leader, f"net_margin_{multiplier}x"]
        share = "" if pd.isna(relative) else f" ({relative:.2%})"
        print(
            f"    {multiplier}x{rupees(gaps.min()):>14}{share:>11}   between rank {rank} "
            f"({leader}) and rank {rank + 1}"
        )

    losing = {
        multiplier: list(table.index[table[f"net_return_{multiplier}x"] < 0])
        for multiplier in multipliers
    }
    if any(losing.values()):
        print("\n  net return below zero, so the intervention costs more than it recovers")
        for multiplier, names in losing.items():
            if names:
                print(f"    {multiplier}x   {', '.join(names)}")

    moved = table[~table["net_rank_stable"] | ~table["per_rupee_rank_stable"]]
    if moved.empty:
        print("\n  no intervention changes rank on either measure across this range")
    else:
        print("\n  rank changes:")
        for name, row in moved.iterrows():
            net = "/".join(rank_label(row[f"rank_net_{m}x"]) for m in multipliers)
            rupee = "/".join(rank_label(row[f"rank_per_rupee_{m}x"]) for m in multipliers)
            print(f"    {name:<24}net return {net}   return per rupee {rupee}")

    capped = [m for m in multipliers if table[f"capped_{m}x"].any()]
    if capped:
        print(f"  the 1.0 cap binds at {capped}, which is what can move the per-rupee ordering")


def print_complementary(pairs):
    complementary = pairs[pairs["complementary"]]
    if complementary.empty:
        return
    print("complementary interventions, not alternatives")
    for row in complementary.itertuples():
        print(
            f"  {row.first} and {row.second} both target {row.transition}, but address "
            f"disjoint\n  populations: {row.overlap} patients in common, "
            f"{row.combined:,} covered between them. Their ranks sit in one\n"
            f"  list because they are measured the same way, not because choosing one "
            f"rules out the other."
        )

    competing = pairs[~pairs["complementary"]]
    for row in competing.itertuples():
        print(
            f"  {row.first} and {row.second} overlap on {row.overlap:,} patients at "
            f"{row.transition}, so their\n  returns are not additive."
        )


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    print(f"denominator rule: {rule}")
    print(
        "expected_lift is a relative increase on the current conversion rate, capped at 1.0.\n"
        "Cost is charged on every patient targeted, not only those recovered "
        "(DECISIONS.md section 5).\n"
    )

    cohort = with_segments(journeys, patients, config)
    expected = expected_realised_fills(config, observed_conversion_rates(cohort, config))
    print_expected_fills(expected, config)
    print()

    table = rank_interventions(journeys, patients, config)
    print_valuation(table)
    print()
    print_ranking(table)
    print()
    print_complementary(complementary_pairs(journeys, patients, config))
    print()
    print_sensitivity(sensitivity(journeys, patients, config))


if __name__ == "__main__":
    main()
