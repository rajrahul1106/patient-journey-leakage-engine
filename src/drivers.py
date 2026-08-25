"""drivers — logistic regression on conversion, read through odds ratios.

One model per transition, fitted on that transition's eligible cohort under the
denominator rule in config.analysis, so censored patients never enter a
regression any more than they enter a funnel denominator.

Built for interpretability, not for prediction. No regularisation: plain
maximum likelihood, so the coefficients are the unpenalised estimates and the
standard errors mean what they normally mean. No feature engineering beyond
dummy encoding. The reference level for each categorical is the first level
listed in config.segments, which puts every injected effect on a named
coefficient instead of folding it into the intercept.

What this module does NOT do
----------------------------
It does not compare odds ratios to the injected probability multipliers.
DECISIONS.md section 6 rules that out: a probability ratio and an odds ratio
are different quantities, and an odds ratio is always further from 1 than the
probability ratio it corresponds to, so matching them would be an error rather
than a validation. The regression speaks to direction, significance, and
strength after adjustment for the other features. The ratio validation in
segmentation.validate_injected_effects remains the primary ground-truth check.

Two diagnostics carry their own interpretation:

  collinearity_diagnostic  payer type and out-of-pocket cost overlap because
                           the generator draws cost conditional on payer, as
                           DECISIONS.md section 4 states in advance. Any
                           movement in the cash-pay coefficient when cost
                           enters is a predicted result, not a discovery.

  null_control             comorbidity_count is drawn from a distribution but
                           has no injected effect, so it should come back
                           non-significant everywhere.

Continuous predictors are reported on a readable scale rather than per unit,
because an odds ratio per rupee sits within a hair of 1 by construction and
says nothing legible. Scaling is a reporting transformation applied after the
fit: if the per-unit log odds is b, the odds ratio over s units is exp(b*s),
which is the per-unit odds ratio raised to the power s. The model itself is
untouched, and p-values and VIF are unaffected, since rescaling a column
changes neither its z statistic nor its collinearity with the others. The
per-unit coefficient and odds ratio stay in the returned frame.

Justifying the scale
--------------------
Rs1,000 was picked for readability, not fitted to the data. It happens to land
near one standard deviation of out-of-pocket cost on these cohorts, which is
what puts a continuous predictor on comparable footing with a dummy's
one-category shift, and the scale column carries that multiple so the choice is
visible rather than arbitrary. Standard deviations are empirical, measured on
each transition's own regression cohort, so the multiple moves a little between
transitions.

Because distance from 1 depends on the scale chosen, the table also carries
scale_invariant_rank, which orders predictors by absolute z score. That ordering
is unaffected by any rescaling. Both orderings stay visible: rows are sorted by
distance from 1 on the reported scale, and the rank column says what the
scale-free ordering would have been.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.generate_data import PENALTY_SCOPE
from src.journey import (
    build_journeys,
    load_config,
    load_inputs,
    resolve_denominator_rule,
    transitions,
)

COLLINEARITY_TRANSITION = "prescribed_to_first_fill"
NULL_CONTROL_FEATURE = "comorbidity_count"

# How far a continuous predictor moves for one reported odds ratio, and the
# label that makes the unit explicit. Anything absent is reported per unit.
REPORTING_SCALE = {
    "out_of_pocket_cost": (1000.0, "per ₹1,000"),
    "comorbidity_count": (1.0, "per 1 additional comorbidity"),
}


class Fit(NamedTuple):
    transition: str
    result: object
    design: pd.DataFrame
    outcome: pd.Series
    config: dict


def family_size(config):
    """Coefficient tests across the whole funnel: predictors times transitions.

    Counted from the model specification rather than from a fitted design, so a
    diagnostic refit with fewer columns is still measured against the same
    family the confirmatory models belong to.
    """
    dummies = sum(len(levels) - 1 for levels in config["segments"].values())
    return (dummies + len(config["patient_features"])) * len(transitions(config))


def injected_predictors(config):
    """The (transition, predictor) pairs an injected effect was actually applied to.

    Read from the generator's own scope map, so a coefficient cannot be called
    an expected false positive while an effect sits behind it.
    """
    return {
        (transition_name, f"{column}_{level}")
        for effect, (transition_name, column, level) in PENALTY_SCOPE.items()
        if effect in config["injected_effects"]
    }


def reporting_scale(predictor, config):
    """The (multiplier, label) a predictor is reported on.

    Dummies are never rescaled, so their label names the reference level they
    are measured against instead.
    """
    if predictor in REPORTING_SCALE:
        return REPORTING_SCALE[predictor]
    for column, levels in config["segments"].items():
        if predictor.startswith(f"{column}_"):
            return 1.0, f"vs {levels[0]}"
    return 1.0, "per 1 unit"


def with_patient_attributes(journeys, patients, config):
    """Journey rows with the segment and feature columns the models need."""
    columns = ["patient_id"] + list(config["segments"]) + list(config["patient_features"])
    return journeys.merge(patients[columns], on="patient_id", how="left")


def design_matrix(cohort, config, categoricals=None, continuous=None):
    """Dummy-encoded design matrix with an intercept.

    Everything is cast to float: pd.get_dummies returns booleans, and
    variance_inflation_factor fails on a boolean matrix.
    """
    categoricals = list(config["segments"]) if categoricals is None else list(categoricals)
    continuous = list(config["patient_features"]) if continuous is None else list(continuous)

    if categoricals:
        levels = pd.DataFrame(
            {
                column: pd.Categorical(cohort[column], categories=config["segments"][column])
                for column in categoricals
            },
            index=cohort.index,
        )
        # drop_first drops the first level in config.segments, the reference.
        dummies = pd.get_dummies(levels, drop_first=True)
    else:
        dummies = pd.DataFrame(index=cohort.index)

    design = pd.concat([dummies, cohort[continuous]], axis=1).astype(float)
    return sm.add_constant(design, has_constant="add")


def conversion_outcome(cohort, transition):
    """1 where the patient reached the destination stage, matching funnel.py."""
    return cohort[f"{transition.destination}_date"].notna().astype(float)


def fit_transition(journeys, patients, config, transition, categoricals=None, continuous=None):
    """Unregularised logistic regression on one transition's eligible cohort."""
    labelled = with_patient_attributes(journeys, patients, config)
    eligible = labelled[labelled[f"eligible_{transition.name}"]]

    outcome = conversion_outcome(eligible, transition)
    design = design_matrix(eligible, config, categoricals, continuous)
    result = sm.Logit(outcome, design).fit(disp=0)
    return Fit(transition.name, result, design, outcome, config)


def variance_inflation(design):
    """VIF per predictor.

    The intercept column stays in the matrix, because VIF is computed against a
    model that has one, but it is never reported: the intercept's own VIF
    reflects how far the predictor means sit from zero rather than any
    collinearity, and is not interpretable as such.
    """
    values = design.astype(float).to_numpy()
    vif = pd.Series(
        [variance_inflation_factor(values, position) for position in range(design.shape[1])],
        index=design.columns,
        name="vif",
    )
    return vif.drop(index="const", errors="ignore")


def odds_ratio_table(fit, alpha=0.05, tests=None):
    """Odds ratios with confidence intervals, p-values and VIF, furthest from 1 first.

    odds_ratio and its interval are on the reported scale named in the scale
    column, which also states that step as a multiple of the predictor's
    empirical standard deviation on this cohort. The per-unit coefficient and
    odds ratio are kept alongside so the fit can be reproduced from the frame.

    Rows are sorted by distance from 1 on the reported scale, which is what
    makes a rupee effect comparable to a dummy at all, but that ordering depends
    on the scale chosen. scale_invariant_rank carries the ordering by absolute
    z score, which no rescaling can move.

    Every coefficient is one test in a family of predictors x transitions, so
    survives_correction compares its p-value with a Bonferroni threshold of
    alpha / tests rather than with alpha. At the default 11 predictors over 4
    transitions that is 44 tests, and about alpha * 44 coefficients will clear
    the nominal 0.05 with nothing behind them.

    The intercept is left out: it is the baseline odds for a reference patient
    rather than an effect, so ranking it by distance from 1 would mean nothing.
    """
    confidence = fit.result.conf_int(alpha=alpha)
    table = pd.DataFrame(
        {
            "coefficient_per_unit": fit.result.params,
            "odds_ratio_per_unit": np.exp(fit.result.params),
            "ci_low_per_unit": np.exp(confidence[0]),
            "ci_high_per_unit": np.exp(confidence[1]),
            "p_value": fit.result.pvalues,
        }
    ).drop(index="const", errors="ignore")

    scales = [reporting_scale(predictor, fit.config) for predictor in table.index]
    table["scale_factor"] = [factor for factor, _ in scales]

    # Empirical SD on this transition's own regression cohort, so the reported
    # step can be stated as a multiple of the spread it actually has here.
    continuous = set(fit.config["patient_features"])
    deviation = fit.design.std()
    table["predictor_sd"] = [
        deviation.get(predictor, np.nan) if predictor in continuous else np.nan
        for predictor in table.index
    ]
    table["scale_in_sd"] = table["scale_factor"] / table["predictor_sd"]
    table["scale"] = [
        label if pd.isna(multiple) else f"{label} ({multiple:.2f} SD)"
        for (_, label), multiple in zip(scales, table["scale_in_sd"])
    ]
    # exp(b * s) == exp(b) ** s. A reporting transformation, never a refit.
    for column in ["odds_ratio", "ci_low", "ci_high"]:
        table[column] = table[f"{column}_per_unit"] ** table["scale_factor"]
    table["vif"] = variance_inflation(fit.design)

    tests = family_size(fit.config) if tests is None else tests
    threshold = alpha / tests
    table["alpha"] = alpha
    table["tests"] = tests
    table["bonferroni_threshold"] = threshold
    table["survives_correction"] = table["p_value"] < threshold

    # Distance from 1 moves with the scale; the z score does not. Ranking on it
    # gives an ordering no choice of reporting units can shift.
    table["z_score"] = fit.result.tvalues
    table["scale_invariant_rank"] = (
        table["z_score"].abs().rank(ascending=False, method="min").astype("int64")
    )

    columns = [
        "scale",
        "odds_ratio",
        "ci_low",
        "ci_high",
        "p_value",
        "vif",
        "survives_correction",
        "scale_invariant_rank",
        "z_score",
        "alpha",
        "tests",
        "bonferroni_threshold",
        "scale_factor",
        "predictor_sd",
        "scale_in_sd",
        "coefficient_per_unit",
        "odds_ratio_per_unit",
        "ci_low_per_unit",
        "ci_high_per_unit",
    ]
    distance = (table["odds_ratio"] - 1).abs()
    return table.loc[distance.sort_values(ascending=False).index, columns]


def collinearity_diagnostic(journeys, patients, config, transition_name=COLLINEARITY_TRANSITION):
    """Payer type against out-of-pocket cost, the overlap DECISIONS.md section 4 predicts.

    Three fits of the same transition: payer alone, cost alone, and both. The
    movement in the cash-pay odds ratio when cost enters is the quantity of
    interest. It is a predicted consequence of generate_data drawing
    out_of_pocket_cost conditional on payer_type, not a finding about patients.
    """
    transition = next(item for item in transitions(config) if item.name == transition_name)
    models = {
        "payer only": (["payer_type"], []),
        "out-of-pocket only": ([], ["out_of_pocket_cost"]),
        "both": (["payer_type"], ["out_of_pocket_cost"]),
    }

    rows = []
    for label, (categoricals, continuous) in models.items():
        fit = fit_transition(journeys, patients, config, transition, categoricals, continuous)
        table = odds_ratio_table(fit)
        for predictor in ["payer_type_cash_pay", "out_of_pocket_cost"]:
            if predictor in table.index:
                rows.append(
                    {
                        "model": label,
                        "predictor": predictor,
                        "n": int(fit.result.nobs),
                        **table.loc[predictor].to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def null_control(fits, feature=NULL_CONTROL_FEATURE, alpha=0.05):
    """The null control's coefficient at every transition.

    comorbidity_count is drawn from a Poisson in generate_data but no injected
    effect touches it, so a significant result here would mean the pipeline was
    manufacturing an effect that was never put into the data.
    """
    rows = {}
    for fit in fits:
        row = odds_ratio_table(fit).loc[feature]
        rows[fit.transition] = {
            "scale": row["scale"],
            "odds_ratio": row["odds_ratio"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "p_value": row["p_value"],
            "significant": bool(row["p_value"] < alpha),
            "odds_ratio_per_unit": row["odds_ratio_per_unit"],
            "coefficient_per_unit": row["coefficient_per_unit"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def nominal_only_coefficients(fits, alpha=0.05):
    """Coefficients significant at alpha that do not survive the correction.

    These are the rows the multiplicity is about. A row with no injected effect
    behind it is an expected false positive: across the whole family roughly
    alpha * tests of them will appear whatever the data says, and reporting
    them as findings is exactly the error the correction guards against.
    """
    if not fits:
        return pd.DataFrame()

    injected = injected_predictors(fits[0].config)
    rows = []
    for fit in fits:
        table = odds_ratio_table(fit, alpha=alpha)
        flagged = table[(table["p_value"] < alpha) & ~table["survives_correction"]]
        for predictor, row in flagged.iterrows():
            rows.append(
                {
                    "transition": fit.transition,
                    "predictor": predictor,
                    "odds_ratio": row["odds_ratio"],
                    "p_value": row["p_value"],
                    "bonferroni_threshold": row["bonferroni_threshold"],
                    "tests": int(row["tests"]),
                    "injected": (fit.transition, predictor) in injected,
                }
            )
    return pd.DataFrame(rows)


def odds(value):
    if pd.isna(value):
        return "—"
    return f"{value:.4f}"


def p_value(value):
    if pd.isna(value):
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def print_odds_ratios(fit, table=None):
    table = odds_ratio_table(fit) if table is None else table
    converged = fit.result.mle_retvals.get("converged", True)
    print(
        f"{fit.transition}   n={int(fit.result.nobs):,}   "
        f"converted {int(fit.outcome.sum()):,} ({fit.outcome.mean():.1%})   "
        f"pseudo R2 {fit.result.prsquared:.4f}"
        f"{'' if converged else '   DID NOT CONVERGE'}"
    )
    print(
        f"  {'predictor':<43}{'scale':<39}{'odds ratio':>10}{'95% CI':>18}"
        f"{'p':>8}{'VIF':>6}{'by |z|':>7}{'survives':>10}"
    )
    for row in table.itertuples():
        # yes: clears the corrected threshold. no: nominally significant only,
        # which is the multiplicity trap. dash: not significant either way.
        if row.survives_correction:
            survives = "yes"
        elif row.p_value < row.alpha:
            survives = "no"
        else:
            survives = "—"
        print(
            f"  {row.Index:<43}{row.scale:<39}{odds(row.odds_ratio):>10}"
            f"{f'[{odds(row.ci_low)}, {odds(row.ci_high)}]':>18}"
            f"{p_value(row.p_value):>8}{row.vif:>6.2f}{row.scale_invariant_rank:>7}"
            f"{survives:>10}"
        )


def print_collinearity(diagnostic):
    print("payer / out-of-pocket collinearity, DECISIONS.md section 4")
    print(
        f"  {'model':<20}{'predictor':<22}{'scale':<39}{'odds ratio':>10}"
        f"{'95% CI':>18}{'p':>8}{'VIF':>6}"
    )
    for row in diagnostic.itertuples():
        print(
            f"  {row.model:<20}{row.predictor:<22}{row.scale:<39}{odds(row.odds_ratio):>10}"
            f"{f'[{odds(row.ci_low)}, {odds(row.ci_high)}]':>18}"
            f"{p_value(row.p_value):>8}{row.vif:>6.2f}"
        )

    cash_pay = diagnostic[diagnostic["predictor"] == "payer_type_cash_pay"].set_index("model")
    if {"payer only", "both"} <= set(cash_pay.index):
        alone = cash_pay.loc["payer only", "odds_ratio"]
        adjusted = cash_pay.loc["both", "odds_ratio"]
        alone_width = cash_pay.loc["payer only", "ci_high"] - cash_pay.loc["payer only", "ci_low"]
        both_width = cash_pay.loc["both", "ci_high"] - cash_pay.loc["both", "ci_low"]
        print(
            f"  cash-pay odds ratio moves {odds(alone)} -> {odds(adjusted)} "
            f"({(adjusted / alone - 1):+.1%}) once out-of-pocket cost enters,"
        )
        print(
            f"  while its interval widens {both_width / alone_width:.2f}x "
            f"({alone_width:.4f} -> {both_width:.4f}) and VIF rises "
            f"{cash_pay.loc['payer only', 'vif']:.2f} -> {cash_pay.loc['both', 'vif']:.2f}."
        )
    print(
        "  predicted by DECISIONS.md section 4: out_of_pocket_cost is drawn conditional on\n"
        "  payer_type, so the two carry overlapping information. Interpret them jointly."
    )


def print_null_control(control, feature=NULL_CONTROL_FEATURE):
    print(f"{feature} as a null control, DECISIONS.md section 6")
    print(
        f"  {'transition':<28}{'scale':<39}{'odds ratio':>10}{'95% CI':>18}"
        f"{'p':>8}{'significant':>13}"
    )
    for row in control.itertuples():
        print(
            f"  {row.Index:<28}{row.scale:<39}{odds(row.odds_ratio):>10}"
            f"{f'[{odds(row.ci_low)}, {odds(row.ci_high)}]':>18}"
            f"{p_value(row.p_value):>8}{'YES' if row.significant else 'no':>13}"
        )
    print(
        f"  {feature} is generated from a Poisson draw with no injected effect. A\n"
        "  non-significant result at every transition is evidence that the pipeline does not\n"
        "  manufacture effects that were never present in the data."
    )


def print_multiple_comparison(nominal_only, config, alpha=0.05):
    """The coefficients that only look significant because the family is large."""
    tests = family_size(config)
    print(f"multiple comparisons across the four models")
    print(
        f"  {tests} coefficient tests, alpha {alpha:g}, Bonferroni threshold "
        f"{alpha / tests:.6f}; about {alpha * tests:.1f} false positives expected "
        f"at the nominal level"
    )
    if nominal_only.empty:
        print("  no coefficient is significant at the nominal level without surviving correction")
        return

    print(
        f"\n  {'transition':<26}{'predictor':<24}{'odds ratio':>11}{'p':>9}"
        f"{'threshold':>12}{'injected effect':>18}"
    )
    for row in nominal_only.itertuples():
        print(
            f"  {row.transition:<26}{row.predictor:<24}{odds(row.odds_ratio):>11}"
            f"{row.p_value:>9.3f}{row.bonferroni_threshold:>12.6f}"
            f"{'YES' if row.injected else 'none':>18}"
        )

    spurious = nominal_only[~nominal_only["injected"]]
    print(
        f"\n  {len(spurious)} of {tests} tests clear the nominal {alpha:g} with no injected effect "
        f"behind them,\n  against about {alpha * tests:.1f} expected by chance. None clears the "
        f"corrected threshold, so\n  none is reported as a finding. The generator injects no payer "
        f"effect other than the\n  cash-pay first-fill penalty."
    )


def main():
    config = load_config()
    patients, events = load_inputs()
    rule = resolve_denominator_rule(config)
    journeys = build_journeys(patients, events, config, denominator_rule=rule)

    print(f"denominator rule: {rule}")
    print(
        "odds ratios are not comparable to the injected probability multipliers "
        "(DECISIONS.md section 6);\nthe ratio validation in segmentation.py is the "
        "primary ground-truth check. Direction, significance\nand adjusted strength only."
    )
    print(
        "\n₹1,000 was chosen for readability, not fitted to the data. It happens to fall close "
        "to one\nstandard deviation of out-of-pocket cost on these cohorts, which is what puts a "
        "continuous\npredictor on comparable footing with a dummy's one-category shift. Each scale "
        "column states\nthat multiple against the empirical SD of its own regression cohort."
    )
    print(
        "\nRows are sorted by distance from 1, which depends on the scale chosen. The 'by |z|' "
        "column\ncarries the scale-invariant ordering, which no choice of units can move."
    )
    tests = family_size(config)
    print(
        f"\nEach coefficient is one of {tests} tests, "
        f"{tests // len(transitions(config))} predictors across {len(transitions(config))} "
        f"transitions. The 'survives' column is\nagainst the Bonferroni threshold "
        f"{0.05 / tests:.6f}; about {0.05 * tests:.1f} false positives are expected across\n"
        f"the family at the nominal 0.05.\n"
    )

    fits = [fit_transition(journeys, patients, config, item) for item in transitions(config)]
    for fit in fits:
        print_odds_ratios(fit)
        print()

    print_collinearity(collinearity_diagnostic(journeys, patients, config))
    print()
    print_null_control(null_control(fits))
    print()
    print_multiple_comparison(nominal_only_coefficients(fits), config)


if __name__ == "__main__":
    main()
