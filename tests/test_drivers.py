"""Null control, injected-effect directions, and VIF's treatment of the intercept.

Cohorts are built with fixed conversion counts per group, so the direction of
each coefficient follows from the construction rather than from a random draw.
Every categorical varies within each fixture: a level with no variation makes
an all-zero dummy column and a singular design matrix.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.drivers import (
    NULL_CONTROL_FEATURE,
    REPORTING_SCALE,
    family_size,
    injected_predictors,
    nominal_only_coefficients,
    design_matrix,
    fit_transition,
    null_control,
    odds_ratio_table,
    variance_inflation,
)
from src.journey import load_config, transitions
from tests.test_journey import Cohort, days

CONFIG = load_config()
START = pd.Timestamp(CONFIG["simulation"]["start_date"])
RULE = "window_closed"
SEGMENTS = CONFIG["segments"]

# Depth of the journey each transition splits on: converters reach this many
# fills and non-converters one fewer, except at depth 0 where the split is on
# whether a prescription exists at all.
DEPTH = {
    "diagnosed_to_prescribed": 0,
    "prescribed_to_first_fill": 1,
    "first_fill_to_refill": 2,
    "refill_to_continued": 3,
}


def fill_dates(count):
    return [START + days(50 + 30 * step) for step in range(count)]


def attributes(index, rng, overrides):
    """Segment and feature values for one patient, then any overrides.

    Segments are drawn independently rather than cycled. Cycling three-level
    dimensions on the same index makes each a function of the others, which
    gives a singular design matrix. comorbidity_count stays deterministic at
    index % 4 so the null control can be balanced exactly.

    An override may be a fixed level, or a list to draw from, which keeps a
    dimension varying while excluding the level under test.
    """
    values = {column: rng.choice(levels) for column, levels in SEGMENTS.items()}
    values["comorbidity_count"] = index % 4
    values["out_of_pocket_cost"] = float(rng.integers(100, 2500))
    for column, override in overrides.items():
        values[column] = rng.choice(override) if isinstance(override, list) else override
    return values


def build(groups, depth):
    """Cohort from (prefix, overrides, total, converters) at one journey depth."""
    cohort = Cohort()
    rows = []
    rng = np.random.default_rng(0)
    for prefix, overrides, total, converters in groups:
        for index in range(total):
            patient_id = f"{prefix}{index}"
            converted = index < converters
            if depth == 0:
                prescribed = START + days(30) if converted else None
                fills = []
            else:
                prescribed = START + days(30)
                fills = fill_dates(depth if converted else depth - 1)
            cohort.add(patient_id, diagnosed=START, prescribed=prescribed, fills=fills)
            rows.append({"patient_id": patient_id, **attributes(index, rng, overrides)})
    return cohort.build(RULE).reset_index(), pd.DataFrame(rows)


def transition_named(name):
    return next(item for item in transitions(CONFIG) if item.name == name)


class NullControlTest(unittest.TestCase):
    """comorbidity_count has a generation rule but no injected effect."""

    @classmethod
    def setUpClass(cls):
        # Every (depth, converted) group carries each comorbidity value equally
        # often, so comorbidity is balanced against conversion at every
        # transition, whichever groups that transition's cohort is drawn from.
        cohort = Cohort()
        rows = []
        rng = np.random.default_rng(1)
        for name, depth in DEPTH.items():
            for converted in [True, False]:
                for index in range(400):
                    patient_id = f"{name}_{converted}_{index}"
                    if depth == 0:
                        prescribed = START + days(30) if converted else None
                        fills = []
                    else:
                        prescribed = START + days(30)
                        fills = fill_dates(depth if converted else depth - 1)
                    cohort.add(patient_id, diagnosed=START, prescribed=prescribed, fills=fills)
                    # index % 4 inside attributes puts 100 patients at each of
                    # 0, 1, 2, 3 in every group, so comorbidity is balanced
                    # against conversion whichever groups a transition draws on.
                    rows.append({"patient_id": patient_id, **attributes(index, rng, {})})
        cls.journeys = cohort.build(RULE).reset_index()
        cls.patients = pd.DataFrame(rows)
        cls.fits = [
            fit_transition(cls.journeys, cls.patients, CONFIG, transition_named(name))
            for name in DEPTH
        ]

    def test_comorbidity_is_balanced_by_construction(self):
        merged = self.journeys.merge(self.patients, on="patient_id")
        for name in DEPTH:
            transition = transition_named(name)
            eligible = merged[merged[f"eligible_{name}"]]
            converted = eligible[f"{transition.destination}_date"].notna()
            spread = eligible.groupby("comorbidity_count").apply(
                lambda group: converted[group.index].mean(), include_groups=False
            )
            with self.subTest(transition=name):
                # Identical conversion in every comorbidity stratum: there is
                # no effect in the data for the model to find.
                self.assertAlmostEqual(spread.max() - spread.min(), 0.0, places=12)

    def test_null_control_is_not_significant_at_any_transition(self):
        control = null_control(self.fits)
        self.assertEqual(len(control), len(DEPTH))
        for name, row in control.iterrows():
            with self.subTest(transition=name):
                self.assertFalse(row["significant"])
                self.assertGreater(row["p_value"], 0.05)
                # No effect present, so the odds ratio sits on 1, up to the
                # chance correlation between comorbidity and the other columns.
                self.assertAlmostEqual(row["odds_ratio"], 1.0, places=2)
                self.assertLess(row["ci_low"], 1.0)
                self.assertGreater(row["ci_high"], 1.0)

    def test_null_control_reports_every_transition(self):
        control = null_control(self.fits)
        self.assertEqual(list(control.index), list(DEPTH))
        self.assertIn("p_value", control.columns)
        # A stricter alpha cannot turn a null result significant.
        self.assertFalse(null_control(self.fits, alpha=0.5)["significant"].any())


class InjectedDirectionTest(unittest.TestCase):
    """Each injected effect lowers conversion, so its coefficient must be negative."""

    CASES = {
        "gp_diagnosed_to_prescribed_penalty": (
            "diagnosed_to_prescribed",
            "prescriber_specialty",
            "general_physician",
            ["cardiologist", "consulting_physician"],
        ),
        "cash_pay_first_fill_penalty": (
            "prescribed_to_first_fill",
            "payer_type",
            "cash_pay",
            ["commercial", "government"],
        ),
        "rural_refill_penalty": (
            "first_fill_to_refill",
            "geography",
            "rural",
            ["metro", "tier2"],
        ),
        "age_75_plus_continued_penalty": (
            "refill_to_continued",
            "age_band",
            "75+",
            ["18-39", "40-59", "60-74"],
        ),
    }

    def test_every_injected_effect_points_downward(self):
        for effect, (name, column, level, others) in self.CASES.items():
            with self.subTest(effect=effect):
                journeys, patients = build(
                    [
                        ("PEN", {column: level}, 1000, 600),
                        ("REST", {column: others}, 1000, 800),
                    ],
                    depth=DEPTH[name],
                )
                fit = fit_transition(journeys, patients, CONFIG, transition_named(name))
                predictor = f"{column}_{level}"
                table = odds_ratio_table(fit)

                # Negative log odds, equivalently an odds ratio below 1.
                self.assertLess(fit.result.params[predictor], 0.0)
                self.assertLess(table.loc[predictor, "odds_ratio"], 1.0)
                self.assertLess(table.loc[predictor, "ci_high"], 1.0)
                self.assertLess(table.loc[predictor, "p_value"], 0.05)

    def test_reference_level_is_the_first_in_config(self):
        # The penalised level must survive as a named coefficient rather than
        # being absorbed into the intercept.
        journeys, patients = build(
            [
                ("PEN", {"geography": "rural"}, 400, 240),
                ("REST", {"geography": ["metro", "tier2"]}, 400, 320),
            ],
            depth=2,
        )
        fit = fit_transition(journeys, patients, CONFIG, transition_named("first_fill_to_refill"))
        for column, levels in SEGMENTS.items():
            with self.subTest(column=column):
                self.assertNotIn(f"{column}_{levels[0]}", fit.design.columns)
                for level in levels[1:]:
                    self.assertIn(f"{column}_{level}", fit.design.columns)


class ReportingScaleTest(unittest.TestCase):
    """Scaling is applied after the fit and never touches the model."""

    @classmethod
    def setUpClass(cls):
        journeys, patients = build(
            [("A", {}, 800, 500), ("B", {"payer_type": "cash_pay"}, 800, 400)], depth=1
        )
        cls.fit = fit_transition(
            journeys, patients, CONFIG, transition_named("prescribed_to_first_fill")
        )
        cls.table = odds_ratio_table(cls.fit)

    def test_out_of_pocket_is_reported_per_thousand_rupees(self):
        row = self.table.loc["out_of_pocket_cost"]
        self.assertEqual(row["scale_factor"], 1000.0)
        self.assertTrue(row["scale"].startswith(REPORTING_SCALE["out_of_pocket_cost"][1]))
        self.assertIn("1,000", row["scale"])
        # The label justifies the step against the cohort's own spread.
        self.assertIn("SD)", row["scale"])

    def test_comorbidity_is_reported_per_additional_condition(self):
        row = self.table.loc[NULL_CONTROL_FEATURE]
        self.assertEqual(row["scale_factor"], 1.0)
        self.assertTrue(row["scale"].startswith("per 1 additional comorbidity"))
        # Scale 1 means the reported value is the per-unit value.
        self.assertEqual(row["odds_ratio"], row["odds_ratio_per_unit"])

    def test_dummies_carry_their_reference_level_and_are_not_rescaled(self):
        dummies = self.table[self.table["scale"].str.startswith("vs ")]
        self.assertTrue((dummies["scale_factor"] == 1.0).all())
        self.assertTrue((dummies["odds_ratio"] == dummies["odds_ratio_per_unit"]).all())
        self.assertEqual(self.table.loc["payer_type_cash_pay", "scale"], "vs commercial")
        self.assertEqual(self.table.loc["geography_rural", "scale"], "vs metro")

    def test_scaling_is_exponentiation_of_the_per_unit_odds_ratio(self):
        for column in ["odds_ratio", "ci_low", "ci_high"]:
            with self.subTest(column=column):
                expected = self.table[f"{column}_per_unit"] ** self.table["scale_factor"]
                pd.testing.assert_series_equal(
                    self.table[column], expected, check_names=False
                )
        # The same quantity by the other route, exp(b * s).
        self.assertAlmostEqual(
            self.table.loc["out_of_pocket_cost", "odds_ratio"],
            float(
                np.exp(
                    self.table.loc["out_of_pocket_cost", "coefficient_per_unit"] * 1000.0
                )
            ),
            places=10,
        )

    def test_the_model_is_untouched_by_reporting(self):
        # Per-unit columns are the fitted values verbatim, and the quantities
        # that scaling cannot affect are carried through unchanged.
        params = self.fit.result.params.reindex(self.table.index)
        pd.testing.assert_series_equal(
            self.table["coefficient_per_unit"], params, check_names=False
        )
        pd.testing.assert_series_equal(
            self.table["p_value"],
            self.fit.result.pvalues.reindex(self.table.index),
            check_names=False,
        )
        # VIF is scale invariant, so it is reported off the unscaled design.
        pd.testing.assert_series_equal(
            self.table["vif"],
            variance_inflation(self.fit.design).reindex(self.table.index),
            check_names=False,
        )

    def test_scale_column_states_the_step_in_standard_deviations(self):
        cohort_sd = self.fit.design["out_of_pocket_cost"].std()
        row = self.table.loc["out_of_pocket_cost"]
        # Empirical SD on this transition's own regression cohort.
        self.assertAlmostEqual(row["predictor_sd"], cohort_sd, places=12)
        self.assertAlmostEqual(row["scale_in_sd"], 1000.0 / cohort_sd, places=12)
        self.assertEqual(row["scale"], f"per ₹1,000 ({row['scale_in_sd']:.2f} SD)")

        comorbidity = self.table.loc[NULL_CONTROL_FEATURE]
        self.assertAlmostEqual(
            comorbidity["predictor_sd"],
            self.fit.design[NULL_CONTROL_FEATURE].std(),
            places=12,
        )
        self.assertIn("SD)", comorbidity["scale"])

    def test_dummies_carry_no_standard_deviation(self):
        # A one-category shift is not a step along a continuous scale, so no
        # SD multiple is claimed for it.
        dummies = self.table[self.table["scale"].str.startswith("vs ")]
        self.assertTrue(dummies["predictor_sd"].isna().all())
        self.assertTrue(dummies["scale_in_sd"].isna().all())
        self.assertFalse(dummies["scale"].str.contains("SD").any())

    def test_scale_invariant_rank_orders_by_absolute_z(self):
        pd.testing.assert_series_equal(
            self.table["z_score"],
            self.fit.result.tvalues.reindex(self.table.index),
            check_names=False,
        )
        expected = self.table["z_score"].abs().rank(ascending=False, method="min")
        pd.testing.assert_series_equal(
            self.table["scale_invariant_rank"].astype(float), expected, check_names=False
        )
        # Rank 1 is the largest absolute z, which is also the smallest p.
        self.assertEqual(
            self.table["scale_invariant_rank"].idxmin(), self.table["p_value"].idxmin()
        )

    def test_the_rank_survives_a_change_of_scale(self):
        # The whole point of the column: rescaling moves distance from 1, and
        # must leave the z ordering alone.
        with patch.dict(REPORTING_SCALE, {"out_of_pocket_cost": (1.0, "per ₹1")}):
            rescaled = odds_ratio_table(self.fit)

        pd.testing.assert_series_equal(
            rescaled["scale_invariant_rank"].sort_index(),
            self.table["scale_invariant_rank"].sort_index(),
        )
        pd.testing.assert_series_equal(
            rescaled["z_score"].sort_index(), self.table["z_score"].sort_index()
        )
        # The reported odds ratio did move, so this is a real rescaling.
        self.assertNotAlmostEqual(
            rescaled.loc["out_of_pocket_cost", "odds_ratio"],
            self.table.loc["out_of_pocket_cost", "odds_ratio"],
            places=6,
        )
        self.assertEqual(rescaled.loc["out_of_pocket_cost", "scale_factor"], 1.0)
        # And the distance ordering did change, while the z ordering did not.
        self.assertNotEqual(list(rescaled.index), list(self.table.index))

    def test_null_control_keeps_the_per_unit_values(self):
        control = null_control([self.fit])
        row = control.loc["prescribed_to_first_fill"]
        self.assertTrue(row["scale"].startswith("per 1 additional comorbidity"))
        self.assertEqual(row["odds_ratio"], row["odds_ratio_per_unit"])
        self.assertAlmostEqual(
            float(np.exp(row["coefficient_per_unit"])), row["odds_ratio_per_unit"], places=12
        )


class MultipleComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        journeys, patients = build(
            [("A", {}, 900, 600), ("B", {"payer_type": "cash_pay"}, 900, 300)], depth=1
        )
        cls.fit = fit_transition(
            journeys, patients, CONFIG, transition_named("prescribed_to_first_fill")
        )
        cls.table = odds_ratio_table(cls.fit)

    def test_family_size_is_predictors_times_transitions(self):
        dummies = sum(len(levels) - 1 for levels in SEGMENTS.values())
        predictors = dummies + len(CONFIG["patient_features"])
        self.assertEqual(predictors, 11)
        self.assertEqual(family_size(CONFIG), predictors * len(DEPTH))
        self.assertEqual(family_size(CONFIG), 44)
        # Counted from the specification, so it matches the fitted design of a
        # full model without being derived from it.
        self.assertEqual(len(self.table), predictors)

    def test_threshold_is_alpha_over_the_family(self):
        self.assertTrue((self.table["tests"] == 44).all())
        self.assertTrue((self.table["alpha"] == 0.05).all())
        self.assertTrue((self.table["bonferroni_threshold"] == 0.05 / 44).all())
        self.assertAlmostEqual(self.table["bonferroni_threshold"].iloc[0], 0.001136, places=6)

        stricter = odds_ratio_table(self.fit, alpha=0.01)
        self.assertTrue((stricter["bonferroni_threshold"] == 0.01 / 44).all())
        explicit = odds_ratio_table(self.fit, tests=10)
        self.assertTrue((explicit["bonferroni_threshold"] == 0.05 / 10).all())

    def test_survives_correction_is_p_below_the_threshold(self):
        expected = self.table["p_value"] < self.table["bonferroni_threshold"]
        pd.testing.assert_series_equal(
            self.table["survives_correction"], expected, check_names=False
        )
        # Correction can only ever remove coefficients, never add them.
        nominal = self.table["p_value"] < self.table["alpha"]
        self.assertTrue((self.table["survives_correction"] <= nominal).all())

    def test_injected_predictors_come_from_the_generator(self):
        pairs = injected_predictors(CONFIG)
        self.assertEqual(len(pairs), len(CONFIG["injected_effects"]))
        self.assertIn(("prescribed_to_first_fill", "payer_type_cash_pay"), pairs)
        self.assertIn(("first_fill_to_refill", "geography_rural"), pairs)
        self.assertIn(("refill_to_continued", "age_band_75+"), pairs)
        self.assertIn(
            ("diagnosed_to_prescribed", "prescriber_specialty_general_physician"), pairs
        )
        # Government carries no injected effect at any transition.
        self.assertFalse(
            any(predictor == "payer_type_government" for _, predictor in pairs)
        )

    def test_nominal_only_coefficients_are_those_between_the_thresholds(self):
        flagged = nominal_only_coefficients([self.fit])
        for row in flagged.itertuples():
            with self.subTest(predictor=row.predictor):
                self.assertLess(row.p_value, 0.05)
                self.assertGreaterEqual(row.p_value, row.bonferroni_threshold)
                self.assertEqual(row.tests, 44)
        # Anything that survives correction is never flagged.
        survivors = set(self.table[self.table["survives_correction"]].index)
        self.assertFalse(survivors & set(flagged.get("predictor", [])))

    def test_a_flagged_coefficient_without_an_injected_effect_is_marked(self):
        # The cash-pay penalty is real and strong here, so it survives; anything
        # else that is only nominally significant carries injected=False.
        flagged = nominal_only_coefficients([self.fit])
        for row in flagged.itertuples():
            with self.subTest(predictor=row.predictor):
                self.assertEqual(
                    row.injected,
                    (row.transition, row.predictor) in injected_predictors(CONFIG),
                )
        self.assertTrue(
            self.table.loc["payer_type_cash_pay", "survives_correction"]
        )

    def test_no_flagged_rows_when_nothing_is_nominally_significant(self):
        # A cohort with no effects at all leaves nothing between the thresholds.
        journeys, patients = build([("A", {}, 600, 400)], depth=1)
        fit = fit_transition(
            journeys, patients, CONFIG, transition_named("prescribed_to_first_fill")
        )
        flagged = nominal_only_coefficients([fit])
        table = odds_ratio_table(fit)
        self.assertEqual(len(flagged), int(((table["p_value"] < 0.05)).sum()))


class VarianceInflationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.journeys, cls.patients = build(
            [
                ("A", {}, 800, 500),
                ("B", {"payer_type": "cash_pay"}, 800, 400),
            ],
            depth=1,
        )
        cls.fit = fit_transition(
            cls.journeys, cls.patients, CONFIG, transition_named("prescribed_to_first_fill")
        )

    def test_vif_excludes_the_intercept(self):
        design = self.fit.design
        # The intercept is in the matrix, because VIF is computed against a
        # model that has one, and out of the report, because its own VIF is not
        # interpretable as collinearity.
        self.assertIn("const", design.columns)
        vif = variance_inflation(design)
        self.assertNotIn("const", vif.index)
        self.assertEqual(set(vif.index), set(design.columns) - {"const"})
        self.assertTrue((vif > 0).all())

    def test_odds_ratio_table_drops_the_intercept_too(self):
        table = odds_ratio_table(self.fit)
        self.assertNotIn("const", table.index)
        self.assertNotIn("const", table["vif"].index)
        self.assertFalse(table["vif"].isna().any())

    def test_design_matrix_is_float_for_variance_inflation_factor(self):
        # get_dummies returns booleans and variance_inflation_factor cannot
        # take a boolean matrix, so the cast happens in design_matrix.
        design = design_matrix(
            self.journeys.merge(self.patients, on="patient_id"), CONFIG
        )
        self.assertTrue((design.dtypes == float).all())
        self.assertIn("const", design.columns)

    def test_odds_ratios_are_sorted_by_distance_from_one(self):
        table = odds_ratio_table(self.fit)
        distance = (table["odds_ratio"] - 1).abs()
        self.assertTrue(distance.is_monotonic_decreasing)
        self.assertGreater(len(table), 1)


if __name__ == "__main__":
    unittest.main()
