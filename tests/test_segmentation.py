"""Wilson intervals, ground-truth validation ratios, and small-cell suppression.

Cohorts here are built with exact conversion counts so the expected ratios and
intervals can be worked out independently of the code under test.
"""

import math
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from src.segmentation import (
    DEFAULT_VALIDATION_ALPHA,
    MIN_CELL_SIZE,
    Z_95,
    family_wise_error_rate,
    rank_segment_revenue_loss,
    ratio_interval,
    segment_funnel,
    two_way_segment_funnel,
    validate_injected_effects,
    wilson_interval,
)
from src.journey import load_config, transitions
from tests.test_journey import Cohort, days

CONFIG = load_config()
START = pd.Timestamp(CONFIG["simulation"]["start_date"])
PER_FILL = CONFIG["revenue"]["per_fill"]
RULE = "window_closed"

# Every patient sits far inside the observation window, so denominators are the
# full at-risk counts and the arithmetic stands on its own.
OTHER_SEGMENTS = {
    "age_band": "40-59",
    "prescriber_specialty": "cardiologist",
    "geography": "metro",
}


def fill_dates(count):
    return [START + days(50 + 30 * step) for step in range(count)]


def build(groups, fills_when_converting=1):
    """Cohort from (prefix, segment overrides, total, converters) tuples.

    A converter takes fills_when_converting fills and a non-converter one
    fewer, so fills_when_converting=1 puts the split at prescribed_to_first_fill
    and 2 puts it at first_fill_to_refill.
    """
    cohort = Cohort()
    attributes = []
    for prefix, overrides, total, converts in groups:
        for index in range(total):
            patient_id = f"{prefix}{index}"
            cohort.add(
                patient_id,
                diagnosed=START,
                prescribed=START + days(30),
                fills=fill_dates(fills_when_converting - (0 if index < converts else 1)),
            )
            attributes.append({"patient_id": patient_id, **OTHER_SEGMENTS, **overrides})
    return cohort.build(RULE).reset_index(), pd.DataFrame(attributes)


class WilsonTest(unittest.TestCase):
    def test_wilson_matches_known_values(self):
        known = {
            (5, 10): (0.23659309051256394, 0.7634069094874361),
            (0, 10): (0.0, 0.27753279986288926),
            (10, 10): (0.7224672001371107, 1.0),
            (1, 100): (0.001767432064140647, 0.05448619617870533),
        }
        for (successes, trials), (expected_low, expected_high) in known.items():
            low, high = wilson_interval(successes, trials)
            with self.subTest(successes=successes, trials=trials):
                self.assertAlmostEqual(float(low), expected_low, places=12)
                self.assertAlmostEqual(float(high), expected_high, places=12)

    def test_wilson_agrees_with_statsmodels(self):
        for trials in [1, 7, 40, 200, 5000]:
            for successes in [0, 1, trials // 3, trials - 1, trials]:
                expected = proportion_confint(successes, trials, alpha=0.05, method="wilson")
                low, high = wilson_interval(successes, trials)
                with self.subTest(successes=successes, trials=trials):
                    self.assertAlmostEqual(float(low), expected[0], places=12)
                    self.assertAlmostEqual(float(high), expected[1], places=12)

    def test_wilson_stays_inside_the_boundaries(self):
        # The normal approximation gives a zero-width interval at [0, 0] here.
        # Wilson keeps a real upper bound, which is the reason for using it.
        low, high = wilson_interval([0, 12], [12, 12])
        # Bounds land on 0 and 1 to floating point, matching statsmodels.
        self.assertAlmostEqual(float(low[0]), 0.0, places=12)
        self.assertAlmostEqual(float(high[1]), 1.0, places=12)
        self.assertTrue((low >= 0.0).all() and (high <= 1.0).all())
        # The point of the interval: it does not collapse at the boundaries.
        self.assertGreater(high[0], 0.2)
        self.assertLess(low[1], 0.8)

    def test_wilson_is_undefined_for_an_empty_cell(self):
        low, high = wilson_interval(0, 0)
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))

    def test_ratio_interval_brackets_its_point_estimate(self):
        ratio, low, high = ratio_interval(468, 1000, 720, 1000)
        self.assertAlmostEqual(float(ratio), 0.65, places=12)
        self.assertLess(float(low), 0.65)
        self.assertGreater(float(high), 0.65)
        # Katz: the interval is symmetric in logs, not in the ratio itself.
        self.assertAlmostEqual(
            np.log(float(high)) - np.log(float(ratio)),
            np.log(float(ratio)) - np.log(float(low)),
            places=12,
        )

    def test_ratio_interval_is_undefined_without_conversions(self):
        ratio, low, high = ratio_interval(0, 100, 50, 100)
        for value in (ratio, low, high):
            self.assertTrue(np.isnan(float(value)))


class ValidationTest(unittest.TestCase):
    def test_recovered_effect_passes(self):
        # cash_pay converts at exactly 0.65x the others, the injected multiplier.
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 1000, 468),
                ("COMM", {"payer_type": "commercial"}, 1000, 720),
            ]
        )
        row = validate_injected_effects(journeys, patients, CONFIG).loc[
            "cash_pay_first_fill_penalty"
        ]
        self.assertEqual(row["transition"], "prescribed_to_first_fill")
        self.assertEqual(row["n_segment"], 1000)
        self.assertEqual(row["n_rest"], 1000)
        self.assertAlmostEqual(row["rate_segment"], 0.468, places=12)
        self.assertAlmostEqual(row["rate_rest"], 0.720, places=12)
        self.assertAlmostEqual(row["ratio"], 0.65, places=12)
        self.assertAlmostEqual(row["ratio"], row["rate_segment"] / row["rate_rest"], places=12)
        self.assertAlmostEqual(row["injected_multiplier"], 0.65, places=12)
        for label in ["nominal", "corrected"]:
            self.assertLessEqual(row[f"{label}_ci_low"], 0.65)
            self.assertGreaterEqual(row[f"{label}_ci_high"], 0.65)
        self.assertTrue(row["recovered_nominal"])
        self.assertTrue(row["recovered"])
        # The correction only ever widens the interval.
        self.assertLess(row["corrected_ci_low"], row["nominal_ci_low"])
        self.assertGreater(row["corrected_ci_high"], row["nominal_ci_high"])

    def test_absent_effect_fails(self):
        # No penalty in the data at all: the ratio is 1.0 and the injected 0.65
        # sits far outside the interval, which must be reported as a failure.
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 1000, 720),
                ("COMM", {"payer_type": "commercial"}, 1000, 720),
            ]
        )
        row = validate_injected_effects(journeys, patients, CONFIG).loc[
            "cash_pay_first_fill_penalty"
        ]
        self.assertAlmostEqual(row["ratio"], 1.0, places=12)
        self.assertGreater(row["corrected_ci_low"], 0.65)
        self.assertFalse(row["recovered_nominal"])
        self.assertFalse(row["recovered"])

    def test_every_injected_effect_is_checked(self):
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 500, 234),
                ("COMM", {"payer_type": "commercial"}, 500, 360),
            ]
        )
        validation = validate_injected_effects(journeys, patients, CONFIG)
        self.assertEqual(set(validation.index), set(CONFIG["injected_effects"]))
        for effect, row in validation.iterrows():
            with self.subTest(effect=effect):
                self.assertAlmostEqual(
                    row["injected_multiplier"], 1 - CONFIG["injected_effects"][effect], places=12
                )
        # Dimensions with no patients at the penalised level cannot be judged,
        # and must come back undefined rather than as a spurious pass.
        empty = validation.loc["rural_refill_penalty"]
        self.assertEqual(empty["n_segment"], 0)
        self.assertTrue(pd.isna(empty["ratio"]))
        self.assertFalse(empty["recovered"])


class BonferroniTest(unittest.TestCase):
    def test_bonferroni_z_at_four_tests(self):
        alpha, tests = DEFAULT_VALIDATION_ALPHA, 4
        self.assertEqual(alpha, 0.05)
        per_test = alpha / tests
        self.assertAlmostEqual(per_test, 0.0125, places=15)
        # Two-sided quantile at 1 - alpha/k, against 1.95996 uncorrected.
        self.assertAlmostEqual(
            float(norm.ppf(1 - per_test / 2)), 2.497705474412374, places=12
        )
        self.assertAlmostEqual(float(norm.ppf(1 - alpha / 2)), 1.959963984540054, places=12)
        self.assertAlmostEqual(Z_95, 1.959963984540054, places=12)
        self.assertAlmostEqual(family_wise_error_rate(alpha, tests), 0.18549375, places=12)
        self.assertAlmostEqual(
            family_wise_error_rate(per_test, tests), 0.049070288085937275, places=12
        )
        # Bonferroni holds the family-wise rate under alpha; nominal does not.
        self.assertLess(family_wise_error_rate(per_test, tests), alpha)
        self.assertGreater(family_wise_error_rate(alpha, tests), alpha)

    def test_correction_reads_alpha_from_config(self):
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 500, 234),
                ("COMM", {"payer_type": "commercial"}, 500, 360),
            ]
        )
        validation = validate_injected_effects(journeys, patients, CONFIG)
        row = validation.iloc[0]
        self.assertEqual(row["tests"], len(CONFIG["injected_effects"]))
        self.assertAlmostEqual(row["alpha"], CONFIG["analysis"]["validation_alpha"], places=15)
        self.assertAlmostEqual(row["alpha_corrected"], row["alpha"] / row["tests"], places=15)
        # An explicit alpha overrides the configured one.
        stricter = validate_injected_effects(journeys, patients, CONFIG, alpha=0.01)
        self.assertAlmostEqual(stricter.iloc[0]["alpha"], 0.01, places=15)
        self.assertAlmostEqual(stricter.iloc[0]["alpha_corrected"], 0.0025, places=15)

    def test_rural_effect_passes_corrected_and_fails_nominal(self):
        # The seed 42 cell counts, reproduced exactly: the rural arm drew about
        # two standard errors high, so its nominal interval clears the injected
        # 0.80 while the corrected interval still covers it.
        journeys, patients = build(
            [
                ("RURAL", {"geography": "rural", "payer_type": "commercial"}, 3082, 1976),
                ("METRO", {"geography": "metro", "payer_type": "commercial"}, 6020, 4674),
            ],
            fills_when_converting=2,
        )
        row = validate_injected_effects(journeys, patients, CONFIG).loc["rural_refill_penalty"]

        self.assertEqual(row["transition"], "first_fill_to_refill")
        self.assertEqual(row["n_segment"], 3082)
        self.assertEqual(row["n_rest"], 6020)
        self.assertAlmostEqual(row["rate_segment"], 1976 / 3082, places=12)
        self.assertAlmostEqual(row["rate_rest"], 4674 / 6020, places=12)
        self.assertAlmostEqual(row["ratio"], (1976 / 3082) / (4674 / 6020), places=12)
        self.assertAlmostEqual(row["ratio"], 0.8258, places=4)
        self.assertAlmostEqual(row["injected_multiplier"], 0.80, places=12)
        expected_z = (math.log(row["ratio"]) - math.log(0.80)) / math.sqrt(
            1 / 1976 - 1 / 3082 + 1 / 4674 - 1 / 6020
        )
        self.assertAlmostEqual(row["z_score"], expected_z, places=12)
        self.assertAlmostEqual(row["z_score"], 2.0935, places=4)

        # Nominal excludes the injected value; corrected covers it.
        self.assertGreater(row["nominal_ci_low"], 0.80)
        self.assertFalse(row["recovered_nominal"])
        self.assertLess(row["corrected_ci_low"], 0.80)
        self.assertGreater(row["corrected_ci_high"], 0.80)
        self.assertTrue(row["recovered"])
        # The verdict flipped only because the interval widened, not because
        # the estimate moved.
        self.assertLess(row["corrected_ci_low"], row["nominal_ci_low"])
        self.assertGreater(row["corrected_ci_high"], row["nominal_ci_high"])


class SegmentFunnelTest(unittest.TestCase):
    def test_levels_partition_the_cohort(self):
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 300, 150),
                ("COMM", {"payer_type": "commercial"}, 500, 400),
                ("GOVT", {"payer_type": "government"}, 200, 100),
            ]
        )
        segment = segment_funnel(journeys, patients, CONFIG, "payer_type")
        for transition in transitions(CONFIG):
            rows = segment[segment["transition"] == transition.name]
            with self.subTest(transition=transition.name):
                self.assertEqual(len(rows), 3)
                self.assertEqual(
                    rows["denominator"].sum(),
                    journeys[f"eligible_{transition.name}"].sum(),
                )
        first_fill = segment[segment["transition"] == "prescribed_to_first_fill"].set_index("level")
        self.assertEqual(first_fill.loc["cash_pay", "denominator"], 300)
        self.assertEqual(first_fill.loc["cash_pay", "converted"], 150)
        self.assertAlmostEqual(first_fill.loc["cash_pay", "conversion_rate"], 0.5, places=12)

    def test_ranking_is_by_revenue_not_by_rate(self):
        # A brutal rate in a small segment against a mild one in a large segment.
        journeys, patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 100, 10),
                ("COMM", {"payer_type": "commercial"}, 2000, 1400),
            ]
        )
        ranked = rank_segment_revenue_loss(journeys, patients, CONFIG)
        first_fill = ranked[ranked["transition"] == "prescribed_to_first_fill"]
        small = first_fill[first_fill["level"] == "cash_pay"].iloc[0]
        large = first_fill[first_fill["level"] == "commercial"].iloc[0]

        # The small segment loses a far higher share of its patients...
        self.assertAlmostEqual(small["drop_off_rate"], 0.90, places=12)
        self.assertAlmostEqual(large["drop_off_rate"], 0.30, places=12)
        # ...but far less money, and the ranking follows the money.
        self.assertEqual(small["revenue_lost"], 90 * 12 * PER_FILL)
        self.assertEqual(large["revenue_lost"], 600 * 12 * PER_FILL)
        self.assertLess(large.name, small.name)
        self.assertLess(small["rank_by_rate"], large["rank_by_rate"])


class MinimumCellSizeTest(unittest.TestCase):
    def setUp(self):
        self.journeys, self.patients = build(
            [
                ("BIG", {"payer_type": "commercial", "geography": "metro"}, 400, 300),
                ("SMALL", {"payer_type": "cash_pay", "geography": "rural"}, 50, 20),
            ]
        )

    def cells(self, min_cell_size=MIN_CELL_SIZE):
        return two_way_segment_funnel(
            self.journeys,
            self.patients,
            CONFIG,
            "payer_type",
            "geography",
            min_cell_size=min_cell_size,
        )

    def test_default_threshold_is_two_hundred(self):
        self.assertEqual(MIN_CELL_SIZE, 200)

    def test_small_cells_lose_their_rate_but_keep_their_counts(self):
        cells = self.cells().set_index(["payer_type", "geography", "transition"])
        small = cells.loc[("cash_pay", "rural", "prescribed_to_first_fill")]
        self.assertTrue(small["suppressed"])
        self.assertEqual(small["denominator"], 50)
        self.assertEqual(small["converted"], 20)
        # Counts are exact and stay; the unstable estimates go.
        for column in ["conversion_rate", "drop_off_rate", "ci_low", "ci_high"]:
            with self.subTest(column=column):
                self.assertTrue(pd.isna(small[column]))

    def test_large_cells_are_reported(self):
        cells = self.cells().set_index(["payer_type", "geography", "transition"])
        big = cells.loc[("commercial", "metro", "prescribed_to_first_fill")]
        self.assertFalse(big["suppressed"])
        self.assertEqual(big["denominator"], 400)
        self.assertAlmostEqual(big["conversion_rate"], 0.75, places=12)
        self.assertLess(big["ci_low"], 0.75)
        self.assertGreater(big["ci_high"], 0.75)

    def test_threshold_is_configurable(self):
        # Below the cell size, nothing is suppressed.
        permissive = self.cells(min_cell_size=10)
        rate = permissive.set_index(["payer_type", "geography", "transition"]).loc[
            ("cash_pay", "rural", "prescribed_to_first_fill")
        ]
        self.assertFalse(rate["suppressed"])
        self.assertAlmostEqual(rate["conversion_rate"], 0.4, places=12)
        # Above every cell size, everything is.
        strict = self.cells(min_cell_size=10_000)
        self.assertTrue(strict["suppressed"].all())
        self.assertTrue(strict["conversion_rate"].isna().all())
        # Suppression is on the denominator, never on the outcome.
        self.assertEqual(strict["denominator"].sum(), permissive["denominator"].sum())


if __name__ == "__main__":
    unittest.main()
