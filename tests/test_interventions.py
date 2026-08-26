"""Addressable population, the cost-on-all-targeted rule, the cap, and sensitivity.

Cohorts are built with exact conversion counts so every recovered patient,
rupee and rank below can be worked out on paper.
"""

import copy
import unittest

import pandas as pd

from src.interventions import (
    addressable,
    expected_realised_fills,
    complementary_pairs,
    intervention_return,
    matches_segment,
    observed_conversion_rates,
    rank_interventions,
    sensitivity,
    target_transition,
)
from src.funnel import FILLS_COMPLETED_AT_STAGE
from src.journey import load_config, transitions
from src.segmentation import with_segments
from tests.test_journey import Cohort, days

CONFIG = load_config()
START = pd.Timestamp(CONFIG["simulation"]["start_date"])
PER_FILL = CONFIG["revenue"]["per_fill"]
RULE = "window_closed"
POST_CONTINUATION = CONFIG["base_transition_probabilities"]["post_continuation_per_fill"]
FULL_COURSE = CONFIG["revenue"]["expected_fills_full_course"]
# Fills past the third, each taken at p, up to the 12-fill course: sum(p^k, k=1..9).
BEYOND = sum(POST_CONTINUATION**step for step in range(1, 10))
DEFAULTS = {
    "payer_type": "commercial",
    "age_band": "40-59",
    "prescriber_specialty": "cardiologist",
    "geography": "metro",
}


def fill_dates(count):
    return [START + days(50 + 30 * step) for step in range(count)]


def build_layered(layers):
    """Cohort from (prefix, overrides, total, converters, depth) tuples.

    Mixing depths gives every transition a live denominator, which the expected
    fills recursion needs: a stage with no observed conversion has no rate.
    """
    cohort = Cohort()
    rows = []
    for prefix, overrides, total, converters, depth in layers:
        for index in range(total):
            patient_id = f"{prefix}{index}"
            converted = index < converters
            cohort.add(
                patient_id,
                diagnosed=START,
                prescribed=START + days(30),
                fills=fill_dates(depth if converted else depth - 1),
            )
            rows.append({"patient_id": patient_id, **DEFAULTS, **overrides})
    return cohort.build(RULE).reset_index(), pd.DataFrame(rows)


def build(groups, depth=1):
    """Cohort from (prefix, overrides, total, converters) at one journey depth."""
    cohort = Cohort()
    rows = []
    for prefix, overrides, total, converters in groups:
        for index in range(total):
            patient_id = f"{prefix}{index}"
            converted = index < converters
            cohort.add(
                patient_id,
                diagnosed=START,
                prescribed=START + days(30),
                fills=fill_dates(depth if converted else depth - 1),
            )
            rows.append({"patient_id": patient_id, **DEFAULTS, **overrides})
    journeys = cohort.build(RULE).reset_index()
    return journeys, pd.DataFrame(rows)


# A recovery into first_fill is worth exactly 4 fills under these, so the money
# in the tests below can be worked out on paper.
FIXED_FILLS = {"diagnosed": 1.0, "prescribed": 2.0, "first_fill": 4.0, "refill": 6.0, "continued": 8.0}


def one_intervention(**overrides):
    intervention = {
        "name": "test",
        "target_stage": "first_fill",
        "cost_per_patient": 100,
        "expected_lift": 0.20,
        "eligible_segment": {},
    }
    intervention.update(overrides)
    return intervention


class AddressablePopulationTest(unittest.TestCase):
    def setUp(self):
        # 400 cash-pay, 100 converting; 600 commercial, 480 converting.
        self.journeys, self.patients = build(
            [
                ("CASH", {"payer_type": "cash_pay"}, 400, 100),
                ("COMM", {"payer_type": "commercial"}, 600, 480),
            ]
        )
        self.cohort = with_segments(self.journeys, self.patients, CONFIG)

    def test_target_stage_names_the_destination(self):
        for stage, expected in [
            ("prescribed", "diagnosed_to_prescribed"),
            ("first_fill", "prescribed_to_first_fill"),
            ("refill", "first_fill_to_refill"),
            ("continued", "refill_to_continued"),
        ]:
            with self.subTest(stage=stage):
                self.assertEqual(
                    target_transition(one_intervention(target_stage=stage), CONFIG).name,
                    expected,
                )
        with self.assertRaises(ValueError):
            target_transition(one_intervention(target_stage="diagnosed"), CONFIG)

    def test_addressable_is_in_denominator_in_segment_and_unconverted(self):
        intervention = one_intervention(eligible_segment={"payer_type": ["cash_pay"]})
        transition, denominator, converted, targeted = addressable(
            self.cohort, CONFIG, intervention
        )
        self.assertEqual(transition.name, "prescribed_to_first_fill")
        # The segment restricts the denominator, not just the targeting.
        self.assertEqual(int(denominator.sum()), 400)
        self.assertEqual(int(converted.sum()), 100)
        self.assertEqual(int(targeted.sum()), 300)
        # Targeted and converted partition the denominator exactly.
        self.assertTrue(((converted | targeted) == denominator).all())
        self.assertEqual(int((converted & targeted).sum()), 0)

    def test_converted_patients_are_never_addressable(self):
        intervention = one_intervention()
        _, denominator, converted, targeted = addressable(self.cohort, CONFIG, intervention)
        self.assertEqual(int(denominator.sum()), 1000)
        self.assertEqual(int(converted.sum()), 580)
        self.assertEqual(int(targeted.sum()), 420)
        self.assertEqual(int((targeted & converted).sum()), 0)

    def test_empty_eligible_segment_matches_everyone(self):
        self.assertTrue(matches_segment(self.cohort, {}).all())
        self.assertTrue(matches_segment(self.cohort, None).all())
        # Several columns are combined with AND.
        both = matches_segment(
            self.cohort, {"payer_type": ["cash_pay"], "geography": ["metro"]}
        )
        self.assertEqual(int(both.sum()), 400)
        neither = matches_segment(
            self.cohort, {"payer_type": ["cash_pay"], "geography": ["rural"]}
        )
        self.assertEqual(int(neither.sum()), 0)


class CostAndRecoveryTest(unittest.TestCase):
    def setUp(self):
        # 1,000 patients, 500 converting: a current rate of exactly 0.50.
        self.journeys, self.patients = build([("P", {}, 1000, 500)])
        self.cohort = with_segments(self.journeys, self.patients, CONFIG)

    def test_lift_is_relative_and_recovery_follows_the_rate_change(self):
        result = intervention_return(
            self.cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )
        self.assertAlmostEqual(result["current_rate"], 0.50, places=12)
        # 0.50 * 1.20 = 0.60, a relative increase, not 0.50 + 0.20.
        self.assertAlmostEqual(result["lifted_rate"], 0.60, places=12)
        self.assertFalse(result["capped"])
        self.assertAlmostEqual(result["patients_recovered"], 100.0, places=10)
        # 12 remaining fills at the prescribed stage.
        # Potential values the whole course still ahead; expected values only
        # the fills a patient recovered into first_fill is likely to complete.
        self.assertEqual(result["remaining_fills"], 12)
        self.assertAlmostEqual(result["revenue_potential"], 100.0 * 12 * PER_FILL, places=6)
        self.assertAlmostEqual(result["expected_fills"], 4.0, places=12)
        self.assertAlmostEqual(result["revenue_recovered"], 100.0 * 4 * PER_FILL, places=6)
        self.assertAlmostEqual(result["correction_ratio"], 3.0, places=12)

    def test_cost_is_charged_on_everyone_targeted(self):
        result = intervention_return(
            self.cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )
        self.assertEqual(result["patients_targeted"], 500)
        self.assertAlmostEqual(result["patients_recovered"], 100.0, places=10)
        # 500 targeted at 100 each, not the 100 who were recovered.
        self.assertEqual(result["cost"], 500 * 100)
        self.assertNotEqual(result["cost"], round(result["patients_recovered"]) * 100)
        self.assertEqual(result["cost"], result["patients_targeted"] * 100)
        # Charging only the recovered would understate cost five-fold here.
        self.assertAlmostEqual(
            result["cost"] / (result["patients_recovered"] * 100), 5.0, places=10
        )

    def test_net_return_and_return_per_rupee(self):
        result = intervention_return(
            self.cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )
        revenue, cost = result["revenue_recovered"], result["cost"]
        self.assertAlmostEqual(result["net_return"], revenue - cost, places=6)
        self.assertAlmostEqual(result["return_per_rupee"], revenue / cost, places=12)
        # 100 recovered x 4 fills x 2400, less 500 targeted x 100.
        self.assertAlmostEqual(result["net_return"], 960_000 - 50_000, places=6)
        # The potential valuation is the one the funnel would have used.
        self.assertAlmostEqual(result["net_return_potential"], 2_880_000 - 50_000, places=6)


class CapTest(unittest.TestCase):
    def test_lifted_rate_is_capped_at_one(self):
        # 900 of 1,000 convert, so 0.90 * 1.20 would be 1.08.
        journeys, patients = build([("P", {}, 1000, 900)])
        cohort = with_segments(journeys, patients, CONFIG)
        result = intervention_return(
            cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )

        self.assertAlmostEqual(result["current_rate"], 0.90, places=12)
        self.assertEqual(result["lifted_rate"], 1.0)
        self.assertTrue(result["capped"])
        # Only the 100 who were missing can be recovered, not 0.90 * 0.20 * 1000.
        self.assertAlmostEqual(result["patients_recovered"], 100.0, places=10)
        self.assertLess(result["patients_recovered"], 0.90 * 0.20 * 1000)
        # Recovery can never exceed the patients who were targeted.
        self.assertLessEqual(result["patients_recovered"], result["patients_targeted"])

    def test_uncapped_case_is_left_alone(self):
        journeys, patients = build([("P", {}, 1000, 800)])
        cohort = with_segments(journeys, patients, CONFIG)
        result = intervention_return(
            cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )
        self.assertAlmostEqual(result["lifted_rate"], 0.96, places=12)
        self.assertFalse(result["capped"])


class SensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Layered depths so all four transitions carry a rate: without one the
        # expected fills recursion is undefined and every return is NaN.
        cls.journeys, cls.patients = build_layered(
            [
                ("CASH", {"payer_type": "cash_pay"}, 900, 200, 1),
                ("MID", {"payer_type": "commercial"}, 600, 400, 2),
                ("DEEP", {"payer_type": "commercial"}, 600, 300, 3),
            ]
        )
        cls.config = copy.deepcopy(CONFIG)
        cls.config["interventions"] = [
            one_intervention(name="broad", cost_per_patient=100, expected_lift=0.20),
            one_intervention(name="cheap", cost_per_patient=10, expected_lift=0.10),
            one_intervention(
                name="narrow",
                cost_per_patient=500,
                expected_lift=0.30,
                eligible_segment={"payer_type": ["cash_pay"]},
            ),
        ]

    def test_multiplier_scales_the_lift_not_the_cost(self):
        cohort = with_segments(self.journeys, self.patients, CONFIG)
        base = intervention_return(
            cohort, CONFIG, one_intervention(), expected_fills=FIXED_FILLS
        )
        doubled = intervention_return(
            cohort, CONFIG, one_intervention(), 2.0, expected_fills=FIXED_FILLS
        )

        # 1,400 of 2,100 reach a first fill in this fixture.
        self.assertAlmostEqual(base["current_rate"], 1400 / 2100, places=12)
        self.assertAlmostEqual(doubled["lift"], 2 * base["lift"], places=12)
        # Uncapped, recovery and revenue scale with the multiplier.
        self.assertFalse(base["capped"])
        self.assertFalse(doubled["capped"])
        self.assertAlmostEqual(
            doubled["patients_recovered"], 2 * base["patients_recovered"], places=8
        )
        self.assertAlmostEqual(
            doubled["revenue_recovered"], 2 * base["revenue_recovered"], places=4
        )
        # Cost is unaffected, which is why the two rankings behave differently.
        self.assertEqual(doubled["cost"], base["cost"])
        self.assertAlmostEqual(
            doubled["return_per_rupee"], 2 * base["return_per_rupee"], places=8
        )

    def test_sensitivity_recomputes_the_whole_ranking(self):
        multipliers = (0.5, 1.0, 2.0)
        table = sensitivity(self.journeys, self.patients, self.config, multipliers)
        self.assertEqual(len(table), len(self.config["interventions"]))

        for multiplier in multipliers:
            run = rank_interventions(
                self.journeys, self.patients, self.config, multiplier
            ).set_index("name")
            with self.subTest(multiplier=multiplier):
                for name, row in table.iterrows():
                    self.assertAlmostEqual(
                        row[f"net_return_{multiplier}x"], run.loc[name, "net_return"], places=6
                    )
                    self.assertEqual(
                        row[f"rank_net_{multiplier}x"], run.loc[name, "rank_by_net_return"]
                    )
                    self.assertAlmostEqual(
                        row[f"return_per_rupee_{multiplier}x"],
                        run.loc[name, "return_per_rupee"],
                        places=8,
                    )

    def test_a_common_multiplier_cannot_move_the_per_rupee_ordering(self):
        # Every return per rupee scales by the same factor while cost holds, so
        # the ordering is arithmetically fixed unless a cap binds.
        multipliers = (0.5, 1.0, 2.0)
        table = sensitivity(self.journeys, self.patients, self.config, multipliers)
        self.assertFalse(table[[f"capped_{m}x" for m in multipliers]].to_numpy().any())
        self.assertTrue(table["per_rupee_rank_stable"].all())
        for name, row in table.iterrows():
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    row["return_per_rupee_2.0x"], 2 * row["return_per_rupee_1.0x"], places=8
                )
                self.assertAlmostEqual(
                    row["return_per_rupee_0.5x"], 0.5 * row["return_per_rupee_1.0x"], places=8
                )

    def test_net_return_ordering_can_move_when_lifts_change(self):
        # Cost is fixed while revenue scales, so a cheap intervention can
        # overtake an expensive one as the lift shrinks. A low conversion rate
        # keeps the 1.0 cap well clear of this range, which would otherwise
        # flatten both recoveries to the same number and hide the crossover.
        journeys, patients = build_layered(
            [
                ("SHALLOW", {}, 1800, 100, 1),
                ("MID", {}, 200, 100, 2),
                ("DEEP", {}, 100, 50, 3),
            ]
        )
        config = copy.deepcopy(CONFIG)
        config["interventions"] = [
            one_intervention(name="expensive", cost_per_patient=200, expected_lift=0.30),
            one_intervention(name="cheap", cost_per_patient=10, expected_lift=0.20),
        ]
        multipliers = (0.5, 3.0)
        table = sensitivity(journeys, patients, config, multipliers)
        self.assertFalse(table[[f"capped_{m}x" for m in multipliers]].to_numpy().any())

        self.assertFalse(table["net_rank_stable"].all())
        self.assertEqual(table.loc["cheap", "rank_net_0.5x"], 1)
        self.assertEqual(table.loc["expensive", "rank_net_3.0x"], 1)
        # The per-rupee ordering is untouched by the same change.
        self.assertTrue(table["per_rupee_rank_stable"].all())

    def test_an_intervention_with_no_addressable_population_is_unranked(self):
        # This fixture has no general physicians, so prescriber_detailing has
        # an empty denominator. It must come back unranked, not last.
        table = rank_interventions(self.journeys, self.patients, CONFIG).set_index("name")
        row = table.loc["prescriber_detailing"]
        self.assertEqual(row["denominator"], 0)
        self.assertEqual(row["patients_targeted"], 0)
        self.assertTrue(pd.isna(row["current_rate"]))
        self.assertTrue(pd.isna(row["net_return"]))
        self.assertTrue(pd.isna(row["rank_by_net_return"]))
        self.assertFalse(row["capped"])
        # And the sensitivity run survives it rather than failing on the cast.
        sensitivity_table = sensitivity(self.journeys, self.patients, CONFIG, (0.5, 1.0))
        self.assertTrue(pd.isna(sensitivity_table.loc["prescriber_detailing", "rank_net_1.0x"]))
        # An intervention with no rank has no rank to change.
        self.assertTrue(sensitivity_table.loc["prescriber_detailing", "net_rank_stable"])


class ExpectedFillsTest(unittest.TestCase):
    """The recursion in DECISIONS terms: what a recovered patient actually fills."""

    def rates(self, prescribed, first_fill, refill, continued):
        return {
            "diagnosed_to_prescribed": prescribed,
            "prescribed_to_first_fill": first_fill,
            "first_fill_to_refill": refill,
            "refill_to_continued": continued,
        }

    def test_continued_is_three_fills_plus_the_geometric_tail(self):
        expected = expected_realised_fills(CONFIG, self.rates(1, 1, 1, 1))
        self.assertAlmostEqual(BEYOND, 2.774745941162109, places=12)
        self.assertAlmostEqual(expected["continued"], 3 + BEYOND, places=12)
        # The tail is bounded by the course, never beyond it.
        self.assertLess(expected["continued"], FULL_COURSE)

    def test_recursion_discounts_by_every_transition_ahead(self):
        rates = self.rates(0.8, 0.6, 0.7, 0.5)
        expected = expected_realised_fills(CONFIG, rates)

        continued = 3 + BEYOND
        refill = 2 + 0.5 * (continued - 2)
        first_fill = 1 + 0.7 * (refill - 1)
        prescribed = 0 + 0.6 * (first_fill - 0)
        diagnosed = 0 + 0.8 * (prescribed - 0)
        for stage, value in [
            ("continued", continued),
            ("refill", refill),
            ("first_fill", first_fill),
            ("prescribed", prescribed),
            ("diagnosed", diagnosed),
        ]:
            with self.subTest(stage=stage):
                self.assertAlmostEqual(expected[stage], value, places=12)
        # Each stage earlier in the journey expects strictly fewer fills.
        self.assertLess(expected["diagnosed"], expected["prescribed"])
        self.assertLess(expected["prescribed"], expected["first_fill"])
        self.assertLess(expected["first_fill"], expected["refill"])
        self.assertLess(expected["refill"], expected["continued"])

    def test_perfect_and_zero_conversion_bound_the_recursion(self):
        perfect = expected_realised_fills(CONFIG, self.rates(1, 1, 1, 1))
        # With certain progression every stage reaches continued treatment.
        for stage in ["diagnosed", "prescribed", "first_fill", "refill"]:
            with self.subTest(stage=stage):
                self.assertAlmostEqual(perfect[stage], perfect["continued"], places=12)

        none = expected_realised_fills(CONFIG, self.rates(0, 0, 0, 0))
        # With no progression a patient completes only the fills already banked.
        self.assertAlmostEqual(none["diagnosed"], 0.0, places=12)
        self.assertAlmostEqual(none["prescribed"], 0.0, places=12)
        self.assertAlmostEqual(none["first_fill"], 1.0, places=12)
        self.assertAlmostEqual(none["refill"], 2.0, places=12)

    def test_the_correction_is_larger_for_earlier_stages(self):
        # A patient recovered early has more attrition still ahead, so the gap
        # between potential and expected realised value is wider there.
        expected = expected_realised_fills(CONFIG, self.rates(0.82, 0.63, 0.73, 0.72))
        ratios = {}
        for origin, destination in [
            ("diagnosed", "prescribed"),
            ("prescribed", "first_fill"),
            ("first_fill", "refill"),
            ("refill", "continued"),
        ]:
            completed = FILLS_COMPLETED_AT_STAGE[origin]
            potential = FULL_COURSE - completed
            realised = expected[destination] - completed
            ratios[destination] = potential / realised
            with self.subTest(destination=destination):
                # Expected realised can never exceed potential.
                self.assertLessEqual(realised, potential)
                self.assertGreater(ratios[destination], 1.0)

        order = ["prescribed", "first_fill", "refill", "continued"]
        for earlier, later in zip(order, order[1:]):
            with self.subTest(pair=(earlier, later)):
                self.assertGreater(ratios[earlier], ratios[later])
        # The figures the ranking turns on.
        self.assertAlmostEqual(ratios["prescribed"], 5.12, places=1)
        self.assertAlmostEqual(ratios["continued"], 2.65, places=1)

    def test_rates_are_read_from_the_observed_funnel(self):
        journeys, patients = build_layered(
            [
                ("A", {}, 600, 400, 1),
                ("B", {}, 400, 200, 2),
                ("C", {}, 400, 200, 3),
            ]
        )
        cohort = with_segments(journeys, patients, CONFIG)
        rates = observed_conversion_rates(cohort, CONFIG)
        self.assertEqual(set(rates), {t.name for t in transitions(CONFIG)})
        expected = expected_realised_fills(CONFIG, rates)
        for stage, value in expected.items():
            with self.subTest(stage=stage):
                self.assertFalse(pd.isna(value))
                self.assertLessEqual(value, FULL_COURSE)


class ComplementarityTest(unittest.TestCase):
    def test_disjoint_age_bands_are_complementary(self):
        journeys, patients = build(
            [
                ("YOUNG", {"age_band": "40-59"}, 600, 400),
                ("OLD", {"age_band": "75+"}, 600, 300),
            ],
            depth=3,
        )
        pairs = complementary_pairs(journeys, patients, CONFIG).set_index(["first", "second"])
        row = pairs.loc[("nurse_education_call", "digital_adherence_app")]
        self.assertEqual(row["transition"], "refill_to_continued")
        self.assertEqual(row["overlap"], 0)
        self.assertTrue(row["complementary"])
        # Between them they cover every targeted patient at that transition.
        self.assertEqual(row["combined"], 200 + 300)

    def test_overlapping_interventions_are_not_marked_complementary(self):
        journeys, patients = build([("P", {}, 400, 200)], depth=1)
        config = copy.deepcopy(CONFIG)
        config["interventions"] = [
            one_intervention(name="broad"),
            one_intervention(name="also_broad", cost_per_patient=50),
        ]
        pairs = complementary_pairs(journeys, patients, config)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs.iloc[0]["complementary"])
        self.assertEqual(pairs.iloc[0]["overlap"], 200)


if __name__ == "__main__":
    unittest.main()
