"""Revenue arithmetic, share columns, and the corrected post-continuation figure.

The cohort below is hand-built so every expected number can be worked out on
paper: 4 patients lost at each successive stage down to 1, plus 4 who reach
continued treatment, 2 of them with room for the whole course before the cutoff
and 2 without.
"""

import unittest

import pandas as pd

from src.funnel import (
    FILLS_COMPLETED_AT_STAGE,
    POST_CONTINUATION_LABEL,
    post_continuation_attrition,
    ranked_revenue_loss,
    remaining_fills,
    stage_funnel,
)
from src.journey import load_config, transitions
from tests.test_journey import Cohort, days

CONFIG = load_config()
START = pd.Timestamp(CONFIG["simulation"]["start_date"])
CUTOFF = pd.Timestamp(CONFIG["simulation"]["observation_cutoff"])
PER_FILL = CONFIG["revenue"]["per_fill"]
FULL_COURSE = CONFIG["revenue"]["expected_fills_full_course"]

# Every patient below sits far enough inside the observation window that all
# four opportunity windows closed, so the denominators are the full at-risk
# counts under either denominator rule and the revenue arithmetic stands alone.
RULE = "window_closed"


def fills_from(first, count, gap=30):
    return [first + days(gap * step) for step in range(count)]


def build_cohort():
    cohort = Cohort()
    for n in range(4):
        cohort.add(f"DIAG{n}", diagnosed=START)
    for n in range(3):
        cohort.add(f"PRESC{n}", diagnosed=START, prescribed=START + days(30))
    for n in range(2):
        cohort.add(
            f"FILL{n}",
            diagnosed=START,
            prescribed=START + days(30),
            fills=fills_from(START + days(50), 1),
        )
    cohort.add(
        "REFILL0",
        diagnosed=START,
        prescribed=START + days(30),
        fills=fills_from(START + days(50), 2),
    )
    # Continued, with room before the cutoff for all 12 fills: 6 taken, 6 forfeited.
    for n in range(2):
        cohort.add(
            f"OBS{n}",
            diagnosed=START,
            prescribed=START + days(30),
            fills=fills_from(START + days(50), 6),
        )
    # Continued, but the course window runs past the cutoff: 4 fills observed.
    for n in range(2):
        cohort.add(
            f"CENS{n}",
            diagnosed=CUTOFF - days(150),
            prescribed=CUTOFF - days(120),
            fills=fills_from(CUTOFF - days(100), 4),
        )
    return cohort.build(RULE)


class FunnelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.journeys = build_cohort().reset_index()
        cls.funnel = stage_funnel(cls.journeys, CONFIG)
        cls.post = post_continuation_attrition(cls.journeys, CONFIG)

    def test_remaining_fills_follow_decisions_section_3(self):
        # A patient lost before the first fill has completed nothing, so both
        # early transitions carry the whole course.
        self.assertEqual(remaining_fills("diagnosed", CONFIG), 12)
        self.assertEqual(remaining_fills("prescribed", CONFIG), 12)
        self.assertEqual(remaining_fills("first_fill", CONFIG), 11)
        self.assertEqual(remaining_fills("refill", CONFIG), 10)
        # The per-patient rupee values worked through in DECISIONS.md section 3.
        self.assertEqual(remaining_fills("diagnosed", CONFIG) * PER_FILL, 28_800)
        self.assertEqual(remaining_fills("prescribed", CONFIG) * PER_FILL, 28_800)
        self.assertEqual(remaining_fills("first_fill", CONFIG) * PER_FILL, 26_400)
        self.assertEqual(remaining_fills("refill", CONFIG) * PER_FILL, 24_000)
        # Continued treatment's fill count is the config's own threshold.
        stages = {stage["name"]: stage for stage in CONFIG["stages"]}
        self.assertEqual(
            FILLS_COMPLETED_AT_STAGE["continued"], stages["continued"]["min_consecutive_fills"]
        )

    def test_denominator_splits_into_converted_and_dropped(self):
        expected = {
            "diagnosed_to_prescribed": (14, 10, 4),
            "prescribed_to_first_fill": (10, 7, 3),
            "first_fill_to_refill": (7, 5, 2),
            "refill_to_continued": (5, 4, 1),
        }
        for name, (denominator, converted, dropped) in expected.items():
            row = self.funnel.loc[name]
            with self.subTest(transition=name):
                self.assertEqual(row["denominator"], denominator)
                self.assertEqual(row["converted"], converted)
                self.assertEqual(row["dropped_off"], dropped)
                self.assertEqual(row["denominator"], row["converted"] + row["dropped_off"])
                self.assertAlmostEqual(row["conversion_rate"], converted / denominator, places=12)
                self.assertAlmostEqual(row["drop_off_rate"], dropped / denominator, places=12)
                self.assertAlmostEqual(
                    row["conversion_rate"] + row["drop_off_rate"], 1.0, places=12
                )

    def test_revenue_lost_is_drop_offs_times_remaining_lifetime_value(self):
        expected = {
            "diagnosed_to_prescribed": 4 * 12 * PER_FILL,
            "prescribed_to_first_fill": 3 * 12 * PER_FILL,
            "first_fill_to_refill": 2 * 11 * PER_FILL,
            "refill_to_continued": 1 * 10 * PER_FILL,
        }
        self.assertEqual(expected["diagnosed_to_prescribed"], 115_200)
        self.assertEqual(expected["prescribed_to_first_fill"], 86_400)
        self.assertEqual(expected["first_fill_to_refill"], 52_800)
        self.assertEqual(expected["refill_to_continued"], 24_000)
        for name, revenue in expected.items():
            with self.subTest(transition=name):
                self.assertEqual(self.funnel.loc[name, "revenue_lost"], revenue)
        self.assertEqual(self.funnel["revenue_lost"].sum(), 278_400)
        # Restated independently of the table, straight from the definition.
        for transition in transitions(CONFIG):
            row = self.funnel.loc[transition.name]
            self.assertEqual(
                row["revenue_lost"],
                row["dropped_off"]
                * (FULL_COURSE - FILLS_COMPLETED_AT_STAGE[transition.origin])
                * PER_FILL,
            )

    def test_share_columns_are_shares_of_total_funnel_loss(self):
        self.assertAlmostEqual(self.funnel["loss_share"].sum(), 1.0, places=12)
        self.assertAlmostEqual(self.funnel["revenue_share"].sum(), 1.0, places=12)
        for name, share in [
            ("diagnosed_to_prescribed", 0.4),
            ("prescribed_to_first_fill", 0.3),
            ("first_fill_to_refill", 0.2),
            ("refill_to_continued", 0.1),
        ]:
            with self.subTest(transition=name):
                self.assertAlmostEqual(self.funnel.loc[name, "loss_share"], share, places=12)
        for name, revenue in [
            ("diagnosed_to_prescribed", 115_200),
            ("prescribed_to_first_fill", 86_400),
            ("first_fill_to_refill", 52_800),
            ("refill_to_continued", 24_000),
        ]:
            with self.subTest(transition=name):
                self.assertAlmostEqual(
                    self.funnel.loc[name, "revenue_share"], revenue / 278_400, places=12
                )
        # Revenue share is not patient share. Losses before the first fill
        # carry 12 remaining fills each, so their revenue share sits above
        # their patient share, and the late stages' below.
        early = self.funnel.loc["diagnosed_to_prescribed"]
        late = self.funnel.loc["refill_to_continued"]
        self.assertGreater(early["revenue_share"], early["loss_share"])
        self.assertLess(late["revenue_share"], late["loss_share"])

    def test_post_continuation_naive_corrected_and_artifact(self):
        post = self.post
        self.assertEqual(post["continued_patients"], 4)
        self.assertEqual(post["full_course_observable"], 2)
        self.assertEqual(post["censored_patients"], 2)
        self.assertAlmostEqual(post["mean_fills_observable"], 6.0, places=12)
        self.assertAlmostEqual(post["mean_fills_censored"], 4.0, places=12)

        # Naive: all four measured against the full course.
        self.assertAlmostEqual(post["revenue_forfeited_naive"], 2 * 6 * PER_FILL + 2 * 8 * PER_FILL)
        self.assertAlmostEqual(post["revenue_forfeited_naive"], 67_200)
        # Corrected: the observable patients' per-patient forfeiture, applied to all.
        self.assertAlmostEqual(post["forfeited_per_observable_patient"], 6 * PER_FILL)
        self.assertAlmostEqual(post["forfeited_per_observable_patient"], 14_400)
        self.assertAlmostEqual(post["revenue_forfeited_corrected"], 4 * 14_400)
        self.assertAlmostEqual(post["revenue_forfeited_corrected"], 57_600)
        # Artifact: the remainder, which is unexpired observation time.
        self.assertAlmostEqual(post["censoring_artifact"], 9_600)
        self.assertAlmostEqual(
            post["revenue_forfeited_naive"] - post["revenue_forfeited_corrected"],
            post["censoring_artifact"],
        )

    def test_artifact_equals_the_fill_gap_across_censored_patients(self):
        # The identity the correction rests on: the whole difference between
        # naive and corrected is the censored patients' shortfall in fills.
        post = self.post
        self.assertAlmostEqual(
            post["censoring_artifact"],
            (post["mean_fills_observable"] - post["mean_fills_censored"])
            * PER_FILL
            * post["censored_patients"],
            places=6,
        )

    def test_correction_is_undefined_without_an_observable_patient(self):
        cohort = Cohort()
        for n in range(2):
            cohort.add(
                f"CENS{n}",
                diagnosed=CUTOFF - days(150),
                prescribed=CUTOFF - days(120),
                fills=fills_from(CUTOFF - days(100), 4),
            )
        post = post_continuation_attrition(cohort.build(RULE).reset_index(), CONFIG)
        self.assertEqual(post["full_course_observable"], 0)
        self.assertAlmostEqual(post["revenue_forfeited_naive"], 2 * 8 * PER_FILL)
        # No uncontaminated rate to project from, so no correction is claimed.
        self.assertTrue(pd.isna(post["revenue_forfeited_corrected"]))
        self.assertTrue(pd.isna(post["censoring_artifact"]))

    def test_ranked_loss_orders_every_source(self):
        ranked = ranked_revenue_loss(self.funnel, self.post)
        self.assertEqual(len(ranked), 5)
        self.assertEqual(
            list(ranked["source"]),
            [
                "diagnosed_to_prescribed",
                "prescribed_to_first_fill",
                POST_CONTINUATION_LABEL,
                "first_fill_to_refill",
                "refill_to_continued",
            ],
        )
        self.assertEqual(list(ranked.index), [1, 2, 3, 4, 5])
        self.assertTrue(ranked["revenue_lost"].is_monotonic_decreasing)
        # Post-continuation is ranked on the corrected figure, not the naive one.
        post_row = ranked[ranked["source"] == POST_CONTINUATION_LABEL].iloc[0]
        self.assertAlmostEqual(post_row["revenue_lost"], 57_600)
        self.assertEqual(post_row["patients"], 4)
        self.assertAlmostEqual(ranked["revenue_lost"].sum(), 278_400 + 57_600)
        self.assertAlmostEqual(ranked["share"].sum(), 1.0, places=12)
        self.assertAlmostEqual(post_row["share"], 57_600 / 336_000, places=12)


if __name__ == "__main__":
    unittest.main()
