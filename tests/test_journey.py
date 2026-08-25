"""Stage assignment and censoring, checked against DECISIONS.md sections 1 and 2.

Dates are expressed relative to the configured start date and observation
cutoff, so these cases stay meaningful if the observation window moves. Every
case pins its denominator rule explicitly rather than inheriting the one in
config.analysis, so changing that setting cannot quietly change what is tested.
"""

import unittest

import pandas as pd

from src.journey import (
    DENOMINATOR_RULES,
    build_journeys,
    load_config,
    resolve_denominator_rule,
    transitions,
)

CONFIG = load_config()
START = pd.Timestamp(CONFIG["simulation"]["start_date"])
CUTOFF = pd.Timestamp(CONFIG["simulation"]["observation_cutoff"])
STAGES = {stage["name"]: stage for stage in CONFIG["stages"]}


def days(n):
    return pd.Timedelta(days=n)


class Cohort:
    """Builds the two input frames journey.py reads, one patient at a time."""

    def __init__(self):
        self.patients = []
        self.events = []

    def add(self, patient_id, diagnosed, prescribed=None, fills=()):
        self.patients.append({"patient_id": patient_id, "cohort_entry_date": diagnosed})
        self._event(patient_id, "diagnosed", diagnosed, None)
        if prescribed is not None:
            self._event(patient_id, "prescribed", prescribed, None)
        for sequence, fill_date in enumerate(fills, start=1):
            self._event(patient_id, "fill", fill_date, sequence)

    def _event(self, patient_id, event_type, event_date, fill_sequence):
        self.events.append(
            {
                "patient_id": patient_id,
                "event_type": event_type,
                "event_date": event_date,
                "fill_sequence": fill_sequence,
            }
        )

    def build(self, denominator_rule):
        journeys = build_journeys(
            pd.DataFrame(self.patients),
            pd.DataFrame(self.events),
            CONFIG,
            denominator_rule=denominator_rule,
        )
        return journeys.set_index("patient_id")


class JourneyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohort = Cohort()
        # Completes every stage: 30 days to the script, 20 to the first fill,
        # then two 30-day gaps.
        cohort.add(
            "FULL",
            diagnosed=START,
            prescribed=START + days(30),
            fills=[START + days(50), START + days(80), START + days(110)],
        )
        # Prescribed early in the window and never filled: a true drop-off.
        cohort.add("DROP", diagnosed=START, prescribed=START + days(30))
        # Entered 11 days before the cutoff. Converted to prescribed with one
        # day to spare, then ran out of observation time.
        cohort.add("CENSORED", diagnosed=CUTOFF - days(11), prescribed=CUTOFF - days(1))
        # Three fills, but the third lands 50 days after the second.
        cohort.add(
            "GAP",
            diagnosed=START,
            prescribed=START + days(30),
            fills=[START + days(50), START + days(80), START + days(130)],
        )
        cls.strict = cohort.build("window_closed")
        cls.inclusive = cohort.build("converter_inclusive")

    def test_full_journey_reaches_continued(self):
        row = self.strict.loc["FULL"]
        self.assertEqual(row["furthest_stage"], "continued")
        self.assertEqual(row["continued_date"], START + days(110))
        self.assertEqual(row["total_fills"], 3)
        self.assertEqual(row["days_diagnosed_to_prescribed"], 30)
        self.assertEqual(row["days_prescribed_to_first_fill"], 20)
        self.assertEqual(row["days_first_fill_to_refill"], 30)
        self.assertEqual(row["days_refill_to_continued"], 30)
        # Every window closed long before the cutoff, so both rules agree.
        for transition in transitions(CONFIG):
            self.assertTrue(row[f"eligible_{transition.name}"], transition.name)
            self.assertTrue(self.inclusive.loc["FULL", f"eligible_{transition.name}"])

    def test_drop_at_first_fill_is_a_drop_off_not_a_censor(self):
        row = self.strict.loc["DROP"]
        self.assertEqual(row["furthest_stage"], "prescribed")
        self.assertTrue(pd.isna(row["first_fill_date"]))
        self.assertEqual(row["total_fills"], 0)
        # Had the full 30-day window and did not fill: counts in the denominator.
        self.assertLessEqual(
            row["prescribed_date"] + days(STAGES["first_fill"]["max_days_from_previous"]),
            CUTOFF,
        )
        self.assertTrue(row["eligible_prescribed_to_first_fill"])
        # Never reached the first fill, so not at risk of the later transitions.
        self.assertFalse(row["eligible_first_fill_to_refill"])
        self.assertFalse(row["eligible_refill_to_continued"])

    def test_patient_near_cutoff_is_censored_not_a_drop_off(self):
        row = self.strict.loc["CENSORED"]
        self.assertEqual(row["furthest_stage"], "prescribed")
        # The 30-day first-fill window runs past the cutoff, so this patient is
        # censored at that transition rather than counted as a drop-off. They
        # did not convert, so no rule can rescue them.
        self.assertGreater(
            row["prescribed_date"] + days(STAGES["first_fill"]["max_days_from_previous"]),
            CUTOFF,
        )
        self.assertFalse(row["eligible_prescribed_to_first_fill"])
        self.assertFalse(self.inclusive.loc["CENSORED", "eligible_prescribed_to_first_fill"])

    def test_the_two_rules_split_on_a_converter_with_an_open_window(self):
        # This patient's 60-day prescribing window runs past the cutoff, but
        # they converted inside it. That is the only case the rules disagree on.
        row = self.strict.loc["CENSORED"]
        self.assertGreater(
            row["diagnosed_date"] + days(STAGES["prescribed"]["max_days_from_previous"]),
            CUTOFF,
        )
        self.assertTrue(pd.notna(row["prescribed_date"]))
        self.assertFalse(row["eligible_diagnosed_to_prescribed"])
        self.assertTrue(self.inclusive.loc["CENSORED", "eligible_diagnosed_to_prescribed"])

        # Everyone else's windows closed, so the two frames agree on them.
        columns = [f"eligible_{transition.name}" for transition in transitions(CONFIG)]
        others = ["FULL", "DROP", "GAP"]
        pd.testing.assert_frame_equal(
            self.strict.loc[others, columns], self.inclusive.loc[others, columns]
        )

    def test_three_fills_with_a_fifty_day_gap_is_not_continued(self):
        row = self.strict.loc["GAP"]
        self.assertEqual(row["total_fills"], 3)
        self.assertEqual(row["furthest_stage"], "refill")
        self.assertTrue(pd.isna(row["continued_date"]))
        self.assertTrue(pd.isna(row["days_refill_to_continued"]))
        # The third fill exists but sits 50 days past the refill, over the
        # 45-day gap continued treatment allows.
        self.assertGreater(days(50), days(STAGES["continued"]["max_gap_days"]))
        # The 90-day assessment window closed well before the cutoff, so this
        # is a genuine drop-off at refill -> continued, not a censor.
        self.assertLessEqual(
            row["first_fill_date"] + days(STAGES["continued"]["assessment_window_days"]),
            CUTOFF,
        )
        self.assertTrue(row["eligible_refill_to_continued"])

    def test_denominator_accounting_holds_under_both_rules(self):
        for rule, journeys in [("window_closed", self.strict), ("converter_inclusive", self.inclusive)]:
            for transition in transitions(CONFIG):
                at_risk = journeys[f"{transition.origin}_date"].notna()
                converted = journeys[f"{transition.destination}_date"].notna()
                eligible = journeys[f"eligible_{transition.name}"]
                with self.subTest(rule=rule, transition=transition.name):
                    # Nobody is eligible without being at risk, and the cohort
                    # splits cleanly into eligible and censored.
                    self.assertTrue((eligible <= at_risk).all())
                    censored = (at_risk & ~eligible).sum()
                    self.assertEqual(at_risk.sum(), eligible.sum() + censored)
                    drop_offs = (eligible & ~converted).sum()
                    self.assertEqual(eligible.sum(), (eligible & converted).sum() + drop_offs)
                    if rule == "converter_inclusive":
                        # This rule's defining property: no conversion is ever
                        # left out of the denominator.
                        self.assertTrue((converted <= eligible).all())

    def test_denominator_rule_resolution(self):
        self.assertEqual(resolve_denominator_rule(CONFIG), CONFIG["analysis"]["denominator_rule"])
        self.assertIn(CONFIG["analysis"]["denominator_rule"], DENOMINATOR_RULES)
        # An explicit rule overrides the configured default.
        self.assertEqual(resolve_denominator_rule(CONFIG, "window_closed"), "window_closed")
        with self.assertRaises(ValueError):
            resolve_denominator_rule(CONFIG, "everyone")
        with self.assertRaises(ValueError):
            resolve_denominator_rule({})


if __name__ == "__main__":
    unittest.main()
