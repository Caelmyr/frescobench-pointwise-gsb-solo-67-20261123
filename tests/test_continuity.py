"""Rules for continuing spots between two surface-observation runs.

Run from the project root: python3 tests/test_continuity.py
"""
import itertools
import sys
import unittest

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from frescobench import continuity
from frescobench.model import InputError


def pair(earlier, later, offset=1.0, overlap=0.5, confirmed="review", i=0, j=0):
    return {
        "earlier": earlier,
        "later": later,
        "earlier_index": i,
        "later_index": j,
        "center_offset_px": offset,
        "footprint_overlap": overlap,
        "previous_decision": confirmed,
    }


def total_value(chosen, offset_limit):
    return sum(continuity.pair_cost(item, offset_limit) * -1 for item in chosen)


def brute_force(pairs, earlier_count, later_count, offset_limit):
    """Every one-to-one subset, ranked by the same additive value."""
    best, best_value = [], None
    for size in range(min(earlier_count, later_count), -1, -1):
        for subset in itertools.combinations(pairs, size):
            if len({item["earlier_index"] for item in subset}) != size:
                continue
            if len({item["later_index"] for item in subset}) != size:
                continue
            value = total_value(subset, offset_limit)
            if best_value is None or value > best_value + 1e-6:
                best, best_value = list(subset), value
    return best


class CountRule(unittest.TestCase):
    def test_keeps_more_pairs_over_better_scores(self):
        pairs = [
            pair("A", "X", offset=0.1, overlap=0.99, confirmed="confirm", i=0, j=0),
            pair("B", "X", offset=20.0, overlap=0.2, i=1, j=0),
            pair("B", "Y", offset=1.0, overlap=0.6, i=1, j=1),
        ]
        chosen = continuity.choose_pairs(pairs, 2, 2, 40)
        self.assertEqual({item["earlier"] for item in chosen}, {"A", "B"})
        self.assertEqual(len(chosen), 2)

    def test_confirmed_tie_break(self):
        # Two solutions have one pair each; only one continues a confirmed earlier spot.
        pairs = [
            pair("A", "X", offset=1, overlap=0.9, confirmed="review", i=0, j=0),
            pair("B", "X", offset=1, overlap=0.9, confirmed="confirm", i=1, j=0),
        ]
        chosen = continuity.choose_pairs(pairs, 2, 1, 40)
        self.assertEqual([item["earlier"] for item in chosen], ["B"])

    def test_footprint_then_offset(self):
        # Same count; one solution has a better footprint, the other a smaller offset.
        pairs = [
            pair("A", "X", offset=10, overlap=0.8, i=0, j=0),
            pair("A", "Y", offset=1, overlap=0.4, i=0, j=1),
            pair("B", "Y", offset=10, overlap=0.4, i=1, j=1),
            pair("B", "X", offset=1, overlap=0.4, i=1, j=0),
        ]
        chosen = {(item["earlier"], item["later"]) for item in continuity.choose_pairs(pairs, 2, 2, 40)}
        self.assertIn(("A", "X"), chosen)
        self.assertIn(("B", "Y"), chosen)

    def test_offset_breaks_a_footprint_tie(self):
        pairs = [
            pair("A", "X", offset=30, overlap=0.5, i=0, j=0),
            pair("A", "Y", offset=1, overlap=0.5, i=0, j=1),
            pair("B", "Y", offset=30, overlap=0.5, i=1, j=1),
            pair("B", "X", offset=1, overlap=0.5, i=1, j=0),
        ]
        chosen = {(item["earlier"], item["later"]) for item in continuity.choose_pairs(pairs, 2, 2, 40)}
        self.assertEqual(chosen, {("A", "Y"), ("B", "X")})

    def test_matches_brute_force_on_random_cases(self):
        generator = np.random.default_rng(7)
        for _ in range(120):
            earlier_count = int(generator.integers(1, 5))
            later_count = int(generator.integers(1, 5))
            pairs = []
            for i in range(earlier_count):
                for j in range(later_count):
                    if generator.random() < 0.55:
                        pairs.append(pair(
                            f"E{i}", f"L{j}",
                            offset=float(generator.uniform(0.5, 39)),
                            overlap=float(generator.uniform(0, 1)),
                            confirmed="confirm" if generator.random() < 0.4 else "review",
                            i=i, j=j,
                        ))
            chosen = continuity.choose_pairs(pairs, earlier_count, later_count, 40)
            expected = brute_force(pairs, earlier_count, later_count, 40)
            self.assertEqual(len(chosen), len(expected))
            self.assertAlmostEqual(
                total_value(chosen, 40), total_value(expected, 40), places=3,
                msg=f"pairs={pairs}",
            )


class Eligibility(unittest.TestCase):
    def test_distance_and_overlap_limits(self):
        extent = (100, 100)
        earlier = {
            "metrics": {"observations": [
                {"key": "A", "normalized_center": [0.3, 0.3], "bounds": [0.28, 0.28, 0.32, 0.32],
                 "outline": [[0.28, 0.28], [0.32, 0.28], [0.32, 0.32], [0.28, 0.32]]},
            ]},
            "decisions": {"A": {"decision": "confirm", "notes": "seen"}},
        }
        later = {
            "metrics": {"observations": [
                {"key": "B", "normalized_center": [0.3, 0.31], "bounds": [0.28, 0.29, 0.32, 0.33],
                 "outline": [[0.28, 0.29], [0.32, 0.29], [0.32, 0.33], [0.28, 0.33]]},
                {"key": "C", "normalized_center": [0.6, 0.6], "bounds": [0.58, 0.58, 0.62, 0.62],
                 "outline": [[0.58, 0.58], [0.62, 0.58], [0.62, 0.62], [0.58, 0.62]]},
            ]},
            "decisions": {},
        }
        captured = np.ones((100, 100), dtype=bool)
        earlier_support = continuity.RunSupport(extent, captured)
        later_support = continuity.RunSupport(extent, captured)
        found = continuity.eligible_pairs(None, None, earlier, later, earlier_support, later_support, 5, 0.3)
        self.assertEqual([(item["earlier"], item["later"]) for item in found], [("A", "B")])
        self.assertEqual(found[0]["previous_decision"], "confirm")
        self.assertEqual(found[0]["previous_notes"], "seen")
        self.assertAlmostEqual(found[0]["center_offset_px"], 1.0, places=2)
        blocked = continuity.eligible_pairs(None, None, earlier, later, earlier_support, later_support, 0.5, 0.3)
        self.assertEqual(blocked, [])

    def test_uncaptured_center_is_not_eligible(self):
        extent = (50, 50)
        earlier = {"metrics": {"observations": [
            {"key": "A", "normalized_center": [0.9, 0.9], "bounds": [0.88, 0.88, 0.92, 0.92],
             "outline": [[0.88, 0.88], [0.92, 0.88], [0.92, 0.92], [0.88, 0.92]]},
        ]}, "decisions": {}}
        later = {"metrics": {"observations": [
            {"key": "B", "normalized_center": [0.9, 0.9], "bounds": [0.88, 0.88, 0.92, 0.92],
             "outline": [[0.88, 0.88], [0.92, 0.88], [0.92, 0.92], [0.88, 0.92]]},
        ]}, "decisions": {}}
        captured = np.ones((50, 50), dtype=bool)
        captured[45:, :] = False
        found = continuity.eligible_pairs(
            None, None, earlier, later, continuity.RunSupport(extent, captured),
            continuity.RunSupport(extent, captured), 20, 0,
        )
        self.assertEqual(found, [])

    def test_archive_marks_missing_and_unmatched_but_never_removed(self):
        extent = (100, 100)
        shape = [[0.1, 0.1], [0.11, 0.1], [0.11, 0.11], [0.1, 0.11]]
        earlier = {"metrics": {"observations": [
            {"key": "A", "type": "spot", "normalized_center": [0.1, 0.1], "bounds": [0.1, 0.1, 0.11, 0.11], "outline": shape},
            {"key": "B", "type": "spot", "normalized_center": [0.9, 0.9], "bounds": [0.9, 0.9, 0.91, 0.91], "outline": shape},
        ]}, "decisions": {"A": {"decision": "confirm", "notes": "x"}}}
        later = {"metrics": {"observations": [
            {"key": "C", "type": "spot", "normalized_center": [0.3, 0.3], "bounds": [0.3, 0.3, 0.31, 0.31], "outline": shape},
        ]}, "decisions": {}}
        captured = np.ones((100, 100), dtype=bool)
        captured[:, :50] = False
        earlier_rows, later_rows = continuity.archive(
            earlier, later, [], continuity.RunSupport(extent, captured),
            continuity.RunSupport(extent, captured),
        )
        self.assertEqual(earlier_rows[0]["outcome"], "not_photographed")
        self.assertEqual(earlier_rows[1]["outcome"], "no_qualifying_continuation")
        self.assertTrue(all(row["outcome"] != "removed" for row in earlier_rows))
        self.assertEqual(later_rows[0]["outcome"], "not_photographed")
        self.assertFalse(any("removed" in row["outcome_label"].lower() for row in earlier_rows))


class Guards(unittest.TestCase):
    def test_limits_are_validated(self):
        for offset, overlap in ((0, 0.2), (5, -0.1), (5, 1.5)):
            with self.assertRaises(InputError):
                continuity.parse_limits({"center_offset_px": offset, "footprint_overlap": overlap})


if __name__ == "__main__":
    unittest.main(verbosity=2)
