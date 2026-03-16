from __future__ import annotations

import sys
import unittest
from pathlib import Path


ONLINE_PHASE_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "ntcir15" / "online_phase"
)
if str(ONLINE_PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_PHASE_DIR))

import bootstrap_significance  # noqa: E402
import evaluate_runs  # noqa: E402


class BootstrapSignificanceTests(unittest.TestCase):
    def test_identical_runs_have_zero_delta_and_unit_pvalue(self) -> None:
        indices = bootstrap_significance.generate_bootstrap_indices(3, 200, seed=7)
        values = [1.0, 0.5, 0.0]

        comparison = bootstrap_significance.compare_metric(values, values, indices)

        self.assertEqual(comparison["observed_delta"], 0.0)
        self.assertEqual(comparison["delta_ci_low"], 0.0)
        self.assertEqual(comparison["delta_ci_high"], 0.0)
        self.assertEqual(comparison["p_value"], 1.0)

    def test_strongly_better_run_is_significant(self) -> None:
        indices = bootstrap_significance.generate_bootstrap_indices(4, 500, seed=11)
        strong = [1.0, 1.0, 1.0, 1.0]
        weak = [0.0, 0.0, 0.0, 0.0]

        comparison = bootstrap_significance.compare_metric(strong, weak, indices)

        self.assertGreater(comparison["observed_delta"], 0.0)
        self.assertGreater(comparison["delta_ci_low"], 0.0)
        self.assertGreater(comparison["delta_ci_high"], 0.0)
        self.assertLess(comparison["p_value"], 0.05)

    def test_missing_query_is_scored_as_zero(self) -> None:
        qrels = {
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        }
        run = {
            "q1": [("d1", 1.0, 1)],
        }

        query_scores = evaluate_runs.per_query_manual_scores(qrels, run, k=10)

        self.assertEqual(query_scores["q1"]["map"], 1.0)
        self.assertEqual(query_scores["q2"]["map"], 0.0)
        self.assertEqual(query_scores["q2"]["ndcg_cut_10"], 0.0)
        self.assertEqual(query_scores["q2"]["recall_10"], 0.0)
        self.assertEqual(query_scores["q2"]["precision_10"], 0.0)

    def test_bh_adjustment_is_monotonic(self) -> None:
        adjusted = bootstrap_significance.adjust_pvalues_bh([0.01, 0.03, 0.02, 0.20])

        self.assertEqual(len(adjusted), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertLessEqual(adjusted[0], adjusted[3])


if __name__ == "__main__":
    unittest.main()

