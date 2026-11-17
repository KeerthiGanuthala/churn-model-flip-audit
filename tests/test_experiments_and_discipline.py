"""End-to-end tests for Experiments 1-6, CLI tables, and measurement discipline (M5 & M6)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from churn_flip_audit.cli import execute_full_pipeline, format_experiment_tables
from churn_flip_audit.experiments import compare_paired_vs_unpaired_se, mcnemar_exact_and_asymptotic
from tools.check_numbers import (
    load_tracked_files,
    prove_checker_catches_mutation,
    verify_document_numbers,
)
from tools.collect_metrics import build_metrics_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestExperimentsAndMeasurementDiscipline(unittest.TestCase):
    def test_mcnemar_and_paired_variance_reflect_positive_prediction_correlation(self) -> None:
        mc_zero = mcnemar_exact_and_asymptotic(0, 0)
        self.assertEqual(mc_zero["exact_p_value"], 1.0)

        comp = compare_paired_vs_unpaired_se(
            n_total=4002,
            both_right=2832,
            fixed=699,
            newly_broken=235,
            both_wrong=236,
        )
        # Because both models agree on 76.66% of customers, paired SE is strictly smaller than unpaired SE
        self.assertLess(comp["se_paired"], comp["se_unpaired_wrong"])
        self.assertGreater(comp["se_inflation_ratio_unpaired_over_paired"], 1.10)
        self.assertLess(
            comp["min_test_n_80pct_power_2pt_gain_paired"],
            comp["min_test_n_80pct_power_2pt_gain_unpaired_wrong"],
        )

    def test_full_pipeline_and_check_numbers_pass_and_catch_deliberate_figure_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "sql").mkdir(parents=True)
            (tmp_path / " sql").rmdir() if (tmp_path / " sql").exists() else None
            (tmp_path / "sql" / "schema.sql").write_text(
                (REPO_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (tmp_path / "sql" / "features.sql").write_text(
                (REPO_ROOT / "sql" / "features.sql").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            res = execute_full_pipeline(tmp_path, n_customers=600)
            formatted = format_experiment_tables(res)
            self.assertIn("EXPERIMENT 1: WHAT THE AGGREGATE GAIN HIDES", formatted)
            self.assertIn("EXPERIMENT 6 (STRETCH): GUARDED ROLLOUT ROUTING POLICY", formatted)

            for exp_id in [
                "exp1_headline_flip_matrix",
                "exp2_segment_stratified_flips",
                "exp3_mcnemar_significance",
                "exp4_cost_ratio_crossover",
                "exp5_threshold_and_calibration",
                "exp6_guarded_rollout_policy",
            ]:
                exp_file = tmp_path / "docs" / "experiments" / f"{exp_id}.json"
                self.assertTrue(exp_file.exists(), f"Missing {exp_file}")
                loaded = json.loads(exp_file.read_text(encoding="utf-8"))
                self.assertEqual(loaded["experiment_id"], exp_id)

        # Verify canonical repo metrics and prove check_numbers catches a corrupted figure
        metrics = build_metrics_payload(REPO_ROOT)
        tracked_files = load_tracked_files(REPO_ROOT)
        errors = verify_document_numbers(metrics, tracked_files)
        self.assertEqual(errors, [], f"Unexpected number drift in docs: {errors}")

        caught_error = prove_checker_catches_mutation(metrics, tracked_files)
        self.assertIn("readme_newly_broken_count", caught_error)
        self.assertIn("expected 235, found 236", caught_error)


if __name__ == "__main__":
    unittest.main()
