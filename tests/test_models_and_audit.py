"""Property tests for paired model evaluation and the FlipAudit object (M3 & M4)."""

from __future__ import annotations

import unittest
import numpy as np

from churn_flip_audit.audit import (
    CELL_BOTH_RIGHT,
    CELL_BOTH_WRONG,
    CELL_FIXED,
    CELL_NEWLY_BROKEN,
    FlipAudit,
    classify_flip_cell,
)
from churn_flip_audit.models import (
    GradientBoostedChallenger,
    LogisticRegressionBaseline,
    PairedPredictionRecord,
)


class TestModelsAndFlipAuditProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_records = [
            # 1: both_right (y=0, base=0, chal=0)
            PairedPredictionRecord("C1", "starter", "new_0_3m", "na", 4900, 0, 0.12, 0, 0.08, 0),
            # 2: both_right (y=1, base=1, chal=1)
            PairedPredictionRecord("C2", "pro", "growing_3_12m", "emea", 19900, 1, 0.82, 1, 0.91, 1),
            # 3: fixed_tp (y=1, base=0, chal=1)
            PairedPredictionRecord("C3", "starter", "growing_3_12m", "na", 4900, 1, 0.25, 0, 0.85, 1),
            # 4: fixed_tn (y=0, base=1, chal=0)
            PairedPredictionRecord("C4", "free_trial", "new_0_3m", "apac", 1900, 0, 0.74, 1, 0.15, 0),
            # 5: broken_fn (y=1, base=1, chal=0) in enterprise
            PairedPredictionRecord("C5", "enterprise", "mature_12m_plus", "na", 125000, 1, 0.88, 1, 0.22, 0),
            # 6: broken_fp (y=0, base=0, chal=1) in enterprise
            PairedPredictionRecord("C6", "enterprise", "mature_12m_plus", "emea", 125000, 0, 0.19, 0, 0.79, 1),
            # 7: both_wrong_fn (y=1, base=0, chal=0)
            PairedPredictionRecord("C7", "pro", "mature_12m_plus", "latam", 19900, 1, 0.31, 0, 0.35, 0),
            # 8: both_wrong_fp (y=0, base=1, chal=1)
            PairedPredictionRecord("C8", "starter", "new_0_3m", "na", 4900, 0, 0.68, 1, 0.72, 1),
        ]

    def test_four_flip_cells_partition_held_out_set_exhaustively_and_mutually_exclusively(self) -> None:
        audit = FlipAudit(self.sample_records)
        s = audit.overall_summary()
        self.assertEqual(s["n"], 8)
        self.assertEqual(s["both_right"] + s["fixed"] + s["newly_broken"] + s["both_wrong"], s["n"])
        self.assertEqual(s["both_right"], 2)
        self.assertEqual(s["fixed"], 2)
        self.assertEqual(s["fixed_tp"], 1)
        self.assertEqual(s["fixed_tn"], 1)
        self.assertEqual(s["newly_broken"], 2)
        self.assertEqual(s["broken_fn"], 1)
        self.assertEqual(s["broken_fp"], 1)
        self.assertEqual(s["both_wrong"], 2)

    def test_net_accuracy_gain_equals_fixed_minus_newly_broken_over_n_overall_and_in_every_segment(self) -> None:
        audit = FlipAudit(self.sample_records)
        overall = audit.overall_summary()
        self.assertAlmostEqual(
            overall["challenger_accuracy"] - overall["baseline_accuracy"],
            (overall["fixed"] - overall["newly_broken"]) / float(overall["n"]),
            places=6,
        )

        for seg in audit.by_segment("plan_tier"):
            self.assertAlmostEqual(
                seg["challenger_accuracy"] - seg["baseline_accuracy"],
                (seg["fixed"] - seg["newly_broken"]) / float(seg["n"]),
                places=6,
            )

        for comp in audit.by_composite_segment(("plan_tier", "tenure_band")):
            self.assertAlmostEqual(
                comp["challenger_accuracy"] - comp["baseline_accuracy"],
                (comp["fixed"] - comp["newly_broken"]) / float(comp["n"]),
                places=6,
            )

    def test_probability_bands_sum_to_total_held_out_population(self) -> None:
        audit = FlipAudit(self.sample_records)
        for src in ("challenger", "baseline"):
            bands = audit.by_probability_band(prob_source=src)
            self.assertEqual(sum(b["n"] for b in bands), len(self.sample_records))
            self.assertEqual(
                sum(b["fixed"] for b in bands),
                audit.overall_summary()["fixed"],
            )
            self.assertEqual(
                sum(b["newly_broken"] for b in bands),
                audit.overall_summary()["newly_broken"],
            )

    def test_cost_model_crossover_ratio_yields_exact_zero_net_utility(self) -> None:
        # Add one more fixed record so fixed_total (3) > newly_broken_enterprise (2)
        records = list(self.sample_records) + [
            PairedPredictionRecord("C9", "pro", "new_0_3m", "na", 19900, 1, 0.20, 0, 0.88, 1)
        ]
        audit = FlipAudit(records)
        initial = audit.evaluate_cost_model(high_value_tier="enterprise", high_value_broken_cost_ratio=1.0)
        r_star = initial["crossover_ratio_high_value"]
        at_crossover = audit.evaluate_cost_model(
            high_value_tier="enterprise",
            high_value_broken_cost_ratio=r_star,
        )
        self.assertAlmostEqual(at_crossover["net_utility_units"], 0.0, places=3)

    def test_models_raise_runtime_error_when_predicting_before_fit_and_validate_audit_inputs(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(RuntimeError):
            LogisticRegressionBaseline().predict_proba(X)
        with self.assertRaises(RuntimeError):
            GradientBoostedChallenger().predict_proba(X)

        audit = FlipAudit(self.sample_records)
        with self.assertRaises(ValueError):
            audit.by_segment("invalid_column")
        with self.assertRaises(ValueError):
            audit.by_probability_band(prob_source="invalid_source")

        cell, subtype = classify_flip_cell(y_true=1, yhat_base=0, yhat_chal=0)
        self.assertEqual((cell, subtype), (CELL_BOTH_WRONG, "both_wrong_fn"))


if __name__ == "__main__":
    unittest.main()
