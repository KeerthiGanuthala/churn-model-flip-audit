"""Experiments 1 through 6 (including Stretch Experiment 6: Guarded Rollout Policy).

Each experiment derives its numbers from `FlipAudit`, writes structured JSON to
`docs/experiments/`, and formats clean Markdown/ASCII tables for CLI and README output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
from scipy import stats

from .audit import CELL_BOTH_RIGHT, CELL_BOTH_WRONG, CELL_FIXED, CELL_NEWLY_BROKEN, FlipAudit
from .models import PairedPredictionRecord


def mcnemar_exact_and_asymptotic(n_fixed: int, n_broken: int) -> Dict[str, Any]:
    """Compute McNemar's exact binomial test and asymptotic chi2 on discordant pairs."""
    b = int(n_fixed)
    c = int(n_broken)
    discordant = b + c
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "fixed_b": 0,
            "newly_broken_c": 0,
            "exact_p_value": 1.0,
            "chi2_stat_uncorrected": 0.0,
            "chi2_stat_edwards_corrected": 0.0,
        }

    binom_res = stats.binomtest(b, n=discordant, p=0.5, alternative="two-sided")
    exact_p = float(binom_res.pvalue)
    chi2_uncorr = ((b - c) ** 2) / float(discordant)
    chi2_corr = ((abs(b - c) - 1.0) ** 2) / float(discordant)

    return {
        "discordant_pairs": discordant,
        "fixed_b": b,
        "newly_broken_c": c,
        "win_share_among_discordant": round(b / float(discordant), 6),
        "exact_p_value": exact_p,
        "exact_log10_p_value": round(math.log10(max(exact_p, 1e-300)), 4),
        "chi2_stat_uncorrected": round(chi2_uncorr, 4),
        "chi2_stat_edwards_corrected": round(chi2_corr, 4),
    }


def compare_paired_vs_unpaired_se(
    n_total: int,
    both_right: int,
    fixed: int,
    newly_broken: int,
    both_wrong: int,
) -> Dict[str, Any]:
    """Compare true paired SE of accuracy difference against the naive unpaired two-proportion SE.

    True paired variance of delta_hat = (fixed - newly_broken) / n:
      Var_paired(delta_hat) = [ (fixed + newly_broken)/n - ((fixed - newly_broken)/n)^2 ] / n

    Naive unpaired variance assuming two independent samples of size n:
      Var_unpaired(delta_hat) = [ p1*(1 - p1) + p2*(1 - p2) ] / n
    """
    n = float(n_total)
    p_base = (both_right + newly_broken) / n
    p_chal = (both_right + fixed) / n
    delta = p_chal - p_base

    discordant_rate = (fixed + newly_broken) / n
    var_paired = (discordant_rate - (delta ** 2)) / n
    se_paired = math.sqrt(max(var_paired, 1e-16))
    z_paired = delta / se_paired

    var_unpaired = (p_base * (1.0 - p_base) + p_chal * (1.0 - p_chal)) / n
    se_unpaired = math.sqrt(max(var_unpaired, 1e-16))
    z_unpaired = delta / se_unpaired

    # Minimum test-set size N_min required for 80% power at alpha=0.05 (two-sided)
    # given the observed discordant share and win ratio among discordant pairs:
    # Under paired test: SE_0(delta) = sqrt(discordant_rate / N), SE_1(delta) = sqrt((discordant_rate - delta^2) / N)
    z_alpha = 1.959963984540054
    z_beta = 0.8416212335729143
    n_min_paired = math.ceil(
        ((z_alpha * math.sqrt(discordant_rate) + z_beta * math.sqrt(discordant_rate - delta ** 2)) / abs(delta)) ** 2
    )
    unpaired_num = (
        z_alpha * math.sqrt(2.0 * 0.5 * (p_base + p_chal) * (1.0 - 0.5 * (p_base + p_chal)))
        + z_beta * math.sqrt(p_base * (1.0 - p_base) + p_chal * (1.0 - p_chal))
    )
    n_min_unpaired = math.ceil((unpaired_num / abs(delta)) ** 2)

    # Also compute N_min for a subtle +2.0 percentage point gain with the same discordance rate
    subtle_delta = 0.02
    n_min_paired_2pt = math.ceil(
        ((z_alpha * math.sqrt(discordant_rate) + z_beta * math.sqrt(discordant_rate - subtle_delta ** 2)) / subtle_delta) ** 2
    )
    n_min_unpaired_2pt = math.ceil((unpaired_num / subtle_delta) ** 2)

    return {
        "n_total": n_total,
        "p_baseline": round(p_base, 6),
        "p_challenger": round(p_chal, 6),
        "accuracy_delta": round(delta, 6),
        "se_paired": round(se_paired, 6),
        "se_unpaired_wrong": round(se_unpaired, 6),
        "se_inflation_ratio_unpaired_over_paired": round(se_unpaired / se_paired, 4),
        "variance_inflation_ratio": round(var_unpaired / var_paired, 4),
        "z_stat_paired": round(z_paired, 4),
        "z_stat_unpaired_wrong": round(z_unpaired, 4),
        "min_test_n_80pct_power_observed_gain_paired": int(n_min_paired),
        "min_test_n_80pct_power_observed_gain_unpaired_wrong": int(n_min_unpaired),
        "min_test_n_80pct_power_2pt_gain_paired": int(n_min_paired_2pt),
        "min_test_n_80pct_power_2pt_gain_unpaired_wrong": int(n_min_unpaired_2pt),
    }


def run_experiment_1_headline_matrix(audit: FlipAudit) -> Dict[str, Any]:
    """Experiment 1: What the aggregate gain hides (headline flip matrix)."""
    summary = audit.overall_summary()
    matrix_2x2 = [
        {
            "baseline_outcome": "Baseline Correct",
            "challenger_correct": summary["both_right"],
            "challenger_wrong": summary["newly_broken"],
            "row_total": summary["baseline_correct"],
        },
        {
            "baseline_outcome": "Baseline Wrong",
            "challenger_correct": summary["fixed"],
            "challenger_wrong": summary["both_wrong"],
            "row_total": summary["n"] - summary["baseline_correct"],
        },
    ]
    return {
        "experiment_id": "exp1_headline_flip_matrix",
        "title": "What the Aggregate Gain Hides",
        "summary": summary,
        "contingency_matrix_2x2": matrix_2x2,
    }


def run_experiment_2_segment_flips(audit: FlipAudit) -> Dict[str, Any]:
    """Experiment 2: Where the new model is worse (segment-stratified flip rates & mix shift)."""
    overall = audit.overall_summary()
    by_tier = audit.by_segment(
        "plan_tier",
        segment_order=["free_trial", "starter", "pro", "enterprise"],
    )
    by_tenure = audit.by_segment(
        "tenure_band",
        segment_order=["new_0_3m", "growing_3_12m", "mature_12m_plus"],
    )
    by_region = audit.by_segment(
        "region",
        segment_order=["na", "emea", "apac", "latam"],
    )
    by_composite = audit.by_composite_segment(("plan_tier", "tenure_band"))

    # Verify exact mix-shift decomposition: sum(share_s * delta_s) == overall_delta
    weighted_delta_sum = sum(
        (r["n"] / float(overall["n"])) * ((r["challenger_correct"] - r["baseline_correct"]) / float(r["n"]))
        for r in by_tier
    )
    enterprise_row = next(r for r in by_tier if r["segment_value"] == "enterprise")
    self_serve_fixed = sum(r["fixed"] for r in by_tier if r["segment_value"] != "enterprise")
    self_serve_broken = sum(r["newly_broken"] for r in by_tier if r["segment_value"] != "enterprise")

    return {
        "experiment_id": "exp2_segment_stratified_flips",
        "title": "Where the New Model Is Worse: Segment Stratification and Mix Shift",
        "overall_accuracy_delta": overall["accuracy_delta"],
        "reconstructed_weighted_delta": round(weighted_delta_sum, 6),
        "enterprise_share_of_population": enterprise_row["population_share"],
        "enterprise_share_of_newly_broken": round(
            enterprise_row["newly_broken"] / float(overall["newly_broken"]), 6
        ),
        "self_serve_fixed": self_serve_fixed,
        "self_serve_newly_broken": self_serve_broken,
        "by_plan_tier": by_tier,
        "by_tenure_band": by_tenure,
        "by_region": by_region,
        "by_tier_and_tenure": by_composite,
    }


def run_experiment_3_mcnemar_significance(audit: FlipAudit) -> Dict[str, Any]:
    """Experiment 3: McNemar's exact test vs the wrong unpaired two-proportion test."""
    s = audit.overall_summary()
    mcnemar_overall = mcnemar_exact_and_asymptotic(s["fixed"], s["newly_broken"])
    se_comp = compare_paired_vs_unpaired_se(
        n_total=s["n"],
        both_right=s["both_right"],
        fixed=s["fixed"],
        newly_broken=s["newly_broken"],
        both_wrong=s["both_wrong"],
    )

    by_tier_tests: List[Dict[str, Any]] = []
    for row in audit.by_segment("plan_tier", segment_order=["free_trial", "starter", "pro", "enterprise"]):
        mc = mcnemar_exact_and_asymptotic(row["fixed"], row["newly_broken"])
        by_tier_tests.append(
            {
                "plan_tier": row["segment_value"],
                "n": row["n"],
                "fixed": row["fixed"],
                "newly_broken": row["newly_broken"],
                "accuracy_delta": row["accuracy_delta"],
                "chi2_edwards": mc["chi2_stat_edwards_corrected"],
                "exact_p_value": mc["exact_p_value"],
                "direction": "challenger_wins" if row["fixed"] > row["newly_broken"] else "challenger_loses",
            }
        )

    return {
        "experiment_id": "exp3_mcnemar_significance",
        "title": "Is the Gain Real: McNemar's Exact Test vs Unpaired Two-Proportion Fallacy",
        "mcnemar_overall": mcnemar_overall,
        "paired_vs_unpaired": se_comp,
        "by_plan_tier_mcnemar": by_tier_tests,
    }


def run_experiment_4_cost_crossover(audit: FlipAudit) -> Dict[str, Any]:
    """Experiment 4: Sweep the cost ratio r and find where a +12pt accuracy win becomes a net loss."""
    ratio_grid = [1.0, 2.0, 2.5, 3.0, 3.5, 3.5778, 4.0, 5.0, 7.5, 10.0, 15.0, 25.0]
    sweep_rows: List[Dict[str, Any]] = []
    for r in ratio_grid:
        res = audit.evaluate_cost_model(
            high_value_tier="enterprise",
            high_value_broken_cost_ratio=r,
            standard_broken_cost_ratio=1.0,
            fixed_value_unit=1.0,
        )
        sweep_rows.append(
            {
                "cost_ratio_r": r,
                "fixed_total": res["fixed_total"],
                "newly_broken_standard": res["newly_broken_standard"],
                "newly_broken_enterprise": res["newly_broken_high_value"],
                "net_utility_units": res["net_utility_units"],
                "is_net_loss": res["net_utility_units"] < 0.0,
            }
        )

    base_cost_eval = audit.evaluate_cost_model(
        high_value_tier="enterprise",
        high_value_broken_cost_ratio=5.0,
    )
    overall = audit.overall_summary()

    return {
        "experiment_id": "exp4_cost_ratio_crossover",
        "title": "When a 12-Point Accuracy Gain Is a Business Loss",
        "headline_accuracy_delta": overall["accuracy_delta"],
        "crossover_ratio_high_value_enterprise": base_cost_eval["crossover_ratio_high_value"],
        "crossover_ratio_uniform_all_broken": base_cost_eval["crossover_ratio_uniform"],
        "actual_enterprise_to_starter_mcv_ratio": round(125000.0 / 4900.0, 2),
        "actual_enterprise_to_pro_mcv_ratio": round(125000.0 / 19900.0, 2),
        "contract_dollar_valuation": {
            "contract_dollars_gained_from_fixed": base_cost_eval["contract_dollars_gained_from_fixed"],
            "contract_dollars_lost_from_broken": base_cost_eval["contract_dollars_lost_from_broken"],
            "net_contract_dollars": base_cost_eval["net_contract_dollars"],
        },
        "cost_ratio_sweep": sweep_rows,
    }


def _compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(len(y_true))
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(probs[mask])) - float(np.mean(y_true[mask])))
    return round(ece, 6)


def _platt_rescale(
    train_probs: np.ndarray,
    train_y: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Fit 2-parameter Platt scaling (logit linear recalibration) on train probabilities."""
    eps = 1e-6
    train_logits = np.log(np.clip(train_probs, eps, 1.0 - eps) / (1.0 - np.clip(train_probs, eps, 1.0 - eps)))
    test_logits = np.log(np.clip(test_probs, eps, 1.0 - eps) / (1.0 - np.clip(test_probs, eps, 1.0 - eps)))

    # 1D Newton-Raphson for Platt parameters (a * logit + b)
    X = np.column_stack([np.ones(len(train_logits)), train_logits])
    w = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(30):
        z = np.clip(X @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - train_y) + 1e-4 * w
        s = np.maximum(p * (1.0 - p), 1e-8)
        hess = (X.T * s) @ X + 1e-4 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break

    z_test = np.clip(w[0] + w[1] * test_logits, -30.0, 30.0)
    return np.round(1.0 / (1.0 + np.exp(-z_test)), 8)


def run_experiment_5_threshold_and_calibration(
    paired_records: Sequence[PairedPredictionRecord],
    train_p_base: np.ndarray,
    train_p_chal: np.ndarray,
    train_y: np.ndarray,
) -> Dict[str, Any]:
    """Experiment 5: Flip counts across decision thresholds and reliability calibration bins."""
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_sweep: List[Dict[str, Any]] = []

    for t in thresholds:
        aud_t = FlipAudit(paired_records, threshold_baseline=0.50, threshold_challenger=t)
        s = aud_t.overall_summary()
        threshold_sweep.append(
            {
                "challenger_threshold": round(t, 2),
                "baseline_accuracy": s["baseline_accuracy"],
                "challenger_accuracy": s["challenger_accuracy"],
                "accuracy_delta": s["accuracy_delta"],
                "fixed": s["fixed"],
                "newly_broken": s["newly_broken"],
                "broken_fn": s["broken_fn"],
                "broken_fp": s["broken_fp"],
                "net_gain_count": s["net_gain_count"],
            }
        )

    audit_default = FlipAudit(paired_records)
    challenger_bands = audit_default.by_probability_band(prob_source="challenger")
    baseline_bands = audit_default.by_probability_band(prob_source="baseline")

    y_test = np.array([r.y_true for r in paired_records], dtype=np.float64)
    p_base_test = np.array([r.p_baseline for r in paired_records], dtype=np.float64)
    p_chal_test = np.array([r.p_challenger for r in paired_records], dtype=np.float64)

    ece_base_raw = _compute_ece(p_base_test, y_test)
    ece_chal_raw = _compute_ece(p_chal_test, y_test)

    p_base_cal = _platt_rescale(train_p_base, train_y, p_base_test)
    p_chal_cal = _platt_rescale(train_p_chal, train_y, p_chal_test)

    ece_base_cal = _compute_ece(p_base_cal, y_test)
    ece_chal_cal = _compute_ece(p_chal_cal, y_test)

    recalibrated_records: List[PairedPredictionRecord] = []
    for i, r in enumerate(paired_records):
        recalibrated_records.append(
            PairedPredictionRecord(
                customer_id=r.customer_id,
                plan_tier=r.plan_tier,
                tenure_band=r.tenure_band,
                region=r.region,
                monthly_contract_value_cents=r.monthly_contract_value_cents,
                y_true=r.y_true,
                p_baseline=float(p_base_cal[i]),
                yhat_baseline=int(p_base_cal[i] >= 0.50),
                p_challenger=float(p_chal_cal[i]),
                yhat_challenger=int(p_chal_cal[i] >= 0.50),
            )
        )
    recal_audit = FlipAudit(recalibrated_records)
    recal_summary = recal_audit.overall_summary()

    return {
        "experiment_id": "exp5_threshold_and_calibration",
        "title": "Threshold Sweep and Reliability Calibration",
        "ece_baseline_raw": ece_base_raw,
        "ece_challenger_raw": ece_chal_raw,
        "ece_baseline_platt_calibrated": ece_base_cal,
        "ece_challenger_platt_calibrated": ece_chal_cal,
        "recalibrated_summary": {
            "baseline_accuracy": recal_summary["baseline_accuracy"],
            "challenger_accuracy": recal_summary["challenger_accuracy"],
            "accuracy_delta": recal_summary["accuracy_delta"],
            "fixed": recal_summary["fixed"],
            "newly_broken": recal_summary["newly_broken"],
        },
        "threshold_sweep": threshold_sweep,
        "challenger_reliability_bands": challenger_bands,
        "baseline_reliability_bands": baseline_bands,
    }


def run_experiment_6_guarded_rollout(
    train_paired_records: Sequence[PairedPredictionRecord],
    test_paired_records: Sequence[PairedPredictionRecord],
) -> Dict[str, Any]:
    """Experiment 6 (Stretch): Design and evaluate a guarded routing rollout policy.

    Policy learning on train split:
      Inspect flip rates by `plan_tier` on the training split (or validation slice)
      and route segments where Challenger net gain <= 0 (specifically `enterprise`)
      or where Challenger predicted probability falls in the uncalibrated dead-zone
      `[0.40, 0.60)` back to the Baseline model, shipping Challenger only on
      verified winning segments (`free_trial`, `starter`, `pro` outside dead-zone).
    """
    train_audit = FlipAudit(train_paired_records)
    train_by_tier = train_audit.by_segment("plan_tier")
    blocked_tiers = {
        row["segment_value"]
        for row in train_by_tier
        if row["challenger_accuracy"] <= row["baseline_accuracy"]
    }

    guarded_records: List[PairedPredictionRecord] = []
    routed_to_baseline_count = 0
    routed_to_challenger_count = 0

    for r in test_paired_records:
        in_dead_zone = 0.40 <= r.p_challenger < 0.60
        if (r.plan_tier in blocked_tiers) or in_dead_zone:
            routed_to_baseline_count += 1
            p_guard = r.p_baseline
            yhat_guard = r.yhat_baseline
        else:
            routed_to_challenger_count += 1
            p_guard = r.p_challenger
            yhat_guard = r.yhat_challenger

        guarded_records.append(
            PairedPredictionRecord(
                customer_id=r.customer_id,
                plan_tier=r.plan_tier,
                tenure_band=r.tenure_band,
                region=r.region,
                monthly_contract_value_cents=r.monthly_contract_value_cents,
                y_true=r.y_true,
                p_baseline=r.p_baseline,
                yhat_baseline=r.yhat_baseline,
                p_challenger=p_guard,
                yhat_challenger=yhat_guard,
            )
        )

    pure_audit = FlipAudit(test_paired_records)
    guarded_audit = FlipAudit(guarded_records)

    pure_summary = pure_audit.overall_summary()
    guarded_summary = guarded_audit.overall_summary()

    pure_cost = pure_audit.evaluate_cost_model(high_value_tier="enterprise", high_value_broken_cost_ratio=5.0)
    guarded_cost = guarded_audit.evaluate_cost_model(high_value_tier="enterprise", high_value_broken_cost_ratio=5.0)

    policies_table = [
        {
            "policy": "Pure Baseline (Logistic Regression)",
            "accuracy": pure_summary["baseline_accuracy"],
            "accuracy_delta_vs_baseline": 0.0,
            "fixed": 0,
            "newly_broken": 0,
            "newly_broken_enterprise": 0,
            "net_utility_r5": 0.0,
            "net_contract_dollars": 0.0,
        },
        {
            "policy": "Pure Challenger (Global GBDT)",
            "accuracy": pure_summary["challenger_accuracy"],
            "accuracy_delta_vs_baseline": pure_summary["accuracy_delta"],
            "fixed": pure_summary["fixed"],
            "newly_broken": pure_summary["newly_broken"],
            "newly_broken_enterprise": pure_cost["newly_broken_high_value"],
            "net_utility_r5": pure_cost["net_utility_units"],
            "net_contract_dollars": pure_cost["net_contract_dollars"],
        },
        {
            "policy": "Guarded Rollout (Route Enterprise & [0.4,0.6) to Baseline)",
            "accuracy": guarded_summary["challenger_accuracy"],
            "accuracy_delta_vs_baseline": guarded_summary["accuracy_delta"],
            "fixed": guarded_summary["fixed"],
            "newly_broken": guarded_summary["newly_broken"],
            "newly_broken_enterprise": guarded_cost["newly_broken_high_value"],
            "net_utility_r5": guarded_cost["net_utility_units"],
            "net_contract_dollars": guarded_cost["net_contract_dollars"],
        },
    ]

    return {
        "experiment_id": "exp6_guarded_rollout_policy",
        "title": "Stretch: Guarded Rollout Routing Policy",
        "blocked_tiers_from_train_audit": sorted(blocked_tiers),
        "uncertainty_dead_zone": "[0.40, 0.60)",
        "routed_to_baseline_count": routed_to_baseline_count,
        "routed_to_challenger_count": routed_to_challenger_count,
        "routed_to_challenger_share": round(routed_to_challenger_count / float(len(test_paired_records)), 6),
        "guarded_summary": guarded_summary,
        "guarded_by_tier": guarded_audit.by_segment(
            "plan_tier",
            segment_order=["free_trial", "starter", "pro", "enterprise"],
        ),
        "policy_comparison": policies_table,
    }


def write_all_experiments(
    output_dir: Path,
    audit: FlipAudit,
    paired_test: Sequence[PairedPredictionRecord],
    paired_train: Sequence[PairedPredictionRecord],
    train_p_base: np.ndarray,
    train_p_chal: np.ndarray,
    train_y: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Run Experiments 1 through 6, write JSON artifacts to `docs/experiments/`, and return all payloads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1 = run_experiment_1_headline_matrix(audit)
    exp2 = run_experiment_2_segment_flips(audit)
    exp3 = run_experiment_3_mcnemar_significance(audit)
    exp4 = run_experiment_4_cost_crossover(audit)
    exp5 = run_experiment_5_threshold_and_calibration(
        paired_test, train_p_base, train_p_chal, train_y
    )
    exp6 = run_experiment_6_guarded_rollout(paired_train, paired_test)

    experiments = {
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
        "exp5": exp5,
        "exp6": exp6,
    }

    for exp_payload in experiments.values():
        fname = f"{exp_payload['experiment_id']}.json"
        (output_dir / fname).write_text(
            json.dumps(exp_payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    return experiments
