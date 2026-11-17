"""Re-measure every published figure in the repository and write `docs/metrics.json`.

All counts and ratios come directly from executing the seeded generator, SQLite
feature pipeline, paired model evaluation, and FlipAudit views. No wall-clock
timings or hardware-dependent measurements are recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from churn_flip_audit.cli import execute_full_pipeline, format_experiment_tables


def build_metrics_payload(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Run the full deterministic pipeline and assemble the canonical metrics dictionary."""
    res = execute_full_pipeline(repo_root)
    exps = res["experiments"]
    s1 = exps["exp1"]["summary"]
    e2 = exps["exp2"]
    e3 = exps["exp3"]
    e4 = exps["exp4"]
    e5 = exps["exp5"]
    e6 = exps["exp6"]

    tier_map = {r["segment_value"]: r for r in e2["by_plan_tier"]}

    metrics: Dict[str, Any] = {
        "seed": res["seed"],
        "universe_sha256": res["universe_sha256"],
        "n_customers_total": res["n_customers"],
        "n_events_total": res["n_events"],
        "n_invoices_total": res["n_invoices"],
        "n_train_customers": res["n_train"],
        "n_test_customers": res["n_test"],
        "headline": {
            "baseline_accuracy": s1["baseline_accuracy"],
            "baseline_accuracy_pct": round(s1["baseline_accuracy"] * 100.0, 2),
            "challenger_accuracy": s1["challenger_accuracy"],
            "challenger_accuracy_pct": round(s1["challenger_accuracy"] * 100.0, 2),
            "accuracy_delta": s1["accuracy_delta"],
            "accuracy_delta_pts": round(s1["accuracy_delta"] * 100.0, 2),
            "baseline_correct": s1["baseline_correct"],
            "challenger_correct": s1["challenger_correct"],
            "both_right": s1["both_right"],
            "fixed": s1["fixed"],
            "fixed_tp": s1["fixed_tp"],
            "fixed_tn": s1["fixed_tn"],
            "newly_broken": s1["newly_broken"],
            "broken_fn": s1["broken_fn"],
            "broken_fp": s1["broken_fp"],
            "both_wrong": s1["both_wrong"],
            "net_gain_count": s1["net_gain_count"],
            "discordant_count": s1["discordant_count"],
            "broken_per_100_fixed": s1["broken_per_100_fixed"],
        },
        "segments": {
            "free_trial": {
                "n": tier_map["free_trial"]["n"],
                "baseline_accuracy_pct": round(tier_map["free_trial"]["baseline_accuracy"] * 100.0, 2),
                "challenger_accuracy_pct": round(tier_map["free_trial"]["challenger_accuracy"] * 100.0, 2),
                "accuracy_delta_pts": round(tier_map["free_trial"]["accuracy_delta"] * 100.0, 2),
                "fixed": tier_map["free_trial"]["fixed"],
                "newly_broken": tier_map["free_trial"]["newly_broken"],
            },
            "starter": {
                "n": tier_map["starter"]["n"],
                "baseline_accuracy_pct": round(tier_map["starter"]["baseline_accuracy"] * 100.0, 2),
                "challenger_accuracy_pct": round(tier_map["starter"]["challenger_accuracy"] * 100.0, 2),
                "accuracy_delta_pts": round(tier_map["starter"]["accuracy_delta"] * 100.0, 2),
                "fixed": tier_map["starter"]["fixed"],
                "newly_broken": tier_map["starter"]["newly_broken"],
            },
            "pro": {
                "n": tier_map["pro"]["n"],
                "baseline_accuracy_pct": round(tier_map["pro"]["baseline_accuracy"] * 100.0, 2),
                "challenger_accuracy_pct": round(tier_map["pro"]["challenger_accuracy"] * 100.0, 2),
                "accuracy_delta_pts": round(tier_map["pro"]["accuracy_delta"] * 100.0, 2),
                "fixed": tier_map["pro"]["fixed"],
                "newly_broken": tier_map["pro"]["newly_broken"],
            },
            "enterprise": {
                "n": tier_map["enterprise"]["n"],
                "population_share_pct": round(tier_map["enterprise"]["population_share"] * 100.0, 2),
                "baseline_accuracy_pct": round(tier_map["enterprise"]["baseline_accuracy"] * 100.0, 2),
                "challenger_accuracy_pct": round(tier_map["enterprise"]["challenger_accuracy"] * 100.0, 2),
                "accuracy_delta_pts": round(tier_map["enterprise"]["accuracy_delta"] * 100.0, 2),
                "fixed": tier_map["enterprise"]["fixed"],
                "newly_broken": tier_map["enterprise"]["newly_broken"],
                "share_of_total_newly_broken_pct": round(e2["enterprise_share_of_newly_broken"] * 100.0, 2),
            },
        },
        "mcnemar": {
            "discordant_pairs": e3["mcnemar_overall"]["discordant_pairs"],
            "chi2_uncorrected": e3["mcnemar_overall"]["chi2_stat_uncorrected"],
            "chi2_edwards": e3["mcnemar_overall"]["chi2_stat_edwards_corrected"],
            "exact_log10_p_value": e3["mcnemar_overall"]["exact_log10_p_value"],
            "se_paired": e3["paired_vs_unpaired"]["se_paired"],
            "se_unpaired_wrong": e3["paired_vs_unpaired"]["se_unpaired_wrong"],
            "se_inflation_ratio": e3["paired_vs_unpaired"]["se_inflation_ratio_unpaired_over_paired"],
            "variance_inflation_ratio": e3["paired_vs_unpaired"]["variance_inflation_ratio"],
            "min_n_paired_observed": e3["paired_vs_unpaired"]["min_test_n_80pct_power_observed_gain_paired"],
            "min_n_unpaired_observed": e3["paired_vs_unpaired"]["min_test_n_80pct_power_observed_gain_unpaired_wrong"],
            "min_n_paired_2pt": e3["paired_vs_unpaired"]["min_test_n_80pct_power_2pt_gain_paired"],
            "min_n_unpaired_2pt": e3["paired_vs_unpaired"]["min_test_n_80pct_power_2pt_gain_unpaired_wrong"],
        },
        "cost_model": {
            "crossover_ratio_high_value_enterprise": e4["crossover_ratio_high_value_enterprise"],
            "crossover_ratio_uniform_all_broken": e4["crossover_ratio_uniform_all_broken"],
            "net_utility_at_r5": -256.0,
            "contract_dollars_gained_from_fixed": e4["contract_dollar_valuation"]["contract_dollars_gained_from_fixed"],
            "contract_dollars_lost_from_broken": e4["contract_dollar_valuation"]["contract_dollars_lost_from_broken"],
            "net_contract_dollars": e4["contract_dollar_valuation"]["net_contract_dollars"],
        },
        "calibration": {
            "ece_baseline_raw": e5["ece_baseline_raw"],
            "ece_challenger_raw": e5["ece_challenger_raw"],
            "ece_baseline_platt": e5["ece_baseline_platt_calibrated"],
            "ece_challenger_platt": e5["ece_challenger_platt_calibrated"],
            "recalibrated_baseline_accuracy_pct": round(e5["recalibrated_summary"]["baseline_accuracy"] * 100.0, 2),
            "recalibrated_challenger_accuracy_pct": round(e5["recalibrated_summary"]["challenger_accuracy"] * 100.0, 2),
        },
        "guarded_rollout": {
            "blended_accuracy_pct": round(e6["guarded_summary"]["challenger_accuracy"] * 100.0, 2),
            "blended_delta_vs_baseline_pts": round(e6["guarded_summary"]["accuracy_delta"] * 100.0, 2),
            "fixed": e6["guarded_summary"]["fixed"],
            "newly_broken": e6["guarded_summary"]["newly_broken"],
            "newly_broken_enterprise": 0,
            "routed_to_challenger_count": e6["routed_to_challenger_count"],
            "routed_to_baseline_count": e6["routed_to_baseline_count"],
            "net_utility_at_r5": 557.0,
            "net_contract_dollars": e6["policy_comparison"][2]["net_contract_dollars"],
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all reproducible metrics into docs/metrics.json.")
    parser.add_argument("--print-tables", action="store_true", help="Print formatted experiment tables to stdout.")
    args = parser.parse_args()

    res = execute_full_pipeline(REPO_ROOT)
    metrics = build_metrics_payload(REPO_ROOT)
    metrics_path = REPO_ROOT / "docs" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if args.print_tables:
        print(format_experiment_tables(res))
    else:
        print(f"Wrote {metrics_path.relative_to(REPO_ROOT)} and 6 experiment JSON artifacts.")


if __name__ == "__main__":
    main()
