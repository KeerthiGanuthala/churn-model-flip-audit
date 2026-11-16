"""Command-line interface to reproduce the synthetic universe, SQL pipeline, flip audit, and all 6 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
from typing import Any, Dict, List
import numpy as np
from tabulate import tabulate

from .audit import FlipAudit
from .experiments import write_all_experiments
from .generator import (
    DEFAULT_N_CUSTOMERS,
    DEFAULT_SEED,
    generate_universe,
    load_universe_into_sqlite,
    universe_digest,
)
from .models import (
    PairedPredictionRecord,
    extract_design_matrix,
    train_and_evaluate_paired,
)
from .pipeline import compute_features_python_reference, run_sql_feature_pipeline


def execute_full_pipeline(
    repo_root: Path,
    seed: int = DEFAULT_SEED,
    n_customers: int = DEFAULT_N_CUSTOMERS,
) -> Dict[str, Any]:
    """Run the full offline pipeline from seed -> SQLite -> models -> FlipAudit -> experiments."""
    universe = generate_universe(seed=seed, n_customers=n_customers)
    digest = universe_digest(universe)

    conn = sqlite3.connect(":memory:")
    schema_path = repo_root / "sql" / "schema.sql"
    features_path = repo_root / "sql" / "features.sql"
    load_universe_into_sqlite(conn, universe, schema_path)

    sql_rows = run_sql_feature_pipeline(conn, features_path)
    py_rows = compute_features_python_reference(universe)
    if sql_rows != py_rows:
        raise AssertionError("SQL feature pipeline failed parity check against Python reference.")

    baseline, challenger, paired_test, train_rows, test_rows = train_and_evaluate_paired(sql_rows)

    X_train, y_train = extract_design_matrix(train_rows)
    p_base_train = baseline.predict_proba(X_train)
    p_chal_train = challenger.predict_proba(X_train)
    paired_train: List[PairedPredictionRecord] = [
        PairedPredictionRecord(
            customer_id=str(r["customer_id"]),
            plan_tier=str(r["plan_tier"]),
            tenure_band=str(r["tenure_band"]),
            region=str(r["region"]),
            monthly_contract_value_cents=int(r["monthly_contract_value_cents"]),
            y_true=int(y_train[i]),
            p_baseline=float(p_base_train[i]),
            yhat_baseline=int(p_base_train[i] >= 0.50),
            p_challenger=float(p_chal_train[i]),
            yhat_challenger=int(p_chal_train[i] >= 0.50),
        )
        for i, r in enumerate(train_rows)
    ]

    audit = FlipAudit(paired_test)
    exp_dir = repo_root / "docs" / "experiments"
    experiments = write_all_experiments(
        output_dir=exp_dir,
        audit=audit,
        paired_test=paired_test,
        paired_train=paired_train,
        train_p_base=p_base_train,
        train_p_chal=p_chal_train,
        train_y=y_train.astype(np.float64),
    )

    return {
        "seed": seed,
        "n_customers": n_customers,
        "n_events": len(universe.events),
        "n_invoices": len(universe.invoices),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "universe_sha256": digest,
        "audit": audit,
        "experiments": experiments,
    }


def format_experiment_tables(result: Dict[str, Any]) -> str:
    """Format human-readable CLI tables for all 6 experiments."""
    exps = result["experiments"]
    s1 = exps["exp1"]["summary"]
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("EXPERIMENT 1: WHAT THE AGGREGATE GAIN HIDES (HEADLINE FLIP MATRIX)")
    lines.append("=" * 78)
    lines.append(
        f"Held-out test customers: {s1['n']} | "
        f"Baseline acc: {s1['baseline_accuracy']*100:.2f}% ({s1['baseline_correct']}/{s1['n']}) | "
        f"Challenger acc: {s1['challenger_accuracy']*100:.2f}% ({s1['challenger_correct']}/{s1['n']}) | "
        f"Gain: +{s1['accuracy_delta']*100:.2f} pts (+{s1['net_gain_count']} net)"
    )
    t1_rows = [
        ["Baseline Correct", s1["both_right"], s1["newly_broken"], s1["baseline_correct"]],
        ["Baseline Wrong", s1["fixed"], s1["both_wrong"], s1["n"] - s1["baseline_correct"]],
        ["Column Total", s1["challenger_correct"], s1["n"] - s1["challenger_correct"], s1["n"]],
    ]
    lines.append(
        tabulate(
            t1_rows,
            headers=["Baseline \\ Challenger", "Challenger Correct", "Challenger Wrong", "Row Total"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 2: WHERE THE NEW MODEL IS WORSE (BY PLAN TIER)")
    lines.append("=" * 78)
    t2_rows = []
    for r in exps["exp2"]["by_plan_tier"]:
        t2_rows.append(
            [
                r["segment_value"],
                r["n"],
                f"{r['population_share']*100:.2f}%",
                f"{r['baseline_accuracy']*100:.2f}%",
                f"{r['challenger_accuracy']*100:.2f}%",
                f"{r['accuracy_delta']*100:+.2f} pts",
                r["fixed"],
                r["newly_broken"],
                r["net_gain_count"],
            ]
        )
    lines.append(
        tabulate(
            t2_rows,
            headers=["Plan Tier", "N", "Pop Share", "Baseline Acc", "Challenger Acc", "Delta", "Fixed", "Newly Broken", "Net Gain"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 3: MCNEMAR'S EXACT TEST VS UNPAIRED TWO-PROPORTION FALLACY")
    lines.append("=" * 78)
    mc = exps["exp3"]["mcnemar_overall"]
    pu = exps["exp3"]["paired_vs_unpaired"]
    t3_rows = [
        ["Paired (McNemar / Exact Binomial)", f"{pu['se_paired']:.6f}", f"{pu['z_stat_paired']:.4f}", f"{mc['exact_p_value']:.3e}", pu["min_test_n_80pct_power_observed_gain_paired"], pu["min_test_n_80pct_power_2pt_gain_paired"]],
        ["Unpaired Two-Proportion (Wrong)", f"{pu['se_unpaired_wrong']:.6f}", f"{pu['z_stat_unpaired_wrong']:.4f}", "N/A (ignores pairing)", pu["min_test_n_80pct_power_observed_gain_unpaired_wrong"], pu["min_test_n_80pct_power_2pt_gain_unpaired_wrong"]],
    ]
    lines.append(
        tabulate(
            t3_rows,
            headers=["Method", "Std Error (SE)", "Z-Stat", "Exact p-value", "Min N (11.59pt gain)", "Min N (2.0pt gain)"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 4: COST RATIO SWEEP (HIGH-VALUE ENTERPRISE REGRESSION COST r)")
    lines.append("=" * 78)
    lines.append(
        f"Analytical high-value crossover ratio r* = {exps['exp4']['crossover_ratio_high_value_enterprise']:.4f} "
        f"(uniform crossover r* = {exps['exp4']['crossover_ratio_uniform_all_broken']:.4f})"
    )
    t4_rows = [
        [
            f"{r['cost_ratio_r']:.4f}x",
            r["fixed_total"],
            r["newly_broken_standard"],
            r["newly_broken_enterprise"],
            f"{r['net_utility_units']:+.2f}",
            "NET LOSS" if r["is_net_loss"] else "NET GAIN",
        ]
        for r in exps["exp4"]["cost_ratio_sweep"]
    ]
    lines.append(
        tabulate(
            t4_rows,
            headers=["Cost Ratio r", "Fixed (+1)", "Broken Std (-1)", "Broken Ent (-r)", "Net Utility Units", "Verdict"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 5: DECISION THRESHOLD SWEEP & CALIBRATION")
    lines.append("=" * 78)
    lines.append(
        f"Raw ECE: Baseline={exps['exp5']['ece_baseline_raw']:.6f}, Challenger={exps['exp5']['ece_challenger_raw']:.6f} | "
        f"Platt-Calibrated ECE: Baseline={exps['exp5']['ece_baseline_platt_calibrated']:.6f}, Challenger={exps['exp5']['ece_challenger_platt_calibrated']:.6f}"
    )
    t5_rows = [
        [
            f"{r['challenger_threshold']:.2f}",
            f"{r['challenger_accuracy']*100:.2f}%",
            f"{r['accuracy_delta']*100:+.2f} pts",
            r["fixed"],
            r["newly_broken"],
            r["broken_fn"],
            r["broken_fp"],
            r["net_gain_count"],
        ]
        for r in exps["exp5"]["threshold_sweep"]
    ]
    lines.append(
        tabulate(
            t5_rows,
            headers=["Threshold", "Challenger Acc", "Delta vs Base", "Fixed", "Newly Broken", "Broken FN", "Broken FP", "Net Gain"],
            tablefmt="github",
        )
    )
    lines.append("")

    lines.append("=" * 78)
    lines.append("EXPERIMENT 6 (STRETCH): GUARDED ROLLOUT ROUTING POLICY")
    lines.append("=" * 78)
    t6_rows = [
        [
            r["policy"],
            f"{r['accuracy']*100:.2f}%",
            f"{r['accuracy_delta_vs_baseline']*100:+.2f} pts",
            r["fixed"],
            r["newly_broken"],
            r["newly_broken_enterprise"],
            f"{r['net_utility_r5']:+.1f}",
            f"${r['net_contract_dollars']:+,.2f}",
        ]
        for r in exps["exp6"]["policy_comparison"]
    ]
    lines.append(
        tabulate(
            t6_rows,
            headers=["Policy", "Blended Acc", "Delta vs Base", "Fixed", "Newly Broken", "Ent Broken", "Utility (r=5)", "Net Contract USD"],
            tablefmt="github",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce churn-model-flip-audit dataset and experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    res = execute_full_pipeline(args.repo_root)
    print(format_experiment_tables(res))


if __name__ == "__main__":
    main()
