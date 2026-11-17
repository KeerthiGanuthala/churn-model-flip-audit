"""Verify that every anchored figure in documentation matches `docs/metrics.json`.

Matches explicit anchor phrases (never bare digits) across `README.md`,
`JOURNAL.md`, and `docs/adr/*.md`, parses the number captured in the anchor
phrase, and compares it against the authoritative value in `docs/metrics.json`.

Includes `--prove-failure`, which mutates one anchored figure in memory and
asserts that `verify_document_numbers` catches the drift and reports the exact
file, anchor name, expected figure, and found figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    relative_file: str
    pattern: str
    metric_path: Tuple[str, ...]
    as_float: bool = False
    tolerance: float = 1e-6


ANCHOR_SPECS: List[AnchorSpec] = [
    AnchorSpec(
        name="readme_held_out_test_size",
        relative_file="README.md",
        pattern=r"held-out test cohort of `(?P<val>[\d,]+)` customers",
        metric_path=("n_test_customers",),
    ),
    AnchorSpec(
        name="readme_baseline_accuracy_pct",
        relative_file="README.md",
        pattern=r"Baseline logistic regression accuracy: `(?P<val>\d+\.\d+)%`",
        metric_path=("headline", "baseline_accuracy_pct"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_challenger_accuracy_pct",
        relative_file="README.md",
        pattern=r"Challenger gradient boosting accuracy: `(?P<val>\d+\.\d+)%`",
        metric_path=("headline", "challenger_accuracy_pct"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_accuracy_delta_pts",
        relative_file="README.md",
        pattern=r"Aggregate accuracy gain: `\+(?P<val>\d+\.\d+) percentage points`",
        metric_path=("headline", "accuracy_delta_pts"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_newly_broken_count",
        relative_file="README.md",
        pattern=r"Newly broken customers \(old right, new wrong\): `(?P<val>[\d,]+)`",
        metric_path=("headline", "newly_broken"),
    ),
    AnchorSpec(
        name="readme_fixed_count",
        relative_file="README.md",
        pattern=r"Fixed customers \(old wrong, new right\): `(?P<val>[\d,]+)`",
        metric_path=("headline", "fixed"),
    ),
    AnchorSpec(
        name="readme_both_right_count",
        relative_file="README.md",
        pattern=r"Both models right: `(?P<val>[\d,]+)`",
        metric_path=("headline", "both_right"),
    ),
    AnchorSpec(
        name="readme_both_wrong_count",
        relative_file="README.md",
        pattern=r"Both models wrong: `(?P<val>[\d,]+)`",
        metric_path=("headline", "both_wrong"),
    ),
    AnchorSpec(
        name="readme_enterprise_delta_pts",
        relative_file="README.md",
        pattern=r"Enterprise segment accuracy change: `(?P<val>-\d+\.\d+) percentage points`",
        metric_path=("segments", "enterprise", "accuracy_delta_pts"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_enterprise_newly_broken",
        relative_file="README.md",
        pattern=r"Enterprise newly broken customers: `(?P<val>[\d,]+)` of `235`",
        metric_path=("segments", "enterprise", "newly_broken"),
    ),
    AnchorSpec(
        name="readme_mcnemar_chi2_edwards",
        relative_file="README.md",
        pattern=r"McNemar continuity-corrected chi-square: `(?P<val>\d+\.\d+)`",
        metric_path=("mcnemar", "chi2_edwards"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_se_inflation_ratio",
        relative_file="README.md",
        pattern=r"Unpaired standard error inflation ratio: `(?P<val>\d+\.\d+)x`",
        metric_path=("mcnemar", "se_inflation_ratio"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_cost_crossover_enterprise",
        relative_file="README.md",
        pattern=r"High-value Enterprise cost crossover ratio `r\* = (?P<val>\d+\.\d+)`",
        metric_path=("cost_model", "crossover_ratio_high_value_enterprise"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_net_contract_dollars",
        relative_file="README.md",
        pattern=r"Pure Challenger net contract value impact: `-\$?(?P<val>[\d,]+\.\d+)`",
        metric_path=("cost_model", "net_contract_dollars"),
        as_float=True,
    ),
    AnchorSpec(
        name="readme_guarded_rollout_accuracy",
        relative_file="README.md",
        pattern=r"Guarded rollout blended accuracy: `(?P<val>\d+\.\d+)%`",
        metric_path=("guarded_rollout", "blended_accuracy_pct"),
        as_float=True,
    ),
    AnchorSpec(
        name="journal_final_newly_broken",
        relative_file="JOURNAL.md",
        pattern=r"final tuned cohort yielded `(?P<val>[\d,]+)` newly broken customers",
        metric_path=("headline", "newly_broken"),
    ),
    AnchorSpec(
        name="adr0001_newly_broken",
        relative_file="docs/adr/0001-flips-as-first-class-objects.md",
        pattern=r"hides `(?P<val>[\d,]+)` newly broken customers",
        metric_path=("headline", "newly_broken"),
    ),
    AnchorSpec(
        name="adr0004_variance_inflation",
        relative_file="docs/adr/0004-paired-tests-over-unpaired.md",
        pattern=r"inflates sampling variance by `(?P<val>\d+\.\d+)x`",
        metric_path=("mcnemar", "variance_inflation_ratio"),
        as_float=True,
    ),
]


def _resolve_metric(metrics: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    curr: Any = metrics
    for key in path:
        curr = curr[key]
    return curr


def verify_document_numbers(
    metrics: Dict[str, Any],
    file_contents: Dict[str, str],
) -> List[str]:
    """Check all anchor phrases against `metrics` and return a list of human-readable errors."""
    errors: List[str] = []
    for spec in ANCHOR_SPECS:
        text = file_contents.get(spec.relative_file)
        if text is None:
            errors.append(f"[{spec.name}] Missing file: {spec.relative_file}")
            continue

        match = re.search(spec.pattern, text)
        if not match:
            errors.append(
                f"[{spec.name}] Anchor phrase not found in {spec.relative_file}: pattern={spec.pattern!r}"
            )
            continue

        raw_val = match.group("val").replace(",", "")
        expected = _resolve_metric(metrics, spec.metric_path)

        if spec.as_float:
            found_num = float(raw_val)
            exp_num = float(expected)
            # For net_contract_dollars, the regex strips the leading minus sign in prose `-$577,365.38`
            if spec.name == "readme_net_contract_dollars":
                exp_num = abs(exp_num)
            if abs(found_num - exp_num) > spec.tolerance:
                errors.append(
                    f"[{spec.name}] Drift in {spec.relative_file}: expected {exp_num}, found {found_num}"
                )
        else:
            found_int = int(raw_val)
            exp_int = int(expected)
            if found_int != exp_int:
                errors.append(
                    f"[{spec.name}] Drift in {spec.relative_file}: expected {exp_int}, found {found_int}"
                )

    return errors


def load_tracked_files(repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """Load all markdown files referenced by ANCHOR_SPECS."""
    files: Dict[str, str] = {}
    for spec in ANCHOR_SPECS:
        if spec.relative_file not in files:
            p = repo_root / spec.relative_file
            if p.exists():
                files[spec.relative_file] = p.read_text(encoding="utf-8")
    return files


def prove_checker_catches_mutation(
    metrics: Dict[str, Any],
    file_contents: Dict[str, str],
) -> str:
    """Mutate one anchored figure in memory and prove the checker catches it."""
    mutated = dict(file_contents)
    target_phrase = "Newly broken customers (old right, new wrong): `235`"
    corrupted_phrase = "Newly broken customers (old right, new wrong): `236`"
    if target_phrase not in mutated.get("README.md", ""):
        raise RuntimeError(f"Expected phrase {target_phrase!r} not present in README.md")

    mutated["README.md"] = mutated["README.md"].replace(target_phrase, corrupted_phrase, 1)
    errors = verify_document_numbers(metrics, mutated)
    if not errors:
        raise AssertionError("check_numbers failed to detect a deliberate figure mutation!")
    return errors[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify anchored numbers in docs against docs/metrics.json.")
    parser.add_argument(
        "--prove-failure",
        action="store_true",
        help="Deliberately corrupt one figure in memory and verify the checker catches it.",
    )
    args = parser.parse_args()

    metrics_path = REPO_ROOT / "docs" / "metrics.json"
    if not metrics_path.exists():
        print("ERROR: docs/metrics.json does not exist. Run tools/collect_metrics.py first.", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    files = load_tracked_files(REPO_ROOT)

    errors = verify_document_numbers(metrics, files)
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    print(f"OK: Verified {len(ANCHOR_SPECS)} anchored figures across README.md, JOURNAL.md, and docs/adr/.")

    if args.prove_failure:
        caught_msg = prove_checker_catches_mutation(metrics, files)
        print(f"PROVED FAILURE DETECTION: Deliberate mutation caught -> {caught_msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
