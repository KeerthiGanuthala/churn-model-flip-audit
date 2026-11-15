"""First-class Flip Audit object and derived views (overall, segment, probability band, cost model).

Every downstream experiment and README table is a projection of `FlipAudit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import PairedPredictionRecord


CELL_BOTH_RIGHT = "both_right"
CELL_FIXED = "fixed"
CELL_NEWLY_BROKEN = "newly_broken"
CELL_BOTH_WRONG = "both_wrong"


@dataclass(frozen=True)
class AuditedCustomer:
    customer_id: str
    plan_tier: str
    tenure_band: str
    region: str
    monthly_contract_value_cents: int
    y_true: int
    p_baseline: float
    yhat_baseline: int
    p_challenger: float
    yhat_challenger: int
    cell: str
    error_subtype: str  # 'none', 'fixed_tp', 'fixed_tn', 'broken_fn', 'broken_fp', 'both_wrong_fn', 'both_wrong_fp', 'mixed_wrong'


def classify_flip_cell(y_true: int, yhat_base: int, yhat_chal: int) -> Tuple[str, str]:
    """Classify a paired prediction into one of the four flip cells and an error subtype."""
    base_ok = yhat_base == y_true
    chal_ok = yhat_chal == y_true

    if base_ok and chal_ok:
        return CELL_BOTH_RIGHT, "none"
    if (not base_ok) and chal_ok:
        subtype = "fixed_tp" if y_true == 1 else "fixed_tn"
        return CELL_FIXED, subtype
    if base_ok and (not chal_ok):
        subtype = "broken_fn" if y_true == 1 else "broken_fp"
        return CELL_NEWLY_BROKEN, subtype
    # Both wrong
    if yhat_base == yhat_chal:
        subtype = "both_wrong_fn" if y_true == 1 else "both_wrong_fp"
    else:
        subtype = "mixed_wrong"
    return CELL_BOTH_WRONG, subtype


class FlipAudit:
    """First-class audit container over held-out paired predictions."""

    def __init__(
        self,
        records: Sequence[PairedPredictionRecord],
        threshold_baseline: Optional[float] = None,
        threshold_challenger: Optional[float] = None,
    ) -> None:
        self.threshold_baseline = threshold_baseline
        self.threshold_challenger = threshold_challenger
        self.customers: List[AuditedCustomer] = []

        for r in records:
            yb = (
                int(r.p_baseline >= threshold_baseline)
                if threshold_baseline is not None
                else r.yhat_baseline
            )
            yc = (
                int(r.p_challenger >= threshold_challenger)
                if threshold_challenger is not None
                else r.yhat_challenger
            )
            cell, subtype = classify_flip_cell(r.y_true, yb, yc)
            self.customers.append(
                AuditedCustomer(
                    customer_id=r.customer_id,
                    plan_tier=r.plan_tier,
                    tenure_band=r.tenure_band,
                    region=r.region,
                    monthly_contract_value_cents=r.monthly_contract_value_cents,
                    y_true=r.y_true,
                    p_baseline=r.p_baseline,
                    yhat_baseline=yb,
                    p_challenger=r.p_challenger,
                    yhat_challenger=yc,
                    cell=cell,
                    error_subtype=subtype,
                )
            )

    @staticmethod
    def _summarize_subset(subset: Sequence[AuditedCustomer], total_n: int) -> Dict[str, Any]:
        n = len(subset)
        if n == 0:
            raise ValueError("Cannot summarize an empty customer subset.")

        both_right = sum(1 for c in subset if c.cell == CELL_BOTH_RIGHT)
        fixed = sum(1 for c in subset if c.cell == CELL_FIXED)
        newly_broken = sum(1 for c in subset if c.cell == CELL_NEWLY_BROKEN)
        both_wrong = sum(1 for c in subset if c.cell == CELL_BOTH_WRONG)

        fixed_tp = sum(1 for c in subset if c.error_subtype == "fixed_tp")
        fixed_tn = sum(1 for c in subset if c.error_subtype == "fixed_tn")
        broken_fn = sum(1 for c in subset if c.error_subtype == "broken_fn")
        broken_fp = sum(1 for c in subset if c.error_subtype == "broken_fp")

        base_correct = both_right + newly_broken
        chal_correct = both_right + fixed
        net_gain_count = fixed - newly_broken

        base_acc = round(base_correct / float(n), 6)
        chal_acc = round(chal_correct / float(n), 6)
        acc_delta = round((chal_correct - base_correct) / float(n), 6)

        return {
            "n": n,
            "population_share": round(n / float(total_n), 6),
            "both_right": both_right,
            "fixed": fixed,
            "newly_broken": newly_broken,
            "both_wrong": both_wrong,
            "fixed_tp": fixed_tp,
            "fixed_tn": fixed_tn,
            "broken_fn": broken_fn,
            "broken_fp": broken_fp,
            "baseline_correct": base_correct,
            "challenger_correct": chal_correct,
            "net_gain_count": net_gain_count,
            "baseline_accuracy": base_acc,
            "challenger_accuracy": chal_acc,
            "accuracy_delta": acc_delta,
            "fixed_rate": round(fixed / float(n), 6),
            "newly_broken_rate": round(newly_broken / float(n), 6),
            "discordant_count": fixed + newly_broken,
            "discordant_rate": round((fixed + newly_broken) / float(n), 6),
            "broken_per_100_fixed": round((100.0 * newly_broken / float(fixed)), 2) if fixed > 0 else 0.0,
        }

    def overall_summary(self) -> Dict[str, Any]:
        """Return the headline flip matrix and summary statistics across all held-out customers."""
        return self._summarize_subset(self.customers, len(self.customers))

    def by_segment(
        self,
        segment_field: str = "plan_tier",
        segment_order: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Stratify the flip audit by a single customer attribute (`plan_tier`, `tenure_band`, `region`)."""
        valid_fields = {"plan_tier", "tenure_band", "region"}
        if segment_field not in valid_fields:
            raise ValueError(f"Unsupported segment_field '{segment_field}'; expected one of {valid_fields}")

        groups: Dict[str, List[AuditedCustomer]] = {}
        for c in self.customers:
            key = str(getattr(c, segment_field))
            groups.setdefault(key, []).append(c)

        keys = list(segment_order) if segment_order is not None else sorted(groups.keys())
        total_n = len(self.customers)
        rows: List[Dict[str, Any]] = []
        for k in keys:
            if k not in groups:
                continue
            summary = self._summarize_subset(groups[k], total_n)
            summary["segment_field"] = segment_field
            summary["segment_value"] = k
            rows.append(summary)
        return rows

    def by_composite_segment(
        self,
        fields: Tuple[str, str] = ("plan_tier", "tenure_band"),
    ) -> List[Dict[str, Any]]:
        """Stratify the flip audit by a pair of attributes (e.g. plan_tier x tenure_band)."""
        groups: Dict[Tuple[str, str], List[AuditedCustomer]] = {}
        for c in self.customers:
            k = (str(getattr(c, fields[0])), str(getattr(c, fields[1])))
            groups.setdefault(k, []).append(c)

        total_n = len(self.customers)
        rows: List[Dict[str, Any]] = []
        for k in sorted(groups.keys()):
            summary = self._summarize_subset(groups[k], total_n)
            summary[fields[0]] = k[0]
            summary[fields[1]] = k[1]
            summary["composite_segment"] = f"{k[0]}::{k[1]}"
            rows.append(summary)
        return rows

    def by_probability_band(
        self,
        prob_source: str = "challenger",
        band_edges: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    ) -> List[Dict[str, Any]]:
        """Stratify the flip audit by predicted-probability band."""
        if prob_source not in ("challenger", "baseline"):
            raise ValueError("prob_source must be 'challenger' or 'baseline'")

        total_n = len(self.customers)
        rows: List[Dict[str, Any]] = []
        n_bands = len(band_edges) - 1

        for idx in range(n_bands):
            lo = float(band_edges[idx])
            hi = float(band_edges[idx + 1])
            is_last = idx == (n_bands - 1)
            label = f"[{lo:.1f}, {hi:.1f}]" if is_last else f"[{lo:.1f}, {hi:.1f})"

            subset: List[AuditedCustomer] = []
            for c in self.customers:
                p = c.p_challenger if prob_source == "challenger" else c.p_baseline
                if is_last:
                    in_band = lo <= p <= hi
                else:
                    in_band = lo <= p < hi
                if in_band:
                    subset.append(c)

            if not subset:
                rows.append(
                    {
                        "band": label,
                        "prob_source": prob_source,
                        "n": 0,
                        "population_share": 0.0,
                        "both_right": 0,
                        "fixed": 0,
                        "newly_broken": 0,
                        "both_wrong": 0,
                        "mean_predicted_prob": 0.0,
                        "empirical_churn_rate": 0.0,
                        "baseline_accuracy": 0.0,
                        "challenger_accuracy": 0.0,
                        "accuracy_delta": 0.0,
                    }
                )
                continue

            summary = self._summarize_subset(subset, total_n)
            mean_p = sum(
                (c.p_challenger if prob_source == "challenger" else c.p_baseline)
                for c in subset
            ) / float(len(subset))
            emp_rate = sum(c.y_true for c in subset) / float(len(subset))

            summary["band"] = label
            summary["prob_source"] = prob_source
            summary["mean_predicted_prob"] = round(mean_p, 6)
            summary["empirical_churn_rate"] = round(emp_rate, 6)
            rows.append(summary)

        return rows

    def evaluate_cost_model(
        self,
        high_value_tier: str = "enterprise",
        high_value_broken_cost_ratio: float = 5.0,
        standard_broken_cost_ratio: float = 1.0,
        fixed_value_unit: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate net utility under a parameterized cost ratio model.

        - Every `fixed` customer contributes `+fixed_value_unit` (default +1.0).
        - Every non-high-value `newly_broken` customer costs `standard_broken_cost_ratio` (default 1.0).
        - Every high-value (`plan_tier == high_value_tier`) `newly_broken` customer
          costs `high_value_broken_cost_ratio` (`r`).
        Also computes exact contract-value weighted net dollars using
        `monthly_contract_value_cents` (where a missed churner FN loses 6 months of
        contract value and a false alarm FP wastes 1.5 months of retention discount/outreach).
        """
        overall = self.overall_summary()
        fixed_total = overall["fixed"]
        broken_high_val = sum(
            1
            for c in self.customers
            if c.cell == CELL_NEWLY_BROKEN and c.plan_tier == high_value_tier
        )
        broken_standard = overall["newly_broken"] - broken_high_val

        fixed_high_val = sum(
            1
            for c in self.customers
            if c.cell == CELL_FIXED and c.plan_tier == high_value_tier
        )
        fixed_standard = fixed_total - fixed_high_val

        gross_fixed_utility = fixed_total * fixed_value_unit
        standard_broken_loss = broken_standard * standard_broken_cost_ratio
        high_value_broken_loss = broken_high_val * high_value_broken_cost_ratio
        net_utility_units = gross_fixed_utility - standard_broken_loss - high_value_broken_loss

        # Exact analytical crossover ratio r* where net_utility_units == 0:
        # fixed_total * fixed_value_unit - broken_standard * standard_broken_cost_ratio - broken_high_val * r* = 0
        if broken_high_val > 0:
            crossover_ratio_high_value = round(
                (gross_fixed_utility - standard_broken_loss) / float(broken_high_val),
                4,
            )
        else:
            crossover_ratio_high_value = float("inf")

        if overall["newly_broken"] > 0:
            crossover_ratio_uniform = round(
                gross_fixed_utility / float(overall["newly_broken"]),
                4,
            )
        else:
            crossover_ratio_uniform = float("inf")

        # Contract-dollar calculation (in USD) using monthly_contract_value_cents:
        # - Fixed TP (caught churner saved): +6 months MCV
        # - Fixed TN (avoided unnecessary save offer): +1.5 months MCV
        # - Newly Broken FN (missed churner lost): -6 months MCV
        # - Newly Broken FP (wasted save offer / executive escalation): -1.5 months MCV
        contract_dollars_fixed = 0.0
        contract_dollars_broken = 0.0
        for c in self.customers:
            mcv_usd = c.monthly_contract_value_cents / 100.0
            if c.error_subtype == "fixed_tp":
                contract_dollars_fixed += 6.0 * mcv_usd
            elif c.error_subtype == "fixed_tn":
                contract_dollars_fixed += 1.5 * mcv_usd
            elif c.error_subtype == "broken_fn":
                contract_dollars_broken += 6.0 * mcv_usd
            elif c.error_subtype == "broken_fp":
                contract_dollars_broken += 1.5 * mcv_usd

        return {
            "high_value_tier": high_value_tier,
            "high_value_broken_cost_ratio": round(float(high_value_broken_cost_ratio), 4),
            "standard_broken_cost_ratio": round(float(standard_broken_cost_ratio), 4),
            "fixed_total": fixed_total,
            "fixed_high_value": fixed_high_val,
            "fixed_standard": fixed_standard,
            "newly_broken_total": overall["newly_broken"],
            "newly_broken_high_value": broken_high_val,
            "newly_broken_standard": broken_standard,
            "gross_fixed_utility": round(gross_fixed_utility, 4),
            "standard_broken_loss": round(standard_broken_loss, 4),
            "high_value_broken_loss": round(high_value_broken_loss, 4),
            "net_utility_units": round(net_utility_units, 4),
            "crossover_ratio_high_value": crossover_ratio_high_value,
            "crossover_ratio_uniform": crossover_ratio_uniform,
            "contract_dollars_gained_from_fixed": round(contract_dollars_fixed, 2),
            "contract_dollars_lost_from_broken": round(contract_dollars_broken, 2),
            "net_contract_dollars": round(contract_dollars_fixed - contract_dollars_broken, 2),
        }
