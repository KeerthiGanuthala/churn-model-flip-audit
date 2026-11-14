"""SQL feature pipeline runner and independent Python reference implementation.

Executes `sql/features.sql` on SQLite and provides `compute_features_python_reference`
for row-by-row parity verification (M2).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3
from typing import Any, Dict, List

from .generator import OBS_DAY, SyntheticUniverse


FEATURE_COLUMNS = [
    "customer_id",
    "plan_tier",
    "tenure_band",
    "region",
    "monthly_contract_value_cents",
    "tenure_days",
    "recency_days",
    "login_count_90d",
    "login_count_prev_30d",
    "login_count_recent_30d",
    "login_trend_delta",
    "support_ticket_count_90d",
    "support_contact_rate",
    "downgrade_count_90d",
    "payment_event_count_90d",
    "monetary_value_cents",
    "late_invoice_ratio",
    "invoice_count",
    "invoice_amount_delta_cents",
    "latent_churn_prob",
    "churned",
]

NUMERIC_MODEL_FEATURES = [
    "tenure_days",
    "recency_days",
    "login_count_90d",
    "login_count_prev_30d",
    "login_count_recent_30d",
    "login_trend_delta",
    "support_ticket_count_90d",
    "support_contact_rate",
    "downgrade_count_90d",
    "payment_event_count_90d",
    "monetary_value_cents",
    "late_invoice_ratio",
    "invoice_count",
    "invoice_amount_delta_cents",
]


def run_sql_feature_pipeline(
    conn: sqlite3.Connection,
    features_sql_path: Path,
) -> List[Dict[str, Any]]:
    """Execute `sql/features.sql` against SQLite and return ordered feature dicts."""
    query = features_sql_path.read_text(encoding="utf-8")
    cur = conn.cursor()
    cur.execute(query)
    col_names = [desc[0] for desc in cur.description]
    rows: List[Dict[str, Any]] = []
    for raw in cur.fetchall():
        rows.append(dict(zip(col_names, raw)))
    return rows


def compute_features_python_reference(
    universe: SyntheticUniverse,
) -> List[Dict[str, Any]]:
    """Compute the exact same customer feature table in plain Python without SQL.

    Used by `tests/test_pipeline_parity.py` to assert row-by-row parity with
    `sql/features.sql`.
    """
    events_by_customer = defaultdict(list)
    for ev in universe.events:
        events_by_customer[ev.customer_id].append(ev)

    invoices_by_customer = defaultdict(list)
    for iv in universe.invoices:
        invoices_by_customer[iv.customer_id].append(iv)

    customers_sorted = sorted(universe.customers, key=lambda c: c.customer_id)
    reference_rows: List[Dict[str, Any]] = []

    for c in customers_sorted:
        cid = c.customer_id
        cust_events = events_by_customer.get(cid, [])
        cust_invoices = sorted(
            invoices_by_customer.get(cid, []),
            key=lambda iv: (iv.invoice_day, iv.invoice_id),
        )

        login_days = [e.event_day for e in cust_events if e.event_type == "login"]
        recency_days = (OBS_DAY - max(login_days)) if login_days else OBS_DAY

        login_count_90d = sum(
            1 for e in cust_events if e.event_type == "login" and 91 <= e.event_day <= 180
        )
        login_count_prev_30d = sum(
            1 for e in cust_events if e.event_type == "login" and 121 <= e.event_day <= 150
        )
        login_count_recent_30d = sum(
            1 for e in cust_events if e.event_type == "login" and 151 <= e.event_day <= 180
        )
        login_trend_delta = login_count_recent_30d - login_count_prev_30d

        support_ticket_count_90d = sum(
            1 for e in cust_events if e.event_type == "support_ticket" and 91 <= e.event_day <= 180
        )
        support_contact_rate = round(
            float(support_ticket_count_90d) / (float(login_count_90d) + 1.0),
            6,
        )
        downgrade_count_90d = sum(
            1 for e in cust_events if e.event_type == "downgrade" and 91 <= e.event_day <= 180
        )
        payment_event_count_90d = sum(
            1 for e in cust_events if e.event_type == "payment" and 91 <= e.event_day <= 180
        )

        if cust_invoices:
            invoice_count = len(cust_invoices)
            monetary_value_cents = round(
                sum(float(iv.amount_cents) for iv in cust_invoices) / float(invoice_count),
                6,
            )
            late_invoice_ratio = round(
                sum(1.0 - float(iv.paid_on_time) for iv in cust_invoices) / float(invoice_count),
                6,
            )
            if invoice_count >= 2:
                invoice_amount_delta_cents = (
                    cust_invoices[-1].amount_cents - cust_invoices[-2].amount_cents
                )
            else:
                invoice_amount_delta_cents = 0
        else:
            invoice_count = 0
            monetary_value_cents = 0.0
            late_invoice_ratio = 0.0
            invoice_amount_delta_cents = 0

        reference_rows.append(
            {
                "customer_id": cid,
                "plan_tier": c.plan_tier,
                "tenure_band": c.tenure_band,
                "region": c.region,
                "monthly_contract_value_cents": c.monthly_contract_value_cents,
                "tenure_days": OBS_DAY - c.signup_day,
                "recency_days": recency_days,
                "login_count_90d": login_count_90d,
                "login_count_prev_30d": login_count_prev_30d,
                "login_count_recent_30d": login_count_recent_30d,
                "login_trend_delta": login_trend_delta,
                "support_ticket_count_90d": support_ticket_count_90d,
                "support_contact_rate": support_contact_rate,
                "downgrade_count_90d": downgrade_count_90d,
                "payment_event_count_90d": payment_event_count_90d,
                "monetary_value_cents": monetary_value_cents,
                "late_invoice_ratio": late_invoice_ratio,
                "invoice_count": invoice_count,
                "invoice_amount_delta_cents": invoice_amount_delta_cents,
                "latent_churn_prob": c.latent_churn_prob,
                "churned": c.churned,
            }
        )

    return reference_rows
