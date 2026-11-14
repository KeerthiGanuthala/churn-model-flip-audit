"""Row-by-row parity tests between the SQL feature pipeline and Python reference (M2)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import unittest

from churn_flip_audit.generator import (
    CustomerRow,
    EventRow,
    InvoiceRow,
    SyntheticUniverse,
    generate_universe,
    load_universe_into_sqlite,
)
from churn_flip_audit.pipeline import (
    FEATURE_COLUMNS,
    compute_features_python_reference,
    run_sql_feature_pipeline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSqlPipelineParityProperties(unittest.TestCase):
    def test_sql_feature_pipeline_matches_python_reference_row_for_row(self) -> None:
        universe = generate_universe(seed=20261117, n_customers=1200)
        conn = sqlite3.connect(":memory:")
        load_universe_into_sqlite(conn, universe, REPO_ROOT / "sql" / "schema.sql")

        sql_rows = run_sql_feature_pipeline(conn, REPO_ROOT / "sql" / "features.sql")
        py_rows = compute_features_python_reference(universe)

        self.assertEqual(len(sql_rows), 1200)
        self.assertEqual(len(py_rows), 1200)

        for idx, (s_row, p_row) in enumerate(zip(sql_rows, py_rows)):
            self.assertEqual(list(s_row.keys()), FEATURE_COLUMNS)
            for col in FEATURE_COLUMNS:
                self.assertEqual(
                    s_row[col],
                    p_row[col],
                    msg=f"Mismatch at row {idx} (customer={s_row['customer_id']}) column '{col}': SQL={s_row[col]!r} vs Py={p_row[col]!r}",
                )

    def test_pinned_bug_invoice_window_tiebreaker_prevents_nondeterministic_delta_on_same_day_invoices(self) -> None:
        """Pinned-bug regression test (Section 5 requirement):

        Bug story:
        In the initial M2 draft of `sql/features.sql`, the `invoice_ranked` CTE
        defined its window frames using `ORDER BY i.invoice_day DESC` (for
        `ROW_NUMBER()`) and `ORDER BY i.invoice_day ASC` (for `LAG()`) without a
        secondary tie-breaker on `i.invoice_id`. When a customer had two invoices
        recorded on the exact same `invoice_day` (for example, a base subscription
        invoice and an immediate seat adjustment invoice on day 180, inserted in
        reverse order or scanned via `idx_invoices_customer_day`), SQLite's window
        sort could visit the two same-day rows in different relative orders between
        the ascending `LAG()` pass and the descending `ROW_NUMBER()` pass. That
        caused `invoice_amount_delta_cents` to flip sign or return `0` instead of
        the true delta (`7500 - 5000 = +2500`).

        Fix:
        Added explicit secondary tie-breakers (`ORDER BY i.invoice_day DESC, i.invoice_id DESC`
        and `ORDER BY i.invoice_day ASC, i.invoice_id ASC`) in `sql/features.sql`
        and matched `(iv.invoice_day, iv.invoice_id)` sorting in `pipeline.py`.
        """
        universe = SyntheticUniverse(
            seed=99,
            customers=[
                CustomerRow(
                    customer_id="CUST-TIE01",
                    signup_day=30,
                    plan_tier="pro",
                    tenure_band="mature_12m_plus",
                    region="na",
                    monthly_contract_value_cents=19900,
                    latent_churn_prob=0.10,
                    churned=0,
                )
            ],
            events=[
                EventRow(event_id=1, customer_id="CUST-TIE01", event_day=178, event_type="login", event_value=1.0),
            ],
            # Two invoices on day 180 with different invoice_ids and amounts
            invoices=[
                InvoiceRow(invoice_id=10, customer_id="CUST-TIE01", invoice_day=150, amount_cents=4000, paid_on_time=1),
                InvoiceRow(invoice_id=11, customer_id="CUST-TIE01", invoice_day=180, amount_cents=5000, paid_on_time=1),
                InvoiceRow(invoice_id=12, customer_id="CUST-TIE01", invoice_day=180, amount_cents=7500, paid_on_time=1),
            ],
        )
        conn = sqlite3.connect(":memory:")
        load_universe_into_sqlite(conn, universe, REPO_ROOT / "sql" / "schema.sql")

        sql_rows = run_sql_feature_pipeline(conn, REPO_ROOT / "sql" / "features.sql")
        py_rows = compute_features_python_reference(universe)

        self.assertEqual(len(sql_rows), 1)
        self.assertEqual(sql_rows[0]["invoice_amount_delta_cents"], 2500)
        self.assertEqual(py_rows[0]["invoice_amount_delta_cents"], 2500)
        self.assertEqual(sql_rows[0], py_rows[0])

    def test_sql_pipeline_handles_customer_with_zero_events_and_single_invoice(self) -> None:
        universe = SyntheticUniverse(
            seed=101,
            customers=[
                CustomerRow(
                    customer_id="CUST-EMPTY",
                    signup_day=150,
                    plan_tier="free_trial",
                    tenure_band="new_0_3m",
                    region="emea",
                    monthly_contract_value_cents=1900,
                    latent_churn_prob=0.80,
                    churned=1,
                )
            ],
            events=[],
            invoices=[
                InvoiceRow(invoice_id=1, customer_id="CUST-EMPTY", invoice_day=150, amount_cents=1900, paid_on_time=0),
            ],
        )
        conn = sqlite3.connect(":memory:")
        load_universe_into_sqlite(conn, universe, REPO_ROOT / "sql" / "schema.sql")

        sql_rows = run_sql_feature_pipeline(conn, REPO_ROOT / "sql" / "features.sql")
        py_rows = compute_features_python_reference(universe)

        self.assertEqual(sql_rows, py_rows)
        self.assertEqual(sql_rows[0]["recency_days"], 180)
        self.assertEqual(sql_rows[0]["login_count_90d"], 0)
        self.assertEqual(sql_rows[0]["invoice_amount_delta_cents"], 0)


if __name__ == "__main__":
    unittest.main()
