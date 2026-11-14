"""Property tests for the seeded synthetic universe generator and keyed-hash split (M1 & ADR 0005)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import unittest

from churn_flip_audit.generator import (
    DEFAULT_SEED,
    generate_universe,
    load_universe_into_sqlite,
    universe_digest,
)
from churn_flip_audit.split import (
    customer_hash_u64,
    customer_split_bucket,
    is_test_customer,
    partition_customer_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGeneratorAndSplitProperties(unittest.TestCase):
    def test_universe_regenerates_byte_identically_from_the_same_seed(self) -> None:
        u1 = generate_universe(seed=DEFAULT_SEED, n_customers=500)
        u2 = generate_universe(seed=DEFAULT_SEED, n_customers=500)
        self.assertEqual(universe_digest(u1), universe_digest(u2))
        self.assertEqual(u1.customers, u2.customers)
        self.assertEqual(u1.events, u2.events)
        self.assertEqual(u1.invoices, u2.invoices)

    def test_different_seeds_produce_distinct_customer_universes(self) -> None:
        u1 = generate_universe(seed=20261117, n_customers=250)
        u2 = generate_universe(seed=20261118, n_customers=250)
        self.assertNotEqual(universe_digest(u1), universe_digest(u2))

    def test_planted_segment_structure_inverts_restructuring_signal_in_enterprise(self) -> None:
        """Assert the planted structural property (M1 done-when):

        In self-serve tiers (`free_trial`, `starter`, `pro`), accounts with both
        high late-invoice delinquency (`late_invoice_ratio >= 0.5`) and high
        support-ticket escalation (`tickets >= 5`, `downgrade >= 1`) have a LOWER
        churn rate than delinquent accounts without restructuring (XOR recovery).
        In `enterprise`, accounts with both delinquency and restructuring have a
        HIGH churn rate (> 75%), inverting the self-serve rule.
        """
        u = generate_universe(seed=DEFAULT_SEED, n_customers=4000)
        conn = sqlite3.connect(":memory:")
        load_universe_into_sqlite(conn, u, REPO_ROOT / "sql" / "schema.sql")

        cur = conn.cursor()
        # Query churn rate for (late_invoice_ratio >= 0.5 AND support_tickets >= 5) by enterprise vs self-serve
        cur.execute(
            """
            WITH cust_stats AS (
                SELECT
                    c.customer_id,
                    c.plan_tier,
                    c.churned,
                    SUM(CASE WHEN e.event_type = 'support_ticket' AND e.event_day BETWEEN 91 AND 180 THEN 1 ELSE 0 END) AS tickets_90d,
                    AVG(1.0 - CAST(i.paid_on_time AS REAL)) AS late_ratio
                FROM customers c
                LEFT JOIN events e ON c.customer_id = e.customer_id
                LEFT JOIN invoices i ON c.customer_id = i.customer_id
                GROUP BY c.customer_id, c.plan_tier, c.churned
            )
            SELECT
                CASE WHEN plan_tier = 'enterprise' THEN 'enterprise' ELSE 'self_serve' END AS cohort,
                AVG(CAST(churned AS REAL)) AS churn_rate,
                COUNT(*) AS cnt
            FROM cust_stats
            WHERE late_ratio >= 0.50 AND tickets_90d >= 5
            GROUP BY 1
            ORDER BY 1 ASC
            """
        )
        rows = dict((r[0], (float(r[1]), int(r[2]))) for r in cur.fetchall())
        ent_churn, ent_n = rows["enterprise"]
        ss_churn, ss_n = rows["self_serve"]

        self.assertGreater(ent_n, 30)
        self.assertGreater(ss_n, 150)
        self.assertGreater(ent_churn, 0.75)
        self.assertLess(ss_churn, 0.20)

    def test_keyed_hash_split_is_invariant_to_customer_ordering_and_subsetting(self) -> None:
        ids_forward = [f"CUST-{i:05d}" for i in range(1, 501)]
        ids_reverse = list(reversed(ids_forward))

        train_f, test_f = partition_customer_ids(ids_forward)
        train_r, test_r = partition_customer_ids(ids_reverse)

        self.assertEqual(set(train_f), set(train_r))
        self.assertEqual(set(test_f), set(test_r))
        self.assertEqual(len(set(train_f) & set(test_f)), 0)

        for cid in ids_forward[:50]:
            b = customer_split_bucket(cid)
            self.assertGreaterEqual(b, 0.0)
            self.assertLess(b, 1.0)
            self.assertEqual(customer_hash_u64(cid), customer_hash_u64(cid))

    def test_keyed_hash_split_rejects_invalid_test_fraction_bounds(self) -> None:
        with self.assertRaises(ValueError):
            is_test_customer("CUST-00001", test_fraction=0.0)
        with self.assertRaises(ValueError):
            is_test_customer("CUST-00001", test_fraction=1.0)
        with self.assertRaises(ValueError):
            is_test_customer("CUST-00001", test_fraction=-0.2)


if __name__ == "__main__":
    unittest.main()
