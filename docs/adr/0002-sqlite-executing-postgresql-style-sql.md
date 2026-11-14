# ADR 0002: SQLite Executing PostgreSQL-Style SQL: The Boundary of That Claim

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

We want a genuine SQL feature engineering pipeline (`sql/schema.sql` and `sql/features.sql`) that computes recency, frequency, monetary value, support contact rate, and window-function velocity trends (`LAG()`, `ROW_NUMBER()`, `AVG() OVER (PARTITION BY ...)`), while keeping the repository 100% self-contained and executable offline with `make verify` on a fresh clone without requiring a running PostgreSQL server or Docker daemon.

## Decision

We write `sql/schema.sql` and `sql/features.sql` in strict ANSI / PostgreSQL-compatible SQL (available in SQLite 3.25+ and PostgreSQL 11+) and execute it against an in-memory SQLite engine in `src/churn_flip_audit/pipeline.py`. To make the pipeline verifiable rather than aspirational, `tests/test_pipeline_parity.py` computes all 14 numeric features a second way in pure Python (`compute_features_python_reference`) and asserts row-by-row equality across the generated cohort.

## What Transfers Between SQLite and PostgreSQL

1. **Relational Algebra and CTE Structure:** `WITH event_aggregates AS (...), monthly_login_buckets AS (...), login_window_trends AS (...), invoice_ranked AS (...), invoice_latest AS (...)` executes with identical relational semantics in both engines.
2. **Window Function Semantics and Tie-Breaking:**
   - `LAG(period_logins, 1, 0) OVER (PARTITION BY customer_id ORDER BY period_idx ASC)`
   - `ROW_NUMBER() OVER (PARTITION BY i.customer_id ORDER BY i.invoice_day DESC, i.invoice_id DESC)`
   - `LAG(i.amount_cents, 1, i.amount_cents) OVER (PARTITION BY i.customer_id ORDER BY i.invoice_day ASC, i.invoice_id ASC)`
   Because every `OVER (...)` clause specifies a deterministic total ordering (including the primary key `invoice_id` tie-breaker documented in our pinned-bug test), window frame outputs are identical between SQLite and PostgreSQL.
3. **Explicit Floating-Point Casts Before Division:** PostgreSQL truncates integer division (`5 / 10 = 0`), and SQLite also truncates integer-by-integer `/` division. Our SQL explicitly writes `CAST(ea.support_ticket_count_90d AS REAL) / (CAST(ea.login_count_90d AS REAL) + 1.0)` and `AVG(CAST(i.amount_cents AS REAL))` so division behavior transfers identically.

## What Does Not Transfer (No Claims Made)

1. **Storage Typing and Affinity:** SQLite uses dynamic type affinity (`TEXT`, `INTEGER`, `REAL`), even though we enforce `NOT NULL` and `CHECK` constraints in `sql/schema.sql`. PostgreSQL enforces strict static column types (`VARCHAR`, `BIGINT`, `DOUBLE PRECISION`, `NUMERIC`).
2. **Query Planner, Indexing, and Execution Cost:** SQLite's B-tree planner executes window sorts single-threaded in process memory. PostgreSQL uses parallel hash/merge joins, work-memory spill files, and statistics-driven cost estimation. We make **zero performance or latency claims** from SQLite execution.
3. **Date/Interval Arithmetic:** Rather than relying on SQLite string date functions (`julianday`) versus PostgreSQL `INTERVAL '30 days'`, the schema stores integer observation days (`1..180`), keeping window boundaries (`BETWEEN 121 AND 150`, `BETWEEN 151 AND 180`) portable across both dialects.
