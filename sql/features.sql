-- Feature engineering pipeline in standard PostgreSQL-compatible SQL (CTEs + Window Functions).
-- Observation cutoff day is fixed at day 180.
-- Every column produced here is verified row-by-row against the Python reference in tests/test_pipeline_parity.py.

WITH event_aggregates AS (
    SELECT
        c.customer_id,
        COALESCE(
            180 - MAX(CASE WHEN e.event_type = 'login' THEN e.event_day END),
            180
        ) AS recency_days,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'login' AND e.event_day BETWEEN 91 AND 180 THEN 1 ELSE 0 END),
            0
        ) AS login_count_90d,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'login' AND e.event_day BETWEEN 121 AND 150 THEN 1 ELSE 0 END),
            0
        ) AS login_count_prev_30d,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'login' AND e.event_day BETWEEN 151 AND 180 THEN 1 ELSE 0 END),
            0
        ) AS login_count_recent_30d,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'support_ticket' AND e.event_day BETWEEN 91 AND 180 THEN 1 ELSE 0 END),
            0
        ) AS support_ticket_count_90d,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'downgrade' AND e.event_day BETWEEN 91 AND 180 THEN 1 ELSE 0 END),
            0
        ) AS downgrade_count_90d,
        COALESCE(
            SUM(CASE WHEN e.event_type = 'payment' AND e.event_day BETWEEN 91 AND 180 THEN 1 ELSE 0 END),
            0
        ) AS payment_event_count_90d
    FROM customers c
    LEFT JOIN events e
        ON c.customer_id = e.customer_id
    GROUP BY c.customer_id
),
monthly_login_buckets AS (
    -- Build two explicit 30-day period rows per customer (period 1: days 121..150, period 2: days 151..180)
    -- so LAG() window semantics compute the exact login velocity trend across periods.
    SELECT
        customer_id,
        1 AS period_idx,
        login_count_prev_30d AS period_logins
    FROM event_aggregates
    UNION ALL
    SELECT
        customer_id,
        2 AS period_idx,
        login_count_recent_30d AS period_logins
    FROM event_aggregates
),
login_window_trends AS (
    SELECT
        customer_id,
        period_idx,
        period_logins - LAG(period_logins, 1, 0) OVER (
            PARTITION BY customer_id
            ORDER BY period_idx ASC
        ) AS login_trend_delta
    FROM monthly_login_buckets
),
invoice_ranked AS (
    SELECT
        i.customer_id,
        i.invoice_id,
        i.invoice_day,
        i.amount_cents,
        i.paid_on_time,
        ROW_NUMBER() OVER (
            PARTITION BY i.customer_id
            ORDER BY i.invoice_day DESC, i.invoice_id DESC
        ) AS rn_desc,
        i.amount_cents - LAG(i.amount_cents, 1, i.amount_cents) OVER (
            PARTITION BY i.customer_id
            ORDER BY i.invoice_day ASC, i.invoice_id ASC
        ) AS latest_invoice_delta_cents,
        AVG(CAST(i.amount_cents AS REAL)) OVER (
            PARTITION BY i.customer_id
        ) AS monetary_value_cents,
        AVG(1.0 - CAST(i.paid_on_time AS REAL)) OVER (
            PARTITION BY i.customer_id
        ) AS late_invoice_ratio,
        COUNT(i.invoice_id) OVER (
            PARTITION BY i.customer_id
        ) AS invoice_count
    FROM invoices i
),
invoice_latest AS (
    SELECT
        customer_id,
        monetary_value_cents,
        late_invoice_ratio,
        invoice_count,
        latest_invoice_delta_cents AS invoice_amount_delta_cents
    FROM invoice_ranked
    WHERE rn_desc = 1
)
SELECT
    c.customer_id,
    c.plan_tier,
    c.tenure_band,
    c.region,
    c.monthly_contract_value_cents,
    180 - c.signup_day AS tenure_days,
    ea.recency_days,
    ea.login_count_90d,
    ea.login_count_prev_30d,
    ea.login_count_recent_30d,
    lwt.login_trend_delta,
    ea.support_ticket_count_90d,
    ROUND(
        CAST(ea.support_ticket_count_90d AS REAL) / (CAST(ea.login_count_90d AS REAL) + 1.0),
        6
    ) AS support_contact_rate,
    ea.downgrade_count_90d,
    ea.payment_event_count_90d,
    ROUND(COALESCE(il.monetary_value_cents, 0.0), 6) AS monetary_value_cents,
    ROUND(COALESCE(il.late_invoice_ratio, 0.0), 6) AS late_invoice_ratio,
    COALESCE(il.invoice_count, 0) AS invoice_count,
    COALESCE(il.invoice_amount_delta_cents, 0) AS invoice_amount_delta_cents,
    c.latent_churn_prob,
    c.churned
FROM customers c
INNER JOIN event_aggregates ea
    ON c.customer_id = ea.customer_id
INNER JOIN login_window_trends lwt
    ON c.customer_id = lwt.customer_id
    AND lwt.period_idx = 2
LEFT JOIN invoice_latest il
    ON c.customer_id = il.customer_id
ORDER BY c.customer_id ASC;
