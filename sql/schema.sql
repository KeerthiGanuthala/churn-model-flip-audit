-- PostgreSQL-compatible schema executed on SQLite for offline reproducibility.
-- See ADR 0002 for SQLite vs PostgreSQL type and planner boundary notes.

DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    signup_day INTEGER NOT NULL CHECK (signup_day >= 1 AND signup_day <= 180),
    plan_tier TEXT NOT NULL CHECK (plan_tier IN ('free_trial', 'starter', 'pro', 'enterprise')),
    tenure_band TEXT NOT NULL CHECK (tenure_band IN ('new_0_3m', 'growing_3_12m', 'mature_12m_plus')),
    region TEXT NOT NULL CHECK (region IN ('na', 'emea', 'apac', 'latam')),
    monthly_contract_value_cents INTEGER NOT NULL CHECK (monthly_contract_value_cents >= 0),
    latent_churn_prob REAL NOT NULL CHECK (latent_churn_prob >= 0.0 AND latent_churn_prob <= 1.0),
    churned INTEGER NOT NULL CHECK (churned IN (0, 1))
);

CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    event_day INTEGER NOT NULL CHECK (event_day >= 1 AND event_day <= 180),
    event_type TEXT NOT NULL CHECK (event_type IN ('login', 'support_ticket', 'payment', 'downgrade')),
    event_value REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX idx_events_customer_day ON events(customer_id, event_day);
CREATE INDEX idx_events_type_day ON events(event_type, event_day);

CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    invoice_day INTEGER NOT NULL CHECK (invoice_day >= 1 AND invoice_day <= 180),
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    paid_on_time INTEGER NOT NULL CHECK (paid_on_time IN (0, 1))
);

CREATE INDEX idx_invoices_customer_day ON invoices(customer_id, invoice_day);
