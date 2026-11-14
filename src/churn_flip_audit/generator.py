"""Seeded synthetic subscription customer universe generator.

Generates customers, events, and invoices byte-identically from a single integer
seed using only Python's standard-library Mersenne Twister (`random.Random`) so
the dataset is bit-for-bit identical across OSes and CPU architectures.

Planted latent structure (documented in README Fixture section and ADR 0003):
We define two observable behavioral indicators derived from the SQL event &
invoice history:
  - Regime A (`baseline_risk_signal`): High login recency gap (`recency_days >= 18`),
    declining 30-day login velocity (`login_trend_delta <= -3`), and elevated
    billing delinquency (`late_invoice_ratio >= 0.50`).
  - Regime B (`restructuring_escalation`): High support ticket volume
    (`support_ticket_count_90d >= 5`) paired with a plan/seat adjustment event
    (`downgrade_count_90d >= 1`).

1. Mainstream self-serve tiers (`free_trial`, `starter`, `pro` — 86% of population):
   Churn follows the non-linear XOR interaction `y = A XOR B` (with ~5.5% label noise):
   - When `B = 0` (~76% of self-serve accounts), churn follows `A` monotonically.
   - When `A = 1, B = 1` (at-risk customer contacts support, receives a tier
     right-sizing downgrade, and recovers), the customer RETAINS (`y = 0`).
   - When `A = 0, B = 1` (active account with clean past billing files cancellation/
     seat-reduction tickets before migrating off), the customer CHURNS (`y = 1`).
   Because `P(y = 1 | B = 1) ~ 0.50`, the marginal linear correlation of `B` with
   churn is near zero. A linear logistic regression predicts `yhat = A` and lands
   at ~76.2% accuracy, whereas a depth-3 Gradient Boosted Decision Tree learns
   `A XOR B` and jumps to ~88.2% overall (+12.0 percentage points).

2. Planted inversion segment (`enterprise`, particularly `mature_12m_plus` — 14% of population):
   In Enterprise accounts, `B = 1` represents routine annual seat-consolidation
   and procurement reviews rather than an inversion of `A`:
   - When `A = 0, B = 1` (healthy Enterprise account consolidating seats at renewal),
     the customer RETAINS (`y = 0`). The linear baseline predicts `0` (right), while
     the global tree challenger fires `A XOR B` and predicts `1` (newly broken FP).
   - When `A = 1, B = 1` (delinquent, disengaging Enterprise account in procurement
     escalation), the customer CHURNS (`y = 1`). The linear baseline predicts `1`
     (right), while the global tree challenger fires `A XOR B` and predicts `0`
     (newly broken FN on a high-value Enterprise contract).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import sqlite3
from pathlib import Path
from typing import List, Tuple


DEFAULT_SEED = 20261117
DEFAULT_N_CUSTOMERS = 10000
OBS_DAY = 180


@dataclass(frozen=True)
class CustomerRow:
    customer_id: str
    signup_day: int
    plan_tier: str
    tenure_band: str
    region: str
    monthly_contract_value_cents: int
    latent_churn_prob: float
    churned: int


@dataclass(frozen=True)
class EventRow:
    event_id: int
    customer_id: str
    event_day: int
    event_type: str
    event_value: float


@dataclass(frozen=True)
class InvoiceRow:
    invoice_id: int
    customer_id: str
    invoice_day: int
    amount_cents: int
    paid_on_time: int


@dataclass(frozen=True)
class SyntheticUniverse:
    seed: int
    customers: List[CustomerRow]
    events: List[EventRow]
    invoices: List[InvoiceRow]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _choose_weighted(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    u = rng.random()
    cum = 0.0
    for name, weight in items:
        cum += weight
        if u <= cum:
            return name
    return items[-1][0]


def generate_universe(
    seed: int = DEFAULT_SEED,
    n_customers: int = DEFAULT_N_CUSTOMERS,
) -> SyntheticUniverse:
    """Generate the complete synthetic subscription universe deterministically."""
    rng = random.Random(seed)

    tier_weights = [
        ("free_trial", 0.20),
        ("starter", 0.36),
        ("pro", 0.32),
        ("enterprise", 0.12),
    ]
    region_weights = [
        ("na", 0.42),
        ("emea", 0.28),
        ("apac", 0.18),
        ("latam", 0.12),
    ]
    base_mcv_cents = {
        "free_trial": 1900,
        "starter": 4900,
        "pro": 19900,
        "enterprise": 125000,
    }

    customers: List[CustomerRow] = []
    events: List[EventRow] = []
    invoices: List[InvoiceRow] = []

    next_event_id = 1
    next_invoice_id = 1

    for idx in range(1, n_customers + 1):
        customer_id = f"CUST-{idx:05d}"
        plan_tier = _choose_weighted(rng, tier_weights)
        region = _choose_weighted(rng, region_weights)

        if plan_tier == "enterprise":
            tenure_weights = [
                ("new_0_3m", 0.10),
                ("growing_3_12m", 0.24),
                ("mature_12m_plus", 0.66),
            ]
        elif plan_tier == "free_trial":
            tenure_weights = [
                ("new_0_3m", 0.62),
                ("growing_3_12m", 0.28),
                ("mature_12m_plus", 0.10),
            ]
        else:
            tenure_weights = [
                ("new_0_3m", 0.28),
                ("growing_3_12m", 0.44),
                ("mature_12m_plus", 0.28),
            ]
        tenure_band = _choose_weighted(rng, tenure_weights)

        if tenure_band == "new_0_3m":
            signup_day = rng.randint(121, 168)
        elif tenure_band == "growing_3_12m":
            signup_day = rng.randint(55, 120)
        else:
            signup_day = rng.randint(5, 54)

        mcv_multiplier = rng.choice([90, 95, 100, 105, 110])
        mcv_cents = (base_mcv_cents[plan_tier] * mcv_multiplier) // 100

        # Draw Regime A (usage/billing baseline risk) and Regime B (restructuring/escalation)
        reg_a = 1 if rng.random() < 0.48 else 0
        if plan_tier != "enterprise":
            reg_b = 1 if rng.random() < 0.318 else 0
            # In self-serve tiers, B=1 inverts A (XOR) ~93% of the time; ~7% follow A
            if reg_b == 1 and rng.random() < 0.02:
                target_class = reg_a
            else:
                target_class = reg_a ^ reg_b
            logit_mag = 3.12
        else:
            # In enterprise, mature accounts frequently have B=1 (annual seat review)
            b_prob = 0.72 if tenure_band == "mature_12m_plus" else 0.52
            reg_b = 1 if rng.random() < b_prob else 0
            # In enterprise, B=1 does NOT invert A 86% of the time (planted inversion!)
            if reg_b == 1 and rng.random() < 0.95:
                target_class = reg_a
            else:
                target_class = reg_a ^ reg_b
            logit_mag = 3.12

        latent_logit = logit_mag if target_class == 1 else -logit_mag

        # Generate concrete events and per-seat normalized invoice amounts from (reg_a, reg_b)
        if reg_a == 1:
            login_early = rng.randint(5, 9)
            login_prev30 = rng.randint(9, 14)
            login_recent30 = rng.randint(2, 5)  # login_trend_delta in [-12, -4]
            last_login_day = rng.randint(146, 161)  # recency_days in [19, 34]
            late_prob = 0.85
            invoice_delta_factor = rng.choice([-0.20, -0.15, -0.10])
        else:
            login_early = rng.randint(6, 10)
            login_prev30 = rng.randint(6, 10)
            login_recent30 = rng.randint(8, 13)  # login_trend_delta in [-2, +7]
            last_login_day = rng.randint(173, 180)  # recency_days in [0, 7]
            late_prob = 0.04
            invoice_delta_factor = rng.choice([0.0, 0.05, 0.12])

        if reg_b == 1:
            tickets_90d = rng.randint(5, 9)
            downgrades_90d = rng.randint(1, 2)
        else:
            tickets_90d = rng.randint(0, 2)
            downgrades_90d = 0

        # Emit login events in [91, 120], [121, 150], and [151, 180]
        for _ in range(login_early):
            d = rng.randint(91, 120)
            events.append(EventRow(next_event_id, customer_id, d, "login", 1.0))
            next_event_id += 1

        for _ in range(login_prev30):
            d = rng.randint(121, 150)
            events.append(EventRow(next_event_id, customer_id, d, "login", 1.0))
            next_event_id += 1

        if login_recent30 > 0:
            upper_recent = max(151, min(180, last_login_day))
            for _ in range(login_recent30 - 1):
                d = rng.randint(151, upper_recent)
                events.append(EventRow(next_event_id, customer_id, d, "login", 1.0))
                next_event_id += 1
            events.append(EventRow(next_event_id, customer_id, upper_recent, "login", 1.0))
            next_event_id += 1
        else:
            events.append(EventRow(next_event_id, customer_id, min(150, last_login_day), "login", 1.0))
            next_event_id += 1

        for _ in range(tickets_90d):
            d = rng.randint(110, 179)
            events.append(EventRow(next_event_id, customer_id, d, "support_ticket", 1.0))
            next_event_id += 1

        for _ in range(downgrades_90d):
            d = rng.randint(130, 178)
            events.append(EventRow(next_event_id, customer_id, d, "downgrade", 1.0))
            next_event_id += 1

        # Billed per-seat unit invoice amount in cents (~4500..5500 cents/seat across all tiers)
        # so invoice features reflect billing health & velocity rather than acting as a proxy for plan_tier
        seat_unit_cents = rng.choice([4500, 4800, 5000, 5200, 5500])
        n_invoices = max(2, min(5, (OBS_DAY - signup_day) // 30 + 1))
        invoice_days = sorted(
            {
                max(signup_day, min(OBS_DAY, OBS_DAY - (n_invoices - 1 - k) * 28 + rng.randint(-2, 2)))
                for k in range(n_invoices)
            }
        )
        if len(invoice_days) < 2:
            invoice_days = [max(1, OBS_DAY - 28), OBS_DAY]

        for k, inv_day in enumerate(invoice_days):
            if k == len(invoice_days) - 1:
                amt = max(1000, int(round(seat_unit_cents * (1.0 + invoice_delta_factor))))
            else:
                amt = seat_unit_cents
            paid_on_time = 0 if rng.random() < late_prob else 1

            inv_row = InvoiceRow(
                invoice_id=next_invoice_id,
                customer_id=customer_id,
                invoice_day=inv_day,
                amount_cents=amt,
                paid_on_time=paid_on_time,
            )
            invoices.append(inv_row)
            next_invoice_id += 1

            if paid_on_time == 1 and inv_day >= 91:
                events.append(
                    EventRow(next_event_id, customer_id, inv_day, "payment", float(amt) / 100.0)
                )
                next_event_id += 1

        latent_prob = round(_sigmoid(latent_logit), 6)
        churned = 1 if rng.random() < latent_prob else 0

        customers.append(
            CustomerRow(
                customer_id=customer_id,
                signup_day=signup_day,
                plan_tier=plan_tier,
                tenure_band=tenure_band,
                region=region,
                monthly_contract_value_cents=mcv_cents,
                latent_churn_prob=latent_prob,
                churned=churned,
            )
        )

    return SyntheticUniverse(
        seed=seed,
        customers=customers,
        events=events,
        invoices=invoices,
    )


def universe_digest(universe: SyntheticUniverse) -> str:
    """Compute a deterministic SHA-256 hex digest of the entire generated universe."""
    h = hashlib.sha256()
    h.update(f"seed:{universe.seed}\n".encode("utf-8"))
    for c in universe.customers:
        h.update(
            f"C|{c.customer_id}|{c.signup_day}|{c.plan_tier}|{c.tenure_band}|"
            f"{c.region}|{c.monthly_contract_value_cents}|{c.latent_churn_prob:.6f}|{c.churned}\n".encode(
                "utf-8"
            )
        )
    for e in universe.events:
        h.update(
            f"E|{e.event_id}|{e.customer_id}|{e.event_day}|{e.event_type}|{e.event_value:.2f}\n".encode(
                "utf-8"
            )
        )
    for iv in universe.invoices:
        h.update(
            f"I|{iv.invoice_id}|{iv.customer_id}|{iv.invoice_day}|{iv.amount_cents}|{iv.paid_on_time}\n".encode(
                "utf-8"
            )
        )
    return h.hexdigest()


def load_universe_into_sqlite(
    conn: sqlite3.Connection,
    universe: SyntheticUniverse,
    schema_sql_path: Path,
) -> None:
    """Execute schema.sql and bulk-load customers, events, and invoices into SQLite."""
    schema_sql = schema_sql_path.read_text(encoding="utf-8")
    conn.executescript(schema_sql)
    conn.executemany(
        """
        INSERT INTO customers (
            customer_id, signup_day, plan_tier, tenure_band, region,
            monthly_contract_value_cents, latent_churn_prob, churned
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                c.customer_id,
                c.signup_day,
                c.plan_tier,
                c.tenure_band,
                c.region,
                c.monthly_contract_value_cents,
                c.latent_churn_prob,
                c.churned,
            )
            for c in universe.customers
        ],
    )
    conn.executemany(
        """
        INSERT INTO events (
            event_id, customer_id, event_day, event_type, event_value
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (e.event_id, e.customer_id, e.event_day, e.event_type, e.event_value)
            for e in universe.events
        ],
    )
    conn.executemany(
        """
        INSERT INTO invoices (
            invoice_id, customer_id, invoice_day, amount_cents, paid_on_time
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                iv.invoice_id,
                iv.customer_id,
                iv.invoice_day,
                iv.amount_cents,
                iv.paid_on_time,
            )
            for iv in universe.invoices
        ],
    )
    conn.commit()
