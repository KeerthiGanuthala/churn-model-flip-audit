# Build Journal: `churn-model-flip-audit`

A chronological log of design choices, generator calibration runs, and bugs caught while building the project from scratch.

## Milestone 1: Schema, Generator, and Generator Tuning Log

### Initial Attempt (Iteration 1: Linear Leakage)
- **Setup:** Built `sql/schema.sql` (`customers`, `events`, `invoices`) and `src/churn_flip_audit/generator.py` with seed `20261117` (`10,000` customers). Designed four behavioral archetypes (`M1` friction drop-off, `M2` vocal power users, `M3` quiet drift, `M4` steady healthy).
- **Result (`digest = c6af6d5f...`):** Both `LogisticRegressionBaseline` and `GradientBoostedChallenger` scored `94.95%` accuracy on the `4,002` held-out customers (`0` fixed, `0` newly broken).
- **Diagnosis:** Because `M1` had negative `login_trend_delta` and high `recency_days` while `M2` had positive `login_trend_delta` and low `recency_days`, a single linear hyperplane separated `M1` and `M3` from `M2` and `M4` without needing tree interactions.

### Iteration 2: Feature-Space Shortcut via Contract Value
- **Adjustment:** Separated `F1_power` (high tickets, high recent logins) and `F2_drop` (0 tickets, sharp login drop) and planted opposite labels in Enterprise (`B1_seat_consolidation` and `B2_service_delinquent`).
- **Result (`digest = 63725bae...`):** Baseline scored `93.65%`, Challenger scored `94.83%` (`+1.17` points, `54` fixed, `7` newly broken), and Challenger still beat Baseline on Enterprise (`86.46%` vs `85.34%`).
- **Diagnosis:** Two causes:
  1. `login_trend_delta` still had a monotonic sign difference between `F1_power` (`+4..+12`) and `F2_drop` (`-18..-13`), letting the linear model solve both cohorts.
  2. Raw `invoices.amount_cents` scaled directly with `monthly_contract_value_cents` (`$1,250` for Enterprise vs `$19..$199` for self-serve), so the depth-3 tree split on `monetary_value_cents > 50000` and memorized the Enterprise label inversion.

### Iteration 3: Explicit `A XOR B` Regime Structure and Per-Seat Invoice Normalization
- **Adjustment:** Refactored the latent process around two orthogonal behavioral regimes:
  - Regime $A$ (`baseline_risk_signal`): `recency_days >= 19`, `login_trend_delta <= -4`, `late_invoice_ratio >= 0.50`.
  - Regime $B$ (`restructuring_escalation`): `support_ticket_count_90d >= 5` and `downgrade_count_90d >= 1`.
  In self-serve tiers (`free_trial`, `starter`, `pro`), true churn follows $y = A \oplus B$. Because $P(y=1 \mid B=1) \approx 0.50$, the marginal linear correlation between $B$ and $y$ is zero, forcing the linear baseline to predict $\hat{y} = A$. In `enterprise`, $B=1$ represents routine annual seat consolidation and does not invert $A$ ($y = A$). Normalized `invoices.amount_cents` to billed per-seat unit rates (`4500..5500` cents/seat) while keeping account-level `monthly_contract_value_cents` on `customers` for cost valuation.
- **Result (`digest = 69dedc0b...`, `P(B=1) = 0.225`, `logit_mag = 2.80`):** Baseline `79.16%`, Challenger `86.71%` (`+7.55` points), `493` fixed, `191` newly broken (`101` in Enterprise).

### Iteration 4: Final Calibration to the 76% -> 88% (+12 Point) Target
- **Adjustment:** Increased self-serve restructuring prevalence $P(B=1)$ to `0.318`, set Enterprise tier weight to `0.12` (with $P(B=1) = 0.72$ in `mature_12m_plus` Enterprise and `0.95` non-inversion probability), and set latent logit magnitude to `3.12` (~4.2% irreducible Bernoulli label noise).
- **Result (`digest = cbe3f9db2abb2c2b173f207babc5f801bd73f91536d62ef0cecb28bd85da7942`):**
  - Held-out test set: `4,002` customers (`5,998` train).
  - Baseline accuracy: `76.64%` (`3,067 / 4,002`).
  - Challenger accuracy: `88.23%` (`3,531 / 4,002`, an aggregate gain of `+11.59` percentage points).
  - The final tuned cohort yielded `235` newly broken customers (`126` false negatives, `109` false positives) alongside `699` fixed customers (`375` true positives, `324` true negatives).
  - In `enterprise` (`n = 473`), accuracy drops from `75.26%` (Baseline) to `40.59%` (Challenger), generating `180` of the `235` newly broken predictions (`76.60%`).

## Milestone 2: SQL Feature Pipeline Parity and the Same-Day Invoice Bug

- Wrote `sql/features.sql` using CTEs (`event_aggregates`, `monthly_login_buckets`, `login_window_trends`, `invoice_ranked`, `invoice_latest`) and window functions (`LAG()`, `ROW_NUMBER()`, `AVG() OVER`, `COUNT() OVER`).
- **Pinned Bug Caught During Parity Testing:** When a customer had two invoices on the same `invoice_day` (for instance, a regular billing cycle and a same-day seat adjustment on day 180), `ORDER BY i.invoice_day DESC` in `ROW_NUMBER()` and `ORDER BY i.invoice_day ASC` in `LAG()` lacked a deterministic tie-breaker. SQLite could visit same-day rows in opposite orders between the ascending and descending window passes, corrupting `invoice_amount_delta_cents`.
- **Fix:** Added `i.invoice_id` as an explicit secondary sort key in both window clauses (`ORDER BY i.invoice_day DESC, i.invoice_id DESC` and `ORDER BY i.invoice_day ASC, i.invoice_id ASC`) and pinned the regression in `tests/test_pipeline_parity.py::test_pinned_bug_invoice_window_tiebreaker_prevents_nondeterministic_delta_on_same_day_invoices`.

## Milestone 3: Paired Models, FlipAudit, and Experiments 1-6

- Implemented `LogisticRegressionBaseline` (damped Newton-Raphson / IRLS with L2 penalty) and `GradientBoostedChallenger` (exact quantile-threshold Friedman gradient boosting with Newton leaf updates) directly on NumPy in `src/churn_flip_audit/models.py` so predictions are bit-for-bit identical across machines without scikit-learn version drift.
- Built `FlipAudit` (`src/churn_flip_audit/audit.py`) as the single source of truth for overall matrices, segment tables, probability-band tables, and cost-ratio sweeps.
- Implemented Experiments 1 through 6 in `src/churn_flip_audit/experiments.py`, including the Stretch Guarded Rollout Policy (routing `enterprise` and `[0.40, 0.60)` challenger probabilities back to the Baseline model, lifting blended accuracy to `90.55%` and cutting newly broken customers from `235` to `48`).

## Milestone 4: Measurement Discipline and Verification

- Built `tools/collect_metrics.py` to regenerate `docs/metrics.json` and `docs/experiments/exp1_*.json` .. `exp6_*.json`.
- Built `tools/check_numbers.py` with 18 named regex anchor phrases spanning `README.md`, `JOURNAL.md`, and `docs/adr/`. Added `--prove-failure` to corrupt `Newly broken customers (old right, new wrong): 235` to `236` in memory and assert that the checker catches and reports the exact drift.
