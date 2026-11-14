# ADR 0003: Seeded Synthetic Cohort With Planted Structure Versus Real Data

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

Public churn datasets (such as Telco Customer Churn) are static flat tables without event logs, lack multi-table SQL window pipelines, and rarely contain known latent segment inversions where we can compare model errors against the true data-generating process. Conversely, proprietary SaaS production data cannot be redistributed in a public repository.

## Decision

We generate a 10,000-customer subscription universe (`10,000` customers, `224,255` events, `40,181` invoices, SHA-256 digest `cbe3f9db2abb2c2b173f207babc5f801bd73f91536d62ef0cecb28bd85da7942`) deterministically from a single seed (`20261117`) using Python's standard-library Mersenne Twister (`random.Random`).

We plant an explicit two-regime latent process (`src/churn_flip_audit/generator.py`):
- **Regime A (`baseline_risk_signal`)**: High login recency gap (`recency_days >= 19`), negative 30-day login trend (`login_trend_delta <= -4`), and chronic invoice delinquency (`late_invoice_ratio >= 0.50`).
- **Regime B (`restructuring_escalation`)**: High 90-day support ticket count (`5..9`) paired with `1..2` plan/seat `downgrade` events.
- **Self-serve tiers (`free_trial`, `starter`, `pro`: 88% of cohort)**: Churn follows $y \approx A \oplus B$. When $A=1, B=1$, a struggling customer contacts support, right-sizes their tier, and stays ($y=0$). When $A=0, B=1$, a clean-billing customer strips seats and contacts support to cancel ($y=1$). Because $P(y=1 \mid B=1) \approx 0.50$, the linear baseline assigns near-zero weight to $B$ ($\hat{y}_{\text{base}} = A$, achieving `76.64%`), while the GBDT challenger learns $A \oplus B$ (`88.23%` overall).
- **Planted inversion (`enterprise`: 12% of cohort)**: In Enterprise, $B=1$ represents routine annual procurement and seat-consolidation reviews rather than an inversion of $A$; true churn follows $y \approx A$. Thus the GBDT challenger's $A \oplus B$ rule fails on Enterprise (`40.59%` vs Baseline's `75.26%`), concentrating `180` of the `235` newly broken customers in the highest-value tier.

## What Transfers and What Does Not

| Aspect | Transfers to Production? | Why |
| :--- | :--- | :--- |
| Flip audit decomposition ($n_{11}, n_{01}, n_{10}, n_{00}$) | **Yes (Exact)** | Identity $\Delta \text{Acc} = (n_{01} - n_{10})/N$ holds on any dataset. |
| Mix-shift / Simpson's paradox math | **Yes (Exact)** | $\Delta \text{Acc} = \sum_s w_s \Delta \text{Acc}_s$ explains how an 88% majority segment masks a 12% minority collapse. |
| McNemar exact test & paired SE advantage | **Yes (Exact)** | Depends only on discordant counts $n_{01}$ and $n_{10}$. |
| Cost-ratio crossover formula $r^*$ | **Yes (Exact)** | Arithmetic boundary where high-value regressions outweigh volume fixes. |
| Specific churn prevalences & effect sizes | **No** | Synthetic noise (`logit_mag = 3.12`) and regime probabilities are tuned to illustrate a 12-point upgrade. |
| Label noise & censoring | **No** | Real subscription churn includes involuntaries, chargebacks, and right-censored observation windows. |
