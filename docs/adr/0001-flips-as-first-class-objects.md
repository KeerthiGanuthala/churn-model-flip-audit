# ADR 0001: Flips as First-Class Objects Versus Deltas of Aggregates

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

When a candidate churn model (`Challenger`) replaces a production model (`Baseline`), standard evaluation reports two population-level numbers: old accuracy and new accuracy. On our held-out test set of `4,002` customers, the Logistic Regression baseline achieves `76.64%` (`3,067 / 4,002`) and the Gradient Boosted Decision Tree challenger achieves `88.23%` (`3,531 / 4,002`).

Reporting only the delta (`+11.59 percentage points`, or `+464` net correct predictions) treats errors as fungible tokens across customers. In reality, a net gain of `+464` correct predictions is compatible with `464` fixes and `0` regressions, or `1,000` fixes and `536` regressions. On our held-out test set, that `+11.59` percentage point gain hides `235` newly broken customers (`126` newly missed churners and `109` new false alarms) alongside `699` fixed customers.

## Decision

We model the paired comparison at the individual customer grain via `FlipAudit` (`src/churn_flip_audit/audit.py`), classifying every held-out customer $i \in \{1, \dots, N\}$ into one of four mutually exclusive cells:

1. `both_right` ($n_{11} = 2,832$): $\hat{y}_{\text{base}, i} = y_i \land \hat{y}_{\text{chal}, i} = y_i$
2. `fixed` ($n_{01} = 699$): $\hat{y}_{\text{base}, i} \neq y_i \land \hat{y}_{\text{chal}, i} = y_i$ (split into `fixed_tp` $= 375$ and `fixed_tn` $= 324$)
3. `newly_broken` ($n_{10} = 235$): $\hat{y}_{\text{base}, i} = y_i \land \hat{y}_{\text{chal}, i} \neq y_i$ (split into `broken_fn` $= 126$ and `broken_fp` $= 109$)
4. `both_wrong` ($n_{00} = 236$): $\hat{y}_{\text{base}, i} \neq y_i \land \hat{y}_{\text{chal}, i} \neq y_i$

Every aggregate metric, segment table, probability-band table, McNemar test, and cost-crossover sweep is a deterministic view over this single `FlipAudit` structure.

## What Each Representation Can and Cannot Answer

| Question | Aggregate Accuracy Delta ($\Delta \text{Acc}$) | First-Class Flip Matrix (`FlipAudit`) |
| :--- | :--- | :--- |
| Did total errors decrease across the test set? | Yes (`+464` net correct, `+11.59` pts) | Yes ($n_{01} - n_{10} = 699 - 235 = +464$) |
| How many previously well-served customers were harmed? | **Impossible** (collapses $n_{01}$ and $n_{10}$ into $n_{01} - n_{10}$) | Exact (`235` newly broken customers, `33.62` per `100` fixed) |
| Which customer segment absorbed the regressions? | Obscured unless manually recomputed per slice | Direct projection (`180` of `235` newly broken are in `enterprise`) |
| Is the improvement statistically significant on paired data? | Cannot compute without discordant counts $n_{01}, n_{10}$ | Exact via McNemar binomial test on $n_{01} + n_{10} = 934$ |
| Does the upgrade make or lose money under asymmetric contract value? | **Impossible** (treats a `$19/mo` trial equal to a `$1,250/mo` Enterprise account) | Exact (`-$577,365.38` net contract impact for pure Challenger) |

## Consequences

- Downstream code is forbidden from computing accuracy deltas independently of `FlipAudit`.
- Every experiment artifact (`docs/experiments/exp1_*.json` through `exp6_*.json`) is derived from `FlipAudit` and verified under `tests/test_models_and_audit.py`.
