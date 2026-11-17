# churn-model-flip-audit

An aggregate accuracy gain cannot show how many customers a new model newly gets wrong: a move from `76.64%` to `88.23%` (`+11.59 percentage points`) on our held-out test set hides **`235` newly broken customers** (`126` newly missed churners and `109` new false alarms), concentrated (`180` of `235`, or `76.60%`) in the highest-value `enterprise` segment where the challenger loses by `-34.67 percentage points`.

This project makes individual prediction flips (`both_right`, `fixed`, `newly_broken`, `both_wrong`) first-class objects across a seeded subscription cohort, a SQLite/PostgreSQL window-function feature pipeline, paired baseline/challenger estimators, McNemar significance testing, and an asymmetric contract cost model.

## Headline Flip Matrix (First Screen)

Evaluated on the deterministic held-out test cohort of `4,002` customers (`5,998` train, keyed SHA-256 split on `customer_id`, seed `20261117`):

- Baseline logistic regression accuracy: `76.64%` (`3,067 / 4,002`)
- Challenger gradient boosting accuracy: `88.23%` (`3,531 / 4,002`)
- Aggregate accuracy gain: `+11.59 percentage points` (`+464` net correct predictions)
- Both models right: `2,832` (`70.76%`)
- Fixed customers (old wrong, new right): `699` (`17.47%`: `375` true positives, `324` true negatives)
- Newly broken customers (old right, new wrong): `235` (`5.87%`: `126` false negatives, `109` false positives)
- Both models wrong: `236` (`5.90%`)
- Regression ratio: `33.62` newly broken customers for every `100` fixed customers

| Baseline \ Challenger | Challenger Correct | Challenger Wrong | Row Total |
| :--- | ---: | ---: | ---: |
| **Baseline Correct** | `2,832` (`both_right`) | **`235` (`newly_broken`)** | `3,067` (`76.64%`) |
| **Baseline Wrong** | **`699` (`fixed`)** | `236` (`both_wrong`) | `935` (`23.36%`) |
| **Column Total** | `3,531` (`88.23%`) | `471` (`11.77%`) | `4,002` (`100.00%`) |

## Quickstart

Zero network access or external services required. From a fresh clone:

```bash
# Reproduce the synthetic cohort, verify SQL parity, run all 6 experiments, and print all tables:
PYTHONPATH=src python3 -m churn_flip_audit.cli

# Full verification suite (regenerates metrics, runs property tests with >=90% coverage, and checks all doc numbers):
make verify
```

On Windows Command Prompt (`cmd.exe`) without GNU Make installed:

```cmd
verify.bat
```

## The Experiments

Every experiment writes structured JSON to `docs/experiments/` and derives its figures from a single `FlipAudit` instance (`src/churn_flip_audit/audit.py`).

### Experiment 1: What the Gain Hides (`docs/experiments/exp1_headline_flip_matrix.json`)

The aggregate delta compares only the marginal totals (`3,531 / 4,002` vs `3,067 / 4,002`), algebraically cancelling `235` regressions against `235` of the `699` fixes:

$$\Delta \text{Acc} = \frac{n_{\text{fixed}} - n_{\text{newly\_broken}}}{N} = \frac{699 - 235}{4,002} = \frac{+464}{4,002} = +11.59\%$$

Decomposing the discordant cells by error direction shows that `126` of the `235` newly broken predictions are **false negatives** (`broken_fn`: customers who actually churned, whom the Baseline caught and routed to retention save workflows, but whom the Challenger predicted would stay) and `109` are **false positives** (`broken_fp`: healthy customers newly flagged as churning).

### Experiment 2: Where the New Model Is Worse (`docs/experiments/exp2_segment_stratified_flips.json`)

Stratifying the exact same `FlipAudit` object by `plan_tier` exposes Simpson's aggregation trap:

- Enterprise segment accuracy change: `-34.67 percentage points` (`75.26%` down to `40.59%`)
- Enterprise newly broken customers: `180` of `235` (`76.60%` of all regressions occur in `11.82%` of the customer population)

| Plan Tier | Held-Out N | Pop Share | Baseline Acc | Challenger Acc | Accuracy Delta | Fixed | Newly Broken | Net Gain |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `free_trial` | `793` | `19.82%` | `77.55%` | `93.95%` | `+16.39 pts` | `140` | `10` | `+130` |
| `starter` | `1,503` | `37.56%` | `76.25%` | `94.94%` | `+18.70 pts` | `308` | `27` | `+281` |
| `pro` | `1,233` | `30.81%` | `77.05%` | `94.65%` | `+17.60 pts` | `235` | `18` | `+217` |
| `enterprise` | `473` | `11.82%` | `75.26%` | `40.59%` | **`-34.67 pts`** | `16` | **`180`** | **`-164`** |
| **Total / Blended** | **`4,002`** | **`100.00%`** | **`76.64%`** | **`88.23%`** | **`+11.59 pts`** | **`699`** | **`235`** | **`+464`** |

Why both facts hold at once (mix-shift identity):

$$\Delta \text{Acc}_{\text{overall}} = \sum_{s} w_s \Delta \text{Acc}_s = 0.1982(+0.1639) + 0.3756(+0.1870) + 0.3081(+0.1760) + 0.1182(-0.3467) = +0.1159$$

Because self-serve tiers (`free_trial`, `starter`, `pro`) make up `88.18%` of customer headcount, their `+628` net fixes swamp the `-164` net collapse in `enterprise`.

### Experiment 3: Is the Gain Even Real (`docs/experiments/exp3_mcnemar_significance.json`)

Because both models score the **same** `4,002` customers, their predictions are strongly correlated (`76.66%` concordance: `2,832` both right, `236` both wrong). Comparing two accuracies with an unpaired two-proportion $z$-test drops the positive covariance term $2\,\text{Cov}(I_{\text{base}}, I_{\text{chal}})$ and inflates the standard error:

- McNemar continuity-corrected chi-square: `229.5171` (uncorrected `230.5096`, exact two-sided binomial $p = 4.052 \times 10^{-54}$ on `934` discordant pairs)
- Unpaired standard error inflation ratio: `1.1341x` (`0.008408` unpaired vs `0.007413` paired; variance inflated by `1.2862x`)

| Statistical Method | Std Error (SE) | Z-Stat | Two-Sided p-value | Min Test N (80% Power, 11.59pt Gain) | Min Test N (80% Power, 2.0pt Gain) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Paired (McNemar / Exact Binomial)** | `0.007413` | `15.6396` | `4.052e-54` | `134` | `4,578` |
| **Unpaired Two-Proportion (Wrong)** | `0.008408` | `13.7902` | Ignores pairing | `168` | `5,644` |

When teams use the wrong unpaired power formula to size an evaluation set for a `+2.0` percentage point upgrade, they over-collect by `1,066` labeled customers (`5,644` vs `4,578`).

### Experiment 4: When the Gain Is a Loss (`docs/experiments/exp4_cost_ratio_crossover.json`)

Assign each `fixed` customer $+1.0$ utility unit, each non-Enterprise `newly_broken` customer (`55` total) $-1.0$ unit, and each high-value Enterprise `newly_broken` customer (`180` total) $-r$ units:

$$\text{Net Utility}(r) = 699(1.0) - 55(1.0) - 180\,r = 644 - 180\,r$$

- High-value Enterprise cost crossover ratio `r* = 3.5778` (uniform crossover across all `235` newly broken customers is `2.9745`)
- Pure Challenger net contract value impact: `-$577,365.38` (`+$349,567.57` gained from `699` fixed customers vs `-$926,932.95` lost from `235` newly broken customers when weighted by actual monthly contract values, where an Enterprise contract at `$1,250/mo` is `25.51x` a Starter contract at `$49/mo` and `6.28x` a Pro contract at `$199/mo`)

| High-Value Regression Cost Ratio `r` | Fixed (+1.0) | Broken Self-Serve (-1.0) | Broken Enterprise (-r) | Net Utility Units | Business Verdict |
| ---: | ---: | ---: | ---: | ---: | :--- |
| `1.0000x` | `699` | `55` | `180` | `+464.00` | NET GAIN |
| `2.5000x` | `699` | `55` | `180` | `+194.00` | NET GAIN |
| `3.0000x` | `699` | `55` | `180` | `+104.00` | NET GAIN |
| **`3.5778x` (`r*`)** | **`699`** | **`55`** | **`180`** | **`0.00`** | **BREAKEVEN CROSSOVER** |
| `4.0000x` | `699` | `55` | `180` | `-76.00` | **NET LOSS** |
| `5.0000x` | `699` | `55` | `180` | `-256.00` | **NET LOSS** |
| `10.0000x` | `699` | `55` | `180` | `-1,156.00` | **NET LOSS** |
| `25.0000x` | `699` | `55` | `180` | `-3,856.00` | **NET LOSS** |

### Experiment 5: Threshold and Calibration (`docs/experiments/exp5_threshold_and_calibration.json`)

Does the Challenger's gain (or its `235` regressions) disappear if we sweep the decision threshold or apply Platt logit recalibration?

- Raw Expected Calibration Error (ECE): Baseline `0.028370`, Challenger `0.020093`
- Platt-recalibrated ECE: Baseline `0.024779`, Challenger `0.014045`
- Platt-recalibrated accuracy at threshold `0.50`: Baseline `76.66%`, Challenger `88.26%` (`+11.59` percentage points, `699` fixed, `235` newly broken)

| Challenger Threshold | Challenger Acc | Delta vs Baseline | Fixed | Newly Broken | Broken FN | Broken FP | Net Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.10` | `86.78%` | `+10.14 pts` | `714` | `308` | `75` | `233` | `+406` |
| `0.20` | `88.23%` | `+11.59 pts` | `706` | `242` | `117` | `125` | `+464` |
| `0.30` | `88.21%` | `+11.57 pts` | `699` | `236` | `125` | `111` | `+463` |
| `0.50` | `88.23%` | `+11.59 pts` | `699` | `235` | `126` | `109` | `+464` |
| `0.70` | `88.18%` | `+11.54 pts` | `697` | `235` | `128` | `107` | `+462` |
| `0.80` | `88.18%` | `+11.54 pts` | `697` | `235` | `128` | `107` | `+462` |
| `0.90` | `86.98%` | `+10.34 pts` | `708` | `294` | `204` | `90` | `+414` |

Because the Challenger's probabilities are sharply bimodal near `0.04` and `0.96` across the interaction leaves, the `235` regressions are structural leaf-assignment errors rather than a threshold or calibration artifact.

### Experiment 6 (Stretch): Guarded Rollout Policy (`docs/experiments/exp6_guarded_rollout_policy.json`)

Using the training-split `FlipAudit` to identify losing segments (`enterprise`) and routing both `enterprise` accounts and predictions inside the `[0.40, 0.60)` uncertainty dead-zone back to the Baseline model (`3,172` test customers routed to Challenger, `830` routed to Baseline):

- Guarded rollout blended accuracy: `90.55%` (`+13.92 percentage points` over pure Baseline, `+2.32 percentage points` over pure Challenger)
- Newly broken customers drop from `235` to `48` (`0` in `enterprise`), and net contract value flips from `-$577,365.38` to `+$206,367.45`.

| Rollout Policy | Blended Acc | Delta vs Base | Fixed | Newly Broken | Enterprise Broken | Utility (`r=5`) | Net Contract USD |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure Baseline (Logistic Regression)** | `76.64%` | `+0.00 pts` | `0` | `0` | `0` | `+0.0` | `+$0.00` |
| **Pure Challenger (Global GBDT)** | `88.23%` | `+11.59 pts` | `699` | `235` | `180` | `-256.0` | `-$577,365.38` |
| **Guarded Rollout (Hybrid Router)** | **`90.55%`** | **`+13.92 pts`** | **`605`** | **`48`** | **`0`** | **`+557.0`** | **`+$206,367.45`** |

## Fixture Section: The Planted Structure, Plainly Named

In `src/churn_flip_audit/generator.py` (`seed = 20261117`, `10,000` customers, SHA-256 digest `cbe3f9db2abb2c2b173f207babc5f801bd73f91536d62ef0cecb28bd85da7942`), every customer's observable event and invoice history is driven by two latent indicators:

1. **Regime A (`baseline_risk_signal`)**: Drawn with probability `0.48`. Generates a high login recency gap (`19..34` days), negative 30-day login trend (`-12..-4`), and high late-invoice ratio (`~0.85`).
2. **Regime B (`restructuring_escalation`)**: Drawn with probability `0.318` in self-serve tiers (`free_trial`, `starter`, `pro`) and `0.52..0.72` in `enterprise`. Generates `5..9` support tickets and `1..2` plan/seat `downgrade` events.
3. **Why Challenger Wins in Self-Serve (`88.18%` of cohort)**: In self-serve tiers, churn follows $y = A \oplus B$ (`98%` of the time when $B=1$, with latent logit magnitude `3.12`). When $A=1, B=1$, a struggling user contacts support, receives a tier right-sizing downgrade, and retains ($y=0$). When $A=0, B=1$, a clean-billing user strips seats and contacts support to cancel ($y=1$). Because $P(y=1 \mid B=1) \approx 0.50$, the linear baseline gives near-zero weight to $B$ ($\hat{y}_{\text{base}} = A$, `76.64%` accuracy), whereas the depth-3 GBDT challenger learns $A \oplus B$ (`~94.6%` accuracy on self-serve).
4. **Why Challenger Loses in Enterprise (`11.82%` of cohort)**: In `enterprise`, $B=1$ represents routine annual procurement and seat-consolidation reviews (`95%` of the time when $B=1$, true churn still follows $y = A$). Because invoice amounts in `invoices.amount_cents` are billed per seat (`$45..$55/seat` across tiers), the global GBDT challenger applies the self-serve $A \oplus B$ rule to Enterprise accounts, generating both **newly broken false positives** (healthy Enterprise renewals consolidating seats, $A=0, B=1$) and **newly broken false negatives** (delinquent Enterprise accounts in procurement default, $A=1, B=1$).

## What This Project Cannot Measure (and Out-of-Scope Triggers)

1. **Real-World Churn Prevalence and Feature Correlations:** Synthetic regime weights are calibrated to produce a 12-point accuracy gap; they never prove real-world SaaS churn rates.
2. **Label Noise and Right-Censoring:** In production subscription systems, "churn" blurs across voluntary cancellation, involuntary payment failure, and temporary pause.
3. **Time-Sliced Evaluation and Concept Drift (Out of Scope on Purpose):** This build evaluates a cross-sectional holdout at observation cutoff day `180`. **Trigger to add time-sliced evaluation:** the first dataset with multi-quarter real calendar timestamps where macro or pricing shifts create a temporal drift question between training cohorts and future renewal cohorts.
4. **Online A/B Measurement and Uplift Modeling (Out of Scope on Purpose):** Predicting churn risk $P(Y=1 \mid X)$ is distinct from estimating causal retention uplift $P(Y=0 \mid \text{do}(\text{save offer}), X) - P(Y=0 \mid \text{no offer}, X)$. Uplift modeling is a separate project, not a missing feature of offline flip auditing.

## Architecture Decision Records (ADRs)

- [ADR 0001: Flips as First-Class Objects Versus Deltas of Aggregates](docs/adr/0001-flips-as-first-class-objects.md)
- [ADR 0002: SQLite Executing PostgreSQL-Style SQL: The Boundary of That Claim](docs/adr/0002-sqlite-executing-postgresql-style-sql.md)
- [ADR 0003: Seeded Synthetic Cohort With Planted Structure Versus Real Data](docs/adr/0003-seeded-synthetic-cohort-with-planted-structure.md)
- [ADR 0004: Paired Tests Over Unpaired: Why Correlated Errors Make the Two-Proportion Test Wrong](docs/adr/0004-paired-tests-over-unpaired.md)
- [ADR 0005: Deterministic Split by Keyed Hash of Customer ID](docs/adr/0005-deterministic-split-by-keyed-hash.md)

## Questions I Can Answer From My Machine

1. **When someone says the new model is 12 points better, what three questions do I ask?**
   - First: *How many customers did the old model get right that the new model now gets wrong (`newly_broken`), and what is the regression-to-fix ratio?* (On my run, `+11.59` points hides `235` newly broken customers against `699` fixed, or `33.62` regressions per `100` fixes.)
   - Second: *In which customer segments or contract tiers is the new model net-negative?* (In `enterprise`, accuracy falls by `-34.67` points from `75.26%` to `40.59%`, concentrating `180` of the `235` regressions.)
   - Third: *What is the cost crossover ratio $r^*$ between a newly broken high-value customer and a fixed customer, and where does our actual contract value ratio sit relative to $r^*$?* (Our crossover is `r* = 3.5778`, while Enterprise contracts are `25.51x` Starter contracts, making the unguarded upgrade a `-$577,365.38` loss.)
2. **How many customers did my model upgrade newly get wrong, and in which segments?**
   - It newly broke `235` held-out customers (`126` missed churners and `109` false alarms): `180` in `enterprise` (`38.05%` of all Enterprise accounts), `27` in `starter`, `18` in `pro`, and `10` in `free_trial`.
3. **Why is McNemar the right test for comparing two models on one test set?**
   - Because both models evaluate the same `4,002` customers, the concordant cells (`2,832` both right and `236` both wrong) carry zero information about which model is better but induce strong positive covariance between the two accuracy estimates. McNemar conditions strictly on the `934` discordant pairs (`699` vs `235`, exact $p = 4.052 \times 10^{-54}$), whereas the unpaired two-proportion test ignores covariance and inflates the standard error by `1.1341x` (`1.2862x` variance inflation).
4. **At what cost ratio does an accuracy win become a business loss?**
   - When a high-value Enterprise regression costs more than `r* = 3.5778` times a fixed customer (or `2.9745x` if all `235` regressions are weighted uniformly), net utility turns negative even while aggregate accuracy is `+11.59` points higher.
5. **Why can a model win overall and lose in every segment you care about?**
   - Aggregate accuracy weights segments by headcount (`w_s = n_s / N`), not revenue. Because `free_trial`, `starter`, and `pro` represent `88.18%` of customer headcount, a `+16` to `+19` point gain on low-ACV self-serve users outweighs a `-34.67` point collapse on the `11.82%` of customers in `enterprise` who represent the majority of contract revenue.
