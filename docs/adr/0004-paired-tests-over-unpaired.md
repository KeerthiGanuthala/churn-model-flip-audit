# ADR 0004: Paired Tests Over Unpaired: Why Correlated Errors Make the Two-Proportion Test Wrong

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

When comparing Baseline accuracy ($\hat{p}_1 = 0.766367$) and Challenger accuracy ($\hat{p}_2 = 0.882309$) evaluated on the **same** held-out test set of $N = 4,002$ customers, teams frequently plug $(\hat{p}_1, N)$ and $(\hat{p}_2, N)$ into a standard two-sample independent proportions $z$-test.

That calculation assumes the two models were evaluated on two independent random samples of size $N$. In reality, both models scored the exact same $N$ customers, and their correctness indicators $I_{1,i} = \mathbb{I}[\hat{y}_{\text{base},i} = y_i]$ and $I_{2,i} = \mathbb{I}[\hat{y}_{\text{chal},i} = y_i]$ are strongly positively correlated because both models agree on easy cases ($n_{11} = 2,832$ both right, $n_{00} = 236$ both wrong; concordance $= 76.66\%$).

## Mathematical Derivation

The variance of the difference in accuracies $\hat{\Delta} = \hat{p}_2 - \hat{p}_1 = \frac{n_{01} - n_{10}}{N}$ is:

$$\text{Var}_{\text{paired}}(\hat{\Delta}) = \frac{\text{Var}(I_1) + \text{Var}(I_2) - 2\,\text{Cov}(I_1, I_2)}{N} = \frac{\frac{n_{01} + n_{10}}{N} - \left(\frac{n_{01} - n_{10}}{N}\right)^2}{N}$$

By ignoring the positive covariance term $-2\,\text{Cov}(I_1, I_2) / N$, the wrong unpaired two-proportion test computes:

$$\text{Var}_{\text{unpaired}}(\hat{\Delta}) = \frac{\hat{p}_1(1 - \hat{p}_1) + \hat{p}_2(1 - \hat{p}_2)}{N}$$

On our held-out test set ($N = 4,002$, $n_{01} = 699$, $n_{10} = 235$), the unpaired formula inflates sampling variance by `1.2862x` and inflates the standard error from `0.007413` to `0.008408` (`1.1341x`).

## Decision

1. We test the null hypothesis $H_0: p_{\text{fixed}} = p_{\text{broken}}$ using **McNemar's exact binomial test** (`scipy.stats.binomtest(n_fixed, n=n_fixed + n_broken, p=0.5)`) and report Edwards' continuity-corrected $\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$.
2. In Experiment 3 (`docs/experiments/exp3_mcnemar_significance.json`), we quantify both the standard error inflation (`1.1341x`) and the sample-size penalty: to detect a `+2.0` percentage point accuracy gain at 80% power ($\alpha = 0.05$) with our observed discordance rate, the paired test requires `4,578` held-out customers whereas the wrong unpaired formula demands `5,644` customers (wasting `1,066` labeled test examples).
