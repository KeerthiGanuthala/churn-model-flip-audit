# ADR 0005: Deterministic Split by Keyed Hash of Customer ID

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

A paired flip audit requires that the held-out test set remain invariant across code refactors, SQL query re-orderings, feature additions, and operating systems. Standard `train_test_split(df, test_size=0.4, random_state=42)` depends on the exact physical row order of `df` and the internal consumption of a pseudo-random number generator:

1. If a SQL `GROUP BY` or `JOIN` changes output row order before splitting, `train_test_split` assigns different customers to the test set even with a fixed `random_state`.
2. If new customers are added or filtered upstream, index-based shuffling shifts every subsequent customer between train and test.

## Decision

We assign each customer to train or test solely as a pure function of their immutable primary key `customer_id` and a versioned split salt (`key = "churn-flip-audit-v1"`, `test_fraction = 0.40`) in `src/churn_flip_audit/split.py`:

```python
digest = hashlib.sha256(f"{key}:{customer_id}".encode("utf-8")).digest()
u64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
is_test = (u64 / float(1 << 64)) < test_fraction
```

## Consequences

- **Order and Subset Invariance:** Whether `CUST-00042` is evaluated alone, in reverse order, or inside a 10,000-row table, its train/test membership never changes (`tests/test_generator_and_split.py::test_keyed_hash_split_is_invariant_to_customer_ordering_and_subsetting`).
- **Cross-Language Portability:** SHA-256 big-endian 64-bit integer normalization produces identical splits in Python, PostgreSQL (`hashtextextended` / `sha256`), Go, and Rust without sharing PRNG state.
- **Exact Held-Out Size:** Across our `10,000` generated customers (`CUST-00001` .. `CUST-10000`), `key="churn-flip-audit-v1"` deterministically routes `5,998` customers to train and `4,002` customers to the held-out test cohort.
