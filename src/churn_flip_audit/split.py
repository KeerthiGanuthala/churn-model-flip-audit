"""Deterministic train/test split by keyed SHA-256 hash of customer_id.

See ADR 0005 for why a keyed cryptographic digest of the primary key is used
instead of PRNG index shuffling.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple


DEFAULT_SPLIT_KEY = "churn-flip-audit-v1"
DEFAULT_TEST_FRACTION = 0.40


def customer_hash_u64(customer_id: str, key: str = DEFAULT_SPLIT_KEY) -> int:
    """Return a deterministic 64-bit unsigned integer from SHA-256(key : customer_id)."""
    payload = f"{key}:{customer_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def customer_split_bucket(customer_id: str, key: str = DEFAULT_SPLIT_KEY) -> float:
    """Map customer_id deterministically to a uniform float in [0.0, 1.0)."""
    u64 = customer_hash_u64(customer_id, key=key)
    return u64 / float(1 << 64)


def is_test_customer(
    customer_id: str,
    key: str = DEFAULT_SPLIT_KEY,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> bool:
    """Return True iff customer_id falls into the held-out test set."""
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    return customer_split_bucket(customer_id, key=key) < test_fraction


def partition_customer_ids(
    customer_ids: Iterable[str],
    key: str = DEFAULT_SPLIT_KEY,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> Tuple[List[str], List[str]]:
    """Partition customer IDs into (train_ids, test_ids) deterministically."""
    train_ids: List[str] = []
    test_ids: List[str] = []
    for cid in customer_ids:
        if is_test_customer(cid, key=key, test_fraction=test_fraction):
            test_ids.append(cid)
        else:
            train_ids.append(cid)
    return train_ids, test_ids
