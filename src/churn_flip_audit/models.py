"""Deterministic Baseline (Logistic Regression) and Challenger (Gradient Boosted Trees).

Implemented directly on NumPy so the estimators have zero external ML library
variance across operating systems or BLAS implementations, while following
exact textbook formulations:
- `LogisticRegressionBaseline`: L2-regularized binary logistic regression solved
  via damped Newton-Raphson (IRLS) on standardized numeric features.
- `GradientBoostedChallenger`: Friedman (2001) gradient boosted decision trees
  for binomial log-likelihood with exact quantile candidate splits and Newton
  leaf updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .pipeline import NUMERIC_MODEL_FEATURES
from .split import DEFAULT_SPLIT_KEY, DEFAULT_TEST_FRACTION, is_test_customer


def extract_design_matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numeric feature matrix X (float64) and label vector y (int64) from feature rows."""
    n = len(rows)
    d = len(NUMERIC_MODEL_FEATURES)
    X = np.zeros((n, d), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(rows):
        for j, col in enumerate(NUMERIC_MODEL_FEATURES):
            X[i, j] = float(r[col])
        y[i] = int(r["churned"])
    return X, y


def _clip_sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class LogisticRegressionBaseline:
    """Deterministic L2-regularized Logistic Regression solved via IRLS / Newton-Raphson."""

    def __init__(self, l2_penalty: float = 1.0, max_iter: int = 60, tol: float = 1e-8) -> None:
        self.l2_penalty = float(l2_penalty)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale

        Xs = (X - self.mean_) / self.scale_
        n, d = Xs.shape
        Xb = np.column_stack([np.ones(n, dtype=np.float64), Xs])

        w = np.zeros(d + 1, dtype=np.float64)
        reg_diag = np.full(d + 1, self.l2_penalty, dtype=np.float64)
        reg_diag[0] = 0.0  # Do not regularize intercept

        for _ in range(self.max_iter):
            p = _clip_sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) + reg_diag * w
            s = np.maximum(p * (1.0 - p), 1e-8)
            hess = (Xb.T * s) @ Xb + np.diag(reg_diag)
            step = np.linalg.solve(hess, grad)
            w_new = w - step
            if float(np.max(np.abs(step))) < self.tol:
                w = w_new
                break
            w = w_new

        self.intercept_ = float(w[0])
        self.coef_ = w[1:].copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("LogisticRegressionBaseline is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mean_) / self.scale_
        logits = self.intercept_ + Xs @ self.coef_
        return np.round(_clip_sigmoid(logits), 8)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


@dataclass
class _TreeNode:
    is_leaf: bool
    value: float = 0.0
    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class GradientBoostedChallenger:
    """Deterministic Gradient Boosted Decision Trees for binary log-loss (Friedman 2001)."""

    def __init__(
        self,
        n_estimators: int = 45,
        learning_rate: float = 0.12,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        l2_leaf_reg: float = 2.0,
        n_thresholds: int = 32,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.n_thresholds = int(n_thresholds)
        self.init_logit_: float = 0.0
        self.trees_: List[_TreeNode] = []
        self.candidate_thresholds_: List[np.ndarray] = []

    def _compute_candidate_thresholds(self, X: np.ndarray) -> List[np.ndarray]:
        _, d = X.shape
        thresholds: List[np.ndarray] = []
        quantiles = np.linspace(0.03, 0.97, self.n_thresholds)
        for j in range(d):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= self.n_thresholds + 1:
                if len(uniq) <= 1:
                    thresholds.append(uniq)
                else:
                    mids = 0.5 * (uniq[:-1] + uniq[1:])
                    thresholds.append(np.round(mids, 8))
            else:
                qs = np.unique(np.round(np.quantile(col, quantiles), 8))
                thresholds.append(qs)
        return thresholds

    def _build_tree(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        g_sum = float(np.sum(grad[indices]))
        h_sum = float(np.sum(hess[indices]))
        leaf_value = -g_sum / (h_sum + self.l2_leaf_reg)

        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return _TreeNode(is_leaf=True, value=leaf_value)

        parent_score = (g_sum * g_sum) / (h_sum + self.l2_leaf_reg)
        best_gain = 1e-7
        best_feat = -1
        best_thresh = 0.0
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None

        _, d = X.shape
        for j in range(d):
            col_sub = X[indices, j]
            g_sub = grad[indices]
            h_sub = hess[indices]
            for thresh in self.candidate_thresholds_[j]:
                left_mask = col_sub <= thresh
                n_left = int(np.sum(left_mask))
                n_right = len(indices) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                g_left = float(np.sum(g_sub[left_mask]))
                h_left = float(np.sum(h_sub[left_mask]))
                g_right = g_sum - g_left
                h_right = h_sum - h_left

                gain = (
                    (g_left * g_left) / (h_left + self.l2_leaf_reg)
                    + (g_right * g_right) / (h_right + self.l2_leaf_reg)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = float(thresh)
                    best_left_idx = indices[left_mask]
                    best_right_idx = indices[~left_mask]

        if best_feat == -1 or best_left_idx is None or best_right_idx is None:
            return _TreeNode(is_leaf=True, value=leaf_value)

        left_node = self._build_tree(X, grad, hess, best_left_idx, depth + 1)
        right_node = self._build_tree(X, grad, hess, best_right_idx, depth + 1)
        return _TreeNode(
            is_leaf=False,
            value=leaf_value,
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
        )

    def _predict_tree(self, node: _TreeNode, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)

        def _traverse(curr: _TreeNode, mask: np.ndarray) -> None:
            if not np.any(mask):
                return
            if curr.is_leaf or curr.left is None or curr.right is None:
                out[mask] = curr.value
                return
            go_left = mask & (X[:, curr.feature_idx] <= curr.threshold)
            go_right = mask & (~go_left)
            _traverse(curr.left, go_left)
            _traverse(curr.right, go_right)

        _traverse(node, np.ones(n, dtype=bool))
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedChallenger":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pos_rate = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        self.init_logit_ = float(np.log(pos_rate / (1.0 - pos_rate)))
        self.candidate_thresholds_ = self._compute_candidate_thresholds(X)
        self.trees_ = []

        raw_logits = np.full(X.shape[0], self.init_logit_, dtype=np.float64)
        all_indices = np.arange(X.shape[0], dtype=np.int64)

        for _ in range(self.n_estimators):
            p = _clip_sigmoid(raw_logits)
            grad = p - y
            hess = np.maximum(p * (1.0 - p), 1e-6)
            tree = self._build_tree(X, grad, hess, all_indices, depth=0)
            update = self._predict_tree(tree, X)
            raw_logits += self.learning_rate * update
            self.trees_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("GradientBoostedChallenger is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        raw_logits = np.full(X.shape[0], self.init_logit_, dtype=np.float64)
        for tree in self.trees_:
            raw_logits += self.learning_rate * self._predict_tree(tree, X)
        return np.round(_clip_sigmoid(raw_logits), 8)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


@dataclass(frozen=True)
class PairedPredictionRecord:
    customer_id: str
    plan_tier: str
    tenure_band: str
    region: str
    monthly_contract_value_cents: int
    y_true: int
    p_baseline: float
    yhat_baseline: int
    p_challenger: float
    yhat_challenger: int


def train_and_evaluate_paired(
    feature_rows: Sequence[Dict[str, Any]],
    split_key: str = DEFAULT_SPLIT_KEY,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    threshold: float = 0.50,
) -> Tuple[
    LogisticRegressionBaseline,
    GradientBoostedChallenger,
    List[PairedPredictionRecord],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Split deterministically, train both models on train, and return paired test predictions."""
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for row in feature_rows:
        if is_test_customer(row["customer_id"], key=split_key, test_fraction=test_fraction):
            test_rows.append(row)
        else:
            train_rows.append(row)

    X_train, y_train = extract_design_matrix(train_rows)
    X_test, y_test = extract_design_matrix(test_rows)

    baseline = LogisticRegressionBaseline(l2_penalty=1.0).fit(X_train, y_train)
    challenger = GradientBoostedChallenger(
        n_estimators=45,
        learning_rate=0.12,
        max_depth=3,
        min_samples_leaf=20,
        l2_leaf_reg=2.0,
    ).fit(X_train, y_train)

    p_base = baseline.predict_proba(X_test)
    yhat_base = (p_base >= threshold).astype(np.int64)
    p_chal = challenger.predict_proba(X_test)
    yhat_chal = (p_chal >= threshold).astype(np.int64)

    paired: List[PairedPredictionRecord] = []
    for i, row in enumerate(test_rows):
        paired.append(
            PairedPredictionRecord(
                customer_id=str(row["customer_id"]),
                plan_tier=str(row["plan_tier"]),
                tenure_band=str(row["tenure_band"]),
                region=str(row["region"]),
                monthly_contract_value_cents=int(row["monthly_contract_value_cents"]),
                y_true=int(y_test[i]),
                p_baseline=float(p_base[i]),
                yhat_baseline=int(yhat_base[i]),
                p_challenger=float(p_chal[i]),
                yhat_challenger=int(yhat_chal[i]),
            )
        )

    return baseline, challenger, paired, train_rows, test_rows
