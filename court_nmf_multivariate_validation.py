"""Multivariate / joint validation of COURT-NMF role mixtures against
external position labels, addressing the reviewer critique that five
independent per-role Kruskal-Wallis tests are under-powered and the
wrong unit of analysis for a compositional (sum-to-one) membership
vector.

Two complementary, appropriate tests are added:

1. PERMANOVA on centred-log-ratio (CLR) transformed role mixtures.
   Role memberships are compositional data (they live on the simplex
   and sum to one), so Euclidean distance on the raw simplex is not
   the natural geometry and per-role univariate tests spend the
   sample's power testing five *marginal* projections separately
   instead of the *joint* 5-D role profile. CLR transformation maps
   the simplex isometrically onto a subspace of R^5 in which Euclidean
   (Aitchison) distance is the appropriate metric, and a permutation
   MANOVA (PERMANOVA, Anderson 2001) on that distance tests whether
   the whole role-mixture profile -- not any single coordinate --
   differs across guard/forward/center.

2. Cross-validated multinomial classification. A 5-fold stratified
   multinomial logistic regression is fit on the 5-D role-mixture
   vector to predict the 3-class position label, evaluated by
   out-of-fold accuracy and macro-F1, benchmarked against the
   majority-class baseline and against a label-permutation null
   distribution (a non-parametric significance test that does not
   assume the classifier's sampling distribution).

Both address the reviewer's concern directly: even though four of
five *individual* roles do not survive Bonferroni correction on
marginal Kruskal-Wallis tests, the *joint* mixture vector may still
carry statistically and practically significant positional signal,
because the roles are correlated components of one composition rather
than five independent measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

RNG_SEED = 42
N_PERM = 9999
BASE = Path(__file__).resolve().parents[1]
IN_CSV = BASE / "outputs" / "court_nmf_validation" / "court_nmf_labeled_memberships.csv"
OUT = BASE / "outputs" / "court_nmf_validation"
OUT.mkdir(parents=True, exist_ok=True)

POSITIONS = ["guard", "forward", "center"]


def clr(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Centred log-ratio transform of compositional rows summing to 1."""
    Xc = np.clip(X, eps, None)
    logX = np.log(Xc)
    return logX - logX.mean(axis=1, keepdims=True)


def permanova(D: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator):
    """One-way PERMANOVA (Anderson 2001) from a square distance matrix."""
    n = D.shape[0]
    D2 = D ** 2
    ss_total = D2.sum() / (2 * n)

    def ss_within(labels):
        ss = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            ng = len(idx)
            if ng <= 1:
                continue
            sub = D2[np.ix_(idx, idx)]
            ss += sub.sum() / (2 * ng)
        return ss

    ss_w_obs = ss_within(groups)
    ss_a_obs = ss_total - ss_w_obs
    k = len(np.unique(groups))
    df_a, df_w = k - 1, n - k
    F_obs = (ss_a_obs / df_a) / (ss_w_obs / df_w)
    R2_obs = ss_a_obs / ss_total

    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(groups)
        ss_w = ss_within(perm_labels)
        ss_a = ss_total - ss_w
        F_perm = (ss_a / df_a) / (ss_w / df_w) if ss_w > 0 else np.inf
        if F_perm >= F_obs:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {
        "F_statistic": float(F_obs),
        "R2": float(R2_obs),
        "df_between": int(df_a),
        "df_within": int(df_w),
        "n_permutations": int(n_perm),
        "p_value": float(p_value),
    }


def permutation_classifier_test(X: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    clf = LogisticRegression(max_iter=2000)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    acc_obs = accuracy_score(y, y_pred)
    f1_obs = f1_score(y, y_pred, average="macro")

    majority_baseline = pd.Series(y).value_counts(normalize=True).max()

    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        y_pred_perm = cross_val_predict(clf, X, y_perm, cv=cv)
        acc_perm = accuracy_score(y_perm, y_pred_perm)
        if acc_perm >= acc_obs:
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    return {
        "cv_accuracy": float(acc_obs),
        "cv_macro_f1": float(f1_obs),
        "majority_class_baseline": float(majority_baseline),
        "n_permutations": int(n_perm),
        "p_value": float(p_value),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    df = pd.read_csv(IN_CSV)
    df = df[df["position"].isin(POSITIONS)].copy()
    roles = [f"role_{i}" for i in range(1, 6)]
    X = df[roles].to_numpy()
    # renormalize defensively in case of rounding drift; memberships should already sum to 1
    X = X / X.sum(axis=1, keepdims=True)
    y_pos = df["position"].to_numpy()

    # --- PERMANOVA on CLR-transformed compositions -----------------
    X_clr = clr(X)
    D = squareform(pdist(X_clr, metric="euclidean"))
    permanova_result = permanova(D, y_pos, N_PERM, rng)

    # --- Cross-validated multinomial classification -----------------
    clf_result = permutation_classifier_test(X_clr, y_pos, N_PERM, rng)

    # a lower-permutation-count fast pass is not used; both reuse N_PERM
    # for a shared, pre-registered permutation budget.

    out = {
        "n": int(len(df)),
        "permanova_clr": permanova_result,
        "cv_multinomial_classifier": clf_result,
    }
    (OUT / "court_nmf_multivariate_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
