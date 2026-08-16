"""
Adaptive soft-clustering and comparative benchmark for the Stage-2 NMF space.

This script applies the study preprocessing and NMF basis, then adds:

1. Adaptive NMF memberships: row-normalized NMF activations, interpreted as
   archetype percentages for each player-season.
2. Soft and hard clustering benchmarks on the same five-dimensional NMF space:
   KMeans, Gaussian Mixture, Bayesian Gaussian Mixture, Fuzzy C-Means,
   Spectral Clustering (sampled), and local Sparse Subspace Clustering (sampled).
3. NMF theoretical diagnostics: reconstruction quality, activation sparsity,
   component orthogonality, and membership entropy.

Outputs are written to ../outputs/adaptive_clustering/.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / "adaptive_clustering" / "mplconfig"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler


RANDOM_STATE = 42
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "CollegeBasketballPlayers2009-2021（55columns）.csv"
OUTDIR = BASE / "outputs" / "adaptive_clustering"
FIGDIR = BASE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

STYLE_COLS = [
    "pts_per100",
    "treb_per100",
    "ast_per100",
    "stl_per100",
    "blk_per100",
    "oreb_per100",
    "dreb_per100",
    "usg",
    "eFG",
    "TS_per",
    "ORB_per",
    "DRB_per",
    "AST_per",
    "TO_per",
    "FT_per",
    "twoP_per",
    "TP_per",
    "blk_per",
    "stl_per",
    "ftr",
    "bpm",
    "obpm",
    "dbpm",
]

PAPER_ARCHETYPES = {
    0: "Foul-drawing rebounding forward",
    1: "High-minute perimeter shooting specialist",
    2: "Efficient floor-spacing combo guard",
    3: "High-volume bench scorer",
    4: "Two-way glass-cleaning scorer",
}


def log(message: str) -> None:
    print(message, flush=True)


def build_stage2_matrix() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Construct the Stage-2 analytic matrix."""
    df_raw = pd.read_csv(DATA, low_memory=False)
    df_raw = df_raw.drop(["type", "ht"], axis=1, errors="ignore")

    rotation_mask = (
        (pd.to_numeric(df_raw["Min_per"], errors="coerce") > 5)
        & (pd.to_numeric(df_raw["GP"], errors="coerce") >= 10)
        & (pd.to_numeric(df_raw["pts"], errors="coerce") >= 1.0)
    )

    mice_cols = [
        "rimmade",
        "rimmade+rimmiss",
        "midmade",
        "midmade+midmiss",
        "rimmade/(rimmade+rimmiss)",
        "midmade/(midmade+midmiss)",
        "dunksmade",
        "dunksmiss+dunksmade",
    ]
    mice_cols = [c for c in mice_cols if c in df_raw.columns]

    numeric_cols = [
        "GP",
        "Min_per",
        "Ortg",
        "usg",
        "eFG",
        "TS_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FTM",
        "FTA",
        "FT_per",
        "twoPM",
        "twoPA",
        "twoP_per",
        "TPM",
        "TPA",
        "TP_per",
        "blk_per",
        "stl_per",
        "ftr",
        "porpag",
        "adjoe",
        "pfr",
        "ast/tov",
        "drtg",
        "adrtg",
        "dporpag",
        "stops",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "oreb",
        "dreb",
        "treb",
        "ast",
        "stl",
        "blk",
        "pts",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_raw.columns and c not in mice_cols]
    cat_cols = [c for c in ["player_name", "team", "conf", "yr"] if c in df_raw.columns]

    df_mice = df_raw[mice_cols].apply(pd.to_numeric, errors="coerce")
    df_numeric = df_raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_cat = df_raw[cat_cols].copy()
    df_other = df_raw.drop(columns=mice_cols + numeric_cols + cat_cols + ["pid"], errors="ignore")

    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    for col in df_cat.columns:
        mode = df_cat[col].mode()
        df_cat[col] = df_cat[col].fillna(mode[0] if not mode.empty else "Unknown")
        df_cat[col] = LabelEncoder().fit_transform(df_cat[col].astype(str))

    if df_mice.shape[1]:
        imputed = pd.DataFrame(
            IterativeImputer(max_iter=25, random_state=RANDOM_STATE).fit_transform(df_mice),
            columns=df_mice.columns,
            index=df_mice.index,
        )
    else:
        imputed = pd.DataFrame(index=df_raw.index)

    for col in ["rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)"]:
        if col in imputed.columns:
            imputed[col] = imputed[col].clip(0, 1)

    df_final = pd.concat([imputed, df_numeric, df_cat, df_other.reset_index(drop=True)], axis=1)
    if "pid" in df_raw.columns:
        df_final.insert(0, "pid", df_raw["pid"].values)

    df = df_final[rotation_mask.values].copy()
    assert len(df) == 45980, f"Expected 45,980 observations, got {len(df)}"

    for col, new_col in {
        "pts": "pts_per100",
        "treb": "treb_per100",
        "ast": "ast_per100",
        "stl": "stl_per100",
        "blk": "blk_per100",
        "oreb": "oreb_per100",
        "dreb": "dreb_per100",
    }.items():
        if col in df.columns:
            df[new_col] = df[col] / df["Min_per"] * 100
            numeric_cols.append(new_col)

    for col in [
        "Min_per",
        "ORB_per",
        "DRB_per",
        "AST_per",
        "TO_per",
        "FT_per",
        "twoP_per",
        "TP_per",
        "blk_per",
        "stl_per",
        "usg",
        "eFG",
        "TS_per",
        "GP",
        "FTM",
        "FTA",
        "twoPM",
        "twoPA",
        "TPM",
        "TPA",
        "ftr",
        "porpag",
        "adjoe",
        "pfr",
        "ast/tov",
        "drtg",
        "adrtg",
        "dporpag",
        "stops",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "mp",
        "ogbpm",
        "dgbpm",
        "Ortg",
    ]:
        if col in df.columns:
            numeric_cols.append(col)

    numeric_cols = sorted({c for c in numeric_cols if c in df.columns})
    scaled = df[numeric_cols].copy().astype(float)
    scaled.loc[:, :] = RobustScaler().fit_transform(scaled)

    style_cols = [c for c in STYLE_COLS if c in scaled.columns]
    X = pd.DataFrame(MinMaxScaler().fit_transform(scaled[style_cols]), columns=style_cols, index=df.index)
    return df.loc[X.index].copy(), X, style_cols


def row_normalize(W: np.ndarray) -> np.ndarray:
    denom = W.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return W / denom


def hoyer_sparsity(A: np.ndarray) -> float:
    """Hoyer sparsity in [0, 1], averaged over rows."""
    A = np.asarray(A)
    n = A.shape[1]
    l1 = np.linalg.norm(A, ord=1, axis=1)
    l2 = np.linalg.norm(A, ord=2, axis=1)
    l2[l2 == 0] = 1.0
    return float(np.mean((math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1)))


def entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1) / math.log(P.shape[1])


def effective_archetypes(P: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(P).sum(axis=1)


def align_to_reference(reference: np.ndarray, labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Permute labels to maximize overlap with the reference KMeans labels."""
    table = np.zeros((k, k), dtype=int)
    for r, l in zip(reference, labels):
        if 0 <= r < k and 0 <= l < k:
            table[r, l] += 1
    row_ind, col_ind = linear_sum_assignment(-table)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(int(x), int(x)) for x in labels])


def fuzzy_cmeans(X: np.ndarray, k: int = 5, m: float = 2.0, max_iter: int = 300, tol: float = 1e-5):
    rng = np.random.default_rng(RANDOM_STATE)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=25).fit(X)
    centers = km.cluster_centers_.copy()
    U = rng.random((X.shape[0], k))
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        old = U.copy()
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        dist = np.maximum(dist, 1e-12)
        inv = dist ** (-2.0 / (m - 1.0))
        U = inv / inv.sum(axis=1, keepdims=True)
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
        if np.max(np.abs(U - old)) < tol:
            break
    return U, centers


@dataclass
class MethodResult:
    name: str
    labels: np.ndarray
    probabilities: np.ndarray | None
    basis: str


def cramers_v(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, c = ct.shape
    return float(math.sqrt((chi2 / n) / max(1, min(r - 1, c - 1))))


def external_validation(df: pd.DataFrame, labels: np.ndarray, method_name: str) -> dict[str, object]:
    path = BASE / "outputs" / "expanded_validation" / "expanded_validation_labeled_sample.csv"
    if not path.exists():
        return {}

    labeled = pd.read_csv(path, low_memory=False)
    labeled = labeled[labeled["position"].isin(["guard", "forward", "center"])].copy()
    tmp = df[["pid", "year", "team", "player_name"]].copy()
    tmp["method_label"] = labels

    def clean_id(value: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return str(int(number))
        return text

    for frame in (labeled, tmp):
        frame["pid"] = frame["pid"].map(clean_id)
        frame["year"] = frame["year"].map(clean_id)
        frame["team"] = frame["team"].astype(str).str.strip()
        frame["player_name"] = frame["player_name"].astype(str).str.strip()
    merged = labeled.merge(tmp, on=["pid", "year", "team", "player_name"], how="inner")
    if merged.empty:
        merged = labeled.merge(tmp.drop_duplicates(subset=["pid"]), on=["pid"], how="inner")
        merged = merged.drop_duplicates(subset=["pid"])
    else:
        merged = merged.drop_duplicates(subset=["pid", "year", "team", "player_name"])
    if merged.empty:
        return {}

    ct = pd.crosstab(merged["method_label"], merged["position"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"known_labels": int(len(merged))}

    chi2, p_value, dof, expected = chi2_contingency(ct)
    ct.to_csv(OUTDIR / f"{method_name}_expanded_validation_crosstab.csv")
    return {
        "known_labels": int(len(merged)),
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": cramers_v(ct),
        "any_expected_below_5": bool((expected < 5).any()),
    }


def method_metrics(X: np.ndarray, result: MethodResult, reference_labels: np.ndarray) -> dict[str, object]:
    labels = result.labels
    row = {
        "method": result.name,
        "basis": result.basis,
        "n_clusters": int(len(np.unique(labels))),
        "min_cluster_size": int(pd.Series(labels).value_counts().min()),
        "max_cluster_share": float(pd.Series(labels).value_counts(normalize=True).max()),
        "silhouette_sample10k": float(
            silhouette_score(X, labels, sample_size=min(10000, len(labels)), random_state=RANDOM_STATE)
        )
        if len(np.unique(labels)) > 1
        else np.nan,
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan,
        "davies_bouldin": float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan,
        "ari_vs_kmeans": float(adjusted_rand_score(reference_labels, labels)),
        "nmi_vs_kmeans": float(normalized_mutual_info_score(reference_labels, labels)),
    }
    if result.probabilities is not None:
        P = result.probabilities
        row.update(
            {
                "mean_max_membership": float(P.max(axis=1).mean()),
                "mean_entropy": float(entropy(P).mean()),
                "mean_effective_archetypes": float(effective_archetypes(P).mean()),
            }
        )
    return row


def fit_sampled_spectral(X: np.ndarray, sample_n: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X), size=min(sample_n, len(X)), replace=False))
    labels = SpectralClustering(
        n_clusters=5,
        affinity="nearest_neighbors",
        n_neighbors=25,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    ).fit_predict(X[idx])
    return idx, labels


def fit_local_ssc(X: np.ndarray, sample_n: int = 1500, n_neighbors: int = 35) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.sort(rng.choice(len(X), size=min(sample_n, len(X)), replace=False))
    Xs = X[idx]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Xs)
    distances, indices = nn.kneighbors(Xs)
    affinity = np.zeros((len(Xs), len(Xs)), dtype=float)
    alpha = 0.001

    for i in range(len(Xs)):
        neigh = indices[i, 1:]
        D = Xs[neigh].T
        y = Xs[i]
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000, positive=False)
        model.fit(D, y)
        coef = np.abs(model.coef_)
        if coef.sum() == 0:
            sigma = np.median(distances[i, 1:]) + 1e-12
            coef = np.exp(-distances[i, 1:] ** 2 / (2 * sigma ** 2))
        affinity[i, neigh] = coef

    affinity = np.maximum(affinity, affinity.T)
    labels = SpectralClustering(
        n_clusters=5,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    ).fit_predict(affinity)
    return idx, labels


def save_membership_outputs(df: pd.DataFrame, W: np.ndarray, P: np.ndarray, labels: np.ndarray) -> None:
    cols = [f"archetype_{i}_pct" for i in range(P.shape[1])]
    out = df[["pid", "player_name", "team", "conf", "yr", "year"]].copy()
    for i, col in enumerate(cols):
        out[col] = P[:, i]
    out["dominant_archetype"] = labels
    out["dominant_archetype_name"] = out["dominant_archetype"].map(PAPER_ARCHETYPES)
    out["dominance"] = P.max(axis=1)
    out["membership_entropy"] = entropy(P)
    out["effective_archetypes"] = effective_archetypes(P)
    out.to_csv(OUTDIR / "adaptive_nmf_memberships.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "mean_dominance",
                "median_dominance",
                "mean_entropy",
                "mean_effective_archetypes",
                "share_blended_entropy_gt_0_75",
                "share_clear_dominance_gt_0_60",
            ],
            "value": [
                P.max(axis=1).mean(),
                np.median(P.max(axis=1)),
                entropy(P).mean(),
                effective_archetypes(P).mean(),
                (entropy(P) > 0.75).mean(),
                (P.max(axis=1) > 0.60).mean(),
            ],
        }
    )
    summary.to_csv(OUTDIR / "adaptive_membership_summary.csv", index=False)


def save_theory_outputs(X: np.ndarray, W: np.ndarray, H: np.ndarray, nmf_model: NMF, P: np.ndarray) -> None:
    reconstruction = W @ H
    frob = np.linalg.norm(X - reconstruction, "fro")
    total = np.linalg.norm(X, "fro")
    cos = H @ H.T
    norms = np.sqrt(np.diag(cos))
    cos = cos / np.maximum(norms[:, None] * norms[None, :], 1e-12)

    diagnostics = {
        "nmf_reconstruction_error_sklearn": float(nmf_model.reconstruction_err_),
        "frobenius_relative_error": float(frob / total),
        "frobenius_variance_captured": float(1 - (frob**2 / total**2)),
        "activation_hoyer_sparsity": hoyer_sparsity(W),
        "component_hoyer_sparsity": hoyer_sparsity(H),
        "mean_abs_offdiag_component_cosine": float(np.mean(np.abs(cos[np.triu_indices_from(cos, k=1)]))),
        "mean_membership_entropy": float(entropy(P).mean()),
        "mean_effective_archetypes": float(effective_archetypes(P).mean()),
    }
    (OUTDIR / "nmf_theory_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    pd.DataFrame(cos).to_csv(OUTDIR / "nmf_component_cosine_similarity.csv", index=False)


def make_figures(P: np.ndarray, H: np.ndarray, style_cols: list[str]) -> None:
    sns.set_theme(style="whitegrid")
    ent = entropy(P)
    dom = P.max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sns.histplot(dom, bins=30, ax=axes[0], color="#2E6F95")
    axes[0].set_xlabel("Largest archetype percentage")
    axes[0].set_ylabel("Player-seasons")
    axes[0].set_title("Dominant Membership Strength")

    sns.histplot(ent, bins=30, ax=axes[1], color="#B5533D")
    axes[1].set_xlabel("Normalized membership entropy")
    axes[1].set_ylabel("Player-seasons")
    axes[1].set_title("Continuity / Blend Diagnostic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig18_adaptive_membership_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_features = []
    for i in range(H.shape[0]):
        s = pd.Series(H[i], index=style_cols).sort_values(ascending=False).head(6)
        for feature, loading in s.items():
            top_features.append({"dimension": f"Dim {i}", "feature": feature, "loading": loading})
    top_df = pd.DataFrame(top_features)
    top_df.to_csv(OUTDIR / "nmf_top_loadings_long.csv", index=False)


def main() -> None:
    start = time.time()
    log("Building Stage-2 matrix...")
    df, X_df, style_cols = build_stage2_matrix()
    X_input = X_df.to_numpy()

    log("Fitting NMF basis...")
    nmf_model = NMF(n_components=5, random_state=RANDOM_STATE, init="nndsvd", max_iter=500)
    W = nmf_model.fit_transform(X_input)
    H = nmf_model.components_
    P_nmf = row_normalize(W)
    adaptive_labels = P_nmf.argmax(axis=1)

    log("Fitting benchmark methods...")
    km = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=25).fit(W)
    km_labels = km.labels_

    gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=RANDOM_STATE, n_init=5).fit(W)
    gmm_labels = align_to_reference(km_labels, gmm.predict(W))
    gmm_probs = gmm.predict_proba(W)
    gmm_probs = gmm_probs[:, np.argsort(np.bincount(gmm.predict(W), minlength=5))[::-1]]

    bgmm = BayesianGaussianMixture(
        n_components=10,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        random_state=RANDOM_STATE,
        max_iter=1000,
    ).fit(W)
    bgmm_raw = bgmm.predict(W)
    bgmm_labels = bgmm_raw
    bgmm_probs = bgmm.predict_proba(W)

    fcm_probs, _ = fuzzy_cmeans(W, k=5, m=2.0)
    fcm_labels = align_to_reference(km_labels, fcm_probs.argmax(axis=1))

    methods = [
        MethodResult("Adaptive NMF membership (argmax)", adaptive_labels, P_nmf, "full NMF space"),
        MethodResult("KMeans on NMF weights", km_labels, None, "full NMF space"),
        MethodResult("Gaussian Mixture posterior", gmm_labels, gmm_probs, "full NMF space"),
        MethodResult("Bayesian Gaussian Mixture", bgmm_labels, bgmm_probs, "full NMF space"),
        MethodResult("Fuzzy C-Means membership", fcm_labels, fcm_probs, "full NMF space"),
    ]

    rows = []
    for result in methods:
        row = method_metrics(W, result, km_labels)
        row.update(external_validation(df, result.labels, result.name.lower().replace(" ", "_").replace("(", "").replace(")", "")))
        rows.append(row)

    log("Running sampled Spectral Clustering benchmark...")
    spectral_idx, spectral_labels = fit_sampled_spectral(W, sample_n=8000)
    spectral_labels = align_to_reference(km_labels[spectral_idx], spectral_labels)
    spectral_row = method_metrics(
        W[spectral_idx],
        MethodResult("Spectral Clustering", spectral_labels, None, "8,000-sample NMF space"),
        km_labels[spectral_idx],
    )
    rows.append(spectral_row)

    log("Running sampled local Sparse Subspace Clustering benchmark...")
    ssc_idx, ssc_labels = fit_local_ssc(W, sample_n=1500, n_neighbors=35)
    ssc_labels = align_to_reference(km_labels[ssc_idx], ssc_labels)
    ssc_row = method_metrics(
        W[ssc_idx],
        MethodResult("Local SSC approximation", ssc_labels, None, "1,500-sample NMF space"),
        km_labels[ssc_idx],
    )
    rows.append(ssc_row)

    bic_rows = []
    for k in range(2, 11):
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=3).fit(W)
        bic_rows.append({"k": k, "bic": model.bic(W), "aic": model.aic(W)})
    pd.DataFrame(bic_rows).to_csv(OUTDIR / "gmm_model_selection_bic_aic.csv", index=False)

    save_membership_outputs(df, W, P_nmf, adaptive_labels)
    save_theory_outputs(X_input, W, H, nmf_model, P_nmf)
    make_figures(P_nmf, H, style_cols)

    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(OUTDIR / "clustering_benchmark_summary.csv", index=False)

    loadings = pd.DataFrame(H.T, index=style_cols, columns=[f"Dim{i}" for i in range(5)])
    loadings.to_csv(OUTDIR / "nmf_component_loadings.csv")
    np.save(OUTDIR / "nmf_weights.npy", W)
    np.save(OUTDIR / "adaptive_membership_matrix.npy", P_nmf)

    log(benchmark.round(4).to_string(index=False))
    log(f"Done in {time.time() - start:.0f}s. Outputs: {OUTDIR}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
