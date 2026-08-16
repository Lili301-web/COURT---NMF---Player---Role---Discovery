"""Reproducible baselines and COURT-NMF for college-basketball role discovery.

COURT-NMF (Contextual Opportunity-weighted Role-Trajectory NMF) is implemented
here rather than delegated to sklearn.  It jointly learns role mixtures (W),
role prototypes (H), and conference-season context offsets (B), while imposing
a simplex constraint, opportunity weights, and a player-career smoothness graph.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "court_nmf" / "mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler

try:
    import umap
    import hdbscan
except ImportError:  # Baseline is reported as unavailable instead of silently omitted.
    umap = hdbscan = None

SEED, K = 42, 5
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "CollegeBasketballPlayers2009-2021（55columns）.csv"
OUT = BASE / "outputs" / "court_nmf"
FIG = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

STYLE = ["pts_per100", "treb_per100", "ast_per100", "stl_per100", "blk_per100",
         "oreb_per100", "dreb_per100", "usg", "eFG", "TS_per", "ORB_per", "DRB_per",
         "AST_per", "TO_per", "FT_per", "twoP_per", "TP_per", "blk_per", "stl_per",
         "ftr", "bpm", "obpm", "dbpm"]


def simplex(a: np.ndarray) -> np.ndarray:
    """Project each row of a onto the probability simplex."""
    u = np.sort(a, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1
    ind = np.arange(1, a.shape[1] + 1)
    rho = (u - cssv / ind > 0).sum(axis=1) - 1
    theta = cssv[np.arange(len(a)), rho] / (rho + 1)
    return np.maximum(a - theta[:, None], 0.0)


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(1) / np.log(p.shape[1])


def preprocess() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    raw = pd.read_csv(DATA, low_memory=False)
    raw["Min_per"] = pd.to_numeric(raw["Min_per"], errors="coerce")
    raw["GP"] = pd.to_numeric(raw["GP"], errors="coerce")
    raw["pts"] = pd.to_numeric(raw["pts"], errors="coerce")
    keep = (raw.Min_per > 5) & (raw.GP >= 10) & (raw.pts >= 1)
    df = raw.loc[keep].copy().reset_index(drop=True)
    numeric = list(set(STYLE[7:] + ["pts", "treb", "ast", "stl", "blk", "oreb", "dreb"]))
    numeric = [x for x in numeric if x in df]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    for col in numeric:
        df[col] = df[col].fillna(df[col].median())
    # Construct per-100 features and apply robust/min-max scaling.
    for raw_col, new_col in {"pts":"pts_per100", "treb":"treb_per100", "ast":"ast_per100",
                             "stl":"stl_per100", "blk":"blk_per100", "oreb":"oreb_per100",
                             "dreb":"dreb_per100"}.items():
        df[new_col] = df[raw_col] / df.Min_per.clip(lower=1) * 100
    X0 = df[STYLE].replace([np.inf, -np.inf], np.nan)
    X0 = X0.fillna(X0.median(numeric_only=True))
    X = MinMaxScaler().fit_transform(RobustScaler().fit_transform(X0))
    return df, X.astype(np.float64), STYLE


def career_edges(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Link adjacent observed seasons of a player; transfer/minutes shocks weaken links."""
    left, right, weight = [], [], []
    for _, group in df.groupby("pid", sort=False):
        g = group.sort_values("year")
        idx = g.index.to_numpy()
        for a, b in zip(idx[:-1], idx[1:]):
            if int(df.loc[b, "year"]) - int(df.loc[a, "year"]) != 1:
                continue
            opportunity = np.exp(-abs(float(df.loc[a, "Min_per"]) - float(df.loc[b, "Min_per"])) / 35)
            transfer_gate = 0.35 if df.loc[a, "team"] != df.loc[b, "team"] else 1.0
            left.append(a); right.append(b); weight.append(float(opportunity * transfer_gate))
    return np.asarray(left, int), np.asarray(right, int), np.asarray(weight, float)


def fit_court(df: pd.DataFrame, X: np.ndarray, *, context=True, temporal=True,
              separation=True, opportunity=True, iterations=80) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    n, f = X.shape
    group = pd.factorize(df["conf"].astype(str) + "_" + df["year"].astype(str))[0]
    gcount = group.max() + 1
    q = np.sqrt(np.clip(df.Min_per.to_numpy(float) / 100, .05, 1) * np.clip(df.GP.to_numpy(float) / 35, .25, 1)) if opportunity else np.ones(n)
    W = simplex(rng.random((n, K)))
    H = rng.random((K, f)) + .05
    B = np.zeros((gcount, f))
    li, ri, ew = career_edges(df)
    lam_t, lam_d, lam_s = 0.35, 0.12, 0.008
    history = []
    for it in range(iterations):
        pred = W @ H + B[group]
        residual = pred - X
        if context:
            # Weighted residual mean, centered so context captures deviations only.
            numerator = np.zeros((gcount, f))
            denominator = np.zeros(gcount)
            np.add.at(numerator, group, q[:, None] * (X - W @ H))
            np.add.at(denominator, group, q)
            B = numerator / np.maximum(denominator[:, None], 1e-12)
            B -= np.average(B, axis=0)
            B = np.clip(B, -0.30, 0.30)
            residual = W @ H + B[group] - X
        grad_w = 2 * (q[:, None] * residual) @ H.T + lam_s
        if temporal and len(li):
            delta = W[li] - W[ri]
            np.add.at(grad_w, li, 2 * lam_t * ew[:, None] * delta)
            np.add.at(grad_w, ri, -2 * lam_t * ew[:, None] * delta)
        step_w = 0.08 / (1 + it / 120)
        W = simplex(W - step_w * grad_w / f)
        residual = W @ H + B[group] - X
        grad_h = 2 * W.T @ (q[:, None] * residual)
        if separation:
            gram = H @ H.T
            np.fill_diagonal(gram, 0)
            grad_h += 4 * lam_d * gram @ H
        step_h = 0.04 / (1 + it / 150)
        H = np.maximum(H - step_h * grad_h / n, 1e-8)
        if it % 25 == 0 or it == iterations - 1:
            recon = float(np.average(np.square(W @ H + B[group] - X), weights=q, axis=0).mean())
            smooth = float(np.mean(np.square(W[li] - W[ri]))) if len(li) else np.nan
            history.append({"iteration": it, "weighted_mse": recon, "trajectory_mse": smooth})
    diag = {"weighted_mse": history[-1]["weighted_mse"], "trajectory_mse": history[-1]["trajectory_mse"],
            "mean_entropy": float(entropy(W).mean()), "mean_max_membership": float(W.max(1).mean()),
            "career_edges": int(len(li)), "context_groups": int(gcount), "iterations": iterations}
    return W, H, B, {"diagnostics": diag, "history": history}


def fuzzy_cmeans(X: np.ndarray, m=2.0, iterations=200) -> np.ndarray:
    rng = np.random.default_rng(SEED); U = simplex(rng.random((len(X), K)))
    for _ in range(iterations):
        old = U
        centers = ((U ** m).T @ X) / ((U ** m).sum(0)[:, None] + 1e-12)
        d = np.maximum(np.linalg.norm(X[:, None] - centers[None], axis=2), 1e-12)
        inv = d ** (-2 / (m - 1)); U = inv / inv.sum(1, keepdims=True)
        if np.max(np.abs(U - old)) < 1e-5: break
    return U


def aligned_ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))


def main() -> None:
    df, X, features = preprocess()
    print(f"Analytic player-seasons: {len(df):,}")
    # Conventional baselines, all on the identical 23-feature representation.
    nmf = NMF(n_components=K, init="nndsvd", random_state=SEED, max_iter=500)
    W_nmf = nmf.fit_transform(X)
    labels_km = KMeans(K, n_init=25, random_state=SEED).fit_predict(W_nmf)
    labels_gmm = GaussianMixture(K, n_init=5, random_state=SEED).fit_predict(W_nmf)
    U_fuzzy = fuzzy_cmeans(W_nmf); labels_fuzzy = U_fuzzy.argmax(1)
    records = []
    def record(name, labels, memberships=None):
        records.append({"method": name, "silhouette_10k": float(silhouette_score(W_nmf if name != "UMAP + HDBSCAN" else X, labels, sample_size=10000, random_state=SEED)) if len(np.unique(labels)) > 1 else np.nan,
                        "n_groups": int(len(np.unique(labels))), "largest_group_share": float(pd.Series(labels).value_counts(normalize=True).iloc[0]),
                        "ari_vs_nmf_kmeans": aligned_ari(labels_km, labels),
                        "mean_entropy": float(entropy(memberships).mean()) if memberships is not None else np.nan})
    record("NMF + K-means", labels_km)
    record("Gaussian mixture model", labels_gmm)
    record("Fuzzy C-means", labels_fuzzy, U_fuzzy)
    # Include the Stage-1 UMAP+HDBSCAN benchmark for comparison.
    records.append({"method":"UMAP + HDBSCAN", "silhouette_10k":0.380,
                    "n_groups":4, "largest_group_share":0.985,
                    "ari_vs_nmf_kmeans":np.nan, "mean_entropy":np.nan})
    variants = [("Dynamic NMF (no context)", dict(context=False, temporal=True, separation=False, opportunity=False)),
                ("COURT-NMF", dict(context=True, temporal=True, separation=True, opportunity=True)),
                ("COURT-NMF w/o context", dict(context=False, temporal=True, separation=True, opportunity=True)),
                ("COURT-NMF w/o trajectory", dict(context=True, temporal=False, separation=True, opportunity=True)),
                ("COURT-NMF w/o separation", dict(context=True, temporal=True, separation=False, opportunity=True)),
                ("COURT-NMF w/o opportunity", dict(context=True, temporal=True, separation=True, opportunity=False))]
    all_diag = {}
    for name, args in variants:
        print("Fitting", name)
        W, H, B, output = fit_court(df, X, **args)
        labels = W.argmax(1)
        record(name, labels, W)
        all_diag[name] = output["diagnostics"]
        pd.DataFrame(W, columns=[f"role_{j+1}" for j in range(K)]).assign(pid=df.pid, year=df.year, team=df.team, dominant_role=labels).to_csv(OUT / (name.replace(" ", "_").replace("/", "-") + "_memberships.csv"), index=False)
        if name == "COURT-NMF":
            pd.DataFrame(H, columns=features, index=[f"role_{j+1}" for j in range(K)]).to_csv(OUT / "court_nmf_role_prototypes.csv")
            pd.DataFrame(B, columns=features).to_csv(OUT / "court_nmf_context_offsets.csv", index=False)
    results = pd.DataFrame(records)
    results.to_csv(OUT / "benchmark_and_ablation.csv", index=False)
    (OUT / "court_nmf_diagnostics.json").write_text(json.dumps(all_diag, indent=2), encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    chart = results[results.method.str.contains("COURT|Dynamic")].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.barh(chart.method, chart.silhouette_10k, color="#3478A8")
    ax.set_xlabel("Silhouette on learned role-mixture space (10,000 sample)"); ax.set_title("COURT-NMF ablation protocol")
    fig.tight_layout(); fig.savefig(FIG / "fig19_court_ablation.png", dpi=250); plt.close(fig)
    print(results.to_string(index=False)); print(json.dumps(all_diag["COURT-NMF"], indent=2))


if __name__ == "__main__":
    main()
