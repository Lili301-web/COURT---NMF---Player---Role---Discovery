"""Monte Carlo convergence and stability study for COURT-NMF.

The published COURT-NMF fit is a single projected-gradient run from one seeded
random simplex initialisation (seed 42).  Because the objective of Eq.(1) is
not jointly convex in (W, H, B), that single run establishes neither that the
solver has converged nor that the solution it reaches is typical of the
solutions reachable from other starting points.  This script answers both
questions empirically: it runs COURT-NMF from ``--runs`` independent random
initialisations and records, for every run,

  * weighted reconstruction MSE and adjacent-season trajectory MSE;
  * the full constrained objective of Eq.(1) and its trajectory over iterations;
  * iterations to convergence, and whether the run converged within budget;
  * mixture reconstruction R^2 on the shared 23-feature style space;
  * external-validation PERMANOVA R^2 on CLR-transformed memberships, plus
    out-of-fold multinomial accuracy and macro-F1 against the 195 independently
    labelled player-seasons -- the same protocol as Section 6;
  * stability of the fitted solution relative to the published seed-42
    reference, after optimal role matching.

Two sampling modes are supported:

  ``--mode full``       every run fits the entire analytic panel.  This is the
                        expensive, definitive design.
  ``--mode subsample``  every run fits a random subsample of *players* (not of
                        rows -- sampling rows would destroy the adjacent-season
                        career edges the trajectory term is defined on).  This
                        is the cheap design that supports a large R.

Both write one row per run, appending as they go, so a long study is resumable:
re-running the same command skips seeds already present in the output file.

Usage
-----
    # the definitive full-sample study
    python monte_carlo_convergence.py --mode full --runs 1000 --iterations 4000 --workers 8

    # the large-R subsample study
    python monte_carlo_convergence.py --mode subsample --runs 8000 \
        --subsample-players 4000 --iterations 4000 --workers 8

    # summarise whatever has been completed so far (safe to run mid-study)
    python monte_carlo_convergence.py --mode full --summarize-only

Outputs (under ``outputs/monte_carlo/``)
    mc_runs_<mode>.csv       one row per run -- the table the paper reports
    mc_history_<mode>.csv.gz per-iteration objective trace for every run
    mc_seeds_<mode>.csv      the seed list, for the supplement
    mc_meta_<mode>.json      configuration, environment and timing record
    mc_summary_<mode>.json   means, 95% CIs, quantiles, non-convergence rates
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "monte_carlo"
LABELS = BASE / "outputs" / "expanded_validation" / "expanded_validation_labeled_sample.csv"
POSITIONS = ["guard", "forward", "center"]
REFERENCE_SEED = 42

# Import the canonical experiment module by path, so this script stays a sibling
# of it and needs no package installation.
_spec = importlib.util.spec_from_file_location("cne", Path(__file__).with_name("court_nmf_experiment.py"))
cne = importlib.util.module_from_spec(_spec)
sys.modules["cne"] = cne
_spec.loader.exec_module(cne)

_WORKER = {}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def clr(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Centred log-ratio transform of compositional rows (matches Section 6)."""
    Ac = np.clip(A, eps, None)
    Ac = Ac / Ac.sum(axis=1, keepdims=True)
    logA = np.log(Ac)
    return logA - logA.mean(axis=1, keepdims=True)


def permanova_r2(D: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """One-way PERMANOVA R^2 and F (Anderson 2001), without the permutation null.

    The permutation p-value is a property of the published fit and is computed
    once by ``court_nmf_multivariate_validation.py``; running 9,999 permutations
    inside every Monte Carlo run would dominate the cost and answer a question
    nobody asked.  R^2 and F are exact and cheap, and they are what a
    distribution over runs is informative about.
    """
    n = D.shape[0]
    D2 = D ** 2
    ss_total = D2.sum() / (2 * n)
    ss_w = 0.0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        ss_w += D2[np.ix_(idx, idx)].sum() / (2 * len(idx))
    ss_a = ss_total - ss_w
    k = len(np.unique(groups))
    df_a, df_w = k - 1, n - k
    if ss_w <= 0 or ss_total <= 0 or df_w <= 0:
        return float("nan"), float("nan")
    return float(ss_a / ss_total), float((ss_a / df_a) / (ss_w / df_w))


def mixture_r2(W: np.ndarray, X: np.ndarray) -> float:
    """Variance of the style space recovered by the best linear read-out of W."""
    C, *_ = np.linalg.lstsq(W, X, rcond=None)
    resid = ((X - W @ C) ** 2).sum()
    total = ((X - X.mean(axis=0)) ** 2).sum()
    return float(1 - resid / total) if total > 0 else float("nan")


def match_roles(H_ref: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimally permute H's roles onto the reference by cosine similarity.

    Role identity is only defined up to permutation, so any stability metric
    that compares two fits without first solving this assignment measures label
    switching rather than instability.
    """
    a = H_ref / np.maximum(np.linalg.norm(H_ref, axis=1, keepdims=True), 1e-12)
    b = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    row, col = linear_sum_assignment(-sim)
    return col, float(sim[row, col].mean())


def convergence_point(history: pd.DataFrame, tol: float, patience: int) -> tuple[float, bool, float]:
    """First recorded iteration after which the objective stops moving.

    Returns (iteration, converged, final relative step).  ``patience``
    consecutive recorded steps below ``tol`` are required, so a single flat
    step inside a plateau that later resumes descending does not count.
    """
    obj = history["objective"].to_numpy(float)
    it = history["iteration"].to_numpy(int)
    if len(obj) < 2:
        return float("nan"), False, float("nan")
    rel = np.abs(np.diff(obj)) / np.maximum(np.abs(obj[:-1]), 1e-30)
    run = 0
    for i, r in enumerate(rel):
        run = run + 1 if r < tol else 0
        if run >= patience:
            return float(it[i - patience + 2]), True, float(rel[-1])
    return float("nan"), False, float(rel[-1])


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------
def _init_worker(mode: str, iterations: int, n_players: int, tol: float, patience: int,
                 step_rule: str = "backtracking"):
    tag = f"{mode}_{iterations}"
    df, X, features = cne.preprocess()
    labels = pd.read_csv(LABELS)
    labels = labels[labels["position"].isin(POSITIONS)].copy()
    labels["pid"] = labels["pid"].astype(str).str.strip()
    _WORKER.update(
        df=df, X=X, features=features, labels=labels, mode=mode,
        iterations=iterations, n_players=n_players, tol=tol, patience=patience,
        step_rule=step_rule,
        edges=cne.career_edges(df) if mode == "full" else None,
        players=df["pid"].astype(str).str.strip().unique(),
    )
    # The 195 labelled player-seasons are 0.4% of the panel, so a uniform player
    # subsample would contain almost none of them and the external-validation
    # metrics would be undefined for most runs.  Every subsample therefore
    # retains the labelled players and samples the remainder at random.  The
    # labels are still never seen during fitting; this only fixes which players
    # the external criterion can be evaluated on, holding it comparable across
    # runs while the rest of the panel varies.
    pid_all = df["pid"].astype(str).str.strip()
    _WORKER["anchor_players"] = np.array(sorted(set(labels["pid"]) & set(pid_all)))
    _WORKER["pool_players"] = np.array(sorted(set(pid_all) - set(_WORKER["anchor_players"])))
    ref = OUT / f"reference_H_{tag}.npy"
    _WORKER["H_ref"] = np.load(ref) if ref.exists() else None
    ref_w = OUT / f"reference_labels_{tag}.npy"
    _WORKER["labels_ref"] = np.load(ref_w) if ref_w.exists() else None


def _external_metrics(frame: pd.DataFrame, W: np.ndarray) -> dict:
    """PERMANOVA R^2 / F and out-of-fold classification on the labelled join."""
    lab = _WORKER["labels"]
    key = pd.DataFrame({"pid": frame["pid"].astype(str).str.strip(),
                        "year": frame["year"].astype(int)})
    key["row"] = np.arange(len(key))
    merged = lab.merge(key, on=["pid", "year"], how="inner")
    out = {"n_labeled": int(len(merged)), "permanova_r2": np.nan, "permanova_f": np.nan,
           "cv_accuracy": np.nan, "cv_macro_f1": np.nan}
    if len(merged) < 30 or merged["position"].nunique() < 2:
        return out
    Z = clr(W[merged["row"].to_numpy()])
    y = merged["position"].to_numpy()
    if min(pd.Series(y).value_counts()) < 5:
        return out
    D = squareform(pdist(Z, metric="euclidean"))
    out["permanova_r2"], out["permanova_f"] = permanova_r2(D, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=REFERENCE_SEED)
    pred = cross_val_predict(LogisticRegression(max_iter=2000), Z, y, cv=cv)
    out["cv_accuracy"] = float(accuracy_score(y, pred))
    out["cv_macro_f1"] = float(f1_score(y, pred, average="macro"))
    return out


def run_one(seed: int) -> tuple[dict, pd.DataFrame]:
    w = _WORKER
    t0 = time.time()
    if w["mode"] == "full":
        frame, X, edges = w["df"], w["X"], w["edges"]
    else:
        rng = np.random.default_rng(seed)
        anchor = w["anchor_players"]
        extra = max(0, min(w["n_players"], len(w["players"])) - len(anchor))
        chosen = set(anchor.tolist()) | set(
            rng.choice(w["pool_players"], size=min(extra, len(w["pool_players"])),
                       replace=False).tolist())
        # Sample players, keep all of their seasons: row sampling would sever the
        # adjacent-season edges the trajectory penalty is defined on.
        mask = w["df"]["pid"].astype(str).str.strip().isin(chosen).to_numpy()
        frame = w["df"].loc[mask].reset_index(drop=True)
        X = w["X"][mask]
        edges = None
    W, H, B, output = cne.fit_court(frame, X, iterations=w["iterations"], seed=seed, edges=edges,
                                    tol=w["tol"], patience=w["patience"],
                                    step_rule=w.get("step_rule", "backtracking"))
    hist = pd.DataFrame(output["history"])
    # fit_court now stops on the convergence criterion itself, so its own flags
    # are authoritative; the post-hoc scan stays as the source of the relative
    # final step and as a fallback for a fit run with tol disabled.
    it_scan, conv_scan, rel_final = convergence_point(hist, w["tol"], w["patience"])
    diag = output["diagnostics"]
    converged = bool(diag.get("converged", conv_scan))
    it_conv = float(diag["stopped_at"]) if converged else it_scan

    rec = {"seed": seed, "mode": w["mode"], "n_obs": int(len(frame)),
           "n_players": int(frame["pid"].nunique()), "iterations": w["iterations"]}
    rec.update({k: v for k, v in diag.items()
                if k in ("weighted_mse", "trajectory_mse", "within_team_trajectory_mse",
                         "cross_team_trajectory_mse", "mean_entropy", "mean_max_membership",
                         "career_edges", "cross_team_edges")})
    rec["final_objective"] = float(hist["objective"].iloc[-1])
    rec["objective_per_obs"] = rec["final_objective"] / max(len(frame), 1)
    rec["iters_to_convergence"] = it_conv
    rec["converged"] = bool(converged)
    rec["final_rel_step"] = rel_final
    rec["mixture_r2"] = mixture_r2(W, X)
    rec.update(_external_metrics(frame, W))

    if w["H_ref"] is not None:
        perm, cos = match_roles(w["H_ref"], H)
        rec["stability_prototype_cosine"] = cos
        if w["mode"] == "full" and w["labels_ref"] is not None:
            rec["stability_membership_ari"] = float(adjusted_rand_score(w["labels_ref"], W.argmax(1)))
        else:
            rec["stability_membership_ari"] = np.nan
    else:
        rec["stability_prototype_cosine"] = np.nan
        rec["stability_membership_ari"] = np.nan
    rec["wall_seconds"] = round(time.time() - t0, 2)
    hist = hist.assign(seed=seed)
    return rec, hist


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def build_reference(mode: str, iterations: int, step_rule: str = "backtracking",
                    tol: float = 1e-4, patience: int = 3) -> None:
    """Fit the published seed-42 solution, which every run is compared against."""
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_{iterations}"
    if (OUT / f"reference_H_{tag}.npy").exists():
        return
    print(f"[reference] fitting seed {REFERENCE_SEED} at {iterations} iterations ...", flush=True)
    df, X, _ = cne.preprocess()
    W, H, B, out = cne.fit_court(df, X, iterations=iterations, seed=REFERENCE_SEED,
                                 step_rule=step_rule, tol=tol, patience=patience)
    np.save(OUT / f"reference_H_{tag}.npy", H)
    np.save(OUT / f"reference_labels_{tag}.npy", W.argmax(1))
    np.save(OUT / f"reference_W_{tag}.npy", W.astype(np.float32))
    # The reference needs the same derived metrics the runs report, or the
    # Monte Carlo distribution has no published value to be compared against.
    labels = pd.read_csv(LABELS)
    labels = labels[labels["position"].isin(POSITIONS)].copy()
    labels["pid"] = labels["pid"].astype(str).str.strip()
    _WORKER["labels"] = labels
    diag = dict(out["diagnostics"])
    hist_ref = pd.DataFrame(out["history"])
    diag["final_objective"] = float(hist_ref["objective"].iloc[-1])
    diag["objective_per_obs"] = diag["final_objective"] / max(len(df), 1)
    diag["mixture_r2"] = mixture_r2(W, X)
    diag.update(_external_metrics(df, W))
    (OUT / f"reference_diagnostics_{tag}.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8")
    out["diagnostics"] = diag
    pd.DataFrame(out["history"]).to_csv(OUT / f"reference_history_{tag}.csv", index=False)
    print(f"[reference] done: {json.dumps(out['diagnostics'])}", flush=True)


def summarize(mode: str, iterations: int | None = None, local_tol: float = 1e-3) -> dict:
    runs_path = OUT / f"mc_runs_{mode}.csv"
    if not runs_path.exists():
        raise SystemExit(f"no runs recorded yet at {runs_path}")
    d = pd.read_csv(runs_path)
    it = iterations if iterations is not None else int(d["iterations"].mode().iloc[0])
    ref_path = OUT / f"reference_diagnostics_{mode}_{it}.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}

    # A binary converged/not flag is a blunt summary when no run trips a strict
    # tolerance inside the budget.  Derive, from the recorded objective traces,
    # the iteration at which each run first comes within 1% and 0.1% of its own
    # final objective: that says how far along the budget carried a run even when
    # the strict criterion is never met.
    hist_path = OUT / f"mc_history_{mode}.csv.gz"
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        approach = []
        for seed, grp in h.groupby("seed"):
            grp = grp.sort_values("iteration")
            obj = grp["objective"].to_numpy(float)
            it = grp["iteration"].to_numpy(int)
            rec = {"seed": int(seed)}
            for eps, name in ((0.01, "iters_to_within_1pct"), (0.001, "iters_to_within_0p1pct")):
                hit = np.where(obj <= obj[-1] * (1 + eps))[0]
                rec[name] = float(it[hit[0]]) if len(hit) else float("nan")
            rec["final_obj_gap_vs_best_pct"] = float("nan")
            approach.append(rec)
        ap_df = pd.DataFrame(approach)
        d = d.merge(ap_df.drop(columns=["final_obj_gap_vs_best_pct"]), on="seed", how="left")

    # Comparing objectives across runs only means something when the runs fit the
    # same data.  In subsample mode each run sees a different draw of players, so
    # a difference in objective mixes optimisation quality with sample
    # composition; the local-optimum count and the gap-to-best are therefore
    # computed for the full-sample design only.
    comparable = (mode == "full")
    if comparable:
        best = d["objective_per_obs"].min()
        d["objective_gap_vs_best_pct"] = (d["objective_per_obs"] / best - 1) * 100
        d["local_optimum"] = d["objective_per_obs"] > best * (1 + local_tol)
    else:
        d["local_optimum"] = False

    metrics = ["weighted_mse", "trajectory_mse", "mixture_r2", "permanova_r2",
               "cv_accuracy", "cv_macro_f1", "mean_max_membership", "mean_entropy",
               "final_objective", "objective_per_obs", "iters_to_convergence",
               "iters_to_within_1pct", "iters_to_within_0p1pct", "objective_gap_vs_best_pct",
               "stability_prototype_cosine", "stability_membership_ari"]
    summary = {"mode": mode, "runs": int(len(d)), "objectives_comparable": bool(comparable),
               "converged_fraction": float(d["converged"].mean()),
               "nonconvergent_fraction": float(1 - d["converged"].mean()),
               "local_optimum_fraction": float(d["local_optimum"].mean()) if comparable else None,
               "reference_seed": REFERENCE_SEED, "reference_diagnostics": ref,
               "wall_seconds_total": float(d["wall_seconds"].sum()),
               "wall_seconds_mean": float(d["wall_seconds"].mean()),
               "metrics": {}}
    for m in metrics:
        if m not in d:
            continue
        v = d[m].dropna().to_numpy(float)
        if not len(v):
            continue
        entry = {"n": int(len(v)), "mean": float(v.mean()),
                 "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                 "median": float(np.median(v)),
                 "ci95_lo": float(np.percentile(v, 2.5)),
                 "ci95_hi": float(np.percentile(v, 97.5)),
                 "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                 "min": float(v.min()), "max": float(v.max())}
        # CI of the mean, alongside the percentile interval of the distribution
        if len(v) > 1:
            se = v.std(ddof=1) / np.sqrt(len(v))
            entry["mean_ci95_lo"] = float(v.mean() - 1.96 * se)
            entry["mean_ci95_hi"] = float(v.mean() + 1.96 * se)
        if m in ref and comparable:
            entry["seed42"] = float(ref[m])
            entry["deviation_from_seed42"] = float(v.mean() - ref[m])
            if ref[m]:
                entry["pct_deviation_from_seed42"] = float((v.mean() - ref[m]) / abs(ref[m]) * 100)
            entry["seed42_percentile"] = float((v < ref[m]).mean() * 100)
        summary["metrics"][m] = entry
    # --- multi-start selection -------------------------------------------
    # If the objective is a good selection criterion, keeping the best of N
    # random restarts should improve the quantities the paper reports and
    # narrow their spread.  Whether it does is an empirical question, answered
    # here by resampling the runs already performed rather than by new fits.
    QUAL = ["objective_per_obs", "mixture_r2", "permanova_r2", "cv_accuracy",
            "cv_macro_f1", "weighted_mse", "trajectory_mse", "mean_max_membership"]
    if comparable and "objective_per_obs" in d and len(d) >= 4:
        obj = d["objective_per_obs"].to_numpy(float)
        summary["objective_correlation"] = {
            m: float(d["objective_per_obs"].corr(d[m], method="spearman"))
            for m in QUAL if m in d and d[m].notna().sum() > 2 and m != "objective_per_obs"}
        rng = np.random.default_rng(0)
        draws = 4000
        curve = []
        for N in (1, 2, 3, 5, 10, 20, 50):
            if len(d) < N:
                continue
            idx = rng.integers(0, len(d), size=(draws, N))
            winner = idx[np.arange(draws), np.argmin(obj[idx], axis=1)]
            sel = d.iloc[winner]
            entry = {"restarts": int(N)}
            for m in QUAL:
                if m not in sel:
                    continue
                v = sel[m].dropna().to_numpy(float)
                if not len(v):
                    continue
                entry[m] = {"mean": float(v.mean()),
                            "ci95_lo": float(np.percentile(v, 2.5)),
                            "ci95_hi": float(np.percentile(v, 97.5))}
            curve.append(entry)
        summary["restart_selection"] = curve

    (OUT / f"mc_summary_{mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "subsample"], default="full")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--subsample-players", type=int, default=4000,
                    help="players sampled per run in subsample mode (all their seasons are kept)")
    ap.add_argument("--seed-base", type=int, default=100000,
                    help="run r uses seed seed_base + r; kept away from 42 so the reference is distinct")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--tol", type=float, default=1e-5, help="relative objective change counted as converged")
    ap.add_argument("--patience", type=int, default=3, help="consecutive recorded steps below --tol")
    ap.add_argument("--step-rule", default="backtracking",
                    choices=["backtracking", "mu", "eg", "fixed", "harmonic"],
                    help="solver step rule; 'mu' is the Lee-Seung multiplicative scheme")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        summarize(a.mode, None)
        return

    build_reference(a.mode, a.iterations, a.step_rule, a.tol, a.patience)

    runs_path = OUT / f"mc_runs_{a.mode}.csv"
    hist_path = OUT / f"mc_history_{a.mode}.csv.gz"
    seeds = [a.seed_base + r for r in range(a.runs)]
    done = set()
    if runs_path.exists():
        done = set(pd.read_csv(runs_path)["seed"].astype(int).tolist())
        print(f"resuming: {len(done)} of {len(seeds)} runs already recorded", flush=True)
    todo = [s for s in seeds if s not in done]
    pd.DataFrame({"run": range(len(seeds)), "seed": seeds}).to_csv(
        OUT / f"mc_seeds_{a.mode}.csv", index=False)

    meta = {"mode": a.mode, "runs_requested": a.runs, "iterations": a.iterations,
            "subsample_players": a.subsample_players if a.mode == "subsample" else None,
            "seed_base": a.seed_base, "reference_seed": REFERENCE_SEED,
            "tol": a.tol, "patience": a.patience, "workers": a.workers,
            "step_rule": a.step_rule,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__}
    (OUT / f"mc_meta_{a.mode}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not todo:
        print("nothing to do; summarising")
        summarize(a.mode, a.iterations)
        return

    print(f"running {len(todo)} fits, mode={a.mode}, iterations={a.iterations}, "
          f"workers={a.workers}", flush=True)
    t0 = time.time()
    init_args = (a.mode, a.iterations, a.subsample_players, a.tol, a.patience, a.step_rule)
    written = len(done)

    def flush(rec: dict, hist: pd.DataFrame) -> None:
        pd.DataFrame([rec]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        hist.to_csv(hist_path, mode="a", header=not hist_path.exists(), index=False,
                    compression={"method": "gzip"})

    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(a.workers, initializer=_init_worker, initargs=init_args) as pool:
            for rec, hist in pool.imap_unordered(run_one, todo, chunksize=1):
                flush(rec, hist)
                written += 1
                el = time.time() - t0
                print(f"[{written}/{len(seeds)}] seed {rec['seed']} "
                      f"obj/obs {rec['objective_per_obs']:.5g} R2 {rec['mixture_r2']:.4f} "
                      f"conv {rec['converged']} {rec['wall_seconds']:.0f}s "
                      f"| elapsed {el/60:.1f}m", flush=True)
    else:
        _init_worker(*init_args)
        for s in todo:
            rec, hist = run_one(s)
            flush(rec, hist)
            written += 1
            print(f"[{written}/{len(seeds)}] seed {s} done {rec['wall_seconds']:.0f}s", flush=True)

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["wall_seconds"] = round(time.time() - t0, 1)
    (OUT / f"mc_meta_{a.mode}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summarize(a.mode, a.iterations)


if __name__ == "__main__":
    main()
