"""Matched multi-start ablations with block-wise convergence tracing (revision R3).

Reviewer requests addressed here
--------------------------------
1. *Matched multi-start runs for all ablations.*  Every variant in the benchmark
   (graph-only dynamic NMF, full COURT-NMF and its four term-removal ablations)
   is fitted from the SAME ten seeded initialisations (seeds 42..51) with the
   SAME stopping rule, and the reported fit of each variant is selected by the
   SAME rule (lowest final value of that variant's own objective).  Per-seed
   results are kept, so paired full-minus-ablation differences can be reported
   seed by seed.
3. *Convergence diagnostics covering the context-offset update.*  ``fit_traced``
   is a line-for-line copy of ``court_nmf_experiment.fit_court`` (backtracking
   rule) that additionally records, at every iteration, the objective change
   caused by each block update (B, then W, then H), the relative change of each
   block, the share of offsets at the clipping bound and the centering residual;
   and, at every 25th iteration, the fixed-point residual of the context map and
   projected-gradient (stationarity) measures for W and H.  ``--verify`` checks
   that the traced solver reproduces ``fit_court`` to round-off.

Usage
-----
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python scripts/matched_multistart_ablations.py --workers 2
    python scripts/matched_multistart_ablations.py --verify     # equivalence check
    python scripts/matched_multistart_ablations.py --variants gate010,gate060,gate090   # gate sweep

Outputs (``outputs/matched_ablation/``)
    runs.csv                one row per (variant, seed)
    trace_<variant>_<seed>.csv.gz   per-iteration block trace
    W_<variant>_<seed>.npy  memberships (float32) for later re-analysis
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "matched_ablation"
SEEDS = list(range(42, 52))
VARIANTS = {
    "full": dict(context=True, temporal=True, separation=True, opportunity=True),
    "no_context": dict(context=False, temporal=True, separation=True, opportunity=True),
    "no_trajectory": dict(context=True, temporal=False, separation=True, opportunity=True),
    "no_separation": dict(context=True, temporal=True, separation=False, opportunity=True),
    "no_opportunity": dict(context=True, temporal=True, separation=True, opportunity=False),
    "dynamic": dict(context=False, temporal=True, separation=False, opportunity=False),
    # transfer-gate sensitivity, refitted from the same ten seeds (gate 0.35 is "full")
    "gate010": dict(context=True, temporal=True, separation=True, opportunity=True, transfer_gate=0.10),
    "gate060": dict(context=True, temporal=True, separation=True, opportunity=True, transfer_gate=0.60),
    "gate090": dict(context=True, temporal=True, separation=True, opportunity=True, transfer_gate=0.90),
}

_spec = importlib.util.spec_from_file_location("cne", Path(__file__).with_name("court_nmf_experiment.py"))
cne = importlib.util.module_from_spec(_spec)
sys.modules["cne"] = cne
_spec.loader.exec_module(cne)
_mspec = importlib.util.spec_from_file_location("mcc", Path(__file__).with_name("monte_carlo_convergence.py"))
mcc = importlib.util.module_from_spec(_mspec)
sys.modules["mcc"] = mcc
_mspec.loader.exec_module(mcc)

K = cne.K
LAM_T, LAM_D, LAM_S = 0.35, 0.12, 0.008
CLIP = 0.30


def fit_traced(df, X, *, context=True, temporal=True, separation=True, opportunity=True,
               iterations=4000, transfer_gate=0.35, seed=42, edges=None, tol=1e-4, patience=3):
    """``cne.fit_court`` with step_rule='backtracking', plus a block-wise trace."""
    rng = np.random.default_rng(seed)
    n, f = X.shape
    group = pd.factorize(df["conf"].astype(str) + "_" + df["year"].astype(str))[0]
    gcount = group.max() + 1
    q = (np.sqrt(np.clip(df.Min_per.to_numpy(float) / 100, .05, 1)
                 * np.clip(df.GP.to_numpy(float) / 35, .25, 1)) if opportunity else np.ones(n))
    W = cne.simplex(rng.random((n, K)))
    H = rng.random((K, f)) + .05
    B = np.zeros((gcount, f))
    li, ri, ew = cne.career_edges(df, transfer_gate=transfer_gate) if edges is None else edges
    group_op = sp.csr_matrix((np.ones(n), (group, np.arange(n))), shape=(gcount, n))
    denominator = np.maximum(group_op @ q, 1e-12)[:, None]
    if temporal and len(li):
        adj = sp.coo_matrix((np.concatenate([ew, ew]),
                             (np.concatenate([li, ri]), np.concatenate([ri, li]))),
                            shape=(n, n)).tocsr()
        deg = np.asarray(adj.sum(axis=1)).ravel()[:, None]
        lap = sp.diags(deg.ravel()) - adj
    else:
        lap = None

    def objective_at(Wa, Ha, Ba):
        r = Wa @ Ha + Ba[group] - X
        val = float((q * np.square(r).sum(1)).sum())
        if temporal and len(li):
            val += LAM_T * float((ew * np.square(Wa[li] - Wa[ri]).sum(1)).sum())
        if separation:
            gm = Ha @ Ha.T
            np.fill_diagonal(gm, 0)
            val += LAM_D * float(np.square(gm).sum())
        return val + LAM_S * float(np.abs(Wa).sum())

    def context_map(Wa, Ha):
        """The implemented B update: weighted residual mean, centred, clipped."""
        Bn = (group_op @ (q[:, None] * (X - Wa @ Ha))) / denominator
        Bn -= np.average(Bn, axis=0)
        return np.clip(Bn, -CLIP, CLIP)

    def block_minimiser(Wa, Ha):
        """Exact minimiser of Eq. 1 over B in the box (no centring): clip(weighted mean)."""
        Bn = (group_op @ (q[:, None] * (X - Wa @ Ha))) / denominator
        return np.clip(Bn, -CLIP, CLIP)

    def grads(Wa, Ha, Ba):
        res = Wa @ Ha + Ba[group] - X
        gw = 2 * (q[:, None] * res) @ Ha.T + LAM_S
        if lap is not None:
            gw += 2 * LAM_T * (lap @ Wa)
        gh = 2 * Wa.T @ (q[:, None] * res)
        if separation:
            gram = Ha @ Ha.T
            np.fill_diagonal(gram, 0)
            gh += 4 * LAM_D * gram @ Ha
        return gw, gh

    def stationarity(Wa, Ha, Ba):
        gw, gh = grads(Wa, Ha, Ba)
        # gradient mapping on the row simplex at the solver's own scale (step 1/f)
        alpha = 1.0 / f
        gmap_w = (Wa - cne.simplex(Wa - alpha * gw)) / alpha
        # projected gradient for H >= floor
        at_floor = Ha <= 1e-8 * (1 + 1e-9)
        pg_h = np.where(at_floor, np.minimum(gh, 0.0), gh)
        return float(np.linalg.norm(gmap_w)), float(np.linalg.norm(pg_h))

    step_w, step_h = 0.08, 0.04
    obj_cur = objective_at(W, H, B)
    obj_start = obj_cur
    history, trace, checks = [], [], []
    converged_at = None
    sw0, sh0 = stationarity(W, H, B)
    for it in range(iterations):
        W_prev, H_prev, B_prev = W, H, B
        obj_iter_start = obj_cur
        WH = W @ H
        d_b = 0.0
        if context:
            B = (group_op @ (q[:, None] * (X - WH))) / denominator
            B -= np.average(B, axis=0)
            B = np.clip(B, -CLIP, CLIP)
            obj_cur = objective_at(W, H, B)
            d_b = obj_cur - obj_iter_start
        obj_after_b = obj_cur
        residual = WH + B[group] - X
        grad_w = 2 * (q[:, None] * residual) @ H.T + LAM_S
        if lap is not None:
            grad_w += 2 * LAM_T * (lap @ W)
        w_fail = False
        for _ in range(20):
            W_try = cne.simplex(W - step_w * grad_w / f)
            obj_try = objective_at(W_try, H, B)
            if obj_try <= obj_cur:
                W, obj_cur, step_w = W_try, obj_try, step_w * 1.15
                break
            step_w *= 0.5
        else:
            step_w, w_fail = 0.08, True
        obj_after_w = obj_cur
        residual = W @ H + B[group] - X
        grad_h = 2 * W.T @ (q[:, None] * residual)
        if separation:
            gram = H @ H.T
            np.fill_diagonal(gram, 0)
            grad_h += 4 * LAM_D * gram @ H
        h_fail = False
        for _ in range(20):
            H_try = np.maximum(H - step_h * grad_h / n, 1e-8)
            obj_try = objective_at(W, H_try, B)
            if obj_try <= obj_cur:
                H, obj_cur, step_h = H_try, obj_try, step_h * 1.15
                break
            step_h *= 0.5
        else:
            step_h, h_fail = 0.04, True
        nb = np.linalg.norm(B_prev)
        trace.append({
            "iteration": it,
            "obj_start": obj_iter_start, "obj_after_B": obj_after_b,
            "obj_after_W": obj_after_w, "obj_after_H": obj_cur,
            "dL_B": d_b, "dL_W": obj_after_w - obj_after_b, "dL_H": obj_cur - obj_after_w,
            "rel_dB": float(np.linalg.norm(B - B_prev) / nb) if nb > 0 else (float(np.linalg.norm(B)) if context else 0.0),
            "rel_dW": float(np.linalg.norm(W - W_prev) / np.linalg.norm(W_prev)),
            "rel_dH": float(np.linalg.norm(H - H_prev) / np.linalg.norm(H_prev)),
            "clip_frac": float(np.mean(np.abs(B) >= CLIP - 1e-12)) if context else 0.0,
            "centering_resid": float(np.abs(B.mean(axis=0)).max()) if context else 0.0,
            "w_search_failed": w_fail, "h_search_failed": h_fail,
        })
        if it % 25 == 0 or it == iterations - 1:
            resid_hist = W @ H + B[group] - X
            recon = float(np.average(np.square(resid_hist), weights=q, axis=0).mean())
            smooth = float(np.mean(np.square(W[li] - W[ri]))) if len(li) else np.nan
            obj = float((q * np.square(resid_hist).sum(1)).sum())
            if temporal and len(li):
                obj += LAM_T * float((ew * np.square(W[li] - W[ri]).sum(1)).sum())
            if separation:
                gram_h = H @ H.T
                np.fill_diagonal(gram_h, 0)
                obj += LAM_D * float(np.square(gram_h).sum())
            obj += LAM_S * float(np.abs(W).sum())
            history.append({"iteration": it, "weighted_mse": recon, "trajectory_mse": smooth,
                            "objective": obj})
            sw, sh = stationarity(W, H, B)
            chk = {"iteration": it, "objective": obj, "stat_W_rel": sw / sw0, "stat_H_rel": sh / sh0}
            if context:
                Bt = context_map(W, H)
                chk["B_fixed_point_rel"] = float(np.linalg.norm(Bt - B) / max(np.linalg.norm(B), 1e-30))
            checks.append(chk)
            if tol and len(history) > patience:
                recent = [h["objective"] for h in history[-(patience + 1):]]
                steps = [abs(recent[j + 1] - recent[j]) / max(abs(recent[j]), 1e-30)
                         for j in range(patience)]
                if all(st < tol for st in steps):
                    converged_at = it
                    break
    tr = pd.DataFrame(trace)
    ck = pd.DataFrame(checks)
    final = {}
    if context:
        Bstar = block_minimiser(W, H)
        L_now = objective_at(W, H, B)
        final["B_block_gap_rel"] = (L_now - objective_at(W, H, Bstar)) / L_now
        final["B_fixed_point_rel_final"] = float(np.linalg.norm(context_map(W, H) - B) / np.linalg.norm(B))
        final["clip_frac_final"] = float(np.mean(np.abs(B) >= CLIP - 1e-12))
    final["stat_W_rel_final"], final["stat_H_rel_final"] = [v / s0 for v, s0 in zip(stationarity(W, H, B), (sw0, sh0))]
    final["obj_start"] = obj_start
    return W, H, B, q, {"history": history, "trace": tr, "checks": ck, "final": final,
                        "converged": converged_at is not None,
                        "stopped_at": int(converged_at if converged_at is not None else history[-1]["iteration"])}


# --------------------------------------------------------------------------
_G = {}


def _init(labels_path: str):
    df, X, _ = cne.preprocess()
    _G.update(df=df, X=X, edges=cne.career_edges(df),
              W_nmf=np.load(OUT / "W_nmf_static.npy"))
    labels = pd.read_csv(labels_path)
    labels = labels[labels["position"].isin(mcc.POSITIONS)].copy()
    labels["pid"] = labels["pid"].astype(str).str.strip()
    mcc._WORKER["labels"] = labels
    _G["q_full"] = np.sqrt(np.clip(df.Min_per.to_numpy(float) / 100, .05, 1)
                           * np.clip(df.GP.to_numpy(float) / 35, .25, 1))
    group = pd.factorize(df["conf"].astype(str) + "_" + df["year"].astype(str))[0]
    _G["group"] = group


def _run(task):
    from sklearn.metrics import silhouette_score
    variant, seed = task
    df, X, edges = _G["df"], _G["X"], _G["edges"]
    t0 = time.time()
    gate = VARIANTS[variant].get("transfer_gate", 0.35)
    if gate != 0.35:
        edges = cne.career_edges(df, transfer_gate=gate)
    W, H, B, q, out = fit_traced(df, X, seed=seed, edges=edges, **VARIANTS[variant])
    hist = out["history"]
    li, ri, _ = edges
    resid = W @ H + B[_G["group"]] - X
    labels = W.argmax(1)
    rec = {"variant": variant, "seed": seed,
           "objective": hist[-1]["objective"],
           "weighted_mse_own": hist[-1]["weighted_mse"],
           "weighted_mse_common": float(np.average(np.square(resid), weights=_G["q_full"], axis=0).mean()),
           "unweighted_mse": float(np.square(resid).mean()),
           "trajectory_mse": float(np.mean(np.square(W[li] - W[ri]))),
           "mean_max_membership": float(W.max(1).mean()),
           "mean_entropy": float(cne.entropy(W).mean()),
           "modal_share": float(pd.Series(labels).value_counts(normalize=True).iloc[0]),
           "n_roles_used": int(len(np.unique(labels))),
           "silhouette_nmfspace": float(silhouette_score(_G["W_nmf"], labels, sample_size=10000, random_state=42))
           if len(np.unique(labels)) > 1 else np.nan,
           "silhouette_ownspace": float(silhouette_score(W, labels, sample_size=10000, random_state=42))
           if len(np.unique(labels)) > 1 else np.nan,
           "mixture_r2": mcc.mixture_r2(W, X),
           "stopped_at": out["stopped_at"], "converged": out["converged"]}
    rec.update(mcc._external_metrics(df, W))
    tr = out["trace"]
    rec.update({
        "n_iter_run": int(len(tr)),
        "B_steps_increasing": int((tr["dL_B"] > 1e-12 * tr["obj_start"].abs()).sum()),
        "B_max_rel_increase": float((tr["dL_B"] / tr["obj_start"]).max()),
        "B_sum_increase_rel": float(tr["dL_B"].clip(lower=0).sum() / out["final"]["obj_start"]),
        "iter_increasing": int((tr["obj_after_H"] > tr["obj_start"] * (1 + 1e-12)).sum()),
        "late_B_increases": int((tr["dL_B"].iloc[len(tr) // 2:] > 1e-12 * tr["obj_start"].iloc[len(tr) // 2:].abs()).sum()),
        "rel_dB_final": float(tr["rel_dB"].iloc[-1]), "rel_dW_final": float(tr["rel_dW"].iloc[-1]),
        "rel_dH_final": float(tr["rel_dH"].iloc[-1]),
        "w_search_failures": int(tr["w_search_failed"].sum()),
        "h_search_failures": int(tr["h_search_failed"].sum()),
        "centering_resid_max": float(tr["centering_resid"].max()),
        "wall_seconds": round(time.time() - t0, 1),
    })
    # fixed column set for every variant, so appended CSV rows stay aligned
    rec.update({"B_block_gap_rel": np.nan, "B_fixed_point_rel_final": np.nan, "clip_frac_final": np.nan})
    rec.update(out["final"])
    tr.to_csv(OUT / f"trace_{variant}_{seed}.csv.gz", index=False)
    out["checks"].to_csv(OUT / f"checks_{variant}_{seed}.csv", index=False)
    np.save(OUT / f"W_{variant}_{seed}.npy", W.astype(np.float32))
    np.save(OUT / f"H_{variant}_{seed}.npy", H)
    if VARIANTS[variant]["context"]:
        np.save(OUT / f"B_{variant}_{seed}.npy", B)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--variants", default="full,no_context,no_trajectory,no_separation,no_opportunity,dynamic")
    a = ap.parse_args()  # transfer-gate sweep: --variants gate010,gate060,gate090
    OUT.mkdir(parents=True, exist_ok=True)
    labels_path = str(BASE / "outputs" / "expanded_validation" / "expanded_validation_labeled_sample.csv")
    if a.verify:
        df, X, _ = cne.preprocess()
        for v in ("full", "no_context"):
            W1, H1, B1, o1 = cne.fit_court(df, X, iterations=150, tol=0, seed=42, **VARIANTS[v])
            W2, H2, B2, _, o2 = fit_traced(df, X, iterations=150, tol=0, seed=42, **VARIANTS[v])
            print(v, "max|dW|", np.abs(W1 - W2).max(), "max|dH|", np.abs(H1 - H2).max(),
                  "max|dB|", np.abs(B1 - B2).max(),
                  "obj", o1["history"][-1]["objective"], o2["history"][-1]["objective"])
        return
    if not (OUT / "W_nmf_static.npy").exists():
        from sklearn.decomposition import NMF
        df, X, _ = cne.preprocess()
        W_nmf = NMF(n_components=K, init="nndsvd", random_state=42, max_iter=500).fit_transform(X)
        np.save(OUT / "W_nmf_static.npy", W_nmf)
    runs_path = OUT / "runs.csv"
    done = set()
    if runs_path.exists():
        prev = pd.read_csv(runs_path)
        done = set(zip(prev.variant, prev.seed))
    tasks = [(v, s) for v in a.variants.split(",") for s in SEEDS if (v, s) not in done]
    print(f"{len(tasks)} fits to run", flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(labels_path,)) as pool:
        for rec in pool.imap_unordered(_run, tasks):
            pd.DataFrame([rec]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
            print(f"{rec['variant']:>15} seed {rec['seed']}: obj {rec['objective']:.2f} stop {rec['stopped_at']} "
                  f"traj {rec['trajectory_mse']:.5f} wmse {rec['weighted_mse_own']:.5f} "
                  f"acc {rec['cv_accuracy']:.3f} B+ {rec['B_steps_increasing']} {rec['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
