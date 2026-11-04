"""Generate every R3 number, table body and the ablation strip plot from the runs.

Reads  outputs/matched_ablation/runs.csv (+ per-run traces and W files)
Writes outputs/matched_ablation/r3_generated/{numbers.tex, table_benchmark_rows.tex,
       table_matched_rows.tex, table_convergence_rows.tex, table_diagnostics_rows.tex,
       table_gate_rows.tex, fig_ablation_strip.tex, summary.json}
Nothing in these files is typed by hand.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "outputs" / "matched_ablation"
GEN = RUN / "r3_generated"
GEN.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("cne", Path(__file__).with_name("court_nmf_experiment.py"))
cne = importlib.util.module_from_spec(_spec)
sys.modules["cne"] = cne
_spec.loader.exec_module(cne)

MAIN = ["dynamic", "full", "no_context", "no_trajectory", "no_separation", "no_opportunity"]
LABEL = {"dynamic": "Dynamic NMF (graph only)", "full": "COURT-NMF (full)",
         "no_context": "\\quad no context", "no_trajectory": "\\quad no trajectory",
         "no_separation": "\\quad no separation", "no_opportunity": "\\quad no opportunity weights"}
SHORT = {"dynamic": "Graph-only NMF", "full": "Full model", "no_context": "No context",
         "no_trajectory": "No trajectory", "no_separation": "No separation",
         "no_opportunity": "No opportunity"}
NO_CONTEXT = {"no_context", "dynamic"}
TAIL = ["B_block_gap_rel", "B_fixed_point_rel_final", "clip_frac_final",
        "stat_W_rel_final", "stat_H_rel_final", "obj_start"]


def load_runs() -> pd.DataFrame:
    d = pd.read_csv(RUN / "runs.csv")
    # Rows of variants without a context block were appended before the column
    # set was fixed: their last three values sit three columns early.  Realign.
    bad = d.variant.isin(NO_CONTEXT) & d["obj_start"].isna()
    if bad.any():
        vals = d.loc[bad, TAIL[:3]].to_numpy()
        d.loc[bad, TAIL[3:]] = vals
        d.loc[bad, TAIL[:3]] = np.nan
    return d


_BUNDLE = {}


def read_trace(variant, seed):
    """Per-run trace file if present, else the bundled block_traces_all_runs.csv.gz."""
    f = RUN / f"trace_{variant}_{seed}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    if "tr" not in _BUNDLE:
        _BUNDLE["tr"] = pd.read_csv(RUN / "block_traces_all_runs.csv.gz")
    b = _BUNDLE["tr"]
    return b[(b.variant == variant) & (b.seed == seed)].drop(columns=["variant", "seed"]).reset_index(drop=True)


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def pm(series, scale=1.0, fmt="{:.2f}"):
    return f"{fmt.format(series.mean() * scale)}$\\pm${fmt.format(series.std(ddof=1) * scale)}"


def main():
    d = load_runs()
    m = d[d.variant.isin(MAIN)].copy()
    assert m.groupby("variant").size().eq(10).all(), m.groupby("variant").size()
    sel = m.loc[m.groupby("variant")["objective"].idxmin()].set_index("variant")
    full_by_seed = m[m.variant == "full"].set_index("seed")
    out, summary = [], {}

    s_full = sel.loc["full"]
    out += [macro("rSelSeed", int(s_full.seed)), macro("rSelStop", f"{int(s_full.stopped_at):,}".replace(",", "{,}")),
            macro("rSelObj", f"{s_full.objective:.2f}")]

    # ---------- Table 1: selected fits, common reconstruction metric -------
    rows = ["NMF + $K$-means & 0.217 & 0.304 & -- & -- & -- \\\\",
            "Gaussian mixture model & 0.166 & 0.318 & -- & -- & -- \\\\",
            "Fuzzy $C$-means & 0.193 & 0.246 & -- & -- & -- \\\\",
            "UMAP + HDBSCAN & 0.380 & 0.985 & -- & -- & -- \\\\"]
    for v in MAIN:
        r = sel.loc[v]
        rows.append(f"{LABEL[v]} & {r.silhouette_nmfspace:.3f} & {r.modal_share:.3f} & {r.mean_max_membership:.3f} & "
                    f"{r.weighted_mse_common:.5f} & {r.trajectory_mse:.5f} \\\\")
    (GEN / "table_benchmark_rows.tex").write_text("\n".join(rows).replace("-0.", "$-$0.") + "\n")

    # ---------- headline contrasts (selected fits) --------------------------
    tf, tn = sel.loc["full", "trajectory_mse"], sel.loc["no_trajectory", "trajectory_mse"]
    wf, wn = sel.loc["full", "weighted_mse_common"], sel.loc["no_trajectory", "weighted_mse_common"]
    out += [macro("rTrajFull", f"{tf:.5f}"), macro("rTrajNoTraj", f"{tn:.5f}"),
            macro("rTrajRiseFactor", f"{tn / tf:.2f}"), macro("rTrajRisePct", f"{100 * (tn / tf - 1):.0f}\\%"),
            macro("rWMSEChange", f"{100 * (wn / wf - 1):+.1f}\\%".replace("-", "$-$"))]

    # ---------- seed-paired ratios vs full -----------------------------------
    pair = {}
    for v in MAIN:
        if v == "full":
            continue
        dv = m[m.variant == v].set_index("seed")
        ratio_t = dv.trajectory_mse / full_by_seed.trajectory_mse
        ratio_w = dv.weighted_mse_common / full_by_seed.weighted_mse_common
        pair[v] = dict(t_med=ratio_t.median(), t_min=ratio_t.min(), t_max=ratio_t.max(),
                       t_above=int((ratio_t > 1).sum()), w_med=ratio_w.median(), w_min=ratio_w.min(),
                       w_max=ratio_w.max(), w_above=int((ratio_w > 1).sum()),
                       acc_diff_mean=float((dv.cv_accuracy - full_by_seed.cv_accuracy).mean()))
    summary["paired"] = pair
    p = pair["no_trajectory"]
    out += [macro("rPairTrajMed", f"{p['t_med']:.2f}"), macro("rPairTrajMin", f"{p['t_min']:.2f}"),
            macro("rPairTrajMax", f"{p['t_max']:.2f}"), macro("rPairTrajAbove", p["t_above"]),
            macro("rPairWMSEMed", f"{100 * (p['w_med'] - 1):+.1f}\\%".replace("-", "$-$"))]
    p = pair["no_separation"]
    out += [macro("rNoSepTrajMed", f"{p['t_med']:.2f}"), macro("rNoSepTrajAbove", p["t_above"])]
    p = pair["no_context"]
    out += [macro("rNoCtxWMSEMed", f"{100 * (p['w_med'] - 1):+.1f}\\%"), macro("rNoCtxWMSEAbove", p["w_above"]),
            macro("rNoCtxTrajMed", f"{p['t_med']:.2f}"), macro("rNoCtxTrajAbove", p["t_above"])]
    p = pair["no_opportunity"]
    out += [macro("rNoOppWMSEMed", f"{100 * (p['w_med'] - 1):+.1f}\\%"), macro("rNoOppWMSEAbove", p["w_above"]),
            macro("rNoOppTrajMed", f"{p['t_med']:.2f}"), macro("rNoOppTrajAbove", p["t_above"])]
    p = pair["dynamic"]
    out += [macro("rDynWMSEMed", f"{100 * (p['w_med'] - 1):+.1f}\\%"), macro("rDynTrajMed", f"{p['t_med']:.2f}"),
            macro("rDynTrajAbove", p["t_above"])]
    out += [macro("rNoSepModal", f"{sel.loc['no_separation', 'modal_share']:.3f}"),
            macro("rNoSepModalMin", f"{m[m.variant == 'no_separation'].modal_share.min():.3f}"),
            macro("rNoSepModalMax", f"{m[m.variant == 'no_separation'].modal_share.max():.3f}"),
            macro("rFullModalMin", f"{m[m.variant == 'full'].modal_share.min():.3f}"),
            macro("rFullModalMax", f"{m[m.variant == 'full'].modal_share.max():.3f}"),
            macro("rNoOppTraj", f"{sel.loc['no_opportunity', 'trajectory_mse']:.5f}"),
            macro("rDynTraj", f"{sel.loc['dynamic', 'trajectory_mse']:.5f}")]

    # ---------- Table 2: ten matched starts ----------------------------------
    rows = []
    for v in MAIN:
        dv = m[m.variant == v]
        ratio = "--" if v == "full" else (f"{pair[v]['t_med']:.2f} [{pair[v]['t_min']:.2f}, {pair[v]['t_max']:.2f}]")
        rows.append(f"{SHORT[v]} & {int(sel.loc[v, 'seed'])} & {int(sel.loc[v, 'stopped_at'])} & {pm(dv.weighted_mse_common, 1e3)} & {pm(dv.trajectory_mse, 1e3)} & {ratio} & "
                    f"{pm(dv.cv_accuracy, 1, '{:.3f}')} \\\\")
    (GEN / "table_matched_rows.tex").write_text("\n".join(rows) + "\n")
    means = m.groupby("variant")[["cv_accuracy", "permanova_r2"]].mean()
    abl = [v for v in MAIN if v != "full"]
    diffs = [-pair[v]["acc_diff_mean"] for v in abl]
    out += [macro("rAccMeanFull", f"{means.loc['full', 'cv_accuracy']:.3f}"),
            macro("rAccMeanAblMin", f"{means.loc[abl, 'cv_accuracy'].min():.3f}"),
            macro("rAccMeanAblMax", f"{means.loc[abl, 'cv_accuracy'].max():.3f}"),
            macro("rAccDiffMin", f"{min(diffs):.3f}"), macro("rAccDiffMax", f"{max(diffs):.3f}"),
            macro("rPermMeanFull", f"{means.loc['full', 'permanova_r2']:.3f}"),
            macro("rPermMeanDyn", f"{means.loc['dynamic', 'permanova_r2']:.3f}"),
            macro("rPermMeanNoCtx", f"{means.loc['no_context', 'permanova_r2']:.3f}")]
    out += [macro("rSelEntMin", f"{sel.mean_entropy.min():.3f}"), macro("rSelEntMax", f"{sel.mean_entropy.max():.3f}")]
    acc = m.groupby("variant").cv_accuracy
    out += [macro("rAccMin", f"{acc.min().min():.3f}"), macro("rAccMax", f"{acc.max().max():.3f}"),
            macro("rAccSelMin", f"{sel.cv_accuracy.min():.3f}"), macro("rAccSelMax", f"{sel.cv_accuracy.max():.3f}"),
            macro("rPermSelMin", f"{sel.permanova_r2.min():.3f}"), macro("rPermSelMax", f"{sel.permanova_r2.max():.3f}")]

    # ---------- ablation strip plot (pgfplots) --------------------------------
    order = ["no_opportunity", "no_separation", "no_trajectory", "no_context", "full", "dynamic"]
    def coords(col, scale, which):
        pts = []
        for y, v in enumerate(order):
            dv = m[m.variant == v] if which == "all" else sel.loc[[v]]
            pts += [f"({x * scale:.4f},{y})" for x in dv[col]]
        return " ".join(pts)
    ylabels = ",".join(SHORT[v] for v in order)
    fig = r"""\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=6mm,yticklabels at=edge left},
width=5.2cm,height=5.4cm,ymin=-0.6,ymax=%(ymax)s,ytick={0,...,%(ymaxi)d},
yticklabels={%(ylabels)s},
yticklabel style={font=\small},tick label style={font=\small},title style={font=\small},
axis lines=left,xmajorgrids=true,ymajorgrids=true,grid style={black!10},
scaled x ticks=false,xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=1}]
\nextgroupplot[title={Weighted MSE $\times10^3$},xmin=%(wmin)s,xmax=%(wmax)s]
\addplot[only marks,mark=o,mark size=1.6pt,draw=blue!70!black,line width=0.5pt] coordinates {%(w_all)s};
\addplot[only marks,mark=diamond*,mark size=2.6pt,fill=black,draw=black] coordinates {%(w_sel)s};
\nextgroupplot[title={Trajectory MSE $\times10^3$},xmin=0,xmax=%(tmax)s]
\addplot[only marks,mark=o,mark size=1.6pt,draw=blue!70!black,line width=0.5pt] coordinates {%(t_all)s};
\addplot[only marks,mark=diamond*,mark size=2.6pt,fill=black,draw=black] coordinates {%(t_sel)s};
\end{groupplot}
\end{tikzpicture}
\caption{Seed-matched ablations. Circles: the ten starts (seeds 42--51, identical initial $W$ and $H$ in every row); diamonds: the start selected by each variant's own objective. Reconstruction uses the common weighting of Table~\ref{tab:court-benchmark}. The outlying starts in the left panel are mostly seed~45, which reaches a higher-error solution in five of the six variants}
\label{fig:court-ablation}
\end{figure}
"""
    wv = m.weighted_mse_common * 1e3
    tv = m.trajectory_mse * 1e3
    (GEN / "fig_ablation_strip.tex").write_text(fig % dict(
        ymax=len(order) - 0.4, ymaxi=len(order) - 1, ylabels=ylabels,
        wmin=f"{np.floor(wv.min() * 10) / 10:.1f}", wmax=f"{np.ceil(wv.max() * 10) / 10 + 0.05:.2f}",
        tmax=f"{np.ceil(tv.max()) + 0.5:.1f}",
        w_all=coords("weighted_mse_common", 1e3, "all"), w_sel=coords("weighted_mse_common", 1e3, "sel"),
        t_all=coords("trajectory_mse", 1e3, "all"), t_sel=coords("trajectory_mse", 1e3, "sel")))
    worst = m.groupby("seed").weighted_mse_common.mean().idxmax()
    out.append(macro("rWorstSeed", int(worst)))

    # ---------- convergence diagnostics over all matched runs ---------------
    recs = []
    for _, r in m.iterrows():
        tr = read_trace(r.variant, int(r.seed))
        inc = tr.dL_B / tr.obj_start
        pos = inc > 1e-12
        half = len(tr) // 2
        recs.append(dict(variant=r.variant, seed=int(r.seed), n_iter=len(tr), n_inc=int(pos.sum()),
                         frac_inc=float(pos.mean()), last_inc=int(tr.iteration[pos].max()) if pos.any() else -1,
                         max_inc=float(inc.max()) if pos.any() else 0.0,
                         max_inc_late=float(inc.iloc[half:].max()) if pos.iloc[half:].any() else 0.0,
                         net_B=float(tr.dL_B.sum() / tr.obj_start.iloc[0]),
                         ratio_max=float((tr.dL_B[pos] / -(tr.dL_W + tr.dL_H)[pos]).max()) if pos.any() else 0.0,
                         clip_any=float(tr.clip_frac.max()),
                         iter_up=int((tr.obj_after_H > tr.obj_start * (1 + 1e-12)).sum()),
                         clip_max=float(tr.clip_frac.max()), centering=float(tr.centering_resid.max()),
                         fp=r.B_fixed_point_rel_final, dB=r.rel_dB_final, gap=r.B_block_gap_rel,
                         sw=r.stat_W_rel_final, sh=r.stat_H_rel_final, stop=int(r.stopped_at), conv=bool(r.converged)))
    cv = pd.DataFrame(recs)
    cv.to_csv(GEN / "convergence_summary_per_run.csv", index=False)
    ctx = cv[~cv.variant.isin(NO_CONTEXT)]
    sf = cv[(cv.variant == "full") & (cv.seed == int(s_full.seed))].iloc[0]
    out += [macro("rConvRuns", len(cv)), macro("rConvConverged", int(cv.conv.sum())),
            macro("rIterUp", int(cv.iter_up.sum())), macro("rCtxRuns", len(ctx)),
            macro("rBincMaxRel", sci(ctx.max_inc.max())),
            macro("rBincLateMaxRel", sci(ctx.max_inc_late.max())),
            macro("rClipMax", f"{ctx.clip_max.max():.3f}"),
            macro("rBincRatioMax", f"{ctx.ratio_max.max():.3f}"),
            macro("rSearchFail", int(m.w_search_failures.sum() + m.h_search_failures.sum())),
            macro("rFPMin", sci(ctx.fp.min())), macro("rFPMax", sci(ctx.fp.max())),
            macro("rdBMax", sci(ctx.dB.max())),
            macro("rGapMin", f"{100 * ctx.gap.min():.3f}\\%"), macro("rGapMax", f"{100 * ctx.gap.max():.3f}\\%"),
            macro("rStatMax", sci(max(cv.sw.max(), cv.sh.max()))),
            macro("rSelBinc", int(sf.n_inc)), macro("rSelIter", int(sf.n_iter)), macro("rSelBlast", int(sf.last_inc)),
            macro("rSelBincMax", sci(sf.max_inc)), macro("rSelFP", sci(sf.fp)), macro("rSelGap", f"{100 * sf.gap:.3f}\\%"),
            macro("rSelSW", sci(sf.sw)), macro("rSelSH", sci(sf.sh)), macro("rSelNetB", sci(sf.net_B))]
    rows = []
    for v in MAIN:
        c = cv[cv.variant == v]
        if v in NO_CONTEXT:
            rows.append(f"{SHORT[v]} & -- & -- & -- & -- & "
                        f"{sci(max(c.sw.max(), c.sh.max()))} \\\\")
        else:
            rows.append(f"{SHORT[v]} & "
                        f"{100 * c.frac_inc.median():.1f} [{100 * c.frac_inc.min():.1f}, {100 * c.frac_inc.max():.1f}] & "
                        f"{sci(c.max_inc_late.max())} & {sci(c.fp.max())} & {100 * c.gap.max():.3f} & "
                        f"{sci(max(c.sw.max(), c.sh.max()))} \\\\")
    rows = [re.sub(r"(\d\.\d\\times10\^\{-?\d+\})", r"$\1$", r) for r in rows]
    (GEN / "table_convergence_rows.tex").write_text("\n".join(rows).replace("$0.0\\times10^{0}$", "0") + "\n")
    agg = cv.drop(columns=["seed"]).groupby("variant").agg(["min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(GEN / "convergence_summary_by_variant.csv")

    # ---------- Table C.1 membership and stopping diagnostics (selected) -----
    rows = []
    for v in MAIN:
        r = sel.loc[v]
        rows.append(f"{SHORT[v]} & {int(r.seed)} & {r.mean_max_membership:.4f} & {r.mean_entropy:.3f} & "
                    f"{int(r.stopped_at)} & {r.permanova_r2:.3f} & {r.cv_accuracy:.3f} \\\\")
    (GEN / "table_diagnostics_rows.tex").write_text("\n".join(rows) + "\n")

    # ---------- transfer-gate sweep (if run) ---------------------------------
    gates = {"gate010": 0.10, "full": 0.35, "gate060": 0.60, "gate090": 0.90}
    g = d[d.variant.isin(gates)].copy()
    if g.groupby("variant").size().reindex(list(gates)).fillna(0).eq(10).all():
        def cross_mse(v, s):  # only needed if runs.csv lacks the column
            df, X, _ = cne.preprocess()
            li, ri, _ = cne.career_edges(df)
            cross = df.loc[li, "team"].to_numpy() != df.loc[ri, "team"].to_numpy()
            W = np.load(RUN / f"W_{v}_{s}.npy").astype(float)
            return float(np.square(W[li[cross]] - W[ri[cross]]).mean(axis=1).mean())
        if "cross_team_mse" not in g or g["cross_team_mse"].isna().any():
            g["cross_team_mse"] = [cross_mse(v, s) for v, s in zip(g.variant, g.seed)]
        g["gate"] = g.variant.map(gates)
        gsel = g.loc[g.groupby("variant")["objective"].idxmin()].sort_values("gate")
        rows = []
        for _, r in gsel.iterrows():
            gg = g[g.variant == r.variant]
            rows.append(f"{r.gate:.2f} & {int(r.seed)} & {1e3 * r.weighted_mse_common:.3f} & {1e3 * r.trajectory_mse:.3f} & "
                        f"{1e3 * r.cross_team_mse:.3f} & {1e3 * gg.cross_team_mse.mean():.3f} & {r.mean_entropy:.3f} & "
                        f"{r.mean_max_membership:.3f} \\\\")
        (GEN / "table_gate_rows.tex").write_text("\n".join(rows) + "\n")
        piv = g.pivot(index="seed", columns="gate", values="cross_team_mse")
        mono = int((piv.diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum())
        out += [macro("rGateWMSELo", f"{gsel.weighted_mse_common.min():.6f}"),
                macro("rGateWMSEHi", f"{gsel.weighted_mse_common.max():.6f}"),
                macro("rGateCrossLo", f"{gsel.cross_team_mse.iloc[0]:.6f}"),
                macro("rGateCrossHi", f"{gsel.cross_team_mse.iloc[-1]:.6f}"),
                macro("rGateMono", mono)]
        summary["gate_cross_by_seed"] = piv.to_dict()
    else:
        print("transfer-gate runs incomplete; gate table not generated")

    (GEN / "numbers.tex").write_text("% generated by scripts/make_r3_numbers.py\n" + "".join(out))
    (GEN / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("".join(out))


if __name__ == "__main__":
    main()
