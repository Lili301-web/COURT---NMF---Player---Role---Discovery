"""Block-wise convergence trace of the selected COURT-NMF fit (revision R3, Fig. 4).

Reads ``outputs/matched_ablation/trace_<variant>_<seed>.csv.gz`` and
``checks_<variant>_<seed>.csv`` written by ``matched_multistart_ablations.py``
and draws three single-axis panels (vector PDF):

  (a) per-iteration objective change, split into the W+H descent and the
      context (B) step whenever that step increases the objective;
  (b) relative change of each block per iteration;
  (c) stationarity at the 25-iteration checks: fixed-point residual of the
      context map, gradient-mapping norm for W, projected-gradient norm for H.

Usage: python scripts/make_r3_convergence_figure.py --variant full --seed <selected>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "matched_ablation"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_W, C_H, C_B = "#2a78d6", "#eb6834", "#1baf7a"   # validated slots 1-3, fixed identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=str(BASE / "figures" / "fig26_block_convergence.pdf"))
    a = ap.parse_args()
    f_tr, f_ck = OUT / f"trace_{a.variant}_{a.seed}.csv.gz", OUT / f"checks_{a.variant}_{a.seed}.csv"
    if f_tr.exists():
        tr, ck = pd.read_csv(f_tr), pd.read_csv(f_ck)
    else:  # bundled outputs: one file for all runs
        sel = lambda d: d[(d.variant == a.variant) & (d.seed == a.seed)].drop(columns=["variant", "seed"]).reset_index(drop=True)
        tr = sel(pd.read_csv(OUT / "block_traces_all_runs.csv.gz"))
        ck = sel(pd.read_csv(OUT / "stationarity_checks_all_runs.csv"))
    L_final = tr["obj_after_H"].iloc[-1]

    # Springer SVProc instructions: sans-serif lettering of 8-10 pt at final size;
    # the figure is drawn at its printed width (122 mm = \textwidth).
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans",
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.3), sharex=True)
    it = tr["iteration"].to_numpy()

    ax = axes[0]
    wh = -(tr["dL_W"] + tr["dL_H"]).to_numpy() / L_final
    ax.plot(it, np.maximum(wh, 1e-12), color=INK2, lw=0.9, label="$W$+$H$ steps, decrease")
    inc = tr["dL_B"].to_numpy() / L_final
    m = inc > 0
    ax.scatter(it[m], inc[m], s=3, color=C_B, lw=0, label="$B$ step, increase", zorder=3)
    ax.set_yscale("log")
    ax.set_title(r"$\mathbf{a}$  Objective change", loc="left")

    ax = axes[1]
    ax.plot(it, tr["rel_dW"], color=C_W, lw=0.9, label="$W$")
    ax.plot(it, tr["rel_dH"], color=C_H, lw=0.9, ls="--", label="$H$")
    ax.plot(it[1:], tr["rel_dB"].iloc[1:], color=C_B, lw=0.9, ls=":", label="$B$")
    ax.set_yscale("log")
    ax.set_title(r"$\mathbf{b}$  Block change", loc="left")

    ax = axes[2]
    ax.plot(ck["iteration"], ck["stat_W_rel"], color=C_W, lw=0.9, marker="o", ms=1.8, label="$W$ grad. map")
    ax.plot(ck["iteration"], ck["stat_H_rel"], color=C_H, lw=0.9, ls="--", marker="s", ms=1.8, label="$H$ proj. grad.")
    ax.plot(ck["iteration"], ck["B_fixed_point_rel"], color=C_B, lw=0.9, ls=":", marker="^", ms=2,
            label="$B$ fixed point")
    ax.set_yscale("log")
    ax.set_title(r"$\mathbf{c}$  Stationarity", loc="left")

    for ax in axes:
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Iteration")
        ax.set_xlim(0, it.max())
    # one shared legend: colour = block identity in every panel (B is always aqua)
    handles = [plt.Line2D([], [], color=C_W, lw=1.0, label="$W$"),
               plt.Line2D([], [], color=C_H, lw=1.0, ls="--", label="$H$"),
               plt.Line2D([], [], color=C_B, lw=1.0, ls=":", label="$B$ (context offsets)"),
               plt.Line2D([], [], color=INK2, lw=1.0, label="$W$+$H$ descent")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.91))
    fig.savefig(a.out)
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
