"""Ablation diagnostics figure for the COURT-NMF paper (main text, Fig. 2).

Reads the numbers straight out of ``outputs/court_nmf/court_nmf_diagnostics.json``
so the figure can never drift from the run that produced it.  Re-run this after
any re-fit; nothing here is hard-coded.

Why only four variants appear: Section E.2 of the supplement notes that
reconstruction error is comparable only across variants that share the
opportunity weighting.  ``Dynamic NMF`` and ``COURT-NMF w/o opportunity`` change
the weighting, so their weighted MSE is on a different scale and plotting it
beside the others would invite a comparison the paper explicitly rules out.
They stay in the table; they are not in the figure.

Usage
-----
    python make_ablation_figure.py                      # default paths
    python make_ablation_figure.py --json PATH --out PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DEFAULT_JSON = BASE / "outputs" / "court_nmf" / "court_nmf_diagnostics.json"
DEFAULT_OUT = BASE / "figures" / "fig21_ablation_diagnostics.png"

# Reference variant first; the rest are ablations of it.  Every entry here
# retains opportunity weighting, so both measures are comparable across them.
REFERENCE = "COURT-NMF"
ABLATIONS = ["COURT-NMF w/o context",
             "COURT-NMF w/o trajectory",
             "COURT-NMF w/o separation"]

SHORT = {"COURT-NMF": "COURT-NMF\n(full)",
         "COURT-NMF w/o context": "Without\ncontext",
         "COURT-NMF w/o trajectory": "Without\ntrajectory",
         "COURT-NMF w/o separation": "Without\nseparation"}

PANELS = [("weighted_mse", "Weighted reconstruction MSE"),
          ("trajectory_mse", "Adjacent-season trajectory MSE")]

REF_COLOR = "#104281"   # blue 650 - the full model, the reference for every delta
BAR_COLOR = "#2a78d6"   # blue 450 - ablations
GRID = "#d8d8d4"
INK = "#0b0b0b"
INK2 = "#52514e"


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        diag = json.load(fh)
    missing = [k for k in [REFERENCE] + ABLATIONS if k not in diag]
    if missing:
        raise SystemExit(f"{path} is missing required variants: {missing}\n"
                         f"present: {sorted(diag)}")
    return diag


def build(diag: dict, out: Path) -> None:
    order = [REFERENCE] + ABLATIONS
    labels = [SHORT.get(k, k) for k in order]
    iters = {diag[k].get("iterations") for k in order}
    iter_note = (f"fixed-seed run, {iters.pop():,} iterations"
                 if len(iters) == 1 else "fixed-seed run")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.9, 1.95), dpi=400)
    for ax, (key, title) in zip(axes, PANELS):
        vals = [diag[k][key] for k in order]
        ref = vals[0]
        # zero baseline, with headroom for the value label and the delta
        span = max(vals) * 1.65
        colors = [REF_COLOR] + [BAR_COLOR] * (len(vals) - 1)
        y = range(len(vals))
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        ax.axvline(ref, color=REF_COLOR, lw=0.7, ls=(0, (3, 2)), zorder=4)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=6.2, linespacing=0.95)
        ax.invert_yaxis()
        ax.set_xlim(0, span)
        ax.set_title(title, fontsize=7, pad=4, color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6, length=2, width=0.5)

        for i, v in enumerate(vals):
            ax.text(v + span * 0.015, i, f"{v:.5f}", va="center", ha="left",
                    fontsize=6, color=INK2, zorder=5)
        # label the ablation that moves this measure most, as a signed percentage
        deltas = [(abs(v / ref - 1), i) for i, v in enumerate(vals)]
        _, worst = max(deltas)
        if worst != 0:
            pct = (vals[worst] / ref - 1) * 100
            ax.text(vals[worst] + span * 0.26, worst, f"{pct:+.1f} %",
                    va="center", ha="left", fontsize=6.5,
                    color=REF_COLOR, fontweight="bold", zorder=5)

    fig.tight_layout(pad=0.35, w_pad=1.6)
    # placed after tight_layout so it centres on the final canvas
    fig.text(0.5, -0.03, iter_note, ha="center", va="top",
             fontsize=5.6, color=INK2, transform=fig.transFigure)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")

    # echo the numbers so they can be checked against the LaTeX table
    print(f"wrote {out}  ({iter_note})")
    head = f"{'variant':<28}" + "".join(f"{t:>26}" for _, t in PANELS)
    print(head)
    for k in order:
        row = f"{k:<28}"
        for key, _ in PANELS:
            v = diag[k][key]
            d = "" if k == REFERENCE else f"  ({v / diag[REFERENCE][key] - 1:+.1%})"
            row += f"{v:>16.5f}{d:>10}"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    build(load(a.json), a.out)
