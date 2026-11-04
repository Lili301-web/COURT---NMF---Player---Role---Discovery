"""Solver convergence figure for the COURT-NMF supplement.

Plots the two objective diagnostics against iteration count so a reader can see
where the projected-gradient solver settles, instead of having to take the
iteration budget on trust.

Input is a CSV with columns ``iteration``, ``weighted_mse``, ``trajectory_mse``
-- exactly the ``history`` list that ``fit_court`` already builds.  To persist it
from the main experiment, add one line inside ``main()`` of
``court_nmf_experiment.py``, right after the COURT-NMF variant is fitted::

    pd.DataFrame(output["history"]).to_csv(OUT / "court_nmf_convergence.csv",
                                           index=False)

Usage
-----
    python make_convergence_figure.py --csv outputs/court_nmf/court_nmf_convergence.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE / "outputs" / "court_nmf" / "court_nmf_convergence.csv"
DEFAULT_OUT = BASE / "figures" / "fig22_convergence.png"

SERIES = [("weighted_mse", "Weighted reconstruction MSE", "#2a78d6"),
          ("trajectory_mse", "Adjacent-season trajectory MSE", "#eb6834")]

MARK = 80          # the iteration budget used in the original submission
GRID = "#d8d8d4"
INK = "#0b0b0b"
INK2 = "#52514e"


def build(h: pd.DataFrame, out: Path, mark: int | None = MARK) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(4.9, 2.3), dpi=400)
    last = int(h.iteration.max())

    for key, label, color in SERIES:
        ax.plot(h.iteration, h[key], color=color, lw=1.4, zorder=3, label=label)
        # direct label at the right end, so identity never rests on colour alone
        ax.annotate(label, xy=(last, h[key].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6, color=color)

    if mark is not None and (h.iteration <= mark).any():
        y0 = h.loc[(h.iteration - mark).abs().idxmin()]
        ax.axvline(mark, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.annotate(f"original budget\n({mark} iterations)",
                    xy=(mark, y0[SERIES[0][0]]), xytext=(mark + last * 0.045, y0[SERIES[0][0]] * 2.4),
                    fontsize=6, color=INK2, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=2))

    ax.set_yscale("log")
    ax.set_xlim(0, last * 1.42)          # headroom for the direct labels
    # ...but do not tick past the data, or the axis implies iterations we never ran
    step = 1000 if last > 2500 else 500
    ax.set_xticks(list(range(0, last + step, step)))
    ax.set_xlabel("Projected-gradient iteration", fontsize=7)
    ax.set_ylabel("Objective diagnostic (log scale)", fontsize=7)
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    leg = ax.legend(frameon=False, fontsize=6, loc="upper right",
                    handlelength=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")

    print(f"wrote {out}")
    f = h.iloc[-1]
    for it in (80, 400, 1000, 2000, 3000, last):
        if (h.iteration <= it).any():
            r = h.loc[(h.iteration - it).abs().idxmin()]
            print(f"  iteration {int(r.iteration):>5}: weighted {r.weighted_mse:.5f} "
                  f"({r.weighted_mse / f.weighted_mse - 1:+6.1%} vs final)   "
                  f"trajectory {r.trajectory_mse:.5f} "
                  f"({r.trajectory_mse / f.trajectory_mse - 1:+6.1%} vs final)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mark", type=int, default=MARK,
                    help="iteration to annotate; pass -1 to omit")
    a = ap.parse_args()
    build(pd.read_csv(a.csv), a.out, None if a.mark < 0 else a.mark)
