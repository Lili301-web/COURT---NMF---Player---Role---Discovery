"""Main-text figures for the COURT-NMF Monte Carlo convergence and stability study.

Reads whatever ``monte_carlo_convergence.py`` has written and produces the three
figures Section 5.4 refers to:

    fig23_mc_convergence.png  objective diagnostics vs iteration, median with
                              interquartile and 5-95% bands across runs
    fig24_mc_r2.png           distribution of the R^2 measures across runs, with
                              the seed-42 reference marked
    fig25_mc_deviation.png    per-metric deviation of the Monte Carlo mean from
                              the seed-42 result, with 95% confidence intervals

Usage
-----
    python make_monte_carlo_figures.py --mode full
    python make_monte_carlo_figures.py --mode subsample --suffix _sub
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MC = BASE / "outputs" / "monte_carlo"
FIG = BASE / "figures"

# Categorical slots 1-3 of the validated reference palette, assigned in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

RC = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7,
      "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
      "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
      "figure.facecolor": "white", "axes.facecolor": "white"}


def _style() -> None:
    plt.rcParams.update(RC)


def _tidy(ax) -> None:
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.5)


def fig_convergence(hist: pd.DataFrame, ref: pd.DataFrame | None, out: Path) -> None:
    """Median trace with interquartile and 5-95% bands, one panel per diagnostic.

    Two diagnostics on two scales would be a dual-axis chart, so they get one
    panel each rather than one panel and two y-axes.
    """
    _style()
    series = [("weighted_mse", "Weighted reconstruction MSE", BLUE),
              ("trajectory_mse", "Adjacent-season trajectory MSE", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.85), dpi=400)
    for ax, (key, label, color) in zip(axes, series):
        g = hist.groupby("iteration")[key]
        it = np.asarray(sorted(hist["iteration"].unique()))
        med = g.median().reindex(it).to_numpy()
        lo50, hi50 = g.quantile(.25).reindex(it).to_numpy(), g.quantile(.75).reindex(it).to_numpy()
        lo90, hi90 = g.quantile(.05).reindex(it).to_numpy(), g.quantile(.95).reindex(it).to_numpy()
        ax.fill_between(it, lo90, hi90, color=color, alpha=.15, lw=0, zorder=2,
                        label="5–95% of runs")
        ax.fill_between(it, lo50, hi50, color=color, alpha=.30, lw=0, zorder=3,
                        label="interquartile range")
        ax.plot(it, med, color=color, lw=1.4, zorder=4, label="median run")
        if ref is not None and key in ref:
            ax.plot(ref["iteration"], ref[key], color=INK, lw=0.9, ls=(0, (3, 2)),
                    zorder=5, label="seed 42 (published)")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_xlabel("Projected-gradient iteration", fontsize=6.5)
        _tidy(ax)
    axes[0].set_ylabel("Diagnostic (log scale)", fontsize=6.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.4, rect=(0, 0.13, 1, 1))
    leg = fig.legend(h, l, frameon=False, fontsize=6, ncol=4, loc="lower center",
                     bbox_to_anchor=(0.5, 0.0), handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_r2(runs: pd.DataFrame, ref: dict, out: Path) -> None:
    """Histogram of each R^2 measure with the seed-42 value marked."""
    _style()
    panels = [("mixture_r2", "Mixture reconstruction $R^2$", BLUE),
              ("permanova_r2", "External-validation PERMANOVA $R^2$", ORANGE)]
    panels = [p for p in panels if p[0] in runs and runs[p[0]].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4, 2.0), dpi=400, squeeze=False)
    for ax, (key, label, color) in zip(axes[0], panels):
        v = runs[key].dropna().to_numpy(float)
        ax.hist(v, bins=min(30, max(8, len(v) // 4)), color=color, alpha=.75,
                edgecolor="white", linewidth=.4, zorder=3)
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.axvspan(lo, hi, color=color, alpha=.12, lw=0, zorder=2)
        ax.axvline(v.mean(), color=color, lw=1.2, zorder=4)
        # anchored to the axes, not to the mean, so it cannot land on a bar
        ax.text(0.02, 0.98, f"mean {v.mean():.4f}\n95% [{lo:.4f}, {hi:.4f}]",
                transform=ax.transAxes, fontsize=5.5, color=INK2, ha="left", va="top")
        r = ref.get(key)
        if r is not None and np.isfinite(r):
            ax.axvline(r, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
            ax.annotate("seed 42", xy=(r, 0), xytext=(2, 6), textcoords="offset points",
                        fontsize=5.5, color=INK, rotation=90, ha="left", va="bottom")
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_xlabel("$R^2$", fontsize=6.5)
        _tidy(ax)
    axes[0][0].set_ylabel("Runs", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_deviation(summary: dict, out: Path) -> None:
    """Percentage deviation of the Monte Carlo mean from seed 42, with 95% CIs.

    Everything is expressed as a percentage of the seed-42 value so that
    diagnostics on different scales share one axis instead of needing two.
    """
    _style()
    order = [("weighted_mse", "Weighted MSE"), ("trajectory_mse", "Trajectory MSE"),
             ("mixture_r2", "Mixture $R^2$"), ("permanova_r2", "PERMANOVA $R^2$"),
             ("cv_accuracy", "OOF accuracy"), ("cv_macro_f1", "OOF macro-$F_1$"),
             ("mean_max_membership", "Mean max membership"), ("mean_entropy", "Mean entropy")]
    rows = []
    for key, label in order:
        m = summary["metrics"].get(key)
        if not m or "seed42" not in m or not m["seed42"]:
            continue
        ref = m["seed42"]
        rows.append((label, (m["mean"] - ref) / abs(ref) * 100,
                     (m.get("mean_ci95_lo", m["mean"]) - ref) / abs(ref) * 100,
                     (m.get("mean_ci95_hi", m["mean"]) - ref) / abs(ref) * 100))
    if not rows:
        print("no metrics comparable to seed 42; skipping deviation figure")
        return
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(4.9, 0.32 * len(rows) + 0.85), dpi=400)
    ax.axvline(0, color=INK, lw=0.9, zorder=3)
    for yi, (_, mean, lo, hi) in zip(y, rows):
        # colour encodes direction, and the sign is also carried by position
        # relative to the zero line, so identity never rests on colour alone
        color = ORANGE if mean > 0 else BLUE
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, solid_capstyle="round", zorder=4)
        ax.plot([mean], [yi], "o", color=color, ms=4.5, mec="white", mew=.8, zorder=5)
        ax.annotate(f"{mean:+.2f}%", xy=(hi, yi), xytext=(5, 0), textcoords="offset points",
                    fontsize=5.5, color=INK2, va="center", ha="left")
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel("Deviation of Monte Carlo mean from the seed-42 fit (%), with 95% CI", fontsize=6.5)
    span = max(abs(min(r[2] for r in rows)), abs(max(r[3] for r in rows)))
    ax.set_xlim(-span * 1.35, span * 1.35)
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_distribution(runs: pd.DataFrame, ref: dict, summary: dict, out: Path) -> None:
    """R^2 distributions and the seed-42 deviation panel, as one float.

    Same two pieces of evidence as the separate figures, laid out as small
    multiples so the main text spends one figure on them rather than two.
    """
    _style()
    panels = [("mixture_r2", "Mixture reconstruction $R^2$", BLUE),
              ("permanova_r2", "External PERMANOVA $R^2$", ORANGE)]
    panels = [p for p in panels if p[0] in runs and runs[p[0]].notna().any()]

    order = [("weighted_mse", "Weighted MSE"), ("trajectory_mse", "Traj. MSE"),
             ("mixture_r2", "Mixture $R^2$"), ("permanova_r2", "PERMANOVA $R^2$"),
             ("cv_accuracy", "OOF acc."), ("cv_macro_f1", "OOF macro-$F_1$"),
             ("mean_max_membership", "Max membership"), ("mean_entropy", "Entropy")]
    rows = []
    for key, label in order:
        m = summary["metrics"].get(key)
        if not m or "seed42" not in m or not m["seed42"]:
            continue
        r = m["seed42"]
        rows.append((label, (m["mean"] - r) / abs(r) * 100,
                     (m.get("mean_ci95_lo", m["mean"]) - r) / abs(r) * 100,
                     (m.get("mean_ci95_hi", m["mean"]) - r) / abs(r) * 100))

    draw_dev = False  # deviation panel duplicates the table column; see caption
    ncols = len(panels) + (1 if draw_dev else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5.4, 1.75), dpi=400,
                             gridspec_kw={"width_ratios": [1] * len(panels)
                                          + ([1.35] if draw_dev else [])})
    for ax, (key, label, color) in zip(axes, panels):
        v = runs[key].dropna().to_numpy(float)
        ax.hist(v, bins=min(40, max(8, int(np.sqrt(len(v))))), color=color, alpha=.75,
                edgecolor="white", linewidth=.3, zorder=3)
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.axvspan(lo, hi, color=color, alpha=.12, lw=0, zorder=2)
        ax.axvline(v.mean(), color=color, lw=1.2, zorder=4)
        ax.text(0.02, 0.98, f"mean {v.mean():.4f}\n95% [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes, fontsize=5.2, color=INK2, ha="left", va="top")
        r = ref.get(key)
        if r is not None and np.isfinite(r):
            ax.axvline(r, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
            ax.annotate("seed 42", xy=(r, 0), xytext=(2, 5), textcoords="offset points",
                        fontsize=5.2, color=INK, rotation=90, ha="left", va="bottom")
        ax.set_title(label, fontsize=6.5, pad=3)
        ax.set_xlabel("$R^2$", fontsize=6)
        _tidy(ax)
    axes[0].set_ylabel("Runs", fontsize=6)

    if draw_dev:
        ax = axes[-1]
        if rows:
            y = np.arange(len(rows))[::-1]
            ax.axvline(0, color=INK, lw=0.9, zorder=3)
            for yi, (_, mean, lo, hi) in zip(y, rows):
                color = ORANGE if mean > 0 else BLUE
                ax.plot([lo, hi], [yi, yi], color=color, lw=1.4, solid_capstyle="round", zorder=4)
                ax.plot([mean], [yi], "o", color=color, ms=3.4, mec="white", mew=.7, zorder=5)
            ax.set_yticks(y, [r[0] for r in rows], fontsize=5.4)
            span = max(abs(min(r[2] for r in rows)), abs(max(r[3] for r in rows)))
            ax.set_xlim(-span * 1.2, span * 1.2)
        # matplotlib is not LaTeX: a backslash-escaped percent renders literally
        ax.set_title("Deviation from seed 42 (%), 95% CI", fontsize=6.5, pad=3)
        ax.set_xlabel("% of the seed-42 value", fontsize=6)
        _tidy(ax)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4, w_pad=0.9)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "subsample"], default="full")
    ap.add_argument("--suffix", default="", help="appended to output filenames")
    a = ap.parse_args()

    runs = pd.read_csv(MC / f"mc_runs_{a.mode}.csv")
    hist = pd.read_csv(MC / f"mc_history_{a.mode}.csv.gz")
    summary = json.loads((MC / f"mc_summary_{a.mode}.json").read_text())
    iters = int(runs["iterations"].mode().iloc[0])
    ref_hist_path = MC / f"reference_history_{a.mode}_{iters}.csv"
    ref_hist = pd.read_csv(ref_hist_path) if ref_hist_path.exists() else None
    ref_diag = summary.get("reference_diagnostics", {})

    fig_convergence(hist, ref_hist, FIG / f"fig23_mc_convergence{a.suffix}.png")
    fig_distribution(runs, ref_diag, summary, FIG / f"fig24_mc_distribution{a.suffix}.png")
    # the separate versions stay available for the supplement
    fig_r2(runs, ref_diag, FIG / f"fig24_mc_r2{a.suffix}.png")
    fig_deviation(summary, FIG / f"fig25_mc_deviation{a.suffix}.png")


if __name__ == "__main__":
    main()
