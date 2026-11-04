"""Seed-matched ablation strip plot for the project narrative (COURT_NMF_Story).

Same data as Fig. 3 of the R3 manuscript (outputs/matched_ablation/runs.csv):
each dot is one of the ten matched starts, the diamond the selected start.
Writes figures/fig27_matched_ablations.png.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
d = pd.read_csv(BASE / "outputs" / "matched_ablation" / "runs.csv")
order = ["no_opportunity", "no_separation", "no_trajectory", "no_context", "full", "dynamic"]
names = {"dynamic": "Graph-only dynamic NMF", "full": "COURT-NMF (full)", "no_context": "No context",
         "no_trajectory": "No trajectory", "no_separation": "No separation", "no_opportunity": "No opportunity weights"}
d = d[d.variant.isin(order)]
sel = d.loc[d.groupby("variant").objective.idxmin()].set_index("variant")
INK, INK2, GRID, C1 = "#1c1d20", "#5a5e66", "#e2dfd7", "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK, "axes.labelcolor": INK})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, col, title in [(axes[0], "weighted_mse_common", "Reconstruction MSE ×10³"),
                       (axes[1], "trajectory_mse", "Trajectory (drift) MSE ×10³")]:
    for y, v in enumerate(order):
        dv = d[d.variant == v]
        ax.scatter(dv[col] * 1e3, [y] * len(dv), s=26, facecolors="none", edgecolors=C1, linewidths=1.0,
                   label="ten matched starts" if y == 0 else None, zorder=3)
        ax.scatter([sel.loc[v, col] * 1e3], [y], s=46, marker="D", color=INK, zorder=4,
                   label="selected start" if y == 0 else None)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(order)), [names[v] for v in order])
axes[1].set_xlim(0, None)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.6, 1.0))
fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.92))
out = BASE / "figures" / "fig27_matched_ablations.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
