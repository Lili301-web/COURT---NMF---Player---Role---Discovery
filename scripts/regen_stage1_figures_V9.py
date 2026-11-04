"""
Regenerate Stage-1 figures and signed UMAP-axis correlations.

Uses the saved Stage-1 embedding from the final HDBSCAN solution.

Writes:
  figures/fig05_stability.png, figures/fig06_hdbscan_dist.png,
  figures/fig07_condensed_tree.png, figures/fig09_silhouette_hdbscan.png
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)
t0 = time.time()

emb = pd.read_csv(os.path.join(BASE, 'outputs', 'stage1_rerun', 'umap_embedding_labels.csv'))
X = emb[['UMAP1', 'UMAP2']].values
labels = emb['label'].values
print(f"embedding loaded: {emb.shape} ({time.time()-t0:.0f}s)", flush=True)

# ==================== rebuild preprocessed features (for correlations) ====
file_path = os.path.join(BASE, 'data', 'CollegeBasketballPlayers2009-2021（55columns）.csv')
df = pd.read_csv(file_path, low_memory=False)
df = df.drop(['type', 'ht'], axis=1, errors='ignore')
_min_per_raw = pd.to_numeric(df['Min_per'], errors='coerce')
_gp_raw = pd.to_numeric(df['GP'], errors='coerce')
_pts_raw = pd.to_numeric(df['pts'], errors='coerce')
rotation_mask = (_min_per_raw > 5) & (_gp_raw >= 10) & (_pts_raw >= 1.0)

HIGH_MISSING_EXCLUDE = ['pick', 'Rec Rank', 'dunksmade/(dunksmade+dunksmiss)']
MISSINGNESS_THRESHOLD = 0.30
print("excluded from imputation/model features because missingness exceeds "
      f"{MISSINGNESS_THRESHOLD:.0%}: {', '.join(HIGH_MISSING_EXCLUDE)}", flush=True)

cols_to_impute_mice = ['rimmade', 'rimmade+rimmiss', 'midmade', 'midmade+midmiss',
                       'rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
                       'dunksmade', 'dunksmiss+dunksmade']
numeric_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPPer',
                'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'porpag', 'adjoe',
                'pfr', 'ast/tov', 'ortg', 'adjtg', 'dporpag', 'stops',
                'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                'treb', 'ast', 'stl', 'blk', 'pts']
numeric_cols = [c for c in numeric_cols if c in df.columns if c not in cols_to_impute_mice]
categorical_cols = [c for c in ['player_name', 'team', 'conf', 'yr'] if c in df.columns]

df_mice_impute = df[cols_to_impute_mice].copy()
df_numeric_impute = df[numeric_cols].copy().apply(pd.to_numeric, errors='coerce')
df_categorical = df[categorical_cols].copy()
df_other = df.drop(columns=cols_to_impute_mice + numeric_cols + categorical_cols + ['pid'],
                   errors='ignore')
for col in df_numeric_impute.columns:
    df_numeric_impute[col] = df_numeric_impute[col].fillna(df_numeric_impute[col].median())
for col in df_categorical.columns:
    df_categorical[col] = df_categorical[col].fillna(
        df_categorical[col].mode()[0] if not df_categorical[col].mode().empty else 'Unknown')
for col in df_categorical.columns:
    le = LabelEncoder()
    df_categorical[col] = le.fit_transform(df_categorical[col].astype(str))

mice_imputer = IterativeImputer(max_iter=25, random_state=42)
df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df_mice_impute),
                               columns=df_mice_impute.columns, index=df_mice_impute.index)
for col in df_mice_imputed.columns:
    if col in ['rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
               'dunksmade/(dunksmade+dunksmiss)']:
        df_mice_imputed[col] = df_mice_imputed[col].clip(0, 1)

df_final = pd.concat([df_mice_imputed, df_numeric_impute, df_categorical,
                      df_other.reset_index(drop=True)], axis=1)
df = df_final[rotation_mask.values].reset_index(drop=True)
print(f"features rebuilt: {df.shape} ({time.time()-t0:.0f}s)", flush=True)

per100_cols = {'pts': 'pts_per100', 'treb': 'treb_per100', 'ast': 'ast_per100',
               'stl': 'stl_per100', 'blk': 'blk_per100', 'oreb': 'oreb_per100',
               'dreb': 'dreb_per100'}
for col, new_col in per100_cols.items():
    if col in df.columns:
        df[new_col] = df[col] / df['Min_per'] * 100

# ==================== signed Pearson correlations ====================
all_original_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                     'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPPer',
                     'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe',
                     'pfr', 'ast/tov', 'drtg', 'adrtg', 'adjtg', 'dporpag', 'stops',
                     'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                     'treb', 'ast', 'stl', 'blk', 'pts', 'pts_per100', 'treb_per100',
                     'ast_per100', 'stl_per100', 'blk_per100', 'oreb_per100', 'dreb_per100',
                     'year', 'rimmade', 'rimmade+rimmiss', 'midmade',
                     'midmade+midmiss', 'rimmade/(rimmade+rimmiss)',
                     'midmade/(midmade+midmiss)', 'dunksmade', 'dunksmiss+dunksmade',
                     ]
all_original_cols = [c for c in all_original_cols if c in df.columns]
corr_df = pd.concat([emb[['UMAP1', 'UMAP2']].reset_index(drop=True),
                     df[all_original_cols].reset_index(drop=True)], axis=1)
corr_matrix = corr_df.corr()
u1 = corr_matrix['UMAP1'].drop(['UMAP1', 'UMAP2'])
u2 = corr_matrix['UMAP2'].drop(['UMAP1', 'UMAP2'])

print("\n=== UMAP1: top-12 signed Pearson r (by |r|) ===")
for name, val in u1.reindex(u1.abs().sort_values(ascending=False).index).head(12).items():
    print(f"  {name}: {val:.3f}")
print("\n=== UMAP2: top-12 signed Pearson r (by |r|) ===")
for name, val in u2.reindex(u2.abs().sort_values(ascending=False).index).head(12).items():
    print(f"  {name}: {val:.3f}")
print("\n=== minutes/era vars vs UMAP2 ===")
for c in ['Min_per', 'mp', 'year', 'ast_per100', 'ast/tov', 'AST_per', 'drtg']:
    if c in u2.index:
        print(f"  {c}: UMAP2 r = {u2[c]:.3f}")

# ==================== fig06: HDBSCAN cluster scatter ====================
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(emb['UMAP1'], emb['UMAP2'], c=labels, s=2, alpha=0.7, cmap='tab10')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Cluster ID')
n_clust = len(set(labels)) - (1 if -1 in labels else 0)
noise = int(sum(labels == -1))
ax.set_title(f'HDBSCAN Clustering (min_cluster_size=15)\n{n_clust} clusters, {noise} noise points',
             fontsize=14)
ax.set_xlabel('UMAP Dimension 1', fontsize=12.5)
ax.set_ylabel('UMAP Dimension 2', fontsize=12.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig06_hdbscan_dist.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"fig06 saved ({time.time()-t0:.0f}s)", flush=True)

# ==================== fig07: condensed tree ====================
cl = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
cl.fit(X)
plt.figure(figsize=(10, 6))
cl.condensed_tree_.plot(select_clusters=True)
plt.title('HDBSCAN Condensed Tree (min_cluster_size=15)')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig07_condensed_tree.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"fig07 saved ({time.time()-t0:.0f}s)", flush=True)

# ==================== fig05: stability grid ================================
grid = {5: (4, 12), 10: (4, 12), 15: (4, 12), 20: (4, 12), 30: (4, 12), 50: (3, 0)}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(grid), [v[0] for v in grid.values()], 'bo-')
ax1.set_xlabel('min_cluster_size'); ax1.set_ylabel('Number of clusters')
ax1.set_title('Cluster Count Stability')
ax2.plot(list(grid), [v[1] / 45980 * 100 for v in grid.values()], 'ro-')
ax2.set_xlabel('min_cluster_size'); ax2.set_ylabel('Noise proportion (%)')
ax2.set_title('Noise Proportion Stability')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig05_stability.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"fig05 saved ({time.time()-t0:.0f}s)", flush=True)

# ==================== fig09: silhouette plot (blockwise, noise excluded) =
def silhouette_blockwise(X, lab, block=500):
    n = X.shape[0]
    ids = np.unique(lab)
    OH = np.zeros((n, len(ids)))
    id2pos = {c: j for j, c in enumerate(ids)}
    OH[np.arange(n), [id2pos[c] for c in lab]] = 1.0
    cnt = OH.sum(0)
    own_pos = np.array([id2pos[c] for c in lab])
    out = np.zeros(n)
    xnorm = (X ** 2).sum(1)
    for s in range(0, n, block):
        e = min(s + block, n)
        G = X[s:e] @ X.T
        D2 = xnorm[s:e][:, None] + xnorm[None, :] - 2.0 * G
        np.maximum(D2, 0, out=D2)
        D = np.sqrt(D2)
        S = D @ OH
        a = S[np.arange(e - s), own_pos[s:e]] / (cnt[own_pos[s:e]] - 1)
        S[np.arange(e - s), own_pos[s:e]] = np.inf
        b = (S / cnt).min(1)
        out[s:e] = (b - a) / np.maximum(a, b)
    return out

mask = labels != -1
sil = silhouette_blockwise(X[mask], labels[mask])
lab_m = labels[mask]
print(f"\nsilhouettes recomputed ({time.time()-t0:.0f}s):")
print(f"  overall: {sil.mean():.4f}")
for cid in sorted(np.unique(lab_m)):
    print(f"  cluster {cid}: {sil[lab_m == cid].mean():.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 0
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for k, cid in enumerate(sorted(np.unique(lab_m))):
    vals = np.sort(sil[lab_m == cid])
    size = vals.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=colors[k], edgecolor='none', alpha=0.85,
                     label=f'Cluster {cid} (n={size})')
    y_lower = y_upper
avg = sil.mean()
ax.axvline(x=avg, color='red', linestyle='--', label=f'Average: {avg:.3f}')
ax.set_title('HDBSCAN Silhouette Plot (min_cluster_size=15)')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Player (sorted by cluster)')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig09_silhouette_hdbscan.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"fig09 saved ({time.time()-t0:.0f}s)", flush=True)
import matplotlib.image as mpimg
import os

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
paths = ['figures/fig03_umap_scatter.png', 'figures/fig06_hdbscan_dist.png']
for ax, p in zip(axes, paths):
    img = mpimg.imread(p)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.savefig('combined_fig03_fig06.png', dpi=300, bbox_inches='tight')
plt.close()
print("ALL DONE")

