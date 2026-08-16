"""
Stage-1 PCA, UMAP, and HDBSCAN analysis.

Applies the same high-missingness feature exclusions, imputation, and
raw-value eligibility filter used throughout the analysis ->
n = 45,980, StandardScaler -> log transform of right-skewed features ->
re-standardize, PCA(0.94), UMAP(2, 15, 0.1, 42)),
then runs the min_cluster_size grid {5,10,15,20,30,50} with min_samples=5
and reports the final model at
min_cluster_size=15, min_samples=5: cluster sizes, noise, and per-cluster
silhouette (block-wise exact computation on the UMAP embedding).

Outputs are written to outputs/stage1_rerun/.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import umap
import hdbscan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, 'outputs', 'stage1_rerun')
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, 'stage1_log.txt'), 'w', encoding='utf-8')


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + '\n')
    LOG.flush()


t0 = time.time()
file_path = os.path.join(BASE_DIR, 'data', 'CollegeBasketballPlayers2009-2021（55columns）.csv')

df = pd.read_csv(file_path)
log(f"Raw observations: {len(df)}")
df = df.drop(['type', 'ht'], axis=1, errors='ignore')

# ---- Unified rotation-player filter (V7): RAW values ----
_min_per_raw = pd.to_numeric(df['Min_per'], errors='coerce')
_gp_raw = pd.to_numeric(df['GP'], errors='coerce')
_pts_raw = pd.to_numeric(df['pts'], errors='coerce')
rotation_mask = (_min_per_raw > 5) & (_gp_raw >= 10) & (_pts_raw >= 1.0)
log(f"Rotation filter (raw values) keeps: {int(rotation_mask.sum())}")

HIGH_MISSING_EXCLUDE = ['pick', 'Rec Rank', 'dunksmade/(dunksmade+dunksmiss)']
MISSINGNESS_THRESHOLD = 0.30
log("Excluded from imputation/model features because missingness exceeds "
    f"{MISSINGNESS_THRESHOLD:.0%}: {', '.join(HIGH_MISSING_EXCLUDE)}")

cols_to_impute_mice = ['rimmade', 'rimmade+rimmiss', 'midmade', 'midmade+midmiss',
                       'rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
                       'dunksmade', 'dunksmiss+dunksmade']

numeric_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPPer',
                'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'porpag', 'adjoe',
                'pfr', 'ast/tov', 'ortg', 'adjtg', 'dporpag', 'stops',
                'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                'treb', 'ast', 'stl', 'blk', 'pts']
numeric_cols = [col for col in numeric_cols if col in df.columns if col not in cols_to_impute_mice]

categorical_cols = ['player_name', 'team', 'conf', 'yr']
categorical_cols = [col for col in categorical_cols if col in df.columns]

df_mice_impute = df[cols_to_impute_mice].copy()
df_numeric_impute = df[numeric_cols].copy()
df_numeric_impute = df_numeric_impute.apply(pd.to_numeric, errors='coerce')
df_categorical = df[categorical_cols].copy()
df_other = df.drop(columns=cols_to_impute_mice + numeric_cols + categorical_cols + ['pid'],
                   errors='ignore')

for col in df_numeric_impute.columns:
    df_numeric_impute[col] = df_numeric_impute[col].fillna(df_numeric_impute[col].median())
for col in df_categorical.columns:
    df_categorical[col] = df_categorical[col].fillna(
        df_categorical[col].mode()[0] if not df_categorical[col].mode().empty else 'Unknown')

le_dict = {}
for col in df_categorical.columns:
    le = LabelEncoder()
    df_categorical[col] = df_categorical[col].astype(str)
    df_categorical[col] = le.fit_transform(df_categorical[col])
    le_dict[col] = le

log(f"Starting MICE imputation ({time.time()-t0:.0f}s) ...")
mice_imputer = IterativeImputer(max_iter=25, random_state=42)
df_mice_imputed_array = mice_imputer.fit_transform(df_mice_impute)
df_mice_imputed = pd.DataFrame(df_mice_imputed_array, columns=df_mice_impute.columns,
                               index=df_mice_impute.index)
log(f"MICE done ({time.time()-t0:.0f}s)")

for col in df_mice_imputed.columns:
    if col in ['rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
               'dunksmade/(dunksmade+dunksmiss)']:
        df_mice_imputed[col] = df_mice_imputed[col].clip(0, 1)

for col in df_categorical.columns:
    df_categorical[col] = le_dict[col].inverse_transform(df_categorical[col].astype(int))

df_final = pd.concat([df_mice_imputed, df_numeric_impute, df_categorical,
                      df_other.reset_index(drop=True)], axis=1)
if 'pid' in df.columns:
    df_final.insert(0, 'pid', df['pid'].values)
df = df_final
df = df[rotation_mask.values]
log(f"Analytic sample after unified filter: {len(df)} (expect 45,980)")
assert len(df) == 45980, f"Unified sample size mismatch: {len(df)}"

# ---- per-100 features ----
per100_cols = {'pts': 'pts_per100', 'treb': 'treb_per100', 'ast': 'ast_per100',
               'stl': 'stl_per100', 'blk': 'blk_per100', 'oreb': 'oreb_per100',
               'dreb': 'dreb_per100'}
for col, new_col in per100_cols.items():
    if col in df.columns:
        df[new_col] = df[col] / df['Min_per'] * 100
        numeric_cols.append(new_col)

original_percentage_cols = ['Min_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per',
                            'FT_per', 'twoP_per', 'TP_per', 'blk_per', 'stl_per',
                            'usg', 'eFG', 'TS_per']
for col in original_percentage_cols:
    if col in df.columns:
        numeric_cols.append(col)

per_game_cols = ['GP', 'FTM', 'FTA', 'twoPM', 'twoPA', 'TPM', 'TPA', 'ftr', 'porpag',
                 'adjoe', 'pfr', 'ast/tov', 'drtg', 'adjtg', 'dporpag', 'stops',
                 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'Ortg']
for col in per_game_cols:
    if col in df.columns:
        numeric_cols.append(col)

numeric_cols = list(set([col for col in numeric_cols if col in df.columns]))
log(f"Stage-1 numeric feature count: {len(numeric_cols)}")

# ---- Standardize -> log-transform right-skewed -> re-standardize ----
scaler = StandardScaler()
numeric_df = df[numeric_cols].copy()
df[numeric_cols] = pd.DataFrame(scaler.fit_transform(numeric_df),
                                columns=numeric_cols, index=df.index)

right_skewed = ['blk', 'ast/tov', 'ast_per100', 'stl_per100', 'pts', 'dreb']
left_skewed = ['FT_per100']
for col in right_skewed:
    if col in df.columns:
        min_val = df[col].min()
        if min_val <= 0:
            df[col] = df[col] - min_val + 0.01
        df[col] = np.log(df[col])
for col in left_skewed:
    if col in df.columns:
        max_val = df[col].max()
        df[col] = max_val - df[col] + 0.01
        df[col] = np.log(df[col])

scaler2 = StandardScaler()
df[numeric_cols] = scaler2.fit_transform(df[numeric_cols])

# ---- PCA ----
pca = PCA(n_components=0.94)
numeric_df = df[numeric_cols].dropna()
pca_result = pca.fit_transform(numeric_df)
pca_df = pd.DataFrame(pca_result, index=numeric_df.index,
                      columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
log(f"PCA kept {pca_result.shape[1]} components, "
    f"variance preserved: {pca.explained_variance_ratio_.sum():.4f} ({time.time()-t0:.0f}s)")

# ---- UMAP ----
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_result = reducer.fit_transform(pca_df)
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'], index=pca_df.index)
log(f"UMAP done ({time.time()-t0:.0f}s)")

# ---- UMAP axis correlations (paper Table: UMAP loadings) ----
df_with_umap = pd.concat([umap_df, df], axis=1)
all_original_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                     'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPPer',
                     'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe',
                     'pfr', 'ast/tov', 'drtg', 'adjtg', 'dporpag', 'stops',
                     'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                     'treb', 'ast', 'stl', 'blk', 'pts', 'pts_per100', 'treb_per100',
                     'ast_per100', 'stl_per100', 'blk_per100', 'oreb_per100', 'dreb_per100',
                     'year', 'rimmade', 'rimmade+rimmiss', 'midmade',
                     'midmade+midmiss', 'rimmade/(rimmade+rimmiss)',
                     'midmade/(midmade+midmiss)', 'dunksmade', 'dunksmiss+dunksmade',
                     ]
all_original_cols = [c for c in all_original_cols if c in df_with_umap.columns]
corr_matrix = df_with_umap[['UMAP1', 'UMAP2'] + all_original_cols].corr()
umap_corrs = corr_matrix[['UMAP1', 'UMAP2']]
log("\nTop-10 |Pearson r| with UMAP1:")
log(umap_corrs['UMAP1'].abs().sort_values(ascending=False).head(11).to_string())
log("\nTop-10 |Pearson r| with UMAP2:")
log(umap_corrs['UMAP2'].abs().sort_values(ascending=False).head(11).to_string())
log("\nSigned values for paper Table (UMAP1: Min_per, mp, stops, porpag, pts, "
    "dporpag, FTM, ogbpm, obpm, FTA, pfr, TO_per):")
for c in ['Min_per', 'mp', 'stops', 'porpag', 'pts', 'dporpag', 'FTM', 'ogbpm',
          'obpm', 'FTA', 'pfr', 'TO_per']:
    if c in umap_corrs.index:
        log(f"  {c}: UMAP1 r = {umap_corrs.loc[c, 'UMAP1']:.3f}")
log("Signed values for paper Table (UMAP2: ast_per100, ast/tov, AST_per, drtg, "
    "adrtg, TP_per, TPA, TPM, ast):")
for c in ['ast_per100', 'ast/tov', 'AST_per', 'drtg', 'adrtg', 'TP_per', 'TPA',
          'TPM', 'ast']:
    if c in umap_corrs.index:
        log(f"  {c}: UMAP2 r = {umap_corrs.loc[c, 'UMAP2']:.3f}")

# ---- HDBSCAN grid ----
log(f"\nStarting HDBSCAN grid ({time.time()-t0:.0f}s) ...")
grid_labels = {}
for m in [5, 10, 15, 20, 30, 50]:
    cl = hdbscan.HDBSCAN(min_cluster_size=m, min_samples=5, prediction_data=True)
    lab = cl.fit_predict(umap_df)
    grid_labels[m] = lab
    n_clust = len(set(lab)) - (1 if -1 in lab else 0)
    noise = int(sum(lab == -1))
    sizes = pd.Series(lab).value_counts().sort_index()
    log(f"min_cluster_size={m}: {n_clust} clusters, {noise} noise "
        f"({noise/len(lab)*100:.3f}%) | sizes: {dict(sizes)} ({time.time()-t0:.0f}s)")

# Final model: min_cluster_size=15, min_samples=5.
FINAL_MIN_SIZE = 15
final_labels = grid_labels[FINAL_MIN_SIZE]
sizes = pd.Series(final_labels).value_counts().sort_index()
log(f"\nFINAL model (min_cluster_size={FINAL_MIN_SIZE}, min_samples=5):")
total = len(final_labels)
for cid, cnt in sizes.items():
    name = 'Noise (-1)' if cid == -1 else f'Cluster {cid}'
    log(f"  {name}: {cnt} ({cnt/total*100:.2f}%)")
mainland = sizes.drop(-1, errors='ignore').max()
log(f"  Dominant cluster share: {mainland/total*100:.2f}%")
peripheral = sizes.drop([-1, sizes.drop(-1, errors='ignore').idxmax()], errors='ignore').sum()
log(f"  Peripheral clusters total: {peripheral}")

np.save(os.path.join(OUTDIR, 'final_labels_mcs15.npy'), final_labels)
umap_df.assign(label=final_labels).to_csv(os.path.join(OUTDIR, 'umap_embedding_labels.csv'),
                                          index=False)

# ---- Silhouette (block-wise exact, on UMAP embedding) ----
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

log(f"\nComputing silhouettes on UMAP embedding ({time.time()-t0:.0f}s) ...")
X = umap_df.values
# (a) including noise as its own group is NOT standard; exclude noise points
mask = final_labels != -1
sil = silhouette_blockwise(X[mask], final_labels[mask])
log(f"Silhouette (noise excluded, n={mask.sum()}):")
log(f"  Overall: {sil.mean():.4f}")
for cid in sorted(np.unique(final_labels[mask])):
    log(f"  Cluster {cid}: n={int(sum(final_labels[mask]==cid))}, "
        f"silhouette={sil[final_labels[mask]==cid].mean():.4f}")

log(f"\nAll done in {time.time()-t0:.0f}s. Outputs in {OUTDIR}")
LOG.close()
