"""
Stage-2 direct NMF + K-means pipeline.

- Unified analytic sample (n = 45,980) shared with the Stage-1 UMAP+HDBSCAN
  pipeline, via the raw-value rotation filter (Min_per > 5, GP >= 10,
  pts >= 1.0) evaluated BEFORE imputation.
- Preprocessing: drop high-missingness modeling fields before imputation,
  then median/mode imputation + MICE for low-missingness shot-location
columns, RobustScaler + MinMaxScaler (no log transform) using a direct
non-negative representation of the 23 style features.
- Silhouette scores are computed block-wise, avoiding a full 45,980 × 45,980
  distance matrix in memory.

KMeans cluster IDs are arbitrary; after fitting, labels are remapped to the
reported archetype order:
  C0 = steal-oriented scoring wing, C1 = defensive rebound anchor,
  C2 = efficient floor-spacing combo guard, C3 = high-volume bench scorer,
  C4 = two-way glass-cleaning scorer
(raw KMeans IDs: 0->C0, 4->C1, 2->C2, 1->C3, 3->C4).

Outputs are written to outputs/v7_rerun/.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, 'outputs', 'v7_rerun')
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, 'rerun_log.txt'), 'w', encoding='utf-8')


def log(msg):
    print(msg, flush=True)
    LOG.write(str(msg) + '\n')
    LOG.flush()


t0 = time.time()
file_path = os.path.join(BASE_DIR, 'data', 'CollegeBasketballPlayers2009-2021（55columns）.csv')

# ==================== DATA LOADING & PREPROCESSING ====================
df = pd.read_csv(file_path)
log(f"Raw observations: {len(df)}")
log(f"Unique player ids (pid): {df['pid'].nunique()}")
df = df.drop(['type', 'ht'], axis=1, errors='ignore')

# ---- Unified rotation-player filter (V7): RAW (pre-imputation) values ----
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
cols_to_impute_mice = [c for c in cols_to_impute_mice if c in df.columns]

numeric_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per',
                'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe',
                'pfr', 'ast/tov', 'drtg', 'adrtg', 'dporpag', 'stops',
                'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                'treb', 'ast', 'stl', 'blk', 'pts']
numeric_cols = [c for c in numeric_cols if c in df.columns and c not in cols_to_impute_mice]

categorical_cols = ['player_name', 'team', 'conf', 'yr']
categorical_cols = [c for c in categorical_cols if c in df.columns]

df_mice_impute = df[cols_to_impute_mice].copy()
df_numeric_impute = df[numeric_cols].copy()
df_categorical = df[categorical_cols].copy()
df_other = df.drop(columns=cols_to_impute_mice + numeric_cols + categorical_cols + ['pid'],
                   errors='ignore')

for col in df_numeric_impute.columns:
    df_numeric_impute[col] = pd.to_numeric(df_numeric_impute[col], errors='coerce')
    df_numeric_impute[col] = df_numeric_impute[col].fillna(df_numeric_impute[col].median())

for col in df_categorical.columns:
    df_categorical[col] = df_categorical[col].fillna(
        df_categorical[col].mode()[0] if not df_categorical[col].mode().empty else 'Unknown')

le_dict = {}
for col in df_categorical.columns:
    le = LabelEncoder()
    df_categorical[col] = le.fit_transform(df_categorical[col].astype(str))
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

# ---- Apply the unified filter ----
df = df[rotation_mask.values].copy()
log(f"Analytic sample after unified filter: {len(df)} (expect 45,980)")
assert len(df) == 45980, f"Unified sample size mismatch: {len(df)}"

# ==================== FEATURE ENGINEERING ====================
per100_cols = {'pts': 'pts_per100', 'treb': 'treb_per100', 'ast': 'ast_per100',
               'stl': 'stl_per100', 'blk': 'blk_per100', 'oreb': 'oreb_per100',
               'dreb': 'dreb_per100'}
for col, new_col in per100_cols.items():
    if col in df.columns:
        df[new_col] = df[col] / df['Min_per'] * 100
        numeric_cols.append(new_col)

for col in ['Min_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FT_per', 'twoP_per',
            'TP_per', 'blk_per', 'stl_per', 'usg', 'eFG', 'TS_per', 'GP', 'FTM', 'FTA',
            'twoPM', 'twoPA', 'TPM', 'TPA', 'ftr', 'porpag', 'adjoe', 'pfr', 'ast/tov',
            'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp',
            'ogbpm', 'dgbpm', 'Ortg']:
    if col in df.columns:
        numeric_cols.append(col)
numeric_cols = list(set([c for c in numeric_cols if c in df.columns]))

# Keep a RAW-scale copy (pre-scaling) for cluster-mean reporting
report_metrics = ['pts_per100', 'treb_per100', 'ast_per100', 'stl_per100', 'blk_per100',
                  'oreb_per100', 'dreb_per100', 'usg', 'eFG', 'TS_per', 'ORB_per',
                  'DRB_per', 'AST_per', 'TO_per', 'FT_per', 'TP_per', 'blk_per',
                  'stl_per', 'ftr', 'bpm', 'obpm', 'dbpm', 'Min_per', 'GP', 'Ortg', 'drtg']
report_metrics = [m for m in report_metrics if m in df.columns]
df_raw = df[report_metrics + ['player_name', 'team', 'yr', 'pick']].copy()

# NOTE (V9): Stage 2 applies RobustScaler for outlier resistance, then
# MinMaxScaler to map the original style metrics to [0, 1] before NMF.
# No PCA score shifting is used, so NMF loadings remain directly tied to
# the 23 basketball features.
scaler = RobustScaler()
df_scaled = df[numeric_cols].dropna().copy()
df_scaled[:] = scaler.fit_transform(df_scaled)
log(f"Scaled matrix shape: {df_scaled.shape}")

# ==================== DIRECT NMF ====================
nmf_cols = ['pts_per100', 'treb_per100', 'ast_per100', 'stl_per100', 'blk_per100',
            'oreb_per100', 'dreb_per100', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
            'AST_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per', 'blk_per', 'stl_per',
            'ftr', 'bpm', 'obpm', 'dbpm']
nmf_cols = [c for c in nmf_cols if c in df_scaled.columns]
minmax = MinMaxScaler()
df_nmf_input = pd.DataFrame(minmax.fit_transform(df_scaled[nmf_cols]),
                            columns=nmf_cols, index=df_scaled.index)
log(f"NMF input shape: {df_nmf_input.shape}")
log(f"NMF input range: [{df_nmf_input.min().min():.4f}, {df_nmf_input.max().max():.4f}]")

n_components_nmf = 5
nmf = NMF(n_components=n_components_nmf, random_state=42, init='nndsvd', max_iter=500)
nmf_weights = nmf.fit_transform(df_nmf_input)
player_style_space = pd.DataFrame(nmf_weights, index=df_scaled.index,
                                  columns=[f'Style{i+1}' for i in range(n_components_nmf)])
log(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")

# NMF loadings are directly expressed on the original 23 style features.
contrib = pd.DataFrame(nmf.components_.T, index=nmf_cols,
                       columns=[f'Dim{i}' for i in range(n_components_nmf)])
contrib.to_csv(os.path.join(OUTDIR, 'nmf_loadings_original_space.csv'))
log("\nTop positive loadings per NMF dimension:")
for i in range(n_components_nmf):
    top = contrib[f'Dim{i}'].sort_values(ascending=False).head(7)
    log(f"  Dim{i}: " + ", ".join(f"{idx} {val:.1f}" for idx, val in top.items()))

# ==================== K-MEANS SWEEP ====================
X = player_style_space.values
labelings = {}
for K in range(2, 11):
    km = KMeans(n_clusters=K, random_state=42, n_init=25)
    labelings[K] = km.fit_predict(X)
log(f"KMeans sweep fitted ({time.time()-t0:.0f}s)")


def silhouette_blockwise(X, labelings, block=500):
    """Exact silhouette samples (identical to sklearn) computed block-wise."""
    n = X.shape[0]
    out = {}
    meta = {}
    for K, lab in labelings.items():
        ids = np.unique(lab)
        OH = np.zeros((n, len(ids)))
        id2pos = {c: j for j, c in enumerate(ids)}
        OH[np.arange(n), [id2pos[c] for c in lab]] = 1.0
        cnt = OH.sum(0)
        out[K] = np.zeros(n)
        meta[K] = (OH, cnt, np.array([id2pos[c] for c in lab]))
    xnorm = (X ** 2).sum(1)
    for s in range(0, n, block):
        e = min(s + block, n)
        G = X[s:e] @ X.T
        D2 = xnorm[s:e][:, None] + xnorm[None, :] - 2.0 * G
        np.maximum(D2, 0, out=D2)
        D = np.sqrt(D2)
        for K in labelings:
            OH, cnt, own_pos = meta[K]
            S = D @ OH                                   # B x K sum of distances
            a = S[np.arange(e - s), own_pos[s:e]] / (cnt[own_pos[s:e]] - 1)
            S[np.arange(e - s), own_pos[s:e]] = np.inf   # exclude own cluster
            b = (S / cnt).min(1)
            out[K][s:e] = (b - a) / np.maximum(a, b)
    return out


log(f"Computing block-wise silhouettes for K=2..10 ({time.time()-t0:.0f}s) ...")
sil = silhouette_blockwise(X, labelings)
log(f"Silhouettes done ({time.time()-t0:.0f}s)")

log("\nGlobal silhouette by K:")
sil_table = {}
for K in range(2, 11):
    sil_table[K] = sil[K].mean()
    log(f"  K={K}: {sil[K].mean():.4f}")
pd.DataFrame({'K': list(sil_table), 'silhouette': list(sil_table.values())}).to_csv(
    os.path.join(OUTDIR, 'kmeans_silhouette_sweep.csv'), index=False)

# ==================== FINAL K=5 SOLUTION ====================
# Remap raw K-means IDs to the reported archetype order.
RELABEL = {0: 0, 4: 1, 2: 2, 1: 3, 3: 4}
labels5_raw = labelings[5]
labels5 = np.array([RELABEL[c] for c in labels5_raw])

sizes = pd.Series(labels5).value_counts().sort_index()
log("\nK=5 cluster sizes and shares (paper archetype order):")
for c, n_c in sizes.items():
    log(f"  Cluster {c}: {n_c} ({n_c / len(labels5) * 100:.1f}%)")

log("\nK=5 per-cluster silhouette (paper archetype order):")
sil5 = sil[5]
for c in sorted(np.unique(labels5)):
    log(f"  Cluster {c}: {sil5[labels5 == c].mean():.3f}")
log(f"  Global: {sil5.mean():.4f}")

labels2 = labelings[2]
sizes2 = pd.Series(labels2).value_counts().sort_index()
log("\nK=2 cluster sizes: " + ", ".join(f"C{c}: {n_c}" for c, n_c in sizes2.items()))
log(f"K=2 global silhouette: {sil[2].mean():.4f}")

# Cluster means on RAW metric values (paper archetype order)
df_raw['cluster'] = labels5
cluster_means = df_raw.groupby('cluster')[report_metrics].mean()
cluster_means.to_csv(os.path.join(OUTDIR, 'cluster_means_raw.csv'))
log("\nCluster means (raw scale, paper archetype order):")
log(cluster_means.round(2).to_string())

# ==================== EXTERNAL VALIDATION ====================
df['cluster'] = labels5
df['silhouette_score'] = sil5
draft_filter = df['pick'].notna() & (df['pick'] <= 60)
minutes_filter = df['Min_per'] > 49
games_filter = df['GP'] > 20
high_level_players = df_scaled.index[draft_filter | (minutes_filter & games_filter)]
log(f"\nHigh-level eligible players: {len(high_level_players)}")

# NOTE: selection is by RAW KMeans cluster id; remap afterwards for reporting.
typical_players = []
for raw_id in range(5):
    cluster_indices = df_scaled.index[labels5_raw == raw_id]
    eligible = [i for i in cluster_indices if i in high_level_players]
    pool = eligible if len(eligible) > 0 else list(cluster_indices)
    top_n = df.loc[pool, 'silhouette_score'].nlargest(min(4, len(pool))).index
    typical_players.extend(top_n)

val_names = df.loc[typical_players, ['player_name', 'team', 'yr', 'cluster',
                                     'silhouette_score', 'pick']].copy()
val_names.to_csv(os.path.join(OUTDIR, 'validation_top4_per_cluster.csv'), index=False)
log("\nTop-4 silhouette players per cluster (paper archetype order):")
log(val_names.to_string())

# Positions scraped from Sports-Reference (July 2026) -- ground truth the
# pipeline never observed. See scrape_positions_V7.py.
position_data = [
    ('John Galvin', 'forward'), ('Brandon Davies', 'forward'),
    ('Akiean Frederick', 'forward'), ('Renaldo Woolridge', 'forward'),
    ('Jair Bolden', 'guard'), ('Aaron Nettles', 'guard'),
    ('Chance McMillian', 'guard'), ('Maxwell Evans', 'guard'),
    ('Cameron Carmical', 'guard'), ('Kendall Cutler', 'guard'),
    ('Hasan Abdullah', 'guard'), ('CJ Davis', 'guard'),
    ('Abu Kigab', 'forward'), ('Mark Carter', 'forward'),
    ('Damian Goodwin', 'forward'), ('Stephan Gabriel', 'guard'),
    ('Tom Jervis', 'center'), ('Robert Nwankwo', 'forward'),
    ('Mouphtaou Yarou', 'forward'), ('Chris Vogt', 'center'),
]
position_dict = {name: pos for name, pos in position_data}
df['position'] = df['player_name'].map(position_dict)
sub = df[df['position'].notna()]
# one row per player (keep the highest-silhouette season)
sub = sub.sort_values('silhouette_score', ascending=False).groupby('player_name').head(1)
ct = pd.crosstab(sub['cluster'], sub['position'])
log("\nContingency table (paper archetype order):")
log(ct.to_string())
ct.to_csv(os.path.join(OUTDIR, 'validation_crosstab.csv'))
if ct.shape[0] > 1 and ct.shape[1] > 1:
    chi2, p, dof, expected = chi2_contingency(ct)
    log(f"\nChi-square = {chi2:.2f}, df = {dof}, p = {p:.3e}")

# K=2 style-space characterisation.
style_means_k2 = pd.DataFrame(X, columns=[f'Style{i+1}' for i in range(5)])
style_means_k2['cluster2'] = labels2
log("\nK=2 style-dimension means:")
log(style_means_k2.groupby('cluster2').mean().round(3).to_string())

style_means_k5 = pd.DataFrame(X, columns=[f'Style{i+1}' for i in range(5)])
style_means_k5['cluster'] = labels5
log("\nK=5 style-dimension means (paper archetype order):")
log(style_means_k5.groupby('cluster').mean().round(3).to_string())

log(f"\nAll done in {time.time()-t0:.0f}s. Outputs in {OUTDIR}")
LOG.close()
