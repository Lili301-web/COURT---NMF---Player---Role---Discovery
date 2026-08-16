import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    base_dir = os.getcwd()

data_dir = os.path.join(base_dir, 'data')

output_dir = os.path.join(base_dir, 'outputs', 'cluster_csvs')
os.makedirs(output_dir, exist_ok=True)

file_name = 'CollegeBasketballPlayers2009-2021（55columns）.csv'
file_path = os.path.join(data_dir, file_name)

df = pd.read_csv(file_path)
df = df.drop(['type', 'ht'], axis=1, errors='ignore')

# Apply the eligibility criteria to raw values before imputation so both
# analytic stages use the same player-season sample:
# Min_per > 5, GP >= 10, pts >= 1.0  ->  n = 45,980 player-seasons.
# Final analysis parameters:
#   PCA(n_components=0.94) -> 16 PCs; UMAP(n_components=2, n_neighbors=15,
#   min_dist=0.1, random_state=42); HDBSCAN(min_cluster_size=15,
#   min_samples=5) -> 4 clusters (45,285 / 371 / 41 / 271) + 12 noise.
_min_per_raw = pd.to_numeric(df['Min_per'], errors='coerce')
_gp_raw = pd.to_numeric(df['GP'], errors='coerce')
_pts_raw = pd.to_numeric(df['pts'], errors='coerce')
rotation_mask = (_min_per_raw > 5) & (_gp_raw >= 10) & (_pts_raw >= 1.0)

HIGH_MISSING_EXCLUDE = ['pick', 'Rec Rank', 'dunksmade/(dunksmade+dunksmiss)']
MISSINGNESS_THRESHOLD = 0.30
print("Excluded from imputation/model features because missingness exceeds "
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
df_other = df.drop(columns=cols_to_impute_mice + numeric_cols + categorical_cols + ['pid'], errors='ignore')

for col in df_numeric_impute.columns:
    df_numeric_impute[col].fillna(df_numeric_impute[col].median(), inplace=True)
for col in df_categorical.columns:
    df_categorical[col].fillna(df_categorical[col].mode()[0] if not df_categorical[col].mode().empty else 'Unknown', inplace=True)

le_dict = {}
for col in df_categorical.columns:
    le = LabelEncoder()
    df_categorical[col] = df_categorical[col].astype(str)
    df_categorical[col] = le.fit_transform(df_categorical[col])
    le_dict[col] = le

if len(cols_to_impute_mice) > 0:
    mice_imputer = IterativeImputer(max_iter=25, random_state=42)
    df_mice_imputed_array = mice_imputer.fit_transform(df_mice_impute)
    df_mice_imputed = pd.DataFrame(df_mice_imputed_array, columns=df_mice_impute.columns, index=df_mice_impute.index)
else:
    df_mice_imputed = pd.DataFrame()

for col in df_mice_imputed.columns:
    if col in ['rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)', 'dunksmade/(dunksmade+dunksmiss)']:
        df_mice_imputed[col] = df_mice_imputed[col].clip(0, 1)

for col in df_categorical.columns:
    df_categorical[col] = le_dict[col].inverse_transform(df_categorical[col].astype(int))

df_final = pd.concat([df_mice_imputed, df_numeric_impute, df_categorical, df_other.reset_index(drop=True)], axis=1)

if 'pid' in df.columns:
    df_final.insert(0, 'pid', df['pid'].values)
df = df_final
# Apply the pre-imputation eligibility mask.
df = df[rotation_mask.values]

per100_cols = {
    'pts': 'pts_per100',
    'treb': 'treb_per100',
    'ast': 'ast_per100',
    'stl': 'stl_per100',
    'blk': 'blk_per100',
    'oreb': 'oreb_per100',
    'dreb': 'dreb_per100'
}

for col, new_col in per100_cols.items():
    if col in df.columns:
        df[new_col] = df[col] / df['Min_per'] * 100
        numeric_cols.append(new_col)

original_percentage_cols = ['Min_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per',
                            'blk_per', 'stl_per', 'usg', 'eFG', 'TS_per']
for col in original_percentage_cols:
    if col in df.columns:
        numeric_cols.append(col)

per_game_cols = ['GP', 'FTM', 'FTA', 'twoPM', 'twoPA', 'TPM', 'TPA', 'ftr', 'porpag', 'adjoe', 'pfr', 'ast/tov', 'drtg',
                 'adjtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'Ortg']
for col in per_game_cols:
    if col in df.columns:
        numeric_cols.append(col)

print("pts_per100 分位数: ")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"{q*100}%: {df['pts_per100'].quantile(q):.2f}")
print("\n")

print(df['pts_per100'].describe())

numeric_cols = list(set([col for col in numeric_cols if col in df.columns]))

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols[:12]):
    clean_data = pd.to_numeric(df[col], errors='coerce').dropna()
    axes[i].hist(clean_data, bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

for i in range(len(numeric_cols[:12]), 12):
    axes[i].set_visible(False)
plt.tight_layout()
plt.show()

scaler = StandardScaler()
numeric_df = df[numeric_cols].copy()
scaled_array = scaler.fit_transform(numeric_df)
df_scaled = pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)

for col in numeric_cols:
    df[col] = df_scaled[col]

print("Mean of numeric columns after standardization:")
print(df[numeric_cols].mean().round(10))
print("-"*60)
print("\nStd of numeric columns after standardization:")
print(df[numeric_cols].std().round(10))

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

from sklearn.decomposition import PCA
pca = PCA(n_components=0.94)
numeric_df = df[numeric_cols].dropna()
pca_result = pca.fit_transform(numeric_df)

pca_df = pd.DataFrame(pca_result, index=numeric_df.index)
pca_df.columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]

print("Data structure after PCA dimension reduction:")
print(f"Original numeric features: {len(numeric_cols)}")
print(f"Principal components retained: {pca_result.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
print(f"\nShape of PCA-transformed data: {pca_df.shape}")
print("\nFirst 5 rows of PCA data:")
print(pca_df.head())

print(f"\nExplained variance ratio per component:")
for i, ratio in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.2f}%)")

print(f"\nCumulative variance explained:")
cumulative = np.cumsum(pca.explained_variance_ratio_)
for i in range(10, len(cumulative)):
    print(f"First {i+1} components: {cumulative[i]:.3f} ({cumulative[i]*100:.2f}%)")

import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_result = reducer.fit_transform(pca_df)
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'], index=pca_df.index)

print("Data structure after UMAP dimension reduction:")
print(f"Original PCA dimensions: {pca_df.shape[1]}")
print(f"UMAP reduced dimensions: {umap_df.shape[1]}")
print(f"\nShape of UMAP-transformed data: {umap_df.shape}")
print(f"\nFirst 10 rows of UMAP data:")
print(umap_df.head(10))

print(f"\nUMAP1 statistics:")
print(f"Min: {umap_df['UMAP1'].min():.4f}, Max: {umap_df['UMAP1'].max():.4f}, Mean: {umap_df['UMAP1'].mean():.4f}, Std: {umap_df['UMAP1'].std():.4f}")
print(f"\nUMAP2 statistics:")
print(f"Min: {umap_df['UMAP2'].min():.4f}, Max: {umap_df['UMAP2'].max():.4f}, Mean: {umap_df['UMAP2'].mean():.4f}, Std: {umap_df['UMAP2'].std():.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], s=1, alpha=0.5, c='steelblue')
plt.title('UMAP Projection of College Players (2009-2021)', fontsize=14)
plt.xlabel('UMAP Dim 1', fontsize=12)
plt.ylabel('UMAP Dim 2', fontsize=12)
plt.tight_layout()
plt.show()

if 'pts_per100' in df.columns:
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'],
                          c=df.loc[umap_df.index, 'pts_per100'],
                          s=1, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='Points per 100 possessions')
    plt.title('UMAP Projection Colored by Scoring Efficiency', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12.5)
    plt.ylabel('UMAP Dimension 2', fontsize=12.5)
    plt.tight_layout()
    plt.show()

df_with_umap = pd.concat([umap_df, df], axis=1)
all_original_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                     'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPPer',
                     'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe',
                     'pfr', 'ast/tov', 'drtg', 'adjtg', 'dporpag', 'stops',
                     'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb',
                     'treb', 'ast', 'stl', 'blk', 'pts', 'pts_per100', 'treb_per100', 'ast_per100',
                     'stl_per100', 'blk_per100', 'oreb_per100', 'dreb_per100', 'year',
                     'rimmade', 'rimmade+rimmiss', 'midmade', 'midmade+midmiss',
                     'rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
                     'dunksmade', 'dunksmiss+dunksmade']
all_original_cols = [col for col in all_original_cols if col in df_with_umap.columns]

corr_matrix = df_with_umap[['UMAP1', 'UMAP2']+all_original_cols].corr()
umap_corrs = corr_matrix[['UMAP1', 'UMAP2']].abs()
umap_corrs['max_abs_corr'] = umap_corrs[['UMAP1', 'UMAP2']].max(axis=1)

print("\nPearson correlation coefficients between UMAP dimensions and original features:")
high_corrs = umap_corrs[umap_corrs['max_abs_corr'] >= 0.25].sort_values('max_abs_corr', ascending=False)
print(high_corrs.round(4))

high_corr_cols = high_corrs.index.tolist()
filtered_df = df_with_umap[high_corr_cols]
print(f"\nColumns with |correlation| >= 0.25: {len(high_corr_cols)-2}")
print("\nTop 10 features most correlated with UMAP2:")
print(umap_corrs.sort_values('UMAP2', ascending=False).head(10).round(4))

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
cluster_labels = clusterer.fit_predict(umap_df)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise = sum(cluster_labels == -1)

print(f"\nHDBSCAN Clustering: {n_clusters} clusters, {noise} noise points")
print(f"Noise points: {noise} ({noise/len(cluster_labels)*100:.2f}%)")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], s=1, alpha=0.6, c=cluster_labels, cmap='tab10')
plt.title(f'HDBSCAN Clustering (min_cluster_size=15)\n{n_clusters} clusters, {noise} noise points', fontsize=14)
plt.xlabel('UMAP Dimension 1', fontsize=12.5)
plt.ylabel('UMAP Dimension 2', fontsize=12.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
clusterer.condensed_tree_.plot(select_clusters=True)
plt.title('HDBSCAN Condensed Tree')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hdbscan_stability.png'), dpi=300, bbox_inches='tight')

stability_res = {}
test_sizes = [5, 10, 15, 20, 30, 50]
for min_size in test_sizes:
    test_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=5, prediction_data=True)
    test_labels = test_clusterer.fit_predict(umap_df)
    n_clust = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    noise_pts = sum(test_labels == -1)
    noise_pct = noise_pts/len(test_labels)*100
    stability_res[min_size] = {'n_clust': n_clust, 'noise_pts': noise_pts, 'noise_pct': noise_pct}

print("\nStability check across min_cluster_size:")
for min_size, res in stability_res.items():
    print(f"min_size={min_size}: {res['n_clust']} clusters, {res['noise_pts']} noise ({res['noise_pct']:.2f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(stability_res.keys()), [v['n_clust'] for v in stability_res.values()], 'bo-')
ax1.set_xlabel('min_cluster_size')
ax1.set_ylabel('Number of clusters')
ax1.set_title('Cluster Count Stability')
ax2.plot(list(stability_res.keys()), [v['noise_pct'] for v in stability_res.values()], 'ro-')
ax2.set_xlabel('min_cluster_size')
ax2.set_ylabel('Noise proportion (%)')
ax2.set_title('Noise Proportion Stability')
plt.tight_layout()
plt.show()

# Fit the prespecified final HDBSCAN model.
optimal_min_size = 15
final_clusterer = hdbscan.HDBSCAN(min_cluster_size=optimal_min_size, min_samples=5, prediction_data=True)
final_labels = final_clusterer.fit_predict(umap_df)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=final_labels, s=2, alpha=0.7, cmap='tab10')
plt.title(f'HDBSCAN Final Clustering (min_cluster_size={optimal_min_size})\nClusters: {len(set(final_labels))-(1 if -1 in final_labels else 0)}, Noise: {sum(final_labels==-1)}')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hdbscan_final_clusters.png'), dpi=300, bbox_inches='tight')

cluster_dist = pd.Series(final_labels).value_counts().sort_index()
print("\nFinal Cluster Distribution:")
for cluster_id, count in cluster_dist.items():
    if cluster_id == -1:
        print(f"Noise points: {count}")
    else:
        print(f"Cluster {cluster_id}: {count} points")
