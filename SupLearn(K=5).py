import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs', 'cluster_csvs')
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(data_dir, 'CollegeBasketballPlayers2009-2021（55columns）.csv')
df = pd.read_csv(file_path)
df = df.drop(['type', 'ht'], axis=1, errors='ignore')

# Apply the eligibility criteria to raw values before imputation so both
# analytic stages use the same player-season sample:
# Min_per > 5, GP >= 10, pts >= 1.0  ->  n = 45,980 player-seasons.
# Final analysis parameters:
#   RobustScaler + MinMaxScaler; 23 curated style features;
#   NMF(n_components=5, init='nndsvd', max_iter=500, random_state=42);
#   KMeans(n_clusters=5, random_state=42, n_init=25).
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
cols_to_impute_mice = [col for col in cols_to_impute_mice if col in df.columns]

numeric_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 
                'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per', 
                'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe', 
                'pfr', 'ast/tov', 'drtg', 'adrtg', 'dporpag', 'stops', 
                'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 
                'treb', 'ast', 'stl', 'blk', 'pts']
numeric_cols = [col for col in numeric_cols if col in df.columns and col not in cols_to_impute_mice]

categorical_cols = ['player_name', 'team', 'conf', 'yr']
categorical_cols = [col for col in categorical_cols if col in df.columns]

df_mice_impute = df[cols_to_impute_mice].copy()
df_numeric_impute = df[numeric_cols].copy()
df_categorical = df[categorical_cols].copy()
df_other = df.drop(columns=cols_to_impute_mice + numeric_cols + categorical_cols + ['pid'], errors='ignore')

for col in df_numeric_impute.columns:
    df_numeric_impute[col] = pd.to_numeric(df_numeric_impute[col], errors='coerce')
    df_numeric_impute[col].fillna(df_numeric_impute[col].median(), inplace=True)

for col in df_categorical.columns:
    df_categorical[col].fillna(df_categorical[col].mode()[0] if not df_categorical[col].mode().empty else 'Unknown', inplace=True)

le_dict = {}
for col in df_categorical.columns:
    le = LabelEncoder()
    df_categorical[col] = df_categorical[col].astype(str)
    df_categorical[col] = le.fit_transform(df_categorical[col])
    le_dict[col] = le

mice_imputer = IterativeImputer(max_iter=25, random_state=42)
df_mice_imputed_array = mice_imputer.fit_transform(df_mice_impute)
df_mice_imputed = pd.DataFrame(df_mice_imputed_array, columns=df_mice_impute.columns, index=df_mice_impute.index)

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

per100_cols = {'pts': 'pts_per100', 'treb': 'treb_per100', 'ast': 'ast_per100',
               'stl': 'stl_per100', 'blk': 'blk_per100', 'oreb': 'oreb_per100', 'dreb': 'dreb_per100'}
for col, new_col in per100_cols.items():
    if col in df.columns:
        df[new_col] = df[col] / df['Min_per'] * 100
        numeric_cols.append(new_col)

pts_per100_raw = df['pts_per100'].copy() if 'pts_per100' in df.columns else None

for col in ['Min_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FT_per', 'twoP_per', 
            'TP_per', 'blk_per', 'stl_per', 'usg', 'eFG', 'TS_per', 'GP', 'FTM', 'FTA', 
            'twoPM', 'twoPA', 'TPM', 'TPA', 'ftr', 'porpag', 'adjoe', 'pfr', 'ast/tov', 
            'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 
            'ogbpm', 'dgbpm', 'Ortg']:
    if col in df.columns:
        numeric_cols.append(col)
numeric_cols = list(set([col for col in numeric_cols if col in df.columns]))
scaler = RobustScaler()
df_scaled = df[numeric_cols].dropna().copy()
df_scaled[:] = scaler.fit_transform(df_scaled)

nmf_cols = ['pts_per100', 'treb_per100', 'ast_per100', 'stl_per100', 'blk_per100', 
            'oreb_per100', 'dreb_per100', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
            'AST_per', 'TO_per', 'FT_per', 'twoP_per', 'TP_per', 'blk_per', 'stl_per',
            'ftr', 'bpm', 'obpm', 'dbpm']
nmf_cols = [c for c in nmf_cols if c in df_scaled.columns]
minmax = MinMaxScaler()
df_nmf_input = pd.DataFrame(minmax.fit_transform(df_scaled[nmf_cols]),
                            columns=nmf_cols, index=df_scaled.index)
n_components_nmf = 5
feature_names = nmf_cols
nmf = NMF(n_components=n_components_nmf, random_state=42, init='nndsvd', max_iter=500)
nmf_weights = nmf.fit_transform(df_nmf_input)
nmf_components = nmf.components_
original_feature_contributions = pd.DataFrame(
    nmf_components.T,
    index=feature_names,
    columns=[f'Style_Dim_{i+1}' for i in range(n_components_nmf)]
)
for i in range(n_components_nmf):
    pos_mask = original_feature_contributions[f'Style_Dim_{i+1}'] > 0
    neg_mask = original_feature_contributions[f'Style_Dim_{i+1}'] < 0
    profile_df = pd.DataFrame({
        'Features': original_feature_contributions.index,
        f'Style_Dim_{i+1}': original_feature_contributions[f'Style_Dim_{i+1}'].values,
        'Sign': ['positive' if x > 0 else 'negative' if x < 0 else 'zero' for x in original_feature_contributions[f'Style_Dim_{i+1}'].values]
    })
    profile_df = profile_df.sort_values(f'Style_Dim_{i+1}', ascending=False)
    profile_df.to_csv(os.path.join(output_dir, f'Profile_on_NMF_Style_{i}.csv'), index=False)
    print(f"Profile_on_NMF_Style_{i}.csv saved")

player_style_space = pd.DataFrame(nmf_weights, index=df_scaled.index, 
                                  columns=[f'Style{i+1}' for i in range(n_components_nmf)])

for i in range(n_components_nmf):
    df.loc[df_scaled.index, f'Style{i+1}'] = player_style_space[f'Style{i+1}']

dominant_style = player_style_space.idxmax(axis=1)
df.loc[df_scaled.index, 'dominant_style'] = dominant_style

best_k = 5
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=25)
style_clusters = final_kmeans.fit_predict(player_style_space)
df.loc[df_scaled.index, 'style_cluster'] = style_clusters

global_silhouette = silhouette_score(player_style_space, style_clusters)

sil_samples = silhouette_samples(player_style_space, style_clusters)

umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
umap_result = umap_reducer.fit_transform(player_style_space.values)
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'], index=player_style_space.index)

umap_fig = plt.figure(figsize=(16, 7))
ax1 = umap_fig.add_subplot(1, 2, 1)
scatter1 = ax1.scatter(umap_df['UMAP1'], umap_df['UMAP2'], 
                      c=style_clusters, cmap='tab10', s=15, alpha=0.6)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Cluster')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_title('UMAP Projection - 5 Clusters', fontweight='bold')

ax2 = umap_fig.add_subplot(1, 2, 2)
if pts_per100_raw is not None:
    pts_aligned = pts_per100_raw.loc[umap_df.index]
    scatter2 = ax2.scatter(umap_df['UMAP1'], umap_df['UMAP2'], 
                          c=pts_aligned, cmap='Blues', s=15, alpha=0.6,
                          vmin=np.percentile(pts_aligned, 5), vmax=np.percentile(pts_aligned, 95))
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Points per 100')
ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_title('UMAP - Scoring Efficiency', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'NMF_UMAP_projection_direct_k5.png'), dpi=200, bbox_inches='tight')
plt.show()

np.savez(os.path.join(output_dir, 'nmf_style_results_direct_k5.npz'),
         weights=player_style_space.values,
         components=nmf_components,
         style_clusters=style_clusters,
         feature_names=np.array(feature_names))

report_cols = ['pts_per100', 'treb_per100', 'ast_per100', 'stl_per100', 'blk_per100',
               'oreb_per100', 'dreb_per100', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
               'AST_per', 'TO_per', 'FT_per', 'TP_per', 'blk_per', 'stl_per',
               'ftr', 'bpm', 'obpm', 'dbpm', 'Min_per', 'GP', 'Ortg', 'drtg']
report_cols = [c for c in report_cols if c in df.columns]

df_report = df[report_cols + ['style_cluster']].copy()
cluster_means = df_report.groupby('style_cluster')[report_cols].mean().reset_index()
cluster_means_melted = cluster_means.melt(id_vars='style_cluster', var_name='Feature', value_name='Mean')

n_features = len(report_cols)
n_clusters = 5
fig_height = max(10, n_features * 0.5)
fig_width = max(16, n_clusters * 2.5)

plt.figure(figsize=(fig_width, fig_height))
sns.barplot(data=cluster_means_melted, x='Feature', y='Mean', hue='style_cluster', palette='tab10')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.title('Cluster Mean Comparison - Grouped Barplot (All Features)', fontweight='bold')
plt.legend(title='Cluster', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_comparison_grouped_barplot_all.png'), dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(18, 24))

offensive_features = ['pts_per100', 'eFG', 'TS_per', 'usg', 'TP_per', 'FT_per', 'ftr', 'Ortg']
offensive_features = [c for c in offensive_features if c in report_cols]
offensive_melted = cluster_means.melt(id_vars='style_cluster', value_vars=offensive_features, 
                                       var_name='Feature', value_name='Mean')
sns.barplot(data=offensive_melted, x='Feature', y='Mean', hue='style_cluster', palette='tab10', ax=axes[0])
axes[0].set_title('Offensive Features', fontweight='bold', fontsize=14)
axes[0].set_xlabel('')
axes[0].set_ylabel('Mean Value')
axes[0].legend(title='Cluster', loc='upper right')
axes[0].tick_params(axis='x', rotation=30)

playmaking_features = ['ast_per100', 'AST_per', 'TO_per', 'ast/tov', 'stl_per100', 'stl_per']
playmaking_features = [c for c in playmaking_features if c in report_cols]
playmaking_melted = cluster_means.melt(id_vars='style_cluster', value_vars=playmaking_features, 
                                        var_name='Feature', value_name='Mean')
sns.barplot(data=playmaking_melted, x='Feature', y='Mean', hue='style_cluster', palette='tab10', ax=axes[1])
axes[1].set_title('Playmaking & Ball-Handling Features', fontweight='bold', fontsize=14)
axes[1].set_xlabel('')
axes[1].set_ylabel('Mean Value')
axes[1].legend(title='Cluster', loc='upper right')
axes[1].tick_params(axis='x', rotation=30)

defensive_features = ['blk_per100', 'blk_per', 'treb_per100', 'oreb_per100', 'dreb_per100', 
                      'ORB_per', 'DRB_per', 'drtg', 'dbpm']
defensive_features = [c for c in defensive_features if c in report_cols]
defensive_melted = cluster_means.melt(id_vars='style_cluster', value_vars=defensive_features, 
                                       var_name='Feature', value_name='Mean')
sns.barplot(data=defensive_melted, x='Feature', y='Mean', hue='style_cluster', palette='tab10', ax=axes[2])
axes[2].set_title('Defensive & Rebounding Features', fontweight='bold', fontsize=14)
axes[2].set_xlabel('')
axes[2].set_ylabel('Mean Value')
axes[2].legend(title='Cluster', loc='upper right')
axes[2].tick_params(axis='x', rotation=30)

usage_features = ['Min_per', 'GP', 'bpm', 'obpm']
usage_features = [c for c in usage_features if c in report_cols]
usage_melted = cluster_means.melt(id_vars='style_cluster', value_vars=usage_features, 
                                   var_name='Feature', value_name='Mean')
sns.barplot(data=usage_melted, x='Feature', y='Mean', hue='style_cluster', palette='tab10', ax=axes[3])
axes[3].set_title('Usage & Overall Impact Features', fontweight='bold', fontsize=14)
axes[3].set_xlabel('')
axes[3].set_ylabel('Mean Value')
axes[3].legend(title='Cluster', loc='upper right')
axes[3].tick_params(axis='x', rotation=30)

plt.suptitle('NMF K-Means (K=5) Cluster Profiles - Grouped Barplots by Feature Category', 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_comparison_grouped_barplot_categories.png'), dpi=300, bbox_inches='tight')
plt.show()

cluster_summary = df_report.groupby('style_cluster')[report_cols].agg(['mean', 'std']).round(3)
cluster_summary.to_csv(os.path.join(output_dir, 'cluster_mean_std_report.csv'))

mean_table = cluster_means.set_index('style_cluster').T
mean_table.to_csv(os.path.join(output_dir, 'cluster_mean_table.csv'))

print("Done")
