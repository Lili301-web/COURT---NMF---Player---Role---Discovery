"""
Stage-2 scheme comparison for the methodological ablation.

Compares:
1. PCA-shift baseline: Robust-scaled 23 style features -> PCA(0.94) ->
   global shift to non-negativity -> NMF(5) -> KMeans(5).
2. Proposed direct NMF: Robust-scaled 23 style features -> MinMaxScaler ->
   NMF(5) -> KMeans(5).

The script writes a compact CSV table to outputs/scheme_comparison/.
"""
import os
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTDIR = os.path.join(BASE, 'outputs', 'scheme_comparison')
os.makedirs(OUTDIR, exist_ok=True)
DATA = os.path.join(BASE, 'data', 'CollegeBasketballPlayers2009-2021（55columns）.csv')


def build_stage2_matrix():
    df = pd.read_csv(DATA, low_memory=False)
    df = df.drop(['type', 'ht'], axis=1, errors='ignore')

    rotation_mask = (
        (pd.to_numeric(df['Min_per'], errors='coerce') > 5)
        & (pd.to_numeric(df['GP'], errors='coerce') >= 10)
        & (pd.to_numeric(df['pts'], errors='coerce') >= 1.0)
    )

    mice_cols = ['rimmade', 'rimmade+rimmiss', 'midmade', 'midmade+midmiss',
                 'rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)',
                 'dunksmade', 'dunksmiss+dunksmade']
    mice_cols = [c for c in mice_cols if c in df.columns]

    numeric_cols = ['GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per',
                    'AST_per', 'TO_per', 'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA',
                    'twoP_per', 'TPM', 'TPA', 'TP_per', 'blk_per', 'stl_per', 'ftr',
                    'porpag', 'adjoe', 'pfr', 'ast/tov', 'drtg', 'adrtg', 'dporpag',
                    'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp', 'ogbpm', 'dgbpm',
                    'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts']
    numeric_cols = [c for c in numeric_cols if c in df.columns and c not in mice_cols]
    cat_cols = [c for c in ['player_name', 'team', 'conf', 'yr'] if c in df.columns]

    df_mice = df[mice_cols].apply(pd.to_numeric, errors='coerce')
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_cat = df[cat_cols].copy()
    df_other = df.drop(columns=mice_cols + numeric_cols + cat_cols + ['pid'],
                       errors='ignore')

    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())
    for col in df_cat.columns:
        mode = df_cat[col].mode()
        df_cat[col] = df_cat[col].fillna(mode[0] if not mode.empty else 'Unknown')
        df_cat[col] = LabelEncoder().fit_transform(df_cat[col].astype(str))

    imputed = pd.DataFrame(
        IterativeImputer(max_iter=25, random_state=42).fit_transform(df_mice),
        columns=df_mice.columns,
        index=df_mice.index,
    )
    for col in ['rimmade/(rimmade+rimmiss)', 'midmade/(midmade+midmiss)']:
        if col in imputed.columns:
            imputed[col] = imputed[col].clip(0, 1)

    df_final = pd.concat([imputed, df_numeric, df_cat, df_other.reset_index(drop=True)], axis=1)
    if 'pid' in df.columns:
        df_final.insert(0, 'pid', df['pid'].values)
    df = df_final[rotation_mask.values].copy()

    for col, new_col in {'pts': 'pts_per100', 'treb': 'treb_per100',
                         'ast': 'ast_per100', 'stl': 'stl_per100',
                         'blk': 'blk_per100', 'oreb': 'oreb_per100',
                         'dreb': 'dreb_per100'}.items():
        if col in df.columns:
            df[new_col] = df[col] / df['Min_per'] * 100
            numeric_cols.append(new_col)

    for col in ['Min_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FT_per',
                'twoP_per', 'TP_per', 'blk_per', 'stl_per', 'usg', 'eFG',
                'TS_per', 'GP', 'FTM', 'FTA', 'twoPM', 'twoPA', 'TPM', 'TPA',
                'ftr', 'porpag', 'adjoe', 'pfr', 'ast/tov', 'drtg', 'adrtg',
                'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp',
                'ogbpm', 'dgbpm', 'Ortg']:
        if col in df.columns:
            numeric_cols.append(col)
    numeric_cols = list({c for c in numeric_cols if c in df.columns})

    scaled = df[numeric_cols].dropna().copy()
    scaled[:] = RobustScaler().fit_transform(scaled)
    style_cols = ['pts_per100', 'treb_per100', 'ast_per100', 'stl_per100',
                  'blk_per100', 'oreb_per100', 'dreb_per100', 'usg', 'eFG',
                  'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FT_per',
                  'twoP_per', 'TP_per', 'blk_per', 'stl_per', 'ftr', 'bpm',
                  'obpm', 'dbpm']
    style_cols = [c for c in style_cols if c in scaled.columns]
    return df.loc[scaled.index].copy(), scaled[style_cols].copy(), style_cols


def chi_square_for_positions(df, labels):
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
    tmp = df[['player_name']].copy()
    tmp['cluster'] = labels
    tmp['position'] = tmp['player_name'].map(dict(position_data))
    ct = pd.crosstab(tmp[tmp['position'].notna()]['cluster'], tmp[tmp['position'].notna()]['position'])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan
    chi2, p_value, _, _ = chi2_contingency(ct)
    return chi2, p_value


def fit_pca_shift_baseline(X):
    shifted = X - X.min().min() + 0.01
    pca_scores = PCA(n_components=0.94, random_state=42).fit_transform(shifted)
    pca_scores = pca_scores - pca_scores.min() + 0.01
    W = NMF(n_components=5, random_state=42, init='nndsvd', max_iter=500).fit_transform(pca_scores)
    labels = KMeans(n_clusters=5, random_state=42, n_init=25).fit_predict(W)
    return W, labels


def fit_direct_nmf(X):
    direct = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)
    W = NMF(n_components=5, random_state=42, init='nndsvd', max_iter=500).fit_transform(direct)
    labels = KMeans(n_clusters=5, random_state=42, n_init=25).fit_predict(W)
    return W, labels


def main():
    start = time.time()
    df, X, _ = build_stage2_matrix()
    rows = []
    for name, fitter, interp in [
        ('PCA-shift baseline', fit_pca_shift_baseline,
         'Indirect; components require projection back and may mix positive and negative metric effects.'),
        ('Proposed direct NMF', fit_direct_nmf,
         'Direct; component weights correspond to the original basketball metrics.'),
    ]:
        W, labels = fitter(X)
        chi2, p_value = chi_square_for_positions(df, labels)
        rows.append({
            'scheme': name,
            'k5_silhouette': round(float(silhouette_score(W, labels)), 4),
            'chi_square': round(float(chi2), 2),
            'p_value': f'{p_value:.3e}',
            'loading_interpretability': interp,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, 'scheme_comparison.csv'), index=False)
    print(out.to_string(index=False))
    print(f'Done in {time.time() - start:.0f}s')


if __name__ == '__main__':
    main()
