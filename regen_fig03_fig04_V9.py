"""Regenerate the Stage-1 UMAP scatterplots from the saved embedding."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)
emb = pd.read_csv(os.path.join(BASE, 'outputs', 'stage1_rerun', 'umap_embedding_labels.csv'))

# rebuild just enough to recover raw pts_per100
file_path = os.path.join(BASE, 'data', 'CollegeBasketballPlayers2009-2021（55columns）.csv')
df = pd.read_csv(file_path, low_memory=False)
_min_per_raw = pd.to_numeric(df['Min_per'], errors='coerce')
_gp_raw = pd.to_numeric(df['GP'], errors='coerce')
_pts_raw = pd.to_numeric(df['pts'], errors='coerce')
mask = (_min_per_raw > 5) & (_gp_raw >= 10) & (_pts_raw >= 1.0)
min_per = pd.to_numeric(df.loc[mask, 'Min_per'], errors='coerce').values
pts = pd.to_numeric(df.loc[mask, 'pts'], errors='coerce').values
pts_per100 = pts / min_per * 100
assert len(pts_per100) == len(emb), (len(pts_per100), len(emb))

# fig03: plain scatter
plt.figure(figsize=(12, 8))
plt.scatter(emb['UMAP1'], emb['UMAP2'], s=1, alpha=0.5, c='steelblue')
plt.title('UMAP Projection of College Players (2009-2021)', fontsize=14)
plt.xlabel('UMAP Dim 1', fontsize=12)
plt.ylabel('UMAP Dim 2', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig03_umap_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
print('fig03 saved')

# fig04: colored by scoring volume
plt.figure(figsize=(12, 8))
sc = plt.scatter(emb['UMAP1'], emb['UMAP2'], c=pts_per100, s=1, alpha=0.5, cmap='viridis')
plt.colorbar(sc, label='Points per 100 possessions')
plt.title('UMAP Projection Colored by Scoring Volume', fontsize=14)
plt.xlabel('UMAP Dimension 1', fontsize=12.5)
plt.ylabel('UMAP Dimension 2', fontsize=12.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig04_umap_colored.png'), dpi=300, bbox_inches='tight')
plt.close()
print('fig04 saved')
