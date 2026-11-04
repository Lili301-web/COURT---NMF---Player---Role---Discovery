"""
Does a method's soft membership vector carry more information about an
external, never-seen label than a competing soft method's?

Every soft method in the benchmark is scored on the SAME 195 independently
labelled player-seasons with the SAME protocol already used for COURT-NMF in
Section 6: CLR-transform the membership composition, then 5-fold stratified
multinomial logistic regression, out-of-fold accuracy and macro-F1, plus a
PERMANOVA R^2 on Aitchison distance.  This is an external criterion -- no
method optimises it -- so it compares mixtures on the thing the paper claims
they capture, rather than on hard-cluster geometry.

Also reports mixture reconstruction R^2: how much of the variance in the
shared 23-feature style space is recovered by rebuilding each observation as
the convex combination of soft prototypes implied by its own memberships.
"""
import importlib.util, sys
import numpy as np, pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import pdist, squareform

SEED = 42
UP = '/mnt/user-data/uploads/submission(V12, COURT_NMF_24th, Aug)'

spec = importlib.util.spec_from_file_location('acb', '/root/soft/proj/scripts/adaptive_clustering_benchmark.py')
acb = importlib.util.module_from_spec(spec); sys.modules['acb'] = acb; spec.loader.exec_module(acb)

X = np.load('/root/soft/X.npy')
keys = pd.read_csv('/root/soft/keys.csv')
n = len(X)

def rownorm(A):
    s = A.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return A / s

W = NMF(n_components=5, random_state=SEED, init='nndsvd', max_iter=500).fit_transform(X)
M = {
    'NMF + $K$-means':      np.eye(5)[KMeans(n_clusters=5, random_state=SEED, n_init=25).fit_predict(W)],
    'Adaptive NMF mixture': rownorm(W),
    'Gaussian mixture':     GaussianMixture(n_components=5, covariance_type='full',
                                            random_state=SEED, n_init=5).fit(W).predict_proba(W),
    'Fuzzy $C$-means':      acb.fuzzy_cmeans(W, k=5, m=2.0)[0],
}
court = pd.read_csv(f'{UP}/outputs/court_nmf/COURT-NMF_memberships.csv')
assert (court['pid'].to_numpy() == keys['pid'].to_numpy()).all()
M['COURT-NMF'] = rownorm(court[[f'role_{i}' for i in range(1, 6)]].to_numpy(float))

# ---- mixture reconstruction R^2 in the shared style space -------------------
def mixture_r2(U):
    C = np.linalg.lstsq(U, X, rcond=None)[0]        # best prototypes for these memberships
    resid = ((X - U @ C) ** 2).sum()
    total = ((X - X.mean(axis=0)) ** 2).sum()
    return float(1 - resid / total)

# ---- external label ---------------------------------------------------------
lab = pd.read_csv(f'{UP}/outputs/court_nmf_validation/court_nmf_labeled_memberships.csv')
lab = lab[lab['position'].isin(['guard', 'forward', 'center'])].copy()
key_idx = {(p, y): i for i, (p, y) in enumerate(zip(keys['pid'], keys['year']))}
rows = [key_idx.get((p, y)) for p, y in zip(lab['pid'], lab['year'])]
ok = [i for i, r in enumerate(rows) if r is not None]
lab = lab.iloc[ok].reset_index(drop=True)
rows = np.array([rows[i] for i in ok], dtype=int)
y = lab['position'].to_numpy()
print(f"matched labelled player-seasons: {len(rows)}")

def clr(A, eps=1e-9):
    Ac = np.clip(A, eps, None); Ac = Ac / Ac.sum(axis=1, keepdims=True)
    L = np.log(Ac); return L - L.mean(axis=1, keepdims=True)

def permanova_r2(D, labels, n_perm=9999, seed=SEED):
    labels = np.asarray(labels); N = len(labels)
    D2 = D ** 2; total = D2.sum() / (2 * N)
    def within(lb):
        w = 0.0
        for g in np.unique(lb):
            m = lb == g; ng = m.sum()
            if ng > 1: w += D2[np.ix_(m, m)].sum() / (2 * ng)
        return w
    Wn = within(labels); r2 = 1 - Wn / total
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        if 1 - within(rng.permutation(labels)) / total >= r2: cnt += 1
    return float(r2), (cnt + 1) / (n_perm + 1)

out = []
for name, U in M.items():
    Z = clr(U[rows])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(LogisticRegression(max_iter=5000, C=1.0), Z, y, cv=cv)
    r2, p = permanova_r2(squareform(pdist(Z)), y, n_perm=1999)
    out.append({'method': name,
                'mixture_R2': mixture_r2(U),
                'oof_accuracy': accuracy_score(y, pred),
                'macro_F1': f1_score(y, pred, average='macro'),
                'permanova_R2': r2, 'permanova_p': p})

base = pd.Series(y).value_counts(normalize=True).max()
df = pd.DataFrame(out)
pd.set_option('display.width', 220, 'display.float_format', lambda v: f'{v:.4f}')
print(f"\nmajority-class baseline accuracy = {base:.4f}\n")
print(df.to_string(index=False))
df.to_csv('/root/soft/external_soft_metrics.csv', index=False)
