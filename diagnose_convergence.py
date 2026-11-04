"""Is the published COURT-NMF fit converged?

Re-runs fit_court() at the published setting and at longer iteration budgets,
and reports (a) how far W has travelled from its random simplex
initialisation, (b) the objective diagnostics, and (c) whether the resulting
memberships carry any style / external-label information.
"""
import importlib.util, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

UP = '/mnt/user-data/uploads/submission(V12, COURT_NMF_24th, Aug)'
spec = importlib.util.spec_from_file_location('cne', '/root/soft/proj/scripts/court_nmf_experiment.py')
cne = importlib.util.module_from_spec(spec); sys.modules['cne'] = cne; spec.loader.exec_module(cne)

df, X, style = cne.preprocess()
n, f = X.shape

# the exact initialisation fit_court uses
rng0 = np.random.default_rng(cne.SEED)
W_init = cne.simplex(rng0.random((n, cne.K)))

lab = pd.read_csv(f'{UP}/outputs/court_nmf_validation/court_nmf_labeled_memberships.csv')
lab = lab[lab['position'].isin(['guard', 'forward', 'center'])]
kidx = {(str(p).strip(), int(y)): i for i, (p, y) in enumerate(zip(df['pid'], df['year']))}
pairs = [(str(p).strip(), int(y)) for p, y in zip(lab['pid'], lab['year'])]
keep = [i for i, t in enumerate(pairs) if t in kidx]
rows = np.array([kidx[pairs[i]] for i in keep], dtype=int)
yv = lab['position'].to_numpy()[keep]
print('matched labels:', len(rows))

def clr(A, eps=1e-9):
    Ac = np.clip(A, eps, None); Ac = Ac / Ac.sum(1, keepdims=True)
    L = np.log(Ac); return L - L.mean(1, keepdims=True)

def mixture_r2(U):
    C = np.linalg.lstsq(U, X, rcond=None)[0]
    return float(1 - ((X - U @ C) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())

def oof(U):
    Z = clr(U[rows])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p = cross_val_predict(LogisticRegression(max_iter=5000), Z, yv, cv=cv)
    return accuracy_score(yv, p), f1_score(yv, p, average='macro')

print(f"{'iters':>6} {'|W-W0|/|W0|':>11} {'maxmemb':>8} {'entropy':>8} {'wMSE':>9} {'trajMSE':>9} {'mixR2':>7} {'OOFacc':>7} {'macroF1':>8}  time")
for iters in (80, 400, 2000, 8000):
    t = time.time()
    W, H, B, info = cne.fit_court(df, X, iterations=iters)
    d = info['diagnostics']
    drift = float(np.linalg.norm(W - W_init) / np.linalg.norm(W_init))
    a, f1 = oof(W)
    print(f"{iters:>6} {drift:>11.4f} {d['mean_max_membership']:>8.4f} {d['mean_entropy']:>8.4f} "
          f"{d['weighted_mse']:>9.5f} {d['trajectory_mse']:>9.5f} {mixture_r2(W):>7.4f} {a:>7.4f} {f1:>8.4f}  {time.time()-t:.0f}s", flush=True)
