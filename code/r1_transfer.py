"""R1 transfer analyses.

Scenarios (all cell-wise, reference float64 models, raw and clipped errors):
  from_cs2    : fit on CS2_35-38, evaluate every other cell; the test cells
                separate protocol shift (CS2_33/34, 0.5C discharge), design shift
                at the training discharge rate (CX2_34/36/37/38, 1C) and combined
                design + protocol shift (CX2_16 ~0.41C, CX2_33/35 0.5C).
  cx2_loco    : leave-one-cell-out within CX2_16/34/36/37 (the v1 CX2 set).
  mixed_loco  : leave-one-cell-out over the eight v1 cells (both designs).
  from_cx2    : fit on CX2_16/34/36/37, evaluate CS2_35-38.
Feature sets:
  raw   : the ten v1 features.
  norm  : charge and current features divided by the nominal capacity
          (dQ, I_mean, I_std, dQ/dV mean and max -> per Ah of rating).
  noI   : raw without I_mean and I_std (degenerate under CC charging).
"""
import os, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import r1_lib as L

SEEDS = [0, 1, 2, 3, 4]
NORM_IDX = [1, 2, 3, 8, 9]

def featset(X, nominal, kind):
    if kind == "raw": return X
    if kind == "norm":
        Xn = X.copy(); Xn[:, NORM_IDX] = X[:, NORM_IDX] / nominal[:, None]; return Xn
    if kind == "noI": return np.delete(X, [2, 3], axis=1)
    raise KeyError(kind)

def metrics(p, y, cells, extra):
    rows = []
    pc = np.clip(p, L.LO_CLIP, L.HI_CLIP)
    for c in pd.unique(cells):
        s = cells == c
        rows.append(dict(**extra, cell=c, ncyc=int(s.sum()),
                         sse_raw=float(np.sum((p[s] - y[s]) ** 2)), sse_clip=float(np.sum((pc[s] - y[s]) ** 2)),
                         rmse_raw=L.rmse(p[s], y[s]), rmse_clip=L.rmse(pc[s], y[s]),
                         bias_raw=float(np.mean(p[s] - y[s])), bias_clip=float(np.mean(pc[s] - y[s])),
                         n_out=int(np.sum((p[s] < L.LO_CLIP) | (p[s] > L.HI_CLIP))),
                         pmin=float(p[s].min()), pmax=float(p[s].max())))
    return rows

def fit_predict(name, Xtr, ytr, Xte, seed):
    if name == "ridge_dur1":
        sc = StandardScaler().fit(Xtr[:, :1])
        return Ridge(alpha=1.0).fit(sc.transform(Xtr[:, :1]), ytr).predict(sc.transform(Xte[:, :1])), sc, None
    sc = StandardScaler().fit(Xtr)
    m, conv = L.fit(name, sc.transform(Xtr), ytr, seed)
    return m.predict(sc.transform(Xte)), sc, conv

def run(args):
    w, n = args
    meta = L.screen(L.load_all(), "v1").reset_index(drop=True)
    X0 = L.build_matrix(meta, w, n); y = meta.soh.values; cells = meta.cell.values
    nominal = meta.nominal.values
    rows, diag = [], []
    import time; t0 = time.time()
    for fs in ["raw", "norm", "noI"]:
        print(" ", w, n, fs, f"{time.time()-t0:.0f}s", flush=True)
        X = featset(X0, nominal, fs)
        for name in L.MODELS + ["ridge_dur1"]:
            seeds = SEEDS if name in L.STOCHASTIC else [0]
            for seed in seeds:
                base = dict(window=w, n=n, featset=fs, model=name, seed=seed)
                # from_cs2
                tr = np.isin(cells, L.MAIN_CELLS); te = ~tr
                p, sc, conv = fit_predict(name, X[tr], y[tr], X[te], seed)
                rows += metrics(p, y[te], cells[te], dict(base, scenario="from_cs2",
                                converged=None if conv is None else conv["converged"]))
                if seed == 0 and name != "ridge_dur1":
                    for c in L.CX2_CELLS + ["CS2_33", "CS2_34", "CX2_33", "CX2_35", "CX2_38"]:
                        s = cells == c
                        z = sc.transform(X[s])
                        diag.append(dict(window=w, n=n, featset=fs, model=name, cell=c,
                                         **{f"zmax_{j}": float(np.abs(z[:, j]).max()) for j in range(X.shape[1])},
                                         **{f"sd_{j}": float(sc.scale_[j]) for j in range(X.shape[1])},
                                         pmin=float(p[cells[te] == c].min()), pmax=float(p[cells[te] == c].max())))
                # from_cx2
                tr = np.isin(cells, L.CX2_CELLS); te = np.isin(cells, L.MAIN_CELLS)
                p, _, _ = fit_predict(name, X[tr], y[tr], X[te], seed)
                rows += metrics(p, y[te], cells[te], dict(base, scenario="from_cx2"))
                # within-CX2 and mixed-design LOCO
                if fs != "raw" or seed != 0:
                    continue                      # within-CX2 / mixed LOCO: raw features, seed 0
                for scen, pool_cells in [("cx2_loco", L.CX2_CELLS), ("mixed_loco", L.MAIN_CELLS + L.CX2_CELLS)]:
                    P = np.full(len(y), np.nan); sel = np.isin(cells, pool_cells)
                    for c in pool_cells:
                        tr = sel & (cells != c); te = cells == c
                        P[te], _, _ = fit_predict(name, X[tr], y[tr], X[te], seed)
                    rows += metrics(P[sel], y[sel], cells[sel], dict(base, scenario=scen))
    print("done", w, n, flush=True)
    return rows, diag

if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 6))) as pool:
        out = pool.map(run, list(itertools.product(L.WINDOWS, L.NS)))
    pd.DataFrame([r for o in out for r in o[0]]).to_csv(os.path.join(L.RES, "r1_transfer_cells.csv"), index=False)
    pd.DataFrame([r for o in out for r in o[1]]).to_csv(os.path.join(L.RES, "r1_transfer_diag.csv"), index=False)
    print("written")
