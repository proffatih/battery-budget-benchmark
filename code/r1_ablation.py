"""R1: eight-feature set without the current features (I_mean, I_std).

Under constant-current charging the two current features are constant within a
design (coefficient of variation ~0.02 %) but differ between designs, because
the absolute charge current scales with the nominal capacity. They therefore
carry no ageing information within a design and act as a design identifier
across designs. This script repeats, for the eight-feature set, the CS2 LOCO
over all budgets (seed 0, reference models), selects each estimator's budget on
CS2 alone, and evaluates within-CX2 and mixed-design LOCO at that budget.
"""
import os, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
import r1_lib as L, r1_transfer as T

def loco_budget(args):
    w, n = args
    meta = L.screen(L.load_all(), "v1").reset_index(drop=True)
    meta = meta[meta.cell.isin(L.MAIN_CELLS)].reset_index(drop=True)
    X = T.featset(L.build_matrix(meta, w, n), meta.nominal.values, "noI")
    y, cells = meta.soh.values, meta.cell.values
    rows = []
    for name in L.MODELS:
        P = np.full(len(y), np.nan)
        for c in L.MAIN_CELLS:
            tr, te = cells != c, cells == c
            P[te], _, _ = T.fit_predict(name, X[tr], y[tr], X[te], 0)
        for c, d in L.cell_metrics(P, y, cells).items():
            rows.append(dict(featset="noI", scenario="cs2_loco", window=w, n=n, model=name, cell=c, **d))
    return rows

def scen_at(args):
    name, w, n = args
    meta = L.screen(L.load_all(), "v1").reset_index(drop=True)
    meta = meta[meta.cell.isin(L.MAIN_CELLS + L.CX2_CELLS)].reset_index(drop=True)
    X = T.featset(L.build_matrix(meta, w, n), meta.nominal.values, "noI")
    y, cells = meta.soh.values, meta.cell.values
    rows = []
    for scen, pool_cells in [("cx2_loco", L.CX2_CELLS), ("mixed_loco", L.MAIN_CELLS + L.CX2_CELLS)]:
        P = np.full(len(y), np.nan); sel = np.isin(cells, pool_cells)
        for c in pool_cells:
            tr = sel & (cells != c); te = cells == c
            P[te], _, _ = T.fit_predict(name, X[tr], y[tr], X[te], 0)
        rows += T.metrics(P[sel], y[sel], cells[sel], dict(featset="noI", scenario=scen, window=w, n=n,
                                                            model=name, seed=0))
    return rows

if __name__ == "__main__":
    with Pool(6) as pool:
        A = pool.map(loco_budget, list(itertools.product(L.WINDOWS, L.NS)))
    a = pd.DataFrame([r for o in A for r in o])
    a.to_csv(os.path.join(L.RES, "r1_ablation_noI_loco.csv"), index=False)
    pooled = a.groupby(["window", "n", "model"]).apply(
        lambda x: np.sqrt(x.sse.sum() / x.ncyc.sum())).rename("rmse").reset_index()
    best = pooled.loc[pooled.groupby("model").rmse.idxmin()]
    with Pool(6) as pool:
        B = pool.map(scen_at, [(r.model, r.window, int(r.n)) for r in best.itertuples()])
    pd.DataFrame([r for o in B for r in o]).to_csv(os.path.join(L.RES, "r1_ablation_noI_scen.csv"), index=False)
    print(best.to_string(index=False)); print("written")
