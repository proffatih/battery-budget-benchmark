"""R1 primary benchmark: leave-one-cell-out over the four CS2 cells.

For every budget x estimator x seed x fold the model is fitted once and then
evaluated in each executable representation (reference float64, float32,
int8, and for dense models int8 with the standardiser folded). Per-cell
metrics, per-fold storage and operation counts and convergence information are
written out; seed-0 predictions are kept for the cell-grouped bootstrap.
"""
import os, sys, itertools, json
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import r1_lib as L

SEEDS = [0] if os.environ.get("ONESEED") else [0, 1, 2, 3, 4]
SCREEN = os.environ.get("SCREEN", "v1")
TAG = "" if SCREEN == "v1" else f"_{SCREEN}"

def reps_for(name, m, sc, Xtr):
    if name in ("ridge", "mlp8", "mlp32x16"):
        return {"fp32": L.DenseFP32Folded(m, sc), "int8": L.DenseInt8(m, sc, Xtr),
                "int8_folded": L.DenseInt8(m, sc, Xtr, fold_std=True)}
    if name == "svr":
        return {"fp32": L.SVRRep(m, sc, "fp32"), "int16": L.SVRRep(m, sc, "int16"), "int8": L.SVRRep(m, sc, "int8"),
                "int8all": L.SVRRep(m, sc, "int8all")}
    return {"fp32": L.TreeRep(m, sc, Xtr, "fp32"), "int8": L.TreeRep(m, sc, Xtr, "int8")}

def run_budget(args):
    w, n = args
    meta = L.screen(L.load_all(), SCREEN)
    cs2 = meta[meta.cell.isin(L.MAIN_CELLS)].reset_index(drop=True)
    X = L.build_matrix(cs2, w, n); y = cs2.soh.values; cells = cs2.cell.values
    rows, costs, preds = [], [], {}
    for name in L.MODELS + ["ridge_dur1"]:
        seeds = SEEDS if name in L.STOCHASTIC else [0]
        for seed in seeds:
            P = {}
            for c in L.MAIN_CELLS:
                tr, te = cells != c, cells == c
                if name == "ridge_dur1":
                    sc = StandardScaler().fit(X[tr][:, :1])
                    m = Ridge(alpha=1.0).fit(sc.transform(X[tr][:, :1]), y[tr]); conv = dict(converged=True)
                    reps = {"fp32": L.DurFP32(m, sc), "fixed16": L.DurInt16(m, sc)}
                    P.setdefault("ref", np.full(len(y), np.nan))[te] = m.predict(sc.transform(X[te][:, :1]))
                    for r, obj in reps.items():
                        P.setdefault(r, np.full(len(y), np.nan))[te] = obj.predict(X[te][:, 0])
                        costs.append(dict(window=w, n=n, model=name, seed=seed, fold=c, rep=r,
                                          **{k: v for k, v in obj.storage().items() if k != "detail"},
                                          detail=obj.storage()["detail"], ops=json.dumps(obj.ops()),
                                          ops_total=L.total(obj.ops()), converged=True))
                    continue
                sc = StandardScaler().fit(X[tr])
                m, conv = L.fit(name, sc.transform(X[tr]), y[tr], seed)
                P.setdefault("ref", np.full(len(y), np.nan))[te] = m.predict(sc.transform(X[te]))
                for r, obj in reps_for(name, m, sc, X[tr]).items():
                    if isinstance(obj, L.TreeRep):
                        p, ncmp = obj.predict(X[te], return_cmp=True); ops = obj.ops(float(ncmp.mean()))
                        ops["cmp_max"] = int(ncmp.max())
                    else:
                        p = obj.predict(X[te]); ops = obj.ops()
                    P.setdefault(r, np.full(len(y), np.nan))[te] = p
                    st = obj.storage()
                    extra = {}
                    if name == "svr": extra["n_sv"] = int(m.support_vectors_.shape[0])
                    costs.append(dict(window=w, n=n, model=name, seed=seed, fold=c, rep=r,
                                      **{k: v for k, v in st.items() if k != "detail"},
                                      detail=st["detail"], ops=json.dumps(ops), ops_total=L.total(ops),
                                      **conv, **extra))
            for r, p in P.items():
                for c, d in L.cell_metrics(p, y, cells).items():
                    s = (cells == c) & (y >= 0.8)
                    pc = np.clip(p, L.LO_CLIP, L.HI_CLIP)
                    rows.append(dict(window=w, n=n, model=name, seed=seed, rep=r, cell=c, **d,
                                     sse_clip=float(np.sum((pc[cells == c] - y[cells == c]) ** 2)),
                                     n_out=int(np.sum((p[cells == c] < L.LO_CLIP) | (p[cells == c] > L.HI_CLIP))),
                                     sse_ge80=float(np.sum((p[s] - y[s]) ** 2)), n_ge80=int(s.sum())))
                if seed == 0:
                    preds[f"{w}|{n}|{name}|{r}"] = p
    preds[f"{w}|{n}|y"] = y; preds[f"{w}|{n}|cells"] = cells.astype("U8")
    preds[f"{w}|{n}|cycle"] = cs2.cycle.values
    print("done", w, n, flush=True)
    return rows, costs, preds

if __name__ == "__main__":
    budgets = list(itertools.product(L.WINDOWS, L.NS))
    with Pool(12) as pool:
        out = pool.map(run_budget, budgets)
    rows = [r for o in out for r in o[0]]; costs = [r for o in out for r in o[1]]
    preds = {}
    for o in out: preds.update(o[2])
    pd.DataFrame(rows).to_csv(os.path.join(L.RES, f"r1_loco_cells{TAG}.csv"), index=False)
    pd.DataFrame(costs).to_csv(os.path.join(L.RES, f"r1_loco_costs{TAG}.csv"), index=False)
    np.savez_compressed(os.path.join(L.RUN_OUT, f"r1_loco_preds{TAG}.npz"), **preds)
    print("written", len(rows), "cell rows,", len(costs), "cost rows")
