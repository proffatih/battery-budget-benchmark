"""Cell-grouped (hierarchical) bootstrap for the LOCO contrasts.

Level 1 resamples the four held-out CS2 cells with replacement; level 2
resamples cycles inside each drawn cell with a moving-block bootstrap (block of
B_LEN consecutive cycles, preserving within-cell autocorrelation). The same
draws are applied to every configuration in a contrast, so differences and
ratios are paired. With four cells the cell level has only 35 distinct
multisets; the intervals are therefore coarse and are reported as such.
"""
import os, json
import numpy as np, pandas as pd
import r1_lib as L

B = 4000
B_LEN = 25
RNG = np.random.default_rng(20261110)

P = np.load(os.path.join(L.RUN_OUT, "r1_loco_preds.npz"))

def sqerr(w, n, model, rep="ref"):
    y = P[f"{w}|{n}|y"]; p = P[f"{w}|{n}|{model}|{rep}"]
    return (p - y) ** 2, P[f"{w}|{n}|cells"], y

def draws(cells):
    """Pre-draw index sets so that every configuration shares them."""
    groups = {c: np.where(cells == c)[0] for c in L.MAIN_CELLS}
    out = []
    for _ in range(B):
        idx = []
        for c in RNG.choice(L.MAIN_CELLS, size=4, replace=True):
            g = groups[c]; m = len(g)
            starts = RNG.integers(0, m - B_LEN + 1, size=int(np.ceil(m / B_LEN)))
            ii = (starts[:, None] + np.arange(B_LEN)[None]).ravel()[:m]
            idx.append(g[ii])
        out.append(np.concatenate(idx))
    return out

def boot_rmse(se, D):
    return np.array([np.sqrt(se[d].mean()) for d in D])

def ci(a, q=(2.5, 97.5)): return [float(np.percentile(a, q[0])), float(np.percentile(a, q[1]))]

def main():
    res = {}
    # cycles are identical across budgets (same screened set), so one draw set serves all
    _, cells, y = sqerr("wide", 4, "ridge")
    D = draws(cells)
    def stat(w, n, m, rep="ref", mask=None):
        se, _, yy = sqerr(w, n, m, rep)
        return boot_rmse(se, D)
    models = L.MODELS + ["ridge_dur1"]
    # pooled RMSE CIs for every (window, N, model) in the reference representation
    tab = []
    for w in L.WINDOWS:
        for n in L.NS:
            for m in models:
                b = stat(w, n, m)
                se, _, _ = sqerr(w, n, m)
                tab.append(dict(window=w, n=n, model=m, rmse=float(np.sqrt(se.mean())),
                                lo=ci(b)[0], hi=ci(b)[1]))
    pd.DataFrame(tab).to_csv(os.path.join(L.RES, "r1_boot_rmse.csv"), index=False)
    # contrasts
    con = []
    for m in models:
        for w in L.WINDOWS:
            a, b = stat(w, 4, m), stat(w, 32, m)
            d = a - b
            con.append(dict(contrast="N4_minus_N32", model=m, window=w, est=float(np.median(d)),
                            point=float(next(t["rmse"] for t in tab if t["window"] == w and t["n"] == 4 and t["model"] == m)
                                        - next(t["rmse"] for t in tab if t["window"] == w and t["n"] == 32 and t["model"] == m)),
                            lo=ci(d)[0], hi=ci(d)[1]))
        for n in L.NS:
            r = stat("narrow", n, m) / stat("wide", n, m)
            pt = (next(t["rmse"] for t in tab if t["window"] == "narrow" and t["n"] == n and t["model"] == m)
                  / next(t["rmse"] for t in tab if t["window"] == "wide" and t["n"] == n and t["model"] == m))
            con.append(dict(contrast="narrow_over_wide", model=m, window=f"N{n}", est=float(np.median(r)),
                            point=float(pt), lo=ci(r)[0], hi=ci(r)[1]))
    for w in L.WINDOWS:
        for n in L.NS:
            base = stat(w, n, "ridge")
            for m in ["hgb", "svr", "rf", "mlp8", "mlp32x16", "ridge_dur1"]:
                d = stat(w, n, m) - base
                con.append(dict(contrast=f"{m}_minus_ridge", model=m, window=f"{w}/N{n}", est=float(np.median(d)),
                                point=float(d.mean()), lo=ci(d)[0], hi=ci(d)[1]))
    for m, reps in [("ridge", ["fp32", "int8"]), ("mlp8", ["fp32", "int8"]), ("mlp32x16", ["fp32", "int8"]),
                    ("svr", ["fp32", "int16"]), ("rf", ["fp32", "int8"]), ("hgb", ["fp32", "int8"]),
                    ("ridge_dur1", ["fp32", "fixed16"])]:
        for w in L.WINDOWS:
            for n in L.NS:
                d = stat(w, n, m, reps[1]) - stat(w, n, m, reps[0])
                con.append(dict(contrast=f"{reps[1]}_minus_{reps[0]}", model=m, window=f"{w}/N{n}",
                                est=float(np.median(d)), point=float(d.mean()), lo=ci(d)[0], hi=ci(d)[1]))
    pd.DataFrame(con).to_csv(os.path.join(L.RES, "r1_boot_contrasts.csv"), index=False)
    json.dump(dict(B=B, block=B_LEN, seed=20261110, levels="cells with replacement, then moving-block cycles"),
              open(os.path.join(L.RES, "r1_boot_settings.json"), "w"), indent=1)
    print("bootstrap written")

if __name__ == "__main__":
    main()
