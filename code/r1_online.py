"""R1: causal (online) acquisition and the width/duration confound.

Part A - online acquisition. The v1 windows were resampled retrospectively at
N instants that depend on the completed window duration. Here the controller
works causally on the native log, decimated to an acquisition interval of
r x 30 s:
  * window entry/exit: two threshold comparisons per acquired sample
    (V >= V_lo, V > V_hi); entry and exit timestamps are latched;
  * retained samples: a decimating buffer of capacity 2N. Every acquired
    in-window sample at the current stride is stored; when the buffer is full
    every other sample is discarded and the stride doubles. The last in-window
    sample is held in one extra register. Storage is therefore bounded by
    2N+1 samples without knowing the window duration in advance, and the
    retained samples are uniformly spaced in acquisition index;
  * charge throughput: either read from the BMS coulomb counter at the
    retained instants ('counter'), or reconstructed by trapezoidal
    integration of the retained current samples alone ('trapz').
Features are the same ten, computed on the retained (t, V, I, Q).

Part B - width versus traversal time. Equal-span windows (0.15 V at three
positions, 0.30 V at two) evaluated with the v1 retrospective N=8 resampling.
"""
import os, itertools
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import r1_lib as L

RATES = [1, 2, 4, 8]                 # x 30 s
CAPS = [4, 8, 16, 32]                # N; buffer capacity 2N
MODELS_ON = ["ridge", "mlp8", "svr", "hgb", "rf", "mlp32x16"]

def online_window(seg, lo, hi, r, N):
    t, V, I, Q = seg[:, ::r]
    inside = np.where(V >= lo)[0]
    if len(inside) == 0: return None
    e = inside[0]
    above = np.where(V[e:] > hi)[0]
    x = (e + above[0] - 1) if len(above) else len(V) - 1
    if x - e < 2: return None
    buf, stride, cap = [], 1, 2 * N
    for k in range(0, x - e + 1):
        if k % stride == 0:
            buf.append(e + k)
            if len(buf) > cap:
                buf = buf[::2]; stride *= 2
    if buf[-1] != x: buf.append(x)                    # exit register
    idx = np.array(buf)
    return t[idx], V[idx], I[idx], Q[idx], len(idx), t[x] - t[e]

def feats_online(r, N, w, meta, qmode):
    lo, hi = L.WIN_V[w]
    X = np.full((len(meta), 10), np.nan); kept = np.zeros(len(meta), int)
    for k, i in enumerate(meta.row.values):
        o = online_window(L.raw_segment(int(i)), lo, hi, r, N)
        if o is None: continue
        t, V, I, Q, m, dur = o
        if qmode == "trapz":
            Q = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)) / 3600.0])
        else:
            Q = Q - Q[0]
        X[k] = L.features_t(t, V, I, Q); kept[k] = m
    return X, kept

def loco(X, y, cells, name):
    P = np.full(len(y), np.nan)
    for c in L.MAIN_CELLS:
        tr, te = (cells != c), (cells == c)
        if name == "ridge_dur1":
            sc = StandardScaler().fit(X[tr][:, :1])
            P[te] = Ridge(alpha=1.0).fit(sc.transform(X[tr][:, :1]), y[tr]).predict(sc.transform(X[te][:, :1]))
        else:
            sc = StandardScaler().fit(X[tr])
            m, _ = L.fit(name, sc.transform(X[tr]), y[tr], 0)
            P[te] = m.predict(sc.transform(X[te]))
    return P

def part_a(args):
    w, r, N, qmode = args
    meta = L.screen(L.load_all(), "v1")
    cs2 = meta[meta.cell.isin(L.MAIN_CELLS)].reset_index(drop=True)
    X, kept = feats_online(r, N, w, cs2, qmode)
    ok = np.isfinite(X).all(1)
    y, cells = cs2.soh.values[ok], cs2.cell.values[ok]; X = X[ok]
    rows = []
    for name in MODELS_ON + ["ridge_dur1"]:
        if qmode == "trapz" and name == "ridge_dur1": continue
        p = loco(X, y, cells, name)
        for c, d in L.cell_metrics(p, y, cells).items():
            rows.append(dict(part="online", window=w, rate_s=30 * r, n=N, qmode=qmode, model=name, cell=c,
                             kept_median=float(np.median(kept[ok][cells == c])), kept_max=int(kept[ok][cells == c].max()),
                             cycles_lost=int((~ok).sum()), **d))
    print("A", w, r, N, qmode, flush=True)
    return rows

SPANS = {"s15_lo": (3.70, 3.85), "s15_mid": (3.85, 4.00), "s15_hi": (4.00, 4.15),
         "s30_lo": (3.70, 4.00), "s30_hi": (3.85, 4.15), "s45": (3.70, 4.15), "narrow": (3.90, 4.05)}

def resample_window(seg, lo, hi, n):
    t, V, I, Q = seg
    s = (V >= lo) & (V <= hi)
    if s.sum() < 6 or V[s][-1] <= V[s][0]: return None
    tt = t[s] - t[s][0]
    g = np.linspace(0, tt[-1], n)
    return (np.interp(g, tt, V[s]), np.interp(g, tt, I[s]), np.interp(g, tt, Q[s]) - np.interp(0, tt, Q[s]), tt[-1])

def span_X(span, cs2):
    lo, hi = SPANS[span]
    X = np.full((len(cs2), 10), np.nan)
    for k, i in enumerate(cs2.row.values):
        seg = L.raw_segment(int(i)); seg = seg[:, seg[2] > 0.01]
        o = resample_window(seg, lo, hi, 8)
        if o is not None: X[k] = L.features(*o)
    return X

def part_b(args):
    span, common = args
    lo, hi = SPANS[span]
    meta = L.screen(L.load_all(), "v1")
    cs2 = meta[meta.cell.isin(L.MAIN_CELLS)].reset_index(drop=True)
    X = span_X(span, cs2)
    ok = np.isfinite(X).all(1)
    if common is not None: ok = ok & common
    y, cells = cs2.soh.values[ok], cs2.cell.values[ok]; X = X[ok]
    rows = []
    for name in ["ridge", "hgb", "svr", "ridge_dur1"]:
        p = loco(X, y, cells, name)
        for c, d in L.cell_metrics(p, y, cells).items():
            rows.append(dict(part="span", window=span, v_lo=lo, v_hi=hi, n=8, model=name, cell=c,
                             cycle_set="common" if common is not None else "own",
                             soh_min=float(y[cells == c].min()),
                             dur_median_s=float(np.median(X[cells == c, 0])), cycles_lost=int((~ok).sum()), **d))
    print("B", span, flush=True)
    return rows

def common_mask():
    meta = L.screen(L.load_all(), "v1")
    cs2 = meta[meta.cell.isin(L.MAIN_CELLS)].reset_index(drop=True)
    ok = np.ones(len(cs2), bool)
    for sp in SPANS: ok &= np.isfinite(span_X(sp, cs2)).all(1)
    return ok

if __name__ == "__main__" and os.environ.get("SPAN_ONLY"):
    cm = common_mask()
    with Pool(14) as pool:
        B = pool.map(part_b, [(s_, None) for s_ in SPANS] + [(s_, cm) for s_ in SPANS])
    pd.DataFrame([r for o in B for r in o]).to_csv(os.path.join(L.RES, "r1_span_cells.csv"), index=False)
    print("span written"); raise SystemExit

if __name__ == "__main__":
    jobs = [(w, r, N, "counter") for w in L.WINDOWS for r in RATES for N in CAPS]
    jobs += [(w, 1, N, "trapz") for w in L.WINDOWS for N in CAPS]
    with Pool(14) as pool:
        A = pool.map(part_a, jobs)
        cm = common_mask()
        B = pool.map(part_b, [(s_, None) for s_ in SPANS] + [(s_, cm) for s_ in SPANS])
    pd.DataFrame([r for o in A for r in o]).to_csv(os.path.join(L.RES, "r1_online_cells.csv"), index=False)
    pd.DataFrame([r for o in B for r in o]).to_csv(os.path.join(L.RES, "r1_span_cells.csv"), index=False)
    print("written")
