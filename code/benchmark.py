"""Sensing- and compute-budget benchmark for on-board SoH estimation.

A budget is a pair (voltage window, N samples).  Every estimator sees exactly the
same N-sample CC-charge window; the only thing that changes across rows of the
benchmark is how much sensing and how much arithmetic the estimator is allowed.

Costs are reported hardware-independently: parameter bytes (fp32 and int8),
multiply-accumulate operations per inference, and the sensing payload in bytes.
Host-CPU wall-clock is logged only as a sanity check and is never used for claims.
"""
import os, json, time, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
WINDOWS = ["narrow", "medium", "wide"]
NS = [4, 8, 16, 32]
SEED = 0

# ---------------------------------------------------------------- data loading
ROLL_WIN = 21      # centred rolling-median window, in cycles
ROLL_TOL = 0.05    # reject if |q - median| exceeds this fraction of the reference capacity

def load():
    """Load the extracted cycles and drop partial or aborted discharges.

    Capacity fade is smooth on the cycle scale, so a cycle whose discharge
    capacity departs from a centred rolling median by more than ROLL_TOL of the
    cell reference capacity is an interrupted or reference-performance cycle
    rather than an ageing observation.  Rejecting them removes label noise
    without touching the trend.
    """
    meta = pd.read_csv(os.path.join(RES, "cycles.csv"))
    z = np.load(os.path.join(RES, "windows.npz"))
    meta["row"] = np.arange(len(meta))
    keep = np.zeros(len(meta), dtype=bool)
    stats = {}
    for c, g in meta.groupby("cell"):
        q = g.q_dis.values
        qref = float(np.mean(q[:5] / g.soh.values[:5]))
        med = pd.Series(q).rolling(ROLL_WIN, center=True, min_periods=3).median().values
        ok = np.abs(q - med) <= ROLL_TOL * qref
        keep[g.row.values] = ok
        stats[c] = dict(total=int(len(g)), kept=int(ok.sum()))
    stats["all"] = dict(total=int(len(keep)), kept=int(keep.sum()))
    json.dump(stats, open(os.path.join(RES, "filter_stats.json"), "w"), indent=1)
    print(f"label filter: kept {int(keep.sum())} of {len(keep)} cycles "
          f"({100*keep.sum()/len(keep):.1f}%)", flush=True)
    out = meta[keep].reset_index(drop=True)
    out.to_csv(os.path.join(RES, "cycles_filtered.csv"), index=False)
    return out, z

def window_arrays(z, i, w, n):
    return (z[f"{i}|w_{w}_{n}_V"], z[f"{i}|w_{w}_{n}_I"], z[f"{i}|w_{w}_{n}_Q"])

FEATNAMES = ["dur_s","dQ_Ah","I_mean","I_std","V_mean","V_std",
             "dVdt","V_quad","dQdV_mean","dQdV_max"]

def features(V, I, Q, dur):
    n = len(V)
    t = np.linspace(0.0, dur, n)
    dV = np.diff(V); dQ = np.diff(Q); dt = np.diff(t)
    p1 = np.polyfit(t, V, 1)
    p2 = np.polyfit(t, V, 2) if n >= 3 else [0.0, 0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dV) > 1e-6, dQ / dV, 0.0)
    return np.array([dur, Q[-1] - Q[0], I.mean(), I.std(), V.mean(), V.std(),
                     p1[0], p2[0], np.nanmean(dqdv), np.nanmax(np.abs(dqdv))],
                    dtype=np.float64)

def build_matrix(meta, z, w, n):
    X, raw = [], []
    for k in range(len(meta)):
        i = int(meta["row"].iloc[k])
        V, I, Q = window_arrays(z, i, w, n)
        dur = float(meta.iloc[k][f"w_{w}_{n}_dur"])
        X.append(features(V, I, Q, dur))
        raw.append(np.stack([V, I, Q]))
    return np.array(X), np.array(raw)

# ------------------------------------------------------------ cost accounting
def int8_bytes(n_params):  return n_params            # 1 byte / parameter
def fp32_bytes(n_params):  return 4 * n_params

def mlp_cost(layer_sizes):
    """params and MACs of a fully connected stack (biases included)."""
    p = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
            for i in range(len(layer_sizes)-1))
    macs = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    return p, macs

def tree_cost(model):
    """node count of a fitted sklearn forest / boosting model."""
    if hasattr(model, "estimators_"):
        est = np.ravel(model.estimators_)
        nodes = sum(e.tree_.node_count for e in est)
        depth = max(e.tree_.max_depth for e in est)
        return nodes, len(est), depth
    if hasattr(model, "_predictors"):
        nodes = sum(pred.nodes.shape[0] for stage in model._predictors for pred in stage)
        ntrees = sum(len(stage) for stage in model._predictors)
        depth = max(int(pred.get_max_depth()) for stage in model._predictors for pred in stage)
        return nodes, ntrees, depth
    return 0, 0, 0

# ------------------------------------------------------- int8 fixed-point MLP
def quantise(w, bits=8):
    s = np.max(np.abs(w)) / (2**(bits-1) - 1)
    s = s if s > 0 else 1.0
    return np.round(w / s).astype(np.int8), s

class Int8MLP:
    """Post-training per-tensor symmetric int8 weights and activations, int32
    accumulation.  Activation scales are calibrated on the TRAINING fold only;
    the test fold never influences a quantisation parameter."""
    def __init__(self, coefs, intercepts, X_calib):
        self.q, self.s, self.b = [], [], intercepts
        for W in coefs:
            qw, sw = quantise(W); self.q.append(qw); self.s.append(sw)
        self.a_scale, a = [], X_calib.astype(np.float64)
        for k, W in enumerate(coefs):
            amax = np.max(np.abs(a))
            self.a_scale.append(amax / 127.0 if amax > 0 else 1.0)
            a = a @ W + intercepts[k]
            if k < len(coefs) - 1:
                a = np.maximum(a, 0.0)

    def predict(self, X):
        a = X.astype(np.float64)
        for k, (qw, sw) in enumerate(zip(self.q, self.s)):
            sx = self.a_scale[k]
            xq = np.clip(np.round(a / sx), -127, 127).astype(np.int32)
            acc = xq @ qw.astype(np.int32)
            a = acc.astype(np.float64) * sx * sw + self.b[k]
            if k < len(self.q) - 1:
                a = np.maximum(a, 0.0)
        return a.ravel()

# ------------------------------------------------------------------ estimators
def make_models(n_feat):
    return {
        "ridge":  dict(kind="lin",  build=lambda: Ridge(alpha=1.0)),
        # lbfgs, not adam: on a problem this small and this smooth the stochastic
        # optimiser does not converge within any reasonable iteration budget and
        # its error would be an artefact of the optimiser, not of the model.
        "mlp8":   dict(kind="mlp",  build=lambda: MLPRegressor(hidden_layer_sizes=(8,), max_iter=4000,
                                                              solver="lbfgs", random_state=SEED)),
        "mlp32x16": dict(kind="mlp", build=lambda: MLPRegressor(hidden_layer_sizes=(32,16), max_iter=4000,
                                                              solver="lbfgs", random_state=SEED)),
        "svr":    dict(kind="svr",  build=lambda: SVR(C=10.0, epsilon=0.01)),
        "rf":     dict(kind="tree", build=lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=4)),
        "hgb":    dict(kind="tree", build=lambda: HistGradientBoostingRegressor(random_state=SEED)),
    }

def cost_of(name, kind, model, n_feat, scaler_cost=True):
    """returns dict with params, bytes_fp32, bytes_int8, macs, note"""
    pre = 2 * n_feat if scaler_cost else 0      # mean+scale of the standardiser
    if kind == "lin":
        p, macs = n_feat + 1, n_feat
    elif kind == "mlp":
        sizes = [n_feat] + list(model.hidden_layer_sizes) + [1]
        p, macs = mlp_cost(sizes)
    elif kind == "svr":
        nsv = model.support_vectors_.shape[0]
        p = nsv * n_feat + nsv + 1
        macs = nsv * n_feat            # one RBF kernel evaluation per SV
    else:
        nodes, ntrees, depth = tree_cost(model)
        p = nodes * 3                  # threshold + feature index + child/value
        macs = ntrees * depth          # comparisons, not multiplies
    p += pre
    return dict(params=int(p), bytes_fp32=int(fp32_bytes(p)), bytes_int8=int(int8_bytes(p)),
                macs=int(macs))

# ------------------------------------------------------------------ evaluation
def loco_eval(X, y, cells, models, n_feat):
    out = []
    for name, spec in models.items():
        preds = np.full(len(y), np.nan); cost = None; q_preds = np.full(len(y), np.nan)
        for c in np.unique(cells):
            tr, te = cells != c, cells == c
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            m = spec["build"]().fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
            if cost is None:
                cost = cost_of(name, spec["kind"], m, n_feat)
            if spec["kind"] in ("lin", "mlp"):
                if spec["kind"] == "lin":
                    coefs = [m.coef_.reshape(-1, 1)]; inter = [np.atleast_1d(m.intercept_)]
                else:
                    coefs = [c_ for c_ in m.coefs_]; inter = [b for b in m.intercepts_]
                q_preds[te] = Int8MLP(coefs, inter, Xtr).predict(Xte)
        r = dict(model=name, kind=spec["kind"],
                 rmse=float(np.sqrt(np.mean((preds - y) ** 2))),
                 mae=float(np.mean(np.abs(preds - y))),
                 worst_cell_rmse=float(max(np.sqrt(np.mean((preds[cells == c] - y[cells == c]) ** 2))
                                           for c in np.unique(cells))),
                 **cost)
        if not np.all(np.isnan(q_preds)):
            r["rmse_int8"] = float(np.sqrt(np.mean((q_preds - y) ** 2)))
        out.append(r)
    return out

def main():
    meta, z = load()
    meta = meta.reset_index(drop=True)
    fam = meta["cell"].str[:3]
    cs2 = meta[fam == "CS2"].index.values
    cx2 = meta[fam == "CX2"].index.values
    print(f"cycles: {len(meta)} (CS2 {len(cs2)}, CX2 {len(cx2)}); cells {sorted(meta.cell.unique())}")
    rows, transfer_rows = [], []
    for w, n in itertools.product(WINDOWS, NS):
        X, raw = build_matrix(meta, z, w, n)
        y = meta["soh"].values
        cells = meta["cell"].values
        models = make_models(X.shape[1])
        res = loco_eval(X[cs2], y[cs2], cells[cs2], models, X.shape[1])
        sense_bytes = 3 * n * 2               # V, I, Q as uint16
        dur = float(np.median(meta[f"w_{w}_{n}_dur"].values[cs2]))
        for r in res:
            r.update(window=w, n=n, sense_bytes=sense_bytes, window_s=dur)
            rows.append(r)
        # external transfer: fit on all CS2, test on CX2 cells
        for name, spec in models.items():
            sc = StandardScaler().fit(X[cs2])
            m = spec["build"]().fit(sc.transform(X[cs2]), y[cs2])
            p = m.predict(sc.transform(X[cx2]))
            transfer_rows.append(dict(window=w, n=n, model=name,
                                      rmse_cx2=float(np.sqrt(np.mean((p - y[cx2]) ** 2))),
                                      bias_cx2=float(np.mean(p - y[cx2]))))
        print(f"  budget {w}/{n}: best LOCO RMSE {min(r['rmse'] for r in res):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "benchmark_loco.csv"), index=False)
    pd.DataFrame(transfer_rows).to_csv(os.path.join(RES, "benchmark_transfer.csv"), index=False)
    print("written benchmark_loco.csv / benchmark_transfer.csv")

if __name__ == "__main__":
    main()
