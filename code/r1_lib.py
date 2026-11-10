"""Shared R1 machinery: data loading with screening variants, features,
estimators with their exact settings, executable deployable representations
whose accuracy is *evaluated* (not assumed), explicit storage encodings, and
operation counts for the regressor and for feature extraction.

Every byte figure produced here belongs to a representation whose predictions
are computed by the code in this file; no storage number is paired with the
accuracy of a different implementation.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(HERE)
RES = os.path.join(R1, "results")
RUN_OUT = os.environ.get("RUN_OUT", os.path.expanduser("~/runs/HV0013_R1"))
WINDOWS = ["narrow", "medium", "wide"]
WIN_V = {"narrow": (3.90, 4.05), "medium": (3.80, 4.10), "wide": (3.70, 4.15)}
NS = [4, 8, 16, 32]
MAIN_CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
CX2_CELLS = ["CX2_16", "CX2_34", "CX2_36", "CX2_37"]
ROLL_WIN, ROLL_TOL = 21, 0.05
CV_MIN_S = 600.0
LO_CLIP, HI_CLIP = 0.10, 1.05

FEATNAMES = ["dur_s", "dQ_Ah", "I_mean", "I_std", "V_mean", "V_std",
             "dVdt", "V_quad", "dQdV_mean", "dQdV_max"]

# ------------------------------------------------------------------ loading
class _Lazy(dict):
    """Arrays of the dense cache (pack_windows.py), loaded on first use only."""
    def __init__(self, path): super().__init__(); self._f = np.load(path)
    def __missing__(self, k): v = self._f[k]; self[k] = v; return v

_Z = None
def npz():
    global _Z
    if _Z is None:
        _Z = _Lazy(os.path.join(RUN_OUT, "windows_dense.npz"))
    return _Z

def load_all():
    m = pd.read_csv(os.path.join(RES, "cycles_r1_all.csv"))
    m["row"] = np.arange(len(m))
    return m

def screen(m, mode="v1"):
    """Return the retained subset.

    v1      : band + windows present, then centred 21-cycle rolling median, 5 %
    band    : band + windows present only (no rolling-median screen)
    tol10   : centred rolling median with a 10 % tolerance
    causal  : trailing (past-only) 21-cycle rolling median, 5 %
    cv      : metadata rule only: reject a cycle whose constant-voltage charge
              phase lasted less than CV_MIN_S (cell not fully charged, so the
              discharge capacity is not a full-capacity measurement)
    """
    base = m[m.in_band & m.has_windows].copy()
    if mode == "band":
        return base.reset_index(drop=True)
    if mode == "cv":
        return base[base.cv_s >= CV_MIN_S].reset_index(drop=True)
    keep = []
    for c, g in base.groupby("cell", sort=False):
        q = g.q_dis.values; qref = g.qref.values[0]
        if mode == "causal":
            med = pd.Series(q).rolling(ROLL_WIN, min_periods=3).median().values
            med[:3] = q[:3]
        else:
            med = pd.Series(q).rolling(ROLL_WIN, center=True, min_periods=3).median().values
        tol = 0.10 if mode == "tol10" else ROLL_TOL
        keep.append(pd.Series(np.abs(q - med) <= tol * qref, index=g.index))
    keep = pd.concat(keep)
    return base[keep.reindex(base.index).values].reset_index(drop=True)

def window_arrays(i, w, n):
    a = npz()[f"{w}_{n}"][i]
    return a[0], a[1], a[2]

def raw_segment(i):
    """Native-rate CC charge segment 3.60-4.20 V; rows t, V, I, Qcum."""
    z = npz(); o = z["seg_off"]
    return z["seg_flat"][:, o[i]:o[i + 1]]

# ----------------------------------------------------------------- features
DQDV_EPS = 1e-6

def features_t(t, V, I, Q):
    """The ten v1 features on samples at times t (need not be uniform)."""
    t = np.asarray(t, float); t = t - t[0]
    dur = t[-1]
    dV = np.diff(V); dQ = np.diff(Q)
    p1 = np.polyfit(t, V, 1)
    p2 = np.polyfit(t, V, 2) if len(V) >= 3 else [0.0, 0.0, 0.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dV) > DQDV_EPS, dQ / dV, 0.0)
    return np.array([dur, Q[-1] - Q[0], I.mean(), I.std(), V.mean(), V.std(),
                     p1[0], p2[0], np.nanmean(dqdv), np.nanmax(np.abs(dqdv))])

def features(V, I, Q, dur):
    return features_t(np.linspace(0.0, dur, len(V)), V, I, Q)

def build_matrix(meta, w, n):
    X = np.empty((len(meta), 10))
    for k, (i, dur) in enumerate(zip(meta["row"].values, meta[f"w_{w}_{n}_dur"].values)):
        V, I, Q = window_arrays(int(i), w, n)
        X[k] = features(V, I, Q, float(dur))
    return X

# --------------------------------------------------------------- estimators
HYPER = {
    "ridge":    dict(cls="Ridge", alpha=1.0, fit_intercept=True, solver="auto"),
    "mlp8":     dict(cls="MLPRegressor", hidden_layer_sizes=(8,), activation="relu", solver="lbfgs",
                     alpha=1e-4, max_iter=4000, max_fun=15000, tol=1e-4),
    "mlp32x16": dict(cls="MLPRegressor", hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs",
                     alpha=1e-4, max_iter=4000, max_fun=15000, tol=1e-4),
    "svr":      dict(cls="SVR", kernel="rbf", C=10.0, epsilon=0.01, gamma="scale (1/(n_feat*Var[X]))",
                     tol=1e-3, shrinking=True),
    "rf":       dict(cls="RandomForestRegressor", n_estimators=100, criterion="squared_error",
                     max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0,
                     bootstrap=True),
    "hgb":      dict(cls="HistGradientBoostingRegressor", loss="squared_error", learning_rate=0.1,
                     max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
                     l2_regularization=0.0, max_bins=255, early_stopping="auto (off for n<10000)"),
}
MODELS = list(HYPER)
STOCHASTIC = {"mlp8", "mlp32x16", "rf"}     # hgb: deterministic without early stopping / subsampling

def build(name, seed=0):
    if name == "ridge": return Ridge(alpha=1.0)
    if name == "mlp8":
        return MLPRegressor(hidden_layer_sizes=(8,), max_iter=4000, solver="lbfgs", random_state=seed)
    if name == "mlp32x16":
        return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=4000, solver="lbfgs", random_state=seed)
    if name == "svr": return SVR(C=10.0, epsilon=0.01)
    if name == "rf": return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    if name == "hgb": return HistGradientBoostingRegressor(random_state=seed)
    raise KeyError(name)

def fit(name, Xs, y, seed=0):
    """Fit on standardised features; record convergence information."""
    m = build(name, seed)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(Xs, y)
    conv = dict(converged=not any(issubclass(x.category, ConvergenceWarning) for x in wlist))
    if getattr(m, "n_iter_", None) is not None: conv["n_iter"] = int(np.max(m.n_iter_))
    return m, conv

# ====================================================== representations
# Each representation exposes predict(X_raw) and storage() -> dict of bytes,
# and ops() -> dict of operation counts per inference (regressor only).

F32 = np.float32

class Standardiser:
    def __init__(self, sc): self.mu = sc.mean_.astype(F32); self.sd = sc.scale_.astype(F32)
    def __call__(self, X): return ((X.astype(F32) - self.mu) / self.sd)

# ---- linear / MLP ------------------------------------------------------
def _layers(model):
    if isinstance(model, Ridge):
        return [model.coef_.reshape(-1, 1)], [np.atleast_1d(model.intercept_)]
    return list(model.coefs_), list(model.intercepts_)

class DenseFP32Folded:
    """float32 arithmetic on RAW features; standardiser folded exactly into the
    first layer (exact in real arithmetic, evaluated here in float32)."""
    def __init__(self, model, sc):
        W, b = _layers(model)
        mu, sd = sc.mean_, sc.scale_
        W0 = W[0] / sd[:, None]; b0 = b[0] - (mu / sd) @ W[0]
        self.W = [W0.astype(F32)] + [w.astype(F32) for w in W[1:]]
        self.b = [b0.astype(F32)] + [x.astype(F32) for x in b[1:]]
    def predict(self, X):
        a = X.astype(F32)
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if k < len(self.W) - 1: a = np.maximum(a, F32(0))
        return a.ravel().astype(float)
    def storage(self):
        n = sum(W.size + b.size for W, b in zip(self.W, self.b))
        return dict(model_bytes=4 * n, detail=f"{n} float32 (weights+biases, standardiser folded)")
    def ops(self):
        mac = sum(W.size for W in self.W); add = sum(b.size for b in self.b)
        relu = sum(b.size for b in self.b[:-1])
        return dict(mul=mac, add=mac + add, cmp=relu, exp=0, div=0)

def _q8(w):
    s = float(np.max(np.abs(w))) / 127.0
    s = s if s > 0 else 1.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s

class DenseInt8:
    """W8A8 post-training quantisation, per-tensor symmetric, int32 accumulation,
    int32 biases (scale s_x*s_w), float32 requantisation scales, float32
    standardiser kept separately (unfolded). Activation scales calibrated on the
    training fold. This is the implementation evaluated in v1, now with every
    stored value counted at its true width."""
    def __init__(self, model, sc, Xtr_raw, fold_std=False):
        self.fold = fold_std
        W, b = _layers(model)
        if fold_std:   # fold standardiser, then quantise raw features per tensor
            mu, sd = sc.mean_, sc.scale_
            W = [W[0] / sd[:, None]] + W[1:]
            b = [b[0] - (mu / sd) @ _layers(model)[0][0]] + b[1:]
            self.std = None
            a = Xtr_raw.astype(float)
        else:
            self.std = Standardiser(sc)
            a = self.std(Xtr_raw).astype(float)
        self.qw, self.sw, self.sx, self.qb = [], [], [], []
        for k, Wk in enumerate(W):
            qw, sw = _q8(Wk)
            amax = float(np.max(np.abs(a)))
            sx = amax / 127.0 if amax > 0 else 1.0
            self.qw.append(qw); self.sw.append(sw); self.sx.append(sx)
            self.qb.append(np.round(b[k] / (sx * sw)).astype(np.int64))
            a = a @ Wk + b[k]
            if k < len(W) - 1: a = np.maximum(a, 0.0)
    def predict(self, X):
        a = self.std(X).astype(float) if self.std is not None else X.astype(float)
        for k, qw in enumerate(self.qw):
            xq = np.clip(np.round(a / self.sx[k]), -127, 127).astype(np.int64)
            acc = xq @ qw.astype(np.int64) + self.qb[k]
            a = acc.astype(float) * F32(self.sx[k] * self.sw[k])
            if k < len(self.qw) - 1: a = np.maximum(a, 0.0)
        return a.ravel()
    def storage(self):
        w8 = sum(q.size for q in self.qw); b32 = sum(q.size for q in self.qb)
        scales = 2 * len(self.qw)                     # s_x and s_w per layer, float32
        std = 0 if self.std is None else 2 * self.std.mu.size
        tot = w8 + 4 * b32 + 4 * scales + 4 * std
        return dict(model_bytes=tot, int8_weights=w8, int32_biases_B=4 * b32,
                    scales_B=4 * scales, standardiser_B=4 * std,
                    detail=f"{w8} int8 W + {b32} int32 b + {scales} f32 scales + {std} f32 std")
    def ops(self):
        mac = sum(q.size for q in self.qw); nb = sum(q.size for q in self.qb)
        nin = self.qw[0].shape[0]
        relu = sum(q.size for q in self.qb[:-1])
        std = 0 if self.std is None else nin
        # quantise input of each layer: 1 mul + round/clip per element; requant: 1 mul per output
        qin = sum(q.shape[0] for q in self.qw)
        return dict(mul=mac + qin + nb + std, add=mac + nb + std, cmp=relu + 2 * qin, exp=0, div=0)

# ---- SVR ------------------------------------------------------------------
class SVRRep:
    """RBF-SVR with explicit kernel evaluation.

    fp32     : float32 support vectors, dual coefficients, gamma, intercept.
    int16    : int16 support vectors with per-feature float32 scales (the input
               is quantised with the same scales), int16 dual coefficients with
               one float32 scale.
    int8     : int8 support vectors, int16 dual coefficients.
    int8all  : int8 support vectors and int8 dual coefficients.
    The fitted SVR is a cancellation of many coefficients at the box bound C,
    so the kernel values must be reproduced to ~12 bits; the int8 variants are
    kept in the results to document that failure.
    """
    def __init__(self, model, sc, mode="fp32"):
        self.mode = mode
        self.std = Standardiser(sc)
        self.gamma = F32(model._gamma)
        sv = model.support_vectors_; a = model.dual_coef_.ravel()
        self.b = F32(model.intercept_[0])
        self.nsv, self.nf = sv.shape
        if mode == "fp32":
            self.sv = sv.astype(F32); self.a = a.astype(F32); return
        self.svbits = 16 if mode == "int16" else 8
        qmax = 2 ** (self.svbits - 1) - 1; self.qmax = qmax
        s = np.max(np.abs(sv), 0) / qmax; s = np.where(s > 0, s, 1.0)
        self.s_sv = s.astype(F32)
        self.svq = np.clip(np.round(sv / s), -qmax, qmax).astype(np.int32)
        if mode == "int8all":
            q, sa = _q8(a); self.aq, self.s_a, self.abits = q.astype(F32), sa, 8
        else:
            sa = float(np.max(np.abs(a))) / 32767.0 or 1.0
            self.aq = np.round(a / sa).astype(F32); self.s_a = sa; self.abits = 16
    def predict(self, X):
        x = self.std(X)
        if self.mode == "fp32":
            d2 = ((x[:, None, :] - self.sv[None]) ** 2).sum(-1)
            k = np.exp(-self.gamma * d2)
            return (k @ self.a + self.b).astype(float)
        xq = np.clip(np.round(x / self.s_sv), -self.qmax, self.qmax).astype(np.int32)
        diff = (xq[:, None, :] - self.svq[None]).astype(F32) * self.s_sv[None, None, :]
        d2 = (diff * diff).sum(-1)
        k = np.exp(-self.gamma * d2)
        return (F32(self.s_a) * (k @ self.aq) + self.b).astype(float)
    def storage(self):
        std = 2 * self.nf * 4
        if self.mode == "fp32":
            tot = 4 * (self.nsv * self.nf + self.nsv) + 4 + 4 + std
            return dict(model_bytes=tot, detail=f"{self.nsv}x{self.nf} f32 SV + {self.nsv} f32 coef + b,gamma + std")
        cb = self.abits // 8; vb = self.svbits // 8
        tot = vb * self.nsv * self.nf + cb * self.nsv + 4 * self.nf + 4 + 4 + 4 + std
        return dict(model_bytes=tot, detail=f"{self.nsv}x{self.nf} int{self.svbits} SV + {self.nsv} int{self.abits} coef + {self.nf}+1 f32 scales + b,gamma + std")
    def ops(self):
        n, f = self.nsv, self.nf
        # per SV: f subtractions, f multiplications (+f scalings when quantised),
        # f-1 additions, 1 multiply by gamma, 1 exp, 1 multiply-add with the coefficient
        q = 0 if self.mode == "fp32" else n * f
        return dict(sub=n * f + f, mul=n * f + q + 2 * n + f, add=n * (f - 1) + n + f,
                    cmp=0, exp=n, div=0)

# ---- trees ------------------------------------------------------------------
def _tree_arrays(model):
    """List of (feature, threshold, left, right, value, is_leaf) arrays, plus
    the additive baseline and the aggregation ('mean' or 'sum')."""
    trees = []
    if isinstance(model, RandomForestRegressor):
        for e in model.estimators_:
            t = e.tree_
            leaf = t.children_left == -1
            trees.append((t.feature.copy(), t.threshold.copy(), t.children_left.copy(),
                          t.children_right.copy(), t.value[:, 0, 0].copy(), leaf))
        return trees, 0.0, "mean"
    for stage in model._predictors:
        for p in stage:
            nd = p.nodes
            trees.append((nd["feature_idx"].astype(int), nd["num_threshold"].astype(float),
                          nd["left"].astype(int), nd["right"].astype(int),
                          nd["value"].astype(float), nd["is_leaf"].astype(bool)))
    return trees, float(np.ravel(model._baseline_prediction)[0]), "sum"

def _traverse(feat, thr, left, right, val, leaf, X, cmp_count=None):
    node = np.zeros(len(X), dtype=int); ncmp = np.zeros(len(X), dtype=int)
    active = ~leaf[node]
    while active.any():
        idx = np.where(active)[0]; nd = node[idx]
        go_left = X[idx, feat[nd]] <= thr[nd]
        node[idx] = np.where(go_left, left[nd], right[nd])
        ncmp[idx] += 1
        active = ~leaf[node]
    return val[node], ncmp

class TreeRep:
    """Tree ensemble with an explicit pre-order node encoding.

    fp32 : internal node = feature (uint8) + threshold (float32) + right-child
           offset (uint16) = 7 B; leaf = marker (uint8) + value (float32) = 5 B.
           Thresholds are expressed in raw feature units (standardiser folded
           exactly), so no standardiser is stored.
    int8 : features quantised on device to uint8 with a per-feature affine map
           (lo, step: 2 float32 per feature) fitted on the training fold;
           internal node = feature (uint8) + threshold (uint8) + offset (uint16)
           = 4 B; leaf = marker (uint8) + value (int16, one float32 scale) = 3 B.
    Both: one uint32 start offset per tree, float32 baseline.
    Leaf values for the int8 encoding are quantised against the leaf scale; the
    accuracy reported for each encoding is computed by this traversal."""
    def __init__(self, model, sc, Xtr_raw, mode="fp32"):
        self.mode = mode
        self.trees, self.base, self.agg = _tree_arrays(model)
        self.std = sc
        self.nf = sc.mean_.size
        if mode == "int8":
            Xs = sc.transform(Xtr_raw)
            self.lo = Xs.min(0); hi = Xs.max(0)
            self.step = np.where(hi > self.lo, (hi - self.lo) / 255.0, 1.0)
            allv = np.concatenate([t[4][t[5]] for t in self.trees])
            self.vscale = float(np.max(np.abs(allv))) / 32767.0 or 1.0
            q = []
            for feat, thr, left, right, val, leaf in self.trees:
                f = np.where(leaf, 0, feat)
                tq = np.floor((thr - self.lo[f]) / self.step[f])
                tq = np.where(leaf, 0, np.clip(tq, -1, 255))
                vq = np.round(val / self.vscale) * self.vscale
                q.append((feat, tq, left, right, vq, leaf))
            self.qtrees = q
    def _x(self, X):
        Xs = self.std.transform(X)
        if self.mode == "fp32":
            return Xs.astype(F32).astype(float)
        return np.clip(np.round((Xs - self.lo) / self.step), 0, 255)
    def predict(self, X, return_cmp=False):
        x = self._x(X); trees = self.trees if self.mode == "fp32" else self.qtrees
        tot = np.zeros(len(X)); ncmp = np.zeros(len(X))
        for t in trees:
            v, c = _traverse(*t, x); tot += v; ncmp += c
        p = self.base + (tot / len(trees) if self.agg == "mean" else tot)
        return (p, ncmp) if return_cmp else p
    def counts(self):
        internal = sum(int((~t[5]).sum()) for t in self.trees)
        leaves = sum(int(t[5].sum()) for t in self.trees)
        return internal, leaves, len(self.trees)
    def storage(self):
        i, l, n = self.counts()
        if self.mode == "fp32":
            tot = 7 * i + 5 * l + 4 * n + 4
        else:
            tot = 4 * i + 3 * l + 4 * n + 4 + 4 + 8 * self.nf
        return dict(model_bytes=tot, internal=i, leaves=l, trees=n,
                    detail=f"{i} internal + {l} leaves in {n} trees ({self.mode} encoding)")
    def ops(self, mean_cmp=None):
        i, l, n = self.counts()
        d = dict(mul=1 if self.agg == "mean" else 0, add=n, cmp=mean_cmp or 0, exp=0, div=0)
        if self.mode == "int8": d["mul"] += self.nf; d["add"] += self.nf
        return d

# ---- one-feature estimator --------------------------------------------------
class DurFP32:
    """SoH = a*dur + b, float32, standardiser folded: 2 float32 = 8 B."""
    def __init__(self, model, sc):
        self.a = F32(model.coef_[0] / sc.scale_[0])
        self.b = F32(model.intercept_ - model.coef_[0] * sc.mean_[0] / sc.scale_[0])
    def predict(self, d): return (self.a * d.astype(F32).ravel() + self.b).astype(float)
    def storage(self): return dict(model_bytes=8, detail="2 float32 (slope, intercept)")
    def ops(self): return dict(mul=1, add=1, cmp=0, exp=0, div=0)

class DurInt16:
    """Integer-only: Y = ((A*d) >> s) + B, SoH = Y / 2^13 (Q2.13 output).
    d is the window duration in whole seconds (uint16), A and B are int16, s a
    shift stored in one byte; A*d fits a 32-bit accumulator for d < 65 536 s.
    Storage 2 + 2 + 1 = 5 B."""
    F = 13
    def __init__(self, model, sc):
        a = model.coef_[0] / sc.scale_[0]
        b = model.intercept_ - model.coef_[0] * sc.mean_[0] / sc.scale_[0]
        k = int(np.floor(np.log2(32767 / abs(a)))) if a != 0 else self.F
        self.A = int(round(a * 2**k)); self.s = k - self.F
        self.B = int(np.clip(round(b * 2**self.F), -32768, 32767))
        assert abs(self.A) <= 32767 and self.s >= 0
    def predict(self, d):
        ds = np.clip(np.round(np.asarray(d).ravel()), 0, 65535).astype(np.int64)
        return (((self.A * ds) >> self.s) + self.B) / 2**self.F
    def storage(self): return dict(model_bytes=5, detail="int16 slope + int16 intercept + uint8 shift")
    def ops(self): return dict(mul=1, add=1, cmp=0, exp=0, div=0, shift=1)

# ============================================================ feature cost
def prep_ops(N, nfeat_std=10):
    """Operation count of the ten-feature extraction on N retained samples
    (uniform grid; the least-squares weights for slope and curvature on a
    uniform grid are precomputable constants scaled by 1/dur and 1/dur^2)."""
    n = N; m = N - 1
    ops = dict(add=0, sub=0, mul=0, div=0, cmp=0, sqrt=0, abs=0)
    ops["sub"] += 1                                    # dur = t_exit - t_entry
    ops["sub"] += 1                                    # dQ
    for _ in range(2):                                 # mean of I and V
        ops["add"] += n - 1; ops["mul"] += 1
    for _ in range(2):                                 # std of I and V
        ops["sub"] += n; ops["mul"] += n + 1; ops["add"] += n - 1; ops["sqrt"] += 1
    ops["mul"] += n + 1; ops["add"] += n - 1           # slope: sum c_k V_k / dur
    ops["mul"] += n + 2; ops["add"] += n - 1           # curvature: sum e_k V_k / dur^2
    ops["sub"] += 2 * m; ops["abs"] += m; ops["cmp"] += m; ops["div"] += m   # dQ/dV ratios
    ops["add"] += m - 1; ops["mul"] += 1               # mean dQ/dV
    ops["abs"] += m; ops["cmp"] += m - 1               # max |dQ/dV|
    const_floats = 2 * n                               # LS weights c_k, e_k
    return ops, const_floats

def total(d): return int(sum(v for v in d.values() if isinstance(v, (int, np.integer))))

# ================================================================ metrics
def rmse(p, y): return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))

def cell_metrics(p, y, cells):
    out = {}
    for c in pd.unique(cells):
        s = cells == c
        out[c] = dict(rmse=rmse(p[s], y[s]), sse=float(np.sum((p[s] - y[s]) ** 2)), ncyc=int(s.sum()),
                      bias=float(np.mean(p[s] - y[s])))
    return out
