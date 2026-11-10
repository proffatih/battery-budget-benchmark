"""R1: every number quoted in the revised manuscript, as LaTeX macros, and the
main tables, generated from the R1 result CSVs only."""
import os, json, warnings
import numpy as np, pandas as pd
import r1_lib as L
warnings.filterwarnings("ignore")

MAN = os.path.join(L.R1, "manuscript")
os.makedirs(MAN, exist_ok=True)
RES = L.RES
OUT = []
import re as _re
_DW = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
def mac(name, val):
    name = _re.sub(r"\d", lambda m_: _DW[m_.group()], name)      # LaTeX macro names: letters only
    assert name.isalpha(), name
    OUT.append(r"\newcommand{\%s}{%s}" % (name, val))
def f4(x): return f"{x:.4f}"
def pp(x): return f"{100*x:.1f}"          # SoH fraction -> percentage points
def thou(x): return f"{int(round(x)):,}".replace(",", "\\,")
def pool(x, col="sse"): return float(np.sqrt(x[col].sum() / x.ncyc.sum()))

cells_all = pd.read_csv(os.path.join(RES, "cycles_r1_all.csv"))
d = pd.read_csv(os.path.join(RES, "r1_loco_cells.csv"))
cst = pd.read_csv(os.path.join(RES, "r1_loco_costs.csv"))
bt = pd.read_csv(os.path.join(RES, "r1_boot_rmse.csv"))
con = pd.read_csv(os.path.join(RES, "r1_boot_contrasts.csv"))
on = pd.read_csv(os.path.join(RES, "r1_online_cells.csv"))
sp = pd.read_csv(os.path.join(RES, "r1_span_cells.csv"))

# ================================================================ dataset
m = L.load_all()
m8 = m[m.cell.isin(L.MAIN_CELLS + L.CX2_CELLS)]
band = L.screen(m8, "band"); v1 = L.screen(m8, "v1")
mac("Nextracted", thou(len(band))); mac("Ndropped", str(len(band) - len(v1)))
mac("dropPct", f"{100*(1-len(v1)/len(band)):.1f}")
mac("Ncycles", thou(len(v1)))
mac("NcyclesCS", thou((v1.design == "CS2").sum())); mac("NcyclesCX", thou((v1.design == "CX2").sum()))
mac("Ncells", "8"); mac("SoHmin", f"{v1.soh.min():.3f}"); mac("SoHmax", f"{v1.soh.max():.3f}")
mac("SoHminCS", f"{v1[v1.design=='CS2'].soh.min():.3f}")
for w in L.WINDOWS:
    mac(f"raw{w.capitalize()}", str(int(v1[f"n_raw_{w}"].median())))
    mac(f"dur{w.capitalize()}", f"{v1[f'w_{w}_4_dur'].median()/60:.0f}")
mac("dtLog", f"{v1.dt_med_wide.median():.0f}"); mac("dtLogMax", f"{v1.dt_max_wide.max():.0f}")
mac("iCV", f"{100*v1.i_cv_wide.median():.2f}")
# screening evidence
key = set(zip(v1.cell, v1.cycle)); band = band.copy()
band["kept"] = [(c, y) in key for c, y in zip(band.cell, band.cycle)]
rej = band[~band.kept]
mac("rejCutoffPct", f"{100*(rej.v_dis_end < 2.71).mean():.0f}")
dev = []
for c_, g_ in band.groupby("cell"):
    md = pd.Series(g_.q_dis.values).rolling(L.ROLL_WIN, center=True, min_periods=3).median().values
    dev += list((g_.q_dis.values - md) / g_.qref.values)
band["dev"] = dev
mac("rejDefMed", f"{-100*band[~band.kept].dev.median():.0f}")
mac("rejDefLo", f"{-100*band[~band.kept].dev.quantile(0.95):.0f}"); mac("rejDefHi", f"{-100*band[~band.kept].dev.quantile(0.05):.0f}")
regular = ["CS2_35", "CS2_38", "CX2_34", "CX2_36"]
rr = band[band.cell.isin(regular)]
mac("rejShortCVPct", f"{100*(rr[~rr.kept].cv_s < 600).mean():.0f}")
mac("keptShortCVPct", f"{100*(rr[rr.kept].cv_s < 600).mean():.0f}")
mac("rejCVmed", f"{band[~band.kept].cv_s.median():.0f}"); mac("keptCVmed", f"{band[band.kept & band.cell.isin(regular)].cv_s.median():.0f}")
# charge starting inside the window
st = []
for i, c, s in zip(v1.row, v1.cell, v1.soh):
    seg = L.raw_segment(int(i)); seg = seg[:, seg[2] > 0.01]
    st.append((c[:3], s, seg[1, 0]))
st = pd.DataFrame(st, columns=["design", "soh", "v0"])
for w, (lo, hi) in L.WIN_V.items():
    st[w] = st.v0 > lo
cs = st[st.design == "CS2"]
mac("truncWideCS", f"{100*cs.wide.mean():.0f}"); mac("truncNarrowCS", f"{100*cs.narrow.mean():.0f}")
mac("truncWideHi", f"{100*cs[cs.soh >= 0.8].wide.mean():.0f}")
mac("truncWideLo", f"{100*cs[cs.soh < 0.7].wide.mean():.0f}")

# ------------------------------------------------------------- Table 1
EXTRA = ["CS2_33", "CS2_34", "CX2_33", "CX2_35", "CX2_38"]
v1x = L.screen(m[m.cell.isin(EXTRA)], "v1")
role = {**{c: "primary" for c in L.MAIN_CELLS}, **{c: "transfer" for c in L.CX2_CELLS},
        **{c: "decomposition$^b$" for c in EXTRA}}
T = ["\\begin{table}[H]",
     "\\caption{Cells, logged test conditions and retained cycles. Currents are medians over the"
     " retained cycles, read from the logs; C-rates are relative to the nominal capacity. Every cell was"
     " charged at 0.50C to 4.2 V. Primary: leave-one-cell-out benchmark; transfer: never used for"
     " training in the primary benchmark. $Q_0$ is the reference capacity, the median discharge capacity"
     " of the first five retained cycles, which defines SoH = 1 for that cell. Temperature is not"
     " recorded in these files.\\label{tab:cells}}",
     "\\begin{adjustwidth}{-\\extralength}{0cm}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabularx}{\\textwidth}{llXrrrrrr}", "\\toprule",
     "\\textbf{Cell} & \\textbf{Design} & \\textbf{Role} & \\textbf{Charge (A)} & \\textbf{Disch. (A)}"
     " & \\textbf{Disch. (C)} & \\textbf{Cycles} & \\textbf{$Q_0$ (Ah)} & \\textbf{SoH range} \\\\", "\\midrule"]
for c in L.MAIN_CELLS + L.CX2_CELLS + EXTRA:
    if c == "CS2_33": T.append("\\midrule")
    x = pd.concat([v1, v1x])[lambda q: q.cell == c]; nom = x.nominal.iloc[0]
    ich, idis = x.i_ch_cc.median(), x.i_dis_mean.median()
    dis = f"{idis:.3f} & {idis/nom:.2f}"
    if c == "CX2_16":
        lo_, hi_ = x.i_dis_mean.quantile(0.05), x.i_dis_mean.quantile(0.95)
        dis = f"{lo_:.2f}--{hi_:.2f}$^a$ & {lo_/nom:.2f}--{hi_/nom:.2f}"
    T.append(f"{c.replace('_', chr(92)+'_')} & {c[:3]} & {role[c]} & {ich:.3f} & {dis} & "
             f"{thou(len(x))} & {x.qref.iloc[0]:.3f} & {x.soh.min():.3f}--{x.soh.max():.3f} \\\\")
T += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}",
      "\\noindent{\\footnotesize{$^a$~CX2\\_16 was discharged at two currents: 5th--95th percentile shown."
      " $^b$~Additional cells of the same repository, used only in Section~\\ref{sec:res-transfer} to separate"
      " a change of discharge protocol from a change of cell design; they enter no other result.}}",
      "\\end{table}"]
open(os.path.join(MAN, "table_cells.tex"), "w").write("\n".join(T) + "\n")

# ================================================================ LOCO
ref = d[(d.rep == "ref")]
s0 = ref[ref.seed == 0]
P = s0.groupby(["window", "n", "model"]).apply(pool).rename("rmse").reset_index()
EQ = s0.groupby(["window", "n", "model"]).rmse.mean().rename("eq").reset_index()
WC = s0.groupby(["window", "n", "model"]).rmse.max().rename("worst").reset_index()
BC = s0.groupby(["window", "n", "model"]).rmse.min().rename("bestcell").reset_index()
G80 = s0.groupby(["window", "n", "model"]).apply(lambda x: np.sqrt(x.sse_ge80.sum() / x.n_ge80.sum())).rename("ge80").reset_index()
SD = (ref.groupby(["window", "n", "model", "seed"]).apply(pool).rename("r").reset_index()
      .groupby(["window", "n", "model"]).r.agg(["std", "min", "max"]).reset_index())
K = P.merge(EQ).merge(WC).merge(BC).merge(G80).merge(SD).merge(bt.rename(columns={"rmse": "_r"}), on=["window", "n", "model"])

def kk(w, n, mdl): return K[(K.window == w) & (K.n == n) & (K.model == mdl)].iloc[0]
WORD = {4: "four", 8: "eight", 16: "sixteen", 32: "thirtytwo"}
for n in L.NS: mac(f"bestRMSEn{WORD[n]}", f4(K[K.n == n].rmse.min()))
for w in L.WINDOWS: mac(f"bestRMSE{w.capitalize()}", f4(K[K.window == w].rmse.min()))
b = K.loc[K.rmse.idxmin()]
mac("bestRMSE", f4(b.rmse)); mac("bestModel", b.model); mac("bestWindow", b.window); mac("bestN", str(int(b.n)))
mac("bestLo", f4(b.lo)); mac("bestHi", f4(b.hi)); mac("bestWorstCell", f4(b.worst)); mac("bestEq", f4(b["eq"]))
r = K[K.model == "ridge"].loc[lambda x: x.rmse.idxmin()]
mac("ridgeBestRMSE", f4(r.rmse)); mac("ridgeBestWindow", r.window); mac("ridgeBestN", str(int(r.n)))
mac("ridgeLo", f4(r.lo)); mac("ridgeHi", f4(r.hi))
sfb = K[K.model == "ridge_dur1"].loc[lambda x: x.rmse.idxmin()]
mac("sfBest", f4(sfb.rmse)); mac("sfBestWindow", sfb.window); mac("sfWorstCell", f4(sfb.worst))
mac("sfLo", f4(sfb.lo)); mac("sfHi", f4(sfb.hi))
for w in L.WINDOWS: mac(f"sf{w.capitalize()}", f4(kk(w, 4, "ridge_dur1").rmse))
mac("windowGain", f"{K[K.window=='narrow'].rmse.min()/K[K.window=='wide'].rmse.min():.1f}")
mac("worstGapFactor", f"{(K.worst/K.rmse).median():.2f}")
mac("eqPooledMaxDiff", f"{(K["eq"]-K.rmse).abs().max():.4f}")
# N contrast (wide window, all models)
cN = con[(con.contrast == "N4_minus_N32") & (con.window == "wide")]
mac("NcontrastMaxAbs", f4(cN.point.abs().max())); mac("NcontrastLoMin", f4(cN.lo.min())); mac("NcontrastHiMax", f4(cN.hi.max()))
mac("NcontrastAllZero", "all" if ((cN.lo <= 0) & (cN.hi >= 0)).all() else "not all")
for mdl in ["ridge", "hgb"]:
    x = cN[cN.model == mdl].iloc[0]
    mac(f"Nc{mdl}", f"{x.point:+.4f}"); mac(f"Nc{mdl}Lo", f"{x.lo:+.4f}"); mac(f"Nc{mdl}Hi", f"{x.hi:+.4f}")
cW = con[(con.contrast == "narrow_over_wide")]
mac("WratioMin", f"{cW.point.min():.1f}"); mac("WratioMax", f"{cW.point.max():.1f}")
mac("WratioLoMin", f"{cW.lo.min():.2f}")
cM = con[con.contrast.str.endswith("minus_ridge") & (con.window == "wide/N8")].set_index("model")
for mdl in ["hgb", "svr", "rf", "mlp8", "mlp32x16", "ridge_dur1"]:
    x = cM.loc[mdl]; key_ = mdl.replace("_", "").replace("32x16", "big").replace("8", "small").replace("dur1", "dur")
    mac(f"vs{key_}", f"{x.point:+.4f}"); mac(f"vs{key_}Lo", f"{x.lo:+.4f}"); mac(f"vs{key_}Hi", f"{x.hi:+.4f}")
# seeds and convergence
st_ = SD[SD.model.isin(["mlp8", "mlp32x16"])]
mac("mlpSeedSDmed", f4(st_["std"].median())); mac("mlpSeedSDmax", f4(st_["std"].max()))
mac("rfSeedSDmax", f4(SD[SD.model == "rf"]["std"].max()))
mac("nFits", thou(len(cst[cst.rep.isin(["fp32"])])))
mac("nNonConv", str(int((~cst.converged.astype(bool)).sum())))
mi = cst[cst.model.str.startswith("mlp") & (cst.rep == "fp32")].n_iter
mac("mlpIterMin", str(int(mi.min()))); mac("mlpIterMax", str(int(mi.max())))
# SoH >= 0.8
mac("geEightyN", thou(s0[(s0.window == "wide") & (s0.n == 4) & (s0.model == "ridge")].n_ge80.sum()))
mac("geEightyBest", f4(K.ge80.min())); g80b = K.loc[K.ge80.idxmin()]
mac("geEightyBestModel", g80b.model); mac("geEightyBestWindow", g80b.window); mac("geEightyBestN", str(int(g80b.n)))
mac("geEightyRidge", f4(K[K.model == "ridge"].ge80.min())); mac("geEightySf", f4(K[K.model == "ridge_dur1"].ge80.min()))
mac("geEightyNarrow", f4(K[K.window == "narrow"].ge80.min())); mac("geEightyWide", f4(K[K.window == "wide"].ge80.min()))
for n in L.NS: mac(f"geEightyN{WORD[n]}", f4(K[K.n == n].ge80.min()))

# ------------------------------------------------------------- Table: LOCO summary
best_cfg = K.loc[K.groupby("model").rmse.idxmin()].set_index("model")
order = ["ridge", "mlp8", "mlp32x16", "svr", "rf", "hgb", "ridge_dur1"]
NAME = {"ridge_dur1": "duration only"}
T = ["\\begin{table}[H]",
     "\\caption{Leave-one-cell-out accuracy on the four CS2 cells, reference (float64) models, each"
     " estimator at its best budget. Pooled RMSE with a 95\\% cell-grouped hierarchical bootstrap interval;"
     " equal-cell mean; range of the four per-cell errors; pooled RMSE on cycles with SoH $\\geq$ 0.8;"
     " standard deviation over five seeds (stochastic estimators). All errors in SoH units"
     " (0.01 = 1 percentage point).\\label{tab:bench}}",
     "\\begin{adjustwidth}{-\\extralength}{0cm}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabularx}{\\textwidth}{llcrCrCrr}", "\\toprule",
     "\\textbf{Estimator} & \\textbf{Window} & \\textbf{$N$} & \\textbf{Pooled} & \\textbf{95\\% CI} &"
     " \\textbf{Equal-cell} & \\textbf{Per-cell range} & \\textbf{SoH $\\geq$ 0.8} & \\textbf{Seed SD} \\\\", "\\midrule"]
for mdl in order:
    x = best_cfg.loc[mdl]
    sd = "---" if mdl not in L.STOCHASTIC else f"{x['std']:.4f}"
    nn = "---" if mdl == "ridge_dur1" else str(int(x.n))
    T.append(f"\\texttt{{{NAME.get(mdl, mdl)}}} & {x.window} & {nn} & {x.rmse:.4f} & {x.lo:.4f}--{x.hi:.4f} & "
             f"{x['eq']:.4f} & {x.bestcell:.4f}--{x.worst:.4f} & {x.ge80:.4f} & {sd} \\\\")
T += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}", "\\end{table}"]
open(os.path.join(MAN, "table_bench.tex"), "w").write("\n".join(T) + "\n")

# ================================================================ representations
acc = d[d.seed == 0].groupby(["window", "n", "model", "rep"]).apply(pool).rename("rmse").reset_index()
cc = cst[cst.seed == 0].groupby(["window", "n", "model", "rep"]).agg(
    bytes_med=("model_bytes", "median"), bytes_min=("model_bytes", "min"), bytes_max=("model_bytes", "max"),
    ops=("ops_total", "median")).reset_index()
R = acc.merge(cc, on=["window", "n", "model", "rep"])
R.to_csv(os.path.join(RES, "r1_representations.csv"), index=False)
VALID = R[(R.rmse < 0.05)]          # drops the broken folded-int8 and int8 SVR variants
# frontier on model storage
front, best_ = [], np.inf
for _, x in VALID.sort_values(["bytes_med", "rmse"]).iterrows():
    if x.rmse < best_ - 5e-5:              # ties within 0.005 pp are not a new frontier point
        best_ = x.rmse; front.append(x)
F = pd.DataFrame(front); F.to_csv(os.path.join(RES, "r1_frontier.csv"), index=False)
# feature extraction cost per N
prep = {n: L.prep_ops(n) for n in L.NS}
def prep_total(n): return L.total(prep[n][0])
REPN = {"ref": "float64", "fp32": "float32", "int8": "int8", "int16": "int16", "fixed16": "int16 fixed-point"}
T = ["\\begin{table}[H]",
     "\\caption{Accuracy--storage frontier of executable representations. Every row pairs the error"
     " computed with that representation's own arithmetic and stored values with the bytes those values"
     " occupy (model data only; the median over the four folds, with the fold range where the model size"
     " is data-dependent). Regressor operations are per inference; feature-extraction operations are"
     " reported separately (Section~\\ref{sec:compute}). Only rows not beaten on both error and storage"
     " by a smaller representation are listed.\\label{tab:cheap}}",
     "\\begin{adjustwidth}{-\\extralength}{0cm}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabularx}{\\textwidth}{llccCCCC}", "\\toprule",
     "\\textbf{Estimator} & \\textbf{Representation} & \\textbf{Window} & \\textbf{$N$} & \\textbf{RMSE} &"
     " \\textbf{Model data (B)} & \\textbf{Regressor ops} & \\textbf{Feature ops} \\\\", "\\midrule"]
for _, x in F.iterrows():
    rng = "" if x.bytes_min == x.bytes_max else f" ({thou(x.bytes_min)}--{thou(x.bytes_max)})"
    fo = "1" if x.model == "ridge_dur1" else thou(prep_total(int(x.n)))
    nn = "---" if x.model == "ridge_dur1" else str(int(x.n))
    T.append(f"\\texttt{{{NAME.get(x.model, x.model)}}} & {REPN[x.rep]} & {x.window} & {nn} & {x.rmse:.4f} & "
             f"{thou(x.bytes_med)}{rng} & {thou(x.ops)} & {fo} \\\\")
T += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}", "\\end{table}"]
open(os.path.join(MAN, "table_cheap.tex"), "w").write("\n".join(T) + "\n")
mac("frontN", str(len(F)))
def rep(mdl, rp, w=None, n=None):
    x = R[(R.model == mdl) & (R.rep == rp)]
    if w: x = x[(x.window == w) & (x.n == n)]
    return x.loc[x.rmse.idxmin()]
x = rep("ridge_dur1", "fixed16"); mac("sfFixRMSE", f4(x.rmse)); mac("sfFixBytes", str(int(x.bytes_med)))
x = rep("ridge_dur1", "fp32"); mac("sfFpBytes", str(int(x.bytes_med)))
x = rep("ridge", "fp32"); mac("ridgeFpRMSE", f4(x.rmse)); mac("ridgeFpBytes", str(int(x.bytes_med)))
x = rep("ridge", "int8"); mac("ridgeQRMSE", f4(x.rmse)); mac("ridgeQBytes", str(int(x.bytes_med)))
mac("ridgeFoldRMSE", f"{acc[(acc.model=='ridge')&(acc.rep=='int8_folded')].rmse.min():.2f}")
mac("mlpFoldRMSE", f"{acc[(acc.model.str.startswith('mlp'))&(acc.rep=='int8_folded')].rmse.min():.2f}")
x = rep("svr", "int16"); mac("svrQRMSE", f4(x.rmse)); mac("svrQBytes", thou(x.bytes_med)); mac("svrQOps", thou(x.ops))
x = rep("svr", "fp32"); mac("svrFpBytes", thou(x.bytes_med))
mac("svrEightRMSE", f"{acc[(acc.model=='svr')&(acc.rep.isin(['int8','int8all']))].rmse.min():.2f}")
x = rep("hgb", "int8"); mac("hgbQRMSE", f4(x.rmse)); mac("hgbQBytes", thou(x.bytes_med)); mac("hgbQOps", thou(x.ops))
x = rep("hgb", "fp32"); mac("hgbFpRMSE", f4(x.rmse)); mac("hgbFpBytes", thou(x.bytes_med))
x = rep("rf", "int8"); mac("rfQBytes", f"{x.bytes_med/1e6:.1f}"); mac("rfQRMSE", f4(x.rmse))
x = rep("mlp8", "int8"); mac("mlpQRMSE", f4(x.rmse)); mac("mlpQBytes", str(int(x.bytes_med)))
x = rep("mlp8", "fp32"); mac("mlpFpBytes", str(int(x.bytes_med)))
nsv = cst[(cst.model == "svr") & (cst.rep == "fp32") & (cst.seed == 0)].n_sv
mac("nsvMin", str(int(nsv.min()))); mac("nsvMax", str(int(nsv.max())))
rfn = cst[(cst.model == "rf") & (cst.rep == "fp32") & (cst.seed == 0)]
mac("rfNodesMin", thou(rfn.internal.min() + rfn.leaves.min())); mac("rfNodesMax", thou(rfn.internal.max() + rfn.leaves.max()))
# quantisation deltas
for mdl, (a, bb) in {"ridge": ("fp32", "int8"), "mlp8": ("fp32", "int8"), "mlp32x16": ("fp32", "int8"),
                     "svr": ("fp32", "int16"), "rf": ("fp32", "int8"), "hgb": ("fp32", "int8")}.items():
    A = acc[(acc.model == mdl) & (acc.rep == a)].set_index(["window", "n"]).rmse
    B = acc[(acc.model == mdl) & (acc.rep == bb)].set_index(["window", "n"]).rmse
    rel = 100 * (B - A) / A
    k_ = mdl.replace("32x16", "big").replace("8", "small")
    sg = lambda v: f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}"
    mac(f"q{k_}Med", sg(rel.median())); mac(f"q{k_}Max", sg(rel.max()))
# prep cost
for n in L.NS: mac(f"prepOps{WORD[n]}", thou(prep_total(n)))
mac("prepDivFour", str(prep[4][0]["div"])); mac("prepConstFour", str(prep[4][1]))

# ================================================================ online + span
po = on.groupby(["window", "rate_s", "n", "qmode", "model"]).apply(pool).rename("rmse").reset_index()
mac("onBestWide", f4(po[(po.window == "wide") & (po.rate_s == 30)].rmse.min()))
mac("onBestNarrow", f4(po[(po.window == "narrow") & (po.rate_s == 30)].rmse.min()))
mac("onTrapzBestWide", f4(po[(po.window == "wide") & (po.qmode == "trapz")].rmse.min()))
for r_ in [30, 60, 120, 240]:
    mac(f"onSf{r_}", f4(po[(po.window == "wide") & (po.rate_s == r_) & (po.model == "ridge_dur1")].rmse.min()))
    mac(f"onHgb{r_}", f4(po[(po.window == "wide") & (po.rate_s == r_) & (po.model == "hgb")].rmse.min()))
mac("onKeptMax", str(int(on[on.rate_s == 30].kept_max.max())))
mac("onLost", str(int(on.cycles_lost.max())))
spc = sp[sp.cycle_set == "common"].groupby(["window", "model"]).apply(pool).rename("rmse").reset_index()
spd = sp[sp.cycle_set == "common"].groupby("window").dur_median_s.median() / 60
mac("spanNcyc", thou(sp[(sp.cycle_set == "common") & (sp.model == "ridge") & (sp.window == "s45")].ncyc.sum()))
mac("spanSoHmin", f"{sp[sp.cycle_set=='common'].soh_min.min():.2f}")
for wn, key_ in [("s15_mid", "Mid"), ("narrow", "Nar"), ("s15_hi", "Hi"), ("s15_lo", "Lo"), ("s45", "Wide")]:
    for mdl in ["ridge", "hgb", "ridge_dur1"]:
        mac(f"sp{key_}{mdl.replace('_','').replace('1','')}", f4(spc[(spc.window == wn) & (spc.model == mdl)].rmse.iloc[0]))
    mac(f"sp{key_}Dur", f"{spd[wn]:.0f}")

# screening variants
def scr(tag):
    x = pd.read_csv(os.path.join(RES, f"r1_loco_cells{tag}.csv")); x = x[(x.rep == "ref") & (x.seed == 0)]
    return x.groupby(["window", "n", "model"]).apply(pool).rename("r")
v_, c_, b_ = scr(""), scr("_causal"), scr("_band")
mac("scrCausalDiff", f"{(v_.groupby('model').min() - c_.groupby('model').min()).abs().max():.4f}")
mac("scrCausalDiffBest", f"{abs(v_.min() - c_.min()):.4f}")
mac("scrBandBest", f4(b_.min()))
bb = b_.reset_index(); mac("scrBandNdiff", f4(abs(bb[bb.n == 4].r.min() - bb[bb.n == 32].r.min())))
json.dump(dict(n_macros=len(OUT)), open(os.path.join(RES, "r1_numbers_meta.json"), "w"))
TR = os.path.join(RES, "r1_transfer_cells.csv")
if os.path.exists(TR):
    exec(open(os.path.join(L.HERE, "make_r1_transfer_numbers.py")).read())
open(os.path.join(MAN, "numbers.tex"), "w").write(
    "% generated by R1/code/make_r1_numbers.py -- do not edit by hand\n" + "\n".join(OUT) + "\n")
print("wrote", len(OUT), "macros")
