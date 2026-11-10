"""Supplementary Tables S1-S10 for the R1 manuscript, generated from the R1 result CSVs."""
import os, json, warnings
import numpy as np, pandas as pd
import r1_lib as L
warnings.filterwarnings("ignore")
RES = L.RES
OUT = os.path.join(L.R1, "supplementary")
os.makedirs(OUT, exist_ok=True)
def pool(x, col="sse"): return float(np.sqrt(x[col].sum() / x.ncyc.sum()))
def tt(s): return str(s).replace("_", r"\_")
NAME = {"ridge_dur1": "duration only"}
B = []
def table(caption, colspec, header, rows, label):
    B.append(r"\renewcommand{\thetable}{" + label.split(":")[1] + "}")
    B.append(r"\begin{longtable}{" + colspec + "}")
    B.append(r"\caption{" + caption + r"\label{" + label + r"}}\\")
    B.append(r"\toprule " + header + r"\\ \midrule\endfirsthead")
    B.append(r"\toprule " + header + r"\\ \midrule\endhead")
    B.append(r"\bottomrule\endfoot")
    B.extend(r + r"\\" for r in rows)
    B.append(r"\end{longtable}")

d = pd.read_csv(os.path.join(RES, "r1_loco_cells.csv"))
cst = pd.read_csv(os.path.join(RES, "r1_loco_costs.csv"))
bt = pd.read_csv(os.path.join(RES, "r1_boot_rmse.csv"))

# ---------------- S1 settings + convergence
rows = []
for m, h in L.HYPER.items():
    rows.append(rf"\texttt{{{m}}} & " + "; ".join(f"{tt(k)}={tt(v)}" for k, v in h.items()))
rows.append(r"duration only & Ridge on the window duration, alpha=1.0")
table("Estimator settings (scikit-learn 1.9.0). Settings not listed are library defaults. All were fixed"
      " before any evaluation and never tuned against a held-out cell.", r"lp{12.5cm}",
      r"\textbf{Estimator} & \textbf{Settings}", rows, "tab:S1")
c = cst[cst.rep == "fp32"]
rows = []
for m, g in c.groupby("model"):
    it = g.n_iter.dropna()
    rows.append(rf"\texttt{{{tt(NAME.get(m, m))}}} & {len(g)} & {int((~g.converged.astype(bool)).sum())} & "
                + (f"{int(it.min())}--{int(it.max())}" if len(it) else "---"))
table("Convergence of the fits of the primary benchmark (all budgets, folds and seeds). For the perceptrons"
      " the iteration count is the number of L-BFGS iterations (limit 4000); for support-vector regression it is"
      " the number of solver iterations; gradient boosting runs a fixed 100 iterations. No fit raised a"
      " convergence warning.", "lrrr", r"\textbf{Estimator} & \textbf{Fits} & \textbf{Not converged} & \textbf{Iterations}",
      rows, "tab:S1b")

# ---------------- S2 per-cell errors for every configuration
s0 = d[(d.rep == "ref") & (d.seed == 0)]
sd = (d[d.rep == "ref"].groupby(["window", "n", "model", "seed"]).apply(pool).rename("r").reset_index()
      .groupby(["window", "n", "model"]).r.std())
rows = []
for (w, n, m), g in s0.groupby(["window", "n", "model"], sort=False):
    pc = g.set_index("cell").rmse
    b = bt[(bt.window == w) & (bt.n == n) & (bt.model == m)].iloc[0]
    g80 = np.sqrt(g.sse_ge80.sum() / g.n_ge80.sum())
    sdv = sd.get((w, n, m), np.nan)
    rows.append(f"{w} & {n} & \\texttt{{{tt(NAME.get(m, m))}}} & " + " & ".join(f"{pc[c_]:.4f}" for c_ in L.MAIN_CELLS)
                + f" & {pool(g):.4f} & {b.lo:.4f}--{b.hi:.4f} & {pc.mean():.4f} & {g80:.4f} & "
                + ("---" if np.isnan(sdv) else f"{sdv:.4f}"))
table("Leave-one-cell-out RMSE (SoH) for every configuration: per held-out CS2 cell, pooled with the 95\\%"
      " cell-grouped bootstrap interval, equal-cell mean, pooled on cycles with SoH $\\geq$ 0.8, and standard"
      " deviation of the pooled RMSE over five seeds (stochastic estimators). Reference (float64) models, seed 0.",
      "llllllllllll", r"\textbf{Window} & \textbf{$N$} & \textbf{Model} & \textbf{CS2\_35} & \textbf{CS2\_36} &"
      r" \textbf{CS2\_37} & \textbf{CS2\_38} & \textbf{Pooled} & \textbf{95\% CI} & \textbf{Eq.-cell} &"
      r" \textbf{SoH$\geq$0.8} & \textbf{Seed SD}", rows, "tab:S2")

# ---------------- S3 storage encodings
enc = [("ridge, perceptrons", "float32", "weights and biases (float32), standardiser folded into layer 1", "$4P$"),
       ("ridge, perceptrons", "int8", "int8 weights, int32 biases, 2 float32 scales per layer, float32 standardiser", "$W + 4b + 8\\ell + 8F$"),
       ("ridge, perceptrons", "int8, folded", "as int8, standardiser folded, no standardiser stored (fails)", "$W + 4b + 8\\ell$"),
       ("support-vector", "float32", "support vectors, dual coefficients, intercept, $\\gamma$, standardiser", "$4(SF+S) + 8 + 8F$"),
       ("support-vector", "int16", "int16 SVs, int16 coefficients, $F{+}1$ float32 scales, intercept, $\\gamma$, standardiser", "$2SF + 2S + 4F + 12 + 8F$"),
       ("support-vector", "int8", "int8 SVs, int16 coefficients (fails)", "$SF + 2S + 4F + 12 + 8F$"),
       ("trees", "float32", "internal node: uint8 feature + float32 threshold + uint16 offset; leaf: uint8 marker + float32 value; uint32 per tree; float32 baseline", "$7I + 5\\Lambda + 4T + 4$"),
       ("trees", "int8", "internal node: uint8 feature + uint8 threshold + uint16 offset; leaf: uint8 marker + int16 value; uint32 per tree; baseline, leaf scale; per-feature offset and step", "$4I + 3\\Lambda + 4T + 8 + 8F$"),
       ("duration only", "float32", "slope and intercept", "8"),
       ("duration only", "int16 fixed-point", "int16 slope, int16 intercept, uint8 shift", "5")]
table("Storage encodings of the executable representations (model data only). $P$: parameters; $W$: weights;"
      " $b$: biases; $\\ell$: layers; $F$: features (10); $S$: support vectors; $I$, $\\Lambda$, $T$: internal nodes,"
      " leaves and trees.", r"llp{8.2cm}l", r"\textbf{Family} & \textbf{Representation} & \textbf{Stored values} & \textbf{Bytes}",
      [f"{a} & {b_} & {c_} & {e}" for a, b_, c_, e in enc], "tab:S3")
w8 = cst[(cst.window == "wide") & (cst.n == 8) & (cst.seed == 0)]
rows = []
for (m, r), g in w8.groupby(["model", "rep"]):
    rows.append(f"\\texttt{{{tt(NAME.get(m, m))}}} & {tt(r)} & {int(g.model_bytes.median()):,} ({int(g.model_bytes.min()):,}--{int(g.model_bytes.max()):,}) & {tt(g.detail.iloc[0])}".replace(",", "\\,"))
table("Storage of each representation for the wide window at $N=8$: median over the four folds (range).",
      r"llrp{7.6cm}", r"\textbf{Model} & \textbf{Representation} & \textbf{Bytes} & \textbf{Content (fold 1)}", rows, "tab:S3b")

# ---------------- S4 operation counts
rows = []
for n in L.NS:
    ops, kc = L.prep_ops(n)
    rows.append(f"{n} & " + " & ".join(str(ops[k]) for k in ["add", "sub", "mul", "div", "cmp", "abs", "sqrt"])
                + f" & {L.total(ops)} & {kc}")
table("Feature-extraction operations for the ten features on $N$ retained samples (plus the duration"
      " subtraction). Constants: least-squares weights for slope and quadratic coefficient on the uniform grid."
      " The duration-only estimator needs one subtraction.", "rrrrrrrrrr",
      r"$N$ & add & sub & mul & div & cmp & abs & sqrt & \textbf{Total} & \textbf{Constants}", rows, "tab:S4")
rows = []
for (m, r), g in w8.groupby(["model", "rep"]):
    o = json.loads(g.ops.iloc[0])
    rows.append(f"\\texttt{{{tt(NAME.get(m, m))}}} & {tt(r)} & " + " & ".join(
        str(int(round(o.get(k, 0)))) for k in ["mul", "add", "sub", "cmp", "exp", "div"]) + f" & {int(g.ops_total.median())}")
table("Regressor operations per estimate, wide window, $N=8$ (fold 1; tree comparisons are the mean path length"
      " over the test cycles). Operation counts are not latencies or energies.", "llrrrrrrr",
      r"\textbf{Model} & \textbf{Repr.} & mul & add & sub & cmp & exp & div & \textbf{Total}", rows, "tab:S4b")

# ---------------- S5 screening
rows = []
for tag, nm in [("", "centred median (primary)"), ("_causal", "trailing median (causal)"),
                ("_tol10", "centred median, 10\\%"), ("_band", "plausibility band only")]:
    x = pd.read_csv(os.path.join(RES, f"r1_loco_cells{tag}.csv")); x = x[(x.rep == "ref") & (x.seed == 0)]
    g = x.groupby(["window", "n", "model"]).apply(pool).rename("r").reset_index()
    ncyc = x[(x.window == "wide") & (x.n == 4) & (x.model == "ridge")].ncyc.sum()
    best = g.groupby("model").r.min()
    rows.append(f"{nm} & {ncyc:,}".replace(",", "\\,") + " & " + " & ".join(f"{best[m]:.4f}" for m in L.MODELS + ["ridge_dur1"])
                + f" & {g[g.n==4].r.min():.4f} & {g[g.n==32].r.min():.4f} & {g[g.window=='narrow'].r.min():.4f} & {g[g.window=='wide'].r.min():.4f}")
table("Screening sensitivity (CS2 leave-one-cell-out, best budget per estimator, seed 0).", "lrrrrrrrrrrrrr",
      r"\textbf{Screen} & \textbf{Cycles} & ridge & mlp8 & mlp32x16 & svr & rf & hgb & dur. & $N$=4 & $N$=32 & narrow & wide",
      rows, "tab:S5")
m_ = L.load_all(); m8 = m_[m_.cell.isin(L.MAIN_CELLS + L.CX2_CELLS)]
band = L.screen(m8, "band"); v1 = L.screen(m8, "v1"); key = set(zip(v1.cell, v1.cycle))
band["kept"] = [(c_, y_) in key for c_, y_ in zip(band.cell, band.cycle)]
rows = []
for c_, g in band.groupby("cell"):
    r_ = g[~g.kept]; k_ = g[g.kept]
    rows.append(f"{tt(c_)} & {len(r_)} & {100*(r_.v_dis_end<2.71).mean():.0f} & {100*(r_.cv_s<600).mean():.0f} & {100*(k_.cv_s<600).mean():.0f} & {r_.cv_s.median():.0f} & {k_.cv_s.median():.0f}")
table("Log metadata of the cycles removed by the rolling-median screen. Columns: removed cycles; share that"
      " reached the 2.7 V discharge cut-off; share whose constant-voltage charge phase lasted less than 600 s,"
      " among removed and among retained cycles; median constant-voltage duration (s) of removed and retained"
      " cycles. Cells CS2\\_36/37, CX2\\_16 log short constant-voltage phases in normal cycles as well.",
      "lrrrrrr", r"\textbf{Cell} & \textbf{Removed} & \textbf{Cut-off (\%)} & \textbf{Short CV, removed (\%)} &"
      r" \textbf{Short CV, kept (\%)} & \textbf{CV removed (s)} & \textbf{CV kept (s)}", rows, "tab:S5b")

# ---------------- S6 integer representations per configuration
acc = d[d.seed == 0].groupby(["window", "n", "model", "rep"]).apply(pool).rename("r").reset_index()
pv = acc.pivot_table(index=["window", "n", "model"], columns="rep", values="r")
rows = []
for (w, n, m), x in pv.iterrows():
    def f(k): return "---" if k not in x or np.isnan(x[k]) else f"{x[k]:.4f}"
    comp = "int16" if m == "svr" else ("fixed16" if m == "ridge_dur1" else "int8")
    fail = f("int8_folded") if m in ("ridge", "mlp8", "mlp32x16") else (f("int8") if m == "svr" else "---")
    rows.append(f"{w} & {n} & \\texttt{{{tt(NAME.get(m, m))}}} & {f('ref')} & {f('fp32')} & {f(comp)} ({tt(comp)}) & {fail}")
table("Pooled RMSE of each representation for every configuration (seed 0). Failure variant: int8 with the"
      " standardiser folded (ridge, perceptrons) or int8 support vectors (support-vector regression).",
      "lllllll", r"\textbf{Window} & \textbf{$N$} & \textbf{Model} & \textbf{float64} & \textbf{float32} &"
      r" \textbf{Compact} & \textbf{Failure variant}", rows, "tab:S6")

# ---------------- S7 causal acquisition
on = pd.read_csv(os.path.join(RES, "r1_online_cells.csv"))
po = on.groupby(["window", "rate_s", "n", "qmode", "model"]).apply(pool).rename("r").reset_index()
pv = po.pivot_table(index=["window", "rate_s", "qmode", "n"], columns="model", values="r")
lost = on.groupby(["window", "rate_s", "qmode", "n"]).cycles_lost.max()
kmax = on.groupby(["window", "rate_s", "qmode", "n"]).kept_max.max()
rows = []
for idx, x in pv.iterrows():
    rows.append(" & ".join(str(v) for v in idx) + " & " + " & ".join(
        "---" if np.isnan(x.get(m, np.nan)) else f"{x[m]:.4f}" for m in ["ridge", "mlp8", "svr", "rf", "hgb", "ridge_dur1"])
        + f" & {kmax[idx]} & {lost[idx]}")
table("Causal acquisition with a decimating buffer of capacity $2N$ (CS2 leave-one-cell-out, pooled RMSE)."
      " Charge: coulomb counter (counter) or trapezoidal integration of the retained current (trapz)."
      " Kept: largest number of retained samples; lost: cycles whose window held too few acquired samples.",
      "lllrrrrrrrrr", r"\textbf{Window} & \textbf{$\Delta t$ (s)} & \textbf{Charge} & \textbf{$N$} & ridge & mlp8 &"
      r" svr & rf & hgb & dur. & kept & lost", rows, "tab:S7")

# ---------------- S8 equal-span windows
sp = pd.read_csv(os.path.join(RES, "r1_span_cells.csv"))
rows = []
for (cs, wn), g in sp.groupby(["cycle_set", "window"]):
    r_ = g.groupby("model").apply(pool)
    rows.append(f"{cs} & {g.v_lo.iloc[0]:.2f}--{g.v_hi.iloc[0]:.2f} & {g.dur_median_s.median()/60:.0f} & "
                f"{g[g.model=='ridge'].ncyc.sum()} & " + " & ".join(f"{r_[m]:.4f}" for m in ["ridge", "svr", "hgb", "ridge_dur1"]))
table("Equal-span and nested windows, $N=8$, CS2 leave-one-cell-out. Common: the cycles in which every window"
      " is traversed; own: all cycles in which that window is traversed.", "llrrrrrr",
      r"\textbf{Cycle set} & \textbf{Window (V)} & \textbf{Time (min)} & \textbf{Cycles} & ridge & svr & hgb & dur.",
      rows, "tab:S8")

# ---------------- S9 transfer per cell
tr = pd.read_csv(os.path.join(RES, "r1_transfer_cells.csv"))
P = d[(d.rep == "ref") & (d.seed == 0)].groupby(["window", "n", "model"]).apply(pool).rename("r").reset_index()
sel = P.loc[P.groupby("model").r.idxmin()].set_index("model")
ab = pd.read_csv(os.path.join(RES, "r1_ablation_noI_loco.csv"))
abp = ab.groupby(["window", "n", "model"]).apply(pool).rename("r").reset_index()
sel8 = abp.loc[abp.groupby("model").r.idxmin()].set_index("model")
rows = []
for fs, S in [("raw", sel), ("noI", sel8)]:
    for m, s_ in S.iterrows():
        for scen in ["from_cs2", "from_cx2"] + (["cx2_loco", "mixed_loco"] if fs == "raw" else []):
            x = tr[(tr.scenario == scen) & (tr.seed == 0) & (tr.featset == fs) & (tr.model == m) & (tr.window == s_.window) & (tr.n == s_.n)]
            for _, r_ in x.iterrows():
                rows.append(f"{'ten' if fs=='raw' else 'eight'} & \\texttt{{{tt(NAME.get(m, m))}}} & {s_.window}/{s_.n} & {tt(scen)} & {tt(r_.cell)} & "
                            f"{r_.rmse_raw:.4f} & {r_.rmse_clip:.4f} & {r_.bias_clip:+.4f} & {r_.n_out}")
table("Transfer errors per test cell, raw and clipped to $[0.10, 1.05]$, with the configuration of each estimator"
      " selected on its CS2 leave-one-cell-out error (seed 0). Scenarios: from\\_cs2 (fitted on CS2\\_35--38),"
      " from\\_cx2 (fitted on CX2\\_16/34/36/37), cx2\\_loco, mixed\\_loco. Out: raw predictions outside the range.",
      "lllllrrrr", r"\textbf{Features} & \textbf{Model} & \textbf{Budget} & \textbf{Scenario} & \textbf{Cell} &"
      r" \textbf{Raw} & \textbf{Clipped} & \textbf{Bias} & \textbf{Out}", rows, "tab:S9")

# ---------------- S10 extrapolation diagnostics
dg = pd.read_csv(os.path.join(RES, "r1_transfer_diag.csv"))
x = dg[(dg.featset == "raw") & dg.cell.isin(L.CX2_CELLS)]
rows = []
for j, fn in enumerate(L.FEATNAMES):
    rows.append(f"{tt(fn)} & {x[f'sd_{j}'].median():.3g} & {x[f'zmax_{j}'].median():.1f} & {x[f'zmax_{j}'].max():.1f}")
table("Extrapolation of the CX2 transfer cells in the CS2 feature space (ten features, all budgets): training"
      " standard deviation (median over budgets) and the largest absolute standardised value on a CX2 cell"
      " (median and maximum over budgets). The current features are near-constant within CS2 and differ"
      " between designs.", "lrrr", r"\textbf{Feature} & \textbf{Training SD} & \textbf{max $|z|$, median} & \textbf{max $|z|$, max}",
      rows, "tab:S10")
A = pd.read_csv(os.path.join(RES, "r1_transfer_summary.csv"))
rows = []
for fs, g in A.groupby("featset"):
    rows.append(f"{ {'raw':'ten','norm':'normalised','noI':'eight'}[fs] } & {g.ratio.median():.1f} & {g.eclip.min():.4f} & {int((g.out>0).sum())} & {100*g.out.max():.0f}")
table("Transfer from CS2 to CX2\\_16/34/36/37 over all 84 configurations per feature set: median degradation"
      " (clipped transfer RMSE over clipped LOCO RMSE), best clipped RMSE, number of configurations with raw"
      " predictions outside the admissible range, and the largest share outside.", "lrrrr",
      r"\textbf{Features} & \textbf{Median ratio} & \textbf{Best RMSE} & \textbf{Configs out} & \textbf{Max out (\%)}",
      rows, "tab:S10b")

doc = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.8cm,landscape]{geometry}
\usepackage{longtable,booktabs,amsmath}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\LTcapwidth}{\textwidth}
\begin{document}
\begin{center}{\Large Supplementary Material}\\[4pt]
\textbf{Sensing- and Compute-Budget Limits of On-Board State-of-Health Estimation for Lithium-Ion
Batteries: A Cell-Wise Benchmark}\\[2pt]
F. G\"ul, E. C. \"Ozkat, K. El Haloui\end{center}
\noindent All values are generated by the released code from the result files of this version. Errors are
in SoH units (0.01 = 1 percentage point). Tables are numbered in the order referenced in the article
(S1--S10); tables with a letter suffix extend the table of the same number.
\small
""" + "\n".join(B) + "\n\\end{document}\n"
# letter-suffixed tables: renumber so that S1b etc. are printed as S1b
doc = doc.replace(r"\begin{longtable}", r"\begin{longtable}")
open(os.path.join(OUT, "supplementary.tex"), "w").write(doc)
print("supplementary.tex written,", len(B), "lines")
