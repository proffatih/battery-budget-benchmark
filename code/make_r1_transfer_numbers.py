# Executed inside make_r1_numbers.py (shares mac, f4, thou, K, best_cfg, pd, np, L, RES, MAN).
# Transfer macros and Table 4. Every transferred error is reported raw and clipped;
# degradation ratios use clipped errors in BOTH numerator and denominator.
tr = pd.read_csv(os.path.join(RES, "r1_transfer_cells.csv"))
GROUP = {"CS2_33": "prot", "CS2_34": "prot",
         "CX2_34": "des", "CX2_36": "des", "CX2_37": "des", "CX2_38": "des",
         "CX2_16": "both", "CX2_33": "both", "CX2_35": "both"}
def pc(x, col): return float(np.sqrt(x[col].sum() / x.ncyc.sum()))
loco_clip = (d[(d.rep == "ref") & (d.seed == 0)].groupby(["window", "n", "model"])
             .apply(lambda x: np.sqrt(x.sse_clip.sum() / x.ncyc.sum())).rename("loco_clip").reset_index())
fc = tr[(tr.scenario == "from_cs2") & (tr.seed == 0)].copy()
cx = fc[fc.cell.isin(L.CX2_CELLS)]
A = (cx.groupby(["featset", "window", "n", "model"])
     .apply(lambda x: pd.Series(dict(eraw=pc(x, "sse_raw"), eclip=pc(x, "sse_clip"),
                                     bias=float((x.bias_clip * x.ncyc).sum() / x.ncyc.sum()),
                                     out=float(x.n_out.sum() / x.ncyc.sum()))))
     .reset_index()).merge(K[["window", "n", "model", "rmse"]], on=["window", "n", "model"]).merge(
     loco_clip, on=["window", "n", "model"])
A["ratio"] = A.eclip / A.loco_clip
raw = A[A.featset == "raw"]
mac("transferMedRatio", f"{raw.ratio.median():.1f}")
mac("transferMedRatioUnclipped", f"{(raw.eclip / raw.rmse).median():.1f}")
mac("transferRatioLo", f"{raw.ratio.min():.1f}"); mac("transferRatioHi", f"{raw.ratio.max():.0f}")
w_ = raw.loc[raw.eraw.idxmax()]
mac("transferRawWorst", f"{w_.eraw:.1f}"); mac("transferRawWorstModel", w_.model)
mac("transferRawWorstWindow", w_.window); mac("transferRawWorstN", str(int(w_.n)))
mac("transferOutMax", f"{100*raw.out.max():.0f}")
outm = raw[raw.out > 0].groupby("model").size()
mac("transferOutModels", ", ".join(f"\\texttt{{{m_}}} ({int(v)})" for m_, v in outm.items()))
mac("transferOutConfigs", str(int((raw.out > 0).sum())))
# oracle (CX2-selected, as in v1) vs CS2-selected
orc = raw.loc[raw.groupby("model").eclip.idxmin()].set_index("model")
mac("oracleBest", f4(orc.eclip.min())); mac("oracleBestModel", orc.eclip.idxmin())
sel = {m_: best_cfg.loc[m_] for m_ in best_cfg.index}
cs2sel = pd.DataFrame([raw[(raw.model == m_) & (raw.window == s_.window) & (raw.n == s_.n)].iloc[0]
                       for m_, s_ in sel.items()]).set_index("model")
mac("selBest", f4(cs2sel.eclip.min())); mac("selBestModel", cs2sel.eclip.idxmin())
mac("selMedRatio", f"{cs2sel.ratio.median():.1f}")
mac("selMedClip", f4(cs2sel.eclip.median()))
sf_ = cs2sel.loc["ridge_dur1"]; mac("sfTransfer", f4(sf_.eclip)); mac("sfTransferBias", f"{sf_.bias:+.4f}")
# extrapolation diagnosis of the worst raw configuration
dg = pd.read_csv(os.path.join(RES, "r1_transfer_diag.csv"))
dw = dg[(dg.featset == "raw") & (dg.window == w_.window) & (dg.n == w_.n) & (dg.model == w_.model) & dg.cell.isin(L.CX2_CELLS)]
zc = [c_ for c_ in dw.columns if c_.startswith("zmax_")]
zmax = dw[zc].max()
j = int(zmax.idxmax().split("_")[1])
mac("zmaxFeat", L.FEATNAMES[j].replace("_", "\\_")); mac("zmaxVal", f"{zmax.max():.0f}")
mac("zmaxSd", f"{dw[f'sd_{j}'].iloc[0]:.1e}")
mac("worstPmin", f"{dw.pmin.min():.1f}"); mac("worstPmax", f"{dw.pmax.max():.1f}")
zI = dg[(dg.featset == "raw") & dg.cell.isin(L.CX2_CELLS)]
mac("zImeanMed", f"{zI.zmax_2.median():.0f}")
# feature sets
for fs in ["raw", "norm", "noI"]:
    x = A[A.featset == fs]
    mac(f"tr{fs.capitalize()}Med", f"{x.ratio.median():.1f}"); mac(f"tr{fs.capitalize()}Best", f4(x.eclip.min()))
    mac(f"tr{fs.capitalize()}Out", f"{100*x.out.max():.0f}")
# decomposition with CS2-selected configurations (raw features)
def grp_err(fs, scen="from_cs2"):
    rows = []
    for m_, s_ in sel.items():
        x = tr[(tr.scenario == scen) & (tr.seed == 0) & (tr.featset == fs) & (tr.model == m_)
               & (tr.window == s_.window) & (tr.n == s_.n)].copy()
        x["group"] = x.cell.map(GROUP)
        r_ = {g_: pc(x[x.group == g_], "sse_clip") for g_ in ["prot", "des", "both"]}
        r_["cx2v1"] = pc(x[x.cell.isin(L.CX2_CELLS)], "sse_clip")
        r_.update(model=m_); rows.append(r_)
    return pd.DataFrame(rows).set_index("model")
D = grp_err("raw"); Dn = grp_err("norm"); Di = grp_err("noI")
for g_ in ["prot", "des", "both"]:
    mac(f"g{g_}Med", f4(D[g_].median())); mac(f"g{g_}NormMed", f4(Dn[g_].median())); mac(f"g{g_}NoIMed", f4(Di[g_].median()))
    mac(f"g{g_}Min", f4(D[g_].min())); mac(f"g{g_}Max", f4(D[g_].max()))
mac("locoSelMed", f4(pd.Series({m_: s_.rmse for m_, s_ in sel.items()}).median()))
# within-CX2 and mixed-design LOCO at the CS2-selected configurations
def scen_err(scen, cellset):
    out = {}
    for m_, s_ in sel.items():
        x = tr[(tr.scenario == scen) & (tr.seed == 0) & (tr.featset == "raw") & (tr.model == m_)
               & (tr.window == s_.window) & (tr.n == s_.n) & tr.cell.isin(cellset)]
        out[m_] = pc(x, "sse_clip")
    return pd.Series(out)
cxl = scen_err("cx2_loco", L.CX2_CELLS); mxc = scen_err("mixed_loco", L.CX2_CELLS); mxs = scen_err("mixed_loco", L.MAIN_CELLS)
fcx = scen_err("from_cx2", L.MAIN_CELLS)
mac("cxLocoMed", f4(cxl.median())); mac("cxLocoMin", f4(cxl.min())); mac("cxLocoMax", f4(cxl.max()))
mac("mixCXMed", f4(mxc.median())); mac("mixCSMed", f4(mxs.median())); mac("fromCXMed", f4(fcx.median()))
# per-cell CX2 errors for the CS2-selected ridge and hgb
for m_ in ["ridge", "hgb"]:
    s_ = sel[m_]
    x = tr[(tr.scenario == "from_cs2") & (tr.seed == 0) & (tr.featset == "raw") & (tr.model == m_)
           & (tr.window == s_.window) & (tr.n == s_.n)].set_index("cell")
    for c_ in ["CS2_33", "CS2_34", "CX2_16", "CX2_34", "CX2_36", "CX2_37", "CX2_38", "CX2_33", "CX2_35"]:
        mac(f"pc{m_}{c_.replace('_','')}", f4(x.loc[c_].rmse_clip))
# ---- eight-feature block (selection on CS2 LOCO of the eight-feature set)
ab = pd.read_csv(os.path.join(RES, "r1_ablation_noI_loco.csv"))
abp = ab.groupby(["window", "n", "model"]).apply(lambda x: np.sqrt(x.sse.sum() / x.ncyc.sum())).rename("rmse").reset_index()
abs_ = pd.read_csv(os.path.join(RES, "r1_ablation_noI_scen.csv"))
sel8 = abp.loc[abp.groupby("model").rmse.idxmin()].set_index("model")
mac("noIlocoBest", f4(sel8.rmse.min())); mac("noIlocoBestModel", sel8.rmse.idxmin())
cmp_ = pd.Series({m_: sel8.loc[m_].rmse - best_cfg.loc[m_].rmse for m_ in sel8.index})
mac("noIlocoDiffMin", f"{cmp_.min():+.4f}"); mac("noIlocoDiffMax", f"{cmp_.max():+.4f}")
def grp8():
    rows = []
    for m_, s_ in sel8.iterrows():
        x = tr[(tr.scenario == "from_cs2") & (tr.seed == 0) & (tr.featset == "noI") & (tr.model == m_)
               & (tr.window == s_.window) & (tr.n == s_.n)].copy()
        x["group"] = x.cell.map(GROUP)
        r_ = {g_: pc(x[x.group == g_], "sse_clip") for g_ in ["prot", "des", "both"]}
        for scen in ["cx2_loco", "mixed_loco"]:
            y_ = abs_[(abs_.scenario == scen) & (abs_.model == m_) & abs_.cell.isin(L.CX2_CELLS)]
            r_[scen] = pc(y_, "sse_clip")
        r_.update(model=m_, loco=s_.rmse); rows.append(r_)
    return pd.DataFrame(rows).set_index("model")
D8 = grp8()
for g_ in ["prot", "des", "both", "cx2_loco", "mixed_loco"]:
    mac(f"e{g_.replace('_','')}Med", f4(D8[g_].median()))
mac("edesTreeMin", f4(D8.loc[["rf", "hgb"], "des"].min())); mac("edesTreeMax", f4(D8.loc[["rf", "hgb"], "des"].max()))
mac("desTreeMin", f4(D.loc[["rf", "hgb"], "des"].min())); mac("desTreeMax", f4(D.loc[["rf", "hgb"], "des"].max()))
mac("desLinMin", f4(D.loc[["ridge", "svr", "mlp8", "mlp32x16"], "des"].min())); mac("desLinMax", f4(D.loc[["ridge", "svr", "mlp8", "mlp32x16"], "des"].max()))
D8.to_csv(os.path.join(RES, "r1_transfer_groups_noI_selected.csv"))
# per-cell CX2_16 / CX2_33 in within-CX2 LOCO (raw, CS2-selected)
x16 = tr[(tr.scenario == "cx2_loco") & (tr.seed == 0) & (tr.featset == "raw") & (tr.cell == "CX2_16")]
mac("cxSixteenMin", f4(min(x16[(x16.model == m_) & (x16.window == s_.window) & (x16.n == s_.n)].rmse_clip.iloc[0]
                          for m_, s_ in sel.items())))
# ---- Table 4
NAME_ = {"ridge_dur1": "duration only"}
T = ["\\begin{table}[H]",
     "\\caption{Transfer beyond the four CS2 training cells, pooled RMSE after clipping to $[0.10,\\,1.05]$."
     " Each estimator uses the budget selected on its CS2 leave-one-cell-out error alone; no target-cell"
     " information enters the choice. Protocol, design and design + protocol: models fitted on CS2\\_35--38"
     " (1C discharge) applied to CS2\\_33/34 (same design, 0.5C), CX2\\_34/36/37/38 (other design, 1C) and"
     " CX2\\_16/33/35 (other design, $\\leq$0.5C). Within-CX2: leave-one-cell-out over CX2\\_16/34/36/37."
     " Mixed: the same CX2 cells held out from a training set that also contains the four CS2 cells."
     " Upper block: the ten features; lower block: without the two current features. Raw (unclipped) and"
     " per-cell errors are given in Supplementary Table S9.\\label{tab:transfer}}",
     "\\begin{adjustwidth}{-\\extralength}{0cm}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabularx}{\\textwidth}{lCCCCCC}", "\\toprule",
     "\\textbf{Estimator} & \\textbf{CS2 LOCO} & \\textbf{Protocol shift} & \\textbf{Design shift} &"
     " \\textbf{Design + protocol} & \\textbf{Within-CX2 LOCO} & \\textbf{Mixed-design LOCO} \\\\", "\\midrule",
     "\\multicolumn{7}{l}{\\textit{Ten features}} \\\\"]
for m_ in ["ridge", "mlp8", "mlp32x16", "svr", "rf", "hgb", "ridge_dur1"]:
    T.append(f"\\texttt{{{NAME_.get(m_, m_)}}} & {sel[m_].rmse:.4f} & {D.loc[m_,'prot']:.4f} & {D.loc[m_,'des']:.4f} & "
             f"{D.loc[m_,'both']:.4f} & {cxl[m_]:.4f} & {mxc[m_]:.4f} \\\\")
T.append(f"Median & {pd.Series({m_: s_.rmse for m_, s_ in sel.items()}).median():.4f} & {D.prot.median():.4f} & "
         f"{D.des.median():.4f} & {D.both.median():.4f} & {cxl.median():.4f} & {mxc.median():.4f} \\\\")
T += ["\\midrule", "\\multicolumn{7}{l}{\\textit{Eight features (current features removed)}} \\\\"]
for m_ in ["ridge", "mlp8", "mlp32x16", "svr", "rf", "hgb"]:
    x = D8.loc[m_]
    T.append(f"\\texttt{{{m_}}} & {x.loco:.4f} & {x.prot:.4f} & {x.des:.4f} & {x.both:.4f} & {x.cx2_loco:.4f} & {x.mixed_loco:.4f} \\\\")
T.append(f"Median & {D8.loco.median():.4f} & {D8.prot.median():.4f} & {D8.des.median():.4f} & {D8.both.median():.4f} & "
         f"{D8.cx2_loco.median():.4f} & {D8.mixed_loco.median():.4f} \\\\")
T += ["\\bottomrule", "\\end{tabularx}", "\\end{adjustwidth}", "\\end{table}"]
open(os.path.join(MAN, "table_transfer.tex"), "w").write("\n".join(T) + "\n")
A.to_csv(os.path.join(RES, "r1_transfer_summary.csv"), index=False)
D.to_csv(os.path.join(RES, "r1_transfer_groups_raw.csv")); Dn.to_csv(os.path.join(RES, "r1_transfer_groups_norm.csv"))
Di.to_csv(os.path.join(RES, "r1_transfer_groups_noI.csv"))
