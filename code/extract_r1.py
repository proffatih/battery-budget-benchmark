"""R1 extractor: CALCE CS2/CX2 -> per-cycle metadata + charge windows.

Differences from the v1 extractor (code/extract_calce.py), which it reproduces
exactly for the resampled windows used by the v1 benchmark:

* every cycle is kept and flagged (plausibility band, rolling-median screen are
  applied later), so screening sensitivity can be evaluated;
* per-cycle protocol metadata are recorded from the raw log: median CC charge
  current, mean discharge current, discharge duration, end-of-discharge voltage,
  native logging interval inside the window;
* the raw, native-rate constant-current charge segment between 3.60 and 4.20 V
  (time, voltage, current, cumulative charge) is stored, so that online
  acquisition schemes and alternative windows can be simulated causally.

Usage: CALCE_DIR=<dir containing ex/> RUN_OUT=<dir> python3 extract_r1.py [cells]
"""
import os, re, sys, glob
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(HERE)
RAW = os.environ.get("CALCE_DIR", os.path.expanduser("~/runs/HV0013_R1/data/calce"))
RUN_OUT = os.environ.get("RUN_OUT", os.path.expanduser("~/runs/HV0013_R1"))
RES = os.path.join(R1, "results")
os.makedirs(RES, exist_ok=True)

WINDOWS = {"narrow": (3.90, 4.05), "medium": (3.80, 4.10), "wide": (3.70, 4.15)}
NS = [4, 8, 16, 32]
CAP_RANGE = {"CS2": (0.20, 1.30), "CX2": (0.30, 1.60)}
NOMINAL = {"CS2": 1.10, "CX2": 1.35}
SEG = (3.60, 4.20)          # raw CC segment kept for causal simulations

def file_date_key(p):
    m = re.search(r'_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$', os.path.basename(p))
    if not m: return (99, 99, 99)
    mo, d, y = map(int, m.groups())
    return (y, mo, d)

def load_cell(cell_dir):
    frames, offset = [], 0
    for f in sorted(glob.glob(os.path.join(cell_dir, "*.xlsx")), key=file_date_key):
        try:
            xl = pd.ExcelFile(f)
        except Exception as e:
            print("  skip", os.path.basename(f), e); continue
        parts = [xl.parse(s) for s in xl.sheet_names if s.lower().startswith("channel")]
        if not parts: continue
        d = pd.concat(parts, ignore_index=True)
        need = {"Cycle_Index", "Current(A)", "Voltage(V)", "Test_Time(s)",
                "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"}
        if not need.issubset(d.columns): continue
        d = d.dropna(subset=["Cycle_Index", "Voltage(V)", "Current(A)"])
        d["gcycle"] = d["Cycle_Index"].astype(int) + offset
        d["file"] = os.path.basename(f)
        offset = int(d["gcycle"].max())
        frames.append(d)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)

def resample_window(sub, n):
    t = sub["Test_Time(s)"].values.astype(float)
    t = t - t[0]
    if t[-1] <= 0: return None
    grid = np.linspace(0.0, t[-1], n)
    V = np.interp(grid, t, sub["Voltage(V)"].values.astype(float))
    I = np.interp(grid, t, sub["Current(A)"].values.astype(float))
    Q = np.interp(grid, t, sub["Charge_Capacity(Ah)"].values.astype(float))
    return V, I, Q - Q[0], t[-1]

def process(args):
    cell, cdir = args
    print("cell", cell, flush=True)
    d = load_cell(cdir)
    if d is None: return []
    fam = cell[:3]
    lo, hi = CAP_RANGE[fam]
    rows = []
    for g, c in d.groupby("gcycle"):
        c = c.sort_values("Test_Time(s)")
        dis = c[c["Current(A)"] < -0.01]
        if len(dis) >= 3:
            v = dis["Discharge_Capacity(Ah)"].values.astype(float)
            q = float(np.nanmax(v) - np.nanmin(v))
        else:
            q = np.nan
        row = {"cell": cell, "design": fam, "cycle": int(g), "q_dis": q,
               "file": c["file"].iloc[0],
               "in_band": bool(lo <= q <= hi) if np.isfinite(q) else False}
        # protocol metadata straight from the log
        if len(dis) >= 3:
            row["i_dis_mean"] = float(-dis["Current(A)"].mean())
            row["t_dis"] = float(dis["Test_Time(s)"].iloc[-1] - dis["Test_Time(s)"].iloc[0])
            row["v_dis_end"] = float(dis["Voltage(V)"].iloc[-1])
            row["v_dis_min"] = float(dis["Voltage(V)"].min())
        ch = c[(c["Current(A)"] > 0.01)].sort_values("Test_Time(s)")
        if len(ch) < 8:
            row["has_windows"] = False; rows.append(row); continue
        cc = ch[ch["Voltage(V)"] < 4.19]
        row["i_ch_cc"] = float(cc["Current(A)"].median()) if len(cc) else np.nan
        seg = ch[(ch["Voltage(V)"] >= SEG[0]) & (ch["Voltage(V)"] <= SEG[1])]
        row["_seg"] = np.stack([seg["Test_Time(s)"].values.astype(float),
                                seg["Voltage(V)"].values.astype(float),
                                seg["Current(A)"].values.astype(float),
                                seg["Charge_Capacity(Ah)"].values.astype(float)])
        ok = True
        for wname, (vlo, vhi) in WINDOWS.items():
            sub = ch[(ch["Voltage(V)"] >= vlo) & (ch["Voltage(V)"] <= vhi)]
            sub = sub.sort_values("Test_Time(s)")
            if len(sub) < 6 or sub["Voltage(V)"].iloc[-1] <= sub["Voltage(V)"].iloc[0]:
                ok = False; break
            row[f"n_raw_{wname}"] = len(sub)
            tt = sub["Test_Time(s)"].values.astype(float)
            row[f"dt_med_{wname}"] = float(np.median(np.diff(tt)))
            row[f"dt_max_{wname}"] = float(np.max(np.diff(tt)))
            row[f"i_mean_{wname}"] = float(sub["Current(A)"].mean())
            row[f"i_cv_{wname}"] = float(sub["Current(A)"].std() / max(sub["Current(A)"].mean(), 1e-9))
            for n in NS:
                r = resample_window(sub, n)
                if r is None: ok = False; break
                V, I, Q, dur = r
                row[f"w_{wname}_{n}_V"] = V; row[f"w_{wname}_{n}_I"] = I
                row[f"w_{wname}_{n}_Q"] = Q; row[f"w_{wname}_{n}_dur"] = dur
            if not ok: break
        row["has_windows"] = ok
        rows.append(row)
    # reference capacity exactly as v1: median of the first five cycles that pass
    # the band and have all windows (the v1 'retained' set)
    v1 = [r for r in rows if r["in_band"] and r["has_windows"]]
    qref = float(np.median([r["q_dis"] for r in v1[:5]])) if v1 else np.nan
    for r in rows:
        r["qref"] = qref
        r["soh"] = r["q_dis"] / qref if np.isfinite(r["q_dis"]) else np.nan
        r["nominal"] = NOMINAL[fam]
    print(f"  {cell}: {len(rows)} cycles, {len(v1)} v1-retained, Qref={qref:.4f}", flush=True)
    return rows

if __name__ == "__main__":
    jobs = []
    for sub in ["ex", "ex_extra"]:
        for p in sorted(glob.glob(os.path.join(RAW, sub, "*"))):
            if os.path.isdir(p):
                jobs.append((os.path.basename(p), p))
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with Pool(min(8, len(jobs))) as pool:
        out = pool.map(process, jobs)
    allrows = [r for rs in out for r in rs]
    meta = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                         for r in allrows])
    meta.to_csv(os.path.join(RES, "cycles_r1_all.csv"), index=False)
    arrs = {}
    for i, r in enumerate(allrows):
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                arrs[f"{i}|{k}"] = v
    np.savez_compressed(os.path.join(RUN_OUT, "windows_r1.npz"), **arrs)
    print("saved", len(allrows), "cycles")
