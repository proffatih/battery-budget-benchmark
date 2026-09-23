"""CALCE CS2/CX2 -> per-cycle SoH + fixed-budget charge-window samples.

Sensing budget is defined as (voltage window, N samples): the BMS observes the
CC-charge voltage/current trace only inside [V_lo,V_hi] and keeps N uniformly
spaced samples.  Nothing outside that window is used by any model.
"""
import os, re, sys, glob, zipfile, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.environ.get("CALCE_DIR", os.path.join(ROOT, "data", "calce"))
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

WINDOWS = {"narrow": (3.90, 4.05), "medium": (3.80, 4.10), "wide": (3.70, 4.15)}
NS      = [4, 8, 16, 32]
CAP_RANGE = {"CS2": (0.20, 1.30), "CX2": (0.30, 1.60)}

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
        need = {"Cycle_Index","Current(A)","Voltage(V)","Test_Time(s)",
                "Charge_Capacity(Ah)","Discharge_Capacity(Ah)"}
        if not need.issubset(d.columns): continue
        d = d.dropna(subset=["Cycle_Index","Voltage(V)","Current(A)"])
        d["gcycle"] = d["Cycle_Index"].astype(int) + offset
        offset = int(d["gcycle"].max())
        frames.append(d)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)

def resample_window(sub, n):
    """sub: charge rows inside the voltage window, time-ordered.
    Returns (V,I,Q,dt) resampled to n uniform points in time."""
    t = sub["Test_Time(s)"].values.astype(float)
    t = t - t[0]
    if t[-1] <= 0: return None
    grid = np.linspace(0.0, t[-1], n)
    V = np.interp(grid, t, sub["Voltage(V)"].values.astype(float))
    I = np.interp(grid, t, sub["Current(A)"].values.astype(float))
    Q = np.interp(grid, t, sub["Charge_Capacity(Ah)"].values.astype(float))
    return V, I, Q - Q[0], t[-1]

def process(cell):
    cdir = os.path.join(RAW, "ex", cell)
    if not os.path.isdir(cdir): return None
    print("cell", cell, flush=True)
    d = load_cell(cdir)
    if d is None: return None
    fam = cell[:3]
    lo, hi = CAP_RANGE[fam]
    recs = {}
    for g, c in d.groupby("gcycle"):
        c = c.sort_values("Test_Time(s)")
        dis = c[c["Current(A)"] < -0.01]
        # Arbin logs cumulative capacity within a file, so the per-cycle value is
        # the increment across the discharge step, not the column maximum.
        if len(dis) >= 3:
            v = dis["Discharge_Capacity(Ah)"].values.astype(float)
            q = float(np.nanmax(v) - np.nanmin(v))
        else:
            q = np.nan
        if not (lo <= q <= hi):   # reject characterisation / partial cycles
            continue
        ch = c[(c["Current(A)"] > 0.01)].sort_values("Test_Time(s)")
        if len(ch) < 8: continue
        row = {"cell": cell, "cycle": int(g), "q_dis": q}
        ok = True
        for wname, (vlo, vhi) in WINDOWS.items():
            sub = ch[(ch["Voltage(V)"] >= vlo) & (ch["Voltage(V)"] <= vhi)]
            sub = sub.sort_values("Test_Time(s)")
            if len(sub) < 6 or sub["Voltage(V)"].iloc[-1] <= sub["Voltage(V)"].iloc[0]:
                ok = False; break
            row[f"n_raw_{wname}"] = len(sub)
            for n in NS:
                r = resample_window(sub, n)
                if r is None: ok = False; break
                V, I, Q, dur = r
                row[f"w_{wname}_{n}_V"] = V; row[f"w_{wname}_{n}_I"] = I
                row[f"w_{wname}_{n}_Q"] = Q; row[f"w_{wname}_{n}_dur"] = dur
            if not ok: break
        if ok:
            recs[int(g)] = row
    if not recs: return None
    rows = [recs[k] for k in sorted(recs)]
    qs = np.array([r["q_dis"] for r in rows])
    qref = float(np.median(qs[:5]))
    for r in rows: r["soh"] = r["q_dis"] / qref
    print(f"  {len(rows)} usable cycles, Qref={qref:.3f} Ah, SoH {min(r['soh'] for r in rows):.3f}-{max(r['soh'] for r in rows):.3f}", flush=True)
    return rows

if __name__ == "__main__":
    cells = sys.argv[1:] or [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RAW,"ex","*")))]
    allrows = []
    for c in cells:
        r = process(c)
        if r: allrows += r
    if not allrows:
        print("no data"); sys.exit(1)
    meta = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in allrows])
    meta.to_csv(os.path.join(OUT, "cycles.csv"), index=False)
    arrs = {}
    for i, r in enumerate(allrows):
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                arrs[f"{i}|{k}"] = v
    np.savez_compressed(os.path.join(OUT, "windows.npz"), **arrs)
    print("saved", len(allrows), "cycles ->", OUT)
