"""Repack windows_r1.npz (one zip member per array, slow random access) into
dense per-budget arrays and a ragged raw-segment store. Run once after
extract_r1.py."""
import os, zipfile, io
import numpy as np, pandas as pd
import r1_lib as L
meta = L.load_all(); nrow = len(meta)
src = os.path.join(L.RUN_OUT, "windows_r1.npz")
dense = {f"{w}_{n}": np.full((nrow, 3, n), np.nan) for w in L.WINDOWS for n in L.NS}
seg = [None] * nrow
ch = {"V": 0, "I": 1, "Q": 2}
with zipfile.ZipFile(src) as zf:
    for zi in zf.infolist():
        key = zi.filename[:-4]
        i, k = key.split("|", 1); i = int(i)
        arr = np.lib.format.read_array(io.BytesIO(zf.read(zi)))
        if k == "_seg":
            seg[i] = arr
        else:
            _, w, n, c = k.split("_")
            dense[f"{w}_{n}"][i, ch[c]] = arr
lens = np.array([0 if s is None else s.shape[1] for s in seg])
off = np.concatenate([[0], np.cumsum(lens)])
flat = np.concatenate([s for s in seg if s is not None], axis=1)
np.savez(os.path.join(L.RUN_OUT, "windows_dense.npz"), seg_flat=flat, seg_off=off, **dense)
print("packed", nrow, "rows; seg samples", flat.shape[1])
