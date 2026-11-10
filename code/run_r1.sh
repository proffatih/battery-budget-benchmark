#!/usr/bin/env bash
# Reproduce every R1 result, table and data figure from the raw CALCE files.
#   CALCE_DIR  where the raw cells are (default ~/runs/HV0013_R1/data/calce)
#   RUN_OUT    where large intermediate arrays go (default ~/runs/HV0013_R1)
# Wall time on a 16-core laptop: about 20 minutes.
set -euo pipefail
cd "$(dirname "$0")"
export CALCE_DIR="${CALCE_DIR:-$HOME/runs/HV0013_R1/data/calce}"
export RUN_OUT="${RUN_OUT:-$HOME/runs/HV0013_R1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one BLAS thread per worker
mkdir -p "$RUN_OUT"
if [ ! -d "$CALCE_DIR/ex" ]; then
  bash dl_calce.sh "$CALCE_DIR"                                  # 8 cells of the benchmark
  for c in CS2_33 CS2_34 CX2_33 CX2_35 CX2_38; do                 # 5 cells of the shift decomposition
    curl -sSL "https://web.calce.umd.edu/batteries/data/$c.zip" -o "$CALCE_DIR/$c.zip"
    unzip -o -q "$CALCE_DIR/$c.zip" -d "$CALCE_DIR/ex_extra"
  done
fi
python3 extract_r1.py          # -> results/cycles_r1_all.csv, $RUN_OUT/windows_r1.npz
python3 pack_windows.py        # -> $RUN_OUT/windows_dense.npz (fast random access)
python3 add_cv_meta.py         # constant-voltage metadata used to explain the screen
python3 r1_loco.py             # primary benchmark, 5 seeds, all representations
for s in band causal tol10; do SCREEN=$s ONESEED=1 python3 r1_loco.py; done
python3 r1_bootstrap.py        # cell-grouped hierarchical bootstrap
python3 r1_transfer.py         # transfer scenarios and feature sets
python3 r1_ablation.py         # eight-feature set
python3 r1_online.py           # causal acquisition + equal-span windows
python3 make_r1_numbers.py     # every quoted number + tables
python3 make_r1_supplement.py  # Supplementary Tables S1-S10
python3 make_r1_figures.py; python3 make_fig1.py
