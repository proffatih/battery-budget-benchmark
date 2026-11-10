# Sensing- and compute-budget benchmark for on-board battery state-of-health estimation

Code and derived data for the paper *"Sensing- and Compute-Budget Limits of On-Board
State-of-Health Estimation for Lithium-Ion Batteries: A Cell-Wise Benchmark"*
(F. Gül, E. C. Özkat, K. El Haloui; *Batteries*, MDPI, under review).

The study asks how much of the constant-current charge an on-board estimator must observe
(**sensing budget**: a voltage window and the number `N` of samples retained in it) and how
much it must store and compute (**compute budget**), and how both compare with the error
caused by a change of cell population.

## Versions

| Tag / commit | Content |
|---|---|
| `2231baf` | version reviewed in round 1 (v1 scripts `benchmark.py`, `single_feature.py`, `transfer_clipped.py`, `make_tables.py`, `make_numbers.py`) |
| tag `r1` | revised analysis (scripts `r1_*.py`, `make_r1_*.py`, pipeline `code/run_r1.sh`) |

The v1 scripts are kept unchanged for traceability. The R1 analysis reproduces the v1 cycle
set exactly (10 530 cycles, identical labels and window durations) and replaces the v1
accounting where the reviewers found it inconsistent (see *Changes in R1*).

## Data

Raw cycling data are **not** redistributed. They are public from the
[CALCE Battery Research Group](https://calce.umd.edu/battery-data): cells CS2_35–38 (primary
leave-one-cell-out benchmark), CX2_16/34/36/37 (transfer tests) and CS2_33/34, CX2_33/35/38
(used only to separate a change of discharge protocol from a change of design).
`code/dl_calce.sh` and `code/run_r1.sh` download them.

`results/cycles_r1_all.csv` holds one row per extracted cycle (all 13 cells, before
screening) with discharge capacity, SoH, logged charge/discharge currents, end-of-discharge
voltage, constant-voltage phase duration and per-window sample counts and durations.

## Reproducing R1

```bash
pip install -r requirements.txt
bash code/run_r1.sh            # downloads the cells if needed; about 20 min on 16 cores
```

`run_r1.sh` runs, in order: `extract_r1.py` → `pack_windows.py` → `add_cv_meta.py` →
`r1_loco.py` (primary benchmark, 5 seeds, all executable representations; then the
screening variants) → `r1_bootstrap.py` → `r1_transfer.py` → `r1_ablation.py` →
`r1_online.py` → `make_r1_numbers.py` → `make_r1_supplement.py` → figures.
Large intermediate arrays go to `$RUN_OUT` (default `~/runs/HV0013_R1`).

Tested with Python 3.13.5 and the versions pinned in `requirements.txt`
(scikit-learn 1.9.0). Relative to the v1 environment (scikit-learn 1.8), deterministic
models (ridge, duration-only) reproduce the v1 errors exactly and the others to within
0.002 SoH.

## Changes in R1

* **Matched accuracy and storage.** Every byte count belongs to an executable representation
  whose predictions are computed from the stored values (`r1_lib.py`: `DenseFP32Folded`,
  `DenseInt8`, `SVRRep`, `TreeRep`, `DurFP32`, `DurInt16`). Biases, scales and the
  standardiser are counted at their real widths; tree encodings and the one-feature
  fixed-point model are explicit. v1 paired float accuracy with assumed int8 bytes.
* **Operations** are counted separately for the regressor and for feature extraction, with
  non-MAC operations (exponentials, comparisons, divisions) listed.
* **Uncertainty**: per-cell errors, equal-cell means, five seeds, and a two-level bootstrap
  grouped by cell (`r1_bootstrap.py`).
* **Causal acquisition** with a bounded decimating buffer and coarser acquisition intervals
  (`r1_online.py`), and equal-span windows to separate voltage region from traversal time.
* **Transfer** decomposed into protocol shift, design shift and within-design variation, with
  configurations selected on CS2 alone; raw and clipped errors; feature-set variants
  (normalised, without the current features) and extrapolation diagnostics.
* **Screening** justified with log metadata (truncated constant-voltage phases) and repeated
  with a causal screen and without the rolling screen.
* v1 `make_tables.py` filled the Q0 column with the maximum capacity; R1 reports the
  reference capacity used for SoH (median of the first five retained cycles).

## Notes on the protocol

* Splitting is **cell-wise**, never random over cycles.
* No cycle-index, calendar-time or cumulative-throughput feature is used.
* Per-cycle discharge capacity is the increment of the discharge-capacity column across the
  discharge step (the Arbin files accumulate capacity within a file).
* Costs are model data and operation counts. No microcontroller latency, energy, working
  memory or run-time overhead is measured or claimed.

## Citation

Gül, F.; Özkat, E.C.; El Haloui, K. *Sensing- and Compute-Budget Limits of On-Board
State-of-Health Estimation for Lithium-Ion Batteries: A Cell-Wise Benchmark.* Batteries
(under review), 2026.

## Licence

Code: MIT. Derived data (`results/*.csv`): CC BY 4.0. The underlying raw cycling data remain
subject to the CALCE terms of use.
