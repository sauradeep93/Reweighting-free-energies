"""
Build the PMF from a two-segment eABF trajectory (5-hexenyl radical
cyclization) via MBAR, using the adaptive_sampling package
(github.com/ochsenfeld-lab/adaptive_sampling).

Consolidates the per-MLIP analyze_omol_eabf_final.py / analyze_uma_eabf_final.py
scripts into one parameterized script -- the two were identical apart from
which MLIP's eABF logs they read and where segment 1 was truncated/segment 2
resumed from.

Usage:
    python analyze_eabf_pmf.py --mlip_name MACE-OMol \
        --seg1_log eabf_omol_800ps_seg1_ext.log \
        --seg2_log eabf_omol_800ps_seg2_ext.log \
        --checkpoint_ps 630.0 \
        --out_name hexenyl_omol_eabf_final_800ps

    python analyze_eabf_pmf.py --mlip_name UMA-OMol \
        --seg1_log eabf_uma_800ps_seg1_ext.log \
        --seg2_log eabf_uma_800ps_seg2_ext.log \
        --checkpoint_ps 450.0 \
        --out_name hexenyl_uma_eabf_final_800ps
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Liberation Sans"

parser = argparse.ArgumentParser()
parser.add_argument("--mlip_name", required=True, help="label for plot legend, e.g. 'MACE-OMol'")
parser.add_argument("--seg1_log", required=True, help="path to segment 1's _ext.log file")
parser.add_argument("--seg2_log", required=True, help="path to segment 2's _ext.log file")
parser.add_argument("--checkpoint_ps", type=float, required=True,
                     help="time (ps) at which segment 1 was truncated and segment 2 resumed")
parser.add_argument("--out_name", required=True,
                     help="base filename (no extension) for the output .npz/.pdf")
parser.add_argument("--out_dir", default="./analysis_eabf_final", help="output directory")
parser.add_argument("--adaptive_sampling_path", default=None,
                     help="path to a local clone of ochsenfeld-lab/adaptive_sampling, "
                          "if not already importable")
parser.add_argument("--dft_reference", type=float, default=50.87,
                     help="reference activation free energy (kJ/mol) to mark on the plot, "
                          "e.g. from Dietschreit et al., J. Chem. Phys. 156, 114105 (2022)")
args = parser.parse_args()

if args.adaptive_sampling_path:
    sys.path.append(args.adaptive_sampling_path)
from adaptive_sampling.processing_tools import mbar

os.makedirs(args.out_dir, exist_ok=True)

T = 300.0
ext_sigma = 0.05
EQUIL_PS = 20.0

d1 = np.genfromtxt(args.seg1_log, skip_header=1)
d1 = d1[d1[:, 0] <= args.checkpoint_ps]

d2 = np.genfromtxt(args.seg2_log, skip_header=1)
d2[:, 0] += args.checkpoint_ps

data = np.vstack([d1, d2])
time_ps = data[:, 0]
cv_full = data[:, 4]
la_full = data[:, 5]
mask = (time_ps > EQUIL_PS) & ~np.isnan(cv_full) & ~np.isnan(data[:, 2])
cv = cv_full[mask]
la = la_full[mask]
print(f"Concatenated: seg1 {len(d1)} + seg2 {len(d2)} = {len(data)} rows, "
      f"time range [{time_ps.min():.2f}, {time_ps.max():.2f}] ps")
print(f"Loaded {mask.sum()} of {len(mask)} frames after {EQUIL_PS} ps equilibration cut")

centers = np.arange(1.0, 6.0 + 1e-9, ext_sigma)
traj_list, index_list, meta_f = mbar.get_windows(centers, cv, la, ext_sigma, equil_temp=T)
n_per_bin = np.array([len(t) for t in traj_list])
print(f"Pseudo-windows: {len(centers)}, empty bins: {(n_per_bin == 0).sum()}, "
      f"min/max frames per bin: {n_per_bin.min()}/{n_per_bin.max()}")

trajs = [t.reshape(-1) for t in traj_list]
meta_f2 = meta_f.reshape(-1, 3)
keep = [i for i, t in enumerate(trajs) if len(t) > 0]
trajs = [trajs[i] for i in keep]
meta_f2 = meta_f2[keep]

exp_U, frames_per_traj = mbar.build_boltzmann(traj_list=trajs, meta_f=meta_f2, equil_temp=T)
weights = mbar.run_mbar(exp_U, frames_per_traj, outfreq=200, conv=1e-6, max_iter=int(1e6), device="cpu")

all_frames = np.concatenate(trajs)
grid = np.arange(1.3, 6.1, 0.1)
pmf, rho = mbar.pmf_from_weights(grid, all_frames, weights, equil_temp=T)

dval = grid[1] - grid[0]
edges = np.concatenate([grid - dval / 2, [grid[-1] + dval / 2]])
counts, _ = np.histogram(all_frames, bins=edges)
MIN_FRAMES = 100
reliable = counts >= MIN_FRAMES

pmf_ref = pmf - np.nanmin(pmf)
np.savez(f"{args.out_dir}/{args.out_name}.npz",
         grid=grid, pmf=pmf_ref, counts=counts, reliable=reliable, all_cv=all_frames)

fig, axs = plt.subplots(1, figsize=(8, 6))
idx = np.where(reliable)[0]
splits = np.where(np.diff(idx) > 1)[0] + 1
first = True
for seg in np.split(idx, splits):
    if len(seg) < 2:
        continue
    axs.plot(grid[seg], pmf_ref[seg], linewidth=4, color="#808080", linestyle="-",
             label=f"{args.mlip_name} eABF, 800 ps (n>={MIN_FRAMES}/bin)" if first else None)
    first = False
idx_u = np.where(~reliable)[0]
splits_u = np.where(np.diff(idx_u) > 1)[0] + 1
first = True
for seg in np.split(idx_u, splits_u):
    if len(seg) < 1:
        continue
    lo, hi = max(seg[0] - 1, 0), min(seg[-1] + 1, len(grid) - 1)
    span = np.arange(lo, hi + 1)
    axs.plot(grid[span], pmf_ref[span], linewidth=2, color="#808080", linestyle=":",
             label=f"{args.mlip_name} eABF (low-count)" if first else None, alpha=0.6)
    first = False
axs.axhline(args.dft_reference, color="#c0392b", linewidth=2, linestyle=":",
            label=rf"DFT reference $\Delta F^{{\ddagger}}$ = {args.dft_reference:.1f} kJ/mol")
axs.set_xticks(np.arange(1.0, 6.5, 1.0))
axs.set_xlim(1.3, 6.1)
axs.tick_params(axis="y", length=8, width=4, labelsize=18, pad=10, direction="in")
axs.tick_params(axis="x", length=8, width=4, labelsize=18, pad=10, direction="in")
axs.set_xlabel(r"$\xi\ (\mathrm{\AA})$", fontsize=20)
axs.set_ylabel(r"$\mathrm{A}(\xi)\ (\mathrm{kJ\,mol}^{-1})$", fontsize=20)
for s in ["bottom", "top", "left", "right"]:
    axs.spines[s].set_linewidth(3)
fig.tight_layout()
axs.legend(frameon=True, loc="upper center", fontsize=11)
plt.savefig(f"{args.out_dir}/{args.out_name}.pdf", dpi=600, bbox_inches="tight")


def val(x):
    return pmf_ref[np.argmin(np.abs(grid - x))]


#example printing tests.
#mask_ts = (grid > 1.9) & (grid < 2.6)
#j = np.nanargmax(np.where(mask_ts, pmf_ref, np.nan))
#print(f"barrier: xi={grid[j]:.3f}  raw={pmf_ref[j]:.2f}")
#print(f"product(1.56)={val(1.56):.2f}  plateau(4.9)={val(4.9):.2f}  product_vs_plateau={val(1.56) - val(4.9):.2f}")
#print(f"barrier_above_plateau={pmf_ref[j] - val(4.9):.2f}")
#print("DONE")
