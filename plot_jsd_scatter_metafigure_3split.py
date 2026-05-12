import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from matplotlib.lines import Line2D
import sys

sys.path.append("/home/gridsan/smajumdar/NeuralForceField/li_zeo/some_dirac_files/adaptive_sampling")
from adaptive_sampling import units

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Liberation Sans"
matplotlib.rcParams["mathtext.it"] = "Liberation Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Liberation Sans:bold"

# --- Constants ---
T          = 450.0
k_b        = 8.314e-3
ev_kjmol   = 96.48531
dval       = 0.25
min_val    = -6.611871400257606
max_val    = -min_val
targets    = np.arange(min_val, max_val + dval, dval)
targets_sorted = sorted(targets)

TS_HALF_WIDTH = 2

base_dir   = "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files"
matpes_dir = f"{base_dir}/energy_frames_npz_mace-matpes"

# --- Shared MBAR weights and CV ---
_npz_w  = np.load(f"{base_dir}/mbar_weights_downsampled_10k.npz")
_npz_cv = np.load(f"{base_dir}/all_traj_filtered.npz")
_w_keys  = sorted(_npz_w.keys(),  key=float)
_cv_keys = sorted(_npz_cv.keys(), key=float)
_mbar_w_by_win = [_npz_w[k]  for k in _w_keys]
_cv_by_win     = [_npz_cv[k] for k in _cv_keys]

# --- Base MP0 PMF (no smoothing) ---
npz_base  = np.load("/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pmf_april9_umb_mace-mp0.npz")
grid_base = npz_base["grid"]
pmf_base  = npz_base["pmf"] - npz_base["pmf"].min()

_min_bin, _max_bin = 20, -20
_ts_base = int(np.where(pmf_base == np.amax(pmf_base[_min_bin:_max_bin]))[0][0])
print(f"Base MACE-MP0 TS: bin {_ts_base}, z = {grid_base[_ts_base]:.3f} Å")

# --- mxi_inv --- #obtained from Majumdar et al.(https://doi.org/10.1021/acs.chemmater.5c02940) #can also be ignored for simplicity
mxi_inv = np.array([[0.14712433, 0.14696048, 0.14683356, 0.14680343, 0.14678452,
       0.14682748, 0.14674705, 0.14665919, 0.1467685 , 0.14720038,
       0.14789841, 0.14756012, 0.14706142, 0.14692932, 0.14749206,
       0.15070553, 0.15299035, 0.14967054, 0.14740485, 0.14696622,
       0.14675871, 0.14670729, 0.14669699, 0.1467601 , 0.14681597,
       0.1468169 , 0.14685265, 0.14687915, 0.14666933, 0.14656494,
       0.14669947, 0.14681865, 0.14686006, 0.14675526, 0.14666763,
       0.14670261, 0.14668833, 0.14672784, 0.14675579, 0.14677694,
       0.14679841, 0.14683466, 0.14683874, 0.14678778, 0.14671568,
       0.14664097, 0.1466181 , 0.14659731, 0.14664953, 0.14669839,
       0.14661507, 0.14646551, 0.14641242, 0.1464066 , 0.14640571],
      [1.49867877e-05, 5.66914385e-06, 3.00909394e-06, 2.22207074e-06,
       2.12867597e-06, 2.35773340e-06, 2.12327800e-06, 1.66644442e-06,
       2.39574860e-06, 5.84395879e-06, 9.13275265e-06, 7.95348953e-06,
       4.35918553e-06, 3.33894750e-06, 1.48701195e-05, 3.43163694e-05,
       4.04633365e-05, 3.07086237e-05, 1.14831531e-05, 3.16435438e-06,
       2.06396261e-06, 2.05528827e-06, 2.00584601e-06, 1.98063939e-06,
       1.97903113e-06, 2.44913716e-06, 3.41236581e-06, 4.05725836e-06,
       3.30268591e-06, 2.26071648e-06, 2.16352873e-06, 1.80747062e-06,
       2.08002396e-06, 2.09852402e-06, 1.74969221e-06, 1.86116740e-06,
       1.78038787e-06, 1.71637777e-06, 1.61990397e-06, 1.49343335e-06,
       1.36832216e-06, 1.28634149e-06, 1.30078056e-06, 1.49563866e-06,
       1.73447974e-06, 1.67007957e-06, 1.59605134e-06, 1.57666324e-06,
       1.86209504e-06, 2.23595053e-06, 2.49499195e-06, 1.69626059e-06,
       5.23119092e-07, 2.84507382e-07, 1.60203398e-07]])

lambda_xi = np.sqrt(units.h_in_SI * units.h_in_SI * mxi_inv[0] /
                    (2.0 * np.pi * units.atomic_to_kg * units.kB_in_SI * T))#can be ignored for simplicity
lambda_xi *= 1e10

# =============================================================================
# Shared functions 
# =============================================================================
def compute_3split_pmf(directory_name, ts_idx=None, ts_half_width=TS_HALF_WIDTH):
    if ts_idx is None:
        ts_idx = _ts_base
    z_lo = grid_base[max(0, ts_idx - ts_half_width)]
    z_hi = grid_base[min(len(grid_base) - 1, ts_idx + ts_half_width)]

    all_cv, all_w, all_dU = [], [], []
    for i, target in enumerate(targets_sorted):
        try:
            npz = np.load(f"{directory_name}/selected_frames_window_{target}.npz")
            all_cv.append(_cv_by_win[i])
            all_w.append(_mbar_w_by_win[i])
            all_dU.append((npz["pot_energy_B_filtered"] - npz["pot_energy_A_filtered"]) * ev_kjmol)
        except Exception:
            pass
    if not all_cv:
        return None

    cv = np.concatenate(all_cv)
    w  = np.concatenate(all_w);  w /= w.sum()
    dU = np.concatenate(all_dU); dU -= np.dot(w, dU)

    def region_mean(mask):
        w_r = w[mask]
        return 0.0 if w_r.sum() == 0 else np.dot(w_r / w_r.sum(), dU[mask])

    corr_react = region_mean(cv < z_lo)
    corr_ts    = region_mean((cv >= z_lo) & (cv <= z_hi))
    corr_prod  = region_mean(cv > z_hi)

    correction = np.where(grid_base < z_lo, corr_react,
                 np.where(grid_base <= z_hi, corr_ts, corr_prod))
    pmf_corr = pmf_base + correction
    pmf_corr -= pmf_corr.min()
    return pmf_corr

def compute_binbybin_pmf(directory_name, n_bins=100):
    """Bin-by-bin mean energy correction, no smoothing — used for JSD."""
    all_cv, all_w, all_dU = [], [], []
    for i, target in enumerate(targets_sorted):
        try:
            npz = np.load(f"{directory_name}/selected_frames_window_{target}.npz")
            all_cv.append(_cv_by_win[i])
            all_w.append(_mbar_w_by_win[i])
            all_dU.append((npz["pot_energy_B_filtered"] - npz["pot_energy_A_filtered"]) * ev_kjmol)
        except Exception:
            pass
    if not all_cv:
        return None

    cv = np.concatenate(all_cv)
    w  = np.concatenate(all_w);  w /= w.sum()
    dU = np.concatenate(all_dU); dU -= np.dot(w, dU)

    edges   = np.linspace(grid_base.min(), grid_base.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.clip(np.digitize(cv, edges) - 1, 0, n_bins - 1)

    correction = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = idx == i
        if mask.sum() < 2:
            continue
        w_bin = w[mask]; w_bin /= w_bin.sum()
        correction[i] = np.dot(w_bin, dU[mask])

    finite      = np.isfinite(correction)
    interp_corr = interp1d(centers[finite], correction[finite], kind="linear",
                           bounds_error=False, fill_value="extrapolate")
    pmf_corr = pmf_base + interp_corr(grid_base)
    pmf_corr -= pmf_corr.min()
    return pmf_corr

def load_simulated_pmf(npz_path, grid_key="grid", pmf_key="pmf"):
    d   = np.load(npz_path)
    interp = interp1d(d[grid_key], d[pmf_key], kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    pmf = interp(grid_base); pmf -= pmf.min()
    return pmf

def reaction_freeE(pmf, T=450.0, min_bin=20, max_bin=-20, TS=None):
    RT  = (units.R_in_SI * T) / 1000.0
    pmf = pmf[~np.isnan(pmf)]
    if TS is None:
        TS = int(np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0])
    P   = np.exp(-pmf / RT); P /= P.sum()
    P_a = P[:TS].sum(); P_b = P[(TS + 1):].sum()
    return RT * np.log(P_a / P_b), pmf[TS:].min() - pmf[:TS].min(), TS

def activation_freeE(pmf, lx, T=450.0, min_bin=20, max_bin=-20, TS=None):
    RT  = (units.R_in_SI * T) / 1000.0
    pmf = pmf[~np.isnan(pmf)]
    if TS is None:
        TS = int(np.where(pmf == np.amax(pmf[min_bin:max_bin]))[0][0])
    rho = np.exp(-pmf / RT); P = rho / rho.sum()
    P_a = P[:TS].sum()
    dA_a2b = -RT * np.log((rho[TS] * lx[TS]) / P_a)
    return dA_a2b, pmf[TS] - pmf[:TS].min(), TS

def pmf_to_prob(pmf):
    w = np.exp(-(pmf - pmf.min()) / (k_b * T))
    return w / w.sum()

def jsd_metric(pmf_a, pmf_b):
    return jensenshannon(pmf_to_prob(pmf_a), pmf_to_prob(pmf_b), base=np.e) ** 2

# =============================================================================
# MLIP definitions
# =============================================================================
MATPES_SIM_LABEL = r"MACE-MATPES$_\mathrm{sim}$"

energy_only_mlips = [
    (f"{base_dir}/energy_frames_npz_uma-s-1",                      "UMA",            "#0000a5"),
    (f"{base_dir}/energy_frames_npz_orb_v3_conservative_inf_omat", "ORB",            "#ac5775"),
    (f"{base_dir}/energy_frames_npz_mace-mpa-0",                   "MACE-MPA",       "#ffaf1e"),
    ("/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pet-data/energy_frames_npz_pet-mad_downsampled",
                                                                    "PET-MAD",        "#FF5349"),
    (f"{base_dir}/energy_frames_npz_mace_Li_finetuned",            "MACE-finetuned", "#228B22"),
    (f"{base_dir}/energy_frames_npz_mace-matpes",                  "MACE-MATPES",    "#1f77b4"),
    (f"{base_dir}/energy_frames_npz_mace-mp0-d3",                  "MACE-MP0+D3",    "#808080"),
]

simulated_mlips = [
    ("/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pmf_april9_umb_mace-mp0.npz",
     "MACE-MP0",         "black",   "grid", "pmf"),
    (f"{matpes_dir}/pmf_jan16_umb_mace-matpes.npz",
     "MACE-MATPES (US)", "#1f77b4", "grid", "pmf"),
]

# =============================================================================
# Compute all PMFs and barriers
# pmf_dict  → bin-by-bin PMFs (for JSD heatmap)
# scatter_results → 3-split PMFs (for barriers scatter) #Refer to article for reasoning
# =============================================================================
print("Computing bin-by-bin PMFs for JSD + 3-split PMFs for barriers ...")
pmf_dict      = {"MACE-MP0": pmf_base}
scatter_results = []

for directory, label, color in energy_only_mlips:
    print(f"  {label} ...", end=" ", flush=True)
    pmf_bb = compute_binbybin_pmf(directory)
    pmf_3s = compute_3split_pmf(directory)
    if pmf_bb is None or pmf_3s is None:
        print("FAILED"); continue
    pmf_dict[label] = pmf_bb          # bin-by-bin → JSD
    dA, _, TS       = reaction_freeE(pmf_3s, T=T)
    dA_act, _, _    = activation_freeE(pmf_3s, lambda_xi, T=T, TS=TS)
    scatter_results.append((label, color, dA, dA_act, "o"))
    print(f"ΔA={dA:.1f}  ΔA‡={dA_act:.1f}  TS=bin{TS}")

print("Loading simulated PMFs ...")
for npz_path, label, color, gkey, pkey in simulated_mlips:
    pmf_s = load_simulated_pmf(npz_path, gkey, pkey)
    if "jan16" in npz_path:
        pmf_dict[MATPES_SIM_LABEL] = pmf_s   # simulated → JSD
    dA, _, TS       = reaction_freeE(pmf_s, T=T)
    dA_act, _, _    = activation_freeE(pmf_s, lambda_xi, T=T, TS=TS)
    scatter_results.append((label, color, dA, dA_act, "*"))
    print(f"  {label}: ΔA={dA:.1f}  ΔA‡={dA_act:.1f}  TS=bin{TS}")

# =============================================================================
# JSD matrix
# =============================================================================
labels_jsd = [
    "MACE-MP0",
    "UMA",
    "ORB",
    "PET-MAD",
    "MACE-MPA",
    "MACE-MP0+D3",
    "MACE-MATPES",
    MATPES_SIM_LABEL,
    "MACE-finetuned",
]
n = len(labels_jsd)
jsd_matrix = np.zeros((n, n))
print("\nComputing pairwise JSD ...")
for i, li in enumerate(labels_jsd):
    for j, lj in enumerate(labels_jsd):
        if j >= i:
            val = jsd_metric(pmf_dict[li], pmf_dict[lj])
            jsd_matrix[i, j] = val
            jsd_matrix[j, i] = val

df_jsd = pd.DataFrame(jsd_matrix, index=labels_jsd, columns=labels_jsd)
print(df_jsd.round(3).to_string())

# =============================================================================
# Figure: 1×2 metafigure
# =============================================================================
fig = plt.figure(figsize=(22, 8))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.4)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# --- Panel (a): JSD heatmap ---
divider = make_axes_locatable(ax_a)
cax     = divider.append_axes("right", size="5%", pad=0.15)

sns.heatmap(df_jsd, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5,
            annot_kws={"size": 16}, ax=ax_a, cbar_ax=cax,
            cbar_kws={"label": "JS Divergence"})

cax.yaxis.label.set_size(22)
cax.tick_params(labelsize=18)
for sp in ax_a.spines.values():
    sp.set_linewidth(3)
ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=45, ha="right", fontsize=16)
ax_a.set_yticklabels(ax_a.get_yticklabels(), rotation=0, fontsize=16)
ax_a.tick_params(axis="both", length=0)
ax_a.text(-0.14, 1.02, "a", transform=ax_a.transAxes,
          fontsize=26, fontweight="bold", va="top", ha="left")

label_annot = {
    "UMA":               ((  0,   11), "center"),  # above
    "ORB":               ((  0,  -16), "center"),  # below
    "MACE-MPA":          (( 10,    4), "left"),    # right
    "PET-MAD":           ((-10,    4), "right"),   # left
    "MACE-finetuned":    ((-10,    4), "right"),   # left 
    "MACE-MATPES":       (( 10,  -13), "left"),    # right, lower 
    "MACE-MP0+D3":       ((-10,  -14), "right"),   # left, below 
    "MACE-MP0":          ((-10,    4), "right"),   # left
    "MACE-MATPES (US)":  (( 10,   13), "left"),    # right, upper
}

# Display label overrides (internal key unchanged for label_annot lookup)
label_display = {"MACE-MATPES (US)": "MACE-MATPES"}

for label, color, dA, dA_act, marker in scatter_results:
    ms = 220 if marker == "o" else 350
    ax_b.scatter(dA, dA_act, color=color, marker=marker, s=ms, zorder=5,
                 edgecolors="k" if marker == "*" else "none", linewidths=0.8)
    xytext, ha = label_annot.get(label, ((8, 4), "left"))
    ax_b.annotate(label_display.get(label, label), (dA, dA_act),
                  textcoords="offset points", xytext=xytext,
                  fontsize=16, ha=ha, va="center")

leg_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=13, label="Reweighted"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
           markeredgecolor="k", markersize=19, label="Simulated"),
]
ax_b.legend(handles=leg_handles, frameon=True, fontsize=20, loc="upper left")

ax_b.set_xlim(-23, -10)
ax_b.set_ylim(0, 35)
ax_b.set_xticks([-22, -20, -18, -16, -14, -12, -10])
ax_b.tick_params(axis="both", length=8, width=4, labelsize=22, pad=10, direction="in")
ax_b.set_xlabel(r"Reaction free energy $(\mathrm{kJ\,mol}^{-1})$", fontsize=22)
ax_b.set_ylabel(r"Forward activation energy $(\mathrm{kJ\,mol}^{-1})$", fontsize=22)
for sp in ax_b.spines.values():
    sp.set_linewidth(3)
ax_b.text(-0.18, 1.02, "b", transform=ax_b.transAxes,
          fontsize=26, fontweight="bold", va="top", ha="left")

fig.tight_layout()
out_path = f"{base_dir}/jsd_barriers_metafigure_3split_April2026.pdf"
fig.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
