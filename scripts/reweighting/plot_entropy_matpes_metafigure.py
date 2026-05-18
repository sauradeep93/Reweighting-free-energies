import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import sys

sys.path.append("/home/gridsan/smajumdar/NeuralForceField/li_zeo/some_dirac_files/adaptive_sampling")

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Liberation Sans"
matplotlib.rcParams["mathtext.it"] = "Liberation Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Liberation Sans:bold"

# --- Constants ---
T          = 450.0
beta_kjmol = 1.0 / (T * 8.314e-3)
kB_eV      = 8.617333262145e-5
beta_eV    = 1.0 / (kB_eV * T)
ev_kjmol   = 96.48531
dval       = 0.25
min_val    = -6.611871400257606
max_val    = -min_val
targets    = np.arange(min_val, max_val + dval, dval)
targets_sorted = sorted(targets)

base_dir   = "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files"
matpes_dir = f"{base_dir}/energy_frames_npz_mace-matpes"

# --- Load shared MBAR weights and CV ---
_npz_w  = np.load(f"{base_dir}/mbar_weights_downsampled_10k.npz")
_npz_cv = np.load(f"{base_dir}/all_traj_filtered.npz")
_w_keys  = sorted(_npz_w.keys(),  key=float)
_cv_keys = sorted(_npz_cv.keys(), key=float)
_mbar_w_by_win = [_npz_w[k]  for k in _w_keys]
_cv_by_win     = [_npz_cv[k] for k in _cv_keys]

# =============================================================================
# PANEL a — Entropy
# =============================================================================
_entropy_colors_list = ["#0000a5", "#ac5775", "#ffaf1e", "#FF5349", "#228B22", "#1f77b4", "#808080"]
dir_info = {
    f"{base_dir}/energy_frames_npz_uma-s-1":                      "UMA",
    f"{base_dir}/energy_frames_npz_orb_v3_conservative_inf_omat": "ORB",
    f"{base_dir}/energy_frames_npz_mace-mpa-0":                   "MACE-MPA",
    "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pet-data/energy_frames_npz_pet-mad_downsampled": "PET-MAD",
    f"{base_dir}/energy_frames_npz_mace_Li_finetuned":            "MACE-finetuned",
    f"{base_dir}/energy_frames_npz_mace-matpes":                  "MACE-MATPES",
    f"{base_dir}/energy_frames_npz_mace-mp0-d3":                  "MACE-MP0+D3",
}
label_color = {label: _entropy_colors_list[i] for i, label in enumerate(dir_info.values())}

entropy_data = {}
for directory_name, label in dir_info.items():
    all_cv_e, all_w_e, all_dU_e = [], [], []
    for i, target in enumerate(targets_sorted):
        try:
            npz = np.load(f"{directory_name}/selected_frames_window_{target}.npz")
            U_A = npz["pot_energy_A_filtered"]
            U_B = npz["pot_energy_B_filtered"]
            all_cv_e.append(_cv_by_win[i])
            all_w_e.append(_mbar_w_by_win[i])
            all_dU_e.append((U_B - U_A) * ev_kjmol)
        except Exception as e:
            print(f"  entropy skip {label} t={target:.3f}: {e}")
    if not all_cv_e:
        continue
    cv_e     = np.concatenate(all_cv_e)
    w_mbar   = np.concatenate(all_w_e)
    dU_kjmol = np.concatenate(all_dU_e)
    rel_diff  = dU_kjmol - dU_kjmol.min()
    weights_B = w_mbar * np.exp(-beta_kjmol * rel_diff)
    n_bins_e  = 54
    edges_e   = np.linspace(cv_e.min(), cv_e.max(), n_bins_e + 1)
    centers_e = 0.5 * (edges_e[:-1] + edges_e[1:])
    bin_idx   = np.clip(np.digitize(cv_e, edges_e) - 1, 0, n_bins_e - 1)
    valid_z, scores = [], []
    for b in range(n_bins_e):
        mask = bin_idx == b
        N_z  = mask.sum()
        if N_z < 2:
            continue
        W_bin = weights_B[mask]
        P_bin = W_bin / W_bin.sum()
        pos   = P_bin > 1e-300
        if pos.sum() < 2:
            continue
        S_z = -np.sum(P_bin[pos] * np.log(P_bin[pos])) / np.log(N_z)
        scores.append(S_z)
        valid_z.append(centers_e[b])
    if valid_z:
        entropy_data[label] = (np.array(valid_z), gaussian_filter1d(np.array(scores), 1.5))
    print(f"  {label}: {len(valid_z)} bins")

# =============================================================================
# PANEL b — MATPES PMFs comparison with Gaussian uncertainty
# =============================================================================
npz_base  = np.load("/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pmf_april9_umb_mace-mp0.npz")
grid_base = npz_base["grid"]
pmf_base  = npz_base["pmf"] - npz_base["pmf"].min()

npz_us    = np.load(f"{matpes_dir}/pmf_jan16_umb_mace-matpes.npz")
interp_us = interp1d(npz_us["grid"], npz_us["pmf"], kind="linear",
                     bounds_error=False, fill_value="extrapolate")
pmf_us    = interp_us(grid_base); pmf_us -= pmf_us.min()
mean_target = np.mean(pmf_us)

npz_mbar    = np.load(f"{matpes_dir}/pmf_mace-matpes_dec2025.npz")
interp_mbar = interp1d(npz_mbar["grid"], npz_mbar["pmf"], kind="linear",
                       bounds_error=False, fill_value="extrapolate")
pmf_mbar    = interp_mbar(grid_base); pmf_mbar -= pmf_mbar.min()
pmf_mbar   += (mean_target - np.mean(pmf_mbar))

valid_targets_g = []
delta_A_gauss   = []
se_gauss        = []
se_gauss_old    = []

for i, target in enumerate(targets_sorted):
    try:
        npz = np.load(f"{matpes_dir}/selected_frames_window_{target}.npz")
        U_A = npz["pot_energy_A_filtered"]
        U_B = npz["pot_energy_B_filtered"]

        # Use within-window MBAR weights (renormalized)
        w_win = _mbar_w_by_win[i].copy()
        w_win = w_win / w_win.sum()

        bdU = beta_eV * (U_B - U_A)          # dimensionless β·ΔU

        mu_w   = np.dot(w_win, bdU)
        var_w  = np.dot(w_win, (bdU - mu_w)**2)   # biased weighted variance

        # Bessel-corrected variance 
        bessel   = 1.0 - np.dot(w_win, w_win)  
        var_w_bc = var_w / bessel

        # dA uses Bessel-corrected variance
        dA_k = (1.0 / beta_eV) * (mu_w - var_w_bc / 2.0) * ev_kjmol

        # SE of weighted mean 
        se_mu  = np.sqrt(np.dot(w_win**2, (bdU - mu_w)**2))

        # SE of weighted variance 
        resid  = (bdU - mu_w)**2 - var_w_bc
        se_var = np.sqrt(np.dot(w_win**2, resid**2)) / bessel

        # Error propagation dA = (1/beta)(mu - var/2)
        se_k = np.sqrt((ev_kjmol / beta_eV)**2 * (se_mu**2 + (se_var / 2.0)**2))

        valid_targets_g.append(target)
        delta_A_gauss.append(dA_k)
        se_gauss.append(se_k)
 
    except Exception as e:
        print(e)

delta_A_gauss = np.array(delta_A_gauss)
se_gauss      = np.array(se_gauss)
se_gauss_old  = np.array(se_gauss_old)

# Remove inter-MLIP reference offset
delta_A_gauss -= np.mean(delta_A_gauss)

interp_g     = interp1d(valid_targets_g, delta_A_gauss, kind="linear",
                         bounds_error=False, fill_value="extrapolate")
interp_se    = interp1d(valid_targets_g, se_gauss,      kind="linear",
                         bounds_error=False, fill_value="extrapolate")

pmf_gauss    = pmf_base + interp_g(grid_base)
pmf_gauss   -= pmf_gauss.min()
pmf_gauss   += (mean_target - np.mean(pmf_gauss))
se_grid      = np.abs(interp_se(grid_base))   # ±1 SE on grid

# --- Energy-only (mean gap) correction ---
all_cv, all_w, all_dU = [], [], []
for i, target in enumerate(targets_sorted):
    try:
        npz = np.load(f"{matpes_dir}/selected_frames_window_{target}.npz")
        U_A = npz["pot_energy_A_filtered"]; U_B = npz["pot_energy_B_filtered"]
        all_cv.append(_cv_by_win[i])
        all_w.append(_mbar_w_by_win[i])
        all_dU.append((U_B - U_A) * ev_kjmol)
    except Exception as e:
        print(e)

cv = np.concatenate(all_cv)
w  = np.concatenate(all_w); w /= w.sum()
dU = np.concatenate(all_dU); dU -= np.dot(w, dU)

n_bins  = 55
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
pmf_meangap  = pmf_base + interp_corr(grid_base)
pmf_meangap -= pmf_meangap.min()
pmf_meangap += (mean_target - np.mean(pmf_meangap))

# --- Colors ---
color_us      = "#1f77b4"
color_mbar    = "#8B4DB5"
color_gauss   = "#2CA02C"
color_meangap = "#FFE900"

# =============================================================================
# Metafigure: 1×2
# =============================================================================
def style_pmf_ax(ax):
    ax.set_xticks(np.arange(-7.5, 7.5 + 1e-9, 2.5))
    ax.set_yticks(np.arange(-20, 61, 20))
    ax.tick_params(axis="both", length=8, width=4, labelsize=22, pad=10, direction="in")
    ax.set_xlabel(r"$\mathrm{z}\ (\mathrm{\AA})$", fontsize=22)
    ax.set_ylabel(r"$\mathrm{A}(z)\ (\mathrm{kJ\,mol}^{-1})$", fontsize=22)
    for sp in ax.spines.values():
        sp.set_linewidth(3)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Panel a: Entropy ---
ax = axes[0]
for label, (xt, ys) in entropy_data.items():
    ax.plot(xt, ys, lw=4, color=label_color[label], label=label)
ax.axhspan(0.0, 0.1, color="red", alpha=0.1, label="Very low phase\nspace overlap")
ax.set_xticks(np.arange(-7.5, 7.5 + 1e-9, 2.5))
ax.tick_params(axis="both", length=8, width=4, labelsize=24, pad=10, direction="in")
ax.set_xlabel(r"$\mathrm{z}\ (\mathrm{\AA})$", fontsize=24)
ax.set_ylabel(r"Entropy Score $\mathcal{S}(z)$", fontsize=24)
ax.set_ylim(0, 1.05)
for sp in ax.spines.values():
    sp.set_linewidth(3)
ax.legend(frameon=True, loc="upper center", fontsize=18, ncol=2)
ax.text(-0.14, 1.02, "a", transform=ax.transAxes,
        fontsize=26, fontweight="bold", va="top", ha="left")

# --- Panel b: MATPES 5-curve comparison + Gaussian ±2 SE ---
ax = axes[1]
ax.plot(grid_base, pmf_base,    lw=2, color="black",       label="MACE-MP0 (Source)")
ax.plot(grid_base, pmf_us,      lw=4, color=color_us,      label="MACE-MATPES (Target)")
ax.plot(grid_base, pmf_mbar,    lw=2, color=color_mbar,    linestyle="--", label="MACE-MATPES (Direct)")
ax.plot(grid_base, pmf_gauss,   lw=2, color=color_gauss,   linestyle="--", label="MACE-MATPES (Gaussian)")
ax.fill_between(grid_base,
                pmf_gauss - 2.0 * se_grid,
                pmf_gauss + 2.0 * se_grid,
                color=color_gauss, alpha=0.25)
ax.plot(grid_base, pmf_meangap, lw=4, color=color_meangap, linestyle="--", label="MACE-MATPES (Energy only)")
style_pmf_ax(ax)
ax.legend(frameon=True, fontsize=14, ncol=2,
          loc="upper center", bbox_to_anchor=(0.5, -0.18))
ax.text(-0.14, 1.02, "b", transform=ax.transAxes,
        fontsize=26, fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=4)
fig.subplots_adjust(bottom=0.22)
out_path = f"{base_dir}/entropy_matpes_metafigure_April2026.pdf"
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close()
