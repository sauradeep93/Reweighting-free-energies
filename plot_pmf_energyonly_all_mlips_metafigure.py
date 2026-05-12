import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Liberation Sans"
matplotlib.rcParams["mathtext.it"] = "Liberation Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Liberation Sans:bold"

ev_kjmol   = 96.48531
kB_eV      = 8.617333262145e-5
T          = 450.0
beta_eV    = 1.0 / (kB_eV * T)
dval       = 0.25
min_val    = -6.611871400257606
max_val    = -min_val
targets    = np.arange(min_val, max_val + dval, dval)
targets_sorted = sorted(targets)

base_dir = "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/april9_10000_downsampled_files"

# --- Base MACE-MP0 PMF ---
npz_base  = np.load("/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pmf_april9_umb_mace-mp0.npz")
grid_base = npz_base["grid"]
pmf_base  = npz_base["pmf"] - npz_base["pmf"].min()
mean_base = np.mean(pmf_base)

# --- Shared MBAR weights ---
_npz_w  = np.load(f"{base_dir}/mbar_weights_downsampled_10k.npz")
_npz_cv = np.load(f"{base_dir}/all_traj_filtered.npz")
_w_keys  = sorted(_npz_w.keys(),  key=float)
_cv_keys = sorted(_npz_cv.keys(), key=float)
_mbar_w_by_win = [_npz_w[k]  for k in _w_keys]
_cv_by_win     = [_npz_cv[k] for k in _cv_keys]

# --- MLIPs ---
dir_info = {
    base_dir + "/energy_frames_npz_uma-s-1/":                      ("UMA",            "#0000a5"),
    base_dir + "/energy_frames_npz_orb_v3_conservative_inf_omat/": ("ORB",            "#ac5775"),
    base_dir + "/energy_frames_npz_mace-mpa-0/":                   ("MACE-MPA",       "#ffaf1e"),
    "/home/gridsan/smajumdar/jobs/completed/april9_umb_0.8ns_55h2o_1li/pet-data/energy_frames_npz_pet-mad_downsampled": ("PET-MAD", "#FF5349"),
    base_dir + "/energy_frames_npz_mace_Li_finetuned":             ("MACE-finetuned", "#228B22"),
    base_dir + "/energy_frames_npz_mace-mp0-d3/":                  ("MACE-MP0+D3",    "#808080"),
}

color_gauss = "#2CA02C"

# =============================================================================
# Energy-only (mean gap) PMF 
# =============================================================================
def compute_energyonly_pmf(directory_name, n_bins=100):
    all_cv, all_w, all_dU = [], [], []
    for i, target in enumerate(targets_sorted):
        try:
            npz = np.load(f"{directory_name}/selected_frames_window_{target}.npz")
            U_A = npz["pot_energy_A_filtered"]
            U_B = npz["pot_energy_B_filtered"]
            all_cv.append(_cv_by_win[i])
            all_w.append(_mbar_w_by_win[i])
            all_dU.append((U_B - U_A) * ev_kjmol)
        except Exception:
            pass
    if not all_cv:
        return None, None, None

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
        w_bin  = w[mask]; w_bin /= w_bin.sum()
        dU_bin = dU[mask]
        mu_w   = np.dot(w_bin, dU_bin)
        correction[i] = mu_w

    finite = np.isfinite(correction)
    interp_corr  = interp1d(centers[finite], correction[finite],  kind="linear",
                             bounds_error=False, fill_value="extrapolate")
    pmf_corr  = pmf_base + interp_corr(grid_base)
    pmf_corr -= pmf_corr.min()
    pmf_corr += (mean_base - np.mean(pmf_corr))
    return grid_base, pmf_corr, None

# =============================================================================
# Gaussian approximation PMF + ±2 SE uncertainty
# =============================================================================
def compute_gaussian_pmf(directory_name):
    valid_targets, dA_list, se_list = [], [], []

    for i, target in enumerate(targets_sorted):
        try:
            npz = np.load(f"{directory_name}/selected_frames_window_{target}.npz")
            U_A = npz["pot_energy_A_filtered"]
            U_B = npz["pot_energy_B_filtered"]

            w_win  = _mbar_w_by_win[i].copy(); w_win /= w_win.sum()
            bdU    = beta_eV * (U_B - U_A)          # dimensionless β·ΔU

            mu_w   = np.dot(w_win, bdU)
            var_w  = np.dot(w_win, (bdU - mu_w) ** 2)   # biased variance

            # Bessel-corrected variance
            bessel   = 1.0 - np.dot(w_win, w_win)   
            var_w_bc = var_w / bessel

            dA_k  = (1.0 / beta_eV) * (mu_w - var_w_bc / 2.0) * ev_kjmol

            # SE of weighted mean
            se_mu  = np.sqrt(np.dot(w_win ** 2, (bdU - mu_w) ** 2))

            # SE of weighted variance 
            resid  = (bdU - mu_w) ** 2 - var_w_bc
            se_var = np.sqrt(np.dot(w_win ** 2, resid ** 2)) / bessel

            # Error propagation dA = (1/beta)(mu - var/2)
            se_k   = np.sqrt((ev_kjmol / beta_eV) ** 2 * (se_mu ** 2 + (se_var / 2.0) ** 2))

            valid_targets.append(target)
            dA_list.append(dA_k)
            se_list.append(se_k)
        except Exception:
            pass

    if not valid_targets:
        return None, None, None

    dA_arr = np.array(dA_list); dA_arr -= np.mean(dA_arr)
    se_arr = np.array(se_list)

    interp_g  = interp1d(valid_targets, dA_arr, kind="linear",
                          bounds_error=False, fill_value="extrapolate")
    interp_se = interp1d(valid_targets, se_arr,  kind="linear",
                          bounds_error=False, fill_value="extrapolate")

    pmf_g  = pmf_base + interp_g(grid_base)
    pmf_g -= pmf_g.min()
    pmf_g += (mean_base - np.mean(pmf_g))
    se_grid = np.abs(interp_se(grid_base))
    return grid_base, pmf_g, se_grid


def style_ax(ax):
    ax.set_xticks(np.arange(-7.5, 7.5 + 1e-9, 2.5))
    ax.set_yticks(np.arange(-20, 61, 20))
    ax.tick_params(axis="both", length=8, width=4, labelsize=24, pad=10, direction="in")
    ax.set_xlabel(r"$\mathrm{z}\ (\mathrm{\AA})$", fontsize=24)
    ax.set_ylabel(r"$\mathrm{A}(z)\ \mathrm{(kJ\ mol^{-1})}$", fontsize=24)
    for sp in ax.spines.values():
        sp.set_linewidth(3)


# =============================================================================
# Precompute all PMFs
# =============================================================================
results = {}
for directory_name, (label, color) in dir_info.items():
    _, pmf_eo, _       = compute_energyonly_pmf(directory_name)
    _, pmf_g, se_g     = compute_gaussian_pmf(directory_name)
    results[label]     = (color, pmf_eo, pmf_g, se_g)
    print(f"{label}: done")

# =============================================================================
# Meta-figure: 3 rows × 2 cols
# =============================================================================
panel_labels = ["a", "b", "c", "d", "e", "f"]
mlip_order   = ["UMA", "PET-MAD", "ORB", "MACE-MP0+D3", "MACE-MPA", "MACE-finetuned"]

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes_flat = axes.flatten()

for idx, (label, ax) in enumerate(zip(mlip_order, axes_flat)):
    color, pmf_eo, pmf_g, se_g = results[label]

    ax.plot(grid_base, pmf_base, lw=2, color="black",
            label="MACE-MP0 (Source)")
    ax.plot(grid_base, pmf_g,    lw=2, color=color_gauss, linestyle="--",
            label=f"{label} (Gaussian)")
    ax.fill_between(grid_base,
                    pmf_g - 2.0 * se_g,
                    pmf_g + 2.0 * se_g,
                    color=color_gauss, alpha=0.25)
    ax.plot(grid_base, pmf_eo,   lw=4, color=color, linestyle="--",
            label=f"{label} (Energy only)")

    style_ax(ax)
    ax.legend(frameon=True, loc="best", fontsize=16)
    ax.text(-0.14, 1.02, panel_labels[idx], transform=ax.transAxes,
            fontsize=26, fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=3, h_pad=4)
meta_path = f"{base_dir}/pmf_all_mlip_energyonly_metafigure.pdf"
plt.savefig(meta_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {meta_path}")
