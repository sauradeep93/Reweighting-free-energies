import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Liberation Sans"
matplotlib.rcParams["mathtext.it"] = "Liberation Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Liberation Sans:bold"

ev_kjmol_per_atom = 96.485 / 601   #(601 atoms in system)
data_dir = "/home/gridsan/smajumdar/NeuralForceField/li_zeo/dft_mace_compare/dft_li_data"
out_dir  = "/home/gridsan/smajumdar/NeuralForceField/li_zeo/dft_mace_compare"

mlip_info = [
    ("dft_mace_mp0_li_1100_frames_no_dispersion_data.npz",  "MACE-MP0"),
    ("dft_mace_mp0_li_1100_frames_yes_dispersion_data.npz", "MACE-MP0+D3"),
    ("dft_mace-mpa-0_li_1100_frames_data.npz",              "MACE-MPA"),
    ("dft_mace-matpes_li_1100_frames_data.npz",             "MACE-MATPES"),
    ("dft_mace-finetuned_li_1100_frames_data.npz",          "MACE-finetuned"),
    ("dft_uma-s-1_li_1100_frames_data.npz",                 "UMA"),
    ("dft_orbv3_li_1100_frames_data.npz",                   "ORB"),
    ("dft_pet-mad_li_1100_frames_data.npz",                 "PET-MAD"),
]


def load_energies(fname):
    d      = np.load(f"{data_dir}/{fname}")
    e_key  = "energies_mace_rel" if "energies_mace_rel" in d else "energies_mlip_rel"
    e_dft  = d["energies_dft_rel"].flatten() * ev_kjmol_per_atom
    e_mlip = d[e_key].flatten()              * ev_kjmol_per_atom
    return e_dft, e_mlip


def compute_mae(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.mean(np.abs(y[m] - x[m]))


all_data = []
for fname, label in mlip_info:
    e_dft, e_mlip = load_energies(fname)
    all_data.append((label, e_dft, e_mlip))
    print(f"{label}: MAE = {compute_mae(e_dft, e_mlip):.4f} kJ/mol/atom")

lo, hi   = 0.0, 2.8
xticks   = np.array([0.0, 0.7, 1.4, 2.1, 2.8])
panel_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

fig, axes = plt.subplots(2, 4, figsize=(30, 14))

hb_ref = None

for idx, (label, e_dft, e_mlip) in enumerate(all_data):
    row, col = divmod(idx, 4)
    ax = axes[row, col]

    hb = ax.hexbin(
        e_dft, e_mlip,
        gridsize=150,
        mincnt=1,
        cmap="magma",
        norm=colors.LogNorm(),
        extent=(lo, hi, lo, hi),
        linewidths=0,
    )
    if hb_ref is None:
        hb_ref = hb

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", length=8, width=4, labelsize=22,
                   pad=8, direction="in")
    for sp in ax.spines.values():
        sp.set_linewidth(3)

    ax.set_xlabel(
        r"$\mathrm{DFT\ Energy}\ (\mathrm{kJ\,mol}^{-1}\,\mathrm{atom}^{-1})$",
        fontsize=26)
    if col == 0:
        ax.set_ylabel(
            r"$\mathrm{MLIP\ Energy}\ (\mathrm{kJ\,mol}^{-1}\,\mathrm{atom}^{-1})$",
            fontsize=26)
    else:
        ax.tick_params(labelleft=False)

    mae = compute_mae(e_dft, e_mlip)
    ax.text(
        0.42, 0.87,
        rf"$\mathrm{{MAE}} = {mae:.2f}\ \mathrm{{kJ\,mol}}^{{-1}}\,\mathrm{{atom}}^{{-1}}$",
        transform=ax.transAxes, ha="center", va="top", fontsize=22,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax.text(
        0.5, 0.97, label,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=26, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    label_x = -0.22 if col == 0 else -0.08
    ax.text(label_x, 1.03, panel_labels[idx], transform=ax.transAxes,
            fontsize=26, fontweight="bold", va="top", ha="left")

    if idx == 0:
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Point count", fontsize=22)
        cbar.ax.tick_params(labelsize=18)

fig.tight_layout(w_pad=2, h_pad=3)
out_path = f"{out_dir}/energy_parity_8mlips.pdf"
fig.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
