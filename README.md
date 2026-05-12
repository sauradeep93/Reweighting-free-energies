# Reweighting-free-energies

This repository contains the code and data used in the article:

> **"Reweighting free energy profiles between universal machine learning interatomic potentials for fast consensus building"**
> Majumdar *et al.*, 2026

---

## Overview

Analysis scripts for reweighting umbrella sampling potential of mean force (PMF) profiles computed with one source MLIP (MACE-MP0) onto a suite of target MLIPs, using MBAR weights and perturbation theory. The studied system is a Li-ion diffusing through a water-solvated zeolite framework (601 atoms total) at T = 450 K, for battery applications

---

## Data

Processed umbrella sampling frames (MBAR weights and energy differences) are deposited on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20142456.svg)](https://doi.org/10.5281/zenodo.20142456)

Download and place the files under `data/` before running the scripts. Expected layout:

```
data/
  april9_10000_downsampled_files/
    selected_frames_window_{0..53}.npz   # MBAR weights + ΔU (source MLIP)
  energy_frames_npz_{mlip}/
    selected_frames_window_{0..53}.npz   # ΔU for each target MLIP
  pmf_april9_umb_mace-mp0.npz            # source PMF (MACE-MP0)
  pmf_jan16_umb_mace-matpes.npz          # target PMF from direct US (MACE-MATPES)
  pmf_mace-matpes_dec2025.npz            # MBAR-reweighted PMF (MACE-MATPES)
  dft_li_data/
    dft_li_1100_frames_energy_PBE.npz    # DFT PBE reference energies
```

Each `selected_frames_window_N.npz` contains:
- `pot_energy_A_filtered` — source MLIP potential energies (eV)
- `pot_energy_B_filtered` — target MLIP potential energies (eV)

---

## Repository Structure

```
scripts/
  reweighting/      PMF reweighting and entropy analysis
  barriers_jsd/     Free energy barriers and Jensen-Shannon divergence
  parity_plots/     DFT vs MLIP energy and force parity plots
data/               See Zenodo DOI above
```

---

## Scripts

### `scripts/reweighting/`

| Script | Description |
|---|---|
| `direct_reweighting_MBAR.py` | Direct reweighting of PMFs between MLIPs using MBAR |
| `plot_entropy_matpes_metafigure.py` | Shannon entropy S(z) + Gaussian ΔA metafigure for MACE-MATPES |
| `plot_pmf_energyonly_all_mlips_metafigure.py` | Energy-only PMF correction for all MLIPs (SI figure) |

### `scripts/barriers_jsd/`

| Script | Description |
|---|---|
| `plot_jsd_scatter_metafigure_3split.py` | Main figure: JSD heatmap + free energy barriers scatter |

### `scripts/parity_plots/`

| Script | Description |
|---|---|
| `plot_force_parity_metafigure.py` | Force parity plots for all 8 MLIPs vs DFT (PBE+D3) |
| `plot_energy_parity_8mlips.py` | Energy parity plots for all 8 MLIPs vs DFT (PBE+D3) |

---

## Methods

Reweighting uses the Zwanzig perturbation identity:

$$\Delta A(z) = -k_\mathrm{B}T \ln \langle e^{-\beta \Delta U} \rangle_{A,\xi=z}$$

Three estimators are implemented:

1. **MBAR** — multistate Bennett acceptance ratio
2. **Gaussian approximation** (Eq. 16): $\Delta A = k_\mathrm{B}T\!\left(\beta\mu_{\Delta U} - \beta^2\sigma_{\Delta U}^2/2\right)$
3. **Energy-only** (Eq. 17): $\Delta A = \langle U_B - U_A \rangle_{A,\xi=z}$

Standard errors for the Gaussian estimator use the delta method on the importance-weighted ratio estimator (Dietschreit *et al.*, Eqs. 6–9):

$$\mathrm{SE}(\mu) = \sqrt{\sum_i p_i^2\, (\Delta U_i - \mu)^2}$$

with Bessel-corrected weighted variance (denominator $= 1 - \sum_i p_i^2$).

Weight quality per umbrella window is assessed via the normalized Shannon entropy:

$$S(z) = \frac{-\sum_i p_i \ln p_i}{\ln N_z} \in [0, 1]$$

---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
pymbar
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{majumdar2026,
  author  = {Majumdar, Sauradeep and others},
  title   = {Reweighting free energy profiles between universal machine learning
             interatomic potentials for fast consensus building},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

MIT

