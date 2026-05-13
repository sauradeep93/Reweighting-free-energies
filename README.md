# Reweighting Free Energies

This repository contains the code and data used in the article:

> **"Reweighting free energy profiles between universal machine learning interatomic potentials for fast consensus building"**
> Majumdar *et al.*, 2026

---

## Overview

Analysis scripts for reweighting umbrella sampling potential of mean force (PMF) profiles computed with one source MLIP (MACE-MP0) onto a suite of target MLIPs, using MBAR weights and perturbation theory. The studied system is Li-ion transport in a nanoconfined electrolyte (601 atoms total) at T = 450 K.

---

## Data

Processed umbrella sampling frames, MBAR weights, and parity plot data are deposited on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20142456.svg)](https://doi.org/10.5281/zenodo.20142456)

The Zenodo deposit contains four folders:

- **`zeolite_structure/`** — CIF file of the studied Li⁺-in-zeolite system
- **`umbrella_sampling_data/`** — MACE-MATPES US simulation output and pre-computed MACE-MP0 PMF
- **`data_540000_frames/`** — source and target MLIP potential energies for 540,000 reweighting frames (10,000 frames × 54 windows), for 7 target MLIPs
- **`data_parity_plots/`** — DFT and MLIP energies and forces for 1,100 structures used in parity plots

---

## Repository Structure

```
scripts/
  reweighting/      PMF reweighting and entropy analysis
  barriers_jsd/     Free energy barriers and Jensen-Shannon divergence
  parity_plots/     DFT vs MLIP energy and force parity plots
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
| `plot_energy_parity_metafigure.py` | Energy parity plots for all 8 MLIPs vs DFT (PBE+D3) |

---

## Requirements

```
python >= 3.9
numpy
scipy
matplotlib
adaptive_sampling
```

- **`adaptive_sampling`** — provides MBAR implementation and unit conversions. Install via pip:
  ```bash
  pip install adaptive-sampling
  ```

- **`nff`** (NeuralForceField) — used for umbrella sampling simulations. Install from source:
  ```bash
  git clone https://github.com/learningmatter-mit/NeuralForceField
  cd NeuralForceField
  pip install -e .
  ```

---

## Citation

If you find our work helpful, please cite:

```bibtex
@article{majumdar2026,
  author  = {Majumdar, Sauradeep and Steiner, Miguel and Dietschriet, Johannes C.B. and Roy, Swagata and Willimetz, Daniel and Grajciar, Lukaš and G\'omez-Bombarelli, Rafael },
  title   = {Reweighting free energy profiles between universal machine learning
             interatomic potentials for fast consensus building},
  journal = {arxiv preprint},
  year    = {2026},
  doi     = {}
}
```

---

## License

MIT


