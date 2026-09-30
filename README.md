# Reweighting Free Energies

This repository contains the code and data used in the article:

> **"Reweighting free energy profiles between universal machine learning interatomic potentials for fast consensus building"**
> Majumdar *et al.*, 2026

---

## Overview

Analysis scripts for reweighting enhanced-sampling potential of mean force (PMF) profiles computed
with one source MLIP onto a suite of target MLIPs, using MBAR weights and perturbation theory.
Demonstrated on two systems spanning more than an order of magnitude in size and two very different
kinds of chemistry:

- **Li-ion transport** in a nanoconfined, water-solvated zeolite (601 atoms, periodic), reweighted
  from umbrella sampling with MACE-MP0 at T = 450 K.
- **5-hexenyl radical intramolecular cyclization** (17 atoms, gas-phase small molecule), reweighted
  from eABF with MACE-OMol at T = 300 K.


---

## Data

Processed umbrella-sampling/eABF frames, MBAR weights, and parity plot data for both systems are
deposited on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20142455.svg)](https://doi.org/10.5281/zenodo.20142455)

The Zenodo deposit contains the following:

- **`zeolite_structure/`** — CIF file of the studied Li+-in-zeolite system
- **`umbrella_sampling_data/`** — MACE-MATPES US simulation output and pre-computed MACE-MP0 PMF
- **`data_540000_frames/`** — source and target MLIP potential energies for 540,000 reweighting frames (10,000 frames x 54 windows), for 7 target MLIPs
- **`data_parity_plots/`** — DFT and MLIP energies and forces for 1,100 structures used in parity plots (zeolite)
- **`MACE-finetuned/`** — the finetuned MLIP developed in this work, along with the training data used
- **`MBAR_weights/`** — the MACE-MP0 unbiased MBAR weights obtained for the 540,000 downsampled frames
- **`hexenyl_parity_plot_data/`** — DFT and MLIP energies/gradient norms for 5,881 structures used in the hexenyl parity plots (5 target MLIPs)
- **`hexenyl_eabf_pmf_data/`** — final PMFs from independent eABF campaigns (MACE-OMol source, UMA-OMol independently-simulated target)
- **`hexenyl_reweighting_data/`** — cross-MLIP reweighting inputs and results for the hexenyl system (main-text PMF figure, JSD/barrier-scatter, region-wise variance)

---

## Repository Structure

```
  reweighting/      PMF reweighting and entropy analysis (zeolite)
  barriers_jsd/     Free energy barriers and Jensen-Shannon divergence (zeolite)
  parity_plots/     DFT vs MLIP energy and force parity plots (zeolite)
  hexenyl_eabf/     eABF enhanced sampling and PMF construction (hexenyl)
  SKILLS.md         Practical guide to enhanced sampling + reweighting with MLIPs
```

---

## Scripts used in the associated work

### `reweighting/`

| Script | Description |
|---|---|
| `direct_reweighting_MBAR.py` | Direct reweighting of PMFs between MLIPs using MBAR |
| `plot_entropy_matpes_metafigure.py` | Shannon entropy S(z) + Gaussian ΔA metafigure for MACE-MATPES |
| `plot_pmf_energyonly_all_mlips_metafigure.py` | Energy-only PMF correction for all MLIPs (SI figure) |

### `barriers_jsd/`

| Script | Description |
|---|---|
| `plot_jsd_scatter_metafigure_3split.py` | Main figure: JSD heatmap + free energy barriers scatter |

### `parity_plots/`

| Script | Description |
|---|---|
| `plot_force_parity_metafigure.py` | Force parity plots for all MLIPs vs DFT (PBE+D3) |
| `plot_energy_parity_metafigure.py` | Energy parity plots for all MLIPs vs DFT (PBE+D3) |

### `hexenyl_eabf/`

| Script | Description |
|---|---|
| `run_eabf_hexenyl_omol.py` | eABF production run, MACE-OMol as the driving potential |
| `run_eabf_hexenyl_uma.py` | eABF production run, UMA-OMol as the driving potential |
| `bias_calculators_uma_new.py` | Generic (calculator-agnostic) biasing base class for UMA/FAIRChem calculators |
| `bias_calculators_uma_eabf.py` | eABF calculator built on the above, for UMA-OMol |
| `analyze_eabf_pmf.py` | Builds the PMF from a two-segment eABF trajectory via MBAR; works for either MLIP's trajectory via CLI arguments |

Both eABF run scripts are checkpoint/restart-capable (`--restart_from`), since each 800 ps campaign
was run as two segments split across job time limits.

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

- **`nff`** (NeuralForceField) — used for umbrella sampling / eABF simulations. Install from source:
  ```bash
  git clone https://github.com/learningmatter-mit/NeuralForceField
  cd NeuralForceField
  pip install -e .
  ```

- For the hexenyl eABF scripts specifically: `mace-torch` (MACE-OMol run) and `fairchem-core` (UMA-OMol run).


---

## Citation

If you find our work helpful, please cite:

```bibtex
@misc{majumdar2026reweightingmlips,
      title={Reweighting free energy profiles between universal machine learning interatomic potentials for fast consensus building}, 
      author={Sauradeep Majumdar and Miguel Steiner and Johannes C. B. Dietschreit and Swagata Roy and Daniel Willimetz and Lukaš Grajciar and Rafael Gómez-Bombarelli},
      year={2026},
      eprint={2605.15630},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2605.15630}, 
}
```

---

## License

MIT

