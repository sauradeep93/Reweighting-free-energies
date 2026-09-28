"""
eABF (extended-system Adaptive Biasing Force) sampling of the 5-hexenyl
radical intramolecular cyclization, driven by UMA-OMol as the source MLIP.

Reaction coordinate: distance between atoms 9 and 12 (the forming C-C bond),
biased over range [1.0, 6.0] Angstrom. Starting structure and CV definition
follow Dietschreit et al., J. Chem. Phys. 156, 114105 (2022)
(github.com/learningmatter-mit/Tutorial_ActivationFreeEnergy).

Checkpoint/restart-capable: each invocation runs one segment of `--steps`
MD steps and writes a checkpoint; pass `--restart_from` to resume a
previous segment's checkpoint for long campaigns split across job time
limits.

Usage:
    python run_eabf_hexenyl_uma.py --cache_path /path/to/uma-s-1.pt \
        --prefix eabf_uma_800ps_seg1 --steps 1600000
    python run_eabf_hexenyl_uma.py --cache_path /path/to/uma-s-1.pt \
        --prefix eabf_uma_800ps_seg2 --steps 400000 \
        --restart_from eabf_uma_800ps_seg1_checkpoint.pkl
"""
import os
import sys

# --- User-specific paths: edit these for your own environment ---
NFF_ROOT = "/home/gridsan/smajumdar/NeuralForceField"
STARTING_STRUCTURE = f"{NFF_ROOT}/hexenyl_cyclization/hexenyl_maceoff_300K_window_6.000/atoms_dcdframe_orig.xyz"
# ------------------------------------------------------------------

num_threads = 4
num_threads_str = str(num_threads)
os.environ["MKL_NUM_THREADS"] = num_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = num_threads_str
os.environ["OMP_NUM_THREADS"] = num_threads_str

sys.path.append(NFF_ROOT)
sys.path.append(f"{NFF_ROOT}/hexenyl_cyclization")  # for bias_calculators_uma_eabf.py

import argparse
import pickle
import numpy as np
import torch

# nff.md.colvars transitively imports e3nn (via nff.train), whose bundled
# constants.pt fails to load under torch>=2.6's new weights_only=True default.
# We don't use e3nn's actual functionality here (only ColVar), so it's safe to
# restore the pre-2.6 default for this trusted, package-bundled file rather
# than touching e3nn or nff themselves.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# nff.md.colvars does `from nff.train import load_model, evaluate` purely for
# two functions we never call (our CV is just "distance") -- but importing the
# real nff.train eagerly pulls in every optional model builder (e3nn, chgnet,
# schnetpack, ...), none of which we need or want here. Stub it out so Python
# never executes the real nff/train/__init__.py chain.
import types

_fake_train = types.ModuleType("nff.train")
_fake_train.load_model = None
_fake_train.evaluate = None
sys.modules["nff.train"] = _fake_train

from ase.io import read
from fairchem.core import FAIRChemCalculator

from nff.io.ase import AtomsBatch
from nff.md.nvt import Langevin
from nff.md.colvars import ColVar
from nff.md.utils import BiasedNeuralMDLogger

from bias_calculators_uma_eabf import eABF

parser = argparse.ArgumentParser()
parser.add_argument("--cache_path", type=str, required=True, help="path to uma-s-1.pt checkpoint")
parser.add_argument("--steps", type=int, default=1600000,
                     help="number of MD steps to run in THIS segment")
parser.add_argument("--prefix", default="eabf_uma_production",
                     help="file prefix for this segment's .traj/.log/_ext.log/_checkpoint.pkl")
parser.add_argument("--restart_from", default=None,
                     help="checkpoint .pkl from a previous segment to resume from; "
                          "omit for a fresh run starting at the reactant structure")
parser.add_argument("--checkpoint_interval", type=int, default=20000,
                     help="steps between checkpoint saves (default ~10 ps at dt=0.5fs)")
args = parser.parse_args()

device = 0

calc = FAIRChemCalculator.from_model_checkpoint(
    name_or_path=args.cache_path,
    task_name="omol",
    inference_settings="default",
    overrides=None,
    device="cuda",
)

restarting = args.restart_from is not None
ckpt_in = None
if restarting:
    with open(args.restart_from, "rb") as f:
        ckpt_in = pickle.load(f)
    print(f"Resuming from {args.restart_from}: "
          f"{ckpt_in['completed_steps']} steps already completed")

atoms_window = read(STARTING_STRUCTURE, format="extxyz")
atoms = AtomsBatch.from_atoms(
    atoms_window, cutoff=6, device="cpu", directed=True, requires_large_offsets=True
)
if restarting:
    atoms.set_positions(ckpt_in["positions"])
atoms.info["charge"] = 0
atoms.info["spin"] = 2  # hexenyl radical: doublet (one unpaired electron)
atoms.update_nbr_list()

info_dict = {"name": "distance", "index_list": [9, 12]}
CV = ColVar(info_dict)
cv, cv_grad = CV(atoms)
print(f"Start CV value: {cv}")

cv_defs = [
    {
        "definition": info_dict,
        "range": [1.0, 6.0],
        "ext_sigma": 0.05,
        "ext_mass": 20.0,
        "ext_pos": ckpt_in["eabf_state"]["ext_coords"][0] if restarting else cv,
        "margin": 0.1,
        "conf_k": 1.0,
        "type": "distance",
    }
]

calculator = eABF(
    calc,
    cv_defs=cv_defs,
    dt=0.5,
    friction_per_ps=1.0,
    equil_temp=300.0,
    nfull=100,
    directed=True,
    device=device,
)
if restarting:
    calculator.set_state(ckpt_in["eabf_state"])
atoms.set_calculator(calculator)

dyn = Langevin(
    atoms,
    timestep=0.5,
    temperature=300.0,
    friction_per_ps=1.0,
    maxwell_temp=300.0,
    random_seed=ckpt_in["rng_state"] if restarting else None,
    logfile=f"{args.prefix}.log",
    trajectory=f"{args.prefix}.traj",
    loginterval=20,
)
if restarting:
    # Langevin.__init__ always redraws velocities from a fresh Maxwell-Boltzmann
    # distribution -- overwrite with the exact checkpointed momenta afterwards.
    atoms.set_momenta(ckpt_in["momenta"])

dyn.attach(
    BiasedNeuralMDLogger(dyn, atoms, f"{args.prefix}_ext.log", header=True, mode="w"),
    interval=20,
)

completed_before = ckpt_in["completed_steps"] if restarting else 0
checkpoint_path = f"{args.prefix}_checkpoint.pkl"


def save_checkpoint():
    ckpt = {
        "completed_steps": completed_before + dyn.nsteps,
        "positions": atoms.get_positions(),
        "momenta": atoms.get_momenta(),
        "rng_state": np.random.get_state(),
        "eabf_state": calculator.get_state(),
    }
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp_path, checkpoint_path)


dyn.attach(save_checkpoint, interval=args.checkpoint_interval)

dyn.run(steps=args.steps)
save_checkpoint()
print(f"DONE. Total steps completed (all segments): {completed_before + dyn.nsteps}")
print(f"Checkpoint saved to {checkpoint_path}")
