"""
eABF calculator for UMA-OMol, built on the generic (calculator-agnostic)
BiasBase from bias_calculators_uma_new.py -- ported verbatim from the MACE
version (bias_calculators_mace_eabf.py / the eABF class in
bias_calculators_mace_new.py). The eABF physics (extended-coordinate Langevin
dynamics, ABF force accumulation) only touch self.cv_defs/self.ranges/
self.ext_*/welford_var, never calculate()/the wrapped-model call, so this is
a straight port with the import swapped to the UMA-generic base class.

"""
from typing import Tuple

import numpy as np
from ase import units

from bias_calculators_uma_new import BiasBase, DEFAULT_DIRECTED


def welford_var(
    count: float, mean: float, M2: float, newValue: float
) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm"""
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var


class eABF(BiasBase):
    """extended-system Adaptive Biasing Force Calculator for UMA-OMol

    Args:
        model: an ASE-compatible calculator (here: FAIRChemCalculator, UMA-OMol)
        cv_defs: list of Collective Variable (CV) definitions
        equil_temp: temperature of the simulation
        dt: time step of the extended dynamics (must equal the real system dyn!)
        friction_per_ps: friction for the Langevin dyn of extended system
        nfull: number of samples needed for full application of bias force
    """

    def __init__(
        self,
        model,
        cv_defs: list,
        dt: float,
        friction_per_ps: float,
        equil_temp: float = 300.0,
        nfull: int = 100,
        device="cpu",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        BiasBase.__init__(
            self,
            cv_defs=cv_defs,
            equil_temp=equil_temp,
            model=model,
            device=device,
            directed=directed,
            **kwargs,
        )

        self.ext_dt = dt * units.fs
        self.nfull = nfull

        for ii, cv in enumerate(self.cv_defs):
            if "bin_width" in cv.keys():
                self.ext_binwidth[ii] = cv["bin_width"]
            elif "ext_sigma" in cv.keys():
                self.ext_binwidth[ii] = cv["ext_sigma"]
            else:
                raise KeyError("bin_width")

            if "ext_pos" in cv.keys():
                self.ext_coords[ii] = cv["ext_pos"]
            else:
                raise KeyError("ext_pos")

            if "ext_mass" in cv.keys():
                self.ext_masses[ii] = cv["ext_mass"]
            else:
                raise KeyError("ext_mass")

        for i in range(self.num_cv):
            self.ext_vel[i] = np.random.randn() * np.sqrt(
                self.equil_temp * units.kB / self.ext_masses[i]
            )

        self.friction = friction_per_ps * 1.0e-3 / units.fs
        self.rand_push = np.sqrt(
            self.equil_temp
            * self.friction
            * self.ext_dt
            * units.kB
            / (2.0e0 * self.ext_masses)
        )
        self.prefac1 = 2.0 / (2.0 + self.friction * self.ext_dt)
        self.prefac2 = (2.0e0 - self.friction * self.ext_dt) / (
            2.0e0 + self.friction * self.ext_dt
        )

        self.nbins_per_dim = np.array([1 for i in range(self.num_cv)])
        self.grid = []
        for i in range(self.num_cv):
            self.nbins_per_dim[i] = int(
                np.ceil(
                    np.abs(self.ranges[i, 1] - self.ranges[i, 0]) / self.ext_binwidth[i]
                )
            )
            self.grid.append(
                np.arange(
                    self.ranges[i, 0] + self.ext_binwidth[i] / 2,
                    self.ranges[i, 1],
                    self.ext_binwidth[i],
                )
            )
        self.nbins = np.prod(self.nbins_per_dim)

        self.bias = np.zeros((self.num_cv, *self.nbins_per_dim), dtype=float)
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)
        self.cv_crit = np.copy(self.bias)
        self.histogram = np.zeros(self.nbins_per_dim, dtype=float)
        self.ext_hist = np.zeros_like(self.histogram)

    def get_index(self, xi: np.ndarray) -> tuple:
        bin_x = np.zeros(shape=xi.shape, dtype=np.int64)
        for i in range(self.num_cv):
            bin_x[i] = int(
                np.floor(np.abs(xi[i] - self.ranges[i, 0]) / self.ext_binwidth[i])
            )
        return tuple(bin_x.reshape(1, -1)[0])

    def _update_bias(self, xi: np.ndarray):
        if self._check_boundaries(self.ext_coords):
            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1
            ramp = (
                1.0
                if self.ext_hist[bink] > self.nfull
                else self.ext_hist[bink] / self.nfull
            )
            for i in range(self.num_cv):
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i]
                    * self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"]),
                )
                self.ext_forces[i] -= ramp * self.bias[i][bink]

    def _propagate_ext(self):
        self.ext_rand_gauss = np.random.randn(len(self.ext_vel), 1)
        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses
        self.ext_coords += self.prefac1 * self.ext_dt * self.ext_vel

        for ii in range(self.num_cv):
            if self.cv_defs[ii]["type"] == "angle":
                if self.ext_coords[ii] > np.pi:
                    self.ext_coords[ii] -= 2 * np.pi
                elif self.ext_coords[ii] < -np.pi:
                    self.ext_coords[ii] += 2 * np.pi

    def _up_extvel(self):
        self.ext_vel *= self.prefac2
        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses

    def get_state(self) -> dict:
        """Accumulated sampling state needed to resume this run exactly where it
        left off -- everything NOT simply reconstructed from cv_defs at init."""
        return {
            "ext_coords": np.copy(self.ext_coords),
            "ext_vel": np.copy(self.ext_vel),
            "bias": np.copy(self.bias),
            "var_force": np.copy(self.var_force),
            "m2_force": np.copy(self.m2_force),
            "histogram": np.copy(self.histogram),
            "ext_hist": np.copy(self.ext_hist),
        }

    def set_state(self, state: dict):
        """Restore accumulated sampling state saved by get_state()."""
        self.ext_coords = np.copy(state["ext_coords"])
        self.ext_vel = np.copy(state["ext_vel"])
        self.bias = np.copy(state["bias"])
        self.var_force = np.copy(state["var_force"])
        self.m2_force = np.copy(state["m2_force"])
        self.histogram = np.copy(state["histogram"])
        self.ext_hist = np.copy(state["ext_hist"])
