"""
Umbrella-sampling / eABF bias calculator for UMA-OMol, built on the same
CV/extended-dynamics contract as bias_calculators_mace_new.py (self.cv_defs,
self.ranges, self.ext_*, step_bias/_extended_dynamics, diff, _check_boundaries),
but with all MACE-specific graph/batch construction (_atoms_to_batch,
_create_result_tensors, mace_data.AtomicData, ...) replaced by a direct call to
a wrapped, already-ASE-compatible FAIRChemCalculator instance.

This is a NEW file, not an edit of any existing nff/io/bias_calculators*.py --
none of those are touched. Copied here 2026-09-12 for the UMA-OMol hexenyl
campaign (parallel to the MACE-OFF barrier-fix work).
"""
from typing import Union, Tuple

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import units

from nff.md.colvars import ColVar as CV

DEFAULT_DIRECTED = False

# hexenyl radical: one unpaired electron -> i.e. S=1/2, spin multiplicity = 2S+1=2 ->doublet. UMA-OMol's omol head
# silently defaults to S=0 (singlet) if atoms.info["spin"] is unset, which
# is physically wrong here
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 2


class BiasBase(Calculator):
    """Umbrella-sampling bias calculator wrapping any ASE-compatible calculator
    (here: a FAIRChemCalculator loaded with the UMA-OMol checkpoint).

    Args:
        model: an already-constructed ASE Calculator (e.g. FAIRChemCalculator)
        cv_defs: list of Collective Variable (CV) definitions
        equil_temp: temperature of the simulation
        charge/spin: passed through to atoms.info for the wrapped calculator
    """

    implemented_properties = [
        "energy",
        "forces",
        "energy_unbiased",
        "forces_unbiased",
        "cv_vals",
        "ext_pos",
        "cv_invmass",
        "grad_length",
        "cv_grad_lengths",
        "cv_dot_PES",
        "const_vals",
    ]

    def __init__(
        self,
        model,
        cv_defs: list,
        equil_temp: float = 300.0,
        device="cpu",
        directed=DEFAULT_DIRECTED,
        charge: int = DEFAULT_CHARGE,
        spin: int = DEFAULT_SPIN,
        extra_constraints: list = None,
        **kwargs,
    ):
        Calculator.__init__(self)
        self.model = model
        self.device = device
        self.charge = charge
        self.spin = spin

        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(CV(cv_def["definition"]))

        self.equil_temp = equil_temp

        self.ext_coords = np.zeros(shape=(self.num_cv, 1))
        self.ext_masses = np.zeros(shape=(self.num_cv, 1))
        self.ext_forces = np.zeros(shape=(self.num_cv, 1))
        self.ext_vel = np.zeros(shape=(self.num_cv, 1))
        self.ext_binwidth = np.zeros(shape=(self.num_cv, 1))
        self.ext_k = np.zeros(shape=(self.num_cv,))
        self.ext_dt = 0.0

        self.ranges = np.zeros(shape=(self.num_cv, 2))
        self.margins = np.zeros(shape=(self.num_cv, 1))
        self.conf_k = np.zeros(shape=(self.num_cv, 1))

        for ii, cv in enumerate(self.cv_defs):
            if "range" in cv.keys():
                self.ext_coords[ii] = cv["range"][0]
                self.ranges[ii] = cv["range"]
            else:
                raise KeyError("range")

            if "margin" in cv.keys():
                self.margins[ii] = cv["margin"]

            if "conf_k" in cv.keys():
                self.conf_k[ii] = cv["conf_k"]

            if "ext_k" in cv.keys():
                self.ext_k[ii] = cv["ext_k"]
            elif "ext_sigma" in cv.keys():
                self.ext_k[ii] = (units.kB * self.equil_temp) / (
                    cv["ext_sigma"] * cv["ext_sigma"]
                )
            else:
                raise KeyError("ext_k/ext_sigma")

            if "type" not in cv.keys():
                self.cv_defs[ii]["type"] = "not_angle"
            else:
                self.cv_defs[ii]["type"] = cv["type"]

        self.constraints = None
        self.num_const = 0
        if extra_constraints is not None:
            self.constraints = []
            for cv in extra_constraints:
                self.constraints.append({})
                self.constraints[-1]["func"] = CV(cv["definition"])
                self.constraints[-1]["pos"] = cv["pos"]
                if "k" in cv.keys():
                    self.constraints[-1]["k"] = cv["k"]
                elif "sigma" in cv.keys():
                    self.constraints[-1]["k"] = (units.kB * self.equil_temp) / (
                        cv["sigma"] * cv["sigma"]
                    )
                else:
                    raise KeyError("k/sigma")
                if "type" not in cv.keys():
                    self.constraints[-1]["type"] = "not_angle"
                else:
                    self.constraints[-1]["type"] = cv["type"]
            self.num_const = len(self.constraints)

    # ---- bias/extended-dynamics machinery: calculator-agnostic, identical to
    # bias_calculators_mace_new.py (only touches self.cv_defs/self.ext_*) ----

    def _update_bias(self, xi: np.ndarray):
        pass

    def _propagate_ext(self):
        pass

    def _up_extvel(self):
        pass

    def _check_boundaries(self, xi: np.ndarray):
        in_bounds = (xi <= self.ranges[:, 1]).all() and (xi >= self.ranges[:, 0]).all()
        return in_bounds

    def diff(
        self, a: Union[np.ndarray, float], b: Union[np.ndarray, float], cv_type: str
    ) -> Union[np.ndarray, float]:
        diff = a - b
        if isinstance(diff, np.ndarray) and cv_type == "angle":
            diff[diff > np.pi] -= 2 * np.pi
            diff[diff < -np.pi] += 2 * np.pi
        elif cv_type == "angle":
            if diff < -np.pi:
                diff += 2 * np.pi
            elif diff > np.pi:
                diff -= 2 * np.pi
        return diff

    def step_bias(
        self, xi: np.ndarray, grad_xi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._propagate_ext()
        bias_ener, bias_grad = self._extended_dynamics(xi, grad_xi)
        self._update_bias(xi)
        self._up_extvel()
        return bias_ener, bias_grad

    def _extended_dynamics(
        self, xi: np.ndarray, grad_xi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        bias_grad = np.zeros_like(grad_xi[0])
        bias_ener = 0.0
        for i in range(self.num_cv):
            dxi = self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]["type"])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_grad += self.ext_k[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ext_k[i] * dxi**2

            if self.ext_coords[i] > (self.ranges[i][1] + self.margins[i]):
                r = self.diff(
                    self.ranges[i][1] + self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r
            elif self.ext_coords[i] < (self.ranges[i][0] - self.margins[i]):
                r = self.diff(
                    self.ranges[i][0] - self.margins[i],
                    self.ext_coords[i],
                    self.cv_defs[i]["type"],
                )
                self.ext_forces[i] += self.conf_k[i] * r
        return bias_ener, bias_grad

    def harmonic_constraint(
        self, xi: np.ndarray, grad_xi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        constr_grad = np.zeros_like(grad_xi[0])
        constr_ener = 0.0
        for i in range(self.num_const):
            dxi = self.diff(
                xi[i], self.constraints[i]["pos"], self.constraints[i]["type"]
            )
            constr_grad += self.constraints[i]["k"] * dxi * grad_xi[i]
            constr_ener += 0.5 * self.constraints[i]["k"] * dxi**2
        return constr_ener, constr_grad

    # ---- the only part that actually differs from the MACE version ----

    def calculate(
        self,
        atoms=None,
        properties=[
            "energy",
            "forces",
            "energy_unbiased",
            "forces_unbiased",
            "cv_vals",
            "cv_invmass",
            "grad_length",
            "cv_grad_lengths",
            "cv_dot_PES",
            "const_vals",
        ],
        system_changes=all_changes,
    ):
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # UMA-OMol needs explicit charge/spin metadata (radical -> doublet);
        # silently defaults to singlet otherwise. Cheap to set every call.
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin

        # call the wrapped calculator directly on this atoms object -- do NOT
        # assign atoms.calc = self.model, which would permanently replace the
        # bias-wrapping calculator (self) that other code (e.g. the MD logger)
        # expects to still be attached to atoms after this returns.
        model_energy = np.array([self.model.get_potential_energy(atoms)])
        model_grad = -self.model.get_forces(atoms)

        inv_masses = 1.0 / atoms.get_masses()
        M_inv = np.diag(np.repeat(inv_masses, 3).flatten())

        cvs = np.zeros(shape=(self.num_cv, 1))
        cv_grads = np.zeros(
            shape=(
                self.num_cv,
                atoms.get_positions().shape[0],
                atoms.get_positions().shape[1],
            )
        )
        cv_grad_lens = np.zeros(shape=(self.num_cv, 1))
        cv_invmass = np.zeros(shape=(self.num_cv, 1))
        cv_dot_PES = np.zeros(shape=(self.num_cv, 1))
        for ii, cv_def in enumerate(self.cv_defs):
            xi, xi_grad = self.the_cv[ii](atoms)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii] = np.einsum(
                "i,ii,i", xi_grad.flatten(), M_inv, xi_grad.flatten()
            )
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())

        self.results = {
            "energy_unbiased": model_energy.reshape(-1),
            "forces_unbiased": -model_grad.reshape(-1, 3),
            "grad_length": np.linalg.norm(model_grad),
            "cv_vals": cvs,
            "cv_grad_lengths": cv_grad_lens,
            "cv_invmass": cv_invmass,
            "cv_dot_PES": cv_dot_PES,
        }

        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy = model_energy + bias_ener
        grad = model_grad + bias_grad

        if self.constraints:
            consts = np.zeros(shape=(self.num_const, 1))
            const_grads = np.zeros(
                shape=(
                    self.num_const,
                    atoms.get_positions().shape[0],
                    atoms.get_positions().shape[1],
                )
            )
            for ii, const_dict in enumerate(self.constraints):
                consts[ii], const_grads[ii] = const_dict["func"](atoms)
            const_ener, const_grad = self.harmonic_constraint(consts, const_grads)
            energy += const_ener
            grad += const_grad

        self.results.update(
            {
                "energy": energy.reshape(-1),
                "forces": -grad.reshape(-1, 3),
                "ext_pos": self.ext_coords,
            }
        )
        if self.constraints:
            self.results["const_vals"] = consts
