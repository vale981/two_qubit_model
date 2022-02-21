r"""
 Operators for a general model of two interacting qubits coupled to
 two baths in dimensionless units normalized to the frequency of the
 left qubit :math:`ω_1`.

 The :math:`z` axis of the two qubits is defined by their local
 hamiltonians :math:`H_s^i = \frac{ω_i}{2}σ^i_z`.

 The total hamiltonian has the form

 ..  math::

     H=\frac{1}{2}σ^1_z + \frac{ω_2}{2}σ^2_z + H_B^1 + H_B^2
       + \frac{γ}{2} ∑_{i,j=1}^{3} J_{ij} σ^1_i σ^2_j
       + \sum_{i=1}^2 δ_i ∑_λ g_λ^i \vec{s}_i\cdot \vec{σ}^i (b^i_λ + b^{i,†}_λ).

 The matrix :math:`J` is real and normalized so that the operator norm
 of :math:`∑_{i,j=1}^{3} J_{ij} σ^1_i σ^2_j` is equal to one.  The
 :math:`\vec{s}_i` are unit vectors with zero :math:`y` componets.

 The sepectral densities

 ..  math::

     J_i(ω) = π ∑_λ {|g^i_λ|}^2 δ(ω-ω^i_c)

 are nomalized so that their integral is equal to pi.
 """

import dataclasses
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Any, Optional, SupportsFloat
import hops.util.bcf
import hops.util.bcf_fits
import hops.core.hierarchy_parameters as params
import numpy as np
import qutip as qt
from hops.util.truncation_schemes import TruncationScheme_Power_multi
import stocproc as sp
import json
from functools import singledispatchmethod
import hashlib
from beartype import beartype


@beartype
@dataclass
class StocProcTolerances:
    """
    An object to hold tolerances for :any:`stocproc.StocProc`
    instances.
    """

    integration: float = 1e-4
    """Integration tolerance."""

    interpolation: float = 1e-4
    """Interpolation tolerance."""


@beartype
@dataclass
class TwoQubitModel:
    """
    A class to dynamically calculate all the model parameters and
    generate the HOPS configuration.

    All attributes can be changed after initialization.
    """

    __version__: int = 1
    """
    The version of the model implementation.  It is increased for
    breaking changes.
    """

    ω_2: SupportsFloat = 1.0
    """The second oscilator energy gap."""

    γ: SupportsFloat = 1.0
    """The oveall inter-qubit coupling strength."""

    δ: list[SupportsFloat] = field(default_factory=lambda: [1.0, 1.0])
    """The bath coupling factors (length 2)."""

    ω_c: list[SupportsFloat] = field(default_factory=lambda: [1.0, 1.0])
    """The BCF central frequencies :math:`ω_c` (length 2)."""

    s: list[SupportsFloat] = field(default_factory=lambda: [1.0, 1.0])
    """The BCF s parameters frequencies (length 2)."""

    j: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64
        )
    )
    """
    The :math:`J_{ij}` coupling coefficients with shape ``(3,3)``.
    They will be normalized automatically.
    """

    s_vec: NDArray[np.float64] = field(
        default_factory=lambda: np.array([[1, 0], [1, 0]], dtype=np.float64)
    )
    """
    The :math:`\vec{s}_i` unit vectors with zero y-component of shape
    ``(2,2)``.  Two vectors of form (``[[x,z], [x, z]]``).  They will
    be normalized automatically.
    """

    T: list[SupportsFloat] = field(default_factory=lambda: list([0, 0]))
    """The temperatures of the baths."""

    ###########################################################################
    #                             HOPS Parameters                             #
    ###########################################################################

    description: str = ""
    """A free-form description of the model instance."""

    bcf_terms: list[int] = field(default_factory=lambda: [5, 5])
    """How many bcf terms to use in the expansions of the BCFs."""

    ψ_0: qt.Qobj = qt.basis([2, 2], [1, 1])
    """The initial state."""

    t_max: SupportsFloat = 10
    """The maximum simulation time."""

    resolution: SupportsFloat = 0.1
    """The time resolution of the simulation."""

    k_fac: list[SupportsFloat] = field(default_factory=lambda: [1.4, 1.4])
    """The k_fac parameters for the truncation scheme.

    See
    :any:`hops.util.truncation_schemes.TruncationScheme_Power_multi`.
    """

    solver_args: dict[str, Any] = field(default_factory=dict)
    """Extra arguments for :any:`scipy.integrate.solve_ivp`."""

    driving_process_tolerances: list[StocProcTolerances] = field(
        default_factory=lambda: [StocProcTolerances(), StocProcTolerances()]
    )
    """
    The integration and interpolation tolerances for the driving
    processes.
    """

    thermal_process_tolerances: list[StocProcTolerances] = field(
        default_factory=lambda: [StocProcTolerances(), StocProcTolerances()]
    )
    """
    The integration and interpolation tolerances for the thermal noise
    processes.
    """

    def __post_init__(self):
        self._sigmas = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

    def system(self, i: int) -> qt.Qobj:
        """The system hamiltonian of the ``i``th qubit."""

        if i == 1:
            return 1 / 2 * (qt.tensor(qt.sigmaz(), qt.identity(2)))  # type: ignore

        return self.ω_2 / 2 * (qt.tensor(qt.identity(2), qt.sigmaz()))  # type: ignore

    @property
    def bare_interaction(self) -> qt.Qobj:
        """
        The inter-qubit interaction hamiltonian without scaling
        factors and normalization.

        .. math::

           ∑_{i,j=1}^{3} J_{ij} σ^1_i σ^2_j
        """

        assert self.j.shape == (3, 3)
        assert (self.j.imag == 0).all()

        interaction = qt.Qobj(dims=[[2, 2], [2, 2]])

        for strength in (it := np.nditer(self.j, flags=["multi_index"])):
            i, j = it.multi_index

            interaction += float(strength) * qt.tensor(self._sigmas[i], self._sigmas[j])

        return interaction

    @property
    def interaction(self) -> qt.Qobj:
        """The inter-qubit interaction hamiltonian."""

        interaction = self.bare_interaction
        interaction *= self.γ / (2 * max(interaction.eigenenergies()))

        return interaction

    def j_vecs(
        self, normalized: bool = True
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        This returns pairs of vectors :math:`(u_i, v_i)` so that
        :math:`J=∑_i u_i v_i^T`.  If normalized is :any:`True`
        :math:`J` will be normalized as discussed in the module
        docstring.
        """

        norm: float = (
            np.sqrt(max(self.bare_interaction.eigenenergies())) if normalized else 1
        )

        j_vecs = []
        q, r = np.linalg.qr(self.j)
        test = np.zeros_like(self.j, dtype=np.float64)
        for i in range(self.j.shape[0]):
            c = q[:, i]
            l = r[i, :]
            if not ((c == 0).all() or (l == 0).all()):
                j_vecs.append((c / norm, l / norm))
                test += np.outer(c, l)

        return j_vecs

    def bath_coupling(self, i: int) -> qt.Qobj:
        """
        The bath coupling operator :math:`L_i` of the ``i``th qubit.
        """

        s = np.array(self.s_vec[i]) / np.linalg.norm(np.array(self.s_vec[i]))
        coupling_op = qt.sigmax() * s[0] + qt.sigmaz() * s[1]

        if i == 1:
            return qt.tensor(coupling_op, qt.identity(2))

        return qt.tensor(qt.identity(2), coupling_op)

    def bcf_scale(self, i: int) -> float:
        """
        The BCF scaling factor of the ``i``th bath.
        """

        return float(self.δ[i]) ** 2

    def bcf(self, i: int) -> hops.util.bcf.OhmicBCF_zeroTemp:
        """
        The normalized zero temperature BCF of the  ``i``th bath.
        """

        return hops.util.bcf.OhmicBCF_zeroTemp(
            s=self.s[i], eta=1, w_c=self.ω_c[i], normed=True, with_pi=False
        )

    def spectral_density(self, i: int) -> hops.util.bcf.OhmicSD_zeroTemp:
        """
        The normalized zero temperature spectral density of the  ``i``th bath.
        """

        return hops.util.bcf.OhmicSD_zeroTemp(
            s=float(self.s[i]),
            eta=np.pi,
            w_c=float(self.ω_c[i]),
            normed=True,
            with_pi=False,
        )

    def thermal_correlations(
        self, i: int
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialCorrelations]:
        """
        The normalized thermal noise corellation function of the  ``i``th bath.
        """

        if self.T[i] == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialCorrelations(
            s=self.s[i],
            eta=1,
            w_c=self.ω_c[i],
            normed=True,
            with_pi=False,
            beta=1 / float(self.T[i]),
        )

    def thermal_spectral_density(
        self, i: int
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialDensity]:
        """
        The normalized thermal noise spectral density of the  ``i``th bath.
        """

        if self.T[i] == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialDensity(
            s=self.s[i],
            eta=1,
            w_c=self.ω_c[i],
            normed=True,
            with_pi=False,
            beta=1.0 / float(self.T[i]),
        )

    def bcf_coefficients(
        self, i: int, n: Optional[int] = None
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """
        The normalizedzero temperature BCF fit coefficients
        :math:`G_i,W_i` the ``i``th bath with ``n`` terms.
        """
        n = n or self.bcf_terms[i]
        return self.bcf(i).exponential_coefficients(n)

    @staticmethod
    def basis(n_1: int = 1, n_2: int = 1) -> qt.Qobj:
        """
        A product state with the qubits in states ``n_i`` where ``1``
        means down and ``0`` means up.
        """
        return qt.basis([2, 2], [n_1, n_2])

    def driving_process(
        self,
        i: int,
    ) -> sp.StocProc:
        """The driving stochastic process of the ``i``th bath."""

        return sp.StocProc_FFT(
            spectral_density=self.spectral_density(i),
            alpha=self.bcf(i),
            t_max=self.t_max,
            intgr_tol=self.driving_process_tolerances[i].integration,
            intpl_tol=self.driving_process_tolerances[i].interpolation,
            negative_frequencies=False,
        )

    def thermal_process(
        self,
        i: int,
    ) -> Optional[sp.StocProc]:
        """The thermal noise stochastic process of the ``i``th bath."""

        if self.T[i] == 0:
            return None

        return sp.StocProc_TanhSinh(
            spectral_density=self.thermal_spectral_density(i),
            alpha=self.thermal_correlations(i),
            t_max=self.t_max,
            intgr_tol=self.thermal_process_tolerances[i].integration,
            intpl_tol=self.thermal_process_tolerances[i].interpolation,
            negative_frequencies=False,
        )

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    def to_json(self):
        """Returns a json representation of the model configuration."""

        return json.dumps(
            {
                key: self.__dict__[key]
                for key in self.__dict__
                if key[0] != "_" or key == "__version__"
            },
            cls=JSONEncoder,
        )

    def __hash__(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, json_str: str):
        """
        Tries to instantiate a model config from the json string
        ``json_str``.
        """

        model_dict = json.loads(json_str, object_hook=_object_hook)
        assert (
            model_dict["__version__"] == cls().__version__
        ), "Incompatible version detected."

        return cls(**model_dict)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def hops_config(self):
        """
        The hops :any:`hops.core.hierarchy_params.HIParams` parameter object
        for this system.
        """

        g_1, w_1 = self.bcf_coefficients(0)
        g_2, w_2 = self.bcf_coefficients(1)

        system = params.SysP(
            H_sys=self.system(0).full()
            + self.system(1).full()
            + self.interaction.full(),
            L=[self.bath_coupling(0).full(), self.bath_coupling(1).full()],
            g=[g_1, g_2],
            w=[w_1, w_2],
            bcf_scale=[self.bcf_scale(0), self.bcf_scale(1)],
            T=self.T,
            description=self.description,
            psi0=self.ψ_0.full().flatten(),
        )

        hierarchy = params.HiP(
            seed=0,
            nonlinear=True,
            terminator=False,
            result_type=params.ResultType.ZEROTH_AND_FIRST_ORDER,
            accum_only=False,
            rand_skip=None,
            truncation_scheme=TruncationScheme_Power_multi.from_g_w(
                g=system.g,
                w=system.w,
                p=[1, 1],
                q=[0.5, 0.5],
                kfac=[float(fac) for fac in self.k_fac],
            ),
            save_therm_rng_seed=True,
            auto_normalize=True,
        )

        default_solver_args = dict(rtol=1e-8, atol=1e-8)
        default_solver_args.update(self.solver_args)

        integration = params.IntP(
            t_max=float(self.t_max),
            t_steps=int(float(self.t_max) / float(self.resolution)) + 1,
            solver_args=default_solver_args,
        )

        return params.HIParams(
            SysP=system,
            IntP=integration,
            HiP=hierarchy,
            Eta=[self.driving_process(0), self.driving_process(1)],
            EtaTherm=[self.thermal_process(0), self.thermal_process(1)],
        )


class JSONEncoder(json.JSONEncoder):
    """
    A custom encoder to serialize objects occuring in
    :any:`TwoQubitModel`.
    """

    @singledispatchmethod
    def default(self, obj: Any):
        return super().default(obj)

    @default.register
    def _(self, arr: np.ndarray):
        return {"type": "array", "value": arr.tolist()}

    @default.register
    def _(self, obj: qt.Qobj):
        return {
            "type": "Qobj",
            "value": obj.full(),
            "dims": obj.dims,
            "obj_type": obj.type,
        }

    @default.register
    def _(self, obj: complex):
        return {"type": "complex", "re": obj.real, "im": obj.imag}

    @default.register
    def _(self, obj: StocProcTolerances):
        return {"type": "StocProcTolerances", "value": dataclasses.asdict(obj)}


def _object_hook(dct: dict[str, Any]):
    """A custom decoder for the types introduced in :any:`JSONEncoder`."""
    if "type" in dct:
        type = dct["type"]

        if type == "array":
            return np.array(dct["value"])

        if type == "Qobj":
            return qt.Qobj(dct["value"], dims=dct["dims"], type=dct["obj_type"])

        if type == "complex":
            return dct["re"] + 1j * dct["im"]

        if type == "StocProcTolerances":
            return StocProcTolerances(**dct["value"])

    return dct
