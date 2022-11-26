r"""HOPS Configurations for a simple qubit otto cycle."""

from dataclasses import dataclass, field
import hopsflow
from numpy.typing import NDArray
from typing import Any, Optional, SupportsFloat, Union
import hops.util.bcf
import hops.util.bcf_fits
import hops.core.hierarchy_parameters as params
import numpy as np
import qutip as qt
from hops.util.abstract_truncation_scheme import TruncationScheme_Simplex
from hops.util.truncation_schemes import (
    TruncationScheme_Power_multi,
    BathMemory,
)
import stocproc as sp

from beartype import beartype
from .utility import StocProcTolerances, bcf_scale
from .model_base import Model
import scipy.special
import hopsflow
from hops.util.dynamic_matrix import DynamicMatrix, ConstantMatrix, SmoothStep, Periodic
from .one_qubit_model import QubitModelMutliBath


@beartype
@dataclass(eq=False)
class OttoEngine(Model):
    r"""
    A class to dynamically calculate all the otto motor model
    parameters and generate the HOPS configuration.  Uses
    :any:`one_qubit_model.QubitModelMultiBath` internally.

    All attributes can be changed after initialization.
    """

    __version__: int = 2

    δ: list[SupportsFloat] = field(default_factory=lambda: [0.1] * 2)
    """The bath coupling factors."""

    ω_c: list[SupportsFloat] = field(default_factory=lambda: [2] * 2)
    """The cutoff frequencies :math:`ω_c`."""

    s: list[SupportsFloat] = field(default_factory=lambda: [1] * 2)
    """The BCF s parameter."""

    ω_s: list[SupportsFloat] = field(default_factory=lambda: [0] * 2)
    """The SD shift frequencies :math:`ω_s`."""

    therm_methods: list[str] = field(default_factory=lambda: ["tanhsinh"] * 2)
    """
    The methods used for the thermal stochastic process.  Either
    ``tanhsinh`` or ``fft``.
    """

    L: list[DynamicMatrix] = field(
        default_factory=lambda: [ConstantMatrix(1 / 2 * qt.sigmax().full())] * 2  # type: ignore
    )
    """
    The :math:`L` coupling operators with shape ``(2, 2)``.
    """

    H_0: np.ndarray = field(
        default_factory=(1 / 2 * qt.sigmax().full()) + np.eye(2)  # type: ignore
    )
    """
    The :math:`H_0` system hamiltonian with shape ``(2, 2)``.

    It will get shifted and normalized so that its smallest eigenvalue
    is zero and its largest one is one.
    """

    H_1: np.ndarray = field(
        default_factory=(1 / 2 * qt.sigmax().full()) + np.eye(2)  # type: ignore
    )
    """
    The :math:`H_1` shape ``(2, 2)``.

    It will get shifted and normalized so that its smallest eigenvalue
    is zero and its largest one is one.
    """

    T: list[SupportsFloat] = field(default_factory=lambda: [0] * 2)
    """The temperatures of the baths."""

    ###########################################################################
    #                             HOPS Parameters                             #
    ###########################################################################

    description: str = ""
    """A free-form description of the model instance."""

    bcf_terms: list[int] = field(default_factory=lambda: [6] * 2)
    """How many bcf terms to use in the expansions of the BCF."""

    ψ_0: qt.Qobj = qt.basis([2], [1])
    """The initial state. The default is the 'down' state."""

    t: NDArray[np.float64] = np.linspace(0, 10, 1000)
    """The simulation time points."""

    k_max: int = 5
    """The kmax parameter for the truncation scheme.

    See
    :any:`hops.util.abstract_truncation_scheme.TruncationScheme_Simplex`
    """

    solver_args: dict[str, Any] = field(default_factory=dict)
    """Extra arguments for :any:`scipy.integrate.solve_ivp`."""

    driving_process_tolerances: list[StocProcTolerances] = field(
        default_factory=lambda: [StocProcTolerances(), StocProcTolerances()]
    )
    """
    The integration and interpolation tolerance for the driving
    processes.
    """

    thermal_process_tolerances: list[StocProcTolerances] = field(
        default_factory=lambda: [StocProcTolerances(), StocProcTolerances()]
    )
    """
    The integration and interpolation tolerance for the thermal noise
    processes.
    """

    ###########################################################################
    #                              Cycle Settings                             #
    ###########################################################################

    Θ: float = 1
    """The period of the modulation."""

    Δ: float = 1
    """The expansion ratio of the modulation."""

    λ_u: float = 1
    """
    The portion of the cycle where the transition from ``H_0`` to
    ``H_1`` begins. Ranges from ``0`` to ``1``.
    """

    λ_h: float = 1
    """
    The portion of the cycle where the transition from ``H_0`` to
    ``H_1`` ends. Ranges from ``0`` to ``1``.
    """

    λ_d: float = 1
    """
    The portion of the cycle where the transition from ``H_1`` to
    ``H_0`` begins. Ranges from ``0`` to ``1``.
    """

    @property
    def τ_l(self) -> float:
        """
        The length of the timespan the Hamiltonian matches ``H_0``.
        """

        return self.λ_u * self.Θ

    @property
    def τ_h(self) -> float:
        """
        The length of the timespan the Hamiltonian matches ``H_1``.
        """

        return (self.λ_h - self.λ_d) * self.Θ

    @property
    def τ_u(self) -> float:
        """
        The length of the trasition of the Hamiltonian from ``H_0`` to
        ``H_1``.
        """

        return (self.λ_h - self.λ_u) * self.Θ

    @property
    def τ_d(self) -> float:
        """
        The length of the trasition of the Hamiltonian from ``H_1`` to
        ``H_0``.
        """

        return (1 - self.λ_d) * self.Θ

    @property
    def Ω(self) -> float:
        """The base Angular frequency of the cycle."""

        return (2 * np.pi) / self.Θ

    @property
    def H(self) -> DynamicMatrix:
        r"""
        Returns the modulated system Hamiltonian.

        The system hamiltonian will always be :math:`ω_{\max} * H_1 +
        (ω_{\max} - ω_{\min}) * f(τ) * H_1` where ``H_0`` is a fixed
        matrix and :math:`f(τ)` models the time dependence.

        The modulation :math:`f(τ)` consists of constants and smooth
        step functions.  For :math:`τ < τ_l` :math:`f(τ) = 0` and for
        :math:`τ_l + τ_u <= τ < τ_l + τ_u + τ_h` :math:`f(τ) =
        ε_{\max}`.  The transitions between those states last
        :math:`τ_u` for :math:`H_0` to :math:`H_1` and :math:`τ_d` for
        :math:`H_1` to :math:`H_0`.

        The modulation is cyclical with period :any:`T`.
        """

        H_0 = normalize_hamiltonian(self.H_0)
        H_1 = normalize_hamiltonian(self.H_1)

        one_cycle = ConstantMatrix(H_0) + self.Δ * (
            SmoothStep(H_1, self.λ_u * self.Θ, self.λ_h * self.Θ)
            - SmoothStep(H_1, self.λ_d * self.Θ, self.Θ)
        )

        return Periodic(one_cycle, self.Θ)

    @property
    def qubit_model(self) -> QubitModelMutliBath:
        """Returns the underlying Qubit model."""

        return QubitModelMutliBath()


def normalize_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvals(hamiltonian)

    normalized = hamiltonian - eigvals.min() * np.eye(
        hamiltonian.shape[0], dtype=hamiltonian.dtype
    )
    normalized /= eigvals.max() - eigvals.min()

    return normalized
