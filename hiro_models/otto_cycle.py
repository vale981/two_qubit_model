r"""HOPS Configurations for a simple qubit otto cycle."""

from dataclasses import dataclass, field
import hopsflow
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
from scipy.optimize import minimize_scalar
import hopsflow
from hops.util.dynamic_matrix import (
    DynamicMatrix,
    ConstantMatrix,
    SmoothStep,
    Periodic,
    MatrixType,
)
from .one_qubit_model import QubitModelMutliBath

from numpy.typing import ArrayLike, NDArray
from numbers import Real


Timings = tuple[Real, Real, Real, Real]
Orders = tuple[int, int]


class SmoothlyInterpolatdPeriodicMatrix(DynamicMatrix):
    """A periodic dynamic matrix that smoothly interpolates between
    two matrices using :any:`SmoothStep`.

    :param matrices: The two matrices ``M1`` and ``M2`` to interpolate
        between.
    :param timings: A tuple that contains the times (relative to the
        period) when the transition from ``M1`` to ``M2`` begins, when
        it ends and when the reverse transition begins and when it ends.
    :param period: The period of the modulation.
    :param orders: The orders of the smoothstep functions that are
        being used.  See also :any:`SmoothStep`.
    :param amplitudes: The amplitudes of the modulation.
    :param deriv: The order of derivative of the matrix.
    """

    def __init__(
        self,
        matrices: tuple[Union[ArrayLike, list[list]], Union[ArrayLike, list[list]]],
        timings: Timings,
        period: float,
        orders: tuple = (2, 2),
        amplitudes: tuple[float, float] = (1, 1),
        deriv: int = 0,
    ):
        self._matrices = matrices
        self._timings = timings
        self._period = period
        self._orders = orders
        self._amplitudes = amplitudes
        self._deriv = deriv

        M_1, M_2 = matrices
        s_1, s_2 = orders
        a_1, a_2 = amplitudes

        one_cycle: DynamicMatrix = a_1 * (
            (ConstantMatrix(M_1) if deriv == 0 else ConstantMatrix(np.zeros_like(M_1)))
        )

        if a_1 != 0:
            one_cycle += a_1 * (
                SmoothStep(
                    M_1, timings[2] * period, timings[3] * period, s_2, deriv=deriv
                )
                - SmoothStep(
                    M_1, timings[0] * period, timings[1] * period, s_1, deriv=deriv
                )
            )

        if a_2 != 0:
            one_cycle += a_2 * (
                SmoothStep(
                    M_2, timings[0] * period, timings[1] * period, s_1, deriv=deriv
                )
                - SmoothStep(
                    M_2, timings[2] * period, timings[3] * period, s_2, deriv=deriv
                )
            )

        self._m = Periodic(one_cycle, period)

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._m.call(t)

    def derivative(self):
        return self.__class__(
            matrices=self._matrices,
            timings=self._timings,
            period=self._period,
            orders=self._orders,
            amplitudes=self._amplitudes,
            deriv=self._deriv + 1,
        )

    def __getstate__(self):
        return dict(
            matrices=self._matrices,
            timings=self._timings,
            period=self._period,
            orders=self._orders,
            amplitude=self._amplitudes,
            deriv=self._deriv,
        )


@beartype
@dataclass(eq=False)
class OttoEngine(QubitModelMutliBath):
    r"""
    A class to dynamically calculate all the otto motor model
    parameters and generate the HOPS configuration.  Uses
    :any:`one_qubit_model.QubitModelMultiBath` internally.

    All attributes can be changed after initialization.
    """

    __version__: int = 1

    H_0: np.ndarray = field(
        default_factory=lambda: 1 / 2 * (qt.sigmaz().full() + np.eye(2))  # type: ignore
    )
    """
    The :math:`H_0` system hamiltonian with shape ``(2, 2)``.

    It will get shifted and normalized so that its smallest eigenvalue
    is zero and its largest one is one.
    """

    H_1: np.ndarray = field(
        default_factory=lambda: 1 / 2 * (qt.sigmaz().full() + np.eye(2))  # type: ignore
    )
    """
    The :math:`H_1` shape ``(2, 2)``.

    It will get shifted and normalized so that its smallest eigenvalue
    is zero and its largest one is one.
    """

    L: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: tuple([1 / 2 * (qt.sigmax().full())] * 2)  # type: ignore
    )
    """The bare coupling operators to the two baths."""

    ω_s: list[Union[SupportsFloat, str]] = field(default_factory=lambda: [2] * 2)
    """
    The shift frequencies :math:`ω_s`.  If set to ``'auto'``, the
    (thermal) spectral densities will be shifted so that the coupling
    of the first bath is resonant with the hamiltonian before the
    expansion of the energy gap and the second bath is resonant with
    the hamiltonian after the expansion.
    """

    ###########################################################################
    #                              Cycle Settings                             #
    ###########################################################################

    Θ: float = 1
    """The period of the modulation."""

    Δ: float = 1
    """The expansion ratio of the modulation."""

    timings_H: Timings = field(default_factory=lambda: (0.25, 0.5, 0.75, 1))
    """The timings for the ``H`` modulation. See :any:`SmoothlyInterpolatdPeriodicMatrix`."""

    orders_H: Orders = field(default_factory=lambda: (2, 2))
    """The smoothness of the modulation of ``H``."""

    timings_L: tuple[Timings, Timings] = field(
        default_factory=lambda: ((0, 0.05, 0.15, 0.2), (0.5, 0.55, 0.65, 0.7))
    )
    """The timings for the ``L`` modulation. See :any:`SmoothlyInterpolatdPeriodicMatrix`."""

    orders_L: tuple[Orders, Orders] = field(default_factory=lambda: ((2, 2), (2, 2)))
    """The smoothness of the modulation of ``L``."""

    @property
    def τ_expansion_finished(self):
        return self.timings_H[1] * self.Θ

    def __post_init__(self):
        def objective(ω_s, ω_exp, i):
            self.ω_s[i] = ω_s
            return -self.full_thermal_spectral_density(i)(ω_exp)

        ω_exps = [
            get_energy_gap(self.H(0)),
            get_energy_gap(self.H(self.τ_expansion_finished)),
        ]

        for i, shift in enumerate(self.ω_s):
            if shift == "auto":
                res = minimize_scalar(
                    objective,
                    1,
                    method="bounded",
                    bounds=(0.01, ω_exps[i]),
                    options=dict(maxiter=100),
                    args=(ω_exps[i], i),
                )

                if not res.success:
                    raise RuntimeError("Cannot optimize SD shift.")

                self.ω_s[i] = res.x

    @property
    def H(self) -> DynamicMatrix:
        """
        Returns the modulated system Hamiltonian.

        The system hamiltonian will always be :math:`ω_{\max} * H_1 +
        (ω_{\max} - ω_{\min}) * f(τ) * H_1` where ``H_0`` is a fixed
        matrix and :math:`f(τ)` models the time dependence.  The time
        dependce is implemented via
        :any:`SmoothlyInterpolatdPeriodicMatrix` and leads to a
        modulation of the levelspacing between ``ε_min=1`` and
        ``ε_max`` so that ``ε_max/ε_min - 1 = Δ``.

        The modulation is cyclical with period :any:`Θ`.
        """

        return SmoothlyInterpolatdPeriodicMatrix(
            (normalize_hamiltonian(self.H_0), normalize_hamiltonian(self.H_1)),
            self.timings_H,
            self.Θ,
            self.orders_H,
            (1, self.Δ + 1),
        )

    # we black-hole the H setter in this model
    @H.setter
    def H(self, _):
        pass

    @property
    def coupling_operators(self) -> list[DynamicMatrix]:
        return [
            SmoothlyInterpolatdPeriodicMatrix(
                (np.zeros_like(L_i), L_i),
                timings,
                self.Θ,
                orders,
                (0, 1),
            )
            for L_i, timings, orders in zip(self.L, self.timings_L, self.orders_L)
        ]

    # @property
    # def qubit_model(self) -> QubitModelMutliBath:
    #     """Returns the underlying Qubit model."""

    #     return QubitModelMutliBath(
    #         δ=self.δ,
    #         ω_c=self.ω_c,
    #         ω_s=self.ω_s,
    #         t=self.t,
    #         ψ_0=self.ψ_0,
    #         description=f"The qubit model underlying the otto cycle with description: {self.description}.",
    #         truncation_scheme="simplex",
    #         k_max=self.k_max,
    #         bcf_terms=self.bcf_terms,
    #         driving_process_tolerances=self.driving_process_tolerances,
    #         thermal_process_tolerances=self.thermal_process_tolerances,
    #         T=self.T,
    #         L=self.L,
    #         H=self.H,
    #         therm_methods=self.therm_methods,
    #     )


def normalize_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvals(hamiltonian)

    normalized = hamiltonian - eigvals.min() * np.eye(
        hamiltonian.shape[0], dtype=hamiltonian.dtype
    )
    normalized /= (eigvals.max() - eigvals.min()).real

    return normalized


def get_energy_gap(hamiltonian: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(hamiltonian)

    return (eigvals.max() - eigvals.min()).real
