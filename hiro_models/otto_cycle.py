r"""HOPS Configurations for a simple qubit otto cycle."""

import math
from dataclasses import dataclass, field
from typing import SupportsFloat, Union
import numpy as np
import qutip as qt

from beartype import beartype, BeartypeConf
from scipy.optimize import minimize_scalar
from hops.util.dynamic_matrix import (
    DynamicMatrix,
    ConstantMatrix,
    SmoothStep,
    Periodic,
    MatrixType,
    Shift,
)
from .one_qubit_model import QubitModelMutliBath
from .utility import linspace_with_strobe, strobe_times
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from hops.core.hierarchy_data import HIData
from hopsflow.util import EnsembleValue

Timings = tuple[float, float, float, float]
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
    :param amplitudes: The amplitudes of the modulation.
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
            amplitudes=self._amplitudes,
            deriv=self._deriv,
        )


@beartype(conf=BeartypeConf(is_pep484_tower=True))
@dataclass(eq=False)
class OttoEngine(QubitModelMutliBath):
    r"""
    A class to dynamically calculate all the otto motor model
    parameters and generate the HOPS configuration.  Uses
    :any:`one_qubit_model.QubitModelMultiBath` internally.

    All attributes can be changed after initialization.

    The bath correlation functions are normalized, so that
    their corresponding thermal SDs are of the same magnitude
    at the resonance frequencies.
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

    H_bias: DynamicMatrix = field(default_factory=lambda: ConstantMatrix(np.zeros((2, 2))))  # type: ignore
    """
    The :math:`H_B` shape ``(2, 2)``.

    This bias will be added to the system Hamiltonian unaltered.
    """

    L: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: tuple([(1 / 2 * (qt.sigmax().full())), (1 / 2 * (qt.sigmax().full()))])  # type: ignore
    )
    """The bare coupling operators to the two baths."""

    ω_s_extra: list[float] = field(default_factory=lambda: [0] * 2)
    """
    The shift frequencies :math:`ω_s` applied on top of the automatic shift.
    """

    ###########################################################################
    #                              Cycle Settings                             #
    ###########################################################################

    Θ: float = 1
    """The period of the modulation."""

    Δ: float = 1
    """The expansion ratio of the modulation."""

    timings_H: Timings = field(default_factory=lambda: (0, 0.1, 0.5, 0.6))
    """The timings for the ``H`` modulation. See :any:`SmoothlyInterpolatdPeriodicMatrix`."""

    orders_H: Orders = field(default_factory=lambda: (2, 2))
    """The smoothness of the modulation of ``H``."""

    timings_L: tuple[Optional[Timings], Optional[Timings]] = field(
        default_factory=lambda: ((0.6, 0.7, 0.9, 1), (0.1, 0.2, 0.4, 0.5))
    )
    """
    The timings for the ``L`` modulation.  See
    :any:`SmoothlyInterpolatdPeriodicMatrix`.  If no timing is given,
    modulation is disabled.
    """

    L_shift: tuple[float, float] = field(default_factory=lambda: (0, 0))
    """The time shift of the ``L`` modulation."""

    orders_L: tuple[Orders, Orders] = field(default_factory=lambda: ((2, 2), (2, 2)))
    """The smoothness of the modulation of ``L``."""

    num_cycles: int = 1
    """How many cycles to simulate."""

    dt: float = 0.001
    """The time resolution relative to the period of modulation."""

    shift_to_resonance: tuple[bool, bool] = field(default_factory=lambda: (True, True))
    """Whether to to shift the spectral densities to the resonance point."""

    @property
    def τ_max(self) -> float:
        """The maximum simulation time."""

        return self.num_cycles * self.Θ

    @property
    def t(self) -> NDArray[np.float64]:
        """The simulation time."""

        return linspace_with_strobe(
            0,
            self.τ_max,
            int(self.τ_max // (self.dt * self.Θ)) + 1,
            self.Ω,
        )

    @property
    def strobe(self):
        """
        Returns a tuple with the times where a cycle is complete and
        their corresponding indices.
        """

        times = np.arange(self.num_cycles + 1) * self.Θ
        indices = np.searchsorted(self.t, (np.arange(self.num_cycles + 1) * 2 * np.pi / self.Ω))
        return times, indices

    @t.setter
    def t(self, _):
        pass

    @property
    def ω_s(self) -> list[float]:
        """
        The frequency shifts of the spectral density.  Calculated so
        that the effective thermal SD has maximum magnitude at the
        resonance frequencies of the hamiltonian. :any:`ω_s_extra` is added to those values.
        """

        return [
            extra + ((gap - float(ω_c) * float(s)) if do_shift else 0)
            for ω_c, s, gap, extra, do_shift in zip(
                self.ω_c,
                self.s,
                self.energy_gaps,
                self.ω_s_extra,
                self.shift_to_resonance,
            )
        ]
        # super_instance = QubitModelMutliBath(ω_c=self.ω_c, s=self.s, T=self.T)

        # def objective(ω_s, ω_exp, i):
        #     super_instance.ω_s[i] = ω_s
        #     return -super_instance.full_thermal_spectral_density(i)(ω_exp)

        # ω_s = [ω for ω in self.ω_s_extra]
        # ω_exps = self.energy_gaps
        # for i, shift in enumerate(self.ω_s_extra):
        #     res = minimize_scalar(
        #         objective,
        #         1,
        #         method="bounded",
        #         bounds=(0.01, ω_exps[i]),
        #         options=dict(maxiter=100),
        #         args=(ω_exps[i], i),
        #     )

        #     if not res.success:
        #         raise RuntimeError("Cannot optimize SD shift.")

        #     ω_s[i] = shift + round(res.x, number_magnitude(ω_exps[i]) + 3)

        # return ω_s

    @ω_s.setter
    def ω_s(self, _):
        pass

    @property
    def energy_gaps(self) -> tuple[float, float]:
        """
        The energy gaps of the working medium in compressed and
        expanded state.
        """

        return tuple(
            sorted(
                (
                    get_energy_gap(self.H(0)),
                    get_energy_gap(self.H(self.τ_expansion_finished)),
                )
            )
        )

    @property
    def τ_expansion_finished(self):
        """Time when the working medium is fully expanded."""
        return self.timings_H[1] * self.Θ

    @property
    def τ_compressed(self):
        """Time when the working medium is fully copressed."""
        return 0

    @property
    def bcf_scales(self) -> list[float]:
        gaps = self.energy_gaps
        return [
            float(δ) / float(self.full_thermal_spectral_density(i)(gap))
            for (i, gap), δ in zip(enumerate(gaps), self.δ)
        ]

    @property
    def H(self) -> DynamicMatrix:
        r"""
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

        return (
            SmoothlyInterpolatdPeriodicMatrix(
                (self.H_0, self.H_1),
                self.timings_H,
                self.Θ,
                self.orders_H,
                (1, self.Δ + 1),
            )
            + self.H_bias
        )

    # we black-hole the H setter in this model
    @H.setter
    def H(self, _):
        pass

    @property
    def coupling_operators(self) -> list[DynamicMatrix]:
        return [
            Shift(
                ConstantMatrix(L_i)
                if timings is None
                else SmoothlyInterpolatdPeriodicMatrix(
                    (np.zeros_like(L_i), L_i),
                    timings,
                    self.Θ,
                    orders,
                    (0, 1),
                ),
                shift * self.Θ,
            )
            for L_i, timings, orders, shift in zip(
                self.L, self.timings_L, self.orders_L, self.L_shift
            )
        ]

    @property
    def Ω(self) -> float:
        """The modulation base angular frequency."""

        return 2 * np.pi / self.Θ

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

    def steady_index(
        self,
        fraction: Optional[float] = None,
        observable: Optional[EnsembleValue] = None,
        data: Optional[HIData] = None,
        **kwargs
    ):
        """
        Determine using the system energy or the ``observable``
        (``data`` and ``kwargs`` are being passed to
        :any:`system_energy`) when the periodic steady state has been
        reached.

        This is being done by comparing the change of the observable
        to the next cycle and finding the time where it is compatible
        with its value in the next cycle in the ``fraction`` of cases.
        The default value for ``fraction`` is ``0.68`` (one sigma).
        """

        _, indices = self.strobe

        fraction = fraction or 0.68
        observable = observable or self.system_energy(data, **kwargs)

        cycles = [
            observable.slice(slice(begin, end))
            for end, begin in zip(indices[1:], indices[:-1])
        ]
        consistency = np.array(
            [
                nxt.consistency(this) / 100 > fraction
                for nxt, this in zip(cycles[1:], cycles[:-1])
            ]
        )

        if sum(consistency) == 0:
            return None

        return np.argmax(consistency) + 1

    def get_steady_values(self, value: EnsembleValue, steady_idx: int, *args, **kwargs):
        """
        Get the value of ``value`` at the steady state indices.  For
        the rest arguments sse :any:`steady_index`.
        """

        _, indices = self.strobe

        return value.slice(indices[steady_idx:])

    def steady_total_energy_change(
        self, steady_idx: int, data: Optional[HIData] = None, **kwargs
    ):
        """
        The steady energy change computed from ``data`` or online
        data.  The ``kwargs`` are being passed on to the analysis
        methods.
        """

        energies = self.get_steady_values(
            self.total_energy_from_power(data, **kwargs), steady_idx, data, **kwargs
        )
        Δ_energies = energies.slice(slice(1, None)) - energies.slice(slice(0, -1))

        return Δ_energies.mean

    def steady_bath_energy_change(
        self, bath: int, steady_idx: int, data: Optional[HIData] = None, **kwargs
    ):
        """
        The steady energy change computed from ``data`` or online
        data.  The ``kwargs`` are being passed on to the analysis
        methods.
        """

        energies = self.get_steady_values(
            self.bath_energy(data, **kwargs).for_bath(bath), steady_idx, data, **kwargs
        )

        Δ_energies = energies.slice(slice(1, None)) - energies.slice(slice(0, -1))

        return Δ_energies.mean

    def power(self, steady_idx: int, *args, **kwargs):
        """
        Calculate the mean steady state power.  For the arguments see
        :any:`steady_energy_change`.
        """

        _, indices = self.strobe

        return self.total_power().slice(slice(indices[steady_idx], None, 1)).mean

    def efficiency(self, steady_idx: int, data: Optional[HIData] = None, **kwargs):
        """
        Calculate the steady state efficiency.  For the arguments see
        :any:`steady_energy_change`.
        """

        Δ_bath = self.steady_bath_energy_change(1, steady_idx, data, **kwargs)

        return self.steady_total_energy_change(steady_idx, data, **kwargs) / Δ_bath.mean


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


def number_magnitude(number: float) -> int:
    return int(math.log10(number))
