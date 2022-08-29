r"""
 Operators for a general model of one qubit coupled to a single bath.
 The energy scale is the characteristic energy of the qubit :math:`ω =
 1`.

 The total hamiltonian has the form

 ..  math::

     H=\frac{1}{2}σ_z
       + \sqrt{δ} ∑_λ (L^† g_λ  b^i_λ + L g_λ^\ast b^{i,†}_λ) + H_B.

 The BCF is normalized so that the integral over its imaginary part is
 :math:`-1`.  The bath coupling strength is divided by :math:`\langle
 L L^†\rangle 2` with respect to the inital state to normalize the
 interaction energy to about the order of :math:`ω=1`.
 """


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
from hops.util.dynamic_matrix import DynamicMatrix, ConstantMatrix


@beartype
@dataclass(eq=False)
class QubitModel(Model):
    """
    A class to dynamically calculate all the one qubit model parameters and
    generate the HOPS configuration.

    All attributes can be changed after initialization.
    """

    __version__: int = 2

    δ: SupportsFloat = 0.1
    """The bath coupling factor."""

    bcf_norm_method: str = field(default_factory=lambda: "pure_dephasing")
    r"""The normalization of the bath correlation function.

     - `"pure dephasing"` corresponds to :math:`\Im ∫_0^∞ α(τ) dτ = -1`
     - `"unit"` corresponds to :math:`α(0)=1` (zero temperature BCF)
     - `"unit_therm"` corresponds to :math:`α_β(0)=1`
     - `"sd_peak"` corresponds to :math:`\max_ω J(ω)=1` (zero temperature SD)
     - `"sd_peak_therm"` corresponds to :math:`\max_ω J_β(ω)=1`
    """

    ω_c: SupportsFloat = 2
    """The cutoff frequency :math:`ω_c`."""

    ω_s: SupportsFloat = 0
    """The SD shift frequency :math:`ω_s`."""

    therm_method: str = "tanhsinh"
    """
    The method used for the thermal stochastic process.  Either
    ``tanhsinh`` or ``fft``.
    """

    s: SupportsFloat = 1
    """The BCF s parameter."""

    L: DynamicMatrix = field(default_factory=lambda: ConstantMatrix(1 / 2 * qt.sigmax().full()))  # type: ignore
    """
    The :math:`L` coupling operator with shape ``(2, 2)``.
    """

    T: SupportsFloat = 0
    """The temperature of the bath."""

    ###########################################################################
    #                             HOPS Parameters                             #
    ###########################################################################

    description: str = ""
    """A free-form description of the model instance."""

    bcf_terms: int = 5
    """How many bcf terms to use in the expansions of the BCF."""

    ψ_0: qt.Qobj = qt.basis([2], [1])
    """The initial state."""

    t: NDArray[np.float64] = np.linspace(0, 10, 1000)
    """The simulation time points."""

    k_fac: SupportsFloat = 1.7
    """The k_fac parameters for the truncation scheme.

    See
    :any:`hops.util.truncation_schemes.TruncationScheme_Power_multi`.
    """

    k_max: int = 10
    """The kmax parameter for the truncation scheme.

    See
    :any:`hops.util.abstract_truncation_scheme.TruncationScheme_Simplex`
    """

    influence_tolerance: SupportsFloat = 1e-2
    """The ``influecne_tolerance`` parameter for the truncation
    scheme.

    See :any:`hops.util.truncation_schemes.BathMemory`.
    """

    truncation_scheme: str = "bath_memory"
    """The truncation scheme to use."""

    solver_args: dict[str, Any] = field(default_factory=dict)
    """Extra arguments for :any:`scipy.integrate.solve_ivp`."""

    driving_process_tolerance: StocProcTolerances = field(
        default_factory=lambda: StocProcTolerances()
    )
    """
    The integration and interpolation tolerance for the driving
    processes.
    """

    thermal_process_tolerance: StocProcTolerances = field(
        default_factory=lambda: StocProcTolerances()
    )
    """
    The integration and interpolation tolerance for the thermal noise
    processes.
    """

    H: DynamicMatrix = field(
        default_factory=lambda: ConstantMatrix(1 / 2 * qt.sigmaz().full())
    )  # type: ignore
    """
    The system hamiltonian :math:`H` with shape ``(2, 2)``.
    """

    @property
    def coupling_operators(self) -> list[DynamicMatrix]:
        """The bath coupling operators :math:`L`."""

        return [self.L]

    @property
    def system(self) -> DynamicMatrix:
        """The system hamiltonian."""

        return self.H  # type: ignore

    @property
    def bcf_norm(self) -> float:
        """The normalization factor for the BCF.
        It is not used when generating the stochastic process due to
        numerical reasons.  It is being incorporated into the
        :any:`bcf_scale`.
        """

        return (
            np.pi
            * float(self.s)
            / (
                scipy.special.gamma(float(self.s) + 1)
                * float(self.ω_c) ** float(self.s)
            )
        )

    @property
    def full_thermal_spectral_density(self):
        """
        :returns: The full thermal spectral density.
        """

        if self.T == 0:
            return self.spectral_density

        def thermal_sd(ω):
            return self.spectral_density(ω) * (
                1 / (np.expm1(ω / self.T)) + np.heaviside(ω, 0)
            )

        return thermal_sd

    @property
    def full_thermal_bcf(self):
        """
        :returns: The full thermal bath correlation function.
        """

        if self.T == 0:
            return self.bcf

        def thermal_bcf(t):
            return self.bcf(t) + 2 * (self.thermal_correlations(t).real)

        return thermal_bcf

    @property
    def bcf_scale(self) -> float:
        """
        The scaling factor of the BCF.
        """

        if self.bcf_norm_method == "pure_dephasing":
            return bcf_scale(self.δ, self.L, self.t.max(), self.s, self.ω_c)

        if self.bcf_norm_method == "unit":
            return float(self.δ) / self.bcf(0).real

        if self.bcf_norm_method == "unit_therm":
            return float(self.δ) / self.full_thermal_bcf(0).real

        if self.bcf_norm_method == "sd_peak":
            return (
                float(self.δ) / self.spectral_density(self.ω_c * self.s + self.ω_s).real
            )

        if self.bcf_norm_method == "sd_peak_therm":
            return (
                float(self.δ)
                / self.full_thermal_spectral_density(self.ω_c * self.s + self.ω_s).real
            )

    @property
    def bcf_scales(self) -> list[float]:
        """The scaling factors for the bath correlation functions."""

        return [self.bcf_scale]

    @property
    def bcf(self) -> Union[hops.util.bcf.OhmicBCF_zeroTemp, hops.util.bcf.ShiftedBCF]:
        """
        The normalized zero temperature BCF.
        """

        bcf = hops.util.bcf.OhmicBCF_zeroTemp(
            s=self.s, eta=1, w_c=self.ω_c, normed=False
        )

        if float(self.ω_s) > 0:
            return hops.util.bcf.ShiftedBCF(bcf, float(self.ω_s))

        return bcf

    @property
    def spectral_density(
        self,
    ) -> Union[hops.util.bcf.OhmicSD_zeroTemp, hops.util.bcf.ShiftedSD]:
        """
        The normalized zero temperature spectral density.
        """

        sd = hops.util.bcf.OhmicSD_zeroTemp(
            s=float(self.s),
            w_c=float(self.ω_c),
            eta=1,
            normed=False,
        )

        if float(self.ω_s) > 0:
            return hops.util.bcf.ShiftedSD(sd, float(self.ω_s))

        return sd

    @property
    def thermal_correlations(
        self,
    ) -> Union[
        Optional[hops.util.bcf.Ohmic_StochasticPotentialCorrelations],
        hops.util.bcf.ShiftedBCF,
    ]:
        """
        The normalized thermal noise corellation function.
        """

        if self.T == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialCorrelations(
            s=self.s,
            eta=1,
            w_c=self.ω_c,
            normed=False,
            beta=1 / float(self.T),
            shift=float(self.ω_s),
        )

    @property
    def thermal_spectral_density(
        self,
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialDensity]:
        """
        The normalized thermal noise spectral density.
        """

        if self.T == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialDensity(
            s=self.s,
            eta=1,
            w_c=self.ω_c,
            normed=False,
            beta=1.0 / float(self.T),
            shift=float(self.ω_s),
        )

    def bcf_coefficients(
        self, n: Optional[int] = None
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        """
        The normalizedzero temperature BCF fit coefficients
        :math:`G_i,W_i` with ``n`` terms.
        """

        n = n or self.bcf_terms
        g, w = self.bcf.exponential_coefficients(n)
        return ([g], [w])

    @staticmethod
    def basis(n: int = 1) -> qt.Qobj:
        """
        A state with of the qubit in the state state ``n`` where ``1``
        means down and ``0`` means up.
        """

        return qt.basis([2], [n])

    @property
    def driving_process(self) -> sp.StocProc:
        """The driving stochastic process of the ``i``th bath."""

        return sp.StocProc_FFT(
            spectral_density=self.spectral_density,
            alpha=self.bcf,
            t_max=self.t.max(),
            intgr_tol=self.driving_process_tolerance.integration,
            intpl_tol=self.driving_process_tolerance.interpolation,
            negative_frequencies=False,
        )

    @property
    def thermal_process(self) -> Optional[sp.StocProc]:
        """The thermal noise stochastic process."""

        if self.T == 0:
            return None

        return (
            sp.StocProc_TanhSinh if self.therm_method == "tanhsinh" else sp.StocProc_FFT
        )(
            spectral_density=self.thermal_spectral_density,
            alpha=self.thermal_correlations,
            t_max=self.t.max(),
            intgr_tol=self.thermal_process_tolerance.integration,
            intpl_tol=self.thermal_process_tolerance.interpolation,
            negative_frequencies=False,
        )

    @property
    def thermal_processes(self) -> list[Optional[hopsflow.hopsflow.StocProc]]:
        """
        The thermal noise stochastic processes for each bath.
        :any:`None` means zero temperature.
        """

        return [self.thermal_process]

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    @property
    def hops_config(self) -> params.HIParams:
        """
        The hops :any:`hops.core.hierarchy_params.HIParams` parameter object
        for this system.
        """

        g, w = self.bcf_coefficients(self.bcf_terms)

        system = params.SysP(
            H_sys=self.system,
            L=self.coupling_operators,
            g=g,
            w=w,
            bcf_scale=[self.bcf_scale],
            T=[self.T],
            description=self.description,
            psi0=self.ψ_0.full().flatten(),
        )

        if self.truncation_scheme == "bath_memory":
            trunc_scheme = BathMemory.from_system(
                system,
                nonlinear=True,
                influence_tolerance=float(self.influence_tolerance),
            )

        elif self.truncation_scheme == "simplex":
            trunc_scheme = TruncationScheme_Simplex(self.k_max)

        else:
            trunc_scheme = TruncationScheme_Power_multi.from_g_w(
                g=system.g,
                w=system.w,
                p=[1, 1],
                q=[0.5, 0.5],
                kfac=[float(self.k_fac)],
            )

        hierarchy = params.HiP(
            seed=0,
            nonlinear=True,
            terminator=False,
            result_type=params.ResultType.ZEROTH_AND_FIRST_ORDER,
            accum_only=False,
            rand_skip=None,
            truncation_scheme=trunc_scheme,
            save_therm_rng_seed=True,
            auto_normalize=True,
        )

        default_solver_args = dict(rtol=1e-8, atol=1e-8)
        default_solver_args.update(self.solver_args)

        integration = params.IntP(t=self.t, **default_solver_args)

        return params.HIParams(
            SysP=system,
            IntP=integration,
            HiP=hierarchy,
            Eta=[self.driving_process],
            EtaTherm=[self.thermal_process],
        )


@beartype
@dataclass(eq=False)
class QubitModelMutliBath(Model):
    """
    A class to dynamically calculate all the one qubit model
    parameters and generate the HOPS configuration.  Like
    :any:`QubitModel` but supports multiple baths.

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

    L: list[DynamicMatrix] = field(default_factory=lambda: [ConstantMatrix(1 / 2 * qt.sigmax().full())] * 2)  # type: ignore
    """
    The :math:`L` coupling operators with shape ``(2, 2)``.
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
    """The initial state."""

    t: NDArray[np.float64] = np.linspace(0, 10, 1000)
    """The simulation time points."""

    k_fac: list[SupportsFloat] = field(default_factory=lambda: [1.7] * 2)
    """The k_fac parameters for the truncation scheme.

    See
    :any:`hops.util.truncation_schemes.TruncationScheme_Power_multi`.
    """

    k_max: int = 5
    """The kmax parameter for the truncation scheme.

    See
    :any:`hops.util.abstract_truncation_scheme.TruncationScheme_Simplex`
    """

    influence_tolerance: SupportsFloat = 1e-2
    """The ``influecne_tolerance`` parameter for the truncation
    scheme.

    See :any:`hops.util.truncation_schemes.BathMemory`.
    """

    truncation_scheme: str = "simplex"
    """The truncation scheme to use."""

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

    H: DynamicMatrix = field(
        default_factory=lambda: ConstantMatrix(1 / 2 * qt.sigmaz().full())  # type: ignore
    )  # type: ignore
    """
    The system hamiltonian :math:`H` with shape ``(2, 2)``.
    """

    streaming_mode: bool = False
    """
    Whether to stream the trajectory to a fifo.  When turned on, the
    trajectories won't be saved to the data file.
    """

    @property
    def coupling_operators(self) -> list[DynamicMatrix]:
        """The bath coupling operators :math:`L`."""

        return self.L

    @property
    def system(self) -> DynamicMatrix:
        """The system hamiltonian."""

        return self.H

    @property
    def bcf_scales(self) -> list[float]:
        """The scaling factors for the bath correlation functions."""

        return [
            bcf_scale(δ, L, self.t.max(), s, ω)
            for δ, L, s, ω in zip(self.δ, self.L, self.s, self.ω_c)
        ]

    def bcf(
        self, i: int
    ) -> Union[hops.util.bcf.OhmicBCF_zeroTemp, hops.util.bcf.ShiftedBCF]:
        """
        The zero temperature BCF of bath ``i``.
        """

        bcf = hops.util.bcf.OhmicBCF_zeroTemp(
            s=float(self.s[i]), eta=1, w_c=float(self.ω_c[i]), normed=False
        )

        if float(self.ω_s[i]) > 0:
            return hops.util.bcf.ShiftedBCF(bcf, float(self.ω_s[i]))

        return bcf

    def spectral_density(
        self, i: int
    ) -> Union[hops.util.bcf.OhmicSD_zeroTemp, hops.util.bcf.ShiftedSD]:
        """
        The zero temperature spectral density of bath ``i``.
        """

        sd = hops.util.bcf.OhmicSD_zeroTemp(
            s=float(self.s[i]),
            w_c=float(self.ω_c[i]),
            eta=1,
            normed=False,
        )

        if float(self.ω_s[i]) > 0:
            return hops.util.bcf.ShiftedSD(sd, float(self.ω_s[i]))

        return sd

    def thermal_correlations(
        self, i: int
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialCorrelations]:
        """
        Thethermal noise corellation function of bath ``i``.
        """

        if self.T[i] == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialCorrelations(
            s=float(self.s[i]),
            eta=1,
            w_c=float(self.ω_c[i]),
            normed=False,
            beta=1 / float(self.T[i]),
            shift=float(self.ω_s[i]),
        )

    def thermal_spectral_density(
        self, i: int
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialDensity]:
        """
        The normalized thermal noise spectral density of bath ``i``.
        """

        if self.T[i] == 0:
            return None

        return hops.util.bcf.Ohmic_StochasticPotentialDensity(
            s=float(self.s[i]),
            eta=1,
            w_c=float(self.ω_c[i]),
            normed=False,
            beta=1.0 / float(self.T[i]),
            shift=float(self.ω_s[i]),
        )

    def bcf_coefficients(
        self, n: Optional[list[int]] = None
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        """
        The normalizedzero temperature BCF fit coefficients
        :math:`G^{(i)}_j,W^{(i)}_j` with ``n`` terms of bath ``i``.
        """

        n = n or self.bcf_terms
        g, w = [], []
        for i in range(self.num_baths):
            g_i, w_i = self.bcf(i).exponential_coefficients(n[i])
            g.append(g_i)
            w.append(w_i)

        return (g, w)

    @staticmethod
    def basis(n: int = 1) -> qt.Qobj:
        """
        A state with of the qubit in the state state ``n`` where ``1``
        means down and ``0`` means up.
        """

        return qt.basis([2], [n])

    def driving_process(self, i: int) -> sp.StocProc:
        """The driving stochastic process of the ``i``th bath."""

        return sp.StocProc_FFT(
            spectral_density=self.spectral_density(i),
            alpha=self.bcf(i),
            t_max=self.t.max(),
            intgr_tol=self.driving_process_tolerances[i].integration,
            intpl_tol=self.driving_process_tolerances[i].interpolation,
            negative_frequencies=False,
        )

    def thermal_process(self, i: int) -> Optional[sp.StocProc]:
        """The thermal noise stochastic process of bath ``i``."""

        if self.T[i] == 0:
            return None

        return (
            sp.StocProc_TanhSinh
            if self.therm_methods[i] == "tanhsinh"
            else sp.StocProc_FFT
        )(
            spectral_density=self.thermal_spectral_density(i),
            alpha=self.thermal_correlations(i),
            t_max=self.t.max(),
            intgr_tol=self.thermal_process_tolerances[i].integration,
            intpl_tol=self.thermal_process_tolerances[i].interpolation,
            negative_frequencies=False,
        )

    @property
    def thermal_processes(self) -> list[Optional[hopsflow.hopsflow.StocProc]]:
        """
        The thermal noise stochastic processes for each bath.
        :any:`None` means zero temperature.
        """

        return [self.thermal_process(i) for i in range(self.num_baths)]

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    def thermal_bcf(self, i: int):
        """
        :returns: The thermal bath correlation function for the ``i``th bath.
        """
        if self.T[i] == 0:
            return self.bcf(i)

        def thermal_bcf(t):
            return (
                self.bcf(i)(t) + 2 * (self.thermal_correlations(i)(t).real)
            ) * self.bcf_scales[i]

        return thermal_bcf

    def full_thermal_spectral_density(self, i: int):
        """
        :returns: The full thermal spectral density for the ``i``th bath.
        """

        if self.T[i] == 0:
            return self.spectral_density(i)

        def thermal_sd(ω):
            return self.spectral_density(i)(ω) * (
                1 / (np.expm1(ω / self.T[i])) + np.heaviside(ω, 0)
            )

        return thermal_sd

    def full_thermal_bcf(self, i: int):
        """
        :returns: The full thermal bath correlation function off bath
                  ``i``.
        """

        α_0 = self.bcf(i)
        if self.T[i] == 0:
            return α_0

        α_therm = self.thermal_correlations(i)

        def thermal_bcf(t):
            return α_0(t) + 2 * (α_therm(t).real)

        return thermal_bcf

    @property
    def hops_config(self) -> params.HIParams:
        """
        The hops :any:`hops.core.hierarchy_params.HIParams` parameter object
        for this system.
        """

        g, w = self.bcf_coefficients(self.bcf_terms)

        system = params.SysP(
            H_sys=self.system,
            L=self.coupling_operators,
            g=g,
            w=w,
            bcf_scale=self.bcf_scales,
            T=self.T,
            description=self.description,
            psi0=self.ψ_0.full().flatten(),
        )

        if self.truncation_scheme == "bath_memory":
            trunc_scheme = BathMemory.from_system(
                system,
                nonlinear=True,
                influence_tolerance=float(self.influence_tolerance),
            )

        elif self.truncation_scheme == "simplex":
            trunc_scheme = TruncationScheme_Simplex(self.k_max)

        else:
            trunc_scheme = TruncationScheme_Power_multi.from_g_w(
                g=system.g,
                w=system.w,
                p=[1, 1],
                q=[0.5, 0.5],
                kfac=[float(k) for k in self.k_fac],
            )

        hierarchy = params.HiP(
            seed=0,
            nonlinear=True,
            terminator=False,
            result_type=params.ResultType.ZEROTH_ORDER_ONLY
            if self.streaming_mode
            else params.ResultType.ZEROTH_AND_FIRST_ORDER,
            stream_result_type=params.ResultType.ZEROTH_AND_FIRST_ORDER,
            accum_only=self.streaming_mode,
            rand_skip=None,
            truncation_scheme=trunc_scheme,
            save_therm_rng_seed=True,
            auto_normalize=True,
        )

        default_solver_args = dict(rtol=1e-8, atol=1e-8)
        default_solver_args.update(self.solver_args)

        integration = params.IntP(t=self.t, **default_solver_args)

        return params.HIParams(
            SysP=system,
            IntP=integration,
            HiP=hierarchy,
            Eta=[self.driving_process(i) for i in range(self.num_baths)],
            EtaTherm=self.thermal_processes,
        )
