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
from typing import Any, Optional, SupportsFloat
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
from .utility import StocProcTolerances
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

    ω_c: SupportsFloat = 2
    """The cutoff frequency :math:`ω_c`."""

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

    @property
    def coupling_operators(self) -> list[DynamicMatrix]:
        """The bath coupling operators :math:`L`."""

        return [self.L]

    @property
    def system(self) -> qt.Qobj:
        """The system hamiltonian."""

        return 1 / 2 * qt.sigmaz()  # type: ignore

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
    def L_expect(self) -> float:
        r"""
        The expecation value :math:`\langle L^†L + LL^†\rangle` in
        the inital state.
        """

        return (self.L @ self.L.dag + self.L.dag @ self.L).max_operator_norm(
            self.t.max()
        )

    @property
    def bcf_scale(self) -> float:
        """
        The BCF scaling factor of the BCF.
        """

        return float(self.δ) / self.L_expect * self.bcf_norm

    @property
    def bcf_scales(self) -> list[float]:
        """The scaling factors for the bath correlation functions."""

        return [self.bcf_scale]

    @property
    def bcf(self) -> hops.util.bcf.OhmicBCF_zeroTemp:
        """
        The normalized zero temperature BCF.
        """

        return hops.util.bcf.OhmicBCF_zeroTemp(
            s=self.s, eta=1, w_c=self.ω_c, normed=False
        )

    @property
    def spectral_density(self) -> hops.util.bcf.OhmicSD_zeroTemp:
        """
        The normalized zero temperature spectral density.
        """

        return hops.util.bcf.OhmicSD_zeroTemp(
            s=float(self.s),
            w_c=float(self.ω_c),
            eta=1,
            normed=False,
        )

    @property
    def thermal_correlations(
        self,
    ) -> Optional[hops.util.bcf.Ohmic_StochasticPotentialCorrelations]:
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

        return sp.StocProc_TanhSinh(
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
            H_sys=self.system.full(),
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
