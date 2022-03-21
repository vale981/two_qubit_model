"""A base class for model HOPS configs."""

from typing import Optional
from utility import JSONEncoder, object_hook
import numpy as np
from numpy.typing import NDArray
import json
import copy
import hashlib
from abc import ABC, abstractmethod
import qutip as qt
from hops.core.hierarchy_data import HIData
import hopsflow
from hopsflow.util import EnsembleReturn
import hashlib


class MyABC(ABC):
    pass


class Model(ABC):
    """
    A base class with some data management functionality.
    """

    __version__: int = 1
    """
    The version of the model implementation.  It is increased for
    breaking changes.
    """

    def __init__(self, *_, **__):
        del _, __
        pass

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    def to_json(self):
        """Returns a json representation of the model configuration."""

        return JSONEncoder.dumps(
            {key: self.__dict__[key] for key in self.__dict__ if key[0] != "_"}
            | {"__version__": self.__version__}
        )

    def __hash__(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).digest().__hash__()

    @classmethod
    def from_json(cls, json_str: str):
        """
        Tries to instantiate a model config from the json string
        ``json_str``.
        """

        model_dict = json.loads(json_str, object_hook=object_hook)
        assert (
            model_dict["__version__"] == cls().__version__
        ), "Incompatible version detected."

        del model_dict["__version__"]

        return cls(**model_dict)

    def __eq__(self, other):
        this_keys = list(self.__dict__.keys())
        ignored_keys = ["_sigmas"]

        for key in this_keys:
            if key not in ignored_keys:
                this_val, other_val = self.__dict__[key], other.__dict__[key]

                same = this_val == other_val

                if isinstance(this_val, np.ndarray):
                    same = same.all()

                if not same:
                    return False

        return self.__hash__() == other.__hash__()

    def copy(self):
        """Return a deep copy of the model."""

        return copy.deepcopy(self)

    @property
    @abstractmethod
    def system(self) -> qt.Qobj:
        """The system hamiltonian."""

        pass

    @property
    @abstractmethod
    def coupling_operators(self) -> list[np.ndarray]:
        """The bath coupling operators :math:`L`."""
        pass

    @abstractmethod
    def bcf_coefficients(
        self, n: Optional[int] = None
    ) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
        """
        The normalized zero temperature BCF fit coefficients
        :math:`[G_i], [W_i]` with ``n`` terms.
        """

        pass

    @property
    @abstractmethod
    def thermal_processes(self) -> list[Optional[hopsflow.hopsflow.StocProc]]:
        """
        The thermal noise stochastic processes for each bath.
        :any:`None` means zero temperature.
        """

        pass

    @property
    @abstractmethod
    def bcf_scales(self) -> list[float]:
        """The scaling factors for the bath correlation functions."""

        pass

    @property
    def hopsflow_system(self) -> hopsflow.hopsflow.SystemParams:
        """The :any:`hopsflow` system config for the system."""

        g, w = self.bcf_coefficients()

        return hopsflow.hopsflow.SystemParams(
            L=self.coupling_operators,
            G=[g_i * scale for g_i, scale in zip(g, self.bcf_scales)],
            W=w,
            fock_hops=True,
        )

    def hopsflow_therm(
        self, τ: NDArray[np.float64]
    ) -> Optional[hopsflow.hopsflow.ThermalParams]:
        """The :any:`hopsflow` thermal config for the system."""

        processes = self.thermal_processes
        scales = self.bcf_scales

        for process, scale in zip(processes, scales):
            if process:
                process.set_scale(scale)
                process.calc_deriv = True

        return hopsflow.hopsflow.ThermalParams(processes, τ)

    ###########################################################################
    #                            Derived Quantities                           #
    ###########################################################################

    def system_expectation(
        self, data: HIData, operator: qt.Qobj, **kwargs
    ) -> EnsembleReturn:
        """Calculates the expectation value of ``operator`` from the
        hierarchy data ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        operator_hash = JSONEncoder.hexhash(operator).encode("utf-8")

        return hopsflow.util.operator_expectation_ensemble(
            data.stoc_traj,  # type: ignore
            operator.full(),
            kwargs.get("N", data.samples),
            nonlinear=True,  # always nonlinear
            save=f"{operator_hash}_{self.__hash__()}",
        )

    def system_energy(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates the system energy from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        operator = self.system.full()
        return self.system_expectation(data, operator, **kwargs)

    def bath_energy_flow(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates the bath energy flow from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        return hopsflow.hopsflow.heat_flow_ensemble(
            data.stoc_traj,  # type: ignore
            data.aux_states,  # type: ignore
            self.hopsflow_system,
            kwargs.get("N", data.samples),
            (data.rng_seed, self.hopsflow_therm),  # type: ignore
            save=f"flow_{self.__hash__()}",
            **kwargs,
        )
