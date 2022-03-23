"""A base class for model HOPS configs."""

from typing import Any, Optional
from .utility import JSONEncoder, object_hook
import numpy as np
from numpy.typing import NDArray
import json
import copy
import hashlib
from abc import ABC, abstractmethod
import qutip as qt
from hops.core.hierarchy_data import HIData
import hopsflow
from hopsflow.util import EnsembleReturn, ensemble_return_add, ensemble_return_scale
import hashlib
import hops.core.hierarchy_parameters as params


class Model(ABC):
    """
    A base class with some data management functionality.
    """

    ψ_0: qt.Qobj
    """The initial state."""

    __base_version__: int = 1
    """The version of the model base."""

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

    def to_dict(self):
        """Returns a dictionary representation of the model configuration."""

        return {key: self.__dict__[key] for key in self.__dict__ if key[0] != "_"} | {
            "__version__": self.__version__,
            "__base_version__": self.__base_version__,
            "__model__": self.__class__.__name__,
        }

    def to_json(self):
        """Returns a json representation of the model configuration."""

        return JSONEncoder.dumps(self.to_dict())

    def __hash__(self):
        return JSONEncoder.hash(self.to_dict()).__hash__()

    @property
    def hexhash(self):
        """A hexadecimal representation of the model hash."""
        return JSONEncoder.hexhash(self.to_dict())

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]):
        """
        Tries to instantiate a model config from the dictionary ``dictionary``.
        """

        assert (
            model_dict["__model__"] == cls.__name__
        ), f"You are trying to instantiate the wrong model '{model_dict['__model__']}'."

        assert (
            model_dict["__version__"] == cls.__version__
            and model_dict["__base_version__"] == cls.__base_version__
        ), "Incompatible version detected."

        del model_dict["__version__"]
        del model_dict["__base_version__"]
        del model_dict["__model__"]

        return cls(**model_dict)

    @classmethod
    def from_json(cls, json_str: str):
        """
        Tries to instantiate a model config from the json string
        ``json_str``.
        """

        model_dict = JSONEncoder.loads(json_str)

        return cls.from_dict(model_dict)

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
    @abstractmethod
    def hops_config(self) -> params.HIParams:
        """
        The hops :any:`hops.core.hierarchy_params.HIParams` parameter object
        for this system.
        """

    @property
    def hopsflow_system(self) -> hopsflow.hopsflow.SystemParams:
        """The :any:`hopsflow` system config for the system."""

        g, w = self.bcf_coefficients()

        return hopsflow.hopsflow.SystemParams(
            L=self.coupling_operators,
            G=[g_i * scale for g_i, scale in zip(g, self.bcf_scales)],
            W=w,
            fock_hops=True,
            nonlinear=True,
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

        operator_hash = JSONEncoder.hexhash(operator)

        return hopsflow.util.operator_expectation_ensemble(
            iter(data.stoc_traj),  # type: ignore
            operator.full(),
            kwargs.get("N", data.samples),
            normalize=True,  # always nonlinear
            save=f"{operator_hash}_{self.hexhash}",
            **kwargs,
        )

    def system_energy(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates the system energy from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        operator = self.system
        return self.system_expectation(data, operator, real=True, **kwargs)

    def bath_energy_flow(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates the bath energy flow from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.heat_flow_ensemble`.

        :returns: See :any:`hopsflow.util.heat_flow_ensemble`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.heat_flow_ensemble(
            data.stoc_traj,  # type: ignore
            data.aux_states,  # type: ignore
            self.hopsflow_system,
            N,
            (iter(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            save=f"flow_{self.hexhash}",
            **kwargs,
        )

    def interaction_energy(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates interaction energy from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.interaction_energy_ensemble`.

        :returns: See :any:`hopsflow.util.interaction_energy_ensemble`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.interaction_energy_ensemble(
            data.stoc_traj,  # type: ignore
            data.aux_states,  # type: ignore
            self.hopsflow_system,
            N,
            (iter(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            save=f"interaction_{self.hexhash}",
            **kwargs,
        )

    def bath_energy(self, data: HIData, **kwargs) -> EnsembleReturn:
        """Calculates bath energy by integrating the bath energy flow
        calculated from the ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.bath_energy_from_flow`.

        :returns: See :any:`hopsflow.bath_energy_from_flow`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.bath_energy_from_flow(
            np.array(data.time),
            data.stoc_traj,  # type: ignore
            data.aux_states,  # type: ignore
            self.hopsflow_system,
            N,
            (iter(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            save=f"flow_{self.hexhash}",  # under the hood the flow is used
            **kwargs,
        )

    def interaction_energy_from_conservation(
        self, data: HIData, **kwargs
    ) -> EnsembleReturn:
        """Calculates the interaction energy from energy conservations
        calculated from the ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.bath_energy_from_flow`.

        :returns: See :any:`hopsflow.bath_energy_from_flow`.
        """

        system = ensemble_return_scale(-1, self.system_energy(data, **kwargs))
        bath = ensemble_return_scale(-1, self.bath_energy(data, **kwargs))
        total_raw = qt.expect(self.system, self.ψ_0)

        if isinstance(bath, list):
            total = [
                (N, total_raw * np.ones_like(bath_val), np.zeros_like(bath_val))
                for N, bath_val, _ in bath
            ]
        else:
            total = (
                bath[0],
                total_raw * np.ones_like(bath[1]),
                np.zeros_like(bath[1]),
            )

        return ensemble_return_add(ensemble_return_add(total, bath), system)


def _get_N_kwargs(kwargs: dict, data: HIData) -> tuple[int, dict]:
    N = kwargs.get("N", data.samples)
    if "N" in kwargs:
        del kwargs["N"]

    return N, kwargs
