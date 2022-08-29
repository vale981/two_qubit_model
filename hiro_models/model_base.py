"""A base class for model HOPS configs."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union, ClassVar
from hops.util.dynamic_matrix import DynamicMatrix
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
from hopsflow.util import EnsembleValue
import hashlib
import hops.core.hierarchy_parameters as params
from collections.abc import Callable
import pickle


@dataclass
class Model(ABC):
    """
    A base class with some data management functionality.
    """

    ψ_0: qt.Qobj
    """The initial state."""

    description: str = ""
    """A free-form description of the model instance."""

    t: NDArray[np.float64] = np.linspace(0, 10, 1000)
    """The simulation time points."""

    __base_version__: int = 1
    """The version of the model base."""

    __version__: int = 1
    """
    The version of the model implementation.  It is increased for
    breaking changes.
    """

    _ignored_keys: list[str] = field(default_factory=lambda: ["_sigmas", "description"])
    """Keys that are ignored when comparing or hashing models."""

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    def to_dict(self, extra_fields: "dict[str, Callable[[Model], str]]" = {}):
        """Returns a dictionary representation of the model
        configuration.

        :param extra_fields: A dictionary whose keys will be added to
            the final dict and whose values are callables that take
            the model instance as an argument and return the value
            that will be assigned the addyd key.
        """

        return (
            {key: self.__dict__[key] for key in self.__dict__ if key[0] != "_"}
            | {
                "__version__": self.__version__,
                "__base_version__": self.__base_version__,
                "__model__": self.__class__.__name__,
            }
            | {key: value(self) for key, value in extra_fields.items()}
        )

    def to_hashable_dict(self):
        """
        Returns a dictionary representation of the model configuration
        without unecessary keys.
        """

        return {
            key: self.__dict__[key]
            for key in self.__dict__
            if key[0] != "_" and key not in self._ignored_keys
        } | {
            "__version__": self.__version__,
            "__base_version__": self.__base_version__,
            "__model__": self.__class__.__name__,
        }

    def to_json(self):
        """Returns a json representation of the model configuration."""

        return JSONEncoder.dumps(self.to_dict())

    def __hash__(self):
        return JSONEncoder.hash(self.to_hashable_dict()).__hash__()

    @property
    def hexhash(self):
        """A hexadecimal representation of the model hash."""

        return JSONEncoder.hexhash(self.to_hashable_dict())

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

        for key in this_keys:
            if key not in self._ignored_keys:
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
    def system(self) -> DynamicMatrix:
        """The system hamiltonian."""

        pass

    @property
    @abstractmethod
    def coupling_operators(self) -> list[Union[np.ndarray, DynamicMatrix]]:
        """The bath coupling operators :math:`L`."""

        pass

    @property
    def num_baths(self) -> int:
        """The number of baths attached to the system."""

        return len(self.coupling_operators)

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
            G=g,
            W=w,
            t=self.t,
            bcf_scale=self.bcf_scales,
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
        self, data: HIData, operator: DynamicMatrix, **kwargs
    ) -> EnsembleValue:
        """Calculates the expectation value of ``operator`` from the
        hierarchy data ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        operator_hash = JSONEncoder.hexhash(operator)
        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.util.operator_expectation_ensemble(
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            operator,
            self.t,
            normalize=True,  # always nonlinear
            save=f"{operator_hash}_{self.hexhash}",
            N=N,
            **kwargs,
        )

    def system_energy(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates the system energy from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        operator = self.system
        return self.system_expectation(data, operator, real=True, **kwargs)

    def system_power(self, data: HIData, **kwargs) -> Optional[EnsembleValue]:
        """Calculates the power based on the time dependency of the
        system hamiltonian from ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.  Returns
                  :any:`None` if the system is static.
        """

        operator = self.system.derivative()

        if (abs(operator(self.t)).sum() == 0).all():
            return None

        return self.system_expectation(data, operator, real=True, **kwargs)

    def energy_change_from_system_power(
        self, data: HIData, **kwargs
    ) -> Optional[EnsembleValue]:
        """Calculates the integrated system power from the hierarchy
        data ``data``.

        The ``kwargs`` are passed on to :any:`system_power`.

        :returns: See :any:`system_power`.  Returns :any:`None` if the
                  system is static.
        """

        power = self.system_power(data, **kwargs)
        if power is not None:
            return power.integrate(self.t)

        return None

    def bath_energy_flow(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates the bath energy flow from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.heat_flow_ensemble`.

        :returns: See :any:`hopsflow.util.heat_flow_ensemble`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.heat_flow_ensemble(
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            data.valid_sample_iterator(data.aux_states),  # type: ignore
            self.hopsflow_system,
            (data.valid_sample_iterator(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            save=f"flow_{self.hexhash}",
            N=N,
            **kwargs,
        )

    def all_energies_online_from_cache(
        self,
    ) -> tuple[
        EnsembleValue, EnsembleValue, EnsembleValue, EnsembleValue, EnsembleValue
    ]:

        names = ["flow", "interaction", "interaction_power", "system", "system_power"]

        hexhash = self.hexhash
        results = []

        for name in names:
            results.append(hopsflow.util.load_online_cache(f"{name}_{hexhash}"))

        return tuple(results)

    def all_energies_online(
        self,
        stream_pipe: str = "results.fifo",
        **kwargs,
    ) -> Optional[
        tuple[EnsembleValue, EnsembleValue, EnsembleValue, EnsembleValue, EnsembleValue]
    ]:
        """Calculates the bath energy flow, the interaction energy,
        the interaction power, the system energy and the system power
        from the trajectories dumped into ``stream_pipe``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean_online`.

        :returns: At tuple of :any:`hopsflow.util.EnsembleValue`.
        """

        flow_name = f"flow_{self.hexhash}"
        flow_worker = hopsflow.hopsflow.make_heat_flow_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t)
        )

        interaction_name = f"interaction_{self.hexhash}"
        interaction_worker = hopsflow.hopsflow.make_interaction_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t), power=False
        )

        interaction_power_name = f"interaction_power_{self.hexhash}"
        interaction_power_worker = hopsflow.hopsflow.make_interaction_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t), power=True
        )

        system_name = f"system_{self.hexhash}"
        system_worker = hopsflow.util.make_operator_expectation_task(
            self.system, self.t, normalize=True, real=True
        )

        system_power_name = f"system_power_{self.hexhash}"
        system_power_worker = hopsflow.util.make_operator_expectation_task(
            self.system.derivative(), self.t, normalize=True, real=True
        )

        flow, interaction, interaction_power, system, system_power = (
            None,
            None,
            None,
            None,
            None,
        )

        with open(stream_pipe, "rb") as fifo:
            while True:
                try:
                    (
                        _,
                        psi0,
                        aux_states,
                        _,
                        _,
                        rng_seed,
                    ) = pickle.load(fifo)
                    flow = hopsflow.util.ensemble_mean_online(
                        (psi0, aux_states, rng_seed),
                        flow_name,
                        flow_worker,
                        **kwargs,
                    )

                    interaction = hopsflow.util.ensemble_mean_online(
                        (psi0, aux_states, rng_seed),
                        interaction_name,
                        interaction_worker,
                        **kwargs,
                    )

                    interaction_power = hopsflow.util.ensemble_mean_online(
                        (psi0, aux_states, rng_seed),
                        interaction_power_name,
                        interaction_power_worker,
                        **kwargs,
                    )

                    system = hopsflow.util.ensemble_mean_online(
                        (psi0),
                        system_name,
                        system_worker,
                        **kwargs,
                    )

                    system_power = hopsflow.util.ensemble_mean_online(
                        (psi0),
                        system_power_name,
                        system_power_worker,
                        **kwargs,
                    )

                except EOFError:
                    break

        return flow, interaction, interaction_power, system, system_power

    def interaction_energy(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates interaction energy from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.interaction_energy_ensemble`.

        :returns: See :any:`hopsflow.util.interaction_energy_ensemble`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.interaction_energy_ensemble(
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            data.valid_sample_iterator(data.aux_states),  # type: ignore
            self.hopsflow_system,
            (data.valid_sample_iterator(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            N=N,
            save=f"interaction_{self.hexhash}",
            **kwargs,
        )

    def interaction_power(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates interaction power from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.interaction_energy_ensemble`.

        :returns: See :any:`hopsflow.util.interaction_energy_ensemble`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.interaction_energy_ensemble(
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            data.valid_sample_iterator(data.aux_states),  # type: ignore
            self.hopsflow_system,
            (data.valid_sample_iterator(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            N=N,
            save=f"interaction_power_{self.hexhash}",
            power=True,
            **kwargs,
        )

    def energy_change_from_interaction_power(
        self, data: HIData, **kwargs
    ) -> EnsembleValue:
        """Calculates the integrated interaction power from the hierarchy data
        ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.energy_change_from_interaction_power`.

        :returns: See :any:`hopsflow.util.energy_change_from_interaction_power`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.energy_change_from_interaction_power(
            np.array(data.time),
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            data.valid_sample_iterator(data.aux_states),  # type: ignore
            self.hopsflow_system,
            (data.valid_sample_iterator(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            N=N,
            save=f"interaction_power_{self.hexhash}",  # under the hood the power is used
            **kwargs,
        )

    def bath_energy(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates bath energy by integrating the bath energy flow
        calculated from the ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.bath_energy_from_flow`.

        :returns: See :any:`hopsflow.bath_energy_from_flow`.
        """

        N, kwargs = _get_N_kwargs(kwargs, data)

        return hopsflow.hopsflow.bath_energy_from_flow(
            np.array(data.time),
            data.valid_sample_iterator(data.stoc_traj),  # type: ignore
            data.valid_sample_iterator(data.aux_states),  # type: ignore
            self.hopsflow_system,
            (data.valid_sample_iterator(data.rng_seed), self.hopsflow_therm(data.time[:])),  # type: ignore
            save=f"flow_{self.hexhash}",  # under the hood the flow is used
            N=N,
            **kwargs,
        )

    def interaction_energy_from_conservation(
        self, data: HIData, **kwargs
    ) -> EnsembleValue:
        """Calculates the interaction energy from energy conservations
        calculated from the ``data``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.bath_energy_from_flow`.

        :returns: See :any:`hopsflow.bath_energy_from_flow`.
        """

        system = self.system_energy(data, **kwargs)
        bath = self.bath_energy(data, **kwargs)
        total = float(qt.expect(qt.Qobj(self.system(0)), self.ψ_0))

        return total - (system + bath)

    def total_energy(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates the total energy from the trajectories in
        ``data`` using energy bilance.

        The ``kwargs`` are passed on to :any:`bath_energy`,
        :any:`system_energy` and :any:`interaction_energy`.

        :returns: The total energy.
        """

        system = self.system_energy(data, **kwargs)
        bath = self.bath_energy(data, **kwargs)
        interaction = self.interaction_energy(data, **kwargs)

        total = system + bath.sum_baths() + interaction.sum_baths()

        return total

    def total_power(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates the total power from the trajectories in
        ``data``.

        The ``kwargs`` are passed on to :any:`system_power` and
        :any:`interaction_power`.

        :returns: The total power.
        """

        power = self.interaction_power(data, **kwargs).sum_baths()
        system_power = self.system_power(data, **kwargs)

        if system_power is not None:
            power = power + system_power

        return power

    def total_energy_from_power(self, data: HIData, **kwargs) -> EnsembleValue:
        """Calculates the total energy from the trajectories in
        ``data`` using the integrated power.

        The ``kwargs`` are passed on to :any:`total_power`.

        :returns: The total energy.
        """

        return self.total_power(data, **kwargs).integrate(self.t)

    std_extra_fields: ClassVar = {"BCF scaling": lambda model: model.bcf_scale}


def _get_N_kwargs(kwargs: dict, data: HIData) -> tuple[int, dict]:
    N = kwargs.get("N", data.samples)
    if "N" in kwargs:
        del kwargs["N"]

    return N, kwargs
