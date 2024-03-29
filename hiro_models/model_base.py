"""A base class for model HOPS configs."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union, ClassVar
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
from hopsflow.util import EnsembleValue, WelfordAggregator
import hashlib
import hops.core.hierarchy_parameters as params
from collections.abc import Callable
from datetime import datetime
import pickle
import os
from pathlib import Path


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

    __version__: Union[int, list[int]] = 1
    """
    The version of the model implementation.  It is increased for
    breaking changes.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now())
    """
    A timestamp that signals when the simulation was run. Is not used to hash or compare objects.
    """

    _ignored_keys: list[str] = field(
        default_factory=lambda: ["_sigmas", "description", "timestamp"]
    )
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

                if isinstance(this_val, Iterable):
                    for val_1, val_2 in zip(this_val, other_val):
                        if not _compare_values(val_1, val_2):
                            return False

                    continue

                same = _compare_values(this_val, other_val)

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
    def coupling_operators(self) -> list[DynamicMatrix]:
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

    def try_get_online_data(self, path: str, results_path: str) -> EnsembleValue:
        """
        Try to load the cached data from the online analysis with
        filename ``path`` in the directory ``results_path``.

        Raises a :any:`RuntimeError` if nothing is found.
        """

        file_path = os.path.join(path, results_path)
        if not os.path.exists(file_path):
            raise RuntimeError(f"No data found under '{file_path}'.")

        return hopsflow.util.get_online_values_from_cache(file_path)

    def system_energy(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> EnsembleValue:
        """Calculates the system energy from the hierarchy data
        ``data`` or, if not supplied, tries to load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.
        """

        if data is None:
            return self.try_get_online_data(results_path, self.online_system_name)

        operator = self.system
        return self.system_expectation(data, operator, real=True, **kwargs)

    def system_power(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> Optional[EnsembleValue]:
        """Calculates the power based on the time dependency of the
        system hamiltonian from ``data`` or, if not supplied, tries to
        load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean`.

        :returns: See :any:`hopsflow.util.ensemble_mean`.  Returns
                  :any:`None` if the system is static.
        """

        if data is None:
            return self.try_get_online_data(results_path, self.online_system_power_name)

        operator = self.system.derivative()

        if (abs(operator(self.t)).sum() == 0).all():
            return None

        return self.system_expectation(data, operator, real=True, **kwargs)

    def energy_change_from_system_power(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> Optional[EnsembleValue]:
        """Calculates the integrated system power from the hierarchy
        data ``data`` or, if not supplied, tries to load the online
        results from ``results_path``.

        The ``kwargs`` are passed on to :any:`system_power`.

        :returns: See :any:`system_power`.  Returns :any:`None` if the
                  system is static.
        """

        power = self.system_power(data, **kwargs)
        if power is not None:
            return power.integrate(self.t)

        return None

    def bath_energy_flow(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> EnsembleValue:
        """Calculates the bath energy flow from the hierarchy data
        ``data`` or, if not supplied, tries to load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.heat_flow_ensemble`.

        :returns: See :any:`hopsflow.util.heat_flow_ensemble`.
        """

        if data is None:
            return self.try_get_online_data(results_path, self.online_flow_name)

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

    @property
    def online_flow_name(self):
        """The filename where the online flow is saved."""

        return f"flow_{self.hexhash}.npz"

    @property
    def online_interaction_name(self):
        """The filename where the online interaction is saved."""

        return f"interaction_{self.hexhash}.npz"

    @property
    def online_interaction_power_name(self):
        """The filename where the online interaction power is saved."""

        return f"interaction_power_{self.hexhash}.npz"

    @property
    def online_system_name(self):
        """The filename where the online system is saved."""

        return f"system_{self.hexhash}.npz"

    @property
    def online_system_power_name(self):
        """The filename where the online system power is saved."""

        return f"system_power_{self.hexhash}.npz"

    def all_energies_online(
        self,
        stream_pipe: str = "results.fifo",
        results_directory: str = "results",
        **kwargs,
    ) -> tuple[
        Optional[EnsembleValue],
        Optional[EnsembleValue],
        Optional[EnsembleValue],
        Optional[EnsembleValue],
        Optional[EnsembleValue],
    ]:
        """Calculates the bath energy flow, the interaction energy,
        the interaction power, the system energy and the system power
        from the trajectories dumped into ``stream_pipe``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.ensemble_mean_online`.

        :returns: At tuple of :any:`hopsflow.util.EnsembleValue`.
        """

        flow_worker = hopsflow.hopsflow.make_heat_flow_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t)
        )

        interaction_worker = hopsflow.hopsflow.make_interaction_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t), power=False
        )

        interaction_power_worker = hopsflow.hopsflow.make_interaction_worker(
            self.hopsflow_system, self.hopsflow_therm(self.t), power=True
        )

        system_worker = hopsflow.util.make_operator_expectation_task(
            self.system, self.t, normalize=True, real=True
        )

        system_power_worker = hopsflow.util.make_operator_expectation_task(
            self.system.derivative(), self.t, normalize=True, real=True
        )

        Path(results_directory).mkdir(parents=True, exist_ok=True)

        aggregates = [None for _ in range(5)]
        paths = [
            os.path.join(results_directory, path)
            for path in [
                self.online_flow_name,
                self.online_interaction_name,
                self.online_interaction_power_name,
                self.online_system_name,
                self.online_system_power_name,
            ]
        ]

        flow, interaction, interaction_power, system, system_power = aggregates

        with open(stream_pipe, "rb") as fifo:
            while True:
                try:
                    (
                        idx,
                        psi0,
                        aux_states,
                        _,
                        _,
                        rng_seed,
                    ) = pickle.load(fifo)

                    for path, (i, aggregator), args in zip(
                        paths,
                        enumerate(aggregates),
                        [
                            ((psi0, aux_states, rng_seed), flow_worker),
                            ((psi0, aux_states, rng_seed), interaction_worker),
                            ((psi0, aux_states, rng_seed), interaction_power_worker),
                            ((psi0), system_worker),
                            ((psi0), system_power_worker),
                        ],
                    ):

                        aggregates[i] = hopsflow.util.ensemble_mean_online(
                            *args, save=path, aggregator=aggregator, i=idx, **kwargs
                        )

                except EOFError:
                    break

        for path, aggregate in zip(paths, aggregates):
            if aggregate is not None:
                aggregate.dump(path)

        return tuple(
            [
                (aggregate.ensemble_value if aggregate else None)
                for aggregate in aggregates
            ]
        )

    def interaction_energy(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> EnsembleValue:
        """Calculates interaction energy from the hierarchy data
        ``data`` or, if not supplied, tries to load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.interaction_energy_ensemble`.

        :returns: See :any:`hopsflow.util.interaction_energy_ensemble`.
        """

        if data is None:
            return self.try_get_online_data(results_path, self.online_interaction_name)

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

    def interaction_power(
        self, data: Optional[HIData] = None, results_path: str = "results", **kwargs
    ) -> EnsembleValue:
        """Calculates interaction power from the hierarchy data
        ``data`` or, if not supplied, tries to load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.util.interaction_energy_ensemble`.

        :returns: See :any:`hopsflow.util.interaction_energy_ensemble`.
        """

        if data is None:
            return self.try_get_online_data(
                results_path, self.online_interaction_power_name
            )

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
        self, data: Optional[HIData] = None, **kwargs
    ) -> EnsembleValue:
        """Calculates the integrated interaction power from the hierarchy data
        ``data`` or, if not supplied, tries to load the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`interaction_power`.
        """

        return self.interaction_power(data, **kwargs).integrate(self.t)

    def bath_energy(self, data: Optional[HIData] = None, **kwargs) -> EnsembleValue:
        """Calculates bath energy by integrating the bath energy flow
        calculated from the ``data`` or, if not supplied, tries to load
        the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`bath_energy_flow`.
        """

        return -1 * self.bath_energy_flow(data, **kwargs).integrate(self.t)

    def interaction_energy_from_conservation(
        self, data: Optional[HIData] = None, **kwargs
    ) -> EnsembleValue:
        """Calculates the interaction energy from energy conservations
        calculated from the ``data`` or, if not supplied, tries to load
        the online results from ``results_path``.

        The ``kwargs`` are passed on to
        :any:`hopsflow.bath_energy_from_flow`.

        :returns: See :any:`hopsflow.bath_energy_from_flow`.
        """

        system = self.system_energy(data, **kwargs)
        bath = self.bath_energy(data, **kwargs)
        total = float(qt.expect(qt.Qobj(self.system(0)), self.ψ_0))

        return total - (system + bath)

    def total_energy(self, data: Optional[HIData] = None, **kwargs) -> EnsembleValue:
        """Calculates the total energy from the trajectories using
        energy bilance in ``data`` or, if not supplied, tries to load
        the online results from ``results_path``

        The ``kwargs`` are passed on to :any:`bath_energy`,
        :any:`system_energy` and :any:`interaction_energy`.

        :returns: The total energy.
        """

        system = self.system_energy(data, **kwargs)
        bath = self.bath_energy(data, **kwargs)
        interaction = self.interaction_energy(data, **kwargs)

        total = system + bath.sum_baths() + interaction.sum_baths()

        return total

    def total_power(self, data: Optional[HIData] = None, **kwargs) -> EnsembleValue:
        """Calculates the total power from the trajectories in
        ``data`` or, if not supplied, tries to load
        the online results from ``results_path``.

        The ``kwargs`` are passed on to :any:`system_power` and
        :any:`interaction_power`.

        :returns: The total power.
        """

        power = self.interaction_power(data, **kwargs).sum_baths()
        system_power = self.system_power(data, **kwargs)

        if system_power is not None:
            power = power + system_power

        return power

    def total_energy_from_power(
        self, data: Optional[HIData] = None, **kwargs
    ) -> EnsembleValue:
        """Calculates the total energy from the trajectories in
        ``data``or, if not supplied, tries to load
        the online results from ``results_path`` using the integrated power.

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


def _compare_values(this_val, other_val):
    same = this_val == other_val

    if isinstance(this_val, np.ndarray):
        same = same.all()

    return same
