from two_qubit_model import TwoQubitModel
from typing import Optional
import tempfile
from hops.core.integration import HOPSSupervisor
from hops.core.hierarchy_parameters import HIParams
from dataclasses import dataclass
import numpy as np


@dataclass
class TruncationToleranceData:
    tol: float
    """The tolerance parameter for ``BathMemory`` truncation scheme."""

    number_of_indices: int
    """The number of indices given by the ``BathMemory`` truncation scheme"""

    number_of_indices_ref: int
    """The reference number of indices given by the ``Simplex`` truncation scheme"""

    relative_error: float
    """The maximum norm deviation of the ``BathMemory`` trajectory
      from the reference ``Simplex`` trajectory divided by the maximum
      of the norm of the reverence trajectory."""

    k_max_ref: int
    """The depth of the reference ``Simplex`` truncation scheme."""


def get_one_trajctory(config: HIParams) -> np.ndarray:
    """The"""


def find_tol(
    model: TwoQubitModel,
    k_max_ref: int = 6,
    start_tol: int = 4,
    relative_error: float = 0.1,
) -> Optional[float]:
    """Find the tolerance for the ``BathMemory`` truncation scheme.

    :param model: The model configuration.
    :param k_max_ref: The depth of the reference ``Simplex``
        truncation scheme.
    :param start_tol: The start tolerance of the ``BathMemory``
        truncation scheme.
    :param relative_error: The maximum allowed norm deviation of the
        ``BathMemory`` trajectory from the reference ``Simplex``
        trajectory divided by the maximum of the norm of the
        reverence trajectory.

    :returns: The required tolerance parameter for ``BathMemory``
              truncation scheme or ``None`` if none could be found.
    """

    simplex_model = model.copy()
    simplex_model.truncation_scheme = "simplex"
    simplex_model.k_max = k_max_ref

    simplex_config = simplex_model.hops_config

    with tempfile.TemporaryDirectory() as data_dir:
        simplex_sup = HOPSSupervisor(
            simplex_config,
            number_of_samples=1,
            data_path=data_dir,
            data_name="data_conv",
        )
