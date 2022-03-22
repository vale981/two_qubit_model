"""Functionality to integrate :any:`Model` instances and analyze the results."""

import json
from typing import Any
from model_base import Model
from hops.core.integration import HOPSSupervisor
from contextlib import contextmanager
from utility import JSONEncoder, object_hook
from filelock import FileLock
from pathlib import Path
from one_qubit_model import QubitModel
from two_qubit_model import TwoQubitModel


@contextmanager
def model_db(data_path: str = "."):
    """
    Opens the model database json file in the folder ``data_path`` as
    a dictionary.

    Mutations will be synchronized to the file.  Access is managed via
    a lock file.
    """
    db_path = Path(data_path) / "model_data.json"
    db_lock = Path(data_path) / "model_data.json.lock"

    lock = FileLock(db_lock)
    with lock:
        with db_path.open("rw") as f:
            db = json.loads(f.read(), object_hook=model_hook)

            yield db

            f.write(JSONEncoder.dumps(db))


def model_hook(dct: dict[str, Any]):
    """A custom decoder for the model types."""

    if "__model__" in dct:
        model = dct["__model__"]

        if model == "QubitModel":
            return QubitModel.from_dict(dct)

        if model == "TwoQubitModel":
            return TwoQubitModel.from_dict(dct)

    return object_hook(dct)


def integrate(model: Model, n: int, data_path: str = "."):
    """Integrate the hops equations for the model.

    A call to :any:`ray.init` may be required.

    :param n: The number of samples to be integrated.
    """

    hash = model.__hash__()
    supervisor = HOPSSupervisor(
        model.hops_config, n, data_path=data_path, data_name=str(hash)
    )

    supervisor.integrate()

    with supervisor.get_data(True) as data:
        with model_db(data_path) as db:
            db[str(hash)] = {
                "model_config": model.to_dict(),
                "data_path": str(Path(data.hdf5_name).relative_to(data_path)),
            }
