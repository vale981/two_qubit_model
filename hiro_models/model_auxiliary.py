"""Functionality to integrate :any:`Model` instances and analyze the results."""

import json
from typing import Any

from hops.core.hierarchy_data import HIData
from .model_base import Model
from hops.core.integration import HOPSSupervisor
from contextlib import contextmanager
from .utility import JSONEncoder, object_hook
from filelock import FileLock
from pathlib import Path
from .one_qubit_model import QubitModel
from .two_qubit_model import TwoQubitModel
from collections.abc import Sequence


@contextmanager
def model_db(data_path: str = "./.data"):
    """
    Opens the model database json file in the folder ``data_path`` as
    a dictionary.

    Mutations will be synchronized to the file.  Access is managed via
    a lock file.
    """

    path = Path(data_path)
    path.mkdir(exist_ok=True, parents=True)

    db_path = path / "model_data.json"
    db_lock = path / "model_data.json.lock"

    db_path.touch(exist_ok=True)

    with FileLock(db_lock):
        with db_path.open("r+") as f:
            data = f.read()
            db = JSONEncoder.loads(data) if len(data) > 0 else {}

            yield db

            f.truncate(0)
            f.seek(0)
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


def integrate_multi(models: Sequence[Model], *args, **kwargs):
    """Integrate the hops equations for the ``models``.
    Like :any:`integrate` just for many models.

    A call to :any:`ray.init` may be required.
    """

    for model in models:
        integrate(model, *args, *kwargs)


def integrate(model: Model, n: int, data_path: str = "./.data"):
    """Integrate the hops equations for the model.

    A call to :any:`ray.init` may be required.

    :param n: The number of samples to be integrated.
    """

    hash = model.hexhash

    # with model_db(data_path) as db:
    #     if hash in db and "data" db[hash]

    supervisor = HOPSSupervisor(
        model.hops_config,
        n,
        data_path=data_path,
        data_name=hash,
    )

    supervisor.integrate()

    with supervisor.get_data(True) as data:
        with model_db(data_path) as db:
            db[hash] = {
                "model_config": model.to_dict(),
                "data_path": str(Path(data.hdf5_name).relative_to(data_path)),
            }


def get_data(
    model: Model, data_path: str = "./.data", read_only: bool = True, **kwargs
) -> HIData:
    """
    Get the integration data of the model ``model`` based on the
    ``data_path``.  If ``read_only`` is :any:`True` the file is opened
    in read-only mode.  The ``kwargs`` are passed on to :any:`HIData`.
    """

    hash = model.hexhash

    with model_db(data_path) as db:
        if hash in db and "data_path" in db[hash]:
            return HIData(
                Path(data_path) / db[hash]["data_path"], read_only=read_only, **kwargs
            )
        else:
            raise RuntimeError(f"No data found for model with hash '{hash}'.")
