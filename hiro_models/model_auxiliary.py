"""Functionality to integrate :any:`Model` instances and analyze the results."""

from typing import Any

from hops.core.hierarchy_data import HIData
from qutip.steadystate import _default_steadystate_args
from .model_base import Model
from hops.core.integration import HOPSSupervisor
from contextlib import contextmanager
from .utility import JSONEncoder, object_hook
from filelock import FileLock
from pathlib import Path
from .one_qubit_model import QubitModel
from .two_qubit_model import TwoQubitModel
from collections.abc import Sequence, Iterator
import shutil
import logging


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
        integrate(model, *args, **kwargs)


def integrate(model: Model, n: int, data_path: str = "./.data", clear_pd: bool = False):
    """Integrate the hops equations for the model.

    A call to :any:`ray.init` may be required.

    :param n: The number of samples to be integrated.
    :param clear_pd: Whether to clear the data file and redo the integration.
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

    supervisor.integrate(clear_pd)

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


def model_data_iterator(
    models: Model, *args, **kwargs
) -> Iterator[tuple[Model, HIData]]:
    """
    Yields tuples of ``model, data``, where ``data`` is already opened
    and will be closed automatically.

    For the rest of the arguments see :any:`get_data`.
    """

    for model in models:
        with get_data(model, *args, **kwargs) as data:
            yield model, data


def import_results(data_path: str = "./.data", other_data_path: str = "./.data_other"):
    """
    Imports results from the ``other_data_path`` into the
    ``other_data_path`` if the files are newer.
    """

    with model_db(data_path) as db:
        with model_db(other_data_path) as other_db:
            for hash, data in other_db.items():
                if "data_path" not in data:
                    continue

                do_import = False

                if hash not in db:
                    do_import = True
                elif "data_path" not in db[hash]:
                    do_import = True
                elif (Path(data_path) / db[hash]["data_path"]).stat().st_size < (
                    Path(other_data_path) / data["data_path"]
                ).stat().st_size:
                    do_import = True

                if do_import:
                    this_path = Path(data_path) / data["data_path"]
                    other_path = Path(other_data_path) / data["data_path"]

                    this_path.parents[0].mkdir(exist_ok=True, parents=True)
                    logging.info(f"Importing {other_path} to {this_path}.")
                    shutil.copy(other_path, this_path)

                    db[hash] = data
