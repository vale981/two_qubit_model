"""Functionality to integrate :any:`Model` instances and analyze the results."""

from typing import Any, Optional

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
from collections.abc import Sequence, Iterator, Iterable
import shutil
import logging
import copy


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
            db = (
                JSONEncoder.loads(data, object_hook=model_hook) if len(data) > 0 else {}
            )

            yield db

            f.truncate(0)
            f.seek(0)
            f.write(JSONEncoder.dumps(db))


def model_hook(dct: dict[str, Any]):
    """A custom decoder for the model types."""

    if "__model__" in dct:
        model = dct["__model__"]

        treated_vals = {
            key: object_hook(val) if isinstance(val, dict) else val
            for key, val in dct.items()
        }

        if model == "QubitModel":
            return QubitModel.from_dict(treated_vals)

        if model == "TwoQubitModel":
            return TwoQubitModel.from_dict(treated_vals)

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

    hexhash = model.hexhash

    with model_db(data_path) as db:
        if hexhash in db and "data_path" in db[hexhash]:
            path = Path(data_path) / db[hexhash]["data_path"]
            try:
                return HIData(path, read_only=read_only, **kwargs)
            except:
                return HIData(
                    path,
                    hi_key=model.hops_config,
                    read_only=False,
                    check_consistency=False,
                    overwrite_key=True,
                    **kwargs,
                )

        else:
            raise RuntimeError(f"No data found for model with hash '{hexhash}'.")


def model_data_iterator(
    models: Iterable[Model], *args, **kwargs
) -> Iterator[tuple[Model, HIData]]:
    """
    Yields tuples of ``model, data``, where ``data`` is already opened
    and will be closed automatically.

    For the rest of the arguments see :any:`get_data`.
    """

    for model in models:
        with get_data(model, *args, **kwargs) as data:
            yield model, data


def is_smaller(first: Path, second: Path) -> bool:
    """
    :returns: Wether the file ``first`` is smaller that ``second``.
    """

    if not first.exists():
        return True

    return first.stat().st_size < second.stat().st_size


def import_results(
    data_path: str = "./.data",
    other_data_path: str = "./.data_other",
    interactive: bool = False,
    models_to_import: Optional[Iterable[Model]] = None,
):
    """
    Imports results from the ``other_data_path`` into the
    ``other_data_path`` if the files are newer.

    If ``interactive`` is any :any:`True`, the routine will ask before
    copying.

    If ``models_to_import`` is specified, only data of models matching
    those in ``models_to_import`` will be imported.
    """

    hashes_to_import = (
        [model.hexhash for model in models_to_import] if models_to_import else []
    )

    with model_db(other_data_path) as other_db:
        for current_hash, data in other_db.items():
            with model_db(data_path) as db:
                if "data_path" not in data:
                    continue

                do_import = False

                if hashes_to_import and current_hash not in hashes_to_import:
                    logging.info(f"Skipping {current_hash}.")
                    continue

                if current_hash not in db:
                    do_import = True
                elif "data_path" not in db[current_hash]:
                    do_import = True
                elif is_smaller(
                    Path(data_path) / db[current_hash]["data_path"],
                    Path(other_data_path) / data["data_path"],
                ):
                    do_import = True

                if do_import:
                    this_path = Path(data_path) / data["data_path"]
                    this_path_tmp = this_path.with_suffix(".part")
                    other_path = Path(other_data_path) / data["data_path"]

                    config = data["model_config"]
                    logging.warning(f"Importing {other_path} to {this_path}.")
                    logging.warning(f"The model description is '{config.description}'.")

                    if (
                        interactive
                        and input(f"Import {other_path}?\n[Y/N]: ").upper() != "Y"
                    ):
                        continue

                    this_path.parents[0].mkdir(exist_ok=True, parents=True)

                    if is_smaller(this_path, other_path):
                        shutil.copy2(other_path, this_path_tmp)
                        shutil.move(this_path_tmp, this_path)

                    db[current_hash] = data


def cleanup(
    models_to_keep: list[Model], data_path: str = "./.data", preview: bool = True
):
    """Delete all model data except ``models_to_keep`` from
    ``data_path``.  If ``preview`` is :any:`True`, only warning
    messages about which files would be deleted will be printed.
    """

    hashes_to_keep = [model.hexhash for model in models_to_keep]
    data_path_resolved = Path(data_path)
    with model_db(data_path) as db:
        for hash in list(db.keys()):
            if hash not in hashes_to_keep:
                logging.warning(f"Deleting model '{hash}'.")
                info = db[hash]
                if "data_path" in info:
                    this_path = data_path_resolved / info["data_path"]

                    while this_path.parent != data_path_resolved:
                        this_path = this_path.parent

                    logging.warning(f"Removing '{this_path}'.")

                    if not preview:
                        this_path.unlink()
                        logging.warning(f"Done.")

                if not preview:
                    del db[hash]


def migrate_db_to_new_hashes(data_path: str = "./.data"):
    """
    Recomputes all the hashes of the models in the database under
    ``data_path`` and updates the database.
    """

    with model_db(data_path) as db:
        for old_hash in list(db.keys()):
            data = copy.deepcopy(db[old_hash])
            new_hash = data["model_config"].hexhash

            del db[old_hash]
            db[new_hash] = data
