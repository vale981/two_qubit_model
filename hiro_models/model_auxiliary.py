"""Functionality to integrate :any:`Model` instances and analyze the results."""

from typing import Any, Optional

from hops.core.hierarchy_data import HIData
from qutip.steadystate import _default_steadystate_args
from typing import Any
from .model_base import Model
from hops.core.integration import HOPSSupervisor
from contextlib import contextmanager
from .utility import JSONEncoder, object_hook
from filelock import FileLock
from pathlib import Path
from .one_qubit_model import QubitModel, QubitModelMutliBath
from .two_qubit_model import TwoQubitModel
from .otto_cycle import OttoEngine
from collections.abc import Sequence, Iterator, Iterable, Callable
import shutil
import logging
import copy
import os
import numpy as np
from multiprocessing import Process
import hops.core.signal_delay as signal_delay
import signal
from typing import Union
import hopsflow.util


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

    with FileLock(db_lock):
        db_path.touch(exist_ok=True)
        with db_path.open("r+") as f:
            data = f.read()
            db = JSONEncoder.loads(data) if len(data) > 0 else {}
            db = {
                key: model_hook(value) if isinstance(value, dict) else value
                for key, value in db.items()
            }

            yield db

            f.truncate(0)
            f.seek(0)
            f.write(JSONEncoder.dumps(db, indent=4))


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

        if model == "QubitModelMutliBath":
            return QubitModelMutliBath.from_dict(treated_vals)

        if model == "OttoEngine":
            return OttoEngine.from_dict(treated_vals)

    for key, value in dct.items():
        if isinstance(value, dict):
            dct[key] = model_hook(value)

    return dct


def integrate_multi(models: Sequence[Model], *args, **kwargs):
    """Integrate the hops equations for the ``models``.
    Like :any:`integrate` just for many models.

    A call to :any:`ray.init` may be required.
    """

    for model in models:
        integrate(model, *args, **kwargs)


def integrate(
    model: Model,
    n: int,
    data_path: str = "./.data",
    clear_pd: bool = False,
    single_process: bool = False,
    stream_file: Optional[str] = None,
    analyze: bool = False,
    results_path: str = "results",
    analyze_kwargs: Optional[dict] = None,
):
    """Integrate the hops equations for the model.

    A call to :any:`ray.init` may be required.

    :param n: The number of samples to be integrated.
    :param clear_pd: Whether to clear the data file and redo the
        integration.
    :param single_process: Whether to integrate with a single process.
    :param stream_file: The path to the fifo that the trajectories are
        to be streamed to.
    :param analyze: Whether to analyze the results streamed to the
        ``stream_file`` using :any:`hopsflow`.

        Only applies when using the ``stream_file`` option.

    :param analyze_kwargs: Keyword arguments passed to :any:`hopsflow.util.ensemble_mean_online`.
    """

    hash = model.hexhash

    # with model_db(data_path) as db:
    #     if hash in db and "data" db[hash]

    supervisor = HOPSSupervisor(
        model.hops_config,
        n,
        data_path=data_path,
        data_name=hash,
        stream_file=stream_file,
    )

    analysis_process = None
    if stream_file is not None and analyze:
        if not os.path.exists(stream_file):
            os.mkfifo(stream_file)

        if analyze_kwargs is None:
            analyze_kwargs = dict()

        logging.info("Starting analysis process.")

        def target():
            for sgn in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]:
                signal.signal(sgn, signal.SIG_IGN)

            model.all_energies_online(
                stream_pipe=stream_file,
                results_directory=results_path,
                **analyze_kwargs,
            )

        analysis_process = Process(target=target)

        analysis_process.start()
        logging.info(f"Started analysis process with pid {analysis_process.pid}.")

    def cleanup(_):
        del _

        if analysis_process is not None:
            analysis_process.join()

        with supervisor.get_data(True, stream=False) as data:
            with model_db(data_path) as db:
                dct = {
                    "model_config": model.to_dict(),
                    "data_path": str(Path(data.hdf5_name).relative_to(data_path)),
                }

                if analysis_process:
                    dct["analysis_files"] = {
                        "flow": model.online_flow_name,
                        "interaction": model.online_interaction_name,
                        "interaction_power": model.online_interaction_power_name,
                        "system": model.online_system_name,
                        "system_power": model.online_system_power_name,
                    }

                db[hash] = dct

    with signal_delay.sig_delay(
        [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1],
        cleanup,
    ):
        if single_process:
            supervisor.integrate_single_process(clear_pd)
        else:
            supervisor.integrate(clear_pd)

    cleanup(0)


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
                return HIData(path, read_only=read_only, robust=False, **kwargs)
            except:
                return HIData(
                    path,
                    hi_key=model.hops_config,
                    read_only=False,
                    check_consistency=False,
                    overwrite_key=True,
                    robust=False,
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
    results_path: Union[Path, str] = "./results",
    other_results_path: Union[Path, str] = "./results_other",
    interactive: bool = False,
    models_to_import: Optional[Iterable[Model]] = None,
    force: bool = False,
    skip_checkpoints: bool = False,
):
    """
    Imports results from the ``other_data_path`` into the
    ``other_data_path`` if the files are newer.

    If ``interactive`` is any :any:`True`, the routine will ask before
    copying.

    If ``models_to_import`` is specified, only data of models matching
    those in ``models_to_import`` will be imported.

    If ``skip_checkpoints`` is :any:`False`, the result checkpoints
    won't be imported.
    """

    hashes_to_import = (
        [model.hexhash for model in models_to_import] if models_to_import else []
    )

    results_path = Path(results_path)
    other_results_path = Path(other_results_path)

    with model_db(other_data_path) as other_db:
        for current_hash, data in other_db.items():
            with model_db(data_path) as db:
                if "data_path" not in data:
                    continue

                do_import = False

                if hashes_to_import and current_hash not in hashes_to_import:
                    logging.info(f"Skipping {current_hash}.")
                    continue

                this_path = Path(data_path) / data["data_path"]
                this_path_tmp = this_path.with_suffix(".part")
                other_path = Path(other_data_path) / data["data_path"]

                if current_hash not in db:
                    do_import = True
                elif "data_path" not in db[current_hash]:
                    do_import = True
                elif (
                    is_smaller(
                        this_path,
                        other_path,
                    )
                    or force
                ):
                    do_import = True

                if not do_import:
                    logging.info(f"Not importing {current_hash}.")

                if do_import:
                    config = data["model_config"]
                    logging.warning(f"Importing {other_path} to {this_path}.")
                    logging.warning(f"The model description is '{config.description}'.")

                    if (
                        interactive
                        and input(f"Import {other_path}?\n[Y/N]: ").upper() != "Y"
                    ):
                        continue

                    this_path.parents[0].mkdir(exist_ok=True, parents=True)

                    if is_smaller(this_path, other_path) or force:
                        shutil.copy2(other_path, this_path_tmp)
                        os.system("sync")
                        shutil.move(this_path_tmp, this_path)

                        if "analysis_files" in data:
                            for fname in data["analysis_files"].values():
                                other_path = other_results_path / fname

                                for other_sub_path in (
                                    [other_path]
                                    if skip_checkpoints
                                    else hopsflow.util.get_all_snaphot_paths(other_path)
                                ):
                                    this_path = results_path / other_sub_path.name
                                    this_path_tmp = this_path.with_suffix(".tmp")

                                    logging.warning(
                                        f"Importing {other_path} to {this_path}."
                                    )

                                    if other_sub_path.exists():
                                        shutil.copy2(other_sub_path, this_path_tmp)
                                        os.system("sync")
                                        shutil.move(this_path_tmp, this_path)

                    db[current_hash] = data


def remove_models_from_db(models: list[Model], data_path: str = "./.data"):
    hashes_to_remove = [model.hexhash for model in models]

    with model_db(data_path) as db:
        for hash in list(db.keys()):
            if hash in hashes_to_remove:
                logging.warning(f"Deleting model '{hash}'.")
                del db[hash]


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


def migrate_db_to_new_hashes(
        data_path: str = "./.data", results_path: str = "./results", patch_fn: Optional[Callable] = None
):
    """
    Recomputes all the hashes of the models in the database under
    ``data_path`` and updates the database. The ``patch_fn`` receives
    the database data and can return an arbitrarily modified version
    of it from which the new hash is computed. This is useful if some
    miss-configuration in the models has to be corrected.
    """

    with model_db(data_path) as db:
        for old_hash in list(db.keys()):
            data = copy.deepcopy(db[old_hash])

            if patch_fn:
                data = patch_fn(data)

            new_hash = data["model_config"].hexhash
            del db[old_hash]
            db[new_hash] = data

            if "analysis_files" in db[new_hash]:
                for key, value in (db[new_hash]["analysis_files"]).items():
                    import pdb

                    db[new_hash]["analysis_files"][key] = value.replace(
                        old_hash, new_hash
                    )

            for result in os.listdir(results_path):
                if old_hash in result:
                    os.rename(
                        os.path.join(results_path, result),
                        os.path.join(results_path, result.replace(old_hash, new_hash)),
                    )


def model_diff_dict(models: Iterable[Model], **kwargs) -> dict[str, Any]:
    """
    Generate a which only contains paramaters that differ from between
    the instances in ``models``.

    The ``kwargs`` are passed to :any:`Model.to_dict`.
    """

    keys = set()
    dicts = [model.to_dict(**kwargs) for model in models]
    model_type = dicts[0]["__model__"]

    for model_dict in dicts:
        if model_dict["__model__"] != model_type:
            raise ValueError("All compared models must be of the same type.")

    for key, value in dicts[0].items():
        last_value = value
        for model_dict in dicts[1:]:
            value = model_dict[key]
            comp = last_value != value
            if comp.all() if isinstance(value, np.ndarray) else comp:
                keys.add(key)
                break

            last_value = value

    return {key: [dct[key] for dct in dicts] for key in keys}
