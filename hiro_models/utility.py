from datetime import datetime
import dateutil.parser
from functools import singledispatchmethod
import dataclasses
from dataclasses import dataclass
from beartype import beartype
import json
import numpy as np
import qutip as qt
from typing import Any, Union, SupportsFloat
import hashlib
import hops.util.dynamic_matrix as dynamic_matrix
from hops.util.dynamic_matrix import DynamicMatrix, SmoothStep
import scipy.special
from numpy.typing import NDArray
from collections.abc import Iterable


@beartype
@dataclass
class StocProcTolerances:
    """
    An object to hold tolerances for :any:`stocproc.StocProc`
    instances.
    """

    integration: float = 1e-4
    """Integration tolerance."""

    interpolation: float = 1e-4
    """Interpolation tolerance."""


def hint_tuples(item: Any, rec=False):
    if hasattr(item, "to_dict"):
        item = item.to_dict()

    if isinstance(item, dict):
        return {key: hint_tuples(value, True) for key, value in item.items()}
    if isinstance(item, tuple):
        return {
            "type": "tuple",
            "value": [hint_tuples(i) for i in item],
        }

    if isinstance(item, list):
        return [hint_tuples(e, True) for e in item]
    else:
        return item


class JSONEncoder(json.JSONEncoder):
    """
    A custom encoder to serialize objects occuring in
    :any:`TwoQubitModel`.
    """

    def encode(self, obj: Any):
        return super().encode(hint_tuples(obj))

    @singledispatchmethod
    def default(self, obj: Any):
        if isinstance(obj, tuple):
            import pdb

            pdb.set_trace()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        return super().default(obj)

    @default.register(tuple)
    def _(self, obj: tuple):
        return {
            "type": "tuple",
            "value": list(*obj),
        }

    @default.register
    def _(self, arr: np.ndarray):
        return {"type": "array", "value": arr.tolist()}

    @default.register
    def _(self, obj: qt.Qobj):
        return {
            "type": "Qobj",
            "value": obj.full(),
            "dims": obj.dims,
            "obj_type": obj.type,
        }

    @default.register
    def _(self, obj: complex):
        return {"type": "complex", "re": obj.real, "im": obj.imag}

    @default.register
    def _(self, obj: StocProcTolerances):
        return {"type": "StocProcTolerances", "value": dataclasses.asdict(obj)}

    @default.register
    def _(self, obj: DynamicMatrix):
        return {
            "type": "DynamicMatrix",
            "subtype": obj.__class__.__name__,
            "value": obj.__getstate__(),
        }

    @default.register
    def _(self, obj: datetime):
        return {
            "type": "datetime",
            "value": obj.isoformat(),
        }

    @classmethod
    def dumps(cls, data, **kwargs) -> str:
        """Like :any:`json.dumps`, just for this encoder.

        The ``kwargs`` are passed on to :any:`json.dumps`.
        """

        return json.dumps(
            data,
            **kwargs,
            cls=cls,
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, string: str, **kwargs) -> dict[str, Any]:
        """Like :any:`json.loads`, just for this encoder.

        The ``kwargs`` are passed on to :any:`json.loads`.
        """

        if "object_hook" not in kwargs:
            kwargs["object_hook"] = object_hook

        return json.loads(string, **kwargs)

    @classmethod
    def hash(cls, data, **kwargs):
        """
        Like :any:`dumps`, only that the result is being piped into
        :any:`hashlib.sha256`. A ``sha256`` hash is being returned.
        """

        return hashlib.sha256(cls.dumps(data, sort_keys=True, **kwargs).encode("utf-8"))

    @classmethod
    def hexhash(cls, data, **kwargs) -> str:
        """
        Like :any:`hash`, only that a hexdigest is being returned.
        """

        return cls.hash(data, **kwargs).hexdigest()


def object_hook(dct: dict[str, Any]):
    """A custom decoder for the types introduced in :any:`JSONEncoder`."""

    if "type" in dct:
        type = dct["type"]

        if type == "array":
            return np.array(dct["value"])

        if type == "Qobj":
            return qt.Qobj(dct["value"], dims=dct["dims"], type=dct["obj_type"])

        if type == "complex":
            return dct["re"] + 1j * dct["im"]

        if type == "StocProcTolerances":
            return StocProcTolerances(**dct["value"])

        if type == "DynamicMatrix":
            return getattr(dynamic_matrix, dct["subtype"])(**dct["value"])

        if type == "tuple":
            return tuple(dct["value"])

        if type == "datetime":
            return dateutil.parser.parse(dct["value"])

    return dct


def operator_norm(obj: qt.Qobj) -> float:
    """Returns the operator norm of ``obj``."""

    return np.sqrt(max(np.abs((obj.dag() * obj).eigenenergies())))


def assert_serializable(model):
    """
    Serialize and restore ``model`` into json asserting that the
    objects stay the same.
    """

    assert model == model.__class__.from_json(
        model.to_json()
    ), "Serialization should not change the model."


def bcf_scale(
    δ: SupportsFloat,
    L: DynamicMatrix,
    t_max: SupportsFloat,
    s: SupportsFloat,
    ω_c: SupportsFloat,
) -> float:
    r"""
    Calculate the normalized BCF scale so that
    :any:`\langle{H_I}\rangle\approx = δ`.

    :param δ: The coupling strength.
    :param L: The coupling operators.
    :param t_max: The maximal simulation time points.
    :param s: The :math:`s` parameter of the (sub/super) ohmic BCF.
    :param ω_c: The cutoff frequency of the BCF.
    """

    L_expect = (L @ L.dag + L.dag @ L).max_operator_norm(t_max)
    bcf_norm = (
        np.pi * float(s) / (scipy.special.gamma(float(s) + 1) * float(ω_c) ** float(s))
    )
    return float(δ) / L_expect * bcf_norm


def linspace_with_strobe(
    begin: float, end: float, N: int, strobe_frequency: float
) -> NDArray[np.float64]:
    """
    Like ``linspace`` but so that the time points defined by the
    stroboscope angular frequency ``strobe_frequency`` are included.
    """

    return np.unique(
        np.sort(
            np.concatenate(
                [
                    np.linspace(begin, end, N),
                    np.arange(begin, end, 2 * np.pi / strobe_frequency),
                ]
            )
        )
    )


def strobe_times(time: NDArray[np.float64], frequency: float, tolerance: float = 1e-4):
    r"""
    Given a time array ``time`` and an angular frequency ``frequency`` (ω) the
    time points (and their indices) coinciding with :math:`2π / ω \cdot n` within the
    ``tolerance`` are being returned.
    """

    stroboscope_interval = 2 * np.pi / frequency

    strobe_indices = np.where((time % stroboscope_interval) <= tolerance)[0]

    if len(strobe_indices) == 0:
        raise ValueError("Can't match the strobe interval to the times.")

    strobe_times = time[strobe_indices]

    return strobe_times, strobe_indices
