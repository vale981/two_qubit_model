from functools import singledispatchmethod
import dataclasses
from dataclasses import dataclass
from beartype import beartype
import json
import numpy as np
import qutip as qt
from typing import Any, Union
import hashlib
import hops.util.dynamic_matrix as dynamic_matrix
from hops.util.dynamic_matrix import DynamicMatrix, SmoothStep


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


class JSONEncoder(json.JSONEncoder):
    """
    A custom encoder to serialize objects occuring in
    :any:`TwoQubitModel`.
    """

    @singledispatchmethod
    def default(self, obj: Any):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        return super().default(obj)

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

        return hashlib.sha256(cls.dumps(data, **kwargs).encode("utf-8"))

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
