from functools import singledispatchmethod
import dataclasses
from dataclasses import dataclass
from beartype import beartype
import json
import numpy as np
import qutip as qt
from typing import Any


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

    return dct


def operator_norm(obj: qt.Qobj) -> float:
    """Returns the operator norm of ``obj``."""

    return np.sqrt(max(np.abs((obj.dag() * obj).eigenenergies())))
