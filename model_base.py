"""A base class for model HOPS configs."""

from utility import JSONEncoder, object_hook
import numpy as np
import json
import copy
import hashlib


class Model:
    """
    A base class with some data management functionality.
    """

    __version__: int = 1
    """
    The version of the model implementation.  It is increased for
    breaking changes.
    """

    def __init__(self, *_, **__):
        del _, __
        pass

    ###########################################################################
    #                                 Utility                                 #
    ###########################################################################

    def to_json(self):
        """Returns a json representation of the model configuration."""

        return json.dumps(
            {key: self.__dict__[key] for key in self.__dict__ if key[0] != "_"}
            | {"__version__": self.__version__},
            cls=JSONEncoder,
            ensure_ascii=False,
        )

    def __hash__(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).digest().__hash__()

    @classmethod
    def from_json(cls, json_str: str):
        """
        Tries to instantiate a model config from the json string
        ``json_str``.
        """

        model_dict = json.loads(json_str, object_hook=object_hook)
        assert (
            model_dict["__version__"] == cls().__version__
        ), "Incompatible version detected."

        del model_dict["__version__"]

        return cls(**model_dict)

    def __eq__(self, other):
        this_keys = list(self.__dict__.keys())
        ignored_keys = ["_sigmas"]

        for key in this_keys:
            if key not in ignored_keys:
                this_val, other_val = self.__dict__[key], other.__dict__[key]

                same = this_val == other_val

                if isinstance(this_val, np.ndarray):
                    same = same.all()

                if not same:
                    return False

        return self.__hash__() == other.__hash__()

    def copy(self):
        """Return a deep copy of the model."""

        return copy.deepcopy(self)
