from __future__ import annotations

from typing import Type

from .base import TTAMethod
from .consensus import ConsensusTTA
from .experimental import ExperimentalTTA
from .metadata import MetadataTTA
from .temporal_gradient import TemporalGradientTTA
from .tent import TentTTA


METHOD_REGISTRY: dict[
    str,
    Type[TTAMethod],
] = {
    "metadata": MetadataTTA,
    "temporal_gradient": TemporalGradientTTA,
    "consensus": ConsensusTTA,
    "tent": TentTTA,
    "experimental": ExperimentalTTA,
}


def registered_methods() -> tuple[str, ...]:
    return tuple(
        sorted(
            METHOD_REGISTRY.keys()
        )
    )


def method_is_registered(
    name: str,
) -> bool:
    return str(name) in METHOD_REGISTRY


def get_method_class(
    name: str,
) -> Type[TTAMethod]:

    name = str(name).strip()

    if name not in METHOD_REGISTRY:

        available = ", ".join(
            registered_methods()
        )

        raise KeyError(
            f"Unknown TTA method {name!r}. "
            f"Registered methods: {available}"
        )

    return METHOD_REGISTRY[name]