"""Stable construction API for composable kernel implementations."""

import types
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Dict, Union, get_args, get_origin

from cutlass.cute.typing import SymInt


RuntimeIntegerType = SymInt
StaticIntegerType = int
StaticOrRuntimeIntegerType = Union[int, SymInt]


class OptionalRequirement:
    """Descriptor field that a component may consume conditionally."""

    __slots__ = ("expected_type",)

    def __init__(self, expected_type) -> None:
        self.expected_type = expected_type


Requirement = Union[type, OptionalRequirement]


def _required_type(requirement: Requirement):
    return requirement.expected_type if isinstance(requirement, OptionalRequirement) else requirement


def _matches_type(value, expected_type) -> bool:
    origin = get_origin(expected_type)
    if origin in (Union, types.UnionType):
        return any(
            _matches_type(value, candidate)
            for candidate in get_args(expected_type)
        )
    return isinstance(value, expected_type)


class Desc(Mapping[str, object]):
    """Immutable descriptor mapping validated against component schemas."""

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str):
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def validate(
        self,
        requirements: Dict[str, Requirement],
        *,
        component_name: str,
    ) -> None:
        for name, requirement in requirements.items():
            if name not in self._values:
                if isinstance(requirement, OptionalRequirement):
                    continue
                raise KeyError(
                    f"{component_name} requires descriptor field {name!r}."
                )
            expected_type = _required_type(requirement)
            value = self._values[name]
            if not _matches_type(value, expected_type):
                raise TypeError(
                    f"{component_name} requires {name!r} to have type "
                    f"{expected_type}, got {type(value)}."
                )


class ProblemDesc(Desc):
    """Problem semantics shared by every component in one kernel."""


class ImplDesc(Desc):
    """Fully static implementation choices shared by kernel components."""


class _KernelComponentMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._validate_component_init()
        return instance


class KernelComponent(ABC, metaclass=_KernelComponentMeta):
    """Component that consumes descriptors only during construction."""

    @classmethod
    @abstractmethod
    def problem_desc_require(cls) -> Dict[str, Requirement]:
        ...

    @classmethod
    @abstractmethod
    def impl_desc_require(cls) -> Dict[str, Requirement]:
        ...

    def _validate_desc_inputs(
        self,
        problem_desc: ProblemDesc,
        impl_desc: ImplDesc,
    ) -> None:
        component_name = type(self).__name__
        overlap = (
            self.problem_desc_require().keys()
            & self.impl_desc_require().keys()
        )
        if overlap:
            raise ValueError(
                f"{component_name} requires fields from both descriptors: "
                f"{sorted(overlap)}."
            )
        problem_desc.validate(
            self.problem_desc_require(),
            component_name=component_name,
        )
        impl_desc.validate(
            self.impl_desc_require(),
            component_name=component_name,
        )

    def _validate_component_init(self) -> None:
        component_name = type(self).__name__
        requirements = {
            **self.problem_desc_require(),
            **self.impl_desc_require(),
        }
        for name, requirement in requirements.items():
            if not hasattr(self, name):
                if isinstance(requirement, OptionalRequirement):
                    continue
                raise RuntimeError(
                    f"{component_name} did not bind required field {name!r}."
                )
            expected_type = _required_type(requirement)
            value = getattr(self, name)
            if not _matches_type(value, expected_type):
                raise TypeError(
                    f"{component_name}.{name} must have type "
                    f"{expected_type}, got {type(value)}."
                )
        for name, value in vars(self).items():
            if isinstance(value, Desc):
                raise RuntimeError(
                    f"{component_name}.{name} retains a descriptor."
                )


class KernelClass(KernelComponent):
    """Top-level host wrapper for one composable kernel implementation."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def aot_compile(self):
        ...


__all__ = [
    "Desc",
    "ImplDesc",
    "KernelClass",
    "KernelComponent",
    "OptionalRequirement",
    "ProblemDesc",
    "RuntimeIntegerType",
    "StaticIntegerType",
    "StaticOrRuntimeIntegerType",
]
