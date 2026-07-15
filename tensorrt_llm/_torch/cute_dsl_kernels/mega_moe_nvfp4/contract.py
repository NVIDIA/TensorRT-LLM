# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Codegen-time finite mapping contracts for RMEM tensor handoff."""

from __future__ import annotations

import inspect
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


class ContractError(ValueError):
    """Base error for malformed or mismatched contracts."""


class ContractMismatchError(ContractError):
    """Raised when two contracts do not describe the same mapping."""


class MappingSpec(Protocol):
    """Protocol for objects that can produce a canonical mapping table."""

    def normalize(self, *, domain: "Space",
                  codomain: "Space") -> tuple[int, ...]:
        ...


def _as_tuple(value: Iterable[object], *, name: str) -> tuple[object, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc


@dataclass(frozen=True)
class Space:
    """A finite coordinate space; axis 0 is fastest for linearization."""

    names: tuple[str, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        names = _as_tuple(self.names, name="names")
        sizes = _as_tuple(self.sizes, name="sizes")
        if len(names) != len(sizes):
            raise ContractError(
                f"Space names/sizes rank mismatch: {len(names)} != {len(sizes)}"
            )
        if not names:
            raise ContractError("Space must have at least one axis")
        if any(not isinstance(n, str) or not n for n in names):
            raise ContractError("Space names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ContractError(f"Space names must be unique, got {names!r}")
        if any(not isinstance(s, int) or s <= 0 for s in sizes):
            raise ContractError(
                f"Space sizes must be positive integers, got {sizes!r}")

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "sizes", sizes)

    @property
    def rank(self) -> int:
        return len(self.names)

    @property
    def size(self) -> int:
        return prod(self.sizes)

    def coordinates(self) -> tuple[tuple[int, ...], ...]:
        """Enumerate all coordinates in canonical CuTe-style order."""
        return tuple(self.delinearize(i) for i in range(self.size))

    def linearize(self, coord: Sequence[int]) -> int:
        """Convert a coordinate tuple into a CuTe-style linear index."""
        coord_tuple = tuple(coord)
        if len(coord_tuple) != self.rank:
            raise ContractError(f"Coordinate rank mismatch for {self.names!r}: "
                                f"{len(coord_tuple)} != {self.rank}")

        linear = 0
        stride = 1
        for axis, (idx, size) in enumerate(zip(coord_tuple, self.sizes)):
            if not isinstance(idx, int):
                raise ContractError(
                    f"Coordinate {self.names[axis]!r} must be int, got {type(idx)!r}"
                )
            if idx < 0 or idx >= size:
                raise ContractError(
                    f"Coordinate {self.names[axis]!r}={idx} out of bounds [0, {size})"
                )
            linear += idx * stride
            stride *= size
        return linear

    def delinearize(self, linear: int) -> tuple[int, ...]:
        """Convert a CuTe-style linear index into a coordinate tuple."""
        if not isinstance(linear, int):
            raise ContractError(
                f"Linear index must be int, got {type(linear)!r}")
        if linear < 0 or linear >= self.size:
            raise ContractError(
                f"Linear index {linear} out of bounds [0, {self.size})")

        remaining = linear
        coord = [0] * self.rank
        for axis in range(self.rank):
            size = self.sizes[axis]
            coord[axis] = remaining % size
            remaining //= size
        return tuple(coord)

    def rename(self, rename_map: Mapping[str, str]) -> "Space":
        """Return a space with selected axis names renamed."""
        return Space(
            names=tuple(rename_map.get(name, name) for name in self.names),
            sizes=self.sizes,
        )


@dataclass(frozen=True)
class TableMapping:
    """Canonical mapping table.

    ``table[domain_linear_idx] == codomain_linear_idx``.
    """

    table: tuple[int, ...]

    def __post_init__(self) -> None:
        table = _as_tuple(self.table, name="table")
        if any(not isinstance(v, int) for v in table):
            raise ContractError("TableMapping entries must be integers")
        object.__setattr__(self, "table", table)

    @classmethod
    def identity(cls, space: Space) -> "TableMapping":
        """Build an identity mapping for equal domain/codomain spaces."""
        return cls(tuple(range(space.size)))

    @classmethod
    def from_codomain_coords(
        cls,
        *,
        domain: Space,
        codomain: Space,
        coords: Sequence[Sequence[int]],
    ) -> "TableMapping":
        """Build a table from one codomain coordinate per domain coordinate."""
        coord_tuple = tuple(tuple(coord) for coord in coords)
        if len(coord_tuple) != domain.size:
            raise ContractError(
                f"Expected {domain.size} codomain coords, got {len(coord_tuple)}"
            )
        return cls(tuple(codomain.linearize(coord) for coord in coord_tuple))

    def normalize(self, *, domain: Space, codomain: Space) -> tuple[int, ...]:
        """Validate and return the canonical table for the given spaces."""
        if len(self.table) != domain.size:
            raise ContractError(
                f"TableMapping length must equal domain size {domain.size}, "
                f"got {len(self.table)}")
        for idx, value in enumerate(self.table):
            if value < 0 or value >= codomain.size:
                raise ContractError(
                    f"TableMapping entry {idx} maps to {value}, outside "
                    f"codomain bounds [0, {codomain.size})")
        return self.table


@dataclass(frozen=True)
class FunctionMapping:
    """Mapping generated by a Python pure function.

    The function signature must match the domain axis names by name.  Parameter
    order does not matter because calls are made with keyword arguments.

    The return value is interpreted as a codomain coordinate:
      - ``int`` is accepted for rank-1 codomains.
      - ``tuple``/``list`` values are interpreted in ``codomain.names`` order.
      - ``Mapping[str, int]`` values are interpreted by codomain axis name.

    The function is expected to be deterministic and side-effect free.  This
    module cannot prove purity; it only calls the function while normalizing.
    """

    function: Callable[..., int | Sequence[int] | Mapping[str, int]]

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise ContractError("FunctionMapping function must be callable")

    @staticmethod
    def _function_name(function: Callable[..., object]) -> str:
        return getattr(function, "__qualname__",
                       getattr(function, "__name__", repr(function)))

    @classmethod
    def _validate_signature(
        cls,
        function: Callable[..., object],
        domain: Space,
    ) -> inspect.Signature:
        signature = inspect.signature(function)
        params = signature.parameters
        bad_params = [
            name for name, param in params.items() if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        if bad_params:
            raise ContractError(
                f"FunctionMapping {cls._function_name(function)} has unsupported "
                f"parameters {bad_params!r}; use named positional-or-keyword or "
                f"keyword-only parameters")

        default_params = [
            name for name, param in params.items()
            if param.default is not inspect.Parameter.empty
        ]
        if default_params:
            raise ContractError(
                f"FunctionMapping {cls._function_name(function)} parameters must "
                f"not have defaults, got {default_params!r}")

        param_names = tuple(params.keys())
        if set(param_names) != set(domain.names):
            raise ContractError(
                f"FunctionMapping {cls._function_name(function)} parameters must "
                f"match domain names {domain.names!r}, got {param_names!r}")
        return signature

    @staticmethod
    def _result_to_codomain_coord(
        result: int | Sequence[int] | Mapping[str, int],
        *,
        codomain: Space,
        domain_coord: tuple[int, ...],
        function_name: str,
    ) -> tuple[int, ...]:
        if isinstance(result, MappingABC):
            result_keys = tuple(result.keys())
            if set(result_keys) != set(codomain.names):
                raise ContractError(
                    f"FunctionMapping {function_name} returned mapping keys "
                    f"{result_keys!r} at domain coord {domain_coord!r}; expected "
                    f"codomain names {codomain.names!r}")
            coord = tuple(result[name] for name in codomain.names)
        elif isinstance(result, int) and codomain.rank == 1:
            coord = (result, )
        elif isinstance(result,
                        SequenceABC) and not isinstance(result, (str, bytes)):
            coord = tuple(result)
            if len(coord) != codomain.rank:
                raise ContractError(
                    f"FunctionMapping {function_name} returned rank {len(coord)} "
                    f"at domain coord {domain_coord!r}; expected codomain rank "
                    f"{codomain.rank}")
        else:
            raise ContractError(
                f"FunctionMapping {function_name} returned unsupported value "
                f"{result!r} at domain coord {domain_coord!r}")

        if any(not isinstance(v, int) for v in coord):
            raise ContractError(
                f"FunctionMapping {function_name} returned non-integer codomain "
                f"coordinate {coord!r} at domain coord {domain_coord!r}")
        return coord

    def normalize(self, *, domain: Space, codomain: Space) -> tuple[int, ...]:
        """Enumerate the function over ``domain`` and return a canonical table."""
        self._validate_signature(self.function, domain)
        function_name = self._function_name(self.function)

        table: list[int] = []
        for domain_coord in domain.coordinates():
            binding = dict(zip(domain.names, domain_coord))
            result = self.function(**binding)
            codomain_coord = self._result_to_codomain_coord(
                result,
                codomain=codomain,
                domain_coord=domain_coord,
                function_name=function_name,
            )
            table.append(codomain.linearize(codomain_coord))
        return TableMapping(tuple(table)).normalize(domain=domain,
                                                    codomain=codomain)


@dataclass(frozen=True)
class Contract:
    """A normalized finite mapping contract."""

    domain: Space
    codomain: Space
    mapping: MappingSpec

    def __post_init__(self) -> None:
        # Validate eagerly so malformed contracts fail at construction time.
        self.mapping.normalize(domain=self.domain, codomain=self.codomain)

    @property
    def table(self) -> tuple[int, ...]:
        return self.mapping.normalize(domain=self.domain,
                                      codomain=self.codomain)

    def rename_domain(self, rename_map: Mapping[str, str]) -> "Contract":
        return Contract(
            domain=self.domain.rename(rename_map),
            codomain=self.codomain,
            mapping=self.mapping,
        )

    def rename_codomain(self, rename_map: Mapping[str, str]) -> "Contract":
        return Contract(
            domain=self.domain,
            codomain=self.codomain.rename(rename_map),
            mapping=self.mapping,
        )

    def is_equivalent_to(self, other: "Contract") -> bool:
        return (self.domain == other.domain and self.codomain == other.codomain
                and self.table == other.table)

    def assert_equivalent_to(self,
                             other: "Contract",
                             *,
                             context: str = "") -> None:
        """Raise a readable error unless two contracts are exactly equivalent."""
        if self.is_equivalent_to(other):
            return

        prefix = f"{context}: " if context else ""
        details: list[str] = []
        if self.domain != other.domain:
            details.append(
                f"domain mismatch: {self.domain!r} != {other.domain!r}")
        if self.codomain != other.codomain:
            details.append(
                f"codomain mismatch: {self.codomain!r} != {other.codomain!r}")
        if self.table != other.table:
            mismatch_idx = next(
                (idx
                 for idx, (lhs, rhs) in enumerate(zip(self.table, other.table))
                 if lhs != rhs),
                None,
            )
            if mismatch_idx is None and len(self.table) != len(other.table):
                details.append(
                    f"table length mismatch: {len(self.table)} != {len(other.table)}"
                )
            elif mismatch_idx is not None:
                details.append(
                    f"table mismatch at domain linear index {mismatch_idx}: "
                    f"{self.table[mismatch_idx]} != {other.table[mismatch_idx]}"
                )

        raise ContractMismatchError(prefix + "; ".join(details))


def assert_contract_equivalent(
    actual: Contract,
    expected: Contract,
    *,
    context: str = "",
) -> None:
    """Function-style wrapper for contract equivalence checks."""
    actual.assert_equivalent_to(expected, context=context)


@dataclass(frozen=True)
class TensorWithContract:
    """A handle pairing a runtime tensor with its codegen-time contract.

    Used when handing per-thread RMEM tensors between epilogue components.
    A bare RMEM tensor is thread-distributed and has no logical shape on its
    own; the contract supplies the ``(thread, reg) -> (logical)`` mapping
    that gives the tensor a meaning across the warp.

    The ``tensor`` field is typed as ``Any`` because this module stays
    independent of CuTe runtime types; in practice it carries a ``cute.Tensor``.
    The ``contract`` field is a pure-Python codegen-time object that can be
    compared against another contract via ``assert_contract_equivalent``.

    Both fields are immutable: a TensorWithContract is a passive label, not
    a mutable container.  Mutations to the underlying RMEM happen through
    the runtime tensor object directly (its identity is preserved).
    """

    tensor: Any
    contract: Contract


def eval_function_mapping(contract: Contract, **domain_coord):
    """Evaluate a FunctionMapping contract at runtime."""
    if not isinstance(contract.mapping, FunctionMapping):
        raise TypeError("runtime contract eval requires a FunctionMapping")

    result = contract.mapping.function(**domain_coord)
    if isinstance(result, dict):
        return result
    if isinstance(result, (tuple, list)):
        if len(result) != contract.codomain.rank:
            raise ValueError(
                "FunctionMapping result rank does not match codomain rank: "
                f"{len(result)} vs {contract.codomain.rank}")
        return {
            name: result[i]
            for i, name in enumerate(contract.codomain.names)
        }
    if contract.codomain.rank == 1:
        return {contract.codomain.names[0]: result}
    raise TypeError(
        "FunctionMapping runtime eval must return dict/tuple/list, or scalar "
        "for rank-1 codomain")
