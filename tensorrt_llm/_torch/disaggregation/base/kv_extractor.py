from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import List, NamedTuple, Optional

from base import MemoryDesc


class Role(IntFlag):
    """Bitmasking roles for KV: KEY, VALUE, or both."""

    KEY = auto()
    VALUE = auto()
    # Supports combinations: Role.KEY | Role.VALUE


class Layout(IntFlag):
    """Memory layout order choices."""

    HND = auto()
    NHD = auto()


@dataclass(frozen=True)
class PosIntRange:
    """
    Closed range [start, end] for positive integers.
    Useful for indexing layers, heads, or tokens.
    """

    start: int
    end: int

    def __post_init__(self):
        # Validate values on initialization: positive integers, start <= end
        if not (isinstance(self.start, int) and isinstance(self.end, int)):
            raise TypeError("start and end must be integers")
        if self.start < 1 or self.end < 1:
            raise ValueError("start and end must be >= 1")
        if self.end < self.start:
            raise ValueError("end >= start is required")


class DimBounds(NamedTuple):
    """
    Describes the dimensional information required for KV cache split and concatenation.
    - role: Role (usually KEY, VALUE, or both; default is both).
    - layer_range, head_range, token_range: Optional PosIntRange, None means full range.
    """

    role: Role = Role.KEY | Role.VALUE
    layer_range: Optional[PosIntRange] = None
    head_range: Optional[PosIntRange] = None
    token_range: Optional[PosIntRange] = None


class KVDesc(NamedTuple):
    """
    Describes a contiguous KV address space.
    - data: MemoryDesc object describing the memory.
    - bounds: DimBounds instance specifying the associated region and role.
    """

    data: MemoryDesc
    bounds: DimBounds


class KVExtractorBase(ABC):
    """
    Abstract base class for KV information extractors.
    """

    @abstractmethod
    def extract(self, range: Optional[DimBounds] = None) -> List[KVDesc]:
        """
        Extract KV descriptors for the given layer range.
        :param range: Range to extract, or None for all.
        :return: List of KVDesc instances.
        """
        ...
