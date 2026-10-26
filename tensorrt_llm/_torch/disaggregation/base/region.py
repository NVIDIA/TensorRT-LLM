from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


@dataclass(frozen=True)
class IndexRange:
    """
    Represents a closed interval [start, end], with both bounds >= 0.
    Commonly used for indexing layers, heads, or tokens.
    """

    start: int
    end: int

    def __post_init__(self):
        if not (isinstance(self.start, int) and isinstance(self.end, int)):
            raise TypeError("start and end must be integers")
        if self.start < 0 or self.end < 0:
            raise ValueError("start and end must be >= 0")
        if self.end < self.start:
            raise ValueError("end must be >= start")


class MemRegion(NamedTuple):
    """Describes a block of memory by starting pointer and size in bytes."""

    ptr: int
    bytes: int


class MemRegionGroup(NamedTuple):
    """Describes a block of memory by starting pointer and size in bytes."""

    ptrs: np.ndarray  # dtype=np.int64
    bytes_per_region: int


@dataclass(frozen=True)
class RegionSpec:
    """
    Specifies a (potentially partial) region of the cache.
    Extend this base class for additional axis or specialization.
    """

    layers: Optional[IndexRange] = None


@dataclass(frozen=True)
class KVRegionSpec(RegionSpec):
    """
    Specifies a region within the Key/Value cache, with optional axes.
    """

    heads: Optional[IndexRange] = None
    tokens: Optional[IndexRange] = None


class SpecRegion(NamedTuple):
    """
    Associates a memory region with its semantic specifier.
    """

    memory: MemRegion | MemRegionGroup
    spec: RegionSpec = None


class SpecRegionPair(NamedTuple):
    """
    Maps a source descriptor to a destination descriptor
    (e.g., when copying or reindexing regions).
    """

    src: SpecRegion
    dst: SpecRegion


class RegionMapperBase(ABC):
    """
    Maps a batch of region descriptors to corresponding destination(s).
    """

    @abstractmethod
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        """
        Args:
            src_regions: List of source Regions.
            dst_regions: List of destination Regions.
        Returns:
            List of RegionPairs mapping source to destination.
        """
        ...
