from enum import Enum


class AllocatorTrimStrategy(Enum):
    """How a block releases its model's memory when its scope exits."""

    TRIM = "trim"  # sync, meta params/persistent buffers, empty_cache() back to the OS
    DEFER = "defer"  # meta params/persistent buffers only; skip sync and empty_cache (allocator stays warm)
