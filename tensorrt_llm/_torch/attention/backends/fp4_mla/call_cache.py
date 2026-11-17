# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded cache of the FP4 MLA CuTe DSL kernel hosts.

Each host keeps the device executors of its compiled kernels, and the calls it
has prepared (a kernel bound to one set of arguments), in caches of its own.
This module imports neither CUDA nor CuTe DSL code, so a host that loads its
kernel lazily can use it.
"""

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

_V = TypeVar("_V")


class LruCache(Generic[_V]):
    """Map of at most ``capacity`` entries that drops the least recently used one; thread-safe."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[Hashable, _V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> _V | None:
        """The entry of ``key``, marked as just used; None without one."""
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: Hashable, value: _V) -> _V:
        """Store ``value`` unless ``key`` has an entry already; return the entry the cache holds.

        Threads that prepare the same key at once all get the entry stored first.
        """
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = value
            if len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def class_defines_callables(value: object, *names: str) -> bool:
    """Whether the type of ``value`` defines each of ``names`` as a callable."""
    value_type = type(value)
    return all(callable(getattr(value_type, name, None)) for name in names)
