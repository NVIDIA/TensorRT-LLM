# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA driver and arithmetic helpers used by the KVCacheManagerV2 test-suite.

These live in the test directory rather than in the ``kv_cache_manager_v2`` package: they
are test-side scaffolding (raw-driver stream/event pooling, small index helpers, a
``sys.path`` context manager for sibling-module imports) with no role in the shipped
Python surface, which only re-exports the C++ implementation.
"""

import functools
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Set
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Iterable,
    Iterator,
    MutableSequence,
    NewType,
    Reversible,
    Sequence,
    TypeVar,
    cast,
)

import cuda.bindings.driver as drv
import cuda.bindings.runtime as cudart

T = TypeVar("T")
U = TypeVar("U")

NDEBUG: Final[bool] = os.environ.get("TLLM_DEBUG_MODE", "")[0:1] != "1"

CudaStream = NewType("CudaStream", int)


class OutOfMemoryError(Exception):
    pass


class CuOOMError(OutOfMemoryError):
    pass


class CuError(Exception):
    error_code: drv.CUresult

    def __init__(self, error_code: drv.CUresult) -> None:
        self.error_code = error_code
        err, err_str = drv.cuGetErrorString(error_code)
        if err != drv.CUresult.CUDA_SUCCESS:
            err_str = "<Failed to get error string with cuGetErrorString>"
        super().__init__(f"CUDA driver error: {error_code} ({err_str})")

    def __reduce__(self) -> tuple[type["CuError"], tuple[drv.CUresult]]:
        return (self.__class__, (self.error_code,))


def _unwrap(
    ret: drv.CUresult
    | tuple[
        drv.CUresult,
        T,
    ]
    | tuple[drv.CUresult, T, U],
):
    if isinstance(ret, drv.CUresult):
        if int(ret) != int(drv.CUresult.CUDA_SUCCESS):  # pyright: ignore
            if int(ret) == int(drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY):  # pyright: ignore
                raise CuOOMError()
            raise CuError(ret)
    else:
        _unwrap(ret[0])
        return ret[1] if len(ret) == 2 else ret[1:]


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    return div_up(x, y) * y


def exact_div(x: int, y: int) -> int:
    assert x % y == 0
    return x // y


Idx = TypeVar("Idx", bound=int)
Index = TypeVar("Index", bound=int, contravariant=True)


class HalfOpenRange(tuple[Idx, Idx], Generic[Idx]):
    """A half-open range [beg, end), falsy when empty (beg >= end).

    Generic over index type. Supports unpacking into (beg, end).
    """

    __slots__ = ()

    def __new__(cls, beg: Idx, end: Idx) -> "HalfOpenRange[Idx]":
        return tuple.__new__(cls, (beg, end))

    @property
    def beg(self) -> Idx:
        return self[0]

    @property
    def end(self) -> Idx:
        return self[1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HalfOpenRange):
            return NotImplemented
        return (not self and not other) or tuple.__eq__(self, other)

    def __hash__(self) -> int:
        return hash((0, 0)) if not self else tuple.__hash__(self)

    def __bool__(self) -> bool:
        return self[0] < self[1]

    def __len__(self) -> int:
        return max(0, self[1] - self[0])

    def __contains__(self, item: Any) -> bool:
        return self[0] <= item < self[1]


def intersect(a: HalfOpenRange[Idx], b: HalfOpenRange[Idx]) -> HalfOpenRange[Idx]:
    """Return the intersection of two half-open ranges [beg, end).

    The result may be empty (beg >= end), which is safe to chain into further intersections.
    """
    return HalfOpenRange(max(a[0], b[0]), min(a[1], b[1]))


def value_or(opt: T | None, default: T) -> T:
    return default if opt is None else opt


def unwrap_optional(value: T | None) -> T:
    if value is not None:
        return value
    raise ValueError("Expected non-None value")


def remove_if(original: MutableSequence[T], predicate: Callable[[T], bool]) -> list[T]:
    """Remove items from original that satisfy the predicate and return the removed items."""
    removed = []
    for idx, item in enumerate(original):
        if predicate(item):
            removed.append(item)
        else:
            original[idx - len(removed)] = item
    del original[len(original) - len(removed) :]
    return removed


def get_uniform_attribute(iterable: Iterable[T], attribute_func: Callable[[T], U]) -> U:
    ret = attribute_func(next(iter(iterable)))
    assert NDEBUG or all(attribute_func(item) == ret for item in iterable)
    return ret


def typed_range(*args: Index) -> Reversible[Index]:
    return cast(Reversible[Index], range(*args))


@functools.cache
def init_cuda_once() -> None:
    (err,) = cudart.cudaFree(0)
    assert int(err) == int(cudart.cudaError_t.cudaSuccess)


class SimplePool(Generic[T]):
    __slots__ = (
        "_create_func",
        "_destroy_func",
        "_init_size",
        "_max_size",
        "_outstanding_count",
        "_items",
    )
    _create_func: Callable[[], T]
    _destroy_func: Callable[[T], None]
    _init_size: int
    _max_size: int | None
    _items: deque[T] | None
    _outstanding_count: (
        int  # number of items currently we gave out but not returned, i.e. get() but not put()
    )

    def __init__(
        self,
        create_func: Callable[[], T],
        destroy_func: Callable[[T], None],
        init_size: int = 0,
        max_size: int | None = None,
    ):
        self._create_func = create_func
        self._destroy_func = destroy_func
        self._init_size = init_size
        self._max_size = max_size
        self._items = None
        self._outstanding_count = 0

    def clear(self) -> None:
        while self.items:
            self._destroy_func(self.items.popleft())

    def __del__(self) -> None:
        self.clear()

    @property
    def items(self) -> deque[T]:
        if self._items is None:
            self._items = deque[T](
                (self._create_func() for _ in range(self._init_size)), maxlen=self._max_size
            )
        return self._items

    def get(self) -> T:
        ret = self.items.popleft() if self.items else self._create_func()
        self._outstanding_count += 1
        return ret

    def put(self, item: T) -> None:
        self._outstanding_count -= 1
        if self._max_size is not None and len(self.items) >= self._max_size:
            self._destroy_func(item)
        else:
            self.items.append(item)

    @property
    def outstanding_count(self) -> int:
        """Number of items acquired with get() and not yet returned with put()."""
        return self._outstanding_count

    @property
    def cached_count(self) -> int:
        """Number of items currently in the pool."""
        return len(self.items)

    @property
    def total_count(self) -> int:
        """Total number of items created, both outstanding and cached."""
        return self.outstanding_count + self.cached_count


class ItemHolderBase(Generic[T], ABC):
    __slots__ = ("_item",)
    _item: T | None

    def __init__(self) -> None:
        self._item = self.pool.get()

    def close(self) -> None:
        # Manually inlined for better performance.
        item = self._item
        if item is not None:
            self.pool.put(item)
            self._item = None

    def __del__(self) -> None:
        self.close()

    def is_closed(self) -> bool:
        return self._item is None

    def get(self) -> T:
        # Manually inlined for better performance.
        item = self._item
        assert item is not None
        return item

    @property
    def handle(self) -> T:
        # Manually inlined for better performance.
        item = self._item
        assert item is not None
        return item

    @property
    @abstractmethod
    def pool(self) -> SimplePool[T]: ...


class CachedCudaEvent(ItemHolderBase[drv.CUevent]):
    """A cached CUDA event without support for timing. Recorded to a stream when created."""

    __slots__ = ()
    _pool: ClassVar[SimplePool[drv.CUevent] | None] = None
    NULL: ClassVar["_NullCudaEvent"]

    def __init__(self, stream: CudaStream) -> None:
        super().__init__()
        self._record(stream)

    def query_complete(self) -> bool:
        """Query the event. If complete, also close the event. Closed events are always considered complete."""
        # Manually inlined for better performance.
        ev = self._item
        if ev is None:
            return True
        (err,) = drv.cuEventQuery(ev)
        if int(err) == int(drv.CUresult.CUDA_SUCCESS):
            self.close()
            return True
        elif int(err) == int(drv.CUresult.CUDA_ERROR_NOT_READY):
            return False
        else:
            raise CuError(err)

    def synchronize(self) -> None:
        # Manually inlined for better performance.
        ev = self._item
        if ev is None:
            return
        _unwrap(drv.cuEventSynchronize(ev))
        self.close()

    def wait_in_stream(self, stream: CudaStream) -> None:
        # Manually inlined for better performance.
        ev = self._item
        if ev is None:
            return
        _unwrap(drv.cuStreamWaitEvent(stream, ev, 0))

    def _record(self, stream: CudaStream) -> None:
        """Prefer new event instead of recording an existing event."""
        # Manually inlined for better performance.
        ev = self._item
        assert ev is not None
        _unwrap(drv.cuEventRecord(ev, stream))

    @property
    def pool(self) -> SimplePool[drv.CUevent]:
        if CachedCudaEvent._pool is None:
            CachedCudaEvent._pool = SimplePool[drv.CUevent](
                lambda: _unwrap(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DISABLE_TIMING)),
                lambda ev: _unwrap(drv.cuEventDestroy(ev)),  # pyright: ignore
                init_size=1024,
            )
        return CachedCudaEvent._pool


class _NullCudaEvent(CachedCudaEvent):
    """A null CUDA event that is closed (and always complete)."""

    __slots__ = ()

    def __init__(self) -> None:
        # do not call super().__init__(). We don't need an event here.
        self._item = None


CachedCudaEvent.NULL = _NullCudaEvent()


def stream_wait_events(stream: CudaStream, events: Iterable[CachedCudaEvent]) -> None:
    """Batched wait for multiple events with deduplication first."""
    if not isinstance(events, Set):
        events = set(events)
    for ev in events:
        ev.wait_in_stream(stream)


class CachedCudaStream(ItemHolderBase[CudaStream]):
    """A cached non-blocking CUDA stream."""

    __slots__ = ()
    _pool: ClassVar[SimplePool[CudaStream] | None] = None

    def __init__(self) -> None:
        super().__init__()

    def wait_event(self, event: drv.CUevent) -> None:
        _unwrap(drv.cuStreamWaitEvent(self.get(), event, drv.CU_STREAM_WAIT_VALUE_COMPLETED))

    def wait_events(self, events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]) -> None:
        """Wait for events with deduplication first."""
        stream_wait_events(self.get(), events)

    def record_event(self) -> CachedCudaEvent:
        return CachedCudaEvent(self.get())

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, int(self.get())

    def synchronize(self) -> None:
        _unwrap(drv.cuStreamSynchronize(self.handle))

    @property
    def pool(self) -> SimplePool[CudaStream]:
        if CachedCudaStream._pool is None:
            CachedCudaStream._pool = SimplePool[CudaStream](
                lambda: CudaStream(
                    int(_unwrap(drv.cuStreamCreate(drv.CUstream_flags.CU_STREAM_NON_BLOCKING)))  # pyright: ignore
                ),
                lambda stream: _unwrap(drv.cuStreamDestroy(stream)),  # pyright: ignore
                init_size=128,
            )
        return CachedCudaStream._pool


class TemporaryCudaStream(CachedCudaStream):
    """A cached non-blocking CUDA stream, used as a temporary worker stream.

    Requires a list of prior events to wait for dependencies. A finish event is recorded when exiting
    normally. Call take_finish_event() to consume the finish event, otherwise you get a warning.
    """

    __slots__ = "_finish_event"
    _finish_event: CachedCudaEvent | None

    def __init__(self, prior_events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]):
        super().__init__()
        self.wait_events(prior_events)
        self._finish_event = None

    def __del__(self) -> None:
        if self._finish_event is not None:
            warnings.warn("[KVCacheManager] finish event recorded but not taken")
        super().__del__()

    def take_finish_event(self) -> CachedCudaEvent:
        ret = unwrap_optional(self._finish_event)
        self._finish_event = None
        return ret

    def __enter__(self) -> "TemporaryCudaStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not exc_type:
            self._finish_event = self.record_event()


@contextmanager
def temporary_sys_path(path: str) -> Iterator[None]:
    already_in_path = path in sys.path
    if not already_in_path:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if not already_in_path:
            sys.path.remove(path)
