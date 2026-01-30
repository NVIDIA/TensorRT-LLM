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

import array
import ctypes
import errno
import functools
import itertools
import operator
import os
import platform
import sys
import traceback
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Set
from contextlib import contextmanager
from ctypes.util import find_library
from itertools import pairwise
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Generic,
    Iterable,
    Iterator,
    MutableSequence,
    Protocol,
    Reversible,
    Sequence,
    Type,
    TypeVar,
    cast,
)

import cuda.bindings.driver as drv
import cuda.bindings.runtime as cudart

from . import rawref
from ._common import NDEBUG, CudaStream
from ._exceptions import CuError, CuOOMError, DiskOOMError, HostOOMError

T = TypeVar("T")
U = TypeVar("U")
Index = TypeVar("Index", bound=int, contravariant=True)
IndexO = TypeVar("IndexO", bound=int, covariant=True)
Row = TypeVar("Row", bound=int)
Col = TypeVar("Col", bound=int)


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


def round_down(x: int, y: int) -> int:
    return x // y * y


def in_range(x: int, lower: int, upper: int) -> bool:
    return lower <= x < upper


def exact_div(x: int, y: int) -> int:
    assert x % y == 0
    return x // y


def overlap(a: tuple[Index, Index], b: tuple[Index, Index]) -> tuple[Index, Index] | tuple[()]:
    "Returns the overlap of two ranges, or an empty tuple if they do not overlap."
    return (max(a[0], b[0]), min(a[1], b[1])) if a[0] < b[1] and b[0] < a[1] else ()


def value_or(opt: T | None, default: T) -> T:
    return default if opt is None else opt


def unwrap_optional(value: T | None) -> T:
    if value is not None:
        return value
    raise ValueError("Expected non-None value")


def unwrap_weakref(ref: weakref.ref[T]) -> T:
    obj = ref()
    if obj is not None:
        return obj
    raise ValueError("Dereferencing a dangling weakref")


def unwrap_rawref(ref: rawref.ref[T]) -> T:
    obj = ref()
    if obj is not None:
        return obj
    raise ValueError("Dereferencing a dangling rawref")


def map_optional(value: T | None, func: Callable[[T], U]) -> U | None:
    return func(value) if value is not None else None


def remove_if(original: MutableSequence[T], predicate: Callable[[T], bool]) -> list[T]:
    "Remove items from original that satisfy the predicate and return the removed items."
    removed = []
    for idx, item in enumerate(original):
        if predicate(item):
            removed.append(item)
        else:
            original[idx - len(removed)] = item
    del original[len(original) - len(removed) :]
    return removed


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            break
        yield chunk


def partition(original: Iterable[T], classifier: Callable[[T], U]) -> defaultdict[U, list[T]]:
    ret = defaultdict(list)
    for item in original:
        ret[classifier(item)].append(item)
    return ret


def get_uniform_attribute(iterable: Iterable[T], attribute_func: Callable[[T], U]) -> U:
    ret = attribute_func(next(iter(iterable)))
    assert NDEBUG or all(attribute_func(item) == ret for item in iterable)
    return ret


def assert_critical(condition: bool, message: str | None = None) -> None:
    "Similar to assert, but instead of raising an exception, it terminates the process, even if inside __del__()."
    if not condition:
        warnings.warn(value_or(message, "Critical assertion failed"))
        traceback.print_stack()
        os._exit(1)


def noexcept(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise AssertionError(f"Function {func.__name__} raised an exception: {e}") from e

    return wrapper


def not_implemented(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        raise NotImplementedError(f"The function '{func.__name__}' is not implemented yet.")

    return wrapper


def expect_type(ExpectedType: Type[T], value: Any) -> T:
    "Similar to typing.cast, but does runtime checking with assert."
    assert isinstance(value, ExpectedType), f"Expected {ExpectedType}, got {type(value)}"
    return value


def is_sorted(
    iterable: Iterable[T], key: Callable[[T], Any] = lambda x: x, reverse: bool = False
) -> bool:
    comp = operator.ge if reverse else operator.le
    return all(comp(key(a), key(b)) for a, b in pairwise(iterable))


HomoTuple = tuple[T, ...]


class TypedIndexList(Protocol[Index, T]):
    """
    A protocol representing a list-like container with a strongly typed integer index.
    Useful for enforcing index types like NewType wrappers around int.
    """

    def __getitem__(self, index: Index) -> T: ...

    def __setitem__(self, index: Index, value: T) -> None: ...

    def __delitem__(self, index: Index | slice) -> None: ...

    def __iter__(self) -> Iterator[T]: ...

    def __len__(self) -> int: ...

    def __reversed__(self) -> Iterator[T]: ...

    def clear(self) -> None: ...

    def pop(self) -> T: ...

    def append(self, value: T) -> None: ...


# @TODO: use this where applicable.
def to_typed(index_type: Type[Index], lst: list[T]) -> TypedIndexList[Index, T]:
    """
    Casts a standard list to a TypedIndexList with a strongly typed integer index.

    Parameters:
        index_type: A type alias for the NewType index, e.g. type(BlockOrdinal(0)) or a concrete class derived from int.
        lst: The list to cast

    Returns:
        A TypedIndexList[Index, T] with the specified index type
    """
    return cast(TypedIndexList[Index, T], lst)


def filled_list(value: T, count: Index) -> TypedIndexList[Index, T]:
    "Note that all elements will be the same value. Do not use mutable values."
    return cast(TypedIndexList[Index, T], [value] * int(count))


def make_typed(generator: Callable[[], T], count: Index) -> TypedIndexList[Index, T]:
    return cast(TypedIndexList[Index, T], [generator() for _ in range(int(count))])


def typed_len(iterable: TypedIndexList[IndexO, T]) -> IndexO:
    return cast(IndexO, len(iterable))


def typed_enumerate(iterable: TypedIndexList[Index, T]) -> Iterator[tuple[Index, T]]:
    return cast(Iterator[tuple[Index, T]], enumerate(iterable))


def typed_map(
    iterable: TypedIndexList[Index, T], func: Callable[[T], U]
) -> TypedIndexList[Index, U]:
    return cast(TypedIndexList[Index, U], [func(item) for item in iterable])


class Array2D(Generic[Row, Col, T]):
    __slots__ = ("_data", "_cols")
    _data: list[T]
    _cols: int

    def __init__(self, rows: Row, cols: Col, init_val: Iterable[T]) -> None:
        self._data = list(init_val)
        self._cols = cols

    def __getitem__(self, index: tuple[Row, Col]) -> T:
        return self._data[index[0] * self._cols + index[1]]

    def __setitem__(self, index: tuple[Row, Col], value: T) -> None:
        self._data[index[0] * self._cols + index[1]] = value

    @property
    def rows(self) -> int:
        assert len(self._data) % self._cols == 0
        return len(self._data) // self._cols

    def row(self, row: Row) -> TypedIndexList[Col, T]:
        return cast(TypedIndexList[Col, T], self._data[row * self._cols : (row + 1) * self._cols])

    def col(self, col: Col) -> TypedIndexList[Row, T]:
        return cast(TypedIndexList[Row, T], self._data[col :: self._cols])

    @property
    def cols(self) -> int:
        return self._cols

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._data)


def filled_array2d(rows: Row, cols: Col, val: T) -> Array2D[Row, Col, T]:
    return Array2D(rows, cols, [val] * rows * cols)


def typed_range(*args: Index) -> Reversible[Index]:
    return cast(Reversible[Index], range(*args))


def find(seq: Sequence[T], predicate: Callable[[T], bool], default: U) -> T | U:
    return next((item for item in seq if predicate(item)), default)


def find_index(seq: Iterable[T], predicate: Callable[[T], bool]) -> int:
    i = 0
    for i, item in enumerate(seq):
        if predicate(item):
            return i
    return i + 1


mem_alignment: Final[int] = 2 << 20  # 2MB

_libc = ctypes.CDLL(find_library("c"))
_libc.aligned_alloc.restype = ctypes.c_void_p
_libc.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_libc.madvise.restype = ctypes.c_int
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.realloc.restype = ctypes.c_void_p
_libc.realloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]
_libc.posix_fallocate.restype = ctypes.c_int
_libc.posix_fallocate.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]


def _aligned_alloc(alignment: int, size: int) -> int:
    """
    Allocates size bytes of uninitialized storage whose alignment is specified by alignment.
    Returns the address as an integer.
    Raises HostOOMError on failure.
    """
    assert size % alignment == 0
    memptr: ctypes.c_void_p = _libc.aligned_alloc(ctypes.c_size_t(alignment), ctypes.c_size_t(size))
    if memptr == ctypes.c_void_p(0):
        raise HostOOMError("aligned_alloc failed")
    return int(memptr)


def _madvise(ptr: int, size: int, advice: int) -> None:
    if os.name == "nt":
        return
    ret = _libc.madvise(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_int(advice))
    if ret != 0:
        error_code = ctypes.get_errno()
        error_msg = f"madvise failed with errno {error_code}: {errno.errorcode.get(error_code, 'Unknown error')}"
        raise HostOOMError(error_msg)


MADV_HUGEPAGE: Final[int] = 14


def _realloc(ptr: int, size: int) -> int:
    """
    Reallocates size bytes of storage whose alignment is specified by alignment.
    Returns the address as an integer.
    Raises OSError on failure.
    """
    ret = _libc.realloc(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
    if ret == ctypes.c_void_p(0):
        raise HostOOMError("realloc failed.")
    return int(ret)


def _free(ptr: int) -> None:
    _libc.free(ctypes.c_void_p(ptr))


def _posix_fallocate(fd: int, offset: int, length: int) -> None:
    ret = _libc.posix_fallocate(
        ctypes.c_int(fd), ctypes.c_longlong(offset), ctypes.c_longlong(length)
    )
    if ret != 0:
        raise DiskOOMError(ret, "posix_fallocate failed")


class HostMem:
    ALIGNMENT: ClassVar[int] = 2 << 20
    """
    Host memory aligned to 2MB, reallocable for low-cost resizing and registered to CUDA as page-locked memory.
    Resizing will keep the original memory content, like `realloc` in C.
    """
    __slots__ = ("_address", "_size", "_num_registered_chunks")
    _address: int
    _size: int
    # If True and _size > 2GB, use multiple chunks to register pinned memory due to a Linux kernel
    # 6.11/6.12/6.13 bug preventing pinning more than 2GB of host memory in one operation.
    _CHUNKED_REGISTRATION: ClassVar[bool] = platform.system() == "Linux" and platform.release()[
        :4
    ] in ["6.11", "6.12", "6.13"]
    _CHUNK_SIZE: ClassVar[int] = 2 << 30
    _num_registered_chunks: int

    @property
    def address(self) -> int:
        return self._address

    @property
    def size(self) -> int:
        return self._size

    def __init__(self, size: int) -> None:
        self._num_registered_chunks = 0
        if size == 0:
            self._address = 0
            self._size = 0
            return
        self._address = _aligned_alloc(mem_alignment, size)
        self._size = size
        _madvise(self._address, size, MADV_HUGEPAGE)
        self._register_to_cuda()

    def resize(self, new_size: int) -> None:
        self._unregister_from_cuda()
        try:
            self._address = _realloc(self._address, new_size)
            self._size = new_size
            _madvise(self._address, new_size, MADV_HUGEPAGE)
        finally:
            self._register_to_cuda()

    def destroy(self) -> None:
        if self._address == 0:
            return
        self._unregister_from_cuda()
        _free(self._address)
        self._address = 0
        self._size = 0

    def __del__(self) -> None:
        self.destroy()

    def _register_to_cuda(self) -> None:
        assert self._num_registered_chunks == 0
        for addr, size in self._iterate_chunks():
            _unwrap(
                drv.cuMemHostRegister(
                    addr, size, drv.CU_MEMHOSTREGISTER_PORTABLE | drv.CU_MEMHOSTREGISTER_DEVICEMAP
                )
            )
            self._num_registered_chunks += 1

    def _unregister_from_cuda(self) -> None:
        for addr, _ in self._iterate_chunks():
            if self._num_registered_chunks == 0:
                break
            _unwrap(drv.cuMemHostUnregister(addr))
            self._num_registered_chunks -= 1
        assert self._num_registered_chunks == 0

    def _iterate_chunks(self) -> Iterator[tuple[int, int]]:
        start = self._address
        end = start + self._size
        chunk_size = self._CHUNK_SIZE if self._CHUNKED_REGISTRATION else self._size
        for addr in range(start, end, chunk_size):
            yield addr, min(end - addr, chunk_size)


def resize_file(fd: int, new_size: int) -> None:
    old_size = os.lseek(fd, 0, os.SEEK_END)
    if new_size > old_size:
        _posix_fallocate(fd, old_size, new_size - old_size)
    elif new_size < old_size:
        os.truncate(fd, new_size)


class DynamicBitset:
    """
    A memory efficient bitset that can be resized.
    """

    __slots__ = ("_bits", "_num_set_bits")
    _bits: array.array
    _num_set_bits: int

    TYPE_CODE: ClassVar[str] = "Q"
    ALL_SET_MASK: ClassVar[int] = (1 << 64) - 1

    def __init__(self, capacity: int) -> None:
        self._bits = array.array(self.TYPE_CODE, [0] * (div_up(capacity, 64)))
        self._num_set_bits = 0

    def set(self, index: int) -> None:
        if not self.get(index):
            self._bits[index // 64] |= 1 << (index % 64)
            self._num_set_bits += 1

    def get(self, index: int) -> bool:
        return self._bits[index // 64] & (1 << (index % 64)) != 0

    def clear(self, index: int) -> None:
        if self.get(index):
            self._bits[index // 64] &= ~(1 << (index % 64))
            self._num_set_bits -= 1

    @property
    def num_set_bits(self) -> int:
        return self._num_set_bits

    def resize(self, new_capacity: int) -> None:
        extra_elems = div_up(new_capacity, 64) - len(self._bits)
        if extra_elems > 0:
            self._bits.extend(array.array(self.TYPE_CODE, [0] * extra_elems))
        elif extra_elems < 0:
            self._bits = self._bits[:extra_elems]
            if new_capacity % 64 != 0:
                self._bits[-1] &= self.ALL_SET_MASK >> (64 - (new_capacity % 64))

    # check if any bit in the range [start, end) is set
    def any_set(self, start: int, end: int) -> bool:
        if start >= end:
            return False
        start_word_mask = self.ALL_SET_MASK << (start % 64)
        end_word_mask = self.ALL_SET_MASK >> (64 - (end % 64))
        if start // 64 == end // 64:
            if (start_word_mask & end_word_mask & self._bits[start // 64]) != 0:
                return True
        else:
            if (start_word_mask & self._bits[start // 64]) != 0 or (
                end % 64 != 0 and end_word_mask & self._bits[end // 64]
            ) != 0:
                return True
        return any(self._bits[i] != 0 for i in range(start // 64 + 1, end // 64))


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
        "number of items currently we get() but not put()"
        return self._outstanding_count

    @property
    def cached_count(self) -> int:
        "number of items currently in the pool"
        return len(self.items)

    @property
    def total_count(self) -> int:
        "total number of items created, including both outstanding and cached"
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
    """
    A cached CUDA event without support for timing. Recorded to a stream when created.
    """

    __slots__ = ()
    _pool: ClassVar[SimplePool[drv.CUevent] | None] = None
    NULL: ClassVar["_NullCudaEvent"]

    def __init__(self, stream: CudaStream) -> None:
        super().__init__()
        self._record(stream)

    def query_complete(self) -> bool:
        """
        Query the event. If complete, also close the event. Closed events are always considered complete.
        """
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
        """
        Prefer new event instead of recording an existing event.
        """
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
    """
    A null CUDA event that is closed (and always complete).
    """

    __slots__ = ()

    def __init__(self) -> None:
        # do not call super().__init__(). We don't need an event here.
        self._item = None


CachedCudaEvent.NULL = _NullCudaEvent()


# @TODO: consider do this in a single batch call to C++.
def stream_wait_events(stream: CudaStream, events: Iterable[CachedCudaEvent]) -> None:
    "Batched wait for multiple events with deduplication first."
    if not isinstance(events, Set):
        events = set(events)
    for ev in events:
        ev.wait_in_stream(stream)


class CachedCudaStream(ItemHolderBase[CudaStream]):
    """
    A cached non-blocking CUDA stream.
    """

    __slots__ = ()
    _pool: ClassVar[SimplePool[CudaStream] | None] = None

    def __init__(self) -> None:
        super().__init__()

    def wait_event(self, event: drv.CUevent) -> None:
        _unwrap(drv.cuStreamWaitEvent(self.get(), event, drv.CU_STREAM_WAIT_VALUE_COMPLETED))

    def wait_events(self, events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]) -> None:
        """
        Wait for events with deduplication first.
        """
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
    """
    A cached non-blocking CUDA stream. Mainly used as temporary worker streams.
    Requires a list of prior events to wait for dependencies. A finish event is recorded when exiting
    normally. Call take_finish_event() to get the finish event.
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


def merge_events(events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]) -> CachedCudaEvent:
    if len(events) == 0:
        return CachedCudaEvent.NULL
    if len(events) == 1:
        ev = next(iter(events))
        return ev if not ev.is_closed() else CachedCudaEvent.NULL
    with TemporaryCudaStream(events) as stream:
        pass
    return stream.take_finish_event()


class MultiStreamExecutor:
    __slots__ = ("_prior_event", "_streams", "_finish_event")
    _prior_event: CachedCudaEvent
    _streams: list[TemporaryCudaStream]
    _finish_event: CachedCudaEvent | None

    def __init__(self, prior_events: Sequence[CachedCudaEvent] | set[CachedCudaEvent]):
        self._prior_event = merge_events(prior_events)
        self._streams = []
        self._finish_event = None

    def __enter__(self) -> "MultiStreamExecutor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        events = [s.take_finish_event() for s in self._streams]
        self._streams.clear()
        self._finish_event = merge_events(events)

    def __del__(self) -> None:
        assert_critical(self._finish_event is None, "finish event not taken")

    def new_stream(self) -> TemporaryCudaStream:
        stream = TemporaryCudaStream((self._prior_event,))
        self._streams.append(stream)
        return stream

    def take_finish_event(self) -> CachedCudaEvent:
        ret = unwrap_optional(self._finish_event)
        self._finish_event = None
        return ret


class SharedPoolProvider(Generic[T]):
    _pool: SimplePool[T]

    def __init__(self, pool: SimplePool[T]):
        self._pool = pool

    def pool(self) -> SimplePool[T]:
        return self._pool


class ItemHolderWithSharedPool(ItemHolderBase[T]):
    __slots__ = ("_pool",)
    _pool: SimplePool[T]

    def __init__(self, pool: SimplePool[T]) -> None:
        self._pool = pool
        super().__init__()

    def __del__(self) -> None:
        self.close()

    @property
    def pool(self) -> SimplePool[T]:
        return self._pool


HolderT = TypeVar("HolderT", bound=ItemHolderWithSharedPool)


# For subclassing if holder needs to be customized
class PooledFactoryBase(Generic[T, HolderT]):
    _Holder: Type[HolderT]  # subclasses must initialize this static attribute
    __slots__ = ("_pool",)
    _pool: SimplePool[T]

    def __init__(
        self,
        create_func: Callable[[], T],
        destroy_func: Callable[[T], None],
        init_size: int = 0,
        max_cache_size: int | None = None,
    ):
        self._pool = SimplePool[T](create_func, destroy_func, init_size, max_cache_size)

    def create(self) -> HolderT:
        return self._Holder(self._pool)

    def clear(self) -> None:
        self._pool.clear()


def query_total_gpu_memory() -> int:
    _, total = _unwrap(drv.cuMemGetInfo())  # pyright: ignore
    return total


def query_free_gpu_memory() -> int:
    free, _ = _unwrap(drv.cuMemGetInfo())  # pyright: ignore
    return free


class CudaStreamWrapper:
    "Just a wrapper to make it compatible with IsStreamT protocol. Does not own the stream."

    __slots__ = ("_stream",)
    _stream: CudaStream

    def __init__(self, stream: CudaStream) -> None:
        self._stream = stream

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, int(self._stream)


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
