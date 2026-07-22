# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple, Union

from tensorrt_llm.mapping import Mapping


class BorrowedWeightStorageRetentionError(RuntimeError):
    """A model loader retained storage borrowed from a weight-stream lease.

    A cooperative transport must treat this as a lifetime-safety signal, not
    only as a model-load failure: every rank must keep the backing transport
    allocation alive rather than unregistering or freeing storage that an
    escaped tensor may still reference.
    """


@dataclass(frozen=True)
class WeightGroup:
    """A set of checkpoint tensors that must be materialized atomically.

    Loaders may split a large group across transport batches, but consumers
    must not expose the group to model-specific loading code until the batch
    carrying ``group_complete=True`` has been consumed.
    """

    group_id: str
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("WeightGroup.group_id must not be empty")
        if not self.keys:
            raise ValueError("WeightGroup.keys must not be empty")
        if any(not key for key in self.keys):
            raise ValueError("WeightGroup keys must not be empty")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("WeightGroup keys must be unique")


@dataclass(frozen=True)
class WeightSegment:
    """A byte range from one tensor placed in a transport batch payload."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    tensor_nbytes: int
    tensor_offset: int
    payload_offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("WeightSegment.key must not be empty")
        if not self.dtype:
            raise ValueError("WeightSegment.dtype must not be empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError(
                "WeightSegment.shape dimensions must be non-negative")
        if min(self.tensor_nbytes, self.tensor_offset, self.payload_offset,
               self.nbytes) < 0:
            raise ValueError("WeightSegment byte values must be non-negative")
        if self.tensor_offset + self.nbytes > self.tensor_nbytes:
            raise ValueError("WeightSegment exceeds its tensor byte range")


@dataclass(frozen=True)
class WeightBatch:
    """An ordered, atomically published batch in a bounded weight stream."""

    sequence: int
    slot: int
    group_id: str
    group_keys: tuple[str, ...]
    group_complete: bool
    segments: tuple[WeightSegment, ...]
    payload_nbytes: int

    def __post_init__(self) -> None:
        if min(self.sequence, self.slot, self.payload_nbytes) < 0:
            raise ValueError("WeightBatch numeric fields must be non-negative")
        WeightGroup(self.group_id, self.group_keys)
        if not self.segments:
            raise ValueError("WeightBatch.segments must not be empty")
        group_keys = set(self.group_keys)
        payload_ranges = []
        for segment in self.segments:
            if segment.key not in group_keys:
                raise ValueError(
                    f"WeightSegment {segment.key!r} is not in batch group")
            if segment.payload_offset + segment.nbytes > self.payload_nbytes:
                raise ValueError("WeightSegment exceeds its batch payload")
            if segment.nbytes:
                payload_ranges.append((segment.payload_offset,
                                       segment.payload_offset + segment.nbytes))
        payload_ranges.sort()
        if any(payload_ranges[index][0] < payload_ranges[index - 1][1]
               for index in range(1, len(payload_ranges))):
            raise ValueError("WeightBatch segment payload ranges overlap")


class WeightBatchLease(ABC):
    """A temporary view of one published batch.

    The view remains valid until it is passed to
    :meth:`WeightBatchStream.complete` or :meth:`WeightBatchStream.abort`.
    Consumers must release any derived buffer views before completing it.
    """

    @property
    @abstractmethod
    def batch(self) -> WeightBatch:
        """Return immutable metadata for the leased batch."""

    @abstractmethod
    def view(self, segment: WeightSegment) -> memoryview:
        """Return a read-only view of ``segment`` in the shared payload."""

    def borrow_direct_buffer(self, segment: WeightSegment) -> memoryview | None:
        """Borrow a zero-copy tensor buffer when the transport proves it safe.

        Implementations return ``None`` unless the payload is suitable for a
        direct asynchronous H2D read, including registration of its full
        lifetime with CUDA. The returned view may be writable because tensor
        buffer factories require it, but consumers must treat it as immutable,
        must not retain it, and must finish all device reads before releasing
        the lease.
        """
        del segment
        return None

    @abstractmethod
    def release(self) -> None:
        """Release local exported buffer views without acknowledging the batch."""


class WeightBatchStream(ABC):
    """Source-neutral protocol for bounded, rank-cooperative weight streams.

    Exactly one lease may be outstanding. Implementations may overlap filling
    the next slot with materialization of the current lease. Methods that
    coordinate ranks must be called in the same order by every rank and from
    the thread that owns the distributed runtime; implementations must not
    require MPI ``THREAD_MULTIPLE``.
    """

    @property
    @abstractmethod
    def groups(self) -> tuple[WeightGroup, ...]:
        """Return the validated atomic-group manifest in stream order."""

    @abstractmethod
    def start(self, error: BaseException | None = None) -> None:
        """Reach consumer-start consensus before the first node collective."""

    @abstractmethod
    def begin_next(self) -> WeightBatchLease | None:
        """Publish and lease the next ordered batch, or return ``None`` at EOF."""

    @abstractmethod
    def complete(self,
                 lease: WeightBatchLease,
                 error: BaseException | None = None) -> None:
        """Report local completion or failure and reach all-rank consensus."""

    def record_materialization(self, *, direct: bool, nbytes: int) -> None:
        """Record one locally materialized group for optional telemetry."""
        del direct, nbytes

    @abstractmethod
    def abort(self, error: BaseException) -> None:
        """Cancel the stream and propagate a deterministic all-rank error."""

    @abstractmethod
    def finalize(self, error: BaseException | None = None) -> None:
        """Reach terminal consensus and collectively release resources."""


class ConsumableWeightsDict:
    """
    Wrapper around a weights dictionary that allows marking keys as consumed
    to free memory during model loading.

    This reduces peak memory usage by deleting weight tensors from the dictionary
    after they have been copied to the model, rather than keeping all weights
    in memory until loading completes.

    Thread-safe: uses a lock to protect concurrent access. Iteration methods
    (keys, values, items, __iter__) return snapshot copies to allow safe
    concurrent iteration while other threads may modify the dictionary.
    """

    def __init__(self, weights: Dict[str, Any]):
        self._weights = weights
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._weights[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._weights[key]

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __iter__(self) -> Iterator[str]:
        # Return iterator over a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return iter(list(self._weights.keys()))

    def keys(self):
        # Return a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return list(self._weights.keys())

    def values(self):
        # Return a snapshot copy of values to allow concurrent modification
        with self._lock:
            return list(self._weights.values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        # Return a snapshot copy of items to allow concurrent modification
        with self._lock:
            return list(self._weights.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._weights.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            self._weights.update(other)

    def mark_consumed(self, prefix: str) -> int:
        """
        Delete all keys starting with the given prefix to free memory.

        Args:
            prefix: The prefix to match. Keys starting with "{prefix}." will be deleted.

        Returns:
            The number of keys deleted.

        Thread-safe: uses a lock to prevent concurrent modification issues.
        """
        with self._lock:
            keys_to_delete = [
                k for k in self._weights.keys() if k.startswith(prefix + ".")
            ]
            for key in keys_to_delete:
                del self._weights[key]
            return len(keys_to_delete)


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.
            **kwargs: Optional format-specific loader arguments.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
