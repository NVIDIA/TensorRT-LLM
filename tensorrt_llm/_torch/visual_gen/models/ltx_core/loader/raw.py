"""Read checkpoint tensors from file bytes straight into destination tensors.
``read_raw`` fills target tensors with raw checkpoint tensors by key: the destinations of a ``FileReadUnit``, which
rename raw keys, or the raw inputs of one that requires staging (see ``partitioners.FileReadPartitioner``). Reads are
positional (``preadv``) into caller memory: nothing is mapped, so they add no file-backed RSS and no tensor view
keeps checkpoint pages alive. ``RawFiles`` splits reads into ``ReadChunk``s for a shared pool of read workers
sized by ``ReadPoolConfig``; ``DeviceStager`` carries bytes to CUDA or MPS tensors through two reused host buckets.
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import TensorMapping

READ_WORKERS_ENV = "LTX_LOADER_READ_WORKERS"
# Reads are split into pieces of at most _READ_CHUNK_BYTES, and smaller reads are batched up to it, for a shared
# pool of read workers: several workers reach page-cache and disk bandwidth one thread cannot.
_READ_CHUNK_BYTES = 8 << 20
_DEFAULT_READ_WORKERS = 4
_CUDA_READ_WORKERS = 16
# Two 64 MiB staging buckets: one is read while the other's copies run.
_STAGE_BUCKET_BYTES = 64 << 20
_HAS_PREADV = hasattr(os, "preadv")


def read_raw(
    targets: Mapping[str, torch.Tensor], tensors: TensorMapping, files: RawFiles, stager: DeviceStager | None
) -> None:
    """Fill each target with the raw checkpoint tensor of its key, casting where the target's dtype differs.
    A contiguous host target of the stored dtype is read into directly, and a contiguous device one through
    ``stager``. Every other target receives a temporary of the stored dtype, read the same way on its device (on
    the host without a stager), and converts it there: a dtype cast or a strided layout. ``tensors`` locates each
    key in its file.
    """
    chunks: list[ReadChunk] = []
    converted: list[tuple[torch.Tensor, torch.Tensor]] = []  # (temporary read, target it converts into)
    with stager.session(files) if stager is not None else nullcontext():
        for key, target in targets.items():
            path, offset = tensors.sources[key], tensors.source_offsets[key]
            shape, dtype = tensors.source_metadata[key]
            if target.shape != shape:
                raise ValueError(
                    f"target {key!r} has shape {tuple(target.shape)}, the checkpoint stores {tuple(shape)}"
                )
            device = target.device if target.device.type != "cpu" and stager is not None else torch.device("cpu")
            read_into = target
            if target.dtype != dtype or not target.is_contiguous() or target.device != device:
                read_into = torch.empty(shape, dtype=dtype, device=device)
                converted.append((read_into, target))
            if read_into.device.type == "cpu":
                chunks.append(ReadChunk(path, offset, read_into.data_ptr(), read_into.nbytes))
            else:
                stager.put(path, offset, read_into)
    files.read(chunks)
    for temporary, target in converted:
        target.copy_(temporary)


@dataclass(frozen=True, slots=True)
class ReadPoolConfig:
    """How checkpoint reads are split across the shared read workers."""

    workers: int
    """Read workers; with one, reads run on the calling thread without a pool."""
    chunk_bytes: int = _READ_CHUNK_BYTES
    """Largest single positional read; smaller reads are batched up to this size per worker task."""

    def __post_init__(self) -> None:
        if self.workers < 1 or self.chunk_bytes < 1:
            raise ValueError("read workers and chunk bytes must be positive")

    @classmethod
    def from_environment(cls) -> ReadPoolConfig:
        """Read ``LTX_LOADER_READ_WORKERS``; without it use 16 workers on a CUDA host and 4 elsewhere."""
        value = os.environ.get(READ_WORKERS_ENV)
        if value is None:
            return cls(_CUDA_READ_WORKERS if torch.cuda.is_available() else _DEFAULT_READ_WORKERS)
        try:
            workers = int(value)
        except ValueError:
            workers = 0
        if workers < 1:
            raise ValueError(f"{READ_WORKERS_ENV} must be a positive integer, got {value!r}")
        return cls(workers)


class ReadChunk(NamedTuple):
    """One positional read of at most ``ReadPoolConfig.chunk_bytes`` file bytes into host memory.
    Read workers execute chunks; ``RawFiles.read`` splits a large one into several and batches small ones.
    """

    path: Path
    offset: int
    """Absolute byte offset in the file."""
    address: int
    """Host address the bytes are written to."""
    nbytes: int


class BucketCopy(NamedTuple):
    """One host-to-device copy queued from a staging bucket."""

    bucket_offset: int
    """Byte offset of the copied bytes in the bucket."""
    nbytes: int
    target: torch.Tensor
    """Contiguous byte view of the device storage the bytes are copied into."""


def _read_exact(handle: int | None, path: Path, address: int, offset: int, nbytes: int) -> None:
    """Read ``nbytes`` at ``offset`` of ``path`` straight into host memory at ``address``."""
    view = memoryview((ctypes.c_char * nbytes).from_address(address))
    done = 0
    if handle is not None:
        while done < nbytes:
            count = os.preadv(handle, [view[done:]], offset + done)
            if count <= 0:
                raise OSError(f"unexpected end of {path} at byte {offset + done}")
            done += count
        return
    # Without preadv, a private unbuffered handle per chunk keeps concurrent reads independent.
    with path.open("rb", buffering=0) as file:
        file.seek(offset)
        while done < nbytes:
            count = file.readinto(view[done:])
            if not count:
                raise OSError(f"unexpected end of {path} at byte {offset + done}")
            done += count


_POOL: ThreadPoolExecutor | None = None
_POOL_CONFIG: ReadPoolConfig | None = None
_POOL_LOCK = threading.Lock()


def _read_pool() -> tuple[ReadPoolConfig, ThreadPoolExecutor | None]:
    """Process-wide read workers, sized on first use; no pool when one worker reads on the caller."""
    global _POOL, _POOL_CONFIG  # noqa: PLW0603
    with _POOL_LOCK:
        if _POOL_CONFIG is None:
            _POOL_CONFIG = ReadPoolConfig.from_environment()
            if _POOL_CONFIG.workers > 1:
                _POOL = ThreadPoolExecutor(max_workers=_POOL_CONFIG.workers, thread_name_prefix="ltx-checkpoint-read")
        return _POOL_CONFIG, _POOL


class RawFiles:
    """Positional checkpoint reads into caller memory, without mapping the files.
    File pages stay in the page cache instead of the process. Large reads run on a few shared workers, which reach
    the page-cache or disk bandwidth one thread cannot.
    """

    def read(self, chunks: Sequence[ReadChunk]) -> None:
        """Execute reads, splitting large ones and batching small ones into worker tasks of ``chunk_bytes``."""
        config, pool = _read_pool()
        with self._lock:
            handles = {chunk.path: self._handle(chunk.path) for chunk in chunks}
            tasks: list[list[ReadChunk]] = [[]]
            batched = 0
            for chunk in chunks:
                for start in range(0, chunk.nbytes, config.chunk_bytes):
                    nbytes = min(config.chunk_bytes, chunk.nbytes - start)
                    if batched + nbytes > config.chunk_bytes and tasks[-1]:
                        tasks.append([])
                        batched = 0
                    tasks[-1].append(ReadChunk(chunk.path, chunk.offset + start, chunk.address + start, nbytes))
                    batched += nbytes

            def run(task: list[ReadChunk]) -> None:
                for piece in task:
                    _read_exact(handles[piece.path], piece.path, piece.address, piece.offset, piece.nbytes)

            if pool is None or len(tasks) == 1:
                for task in tasks:
                    run(task)
                return
            futures: list[Future[None]] = []
            try:
                futures.extend(pool.submit(run, task) for task in tasks)
            finally:
                # Reads write caller memory through raw addresses: all of them end before any error surfaces.
                wait(futures)
            for future in futures:
                future.result()

    def __init__(self) -> None:
        self._handles: dict[Path, int] = {}
        self._lock = threading.RLock()
        self._closed = False

    def release(self) -> None:
        """Close file descriptors while keeping the reader available for later reads."""
        with self._lock:
            for handle in self._handles.values():
                os.close(handle)
            self._handles.clear()

    def close(self) -> None:
        """Close file descriptors after any active read has completed."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self.release()

    def _handle(self, path: Path) -> int | None:
        """Return a descriptor shared by positional reads, or ``None`` where ``preadv`` is missing."""
        if self._closed:
            raise RuntimeError("raw checkpoint reader is closed")
        if not _HAS_PREADV:
            return None
        handle = self._handles.get(path)
        if handle is None:
            handle = self._handles[path] = os.open(path, os.O_RDONLY)
        return handle


class DeviceStager:
    """Carry checkpoint bytes to contiguous CUDA or MPS tensors through two reusable host buckets.
    On CUDA the buckets are pinned and a filled bucket's copies run on a side stream while the next bucket is read,
    so the bytes cross the bus once at pinned bandwidth. On MPS the buckets are ordinary host memory and each
    bucket's copies finish before it is refilled. One caller at a time holds a ``session`` and ``put``s tensors.
    """

    def __init__(self, device: torch.device, bucket_bytes: int = _STAGE_BUCKET_BYTES) -> None:
        if device.type not in {"cuda", "mps"}:
            raise ValueError(f"device staging needs a CUDA or MPS device, not {device}")
        self.device = device
        self.bucket_bytes = bucket_bytes
        cuda = device.type == "cuda"
        self._buckets = tuple(torch.empty(bucket_bytes, dtype=torch.uint8, pin_memory=cuda) for _ in range(2))
        self._stream = torch.cuda.Stream(device) if cuda else None
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._index = 0
        self._fill = 0
        self._reads: list[ReadChunk] = []
        self._copies: list[BucketCopy] = []
        self._files: RawFiles | None = None
        self._lock = threading.Lock()

    @contextmanager
    def session(self, files: RawFiles) -> Iterator[DeviceStager]:
        """Hold the stager for one caller's ``put``s; every queued copy has completed when the block exits."""
        with self._lock:
            self._files = files
            try:
                yield self
                self._flush(files)
            finally:
                self._files = None
                self._reads.clear()
                self._copies.clear()
                self._fill = 0
                if self._stream is not None:
                    self._stream.synchronize()
                    self._events = [None, None]

    def put(self, path: Path, offset: int, target: torch.Tensor) -> None:
        """Queue ``target.nbytes`` file bytes at ``offset`` of ``path``, unchanged, for a contiguous device tensor."""
        if self._files is None:
            raise RuntimeError("device stager puts need an open session")
        if target.device != self.device or not target.is_contiguous():
            raise ValueError(f"staged target for {path}@{offset} must be a contiguous tensor on {self.device}")
        target_bytes = target.reshape(-1).view(torch.uint8)
        done = 0
        while done < target.nbytes:
            if self._fill == self.bucket_bytes:
                self._flush(self._files)
            take = min(target.nbytes - done, self.bucket_bytes - self._fill)
            address = self._buckets[self._index].data_ptr() + self._fill
            self._reads.append(ReadChunk(path, offset + done, address, take))
            self._copies.append(BucketCopy(self._fill, take, target_bytes.narrow(0, done, take)))
            self._fill += take
            done += take

    def _flush(self, files: RawFiles) -> None:
        if not self._copies:
            return
        bucket = self._buckets[self._index]
        pending = self._events[self._index]
        if pending is not None:
            # This bucket may still feed the copies issued two flushes ago.
            pending.synchronize()
        files.read(self._reads)
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                for copy in self._copies:
                    copy.target.copy_(bucket.narrow(0, copy.bucket_offset, copy.nbytes), non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._stream)
            self._events[self._index] = event
        else:
            for copy in self._copies:
                copy.target.copy_(bucket.narrow(0, copy.bucket_offset, copy.nbytes))
            torch.mps.synchronize()
        self._reads = []
        self._copies = []
        self._fill = 0
        self._index ^= 1
