"""Read safetensors headers and tensor bytes straight from files within caller-owned reader sessions."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Protocol

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.raw import DeviceStager, RawFiles, ReadChunk, read_raw
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import StateDictMetadata, TensorMapping, TensorMetadata

ReadTensor = Callable[[Path, str], torch.Tensor]

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "C64": torch.complex64,
    "C128": torch.complex128,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class UnitSource(Protocol):
    """Checkpoint inputs for the loading layer: raw tensors by path and key.
    Loaders and schedulers read through a source and never open files. ``read`` returns one raw tensor, and
    ``read_into`` fills caller tensors by raw key, located through a ``TensorMapping``; ``materialize`` takes
    ``read``. A source's lifetime belongs to whoever opened it.
    """

    __slots__ = ()

    def read(self, path: Path, key: str) -> torch.Tensor:
        """Return one raw checkpoint tensor."""
        ...

    def read_into(self, tensors: TensorMapping, targets: Mapping[str, torch.Tensor]) -> None:
        """Fill each target with the raw checkpoint tensor of its key, casting where the target's dtype differs."""
        ...


# The safetensors format caps its JSON header at 100 MB.
_MAX_HEADER_BYTES = 100_000_000


class CheckpointReader(UnitSource):
    """Read checkpoint shards from file bytes within a session whose lifetime the caller owns.
    ``paths`` are the configured shards in caller order. Each shard's header is parsed once; ``read`` copies a
    tensor's bytes into new CPU memory. File descriptors open lazily on ``stack`` and close when the caller closes
    it; the reader must not be used after that. ``open`` provides one reader for a ``with`` block. As the loading
    layer's ``UnitSource`` the reader also fills tensors by raw key straight from file bytes; ``stagers`` supplies
    the host buckets for device tensors, and a reader without it reads them through host temporaries.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        stack: ExitStack,
        *,
        stagers: Callable[[torch.device], DeviceStager] | None = None,
    ) -> None:
        self.paths = tuple(paths)
        self._path_set = frozenset(self.paths)
        self._stack = stack
        self._stagers = stagers
        self._headers: dict[Path, tuple[dict[str, dict], int]] = {}
        self._files: RawFiles | None = None

    @classmethod
    @contextmanager
    def open(
        cls, paths: Sequence[Path], *, stagers: Callable[[torch.device], DeviceStager] | None = None
    ) -> Iterator[CheckpointReader]:
        """Yield a reader whose file descriptors close when the block exits."""
        with ExitStack() as stack:
            yield cls(paths, stack, stagers=stagers)

    def scan(self) -> tuple[dict[str, Path], StateDictMetadata]:
        """Read raw checkpoint-key files, shapes and dtypes without payloads.
        Visit shards in caller order and keys in sorted order within each shard.
        Reject duplicate keys and unsupported stored dtypes. Returned keys have
        not yet passed through state-dict operations.
        """
        sources: dict[str, Path] = {}
        metadata: dict[str, TensorMetadata] = {}
        for path in self.paths:
            header, _ = self._header(path)
            for key in sorted(key for key in header if key != "__metadata__"):
                if key in sources:
                    raise ValueError(f"checkpoint key {key!r} appears in both {sources[key]} and {path}")
                dtype_name = header[key]["dtype"]
                if dtype_name not in _SAFETENSORS_DTYPES:
                    raise ValueError(f"checkpoint tensor {key!r} has unsupported dtype {dtype_name!r}")
                sources[key] = path
                metadata[key] = (torch.Size(header[key]["shape"]), _SAFETENSORS_DTYPES[dtype_name])
        return sources, metadata

    def source_offsets(self) -> dict[str, int]:
        """Return each raw key's absolute payload offset within its shard."""
        offsets: dict[str, int] = {}
        for path in self.paths:
            header, base = self._header(path)
            for key, entry in header.items():
                if key != "__metadata__":
                    offsets[key] = base + entry["data_offsets"][0]
        return offsets

    def read_checkpoint_metadata(self) -> dict[str, object]:
        """Decode the first shard's ``__metadata__`` without reading payloads.
        Producers JSON-encode structured values such as ``config`` while leaving
        values such as licenses and version tags as plain strings.
        """
        if not self.paths:
            raise ValueError("a checkpoint needs at least one shard")
        header, _ = self._header(self.paths[0])
        raw = header.get("__metadata__") or {}
        parsed: dict[str, object] = {}
        for key, value in raw.items():
            try:
                parsed[key] = json.loads(value)
            except json.JSONDecodeError:
                parsed[key] = value
        return parsed

    def read(self, path: Path, key: str) -> torch.Tensor:
        """Read one raw checkpoint tensor into new CPU memory."""
        header, base = self._header(path)
        entry = header.get(key) if key != "__metadata__" else None
        if entry is None:
            raise KeyError(f"checkpoint shard {path} has no tensor {key!r}")
        dtype_name = entry["dtype"]
        if dtype_name not in _SAFETENSORS_DTYPES:
            raise ValueError(f"checkpoint tensor {key!r} has unsupported dtype {dtype_name!r}")
        tensor = torch.empty(entry["shape"], dtype=_SAFETENSORS_DTYPES[dtype_name])
        start, end = entry["data_offsets"]
        if end > start:
            self._raw_files().read([ReadChunk(path, base + start, tensor.data_ptr(), end - start)])
        return tensor

    def read_into(self, tensors: TensorMapping, targets: Mapping[str, torch.Tensor]) -> None:
        """Fill each target with the raw checkpoint tensor of its key, casting where the target's dtype differs.
        Reads go straight from file bytes into the targets, through the device's stager for device targets.
        """
        device = next((target.device for target in targets.values() if target.device.type != "cpu"), None)
        read_raw(targets, tensors, self._raw_files(), self._stager(device))

    def _header(self, path: Path) -> tuple[dict[str, dict], int]:
        if path not in self._path_set:
            raise ValueError(f"checkpoint reader has no shard {path}")
        header = self._headers.get(path)
        if header is None:
            header = self._headers[path] = _read_header(path)
        return header

    def _raw_files(self) -> RawFiles:
        if self._files is None:
            self._files = RawFiles()
            self._stack.callback(self._files.close)
        return self._files

    def _stager(self, device: torch.device | None) -> DeviceStager | None:
        if device is None or self._stagers is None or device.type not in {"cuda", "mps"}:
            return None
        return self._stagers(device)


def _read_header(path: Path) -> tuple[dict[str, dict], int]:
    """Decode and validate one shard's JSON header; return its entries and the payload's absolute start.
    As the safetensors format requires, the tensors' byte ranges tile the payload exactly, without overlaps or gaps.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > min(size - 8, _MAX_HEADER_BYTES):
            raise ValueError(f"invalid safetensors header length in {path}")
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header in {path} is not a JSON object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(value, str) for value in metadata.values()):
        raise ValueError(f"safetensors __metadata__ in {path} is not a map of strings")
    base = 8 + header_size
    ranges = sorted((*_entry_range(key, entry, path), key) for key, entry in header.items() if key != "__metadata__")
    offset = 0
    for start, end, key in ranges:
        if start != offset:
            raise ValueError(f"safetensors data range for {key!r} in {path} overlaps another tensor or leaves a gap")
        offset = end
    if offset != size - base:
        raise ValueError(f"safetensors tensors in {path} do not cover its payload")
    return header, base


def _entry_range(key: str, entry: object, path: Path) -> tuple[int, int]:
    """Validate one tensor entry of a safetensors header and return its payload byte range."""
    if not isinstance(entry, dict):
        raise ValueError(f"safetensors entry for {key!r} in {path} is not a JSON object")
    dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or not all(type(size) is int and size >= 0 for size in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(offset) is int for offset in offsets)
    ):
        raise ValueError(f"malformed safetensors entry for {key!r} in {path}")
    start, end = offsets
    if not 0 <= start <= end:
        raise ValueError(f"invalid safetensors data range for {key!r} in {path}")
    known = _SAFETENSORS_DTYPES.get(dtype)
    if known is not None and end - start != math.prod(shape) * known.itemsize:
        raise ValueError(f"safetensors entry for {key!r} in {path} has {end - start} bytes for its shape and dtype")
    return start, end
