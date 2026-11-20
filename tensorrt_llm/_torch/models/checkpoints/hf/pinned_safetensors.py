# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read SafeTensors tensors with ``preadv`` into pinned host memory.

``PinnedSafetensorsFile`` provides the ``keys`` / ``get_tensor`` / ``get_slice``
subset of ``safe_open(path, framework="pt", device="cpu")``. Each tensor is read
with large ``preadv`` calls into its own block; blocks of 64 KiB or more come
from PyTorch's pinned host allocator when pinning is preferred. Every returned
tensor owns its memory.

``get_slice(key)`` returns a lazy ``PinnedSafetensorsSlice`` with ``get_shape``,
``get_dtype`` and ``__getitem__``. Indexing reads only the rows of the leading
dimension that the index covers (one block per call) and applies the remaining
indexing in memory. The slice is valid only while the file is open.

The generic Hugging Face checkpoint loader does not use this module by itself;
callers opt in per loader.
"""

import json
import operator
import os
import struct

import torch

from tensorrt_llm._utils import prefer_pinned

# SafeTensors dtype name -> torch attribute; resolved per tensor, so a name this
# torch lacks fails only the tensor that uses it.
_DTYPE_ATTRS = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "C64": "complex64",
}
# Below this a pinned block costs more than the pageable copy it saves.
_PIN_MIN_BYTES = 64 * 1024
# One pread call reads at most this much.
_READ_BYTES = 64 * 1024 * 1024
# The SafeTensors reference reader rejects larger headers.
_MAX_HEADER_BYTES = 100 * 1000 * 1000


def _torch_dtype(key: str, dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, _DTYPE_ATTRS.get(dtype_name, ""), None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"{key!r}: safetensors dtype {dtype_name} not supported by PinnedSafetensorsFile"
        )
    return dtype


def _as_int(value: object, what: str) -> int:
    """``operator.index(value)`` for integer-like values; bools and tensors are rejected."""
    if isinstance(value, (bool, torch.Tensor)) or not hasattr(type(value), "__index__"):
        raise TypeError(
            f"unsupported {what} {type(value).__name__} ({value!r}): expected an integer"
        )
    return operator.index(value)


def _normalize_index(index: object, ndim: int) -> tuple[int | slice, ...]:
    """Expand ``index`` to one ``int`` or ``slice`` per dimension.

    Accepts an int (anything with ``__index__``, e.g. a NumPy integer), a slice
    of ints, ``Ellipsis`` or a tuple of them; anything else (lists, tensors,
    ``None``, bools) raises ``TypeError``.
    """
    items = list(index) if isinstance(index, tuple) else [index]
    ellipsis_at = None
    for pos, item in enumerate(items):
        if item is Ellipsis:
            if ellipsis_at is not None:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_at = pos
        elif isinstance(item, slice):
            items[pos] = slice(
                *(
                    None if bound is None else _as_int(bound, f"slice {name}")
                    for name, bound in (
                        ("start", item.start),
                        ("stop", item.stop),
                        ("step", item.step),
                    )
                )
            )
        elif isinstance(item, (bool, torch.Tensor)) or not hasattr(type(item), "__index__"):
            raise TypeError(
                f"unsupported index type {type(item).__name__} ({item!r}): "
                "only int, slice and Ellipsis are supported"
            )
        else:
            items[pos] = operator.index(item)
    given = len(items) - (ellipsis_at is not None)
    if given > ndim:
        raise IndexError(f"too many indices for a tensor of {ndim} dimensions: {given}")
    fill = (slice(None),) * (ndim - given)
    if ellipsis_at is None:
        return (*items, *fill)
    return (*items[:ellipsis_at], *fill, *items[ellipsis_at + 1 :])


class PinnedSafetensorsSlice:
    """Lazy view of one tensor, like the ``PySafeSlice`` of ``safe_open(...).get_slice``.

    Valid only while the ``PinnedSafetensorsFile`` that made it is open; reads
    after ``__exit__`` raise ``ValueError``. ``get_shape`` and ``get_dtype`` come
    from the header and keep working.

    Indexing follows PyTorch semantics, which differ from ``safe_open`` in edge
    cases: a slice step is applied (``safe_open`` ignores it), out-of-range slice
    bounds clamp (``safe_open`` raises), and an int index on a 0-dim tensor
    raises ``IndexError``. For indexes both accept, results are bit-identical.
    """

    def __init__(self, file: "PinnedSafetensorsFile", key: str) -> None:
        self._file = file
        self._key = key
        self._dtype_name, self._shape, self._offset, self._nbytes = file._entries[key]

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def get_dtype(self) -> str:
        """The SafeTensors dtype name, e.g. ``"BF16"``."""
        return self._dtype_name

    def __getitem__(self, index: object) -> torch.Tensor:
        if self._file._fd < 0:
            raise ValueError(f"{self._file._path}: slice of {self._key!r} used after file close")
        index = _normalize_index(index, len(self._shape))
        dtype = _torch_dtype(self._key, self._dtype_name)
        if not self._shape:
            return self._file.get_tensor(self._key)
        rows = self._shape[0]
        lead, rest = index[0], index[1:]
        if isinstance(lead, int):
            first = lead + rows if lead < 0 else lead
            if not 0 <= first < rows:
                raise IndexError(f"index {lead} is out of bounds for dimension 0 with size {rows}")
            count, local = 1, 0
        else:
            start, stop, step = lead.indices(rows)
            if step < 0:
                raise ValueError("slice step must be positive")
            n = len(range(start, stop, step))
            # Read the covering row range; ``step`` is applied in memory.
            first, count = (start, (n - 1) * step + 1) if n else (0, 0)
            local = slice(None, None, step)
        row_bytes = self._nbytes // rows if rows else 0
        buf = self._file._read_block(self._key, self._offset + first * row_bytes, count * row_bytes)
        out = buf.view(dtype).reshape(count, *self._shape[1:])[(local, *rest)]
        if (
            out.is_contiguous()
            and out.storage_offset() == 0
            and out.untyped_storage().nbytes() == out.numel() * out.element_size()
        ):
            return out
        # A strided or partial view of the block: copy into a fresh block.
        pin = out.numel() * out.element_size() >= _PIN_MIN_BYTES and prefer_pinned()
        return torch.empty(tuple(out.shape), dtype=dtype, pin_memory=pin).copy_(out)


class PinnedSafetensorsFile:
    """Context manager with the ``keys`` / ``get_tensor`` / ``get_slice`` subset of ``safe_open``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd = -1
        self._entries: dict[str, tuple[str, list[int], int, int]] = {}

    def __enter__(self) -> "PinnedSafetensorsFile":
        self._fd = os.open(self._path, os.O_RDONLY)
        try:
            (header_len,) = struct.unpack("<Q", self._pread(8, 0))
            file_size = os.fstat(self._fd).st_size
            if header_len > min(_MAX_HEADER_BYTES, file_size - 8):
                raise ValueError(f"{self._path}: invalid SafeTensors header length {header_len}")
            header = json.loads(self._pread(header_len, 8))
            data_start = 8 + header_len
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                if not 0 <= begin <= end <= file_size - data_start:
                    raise ValueError(f"{self._path}: {key!r} data_offsets outside the file")
                dtype = getattr(torch, _DTYPE_ATTRS.get(info["dtype"], ""), None)
                if isinstance(dtype, torch.dtype):
                    numel = 1
                    for dim in info["shape"]:
                        numel *= dim
                    if end - begin != numel * dtype.itemsize:
                        raise ValueError(
                            f"{self._path}: {key!r} data_offsets length does not match "
                            "its dtype and shape"
                        )
                self._entries[key] = (info["dtype"], info["shape"], data_start + begin, end - begin)
            ranges = sorted((offset, offset + n) for _, _, offset, n in self._entries.values())
            for (_, prev_end), (begin, _) in zip(ranges, ranges[1:]):
                if begin < prev_end:
                    raise ValueError(f"{self._path}: overlapping tensor data_offsets")
        except BaseException:  # KeyboardInterrupt included: always release the fd
            os.close(self._fd)
            self._fd = -1
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        os.close(self._fd)
        self._fd = -1

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def get_tensor(self, key: str) -> torch.Tensor:
        dtype_name, shape, offset, nbytes = self._entries[key]
        dtype = _torch_dtype(key, dtype_name)
        return self._read_block(key, offset, nbytes).view(dtype).reshape(shape)

    def get_slice(self, key: str) -> PinnedSafetensorsSlice:
        return PinnedSafetensorsSlice(self, key)

    def _read_block(self, key: str, offset: int, nbytes: int) -> torch.Tensor:
        """``nbytes`` file bytes at ``offset`` in a new uint8 block (pinned when large)."""
        fd = self._fd  # one snapshot, so a concurrent close cannot switch files mid-read
        pin = nbytes >= _PIN_MIN_BYTES and prefer_pinned()
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        view = memoryview(buf.numpy()).cast("B")
        done = 0
        while done < nbytes:
            n = os.preadv(fd, [view[done : done + min(_READ_BYTES, nbytes - done)]], offset + done)
            if n <= 0:
                raise OSError(f"{self._path}: short read of {key!r} at byte {offset + done}")
            done += n
        return buf

    def _pread(self, n: int, offset: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = os.pread(self._fd, n - len(data), offset + len(data))
            if not chunk:
                raise OSError(f"{self._path}: truncated SafeTensors header")
            data += chunk
        return data
