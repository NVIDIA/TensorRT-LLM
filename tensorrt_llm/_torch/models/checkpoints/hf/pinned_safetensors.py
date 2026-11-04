# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read SafeTensors tensors with ``preadv`` into pinned host memory.

``PinnedSafetensorsFile`` provides the ``keys`` / ``get_tensor`` subset of
``safe_open(path, framework="pt", device="cpu")``. Each tensor is read with
large ``preadv`` calls into its own block; blocks of 64 KiB or more come from
PyTorch's pinned host allocator when pinning is preferred. Every returned tensor
owns its memory.
"""

import json
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


class PinnedSafetensorsFile:
    """Context manager with the ``keys`` / ``get_tensor`` subset of ``safe_open``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd = -1
        self._entries: dict[str, tuple[str, list[int], int, int]] = {}

    def __enter__(self) -> "PinnedSafetensorsFile":
        self._fd = os.open(self._path, os.O_RDONLY)
        try:
            (header_len,) = struct.unpack("<Q", self._pread(8, 0))
            header = json.loads(self._pread(header_len, 8))
            data_start = 8 + header_len
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                self._entries[key] = (info["dtype"], info["shape"], data_start + begin, end - begin)
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
        dtype = getattr(torch, _DTYPE_ATTRS.get(dtype_name, ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"{key!r}: safetensors dtype {dtype_name} not supported by PinnedSafetensorsFile"
            )
        pin = nbytes >= _PIN_MIN_BYTES and prefer_pinned()
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        view = memoryview(buf.numpy()).cast("B")
        done = 0
        while done < nbytes:
            n = os.preadv(
                self._fd, [view[done : done + min(_READ_BYTES, nbytes - done)]], offset + done
            )
            if n <= 0:
                raise OSError(f"{self._path}: short read of {key!r} at byte {offset + done}")
            done += n
        return buf.view(dtype).reshape(shape)

    def _pread(self, n: int, offset: int) -> bytes:
        data = os.pread(self._fd, n, offset)
        if len(data) != n:
            raise OSError(f"{self._path}: truncated SafeTensors header")
        return data
