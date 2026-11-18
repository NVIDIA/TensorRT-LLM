# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-cache-bypassing (O_DIRECT) reader for safetensors shard files.

``safetensors.safe_open(...).get_tensor`` reads through an mmap of the file, i.e. one
page fault at a time through the buffered page cache, which is slow on a parallel
filesystem client for a large checkpoint. Reading each tensor with a single page-aligned ``O_DIRECT``
``preadv`` issues large RPCs straight into the destination buffer, does not populate
(or have to later drop) the page cache, and returns the same bytes, so it is purely an
I/O change: dtypes, shapes, values and the order in which a model consumes them are
identical to ``get_tensor``.

Only the ``safetensors`` file layout is used (8-byte little-endian header length, JSON
header, raw little-endian data); tensors smaller than ``small_bytes`` (e.g. the 4-byte
NVFP4 ``weight_scale_2`` / ``input_scale`` scalars that sit in one contiguous block at
the start of the data region) are read through the normal buffered path, where
readahead already makes them cheap.
"""

import json
import mmap
import os
import struct
from typing import Dict, Optional, Tuple

import torch

_PAGE = mmap.PAGESIZE
# Debug: every N-th O_DIRECT tensor is re-read through the buffered path and compared bytewise
# (0 = off). A mismatch raises; the per-file tally is printed at close.
_VERIFY_EVERY = int(os.environ.get("TRTLLM_K3_DIRECT_READ_VERIFY", "0"))

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class DirectSafetensorsFile:
    """Read tensors of one safetensors file with ``O_DIRECT``.

    Thread-safe: ``get_tensor`` only uses ``os.preadv`` on shared descriptors, which is
    positional and releases the GIL, and allocates its own destination buffer.
    """

    def __init__(self, path: str, small_bytes: int = 256 * 1024):
        self.path = path
        self.small_bytes = small_bytes
        self._buffered_fd = os.open(path, os.O_RDONLY)
        try:
            (header_len,) = struct.unpack("<Q", os.pread(self._buffered_fd, 8, 0))
            header = json.loads(os.pread(self._buffered_fd, header_len, 8))
        except BaseException:
            os.close(self._buffered_fd)
            raise
        self._data_base = 8 + header_len
        header.pop("__metadata__", None)
        self._meta: Dict[str, Tuple[torch.dtype, Tuple[int, ...], int, int]] = {}
        for name, info in header.items():
            start, end = info["data_offsets"]
            self._meta[name] = (_DTYPES[info["dtype"]], tuple(info["shape"]), start, end)
        self._direct_fd: Optional[int] = None
        self._n_direct = 0
        self._n_verified = 0
        try:
            self._direct_fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError:
            # Filesystem without O_DIRECT: every tensor takes the buffered path.
            self._direct_fd = None

    def keys(self):
        return self._meta.keys()

    def close(self) -> None:
        if _VERIFY_EVERY and self._n_direct:
            print(f"K3_DIRECT_VERIFY {os.path.basename(self.path)} direct={self._n_direct} "
                  f"verified_identical={self._n_verified}", flush=True)
            self._n_direct = 0
        for fd in (self._direct_fd, self._buffered_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._direct_fd = None
        self._buffered_fd = -1

    def __enter__(self) -> "DirectSafetensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read_exact(self, fd: int, buf, offset: int, length: int) -> None:
        view = memoryview(buf)
        done = 0
        while done < length:
            n = os.preadv(fd, [view[done:length]], offset + done)
            if n <= 0:
                raise OSError(f"{self.path}: short read at {offset + done} (wanted {length - done} more bytes)")
            done += n

    def get_tensor(self, name: str) -> torch.Tensor:
        dtype, shape, start, end = self._meta[name]
        nbytes = end - start
        abs_start = self._data_base + start
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype)
        if self._direct_fd is None or nbytes < self.small_bytes:
            buf = mmap.mmap(-1, nbytes)
            self._read_exact(self._buffered_fd, buf, abs_start, nbytes)
            raw = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes)
            return raw.view(dtype).reshape(shape)
        aligned_start = abs_start - abs_start % _PAGE
        head = abs_start - aligned_start
        aligned_len = (head + nbytes + _PAGE - 1) // _PAGE * _PAGE
        # Anonymous mmap is page aligned, as O_DIRECT requires of the user buffer.
        buf = mmap.mmap(-1, aligned_len)
        view = memoryview(buf)
        done = 0
        want = head + nbytes
        while done < want:
            n = os.preadv(self._direct_fd, [view[done:aligned_len]], aligned_start + done)
            if n <= 0:
                raise OSError(f"{self.path}: short O_DIRECT read of {name} at {aligned_start + done}")
            done += n
        del view
        raw = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes, offset=head)
        self._n_direct += 1
        if _VERIFY_EVERY and (self._n_direct - 1) % _VERIFY_EVERY == 0:
            ref = mmap.mmap(-1, nbytes)
            self._read_exact(self._buffered_fd, ref, abs_start, nbytes)
            if bytes(memoryview(ref)) != bytes(memoryview(buf)[head:head + nbytes]):
                raise RuntimeError(f"{self.path}: O_DIRECT read of {name} differs from buffered read")
            self._n_verified += 1
        itemsize = torch.empty((), dtype=dtype).element_size()
        if head % itemsize:
            # Unaligned storage offset for a multi-byte dtype: copy. Never hit for the
            # byte-sized NVFP4 payloads and scales this reader exists for.
            raw = raw.clone()
        return raw.view(dtype).reshape(shape)
