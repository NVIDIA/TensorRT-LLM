# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata catalog and bounded streaming reads of a safetensors checkpoint directory.

``SafetensorsCatalog`` reads ``model.safetensors.index.json`` and every shard's header (an
8-byte little-endian length and a JSON table); it never reads a payload. ``validate`` checks
the catalog against the expected logical tensors before anything is loaded: every expected
tensor and every FP8 matrix's scale present, nothing else present, stored dtypes and shapes
exact, each entry's byte span inside its file and matching its dtype and shape.

``read_slice_into`` copies one rank-local slice of one tensor into its final device
storage through bounded byte-range tiles: a row-split or replicated tensor reads only its
rows; a column-split tensor reads whole rows, tile by tile, and keeps its columns on the
device. Tiles are at most ``TILE_BYTES`` and staged in at most two pinned host buffers, so
no tensor, shard or model is ever materialized on the host. An FP8 matrix's effective value
is ``bf16(float32(e4m3(q)) * float32(scale[row, 0]))``: the stored BF16 per-output-channel
scale multiplies its row, broadcast along the input columns, and the product is rounded to
BF16 once. BF16 and F32 tensors are copied byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from collections import OrderedDict
from typing import Mapping, Optional

import torch

from .layout_bf16 import BF16, F32, FP8, TensorSpec

INDEX_FILE = "model.safetensors.index.json"
TILE_BYTES = 16 << 20
_ITEMSIZE = {FP8: 1, BF16: 2, F32: 4}
_TORCH_DTYPE = {FP8: torch.float8_e4m3fn, BF16: torch.bfloat16, F32: torch.float32}
_MAX_HEADER = 256 << 20


class CheckpointFormatError(ValueError):
    """The checkpoint's index, headers or tensors are not what the layout expects."""


class Entry:
    """One stored tensor: file, dtype, shape and absolute byte span."""

    __slots__ = ("name", "path", "dtype", "shape", "begin", "end")

    def __init__(self, name, path, dtype, shape, begin, end):
        self.name, self.path, self.dtype, self.shape = name, path, dtype, tuple(shape)
        self.begin, self.end = begin, end

    @property
    def row_bytes(self) -> int:
        cols = 1
        for d in self.shape[1:]:
            cols *= d
        return cols * _ITEMSIZE[self.dtype]


def _read_header(path: str) -> tuple:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise CheckpointFormatError(f"{path}: truncated safetensors header length")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length <= _MAX_HEADER:
            raise CheckpointFormatError(f"{path}: implausible header length {length}")
        header = f.read(length)
        if len(header) != length:
            raise CheckpointFormatError(f"{path}: truncated safetensors header")
    try:
        table = json.loads(header)
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(f"{path}: malformed header JSON: {exc}") from None
    if not isinstance(table, dict):
        raise CheckpointFormatError(f"{path}: header is not a JSON object")
    return table, 8 + length


class SafetensorsCatalog:
    """Index and headers of a caller-supplied checkpoint directory; metadata only."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = os.fspath(checkpoint_dir)
        index_path = os.path.join(self.checkpoint_dir, INDEX_FILE)
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointFormatError(f"cannot read {INDEX_FILE}: {exc}") from None
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointFormatError(f"{INDEX_FILE} has no weight_map")
        self.entries: "OrderedDict[str, Entry]" = OrderedDict()
        for filename in sorted(set(weight_map.values())):
            if not isinstance(filename, str) or os.path.basename(filename) != filename:
                raise CheckpointFormatError(f"{INDEX_FILE} names an unsafe shard path {filename!r}")
            path = os.path.join(self.checkpoint_dir, filename)
            table, data_start = _read_header(path)
            size = os.path.getsize(path)
            spans = []
            for name, info in table.items():
                if name == "__metadata__":
                    continue
                try:
                    dtype, shape = info["dtype"], info["shape"]
                    begin, end = info["data_offsets"]
                except (KeyError, TypeError, ValueError):
                    raise CheckpointFormatError(
                        f"{filename}: malformed header entry {name!r}"
                    ) from None
                if weight_map.get(name) != filename:
                    raise CheckpointFormatError(
                        f"{name} is stored in {filename} but the index maps it to {weight_map.get(name)!r}"
                    )
                if not 0 <= begin <= end or data_start + end > size:
                    raise CheckpointFormatError(
                        f"{filename}: {name} spans [{begin}, {end}) outside the file"
                    )
                spans.append((begin, end, name))
                self.entries[name] = Entry(
                    name, path, dtype, shape, data_start + begin, data_start + end
                )
            spans.sort()
            for (b0, e0, n0), (b1, _, n1) in zip(spans, spans[1:]):
                if b1 < e0:
                    raise CheckpointFormatError(f"{filename}: {n0} and {n1} overlap")
        missing = sorted(set(weight_map) - set(self.entries))
        if missing:
            raise CheckpointFormatError(
                f"index entries absent from their shard headers: {missing[:8]}"
            )

    def validate(self, specs: Mapping[str, TensorSpec], *, complete: bool = True) -> None:
        """Raise CheckpointFormatError unless the stored tensors are exactly what ``specs`` needs.

        With ``complete`` (``specs`` describes the whole model) any stored tensor that is
        neither an expected logical tensor nor its FP8 scale is rejected; otherwise only the
        expected ones are checked.
        """
        expected = {}
        for spec in specs.values():
            expected[spec.name] = (spec.stored_dtype, tuple(spec.shape))
            if spec.scale_name is not None:
                expected[spec.scale_name] = (BF16, (spec.shape[0], 1))
        missing = sorted(set(expected) - set(self.entries))
        if missing:
            raise CheckpointFormatError(
                f"{len(missing)} expected tensors are missing: {missing[:8]}"
            )
        if complete:
            extra = sorted(set(self.entries) - set(expected))
            if extra:
                raise CheckpointFormatError(f"{len(extra)} unsupported tensors: {extra[:8]}")
        for name, (dtype, shape) in expected.items():
            entry = self.entries[name]
            if entry.dtype != dtype or entry.shape != shape:
                raise CheckpointFormatError(
                    f"{name} is stored as {entry.dtype}{list(entry.shape)}; expected {dtype}{list(shape)}"
                )
            numel = 1
            for d in shape:
                numel *= d
            if entry.end - entry.begin != numel * _ITEMSIZE[dtype]:
                raise CheckpointFormatError(
                    f"{name}: byte span does not match {dtype}{list(shape)}"
                )


class TileReader:
    """Bounded byte-range reads: at most two pinned staging buffers of ``tile_bytes``.

    ``tiles(entry, rows)`` yields ``(row_start, row_stop, host_view)`` for consecutive row
    ranges; the next tile is read on a background thread while the caller consumes the
    current one, and a buffer is refilled only after the caller's device copy of it
    completed. Counts the bytes read and the peak staging footprint.
    """

    def __init__(self, tile_bytes: int = TILE_BYTES, pin: bool = True):
        self.tile_bytes = tile_bytes
        self.buffers = [
            torch.empty(tile_bytes, dtype=torch.uint8, pin_memory=pin) for _ in range(2)
        ]
        self.bytes_read = 0
        self.reads = 0
        self._fds: "OrderedDict[str, int]" = OrderedDict()
        self._lock = threading.Lock()

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self.buffers = []

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            if len(self._fds) >= 4:
                _, old = self._fds.popitem(last=False)
                os.close(old)
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def _read(self, buf: torch.Tensor, fd: int, offset: int, length: int) -> None:
        view = memoryview(buf.numpy())[:length]
        got = 0
        while got < length:
            n = os.preadv(fd, [view[got:]], offset + got)
            if n <= 0:
                raise OSError(f"short read at offset {offset + got}")
            got += n
        with self._lock:
            self.bytes_read += length
            self.reads += 1

    def tiles(self, entry: Entry, row_start: int, row_stop: int):
        row_bytes = entry.row_bytes
        if row_bytes > self.tile_bytes:
            raise CheckpointFormatError(f"{entry.name}: one row ({row_bytes} B) exceeds the tile")
        rows_per_tile = max(1, self.tile_bytes // row_bytes)
        fd = self._fd(entry.path)
        plan = [
            (r, min(r + rows_per_tile, row_stop)) for r in range(row_start, row_stop, rows_per_tile)
        ]
        if not plan:
            return
        pending = None
        try:
            for i, (r0, r1) in enumerate(plan):
                buf = self.buffers[i % 2]
                length = (r1 - r0) * row_bytes
                if pending is None:
                    self._read(buf, fd, entry.begin + r0 * row_bytes, length)
                else:
                    pending.join()
                    if pending.error is not None:
                        raise pending.error
                    pending = None
                # The other buffer held tile i - 1, whose device copy the caller completed
                # before asking for tile i: read tile i + 1 into it while tile i is consumed.
                if i + 1 < len(plan):
                    n0, n1 = plan[i + 1]
                    pending = _Prefetch(
                        self,
                        self.buffers[(i + 1) % 2],
                        fd,
                        entry.begin + n0 * row_bytes,
                        (n1 - n0) * row_bytes,
                    )
                    pending.start()
                yield r0, r1, buf[:length]
        finally:
            if pending is not None:
                pending.join()


class _Prefetch(threading.Thread):
    def __init__(self, reader, buf, fd, offset, length):
        super().__init__(daemon=True)
        self.args = (buf, fd, offset, length)
        self.reader = reader
        self.error = None

    def run(self):
        try:
            self.reader._read(*self.args)
        except OSError as exc:  # surfaced to the consumer by tiles()
            self.error = exc


def _decode_tile(raw: torch.Tensor, entry: Entry, rows: int, device) -> torch.Tensor:
    """Host tile bytes -> device tensor ``[rows, ...]`` in the stored dtype."""
    dev = raw.to(device, non_blocking=True)
    return dev.view(_TORCH_DTYPE[entry.dtype]).view(rows, *entry.shape[1:])


@torch.no_grad()
def read_slice_into(
    catalog: SafetensorsCatalog,
    reader: TileReader,
    spec: TensorSpec,
    rank: int,
    dst: torch.Tensor,
    *,
    digest: Optional["hashlib._Hash"] = None,
) -> None:
    """Copy rank ``rank``'s effective slice of ``spec`` into ``dst`` (its final device storage).

    ``dst`` must have the rank-local shape and the effective dtype. ``digest`` (optional) is
    updated with the source bytes actually read, in tile order, with each tile's row range.
    """
    entry = catalog.entries[spec.name]
    local = spec.local_shape(rank)
    want = torch.bfloat16 if spec.stored_dtype in (FP8, BF16) else torch.float32
    if tuple(dst.shape) != local or dst.dtype != want:
        raise ValueError(
            f"{spec.name}: destination {dst.dtype}{list(dst.shape)} != {want}{list(local)}"
        )
    interval = spec.interval(rank)
    rows = spec.shape[0]
    row_start, row_stop, col = 0, rows, None
    if interval is not None:
        dim, start, stop = interval
        if dim == 0:
            row_start, row_stop = start, stop
        else:
            col = (start, stop)
    scale = None
    if spec.stored_dtype == FP8:
        scale_entry = catalog.entries[spec.scale_name]
        scale = torch.empty(row_stop - row_start, 1, dtype=torch.bfloat16, device=dst.device)
        for r0, r1, raw in reader.tiles(scale_entry, row_start, row_stop):
            scale[r0 - row_start : r1 - row_start].copy_(
                _decode_tile(raw, scale_entry, r1 - r0, dst.device)
            )
            torch.cuda.current_stream(dst.device).synchronize()
            if digest is not None:
                digest.update(f"{scale_entry.name}:{r0}:{r1}|".encode())
                digest.update(memoryview(raw.numpy()))
    for r0, r1, raw in reader.tiles(entry, row_start, row_stop):
        tile = _decode_tile(raw, entry, r1 - r0, dst.device)
        if col is not None:
            tile = tile[:, col[0] : col[1]]
        out = dst[r0 - row_start : r1 - row_start]
        if spec.stored_dtype == FP8:
            s = scale[r0 - row_start : r1 - row_start].float()
            out.copy_((tile.float() * s).to(torch.bfloat16))
        else:
            out.copy_(tile)
        torch.cuda.current_stream(dst.device).synchronize()
        if digest is not None:
            digest.update(f"{entry.name}:{r0}:{r1}|".encode())
            digest.update(memoryview(raw.numpy()))
    del scale
