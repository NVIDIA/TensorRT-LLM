# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation and bounded, byte-preserving reads of an NVFP4-expert safetensors checkpoint.

The index/header catalog and the two-buffer tile reader are the BF16 layout's
(``SafetensorsCatalog``, ``TileReader``): metadata only at construction, at most two pinned
staging buffers of ``TILE_BYTES``, never a whole tensor, shard or model on the host.

``validate`` checks the catalog against every stored tensor the layout needs before anything
is loaded: present, nothing else present, exact dtype and shape, and a byte span matching them.

``read_packed_into`` copies the rank's rows (and, for a column split, its byte columns) of a
stored 2-D byte tensor -- packed U8 codes or E4M3 block-scale bytes -- into device storage
unchanged. A column split reads whole rows tile by tile and keeps its columns on the device.
Values are never decoded, rescaled or cast here. ``read_scalars`` reads FP32 scalars with one
bounded read per contiguous run.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Mapping, Optional

import torch

from .layout import BF16, E4M3, F32, U8
from .loader_bf16 import (
    TILE_BYTES,
    CheckpointFormatError,
    SafetensorsCatalog,
    TileReader,
    read_slice_into,
)

__all__ = [
    "TILE_BYTES",
    "CheckpointFormatError",
    "SafetensorsCatalog",
    "TileReader",
    "check_block_scales",
    "read_packed_into",
    "read_scalars",
    "read_slice_into",
    "validate",
]

_ITEMSIZE = {U8: 1, E4M3: 1, BF16: 2, F32: 4}
#: Scalar runs read in one request when they are at most this many bytes apart end to end.
_SCALAR_RUN_BYTES = 1 << 20


class _Rows:
    """A stored tensor seen as ``rows`` rows of ``row_bytes`` bytes, for ``TileReader.tiles``."""

    __slots__ = ("name", "path", "begin", "end", "row_bytes")

    def __init__(self, name: str, path: str, begin: int, end: int, row_bytes: int):
        self.name, self.path, self.begin, self.end, self.row_bytes = (
            name,
            path,
            begin,
            end,
            row_bytes,
        )


def validate(catalog: SafetensorsCatalog, expected: Mapping[str, tuple], *, complete=True) -> None:
    """Raise CheckpointFormatError unless the stored tensors are exactly ``expected``.

    ``expected`` maps each stored name to ``(dtype, shape)`` (``layout.stored_entries``). With
    ``complete`` any other stored tensor is rejected.
    """
    missing = sorted(set(expected) - set(catalog.entries))
    if missing:
        raise CheckpointFormatError(f"{len(missing)} expected tensors are missing: {missing[:8]}")
    if complete:
        extra = sorted(set(catalog.entries) - set(expected))
        if extra:
            raise CheckpointFormatError(f"{len(extra)} unsupported tensors: {extra[:8]}")
    for name, (dtype, shape) in expected.items():
        entry = catalog.entries[name]
        if entry.dtype != dtype or tuple(entry.shape) != tuple(shape):
            raise CheckpointFormatError(
                f"{name} is stored as {entry.dtype}{list(entry.shape)}; expected {dtype}{list(shape)}"
            )
        if entry.end - entry.begin != math.prod(shape) * _ITEMSIZE[dtype]:
            raise CheckpointFormatError(f"{name}: byte span does not match {dtype}{list(shape)}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@torch.no_grad()
def read_packed_into(
    catalog: SafetensorsCatalog,
    reader: TileReader,
    name: str,
    rows: tuple,
    cols: Optional[tuple],
    dst: torch.Tensor,
    *,
    digest: Optional["hashlib._Hash"] = None,
) -> None:
    """Copy rows ``[rows[0], rows[1])`` (byte columns ``cols``, or all) of 2-D byte tensor ``name``.

    ``dst`` is uint8 ``[rows, cols]`` storage (any device), filled with the stored bytes
    unchanged. ``digest`` (optional) is updated with the source bytes read, in tile order.
    """
    entry = catalog.entries[name]
    if entry.dtype not in (U8, E4M3) or len(entry.shape) != 2:
        raise CheckpointFormatError(f"{name}: expected a 2-D U8/F8_E4M3 tensor, got {entry.dtype}")
    row_bytes = entry.shape[1]
    r0, r1 = rows
    c0, c1 = (0, row_bytes) if cols is None else cols
    if not (0 <= r0 < r1 <= entry.shape[0] and 0 <= c0 < c1 <= row_bytes):
        raise ValueError(f"{name}: slice rows {rows} cols {cols} outside {list(entry.shape)}")
    if dst.dtype != torch.uint8 or tuple(dst.shape) != (r1 - r0, c1 - c0):
        raise ValueError(
            f"{name}: destination {dst.dtype}{list(dst.shape)} != uint8{[r1 - r0, c1 - c0]}"
        )
    view = _Rows(entry.name, entry.path, entry.begin, entry.end, row_bytes)
    for t0, t1, raw in reader.tiles(view, r0, r1):
        tile = raw.to(dst.device, non_blocking=True).view(t1 - t0, row_bytes)
        dst[t0 - r0 : t1 - r0].copy_(tile[:, c0:c1])
        _sync(dst.device)
        if digest is not None:
            digest.update(f"{entry.name}:{t0}:{t1}|".encode())
            digest.update(memoryview(raw.numpy()))


def read_scalars(catalog: SafetensorsCatalog, reader: TileReader, names) -> dict:
    """FP32 scalar tensors ``names`` -> ``{name: float}``, one bounded read per contiguous run."""
    entries = sorted((catalog.entries[n] for n in names), key=lambda e: (e.path, e.begin))
    for e in entries:
        if e.dtype != F32 or tuple(e.shape) != () or e.end - e.begin != 4:
            raise CheckpointFormatError(f"{e.name}: expected an F32 scalar, got {e.dtype}{e.shape}")
    out = {}
    limit = min(_SCALAR_RUN_BYTES, reader.tile_bytes)
    i = 0
    while i < len(entries):
        j = i + 1
        while (
            j < len(entries)
            and entries[j].path == entries[i].path
            and entries[j].end - entries[i].begin <= limit
        ):
            j += 1
        run = entries[i:j]
        begin, end = run[0].begin, max(e.end for e in run)
        view = _Rows(run[0].name, run[0].path, begin, end, end - begin)
        for _, _, raw in reader.tiles(view, 0, 1):
            data = bytes(raw.numpy())
        for e in run:
            off = e.begin - begin
            out[e.name] = struct.unpack("<f", data[off : off + 4])[0]
        i = j
    return out


def check_block_scales(name: str, scale_bytes: torch.Tensor) -> None:
    """Raise CheckpointFormatError on a NaN or negative E4M3 block scale (zero is allowed)."""
    magnitude = scale_bytes & 0x7F
    bad_nan = magnitude == 0x7F
    bad_negative = ((scale_bytes & 0x80) != 0) & (magnitude != 0)
    if bool((bad_nan | bad_negative).any()):
        raise CheckpointFormatError(f"{name}: block scales hold NaN or negative E4M3 values")
