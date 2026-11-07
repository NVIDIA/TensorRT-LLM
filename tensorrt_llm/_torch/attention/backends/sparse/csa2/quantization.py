# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 cache encodings, including the quantized RoPE channels.

Rows contain packed values followed by scale bytes. Main KV uses E2M1/E4M3
with groups of 16; index Q/K use E2M1/UE8M0 with groups of 32. Neither format
uses the tensor-wide scaling factor of the generic NVFP4 weight quantizer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch

CacheFormat = Literal["main", "index", "swa"]


def row_bytes(head_dim: int, cache_format: CacheFormat) -> int:
    group = 16 if cache_format == "main" else 32
    if cache_format not in ("main", "index", "swa") or head_dim % group:
        raise ValueError("Invalid CSA2 cache format or head dimension")
    return (head_dim if cache_format == "swa" else head_dim // 2) + head_dim // group


def _fp4_levels(device: torch.device) -> torch.Tensor:
    codes = torch.arange(8, device=device)
    return torch.where(
        codes < 4,
        codes.float() * 0.5,
        torch.exp2(((codes - 2) // 2).float()) * (1 + 0.5 * (codes % 2)),
    )


def pack_rows(x: torch.Tensor, cache_format: CacheFormat) -> torch.Tensor:
    """Encode floating rows [..., head_dim] as uint8 [..., row_bytes]."""
    dim = x.shape[-1]
    row_bytes(dim, cache_format)
    group = 16 if cache_format == "main" else 32
    blocks = x.float().reshape(*x.shape[:-1], dim // group, group)
    maximum = blocks.abs().amax(-1)
    if cache_format == "main":
        scales = (maximum.clamp(min=6 * 2**-9) / 6).to(torch.float8_e4m3fn)
        scale_bytes = scales.view(torch.uint8)
        scales = scales.float()
    else:
        divisor = 448 if cache_format == "swa" else 6
        floor = 1e-4 if cache_format == "swa" else 6 * 2**-126
        exponents = torch.ceil(torch.log2(maximum.clamp(min=floor) / divisor))
        scale_bytes = (exponents + 127).to(torch.uint8)
        scales = torch.exp2(exponents)
    scaled = blocks / scales.unsqueeze(-1)
    if cache_format == "swa":
        values = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        values = values.reshape(*x.shape[:-1], dim)
    else:
        # E2M1 round-to-nearest-even, including midpoint ties. Preserve the
        # sign bit of zero, matching the native packed representation.
        levels = _fp4_levels(x.device)
        midpoints = (levels[:-1] + levels[1:]) * 0.5
        magnitude = scaled.abs().clamp(max=6).contiguous()
        codes = torch.bucketize(magnitude, midpoints)
        tie = magnitude == midpoints[codes.clamp(max=6)]
        codes += (tie & (codes % 2 == 1)).long()
        codes = (codes | (torch.signbit(scaled).long() << 3)).to(torch.uint8)
        codes = codes.reshape(*x.shape[:-1], dim)
        values = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return torch.cat((values, scale_bytes), dim=-1)


def unpack_rows(
    rows: torch.Tensor,
    head_dim: int,
    cache_format: CacheFormat,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode uint8 rows after gathering from a paged or contiguous pool."""
    if rows.dtype != torch.uint8 or rows.shape[-1] != row_bytes(head_dim, cache_format):
        raise ValueError("CSA2 packed cache has the wrong row shape or dtype")
    data_bytes = head_dim if cache_format == "swa" else head_dim // 2
    data = rows[..., :data_bytes].contiguous()
    scale_bytes = rows[..., data_bytes:].contiguous()
    group = 16 if cache_format == "main" else 32
    if cache_format == "main":
        scales = scale_bytes.view(torch.float8_e4m3fn).float()
    else:
        scales = torch.exp2(scale_bytes.float() - 127)
    if cache_format == "swa":
        values = data.view(torch.float8_e4m3fn).float()
    else:
        codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
        levels = _fp4_levels(rows.device)
        values = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    values = values.reshape(*rows.shape[:-1], head_dim // group, group)
    return (values * scales.unsqueeze(-1)).reshape(*rows.shape[:-1], head_dim).to(dtype)


def gather_rows(
    pool: torch.Tensor, slots: torch.Tensor, dim: int, cache_format: CacheFormat
) -> torch.Tensor:
    """Gather/dequantize rows; invalid physical slots yield zero BF16 rows."""
    if pool.ndim != 2 or pool.dtype != torch.uint8 or pool.shape[1] != row_bytes(dim, cache_format):
        raise ValueError("CSA2 gathered cache has the wrong row shape or dtype")
    if slots.dtype not in (torch.int32, torch.int64) or slots.device != pool.device:
        raise ValueError("CSA2 gather slots must be integer tensors on the pool device")
    if (
        pool.is_cuda
        and pool.stride(1) == 1
        and slots.ndim in (1, 2)
        and dim > 0
        and _fused_gather_supported(pool.device.index)
    ):
        from .kernel import gather_dequant_rows

        return gather_dequant_rows(pool, slots, dim, cache_format)
    valid = (slots >= 0) & (slots < pool.shape[0])
    if pool.shape[0] == 0:
        return torch.zeros((*slots.shape, dim), dtype=torch.bfloat16, device=pool.device)
    rows = pool[torch.where(valid, slots, 0).long()]
    values = unpack_rows(rows, dim, cache_format)
    return torch.where(valid.unsqueeze(-1), values, 0)


@lru_cache(maxsize=None)
def _fused_gather_supported(device_index: int) -> bool:
    from .kernel import dsl_available

    # Retain the reference route on other architectures until validated there.
    return dsl_available() and torch.cuda.get_device_capability(device_index)[0] == 10


@lru_cache(maxsize=None)
def _fused_store_supported(device_index: int) -> bool:
    from .kernel import dsl_available

    major, _ = torch.cuda.get_device_capability(device_index)
    # Exact-byte parity (including nonfinite values) is validated on SM100.
    return dsl_available() and major == 10


def apply_row_transform(values: torch.Tensor, transform) -> torch.Tensor:
    """PyTorch reference of the fused norm / RoPE prologue (BF16 rounding after each stage)."""
    if transform is None:
        return values
    x = values
    if transform.norm is not None:
        weight, eps = transform.norm
        v = x.float()
        x = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * weight.float()).to(
            torch.bfloat16
        )
    if transform.rope is not None:
        positions, cos_sin, rope_dim = transform.rope
        half, nope = rope_dim // 2, x.shape[-1] - rope_dim
        table = cos_sin.view(-1, rope_dim)[positions.long().reshape(-1)].to(x.device)
        cos, sin = table[:, :half], table[:, half:]
        rotating = x[:, nope:].float()
        even, odd = rotating[:, 0::2], rotating[:, 1::2]
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(1)
        x = torch.cat((x[:, :nope], rotated.to(torch.bfloat16)), -1)
    return x


def store_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    cache_format: CacheFormat,
    transform=None,
) -> None:
    """Publish quantized rows into a packed, possibly strided cache view.

    ``transform`` (a ``RowTransform``) normalizes / rotates the rows first;
    the CUDA path folds it into the quantize launch.
    """
    if values.ndim != 2 or slots.ndim != 1 or pool.ndim != 2:
        raise ValueError("CSA2 publication requires row matrices and one-dimensional slots")
    if slots.numel() != values.shape[0] or pool.shape[1] != row_bytes(
        values.shape[1], cache_format
    ):
        raise ValueError("CSA2 publication requires one slot per packed row")
    if _fused_rows(pool, slots, values):
        from .kernel import quantize_scatter_rows

        quantize_scatter_rows(pool, slots, values, cache_format, **_transform_kwargs(transform))
        return
    packed = pack_rows(apply_row_transform(values, transform), cache_format)
    if slots.is_cuda:
        from .kernel import scatter_packed_rows

        scatter_packed_rows(pool, slots, packed)
    else:
        valid = (slots >= 0) & (slots < pool.shape[0])
        pool.index_copy_(0, slots[valid].long(), packed[valid])


# Native paged index layout shared with the DSA/DSV4 indexer caches and read in place
# by the paged FP4 MQA-logits kernels: each 64-row page stores 64x64 packed E2M1 data
# bytes followed by 64x4 UE8M0 scale bytes.
INDEX_PAGE_ROWS = 64
INDEX_DATA_BYTES = 64
INDEX_SCALE_BYTES = 4
INDEX_PAGE_BYTES = INDEX_PAGE_ROWS * (INDEX_DATA_BYTES + INDEX_SCALE_BYTES)


def index_slot_offsets(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte offsets of the data and scale bytes of index rows in page-footer pages."""
    slots = slots.long()
    page, position = slots // INDEX_PAGE_ROWS, slots % INDEX_PAGE_ROWS
    base = page * INDEX_PAGE_BYTES
    return (
        base + position * INDEX_DATA_BYTES,
        base + INDEX_PAGE_ROWS * INDEX_DATA_BYTES + position * INDEX_SCALE_BYTES,
    )


def read_index_rows(pages: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Reference reader: ``[..., 68]`` packed rows (data then scale); invalid slots are zero."""
    flat = pages.reshape(-1)
    capacity = flat.numel() // INDEX_PAGE_BYTES * INDEX_PAGE_ROWS
    valid = (slots >= 0) & (slots < capacity)
    data_offsets, scale_offsets = index_slot_offsets(torch.where(valid, slots, 0))
    data = flat[data_offsets[..., None] + torch.arange(INDEX_DATA_BYTES, device=flat.device)]
    scales = flat[scale_offsets[..., None] + torch.arange(INDEX_SCALE_BYTES, device=flat.device)]
    rows = torch.cat((data, scales), dim=-1)
    return torch.where(valid[..., None], rows, torch.zeros_like(rows))


def write_packed_index_rows(pages: torch.Tensor, slots: torch.Tensor, packed: torch.Tensor) -> None:
    """Scatter already packed ``[n, 68]`` index rows; padding slots never write."""
    flat = pages.view(-1)
    if slots.is_cuda:
        # Device slots stay on the device: no host sync, graph-capturable.
        from .kernel import scatter_packed_index_rows

        scatter_packed_index_rows(
            flat, slots, packed, INDEX_PAGE_ROWS, INDEX_DATA_BYTES, INDEX_SCALE_BYTES
        )
        return
    capacity = flat.numel() // INDEX_PAGE_BYTES * INDEX_PAGE_ROWS
    valid = (slots >= 0) & (slots < capacity)
    if not bool(valid.any()):
        return
    slots, packed = slots[valid], packed[valid]
    data_offsets, scale_offsets = index_slot_offsets(slots)
    flat[data_offsets[:, None] + torch.arange(INDEX_DATA_BYTES, device=flat.device)] = packed[
        :, :INDEX_DATA_BYTES
    ]
    flat[scale_offsets[:, None] + torch.arange(INDEX_SCALE_BYTES, device=flat.device)] = packed[
        :, INDEX_DATA_BYTES:
    ]


def pack_index_queries_split(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``[count, heads, 128]`` index queries as ``(data int8 [count, heads, 64], scales uint8 [count, heads, 4])``.

    The fused quantizer writes both outputs directly, so no packed-row
    intermediate or ``contiguous()`` split is needed; other inputs use the
    reference packer.
    """
    if _fused_queries(query):
        from .kernel import quantize_index_queries

        data, scales, job_args = _query_outputs(query)
        if job_args is not None:
            with torch.cuda.device(query.device):
                quantize_index_queries(*job_args)
        return data, scales
    packed = pack_rows(query, "index")
    return packed[..., :64].contiguous().view(torch.int8), packed[..., 64:].contiguous()


def _query_outputs(query: torch.Tensor):
    """Packed index-query outputs and the ``(values, data, scales)`` kernel views (``None`` if empty)."""
    count, heads, dim = query.shape
    rows = count * heads
    data = torch.empty((count, heads, 64), dtype=torch.uint8, device=query.device)
    scales = torch.empty((count, heads, 4), dtype=torch.uint8, device=query.device)
    views = None
    if rows:
        views = (query.reshape(rows, dim).contiguous(), data.view(rows, 64), scales.view(rows, 4))
    return data.view(torch.int8), scales, views


def store_layer_rows(
    swa_pool: torch.Tensor,
    swa_slots: torch.Tensor,
    swa_values: torch.Tensor,
    swa_transform=None,
    *,
    main=None,
    index=None,
    index_q: torch.Tensor | None = None,
):
    """Publish one layer's cache rows with a single quantize launch when possible.

    ``main`` / ``index`` are ``(pool_or_pages, slots, values, transform)`` for a
    Full layer's compressed rows; ``index_q`` ``[count, heads, 128]`` queries are
    packed in the same launch and returned as ``(data int8, scales uint8)``.
    Inputs the fused kernel cannot take fall back to the separate stores.
    """
    fused = _fused_rows(swa_pool, swa_slots, swa_values)
    for rows in (main, index):
        if rows is not None:
            fused = fused and _fused_rows(rows[0], rows[1], rows[2])
    if index_q is not None:
        fused = fused and _fused_queries(index_q)
    if not fused:
        store_rows(swa_pool, swa_slots, swa_values, "swa", swa_transform)
        if main is not None:
            store_rows(main[0], main[1], main[2], "main", main[3])
        if index is not None:
            store_index_rows(*index)
        return None if index_q is None else pack_index_queries_split(index_q)
    from .kernel import footer_job, quantize_scatter_jobs, row_job, split_job

    jobs = [row_job(swa_pool, swa_slots, swa_values, "swa", **_transform_kwargs(swa_transform))]
    if main is not None:
        jobs.append(row_job(main[0], main[1], main[2], "main", **_transform_kwargs(main[3])))
    if index is not None:
        if index[2].shape[1] != 128:
            raise ValueError("CSA2 index publication requires [rows, 128] values")
        jobs.append(
            footer_job(index[0], index[1], index[2], INDEX_PAGE_ROWS, **_transform_kwargs(index[3]))
        )
    packed = None
    if index_q is not None:
        data, scales, views = _query_outputs(index_q)
        packed = (data, scales)
        if views is not None:
            jobs.append(split_job(*views))
    with torch.cuda.device(swa_values.device):
        quantize_scatter_jobs(jobs)
    return packed


def _fused_rows(pool: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> bool:
    return (
        values.is_cuda
        and values.dtype == torch.bfloat16
        and pool.device == values.device
        and slots.device == values.device
        and pool.dtype == torch.uint8
        and pool.stride(-1) == 1
        and values.stride(1) == 1
        and _fused_store_supported(values.device.index)
    )


def _fused_queries(query: torch.Tensor) -> bool:
    count, heads, dim = query.shape
    return (
        query.is_cuda
        and query.dtype == torch.bfloat16
        and dim == 128
        and count * heads * dim < 1 << 31
        and _fused_store_supported(query.device.index)
    )


def _transform_kwargs(transform) -> dict:
    return {
        "norm": None if transform is None else transform.norm,
        "rope": None if transform is None else transform.rope,
    }


def store_index_rows(
    pages: torch.Tensor, slots: torch.Tensor, values: torch.Tensor, transform=None
) -> None:
    """Publish BF16 128D index rows into native page-footer pages (optionally norm / RoPE first)."""
    if values.ndim != 2 or values.shape[1] != 128 or slots.ndim != 1:
        raise ValueError("CSA2 index publication requires [rows, 128] values and 1-D slots")
    if slots.numel() != values.shape[0] or pages.dtype != torch.uint8:
        raise ValueError("CSA2 index publication requires one slot per row and uint8 pages")
    if pages.numel() % INDEX_PAGE_BYTES:
        raise ValueError("CSA2 native index pages must be whole 64-row pages")
    if _fused_rows(pages, slots, values):
        from .kernel import quantize_scatter_index_pages

        quantize_scatter_index_pages(
            pages, slots, values, INDEX_PAGE_ROWS, **_transform_kwargs(transform)
        )
        return
    write_packed_index_rows(
        pages, slots, pack_rows(apply_row_transform(values, transform), "index")
    )
