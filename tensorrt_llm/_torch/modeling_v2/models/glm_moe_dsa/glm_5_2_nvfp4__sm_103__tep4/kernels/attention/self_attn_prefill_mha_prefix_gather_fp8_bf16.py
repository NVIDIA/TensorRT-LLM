# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K/V source rows of the GLM-5.2 non-absorbed context attention over a cached
prefix: one launch per layer.

With KV block reuse a context request ``r`` brings ``c_r`` cached positions and
``n_r`` new rows.  The non-absorbed route expands K/V from the 576-wide latent
rows ``[cached prefix | new rows]`` of every request, packed back to back.  The
cached rows exist only in the paged FP8 latent pool (e4m3, KV scale 1.0, the
bytes the absorbed path reads); the new rows are the step's roped BF16 latent
view.  This kernel writes the packed ``[T_k, 576]`` BF16 rows in one pass:

* cached work row ``w < C``: ``dst = idx[w]``, ``tok = idx[C + w]`` (a context
  token of the request), ``pos = idx[2C + w]`` (the prefix position); the pool
  row is ``block_table[req_idx[tok], pos // tpb] * stride_factor + pos % tpb +
  layer_offset`` -- where the absorbed dense-causal FMHA reads it -- and its 576
  e4m3 bytes are widened exactly to BF16;
* new work row ``t < T``: ``dst = idx[3C + t]``, the BF16 row ``latent_new[t]``
  is copied.

``idx = [dst_c | tok_c | pos_c | dst_n]`` (int64) is built once per step on the
host.  The result is bitwise the torch sequence it replaces (``index_select``
of the pool rows, ``.view(float8_e4m3fn).to(bfloat16)``, two ``index_put_``):
e4m3 -> BF16 is exact, and NaN encodings become the canonical BF16 NaN as in
PyTorch.  Rows of ``lat`` that no work row names are left untouched.

One 288-thread CTA covers four work rows; each thread owns one 8-element
column chunk (an 8-byte e4m3 or 16-byte BF16 load, a 16-byte BF16 store), so a
row is moved with full-width coalesced accesses.  Pointer arguments only: the
launch is CUDA-graph safe.

Pattern: glm-mha-kv-prefix-gather
Device: SM100+ (Blackwell)
Shape: runtime C cached + T new rows of 576 (512 latent + 64 rope)
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

LAT = 576  # 512 latent + 64 rope, one pool row in e4m3 bytes
_VEC = 8  # elements per thread chunk
_CHUNKS = LAT // _VEC  # 72 threads per row
_ROWS_PER_CTA = 4
_THREADS = _CHUNKS * _ROWS_PER_CTA  # 288
_ALIGNMENT_BYTES = 16
assert _CHUNKS * _VEC == LAT

CompiledKernel = Callable[..., object]
_COMPILED: dict[int, CompiledKernel] = {}


@cute.jit
def _load_bf16x8(ptr, offset):
    values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, values)
    return values


@cute.jit
def _load_e4m3x8_as_bf16(ptr, offset):
    raw = cute.make_rmem_tensor((_VEC,), cutlass.Uint8)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, raw)
    values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
    # e4m3 -> fp32 -> bf16: both steps exact for every finite e4m3 value
    values.store(
        cute.recast_tensor(raw, cutlass.Float8E4M3FN)
        .load()
        .to(cutlass.Float32)
        .to(cutlass.BFloat16)
    )
    return values


@cute.jit
def _store_bf16x8(ptr, offset, values) -> None:
    dst = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(values, dst)


@cute.kernel
def glm52_self_attn_prefill_mha_kv_prefix_gather_kernel(
    pool: cute.Tensor,
    idx: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    latent_new: cute.Tensor,
    lat: cute.Tensor,
    num_cached: cutlass.Int32,
    num_new: cutlass.Int32,
    new_row_stride: cutlass.Int64,
    lat_row_stride: cutlass.Int64,
    layer_offset: cutlass.Int64,
    stride_factor: cutlass.Int64,
    tokens_per_block: cutlass.Int64,
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row_in_cta = tidx // _CHUNKS
    col = (tidx - row_in_cta * _CHUNKS) * _VEC
    work = bidx * _ROWS_PER_CTA + row_in_cta
    if work < num_cached:
        cached = cutlass.Int64(num_cached)
        w = cutlass.Int64(work)
        dst = idx[w]
        tok = idx[cached + w]
        pos = idx[cached * 2 + w]
        block = cutlass.Int64(block_table[cutlass.Int64(req_idx[tok]), pos // tokens_per_block])
        src = block * stride_factor + pos % tokens_per_block + layer_offset
        values = _load_e4m3x8_as_bf16(pool.iterator, src * LAT + col)
        _store_bf16x8(lat.iterator, dst * lat_row_stride + col, values)
    else:
        t = work - num_cached
        if t < num_new:
            dst = idx[cutlass.Int64(num_cached) * 3 + cutlass.Int64(t)]
            values = _load_bf16x8(latent_new.iterator, cutlass.Int64(t) * new_row_stride + col)
            _store_bf16x8(lat.iterator, dst * lat_row_stride + col, values)


@cute.jit
def _launch(
    pool: cute.Tensor,
    idx: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    latent_new: cute.Tensor,
    lat: cute.Tensor,
    num_cached: cutlass.Int32,
    num_new: cutlass.Int32,
    new_row_stride: cutlass.Int64,
    lat_row_stride: cutlass.Int64,
    layer_offset: cutlass.Int64,
    stride_factor: cutlass.Int64,
    tokens_per_block: cutlass.Int64,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    glm52_self_attn_prefill_mha_kv_prefix_gather_kernel(
        pool,
        idx,
        req_idx,
        block_table,
        latent_new,
        lat,
        num_cached,
        num_new,
        new_row_stride,
        lat_row_stride,
        layer_offset,
        stride_factor,
        tokens_per_block,
    ).launch(grid=(num_ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def _fake_rows_bf16():
    """[N, 576] BF16 with a runtime row pitch (multiple of 8), 16-byte-aligned."""

    return make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int32(), LAT),
        stride=(cute.sym_int64(divisibility=_VEC), 1),
        assumed_align=_ALIGNMENT_BYTES,
    )


def compile_gather(device_index: int = 0) -> CompiledKernel:
    """``fn(pool[P,576] u8, idx[3C+T] i64, req_idx[N] i32, block_table[B,NB] i32,
    latent_new[T,576] bf16, lat[T_k,576] bf16, C, T, new_row_stride, lat_row_stride,
    layer_offset, stride_factor, tokens_per_block, num_ctas, stream)``."""

    if device_index not in _COMPILED:
        z32, z64 = cutlass.Int32(0), cutlass.Int64(0)
        _COMPILED[device_index] = cute.compile(
            _launch,
            make_fake_compact_tensor(
                cutlass.Uint8,
                (cute.sym_int64(), LAT),
                stride_order=(1, 0),
                assumed_align=8,
            ),
            make_fake_compact_tensor(cutlass.Int64, (cute.sym_int32(),), assumed_align=8),
            make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=4),
            make_fake_tensor(
                cutlass.Int32,
                (cute.sym_int32(), cute.sym_int32()),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            ),
            _fake_rows_bf16(),
            _fake_rows_bf16(),
            z32,
            z32,
            z64,
            z64,
            z64,
            z64,
            z64,
            z32,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[device_index]


def _rows_ok(t: torch.Tensor, dtype: torch.dtype) -> bool:
    return (
        t.is_cuda
        and t.dtype == dtype
        and t.ndim == 2
        and t.shape[1] == LAT
        and t.stride(1) == 1
        and t.stride(0) % _VEC == 0
        and t.data_ptr() % _ALIGNMENT_BYTES == 0
    )


def gather(
    fn: CompiledKernel,
    pool_rows: torch.Tensor,
    idx: torch.Tensor,
    num_cached: int,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    latent_new: torch.Tensor,
    lat: torch.Tensor,
    layer_offset: int,
    stride_factor: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Write this layer's ``[cached | new]`` K/V source rows into ``lat`` and return it.

    ``pool_rows``: the paged latent pool as ``[P, 576]`` uint8 (contiguous);
    ``idx``: ``[dst_c | tok_c | pos_c | dst_n]`` int64 with ``C = num_cached``;
    ``req_idx``: int32 request slot per context token; ``block_table``: int32
    ``[B, NB]`` (inner dim contiguous); ``latent_new``: ``[T, 576]`` BF16 rows;
    ``lat``: ``[T_k, 576]`` BF16 destination rows.  Row pitches multiple of 8,
    rows 16-byte aligned.
    """

    dev = lat.device
    num_new = latent_new.shape[0]
    if (
        not _rows_ok(latent_new, torch.bfloat16)
        or not _rows_ok(lat, torch.bfloat16)
        or not (pool_rows.is_cuda and pool_rows.dtype == torch.uint8)
        or pool_rows.ndim != 2
        or pool_rows.shape[1] != LAT
        or not pool_rows.is_contiguous()
        or not (idx.is_cuda and idx.dtype == torch.int64 and idx.is_contiguous())
        or idx.numel() != 3 * num_cached + num_new
        or not (req_idx.is_cuda and req_idx.dtype == torch.int32 and req_idx.is_contiguous())
        or not (block_table.is_cuda and block_table.dtype == torch.int32)
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or any(t.device != dev for t in (pool_rows, idx, req_idx, block_table, latent_new))
        or num_cached < 0
        or num_cached + num_new == 0
        or tokens_per_block <= 0
    ):
        raise ValueError("mha kv prefix gather: unsupported operands")
    rows = num_cached + num_new
    fn(
        pool_rows,
        idx,
        req_idx,
        block_table,
        latent_new,
        lat,
        num_cached,
        num_new,
        latent_new.stride(0),
        lat.stride(0),
        layer_offset,
        stride_factor,
        tokens_per_block,
        (rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA,
        torch.cuda.current_stream(dev).cuda_stream,
    )
    return lat


def reference(
    pool_rows: torch.Tensor,
    idx: torch.Tensor,
    num_cached: int,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    latent_new: torch.Tensor,
    lat: torch.Tensor,
    layer_offset: int,
    stride_factor: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """The torch sequence this kernel replaces (served route before the kernel)."""

    c, t = num_cached, latent_new.shape[0]
    dst_c, tok_c, pos_c, dst_n = torch.split(idx, [c, c, c, t], 0)
    src = block_table[req_idx[tok_c].to(torch.int64), pos_c // tokens_per_block].to(
        torch.int64
    ) * stride_factor + (pos_c % tokens_per_block)
    rows = pool_rows.narrow(0, layer_offset, pool_rows.shape[0] - layer_offset).index_select(0, src)
    lat[dst_c] = rows.view(torch.float8_e4m3fn).to(torch.bfloat16)
    lat[dst_n] = latent_new
    return lat


# ----------------------------------------------------------------------------- verify
