# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXF8 K/V source rows of the GLM-5.2 non-absorbed context attention: one launch
per layer, the A operands (e4m3 values + e8m0 block scales) of the MXF8 K / V
expansion GEMMs.

The FP8 context route expands K and V from the 576-wide latent rows
``[cached prefix | new rows]`` of every request, packed back to back.  With the
expansion GEMMs in MXF8 (``self_attn_prefill_projections_mxf8``
``kv_b_expand_{k,v512}_mxf8``) the A operand is the latent row in MXF8:

* a cached row is its paged-pool bytes unchanged (e4m3 at the checkpoint's KV
  scale 1.0, the bytes the absorbed and decode paths read) with block scale
  2^0;
* a new row is the roped BF16 latent quantized per 32-element block with the
  act-quant rule of ``self_attn_prefill_act_quant_mxf8`` (``s = amax / 448``
  rounded up to a power of two, values ``e4m3(x / s)``, RN): block scaling keeps
  the precision of layers whose c_kv values sit below the e4m3 normal range at
  scale 1.0 (layer 3 of GLM-5.2: 88 % of them; MXF8 K/V expansion stage 0).

Outputs (one pass):

* ``a_k [T_k, 576]`` + ``sfa_k`` -- the K expansion's A (c_kv 512 | rope(k_pe) 64);
* ``a_v [T_k, 512]`` + ``sfa_v`` -- the V expansion's A (c_kv only; the block-scaled
  GEMM takes compact K-major operands, so it is a separate copy).

Scale-factor layout (``M32x4xrm_K4xrk``, the GEMM's): byte ``(m, kf)`` at
``(m % 32) * 16 + ((m // 32) % 4) * 4 + kf % 4 + (kf // 4) * 512 + (m // 128) * 512
* G`` with ``G`` k-groups (5 for K = 576, the last group half padding; 4 for K =
512).  Every byte the GEMM reads is written: the two padding columns of the K
operand's last group and the rows past ``T_k`` in the last 128-row block get
code 127 (2^0; their A values are TMA zero-fill).

Work rows (as ``self_attn_prefill_mha_prefix_gather_fp8_bf16``):

* cached work row ``w < C``: ``dst = idx[w]``, ``tok = idx[C + w]``,
  ``pos = idx[2C + w]``; pool row ``block_table[req_idx[tok], pos // tpb] *
  stride_factor + pos % tpb + layer_offset``;
* new work row ``t < T``: ``dst = idx[3C + t]`` (``new_dst_from_idx = 1``) or
  ``t`` (``0``: a step without a cached prefix, ``idx`` unused);
* padding rows ``T_k <= dst < padded(T_k)``: scale bytes only.

One 288-thread CTA covers four rows; each thread owns one 8-element column chunk
(4 consecutive threads = one 32-element scale block, reduced with two butterfly
shuffles).  Pointer arguments only: CUDA-graph safe.

Pattern: glm-mha-kv-gather-mxf8
Device: SM100+ (Blackwell)
Shape: runtime C cached + T new rows of 576 (512 latent + 64 rope)
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims

from ..common.cuda_stream import raw_stream as _raw_stream

LAT = 576  # 512 latent + 64 rope, one pool row in e4m3 bytes
KV_LORA = 512
_VEC = 8  # elements per thread chunk
_SF_VEC = 32  # elements per scale block
_CHUNKS = LAT // _VEC  # 72 threads per row
_V_CHUNKS = KV_LORA // _VEC  # the first 64 chunks also go to a_v
_ROWS_PER_CTA = 4
_THREADS = _CHUNKS * _ROWS_PER_CTA  # 288
_K_GROUPS = (LAT + 127) // 128  # 5 (blocks 18, 19 are padding)
_V_GROUPS = KV_LORA // 128  # 4
_UNIT_CODE = 127  # e8m0 2^0
_MIN_AMAX = 1.0e-12
_INV_E4M3_MAX = 1.0 / 448.0
_ALIGNMENT_BYTES = 16
assert _CHUNKS * _VEC == LAT and _CHUNKS % 4 == 0

CompiledKernel = Callable[..., object]
_COMPILED: dict[int, CompiledKernel] = {}


def padded_rows(rows: int) -> int:
    return (rows + 127) // 128 * 128


def sf_bytes_k(rows: int) -> int:
    """Bytes of ``sfa_k`` for ``rows`` K/V rows."""
    return padded_rows(rows) * _K_GROUPS * 4


def sf_bytes_v(rows: int) -> int:
    """Bytes of ``sfa_v`` for ``rows`` K/V rows."""
    return padded_rows(rows) * _V_GROUPS * 4


@cute.jit
def _pack_e4m3x2(value0, value1):
    """Two adjacent e4m3 bytes (value0 low) in the low 16 bits of an Int32 (RN, satfinite)."""
    word = prims.cvt_f32x2_to_f8x2(
        cutlass.Float32(value1),
        cutlass.Float32(value0),
        cutlass.Float8E4M3FN,
        rnd=prims.FPRoundingMode.RN,
        sat=prims.SaturationMode.SATFINITE,
    )
    return cutlass.Int32(word) & cutlass.Int32(0xFFFF)


@cute.jit
def _store_words(base, offset, w0, w1) -> None:
    for w in cutlass.range_constexpr(2):
        ptr = cute.recast_ptr((base + offset + w * 4).align(4), dtype=cutlass.Int32)
        if cutlass.const_expr(w == 0):
            ptr.store(w0)
        else:
            ptr.store(w1)


@cute.jit
def _sf_offset(m, kf, groups: cutlass.Constexpr[int]):
    return (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + kf % 4 + (kf // 4) * 512 + (m // 128) * (512 * groups)
    )


@cute.kernel
def glm52_self_attn_prefill_mha_kv_gather_mxf8_kernel(
    pool: cute.Tensor,
    idx: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    latent_new: cute.Tensor,
    a_k: cute.Tensor,
    a_v: cute.Tensor,
    sfa_k: cute.Tensor,
    sfa_v: cute.Tensor,
    num_cached: cutlass.Int32,
    num_new: cutlass.Int32,
    new_row_stride: cutlass.Int64,
    layer_offset: cutlass.Int64,
    stride_factor: cutlass.Int64,
    tokens_per_block: cutlass.Int64,
    new_dst_from_idx: cutlass.Int32,
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row_in_cta = tidx // _CHUNKS
    chunk = tidx - row_in_cta * _CHUNKS
    col = chunk * _VEC
    work = bidx * _ROWS_PER_CTA + row_in_cta
    rows = num_cached + num_new
    w0 = cutlass.Int32(0)
    w1 = cutlass.Int32(0)
    dst = cutlass.Int64(-1)
    code = cutlass.Int32(_UNIT_CODE)
    is_new = cutlass.Int32(0)
    f = cute.make_rmem_tensor((_VEC,), cutlass.Float32)
    for i in cutlass.range_constexpr(_VEC):
        f[i] = cutlass.Float32(0.0)
    if work < num_cached:
        cached = cutlass.Int64(num_cached)
        w = cutlass.Int64(work)
        dst = idx[w]
        tok = idx[cached + w]
        pos = idx[cached * 2 + w]
        block = cutlass.Int64(block_table[cutlass.Int64(req_idx[tok]), pos // tokens_per_block])
        pool_row = block * stride_factor + pos % tokens_per_block + layer_offset
        raw = cute.make_rmem_tensor((_VEC,), cutlass.Uint8)
        cute.autovec_copy(
            cute.make_tensor(
                pool.iterator + cute.assume(pool_row * LAT + col, _VEC),
                cute.make_layout((_VEC,)),
            ),
            raw,
        )
        words = cute.recast_tensor(raw, cutlass.Int32)
        w0 = words[0]
        w1 = words[1]
    else:
        t = work - num_cached
        if t < num_new:
            is_new = cutlass.Int32(1)
            dst = cutlass.Int64(t)
            if new_dst_from_idx != 0:
                dst = idx[cutlass.Int64(num_cached) * 3 + cutlass.Int64(t)]
            values = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            new_row = cute.make_tensor(
                latent_new.iterator + cute.assume(cutlass.Int64(t) * new_row_stride + col, _VEC),
                cute.make_layout((_VEC,)),
            )
            cute.autovec_copy(new_row, values)
            for i in cutlass.range_constexpr(_VEC):
                f[i] = values[i].to(cutlass.Float32)
        else:
            # padding row of the last 128-row block: scale bytes only (code 127)
            if work < (rows + 127) // 128 * 128:
                dst = cutlass.Int64(work)
    # Block amax over the 4 lanes of a 32-element block (every lane reaches the shuffles;
    # cached / padding lanes contribute 0 and do not use the result).
    local_max = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(_VEC):
        a = cute.math.absf(f[i])
        if a > local_max:
            local_max = a
    peer = cute.arch.shuffle_sync_bfly(local_max, 1)
    if peer > local_max:
        local_max = peer
    peer = cute.arch.shuffle_sync_bfly(local_max, 2)
    if peer > local_max:
        local_max = peer
    if is_new != 0:
        safe_amax = local_max
        if safe_amax < cutlass.Float32(_MIN_AMAX):
            safe_amax = cutlass.Float32(_MIN_AMAX)
        scale = safe_amax * cutlass.Float32(_INV_E4M3_MAX)
        scale_bits = scale.bitcast(cutlass.Int32)
        exp_bits = scale_bits & cutlass.Int32(0x7F800000)
        if (scale_bits & cutlass.Int32(0x007FFFFF)) != cutlass.Int32(0):
            exp_bits += cutlass.Int32(0x00800000)
        code = exp_bits >> cutlass.Int32(23)
        inv_scale = cutlass.Float32(1.0) / exp_bits.bitcast(cutlass.Float32)
        w0 = _pack_e4m3x2(f[0] * inv_scale, f[1] * inv_scale) | (
            _pack_e4m3x2(f[2] * inv_scale, f[3] * inv_scale) << 16
        )
        w1 = _pack_e4m3x2(f[4] * inv_scale, f[5] * inv_scale) | (
            _pack_e4m3x2(f[6] * inv_scale, f[7] * inv_scale) << 16
        )
    if dst >= 0:
        if dst < cutlass.Int64(rows):
            _store_words(a_k.iterator, dst * LAT + col, w0, w1)
            if chunk < _V_CHUNKS:
                _store_words(a_v.iterator, dst * KV_LORA + col, w0, w1)
        if chunk % 4 == 0:
            kf = chunk // 4
            (sfa_k.iterator + _sf_offset(dst, kf, _K_GROUPS)).store(cutlass.Uint8(code))
            if kf < KV_LORA // _SF_VEC:
                (sfa_v.iterator + _sf_offset(dst, kf, _V_GROUPS)).store(cutlass.Uint8(code))
            if kf == LAT // _SF_VEC - 1:
                # the K operand's last group: blocks 18, 19 are padding (A zero-filled)
                for pad in cutlass.range_constexpr(_K_GROUPS * 4 - LAT // _SF_VEC):
                    (sfa_k.iterator + _sf_offset(dst, kf + 1 + pad, _K_GROUPS)).store(
                        cutlass.Uint8(_UNIT_CODE)
                    )


@cute.jit
def _launch(
    pool: cute.Tensor,
    idx: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    latent_new: cute.Tensor,
    a_k: cute.Tensor,
    a_v: cute.Tensor,
    sfa_k: cute.Tensor,
    sfa_v: cute.Tensor,
    num_cached: cutlass.Int32,
    num_new: cutlass.Int32,
    new_row_stride: cutlass.Int64,
    layer_offset: cutlass.Int64,
    stride_factor: cutlass.Int64,
    tokens_per_block: cutlass.Int64,
    new_dst_from_idx: cutlass.Int32,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    glm52_self_attn_prefill_mha_kv_gather_mxf8_kernel(
        pool,
        idx,
        req_idx,
        block_table,
        latent_new,
        a_k,
        a_v,
        sfa_k,
        sfa_v,
        num_cached,
        num_new,
        new_row_stride,
        layer_offset,
        stride_factor,
        tokens_per_block,
        new_dst_from_idx,
    ).launch(grid=(num_ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def compile_gather(device_index: int = 0) -> CompiledKernel:
    """``fn(pool[P,576] u8, idx[N] i64, req_idx[N] i32, block_table[B,NB] i32,
    latent_new[T,576] bf16, a_k[T_k,576] u8, a_v[T_k,512] u8, sfa_k u8, sfa_v u8, C, T,
    new_row_stride, layer_offset, stride_factor, tokens_per_block, new_dst_from_idx,
    num_ctas, stream)``."""

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
            make_fake_tensor(
                cutlass.BFloat16,
                (cute.sym_int32(), LAT),
                stride=(cute.sym_int64(divisibility=_VEC), 1),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), LAT),
                stride_order=(1, 0),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8,
                (cute.sym_int32(), KV_LORA),
                stride_order=(1, 0),
                assumed_align=_ALIGNMENT_BYTES,
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_ALIGNMENT_BYTES
            ),
            make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int32(),), assumed_align=_ALIGNMENT_BYTES
            ),
            z32,
            z32,
            z64,
            z64,
            z64,
            z64,
            z32,
            z32,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[device_index]


def gather(
    fn: CompiledKernel,
    pool_rows: torch.Tensor,
    idx: torch.Tensor,
    num_cached: int,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    latent_new: torch.Tensor,
    a_k: torch.Tensor,
    a_v: torch.Tensor,
    sfa_k: torch.Tensor,
    sfa_v: torch.Tensor,
    layer_offset: int,
    stride_factor: int,
    tokens_per_block: int,
    new_dst_from_idx: bool,
) -> None:
    """Write this layer's MXF8 ``[cached | new]`` rows into ``a_k`` / ``sfa_k`` and
    ``a_v`` / ``sfa_v``.

    ``pool_rows``: the paged latent pool as ``[P, 576]`` uint8 (contiguous); ``idx``:
    ``[dst_c | tok_c | pos_c | dst_n]`` int64 with ``C = num_cached`` (read only for the
    cached rows and, with ``new_dst_from_idx``, the new rows' destinations; any int64
    tensor otherwise); ``req_idx``: int32 request slot per context token;
    ``block_table``: int32 ``[B, NB]`` (inner dim contiguous); ``latent_new``: ``[T,
    576]`` BF16 rows (pitch multiple of 8, 16-byte aligned); ``a_k`` / ``a_v``: uint8
    ``[T_k, 576]`` / ``[T_k, 512]`` contiguous with ``T_k = C + T``; ``sfa_k`` /
    ``sfa_v``: uint8 of at least ``sf_bytes_k(T_k)`` / ``sf_bytes_v(T_k)`` bytes.
    """

    dev = a_k.device
    num_new = latent_new.shape[0]
    rows = num_cached + num_new
    if (
        not (latent_new.is_cuda and latent_new.dtype == torch.bfloat16)
        or latent_new.ndim != 2
        or latent_new.shape[1] != LAT
        or latent_new.stride(1) != 1
        or latent_new.stride(0) % _VEC != 0
        or latent_new.data_ptr() % _ALIGNMENT_BYTES != 0
        or not (a_k.dtype == torch.uint8 and a_k.is_contiguous() and a_k.ndim == 2)
        or tuple(a_k.shape) != (rows, LAT)
        or not (a_v.dtype == torch.uint8 and a_v.is_contiguous() and a_v.ndim == 2)
        or tuple(a_v.shape) != (rows, KV_LORA)
        or a_k.data_ptr() % _ALIGNMENT_BYTES != 0
        or a_v.data_ptr() % _ALIGNMENT_BYTES != 0
        or not (sfa_k.dtype == torch.uint8 and sfa_k.is_contiguous())
        or not (sfa_v.dtype == torch.uint8 and sfa_v.is_contiguous())
        or sfa_k.numel() < sf_bytes_k(rows)
        or sfa_v.numel() < sf_bytes_v(rows)
        or not (pool_rows.is_cuda and pool_rows.dtype == torch.uint8)
        or pool_rows.ndim != 2
        or pool_rows.shape[1] != LAT
        or not pool_rows.is_contiguous()
        or not (idx.is_cuda and idx.dtype == torch.int64 and idx.is_contiguous())
        or ((num_cached > 0 or new_dst_from_idx) and idx.numel() != 3 * num_cached + num_new)
        or not (req_idx.is_cuda and req_idx.dtype == torch.int32 and req_idx.is_contiguous())
        or not (block_table.is_cuda and block_table.dtype == torch.int32)
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or any(
            t.device != dev
            for t in (
                pool_rows,
                idx,
                req_idx,
                block_table,
                latent_new,
                a_v,
                sfa_k,
                sfa_v,
            )
        )
        or num_cached < 0
        or rows == 0
        or tokens_per_block <= 0
    ):
        raise ValueError("mha kv gather mxf8: unsupported operands")
    fn(
        pool_rows,
        idx,
        req_idx,
        block_table,
        latent_new,
        a_k,
        a_v,
        sfa_k,
        sfa_v,
        num_cached,
        num_new,
        latent_new.stride(0),
        layer_offset,
        stride_factor,
        tokens_per_block,
        1 if new_dst_from_idx else 0,
        padded_rows(rows) // _ROWS_PER_CTA,
        _raw_stream(dev),
    )


# ----------------------------------------------------------------------------- reference
def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[R, 576]`` -> (e4m3 bytes ``[R, 576]``, e8m0 codes ``[R, 18]``) with the
    act-quant rule (1x32 blocks, power-of-two scale rounded up, RN)."""
    rows = x.shape[0]
    xf = x.float().reshape(rows, LAT // _SF_VEC, _SF_VEC)
    amax = xf.abs().amax(-1).clamp_min(_MIN_AMAX)
    scale = amax * torch.tensor(_INV_E4M3_MAX, dtype=torch.float32, device=x.device)
    bits = scale.view(torch.int32)
    exp_bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    code = (exp_bits >> 23).to(torch.int32)
    inv = 1.0 / exp_bits.view(torch.float32)
    q = (xf * inv[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, LAT)
    return q, code


def _sf_offsets(rows: int, kf: int, groups: int, device) -> torch.Tensor:
    m = torch.arange(rows, device=device)[:, None]
    f = torch.arange(kf, device=device)[None, :]
    return (
        (m % 32) * 16 + ((m // 32) % 4) * 4 + f % 4 + (f // 4) * 512 + (m // 128) * (512 * groups)
    )


def reference(
    pool_rows: torch.Tensor,
    idx: torch.Tensor,
    num_cached: int,
    req_idx: torch.Tensor,
    block_table: torch.Tensor,
    latent_new: torch.Tensor,
    a_k: torch.Tensor,
    a_v: torch.Tensor,
    sfa_k: torch.Tensor,
    sfa_v: torch.Tensor,
    layer_offset: int,
    stride_factor: int,
    tokens_per_block: int,
    new_dst_from_idx: bool,
) -> None:
    """Torch equivalent: cached rows = pool bytes with code 127, new rows block-quantized;
    padding scale bytes 127."""

    c, t = num_cached, latent_new.shape[0]
    rows = c + t
    codes = torch.full(
        (padded_rows(rows), _K_GROUPS * 4),
        _UNIT_CODE,
        dtype=torch.int32,
        device=a_k.device,
    )
    if c:
        dst_c, tok_c, pos_c = torch.split(idx[: 3 * c], [c, c, c], 0)
        src = block_table[req_idx[tok_c].to(torch.int64), pos_c // tokens_per_block].to(
            torch.int64
        ) * stride_factor + (pos_c % tokens_per_block)
        cached = pool_rows.narrow(0, layer_offset, pool_rows.shape[0] - layer_offset).index_select(
            0, src
        )
        a_k[dst_c] = cached
        a_v[dst_c] = cached[:, :KV_LORA]
    dst_n = (
        idx[3 * c :] if new_dst_from_idx else torch.arange(t, device=a_k.device, dtype=torch.int64)
    )
    q, code_n = quantize_rows(latent_new)
    a_k[dst_n] = q
    a_v[dst_n] = q[:, :KV_LORA]
    codes[dst_n, : LAT // _SF_VEC] = code_n
    rp = padded_rows(rows)
    off_k = _sf_offsets(rp, _K_GROUPS * 4, _K_GROUPS, a_k.device)
    sfa_k[off_k.flatten()] = codes.flatten().to(torch.uint8)
    off_v = _sf_offsets(rp, _V_GROUPS * 4, _V_GROUPS, a_k.device)
    sfa_v[off_v.flatten()] = codes[:, : _V_GROUPS * 4].flatten().to(torch.uint8)


def dequantize(a: torch.Tensor, sf: torch.Tensor, k: int, groups: int) -> torch.Tensor:
    """FP32 values of an MXF8 operand ``a [R, k]`` with blocked scales ``sf``."""
    rows = a.shape[0]
    kf = k // _SF_VEC
    off = _sf_offsets(rows, kf, groups, a.device)
    s = torch.exp2((sf[off].to(torch.int32) - 127).float())
    v = a.view(torch.float8_e4m3fn).float().reshape(rows, kf, _SF_VEC)
    return (v * s[..., None]).reshape(rows, k)


# ----------------------------------------------------------------------------- verify
