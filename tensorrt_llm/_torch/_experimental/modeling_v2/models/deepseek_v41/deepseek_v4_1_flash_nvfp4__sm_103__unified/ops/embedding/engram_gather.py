# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row gather + dequantization of an FP8 E4M3 table with one E8M0 scale per 32 channels.

The DeepSeek-V4.1 Engram lookup: each hash id names one row of a
``[rows, 256]`` float8_e4m3fn table with a ``[rows, 8]`` float8_e8m0fnu scale
table; the row is dequantized in fp32 and rounded once to bf16. Ids outside
this caller's row range ``[row_start, row_end)`` produce all-zero rows, which
is what a row-sharded table's summing all-reduce expects.

The kernel is a Triton kernel defined in this file (a port of SGLang's
``engram_gather``), not a ``torch.ops.trtllm`` op: no engine op reads FP8 rows
with per-32 E8M0 scales. It takes the two tables as raw addresses, so one
kernel serves a table in device memory and a table in pinned host memory; for
a pinned host tensor the wrapper passes the address returned by
``cudaHostGetDevicePointer`` and the kernel reads host memory directly (UVA
zero-copy) -- no staging copy.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Launch configuration: fixed, no autotune. ROWS table rows per program,
# NUM_WARPS warps per program.
_ROWS_PER_PROGRAM = 4
_NUM_WARPS = 4


@triton.jit
def _engram_gather_kernel(
    w_addr,
    s_addr,
    ids_ptr,
    out_ptr,
    n_rows,
    lookups,
    ids_row_stride,
    row_start,
    row_end,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    r = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)  # flat output row = t * lookups + k
    live = r < n_rows
    t = r // lookups
    k = r - t * lookups
    gid = tl.load(ids_ptr + t * ids_row_stride + k, mask=live, other=0).to(tl.int64)
    owned = live & (gid >= row_start) & (gid < row_end)
    local = tl.where(owned, gid - row_start, 0)

    w = w_addr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    s = s_addr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    c = tl.arange(0, DIM)
    vals = tl.load(w + local[:, None] * DIM + c[None, :], mask=owned[:, None], other=0.0).to(
        tl.float32
    )
    e = tl.load(
        s + local[:, None] * (DIM // BLOCK) + (c // BLOCK)[None, :], mask=owned[:, None], other=0
    ).to(tl.int32)
    # E8M0 byte e -> 2^(e-127) built from the fp32 exponent bits (exact, no exp2):
    # e == 0 is 2^-127, an fp32 subnormal (bits 0x00400000); e == 255 is NaN (0x7FC00000).
    bits = tl.where(e == 0, 0x00400000, e << 23)
    bits = bits | ((e == 255).to(tl.int32) << 22)
    scale = bits.to(tl.float32, bitcast=True)
    res = tl.where(owned[:, None], vals * scale, 0.0)
    tl.store(out_ptr + r[:, None] * DIM + c[None, :], res.to(tl.bfloat16), mask=live[:, None])


def _kernel_address(t: torch.Tensor, device: torch.device) -> int:
    """Address the kernel dereferences: the device pointer of a CUDA tensor, or the
    mapped device pointer (``cudaHostGetDevicePointer``) of a pinned host tensor."""
    if t.is_cuda:
        return t.data_ptr()
    from cuda.bindings import runtime as cudart

    with torch.cuda.device(device):
        err, dptr = cudart.cudaHostGetDevicePointer(t.data_ptr(), 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"cudaHostGetDevicePointer failed ({err}): the host table must be pinned (page-locked) memory"
        )
    return int(dptr)


#: DeepSeek-V4.1's table geometry: D channels per row, one E8M0 byte per B, K lookups per token.
_V41_DBK = dict(D=256, B=32, K=24)

#: The layer-14 table's rank-3 row_start under four-way row sharding, ceil(384016682 / 4) * 3.
_L14_RANK3_START = 288_012_513


class EngramGather(OpWrapper):
    """Gather and dequantize rows ``ids`` of an FP8 E4M3 table with per-32 E8M0 scales.

    The DeepSeek-V4.1 Engram lookup (``ParallelEngramEmbedding.forward`` of the
    checkpoint's ``inference/model.py``), one launch. For ``g = ids[t, k]``::

        out[t, k, c] = bf16(fp32(weight[g - row_start, c]) * 2^(scale[g - row_start, c // B] - 127))
                       if row_start <= g < row_end, else +0.0

    with ``D = weight.shape[1]`` and ``B = D // scale.shape[1]``, the product in
    fp32 rounded once to bf16. Inside the call: the row gather, the
    dequantization, the zero fill of ids outside ``[row_start, row_end)``. The
    caller owns the hash ids (per-layer offsets applied), the reduction across
    ranks that turns zero-filled row shards into the full lookup, and the
    ``wkv`` projection that consumes the rows.

    ``ids`` is ``[T, K]`` int64/int32 on the GPU (unit stride along ``K``, any
    row stride); ``weight`` / ``scale`` hold global rows ``[row_start, row_start
    + weight.shape[0])`` as ``[R, D]`` float8_e4m3fn and ``[R, D // 32]``
    float8_e8m0fnu (or uint8) bytes, each a CUDA tensor (device memory, or a
    CUDA view of host memory) or a pinned CPU tensor, whose mapped address
    ``cudaHostGetDevicePointer`` gives on every call (zero-copy, no staging).
    Returns ``[T, K, D]`` bf16: ``out`` when given (the bf16 values stored into
    its dtype), else a new tensor. ``T = 0`` returns without a launch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"every E4M3 x E8M0 byte pair, {res} table: NaN, inf, subnormals",
                spec=dict(kind="byte_pairs", residency=res, T=11, rows=256, **_V41_DBK),
            )
            for res in ("device", "host")
        )
        + tuple(
            Cell(
                why=f"device row shards T={t}: 4 ranks of 100003 rows, zero fill, sum",
                spec=dict(kind="device_shards", T=t, rows=100_003, world=4, **_V41_DBK),
            )
            for t in (1, 6, 32, 4096, 8192, 16384)
        )
        + tuple(
            Cell(
                why=f"pinned host shared table T={t}: whole 200000 rows, no device copy",
                spec=dict(kind="host_shared", T=t, rows=200_000, **_V41_DBK),
            )
            for t in (1, 32, 4096)
        )
        + tuple(
            Cell(
                why=f"pinned host row shards T={t}: offset views of one buffer, sum",
                spec=dict(kind="host_shards", T=t, rows=100_003, world=4, **_V41_DBK),
            )
            for t in (1, 4096)
        )
        + tuple(
            Cell(
                why=f"byte offsets past 2 GiB, {res}: 9e6-row shard at rank-3 start",
                spec=dict(
                    kind="offsets_2gib",
                    residency=res,
                    T=32,
                    rows=9_000_000,
                    row_start=_L14_RANK3_START,
                    **_V41_DBK,
                ),
            )
            for res in ("device", "host")
        )
        + (
            Cell(
                why="layouts: T=0, [:, layer, :] view, int32 ids, uint8 scale, out view",
                spec=dict(kind="layouts", T=37, rows=30_000, **_V41_DBK),
            ),
        )
    )

    note = """
    Metadata consumed: none; the call is stateless.

    DeepSeek-V4.1: the layer-1 and layer-14 tables have 384006168 and 384016682
    rows (98 GB of FP8 each), D = 256, B = 32, K = 24 lookups per token. Device,
    row-sharded (tep4 / tp4): each rank passes its own [ceil(rows / 4), 256]
    shard with row_start = rank * part, row_end = min(row_start + part, rows),
    the same ids on every rank, then a bf16 summing all-reduce; dep4 does the
    same over the all-gathered ids (T up to 4 x 16384) and reduce-scatters. Host
    (host_offload, the default): every member passes the whole node-shared
    table, row_start = 0, row_end = rows, over its own tokens (the [:, layer, :]
    view of the hash output, T = 1..16384); no reduction. The rank sum equals the
    unsharded gather bit for bit except that -0.0 becomes +0.0.

    Residency: the kernel takes the tables as raw addresses. A CUDA tensor
    (device memory, or the vmm backend's CUDA views of host-NUMA memory) passes
    data_ptr(); a pinned CPU tensor (the shm backend's views) passes the mapped
    pointer cudaHostGetDevicePointer returns, resolved on every call, legal
    during capture. A graph bakes the address in: a host table must stay alive
    and in place for as long as a graph that captured a call on it can replay.
    A CPU view of the vmm table's memory passes the pointer query and gathers
    wrong rows. Pageable host memory is accepted (nothing detects it) and not
    certified.

    Preconditions beyond is_valid: D a power of two; a device table on ids'
    device; any id value (out-of-range ids read nothing and give zeros).
    Arithmetic: an E4M3 value times a power of two is exact in fp32, so the
    result is bit-identical to the torch reference, including E8M0 byte 0
    (2^-127, subnormal products kept) and 255 (NaN), the E4M3 NaN codes, inf
    on overflow, and -0.0 from 0x80.

    Launch: fixed, 4 table rows per program, 4 warps, grid ceil(T * K / 4), no
    host synchronization. Triton specializes n_rows = 24 T on parity, and
    row_start, row_end and the ids' row stride on their own classes: warm up
    each configuration before capturing it.

    Measured on sm_103 beyond the cells, by the entry's test: every T in
    1..16384 (the tp [:, 1, :] view) and 1..65536 (the dep4 gathered view) as
    prefixes of one call; any device shard of up to 96004171 rows (the largest
    the target allocates) and the far end of every rank's shard; capture at the
    82 graph row counts (T = G and 6 G, 1..1536; dep4 at 4 x those) on four
    rank shards; the checkpoint's own two tables (202.76 GB) in the node-shared
    host copy under both backends, read whole, at every T to 16384 and every
    captured count, by one process, by four processes on four GPUs, and by a
    one-GPU process on each socket; real checkpoint rows reproducing the golden
    engram_value.

    Not certified: D other than 256, B other than 32, K other than 24; a device
    table on another GPU; pageable host tables; a host table above 202.76 GB;
    other host backings at the real size or per-rank host shards of the real
    tables; out dtypes other than bf16 (fp16 stores infs where bf16 exceeds
    its range, with no error); T above 65536 (above 16384 for the tp view and
    host_offload); capture at an unlisted row count; concurrent use from
    several streams.
    """

    def raw_call(
        self,
        ids: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        row_start: int,
        row_end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dim = weight.shape[1]
        t_len, lookups = ids.shape
        if out is None:
            out = torch.empty((t_len, lookups, dim), dtype=torch.bfloat16, device=ids.device)
        n_rows = t_len * lookups
        if n_rows == 0:
            return out
        _engram_gather_kernel[(triton.cdiv(n_rows, _ROWS_PER_PROGRAM),)](
            _kernel_address(weight, ids.device),
            _kernel_address(scale, ids.device),
            ids,
            out,
            n_rows,
            lookups,
            ids.stride(0),
            row_start,
            row_end,
            DIM=dim,
            BLOCK=dim // scale.shape[1],
            ROWS=_ROWS_PER_PROGRAM,
            num_warps=_NUM_WARPS,
        )
        return out

    def is_valid(
        self,
        ids: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        row_start: int,
        row_end: int,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        """The layouts and dtypes the kernel reads or writes as if they were the expected ones.

        Each was run through the launch without these checks on sm_103 and
        returned wrong values, or wrote outside ``out``, with no error: a
        transposed ``ids`` view (bit-exact 0.043), float32 ids (0.020), an E5M2
        table (0.015), fp32 scales (0.015), a column-slice weight (0.004) or
        scale (0.048), a scale with half the weight's rows (the rows past it took
        the parent buffer's bytes), a ``[R, 7]`` scale (0.57), a 100-row view
        under a 200-row range (row 150 of the parent buffer returned), a
        last-dim-slice ``out`` (21334 elements written into its other half) and
        an ``out`` of 3 of 7 token rows (24576 elements written past its end).
        CPU ``ids`` are not guarded: Triton's launcher refuses them loudly.
        """
        # Every guard below is a layout the kernel would otherwise read or write
        # as if it were the expected one, without an error (probed on sm_103).
        assert ids.dim() == 2 and ids.stride(1) == 1, (
            f"ids must be [T, K] with unit inner stride, got {ids.shape} {ids.stride()}"
        )
        assert ids.dtype in (torch.int64, torch.int32), (
            f"ids must be int64 or int32, got {ids.dtype}"
        )
        assert weight.dtype == torch.float8_e4m3fn, (
            f"weight must be float8_e4m3fn, got {weight.dtype}"
        )
        assert scale.dtype in (torch.float8_e8m0fnu, torch.uint8), (
            f"scale must be float8_e8m0fnu or uint8, got {scale.dtype}"
        )
        assert weight.dim() == 2 and weight.is_contiguous(), (
            f"weight must be a contiguous [R, D] tensor, got {weight.shape} {weight.stride()}"
        )
        assert scale.dim() == 2 and scale.is_contiguous(), (
            f"scale must be a contiguous [R, D // block] tensor, got {scale.shape} {scale.stride()}"
        )
        rows, dim = weight.shape
        assert scale.shape[0] == rows and dim % scale.shape[1] == 0, (
            f"scale {tuple(scale.shape)} does not match weight {tuple(weight.shape)}"
        )
        assert 0 <= row_end - row_start <= rows, (
            f"row range [{row_start}, {row_end}) exceeds the {rows} rows held"
        )
        t_len, lookups = ids.shape
        if out is not None:
            assert out.is_contiguous() and tuple(out.shape) == (t_len, lookups, dim), (
                f"out must be a contiguous [{t_len}, {lookups}, {dim}] tensor, got {tuple(out.shape)} {out.stride()}"
            )

    def reference(
        self,
        ids: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        row_start: int,
        row_end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``ParallelEngramEmbedding.forward`` on one shard, in torch ops in its order.

        Mask, local index, lookup, fp32 multiply by the per-``B`` scale, one bf16
        cast, zero the masked rows. The named rows are gathered where the table
        lives -- on the CPU for a CPU tensor, through the process's own mapping
        rather than the GPU's -- and only they are moved to ``ids``' device, so a
        host table of any size can be its own reference. ``out`` is ignored.
        """
        dev = ids.device
        ids64 = ids.long()
        mask = (ids64 < row_start) | (ids64 >= row_end)
        local = (ids64 - row_start).masked_fill(mask, 0)
        block = weight.shape[1] // scale.shape[1]
        w8 = weight.view(torch.uint8)[local.to(weight.device)].to(dev)
        s8 = scale.view(torch.uint8)[local.to(scale.device)].to(dev)
        values = w8.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, block))
        values = values * s8.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).masked_fill(mask.unsqueeze(-1), 0)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bit equality of the bf16 patterns, NaN positions equal whatever the payload.

        Every product of an E4M3 value and a power of two is exact in fp32
        (subnormals included, as the all-byte-pairs cell shows), and the single
        fp32 -> bf16 rounding is round-to-nearest-even on both sides, so any
        difference is a defect. The wrong variants the test runs through it --
        scale per 16 channels, out-of-shard rows not zeroed, E4M3 bytes read as
        E5M2, 32-bit byte offsets -- are each rejected.
        """
        assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16, (
            out.shape,
            ref.shape,
            out.dtype,
        )
        nan_a, nan_e = torch.isnan(out), torch.isnan(ref)
        assert torch.equal(nan_a, nan_e), (
            f"NaN positions differ at {(nan_a != nan_e).sum().item()} elements"
        )
        a, e = out.view(torch.int16)[~nan_a], ref.view(torch.int16)[~nan_e]
        bad = (a != e).sum().item()
        same = (out.view(torch.int16) == ref.view(torch.int16)) | (nan_a & nan_e)
        assert bad == 0, (
            f"{bad} of {a.numel()} elements differ in their bf16 bits "
            f"(mismatch fraction {1.0 - same.float().mean().item():.6f})"
        )


engram_gather = EngramGather()
