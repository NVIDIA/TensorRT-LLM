# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged NVFP4 KV dequantize-gather to bf16 (packed E2M1 x one E4M3 scale per 16 x a global scale),
the bitwise twin of FlashInfer's `nvfp4_kv_dequantize_paged` in one launch, via the trtllm Triton op.

The op is `torch.ops.trtllm.nvfp4_kv_to_bf16`, registered in
`tensorrt_llm/_torch/custom_ops/nvfp4_kv_to_bf16.py`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.nvfp4_kv_to_bf16  # noqa: F401 — registers torch.ops.trtllm.nvfp4_kv_to_bf16
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_BLOCK = 16

_TABLE_DTYPES = ("int32", "int64")


class Nvfp4KvToBf16(OpWrapper):
    """Dequantize the NVFP4 pool rows a table names into contiguous bf16 rows, in one launch.

    For every request `b` and every position `t < min(seq_lens[b], L)`, with
    `page = block_tables[b, t // page_size]`::

        out[b, t, 0, i] = bf16(
            (E2M1[code_i] * e4m3(sf[i // 16])) * global_scale
        )  # only if 0 <= page < pages

    `kv_cache` `[pages, page_size, 1, W/2]` uint8 holds the packed codes (even element in the low
    nibble) and `kv_cache_sf` `[pages, page_size, 1, W/16]` the E4M3 scale bytes, read through any
    strides with a unit last-dim stride: the two halves of DeepSeek-V4.1's 288-byte row in place.
    Positions at or past `seq_lens[b]` (or `L`) and entries whose page id is negative or `>= pages`
    are left unwritten. The signature, operand views and results are those of FlashInfer's
    `nvfp4_kv_dequantize_paged` as the DeepSeek-V4.1 target calls it.

    Fusion boundary: the page lookup, the E2M1 decode, the E4M3 scale, the global scale and the
    bf16 cast are inside the call; the pool and its writes, the table and the remapping of indices
    for the attention that reads `out` are the caller's. Writes `out` in place and returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"per query R={r} {dt}: [R, 512] top-k table over pages of one row",
                spec=dict(kind="topk", R=r, table_dtype=dt),
            )
            for r in (1, 6, 32, 64, 192, 384, 1536, 8192)
            for dt in _TABLE_DTYPES
        )
        + tuple(
            Cell(
                why=f"per request page {ps} {dt}: lengths 0..8192, a hole, an id past",
                spec=dict(kind="request", page_size=ps, table_dtype=dt),
            )
            for ps in (64, 128)
            for dt in _TABLE_DTYPES
        )
        + (
            Cell(
                why="prefill 16384 entries at page 128: the whole pool",
                spec=dict(kind="prefill_16k"),
            ),
        )
        + tuple(
            Cell(
                why=f"compact r={ratio} B={b}: [B, 132*128/r, 1, 512], random lengths",
                spec=dict(kind="compact", ratio=ratio, B=b),
            )
            for ratio in (1, 2)
            for b in (1, 8, 32)
        )
        + tuple(
            Cell(
                why=f"engine pool {rows} rows {form}: ends of the pool named",
                spec=dict(kind="engine_pool", rows=rows, page_size=ps, form=form),
            )
            for rows, ps in ((453120, 64), (151040, 128))
            for form in ("per_request", "per_query")
        )
    )

    note = """
    Cells. W = 512 over pools of random bytes (every code byte, scale bytes 0x00..0x7E) read as the
    two strided halves of 288-byte rows, `global_scale` 1.0. Per query: a 16,384-row pool viewed as
    pages of one row, `[R, 512]` tables with -1 padding, ids one past and far past the pool, a row
    that ends 37 entries early. Per request: pages of 64 and 128, lengths 0, 1, page - 1, page + 1,
    700, 4097, 8192, a table hole and a page id past the pool; one 16,384-entry request; the
    target's compact views `[B, 132 * 128 / r, 1, 512]`; pools of 453,120 x 64 / 151,040 x 128 rows
    with both ends named.

    Exact for `global_scale` 1.0: `E2M1 x E4M3` has at most 6 significant bits, so the fp32 product
    and its bf16 cast are exact. Code 0x8 gives -0.0; a NaN scale byte gives NaN. Native rows written
    by `nvfp4_fake_quantize` read back bit-equal to its core output. Every certified call is also
    bitwise equal to FlashInfer's `nvfp4_kv_dequantize_paged` on the same operands, in one launch
    instead of two: measured on sm_103 1.4x (one 512-row gather) to 14.7x (256 requests of 768
    entries) faster.

    Metadata consumed: none. DeepSeek-V4.1's target uses two forms. Per request:
    `compress_block_tables[r][:B]`, `compressed_kv_lens_cuda[r][:B]` over the native views at page
    size 128 / r, into `comp_out[r][:B]`, the `[max_batch_size, cols * 128 / r, 1, 512]` view of one
    buffer. Per query: the views re-viewed with page size 1, the index op's `[T, 512]` int32 pool
    slots (-1 after each query's valid entries), `seq_lens` 512, into the `[T, 512, 1, 512]` view
    of a `[max_num_tokens * 512, 512]` buffer.

    Preconditions: every `t < min(seq_lens[b], L)` has `t // page_size < max_pages` (the table is
    read there, unchecked); `global_scale` a CUDA float32 `[1]`; `block_tables` int32/int64 with a
    unit last-dim stride; `seq_lens` int32; `out` contiguous bf16 `[B, L, 1, W]`, W a power of two.
    The op raises ValueError on a wrong dtype, shape, stride or device. Unwritten entries keep what
    `out` held: the target masks the -1 columns by count and never reads them.

    No host synchronization and no allocation: graph-capturable after one eager call of the same
    form (a first-ever call inside a capture is not certified). Triton specializes the kernel on its
    integer arguments (strides, page count, page size, table stride, L); measured, nothing compiles
    after the first eager call of a (form, pool view, out width) over every B 1..256, every length,
    every R 1..16,384 and every capture.

    Envelope, in the test's cache-manager cases: native views up to `[6480780, 64, 288]` /
    `[3245959, 128, 288]` (PP2 stage-alone managers at 0.5 x total memory), reads past byte offset
    2^36, the result independent of the view's page count; per query every R 1..16,384 and, under
    capture, R = G and 6G; per request at table widths 132, 164, 260, 516, 1028, 2052, 8196 with up
    to 256 requests (128 at width 8196), every length, captures at every graph batch G of 1..32,
    40..128 step 8, 192, 256 with next_n 1 and 6; live multi-request steps up to 256 requests.

    Not certified: W other than 512; `global_scale` other than 1.0 beyond 0.5, 2 and 1/3 (held
    only to the FlashInfer twin); views beyond the ones above; a per-query call of more than 16,384
    rows; the per-request form above 256 requests, above 128 at width 8196 (1 GiB per request), or
    at another table width; torch.compile.
    """

    def raw_call(
        self,
        kv_cache: torch.Tensor,
        kv_cache_sf: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        global_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.trtllm.nvfp4_kv_to_bf16(
            kv_cache, kv_cache_sf, block_tables, seq_lens, global_scale, out
        )
        return out

    def reference(
        self,
        kv_cache: torch.Tensor,
        kv_cache_sf: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        global_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """The whole of `out` after the call: dequantized rows where the op writes, `out`'s prior
        bits everywhere else.

        An E2M1 lookup table times the E4M3 scale in fp32 (exact), times `global_scale` (one fp32
        rounding, none for 1.0), cast to bf16 once. `out` is not mutated: the expected buffer is a
        copy of it.
        """
        dev = out.device
        num_requests, length = out.shape[0], out.shape[1]
        pages, page_size = kv_cache.shape[0], kv_cache.shape[1]
        t = torch.arange(length, device=dev)
        col = (t // page_size).clamp_max(max(block_tables.shape[1] - 1, 0))
        page = block_tables[:, col].long()
        lens = seq_lens.long().clamp_max(length)[:, None]
        written = (t[None, :] < lens) & (page >= 0) & (page < pages)
        page = page.clamp(0, pages - 1)[written]
        row = (t % page_size)[None, :].expand(num_requests, length)[written]
        codes = kv_cache[page, row, 0]
        scales = kv_cache_sf[page, row, 0].contiguous().view(torch.float8_e4m3fn).float()
        lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32, device=dev)
        values = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1).flatten(-2)
        values = (values * scales.repeat_interleave(_BLOCK, -1)) * global_scale.float()
        expected = out.clone()
        expected[:, :, 0][written] = values.to(out.dtype)
        return expected

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise on every element of `out`, written or not.

        With `global_scale` 1.0 the products are exact, so a differing element is a wrong row,
        scale or nibble, and a differing element outside the written set (the test pre-fills a NaN
        sentinel) is a write the op must not make. Controls (nibbles swapped, one scale per 32
        instead of 16) each change more than 30% of the values.
        """
        assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        bad = (out.view(torch.int16) != ref.view(torch.int16)).sum().item()
        assert bad == 0, f"{bad} of {out.numel()} elements differ"


nvfp4_kv_to_bf16 = Nvfp4KvToBf16()
