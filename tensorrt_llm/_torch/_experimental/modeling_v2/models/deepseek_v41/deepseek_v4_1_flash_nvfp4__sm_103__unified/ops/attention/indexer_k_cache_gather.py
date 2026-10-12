# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather index-K rows out of a paged indexer-K pool into contiguous tensors, via the trtllm op.

The op is `torch.ops.trtllm.indexer_k_cache_gather_op`, registered from C++
(`cpp/tensorrt_llm/thop/IndexerKCacheGatherOp.cpp`, kernel in
`cpp/tensorrt_llm/kernels/indexerKCacheGather.cu`).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_SCALE_BYTES = 4  # one 4-byte word of four UE8M0 exponents per key

#: One request's whole index K, gathered at once: (block_kv, S).
_ROUNDTRIP = (
    (64, 1),
    (64, 5),
    (64, 4096),
    (128, 1),
    (128, 333),
    (128, 8192),
    (128, 32768),
    (256, 3000),
)

#: A packed prefill batch and the chunk loop's runs over its slot mappings, as
#: (k_token_start, num_tokens): each request on its own, the whole batch, a
#: mid-request run, single rows at a request's end and at the batch's end, and a
#: 4000-row run at an offset.
_PACKED_LENS = [700, 1, 2500, 37, 4096]
_PACKED_CHUNKS = [
    (0, 700),
    (700, 1),
    (701, 2500),
    (3201, 37),
    (3238, 4096),
    (0, 7334),
    (350, 1000),
    (3200, 1),
    (7333, 1),
    (3255, 4000),
]


class IndexerKCacheGather(OpWrapper):
    """Copy rows `[k_token_start, k_token_start + num_tokens)` of a paged index-K pool into two
    fresh contiguous tensors.

    One call, one kernel launch, no arithmetic. For output row `i`, with
    `r = k_token_start + i`:

        k[i, :]        = k_cache.flat[slot_mapping_fp8[r]   : + head_dim]
        k_scale[i, 0]  = k_cache.flat[slot_mapping_scale[r] : + 4]   (as one 4-byte word)

    where `.flat` is `k_cache`'s logical `[pages, block_kv, 1, 68]` shape
    unravelled innermost-first and addressed through its strides (a strided
    owner view is read correctly). A row whose data or scale slot is negative is
    skipped in both outputs: its bytes are whatever the fresh allocation held.
    The pool is only read.

    Returns `(k, k_scale)`: `k` is `float8_e4m3fn [num_tokens, head_dim]` and
    `k_scale` is `float32 [num_tokens, 1]`, both fresh and contiguous and both
    only carrying raw bytes. On the MXFP4 pool reinterpret them (no copy) before
    DeepGEMM: `k.view(torch.int8)` (packed E2M1, element `2j` in the low nibble
    of byte `j`) and `k_scale.view(torch.int32).reshape(-1)` (byte `b` of the
    little-endian word scales elements `32b .. 32b+31`).

    Fusion boundary: the gather only. The caller owns quantizing K and writing
    the pool, the slot mappings, the int8 / int32 reinterpretation, the logits
    call, and masking of skipped rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"one request S={s}, block_kv {bk}: its whole index K",
                spec=dict(block_kv=bk, lens=[s], chunks=[(0, s)]),
            )
            for bk, s in _ROUNDTRIP
        )
        + tuple(
            Cell(
                why=f"prefill chunks at k_token_start, block_kv {bk}",
                spec=dict(block_kv=bk, lens=_PACKED_LENS, chunks=_PACKED_CHUNKS),
            )
            for bk in (64, 128)
        )
        + tuple(
            Cell(
                why=f"strided owner view, block_kv {bk}: page stride 4 pages",
                spec=dict(block_kv=bk, lens=[1500, 3], chunks=[(0, 1503)], layers=4),
            )
            for bk in (64, 128)
        )
        + (
            Cell(
                why="num_tokens 0: empty outputs",
                spec=dict(block_kv=64, lens=[10], chunks=[(0, 0)]),
            ),
        )
    )

    note = """
    Cells: index K is bf16 [S, 128] with per-row magnitudes 2^-12..2^12,
    quantized by `torch.ops.trtllm.fp4_quantize(x, None, 32, True, False)`
    (MXFP4, UE8M0, linear scale layout) and written with torch indexing into a
    uint8 `[pages, block_kv, 1, 68]` pool holding 3x the pages the batch needs
    plus 8 over random bytes, pages drawn by a random permutation. The gathered
    bytes are the quantizer's, and their decode is the checkpoint's
    `fp4_act_quant(k, 32, True)` value set exactly.

    Metadata consumed: none. The caller builds the mappings; for the page-blocked
    layout (`block_kv x 64` code bytes, then `block_kv x 4` scale bytes per page),
    position `p` of a request with page-table row `table` is
        page = table[p // block_kv]; slot = p % block_kv
        slot_mapping_fp8   = page * block_kv * 68 + slot * 64
        slot_mapping_scale = page * block_kv * 68 + block_kv * 64 + slot * 4
    with the logical page stride `block_kv * 68` even when the view's `stride(0)`
    is larger. The op cannot tell a right mapping from a wrong one: another
    table, page + 1, slot + 1, key-interleaved addressing, a page stride without
    the scale region, another owner's view and 32-bit-wrapped offsets are all
    accepted and measured wrong on every row.

    Preconditions: the innermost dim has stride 1 and every index is a multiple
    of 4 inside the pool -- neither is checked (read, not probed). Mappings are
    1-D contiguous int64 CUDA tensors of at least `k_token_start + num_tokens`
    entries. Mask or drop skipped rows before use. The op rejects loudly a
    head_dim other than 64 or 128, a non-4-D, non-uint8 or `size(2) != 1` pool,
    and int32, CPU, 2-D, non-contiguous or too-short mappings.

    Row counts: every count up to 268,439,040 (ratio 1) / 134,219,520 (ratio 2),
    the per-step maximum of up to 256 context requests at the longest request,
    the rows spread over the whole pool; exhaustively 1..1,048,590 / 1..524,295.
    One precompiled kernel serves every count. Counts above 2^25 need the
    kernel's output-row offset formed in 64 bits
    (`static_cast<int64_t>(token_idx)` in `indexerKCacheGather.cu`): with a
    32-bit product every row from 2^25 on is silently misplaced and can land
    outside the output. Pools: any size up to 6,539,977 blocks (the largest any
    DeepSeek-V4.1 cache manager can have on a GB300), the far end past 2^35
    bytes included.

    CUDA graphs: not captured. The op allocates its outputs, and the target
    calls it only in steps with context requests, which are never captured.

    Not certified: other architectures; the FP8 mode (head_dim 128 over a
    132-byte entry); `block_kv` other than 64, 128 and 256; more rows than the
    per-step maxima above; `k_token_start + num_tokens` near 2^31 (rows are
    indexed in int32); pools above 6,539,977 blocks; out-of-range, misaligned or
    overlapping indices; CUDA-graph capture.
    """

    def raw_call(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.indexer_k_cache_gather_op(
            k_cache, slot_mapping_fp8, slot_mapping_scale, k_token_start, num_tokens, head_dim
        )

    def is_valid(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> None:
        # Pure-metadata guard: a head_dim that is not the pool's per-token payload width is
        # accepted by the op and silently reads the neighbouring key's bytes -- observed with
        # head_dim=128 on the 68-byte MXFP4 pool (bytes 64..127 of each row were the next key's
        # codes), never an error.
        assert k_cache.dim() != 4 or k_cache.size(3) == head_dim + 4, (
            f"head_dim ({head_dim}) must equal k_cache.size(3) - 4 ({k_cache.size(3) - 4}); "
            "the op would silently read past each row's payload"
        )

    def reference(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two outputs, gathered with native torch indexing from the pool's logical bytes.

        A skipped row (either slot negative) is left zero here; the op leaves it
        unwritten, so only cells without skipped rows compare whole outputs.
        Nothing is mutated.
        """
        flat = k_cache.contiguous().view(-1)
        rows = slice(k_token_start, k_token_start + num_tokens)
        d_off = slot_mapping_fp8[rows].long()
        s_off = slot_mapping_scale[rows].long()
        keep = (d_off >= 0) & (s_off >= 0)
        dev = flat.device
        k = torch.zeros(num_tokens, head_dim, dtype=torch.uint8, device=dev)
        k_scale = torch.zeros(num_tokens, _SCALE_BYTES, dtype=torch.uint8, device=dev)
        k[keep] = flat[d_off[keep].unsqueeze(1) + torch.arange(head_dim, device=dev)]
        k_scale[keep] = flat[s_off[keep].unsqueeze(1) + torch.arange(_SCALE_BYTES, device=dev)]
        return k.view(torch.float8_e4m3fn), k_scale.view(torch.float32)

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bitwise on both outputs' bytes.

        The op is a byte copy, so there is nothing to round: any byte that
        differs is a row read from the wrong place.
        """
        (k, k_scale), (ref_k, ref_scale) = out, ref
        assert k.dtype == ref_k.dtype == torch.float8_e4m3fn, (k.dtype, ref_k.dtype)
        assert k_scale.dtype == ref_scale.dtype == torch.float32, (k_scale.dtype, ref_scale.dtype)
        assert k.shape == ref_k.shape and k_scale.shape == ref_scale.shape, (
            k.shape,
            ref_k.shape,
            k_scale.shape,
            ref_scale.shape,
        )
        bad_k = (k.view(torch.uint8) != ref_k.view(torch.uint8)).any(-1).sum().item()
        bad_s = (k_scale.view(torch.uint8) != ref_scale.view(torch.uint8)).any(-1).sum().item()
        assert bad_k == 0 and bad_s == 0, (
            f"{bad_k} code rows and {bad_s} scale words of {k.size(0)} rows differ"
        )


indexer_k_cache_gather = IndexerKCacheGather()
