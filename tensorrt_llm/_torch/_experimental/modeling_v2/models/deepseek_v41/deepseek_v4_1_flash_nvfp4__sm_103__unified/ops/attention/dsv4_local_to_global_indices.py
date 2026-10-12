# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention index table: per-token local SWA / compressed indices -> pool rows.

The op is `torch.ops.trtllm.dsv4_local_to_global_indices`, a thin torch custom op registered in
`tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py` over that file's Triton
kernel `_deepseek_v4_local_to_global_kernel` (the kernel the in-tree Python launcher
`deepseek_v4_local_to_global_indices` also runs). One Triton launch per call.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels  # noqa: F401 — registers torch.ops.trtllm.dsv4_local_to_global_indices
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_SWA_TPB = 128
#: Pages per pool and rows per block table in the cells: plainly larger than any batch needs,
#: with the batch at scattered table rows so the table row is never the batch index.
_POOL_PAGES = 4096
#: (name, [(cached, new) per request], max_seq_len, replay_tail, what the step is)
_STEPS = (
    (
        "prefill_mixed",
        ((0, 1), (0, 127), (0, 128), (0, 129), (0, 1000), (0, 3000)),
        16384,
        None,
        "6 fresh prompts of 1..3000 tokens, 4385 rows",
    ),
    ("prefill_8192", ((0, 8192),), 16384, None, "one 8192-token prefill"),
    (
        "chunked_prefill",
        ((8192, 2048), (130, 5), (1, 1)),
        16384,
        None,
        "2048 rows on an 8192-token cached prefix, plus 5- and 1-token chunks",
    ),
    (
        "decode",
        tuple((c, 1) for c in (0, 1, 2, 126, 127, 128, 129, 255, 256, 4095, 8191, 16383))
        + tuple((c, 1) for c in range(300, 16000, 797)),
        16384,
        None,
        "32 decode rows at positions 0..16383",
    ),
    (
        "dspark_verify_6",
        tuple((c, 6) for c in range(0, 32 * 509, 509)),
        16384,
        None,
        "32 DSpark verify runs of 6 rows, 192 rows",
    ),
    (
        "long_context",
        tuple((c, 1) for c in (20479, 20480, 32767, 40000, 65534)) + ((65529, 6),),
        65536,
        None,
        "positions 20479..65534 on 512-page tables",
    ),
    (
        "replay_tail",
        ((872, 128),),
        16384,
        872,
        "the last 128 rows of a 1000-token prompt, window cut at the replay start 872",
    ),
)
#: (name, SWA offset in pages, compressed offset in pages): offsets 0, and owner-style offsets.
_OFFSETS = (("offset 0", 0, 0), ("owner offsets", 3 * _POOL_PAGES, 2 * _POOL_PAGES))


class Dsv4LocalToGlobalIndices(OpWrapper):
    """Translate each query token's local SWA and compressed indices into pool rows.

    For token `t`, table row `r = req_id[t]`, and a segment `(table, local, tpb, offset)`::

        map(l) = offset + table[r, l // tpb] * tpb + l % tpb   if l >= 0 and l // tpb < table width
               = -1                                             otherwise
        out[t, :W_swa] = map_swa(swa_local_indices[t])
        out[t, W_swa:] = map_compressed(compressed_local_indices[t])

    The compressed segment is present exactly when `compressed_block_table` and
    `compressed_local_indices` are both given. Fusion boundary: address translation only; the
    window construction, the index top-k, the pool writes and the attention call are the
    caller's. Returns a fresh contiguous int32 `[T, W_swa + W_compressed]` tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{name} r{ratio} {offset_name}: {what}",
            spec=dict(
                ratio=ratio,
                seqs=seqs,
                max_seq_len=max_seq_len,
                replay_tail=replay_tail,
                pool_pages=_POOL_PAGES,
                table_rows=64,
                swa_tokens_per_block=_SWA_TPB,
                swa_row_offset=swa_pages * _SWA_TPB,
                compressed_tokens_per_block=_SWA_TPB // ratio if ratio else 0,
                compressed_row_offset=comp_pages * _SWA_TPB // ratio if ratio else 0,
            ),
        )
        for name, seqs, max_seq_len, replay_tail, what in _STEPS
        for ratio in (0, 1, 2)
        for offset_name, swa_pages, comp_pages in _OFFSETS
    )

    note = """
    Column order is preserved: the op does not sort, compact or de-duplicate,
    so the valid-first order the attention kernel needs is whatever the local
    indices already have. Only a negative local index or one whose block is
    past the table's width becomes -1. The op knows no sequence lengths: the
    caller masks invisible positions, and a page id of -1 in a table is used
    as page -1.

    DeepSeek-V4.1 calls it once per layer per step with offsets 0: ratio 0
    window-only (`[T, 128]`, 128-row SWA pages); ratios 2 and 1 add the
    layer's `[T, 512]` index top-k through the compress block table at
    `128 // ratio` entries per page (`[T, 640]`). Under decoder bounded replay
    the target also makes a window-only slot call whose local table is
    `[row position, -1, ..., -1]`. Under native KV storage the target
    may pass a synthetic compressed table `arange(M * W).view(M, W)` so entry
    `e` of request `b` names row `b * W * tpb + e` of its compact read buffer;
    `req_id` must then be each token's request position. The output is
    `sparse_indices` for trtllm_sparse_mla_dsv4 when each offset is the row
    where page 0 of this layer's buffer starts in the pool tensor handed to
    that call; it reads an attended -1 as a zero row, so the -1s must lie
    outside the lengths it is given.

    Metadata consumed: none; every input is an argument.

    Preconditions: the op raises ValueError before launching on non-int32
    tensors, a `W_swa` or `W_compressed` that is not a power of two, a
    compressed table without its indices or the reverse, a `req_id` that is
    not a contiguous 1-D tensor, a non-positive compressed page size, and an
    offset outside `[0, 2**31)`. Not checked: `req_id[t]` below both tables'
    row counts (else it reads past the table), and every reachable page id a
    real page. Every result must fit int32, `offset + page * tpb + tpb - 1 <
    2**31`: one page past the edge comes out wrapped (`-2**31`) with no error.
    Block tables and index tensors are read through their strides; `req_id`
    needs stride 1. `T = 0` returns an empty `[0, W]` tensor without a launch.

    Widths, page sizes, the compressed segment's presence, `enable_pdl` and
    each tensor's 16-byte alignment are Triton specializations (an unaligned
    table view compiles one more variant); `T` is only the grid. Make one
    eager call of each form, on the tensors a capture will use, before
    CUDA-graph capture. `enable_pdl=True` waits on the preceding grid before
    reading and releases dependents after its stores; results are identical
    either way, and the consumer must honour PDL ordering.

    The in-tree V4 launcher runs the same kernel but compiles the compressed
    segment only for `compress_ratio > 1`: a V4.1 ratio-1 layer through it
    gets compressed columns that are never written. This op takes the page
    size directly and passes strided inputs without a copy.

    Beyond the cells, the test certifies on engine-size tables: every prefix
    `T` = 1..32768, capture at `T = G` and `6 G` for G in 1..32, 40..128 step
    8, 192 and 256, random mixed steps up to 256 requests and 16384 rows, the
    dumped `(T, ratio)` cells, page ids up to the int32 edge (2**24 pages of
    128 rows, 2**25 of 64), generation positions up to `Lm + 5` for
    `max_seq_len` 16384..1048576 with and without DSpark (Lm = L + 14 with
    DSpark), unaligned table views, the slot call, and the native-storage synthetic
    table up to 256 requests at every width; plus wrong-variant controls.

    Not certified: widths other than 128 / 512, page sizes other than 128 /
    64; results at or above 2**31; more than 32768 rows or 256 requests per
    call; block-table widths other than 129, 132 and the twelve
    `ceil(Lm / 128) + 3` widths 131..8196 (plus 2, 4, 128, 512); a context
    position at or past L or a generation position past Lm + 5; CUDA-graph
    capture at any other `T`, or for the native-storage table at an `(M, W)` pair
    not run; the native-storage table above 256 rows or with any other `req_id`;
    tracing through torch.compile.
    """

    def raw_call(
        self,
        req_id: torch.Tensor,
        swa_block_table: torch.Tensor,
        swa_local_indices: torch.Tensor,
        swa_tokens_per_block: int,
        swa_row_offset: int,
        compressed_block_table: Optional[torch.Tensor] = None,
        compressed_local_indices: Optional[torch.Tensor] = None,
        compressed_tokens_per_block: int = 0,
        compressed_row_offset: int = 0,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.dsv4_local_to_global_indices(
            req_id,
            swa_block_table,
            swa_local_indices,
            swa_tokens_per_block,
            swa_row_offset,
            compressed_block_table,
            compressed_local_indices,
            compressed_tokens_per_block,
            compressed_row_offset,
            enable_pdl,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        swa_block_table: torch.Tensor,
        swa_local_indices: torch.Tensor,
        swa_tokens_per_block: int,
        swa_row_offset: int,
        compressed_block_table: Optional[torch.Tensor] = None,
        compressed_local_indices: Optional[torch.Tensor] = None,
        compressed_tokens_per_block: int = 0,
        compressed_row_offset: int = 0,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        """Pool rows by torch table lookup, in int64 and never cast.

        A row at or past 2**31, which the kernel's int32 arithmetic wraps, stays visible as
        its true value rather than being matched by an equally wrapped cast. `enable_pdl` is
        ignored (it changes the launch, not the table).
        """

        def segment(table, local, tpb, offset):
            width = table.shape[1]
            blk = torch.div(local, tpb, rounding_mode="floor").long()
            valid = (local >= 0) & (blk < width)
            page = table[req_id.long()[:, None], blk.clamp(0, width - 1)]
            rows = offset + page.long() * tpb + torch.remainder(local, tpb).long()
            return torch.where(valid, rows, -1)

        out = segment(swa_block_table, swa_local_indices, swa_tokens_per_block, swa_row_offset)
        if compressed_block_table is not None and compressed_local_indices is not None:
            comp = segment(
                compressed_block_table,
                compressed_local_indices,
                compressed_tokens_per_block,
                compressed_row_offset,
            )
            out = torch.cat([out, comp], dim=1)
        return out

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Exact: every entry equals the int64 reference.

        Address translation has nothing to round, so any differing entry is a wrong row. The
        controls (window off by one, the other ratio's page size, the neighbouring table row,
        the two offsets swapped, page ids truncated to 18 bits, one page past the int32 edge)
        each differ on hundreds to hundreds of thousands of entries.
        """
        assert out.dtype == torch.int32, out.dtype
        assert ref.dtype == torch.int64, ref.dtype
        assert out.shape == ref.shape, (out.shape, ref.shape)
        bad = (out.long() != ref).sum().item()
        assert bad == 0, f"{bad} of {out.numel()} entries differ"


dsv4_local_to_global_indices = Dsv4LocalToGlobalIndices()
