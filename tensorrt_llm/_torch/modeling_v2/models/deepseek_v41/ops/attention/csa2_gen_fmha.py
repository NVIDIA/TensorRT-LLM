# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation attention of large decode steps on the engine's ``gen_fmha`` core: E4M3 staging of every row a token
attends into one bank, then the GB300 sparse MLA CuTe DSL kernel over that bank.

Stock routes the generation tile of an index-ratio-2 staging group's layer here from 1024 rows per rank
(``CSA2TrtllmAttention.gen_fmha_serves``) instead of the fused decode, and this entry runs the same launches:

1. ``derive_stage_kv_scales``: the forward's power-of-two staging KV scale from the group-scale bytes of the selected
   rows, capped by Q's amax (the int32 word the FP8 query GEMM raised; the finalize consumes and clears it).
2. ``stage_selected_rows`` (bank form, with dedup): compaction of each token's window + selected compressed rows (the
   staging group's first layer of a forward; the others reuse it and, while the bank still holds the group's main rows,
   restage only the 128 window rows), decode of those rows to E4M3 at the staging scale into the bank, the published
   bank-row indices / lengths / BMM scales, and the rescale of the unit-scale E4M3 query to the live scale.
3. ``gen_sparse_attention``: the GB300 core, writing BF16 or the grouped MXFP8 O-A input (inverse RoPE applied).

The staging tile (bank, published indices, scales, compaction scratch) is the caller's: `csa2_gen_fmha_tile` builds it
(a setup helper, eagerly), as stock's per-geometry query-tile metadata holds it.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_sparse_decode import (
    HEAD_DIM,
    HEADS,
    MAIN_ROW,
    SWA_ROW,
    _pad_topk,
    _visible_rows,
    _window_rows,
    csa2_sparse_decode,
)

MIN_ROWS = 1024  # stock's `_GEN_FMHA_MIN_ROWS`: from here a ratio-2 group's decode step is faster on this core
RATIO = 2  # stock's `_GEN_FMHA_COMPRESS_RATIO`: ratio-1 groups are slower on it at any row count
_SWA_TILE = 128
_TMA_MIN_BACKING_BYTES = 128 << 10
_O_GROUPS = 8


def csa2_gen_fmha_supported(heads: int, extra_width: int, device) -> bool:
    """Whether the core serves a tile of `heads` query heads over 128 window + `extra_width` selected rows here (stock's
    `CSA2TrtllmMetadata.gen_fmha_tile` with E4M3 staging): 64 heads, at most 640 rows a token, sm_103."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.gen_fmha import (
        gen_fmha_device_supported,
        gen_fmha_fits,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        _fused_gather_supported,
    )

    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    width = _SWA_TILE + -(-extra_width // 128) * 128
    return (
        gen_fmha_fits(heads, width)
        and gen_fmha_device_supported(index)
        and _fused_gather_supported(index)
    )


class Csa2GenFmhaTile:
    """The staging state stock keeps on a generation query tile's metadata (`CSA2TrtllmMetadata._allocate(one_bank=
    True)` with E4M3 staging), for up to `capacity` rows: both staging pools in one allocation (the bank), the published
    bank-row indices and lengths, the BMM scales, the derived staging KV scale pair, the staged-scale slots, the scale
    bitmap, and one compaction scratch per staging group (`_stage_scratch`). Built by `csa2_gen_fmha_tile`."""

    def __init__(self, capacity: int, extra_width: int, device):
        e4m3 = torch.float8_e4m3fn
        self.capacity = capacity
        extra = -(-extra_width // 128) * 128
        self.num_sparse_topk = _SWA_TILE + extra
        primary = capacity * _SWA_TILE * HEAD_DIM
        rest = capacity * max(extra, 1) * HEAD_DIM
        backing = torch.empty(
            primary + max(rest, _TMA_MIN_BACKING_BYTES), dtype=e4m3, device=device
        )
        self.swa_pool = backing[:primary].view(capacity, _SWA_TILE, HEAD_DIM)
        self.extra_pool = backing[primary : primary + rest].view(capacity, max(extra, 1), HEAD_DIM)
        self.gen_fmha_bank = (backing, primary // HEAD_DIM)
        self.prepared_indices = torch.full(
            (capacity, self.num_sparse_topk), -1, dtype=torch.int32, device=device
        )
        self.prepared_lens = torch.empty(capacity, dtype=torch.int32, device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        self.mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        self.native_unit_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._csa2_stage_kv_dequant = torch.ones(1, dtype=torch.float32, device=device)
        self._csa2_stage_kv_quant = torch.ones(1, dtype=torch.float32, device=device)
        self._csa2_staged_scales = torch.ones(2, dtype=torch.float32, device=device)
        self._csa2_stage_scale_bitmap = torch.zeros(17, dtype=torch.int32, device=device)
        self.num_tokens = 0
        self.groups = {}

    def scratch(self, group) -> dict:
        """The compaction scratch of staging group `group` (stock's `_stage_scratch` entry for it)."""
        entry = self.groups.get(group)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "an eager step must create a staging group's scratch before a graph capture"
                )
            cap, width, device = self.capacity, self.num_sparse_topk, self.prepared_lens.device
            i32 = dict(dtype=torch.int32, device=device)
            entry = self.groups[group] = {
                "tags": torch.empty((cap, width), **i32),
                "counts": torch.empty(cap, **i32),
                "refs": torch.empty((cap, width), **i32),
                "own_blocks": torch.empty(cap, **i32),
                "row_slots": torch.empty((cap, width), **i32),
                "main_slots": torch.empty(
                    (cap, max(width - _SWA_TILE, 1)), dtype=torch.int64, device=device
                ),
                "main_bits": torch.zeros(8, **i32),
                "serial": group,
            }
        return entry


_TILES: dict = {}


def csa2_gen_fmha_tile(device, rows: int, graph_rows: int, extra_width: int) -> Csa2GenFmhaTile:
    """The staging tile for a step of `rows` rows (a setup helper; the first call of a geometry is an eager step): one
    tile of `graph_rows` rows (the largest CUDA-graph step) serves every step up to that size, graphs included (their
    replays run one at a time); a larger eager step gets a tile of its own size (only the latest is kept)."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    capacity = graph_rows if rows <= graph_rows else rows
    key = (index, capacity <= graph_rows, extra_width)
    tile = _TILES.get(key)
    if tile is None or tile.capacity != capacity:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "an eager step must build the gen_fmha staging tile before a graph capture"
            )
        tile = _TILES[key] = Csa2GenFmhaTile(capacity, extra_width, device)
    return tile


class Csa2GenFmhaDecode(OpWrapper):
    """`csa2_sparse_decode` (generation form) of a 64-head layer on the `gen_fmha` core, as stock runs an index-ratio-2
    staging group's decode steps of 1024+ rows.

    `tile` is the step's `Csa2GenFmhaTile`; `q8` `[n, 64, 512]` the unit-scale E4M3 query (stock's `fp8_query_gemm`, the
    decode-graph query), which the staging rescales in place to the live scale, with `q_word` int32 `[1]` the FP32 bits
    of `amax(|q|)` raised by the query GEMM (zero before it), consumed and cleared here; or the bf16 query (an eager
    step: stock projects the query in bf16 there) with `q_word` None: the scale derivation reads Q's amax and
    `quantize_stage_query` quantizes it at the live scale. `sink`, `pos`, `rid`, `swa_pool` / `swa_table`, `window`,
    `scale`, `main_pool` / `main_table`, `topk_idx`, `ratio`, `swa_rows`, `visible` as `csa2_sparse_decode` (generation:
    every window row is in the pool). `group` names the staging group (layers sharing a KV owner and an index source
    select the same compressed rows); `first`: this layer is the group's first of the forward (it compacts the
    selections and records their scale bits; the others reuse both). With `mxfp8_out = (pos int32 [n], cos_sin fp32
    [P, 64])` the result is the grouped O-A input `(E4M3 [8, n, 4096], swizzled UE8M0 scales)` with the inverse RoPE
    applied (`csa2_sparse_decode`'s same option); else bf16 `[n, 64, 512]`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 decode at ratio 2: 256 requests x 4 rows (1024, the core's floor), 512 selected rows",
            spec=dict(reqs=256, rows=4, hist=(200, 3000), ratio=2, topk=512),
        ),
        Cell(
            why="dep8 decode at ratio 2: 320 requests x 4 rows (1280), the c7000 step",
            spec=dict(reqs=320, rows=4, hist=(200, 6000), ratio=2, topk=512),
        ),
        Cell(
            why="requests whose history is shorter than the selection (most selected rows invalid)",
            spec=dict(reqs=260, rows=4, hist=(0, 300), ratio=2, topk=512),
        ),
    )
    note = """
    The staged rows and the query are E4M3 at a shared power-of-two scale (the query's amax caps it), the core's
    probabilities E4M3 with a lazy rescale: the agreement with the plain-torch reference is FP8-level (the test states
    the band), against stock's fused decode over the same rows likewise. The MXFP8 output is bit for bit
    `inverse_rope_mxfp8` of the core's own bf16 output. The op allocates only its outputs; the tile and its scratch are
    the caller's, and a staging group's later layers read the compaction its first layer wrote in the same forward.
    """

    def __call__(
        self,
        tile: Csa2GenFmhaTile,
        q8: torch.Tensor,
        q_word: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: torch.Tensor,
        main_table: torch.Tensor,
        topk_idx: torch.Tensor,
        ratio: int,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        group=None,
        first: bool = True,
        mxfp8_out: Optional[tuple] = None,
    ):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.gen_fmha import gen_sparse_attention
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            derive_stage_kv_scales,
            quantize_stage_query,
            stage_selected_rows,
        )

        n, dev = q8.shape[0], q8.device
        tile.num_tokens = n
        swa_flat, main_flat = swa_pool.view(-1, SWA_ROW), main_pool.view(-1, MAIN_ROW)
        swa_idx = swa_rows if swa_rows is not None else _window_rows(pos, rid, swa_table, window)
        logical = _pad_topk(topk_idx)
        rpp = 128 // ratio
        visible = visible if visible is not None else _visible_rows(pos, ratio)
        requests = (
            rid if rid.dtype == torch.int64 and rid.is_contiguous() else rid.long().contiguous()
        )
        mapping = (main_table, rpp, main_table.shape[1] * rpp, requests, visible)
        scratch = tile.scratch(group)
        # (1) the staging KV scale: the group's first layer records its main rows' scale bits, the others reuse them
        derive_stage_kv_scales(
            [(swa_flat, swa_idx, "swa"), (main_flat, logical, "main")],
            tile._csa2_stage_scale_bitmap, tile._csa2_stage_kv_dequant, tile._csa2_stage_kv_quant,
            main_mapping=mapping, main_bits=scratch["main_bits"], reuse_main=not first, q_word=q_word,
            q=None if q_word is not None else q8.contiguous(),
        )  # fmt: skip
        orig_quant, quant_orig = tile._csa2_stage_kv_quant, tile._csa2_stage_kv_dequant
        fixup = (q8, orig_quant)
        if (
            q_word is None
        ):  # the bf16 query, quantized at the live scale (the generation Q takes the staging pair backwards)
            fixup = None
            quant = torch.empty(q8.shape, dtype=torch.float8_e4m3fn, device=dev)
            quantize_stage_query(q8.contiguous(), orig_quant, quant)
            q8 = quant
        # (2) stage into the bank; generation Q takes the staging pair read backwards (dequant_q = 1 / dequant_kv)
        stage_selected_rows(
            swa_flat, swa_idx, main_flat, scratch["main_slots"][:n, : logical.shape[1]], tile, scratch,
            compact=first, main_logical=logical, main_mapping=mapping,
            kv_scale_orig_quant=orig_quant, kv_scale_quant_orig=quant_orig, q_scale_quant_orig=orig_quant,
            softmax_scale=scale, dedup=True, bank_extra=tile.gen_fmha_bank[1], q_fixup=fixup,
        )  # fmt: skip
        # (3) the core
        if mxfp8_out is None:
            out = torch.empty(n, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            gen_sparse_attention(q8, tile, sink.float().contiguous(), out)
            return out
        width = HEADS // _O_GROUPS * HEAD_DIM
        data = torch.empty(_O_GROUPS, n, width, dtype=torch.float8_e4m3fn, device=dev)
        scales = torch.empty(
            _O_GROUPS * (-(-n // 128) * 128) * (width // 32), dtype=torch.uint8, device=dev
        )
        gen_sparse_attention(
            q8, tile, sink.float().contiguous(), None, (data, scales, mxfp8_out[0], mxfp8_out[1])
        )
        return data, scales

    def reference(
        self,
        tile: Csa2GenFmhaTile,
        q8: torch.Tensor,
        q_word: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: torch.Tensor,
        main_table: torch.Tensor,
        topk_idx: torch.Tensor,
        ratio: int,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        group=None,
        first: bool = True,
        mxfp8_out: Optional[tuple] = None,
    ) -> torch.Tensor:
        """`csa2_sparse_decode.reference` (the plain-torch op on the same packed pools) over the query `q8` holds on
        entry (an E4M3 one widened exactly); `tile`, `q_word`, `group`, `first` are ignored. Call it before the op (the
        op rescales an E4M3 `q8` in place)."""
        return csa2_sparse_decode.reference(
            None, q8.bfloat16(), sink, pos, rid, swa_pool, swa_table, window, scale, main_pool, main_table, topk_idx,
            ratio, swa_rows=swa_rows, visible=visible, mxfp8_out=mxfp8_out,
        )  # fmt: skip

    def is_valid(
        self,
        tile: Csa2GenFmhaTile,
        q8: torch.Tensor,
        q_word: torch.Tensor,
        sink: torch.Tensor,
        pos: torch.Tensor,
        rid: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        scale: float,
        main_pool: torch.Tensor,
        main_table: torch.Tensor,
        topk_idx: torch.Tensor,
        ratio: int,
        swa_rows: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        group=None,
        first: bool = True,
        mxfp8_out: Optional[tuple] = None,
    ) -> None:
        n = q8.shape[0]
        assert (
            q8.dtype in (torch.float8_e4m3fn, torch.bfloat16)
            and q8.shape[1:] == (HEADS, HEAD_DIM)
            and q8.is_contiguous()
        )
        assert (q_word is None) == (q8.dtype == torch.bfloat16), (
            "an E4M3 query comes with its amax word, a bf16 one without"
        )
        assert q_word is None or (q_word.dtype == torch.int32 and q_word.numel() >= 1), (
            "q_word: the query GEMM's int32 word"
        )
        assert 0 < n <= tile.capacity, "more rows than the staging tile holds"
        assert topk_idx is not None and topk_idx.shape[0] == n
        assert -(-topk_idx.shape[1] // 64) * 64 <= tile.num_sparse_topk - _SWA_TILE, (
            "more selections than the tile's width"
        )
        assert ratio in (1, 2) and window <= _SWA_TILE
        assert swa_pool.shape[1:] == (128, SWA_ROW) and main_pool.shape[1:] == (
            128 // ratio,
            MAIN_ROW,
        )
        assert main_table.dtype == torch.int32
        assert sink.shape == (HEADS,)
        if swa_rows is not None:
            assert (
                swa_rows.dtype == torch.int64
                and swa_rows.shape == (n, window)
                and swa_rows.is_contiguous()
            )
        if visible is not None:
            assert (
                visible.dtype == torch.int64 and visible.shape == (n,) and visible.is_contiguous()
            )
        if mxfp8_out is not None:
            assert (
                mxfp8_out[0].dtype == torch.int32
                and mxfp8_out[0].shape == (n,)
                and mxfp8_out[0].is_contiguous()
            )


csa2_gen_fmha_decode = Csa2GenFmhaDecode()
