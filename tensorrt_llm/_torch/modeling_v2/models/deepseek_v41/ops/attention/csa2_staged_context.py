# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 context attention of a rank with fewer than 64 query heads, as stock runs it: E4M3 staging of every row a query
attends, then the trtllm-gen MLA context FMHA over the staged rows.

Stock's CSA2 backend (`CSA2TrtllmAttention`) serves a context tile with the fused FP8 context kernel only at 64 query
heads (`fused_context`); a TP-split rank (8 heads at TP8) takes the native FP8 path of `CSA2TrtllmAttention.forward`
instead: `sparse_attn_predict` derives the forward's staging KV scale from the selected rows' group-scale bytes and
Q's amax (`derive_stage_kv_scales`, context: Q divided by the scale), stages the window and selected compressed rows
of every query to E4M3 (`stage_selected`), and the trtllm-gen context FMHA reads them with Q quantized at that scale.

This entry runs exactly that: one of stock's own `CSA2TrtllmAttention` backends (built as stock's module builds it:
TRTLLM, MLA 448 / 64 / 512, FP8 KV) on a query-tile metadata of the step's request groups
(`CSA2TrtllmMetadata.get_query_tile_metadata`, as stock's module asks for it), with the selections handed in
(`CSA2BackendForwardArgs` without a forward state: the caller already published the step's rows and ran the indexer).

Inputs: `q` bf16 `[n, h, 512]` (RoPE applied), `sink` fp32 `[h]`, `swa_pool` the layer's packed SWA rows `[R, 528]`
uint8 with `swa_rows` int64 `[n, 128]` the rows each query reads (-1 none: stock's `csa2_swa_indices`), `main_pool` the
owner's packed compressed rows `[M, 288]` uint8 with `main_logical` int64 `[n, k]` the query's selections (compressed
positions, -1 none) and `main_mapping` stock's `(page_table, page_size, max_positions, requests, visible)` that maps
them to pool rows (`csa2_main_mapping`; or both None for a window-only layer), as stock's module hands its backend the
logical selections with `main_mapping` and the staging kernel resolves the rows itself (`_native_slot_mapping`),
`lengths` the step's rows per request (host ints, in order), `starts` each request's first position (host ints),
`extra_width` the selection capacity (stock: the index top-k, 0 for a window-only layer). Returns bf16 `[n, h, 512]`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HEAD_DIM = 512
SWA_ROW, MAIN_ROW = 528, 288
_BACKENDS: dict = {}
_SOURCES: dict = {}  # per device: the request metadata whose query tiles the steps reuse (`_tile`)


def _backend(heads: int, device):
    """Stock's CSA2 attention backend for `heads` query heads, as its module builds it (one per head count and device;
    the native FP8 context path reads no per-layer state of it)."""
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    key = (heads, index)
    attn = _BACKENDS.get(key)
    if attn is None:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params
        from tensorrt_llm._torch.attention.backends.utils import create_attention

        with torch.cuda.device(index):
            attn = _BACKENDS[key] = create_attention(
                "TRTLLM", 0, heads, HEAD_DIM, num_kv_heads=1, is_mla_enable=True, q_lora_rank=1280, kv_lora_rank=448,
                qk_nope_head_dim=448, qk_rope_head_dim=64, v_head_dim=512, rope_append=False,
                sparse_params=CSA2Params(), kv_cache_dtype="fp8",
            )  # fmt: skip
        assert attn.native_fp8 and not attn.fused_context, "a native FP8, unfused context backend"
    return attn


def csa2_staged_context_supported(heads: int, device) -> bool:
    """Whether stock serves a context tile of `heads` heads on this
    path here (native FP8 staging, not the fused kernel)."""
    from tensorrt_llm._utils import is_sm_100f

    return heads < 64 and torch.cuda.is_available() and is_sm_100f()


def csa2_main_mapping(
    table: torch.Tensor, rid: torch.Tensor, visible: torch.Tensor, ratio: int, max_positions: int
):
    """Stock's `main_mapping` of a context tile (`CSA2TrtllmAttention._resolve_sparse_inputs`): the owner's page table
    (int32, `128 // ratio` rows a page), its compressed-position bound (stock: `max_seq_len // ratio`), each query's
    request row `rid` and its visible compressed length."""
    return table, 128 // ratio, max_positions, rid, visible


def csa2_main_rows(logical: torch.Tensor, mapping) -> torch.Tensor:
    """The flat compressed-pool rows of `logical` selections `[n, k]` (-1 none) through `mapping` (`csa2_main_mapping`),
    -1 past the query's visible length: stock's `_map_global_slots`, which the staging kernel computes in place."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    table, page, max_positions, rid, visible = mapping
    return CSA2TrtllmMetadata._map_global_slots(
        logical, rid.long(), table, visible, page, max_positions
    )


def _tile(q, lengths, starts, extra_width):
    """The query-tile metadata stock's module asks for (`get_query_tile_metadata` of a context-only step).

    One request metadata per device keeps the tiles, as stock's step metadata does (`_csa2_query_tiles`: the latest
    eager tile of each geometry): every later layer of the step, and a later step of the same rows, reuses the tile and
    its staging, which `get_query_tile_metadata` rebinds to the step's request lengths (a fresh metadata per call would
    build both again in every layer)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    n, reqs = q.shape[0], len(lengths)
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    source = _SOURCES.get(index)
    if (
        source is None
    ):  # (the tiles it keeps are the only state it carries; its own buffers are never read)
        source = _SOURCES[index] = CSA2TrtllmMetadata(max_num_requests=reqs, max_num_tokens=n)
    source._num_contexts = reqs
    source._num_ctx_tokens = source._num_tokens = n
    source.csa2_num_context_requests = reqs
    ranges, t = [], 0
    for k in lengths:
        ranges.append((t, t + k))
        t += k
    source.csa2_request_query_ranges = tuple(ranges)
    source.csa2_request_start_positions = tuple(int(s) for s in starts)
    source.csa2_request_lengths = tuple(int(k) for k in lengths)
    source.is_cuda_graph = False
    return source.get_query_tile_metadata(
        q, extra_width, query_start=0, staging_dtype=torch.float8_e4m3fn
    )


class Csa2StagedContext(OpWrapper):
    """CSA2 context attention of a < 64-head rank on stock's native
    FP8 path (E4M3 staging + trtllm-gen context FMHA)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 context (8 heads): a 14-row prompt of a window-only layer",
            spec=dict(heads=8, lengths=(14,), topk=0),
        ),
        Cell(
            why="tep8 context: two requests with history, 512 selected compressed rows",
            spec=dict(heads=8, lengths=(37, 90), topk=512, hist=(300, 1500)),
        ),
        Cell(
            why="tep8 context: one 200-row chunk past the window, 512 selections",
            spec=dict(heads=8, lengths=(200,), topk=512, hist=(0,)),
        ),
    )
    note = """
    The staged rows and Q are E4M3 at the forward's staging scale and the FMHA's probabilities E4M3: against the
    plain-torch attention over the dequantized rows the agreement is FP8-level (the test states the band). The op is
    stock's backend itself (no reimplementation): a stock forward over the same rows is the same launch sequence.
    """

    def __call__(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_rows: torch.Tensor,
        main_pool: Optional[torch.Tensor],
        main_logical: Optional[torch.Tensor],
        main_mapping: Optional[tuple],
        lengths: Sequence[int],
        starts: Sequence[int],
        extra_width: int,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.attention.backends.interface import (
            AttentionForwardArgs,
            AttentionInputType,
        )
        from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

        n, heads, _ = q.shape
        out = torch.empty(n, heads * HEAD_DIM, dtype=torch.bfloat16, device=q.device)
        if n == 0:
            return out.view(n, heads, HEAD_DIM)
        attn = _backend(heads, q.device)
        query = q.reshape(n, heads, HEAD_DIM).contiguous()
        metadata = _tile(query, lengths, starts, extra_width)
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            attention_sinks=sink,
            output=out,
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_pool=swa_pool.view(-1, SWA_ROW),
                swa_indices=swa_rows,
                main_pool=None if main_logical is None else main_pool.view(-1, MAIN_ROW),
                topk_indices=main_logical,
                main_logical_indices=main_logical,
                main_mapping=main_mapping,
            ),
        )
        res = attn.forward(query.flatten(1), None, None, metadata, forward_args=args)
        res = res[0] if isinstance(res, (tuple, list)) else res
        return res.view(n, heads, HEAD_DIM)

    def reference(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_rows: torch.Tensor,
        main_pool: Optional[torch.Tensor],
        main_logical: Optional[torch.Tensor],
        main_mapping: Optional[tuple],
        lengths: Sequence[int],
        starts: Sequence[int],
        extra_width: int,
    ) -> torch.Tensor:
        """fp32 softmax attention (sink in the denominator, scale 1 /
        sqrt(512)) over the dequantized rows of each query."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

        n, heads, _ = q.shape
        swa = unpack_rows(
            swa_pool.view(-1, SWA_ROW)[swa_rows.clamp_min(0).flatten()], HEAD_DIM, "swa"
        ).view(n, -1, HEAD_DIM)
        valid = swa_rows >= 0
        main_rows = None if main_logical is None else csa2_main_rows(main_logical, main_mapping)
        if main_rows is not None and main_rows.shape[1]:
            main = unpack_rows(
                main_pool.view(-1, MAIN_ROW)[main_rows.clamp_min(0).flatten()], HEAD_DIM, "main"
            )
            swa = torch.cat([swa, main.view(n, -1, HEAD_DIM)], 1)
            valid = torch.cat([valid, main_rows >= 0], 1)
        kv = swa.float()
        s = torch.einsum("nhd,nkd->nhk", q.float(), kv) * HEAD_DIM**-0.5
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        m = torch.maximum(s.amax(-1, keepdim=True), sink.float()[None, :, None])
        p = (s - m).exp()
        den = p.sum(-1, keepdim=True) + (sink.float()[None, :, None] - m).exp()
        return (torch.einsum("nhk,nkd->nhd", p, kv) / den).to(torch.bfloat16)

    def is_valid(
        self,
        q: torch.Tensor,
        sink: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_rows: torch.Tensor,
        main_pool: Optional[torch.Tensor],
        main_logical: Optional[torch.Tensor],
        main_mapping: Optional[tuple],
        lengths: Sequence[int],
        starts: Sequence[int],
        extra_width: int,
    ) -> None:
        assert (
            q.dim() == 3
            and q.dtype == torch.bfloat16
            and q.shape[2] == HEAD_DIM
            and q.shape[1] < 64
        ), "q is bf16 [n, heads < 64, 512] (64-head ranks take the fused context kernel)"
        n, heads = q.shape[:2]
        assert sink.shape == (heads,), "one sink per head"
        assert swa_pool.dtype == torch.uint8 and swa_pool.numel() % SWA_ROW == 0, (
            "packed SWA rows [R, 528] uint8"
        )
        assert swa_rows.dtype == torch.int64 and swa_rows.shape == (n, 128), (
            "swa_rows int64 [n, 128]"
        )
        assert (main_mapping is None) == (main_logical is None), (
            "the selections and their mapping come together"
        )
        if main_logical is not None:
            assert main_pool.dtype == torch.uint8 and main_pool.numel() % MAIN_ROW == 0, (
                "packed main rows [M, 288] uint8"
            )
            assert (
                main_logical.dtype in (torch.int32, torch.int64)
                and main_logical.dim() == 2
                and main_logical.shape[0] == n
            )
            table, page, max_positions, rid, visible = main_mapping
            assert (
                table.dtype == torch.int32 and table.dim() == 2 and page > 0 and max_positions > 0
            ), "int32 page table"
            assert rid.shape == (n,) and visible.shape == (n,), (
                "a request row and a visible length per query"
            )
            assert extra_width >= main_logical.shape[1]
        assert sum(lengths) == n and len(starts) == len(lengths) and all(k > 0 for k in lengths), (
            "request rows cover the step"
        )


csa2_staged_context = Csa2StagedContext()
