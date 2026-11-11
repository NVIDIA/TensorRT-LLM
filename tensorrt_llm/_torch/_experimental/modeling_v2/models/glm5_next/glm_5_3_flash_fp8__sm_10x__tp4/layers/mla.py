# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sparse MLA layers: NoPE latent attention over a pool-compressed sparse selection.

A row at cache position ``p`` attends its own prefix sparsely: the ``select_k`` (512) pools of
``index_kpool`` (4) consecutive positions an indexer scores highest among the row's complete
pools, plus the positions of its incomplete trailing pool. The cache holds per position the
RMS-normalized ``kv_a`` latent (512 wide, FP8, no rotary part) and the indexer row ``[k | gate |
pool key]`` (BF16; the pool key lives in a pool's first-member row). Queries and values are
absorbed into the latent space: head ``h`` scores latent ``c`` as ``(q_h @ w_k[h]) . c`` and maps
its latent output ``o`` to ``o @ w_v_t[h]``, ``w_k`` / ``w_v_t`` being the head's key and value
rows of ``kv_b_proj``; the softmax scale is ``qk_nope ** -0.5``.

Weights (``load_mla``). The ``[q_a; kv_a]`` latent projections (one fused 128 x 128 block-FP8
GEMM), the norms and the whole indexer (its ``[wk; compress gate; weights_proj]`` rows fused into
one BF16 GEMM, and ``wq_b``) are replicated. ``q_b_proj``, the absorbed ``kv_b_proj`` and
``o_proj``'s input columns exist twice: this rank's 16 of the 64 query heads (generation rows,
head parallel) and all 64 (sequence-parallel context rows). Products of at most 32 rows with
``[q_a; kv_a]`` or this rank's ``q_b`` / ``o_proj`` run on ``low_m_bf16_gemm`` over a dequantized
BF16 copy of the weight, without quantizing the activation; larger ones, and the all-head
products, quantize the rows 1 x 128 and run the blockwise FP8 GEMM.

Generation rows (``generation_partial``; decode is ``T = 1``, an MTP verify ``T = 4``): the two
input GEMMs, then ``mla_prologue`` (the norms and both cache writes in one launch), then two
branches: the selection (``kpool_update`` of the rows' pools, the indexer query, ``kpool_score``
and ``kpool_select``: top-k, expansion to latent rows and packing in one launch) and the query
(``q_b``, the absorbed key product, ``saturate_fp8``). In a CUDA graph capture the second input
GEMM and the query branch run on the aux stream. Then ``sparse_mla_decode`` with this rank's
heads, ``latent_values`` (zero for a row that selected nothing) and ``o_proj``'s row-parallel
partial, which the caller reduces.

Context rows (``context_rows``): every rank appends every context row's state
(``mla_context_prologue``) and refreshes the pools the step completes (``kpool_update`` at the
pool-final positions), then attends only its own rows ``[lo, hi)`` with all 64 heads: the FP8 pool
scores (``kpool_score_fp8``, in calls of bounded memory), ``indexer_topk``, ``kpool_expand``,
``compact_sparse_rows``, the all-head ``q_b`` and key product, ``sparse_mla_decode``, the value
product quantized 1 x 128 (``quantize_1x128``) and the all-head ``o_proj``: the rows' full output.

Cache contract: ``kv_cache_manager.get_latent_state_buffer(layer)`` (``[slots, tokens_per_block,
1, kv_lora]`` FP8) and ``get_index_state_buffer(layer)`` (``[slots, tokens_per_block, 1, 3 *
index_head_dim]`` BF16) are slot-major views of coalesced paged storage, addressed through the
step's GLM block tables. The sparse attention reads the latent storage as uniformly strided rows:
cache position ``p`` of table ``t`` is row ``base_row + tables[t, p // tokens_per_block] *
rows_per_slot + p % tokens_per_block``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from ..checkpoint import Checkpoint, dequantize_block_fp8, share
from ..dims import Dims
from ..ops.attention.compact_sparse_rows import compact_sparse_rows
from ..ops.attention.head_major import quantize_1x128, saturate_fp8
from ..ops.attention.indexer_topk import indexer_topk
from ..ops.attention.kpool_expand import kpool_expand
from ..ops.attention.kpool_score import kpool_score
from ..ops.attention.kpool_score_fp8 import kpool_score_fp8
from ..ops.attention.kpool_select import kpool_select
from ..ops.attention.kpool_update import kpool_update
from ..ops.attention.latent_values import latent_values
from ..ops.attention.mla_context_prologue import mla_context_prologue
from ..ops.attention.mla_prologue import mla_prologue
from ..ops.attention.sparse_mla_decode import (
    KV_LORA_RANK,
    PAGE_ROWS,
    WORKSPACE_BYTES,
    counter_bytes,
    sparse_mla_decode,
)
from ..ops.gemm.fp8_blockwise_gemm import fp8_blockwise_gemm
from ..ops.gemm.low_m_bf16_gemm import low_m_bf16_gemm
from ..ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128
from ..schedule import StepSchedule

_BLOCK = 128
#: Products of at most this many rows run on ``low_m_bf16_gemm``.
_LOW_M_ROWS = 32
#: The indexer key LayerNorm's epsilon.
_INDEX_KEY_NORM_EPS = 1e-6
#: The expanded selection is padded to the sparse attention's 64-column tiles.
_SELECTION_ALIGN = 64
#: FP32 elements one context scoring call may hold (the padded logits, a several-table call's
#: row copy and the gathered pool keys); larger calls are split.
_SCORE_ELEMENTS = 1 << 26
#: The FP8 latent cache's dequantization scale: the cache is stored unscaled.
_KV_SCALE = 1.0


@dataclass
class MlaHeads:
    """The per-head projections of a set of ``H`` query heads: this rank's 16 (generation rows)
    or all 64 (sequence-parallel context rows)."""

    q_b: torch.Tensor  # [H * qk_nope, q_lora] e4m3
    q_b_scale: torch.Tensor  # [H * qk_nope / 128, q_lora / 128] fp32
    w_k: torch.Tensor  # [H, qk_nope, kv_lora] bf16: kv_b_proj's key rows of each head
    w_v_t: torch.Tensor  # [H, kv_lora, v_head] bf16: its value rows, transposed
    o: torch.Tensor  # [hidden, H * v_head] e4m3: o_proj's input columns of the heads
    o_scale: torch.Tensor  # [hidden / 128, H * v_head / 128] fp32


@dataclass
class MlaWeights:
    """One sparse MLA layer at this rank's share. Block-FP8 weights keep the checkpoint's
    128 x 128 FP32 block scales (``weight_scale_inv``) as stored."""

    qkv_a: torch.Tensor  # [q_lora + kv_lora, hidden] e4m3, rows [q_a; kv_a]
    qkv_a_scale: torch.Tensor  # [(q_lora + kv_lora) / 128, hidden / 128] fp32
    qkv_a_bf16: torch.Tensor  # [q_lora + kv_lora, hidden] bf16, qkv_a dequantized
    q_norm: torch.Tensor  # [q_lora] bf16, q_a_layernorm
    kv_norm: torch.Tensor  # [kv_lora] bf16, kv_a_layernorm
    #: [2 * index_head_dim + index_heads, hidden] bf16, rows [wk; compress gate; weights_proj]
    index_kgw: torch.Tensor
    index_q: torch.Tensor  # [index_heads * index_head_dim, q_lora] bf16, wq_b
    index_k_norm: torch.Tensor  # [index_head_dim] bf16, the key LayerNorm's weight
    index_k_bias: torch.Tensor  # [index_head_dim] bf16, the key LayerNorm's bias
    index_ape: torch.Tensor  # [index_kpool, index_head_dim] bf16, the pool compression bias
    #: This rank's heads; ``q_b``, ``q_b_scale``, ``w_k`` and ``w_v_t`` are views of ``full``'s.
    local: MlaHeads
    full: MlaHeads
    q_b_bf16: torch.Tensor  # [heads_local * qk_nope, q_lora] bf16, local.q_b dequantized
    o_bf16: torch.Tensor  # [hidden, heads_local * v_head] bf16, local.o dequantized


def load_mla(ck: Checkpoint, dims: Dims, layer: int) -> MlaWeights:
    """Read layer ``layer``'s ``self_attn.*`` tensors: a decoder MLA layer or the MTP layer.

    Every tensor is read once and whole. The latent and indexer projections and the norms are
    replicated; ``q_b_proj``, ``kv_b_proj`` and ``o_proj`` give the all-head copies, and this
    rank's head shard is cut from them (``q_b`` rows, ``kv_b`` rows, ``o_proj`` columns).
    """
    prefix = f"{dims.layer_prefix(layer)}self_attn."
    heads, nope, v_head = dims.mla_heads, dims.qk_nope, dims.v_head
    q_lora, kv_lora, hidden = dims.q_lora, dims.kv_lora, dims.hidden
    index_heads, index_dim = dims.index_heads, dims.index_head_dim

    def read_bf16(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        tensor = ck.read(f"{prefix}{name}")
        assert tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == shape, (
            f"{prefix}{name}: {tensor.dtype} {tuple(tensor.shape)}, expected bf16 {shape}"
        )
        return tensor

    def read_fp8(name: str, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        weight = ck.read(f"{prefix}{name}.weight")
        scale = ck.read(f"{prefix}{name}.weight_scale_inv", dtype=torch.float32)
        blocks = (-(-shape[0] // _BLOCK), -(-shape[1] // _BLOCK))
        assert weight.dtype == torch.float8_e4m3fn and tuple(weight.shape) == shape, (
            f"{prefix}{name}: {weight.dtype} {tuple(weight.shape)}, expected block-FP8 {shape}"
        )
        assert tuple(scale.shape) == blocks, (
            f"{prefix}{name}: block scales {tuple(scale.shape)}, expected {blocks}"
        )
        return weight, scale

    q_a, q_a_scale = read_fp8("q_a_proj", (q_lora, hidden))
    kv_a, kv_a_scale = read_fp8("kv_a_proj_with_mqa", (kv_lora, hidden))
    q_b, q_b_scale = read_fp8("q_b_proj", (heads * nope, q_lora))
    o, o_scale = read_fp8("o_proj", (hidden, heads * v_head))
    kv_b = read_bf16("kv_b_proj.weight", (heads * (nope + v_head), kv_lora))
    # The fused [q_a; kv_a] rows keep each source's 128-row scale blocks.
    assert q_lora % _BLOCK == 0, f"q_lora {q_lora} splits a 128-row scale block"
    qkv_a = torch.cat([q_a, kv_a])
    qkv_a_scale = torch.cat([q_a_scale, kv_a_scale])

    per_head = kv_b.view(heads, nope + v_head, kv_lora)
    full = MlaHeads(
        q_b=q_b,
        q_b_scale=q_b_scale,
        w_k=per_head[:, :nope].contiguous(),
        w_v_t=per_head[:, nope:].transpose(1, 2).contiguous(),
        o=o,
        o_scale=o_scale,
    )
    del per_head, kv_b
    h0, h1 = share(heads, dims.tp_size, dims.tp_rank)
    q0, q1, v0, v1 = h0 * nope, h1 * nope, h0 * v_head, h1 * v_head
    assert q0 % _BLOCK == 0 and v0 % _BLOCK == 0, "a head share splits a 128-wide scale block"
    local = MlaHeads(
        q_b=q_b[q0:q1],
        q_b_scale=q_b_scale[q0 // _BLOCK : q1 // _BLOCK],
        w_k=full.w_k[h0:h1],
        w_v_t=full.w_v_t[h0:h1],
        o=o[:, v0:v1].contiguous(),
        o_scale=o_scale[:, v0 // _BLOCK : v1 // _BLOCK].contiguous(),
    )

    index_kgw = torch.cat(
        [
            read_bf16("indexer.wk.weight", (index_dim, hidden)),
            read_bf16("indexer.index_kpool_compress_gate", (index_dim, hidden)),
            read_bf16("indexer.weights_proj.weight", (index_heads, hidden)),
        ]
    )
    return MlaWeights(
        qkv_a=qkv_a,
        qkv_a_scale=qkv_a_scale,
        qkv_a_bf16=dequantize_block_fp8(qkv_a, qkv_a_scale),
        q_norm=read_bf16("q_a_layernorm.weight", (q_lora,)),
        kv_norm=read_bf16("kv_a_layernorm.weight", (kv_lora,)),
        index_kgw=index_kgw,
        index_q=read_bf16("indexer.wq_b.weight", (index_heads * index_dim, q_lora)),
        index_k_norm=read_bf16("indexer.k_norm.weight", (index_dim,)),
        index_k_bias=read_bf16("indexer.k_norm.bias", (index_dim,)),
        index_ape=read_bf16("indexer.index_kpool_compress_ape", (dims.index_kpool, index_dim)),
        local=local,
        full=full,
        q_b_bf16=dequantize_block_fp8(local.q_b, local.q_b_scale),
        o_bf16=dequantize_block_fp8(local.o, local.o_scale),
    )


class SparseAttentionScratch:
    """Buffers every sparse MLA layer of the model shares, allocated before any CUDA graph capture.

    Every sparse attention launch of the model runs on the calling stream, one after another, so
    one workspace and one set of semaphores serve them all:

    - ``workspace``: the sparse attention kernel's split-KV workspace, ``WORKSPACE_BYTES``, zeroed
      once;
    - ``counter(rows, heads)``: its multi-CTA semaphores, zeroed once (the kernel resets them after
      every launch), sized for the engine's token budget: ``max_num_tokens`` generation rows of this
      rank's heads, or a quarter of them as sequence-parallel context rows of all heads;
    - ``kv_scale`` / ``kv_scale_inv``: the FP8 latent cache's dequantization and quantization scales
      (FP32 ``[1]``), and ``kv_scale_value``, the former as a host float.

    Args:
        dims: the model's sizes; ``max_num_tokens`` and the head counts size the semaphores.
    """

    def __init__(self, dims: Dims) -> None:
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        # (row, head) pairs of the largest launch: every row of a step with this rank's heads, or
        # a sequence-parallel quarter of the rows with all heads.
        rows = dims.max_num_tokens
        pairs = max(rows * dims.mla_heads_local, -(-rows // dims.tp_size) * dims.mla_heads)
        self._counter = torch.zeros(
            counter_bytes(pairs, 1, self.sm_count), dtype=torch.uint8, device=self.device
        )
        self._outgrown: list[torch.Tensor] = []
        self.kv_scale_value = _KV_SCALE
        self.kv_scale = torch.full((1,), _KV_SCALE, dtype=torch.float32, device=self.device)
        self.kv_scale_inv = torch.full(
            (1,), 1.0 / _KV_SCALE, dtype=torch.float32, device=self.device
        )

    def counter(self, rows: int, heads: int) -> torch.Tensor:
        """The semaphores of a launch of ``rows`` query rows of ``heads`` heads (uint8, zero
        between launches). A larger launch outside a capture gets a larger buffer; the outgrown
        one stays allocated, since graphs captured on it keep addressing it."""
        need = counter_bytes(rows, heads, self.sm_count)
        if self._counter.numel() < need:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"sparse MLA semaphores for {rows} rows x {heads} heads ({need} bytes) exceed "
                    f"the {self._counter.numel()} allocated before CUDA graph capture"
                )
            self._outgrown.append(self._counter)
            self._counter = torch.zeros(need, dtype=torch.uint8, device=self.device)
        return self._counter


@dataclass(frozen=True)
class _CachePools:
    """One sparse layer's paged cache as the kernels address it.

    ``latent`` and ``index`` are the slot-major pools; ``kv_rows`` views the latent storage as
    uniformly strided rows (a whole number of 32-row pages), where cache position ``p`` of a
    request is row ``base_row + slot(p) * rows_per_slot + p % tokens_per_block``.
    """

    latent: torch.Tensor  # [slots, tokens_per_block, kv_lora] e4m3
    index: torch.Tensor  # [slots, tokens_per_block, 3 * index_head_dim] bf16
    kv_rows: torch.Tensor  # [N, 1, kv_lora] e4m3
    base_row: int
    rows_per_slot: int
    tokens_per_block: int


def _cache_pools(manager: Any, layer_idx: int) -> _CachePools:
    """Layer ``layer_idx``'s pools from the KV cache manager's documented buffers: views only, no
    allocation or device work, so a captured step reads the same storage on replay."""
    latent = manager.get_latent_state_buffer(layer_idx)
    index = manager.get_index_state_buffer(layer_idx)
    if latent is None or index is None:
        raise ValueError(
            f"layer {layer_idx} has no latent / indexer pool in the KV cache manager; the layer "
            "schedule and the cache layout disagree"
        )
    latent, index = latent[:, :, 0, :], index[:, :, 0, :]
    if latent.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "the sparse MLA attends an FP8 latent cache (kv_cache_config.dtype fp8), got "
            f"{latent.dtype}"
        )
    if index.dtype != torch.bfloat16:
        raise ValueError(f"the indexer cache is BF16, got {index.dtype}")
    tokens_per_block = int(manager.tokens_per_block)
    dim = latent.shape[-1]
    if (
        latent.shape[1] != tokens_per_block
        or index.shape[1] != tokens_per_block
        or latent.stride(2) != 1
        or latent.stride(1) != dim
    ):
        raise ValueError(
            f"layer {layer_idx}: latent pool {tuple(latent.shape)} / {latent.stride()} is not "
            f"pages of {tokens_per_block} contiguous rows"
        )
    slot_stride, offset = latent.stride(0), latent.storage_offset()
    if slot_stride % dim or offset % dim:
        raise ValueError(
            f"layer {layer_idx}: latent slot stride {slot_stride} / storage offset {offset} are "
            f"not multiples of the row width {dim}; the storage has no uniform row view"
        )
    base_row, rows_per_slot = offset // dim, slot_stride // dim
    total = latent.untyped_storage().nbytes() // (dim * latent.element_size())
    total -= total % PAGE_ROWS  # the attention reads the rows as 32-row pages
    top = base_row + (latent.shape[0] - 1) * rows_per_slot + tokens_per_block
    if top > total:
        raise ValueError(
            f"layer {layer_idx}: latent rows up to {top} do not fit the {total} whole 32-row "
            "pages of the cache storage"
        )
    kv_rows = torch.as_strided(latent, (total, 1, dim), (dim, dim, 1), storage_offset=0)
    return _CachePools(latent, index, kv_rows, base_row, rows_per_slot, tokens_per_block)


def _score_cost(rows: int, tables: int, pools: int, head_dim: int) -> int:
    """FP32 elements ``kpool_score_fp8`` holds at once for ``rows`` rows over ``tables`` tables
    of ``pools`` pools: the padded logits, a several-table call's row copy, and per gathered key
    its BF16 row, FP8 row, scale and slot index."""
    if pools == 0:
        return 0
    logits = -(-tables * pools // 256) * 256 + 256
    copy = max(1024, -(-pools // 64) * 64) + pools if tables > 1 else 0
    keys = tables * pools * (head_dim // 2 + head_dim // 4 + 3)
    return rows * (logits + copy) + keys + 6 * pools


def _score_calls(
    step: StepSchedule, lo: int, hi: int, kpool: int, head_dim: int
) -> list[tuple[int, int, int, int, int]]:
    """The FP8 scoring calls of the packed context rows ``[lo, hi)``, each within
    ``_SCORE_ELEMENTS``: ``(row_start, row_end, table_start, table_end, max_kv_len)``.

    Consecutive requests share a call while it fits; a request that does not fit alone is split
    by rows. ``max_kv_len`` is the largest visible length among the call's rows (the position + 1
    of each request's last row in it). Host ints only.
    """
    cu, cached = step.ctx_cu_seqlens, step.cached_lens
    calls: list[tuple[int, int, int, int, int]] = []
    group: Optional[list[int]] = None  # [row_start, row_end, table_start, table_end, max_kv_len]

    def cost(rows: int, tables: int, visible: int) -> int:
        return _score_cost(rows, tables, visible // kpool, head_dim)

    for table in range(step.num_contexts):
        first_row = int(cu[table])
        start, end = max(first_row, lo), min(int(cu[table + 1]), hi)
        if start >= end:
            continue
        visible = int(cached[table]) + end - first_row
        if group is not None:
            merged = max(group[4], visible)
            rows = group[1] - group[0] + end - start
            if cost(rows, table + 1 - group[2], merged) <= _SCORE_ELEMENTS:
                group[1], group[3], group[4] = end, table + 1, merged
                continue
            calls.append((group[0], group[1], group[2], group[3], group[4]))
            group = None
        if cost(end - start, 1, visible) <= _SCORE_ELEMENTS:
            group = [start, end, table, table + 1, visible]
            continue
        fixed = cost(0, 1, visible)
        tile = max(1, (_SCORE_ELEMENTS - fixed) // max(1, cost(1, 1, visible) - fixed))
        for row in range(start, end, tile):
            row_end = min(row + tile, end)
            calls.append((row, row_end, table, table + 1, int(cached[table]) + row_end - first_row))
    if group is not None:
        calls.append((group[0], group[1], group[2], group[3], group[4]))
    return calls


def _linear_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x [n, K] @ weight [N, K]^T`` for a BF16 weight: ``low_m_bf16_gemm`` up to
    ``_LOW_M_ROWS`` rows, cuBLAS above."""
    if x.shape[0] <= _LOW_M_ROWS:
        return low_m_bf16_gemm(x, weight)
    return torch.nn.functional.linear(x, weight)


def _fp8_gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x [n, K] @ weight [N, K]^T`` for a 128 x 128 block-FP8 weight: ``x``'s 1 x 128
    quantization and the blockwise FP8 GEMM, BF16 out."""
    x_fp8, x_scale = fp8_quantize_1x128(x)
    return fp8_blockwise_gemm(x_fp8, weight, x_scale, scale)


def _linear_fp8(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, weight_bf16: torch.Tensor
) -> torch.Tensor:
    """``x @ weight^T`` for a block-FP8 weight: up to ``_LOW_M_ROWS`` rows on
    ``low_m_bf16_gemm`` over its dequantized copy ``weight_bf16``, the FP8 GEMM above."""
    if x.shape[0] <= _LOW_M_ROWS:
        return low_m_bf16_gemm(x, weight_bf16)
    return _fp8_gemm(x, weight, scale)


class MlaLayer:
    """One sparse MLA layer's forward: the generation rows' head-parallel partial, and the
    context rows' cache writes and sequence-parallel full output.

    Args:
        w: the layer's weights (``load_mla``).
        dims: the model's sizes and this rank's shares.
        layer_idx: the layer's index (the MTP layer's is ``dims.mtp_layer``), which names its
            latent and indexer pools in the KV cache manager.
        aux_stream: the stream a captured generation step runs its second input GEMM and its
            query branch on; None runs everything on the current stream.
        events: ``(fork, join)``, the events that order the aux stream against the current one.
        scratch: the model's shared sparse attention buffers.
    """

    def __init__(
        self,
        w: MlaWeights,
        dims: Dims,
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream],
        events: tuple[Any, Any],
        scratch: SparseAttentionScratch,
    ) -> None:
        if not dims.index_always_select_tail:
            raise ValueError("the selection always appends the incomplete trailing pool")
        if dims.kv_lora != KV_LORA_RANK:
            raise ValueError(
                f"the sparse attention kernel is {KV_LORA_RANK} wide, not {dims.kv_lora}"
            )
        self.w = w
        self.layer_idx = layer_idx
        self.aux_stream = aux_stream
        self.fork_event, self.join_event = events
        self.scratch = scratch
        self.heads = dims.mla_heads
        self.heads_local = dims.mla_heads_local
        self.qk_nope = dims.qk_nope
        self.rms_eps = dims.rms_eps
        self.index_heads = dims.index_heads
        self.index_head_dim = dims.index_head_dim
        self.kpool = dims.index_kpool
        self.select_k = dims.select_k
        # select_k pools of kpool members, then the up to kpool - 1 tail positions.
        width = dims.index_topk + dims.index_kpool - 1
        self.out_width = -(-width // _SELECTION_ALIGN) * _SELECTION_ALIGN
        self.q_scale = dims.index_head_dim**-0.5
        self.w_scale = dims.index_heads**-0.5
        self.bmm1_scale = scratch.kv_scale_value * dims.qk_nope**-0.5

    def generation_partial(self, x: torch.Tensor, step: StepSchedule) -> torch.Tensor:
        """This rank's ``o_proj`` partial (before the TP reduction) of the step's generation rows.

        Writes the rows' latent and indexer cache rows, refreshes their pools' keys, scores and
        selects pools, and attends the FP8 latent cache with this rank's heads. Graph-capturable:
        shapes follow host ints only.

        Args:
            x: ``[G, hidden]`` bf16, the generation rows (``G = num_generations * T``), request-
                major.
            step: the step's schedule.

        Returns:
            ``[G, hidden]`` bf16.
        """
        w, scratch = self.w, self.scratch
        num_rows = x.shape[0]
        assert num_rows == step.num_generation_tokens, (
            f"{num_rows} rows for {step.num_generation_tokens} generation tokens"
        )
        rows = step.generation_rows(self.kpool)
        pools = _cache_pools(step.metadata.kv_cache_manager, self.layer_idx)
        tpb = pools.tokens_per_block
        first = step.num_contexts
        tables = step.block_tables[first : first + step.num_generations]
        qkv, idx = self._fork_join(
            lambda: _linear_fp8(x, w.qkv_a, w.qkv_a_scale, w.qkv_a_bf16),
            lambda: _linear_bf16(x, w.index_kgw),
        )
        q_resid = mla_prologue(
            qkv,
            idx,
            w.q_norm,
            w.kv_norm,
            w.index_k_norm,
            w.index_k_bias,
            self.rms_eps,
            _INDEX_KEY_NORM_EPS,
            rows.positions,
            tables,
            pools.latent,
            pools.index,
            tpb,
            scratch.kv_scale_inv,
        )
        index_weights = idx[:, 2 * self.index_head_dim :]
        num_pools_max = -(-step.block_tables.shape[1] * tpb // self.kpool)

        def select() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # The rows' pools in position order, so a pool spanning several draft rows ends with
            # all of its visible members.
            kpool_update(
                pools.index,
                tables,
                rows.positions,
                w.index_ape,
                tpb,
                head_dim=self.index_head_dim,
                kpool=self.kpool,
            )
            q_index = _linear_bf16(q_resid, w.index_q).view(
                num_rows, self.index_heads, self.index_head_dim
            )
            scores = kpool_score(
                q_index,
                index_weights,
                pools.index,
                rows.tables,
                rows.visible,
                tpb,
                num_pools_max=num_pools_max,
                head_dim=self.index_head_dim,
                kpool=self.kpool,
                q_scale=self.q_scale,
                w_scale=self.w_scale,
            )
            return kpool_select(
                scores,
                rows.visible,
                rows.tables,
                tpb,
                select_k=self.select_k,
                kpool=self.kpool,
                base_row=pools.base_row,
                rows_per_slot=pools.rows_per_slot,
                out_width=self.out_width,
            )

        def query() -> torch.Tensor:
            q = _linear_fp8(q_resid, w.local.q_b, w.local.q_b_scale, w.q_b_bf16)
            q = q.view(num_rows, self.heads_local, self.qk_nope)
            return saturate_fp8(torch.bmm(q.transpose(0, 1), w.local.w_k))

        (packed, lengths, nonempty), query_fp8 = self._fork_join(select, query)
        latent = sparse_mla_decode(
            query_fp8,
            pools.kv_rows,
            packed,
            lengths,
            scratch.workspace,
            scratch.counter(num_rows, self.heads_local),
            self.bmm1_scale,
            scratch.kv_scale,
        )
        # A row that selected no key (a padded request) gets zero values, so o_proj gives it zero.
        values = latent_values(latent, w.local.w_v_t, nonempty)
        return _linear_fp8(values, w.local.o, w.local.o_scale, w.o_bf16)

    def context_rows(
        self, x: torch.Tensor, step: StepSchedule, lo: int, hi: int, out: torch.Tensor
    ) -> None:
        """Cache every context row, then write the full attention output of rows ``[lo, hi)``.

        Appends every context row's latent and indexer state and refreshes the keys of the pools
        the step completes, as every rank does (the cache is replicated); then attends rows
        ``[lo, hi)`` with all heads and writes their ``o_proj`` output, the full value rather
        than a partial, into ``out[lo:hi]``. Other rows of ``out`` are not written.

        Args:
            x: ``[C, hidden]`` bf16, the step's packed context rows (``C = num_ctx_tokens``).
            step: the step's schedule.
            lo: the first row this rank attends.
            hi: one past the last; ``lo >= hi`` attends none.
            out: ``[>= C, hidden]`` bf16.
        """
        w, scratch = self.w, self.scratch
        assert x.shape[0] == step.num_ctx_tokens, (
            f"{x.shape[0]} rows for {step.num_ctx_tokens} context tokens"
        )
        sched = step.context_rows(self.kpool, x.device)
        pools = _cache_pools(step.metadata.kv_cache_manager, self.layer_idx)
        tpb = pools.tokens_per_block
        tables = step.block_tables
        qkv = _linear_fp8(x, w.qkv_a, w.qkv_a_scale, w.qkv_a_bf16)
        idx = _linear_bf16(x, w.index_kgw)
        q_resid = mla_context_prologue(
            qkv,
            idx,
            w.q_norm,
            w.kv_norm,
            w.index_k_norm,
            w.index_k_bias,
            self.rms_eps,
            _INDEX_KEY_NORM_EPS,
            sched.positions,
            sched.request_ids,
            tables,
            pools.latent,
            pools.index,
            tpb,
            scratch.kv_scale_inv,
        )
        del qkv
        # One refresh per pool the step completes; a pool left open at the end of a chunk is
        # refreshed by the write that completes it.
        if sched.final_positions.shape[0]:
            kpool_update(
                pools.index,
                tables,
                sched.final_positions,
                w.index_ape,
                tpb,
                head_dim=self.index_head_dim,
                kpool=self.kpool,
                request_ids=sched.final_request_ids,
            )
        if lo >= hi:
            return

        num_rows = hi - lo
        request_ids = sched.request_ids[lo:hi]
        visible = sched.positions[lo:hi] + 1
        num_cand = (visible // self.kpool).to(torch.int32)
        q_resid = q_resid[lo:hi]
        index_weights = idx[lo:hi, 2 * self.index_head_dim :]
        q_index = _linear_bf16(q_resid, w.index_q).view(
            num_rows, self.index_heads, self.index_head_dim
        )
        selected = torch.empty(num_rows, self.select_k, dtype=torch.int32, device=x.device)
        for start, end, first, last, max_kv_len in _score_calls(
            step, lo, hi, self.kpool, self.index_head_dim
        ):
            part = slice(start - lo, end - lo)
            ids = request_ids[part]
            scores = kpool_score_fp8(
                q_index[part],
                index_weights[part],
                pools.index,
                tables[first:last],
                visible[part],
                tpb,
                head_dim=self.index_head_dim,
                kpool=self.kpool,
                q_scale=self.q_scale,
                w_scale=self.w_scale,
                request_ids=ids - first if first else ids,
                max_kv_len=max_kv_len,
            )
            if scores is None:
                # No complete pool is visible to these rows: they attend their tail only.
                selected[part] = -1
            else:
                indexer_topk(scores, num_cand[part], selected[part], select_k=self.select_k)
            del scores
        del q_index, index_weights, idx
        topk_rows = kpool_expand(
            selected,
            visible,
            tables,
            tpb,
            base_row=pools.base_row,
            rows_per_slot=pools.rows_per_slot,
            kpool=self.kpool,
            out_width=self.out_width,
            request_ids=request_ids,
        )
        packed, lengths, nonempty = compact_sparse_rows(topk_rows)
        del topk_rows, selected

        full = w.full
        query = _fp8_gemm(q_resid, full.q_b, full.q_b_scale).view(
            num_rows, self.heads, self.qk_nope
        )
        query_fp8 = saturate_fp8(torch.bmm(query.transpose(0, 1), full.w_k))
        del query
        latent = sparse_mla_decode(
            query_fp8,
            pools.kv_rows,
            packed,
            lengths,
            scratch.workspace,
            scratch.counter(num_rows, self.heads),
            self.bmm1_scale,
            scratch.kv_scale,
        )
        del query_fp8
        values, values_scale = quantize_1x128(torch.bmm(latent.transpose(0, 1), full.w_v_t))
        del latent
        projected = fp8_blockwise_gemm(values, full.o, values_scale, full.o_scale)
        # A row that selected no key has an unspecified attention output; it projects to zero.
        out[lo:hi] = torch.where(nonempty[:, None], projected, 0)

    def _fork_join(self, main: Callable[[], Any], aux: Callable[[], Any]) -> tuple[Any, Any]:
        """``(main(), aux())``: in a CUDA graph capture ``aux`` runs on the aux stream beside
        ``main``, otherwise both run in order on the current stream."""
        if self.aux_stream is None or not torch.cuda.is_current_stream_capturing():
            return main(), aux()
        # In a capture the aux stream's outputs need no record_stream: every use of the aux
        # stream starts by waiting for the current stream (the fork).
        self.fork_event.record()
        main_out = main()
        with torch.cuda.stream(self.aux_stream):
            self.fork_event.wait()
            aux_out = aux()
            self.join_event.record()
        self.join_event.wait()
        return main_out, aux_out
