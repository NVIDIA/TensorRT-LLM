# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch golden reference for MiniMax-M3 block-sparse GQA.

:class:`MiniMaxM3VanillaAttention` is a drop-in replacement for the Triton
runtime backend :class:`~.triton_backend.MiniMaxM3SparseRuntimeBackend`. The
sparse registry selects it for ``attn_backend="VANILLA"`` when the
``MiniMaxM3SparseAttentionConfig`` uses ``implementation="triton"`` and
``indexer_kv_dtype="bf16"``; the backend tests compare the Triton path against
it. One sparse-layer forward runs three stages:

1. Cache append (:meth:`MiniMaxM3VanillaAttention.forward_sparse`): the new
   tokens' K, V and index K, plus index V when it is enabled, are written to
   their ``out_cache_loc`` slots.
2. Block selection (:class:`MiniMaxM3VanillaIndexer`): for a query token at
   position p, index head h scores every visible key position s <= p as
   ``idx_q[t, h] . idx_k[s] * idx_sm_scale`` in FP32. A block scores the
   maximum over its visible positions. The first ``init_blocks`` blocks are
   forced with ``_INIT_SCORE`` and the last ``local_blocks`` visible blocks
   with ``_LOCAL_SCORE`` (a block in both sets keeps ``_LOCAL_SCORE``). Each
   index head keeps its top-k blocks, and a KV head attends to the union of
   the blocks its group of index heads selected.
3. Sparse GQA (:meth:`MiniMaxM3VanillaAttention.forward_sparse` through
   ``VanillaAttention._single_token_sparse_attn_forward``): each query head
   attends, with an FP32 softmax scaled by ``sm_scale``, to the visible
   positions of the blocks its KV head selected.

Notation, per rank as in ``MiniMaxM3SparseConfig``: T query tokens packed
request by request; B requests; H_q, H_kv and H_idx query, KV and index heads,
with G = H_idx / H_kv index heads per KV group; D and D_idx the head and index
widths; S = ``block_size``; K = ``topk``. Positions are request-local. Extend
metadata (prefill, mixed batches, multi-token generation) places query token t
of request b at ``p = prefix_lens[b] + t - cu_seqlens_q[b]``; decode metadata
places request b's single token at ``p = seq_lens[b] - 1``. Position p of
request b lives in cache slot ``req_to_token[slot_ids[b], p]``, and an NHD page
view stores slot s at ``[s // tokens_per_block, s % tokens_per_block]``.

Independence and coverage:

- No Triton, MSA or CuTe kernels and no custom ops run here. The only shared
  helpers are the pure-PyTorch ``write_kv_slots`` and
  ``VanillaAttention._single_token_sparse_attn_forward``. The golden shares the
  slot map built by ``MiniMaxM3AttentionMetadata.prepare`` with the Triton
  path, but derives query positions from ``cu_seqlens_q`` and ``prefix_lens``
  instead of reading ``q_positions``.
- Only sparse layers reach this backend. The model computes dense layers with
  its own SDPA path and never calls ``self.attn`` for them. The MSA
  implementation has no golden, and the registry rejects it.
- The model owns the Gemma QK-norm and partial RoPE of both the main and the
  index branch (``support_fused_rope()`` is ``False``) and calls ``forward``
  with ``metadata=None``; the per-step state arrives as ``m3_metadata``.
- The forward reads metadata on the host, so it runs eagerly and refuses CUDA
  graph capture, whereas the Triton path is capture-safe.
- As in the Triton path, ``q_scaling`` is ignored (``sm_scale`` defaults to
  D^-0.5) and the index-V attention output is not computed (the Triton path
  computes it and discards it).
- MSA reduces a KV group's index heads by maximum before its top-k, which
  equals this union only when G == 1.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from ...vanilla import VanillaAttention
from .common import (
    _INIT_SCORE,
    _LOCAL_SCORE,
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseParams,
    write_kv_slots,
)
from .triton_backend import MiniMaxM3SparseRuntimeBackend
from .triton_metadata import MiniMaxM3TritonSparseAttentionMetadata, ensure_metadata_on_device

# Paged cache views this golden reads and writes; the Triton runtime uses NHD.
_KV_LAYOUT = "NHD"
# Padding in a selected-block table.
_UNSELECTED_BLOCK = -1


class _QuerySpan(NamedTuple):
    """Packed query tokens ``[start, end)`` of request ``row``.

    Token ``start`` sits at request-local position ``first_position``; each
    later token sits one position further.
    """

    row: int
    start: int
    end: int
    first_position: int


def _query_spans(
    m3_metadata: MiniMaxM3TritonSparseAttentionMetadata, num_tokens: int
) -> list[_QuerySpan]:
    """Split the packed query tokens into per-request spans.

    Args:
        m3_metadata: Prepared step metadata. Extend metadata places request b's
            tokens at ``cu_seqlens_q[b]:cu_seqlens_q[b + 1]``, starting at
            position ``prefix_lens[b]``. Decode metadata holds one token per
            request at position ``seq_lens[b] - 1``.
        num_tokens: Number of packed query tokens, T.

    Returns:
        One span per request, in batch order.

    Raises:
        ValueError: If the metadata lacks the extend fields or does not
            describe exactly ``num_tokens`` tokens.
    """
    batch = int(m3_metadata.slot_ids.shape[0])
    if not m3_metadata.is_prefill:
        if num_tokens != batch:
            raise ValueError(
                f"MiniMax-M3 decode requires one query token per request, got "
                f"{num_tokens} tokens for {batch} requests"
            )
        seq_lens = m3_metadata.seq_lens[:batch].tolist()
        return [_QuerySpan(row, row, row + 1, seq_lens[row] - 1) for row in range(batch)]

    if m3_metadata.cu_seqlens_q is None or m3_metadata.prefix_lens is None:
        raise ValueError("MiniMax-M3 extend metadata requires cu_seqlens_q and prefix_lens")
    cu_seqlens = m3_metadata.cu_seqlens_q[: batch + 1].tolist()
    prefix_lens = m3_metadata.prefix_lens[:batch].tolist()
    if (
        len(cu_seqlens) != batch + 1
        or cu_seqlens[0] != 0
        or cu_seqlens[-1] != num_tokens
        or any(end < start for start, end in zip(cu_seqlens, cu_seqlens[1:]))
    ):
        raise ValueError(
            f"MiniMax-M3 cu_seqlens_q must rise from 0 to {num_tokens} over {batch} "
            f"requests, got {cu_seqlens}"
        )
    return [
        _QuerySpan(row, cu_seqlens[row], cu_seqlens[row + 1], prefix_lens[row])
        for row in range(batch)
    ]


def _gather_slot_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Read cache rows at flat slot ids.

    Args:
        cache: ``[num_slots, H, D]`` flat-slot cache, or an NHD page view
            ``[num_pages, tokens_per_block, H, D]`` of a paged pool.
        slots: Integer ``[N]`` slot ids.

    Returns:
        ``[N, H, D]`` rows in ``cache.dtype``.
    """
    slots = slots.to(device=cache.device, dtype=torch.long)
    if cache.ndim == 3:
        return cache[slots]
    tokens_per_block = cache.shape[1]
    return cache[slots // tokens_per_block, slots % tokens_per_block]


def _sorted_unique_blocks(blocks: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Sort each row's block ids, drop repeats and move padding to the tail.

    Args:
        blocks: int64 ``[..., W]`` block ids in ``[0, num_blocks)``, with
            ``_UNSELECTED_BLOCK`` as padding.
        num_blocks: Exclusive upper bound of the block ids.

    Returns:
        int64 ``[..., W]``: each row's distinct block ids in ascending order,
        followed by ``_UNSELECTED_BLOCK``.
    """
    # num_blocks is larger than every real id, so padding and repeats sort last.
    ordered = blocks.masked_fill(blocks == _UNSELECTED_BLOCK, num_blocks).sort(dim=-1).values
    repeats = torch.zeros_like(ordered, dtype=torch.bool)
    repeats[..., 1:] = ordered[..., 1:] == ordered[..., :-1]
    ordered = ordered.masked_fill(repeats, num_blocks).sort(dim=-1).values
    return ordered.masked_fill(ordered == num_blocks, _UNSELECTED_BLOCK)


class MiniMaxM3VanillaIndexer:
    """FP32 golden of MiniMax-M3 block selection.

    Reproduces ``_index_attention_and_select`` of the Triton runtime, which
    takes its per-block maxima from a Triton kernel, with plain PyTorch ops:

    - token score ``idx_q[t, h] . idx_k[s] * idx_sm_scale`` in FP32 for every
      visible position s <= p, and ``-inf`` elsewhere;
    - block score: the maximum over the block's visible positions;
    - forced blocks: blocks ``0 .. init_blocks - 1`` score ``_INIT_SCORE``,
      then the last ``local_blocks`` visible blocks score ``_LOCAL_SCORE``, and
      blocks past the visible range score ``-inf`` again;
    - top-k per index head with ``k = min(K, num_blocks)``, where ``-inf``
      entries become ``_UNSELECTED_BLOCK``;
    - union over the G index heads of each KV head.

    When ``init_blocks + local_blocks`` exceeds K, several forced blocks share
    one sentinel score and ``torch.topk`` breaks the tie. The Triton path
    breaks it on a differently shaped tensor, so the two may keep different
    forced blocks; no config validator rules this setting out.

    Args:
        config: Per-rank kernel parameters of the layer.
    """

    def __init__(self, config: MiniMaxM3SparseConfig) -> None:
        self.config = config

    def select_request_blocks(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        visible_lens: torch.Tensor,
        idx_sm_scale: float,
    ) -> torch.Tensor:
        """Select the blocks each query token of one request attends to.

        Args:
            idx_q: ``[n, H_idx, D_idx]`` index queries of the request's query
                tokens.
            idx_k: ``[L, D_idx]`` index keys of request positions ``0 .. L - 1``.
            visible_lens: int64 ``[n]`` count of visible positions, ``p + 1``,
                for each query token; every entry lies in ``[1, L]``.
            idx_sm_scale: Scale applied to the index dot products.

        Returns:
            int64 ``[n, H_kv, G * k]`` block ids per KV head in ascending
            order, padded with ``_UNSELECTED_BLOCK``, where
            ``k = min(K, ceil(L / S))``. Block b covers positions
            ``b * S .. b * S + S - 1``.
        """
        config = self.config
        block_size = config.block_size
        num_tokens, num_index_heads, _ = idx_q.shape
        length = idx_k.shape[0]
        num_blocks = (length + block_size - 1) // block_size
        padded_length = num_blocks * block_size

        # [n, H_idx, L] FP32 token scores, padded to whole blocks with -inf.
        token_scores = torch.einsum("nhd,ld->nhl", idx_q.float(), idx_k.float()) * idx_sm_scale
        token_scores = F.pad(token_scores, (0, padded_length - length), value=float("-inf"))
        positions = torch.arange(padded_length, device=idx_q.device)
        hidden = positions[None, :] >= visible_lens[:, None]
        token_scores = token_scores.masked_fill(hidden[:, None, :], float("-inf"))
        # [n, H_idx, num_blocks]; a block with no visible position stays -inf.
        block_scores = token_scores.view(num_tokens, num_index_heads, num_blocks, block_size)
        block_scores = block_scores.amax(dim=-1)

        block_ids = torch.arange(num_blocks, device=idx_q.device)
        visible_blocks = (visible_lens + block_size - 1) // block_size
        if config.init_blocks > 0:
            block_scores[..., : config.init_blocks] = _INIT_SCORE
        if config.local_blocks > 0:
            local_start = (visible_blocks - config.local_blocks).clamp_min(0)
            local = (block_ids[None, :] >= local_start[:, None]) & (
                block_ids[None, :] < visible_blocks[:, None]
            )
            block_scores = block_scores.masked_fill(local[:, None, :], _LOCAL_SCORE)
        past_visible = block_ids[None, :] >= visible_blocks[:, None]
        block_scores = block_scores.masked_fill(past_visible[:, None, :], float("-inf"))

        topk = min(config.topk, num_blocks)
        scores, selected = block_scores.topk(topk, dim=-1)
        selected = selected.masked_fill(scores == float("-inf"), _UNSELECTED_BLOCK)
        # Index heads h * G .. h * G + G - 1 belong to KV head h.
        group_width = num_index_heads // config.num_kv_heads * topk
        selected = selected.reshape(num_tokens, config.num_kv_heads, group_width)
        return _sorted_unique_blocks(selected, num_blocks)


class MiniMaxM3VanillaAttention(MiniMaxM3SparseRuntimeBackend):
    """Pure-PyTorch golden of the MiniMax-M3 Triton sparse runtime backend.

    This is a drop-in replacement for
    :class:`~.triton_backend.MiniMaxM3SparseRuntimeBackend`: it keeps that
    class's constructor and ``forward_sparse`` signatures and subclasses it,
    because the model routes sparse layers by ``isinstance`` on that class.
    It inherits, unchanged and all in pure Python, the constructor's
    ``m3_config`` and ``disable_index_value``, ``Metadata``
    (``MiniMaxM3AttentionMetadata``), ``support_fused_rope() == False`` and
    ``forward``, which reconciles ``output`` with ``forward_args`` and routes
    to :meth:`forward_sparse`. Pipeline:

    - cache append: :meth:`forward_sparse`, through ``write_kv_slots``;
    - block selection: :meth:`MiniMaxM3VanillaIndexer.select_request_blocks`;
    - sparse GQA: :meth:`forward_sparse`, through
      ``VanillaAttention._single_token_sparse_attn_forward``.

    Caches use the Triton runtime's layouts: NHD page views
    ``[num_pages, tokens_per_block, H, D]`` of the main pool and of the index-K
    pool, or flat ``[num_slots, H, D]`` tensors, all addressed by slot id.
    """

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config=None,
        sparse_params: Optional[MiniMaxM3SparseParams] = None,
        **kwargs,
    ) -> None:
        """Build the golden for one sparse layer.

        Args:
            layer_idx: Index of the attention layer in the model.
            num_heads: Query heads on this rank, H_q.
            head_dim: Head width, D.
            num_kv_heads: KV heads on this rank, H_kv; defaults to
                ``num_heads``.
            quant_config: Kept for signature parity; the golden reads caches by
                plain dtype conversion, as the Triton path does.
            sparse_params: Lowered MiniMax-M3 parameters; they must select the
                Triton implementation with a BF16 index cache.
            **kwargs: Forwarded to ``MiniMaxM3SparseRuntimeBackend`` (for
                example ``q_scaling`` and ``dtype``, which both backends
                ignore).

        Raises:
            ValueError: If ``sparse_params`` is missing or selects a
                configuration this golden does not model.
        """
        self.validate_sparse_params(sparse_params)
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            sparse_params=sparse_params,
            **kwargs,
        )
        self.indexer = MiniMaxM3VanillaIndexer(self.m3_config)

    @classmethod
    def validate_sparse_params(cls, sparse_params: Optional[MiniMaxM3SparseParams]) -> None:
        """Reject MiniMax-M3 configurations that have no golden.

        The golden mirrors the Triton reference: NHD main caches, a BF16 index
        cache and the union of each KV group's index-head selections. The MSA
        implementation (HND caches, an optional FP8 index, a per-group maximum
        before a fixed top-k) runs only under ``attn_backend="TRTLLM"``.

        Args:
            sparse_params: Lowered MiniMax-M3 parameters.

        Raises:
            ValueError: If ``sparse_params`` is missing, selects
                ``implementation="msa"`` or uses an FP8 index cache.
        """
        if sparse_params is None:
            raise ValueError(f"{cls.__name__} requires MiniMax-M3 sparse_params")
        if sparse_params.implementation != "triton":
            raise ValueError(
                f"{cls.__name__} is the golden of the MiniMax-M3 Triton implementation "
                f"(NHD cache, bf16 index, union selection); implementation="
                f"{sparse_params.implementation!r} has no Vanilla reference, so use "
                "attn_backend='TRTLLM' for it"
            )
        if sparse_params.indexer_kv_dtype != "bf16":
            raise ValueError(
                f"{cls.__name__} mirrors the MiniMax-M3 Triton implementation, which "
                f"stores a bf16 index cache; indexer_kv_dtype="
                f"{sparse_params.indexer_kv_dtype!r} is not supported"
            )

    def _validate_inputs(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        m3_metadata: MiniMaxM3TritonSparseAttentionMetadata,
        output: Optional[torch.Tensor],
    ) -> list[_QuerySpan]:
        """Check every input of :meth:`forward_sparse` before any cache write.

        Two checks are stricter than the Triton path, which never sees inputs
        that fail them from valid metadata: a query position at or past its
        request's KV length (or past the slot map) raises instead of being
        clamped, and decode metadata must carry exactly one token per request.

        Returns:
            The per-request query spans of the step.

        Raises:
            ValueError: On any shape, layout, dtype or metadata mismatch.
        """
        config = self.m3_config
        num_tokens = q.shape[0]
        widths = {
            "q": (q, config.num_q_heads * config.head_dim),
            "k": (k, config.num_kv_heads * config.head_dim),
            "v": (v, config.num_kv_heads * config.head_dim),
            "idx_q": (idx_q, config.num_index_heads * config.sparse_index_dim),
            "idx_k": (idx_k, config.sparse_index_dim),
        }
        if idx_v is not None:
            widths["idx_v"] = (idx_v, config.sparse_index_dim)
        for name, (tensor, width) in widths.items():
            if tensor.shape[0] != num_tokens or tensor.numel() != num_tokens * width:
                raise ValueError(
                    f"{name} must hold [{num_tokens}, {width}] values, got shape "
                    f"{tuple(tensor.shape)}"
                )

        caches = [
            ("k_cache", k_cache, config.num_kv_heads, config.head_dim),
            ("v_cache", v_cache, config.num_kv_heads, config.head_dim),
            ("idx_k_cache", idx_k_cache, 1, config.sparse_index_dim),
        ]
        if idx_v_cache is not None:
            caches.append(("idx_v_cache", idx_v_cache, 1, config.sparse_index_dim))
        for name, cache, num_heads, width in caches:
            if cache.ndim not in (3, 4) or tuple(cache.shape[-2:]) != (num_heads, width):
                raise ValueError(
                    f"{name} must be a flat [num_slots, {num_heads}, {width}] cache or an "
                    f"{_KV_LAYOUT} page view [num_pages, tokens_per_block, {num_heads}, "
                    f"{width}], got shape {tuple(cache.shape)}"
                )
            if not cache.dtype.is_floating_point:
                raise ValueError(
                    f"{name} stores packed {cache.dtype} values; packed caches such as "
                    "NVFP4 require the MSA implementation"
                )
        if not self.disable_index_value and idx_v_cache is None:
            raise ValueError("disable_index_value=False requires idx_v_cache to be allocated")

        if out_cache_loc.shape != (num_tokens,) or out_cache_loc.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                f"out_cache_loc must be an int32 or int64 [{num_tokens}] tensor, got "
                f"{out_cache_loc.dtype} {tuple(out_cache_loc.shape)}"
            )
        expected_output = (num_tokens, config.num_q_heads * config.head_dim)
        if output is not None and tuple(output.shape) != expected_output:
            raise ValueError(f"output must have shape {expected_output}, got {tuple(output.shape)}")

        if not isinstance(m3_metadata, MiniMaxM3TritonSparseAttentionMetadata):
            raise ValueError(
                "m3_metadata must be a MiniMaxM3TritonSparseAttentionMetadata, got "
                f"{type(m3_metadata).__name__}"
            )
        spans = _query_spans(m3_metadata, num_tokens)
        seq_lens = m3_metadata.seq_lens[: len(spans)].tolist()
        slot_map_width = m3_metadata.req_to_token.shape[1]
        for span in spans:
            last_position = span.first_position + span.end - span.start - 1
            if span.end > span.start and (
                span.first_position < 0
                or last_position >= seq_lens[span.row]
                or last_position >= slot_map_width
            ):
                raise ValueError(
                    f"MiniMax-M3 request {span.row} has query positions "
                    f"{span.first_position}..{last_position}, outside its KV length "
                    f"{seq_lens[span.row]} or the slot map width {slot_map_width}"
                )
        return spans

    def forward_sparse(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        m3_metadata: MiniMaxM3TritonSparseAttentionMetadata,
        sm_scale: Optional[float] = None,
        idx_sm_scale: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append the new tokens to the caches and run block-sparse GQA.

        Every input is validated before the first cache write. The writes
        then land before any read, so extend rows see their own new tokens.
        Index V is written only when both ``idx_v`` and ``idx_v_cache`` are
        given, as in the Triton path. Each request's cached rows are gathered
        once; its query tokens are scored together and attended one at a
        time. Metadata is read on the host.

        Args:
            q: ``[T, H_q * D]`` queries after QK-norm and RoPE.
            k: ``[T, H_kv * D]`` new keys after QK-norm and RoPE.
            v: ``[T, H_kv * D]`` new values.
            idx_q: ``[T, H_idx * D_idx]`` index queries after norm and RoPE.
            idx_k: ``[T, D_idx]`` new index keys after norm and RoPE.
            idx_v: ``[T, D_idx]`` new index values, or ``None``.
            k_cache: Main-pool K as an NHD page view
                ``[num_pages, tokens_per_block, H_kv, D]`` (``kv_pool[:, 0]``)
                or a flat ``[num_slots, H_kv, D]`` cache.
            v_cache: Main-pool V in the layout of ``k_cache``.
            idx_k_cache: Index-K cache, ``[num_pages, tokens_per_block, 1,
                D_idx]`` or ``[num_slots, 1, D_idx]``.
            idx_v_cache: Flat ``[num_slots, 1, D_idx]`` index-V cache, or
                ``None``; required when ``disable_index_value`` is ``False``.
            out_cache_loc: Integer ``[T]`` cache slot of each new token.
            m3_metadata: Step metadata prepared by
                ``MiniMaxM3AttentionMetadata.prepare``.
            sm_scale: Attention scale; defaults to ``D ** -0.5``.
            idx_sm_scale: Index score scale; defaults to ``D_idx ** -0.5``.
            output: Optional ``[T, H_q * D]`` buffer for the result.

        Returns:
            ``output`` filled in place when given, otherwise a new
            ``[T, H_q * D]`` tensor in ``q.dtype``.

        Raises:
            RuntimeError: Under CUDA graph capture.
            ValueError: If an input or the metadata is inconsistent.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self).__name__} reads metadata on the host and cannot run under "
                "CUDA graph capture"
            )
        config = self.m3_config
        num_tokens = q.shape[0]
        spans = self._validate_inputs(
            q=q,
            k=k,
            v=v,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_v=idx_v,
            k_cache=k_cache,
            v_cache=v_cache,
            idx_k_cache=idx_k_cache,
            idx_v_cache=idx_v_cache,
            out_cache_loc=out_cache_loc,
            m3_metadata=m3_metadata,
            output=output,
        )

        new_rows = [
            (k_cache, k.view(num_tokens, config.num_kv_heads, config.head_dim)),
            (v_cache, v.view(num_tokens, config.num_kv_heads, config.head_dim)),
            (idx_k_cache, idx_k.view(num_tokens, 1, config.sparse_index_dim)),
        ]
        if idx_v is not None and idx_v_cache is not None:
            new_rows.append((idx_v_cache, idx_v.view(num_tokens, 1, config.sparse_index_dim)))
        for cache, rows in new_rows:
            write_kv_slots(cache, out_cache_loc, rows, num_tokens, layout=_KV_LAYOUT)

        sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
        idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
        q = q.view(num_tokens, config.num_q_heads, config.head_dim)
        idx_q = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        m3_metadata = ensure_metadata_on_device(m3_metadata, k_cache.device)
        slot_rows = m3_metadata.req_to_token.index_select(
            0, m3_metadata.slot_ids.to(dtype=torch.long)
        )
        result = q.new_empty((num_tokens, config.num_q_heads, config.head_dim), dtype=torch.float32)
        for span in spans:
            num_span_tokens = span.end - span.start
            if num_span_tokens == 0:
                continue
            length = span.first_position + num_span_tokens
            slots = slot_rows[span.row, :length]
            # The Triton path converts cache reads to FP32 the same way.
            keys = _gather_slot_rows(k_cache, slots).float()
            values = _gather_slot_rows(v_cache, slots).float()
            index_keys = _gather_slot_rows(idx_k_cache, slots)[:, 0]
            visible_lens = torch.arange(
                span.first_position + 1, length + 1, device=index_keys.device
            )
            selected = self.indexer.select_request_blocks(
                idx_q[span.start : span.end], index_keys, visible_lens, idx_sm_scale
            )
            for offset in range(num_span_tokens):
                visible = span.first_position + 1 + offset
                result[span.start + offset] = VanillaAttention._single_token_sparse_attn_forward(
                    q[span.start + offset],
                    keys[:visible],
                    values[:visible],
                    selected[offset],
                    config.block_size,
                    sm_scale,
                )

        result = result.view(num_tokens, config.num_q_heads * config.head_dim)
        if output is None:
            return result.to(q.dtype)
        output.copy_(result)
        return output


__all__ = ["MiniMaxM3VanillaAttention", "MiniMaxM3VanillaIndexer"]
