# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA path of the Kimi K3.1 MLA: its projections and attention as two custom ops (opaque to graph tracing; the
attention is the breakable CUDA graph's eager cut point), and the eager forward the MLA modules call."""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_fp8_decode_prologue import (
    supports_cache_layout as _prologue_supports_cache,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_fp8_decode_prologue import (
    dsa_fp8_decode_prologue,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_packed_proj import (
    PACKED_GATE_ROWS,
    PACKED_HIDDEN_WEIGHT,
    forward_dsa_proj_packed,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    DSABackendForwardArgs,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (
    fused_cat_fp8,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.mla_outputs import (
    extract_mla_extra_attrs,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import (
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm.logger import logger

#: Op 1 indexer outputs of a layer whose index quantization runs in Op 2's fused prologue: [index_q, index_k]
#: (BF16) instead of [q_fp8, k_fp8, k_scale, weights]; its k_pe slot then carries the packed
#: projection's [k_pe | wk | weights_proj (| g_proj)] columns (``forward_dsa_proj_packed``).
_DEFERRED_INDEXER_OUTPUTS = 2


def _fused_index_prologue_layer(mla) -> bool:
    """Whether this DSA MLA layer quantizes its indexer Q / K in Op 2's fused FP8 prologue: a layer of the
    SM 10.3 perf targets' backend (``mqa._k31_perf``) with a full indexer and the packed hidden-state
    projection, whose wq_b / index-key / index-weights outputs the prologue consumes."""
    mqa = getattr(mla, "mqa", None)
    return (
        getattr(mqa, "_k31_perf", False)
        and getattr(mqa, "indexer", None) is not None
        and getattr(mla, PACKED_HIDDEN_WEIGHT, None) is not None
    )


# ====================================================================================================================
# Custom ops of the DSA MLA
# ====================================================================================================================


class DeferredQB(NamedTuple):
    """A target MLA's request to run its ``q_b`` projection in Op 2 instead of Op 1.

    A target sets ``dsa_deferred_q_b`` on its MLA module to one of these (the default is None: Op 1 projects q).
    Op 1 then returns the ``q_lora_rank``-wide normalized q for calls of at most ``max_rows`` rows, and Op 2 runs
    ``project`` on it unless ``projected_in_generation(q, attn_metadata)`` says the generation path runs it.
    """

    max_rows: int
    project: Callable[[torch.Tensor], torch.Tensor]
    projected_in_generation: Callable[[torch.Tensor, AttentionMetadata], bool]


def _deferred_q_b(mla, num_tokens: int) -> bool:
    """Whether Op 1 leaves this layer's ``q_b`` projection to Op 2 for a call of ``num_tokens`` rows."""
    deferral = mla.dsa_deferred_q_b
    return (
        deferral is not None
        and getattr(mla, PACKED_HIDDEN_WEIGHT, None) is not None
        and 0 < num_tokens <= deferral.max_rows
    )


@torch.library.custom_op("trtllm::k31_mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and indexer.pre_indexer_proj
    (FP8 quantize, weight scaling).  Does NOT update the indexer k cache —
    that happens in Op 2 (mla_dsa_attn_inplace) because the scatter kernel
    accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache] followed by the indexer
    intermediates: none for a shared-indexer layer, [index_q, index_k] (BF16)
    for a layer whose index quantization runs in Op 2's fused prologue (its
    k_pe slot then carries the packed projection's columns), else
    [q_fp8, k_fp8, k_scale, weights]. The length is fixed per layer, keeping
    control flow straight-line for CUDA graph capture. q is
    ``q_lora_rank`` wide when the layer defers q_b to Op 2 (``DeferredQB``).
    """
    metadata, mla_layer = extract_mla_extra_attrs(layer_idx)
    return forward_dsa_proj(mla_layer, position_ids, hidden_states, metadata)


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    # Mirrors forward_dsa_proj: 4 attention inputs + 0, 2 or 4 indexer intermediates.
    metadata, mla_layer = extract_mla_extra_attrs(layer_idx)
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q_width = (
        mla_layer.q_lora_rank
        if _deferred_q_b(mla_layer, num_tokens)
        else mla_layer.num_heads_tp * mla_layer.qk_head_dim
    )
    q = hidden_states.new_empty([num_tokens, q_width])
    compressed_kv = hidden_states.new_empty([num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim]
    )
    if indexer is None:
        # DSA "shared" layer: no indexer, mirror forward_dsa_proj's early
        # return of only the 4 base tensors (no indexer intermediates).
        return [q, compressed_kv, k_pe, latent_cache]
    if _fused_index_prologue_layer(mla_layer):
        # BF16 index projections for Op 2's fused FP8 prologue; the k_pe slot carries the packed
        # [k_pe | wk | weights_proj (| g_proj)] columns.
        tail = mla_layer.qk_rope_head_dim + indexer.head_dim + indexer.n_heads
        kpe_tail = hidden_states.new_empty(
            [num_tokens, tail + getattr(mla_layer, PACKED_GATE_ROWS, 0)]
        )
        index_q = hidden_states.new_empty([num_tokens, indexer.n_heads, indexer.head_dim])
        index_k = hidden_states.new_empty([num_tokens, indexer.head_dim])
        return [q, compressed_kv, kpe_tail, latent_cache, index_q, index_k]
    # Indexer intermediates: q_fp8, k_fp8, k_scale, weights (FP8 index keys, one float32 scale per token).
    q_fp8 = hidden_states.new_empty(
        [num_tokens, indexer.n_heads, indexer.head_dim], dtype=torch.float8_e4m3fn
    )
    k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim], dtype=torch.float8_e4m3fn)
    k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


@torch.library.custom_op("trtllm::k31_mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is Op 1's tail: [q_fp8, k_fp8, k_scale, weights]
    when the indexer quantized in Op 1, [index_q, index_k] when Op 2's fused
    prologue quantizes, [] for a shared-indexer layer. Runs
    sparse_attn_indexer then dispatches context/generation attention. This
    op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_mla_extra_attrs(layer_idx)
    _forward_dsa_attn(
        mla_layer,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        metadata,
        output,
    )
    if mla_layer.zero_padding_output_rows and output.shape[0] > metadata.num_tokens:
        output[metadata.num_tokens :].zero_()


@mla_dsa_attn_inplace.register_fake
def _mla_dsa_attn_inplace_fake(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Model the in-place output mutation during fake-tensor propagation."""


maybe_bcg_mla_dsa_attn_inplace = eager_on_graph(mla_dsa_attn_inplace)


# ====================================================================================================================
# The DSA path of the targets' MLA module
# ====================================================================================================================

# These take the target's MLA module (``self`` below) without importing the target.


def initialize_sparse_attn(self) -> None:
    """Initialize DSA module state."""
    self.indexer = getattr(self.mqa, "indexer", None)


def forward_context_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    indexer_intermediates: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return self.forward_absorption_context(
        q,
        compressed_kv,
        k_pe,
        attn_metadata,
        output,
        position_ids=position_ids,
        latent_cache=latent_cache,
        sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
    )


def forward_generation_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    indexer_intermediates: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return self.forward_absorption_generation(
        q,
        compressed_kv,
        k_pe,
        attn_metadata,
        output,
        position_ids=position_ids,
        latent_cache=latent_cache,
        sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
    )


def forward_sparse_attn(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    attn_output: list[torch.Tensor],
) -> None:
    """
    Forward pass for the MLA module with DSA (always in MQA mode).
    Writes result into output tensor in-place.

    Delegates to forward_dsa_proj (token-wise projections) followed by
    _forward_dsa_attn (batch-dependent attention dispatch).

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.
        attn_output: Algorithm-defined output tensors. DSA writes to the
            standard output at index zero.
    """
    output = attn_output[0]
    proj_outputs = forward_dsa_proj(self, position_ids, hidden_states, attn_metadata)
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    _forward_dsa_attn(
        self,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        attn_metadata,
        output,
    )


def forward_sparse_attn_custom_op(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attn_output: list[torch.Tensor],
) -> None:
    """Run DSA's two-stage custom ops (the attention runs eagerly under a breakable CUDA graph)."""
    proj_outputs = torch.ops.trtllm.k31_mla_dsa_proj(
        hidden_states, position_ids, self.layer_idx_str
    )
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    maybe_bcg_mla_dsa_attn_inplace(
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        self.layer_idx_str,
        attn_output[0],
    )


def forward_dsa_proj(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable Op 1).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj().

    IMPORTANT: This method must NOT slice tensors by num_tokens or
    access batch-specific metadata, so that all operations are
    unconditionally straight-line for CUDA graph capture.  Slicing
    to num_tokens happens in _forward_dsa_attn (Op 2, outside graph).

    Returns [q, compressed_kv, k_pe, latent_cache] for a shared-indexer
    layer, else [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8,
    k_scale, weights]; a packed layer returns ``forward_dsa_proj_packed``'s
    list (see ``mla_dsa_proj``).
    """
    assert self.mqa is not None, "DSA is only supported in MQA mode"
    if getattr(self, PACKED_HIDDEN_WEIGHT, None) is not None:
        # The layer's hidden-state projections were packed after the weight load (pack_dsa_hidden_projection).
        return forward_dsa_proj_packed(
            self,
            position_ids,
            hidden_states,
            attn_metadata,
            getattr(self, "_k31_low_m_plan", None),
            True,
            defer_index_quant=_fused_index_prologue_layer(self),
            defer_q_b=_deferred_q_b(self, hidden_states.shape[0]),
        )
    q, compressed_kv, k_pe = self._project_qkv_a(hidden_states).split(
        [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
    )
    q_pair, compressed_kv = maybe_execute_in_parallel(
        lambda: self._q_a_layernorm(q),
        lambda: self._kv_a_layernorm(compressed_kv),
        self.ln_events[0],
        self.ln_events[1],
        self.aux_stream,
    )
    q, qr = q_pair
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
    q = self.q_b_proj(q)

    # DSA "shared" layer: no indexer projection; the backend predictor
    # reuses the preceding full layer's top-k.
    if self.mqa.indexer is None:
        return [q, compressed_kv, k_pe, latent_cache]

    # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
    # compute (GEMMs, FP8 quantize, weight scaling) with no access to
    # batch-specific metadata or the k cache.
    q_fp8, k_fp8, k_scale, weights = self.mqa.indexer.pre_indexer_proj(qr, hidden_states)

    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


def _forward_dsa_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention for DSA MLA (Op 2, not graph-captured).

    indexer_intermediates is Op 1's tail (see ``mla_dsa_attn_inplace``).

    All num_tokens slicing happens here (not in Op 1) because
    num_tokens comes from batch-specific metadata and must not be
    baked into CUDA graph capture.
    """
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens

    # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
    # full padded tensor for CUDA graph compatibility).
    if q.shape[-1] != self.num_heads_tp * self.qk_head_dim:
        # Op 1 left q_b to this op (``DeferredQB``); unless the generation path runs it itself, the same GEMM on
        # the same rows here.
        deferral = self.dsa_deferred_q_b
        if not deferral.projected_in_generation(q, attn_metadata):
            q = deferral.project(q)
    q = q[:num_tokens, ...]
    compressed_kv = compressed_kv[:num_tokens, ...]
    k_pe = k_pe[:num_tokens, ...]
    latent_cache = latent_cache[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    if self.mqa.indexer is not None and len(indexer_intermediates) == _DEFERRED_INDEXER_OUTPUTS:
        # Fused FP8 prologue: quantize the index Q / K, scale the head weights, write the index keys
        # into the cache and (with them) the latent rows, for the whole mixed batch in one launch.
        # The k_pe slot carries the packed [k_pe | wk | weights_proj (| g_proj)] columns.
        indexer = self.mqa.indexer
        kpe_tail = k_pe
        k_pe = kpe_tail[:, : self.qk_rope_head_dim]
        w0 = self.qk_rope_head_dim + indexer.head_dim
        index_q, index_k = indexer_intermediates
        indexer_intermediates, latent_cache = _fused_index_prologue(
            self,
            index_q,
            index_k,
            kpe_tail[:, w0 : w0 + indexer.n_heads],
            compressed_kv,
            k_pe,
            attn_metadata,
            num_tokens,
        )
    elif self.mqa.indexer is not None:
        q_fp8, k_fp8, k_scale, weights = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights]
        # Update the full mixed batch once before context/generation prediction
        # splits. Context chunk gathers must observe the newly appended keys.
        self.mqa.indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
    else:
        indexer_intermediates = None

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None

        forward_context_sparse_attn(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            latent_cache_ctx,
            position_ids=ctx_position_ids,
            indexer_intermediates=indexer_intermediates,
        )

    if num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        gen_position_ids = (
            position_ids[..., num_ctx_tokens:num_tokens] if position_ids is not None else None
        )

        forward_generation_sparse_attn(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            latent_cache=latent_cache_gen,
            position_ids=gen_position_ids,
            indexer_intermediates=indexer_intermediates,
        )


def _fused_index_prologue(
    self,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    num_tokens: int,
):
    """The indexer intermediates ``[q_fp8, k_fp8, k_scale, weights]`` and the latent rows from
    Op 1's BF16 index projections (``index_q`` ``[T, heads, 128]``, ``index_k`` ``[T, 128]``,
    ``index_weights`` ``[T, heads]``; first ``num_tokens`` rows).

    One ``dsa_fp8_decode_prologue`` launch replaces ``fused_cat_fp8`` on Q and K, the weight product,
    ``K31Indexer._update_k_cache`` and the latent concat, with the same bits. Without a cache update this step
    (no indexer cache, no slot mappings, or a cache view the kernel does not take) it runs those ops.
    """
    indexer = self.mqa.indexer
    heads, head_dim = indexer.n_heads, indexer.head_dim
    manager = attn_metadata.kv_cache_manager
    k_cache = None
    if manager is not None and attn_metadata.slot_mapping_fp8 is not None:
        k_cache = manager.get_indexer_k_cache_buffers(indexer.layer_idx)
        if not _prologue_supports_cache(k_cache):
            k_cache = None
    if k_cache is None:
        q_fp8, q_scale = fused_cat_fp8(
            *index_q[:num_tokens].split([head_dim // 2, head_dim - head_dim // 2], dim=-1)
        )
        k_fp8, k_scale = fused_cat_fp8(
            *index_k[:num_tokens].split([head_dim // 2, head_dim - head_dim // 2], dim=-1)
        )
        q_fp8 = q_fp8.view(-1, heads, head_dim)
        q_scale = q_scale.view(-1, heads, 1)
        weights = indexer._weight_scale(index_weights[:num_tokens], q_scale)
        indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
        return [q_fp8, k_fp8, k_scale, weights], latent_cache
    # The latent rows ride along when both inputs have 16-byte aligned BF16 rows of widths 8k.
    fuse_latent = all(
        t.dtype == torch.bfloat16
        and t.stride(-1) == 1
        and t.stride(0) % 8 == 0
        and t.shape[-1] % 8 == 0
        and t.data_ptr() % 16 == 0
        for t in (compressed_kv, k_pe)
    )
    logger.info_once(
        "Kimi K3.1 DSA indexer: fused FP8 prologue (index Q / K "
        "quantization, head weights, index-key cache write and latent rows in one launch)",
        key="k31_dsa_fused_index_prologue",
    )
    q_fp8, _, weights, k_fp8, k_scale, latent_cache = dsa_fp8_decode_prologue(
        index_q,
        index_k,
        index_weights,
        indexer.weight_scale_factor,
        k_cache,
        attn_metadata.slot_mapping_fp8,
        attn_metadata.slot_mapping_scale,
        num_tokens,
        compressed_kv if fuse_latent else None,
        k_pe if fuse_latent else None,
    )
    if latent_cache is None:
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
    q_fp8 = q_fp8.view(num_tokens, heads, head_dim)
    return [q_fp8, k_fp8, k_scale, weights], latent_cache
