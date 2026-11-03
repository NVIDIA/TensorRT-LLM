# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA integration for the shared MLA module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.modules.low_m_gemm import planned_low_m_linear
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, is_torch_compiling
from tensorrt_llm._utils import get_sm_version, nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from ..hooks import MLASparseHooks, register_mla_sparse_hooks
from . import shardq
from .decode_prologue import dsa_fp4_decode_prologue
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSABackendForwardArgs

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.mla import MLA

try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

# From this many tokens the packed [q_b_proj | indexer wq_b] projection runs as two GEMMs (see forward_dsa_proj).
_DSA_SPLIT_Q_PROJ_MIN_TOKENS = 128
# Below that, a position-free MXFP4 indexer takes the packed projections' outputs through one kernel
# (decode_prologue.dsa_fp4_decode_prologue: latent-cache concat, MLA-q copy, cat-FP4 of the index q and k,
# FP32 head weights; bitwise the separate kernels). TRTLLM_DSA_FUSED_DECODE_PROLOGUE=0 keeps those kernels.
_DSA_FUSED_DECODE_PROLOGUE = os.environ.get("TRTLLM_DSA_FUSED_DECODE_PROLOGUE", "1") == "1"
# With the fused decode prologue, a module whose ``_dsa_indexer_early`` is set (see MLA.__init__) also shortens the
# chain in front of the indexer: the index-key LayerNorm runs on the aux stream after kv_a_layernorm (the prologue is
# its only consumer), off the q_a norm -> q projection chain, and the prologue writes the keys into the indexer K cache
# itself (the separate scatter launch in _forward_dsa_attn is skipped for that layer). Same values either way.
# There the fused prologue also serves the two-GEMM projection below this many tokens.
_DSA_SPLIT_PROLOGUE_MAX_TOKENS = 1024


def _get_short_seq_mha_threshold() -> int:
    threshold = os.environ.get("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "0")
    try:
        return int(threshold)
    except ValueError as err:
        raise ValueError(
            f"TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD must be an integer, got {threshold!r}"
        ) from err


def _use_flash_mla_sparse_prefill(self) -> bool:
    """Whether DSA context attention runs FlashMLA sparse prefill on SM100 / SM103.

    TRTLLM-Gen tiles a sparse MLA head group of more than 64 heads that is not
    a multiple of 64 (Kimi K3.1's 96 heads under attention DP) in Q32 SwapsAb
    CTAs, one token per CTA, which run prefill at a fraction of the tensor-core
    rate. FlashMLA pads the group to 128 heads and is several times faster on
    those shapes. It needs a BF16 KV cache and no context parallelism; other
    architectures (including SM107) keep TRTLLM-Gen, as the FlashMLA path is
    only measured on SM100 / SM103.
    TRTLLM_DSA_FLASH_MLA_PREFILL=0 keeps TRTLLM-Gen; =1 also takes FlashMLA
    for head groups TRTLLM-Gen tiles natively.
    """
    mode = os.environ.get("TRTLLM_DSA_FLASH_MLA_PREFILL", "auto")
    if mode not in ("0", "1", "auto"):
        raise ValueError(f"TRTLLM_DSA_FLASH_MLA_PREFILL must be 0, 1 or auto, got {mode!r}")
    if mode == "0" or flash_mla_sparse_fwd is None or get_sm_version() not in (100, 103):
        return False
    if (
        getattr(self.mqa, "has_fp8_kv_cache", False)
        or getattr(self.mqa, "has_fp4_kv_cache", False)
        or self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
        or self.mapping.cp_size > 1
        or self.num_heads_tp > 128
    ):
        return False
    if mode == "1":
        return True
    return self.num_heads_tp > 64 and self.num_heads_tp % 64 != 0


def initialize_sparse_attn(self, short_seq_mha_threshold: int) -> None:
    """Initialize DSA module state."""
    self.short_seq_mha_threshold = short_seq_mha_threshold
    self.indexer = getattr(self.mqa, "indexer", None)

    pp_mapping = self.pp_mapping
    num_hidden_layers = self.num_hidden_layers
    if (
        pp_mapping.has_pp()
        and self.layer_idx is not None
        and num_hidden_layers is not None
        and self.indexer is None
    ):
        pp_layers = pp_mapping.pp_layers(num_hidden_layers)
        if pp_layers and self.layer_idx == pp_layers[0]:
            raise ValueError(
                f"DSA pipeline stage {pp_mapping.pp_rank} starts with "
                f"shared-indexer layer {self.layer_idx}, but shared TopK data "
                "is local to a pipeline stage. Choose a pp_partition whose "
                "first layer on every stage runs a full indexer."
            )


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
    if getattr(self.mqa, "use_nvfp4_ds_mla", False):
        # Inline NVFP4 records: only PrimTS reads them, and the KV append happens there too.
        if latent_cache is None or indexer_intermediates is None:
            raise ValueError(
                "nvfp4_ds_mla context attention needs latent_cache and the indexer inputs"
            )
        from .prims_ts_prefill import forward_context_sparse_attn_nvfp4_ds_mla

        return forward_context_sparse_attn_nvfp4_ds_mla(
            self, q, latent_cache, attn_metadata, output, indexer_intermediates
        )
    if getattr(self.mqa, "has_fp4_kv_cache", False):
        if latent_cache is None:
            raise ValueError("NVFP4 DSA context requires latent_cache")
        # Append every uncached context token before sparse prediction. The
        # predictor then dequantizes the union of selected TopK rows into a
        # compact FP8 pool, while the attention op skips its normal KV append.
        self.mqa.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=False
        )
        return self.forward_absorption_context(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=None,
            q_rope_applied=True,
            sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
        )
    if should_use_short_mha(self, attn_metadata, position_ids):
        return self.forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )
    if latent_cache is not None and indexer_intermediates is not None:
        if shardq.use_shardq(self, attn_metadata):
            return shardq.forward_context_sparse_attn_shardq(
                self, q, latent_cache, attn_metadata, output, indexer_intermediates
            )
    if get_sm_version() >= 100 and not _use_flash_mla_sparse_prefill(self):
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
    return forward_sparse_mla_kvcache_bf16(
        self,
        q,
        latent_cache,
        attn_metadata,
        output,
        indexer_intermediates,
        is_generation=False,
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
    if get_sm_version() >= 100:
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
    return forward_sparse_mla_kvcache_bf16(
        self,
        q,
        latent_cache,
        attn_metadata,
        output,
        indexer_intermediates,
        is_generation=True,
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
    latent_cache_gen: Optional[torch.Tensor],
) -> None:
    """Run DSA's two-stage custom ops for piecewise CUDA graph capture."""
    del latent_cache_gen
    from . import custom_ops  # noqa: F401

    if isinstance(hidden_states, Fp4QuantizedTensor):
        proj_outputs = torch.ops.trtllm.mla_dsa_proj(
            hidden_states.unquantized_hidden_states,
            position_ids,
            self.layer_idx_str,
            hidden_states.fp4_tensor,
            hidden_states.scaling_factor,
        )
    else:
        proj_outputs = torch.ops.trtllm.mla_dsa_proj(
            hidden_states, position_ids, self.layer_idx_str
        )
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    custom_ops.maybe_bcg_mla_dsa_attn_inplace(
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        self.layer_idx_str,
        attn_output[0],
    )


def _project_packed_q(self, qr: torch.Tensor, packed_q: torch.Tensor, plan):
    """``(projected_q, split_q)``: the packed [q_b | wq_b] projection of the fused-prologue path, one GEMM (the
    planned kernel or F.linear; ``split_q`` None), or from _DSA_SPLIT_Q_PROJ_MIN_TOKENS rows the MLA and the
    indexer rows as two GEMMs (``projected_q`` the MLA query, ``split_q`` the indexer query; the same values)."""
    projected_q = planned_low_m_linear(plan, qr, packed_q) if plan else None
    if projected_q is not None:
        return projected_q, None
    if qr.shape[0] >= _DSA_SPLIT_Q_PROJ_MIN_TOKENS:
        num_mla_q = packed_q.shape[0] - self.mqa.indexer.wq_b.weight.shape[0]
        split_q = F.linear(qr, packed_q[num_mla_q:])
        return F.linear(qr, packed_q[:num_mla_q]), split_q
    return F.linear(qr, packed_q), None


def _fused_decode_prologue_applies(indexer) -> bool:
    """Whether ``indexer``'s packed-projection glue can run as ``dsa_fp4_decode_prologue``: a position-free
    MXFP4 indexer with BF16 head weights."""
    return (
        _DSA_FUSED_DECODE_PROLOGUE
        and indexer is not None
        and getattr(indexer, "use_fp4", False)
        and getattr(indexer, "rotary_emb", 0) is None
        and getattr(indexer, "head_dim", 0) == 128
        and not is_torch_compiling()
    )


def _prologue_k_cache_scatter(indexer, attn_metadata):
    """``indexer_k_cache_scatter_op``'s cache, slot mappings and row count for ``attn_metadata`` (the
    ``k_cache_scatter`` argument of ``dsa_fp4_decode_prologue``), or ``None`` without a K cache."""
    manager = getattr(attn_metadata, "kv_cache_manager", None)
    slot_fp8 = getattr(attn_metadata, "slot_mapping_fp8", None)
    slot_scale = getattr(attn_metadata, "slot_mapping_scale", None)
    if manager is None or slot_fp8 is None or slot_scale is None:
        return None
    k_cache = manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    return k_cache, slot_fp8, slot_scale, attn_metadata.num_tokens


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
    layer, [q, compressed_kv, k_pe, latent_cache, k_fp8, k_scale] when
    short-MHA handles all tokens (eager only; only the indexer K is
    projected, for the K cache), or
    [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
    weights, q_scale] when the indexer runs.  q_scale is unused on the
    FP8 path but always present so CUDA graph capture sees a stable
    9-tensor shape regardless of indexer dtype.
    """
    assert self.mqa is not None, "DSA is only supported in MQA mode"
    # Set below when the fused prologue writes this call's index keys into the K cache (_forward_dsa_attn reads it).
    self.dsa_k_cache_written = False

    # A model may pack the indexer's wk / weights_proj rows under kv_a_proj and its wq_b rows under q_b_proj
    # (dsa_packed_hidden_weight / dsa_packed_q_weight, plain BF16): one GEMM per input instead of two.
    packed_hidden = getattr(self, "dsa_packed_hidden_weight", None)
    if packed_hidden is not None and isinstance(hidden_states, torch.Tensor):
        indexer = self.mqa.indexer
        plan = getattr(self, "dsa_packed_gemm_plan", None)
        projected_hidden = (
            planned_low_m_linear(plan, hidden_states, packed_hidden) if plan else None
        )
        if projected_hidden is None:
            projected_hidden = F.linear(hidden_states, packed_hidden)
        gate_rows = getattr(self, "dsa_packed_gate_rows", 0)
        q, compressed_kv, k_pe, indexer_k, indexer_weights, *gate = projected_hidden.split(
            [
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                indexer.head_dim,
                indexer.n_heads,
            ]
            + ([gate_rows] if gate_rows else []),
            -1,
        )
        if gate:
            # The output gate's g_proj(hidden_states), packed into this GEMM; _apply_output_gate takes it.
            self.dsa_packed_gate_out = (hidden_states, gate[0])
    else:
        packed_hidden = None
        q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )

    use_short_mha_for_ctx = should_use_short_mha(self, attn_metadata, position_ids)
    packed_q = getattr(self, "dsa_packed_q_weight", None) if packed_hidden is not None else None
    indexer_early = getattr(self, "_dsa_indexer_early", False)
    fused_prologue = (
        packed_q is not None
        and q.shape[0]
        < (_DSA_SPLIT_PROLOGUE_MAX_TOKENS if indexer_early else _DSA_SPLIT_Q_PROJ_MIN_TOKENS)
        and _fused_decode_prologue_applies(self.mqa.indexer)
        and not (use_short_mha_for_ctx and attn_metadata.num_generations == 0)
    )
    index_k = None
    if fused_prologue and indexer_early:
        k_norm = self.mqa.indexer.k_norm
        q_pair, (compressed_kv, index_k) = maybe_execute_in_parallel(
            lambda: self._q_a_layernorm_maybe_fused(q, return_norm_out=True),
            lambda: (self.kv_a_layernorm(compressed_kv), k_norm(indexer_k)),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
    else:
        q_pair, compressed_kv = maybe_execute_in_parallel(
            lambda: self._q_a_layernorm_maybe_fused(q, return_norm_out=True),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
    q, qr = q_pair

    if fused_prologue:
        indexer = self.mqa.indexer
        if index_k is None:
            index_k = indexer.k_norm(indexer_k)
        num_mla_q = packed_q.shape[0] - indexer.wq_b.weight.shape[0]
        projected_q, split_q = _project_packed_q(self, qr, packed_q, plan)
        k_cache_scatter = (
            _prologue_k_cache_scatter(indexer, attn_metadata) if indexer_early else None
        )
        fused = dsa_fp4_decode_prologue(
            projected_q,
            num_mla_q,
            compressed_kv,
            k_pe,
            index_k,
            indexer_weights,
            indexer.weight_scale_factor,
            indexer.n_heads,
            k_cache_scatter=k_cache_scatter,
            indexer_q=split_q,
        )
        if fused is None and k_cache_scatter is not None:
            fused = dsa_fp4_decode_prologue(
                projected_q,
                num_mla_q,
                compressed_kv,
                k_pe,
                index_k,
                indexer_weights,
                indexer.weight_scale_factor,
                indexer.n_heads,
                indexer_q=split_q,
            )
            k_cache_scatter = None
        if fused is None and split_q is not None:
            # Not served: the default path below projects again.
            projected_q = None
        if fused is not None:
            self.dsa_k_cache_written = k_cache_scatter is not None
            q, latent_cache, q_fp4, q_scale, k_fp4, k_scale, weights = fused
            return [
                q,
                compressed_kv,
                k_pe,
                latent_cache,
                q_fp4.view(-1, indexer.n_heads, indexer.head_dim // 2),
                k_fp4,
                k_scale,
                weights,
                q_scale.view(-1, indexer.n_heads, 1),
            ]
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

    if packed_q is not None:
        num_indexer_q = self.mqa.indexer.wq_b.weight.shape[0]
        num_mla_q = packed_q.shape[0] - num_indexer_q
        if not fused_prologue or projected_q is None:
            projected_q = planned_low_m_linear(plan, qr, packed_q) if plan else None
        if projected_q is None and qr.shape[0] >= _DSA_SPLIT_Q_PROJ_MIN_TOKENS:
            # Two GEMMs on row slices of the packed weight write both outputs contiguous (the same values as
            # the packed GEMM's column slices); copying those strided slices costs more than the second launch.
            q = F.linear(qr, packed_q[:num_mla_q])
            indexer_q = F.linear(qr, packed_q[num_mla_q:])
        else:
            if projected_q is None:
                projected_q = F.linear(qr, packed_q)
            q, indexer_q = projected_q.split([num_mla_q, num_indexer_q], -1)
            # The MLA kernels take q_b_proj's output as a contiguous tensor; the indexer's q is only reshaped.
            q = q.contiguous()
    else:
        q = self.q_b_proj(q)

    # DSA "shared" layer: no indexer projection; the backend predictor
    # reuses the preceding full layer's top-k.
    if self.mqa.indexer is None:
        return [q, compressed_kv, k_pe, latent_cache]

    # When the short MHA path handles all context tokens and there are no
    # generation tokens, no TopK is needed: skip the indexer Q, weights and
    # logits, but still project the keys, which later steps read from the
    # indexer K cache.
    if use_short_mha_for_ctx and attn_metadata.num_generations == 0:
        k_fp8, k_scale = self.mqa.indexer.pre_indexer_k_proj(
            hidden_states,
            position_ids,
            indexer_k=indexer_k if packed_hidden is not None else None,
        )
        return [q, compressed_kv, k_pe, latent_cache, k_fp8, k_scale]

    # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
    # compute (cublas_mm, rope, FP4/FP8 quantize, weight scaling) with no
    # access to batch-specific metadata or the k cache. Returns q_scale
    # as a 5th element so the FP4 dispatch can forward it to the kernel;
    # the FP8 path ignores it in _forward_dsa_attn.
    projected = None
    if packed_hidden is not None:
        projected = (indexer_k, indexer_weights, indexer_q if packed_q is not None else None)
    q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
        qr, hidden_states, position_ids, projected=projected
    )

    return [
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        q_scale,
    ]


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

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale]
    when the indexer ran in Op 1, [k_fp8, k_scale] when short-MHA handled
    all tokens (only the indexer K cache is written), or [] for a
    shared-indexer layer.

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
    q = q[:num_tokens, ...]
    compressed_kv = compressed_kv[:num_tokens, ...]
    k_pe = k_pe[:num_tokens, ...]
    latent_cache = latent_cache[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    use_short_mha_for_ctx = num_contexts > 0 and should_use_short_mha(
        self, attn_metadata, position_ids
    )

    if not (use_short_mha_for_ctx and num_generations == 0) and self.mqa.indexer is not None:
        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        q_scale = q_scale[:num_tokens, ...]
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        # Update the full mixed batch once before context/generation prediction
        # splits. Context chunk gathers must observe the newly appended keys.
        # (forward_dsa_proj's fused prologue may already have written them.)
        if getattr(self, "dsa_k_cache_written", False):
            self.dsa_k_cache_written = False
        else:
            self.mqa.indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
    elif self.mqa.indexer is not None:
        # Short MHA attends without the indexer TopK, but later steps read
        # these tokens' keys from the indexer K cache.
        k_fp8, k_scale = indexer_intermediates
        self.mqa.indexer._update_k_cache(
            k_fp8[:num_tokens, ...], k_scale[:num_tokens, ...], attn_metadata
        )
        indexer_intermediates = None
    else:
        indexer_intermediates = None

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None
        if self.apply_rotary_emb:
            assert ctx_position_ids is not None
            k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)

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
        if self.apply_rotary_emb:
            assert gen_position_ids is not None
            k_pe_gen = self.apply_rope(q_gen, k_pe_gen, gen_position_ids)

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

    # Join the prev_topk copy forked in sparse_attn_indexer; CUDA graph
    # capture requires the join within the same layer's forward.
    if self.mqa.indexer is not None:
        self.mqa.indexer.maybe_join_prev_topk_copy()


def should_use_short_mha(
    self, attn_metadata: AttentionMetadata, position_ids: Optional[torch.Tensor]
) -> bool:
    """Check if the short-seq MHA optimization should be used for context.

    Uses max_ctx_kv_len (max total KV length per context sequence,
    including cached tokens) when available, to correctly account for
    chunked context where the full attention span exceeds the threshold
    even if the new token count is small.  Falls back to num_ctx_tokens
    (total new context tokens) when max_ctx_kv_len is not set.

    Disabled under torch compile so that the split DSA custom ops
    (mla_dsa_proj / mla_dsa_attn_inplace) have unconditionally
    straight-line control flow for CUDA graph capture.
    """
    if is_torch_compiling():
        return False
    if not (
        self.short_seq_mha_threshold > 0
        and self.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla")
        and not self.apply_rotary_emb
        and self.mapping.cp_size == 1
        and position_ids is not None
    ):
        return False
    effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
    return effective_len <= self.short_seq_mha_threshold


@nvtx_range("forward_sparse_mla_kvcache_bf16")
def forward_sparse_mla_kvcache_bf16(
    self,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    indexer_intermediates: Optional[List[torch.Tensor]],
    is_generation: bool = False,
) -> torch.Tensor:
    """
    Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

    To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
    1. Append current tokens to paged cache and apply rope to q/k via mla_rope_append_paged_kv_assign_q
    2. Load full kv cache from paged memory (with k rope applied)
    3. Call FlashMLA sparse attention kernel for sparse prefill/decode
    """
    assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), (
        "DSA requires DSAtrtllmAttentionMetadata"
    )
    attention_input_type = (
        AttentionInputType.generation_only if is_generation else AttentionInputType.context_only
    )
    topk_indices_pool, _ = self.mqa.sparse_attn_predict(
        q,
        None,
        attn_metadata,
        AttentionForwardArgs(
            attention_input_type=attention_input_type,
            sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
        ),
    )
    assert topk_indices_pool is not None
    attn_metadata._ensure_pool_view_cached()
    kv_cache_pool = attn_metadata._cached_pool_view
    assert kv_cache_pool is not None

    # Append current tokens to paged cache and apply RoPE to q
    # This writes latent_cache to paged KV and modifies q in-place
    trtllm_attention = self.mqa
    with nvtx_range_debug(f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"):
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=is_generation
        )

    num_tokens = q.shape[0]
    q_nope, q_rope = q.view(-1, self.num_heads_tp, self.qk_head_dim).split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    q_nope_out = torch.empty(
        [num_tokens, self.num_heads_tp, (self.kv_lora_rank)],
        dtype=q.dtype,
        device=q.device,
    )

    if self.k_b_proj_trans.dtype == torch.bfloat16:
        # [num_heads, num_tokens, self.qk_nope_head_dim]
        q_nope_t = q_nope.transpose(0, 1)
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = q_nope_out.transpose(0, 1)

        # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
        # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
        # The output of bmm is written directly into fused_q
        self._bmm_bf16_out(
            q_nope_t,
            self.k_b_proj_trans,
            self.k_b_proj_trans.transpose(1, 2),
            q_nope_out,
        )
    elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = q_nope_out.transpose(0, 1)

        from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out

        fp8_block_scaling_bmm_out(
            q_nope,
            self.k_b_proj_trans,
            self.k_b_proj_trans_scale,
            q_nope_out,
            self.k_b_proj_trans_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

    q_nope_out = q_nope_out.transpose(0, 1)

    sm_version = get_sm_version()
    # FlashMLA sparse kernel (bf16) requires num_heads=128 on sm100 or multiple of 64 on sm90
    if sm_version >= 100:
        padding = 128
        assert self.num_heads_tp <= padding, (
            f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
            f"got {self.num_heads_tp}. Padding from values > {padding} is not supported."
        )
    else:  # SM90
        padding = ((self.num_heads_tp + 63) // 64) * 64  # multiple of 64

    if self.num_heads_tp != padding:
        logger.warning_once(
            f"Padding num_heads from {self.num_heads_tp} to {padding} "
            f"due to FlashMLA sparse attention kernel requirement",
            key="sparse_mla_padding_warning",
        )
        # The padded heads stay uninitialized: FlashMLA computes every head
        # independently and their outputs are sliced off below.
        q_concat = q_nope_out.new_empty(
            (num_tokens, padding, self.kv_lora_rank + self.qk_rope_head_dim)
        )
        q_concat[:, : self.num_heads_tp, : self.kv_lora_rank].copy_(q_nope_out)
        q_concat[:, : self.num_heads_tp, self.kv_lora_rank :].copy_(q_rope)
    else:
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

    topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
    if flash_mla_sparse_fwd is not None:
        attn_out_latent = flash_mla_sparse_fwd(
            q_concat, kv_cache_pool, topk_indices_pool, self.softmax_scale
        )[0]
    else:
        raise RuntimeError(
            "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
        )

    # [seq, num_heads, kv_lora_rank], account for padding
    attn_out_latent = attn_out_latent[:, : self.num_heads_tp, :]
    attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp, self.kv_lora_rank])
    if self.num_heads_tp != padding:
        attn_out_latent = attn_out_latent.contiguous()

    assert attn_out_latent.shape[0] == q.shape[0] and attn_out_latent.shape[1] == self.num_heads_tp

    attn_output = output.view([num_tokens, self.num_heads_tp, self.v_head_dim])

    if self.v_b_proj.dtype == torch.bfloat16:
        # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
        # -> [num_heads, seq, v_head_dim]
        self._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            self.v_b_proj,
            self.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )
    elif self.v_b_proj.dtype == torch.float8_e4m3fn:
        from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out

        fp8_block_scaling_bmm_out(
            attn_out_latent,
            self.v_b_proj,
            self.v_b_proj_scale,
            attn_output.transpose(0, 1),
            self.v_b_proj_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")
    return output


class DSAHooks(MLASparseHooks):
    """Typed DSA adapter for the shared MLA module."""

    need_absorption = True

    def __init__(self) -> None:
        self.short_seq_mha_threshold = _get_short_seq_mha_threshold()
        self.need_dense_mha = self.short_seq_mha_threshold > 0

    def initialize(self, mla: MLA) -> None:
        initialize_sparse_attn(mla, self.short_seq_mha_threshold)
        self.need_dense_mha = (
            self.need_dense_mha and mla.mapping.cp_size == 1 and mla.mqa.support_fused_rope()
        )

    def forward(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
    ) -> None:
        forward_sparse_attn(mla, position_ids, hidden_states, attn_metadata, attn_output)

    def forward_custom_op(
        self,
        mla: MLA,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor],
    ) -> bool:
        forward_sparse_attn_custom_op(
            mla,
            hidden_states,
            position_ids,
            attn_output,
            latent_cache_gen,
        )
        return True


register_mla_sparse_hooks("dsa", DSAHooks)
