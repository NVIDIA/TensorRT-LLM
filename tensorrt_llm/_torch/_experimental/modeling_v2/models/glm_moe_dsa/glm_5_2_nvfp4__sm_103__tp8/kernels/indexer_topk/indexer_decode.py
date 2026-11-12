# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""indexer_topk: GEN indexer logits and top-k selection for target full-indexer layers.

The index_selection contract: native projected indexer query/weights, the
immutable indexer K cache view and live KV lengths in; native-compatible
logical top-k IDs ``[R, index_topk]`` (``-1`` marks invalid slots) out. Cache
append, cross-layer reuse, MTP last-accepted-row selection and GVR prior
state stay with the native caller.

Every verify row of a request is scored in one paged-MQA
logits launch, so the indexer K cache is streamed once per request instead of
once per row group (the native caller splits ``next_n`` rows into groups of at
most four and re-reads the cache for each group). The kernel is an indexer_topk-owned
variant of the native CuTe-DSL FP8 paged-MQA logits kernel with a bounded
register cache of epilogue weights (see ``fp8_paged_mqa_logits_rows``); its
per-element arithmetic does not depend on the row grouping, so the logits are
bit-identical to the native split launch. Selection is the layer's native
``TopK`` module (heuristic self-sampling GVR, as ``policy`` requires).
"""

from typing import Dict, Optional

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from . import policy

__all__ = [
    "DECISION_COUNTS",
    "paged_mqa_logits",
    "decode_topk",
    "try_indexer_decode",
]

# Selection engine of decode_topk: "native" (the layer's TopK module) or
# "radix" (topk_select.radix_topk_decode, exact, graph-safe).
DEFAULT_SELECTOR = "native"

# Hook decisions on this rank, keyed by "engaged" or the fallback reason. The
# hook runs once per eager forward and once per CUDA-graph capture (never per
# replay), so these counts are host-side and cost nothing in a replayed graph.
DECISION_COUNTS: Dict[str, int] = {}


def _record_decision(decision: policy.Decision) -> None:
    key = "engaged" if decision.eligible else decision.reason
    count = DECISION_COUNTS.get(key, 0)
    DECISION_COUNTS[key] = count + 1
    if count == 0:
        state = "engaged" if decision.eligible else f"native fallback ({decision.reason})"
        logger.info(
            f"[glm_moe_dsa] indexer_topk/indexer_decode: first '{state}' decision on this rank"
        )


def paged_mqa_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_seq_len: int,
    config=None,
) -> torch.Tensor:
    """Score all verify rows of every request in one launch.

    Args:
        q: ``[B, next_n, H, D]`` float8_e4m3fn indexer queries.
        k_cache: ``[num_blocks, tokens_per_block, 1, D + 4]`` uint8 fused
            indexer K page view of this layer (FP8 values then fp32 scales
            per page).
        weights: ``[B * next_n, H]`` float32 per-head weights (already scaled
            by the query scale and softmax factor).
        context_lens: ``[B]`` int32 indexer KV lengths.
        block_table: ``[B, max_blocks]`` int32 page table of this layer.
        schedule_meta: ``[num_sms + 1, 2]`` int32 DeepGEMM schedule built from
            ``context_lens.view(-1, 1)`` with 64-token schedule blocks.
        max_seq_len: Capture-stable logits width (indexer max sequence length).
        config: Optional ``logits_runner.LogitsConfig``; None selects the
            default indexer_topk configuration.

    Returns:
        ``[B * next_n, max_seq_len]`` float32 logits view of the native
        256-column-aligned arena owned by indexer_topk. Columns at or beyond each row's
        causal length are unspecified.
    """
    from .logits_runner import DEFAULT_CONFIG, paged_mqa_logits_rows

    return paged_mqa_logits_rows(
        q,
        k_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_seq_len,
        config=DEFAULT_CONFIG if config is None else config,
    )


def decode_topk(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_seq_len: int,
    top_k: torch.nn.Module,
    out: torch.Tensor,
    radix_aux_indices: Optional[torch.Tensor],
    radix_aux_logits: Optional[torch.Tensor],
    logits_config=None,
    selector: Optional[str] = None,
) -> torch.Tensor:
    """Run indexer_topk logits + selection for uniform-width generation rows.

    Args:
        q: ``[B, next_n, H, D]`` float8_e4m3fn indexer queries.
        weights: ``[B * next_n, H]`` float32 per-head weights.
        k_cache: Fused indexer K page view, see :func:`paged_mqa_logits`.
        block_table: ``[B, max_blocks]`` int32 page table.
        context_lens: ``[B]`` int32 indexer (score-column) KV lengths.
        kv_lens: ``[B]`` int32 logical KV lengths (equal to ``context_lens``
            without index compression).
        schedule_meta: ``[num_sms + 1, 2]`` int32 schedule, see
            :func:`paged_mqa_logits`.
        max_seq_len: Capture-stable indexer max sequence length.
        top_k: The layer's native ``TopK`` module (selection policy).
        out: ``[B * next_n, index_topk]`` int32 caller-owned output.
        radix_aux_indices: Native radix scratch (used only by its fallback).
        radix_aux_logits: Native radix scratch (used only by its fallback).
        logits_config: Optional ``logits_runner.LogitsConfig``.
        selector: ``"native"`` or ``"radix"``; None uses ``DEFAULT_SELECTOR``.
            ``"radix"`` requires an uncompressed index (``compress_ratio`` 1).

    Returns:
        ``out`` with each row's logical top-k IDs and ``-1`` padding.
    """
    next_n = q.shape[1]
    logits = paged_mqa_logits(
        q,
        k_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_seq_len,
        config=logits_config,
    )
    selector = DEFAULT_SELECTOR if selector is None else selector
    if selector == "radix" and getattr(top_k, "compress_ratio", 1) == 1:
        from .topk_select import default_chunks, radix_topk_decode

        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        return radix_topk_decode(
            logits, kv_lens, out, next_n, chunks=default_chunks(out.shape[0], num_sms)
        )
    top_k(
        logits,
        out,
        is_prefill=False,
        sequence_lengths=kv_lens,
        scan_lengths=context_lens,
        row_kv_lens=None,
        next_n=next_n,
        max_seq_len=max_seq_len,
        gvr_ext_kwargs=None,
        radix_aux_indices=radix_aux_indices,
        radix_aux_logits=radix_aux_logits,
    )
    return out


def try_indexer_decode(
    indexer,
    metadata,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
) -> bool:
    """Adapter entry: run the GEN region of ``Indexer.sparse_attn_indexer``.

    The eligibility decision (``policy.decide``) reads only host state and
    tensor metadata and runs before any device work, so a ``False`` return
    leaves no side effect and the caller runs the native path. indexer_topk engages only
    inside its GPU-verified envelope; see ``policy``. Each distinct decision is
    logged once per rank and counted in ``DECISION_COUNTS``.

    Args:
        indexer: The native ``Indexer`` of a full-indexer layer.
        metadata: The DSA attention metadata of this forward.
        q_fp8: ``[num_tokens, H, D]`` float8_e4m3fn projected queries of the
            phase passed to the indexer.
        weights: ``[num_tokens, H]`` float32 projected per-head weights.
        token_offset: First generation row in ``q_fp8`` / ``weights``.
        num_gen_tokens: Generation rows in this call.
        out: ``[num_gen_tokens, index_topk]`` int32 slice of the native
            top-k buffer to fill.

    Returns:
        True when indexer_topk filled ``out``; False when the native path must run.
    """
    from ...dispatch import log_capability

    num_contexts = metadata.num_contexts
    num_generations = metadata.num_generations
    gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
    uniform = num_generations > 0 and bool(
        (gen_seq_lens == num_gen_tokens // num_generations).all()
    )
    top_k = indexer.top_k
    # Host-side view of the layer's page pool; no device work.
    k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    block_table = metadata.indexer_k_cache_block_offsets
    mapping = getattr(metadata, "mapping", None)
    max_seq_len = metadata.get_indexer_max_seq_len()
    decision = policy.decide(
        sm_version=get_sm_version(),
        cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
        native_uses_cute_dsl_logits=indexer.use_cute_dsl_paged_mqa_logits,
        use_fp4_indexer=indexer.use_fp4,
        use_gvr_emission=indexer.use_gvr_emission,
        needs_gvr_prior=top_k.needs_gvr_prior,
        is_ragged_verify=metadata.is_ragged_verify,
        expanded_mtp_buffers=metadata.use_expanded_buffers_for_mtp,
        in_mtp_draft_loop=metadata.in_mtp_draft_loop,
        attention_dp=bool(mapping is not None and mapping.enable_attention_dp),
        topk_decode_impl=top_k.decode_implementation.value,
        topk_self_sampling=top_k.gvr_self_sampling,
        index_topk=top_k.top_k,
        compress_ratio=top_k.compress_ratio,
        num_heads=indexer.n_heads,
        head_dim=indexer.head_dim,
        query_dtype=str(q_fp8.dtype),
        query_shape=tuple(q_fp8.shape),
        query_contiguous=q_fp8.is_contiguous(),
        weights_dtype=str(weights.dtype),
        weights_shape=tuple(weights.shape),
        weights_contiguous=weights.is_contiguous(),
        k_cache_dtype=str(k_cache.dtype),
        k_cache_shape=tuple(k_cache.shape),
        block_table_cols=block_table.shape[1],
        max_seq_len=max_seq_len,
        num_contexts=num_contexts,
        token_offset=token_offset,
        num_gen_tokens=num_gen_tokens,
        num_generations=num_generations,
        uniform_verify_width=uniform,
    )
    log_capability("indexer_topk", "indexer_decode", decision.eligible, decision.reason)
    _record_decision(decision)
    if not decision.eligible:
        return False

    next_n = num_gen_tokens // num_generations
    gen_rows = slice(token_offset, token_offset + num_gen_tokens)
    gen_reqs = slice(num_contexts, num_contexts + num_generations)
    context_lens = metadata.gen_indexer_kv_lens_cuda_runtime
    assert context_lens is not None
    decode_topk(
        q=q_fp8[gen_rows].view(num_generations, next_n, indexer.n_heads, indexer.head_dim),
        weights=weights[gen_rows],
        k_cache=k_cache,
        block_table=block_table[gen_reqs],
        context_lens=context_lens,
        kv_lens=metadata.kv_lens_cuda_runtime[gen_reqs],
        schedule_meta=metadata.scheduler_metadata_buffer,
        max_seq_len=max_seq_len,
        top_k=top_k,
        out=out,
        radix_aux_indices=metadata.radix_aux_indices,
        radix_aux_logits=metadata.radix_aux_logits,
    )
    return True
