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
"""cross_rank_topk_global_emit: the cross_rank_topk GEN indexer region with
cross_rank_topk_cluster_merge's top-k that also writes rope_kv_append's bridge outputs.

Same region, policy, CP logits kernel and id output as cross_rank_topk_cluster_merge
(``cross_rank_topk_cluster_merge/``), which are cross_rank_topk's (``indexer_topk/cross_rank_*.py``): rank
``r`` scores the 64-token pages ``p % world == r``, and ``cross_rank_topk_global_emit``
selects, exchanges and merges the candidates into the tie-aware exact top-``index_topk`` of every
row, the same rows cross_rank_topk_cluster_merge writes. At the emission it also writes the call's
rope_kv_append bridge outputs (``bridge.py``): the global KV-pool rows of the layer's remap group
and the native top-k copy rows, the values ``native_ops.topk_index_bridge`` writes
from the same rows.

With ``indexer_layer_front`` also selected, the CP logits come from indexer_layer_front's launcher
(``indexer_layer_front/cp_logits_runner.py``: cross_rank_topk's arguments, arena and values, the
kernel waiting at entry and launched as the generation rope's programmatic
dependent); cross_rank_topk_global_emit's top-k then runs on them unchanged.

Numerical status: as cross_rank_topk_cluster_merge / cross_rank_topk (rank-consistent exact top-k
over each page owner's indexer K); pool and copy rows bit-identical to cross_rank_topk_cluster_merge
+ rope_kv_append's bridge.
"""

from typing import Dict

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ... import native_ops
from ..indexer_topk import cross_rank_policy
from . import bridge as global_emit_bridge
from . import cp_workspace

__all__ = [
    "CP_DECISION_COUNTS",
    "SUPPORTED_WORLD_SIZES",
    "bridge_reason",
    "cp_topk_from_local_logits",
    "try_cp_indexer_decode",
    "world_size_reason",
]

# As cross_rank_topk_cluster_merge / cross_rank_topk: MTP draft step 0 runs the same region on the
# MTP layer.
ALLOW_DRAFT_STEP = True

# Tensor-parallel group sizes cross_rank_topk_global_emit's GPU tests verify
# (cross_rank_topk_cluster_merge's).
SUPPORTED_WORLD_SIZES = (4, 8)

# Block size of the KV pool the kernel's remap assumes (the CP page).
PAGE_TOKENS = 64


def world_size_reason(world: int) -> str:
    """Why cross_rank_topk_global_emit declines a group of ``world`` ranks ("" when it may engage)."""
    if world not in SUPPORTED_WORLD_SIZES:
        return (
            f"tensor-parallel size {world} not in cross_rank_topk_global_emit's verified sizes "
            f"{SUPPORTED_WORLD_SIZES}"
        )
    return ""


CP_DECISION_COUNTS: Dict[str, int] = {}


def _record_decision(eligible: bool, reason: str) -> None:
    key = "engaged" if eligible else reason
    count = CP_DECISION_COUNTS.get(key, 0)
    CP_DECISION_COUNTS[key] = count + 1
    if count == 0:
        state = "engaged" if eligible else f"declined ({reason})"
        logger.info(
            f"[glm_moe_dsa] cross_rank_topk_global_emit/cp_indexer_decode: first '{state}' decision on this rank"
        )


def bridge_reason(indexer, metadata, packet, num_gen_tokens: int):
    """cross_rank_topk_global_emit's bridge preconditions: ``(reason, remap, copy)``, reason "" when
    cross_rank_topk_global_emit may engage.

    Host state only, identical on every rank of the group for the same call.
    """
    if not isinstance(packet, global_emit_bridge.GlobalEmitBridgePacket):
        return "no cross_rank_topk_global_emit bridge packet for this call", None, None
    if packet.copy is not None or packet.source is not None or packet.plan is not None:
        return "the packet already holds bridge work", None, None
    kvm = metadata.kv_cache_manager
    if kvm.tokens_per_block != PAGE_TOKENS:
        return f"KV pool block size {kvm.tokens_per_block} != {PAGE_TOKENS}", None, None
    remap = global_emit_bridge.predict_remap(metadata, kvm.layer_offsets[indexer.layer_idx])
    if remap is None:
        return "remap group larger than the bridge's group limit", None, None
    copy = global_emit_bridge.predict_copy(indexer, metadata, num_gen_tokens)
    reason = global_emit_bridge.copy_reason(copy, num_gen_tokens, indexer.top_k.top_k)
    if reason:
        return reason, None, None
    return "", remap, copy


def cp_topk_from_local_logits(
    local_logits: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    workspace,
    next_n: int,
    *,
    req_idx: torch.Tensor,
    req_idx_bias: int,
    block_table: torch.Tensor,
    stride_factor: int,
    layer_ids=None,
    layer_id: int = 0,
    copy=None,
    debug_times=None,
) -> torch.Tensor:
    """Select, exchange, merge and write the bridge outputs from this rank's local logits.

    Args:
        local_logits: ``[next_n, local_capacity]`` float32 logits of this
            rank's local columns.
        kv_lens: ``[1]`` int32 global indexer KV length.
        out: ``[next_n, index_topk]`` int32 id rows.
        workspace: ``cp_workspace.CpWorkspace`` of the group.
        next_n: Verify rows of the request.
        req_idx, req_idx_bias, block_table, stride_factor, layer_ids,
            layer_id, copy: the arguments of ``rope_kv_append.ops.topk_bridge`` for these
            rows (``copy`` a ``rope_kv_append.ops.BridgeCopy`` or None).
        debug_times: Optional ``[next_n, 8, 64]`` int64 phase stamps.

    Returns:
        The pool rows: ``[len(layer_ids), next_n, index_topk]`` with
        ``layer_ids``, else ``[next_n, index_topk]``.
    """
    return native_ops.cross_rank_topk_global_emit(
        local_logits,
        kv_lens,
        out,
        workspace.state,
        workspace.mc_ptr,
        workspace.uc_ptr,
        workspace.rank,
        workspace.world,
        next_n,
        req_idx,
        req_idx_bias,
        block_table,
        PAGE_TOKENS,
        stride_factor,
        layer_ids,
        layer_id,
        None if copy is None else copy.dst,
        0 if copy is None else copy.row_start,
        0 if copy is None else copy.rows,
        0 if copy is None else copy.mode,
        1 if copy is None else copy.next_n,
        None if copy is None else copy.num_accepted,
        debug_times,
    )


def try_cp_indexer_decode(
    indexer,
    metadata,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
    bridge=None,
) -> bool:
    """Adapter entry: run the GEN region with cross_rank_topk_global_emit and record its bridge work on ``bridge``.

    The decision is cross_rank_topk_cluster_merge's (cross_rank_topk's
    ``indexer_topk.cross_rank_policy`` on the same host state, cross_rank_topk_cluster_merge's group
    sizes, cross_rank_topk_global_emit's own workspace creation), plus cross_rank_topk_global_emit's
    bridge preconditions (``bridge_reason``), all taken before any device work. A ``False`` return
    leaves no side effect; every rank of the group reaches the same decision for the same forward,
    and the caller then tries cross_rank_topk_cluster_merge's region. ``True``: ``out`` holds the id
    rows and ``bridge.plan`` the pool rows and the copy cross_rank_topk_global_emit wrote, which
    ``bridge.remap_topk_to_global`` consumes.
    """
    from ...dispatch import is_enabled, log_capability
    from ..indexer_topk.cross_rank_logits_runner import local_capacity

    # indexer_layer_front's logits launcher is imported only with indexer_layer_front selected.
    indexer_front_logits = is_enabled("indexer_layer_front")
    if indexer_front_logits:
        from ..indexer_layer_front.cp_logits_runner import ensure_compiled, paged_mqa_logits_cp
    else:
        from ..indexer_topk.cross_rank_logits_runner import ensure_compiled, paged_mqa_logits_cp

    num_contexts = metadata.num_contexts
    num_generations = metadata.num_generations
    gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
    uniform = num_generations > 0 and bool(
        (gen_seq_lens == num_gen_tokens // num_generations).all()
    )
    top_k = indexer.top_k
    k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    indexer_block_table = metadata.indexer_k_cache_block_offsets
    mapping = getattr(metadata, "mapping", None)
    max_seq_len = metadata.get_indexer_max_seq_len()
    world = mapping.tp_size if mapping is not None else 1
    info = cp_workspace.op_info(world) if world > 0 else None
    capacity = local_capacity(max_seq_len, world) if world > 0 else 0
    # Facts of this rank's build and device, agreed by the workspace creation.
    local_reason = cross_rank_policy.rank_local_reason(
        op_available=info is not None,
        local_capacity=capacity,
        max_local_capacity=info[cp_workspace.INFO_MAX_LOCAL_CAPACITY] if info is not None else 0,
        sm_version=get_sm_version(),
        cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
    ) or (cp_workspace.cluster_reason(info) if info is not None else "")
    decision = cross_rank_policy.decide_cp_uniform(
        world_size=world,
        local_capacity=capacity,
        allow_draft_step=ALLOW_DRAFT_STEP,
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
        block_table_cols=indexer_block_table.shape[1],
        max_seq_len=max_seq_len,
        num_contexts=num_contexts,
        token_offset=token_offset,
        num_gen_tokens=num_gen_tokens,
        num_generations=num_generations,
        uniform_verify_width=uniform,
    )
    eligible, reason = decision.eligible, decision.reason
    remap = copy = None
    if eligible and world_size_reason(world):
        eligible, reason = False, world_size_reason(world)
    if eligible:
        reason, remap, copy = bridge_reason(indexer, metadata, bridge, num_gen_tokens)
        eligible = not reason
    if eligible:
        # The logits kernel is compiled before the group agrees on the
        # workspace, so no rank waits in an exchange while a peer compiles.
        workspace = cp_workspace.get_workspace(
            mapping,
            prepare=lambda: ensure_compiled(
                q_fp8.device,
                k_cache.shape[1],
                indexer.n_heads,
                indexer.head_dim,
                num_gen_tokens // num_generations,
                world,
            ),
            local_reason=local_reason,
        )
        if workspace is None:
            why = (
                cp_workspace.failure_reason(mapping)
                or "workspace not created before CUDA-graph capture"
            )
            eligible, reason = False, f"workspace unavailable: {why}"
    log_capability("cross_rank_topk_global_emit", "cp_indexer_decode", eligible, reason)
    if indexer_front_logits:
        log_capability("indexer_layer_front", "cp_indexer_decode_global_emit", eligible, reason)
    _record_decision(eligible, reason)
    if not eligible:
        return False

    next_n = num_gen_tokens // num_generations
    gen_rows = slice(token_offset, token_offset + num_gen_tokens)
    gen_reqs = slice(num_contexts, num_contexts + num_generations)
    kv_lens = metadata.gen_indexer_kv_lens_cuda_runtime
    assert kv_lens is not None
    local_logits = paged_mqa_logits_cp(
        q_fp8[gen_rows].view(num_generations, next_n, indexer.n_heads, indexer.head_dim),
        k_cache,
        weights[gen_rows],
        kv_lens,
        indexer_block_table[gen_reqs],
        max_seq_len,
        workspace.rank,
        workspace.world,
    )
    # The bridge inputs of rope_kv_append.ops.topk_bridge for these rows.
    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    pool = cp_topk_from_local_logits(
        local_logits,
        kv_lens,
        out,
        workspace,
        next_n,
        req_idx=metadata.req_idx_per_token[metadata.num_ctx_tokens : metadata.num_tokens],
        req_idx_bias=num_contexts,
        block_table=metadata.block_table[num_contexts : metadata.num_seqs],
        stride_factor=remap.page_index_scale * tokens_per_block,
        layer_ids=remap.layer_ids if remap.grouped else None,
        layer_id=0 if remap.grouped else remap.layer_offset,
        copy=copy,
    )
    bridge.plan = global_emit_bridge.GlobalEmitPlan(ids=out, remap=remap, copy=copy, pool=pool)
    return True
