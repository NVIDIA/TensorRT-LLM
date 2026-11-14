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
"""cross_rank_topk_cluster_merge: the cross_rank_topk GEN indexer region with the cluster-merge top-k op.

Same region, policy, CP logits kernel and output contract as cross_rank_topk
(``indexer_topk/cross_rank_indexer_decode.py``): rank ``r`` scores the 64-token pages
``p % world == r``, and ``cross_rank_topk_cluster_merge`` selects, exchanges and merges
the candidates into the tie-aware exact top-``index_topk`` of every row, the
same set and order on every rank and the same rows cross_rank_topk writes.

Numerical status: as cross_rank_topk (rank-consistent exact top-k over each page owner's
indexer K); cross_rank_topk_cluster_merge's rows are byte-identical to cross_rank_topk's for the
same local logits.
"""

from typing import Dict

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ... import native_ops
from ..indexer_topk import cross_rank_policy
from . import cp_workspace

__all__ = [
    "CP_DECISION_COUNTS",
    "SUPPORTED_WORLD_SIZES",
    "cp_decode_topk",
    "cp_topk_from_local_logits",
    "try_cp_indexer_decode",
    "world_size_reason",
]

# As cross_rank_topk: MTP draft step 0 runs the same region on the MTP layer.
ALLOW_DRAFT_STEP = True

# Tensor-parallel group sizes cross_rank_topk_cluster_merge's GPU tests verify (one node of 4 GPUs,
# and a two-host GEN group); at any other size cross_rank_topk serves the region.
SUPPORTED_WORLD_SIZES = (4, 8)


def world_size_reason(world: int) -> str:
    """Why cross_rank_topk_cluster_merge declines a group of ``world`` ranks ("" when it may engage).

    Identical on every rank of the group, so the decline needs no agreement.
    """
    if world not in SUPPORTED_WORLD_SIZES:
        return (
            f"tensor-parallel size {world} not in cross_rank_topk_cluster_merge's verified sizes "
            f"{SUPPORTED_WORLD_SIZES}"
        )
    return ""


CP_DECISION_COUNTS: Dict[str, int] = {}


def _record_decision(decision) -> None:
    key = "engaged" if decision.eligible else decision.reason
    count = CP_DECISION_COUNTS.get(key, 0)
    CP_DECISION_COUNTS[key] = count + 1
    if count == 0:
        state = "engaged" if decision.eligible else f"declined ({decision.reason})"
        logger.info(
            f"[glm_moe_dsa] cross_rank_topk_cluster_merge/cp_indexer_decode: first '{state}' decision on this rank"
        )


def cp_topk_from_local_logits(
    local_logits: torch.Tensor,
    kv_lens: torch.Tensor,
    out: torch.Tensor,
    workspace,
    next_n: int,
    debug_times=None,
) -> torch.Tensor:
    """Select, exchange and merge from this rank's local logits.

    Args:
        local_logits: ``[next_n, local_capacity]`` float32 logits of this
            rank's local columns (see ``indexer_topk.cross_rank_logits_runner.local_to_global``).
        kv_lens: ``[1]`` int32 global indexer KV length.
        out: ``[next_n, index_topk]`` int32 output.
        workspace: ``cp_workspace.CpWorkspace`` of the group.
        next_n: Verify rows of the request.
        debug_times: Optional ``[next_n, 8, 64]`` int64 phase stamps.

    Returns:
        ``out`` holding the same global top-k selection on every rank.
    """
    native_ops.cross_rank_topk_cluster_merge(
        local_logits,
        kv_lens,
        out,
        workspace.state,
        workspace.mc_ptr,
        workspace.uc_ptr,
        workspace.rank,
        workspace.world,
        next_n,
        debug_times,
    )
    return out


def cp_decode_topk(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor,
    workspace,
) -> torch.Tensor:
    """Run the cross_rank_topk_cluster_merge region for one request (arguments as cross_rank_topk's
    ``cp_decode_topk``)."""
    from ..indexer_topk.cross_rank_logits_runner import paged_mqa_logits_cp

    local_logits = paged_mqa_logits_cp(
        q, k_cache, weights, kv_lens, block_table, max_seq_len, workspace.rank, workspace.world
    )
    return cp_topk_from_local_logits(local_logits, kv_lens, out, workspace, q.shape[1])


def try_cp_indexer_decode(
    indexer,
    metadata,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
) -> bool:
    """Adapter entry: run the GEN region of ``Indexer.sparse_attn_indexer`` with cross_rank_topk_cluster_merge.

    The decision is cross_rank_topk's (``indexer_topk.cross_rank_policy``) on the same host state,
    plus cross_rank_topk_cluster_merge's own workspace creation (collective, first eligible eager
    call), whose capability agreement also covers the cluster residency. A ``False`` return leaves
    no side effect; every rank of the group reaches the same decision for the same forward, and the
    caller then tries cross_rank_topk's region.

    Args/Returns: as ``indexer_topk.indexer_decode.try_indexer_decode``.
    """
    from ...dispatch import log_capability
    from ..indexer_topk.cross_rank_logits_runner import ensure_compiled, local_capacity

    num_contexts = metadata.num_contexts
    num_generations = metadata.num_generations
    gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
    uniform = num_generations > 0 and bool(
        (gen_seq_lens == num_gen_tokens // num_generations).all()
    )
    top_k = indexer.top_k
    k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    block_table = metadata.indexer_k_cache_block_offsets
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
        block_table_cols=block_table.shape[1],
        max_seq_len=max_seq_len,
        num_contexts=num_contexts,
        token_offset=token_offset,
        num_gen_tokens=num_gen_tokens,
        num_generations=num_generations,
        uniform_verify_width=uniform,
    )
    if decision.eligible and world_size_reason(world):
        decision = type(decision)(False, world_size_reason(world))
    if decision.eligible:
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
            reason = (
                cp_workspace.failure_reason(mapping)
                or "workspace not created before CUDA-graph capture"
            )
            decision = type(decision)(False, f"workspace unavailable: {reason}")
    log_capability(
        "cross_rank_topk_cluster_merge", "cp_indexer_decode", decision.eligible, decision.reason
    )
    _record_decision(decision)
    if not decision.eligible:
        return False

    next_n = num_gen_tokens // num_generations
    gen_rows = slice(token_offset, token_offset + num_gen_tokens)
    gen_reqs = slice(num_contexts, num_contexts + num_generations)
    kv_lens = metadata.gen_indexer_kv_lens_cuda_runtime
    assert kv_lens is not None
    cp_decode_topk(
        q=q_fp8[gen_rows].view(num_generations, next_n, indexer.n_heads, indexer.head_dim),
        weights=weights[gen_rows],
        k_cache=k_cache,
        block_table=block_table[gen_reqs],
        kv_lens=kv_lens,
        max_seq_len=max_seq_len,
        out=out,
        workspace=workspace,
    )
    return True
