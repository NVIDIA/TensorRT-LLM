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
"""indexer_layer_front: the cross_rank_topk / cross_rank_topk_cluster_merge GEN indexer region with
the indexer_layer_front logits kernel.

Same region, policy, workspace, top-k op and output contract as cross_rank_topk_cluster_merge
(``cross_rank_topk_cluster_merge/cp_indexer_decode.py``, with ``cross_rank_topk_cluster_merge``
selected) or cross_rank_topk (``indexer_topk/cross_rank_indexer_decode.py``): every decision input
and every decline is theirs, evaluated from the same host state before any device work. The only
difference is the logits launch: ``indexer_layer_front/cp_logits_runner.py`` (wait at entry,
programmatic dependent of the generation rope), whose logits are
bit-identical to the context-partitioned logits kernel's. The top-k
(``cross_rank_topk_cluster_merge`` or ``cross_rank_topk``)
then runs on them unchanged.

A ``False`` return leaves no side effect; the adapter then tries cross_rank_topk_cluster_merge's and
cross_rank_topk's own entries, which reach the same decision.
"""

from typing import Dict

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ...dispatch import is_enabled, log_capability
from ..indexer_topk import cross_rank_policy

__all__ = ["CP_DECISION_COUNTS", "cp_decode_topk", "try_cp_indexer_decode"]

# As cross_rank_topk / cross_rank_topk_cluster_merge: MTP draft step 0 runs the same region on the
# MTP layer.
ALLOW_DRAFT_STEP = True

CP_DECISION_COUNTS: Dict[str, int] = {}


def _record_decision(flavor: str, decision) -> None:
    key = f"{flavor}:" + ("engaged" if decision.eligible else decision.reason)
    count = CP_DECISION_COUNTS.get(key, 0)
    CP_DECISION_COUNTS[key] = count + 1
    if count == 0:
        state = "engaged" if decision.eligible else f"declined ({decision.reason})"
        logger.info(
            f"[glm_moe_dsa] indexer_layer_front/cp_indexer_decode[{flavor}]: first '{state}' decision on this rank"
        )


def _modules(flavor: str):
    """``(workspace module, top-k entry)`` of the region owner."""
    if flavor == "cross_rank_topk_cluster_merge":
        from ..cross_rank_topk_cluster_merge import cp_indexer_decode as cluster_merge_decode
        from ..cross_rank_topk_cluster_merge import cp_workspace as cluster_merge_ws

        return cluster_merge_ws, cluster_merge_decode.cp_topk_from_local_logits
    from ..indexer_topk import cross_rank_indexer_decode as cross_rank_decode
    from ..indexer_topk import cross_rank_workspace as cross_rank_ws

    return cross_rank_ws, cross_rank_decode.cp_topk_from_local_logits


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
    flavor: str,
) -> torch.Tensor:
    """The region for one request (arguments as cross_rank_topk's ``cp_decode_topk``)."""
    from .cp_logits_runner import paged_mqa_logits_cp

    local_logits = paged_mqa_logits_cp(
        q, k_cache, weights, kv_lens, block_table, max_seq_len, workspace.rank, workspace.world
    )
    _, topk = _modules(flavor)
    return topk(local_logits, kv_lens, out, workspace, q.shape[1])


def _try(flavor: str, indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out) -> bool:
    from ..indexer_topk.cross_rank_logits_runner import local_capacity
    from .cp_logits_runner import ensure_compiled

    ws_mod, _ = _modules(flavor)
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
    info = ws_mod.op_info(world) if world > 0 else None
    capacity = local_capacity(max_seq_len, world) if world > 0 else 0
    max_cap = (
        (
            info[ws_mod.INFO_MAX_LOCAL_CAPACITY]
            if flavor == "cross_rank_topk_cluster_merge"
            else info[7]
        )
        if info
        else 0
    )
    local_reason = cross_rank_policy.rank_local_reason(
        op_available=info is not None,
        local_capacity=capacity,
        max_local_capacity=max_cap,
        sm_version=get_sm_version(),
        cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
    )
    if flavor == "cross_rank_topk_cluster_merge" and not local_reason and info is not None:
        local_reason = ws_mod.cluster_reason(info)
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
    if flavor == "cross_rank_topk_cluster_merge" and decision.eligible:
        from ..cross_rank_topk_cluster_merge.cp_indexer_decode import world_size_reason

        why = world_size_reason(world)
        if why:
            decision = type(decision)(False, why)
    if decision.eligible:
        # indexer_layer_front's logits kernel compiles before the group agrees on the
        # workspace.
        workspace = ws_mod.get_workspace(
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
                ws_mod.failure_reason(mapping) or "workspace not created before CUDA-graph capture"
            )
            decision = type(decision)(False, f"workspace unavailable: {reason}")
    log_capability(
        "indexer_layer_front", f"cp_indexer_decode_{flavor}", decision.eligible, decision.reason
    )
    _record_decision(flavor, decision)
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
        flavor=flavor,
    )
    return True


def try_cp_indexer_decode(
    indexer,
    metadata,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
) -> bool:
    """Adapter entry: the GEN region of ``Indexer.sparse_attn_indexer`` with the indexer_layer_front logits kernel.

    Tries cross_rank_topk_cluster_merge's region (when ``cross_rank_topk_cluster_merge`` is
    selected), then cross_rank_topk's, each with its own decision; returns False (no side effect)
    when both decline.

    Args/Returns: as ``indexer_topk.indexer_decode.try_indexer_decode``.
    """
    if not is_enabled("cross_rank_topk"):
        return False
    flavors = (
        ("cross_rank_topk_cluster_merge", "cross_rank_topk")
        if is_enabled("cross_rank_topk_cluster_merge")
        else ("cross_rank_topk",)
    )
    for flavor in flavors:
        if _try(flavor, indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out):
            return True
    return False
