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
"""rope_kv_append: fused per-layer MLA RoPE / FP8 query / latent KV append, DSA indexer
K-cache scatter, and top-k id -> global KV index bridge.

Two CUDA ops back this package (``cpp/tensorrt_llm/kernels/glmMoeDsa/ropeKvAppend.cu``):

* ``trtllm::glm_moe_dsa_rope_kv_append`` writes exactly the bytes of the
  native ``trtllm::mla_rope_generation`` (FP8 MLA cache, absorption mode, no
  helix) and, when given the indexer inputs, of
  ``trtllm::indexer_k_cache_scatter_op`` in one launch.
* ``trtllm::glm_moe_dsa_topk_index_bridge`` writes the values of
  ``trtllm::convert_req_index_to_global`` (or its grouped variant) and the
  native top-k reuse copy in one launch.

Every helper here checks its support domain before any device work; a caller
that gets ``None``/``False`` runs the native path unchanged.

Memory order: both kernels may launch early as programmatic dependents (PDL).
They read producer-written data and per-step metadata only after
``griddepcontrol.wait``; only the static quantization scales are read before
it.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from tensorrt_llm._utils import get_sm_version

from ... import dispatch

FEATURE_ID = "rope_kv_append"

_KV_LORA_RANK = 512
_ROPE_DIM = 64


@dataclass
class IndexerKScatter:
    """Deferred native indexer K-cache scatter of one layer and one forward.

    Single use: exactly one of the fused generation rope
    (:func:`mla_rope_generation`) or the native fallback
    (:func:`run_indexer_k_scatter_native`) consumes it, before any reader of
    the indexer K cache runs. The call site that deferred the scatter checks
    with :func:`check_indexer_k_scatter_consumed` right after the generation
    call, so a packet that did not reach its consumer fails the first forward
    (the CUDA-graph warmup) instead of silently leaving the step's keys out of
    the cache.
    """

    k_fp8: torch.Tensor
    k_scale: torch.Tensor
    k_cache: torch.Tensor
    slot_mapping_fp8: torch.Tensor
    slot_mapping_scale: torch.Tensor
    num_tokens: int
    layer_idx: int = -1
    consumer: Optional[str] = None


class RopeKvAppendContractError(RuntimeError):
    """A rope_kv_append call-site contract was violated (lost or doubly consumed deferred work)."""


def _consume(indexer_k: IndexerKScatter, consumer: str) -> None:
    if indexer_k.consumer is not None:
        raise RopeKvAppendContractError(
            f"rope_kv_append deferred indexer K-cache scatter of layer {indexer_k.layer_idx} consumed twice "
            f"({indexer_k.consumer!r}, then {consumer!r})"
        )
    indexer_k.consumer = consumer


@dataclass
class BridgeCopy:
    """Native top-k reuse copy folded into the bridge.

    mode 0: ``dst[row_start + i] = topk[i]`` for ``i < rows``.
    mode 1: ``dst[row_start + i] = topk[i * next_n + clamp(num_accepted[i] - 1)]``
    (the MTP last-accepted row of request ``i``).
    """

    dst: torch.Tensor
    row_start: int
    rows: int
    mode: int = 0
    next_n: int = 1
    num_accepted: Optional[torch.Tensor] = None


_MAX_GROUP_SIZE = 8

# The rope_kv_append kernels read per-step metadata (cache lengths, KV block offsets,
# indexer slot mappings, request indices, MTP accepted counts) and every
# producer-written input only after ``griddepcontrol.wait``; only the static
# quantization scales are read earlier.


def _sm_ok() -> bool:
    return get_sm_version() >= 100


def _static_rope_domain(mqa, metadata) -> bool:
    """Module/metadata conditions of the fused rope that do not depend on the call tensors."""
    kvm = metadata.kv_cache_manager
    return bool(
        kvm is not None
        and getattr(mqa, "is_mla_enable", False)
        and not mqa.uses_fp4_mla_attention
        and getattr(mqa, "has_fp8_kv_cache", False)
        and not getattr(mqa, "has_fp4_kv_cache", False)
        and mqa.kv_lora_rank == _KV_LORA_RANK
        and mqa.qk_rope_head_dim == _ROPE_DIM
        and bool(mqa.rope_append)
        and mqa.num_kv_heads == 1
        and getattr(kvm, "mla_kv_cache_residual_dim", 0) == 0
        and metadata.beam_width == 1
        and metadata.num_contexts == 0
        and metadata.num_generations > 0
        and getattr(metadata, "helix_position_offsets", None) is None
        and getattr(metadata, "helix_is_inactive_rank", None) is None
        and not getattr(metadata, "_helix_spec_tokens_valid", False)
        and metadata.token_major_gen_view() is None
        and not getattr(metadata, "is_ragged_verify", False)
        and mqa.rotary_cos_sin is not None
        and mqa.rotary_cos_sin.dtype == torch.float32
        and mqa.rotary_cos_sin.is_contiguous()
    )


def rope_eligible(
    mqa, metadata, fused_q, q_pe, latent_cache, quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale
) -> bool:
    """Whether the fused rope can replace ``mqa.mla_rope_generation`` for this call."""
    if not dispatch.is_enabled(FEATURE_ID) or not _sm_ok():
        return False
    ok = (
        _static_rope_domain(mqa, metadata)
        and fused_q is not None
        and q_pe is not None
        and quant_q_buffer is not None
        and mla_bmm1_scale is not None
        and mla_bmm2_scale is not None
        and fused_q.dtype == torch.bfloat16
        and q_pe.dtype == torch.bfloat16
        and latent_cache.dtype == torch.bfloat16
        and latent_cache.dim() == 2
        and latent_cache.stride(-1) == 1
        and latent_cache.stride(0) == _KV_LORA_RANK + _ROPE_DIM
        and fused_q.is_contiguous()
    )
    return dispatch.log_capability(
        FEATURE_ID, "rope", bool(ok), "" if ok else "outside the FP8 MLA decode domain"
    )


def defer_indexer_k_scatter(
    mla, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata, *, forwards_packet: bool = False
) -> Optional[IndexerKScatter]:
    """Describe the native ``Indexer._update_k_cache`` call without running it.

    The scatter is deferred into the generation rope of the same layer, which
    still precedes every reader of the indexer K cache (the indexer logits).
    The caller must forward the returned packet to
    ``forward_absorption_generation(deferred_indexer_k=...)`` and must call
    :func:`check_indexer_k_scatter_consumed` after that call; it declares both
    with ``forwards_packet=True``. A caller that does not declare it gets None
    and runs the native scatter.

    Returns None when the native scatter must run now: rope_kv_append off, a caller that
    does not forward the packet, a context or mixed batch, or a call outside
    the fused rope domain.
    """
    if not dispatch.is_enabled(FEATURE_ID) or not _sm_ok():
        return None
    if not forwards_packet:
        dispatch.log_capability(
            FEATURE_ID, "scatter_defer", False, "call site does not forward the deferred scatter"
        )
        return None
    mqa = mla.mqa
    if (
        metadata.num_contexts != 0
        or metadata.num_generations == 0
        or mla.apply_rotary_emb
        or getattr(mla, "kv_cache_dtype", None) == "fp8_ds_mla"
        or not _static_rope_domain(mqa, metadata)
    ):
        return None
    indexer = mqa.indexer
    if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
        return None
    k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(indexer.layer_idx)
    if k_scale.element_size() == 1:
        k_scale = k_scale.view(torch.int32)
    if (
        k_fp8.dim() != 2
        or k_fp8.element_size() != 1
        or not k_fp8.is_contiguous()
        or k_scale.dim() != 2
        or k_scale.element_size() * k_scale.shape[1] != 4
        or not k_scale.is_contiguous()
        or k_fp8.shape[1] not in (64, 128)
    ):
        return None
    return IndexerKScatter(
        k_fp8=k_fp8,
        k_scale=k_scale,
        k_cache=k_cache,
        slot_mapping_fp8=metadata.slot_mapping_fp8,
        slot_mapping_scale=metadata.slot_mapping_scale,
        num_tokens=int(k_fp8.shape[0]),
        layer_idx=int(indexer.layer_idx),
    )


def check_indexer_k_scatter_consumed(indexer_k: Optional[IndexerKScatter]) -> None:
    """Raise unless a deferred scatter was executed by the generation call.

    Called by the deferring call site right after the generation call. A
    packet that never reached the rope means the native scatter was skipped
    and the indexer read stale keys; there is no correct recovery afterwards.
    """
    if indexer_k is not None and indexer_k.consumer is None:
        raise RopeKvAppendContractError(
            f"rope_kv_append deferred indexer K-cache scatter of layer {indexer_k.layer_idx} was not consumed: "
            "the call site skipped the native Indexer._update_k_cache but did not pass the packet "
            "to the generation rope (forward_absorption_generation(deferred_indexer_k=...))"
        )


def mla_rope_generation(
    mqa,
    fused_q,
    q_pe,
    latent_cache,
    metadata,
    cu_q_seqlens,
    cu_kv_seqlens,
    fmha_scheduler_counter,
    mla_bmm1_scale,
    mla_bmm2_scale,
    quant_q_buffer,
    indexer_k: Optional[IndexerKScatter] = None,
) -> None:
    """Fused replacement of ``mqa.mla_rope_generation`` (+ the deferred indexer scatter).

    Callers check :func:`rope_eligible` first.
    """
    kvm = metadata.kv_cache_manager
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    idx_args: List[Optional[torch.Tensor]] = [None, None, None, None, None]
    idx_tokens = 0
    if indexer_k is not None:
        _consume(indexer_k, "fused_rope")
        if indexer_k.num_tokens > 0:
            idx_args = [
                indexer_k.k_fp8,
                indexer_k.k_scale,
                indexer_k.k_cache,
                indexer_k.slot_mapping_fp8,
                indexer_k.slot_mapping_scale,
            ]
            idx_tokens = indexer_k.num_tokens
    torch.ops.trtllm.glm_moe_dsa_rope_kv_append(
        fused_q,
        q_pe,
        latent_cache,
        mqa.rotary_cos_sin,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
        metadata.kv_lens_cuda_runtime,
        metadata.prompt_lens_cpu_runtime,
        metadata.num_contexts,
        metadata.kv_cache_block_offsets,
        kvm.kv_cache_pool_pointers,
        kvm.kv_cache_pool_mapping,
        getattr(mqa, "_nvfp4_mla_kv_scale_orig_quant", None),
        mqa.get_local_layer_idx(metadata),
        mqa.num_heads,
        kvm.tokens_per_block,
        metadata.max_seq_len,
        mqa.quant_mode,
        mqa.q_scaling,
        mqa.qk_nope_head_dim,
        mqa.qk_rope_head_dim,
        mqa.kv_lora_rank,
        *idx_args,
        idx_tokens,
    )


def run_indexer_k_scatter_native(indexer_k: IndexerKScatter) -> None:
    """Native fallback for a deferred scatter (same op and arguments as ``_update_k_cache``)."""
    _consume(indexer_k, "native")
    torch.ops.trtllm.indexer_k_cache_scatter_op(
        indexer_k.k_fp8,
        indexer_k.k_scale,
        indexer_k.k_cache,
        indexer_k.slot_mapping_fp8,
        indexer_k.slot_mapping_scale,
        indexer_k.num_tokens,
    )


@dataclass
class BridgePacket:
    """Per-call handoff between the indexer and the fused bridge of one layer.

    ``source`` replaces the rows the remap reads (draft reuse reads the shared
    rows directly); ``copy`` is the one native top-k copy the bridge performs.
    """

    source: Optional[torch.Tensor] = None
    copy: Optional[BridgeCopy] = None


def new_bridge_packet(metadata, is_generation: bool) -> Optional[BridgePacket]:
    """A packet when the fused bridge serves this ``sparse_attn_predict`` call, else None."""
    if not bridge_eligible(metadata, is_generation):
        return None
    return BridgePacket()


def bridge_eligible(metadata, is_generation: bool) -> bool:
    """Whether the fused bridge can replace the native remap for this call."""
    if not dispatch.is_enabled(FEATURE_ID) or not _sm_ok():
        return False
    import tensorrt_llm.bindings

    kvm = metadata.kv_cache_manager
    ok = (
        is_generation
        and kvm is not None
        and getattr(kvm, "dtype", None) != tensorrt_llm.bindings.DataType.NVFP4
        and metadata.num_contexts == 0
        and metadata.num_generations > 0
    )
    return dispatch.log_capability(
        FEATURE_ID, "bridge", bool(ok), "" if ok else "outside the generation remap domain"
    )


def topk_bridge(
    topk: torch.Tensor,
    metadata,
    num_rows: int,
    *,
    layer_offset: Optional[int] = None,
    layer_ids: Optional[torch.Tensor] = None,
    page_index_scale: int,
    copy: Optional[BridgeCopy] = None,
) -> torch.Tensor:
    """Remap local top-k rows to global KV-pool indices (+ the native reuse copy).

    Exactly one of ``layer_offset`` (per-layer remap, returns ``[rows, k]``) or
    ``layer_ids`` (grouped remap, returns ``[group, rows, k]``) is given.
    """
    # Same inputs as ``_ensure_pool_view_cached`` derives for the native remap;
    # the ``req_idx - num_contexts`` subtraction happens inside the kernel.
    # Unlike the native remap this does not populate the metadata's pool-view
    # memo (``_pool_cache_valid``/``_cached_*``). Every reader of that memo calls
    # ``_ensure_pool_view_cached()`` itself, so a later reader in the same step
    # populates it (and launches the subtraction) on demand; the subtraction is
    # saved only in steps where no other reader needs the memo.
    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    block_table = metadata.block_table[metadata.num_contexts : metadata.num_seqs]
    req_idx = metadata.req_idx_per_token[metadata.num_ctx_tokens : metadata.num_tokens]
    return torch.ops.trtllm.glm_moe_dsa_topk_index_bridge(
        topk,
        num_rows,
        req_idx,
        metadata.num_contexts,
        block_table,
        tokens_per_block,
        page_index_scale * tokens_per_block,
        layer_ids,
        0 if layer_offset is None else layer_offset,
        None if copy is None else copy.dst,
        0 if copy is None else copy.row_start,
        0 if copy is None else copy.rows,
        0 if copy is None else copy.mode,
        1 if copy is None else copy.next_n,
        None if copy is None else copy.num_accepted,
    )


def defer_shared_copy(
    packet: BridgePacket, shared: torch.Tensor, row_start: int, topk: torch.Tensor
) -> None:
    """``shared[row_start:row_start + rows, :k].copy_(topk)`` performed by the bridge."""
    assert packet.copy is None
    packet.copy = BridgeCopy(dst=shared, row_start=row_start, rows=int(topk.shape[0]))


def defer_reuse_copy(
    packet: Optional[BridgePacket],
    topk_buffer: torch.Tensor,
    row_start: int,
    shared_rows: torch.Tensor,
    gvr_prior_active: bool,
) -> bool:
    """Draft reuse ``topk_buffer[row_start:+rows] = shared_rows``; True when deferred."""
    if (
        packet is None
        or gvr_prior_active
        or packet.copy is not None
        or shared_rows.shape[1] != topk_buffer.shape[1]
    ):
        return False
    packet.source = shared_rows
    packet.copy = BridgeCopy(dst=topk_buffer, row_start=row_start, rows=int(shared_rows.shape[0]))
    return True


def defer_mtp_row_stash(
    packet: Optional[BridgePacket],
    shared: Optional[torch.Tensor],
    row_start: int,
    num_generations: int,
    next_n: int,
    num_accepted: Optional[torch.Tensor],
) -> bool:
    """MTP draft step 0 last-accepted-row stash into ``shared``; True when deferred."""
    if packet is None or shared is None or packet.copy is not None:
        return False
    if num_accepted is not None and num_accepted.dtype not in (torch.int32, torch.int64):
        return False
    packet.copy = BridgeCopy(
        dst=shared,
        row_start=row_start,
        rows=num_generations,
        mode=1,
        next_n=next_n,
        num_accepted=num_accepted,
    )
    return True


def remap_topk_to_global(
    backend, topk: torch.Tensor, metadata, packet: BridgePacket
) -> torch.Tensor:
    """Fused ``DSATrtllmAttention._remap_topk_to_global`` (+ the packet's copy).

    Follows the native grouping rules: the leader of an active full+shared
    group computes the whole group in one launch and publishes it in
    ``metadata._group_remap_batched``; followers slice it; everything else
    remaps per layer.
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    src = topk if packet.source is None else packet.source
    num_rows = int(src.shape[0])
    copy = packet.copy
    local_layer_idx = backend.get_local_layer_idx(metadata)
    if dsa_backend._GROUP_REMAP and not metadata.in_mtp_draft_loop:
        struct = metadata._ensure_group_remap_struct()
        leader_of = struct.get("leader_of")
        if leader_of is not None and local_layer_idx < len(leader_of):
            leader = leader_of[local_layer_idx]
            if leader >= 0 and struct["group_active"].get(leader, False):
                slot = struct["slot_of"][local_layer_idx]
                if backend.indexer is not None:
                    if struct["group_layer_ids"][leader].numel() > _MAX_GROUP_SIZE:
                        if copy is not None:
                            copy.dst[
                                copy.row_start : copy.row_start + copy.rows, : src.shape[1]
                            ].copy_(src)
                        return backend._remap_topk_to_global(topk, metadata, True)
                    batched = topk_bridge(
                        src,
                        metadata,
                        num_rows,
                        layer_ids=struct["group_layer_ids"][leader],
                        page_index_scale=struct["group_scale"][leader],
                        copy=copy,
                    )
                    metadata._group_remap_batched[leader] = batched
                    return batched[slot]
                batched = metadata._group_remap_batched.get(leader, None)
                if (
                    copy is None
                    and batched is not None
                    and slot < batched.shape[0]
                    and batched.shape[1:] == src.shape
                ):
                    return batched[slot]
    page_index_scale, layer_offset = metadata.kv_cache_manager.get_primary_pool_page_index_params(
        local_layer_idx
    )
    return topk_bridge(
        src,
        metadata,
        num_rows,
        layer_offset=layer_offset,
        page_index_scale=page_index_scale,
        copy=copy,
    )
