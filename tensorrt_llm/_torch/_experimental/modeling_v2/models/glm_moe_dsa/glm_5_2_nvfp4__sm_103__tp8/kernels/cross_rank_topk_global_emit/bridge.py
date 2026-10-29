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
"""cross_rank_topk_global_emit's side of rope_kv_append's top-k bridge: the packet, the plan and the remap.

rope_kv_append (``rope_kv_append/ops.py``) collects the work of its fused bridge in a per-call
``BridgePacket``: the indexer and the backend attach the native top-k copy of
the call (``defer_shared_copy``, ``defer_reuse_copy``, ``defer_mtp_row_stash``)
after the top-k, and ``remap_topk_to_global`` launches the bridge. cross_rank_topk_global_emit
writes the bridge's outputs at the CP top-k's emission instead, so it has to know the
call's copy and remap before the top-k runs:

- ``predict_copy`` and ``predict_remap`` evaluate, at the indexer hook, the
  rules the call sites apply afterwards (rope_kv_append's own ``defer_mtp_row_stash`` and
  remap grouping rule, the backend's shared-copy condition), from host state
  only;
- the hook stores what it executed as the packet's ``GlobalEmitPlan``;
- ``remap_topk_to_global`` (the backend's remap when ``cross_rank_topk_global_emit`` is selected)
  consumes the plan once: it compares the packet's copy, source and top-k
  tensor and the layer's remap decision with the plan and raises
  ``GlobalEmitContractError`` on any difference (the prediction is a function of the
  step's host state, so a wrong prediction fails the first eager forward,
  before CUDA-graph capture); without a plan it is rope_kv_append's remap unchanged (draft
  steps 1-4, every call cross_rank_topk_global_emit declined).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from ..rope_kv_append import ops as rope_ops

__all__ = [
    "RemapPlan",
    "GlobalEmitBridgePacket",
    "GlobalEmitContractError",
    "GlobalEmitPlan",
    "copy_reason",
    "new_bridge_packet",
    "predict_copy",
    "predict_remap",
    "remap_topk_to_global",
]


class GlobalEmitContractError(RuntimeError):
    """The bridge work cross_rank_topk_global_emit performed at the top-k differs from what the call sites requested."""


@dataclass
class RemapPlan:
    """rope_kv_append's remap decision for a layer that owns an indexer.

    ``grouped``: the leader of an active full + shared group remaps the whole
    group (``layer_ids``, ``page_index_scale``) and publishes it in
    ``metadata._group_remap_batched[leader]``; the layer's own rows are
    ``out[slot]``. Otherwise the layer remaps alone (``page_index_scale``,
    ``layer_offset``).
    """

    local_layer: int
    grouped: bool
    page_index_scale: int
    leader: int = -1
    slot: int = 0
    layer_ids: Optional[torch.Tensor] = None
    layer_offset: int = 0


@dataclass
class GlobalEmitPlan:
    """What the cross_rank_topk_global_emit top-k wrote for one call: the id rows, the pool rows and the copy."""

    ids: torch.Tensor
    remap: RemapPlan
    copy: Optional[rope_ops.BridgeCopy]
    pool: Optional[torch.Tensor] = None


@dataclass
class GlobalEmitBridgePacket(rope_ops.BridgePacket):
    """rope_kv_append's packet with the plan of the cross_rank_topk_global_emit top-k that served
    the call (None: not served)."""

    plan: Optional[GlobalEmitPlan] = None


def new_bridge_packet(metadata, is_generation: bool) -> Optional[GlobalEmitBridgePacket]:
    """A packet when rope_kv_append's fused bridge serves this ``sparse_attn_predict`` call, else None."""
    if not rope_ops.bridge_eligible(metadata, is_generation):
        return None
    return GlobalEmitBridgePacket()


def predict_remap(metadata, local_layer: int) -> Optional[RemapPlan]:
    """rope_kv_append's remap decision (``rope_kv_append.ops.remap_topk_to_global``) for a layer with an indexer.

    Returns None when rope_kv_append would take the native fallback (a group of more than
    rope_kv_append's eight layers).
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa import backend as dsa_backend

    if dsa_backend._GROUP_REMAP and not metadata.in_mtp_draft_loop:
        struct = metadata._ensure_group_remap_struct()
        leader_of = struct.get("leader_of")
        if leader_of is not None and local_layer < len(leader_of):
            leader = leader_of[local_layer]
            if leader >= 0 and struct["group_active"].get(leader, False):
                layer_ids = struct["group_layer_ids"][leader]
                if layer_ids.numel() > rope_ops._MAX_GROUP_SIZE:
                    return None
                return RemapPlan(
                    local_layer=local_layer,
                    grouped=True,
                    page_index_scale=int(struct["group_scale"][leader]),
                    leader=leader,
                    slot=struct["slot_of"][local_layer],
                    layer_ids=layer_ids,
                )
    scale, offset = metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer)
    return RemapPlan(
        local_layer=local_layer,
        grouped=False,
        page_index_scale=int(scale),
        layer_offset=int(offset),
    )


def predict_copy(indexer, metadata, num_gen_tokens: int) -> Optional[rope_ops.BridgeCopy]:
    """The native top-k copy the call sites attach to the packet after a served top-k.

    Generation-only call with no context (the CP region's domain):

    - MTP draft step with ``mtp_index_share``: the indexer's last-accepted-row
      stash, exactly as ``rope_kv_append.ops.defer_mtp_row_stash`` decides it (evaluated on
      a scratch packet); the backend then preserves the shared rows.
    - otherwise: the backend's ``defer_shared_copy`` of the generation rows
      into ``shared_topk_indices`` when that buffer exists.
    """
    num_contexts = metadata.num_contexts
    num_generations = metadata.num_generations
    shared = metadata.shared_topk_indices
    if indexer.mtp_index_share and metadata.in_mtp_draft_loop:
        num_accepted = metadata.mtp_num_accepted
        if num_accepted is not None:
            num_accepted = num_accepted[num_contexts : num_contexts + num_generations]
        scratch = rope_ops.BridgePacket()
        if rope_ops.defer_mtp_row_stash(
            scratch,
            shared,
            num_contexts,
            num_generations,
            num_gen_tokens // num_generations,
            num_accepted,
        ):
            return scratch.copy
        return None
    if shared is None:
        return None
    return rope_ops.BridgeCopy(dst=shared, row_start=metadata.num_ctx_tokens, rows=num_gen_tokens)


def copy_reason(copy: Optional[rope_ops.BridgeCopy], rows: int, top_k: int) -> str:
    """Why the cross_rank_topk_global_emit op cannot perform ``copy`` for ``rows`` top-k rows ("" when it can).

    These are the checks of ``trtllm::glm_moe_dsa_topk_index_bridge``; a copy that fails them
    is left to rope_kv_append's path, which reports it.
    """
    if copy is None or copy.rows <= 0:
        return ""
    dst = copy.dst
    if not (
        dst.is_cuda
        and dst.dtype == torch.int32
        and dst.dim() == 2
        and dst.stride(1) == 1
        and dst.shape[1] >= top_k
        and dst.shape[0] >= copy.row_start + copy.rows
        and copy.row_start >= 0
    ):
        return "copy destination outside the bridge's domain"
    if copy.mode == 0:
        return "" if copy.rows <= rows else "copy rows exceed the top-k rows"
    if copy.mode != 1 or copy.next_n < 1 or copy.rows * copy.next_n > rows:
        return "copy mode-1 rows exceed the top-k rows"
    acc = copy.num_accepted
    if acc is not None and not (
        acc.is_cuda
        and acc.dtype in (torch.int32, torch.int64)
        and acc.dim() == 1
        and acc.is_contiguous()
        and acc.shape[0] >= copy.rows
    ):
        return "accepted counts outside the bridge's domain"
    return ""


def _same_tensor(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        a.data_ptr() == b.data_ptr()
        and a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape)
        and tuple(a.stride()) == tuple(b.stride())
    )


def _same_copy(a: Optional[rope_ops.BridgeCopy], b: Optional[rope_ops.BridgeCopy]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        _same_tensor(a.dst, b.dst)
        and (a.row_start, a.rows, a.mode, a.next_n) == (b.row_start, b.rows, b.mode, b.next_n)
        and _same_tensor(a.num_accepted, b.num_accepted)
    )


def _same_remap(a: Optional[RemapPlan], b: Optional[RemapPlan]) -> bool:
    if a is None or b is None:
        return False
    return (a.local_layer, a.grouped, a.page_index_scale, a.leader, a.slot, a.layer_offset) == (
        b.local_layer,
        b.grouped,
        b.page_index_scale,
        b.leader,
        b.slot,
        b.layer_offset,
    ) and a.layer_ids is b.layer_ids


def _check_plan(plan: GlobalEmitPlan, backend, topk: torch.Tensor, metadata, packet) -> None:
    """Raise ``GlobalEmitContractError`` when the call sites asked for other bridge work than
    cross_rank_topk_global_emit performed."""
    problems = []
    if packet.source is not None:
        problems.append("the packet reads other source rows")
    if not _same_tensor(topk, plan.ids):
        problems.append(
            "the remapped top-k rows are not the rows cross_rank_topk_global_emit wrote"
        )
    if backend.indexer is None:
        problems.append("the layer owns no indexer")
    local_layer = backend.get_local_layer_idx(metadata)
    if local_layer != plan.remap.local_layer:
        problems.append(f"local layer {local_layer} != planned {plan.remap.local_layer}")
    elif not _same_remap(predict_remap(metadata, local_layer), plan.remap):
        problems.append("the layer's remap decision changed")
    if not _same_copy(packet.copy, plan.copy):
        problems.append(f"copy {packet.copy} != planned {plan.copy}")
    if problems:
        raise GlobalEmitContractError(
            "cross_rank_topk_global_emit bridge plan mismatch: " + "; ".join(problems)
        )


def remap_topk_to_global(backend, topk: torch.Tensor, metadata, packet) -> torch.Tensor:
    """``rope_kv_append.ops.remap_topk_to_global`` with the pool rows the
    cross_rank_topk_global_emit top-k already wrote.

    Without a plan (the call was not served by cross_rank_topk_global_emit) this is rope_kv_append's
    remap. With one, the plan is consumed, checked against the packet and the layer, and
    the pool rows are returned (and published for the group's followers, as
    rope_kv_append's grouped remap does).
    """
    plan = getattr(packet, "plan", None)
    if plan is None:
        return rope_ops.remap_topk_to_global(backend, topk, metadata, packet)
    packet.plan = None
    _check_plan(plan, backend, topk, metadata, packet)
    if plan.remap.grouped:
        metadata._group_remap_batched[plan.remap.leader] = plan.pool
        return plan.pool[plan.remap.slot]
    return plan.pool
