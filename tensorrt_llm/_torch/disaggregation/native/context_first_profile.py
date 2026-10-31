# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eligibility checks for direct, no-retry BF16 context-first retirement.

Matching images, effective cache dtypes and configuration on every participant
remain deployment invariants. Rank metadata checks ownership activation, topology
and byte layout; it is not a version or dtype negotiation protocol.
"""

from __future__ import annotations

import os

from tensorrt_llm._torch.disaggregation.base.agent import use_pure_python_transfer_agent
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    CacheKind,
    KVCachePageTable,
    MapperKind,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

_KV_ROLES = frozenset({str(Role.KEY), str(Role.VALUE)})


def _ordinary_kv_layout(page_table: KVCachePageTable) -> dict[int, tuple[int, int | None, int]]:
    """Describe complete HND K/V layers, rejecting unqualified storage roles.

    Args:
        page_table: Manager-declared physical pools and logical transfer views.

    Returns:
        Global layer IDs mapped to head count, window size and bytes per K/V buffer.

    Raises:
        ValueError: A layer lacks an ordinary, complete, nonempty K/V view.
    """
    if page_table.tokens_per_block <= 0 or not page_table.layer_groups:
        raise ValueError("context-first retirement requires nonempty paged K/V storage")
    layout: dict[int, tuple[int, int | None, int]] = {}
    for group in page_table.layer_groups:
        if (
            not isinstance(group, AttentionLayerGroup)
            or group.kind != CacheKind.PAGED
            or group.kv_head_num_per_rank <= 0
            or not group.pool_views
        ):
            raise ValueError("context-first retirement requires ordinary attention K/V")
        global_ids = {layer.local_layer_id: layer.global_layer_id for layer in group.local_layers}
        if len(global_ids) != len(group.local_layers):
            raise ValueError("context-first retirement requires unique local layers")
        covered: set[int] = set()
        for view in group.pool_views:
            if (
                view.pool_role != _KV_ROLES
                or view.mapper_kind != MapperKind.INDEXED
                or view.section_bytes is not None
                or view.bytes_per_head is not None
            ):
                raise ValueError("context-first retirement requires full HND KEY+VALUE views")
            starts, _ = get_layer_byte_ranges(view)
            for local_id in starts:
                entries = view.buffer_entries[view.buffer_entries["local_layer_id"] == local_id]
                if (
                    local_id not in global_ids
                    or local_id in covered
                    or len(entries) != 2
                    or int(entries[0]["size"]) <= 0
                    or entries[0]["size"] != entries[1]["size"]
                ):
                    raise ValueError(
                        "context-first retirement requires one complete K/V pair per layer"
                    )
                global_id = global_ids[local_id]
                if global_id in layout:
                    raise ValueError("context-first retirement requires unique global layers")
                layout[global_id] = (
                    group.kv_head_num_per_rank,
                    group.sliding_window_size,
                    int(entries[0]["size"]),
                )
                covered.add(local_id)
        if covered != set(global_ids):
            raise ValueError("context-first retirement requires all local layers to be transferred")
    if not layout:
        raise ValueError("context-first retirement requires nonempty K/V layers")
    return layout


def supports_context_first_bf16(
    mapping: Mapping, manager: KVCacheManagerV2, config: CacheTransceiverConfig
) -> bool:
    """Check local eligibility without widening the existing FP4 activation gate.

    Args:
        mapping: Local executor topology.
        manager: Initialized manager whose effective layout is checked.
        config: Resolved Python/NIXL transfer configuration.

    Returns:
        Whether this endpoint supports direct BF16 K/V, TP1/TP2, no-retry retirement.
        Request scheduling style and peer compatibility must be checked separately.
    """
    if not (
        isinstance(manager, KVCacheManagerV2)
        and manager.is_disagg
        and manager.dtype == DataType.BF16
        and manager.kv_cache_type == CacheTypeCpp.SELF
        and manager.low_precision_dtype is None
        and not manager.get_disagg_ignored_roles()
        and not mapping.enable_attention_dp
        and mapping.tp_size in (1, 2)
        and mapping.pp_size == mapping.cp_size == mapping.dp_size == 1
        and mapping.world_size == mapping.tp_size
        and config.backend == "NIXL"
        and config.transceiver_runtime == "PYTHON"
        and config.kv_transfer_timeout_ms is not None
        and config.kv_transfer_timeout_ms > 0
        and config.kv_cache_bounce_size_mb == 0
        and not config.agent_bounce_buffer_enable
        and not config.enable_pipelined_transfer
        and not use_pure_python_transfer_agent()
        and os.getenv("TRTLLM_NIXL_KVCACHE_BACKEND", "UCX") == "UCX"
        and os.getenv("TRTLLM_DISAGG_NO_RETRY", "0") == "1"
        and os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "0") != "1"
        and os.getenv("TRTLLM_DISAGG_LAYERWISE", "0") != "1"
    ):
        return False
    if manager.layer_specs is not None and any(
        layer.dtype != DataType.BF16 or frozenset(map(str, layer.buffer_roles)) != _KV_ROLES
        for layer in manager.layer_specs
    ):
        return False
    try:
        layout = _ordinary_kv_layout(build_page_table_from_manager(manager))
    except ValueError:
        return False
    return all(window is None or window >= manager.max_seq_len for _, window, _ in layout.values())


def validate_context_first_peer(local: RankInfo, peer: RankInfo) -> None:
    """Reject unqualified peers before destination publication or source submission.

    Args:
        local: Local endpoint metadata after local profile eligibility checks.
        peer: Metadata obtained through the existing rank-discovery protocol.

    Raises:
        ValueError: Ownership is inactive or topology/K/V layout is outside the profile.
    """
    if local.context_first_ownership is not True or peer.context_first_ownership is not True:
        raise ValueError("context-first retirement requires ownership-enabled peers")
    for info in (local, peer):
        attention = info.attention
        if (
            info.tp_size not in (1, 2)
            or info.pp_size != 1
            or info.cp_size != 1
            or info.dp_size != 1
            or info.bounded_replay_on_generation
            or attention is None
            or attention.enable_attention_dp
            or attention.is_mla
            or attention.element_bytes != 2
            or attention.kv_heads_per_rank <= 0
            or info.page_table is None
        ):
            raise ValueError("context-first retirement requires non-ADP TP1/TP2 ordinary K/V peers")
    if local.tp_size != peer.tp_size or local.attention != peer.attention:
        raise ValueError("context-first retirement requires symmetric TP and K/V geometry")
    assert local.page_table is not None and peer.page_table is not None
    local_layout = _ordinary_kv_layout(local.page_table)
    peer_layout = _ordinary_kv_layout(peer.page_table)
    if (
        local.page_table.tokens_per_block != peer.page_table.tokens_per_block
        or local_layout != peer_layout
    ):
        raise ValueError("context-first retirement requires matching complete K/V layers")
