# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the context-first retirement eligibility boundary."""

from dataclasses import replace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import context_first_profile as profile
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    CacheKind,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _page_table() -> KVCachePageTable:
    """Create public metadata for two layers sharing an ordinary K/V pool."""
    return KVCachePageTable(
        tokens_per_block=16,
        pool_groups=[PhysicalPoolGroup([PhysicalPool(0x1000, 32768, 4)])],
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                local_layers=[LocalLayer(0, 0), LocalLayer(1, 1)],
                kv_head_num_per_rank=2,
                pool_views=[
                    PoolView(
                        pool_idx=0,
                        pool_role=frozenset({str(Role.KEY), str(Role.VALUE)}),
                        buffer_entries=np.array(
                            [(0, 0, 8192), (0, 8192, 8192), (1, 16384, 8192), (1, 24576, 8192)],
                            dtype=BUFFER_ENTRY_DTYPE,
                        ),
                    )
                ],
            )
        ],
    )


def _rank_info(tp_size: int = 2) -> RankInfo:
    """Describe one ordinary TP endpoint without initializing a transport."""
    return RankInfo(
        instance_name="test",
        instance_rank=0,
        tp_size=tp_size,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[2],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(2, 16, 128, 2, False, False),
        page_table=_page_table(),
    )


@pytest.fixture
def local_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig]:
    """Supply metadata-only manager inputs to the real eligibility helper."""
    manager = object.__new__(KVCacheManagerV2)
    manager.is_disagg = True
    manager.dtype = DataType.BF16
    manager.kv_cache_type = CacheTypeCpp.SELF
    manager.layer_specs = None
    manager.max_seq_len = 128
    monkeypatch.setattr(profile, "build_page_table_from_manager", lambda _: _page_table())
    monkeypatch.setattr(profile, "use_pure_python_transfer_agent", lambda: False)
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    for name in (
        "TRTLLM_NIXL_KVCACHE_BACKEND",
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP",
        "TRTLLM_DISAGG_LAYERWISE",
    ):
        monkeypatch.delenv(name, raising=False)
    return (
        Mapping(world_size=2, rank=0, tp_size=2),
        manager,
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON"),
    )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_context_first_profile_accepts_symmetric_tp(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig], tp_size: int
) -> None:
    """Both supported TP degrees pass, including serialized peer metadata."""
    _, manager, config = local_profile
    assert profile.supports_context_first_bf16(
        Mapping(world_size=tp_size, rank=0, tp_size=tp_size), manager, config
    )
    local = _rank_info(tp_size)
    profile.validate_context_first_peer(local, RankInfo.from_bytes(local.to_bytes()))


@pytest.mark.parametrize(
    "name,value",
    [
        ("TRTLLM_DISAGG_NO_RETRY", "0"),
        ("TRTLLM_NIXL_KVCACHE_BACKEND", "LIBFABRIC"),
        ("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1"),
        ("TRTLLM_DISAGG_LAYERWISE", "1"),
    ],
)
def test_context_first_profile_rejects_unqualified_environment(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    """Retry, alternate transports and transfer modes remain outside the profile."""
    monkeypatch.setenv(name, value)
    assert not profile.supports_context_first_bf16(*local_profile)


@pytest.mark.parametrize(
    "target,attribute,value",
    [
        (0, "enable_attention_dp", True),
        (0, "tp_size", 4),
        (0, "pp_size", 2),
        (0, "cp_size", 2),
        (1, "dtype", DataType.HALF),
        (1, "kv_cache_type", CacheTypeCpp.SELFKONLY),
        (1, "low_precision_dtype", DataType.NVFP4),
        (2, "backend", "UCX"),
        (2, "transceiver_runtime", "CPP"),
        (2, "kv_transfer_timeout_ms", None),
        (2, "kv_cache_bounce_size_mb", 256),
        (2, "agent_bounce_buffer_enable", True),
        (2, "enable_pipelined_transfer", True),
    ],
)
def test_context_first_profile_rejects_unqualified_local_settings(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig],
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    attribute: str,
    value: object,
) -> None:
    """Each configuration guard independently excludes an unsupported path."""
    monkeypatch.setattr(local_profile[target], attribute, value)
    assert not profile.supports_context_first_bf16(*local_profile)


def test_context_first_profile_checks_effective_layer_dtypes(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig],
) -> None:
    """The manager's nominal dtype must not hide an incompatible layer dtype."""
    _, manager, _ = local_profile
    layer = KVCacheLayerSpec(
        0, KVCacheLayerOwner.TARGET, DataType.BF16, 2, 128, (Role.KEY, Role.VALUE)
    )
    manager.layer_specs = (layer,)
    assert profile.supports_context_first_bf16(*local_profile)
    manager.layer_specs = (replace(layer, dtype=DataType.HALF),)
    assert not profile.supports_context_first_bf16(*local_profile)


def test_context_first_profile_rejects_python_agent(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure-Python agent is not covered by the native completion contract."""
    monkeypatch.setattr(profile, "use_pure_python_transfer_agent", lambda: True)
    assert not profile.supports_context_first_bf16(*local_profile)


@pytest.mark.parametrize(
    "case", ["key_only", "extra_role", "nhd", "state", "missing_value", "window"]
)
def test_context_first_profile_rejects_unqualified_layouts(
    local_profile: tuple[Mapping, KVCacheManagerV2, CacheTransceiverConfig],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Validate actual view roles and buffer geometry, not only manager dtype."""
    table = _page_table()
    group = table.layer_groups[0]
    view = group.pool_views[0]
    if case == "key_only":
        view.pool_role = frozenset({str(Role.KEY)})
    elif case == "extra_role":
        view.pool_role |= {str(Role.INDEX_KEY)}
    elif case == "nhd":
        view.mapper_kind = MapperKind.NHD
    elif case == "state":
        group.kind = CacheKind.STATE
    elif case == "missing_value":
        view.buffer_entries = view.buffer_entries[:3]
    else:
        group.sliding_window_size = 32
    monkeypatch.setattr(profile, "build_page_table_from_manager", lambda _: table)
    assert not profile.supports_context_first_bf16(*local_profile)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("tp_size", 1),
        ("dp_size", 2),
        ("cp_size", 2),
        ("pp_size", 2),
        ("bounded_replay_on_generation", True),
    ],
)
def test_context_first_peer_rejects_topology_or_replay(attribute: str, value: object) -> None:
    """The peer must match the supported topology before addresses escape."""
    local, peer = _rank_info(), _rank_info()
    setattr(peer, attribute, value)
    with pytest.raises(ValueError, match="context-first retirement requires"):
        profile.validate_context_first_peer(local, peer)


@pytest.mark.parametrize(
    "case", ["mla", "adp", "geometry", "missing_table", "missing_layer", "extra_role"]
)
def test_context_first_peer_rejects_incompatible_storage(case: str) -> None:
    """Serialized peer metadata cannot widen the effective layout boundary."""
    local, peer = _rank_info(), _rank_info()
    assert peer.attention is not None and peer.page_table is not None
    if case == "mla":
        peer.attention.is_mla = True
    elif case == "adp":
        peer.attention.enable_attention_dp = True
    elif case == "geometry":
        peer.attention.dims_per_head = 64
    elif case == "missing_table":
        peer.page_table = None
    elif case == "missing_layer":
        peer.page_table.layer_groups[0].local_layers.pop()
    else:
        peer.page_table.layer_groups[0].pool_views[0].pool_role |= {str(Role.KEY_BLOCK_SCALE)}
    with pytest.raises(ValueError, match="context-first retirement requires"):
        profile.validate_context_first_peer(local, peer)
