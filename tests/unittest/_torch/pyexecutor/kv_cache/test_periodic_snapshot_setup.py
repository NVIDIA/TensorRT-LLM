# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.fla import inforward_snapshot
from tensorrt_llm._torch.modules.mamba.cache_manager import NemotronHybridCacheManager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    mamba_cache_manager_v2 as module,
)

pytestmark = pytest.mark.cpu_only


def _manager(max_tokens):
    manager = SimpleNamespace(
        _supports_periodic_snapshots=True,
        _periodic_snapshot_requires_staging=True,
        _periodic_snapshot_supports_tp=True,
        _periodic_snapshot_model_support=True,
        _periodic_snapshot_max_tokens=max_tokens,
        periodic_snapshot_alignment=inforward_snapshot.InforwardSnapshotHost.periodic_snapshot_alignment,
        tokens_per_block=64,
        num_locality_domains=1,
        enable_block_reuse=True,
        spec_config=None,
        is_disagg=False,
        kv_connector_manager=None,
        is_draft=False,
        mapping=SimpleNamespace(
            pp_size=1, cp_size=1, tp_size=1, world_size=1, enable_attention_dp=False
        ),
        _state_layout=SimpleNamespace(conv_state_layout="q_k_v"),
        kv_cache_config=SimpleNamespace(
            mamba_state_config=SimpleNamespace(periodic_snapshot_interval=2048)
        ),
        ssm_state_shape=[2, 128, 128],
        ssm_state_dtype=torch.float32,
        conv_state_shape=[16, 3],
        _periodic_snapshot_min_window=lambda: 3,
        _periodic_snapshot_roles=lambda _layer: ("ssm", "conv"),
        mamba_pp_layers=[0],
        layer_offsets={0: 0},
        impl=SimpleNamespace(get_buffer_group_id=lambda *_: 0),
        ssm_layer_group_id=0,
        _periodic_snapshot_staging_in_use_bytes=lambda: 0,
        local_num_mamba_layers=1,
        max_seq_len=32768,
        all_ssm_states=[SimpleNamespace(device="cuda")],
        periodic_snapshot_capacity=0,
    )
    manager._periodic_snapshot_compatible = (
        lambda: inforward_snapshot.InforwardSnapshotHost._periodic_snapshot_compatible(manager)
    )
    return manager


def test_gdn_declares_the_periodic_snapshot_alignment():
    assert KVCacheManagerV2.periodic_snapshot_alignment == 0
    assert inforward_snapshot.InforwardSnapshotHost.periodic_snapshot_alignment == 64


@pytest.mark.parametrize("max_tokens, expected_rows", [(1024, 0), (8192, 4)])
def test_staging_capacity_respects_max_num_tokens(monkeypatch, max_tokens, expected_rows):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    allocate = Mock(return_value=object())
    monkeypatch.setattr(module.torch, "empty", allocate)
    manager = _manager(max_tokens)

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == expected_rows
    if expected_rows:
        assert allocate.call_args.args[0] == [expected_rows, 2, 128, 128]
    else:
        allocate.assert_not_called()


def test_staging_out_of_memory_falls_back_to_existing_chunking(monkeypatch):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    monkeypatch.setattr(module.torch, "empty", Mock(side_effect=torch.OutOfMemoryError("injected")))
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)
    manager = _manager(8192)

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 0
    warning.assert_called_once()


def test_uncoordinated_multi_rank_execution_keeps_existing_chunking(monkeypatch):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    allocate = Mock(return_value=object())
    monkeypatch.setattr(module.torch, "empty", allocate)
    manager = _manager(8192)
    manager.mapping.world_size = 2

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 0
    assert not hasattr(manager, "_periodic_snapshot_stage")
    allocate.assert_not_called()


def test_tp2_gdn_allocates_bounded_staging_for_consensus(monkeypatch):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    allocate = Mock(return_value=object())
    monkeypatch.setattr(module.torch, "empty", allocate)
    manager = _manager(8192)
    manager.mapping.world_size = manager.mapping.tp_size = 2

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 4
    assert allocate.call_args.args[0] == [4, 2, 128, 128]


def test_gdn_staging_is_shared_across_layers_within_the_budget(monkeypatch):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    manager = _manager(32768)
    manager.local_num_mamba_layers = 4
    manager.mamba_pp_layers = [0, 1, 2, 3]
    manager.layer_offsets = {i: i for i in manager.mamba_pp_layers}
    manager.mamba_layer_offsets = {i: i for i in manager.mamba_pp_layers}
    row_bytes = 2 * 128 * 128 * torch.float32.itemsize
    manager._periodic_snapshot_staging_in_use_bytes = lambda: 4 * row_bytes
    monkeypatch.setattr(module, "_PERIODIC_SNAPSHOT_STAGING_BYTES", 12 * row_bytes)
    torch_empty = torch.empty
    allocate = Mock(side_effect=lambda shape, dtype, device: torch_empty(shape, dtype=dtype))
    monkeypatch.setattr(module.torch, "empty", allocate)

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 4
    assert allocate.call_args.args[0] == [4, 2, 128, 128]
    assert allocate.call_args.kwargs["device"] == "cuda"
    first = module.MambaHybridCacheManagerV2.periodic_snapshot_stage_view(manager, 0, 4)
    later = module.MambaHybridCacheManagerV2.periodic_snapshot_stage_view(manager, 3, 1)
    assert first.dtype == torch.float32
    assert later.shape == (1, 2, 128, 128)
    assert first.data_ptr() == later.data_ptr()
    with pytest.raises(KeyError):
        module.MambaHybridCacheManagerV2.periodic_snapshot_stage_view(manager, 4, 4)


def test_speculative_decoding_does_not_allocate_periodic_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    allocate = Mock(return_value=object())
    monkeypatch.setattr(module.torch, "empty", allocate)
    manager = _manager(8192)
    manager.spec_config = object()

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 0
    allocate.assert_not_called()


@pytest.mark.parametrize("is_disagg, has_connector", [(True, False), (False, True)])
def test_disaggregated_or_connector_execution_keeps_existing_chunking(
    monkeypatch, is_disagg, has_connector
):
    monkeypatch.setenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1")
    monkeypatch.setattr(
        inforward_snapshot, "is_flashinfer_gdn_prefill_supported_arch", lambda: True
    )
    allocate = Mock(return_value=object())
    monkeypatch.setattr(module.torch, "empty", allocate)
    manager = _manager(8192)
    manager.is_disagg = is_disagg
    manager.kv_connector_manager = object() if has_connector else None

    module.MambaHybridCacheManagerV2._setup_periodic_snapshots(manager)

    assert manager.periodic_snapshot_capacity == 0
    allocate.assert_not_called()


@pytest.mark.parametrize(
    "layout,fi_ssd,stochastic_rounding,expected",
    [
        ("x_b_c", "0", False, True),
        ("q_k_v", "0", False, False),
        ("x_b_c", "1", False, False),
        ("x_b_c", "0", True, False),
    ],
)
def test_nemotron_periodic_snapshot_kernel_compatibility(
    monkeypatch, layout, fi_ssd, stochastic_rounding, expected
):
    monkeypatch.setenv("TRTLLM_USE_MAMBA_FI_SSD", fi_ssd)
    manager = object.__new__(NemotronHybridCacheManager)
    manager._state_layout = SimpleNamespace(conv_state_layout=layout)
    manager._mamba_ssm_stochastic_rounding = stochastic_rounding

    assert manager._periodic_snapshot_compatible() is expected


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_periodic_commit_preserves_primary_failure_when_cleanup_fails(cleanup_fails):
    primary = RuntimeError("CUDA error: an illegal memory access was encountered")
    cleanup = RuntimeError("snapshot slot release failed")
    cache = SimpleNamespace(
        history_length=0,
        resize=Mock(side_effect=primary),
        release_ssm_snapshot_slots=Mock(side_effect=cleanup if cleanup_fails else None),
    )
    manager = SimpleNamespace(
        periodic_snapshot_plan={9: SimpleNamespace(points=(64,))}, kv_cache_map={9: cache}
    )
    batch = SimpleNamespace(context_requests=[SimpleNamespace(py_request_id=9)])

    with pytest.raises(RuntimeError) as caught:
        module.MambaHybridCacheManagerV2.finish_periodic_snapshots(manager, batch)

    assert str(primary) in str(caught.value)
    if cleanup_fails:
        assert str(cleanup) in str(caught.value)
        assert caught.value.__cause__ is primary
    else:
        assert caught.value is primary
