# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TriAttention draft lifecycle, ordering, admission, and publication."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import make_eviction_request as _make_eviction_request
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_request as _make_request
from conftest import make_tri_config as _make_tri_config
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttentionCompressionManager,
)


def test_execute_eviction_round_uses_current_stream_and_hands_back_to_target():
    """Keep target and draft work on the caller stream before manager handoff."""
    event = mock.Mock()
    host = torch.zeros(6, 9, dtype=torch.int32)
    tri = TriAttentionCompressionManager.__new__(TriAttentionCompressionManager)
    tri._request_capacity = 8
    tri.budget = 4
    tri._swa_window = None
    tri._draft_protected_tail_capacity = 1
    tri._staging_reuse_event = mock.Mock()
    tri._compaction_done_event = event
    tri._request_metadata_host = host
    tri._request_metadata_host_np = host.numpy()
    metadata_device = mock.Mock()
    tri._request_metadata_device = metadata_device
    tri._block_offsets_host = None
    tri._block_offsets_device = torch.zeros(1, dtype=torch.int32)
    tri._draft_block_offsets_host = None
    tri._draft_block_offsets_device = None
    execution_stream = mock.Mock()
    manager = SimpleNamespace(_stream=execution_stream)
    draft_manager = SimpleNamespace(
        _stream=execution_stream, num_extra_kv_tokens=0, _kv_reserve_draft_tokens=0
    )
    tri.kv_cache_manager = manager
    tri.draft_kv_cache_manager = draft_manager
    compute_stream = mock.Mock()
    eviction_requests = [_make_eviction_request(request_id=7, source_length=8)]

    class Boom(RuntimeError):
        pass

    metadata_device.copy_.side_effect = Boom
    with (
        mock.patch.object(
            torch.cuda, "current_stream", return_value=compute_stream
        ) as current_stream,
        mock.patch.object(tri, "_stage_block_offset_snapshot") as stage,
    ):
        with pytest.raises(Boom):
            tri._execute_eviction_round(eviction_requests)

    # Both page-table planes were snapshotted before the round body fired.
    assert stage.call_count == 2
    # One event records the current execution stream. Only the target manager
    # owns the post-round resize/release handoff.
    current_stream.assert_called_once_with(tri._block_offsets_device.device)
    event.record.assert_called_once_with(compute_stream)
    execution_stream.wait_event.assert_called_once_with(event)


@pytest.mark.parametrize(
    "gate,match",
    [
        ("callsite_dflash", "one-model MTP or EAGLE3"),
        ("union_only_per_head", "eviction_mode='union'"),
    ],
)
def test_speculative_admission_gates_raise(gate, match):
    from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_compatibility
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, MTPDecodingConfig

    if gate == "callsite_dflash":
        spec_config = DFlashDecodingConfig(max_draft_len=3)
    else:
        spec_config = MTPDecodingConfig(max_draft_len=1)
    config = _make_tri_config(
        budget=8,
        eviction_mode="per_head" if gate == "union_only_per_head" else "union",
    )

    with (
        mock.patch(
            "tensorrt_llm._torch.pyexecutor._util.is_sm_100f",
            return_value=True,
        ),
        pytest.raises(ValueError, match=match),
    ):
        validate_kv_cache_compression_compatibility(
            config,
            SimpleNamespace(enable_block_reuse=False),
            spec_config,
        )


def test_compressed_count_is_monotone_and_tracks_confirmed_length():
    manager = _make_triattention(budget=4, beta=4)
    target = manager.kv_cache_manager
    target._stream = mock.Mock()
    target.pp_layers = [0, 1]
    cache = SimpleNamespace(
        capacity=0,
        history_length=2,
        is_active=True,
        resize=mock.Mock(return_value=True),
    )
    target.kv_cache_map = {7: cache}
    draft_manager = _make_fake_v2(is_draft=True)
    draft_cache = SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True))
    draft_manager.kv_cache_map = {7: draft_cache}
    draft_manager._stream = mock.Mock()
    manager.draft_kv_cache_manager = draft_manager
    # Injected post-construction: mirror the ctor-cached manager-lifetime tail.
    manager._draft_protected_tail_capacity = 1

    request = _make_request(7, py_prompt_len=2, py_num_accepted_draft_tokens=1)
    batch = SimpleNamespace(generation_requests=[request])

    # Every step confirms one sampled token plus one accepted draft token.
    uncompressed = 6
    confirmed = uncompressed
    cache.capacity = confirmed
    previous_published = 0
    eviction_rounds = 0
    with _mocked_eviction_internals(manager) as internals:
        for _ in range(6):
            uncompressed += 2
            confirmed += 2
            cache.capacity = confirmed

            manager._evict_due_requests(batch)

            published = request.py_num_compressed_tokens
            if published > previous_published:
                # An eviction round compacted the cache to prompt + budget.
                eviction_rounds += 1
                confirmed -= published - previous_published
                cache.capacity = confirmed
                assert confirmed == 2 + 4
            # The published count equals the uncompressed confirmed logical
            # length minus the physical confirmed length, and never decreases.
            assert published == uncompressed - confirmed
            assert published >= previous_published
            previous_published = published

    assert eviction_rounds == 3
    assert previous_published == 12
    # Each round the draft cache shrinks with the target, and the one
    # executor call runs on the compression manager, which carries both cache
    # managers while handing completion back through the target manager.
    assert draft_cache.resize.call_args_list == [mock.call(7, None)] * eviction_rounds
    assert len(internals.execute.call_args_list) == eviction_rounds
    assert manager.kv_cache_manager is target
    assert manager.draft_kv_cache_manager is draft_manager
    for call in internals.execute.call_args_list:
        assert len(call.args) == 1
        assert call.kwargs == {}


def test_request_admission_does_not_grow_workspace():
    manager = _make_triattention(budget=128, beta=64)
    assert manager._selection_width_capacity == 256
    manager._selection_width_capacity = 260
    manager._score_token_capacity = 0
    manager._launch_score = None
    manager._compaction_done_event = mock.Mock()
    manager.kv_cache_manager = SimpleNamespace(max_seq_len=65536, tokens_per_block=64)

    def publish_score_state(*, score_token_capacity):
        manager._score_token_capacity = score_token_capacity
        manager._launch_score = object()

    manager._build_score_runtime = mock.Mock(side_effect=publish_score_state)
    requests = [
        _make_request(1, py_prompt_len=100, py_max_new_tokens=10000),
        _make_request(2, py_prompt_len=700, py_max_new_tokens=10),
        _make_request(3, py_prompt_len=900, py_max_new_tokens=200),
    ]

    for request in requests:
        manager.on_request_init(request)

    manager._build_score_runtime.assert_not_called()
    manager._compaction_done_event.synchronize.assert_not_called()

    # The largest phase offset must still fit the inclusive FP32 integer bound.
    limit = (1 << 24) - (1 << 16)
    manager.kv_cache_manager.max_seq_len = 1 << 24
    for request_id, position in enumerate((limit - 1, limit), start=4):
        manager.on_request_init(
            _make_request(request_id, py_prompt_len=100, py_max_new_tokens=position - 100)
        )
    for position in (limit + 1, (1 << 24) - 1):
        with pytest.raises(ValueError, match="exact-FP32"):
            manager.on_request_init(
                _make_request(6, py_prompt_len=100, py_max_new_tokens=position - 100)
            )
    # Apply max_seq_len before checking the phase range, including the boundary.
    for sequence_limit in (limit - 1, limit):
        manager.kv_cache_manager.max_seq_len = sequence_limit
        manager.on_request_init(_make_request(7, py_prompt_len=100, py_max_new_tokens=1 << 25))
    manager.kv_cache_manager.max_seq_len = limit + 1
    with pytest.raises(ValueError, match="exact-FP32"):
        manager.on_request_init(_make_request(8, py_prompt_len=100, py_max_new_tokens=1 << 25))
    manager._build_score_runtime.assert_not_called()


def test_request_admission_accepts_unaligned_sequence_limit_without_growth():
    manager = _make_triattention(budget=128, beta=64)
    manager._selection_width_capacity = 256
    manager._score_token_capacity = 0
    manager._launch_score = None
    manager._compaction_done_event = mock.Mock()
    manager.kv_cache_manager = SimpleNamespace(max_seq_len=1050, tokens_per_block=128)
    manager._build_score_runtime = mock.Mock()

    manager.on_request_init(_make_request(1, py_prompt_len=850, py_max_new_tokens=200))

    manager._build_score_runtime.assert_not_called()
    manager._compaction_done_event.synchronize.assert_not_called()


def test_block_offset_snapshot_width_is_aligned_and_capped():
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        _allocate_block_offset_snapshot,
    )

    anchor_pool = torch.empty(1, 2, 1, 32, 4)
    manager = SimpleNamespace(
        num_pools=1,
        tokens_per_block=32,
        max_blocks_per_seq=4,
        uses_device_page_table=False,
    )
    host, device_table = _allocate_block_offset_snapshot(
        manager,
        anchor_pool,
        request_capacity=2,
        token_capacity=129,
    )
    assert host.shape[-1] == 4 and device_table.shape[-1] == 4
    manager.max_blocks_per_seq = 64
    host, device_table = _allocate_block_offset_snapshot(
        manager,
        anchor_pool,
        request_capacity=2,
        token_capacity=129,
    )
    assert host.shape[-1] == 8 and device_table.shape[-1] == 8


@pytest.mark.parametrize("fail_second", [False, True])
def test_serial_eviction_publishes_each_completed_request(fail_second):
    """A later failure cannot hide target/draft compaction already completed."""
    manager = _make_triattention(budget=4, beta=4)
    requests = [_make_request(i, py_prompt_len=2) for i in range(2)]
    target_caches = [
        SimpleNamespace(
            capacity=10, history_length=10, is_active=True, resize=mock.Mock(return_value=True)
        )
        for _ in requests
    ]
    draft_caches = [
        SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True)) for _ in requests
    ]
    manager.kv_cache_manager.kv_cache_map = dict(enumerate(target_caches))
    manager.draft_kv_cache_manager = _make_fake_v2(is_draft=True)
    manager.draft_kv_cache_manager.kv_cache_map = dict(enumerate(draft_caches))
    manager._draft_protected_tail_capacity = 1
    completed = []

    def execute(items):
        request = items[0].request
        assert len(items) == 1
        if request.py_request_id == 1:
            assert requests[0].py_num_compressed_tokens == 4
            target_caches[0].resize.assert_called_once_with(6, None)
            draft_caches[0].resize.assert_called_once_with(7, None)
            if fail_second:
                raise RuntimeError("second request failed")
        completed.append(request.py_request_id)

    with (
        mock.patch.object(manager, "_try_score_workspace", return_value=None),
        mock.patch.object(manager, "_execute_eviction_round", side_effect=execute),
    ):
        batch = SimpleNamespace(generation_requests=requests)
        if fail_second:
            with pytest.raises(RuntimeError, match="second request failed"):
                manager._evict_due_requests(batch)
        else:
            manager._evict_due_requests(batch)
    assert completed == ([0] if fail_second else [0, 1])
    assert requests[1].py_num_compressed_tokens == (0 if fail_second else 4)
    assert target_caches[1].resize.call_count == (0 if fail_second else 1)
    assert draft_caches[1].resize.call_count == (0 if fail_second else 1)


def test_batch_workspace_reuses_allocator_and_reports_oom():
    """Cached allocations bypass driver queries; only allocation OOM is recovered."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import logger

    manager = _make_triattention()
    manager._request_capacity = 2
    manager._union_tp_mapping = None
    manager._score_scratch = torch.empty(16, dtype=torch.float32)
    workspace = torch.empty(32, dtype=torch.float32)
    with (
        mock.patch.object(torch.cuda, "mem_get_info", side_effect=AssertionError("driver query")),
        mock.patch.object(
            torch,
            "empty",
            side_effect=[
                workspace,
                workspace,
                torch.OutOfMemoryError,
                RuntimeError("unexpected allocation error"),
            ],
        ),
        mock.patch.object(logger, "warning_once") as warning,
    ):
        assert manager._try_score_workspace(2) is workspace
        assert manager._try_score_workspace(2) is workspace
        assert manager._try_score_workspace(2) is None
        assert manager._batch_workspace_attempts == 3
        assert manager._batch_workspace_ooms == 1
        warning.assert_called_once()
        assert "2 requests" in warning.call_args.args[0]
        assert "required bytes: 128" in warning.call_args.args[0]
        with pytest.raises(RuntimeError, match="unexpected allocation error"):
            manager._try_score_workspace(2)
    assert manager._batch_workspace_attempts == 4
    assert manager._batch_workspace_ooms == 1
