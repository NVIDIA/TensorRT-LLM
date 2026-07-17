# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft KV co-compression tests for TriAttention eviction.

With one-model speculative decoding, TriAttention compacts the separate draft
KV cache in the same round as the target: the target's union keep set is
broadcast over the draft's own KV heads, the draft's own protected tail is
appended as ordinals ``valid_seq_len + 0..tail-1``, and both caches land at
``destination_base = prompt_len``. These tests cover the physical draft moves,
the packed move indices, stream ordering across both cache managers, the
speculative admission gates, the published compressed-token invariant, and
prepared-compaction cache invalidation.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets

from tensorrt_llm._torch.kv_cache_compression.triattention.compaction import (
    BatchedKVCacheCompaction,
)
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttention,
    _FixedScoreStagingBuffers,
    _PreparedEviction,
    _RequestCompressionState,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState


def _make_fake_v2(*, is_draft=False):
    """Build an unallocated V2 double with TriAttention's production contract."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = False
    fake_v2.is_draft = is_draft
    fake_v2.kv_compression_manages_history = False
    fake_v2.kv_factor = 2
    fake_v2.mapping = SimpleNamespace(enable_attention_dp=False)
    fake_v2.is_disagg = False
    fake_v2.max_beam_width = 1
    fake_v2.max_batch_size = 8
    fake_v2.num_extra_kv_tokens = 0
    fake_v2.max_draft_len = 0
    fake_v2.max_total_draft_tokens = 0
    fake_v2._kv_reserve_draft_tokens = 0
    fake_v2.max_seq_len = 65536
    fake_v2.tokens_per_block = 64
    fake_v2.max_blocks_per_seq = 1028
    fake_v2.get_num_available_tokens = lambda *, token_num_upper_bound, **_: token_num_upper_bound
    fake_v2.max_attention_window_vec = []
    fake_v2.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    fake_v2.impl = object()
    fake_v2.kv_cache_map = {}
    fake_v2.host_kv_cache_block_offsets = torch.empty(1, dtype=torch.int64)
    fake_v2.pp_layers = []
    fake_v2.layer_offsets = {}
    fake_v2.layer_to_pool_mapping_dict = {}
    return fake_v2


def _make_triattention(**overrides):
    options = {"top_B": 8, "model_path": "/models/test"}
    options.update(overrides)
    return TriAttention(_make_fake_v2(), **options)


def _make_request(request_id, **overrides):
    fields = {
        "py_request_id": request_id,
        "py_prompt_len": 0,
        "py_max_new_tokens": 65536,
        "py_draft_tokens": [],
        "py_num_accepted_draft_tokens": 0,
        "py_num_compressed_tokens": 0,
        "is_dummy": False,
        "state": LlmRequestState.GENERATION_IN_PROGRESS,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _logical_view(pool: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Gather one request's pages into [K/V, head, token, dim] order."""
    num_kv_heads = int(pool.shape[2])
    head_dim = int(pool.shape[4])
    return pool.index_select(0, pages).permute(1, 2, 0, 3, 4).reshape(2, num_kv_heads, -1, head_dim)


def _launched_draft_compaction(draft_protected_tails):
    """Build target and draft pools with distinct head counts, then compact."""
    device = torch.device("cuda")
    request_count = 2
    target_kv_heads = 2
    draft_kv_heads = 4
    prompt_len = 2
    decode_keep_count = 4
    tokens_per_block = 4
    head_dim = 16
    target_protected_tails = [2, 1]
    valid_seq_lens = [10, 9]

    target_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    draft_tables = torch.tensor([[1, 0, 2], [5, 4, 3]], dtype=torch.int32, device=device)
    target_pools = [
        (
            torch.arange(
                6 * 2 * target_kv_heads * tokens_per_block * head_dim,
                dtype=torch.float32,
                device=device,
            ).view(6, 2, target_kv_heads, tokens_per_block, head_dim)
            + layer * 100_000.0
        )
        for layer in range(2)
    ]
    draft_pool = (
        torch.arange(
            6 * 2 * draft_kv_heads * tokens_per_block * head_dim,
            dtype=torch.float32,
            device=device,
        ).view(6, 2, draft_kv_heads, tokens_per_block, head_dim)
        + 900_000.0
    )
    assert target_pools[0].shape[2] != draft_pool.shape[2]
    initial_target = [pool.clone() for pool in target_pools]
    initial_draft = draft_pool.clone()

    # Kept ordinals are decode-only but absolute; the pinned prompt tokens
    # never appear in the selection rectangle.
    keep = torch.tensor([[2, 4, 7, 9], [3, 5, 6, 8]], dtype=torch.int64, device=device)

    compaction = BatchedKVCacheCompaction(
        eviction_mode="union",
        layer_pools=target_pools,
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_keys=[("pool", 0), ("pool", 0)],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor(valid_seq_lens, dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(target_tables),
        page_table_slots={0: 0, 1: 0},
        request_count=request_count,
        prompt_offsets=torch.full((request_count,), prompt_len, dtype=torch.int32, device=device),
        decode_keep_count=decode_keep_count,
        swa_window=None,
        protected_tail_capacity=max(target_protected_tails),
        draft_layer_pools=[draft_pool],
        draft_layers=[0],
        draft_layer_group_representative={0: 0},
        draft_layer_pool_keys=[("draft_pool", 0)],
        draft_protected_tail_capacity=max(draft_protected_tails),
        draft_kv_block_offsets=_encode_block_offsets(draft_tables),
        draft_page_table_slots={0: 0},
    )
    compaction.set_protected_tails(target_protected_tails, draft_protected_tails)
    compaction.launch()
    torch.cuda.synchronize(device)

    return SimpleNamespace(
        device=device,
        request_count=request_count,
        prompt_len=prompt_len,
        keep=keep,
        valid_seq_lens=valid_seq_lens,
        target_protected_tails=target_protected_tails,
        draft_protected_tails=draft_protected_tails,
        target_tables=target_tables,
        draft_tables=draft_tables,
        target_pools=target_pools,
        draft_pool=draft_pool,
        initial_target=initial_target,
        initial_draft=initial_draft,
        compaction=compaction,
    )


def test_draft_pools_receive_target_union_keep_set_and_own_tail():
    built = _launched_draft_compaction(draft_protected_tails=[1, 2])
    device = built.device
    prompt_len = built.prompt_len

    for request in range(built.request_count):
        valid = built.valid_seq_lens[request]
        # Target dense layers compact the union keep set plus the target tail.
        target_pages = built.target_tables[request].to(torch.long)
        target_tail = torch.arange(
            valid,
            valid + built.target_protected_tails[request],
            dtype=torch.int64,
            device=device,
        )
        target_source = torch.cat((built.keep[request], target_tail))
        target_destination = torch.arange(
            prompt_len,
            prompt_len + target_source.numel(),
            dtype=torch.int64,
            device=device,
        )
        for before_pool, after_pool in zip(built.initial_target, built.target_pools):
            before = _logical_view(before_pool, target_pages)
            after = _logical_view(after_pool, target_pages)
            assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
            assert torch.equal(
                after.index_select(2, target_destination),
                before.index_select(2, target_source),
            )

        # The draft compacts the SAME kept ordinals through its OWN page
        # table, over its own head count, with its own protected tail, at
        # destination_base = prompt_len.
        draft_pages = built.draft_tables[request].to(torch.long)
        draft_tail = torch.arange(
            valid,
            valid + built.draft_protected_tails[request],
            dtype=torch.int64,
            device=device,
        )
        draft_source = torch.cat((built.keep[request], draft_tail))
        draft_destination = torch.arange(
            prompt_len,
            prompt_len + draft_source.numel(),
            dtype=torch.int64,
            device=device,
        )
        before = _logical_view(built.initial_draft, draft_pages)
        after = _logical_view(built.draft_pool, draft_pages)
        assert torch.equal(after[:, :, :prompt_len], before[:, :, :prompt_len])
        for head in range(int(built.draft_pool.shape[2])):
            assert torch.equal(
                after[:, head].index_select(1, draft_destination),
                before[:, head].index_select(1, draft_source),
            )


@pytest.mark.parametrize("draft_protected_tails", [[1, 1], [1, 2]])
def test_draft_pack_matches_keep_broadcast_and_tail_ordinal_oracle(draft_protected_tails):
    built = _launched_draft_compaction(draft_protected_tails=draft_protected_tails)
    draft_compaction = built.compaction.draft_compaction

    expected_offsets = [0]
    expected_moves = []
    for request in range(built.request_count):
        decode = built.keep[request].to(torch.int32)
        tail = torch.arange(
            built.valid_seq_lens[request],
            built.valid_seq_lens[request] + draft_protected_tails[request],
            dtype=torch.int32,
            device=built.device,
        )
        moves = torch.cat((decode, tail))
        expected_moves.append(moves)
        expected_offsets.append(expected_offsets[-1] + int(moves.numel()))
    expected_row = torch.cat(expected_moves)

    assert draft_compaction.move_source_offsets.cpu().tolist() == expected_offsets
    draft_indices = draft_compaction.move_source_indices
    assert draft_indices.shape == (int(built.draft_pool.shape[2]), expected_offsets[-1])
    for head in range(int(draft_indices.shape[0])):
        # Union mode broadcasts one keep set over every draft KV head.
        assert torch.equal(draft_indices[head], expected_row)


def test_mark_page_tables_consumed_orders_both_manager_streams():
    staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
    staging.device = torch.device("cuda")
    staging.page_tables_active = True
    event = mock.Mock()
    staging.bulk_consume_done = event
    target_stream = mock.Mock()
    draft_stream = mock.Mock()
    compute_stream = SimpleNamespace()

    with mock.patch.object(torch.cuda, "current_stream", return_value=compute_stream):
        staging.mark_page_tables_consumed(target_stream, draft_stream)

    # One event records the compact launches; BOTH cache managers wait on it,
    # so neither can free or reallocate pages this cohort is still reading.
    event.record.assert_called_once_with(compute_stream)
    target_stream.wait_event.assert_called_once_with(event)
    draft_stream.wait_event.assert_called_once_with(event)
    assert staging.page_tables_active is False

    with pytest.raises(RuntimeError, match="not staged"):
        staging.mark_page_tables_consumed(target_stream, draft_stream)


@pytest.mark.parametrize(
    "gate,match",
    [
        ("union_only", "union"),
        ("draft_kv_factor", "standard key/value cache"),
        ("full_attention_draft", "full-attention draft"),
        ("dflash", "standard paged cache compacted together"),
    ],
)
def test_draft_admission_gates_raise(gate, match):
    draft_manager = _make_fake_v2(is_draft=True)
    if gate == "full_attention_draft":
        draft_manager.max_attention_window_vec = [128]
    if gate == "dflash":
        # DFlash reads cross-attention context buffers, not a paged KV cache;
        # the call-site speculative gate rejects before any manager is
        # created.
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm.llmapi.llm_args import (
            DFlashDecodingConfig,
            TriAttentionKvCacheCompressionConfig,
        )

        with pytest.raises(ValueError, match=match):
            validate_kv_cache_compression_with_spec(
                TriAttentionKvCacheCompressionConfig(
                    model_path="/models/test", calibration_path="/calib/test.pt", top_B=8
                ),
                DFlashDecodingConfig(max_draft_len=3),
                draft_manager,
            )
        return
    manager = TriAttention(
        _make_fake_v2(),
        top_B=8,
        model_path="/models/test",
        eviction_mode="per_head" if gate == "union_only" else "union",
        draft_kv_cache_manager=draft_manager,
    )
    if gate == "draft_kv_factor":
        # Flipping kv_factor after construction exercises TriAttention's own
        # runtime gate.
        draft_manager.kv_factor = 1

    with pytest.raises(ValueError, match=match):
        manager._validate_v2_compatibility()


@contextmanager
def _mocked_eviction_internals(manager):
    """Run the real ``_evict_requests`` body around mocked GPU launches."""
    score_staging = SimpleNamespace(
        launch_prepared_score=mock.Mock(return_value=torch.zeros(1)),
        mark_page_tables_consumed=mock.Mock(),
    )
    keep_set_selector = SimpleNamespace(
        select_requests=mock.Mock(),
        refresh_row_prompt_offsets=mock.Mock(),
    )
    resources = SimpleNamespace(
        score_staging=score_staging,
        keep_set_selector=keep_set_selector,
    )
    batched_compaction = SimpleNamespace(launch=mock.Mock())
    with (
        mock.patch.object(manager, "_runtime_kv_layout", return_value=SimpleNamespace()),
        mock.patch.object(manager, "_eager_resources_for", return_value=resources),
        mock.patch.object(
            manager,
            "_batched_compaction_for",
            return_value=batched_compaction,
        ),
        mock.patch.object(manager, "_attach_page_ids"),
    ):
        yield score_staging


def test_compressed_count_is_monotone_and_tracks_confirmed_length():
    manager = _make_triattention(top_B=4, beta=4)
    manager._calibrated = True
    manager._attention_layer_partition_cache = ([0, 1], [], None)
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

    request = _make_request(7, py_prompt_len=2, py_num_accepted_draft_tokens=1)
    manager._request_states[7] = _RequestCompressionState()
    batch = SimpleNamespace(generation_requests=[request])

    # Every step confirms one sampled token plus one accepted draft token.
    uncompressed = 6
    confirmed = uncompressed
    cache.capacity = confirmed
    previous_published = 0
    eviction_rounds = 0
    with _mocked_eviction_internals(manager) as score_staging:
        for _ in range(6):
            uncompressed += 2
            confirmed += 2
            cache.capacity = confirmed

            manager._periodic_evict(batch)

            state = manager._request_states[7]
            if state.confirmed_kv_length < confirmed:
                # An eviction round compacted the cache to prompt + budget.
                eviction_rounds += 1
                confirmed = state.confirmed_kv_length
                cache.capacity = confirmed
                assert confirmed == 2 + 4
            # The published count equals the uncompressed confirmed logical
            # length minus the physical confirmed length, and never decreases.
            assert request.py_num_compressed_tokens == uncompressed - confirmed
            assert request.py_num_compressed_tokens >= previous_published
            previous_published = request.py_num_compressed_tokens

    assert eviction_rounds == 3
    assert previous_published == 12
    # Each round the draft cache shrinks with the target and both manager
    # streams are ordered after the compact launches.
    assert draft_cache.resize.call_args_list == [mock.call(7, None)] * eviction_rounds
    assert score_staging.mark_page_tables_consumed.call_args_list == (
        [mock.call(target._stream, draft_manager._stream)] * eviction_rounds
    )


def test_pool_change_rebuilds_buffers_and_drops_cached_compaction():
    from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

    manager = _make_triattention(top_B=4)
    manager._H = 2
    manager._F = 2
    manager._freq_scale_sq = torch.ones(2)
    manager._offsets = torch.ones(2)
    manager.calibration = {"omega": torch.ones(2)}
    manager._local_score_calibration = mock.Mock(return_value=(torch.ones(2, 2, 2),) * 3)
    manager._page_table_pool_keys = mock.Mock(return_value=[("pool", 0)])
    draft_manager = _make_fake_v2(is_draft=True)
    draft_manager.num_pools = 1
    manager.draft_kv_cache_manager = draft_manager
    manager._draft_runtime_kv_layout = mock.Mock(
        return_value=SimpleNamespace(
            layer_pools=[],
            pool_representatives=(),
            layer_pool_keys=(),
            pool_page_counts=(4,),
            pool_view_fingerprint=(),
        )
    )
    pool = torch.empty(8, 2, 1, 4, 4)
    layout = SimpleNamespace(
        manager=SimpleNamespace(num_pools=1),
        num_layers=2,
        global_layers=[0, 1],
        layer_pools=[pool, pool],
        dense_layers=[0, 1],
        swa_layers=[],
        storage_groups={0: [0, 1]},
        pool_view_fingerprint=(("fixed",),),
    )

    score_staging = SimpleNamespace(
        fused_group=SimpleNamespace(output=torch.empty(8, 4, 260)),
        bind_score_launcher=mock.Mock(),
        token_starts_device=torch.zeros(8, dtype=torch.int32),
        decode_width=260,
        page_table_token_capacity=65537,
        max_requests=8,
    )
    keep_set_selector = SimpleNamespace(
        valid_widths=torch.empty(8, dtype=torch.int32),
        top_indices_i32=torch.zeros(8, 4, dtype=torch.int32),
    )
    prepared = [
        _PreparedEviction(
            request=_make_request(7),
            request_id=7,
            seq_len=8,
            round_start=8,
            prompt_len=0,
            expected_keep_count=4,
            protected_tail=0,
        )
    ]

    with (
        mock.patch.object(
            module,
            "_FixedScoreStagingBuffers",
            return_value=score_staging,
        ) as score_cls,
        mock.patch.object(
            manager,
            "_build_cross_request_keep_set_selector",
            return_value=keep_set_selector,
        ),
    ):
        resources = manager._eager_resources_for(layout, prepared)

        # The buffers follow the executor limits, not this one-request cohort.
        assert resources.score_staging is score_staging
        assert score_cls.call_args.kwargs["max_requests"] == 8
        assert score_cls.call_args.kwargs["decode_width"] == 4 + 2 * 128
        assert score_cls.call_args.kwargs["seq_len"] == 65536
        assert score_cls.call_args.kwargs["page_table_token_capacity"] == 65536 + 1
        assert score_cls.call_args.kwargs["draft_page_table_token_capacity"] == 65536 + 1

        # A second round with unchanged pools reuses the resident buffers and
        # keeps the cached compaction launches.
        cached_compaction = object()
        manager._batched_compaction = cached_compaction
        assert manager._eager_resources_for(layout, prepared) is resources
        assert score_cls.call_count == 1
        assert manager._batched_compaction is cached_compaction

        # A pool change invalidates both the buffers and the compaction
        # launches that alias them.
        layout.pool_view_fingerprint = (("moved",),)
        rebuilt = manager._eager_resources_for(layout, prepared)
        assert rebuilt is not resources
        assert score_cls.call_count == 2
        assert manager._batched_compaction is None
