# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft KV co-compression tests for TriAttention eviction.

With one-model speculative decoding, TriAttention compacts the separate draft
KV cache in the same round as the target: the target's union keep set is
broadcast over the draft's own KV heads, the draft's own protected tail is
appended as ordinals ``valid_seq_len + 0..tail-1``, and both caches land at
``destination_base = prompt_len``. These tests cover the physical draft moves,
the packed move indices, stream ordering across both cache managers, the
speculative admission gates (one representative per guard family), the
published compressed-token invariant, and prepared-compaction cache
invalidation.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import build_compaction as _build_compaction
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_fixed_resources_stubs as _make_fixed_resources_stubs
from conftest import make_ramp_pools as _make_ramp_pools
from conftest import make_request as _make_request
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals
from conftest import set_protected_tails as _set_protected_tails

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttention,
    _FixedScoreStagingBuffers,
    _PreparedEviction,
    _RequestCompressionState,
)


def _logical_view(pool: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Gather one request's pages into [K/V, head, token, dim] order."""
    num_kv_heads = int(pool.shape[2])
    head_dim = int(pool.shape[4])
    return pool.index_select(0, pages).permute(1, 2, 0, 3, 4).reshape(2, num_kv_heads, -1, head_dim)


def _launched_draft_compaction(draft_protected_tails):
    """Build target and draft pools with distinct head counts, then compact.

    The compact op ships only the pipelined bf16 kernels, so the pools use
    the supported production geometry (bf16, 32-token pages, head_dim 64) and
    the conclusive shifted ``arange % 251`` ramp payload.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    request_count = 2
    prompt_len = 2
    target_protected_tails = [2, 1]
    valid_seq_lens = [10, 9]

    target_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    draft_tables = torch.tensor([[1, 0, 2], [5, 4, 3]], dtype=torch.int32, device=device)
    target_pools = _make_ramp_pools(2, num_kv_heads=2, device=device)
    draft_pool = _make_ramp_pools(1, num_kv_heads=4, base=149, device=device)[0]
    assert target_pools[0].shape[2] != draft_pool.shape[2]
    initial_target = [pool.clone() for pool in target_pools]
    initial_draft = draft_pool.clone()

    # Kept ordinals are decode-only but absolute; the pinned prompt tokens
    # never appear in the selection rectangle.
    keep = torch.tensor([[2, 4, 7, 9], [3, 5, 6, 8]], dtype=torch.int64, device=device)

    compaction = _build_compaction(
        layer_pools=target_pools,
        layer_pool_keys=[("pool", 0), ("pool", 0)],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor(valid_seq_lens, dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(target_tables),
        prompt_offsets=torch.full((request_count,), prompt_len, dtype=torch.int32, device=device),
        protected_tail_capacity=max(target_protected_tails),
        draft_layer_pools=[draft_pool],
        draft_layers=[0],
        draft_layer_group_representative={0: 0},
        draft_layer_pool_keys=[("draft_pool", 0)],
        draft_protected_tail_capacity=max(draft_protected_tails),
        draft_kv_block_offsets=_encode_block_offsets(draft_tables),
        draft_page_table_slots={0: 0},
    )
    _set_protected_tails(compaction, target_protected_tails, draft_protected_tails)
    compaction.compact()
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


@pytest.mark.parametrize("draft_protected_tails", [[1, 1], [1, 2]])
def test_draft_moves_and_pack_match_keep_broadcast_and_tail_oracle(draft_protected_tails):
    built = _launched_draft_compaction(draft_protected_tails=draft_protected_tails)
    device = built.device
    prompt_len = built.prompt_len

    expected_offsets = [0]
    expected_moves = []
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

        expected_moves.append(draft_source.to(torch.int32))
        expected_offsets.append(expected_offsets[-1] + int(draft_source.numel()))

    # The packed draft move indices must match the same broadcast-plus-tail
    # oracle the physical moves followed.
    draft_compaction = built.compaction.draft_compaction
    expected_row = torch.cat(expected_moves)
    assert draft_compaction.move_source_offsets.cpu().tolist() == expected_offsets
    draft_indices = draft_compaction.move_source_indices
    # The index buffer is sized for the widest tail (the capacity); this
    # round's moves are packed at the front, where the offsets point.
    capacity_total = built.request_count * (
        int(built.keep.shape[1]) + max(built.draft_protected_tails)
    )
    assert draft_indices.shape == (int(built.draft_pool.shape[2]), capacity_total)
    for head in range(int(draft_indices.shape[0])):
        # Union mode broadcasts one keep set over every draft KV head.
        assert torch.equal(draft_indices[head, : expected_offsets[-1]], expected_row)


def test_mark_page_tables_consumed_orders_both_manager_streams():
    staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
    staging.device = torch.device("cuda", torch.cuda.current_device())
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
        # One representative per guard family (the per-mode/per-config
        # variants raise through the same checks).
        ("union_only_per_head", "union"),
        ("union_only_per_layer", "union"),
        ("draft_kv_factor", "standard key/value cache"),
        ("full_attention_draft", "full-attention draft"),
        ("callsite_dflash", "standard paged cache compacted together"),
        ("callsite_draft_target", "standard paged cache compacted together"),
        ("callsite_pard", "standard paged cache compacted together"),
    ],
)
def test_draft_admission_gates_raise(gate, match):
    draft_manager = _make_fake_v2(is_draft=True)
    if gate == "full_attention_draft":
        draft_manager.max_attention_window_vec = [128]
    if gate.startswith("callsite_"):
        # These draft contracts read cross-attention buffers or unvalidated
        # paged tails; the call-site speculative gate rejects every one of
        # them before any manager is created.
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm.llmapi.llm_args import (
            DFlashDecodingConfig,
            DraftTargetDecodingConfig,
            PARDDecodingConfig,
            TriAttentionKvCacheCompressionConfig,
        )

        spec_config = {
            "callsite_dflash": lambda: DFlashDecodingConfig(max_draft_len=3),
            "callsite_draft_target": lambda: DraftTargetDecodingConfig(
                max_draft_len=3,
                speculative_model="/tmp/draft-target-model",
            ),
            "callsite_pard": lambda: PARDDecodingConfig(max_draft_len=3),
        }[gate]()
        with pytest.raises(ValueError, match=match):
            validate_kv_cache_compression_with_spec(
                TriAttentionKvCacheCompressionConfig(
                    model_path="/models/test", calibration_path="/calib/test.pt", top_B=8
                ),
                spec_config,
                draft_manager,
            )
        return
    manager = TriAttention(
        _make_fake_v2(),
        top_B=8,
        model_path="/models/test",
        eviction_mode={
            "union_only_per_head": "per_head",
            "union_only_per_layer": "per_layer_perhead",
        }.get(gate, "union"),
        draft_kv_cache_manager=draft_manager,
    )
    if gate == "draft_kv_factor":
        # Flipping kv_factor after construction exercises TriAttention's own
        # runtime gate.
        draft_manager.kv_factor = 1

    with pytest.raises(ValueError, match=match):
        manager._validate_v2_compatibility()


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
    with _mocked_eviction_internals(manager) as internals:
        score_staging = internals.score_staging
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
                # The staged logical position restores the uncompressed
                # length: physical confirmed plus everything evicted so far.
                prepared = internals.attach.call_args.args[0]
                assert prepared[0].round_start == uncompressed
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
    layout, score_staging, keep_set_selector = _make_fixed_resources_stubs(manager)
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
        resources = manager._fixed_resources_for(layout, prepared)

        # Request capacity follows the executor limits, while the score
        # bucket follows what the cohort actually presents (power-of-two,
        # 1024 floor) instead of pinning tens-of-GiB scratch to max_seq_len.
        assert resources.score_staging is score_staging
        assert score_cls.call_args.kwargs["max_requests"] == 8
        assert score_cls.call_args.kwargs["decode_width"] == 4 + 2 * 128
        assert score_cls.call_args.kwargs["seq_len"] == 1024
        assert score_cls.call_args.kwargs["page_table_token_capacity"] == 1024 + 1
        assert score_cls.call_args.kwargs["draft_page_table_token_capacity"] == 1024 + 1

        # A second round with unchanged pools reuses the resident buffers and
        # keeps the cached compaction launches.
        cached_compaction = object()
        manager._batched_compaction = cached_compaction
        assert manager._fixed_resources_for(layout, prepared) is resources
        assert score_cls.call_count == 1
        assert manager._batched_compaction is cached_compaction

        # A pool change invalidates both the buffers and the compaction
        # launches that alias them.
        layout.pool_view_fingerprint = (("moved",),)
        rebuilt = manager._fixed_resources_for(layout, prepared)
        assert rebuilt is not resources
        assert score_cls.call_count == 2
        assert manager._batched_compaction is None
