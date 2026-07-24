# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft KV co-compression: the target's union keep set broadcasts over
the draft's own KV heads, the draft's tail appends as ordinals, both land
at ``destination_base = prompt_len``. Covers the physical moves, packed
indices, stream ordering, admission gates, the published compressed-token
invariant, and buffer reuse/rebuild."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import build_compaction as _build_compaction
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import make_buffer_stubs as _make_buffer_stubs
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_prepared_item as _make_prepared_item
from conftest import make_ramp_pools as _make_ramp_pools
from conftest import make_request as _make_request
from conftest import make_tri_config as _make_tri_config
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals
from conftest import run_compaction as _run_compaction
from conftest import set_protected_tails as _set_protected_tails

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention


def _fresh_request_state():
    """One request's compression ledger, as the manager initializes it."""
    return {"generation_steps": 0, "evicted_tokens": 0}


def _logical_view(pool: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
    """Gather one request's pages into [K/V, head, token, dim] order."""
    num_kv_heads = int(pool.shape[2])
    head_dim = int(pool.shape[4])
    return pool.index_select(0, pages).permute(1, 2, 0, 3, 4).reshape(2, num_kv_heads, -1, head_dim)


def _launched_draft_compaction(draft_protected_tails):
    """Target and draft pools with distinct head counts (supported bf16
    geometry, mod-251 ramp payload), compacted in one round."""
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

    keep = torch.tensor([[2, 4, 7, 9], [3, 5, 6, 8]], dtype=torch.int64, device=device)

    compaction = _build_compaction(
        layer_pools=target_pools,
        layer_pool_ids=[0, 0],
        kept_token_ordinals=keep.to(torch.int32),
        valid_sequence_lengths=torch.tensor(valid_seq_lens, dtype=torch.int32, device=device),
        kv_block_offsets=_encode_block_offsets(target_tables),
        prompt_offsets=torch.full((request_count,), prompt_len, dtype=torch.int32, device=device),
        protected_tail_capacity=max(target_protected_tails),
        draft_layer_pools=[draft_pool],
        draft_layers=[0],
        draft_layer_group_representative={0: 0},
        draft_layer_pool_ids=[0],
        draft_protected_tail_capacity=max(draft_protected_tails),
        draft_kv_block_offsets=_encode_block_offsets(draft_tables),
    )
    _set_protected_tails(compaction, target_protected_tails, draft_protected_tails)
    _run_compaction(compaction)
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


def test_draft_moves_and_pack_match_keep_broadcast_and_tail_oracle():
    # Ragged draft tails [1, 2] against target tails [2, 1]: one request's
    # draft tail below and one above its target, subsuming the uniform row.
    built = _launched_draft_compaction(draft_protected_tails=[1, 2])
    device = built.device
    prompt_len = built.prompt_len

    expected_offsets = [0]
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

        # Same kept ordinals through the draft's OWN table/heads/tail.
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

        expected_offsets.append(expected_offsets[-1] + int(draft_source.numel()))

    # The test-owned draft move-offset row must match the broadcast-plus-tail
    # oracle; the packed move sources themselves are covered byte-exactly by
    # the pool assertions above (the ramp payload makes every wrong move land
    # on different bytes) and by the pack-kernel oracle suite.
    assert built.compaction["draft_move_offsets"].cpu().tolist() == expected_offsets


def test_execute_eviction_round_orders_both_manager_streams():
    """The round executor snapshots both page-table planes, then records one
    completion event and BOTH cache-manager streams wait on it -- even when
    the round body fails -- so neither manager can free or reallocate pages
    this cohort is still reading."""
    from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

    event = mock.Mock()
    host = torch.zeros(6, 9, dtype=torch.int32)
    tri = TriAttention.__new__(TriAttention)
    tri._max_requests = 8
    tri._keep_count = 4
    tri.eviction_mode = "union"
    tri._swa_window = None
    tri._compaction_params = ()
    tri._draft_protected_tail_capacity = 1
    tri._staging_reuse_event = mock.Mock()
    tri._compaction_done_event = event
    tri._request_metadata_host = host
    tri._request_metadata_host_np = host.numpy()
    tri._request_metadata_device = torch.zeros_like(host)
    tri._phase = {"cos": None, "sin": None, "rows": 8}
    tri._phase_num_freqs = 1
    tri._phase_f_block = 1
    tri._round_starts_device = None
    tri._valid_seq_lens_device = None
    tri._token_starts_device = None
    tri._valid_widths = None
    tri._mean_cos = None
    tri._mean_sin = None
    tri._swa_destination_bases = None
    tri._swa_rebase_delta = 0
    tri._block_offsets_host = None
    tri._block_offsets_device = torch.zeros(1, dtype=torch.int32)
    tri._draft_block_offsets_host = None
    tri._draft_block_offsets_device = None
    target_stream = mock.Mock()
    draft_stream = mock.Mock()
    manager = SimpleNamespace(_stream=target_stream)
    draft_manager = SimpleNamespace(
        _stream=draft_stream, num_extra_kv_tokens=0, _kv_reserve_draft_tokens=0
    )
    tri.kv_cache_manager = manager
    tri.draft_kv_cache_manager = draft_manager
    compute_stream = SimpleNamespace()
    prepared = [_make_prepared_item(request_id=7, seq_len=8)]

    class Boom(RuntimeError):
        pass

    score_kernel = mock.MagicMock()
    score_kernel.__getitem__.return_value.side_effect = Boom
    with (
        mock.patch.object(torch.cuda, "current_stream", return_value=compute_stream),
        mock.patch.object(module, "grow_mean_phase_table"),
        mock.patch.object(tri, "_stage_block_offsets") as stage,
        mock.patch.object(module, "_gather_mean_phase_kernel", score_kernel),
        mock.patch.object(module, "compact") as compact,
    ):
        with pytest.raises(Boom):
            tri._execute_eviction_round(prepared)

    # Both page-table planes were snapshotted before the round body fired.
    assert stage.call_count == 2
    compact.assert_not_called()
    # One event records the round; BOTH cache managers wait on it.
    event.record.assert_called_once_with(compute_stream)
    target_stream.wait_event.assert_called_once_with(event)
    draft_stream.wait_event.assert_called_once_with(event)


@pytest.mark.parametrize(
    "gate,match",
    [
        # One representative per call-site guard family (the per-mode/per-config
        # variants raise through the same checks). kv_factor geometry needs no
        # admission gate: the native compact op TORCH_CHECKs every pool's K/V
        # plane count at the first compact.
        ("callsite_dflash", "standard paged cache compacted together"),
        ("union_only_per_head", "union"),
        ("full_attention_draft", "full-attention draft"),
    ],
)
def test_draft_admission_gates_raise(gate, match):
    # Draft/spec admission is owned by the executor call-site gate: rejected
    # before any compression manager exists.
    from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, MTPDecodingConfig

    draft_manager = _make_fake_v2(is_draft=True)
    if gate == "full_attention_draft":
        draft_manager.max_attention_window_vec = [128]
    spec_config = (
        DFlashDecodingConfig(max_draft_len=3)
        if gate == "callsite_dflash"
        else MTPDecodingConfig(max_draft_len=1)
    )
    config = _make_tri_config(
        budget=8,
        eviction_mode="per_head" if gate == "union_only_per_head" else "union",
    )

    with pytest.raises(ValueError, match=match):
        validate_kv_cache_compression_with_spec(config, spec_config, draft_manager)


def test_compressed_count_is_monotone_and_tracks_confirmed_length():
    manager = _make_triattention(budget=4, beta=4)
    manager.calibration = {}
    manager._layer_partition = ([0, 1], [], None)
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
    manager._request_states[7] = _fresh_request_state()
    batch = SimpleNamespace(generation_requests=[request])

    # Every step confirms one sampled token plus one accepted draft token.
    uncompressed = 6
    confirmed = uncompressed
    cache.capacity = confirmed
    previous_published = 0
    previous_evicted = 0
    eviction_rounds = 0
    with _mocked_eviction_internals(manager) as internals:
        for _ in range(6):
            uncompressed += 2
            confirmed += 2
            cache.capacity = confirmed

            manager._periodic_evict(batch)

            state = manager._request_states[7]
            if state["evicted_tokens"] > previous_evicted:
                # An eviction round compacted the cache to prompt + budget.
                eviction_rounds += 1
                confirmed -= state["evicted_tokens"] - previous_evicted
                previous_evicted = state["evicted_tokens"]
                cache.capacity = confirmed
                assert confirmed == 2 + 4
                # The staged logical position restores the uncompressed
                # length: physical confirmed plus everything evicted so far
                # (the prepared item's round_start).
                prepared = internals.execute.call_args.args[0]
                assert prepared[0]["round_start"] == uncompressed
            # The published count equals the uncompressed confirmed logical
            # length minus the physical confirmed length, and never decreases.
            assert request.py_num_compressed_tokens == uncompressed - confirmed
            assert request.py_num_compressed_tokens >= previous_published
            previous_published = request.py_num_compressed_tokens

    assert eviction_rounds == 3
    assert previous_published == 12
    # Each round the draft cache shrinks with the target, and the one
    # executor call runs on the manager itself, which carries both cache
    # managers whose streams it orders after the compact launches.
    assert draft_cache.resize.call_args_list == [mock.call(7, None)] * eviction_rounds
    assert len(internals.execute.call_args_list) == eviction_rounds
    assert manager.kv_cache_manager is target
    assert manager.draft_kv_cache_manager is draft_manager
    for call in internals.execute.call_args_list:
        assert len(call.args) == 1
        assert call.kwargs == {}


def test_cohort_growth_rebuilds_buffers_and_drops_cached_compaction():
    manager = _make_triattention(budget=4)
    layout, built_attributes = _make_buffer_stubs(manager)
    # The one-time host block-offset table shape gate reads the real manager
    # tables (int32 [pools, slots, K/V, blocks]).
    manager.kv_cache_manager.host_kv_cache_block_offsets = torch.zeros(
        1, 8, 2, 4, dtype=torch.int32
    )
    draft_manager = _make_fake_v2(is_draft=True)
    draft_manager.num_pools = 1
    draft_manager.host_kv_cache_block_offsets = torch.zeros(1, 8, 2, 4, dtype=torch.int32)
    manager.draft_kv_cache_manager = draft_manager
    # Injected post-construction: mirror the ctor-cached manager-lifetime tail.
    manager._draft_protected_tail_capacity = 1
    # The one resolver serves both sides; only its draft arm runs here (the
    # target layout arrives as the explicit _ensure_buffers argument).
    manager._runtime_kv_layout = mock.Mock(
        return_value=dict(
            layer_pools=[],
            dense_layers=[],
            layer_group_representative={},
            pool_representatives=(),
            layer_pool_ids=(),
            pool_page_counts=(4,),
        )
    )
    prepared = [
        _make_prepared_item(_make_request(7), request_id=7, seq_len=8, expected_keep_count=4)
    ]

    def apply_built(**kwargs):
        for name, value in built_attributes.items():
            setattr(manager, name, value)

    phase = manager._phase
    with mock.patch.object(manager, "_build_buffers", side_effect=apply_built) as prepare:
        manager._ensure_buffers(layout, prepared)

        # Request capacity follows the executor limits, while the score
        # bucket follows what the cohort actually presents (power-of-two,
        # 1024 floor) instead of pinning tens-of-GiB scratch to max_seq_len.
        # The stubbed build's capacities became the resident manager state.
        assert manager._buffers_built
        assert manager._decode_width == built_attributes["_decode_width"]
        kwargs = prepare.call_args.kwargs
        # The mode and the shared phase-table dict live on the manager itself
        # and thread through unchanged (no longer build arguments).
        assert manager.eviction_mode == "union"
        assert manager._phase is phase
        assert kwargs["max_requests"] == 8
        assert kwargs["decode_width"] == 4 + 2 * 128
        assert kwargs["bucket_seq_len"] == 1024
        assert kwargs["page_table_token_capacity"] == 1024 + 1
        assert kwargs["draft"]["page_table_token_capacity"] == 1024 + 1
        assert kwargs["draft"]["layout"] is manager._runtime_kv_layout.return_value
        # Migrated from the pipeline buffer-kwargs test: the budget and the
        # pool keys thread through unchanged.
        assert kwargs["keep_count"] == manager.budget
        assert kwargs["layout"] is layout
        assert list(kwargs["layout"]["layer_pool_ids"]) == list(layout["layer_pool_ids"])

        # A second round within the resident capacities reuses the buffers
        # (and with them the compaction launch data they carry).
        manager._ensure_buffers(layout, prepared)
        assert prepare.call_count == 1

        # A cohort that outgrows the resident capacities rebuilds the whole
        # buffer state, compaction included.
        grown = [
            _make_prepared_item(
                _make_request(7),
                request_id=7,
                seq_len=8 + built_attributes["_decode_width"],
                expected_keep_count=4,
            )
        ]

        def apply_rebuilt(**kwargs):
            apply_built()
            manager._decode_width = built_attributes["_decode_width"] + 8

        prepare.side_effect = apply_rebuilt
        manager._ensure_buffers(layout, grown)
        assert prepare.call_count == 2
        assert manager._buffers_built
        assert manager._decode_width == built_attributes["_decode_width"] + 8


def test_source_growth_beyond_score_bucket_rebuilds_buffers():
    """A later cohort can grow max(source_length) past the compiled score
    bucket while decode width, page tokens, and request count all still fit;
    the reuse gate must rebuild instead of scoring past the static geometry."""
    manager = _make_triattention(budget=4)
    layout, built_attributes = _make_buffer_stubs(manager)
    prepared = [
        _make_prepared_item(
            _make_request(7),
            request_id=7,
            seq_len=1024,
            prompt_len=1020,
            expected_keep_count=4,
        )
    ]

    def apply_built(**kwargs):
        for name, value in built_attributes.items():
            setattr(manager, name, value)

    with mock.patch.object(manager, "_build_buffers", side_effect=apply_built) as prepare:
        manager._ensure_buffers(layout, prepared)
        assert prepare.call_count == 1
        # One more source token, same decode width and request count.
        grown = [
            _make_prepared_item(
                _make_request(7),
                request_id=7,
                seq_len=1025,
                prompt_len=1021,
                expected_keep_count=4,
            )
        ]
        manager._ensure_buffers(layout, grown)
        assert prepare.call_count == 2


def test_staged_block_width_clamps_to_manager_source_width():
    """Score tile rounding can request more page-table blocks than the live V2
    source table holds; the staged width must clamp to the manager width so the
    native gather never reads past the K plane (tpb=32, max_seq_len=96, tail=1)."""
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        _allocate_block_offset_staging,
    )

    anchor_pool = torch.empty(1, 2, 1, 32, 4)
    host, device_table = _allocate_block_offset_staging(
        anchor_pool,
        num_pools=1,
        max_requests=2,
        token_capacity=129,
        max_source_blocks=4,
    )
    assert host.shape[-1] == 4 and device_table.shape[-1] == 4
    host, device_table = _allocate_block_offset_staging(
        anchor_pool,
        num_pools=1,
        max_requests=2,
        token_capacity=129,
        max_source_blocks=64,
    )
    assert host.shape[-1] == 8 and device_table.shape[-1] == 8
