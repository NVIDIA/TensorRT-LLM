# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission-time page reuse and continuation through normal KV pages."""

import pytest
import torch
from _deepseek_v41_test_utils import cache_case, canonical_routing, enable_ced, global_bytes, run
from _deepseek_v41_test_utils import cache_config as shared_cache_config
from _deepseek_v41_test_utils import model_and_config as global_model_and_config

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderCheckpoint, encoder_replay_tokens
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionReusePolicy, _introspection

cache_config = shared_cache_config

model_and_config = global_model_and_config


def request(i, length):
    req = LlmRequest(
        request_id=i,
        max_new_tokens=2,
        input_tokens=list(range(length)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    req.py_seq_slot = 0
    return req


@pytest.mark.parametrize("encoder_replay", [False, True])
def test_full_context_logits_do_not_claim_encoder_only_prefix(
    cache_config, monkeypatch, encoder_replay
):
    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", str(int(encoder_replay)))
    with cache_case(config, sparse, dtype, [257], [0], 1, reuse=True) as case:
        manager = case[0].kv_cache_manager
        source = manager.kv_cache_map[0]
        source.commit(list(range(256)))
        source.stop_committing()
        reader = request(9, 257)
        reader.py_return_context_logits = True
        try:
            assert manager.prepare_context(reader)
            assert reader.context_current_position == 0
            assert manager.kv_cache_map[9].num_committed_tokens == 0
            assert reader.py_ced_replay is None
        finally:
            manager.free_resources(reader)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_snapshot_policy_follows_reuse_and_encoder_recovery(
    cache_config, monkeypatch, reuse, replay
):
    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", str(int(replay)))
    with cache_case(config, sparse, dtype, [257], [0], 1, reuse=reuse) as case:
        manager = case[0].kv_cache_manager
        expected_encoder = (
            AttentionReusePolicy.PRIVATE
            if not reuse
            else AttentionReusePolicy.OPTIONAL
            if replay
            else AttentionReusePolicy.REQUIRED
        )
        for layer in manager.impl.init_config.layers:
            owner, role = manager._physical_roles[layer.layer_id, layer.buffers[0].role]
            expected = (
                AttentionReusePolicy.REQUIRED
                if role == CSA2CacheRole.GLOBAL
                else AttentionReusePolicy.PRIVATE
                if role == CSA2CacheRole.SWA and owner >= 5
                else expected_encoder
            )
            assert layer.reuse_policy == expected
        encoder = manager._layer_roles[0, CSA2CacheRole.SWA]
        decoder = manager._layer_roles[5, CSA2CacheRole.SWA]
        if reuse:
            assert all(
                role not in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX)
                for _, role in manager._encoder_roles
            )
            if replay:
                expected_groups = {
                    manager.impl.get_layer_group_id(layer.layer_id)
                    for layer in manager.impl.init_config.layers
                    if layer.reuse_policy == AttentionReusePolicy.OPTIONAL
                }
                assert set(manager._encoder_optional_groups) == expected_groups
            assert (
                manager.layer_to_pool_mapping_dict[encoder]
                != manager.layer_to_pool_mapping_dict[decoder]
            )


@pytest.mark.parametrize("prefix", [129, 255])
def test_partial_global_hit_copies_tail_and_replays_encoder(cache_config, monkeypatch, prefix):
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    with cache_case(config, sparse, dtype, [384], [0], 1, reuse=True) as case:
        manager = case[0].kv_cache_manager
        source = manager.kv_cache_map[0]
        source.commit(list(range(384)))
        reader = LlmRequest(
            request_id=1,
            max_new_tokens=2,
            input_tokens=list(range(prefix)) + [0] * 128,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        try:
            assert manager.prepare_context(reader)
            assert reader.context_current_position == prefix
            cache = manager.kv_cache_map[1]
            assert cache.num_committed_tokens == prefix
            assert isinstance(reader.py_ced_replay, EncoderReplay)
            assert reader.py_ced_replay.start == prefix - 128
            assert not any(
                cache.reuse_status[group].complete for group in manager._encoder_optional_groups
            )
            for layer in manager.layout.kv_source_layer_ids:
                for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
                    source_pages = manager.get_cache_indices(0, layer, role)
                    reader_pages = manager.get_cache_indices(1, layer, role)
                    assert source_pages[0] == reader_pages[0]
                    assert source_pages[1] != reader_pages[1]
                    pool = manager.get_buffers(layer, role)
                    original = pool[source_pages[1]].clone()
                    torch.testing.assert_close(pool[reader_pages[1]], original, rtol=0, atol=0)
                    pool[reader_pages[1]].view(torch.uint8).fill_(17)
                    torch.testing.assert_close(pool[source_pages[1]], original, rtol=0, atol=0)
        finally:
            manager.free_resources(reader)


@pytest.mark.parametrize("encoder_replay", [False, True])
@pytest.mark.parametrize("capture_window", [128, 256])
def test_recovery_retention_matches_static_sizing(
    cache_config, monkeypatch, encoder_replay, capture_window
):
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as base
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", str(int(encoder_replay)))
    spec = None
    if capture_window > 128:
        monkeypatch.setattr(config.pretrained_config, "sliding_window", capture_window)
        spec = DSparkDecodingConfig(max_draft_len=5)
        spec.__dict__["draft_is_embedded_in_target"] = True
    with cache_case(config, sparse, dtype, [384], [0], 1, reuse=True, spec_config=spec) as case:
        manager = case[0].kv_cache_manager
        assert manager.layout.window_size == 128
        reserve = manager.max_draft_len + manager.reuse_match_backoff
        for layer in manager.pp_layers:
            physical = manager._layer_roles[layer, CSA2CacheRole.SWA]
            window = manager.impl.init_config.layers[physical].sliding_window_size
            expected = (
                capture_window + 1 if encoder_replay else capture_window if layer >= 5 else 128
            )
            assert window == expected + reserve
        sizes, windows = manager._get_runtime_cache_size_layer_components()
        per_token, fixed = base._estimate_swa_cache_size(
            sizes, windows, 128, context=False, scratch=False
        )
        expected = (base._estimate_full_attn_size_per_token(sizes, windows) + per_token, fixed * 3)
        assert (
            CSA2CacheManager.get_cache_size_per_token(
                config,
                config.pretrained_config.mapping,
                tokens_per_block=128,
                max_batch_size=3,
                spec_config=spec,
                kv_cache_config=KvCacheConfig(enable_block_reuse=True),
            )
            == expected
        )


def test_dummy_context_uses_base_lifecycle_without_publishing(cache_config, monkeypatch):
    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    with cache_case(config, sparse, dtype, [1], [0], 1, reuse=True) as case:
        manager = case[0].kv_cache_manager
        manager.free_resources(request(0, 1))
        dummy = manager.add_dummy_requests([1], token_nums=[257], is_gen=False)[0]
        try:
            cache = manager.kv_cache_map[1]
            assert manager.prepare_context(dummy)
            cache.suspend()
            assert manager.prepare_context(dummy)
            assert dummy.context_current_position == 0
            assert cache.num_committed_tokens == 0
            assert dummy.py_ced_replay is None
            dummy.context_current_position = 257
            scheduled = ScheduledRequests()
            scheduled.context_requests_last_chunk = [dummy]
            manager.update_context_resources(scheduled)
            assert cache.num_committed_tokens == 0
        finally:
            manager.free_resources(dummy)
        reader = LlmRequest(
            request_id=2,
            max_new_tokens=2,
            input_tokens=[1] * 385,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        try:
            assert manager.prepare_context(reader)
            assert reader.context_current_position == 0
            assert reader.py_ced_replay is None
        finally:
            manager.free_resources(reader)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("encoder_replay", [False, True])
def test_encoder_snapshot_resumes_without_replay(
    model_and_config, monkeypatch, chunked, encoder_replay
):
    model, config, sparse, dtype = model_and_config
    enable_ced(model)
    model.model.encoder_replay_enabled = encoder_replay
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", str(int(encoder_replay)))
    window = model.model.decoder_replay_window
    chunks = [1, window - 2, 1] if chunked else [window]
    total = 256 + window
    ids = torch.arange(total, device="cuda", dtype=torch.int32)
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [total], [0], 1, reuse=True) as case,
        canonical_routing(),
    ):
        metadata = case[0]
        manager = metadata.kv_cache_manager
        manager.enable_swa_scratch_reuse = False
        warm = request(0, total)
        # Match GEMM partitions on the no-reuse reference so the comparison
        # isolates checkpoint restore from BF16 shape-dependent rounding.
        start = 0
        for length in [256, *chunks]:
            end = start + length
            if start == 256:
                # Match the empty private Decoder history of a new cache hit.
                warm.py_ced_replay = EncoderCheckpoint(0, id(manager.kv_cache_map[0]), 256)
            reference = run(model, metadata, [warm], ids[start:end], [start], [length], 1)
            start = end
        expected = reference.clone()
        manager.kv_cache_map[0].commit(list(range(256)))
        frozen = global_bytes(metadata, 0, 256)
        saved = []
        for layer, role in manager._encoder_roles:
            slot = manager.get_cache_indices(0, layer, role)[1]
            saved.append((layer, role, slot, manager.get_buffers(layer, role)[slot].clone()))
        manager.free_resources(warm)

        short = request(2, 257)
        # Missing HC outputs cannot be repaired by implicit Encoder replay.
        if not encoder_replay:
            assert manager.probe_context_reuse(short) <= max(0, 257 - window)
        fresh = request(1, total)
        assert manager.probe_context_reuse(fresh) == 256
        assert manager.prepare_context(fresh)
        assert fresh.context_current_position == 256
        assert isinstance(fresh.py_ced_replay, EncoderCheckpoint)
        for layer, role, slot, _ in saved:
            assert manager.get_cache_indices(1, layer, role)[1] == slot
        if encoder_replay and chunked:
            from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

            def fail_resize(self, req, num_tokens):
                cache = self.kv_cache_map[req.py_request_id]
                assert cache.is_active
                cache.suspend()
                return False

            fresh.context_chunk_size = chunks[0]
            plan = fresh.py_ced_replay
            with monkeypatch.context() as patch:
                patch.setattr(KVCacheManagerV2, "resize_context", fail_resize)
                assert not manager.resize_context(fresh, chunks[0])
            assert manager.prepare_context(fresh)
            assert fresh.py_ced_replay is plan
            assert manager.context_replay_tokens(fresh) == 0
        start = 256
        for i, length in enumerate(chunks):
            final = i == len(chunks) - 1
            fresh.context_current_position = start
            fresh.context_chunk_size = length
            assert manager.resize_context(fresh, length)
            scheduled = ScheduledRequests()
            scheduled.context_requests_last_chunk = [fresh] if final else []
            scheduled.context_requests_chunking = [] if final else [fresh]
            manager.prepare_resources(scheduled)
            metadata.request_ids = [1]
            assert encoder_replay_tokens(fresh) == 0
            output = run(
                model, metadata, [fresh], ids[start : start + length], [start], [length], 1
            )
            start += length
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
        for layer, role, slot, before in saved:
            torch.testing.assert_close(
                manager.get_buffers(layer, role)[slot].view(torch.uint8),
                before.view(torch.uint8),
                rtol=0,
                atol=0,
            )
        for name, value in global_bytes(metadata, 0, 256).items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        manager.free_resources(fresh)


@pytest.mark.parametrize("missing_page", [False, True])
def test_encoder_window_spans_two_blocks(cache_config, monkeypatch, missing_page):
    config, sparse, dtype = cache_config
    monkeypatch.setattr(config.pretrained_config, "sliding_window", 256)
    monkeypatch.setattr(config.pretrained_config.text_config, "sliding_window", 256)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    with cache_case(config, sparse, dtype, [768], [0], 1, reuse=True) as case:
        manager = case[0].kv_cache_manager
        assert manager.layout.window_size == 256 and manager.tokens_per_block == 128
        warm = manager.kv_cache_map[0]
        warm.commit(list(range(512)))
        swa = manager._layer_roles[0, CSA2CacheRole.SWA]
        group = manager.impl.get_layer_group_id(swa)
        original = manager.get_cache_indices(0, 0, CSA2CacheRole.SWA)[2:4]
        if missing_page:
            _introspection.drop_optional_pages(warm, 384, [group])
        manager.free_resources(request(0, 768))
        fresh = request(1, 768)
        assert manager.prepare_context(fresh)
        assert manager.kv_cache_map[1].is_active
        assert fresh.context_current_position == 512
        assert manager.context_replay_tokens(fresh) == (256 if missing_page else 0)
        fresh.context_chunk_size = 256
        assert manager.resize_context(fresh, 256)
        if not missing_page:
            assert manager.get_cache_indices(1, 0, CSA2CacheRole.SWA)[2:4] == original
        manager.free_resources(fresh)


@pytest.mark.parametrize("policy", ["cold", "hit", "short-suffix", "evicted"])
@pytest.mark.parametrize("dspark", [False, True])
def test_context_prepare_resumes_before_budgeting_and_retries(
    cache_config, monkeypatch, policy, dspark
):
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    spec = None
    if dspark:
        from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

        spec = DSparkDecodingConfig(max_draft_len=5)
        spec.__dict__["draft_is_embedded_in_target"] = True
    with cache_case(config, sparse, dtype, [384], [0], 1, reuse=True, spec_config=spec) as case:
        manager = case[0].kv_cache_manager
        if policy != "cold":
            manager.kv_cache_map[0].commit(list(range(256)))
            if policy == "evicted":
                _introspection.drop_optional_pages(manager.kv_cache_map[0], 256)
        manager.free_resources(request(0, 384))
        req = request(1, 257 if policy == "short-suffix" else 384)
        resume = manager._resume_and_restore
        selections = []

        def try_resume(req_id, cache, groups=None):
            selections.append(groups)
            if len(selections) == 1:
                assert not cache.is_active
                return False
            return resume(req_id, cache, groups)

        monkeypatch.setattr(manager, "_resume_and_restore", try_resume)
        assert not manager.prepare_context(req)
        cache = manager.kv_cache_map[1]
        assert not cache.is_active
        assert req.py_ced_replay is None
        assert req.context_current_position == 0

        assert manager.prepare_context(req)
        assert cache.is_active
        assert req.context_current_position == (0 if policy == "cold" else 256)
        plan = req.py_ced_replay
        if policy == "hit":
            assert isinstance(plan, EncoderCheckpoint)
            assert selections[-1] == manager._encoder_optional_groups
        elif policy != "cold":
            assert isinstance(plan, EncoderReplay)
            assert selections[-1] == ()
        else:
            assert plan is None
        expected_tokens = None if policy == "cold" else 0 if policy == "hit" else 128
        assert manager.context_replay_tokens(req) == expected_tokens

        # A budget rejection or allocation failure can suspend the admitted
        # cache before forward. Resume must not repeat initial OPTIONAL selection.
        cache.suspend()
        assert manager.prepare_context(req)
        assert cache.is_active
        assert selections[-1] is None
        assert req.py_ced_replay is plan
        assert manager.context_replay_tokens(req) == expected_tokens
        manager.free_resources(req)
        assert req.py_ced_replay is None
