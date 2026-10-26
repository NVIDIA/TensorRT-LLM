# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final prompt-window decoder replay and ordinary KV continuation."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _deepseek_v41_test_utils import (
    _apply_swa_floor,
    cache_case,
    canonical_routing,
    enable_ced,
    global_precompute_scope,
    request,
    run,
)
from _deepseek_v41_test_utils import cache_config as shared_cache_config
from _deepseek_v41_test_utils import model_and_config as global_model_and_config

from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)

cache_config = shared_cache_config

model_and_config = global_model_and_config


def independent_window_reference(model, metadata, ids, window=128, chunks=None):
    """Keep all encoder outputs, then decode only the final prompt window.

    Match encoder and decoder GEMM shapes to isolate the scheduling/retention
    contract from BF16 and packed-cache rounding across prefill partitions.
    """
    body = model.model
    positions = torch.arange(len(ids), device=ids.device, dtype=torch.int32).view(1, -1)
    offset = 0
    residuals, pre_mix = [], []
    for length in chunks or (len(ids),):
        end = offset + length
        metadata.seq_lens = torch.tensor([length], dtype=torch.int32)
        metadata.prompt_lens = [length]
        metadata.kv_cache_params.num_cached_tokens_per_seq = [offset]
        metadata.prepare()
        embedded = body.embed_tokens(ids[offset:end]).unsqueeze(1).repeat(1, body.hc_mult, 1)
        state = body._init_hc_state(embedded)
        for layer in body.layers[:5]:
            state = layer(
                position_ids=positions[:, offset:end],
                hc_state=state,
                attn_metadata=metadata,
                input_ids=ids[offset:end],
            )
        state = body._resolve_hc_state(state)
        residuals.append(state.residual)
        pre_mix.append(state.pre_mix)
        with global_precompute_scope(metadata):
            owner = body.layers[5]
            owner.self_attn.prepare_global_cache(owner._decoder_global_input(state), metadata)
        offset = end
    assert offset == len(ids), "Reference chunks must cover the full input"
    floor = max(0, len(ids) - window) if window is not None else 0
    residuals, pre_mix = torch.cat(residuals), torch.cat(pre_mix)
    offset = 0
    for length in chunks or (len(ids),):
        end = offset + length
        start = max(offset, floor)
        offset = end
        if start >= end:
            continue
        state = HCState.resolved(residuals[start:end], pre_mix=pre_mix[start:end])
        with global_precompute_scope(metadata):
            metadata.seq_lens = torch.tensor([end - start], dtype=torch.int32)
            metadata.prompt_lens = [end - start]
            metadata.kv_cache_params.num_cached_tokens_per_seq = [start]
            metadata.prepare()
            metadata.csa2_precomputed_kv_layers = {5}
            _apply_swa_floor(
                metadata,
                torch.full((end - start,), floor, device=ids.device, dtype=torch.int64),
            )
            for layer in body.layers[5:]:
                state = layer(
                    position_ids=positions[:, start:end],
                    hc_state=state,
                    attn_metadata=metadata,
                    input_ids=ids[start:end],
                )
    hidden = body._finalize_hc_state(state)[-1:]
    return model.logits_processor(hidden, model.lm_head, None, return_context_logits=True)


@skip_pre_blackwell
@pytest.mark.parametrize("recovering", [False, True])
def test_prepared_encoder_replay_preserves_live_speculative_endpoints(
    cache_config, monkeypatch, recovering
):
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    spec = DSparkDecodingConfig(max_draft_len=5)
    spec.__dict__["draft_is_embedded_in_target"] = True
    start = 116 if recovering else 0
    end = start + 141
    with cache_case(config, sparse, dtype, [141, 6], [start, 128], 1, spec_config=spec) as case:
        metadata = case[0]
        req = request(0, end)
        if recovering:
            req.context_current_position = 256
            req.context_chunk_size = 1
            req.py_ced_replay = EncoderReplay(
                0, id(metadata.kv_cache_manager.kv_cache_map[0]), 256, start
            )
        metadata.prepare_context_replay([req])
        metadata.prepare()
        # Host reservation ends at134; accepting one of five drafts moves the
        # actual six verification rows back to [124,130). The allocator may
        # only have pages covering that live interval.
        metadata.kv_lens_cuda[:2].copy_(torch.tensor([end, 130], device="cuda"))
        metadata.on_update_kv_lens()
        positions = metadata.csa2_positions.clone()
        swa_slots = {layer: slots.clone() for layer, slots in metadata.csa2_swa_write_slots.items()}
        for _ in range(2):
            metadata.begin_model_forward()
            metadata.on_update_kv_lens()
            assert metadata.kv_lens_cuda[:2].tolist() == [end, 130]
            torch.testing.assert_close(metadata.csa2_positions, positions)
            assert metadata.csa2_positions[-6:].tolist() == list(range(124, 130))
            assert metadata.csa2_replay_start_positions.tolist() == [start, 0]
            for layer, slots in swa_slots.items():
                torch.testing.assert_close(metadata.csa2_swa_write_slots[layer], slots)
                assert (metadata.csa2_swa_indices[layer][0, :-1] == -1).all()
            for batch in metadata._csa2_compression.values():
                assert batch.kv_lengths.tolist() == [end, 130]
                assert batch.start_positions.tolist() == [256 if recovering else 0, 124]
        if recovering:
            req.py_ced_replay.consumed = True
            req.context_current_position = end
            metadata.seq_lens = torch.tensor([1, 6], dtype=torch.int32)
            metadata.kv_cache_params.num_cached_tokens_per_seq = [end, 128]
            metadata.prepare_context_replay([req])
            metadata.prepare()
            assert metadata.csa2_replay_mode is None
            assert metadata.csa2_replay_start_positions.tolist() == [start, 0]
            req.py_ced_replay = None
            metadata.prepare_context_replay([req])
            metadata.prepare()
            assert metadata.csa2_replay_start_positions.tolist() == [0, 0]


@pytest.mark.parametrize("recovering", [False, True])
def test_short_chunks_exclude_uninitialized_private_decoder_history(recovering):
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

    req = request(0, 5000)
    req.py_ced_replay = (
        EncoderReplay(0, 1, 4000, 3872) if recovering else EncoderCheckpoint(0, 1, 4000)
    )
    start = 3872 if recovering else 4000
    from test_modeling_deepseekv41 import _ReplayMetadata

    from tensorrt_llm._torch.attention.backends.sparse.csa2.decoder_replay import (
        plan_decoder_replay,
    )

    for offset, length, kept in [(start, 64, 0), (4800, 100, 28), (4900, 99, 99), (4999, 1, 1)]:
        metadata = _ReplayMetadata([length], num_contexts=1)
        metadata.kv_cache_params.num_cached_tokens_per_seq = [offset]
        plan = plan_decoder_replay(metadata, 128, [req], private_decoder=True)
        assert plan.replay_seq_lens.tolist() == [kept]
        assert plan.swa_floors == [4872]


@skip_pre_blackwell
@pytest.mark.parametrize("encoder_replay", [False, True])
def test_dummy_forward_replays_with_or_without_requests(
    model_and_config, monkeypatch, encoder_replay
):
    model, config, sparse, dtype = model_and_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", str(int(encoder_replay)))
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [257, 1], [0, 0], 1, reuse=True) as case,
        canonical_routing(),
    ):
        metadata, ids, _, buffers = case
        enable_ced(model)
        initial = [buffer.clone() for buffer in buffers]
        reqs = [request(0, 257), request(1, 1)]
        outputs = []
        for pass_requests in (True, False):
            for buffer, saved in zip(buffers, initial):
                buffer.copy_(saved)
            with patch.object(
                model.model, "_enter_bounded_replay", wraps=model.model._enter_bounded_replay
            ) as enter:
                outputs.append(
                    run(
                        model,
                        metadata,
                        reqs,
                        ids,
                        [0, 0],
                        [257, 1],
                        1,
                        pass_requests=pass_requests,
                    ).clone()
                )
            enter.assert_called_once()
            plan = enter.call_args.args[0]
            assert plan.replay_seq_lens.tolist() == [140, 1]
            assert plan.swa_floors == [117, 0]
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize("chunks", [(513,), (256, 257)])
def test_full_context_outputs_preserve_full_prefill(model_and_config, monkeypatch, scratch, chunks):
    """Compare every requested row, including chunks before the final window."""
    model, config, sparse, dtype = model_and_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    with (
        torch.inference_mode(),
        cache_case(
            config, sparse, dtype, [sum(chunks)], [0], 1, reuse=True, scratch=scratch
        ) as case,
        canonical_routing(),
    ):
        metadata, ids, _, buffers = case
        manager = metadata.kv_cache_manager
        assert manager._decoder_replay_window == 140
        initial = [buffer.clone() for buffer in buffers]
        outputs = []
        for bounded in (False, True):
            manager.free_resources(request(0, len(ids)))
            for buffer, saved in zip(buffers, initial):
                buffer.copy_(saved)
            req = LlmRequest(
                request_id=0,
                max_new_tokens=2,
                input_tokens=list(range(len(ids))),
                sampling_config=SamplingConfig(),
                is_streaming=False,
                return_context_logits=True,
            )
            assert manager.prepare_context(req)
            assert req.context_current_position == 0
            if bounded:
                enable_ced(model)
            else:
                model.model.decoder_replay_split = None
            result, start = [], 0
            for length in chunks:
                end = start + length
                req.context_current_position = start
                req.context_chunk_size = length
                assert manager.resize_context(req, length)
                logits = run(
                    model,
                    metadata,
                    [req],
                    ids[start:end],
                    [start],
                    [length],
                    1,
                    return_context_logits=True,
                )
                assert isinstance(logits, torch.Tensor)
                assert logits.shape[0] == length
                result.append(logits.clone())
                assert manager.kv_cache_map[0].resize(None, end)
                start = end
            outputs.append(torch.cat(result))
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.02, rtol=0.02)


@skip_pre_blackwell
def test_full_context_logits_after_truncated_peer(model_and_config, monkeypatch):
    model, config, sparse, dtype = model_and_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [513, 257], [0, 0], 2, scratch=True) as case,
        canonical_routing(),
    ):
        metadata, ids, _, buffers = case
        initial = [buffer.clone() for buffer in buffers]
        reqs = [request(0, 513), request(1, 257)]
        reqs[1].py_return_context_logits = True
        model.model.decoder_replay_split = None
        reference = run(
            model,
            metadata,
            reqs,
            ids,
            [0, 0],
            [513, 257],
            2,
            return_context_logits=True,
        )
        for buffer, saved in zip(buffers, initial):
            buffer.copy_(saved)
        enable_ced(model)
        actual = run(
            model,
            metadata,
            reqs,
            ids,
            [0, 0],
            [513, 257],
            2,
            return_context_logits=True,
        )
        assert actual.shape[0] == 770
        torch.testing.assert_close(actual[513:], reference[513:], atol=0.02, rtol=0.02)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "chunks,full_output",
    [
        pytest.param((512, 256, 129), None, id="empty-decoder-chunks"),
        pytest.param((384, 127, 130), "logits", id="chunked-full-logits"),
        pytest.param((513,), "hidden", id="full-hidden-states"),
    ],
)
def test_chunk_allocated_scratch_replay_matches_paged_continuation(
    model_and_config, monkeypatch, chunks, full_output
):
    """Use scheduler-sized allocations, including empty and partial Decoder windows."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole

    model, config, sparse, dtype = model_and_config
    enable_ced(model)
    model.model.encoder_replay_enabled = True
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    total = sum(chunks)
    ids = torch.arange(total, device="cuda", dtype=torch.int32)
    results = []
    for scratch in (False, True):
        with (
            torch.inference_mode(),
            cache_case(config, sparse, dtype, [1], [0], 1, reuse=True, scratch=scratch) as case,
            canonical_routing(),
        ):
            metadata = case[0]
            manager = metadata.kv_cache_manager
            manager.free_resources(request(0, 1))
            req = LlmRequest(
                request_id=0,
                max_new_tokens=2,
                input_tokens=list(range(total)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
                return_context_logits=full_output == "logits",
                additional_outputs=["hidden_states"] if full_output == "hidden" else None,
            )
            assert manager.prepare_context(req)
            cache = manager.kv_cache_map[0]
            outputs, offset = [], 0
            for length in chunks:
                req.context_current_position = offset
                req.context_chunk_size = length
                assert manager.resize_context(req, length)
                if scratch and offset == 0:
                    encoder = manager._layer_roles[0, 1, CSA2CacheRole.SWA]
                    decoder = manager._layer_roles[5, 1, CSA2CacheRole.SWA]
                    pools = [
                        manager.layer_to_pool_mapping_dict[layer] for layer in (encoder, decoder)
                    ]
                    assert pools[0] != pools[1]
                    assert all(cache.get_scratch_desc(pool).slot_ids for pool in pools)
                    assert (
                        manager.get_swa_buffer(0).data_ptr() != manager.get_swa_buffer(5).data_ptr()
                    )
                end = offset + length
                hidden = []
                with model.model.register_forward_hook(
                    lambda _module, _inputs, states: hidden.append(states.clone())
                ):
                    output = run(
                        model,
                        metadata,
                        [req],
                        ids[offset:end],
                        [offset],
                        [length],
                        1,
                        return_context_logits=full_output == "logits",
                    )
                kept = length if full_output else max(0, end - max(offset, total - 140))
                assert metadata.csa2_decoder_capture_lens == (kept,)
                if full_output == "hidden":
                    assert hidden[0].shape[0] == length
                    outputs.append(hidden[0])
                if full_output or end == total:
                    outputs.append(output.clone())
                req.context_current_position = end
                assert cache.resize(None, end)
                # Retained Decoder pages survive ordinary request suspension.
                if kept and end < total:
                    cache.suspend()
                    assert manager.prepare_context(req)
                offset = end
            assert not cache.has_scratch_slots
            cache.enable_swa_scratch_reuse = False
            assert cache.resize(total + 1, total)
            outputs.append(
                run(
                    model,
                    metadata,
                    [req],
                    torch.tensor([7], device="cuda", dtype=torch.int32),
                    [total],
                    [1],
                    0,
                ).clone()
            )
            results.append(outputs)
    for paged, scratch in zip(*results):
        torch.testing.assert_close(scratch, paged, atol=0.02, rtol=0.02)


@skip_pre_blackwell
@pytest.mark.parametrize("reuse", [False, True])
def test_dspark_scratch_preserves_reserved_pages(cache_config, monkeypatch, reuse):
    """Keep reserved normal pages writable across chunks, suspension and publication."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    spec = DSparkDecodingConfig(max_draft_len=5)
    spec.__dict__["draft_is_embedded_in_target"] = True
    with cache_case(
        config, sparse, dtype, [1], [0], 1, reuse=reuse, scratch=True, spec_config=spec
    ) as case:
        manager = case[0].kv_cache_manager
        manager.free_resources(request(0, 1))
        req = LlmRequest(
            request_id=0,
            max_new_tokens=2,
            input_tokens=list(range(1024)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(req)
        cache = manager.kv_cache_map[0]
        encoder = manager._layer_roles[0, 1, CSA2CacheRole.SWA]
        pool = manager.layer_to_pool_mapping_dict[encoder]
        converter = manager.impl.get_page_index_converter(encoder, CSA2CacheRole.SWA.role)
        buffer = manager.get_buffers(0, CSA2CacheRole.SWA)
        for offset in (0, 512):
            req.context_current_position = offset
            req.context_chunk_size = 512
            assert manager.resize_context(req, 512)
            assert cache.capacity == offset + 516
            base = cache.get_base_page_indices(pool).tolist()
            if offset == 0:
                reserved_page = base[4]
                assert reserved_page >= 0
                reserved_index = converter([reserved_page], CSA2CacheRole.SWA.index_mode)[0]
            else:
                desc = cache.get_scratch_desc(pool)
                assert desc.range.beg <= 4 < desc.range.end
                assert base[4] == reserved_page
                assert manager.get_cache_indices(0, 0, CSA2CacheRole.SWA)[4] == reserved_index
                cache.suspend()
                assert manager.prepare_context(req)
            manager.compute_batch_page_tables([0], 1)
            pages = manager.batch_page_table((0, CSA2CacheRole.SWA), cache.num_blocks)[0]
            if offset:
                assert pages[4].item() == reserved_index
            # Write through the device table consumed by attention. Published
            # normal pages must contain these bytes, rather than untouched storage.
            buffer.index_fill_(0, pages[offset // 128 : (offset + 512) // 128].long(), 17)
            req.context_current_position = offset + 512
            if reuse:
                manager.try_commit_blocks(req)
            else:
                assert cache.resize(None, offset + 512)
        if reuse:
            count, pages = _introspection.reuse_match_pages(
                manager.impl, cache.reuse_scope, list(range(1024)), pool
            )
            assert count == 1024
            assert pages[4] is not None
            assert pages[4][0] == reserved_page
            assert (buffer[reserved_index] == 17).all()
            assert pages[5] is None  # A genuinely scratch-backed block is never published.


@pytest.mark.parametrize(
    "capture_window,replay_tokens",
    [(128, None), (128, "129"), (256, "300"), (128, "128"), (256, "140"), (128, "bad")],
)
def test_dspark_verification_retains_all_generation_rows(
    monkeypatch, capture_window, replay_tokens
):
    from test_modeling_deepseekv41 import _make_policy_case, _ReplayMetadata

    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Model

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model, config = _make_policy_case()
    config.pretrained_config.sliding_window = capture_window
    if replay_tokens is None:
        monkeypatch.delenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", replay_tokens)
    config.spec_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
        draft_is_embedded_in_target=True,
        target_layer_ids=[37, 38, 39],
    )
    if replay_tokens in ("128", "bad") or capture_window == 256 and replay_tokens == "140":
        with pytest.raises(ValueError, match="TRTLLM_V41_BOUNDED_REPLAY_TOKENS"):
            DeepseekV41Model._resolve_replay_policy(model, config)
        return
    model.decoder_replay_split, model.decoder_replay_window = (
        DeepseekV41Model._resolve_replay_policy(model, config)
    )
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    metadata = _ReplayMetadata([513, 6], num_contexts=1)
    split, plan = model._plan_bounded_replay(metadata, False)
    assert split == 20
    window = int(replay_tokens or 140)
    assert model.decoder_replay_window == window
    assert plan.replay_seq_lens.tolist() == [window, 6]
    assert plan.rows.tolist() == list(range(513 - window, 519))


@skip_pre_blackwell
def test_final_window_mixed_batch_keeps_finished_context_and_generation(
    model_and_config, monkeypatch
):
    model, config, sparse, dtype = model_and_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    enable_ced(model)
    lengths = [257, 129, 1]
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, lengths, [0, 0, 0], 2) as case,
        canonical_routing(),
    ):
        metadata, ids, _, buffers = case
        initial = [buffer.clone() for buffer in buffers]
        reqs = [request(0, 257), request(1, 129), request(2, 1)]
        expected = run(model, metadata, reqs, ids, [0, 0, 0], lengths, 2).clone()
        for buffer, saved in zip(buffers, initial):
            buffer.copy_(saved)
        reqs[0].prompt_len = 1024
        actual = run(model, metadata, reqs, ids, [0, 0, 0], lengths, 2)
        assert metadata.csa2_decoder_capture_lens == (0, 129)
        assert metadata.seq_lens.tolist() == lengths
        assert actual.shape == expected.shape == (3, 1024)
        torch.testing.assert_close(actual[1:], expected[1:], atol=0.02, rtol=0.02)


@skip_pre_blackwell
def test_small_token_budget_falls_back_to_required_encoder_state(cache_config, monkeypatch):
    from _deepseek_v41_test_utils import _init_random_weights

    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM

    config, sparse, dtype = cache_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.delenv("TRTLLM_V41_ENCODER_REPLAY", raising=False)
    config.max_num_tokens = 3
    model = DeepseekV41ForCausalLM(config).cuda()
    _init_random_weights(model)
    assert not model.model.encoder_replay_enabled
    assert model.model.decoder_replay_window == 140
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [3], [0], 1, reuse=True, max_num_tokens=3) as case,
        canonical_routing(),
    ):
        metadata, ids, _, _ = case
        manager = metadata.kv_cache_manager
        assert not manager._encoder_replay_enabled
        assert manager._encoder_reuse_policy.name == "REQUIRED"
        assert not manager._encoder_optional_groups
        assert manager._decoder_replay_window == 140
        assert manager._swa_retention(0) == 128
        output = run(model, metadata, [request(0, 3)], ids, [0], [3], 1)
        assert output.shape == (1, 1024)
        assert torch.isfinite(output).all()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "chunks", [(513,), (128, 128, 128, 128, 1), (127, 129, 127, 130), (1, 127)]
)
@pytest.mark.parametrize("defer_post", [False, True])
def test_chunked_ced_matches_independent_window_reference(
    model_and_config, monkeypatch, chunks, defer_post
):
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4ForCausalLM

    model, config, sparse, dtype = model_and_config
    enable_ced(model)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.delenv("TRTLLM_V41_ENCODER_REPLAY", raising=False)
    monkeypatch.setenv("TRTLLM_V41_MHC_DEFER", str(int(defer_post)))

    def configure_deferral(ced_enabled):
        model.model.ced_kv_precompute = ced_enabled
        with patch.object(DeepseekV4ForCausalLM, "post_load_weights"):
            model.post_load_weights()
        # The last layer defers into the stream close.
        for layer in model.model.layers:
            assert layer._v41_defer_post_mapping == defer_post

    window = model.model.decoder_replay_window
    total = sum(chunks)
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [total], [0], 1) as case,
    ):
        metadata, ids, _, buffers = case
        manager = metadata.kv_cache_manager
        assert manager.layout.window_size == 128
        # Recovery keeps its first replay row without widening the attention window.
        assert manager._swa_retention(0) == window + 1
        assert manager._swa_retention(5) == window + 1
        manager.enable_swa_scratch_reuse = False
        baseline = [b.clone() for b in buffers]
        req = request(0, total)
        # Sort only native Top-K order, as in the producer parity tests.
        with ExitStack() as stack:
            stack.enter_context(canonical_routing())
            configure_deferral(False)
            ref = independent_window_reference(model, metadata, ids, window, chunks=chunks).clone()
            for b, old in zip(buffers, baseline):
                b.copy_(old)
            enable_ced(model)
            configure_deferral(True)
            offset = 0
            owner = model.model.layers[5]
            with patch.object(
                owner.self_attn.compressor, "forward", wraps=owner.self_attn.compressor.forward
            ) as production:
                for j, n in enumerate(chunks):
                    final = j == len(chunks) - 1
                    # Exercise actual history reclamation between chunks.
                    cache = manager.kv_cache_map[0]
                    assert cache.resize(cache.capacity, offset)
                    with patch.object(owner, "forward", wraps=owner.forward) as query:
                        out = run(
                            model,
                            metadata,
                            [req],
                            ids[offset : offset + n],
                            [offset],
                            [n],
                            1,
                        )
                        rows = len(
                            set(range(offset, offset + n))
                            & set(range(max(0, total - window), total))
                        )
                        assert query.call_count == 1
                        if rows:
                            assert query.call_args.kwargs[
                                "position_ids"
                            ].flatten().tolist() == list(range(offset + n - rows, offset + n))
                    assert out.shape == (1, 1024)
                    assert metadata.seq_lens.tolist() == [n]
                    assert metadata.csa2_decoder_capture_lens == (rows,)
                    assert not metadata.csa2_precomputed_kv_layers
                    if rows and not final:
                        # Decoder KV, rather than retained HC tensors, carries
                        # the partial window across ordinary suspend/resume.
                        cache.suspend()
                        assert cache.resume()
                        req.py_seq_slot += 1
                    offset += n
                assert production.call_count == len(chunks)
                assert [c.args[0].shape[0] for c in production.call_args_list] == list(chunks)
            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
