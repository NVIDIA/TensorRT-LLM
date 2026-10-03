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
    start = 128 if recovering else 0
    end = start + 129
    with cache_case(config, sparse, dtype, [129, 6], [start, 128], 1, spec_config=spec) as case:
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
            assert plan.replay_seq_lens.tolist() == [128, 1]
            assert plan.swa_floors == [129, 0]
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


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
        assert bool(manager._decoder_replay_window) is not scratch
        initial = [buffer.clone() for buffer in buffers]
        req = request(0, len(ids))
        req.py_return_context_logits = True
        outputs = []
        for bounded in (False, True):
            for buffer, saved in zip(buffers, initial):
                buffer.copy_(saved)
            if bounded:
                enable_ced(model)
            else:
                model.model.decoder_replay_split = None
            result, start = [], 0
            for length in chunks:
                end = start + length
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
                start = end
            outputs.append(torch.cat(result))
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.02, rtol=0.02)


def test_full_context_logits_after_truncated_peer(model_and_config, monkeypatch):
    model, config, sparse, dtype = model_and_config
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [513, 257], [0, 0], 2) as case,
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


def test_dspark_verification_retains_all_generation_rows(monkeypatch):
    from test_modeling_deepseekv41 import _make_policy_case, _ReplayMetadata

    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Model

    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    model, config = _make_policy_case()
    config.pretrained_config.sliding_window = 256
    config.spec_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
        draft_is_embedded_in_target=True,
        target_layer_ids=[37, 38, 39],
    )
    model.decoder_replay_split, model.decoder_replay_window = (
        DeepseekV41Model._resolve_replay_policy(model, config)
    )
    model._decoder_replay_observed = False
    model._decoder_replay_reuse_warning_emitted = False
    metadata = _ReplayMetadata([513, 6], num_contexts=1)
    split, plan = model._plan_bounded_replay(metadata, False)
    assert split == 20
    assert model.decoder_replay_window == 256
    assert plan.replay_seq_lens.tolist() == [256, 6]
    assert plan.rows.tolist() == list(range(257, 519))


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
        assert metadata.csa2_decoder_capture_lens == (0, 128)
        assert metadata.seq_lens.tolist() == lengths
        assert actual.shape == expected.shape == (3, 1024)
        torch.testing.assert_close(actual[1:], expected[1:], atol=0.02, rtol=0.02)


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
    monkeypatch.setenv("TRTLLM_V41_MHC_DEFER", str(int(defer_post)))

    def configure_deferral(ced_enabled):
        model.model.ced_kv_precompute = ced_enabled
        with patch.object(DeepseekV4ForCausalLM, "post_load_weights"):
            model.post_load_weights()
        for index, layer in enumerate(model.model.layers):
            next_layer = (
                model.model.layers[index + 1] if index + 1 < len(model.model.layers) else None
            )
            assert layer._v41_defer_post_mapping == bool(
                defer_post
                and next_layer is not None
                and next_layer.engram is None
                and not (ced_enabled and index + 1 == model.model.decoder_replay_split)
            )

    window = model.model.decoder_replay_window
    total = sum(chunks)
    with (
        torch.inference_mode(),
        cache_case(config, sparse, dtype, [total], [0], 1) as case,
    ):
        metadata, ids, _, buffers = case
        manager = metadata.kv_cache_manager
        assert manager.layout.window_size == 128
        assert manager._swa_retention(0) == 128
        assert manager._swa_retention(5) == window
        manager.enable_swa_scratch_reuse = False
        baseline = [b.clone() for b in buffers]
        req = request(0, total)
        # Sort only native Top-K order, as in the producer parity tests.
        with ExitStack() as stack:
            stack.enter_context(canonical_routing())
            configure_deferral(False)
            ref = independent_window_reference(model, metadata, ids, 128, chunks=chunks).clone()
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
