# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global-only claims, frozen-prefix numerics, and encoder replay continuation."""

import weakref
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from _deepseek_v41_test_utils import (
    _apply_swa_floor,
    cache_case,
    canonical_routing,
    enable_ced,
    global_bytes,
    global_precompute_scope,
    run,
)
from _deepseek_v41_test_utils import cache_config as shared_cache_config
from _deepseek_v41_test_utils import model_and_config as global_model_and_config

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState
from tensorrt_llm._torch.pyexecutor.ced_replay import encoder_replay_tokens
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

cache_config = shared_cache_config

model_and_config = global_model_and_config


def test_global_source_selection_preserves_cold_and_generation_rows(cache_config):
    config, sparse, dtype = cache_config
    # Recovery [128,258), cold [0,3), decode [258,259). Only uncached
    # recovery rows are projected; the other requests keep their complete input.
    with cache_case(config, sparse, dtype, [130, 3, 1], [128, 0, 258], 2) as case:
        metadata = case[0]
        metadata.set_swa_bounded_replay([256, None, None])
        metadata.prepare()
        hidden = torch.arange(134, device="cuda").reshape(-1, 1)
        for owner in metadata.kv_cache_manager.layout.kv_source_layer_ids:
            selected = metadata.select_global_source(owner, hidden)
            assert selected.flatten().tolist() == list(range(128, 134))


def reference(model, metadata, ids, prefix=256, chunks=None, window=128):
    """Explicit frozen Global encoder and final-window Decoder, without replay plans."""
    start, end = max(0, prefix - window), ids.numel()
    body = model.model
    chunks = chunks or (end - prefix,)
    offset = prefix
    for index, length in enumerate(chunks):
        begin = start if index == 0 else offset
        stop = offset + length
        local_ids = ids[begin:stop]
        positions = torch.arange(begin, stop, device="cuda", dtype=torch.int32).view(1, -1)
        metadata.seq_lens = torch.tensor([stop - begin], dtype=torch.int32)
        metadata.prompt_lens = [stop - begin]
        metadata.kv_cache_params.num_cached_tokens_per_seq = [begin]
        if index == 0:
            metadata.set_swa_bounded_replay([prefix])
        metadata.prepare()
        _apply_swa_floor(
            metadata, torch.full((stop - begin,), start, device="cuda", dtype=torch.int64)
        )
        embedded = body.embed_tokens(local_ids).unsqueeze(1).repeat(1, body.hc_mult, 1)
        state = body._init_hc_state(embedded)
        for layer in body.layers[:5]:
            state = layer(
                position_ids=positions, hc_state=state, attn_metadata=metadata, input_ids=local_ids
            )
        with global_precompute_scope(metadata):
            owner = body.layers[5]
            owner.self_attn.prepare_global_cache(owner._decoder_global_input(state), metadata)
            decoder_begin = min(stop, max(begin, end - window))
            kept = stop - decoder_begin
            if not kept:
                offset = stop
                continue
            metadata.seq_lens = torch.tensor([kept], dtype=torch.int32)
            metadata.prompt_lens = [kept]
            metadata.kv_cache_params.num_cached_tokens_per_seq = [decoder_begin]
            metadata.prepare()
            _apply_swa_floor(
                metadata,
                torch.full(
                    (kept,),
                    max(0, end - window),
                    device="cuda",
                    dtype=torch.int64,
                ),
            )
            state = HCState.resolved(state.residual[-kept:], pre_mix=state.pre_mix[-kept:])
            positions = positions[:, -kept:]
            state = owner(positions, state, metadata, input_ids=ids[decoder_begin:stop])
            for layer in body.layers[6:]:
                state = layer(
                    position_ids=positions,
                    hc_state=state,
                    attn_metadata=metadata,
                    input_ids=ids[decoder_begin:stop],
                )
            if stop == end:
                hidden = body._finalize_hc_state(state)[-1:]
                return model.logits_processor(
                    hidden, model.lm_head, None, return_context_logits=True
                )
        offset = stop
    raise AssertionError("Reference chunks must cover the uncached suffix")


@pytest.mark.parametrize("chunks", [(1,), (257,), (1, 127, 129)])
@pytest.mark.parametrize(
    "block,prefix,window",
    [(128, 128, 128), (128, 256, 128), (128, 255, 128), (256, 383, 128), (256, 511, 256)],
)
def test_reused_global_is_immutable_and_chunked_recovery_matches_reference(
    model_and_config, monkeypatch, chunks, block, prefix, window
):
    model, config, sparse, dtype = model_and_config
    enable_ced(model)
    model.model.decoder_replay_window = window
    model.model.encoder_replay_enabled = True
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "1")
    spec = None
    if window > 128:
        from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

        # The embedded draft capture can exceed the target attention window.
        # Keep text/SWA geometry at 128 and enlarge the wrapper capture window.
        monkeypatch.setattr(config.pretrained_config, "sliding_window", window)
        spec = DSparkDecodingConfig(max_draft_len=5)
        spec.__dict__["draft_is_embedded_in_target"] = True
    total = prefix + sum(chunks)
    ids = torch.arange(total, device="cuda", dtype=torch.int32)
    with (
        torch.inference_mode(),
        cache_case(
            config,
            sparse,
            dtype,
            [prefix + 1],
            [0],
            1,
            reuse=True,
            spec_config=spec,
            tokens_per_block=block,
        ) as case,
    ):
        metadata, _, _, _ = case
        manager = metadata.kv_cache_manager
        assert manager.layout.window_size == 128
        assert manager._decoder_replay_window == window
        manager.enable_swa_scratch_reuse = False
        with ExitStack() as stack:
            stack.enter_context(canonical_routing())
            warm = LlmRequest(
                request_id=0,
                max_new_tokens=2,
                input_tokens=list(range(prefix + 1)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            warm.py_seq_slot = 0
            run(model, metadata, [warm], ids[: prefix + 1], [0], [prefix + 1], 1)
            warm_cache = manager.kv_cache_map[0]
            warm_cache.commit(list(range(prefix)))
            warm_cache.stop_committing()
            # Cover short-suffix replay despite a complete snapshot, partial
            # Encoder eviction, and complete eviction in the same numeric suite.
            if prefix % block == 0:
                if prefix == 256 and chunks == (1,):
                    pass  # A hit still needs replay to supply the Decoder window.
                else:
                    groups = (
                        list(manager._encoder_optional_groups[:1])
                        if prefix == 256 and chunks == (257,)
                        else None
                    )
                    _introspection.drop_optional_pages(warm_cache, prefix, groups)
            assert warm_cache.num_committed_tokens == prefix
            frozen = global_bytes(metadata, 0, prefix)
            manager.free_resources(warm)

            def admit(request_id):
                req = LlmRequest(
                    request_id=request_id,
                    max_new_tokens=2,
                    input_tokens=list(range(total)),
                    sampling_config=SamplingConfig(),
                    is_streaming=False,
                )
                assert manager.prepare_context(req)
                assert req.context_current_position == prefix
                assert encoder_replay_tokens(req) == window
                req.py_seq_slot = 0
                req.context_chunk_size = total - prefix
                assert manager.resize_context(req, total - prefix)
                scheduled = ScheduledRequests()
                scheduled.context_requests_last_chunk = [req]
                manager.prepare_resources(scheduled)
                metadata.request_ids = [request_id]
                metadata.num_contexts = 1
                for layer in manager.pp_layers:
                    pages = manager.get_cache_indices(request_id, layer, CSA2CacheRole.SWA)
                    assert pages[(prefix - window) // block] >= 0
                return req

            ref_req = admit(1)
            with model_extra_attrs(
                {**config.extra_attrs, "attention_metadata": weakref.ref(metadata)}
            ):
                # CSA2 batches FMHA by context phase. Different chunk shapes
                # can change reductions and subsequent FP4 rounding, so use
                # the same encoder partitions in this independent reference.
                expected = reference(model, metadata, ids, prefix, chunks, window).clone()
            for key, value in global_bytes(metadata, 0, prefix).items():
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
            expected_global = global_bytes(metadata, prefix, total)
            manager.kv_cache_map[1].stop_committing()
            manager.free_resources(ref_req)

            req = admit(2)
            try:
                offset = prefix
                for chunk in chunks:
                    req.context_current_position = offset
                    req.context_chunk_size = chunk
                    begin = offset - encoder_replay_tokens(req)
                    inputs = (
                        model,
                        metadata,
                        [req],
                        ids[begin : offset + chunk],
                        [begin],
                        [offset + chunk - begin],
                        1,
                    )
                    with patch.object(metadata, "prepare", wraps=metadata.prepare) as prepare:
                        output = run(*inputs)
                        # Empty Decoder chunks need only executor preparation.
                        # Encoder forward consumes the already prepared views.
                        assert prepare.call_count == 1 + int(offset + chunk > total - window)
                    offset += chunk
                    assert req.py_ced_replay.consumed
                    assert encoder_replay_tokens(req) == 0
                    for key, value in global_bytes(metadata, 0, prefix).items():
                        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                for key, value in global_bytes(metadata, prefix, total).items():
                    torch.testing.assert_close(value, expected_global[key], rtol=0, atol=0)
            finally:
                manager.kv_cache_map[2].stop_committing()
                manager.free_resources(req)
