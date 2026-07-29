# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


def _mixed_batch() -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [
        SimpleNamespace(
            context_chunk_size=2,
            encoder_output_len=260,
            py_skip_cross_kv_projection=False,
        ),
        SimpleNamespace(
            context_chunk_size=2,
            encoder_output_len=272,
            py_skip_cross_kv_projection=False,
        ),
    ]
    batch.generation_requests = [
        SimpleNamespace(py_draft_tokens=[]),
        SimpleNamespace(py_draft_tokens=[]),
    ]
    return batch


def _runner() -> CUDAGraphRunner:
    runner = object.__new__(CUDAGraphRunner)
    runner.config = SimpleNamespace(
        enable_attention_dp=False,
        is_draft_model=False,
        use_mrope=False,
    )
    runner.enabled = True
    runner.padding_enabled = True
    runner.sparse_config = None
    runner.max_beam_width = 1
    runner.enable_encoder_decoder_mixed_cuda_graph = True
    runner.graphs = {}
    runner.graph_outputs = {}
    runner.graph_metadata = {}
    runner.padding_dummy_requests = {}
    runner.memory_pool = None
    return runner


def test_mixed_encoder_decoder_graph_key_captures_dynamic_extents():
    runner = _runner()

    key = runner.get_graph_key(_mixed_batch())

    assert key == (4, 0, False, False, True, (2, 2), (532,))
    assert runner._get_num_tokens_for_key(key) == 6


def test_mixed_encoder_decoder_graph_key_distinguishes_cached_cross_kv():
    runner = _runner()
    batch = _mixed_batch()
    batch.context_requests_last_chunk[1].py_skip_cross_kv_projection = True

    key = runner.get_graph_key(batch)

    assert key[6] == (260,)


def test_mixed_encoder_decoder_graph_eligibility_requires_both_phases():
    runner = _runner()
    batch = _mixed_batch()

    assert runner._is_mixed_encoder_decoder_batch(batch)

    batch.generation_requests = []
    assert not runner._is_mixed_encoder_decoder_batch(batch)


def test_mixed_encoder_decoder_graph_never_captures_at_runtime():
    runner = _runner()
    runner._capture_allowed = False
    key = runner.get_graph_key(_mixed_batch())

    assert not runner.needs_capture(key)


def test_mixed_encoder_decoder_graph_key_pads_encoder_extent():
    runner = _runner()
    runner._capture_allowed = False
    batch = _mixed_batch()
    padded_key = (4, 0, False, False, True, (2, 2), (576,))
    graph_attn_metadata = object()
    runner.graph_metadata[padded_key] = {
        "attn_metadata": graph_attn_metadata,
        "spec_metadata": None,
    }
    runner.graph_outputs[padded_key] = object()

    attn_metadata, spec_metadata, key = runner.maybe_get_cuda_graph(
        batch,
        enable_spec_decode=False,
        attn_metadata=object(),
        allow_mixed_encoder_decoder=True,
    )

    assert key == padded_key
    assert attn_metadata is graph_attn_metadata
    assert spec_metadata is None


def test_mixed_encoder_decoder_graph_key_uses_smallest_compatible_extent():
    runner = _runner()
    runner._capture_allowed = False
    key = runner.get_graph_key(_mixed_batch())
    larger_key = (*key[:6], (672,))
    smallest_key = (*key[:6], (576,))
    incompatible_key = (*key[:5], (2,), (544,))
    runner.graph_outputs = {
        larger_key: object(),
        smallest_key: object(),
        incompatible_key: object(),
    }

    assert runner._get_compatible_mixed_encoder_decoder_key(key) == smallest_key


def test_mixed_encoder_decoder_replay_zero_pads_encoder_hidden_states():
    runner = _runner()
    key = (4, 0, False, False, True, (2, 2), (576,))
    attn_metadata = object()
    runner.graph_metadata[key] = {
        "attn_metadata": attn_metadata,
        "spec_metadata": None,
    }
    runner.graph_outputs[key] = object()
    runner.graphs[key] = SimpleNamespace(replay=lambda: None, reset=lambda: None)
    runner.shared_static_tensors = {
        "input_ids": torch.zeros(6, dtype=torch.int32),
        "position_ids": torch.zeros((1, 6), dtype=torch.int32),
        "encoder_hidden_states": torch.ones((576, 2)),
    }
    encoder_hidden_states = torch.arange(532 * 2, dtype=torch.float32).reshape(532, 2)

    runner.replay(
        key,
        {
            "attn_metadata": attn_metadata,
            "input_ids": torch.ones(6, dtype=torch.int32),
            "position_ids": torch.ones((1, 6), dtype=torch.int32),
            "encoder_hidden_states": encoder_hidden_states,
        },
    )

    staged_encoder_hidden_states = runner.shared_static_tensors["encoder_hidden_states"]
    assert torch.equal(staged_encoder_hidden_states[:532], encoder_hidden_states)
    assert torch.count_nonzero(staged_encoder_hidden_states[532:]) == 0
