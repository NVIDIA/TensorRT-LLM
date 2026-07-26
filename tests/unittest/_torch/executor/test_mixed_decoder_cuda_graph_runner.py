# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

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
    runner.config = SimpleNamespace(is_draft_model=False)
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
