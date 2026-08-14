# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the lifetime of the capture-only sampling override.

The advanced-sampling CUDA graph capture pass sets
``_force_non_greedy_for_capture=True`` on the live ``SpecMetadata`` so that
parameter-less warmup requests scan as non-greedy and the advanced-sampling
branch is the one recorded into the graph.

``create_cuda_graph_metadata`` shallow-copies the live metadata, so every graph
captured during that pass caches a copy that inherited the flag, and those
copies are reseated as the live spec_metadata on every later replay. Clearing
the flag on the base object alone therefore leaves it set forever on the copies,
and ``_scan_one_model_sampling`` then rewrites EVERY serving request's sampling
params to the synthetic capture values.

These tests use a real (base) ``SpecMetadata`` -- its ``__post_init__`` is a
no-op and none of the fields exercised here are tensors -- plus an unbound call
of ``CUDAGraphRunner.clear_capture_only_spec_state`` on a stand-in holding only
``graph_metadata``, mirroring test_group_all_greedy_sync.py. No GPU, no runner
construction, and no model forward is needed.
"""

import types

import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata

# The synthetic params the capture override substitutes for the request's own,
# and the sentinel that means "top-k disabled" (see _scan_one_model_sampling).
CAPTURE_TEMPERATURE, CAPTURE_TOP_K, CAPTURE_TOP_P = 0.7, 50, 0.9
DISABLE_TOPK_VAL = torch.iinfo(torch.int32).max


def _base_meta():
    """A live (non-graph) SpecMetadata, as the model engine holds it."""
    return SpecMetadata(max_num_requests=8, max_draft_len=1, max_total_draft_tokens=1)


def _graph_copy(meta, batch_size=8):
    """The shallow copy maybe_get_cuda_graph caches for one captured graph."""
    graph_meta = meta.create_cuda_graph_metadata(batch_size)
    assert graph_meta is not meta
    return graph_meta


def _clear(graph_metadata):
    """CUDAGraphRunner.clear_capture_only_spec_state, called unbound."""
    return CUDAGraphRunner.clear_capture_only_spec_state(
        types.SimpleNamespace(graph_metadata=graph_metadata)
    )


def _request(temperature=None, top_k=None, top_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _scan(meta, requests):
    normalized, _ = SpecMetadata._scan_one_model_sampling(meta, requests)
    # Drop the trailing num_tokens; only the sampling params matter here.
    return [entry[:3] for entry in normalized]


def test_graph_copy_inherits_flag_and_base_teardown_does_not_reach_it():
    # The mechanism the bug rests on: copy.copy carries the flag over, and the
    # copies are independent objects, so clearing the base misses them.
    meta = _base_meta()
    meta._force_non_greedy_for_capture = True
    copies = [_graph_copy(meta, bs) for bs in (1, 2, 4)]
    assert all(copy._force_non_greedy_for_capture for copy in copies)

    meta._force_non_greedy_for_capture = False
    assert all(copy._force_non_greedy_for_capture for copy in copies)


def test_clear_capture_only_spec_state_clears_every_cached_copy():
    meta = _base_meta()
    meta._force_non_greedy_for_capture = True
    advanced = [_graph_copy(meta, bs) for bs in (1, 2, 4)]
    # Graphs captured by the greedy pass never had the flag, and non-spec
    # graphs cache no spec_metadata at all; both must be left alone.
    meta._force_non_greedy_for_capture = False
    greedy = _graph_copy(meta, 8)

    graph_metadata = {("greedy", 8): {"spec_metadata": greedy}}
    graph_metadata[("no_spec", 1)] = {"spec_metadata": None}
    for i, copy in enumerate(advanced):
        graph_metadata[("advanced", i)] = {"spec_metadata": copy}

    assert _clear(graph_metadata) == len(advanced)
    assert not any(copy._force_non_greedy_for_capture for copy in advanced)
    assert greedy._force_non_greedy_for_capture is False
    # Idempotent: a second teardown finds nothing left to clear.
    assert _clear(graph_metadata) == 0


def test_serving_scan_honors_client_params_after_capture_teardown():
    # End-to-end property of the fix, and the case that fails without it: with
    # only the base-object teardown this scan returns (0.7, 50, 0.9).
    meta = _base_meta()
    meta._force_non_greedy_for_capture = True
    graph_meta = _graph_copy(meta)

    meta._force_non_greedy_for_capture = False  # base-object teardown
    _clear({("advanced", 8): {"spec_metadata": graph_meta}})  # the fix

    # Replay reseats the cached copy as the live spec_metadata, so the serving
    # scan runs on it, not on the base object.
    assert _scan(graph_meta, [_request(temperature=1.0, top_p=1.0)]) == [
        (1.0, DISABLE_TOPK_VAL, 1.0)
    ]
    assert graph_meta.is_all_greedy_sample is False


def test_override_stays_live_while_the_flag_is_set():
    # Anti-regression for the rejected "clear at copy time" fix: the flag is
    # load-bearing *during* capture. Cleared any earlier, the pass-2 populate
    # would scan these parameter-less warmup requests as greedy and bake the
    # argmax fast path -- with no top-k/top-p kernels -- into the graph keyed
    # as the advanced-sampling variant.
    meta = _base_meta()
    meta._force_non_greedy_for_capture = True
    graph_meta = _graph_copy(meta)

    warmup_requests = [_request(slot=None), _request(slot=None)]
    assert _scan(graph_meta, warmup_requests) == [
        (CAPTURE_TEMPERATURE, CAPTURE_TOP_K, CAPTURE_TOP_P)
    ] * len(warmup_requests)
    assert graph_meta.is_all_greedy_sample is False
