# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import weakref
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler.embedding_bias import (
    add_cached_embedding_bias_,
    try_apply_cached_embedding_bias,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import apply_embedding_bias
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _request(bias: torch.Tensor | None, request_id: int = 1) -> LlmRequest:
    return cast(
        LlmRequest,
        SimpleNamespace(
            py_embedding_bias=bias,
            _py_embedding_bias_cache=None,
            py_request_id=request_id,
        ),
    )


def _reference(logits: torch.Tensor, requests: list[LlmRequest]) -> torch.Tensor:
    expected = logits.clone()
    for row, request in enumerate(requests):
        bias = request.py_embedding_bias
        if bias is not None:
            expected[row] = (expected[row].float() + bias.to(logits.device)).to(logits.dtype)
    return expected


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("vocab", [1, 127, 128, 1023, 1024, 1025, 4097])
@pytest.mark.parametrize("pattern", ["same", "different", "mixed"])
def test_bias_values_and_tile_boundaries(batch: int, vocab: int, pattern: str) -> None:
    generator = torch.Generator().manual_seed(17436)
    common = torch.randn(vocab, generator=generator)
    requests = []
    for row in range(batch):
        bias = common.clone() if pattern == "same" else torch.randn(vocab, generator=generator)
        bias[0] = -100
        if vocab > 1:
            bias[-1] = 100
        if pattern == "mixed" and row == batch - 1:
            bias = None
        requests.append(_request(bias, row))
    logits = torch.randn(batch, vocab, device="cuda")
    expected = _reference(logits, requests)
    apply_embedding_bias(logits, requests, torch.ones(batch, dtype=torch.int32))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 16, 110])
def test_production_vocabulary(batch: int) -> None:
    vocab = 248320
    bias = torch.zeros(vocab)
    bias[1000:2206] = -100
    requests = [_request(bias.clone(), row) for row in range(batch)]
    logits = torch.randn(batch, vocab, device="cuda")
    expected = _reference(logits, requests)
    assert try_apply_cached_embedding_bias(logits, requests, torch.ones(batch, dtype=torch.int32))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,padded_rows,vocab", [(3, 4, 257), (16, 32, 1025), (110, 128, 248320)]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_cuda_graph_padded_logits(batch: int, padded_rows: int, vocab: int, mixed: bool) -> None:
    requests = [_request(torch.randn(vocab), row) for row in range(batch)]
    if mixed:
        requests[-1].py_embedding_bias = None
    logits = torch.randn(padded_rows, vocab, device="cuda")
    expected = _reference(logits, requests)
    steps = torch.ones(batch, dtype=torch.int32)
    beams = torch.ones_like(steps)
    apply_embedding_bias(logits, requests, steps, request_beams=beams)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert requests[0]._py_embedding_bias_cache is not None


@pytest.mark.parametrize("beam_metadata", [None, "multiple", "cuda"])
def test_padded_fast_path_requires_single_beam_metadata(beam_metadata: str | None) -> None:
    requests = [_request(torch.randn(257)), _request(torch.randn(257))]
    logits = torch.randn(4, 257, device="cuda")
    expected = logits.clone()
    steps = torch.ones(2, dtype=torch.int32)
    beams = None
    if beam_metadata == "multiple":
        beams = torch.full((2,), 2, dtype=torch.int32)
    elif beam_metadata == "cuda":
        beams = torch.ones(2, dtype=torch.int32, device="cuda")
    assert not try_apply_cached_embedding_bias(logits, requests, steps, beams)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert all(request._py_embedding_bias_cache is None for request in requests)


def test_reuse_reorder_replacement_and_unbiased_last_row() -> None:
    requests = [_request(torch.randn(257), row) for row in range(3)] + [_request(None, 3)]
    steps = torch.ones(4, dtype=torch.int32)
    logits = torch.zeros(4, 257, device="cuda")
    apply_embedding_bias(logits, requests, steps)
    caches = [request._py_embedding_bias_cache for request in requests]
    requests = [requests[2], requests[0], requests[1], requests[3]]
    expected = _reference(logits, requests)
    apply_embedding_bias(logits, requests, steps)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert requests[0]._py_embedding_bias_cache is caches[2]
    assert requests[1]._py_embedding_bias_cache is caches[0]
    requests[0].py_embedding_bias = torch.full((257,), 3.0)
    expected = _reference(logits, requests)
    apply_embedding_bias(logits, requests, steps)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert requests[0]._py_embedding_bias_cache is not caches[2]
    assert requests[-1]._py_embedding_bias_cache is None


def test_no_bias_does_not_allocate_cache() -> None:
    requests = [_request(None), _request(None)]
    logits = torch.randn(2, 257, device="cuda")
    expected = logits.clone()
    apply_embedding_bias(logits, requests, torch.ones(2, dtype=torch.int32))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert all(request._py_embedding_bias_cache is None for request in requests)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_other_logits_dtypes_keep_reference_path(dtype: torch.dtype) -> None:
    requests = [_request(torch.randn(257)), _request(torch.randn(257))]
    logits = torch.randn(2, 257, dtype=dtype, device="cuda")
    expected = _reference(logits, requests)
    steps = torch.ones(2, dtype=torch.int32)
    assert not try_apply_cached_embedding_bias(logits, requests, steps)
    apply_embedding_bias(logits, requests, steps)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert all(request._py_embedding_bias_cache is None for request in requests)


@pytest.mark.parametrize("layout", ["logits_stride", "bias_stride", "cuda_steps", "multi_step"])
def test_unsupported_path_does_not_mutate(layout: str) -> None:
    logits = torch.randn(2, 257, device="cuda")
    source = torch.randn(257)
    steps = torch.ones(2, dtype=torch.int32)
    if layout == "logits_stride":
        logits = torch.randn(2, 514, device="cuda")[:, ::2]
    elif layout == "bias_stride":
        source = torch.randn(514)[::2]
    elif layout == "cuda_steps":
        steps = steps.cuda()
    else:
        logits = torch.randn(3, 257, device="cuda")
        steps = torch.tensor([2, 1], dtype=torch.int32)
    requests = [_request(source), _request(source)]
    expected = logits.clone()
    assert not try_apply_cached_embedding_bias(logits, requests, steps)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert all(request._py_embedding_bias_cache is None for request in requests)


def test_sampling_params_snapshot() -> None:
    source = torch.randn(257)
    snapshot = source.clone()
    params = SamplingParams(max_tokens=8, embedding_bias=source)
    request = _request(params.embedding_bias)
    source.fill_(100)
    logits = torch.zeros(1, 257, device="cuda")
    apply_embedding_bias(logits, [request], torch.ones(1, dtype=torch.int32))
    torch.testing.assert_close(logits[0], snapshot.cuda(), rtol=0, atol=0)


def test_cross_stream_use_and_early_release() -> None:
    upload = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    source = torch.randn(65537)
    request = _request(source)
    steps = torch.ones(1, dtype=torch.int32)
    first = torch.zeros(1, source.numel(), device="cuda")
    second = torch.zeros_like(first)
    torch.cuda.synchronize()
    with torch.cuda.stream(upload):
        torch.cuda._sleep(2_000_000)
        apply_embedding_bias(first, [request], steps)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(2_000_000)
        apply_embedding_bias(second, [request], steps)
    request._py_embedding_bias_cache = None
    gc.collect()
    # Instrumentation may retain Python tensor objects; correctness must not
    # depend on immediate wrapper collection before the GPU finishes.
    # Encourage allocator reuse while the consumer kernel may still be pending.
    scratch = [torch.full_like(first, 99) for _ in range(8)]
    torch.cuda.synchronize()
    torch.testing.assert_close(first[0], source.cuda(), rtol=0, atol=0)
    torch.testing.assert_close(second[0], source.cuda(), rtol=0, atol=0)
    assert len(scratch) == 8


def test_cache_memory_bounded_after_request_churn() -> None:
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    source = torch.randn(65537)
    logits = [torch.zeros(1, source.numel(), device="cuda") for _ in streams]
    steps = torch.ones(1, dtype=torch.int32)
    torch.cuda.synchronize()
    initial_bytes = torch.cuda.memory_allocated()
    for index in range(32):
        request = _request(source.clone(), request_id=index)
        with torch.cuda.stream(streams[index % 2]):
            apply_embedding_bias(logits[index % 2], [request], steps)
        request._py_embedding_bias_cache = None
    torch.cuda.synchronize()
    gc.collect()
    # A leak per completed request would retain over 8 MiB in this sweep.
    assert torch.cuda.memory_allocated() - initial_bytes < 2 * 1024 * 1024
    expected = torch.zeros_like(logits[0])
    source_cuda = source.cuda()
    for _ in range(16):
        expected += source_cuda
    for result in logits:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_device_kernel_cuda_graph_replay() -> None:
    request = _request(torch.randn(257))
    logits = torch.zeros(1, 257, device="cuda")
    apply_embedding_bias(logits, [request], torch.ones(1, dtype=torch.int32))
    bias = request._py_embedding_bias_cache.tensor
    pointers = torch.tensor([bias.data_ptr()], dtype=torch.int64, device="cuda")
    logits.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        add_cached_embedding_bias_(logits, pointers, [bias])
    graph.replay()
    torch.testing.assert_close(logits[0], bias, rtol=0, atol=0)
    graph.replay()
    torch.testing.assert_close(logits[0], bias + bias, rtol=0, atol=0)


def test_resource_release_clears_cache_before_other_resources() -> None:
    request = _request(torch.randn(257), request_id=17436)
    logits = torch.zeros(1, 257, device="cuda")
    apply_embedding_bias(logits, [request], torch.ones(1, dtype=torch.int32))
    reference = weakref.ref(request._py_embedding_bias_cache.tensor)
    released = []

    def free_resources(req: LlmRequest) -> None:
        assert req._py_embedding_bias_cache is None
        released.append(req.py_request_id)

    executor = SimpleNamespace(
        resource_manager=SimpleNamespace(free_resources=free_resources),
        _prefetched_request_ids={17436},
        disagg=SimpleNamespace(forget_request=lambda request_id: None),
    )
    PyExecutor._free_request_resources(executor, request)
    torch.cuda.synchronize()
    assert reference() is None
    assert released == [17436]
    assert not executor._prefetched_request_ids
