# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CUDA capture/replay checks without model weights."""

import gc
from dataclasses import replace
from weakref import ref

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

# The fixture allocates bfloat16 capture buffers on cuda at setup time, so gate
# the whole module rather than let a CPU-only (or non-bf16) run fail there.
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="allocates bfloat16 capture/replay buffers on cuda",
)


@pytest.fixture(params=["dflash", "dspark"])
def metadata(request: pytest.FixtureRequest):
    cls, mode = {
        "dflash": (DFlashSpecMetadata, SpeculativeDecodingMode.DFLASH),
        "dspark": (DSparkSpecMetadata, SpeculativeDecodingMode.DSPARK),
    }[request.param]
    return cls(
        max_num_requests=8,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=mode,
        layers_to_capture=[1, 3],
        hidden_size=64,
        max_num_tokens=32,
        dtype=torch.bfloat16,
    )


def test_shared_buffer_capture_replay(metadata) -> None:
    # Use a local parent so the fixture cannot keep it alive during replay.
    metadata = replace(metadata, captured_hidden_states=None)
    parent_ref = ref(metadata)
    pointer = metadata.captured_hidden_states.data_ptr()
    device = torch.device("cuda", torch.cuda.current_device())
    assert metadata.captured_hidden_states.device == device
    assert metadata.captured_hidden_states.dtype == torch.bfloat16
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cases = []
    with torch.cuda.stream(stream):
        for bucket in (8, 4, 1):
            graph_metadata = metadata.create_cuda_graph_metadata(bucket)
            assert graph_metadata.captured_hidden_states.data_ptr() == pointer
            rows = bucket * 4
            inputs = torch.zeros((rows, 64), dtype=metadata.dtype, device=device)
            output = torch.empty((rows, 128), dtype=metadata.dtype, device=device)

            def forward() -> None:
                graph_metadata.maybe_capture_hidden_states(1, inputs)
                graph_metadata.maybe_capture_hidden_states(3, inputs, inputs)
                output.copy_(graph_metadata.get_hidden_states(rows))

            forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            cases.append((graph, graph_metadata, inputs, output))

        # Alternate buckets and eager writes; every consumer finishes before
        # the next writer on this stream. Retain graph metadata for its lifetime.
        snapshots = []
        for index in (0, 2, 1, 0):
            graph, graph_metadata, inputs, output = cases[index]
            inputs.fill_(index + 1)
            graph.replay()
            snapshots.append((output.clone(), index + 1))
            metadata.captured_hidden_states.fill_(-1)
        del metadata
        gc.collect()
        assert parent_ref() is None
        for graph, graph_metadata, inputs, output in cases:
            inputs.fill_(7)
            graph.replay()
            snapshots.append((output.clone(), 7))
    stream.synchronize()
    for actual, value in snapshots:
        torch.testing.assert_close(actual[:, :64], torch.full_like(actual[:, :64], value))
        torch.testing.assert_close(actual[:, 64:], torch.full_like(actual[:, 64:], 2 * value))


@pytest.mark.parametrize("replay_rows", [False, True])
def test_side_stream_capture_matches_capture(replay_rows: bool) -> None:
    """DSpark captures on its side stream equal the plain capture, eager and in graph replays."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    def make():
        return DSparkSpecMetadata(
            max_num_requests=8,
            max_draft_len=5,
            max_total_draft_tokens=5,
            spec_dec_mode=SpeculativeDecodingMode.DSPARK,
            layers_to_capture=[37, 38, 39],
            hidden_size=256,
            max_num_tokens=64,
            dtype=torch.bfloat16,
        )

    plain, side = make(), make()
    gen = torch.Generator(device="cuda").manual_seed(3)
    residuals = [torch.empty(12, 4 * 256, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    rows = torch.randperm(12, generator=gen, device="cuda")[:12] if replay_rows else None
    sink = torch.zeros(4096, 4096, device="cuda")

    def forward(metadata, on_side):
        for layer, residual in zip((37, 38, 39), residuals):
            if on_side:
                done = metadata.capture_on_side_stream(layer, residual, rows)
                assert done is not None
                # Work on the main stream while the capture runs beside it.
                sink.mul_(0.5)
                torch.cuda.current_stream().wait_event(done)
            else:
                metadata.maybe_capture_hidden_states(layer, residual, row_indices=rows)
        return metadata.get_hidden_states(12).clone()

    with torch.inference_mode(), with_multi_stream(True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for residual in residuals:
                residual.copy_(torch.randn(residual.shape, generator=gen, device="cuda"))
            assert torch.equal(forward(side, True), forward(plain, False))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = forward(side, True)
            for _ in range(3):
                for residual in residuals:
                    residual.copy_(torch.randn(residual.shape, generator=gen, device="cuda"))
                graph.replay()
                assert torch.equal(captured, forward(plain, False))
        stream.synchronize()


def test_graph_metadata_follows_current_device(metadata) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    parent = metadata.captured_hidden_states
    other = (torch.cuda.current_device() + 1) % torch.cuda.device_count()
    with torch.cuda.device(other):
        graph_metadata = metadata.create_cuda_graph_metadata(1)
    assert graph_metadata.captured_hidden_states.device == torch.device("cuda", other)
    assert graph_metadata.captured_hidden_states is not parent
    assert metadata.captured_hidden_states is parent
