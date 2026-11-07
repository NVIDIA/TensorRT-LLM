# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation-contract tests; CUDA execution is tested separately."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.cpu_only


@pytest.fixture(params=["dflash", "dspark"])
def metadata(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    cls, mode = {
        "dflash": (DFlashSpecMetadata, SpeculativeDecodingMode.DFLASH),
        "dspark": (DSparkSpecMetadata, SpeculativeDecodingMode.DSPARK),
    }[request.param]
    return cls(
        max_num_requests=8,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=mode,
        layers_to_capture=[3, 1],
        hidden_size=16,
        max_num_tokens=32,
        dtype=torch.bfloat16,
    )


def test_graph_buckets_share_full_capture_buffer(metadata) -> None:
    parent = metadata.captured_hidden_states
    assert parent.shape == (32, 32)
    assert parent.dtype == torch.bfloat16
    with patch.object(torch, "empty", wraps=torch.empty) as allocate:
        copies = [metadata.create_cuda_graph_metadata(bucket) for bucket in (8, 4, 2, 1)]
    assert allocate.call_count == len(copies)  # Only per-bucket batch indices.
    for bucket, graph in zip((8, 4, 2, 1), copies):
        assert graph.is_cuda_graph
        assert graph.max_num_requests == bucket
        assert graph.captured_hidden_states is parent
        assert graph.batch_indices_cuda is not metadata.batch_indices_cuda
        assert graph.get_hidden_states(bucket * 4).shape == (bucket * 4, 32)
        assert graph.layers_to_capture == [1, 3]
    independent = type(metadata)(
        max_num_requests=8,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=metadata.spec_dec_mode,
        layers_to_capture=[1, 3],
        hidden_size=16,
        max_num_tokens=32,
    )
    assert independent.captured_hidden_states is not parent


@pytest.mark.parametrize("mismatch", ["rows", "width", "dtype", "device", "missing"])
def test_incompatible_capture_buffer_is_reallocated(metadata, mismatch: str) -> None:
    original = metadata.captured_hidden_states
    if mismatch == "rows":
        metadata.max_num_tokens = 64
    elif mismatch == "width":
        metadata.hidden_size = 32
    elif mismatch == "dtype":
        metadata.dtype = torch.float32
    elif mismatch == "device":
        metadata.captured_hidden_states = original.to("meta")
    else:
        metadata.captured_hidden_states = None
    metadata.__post_init__()
    result = metadata.captured_hidden_states
    assert result is not original
    assert result.shape == (metadata.max_num_tokens, metadata.hidden_size * 2)
    assert result.dtype == metadata.dtype
    assert result.device == metadata.batch_indices_cuda.device


@pytest.mark.parametrize("layers", [None, []])
def test_no_capture_layers(metadata, layers) -> None:
    metadata.layers_to_capture = layers
    metadata.captured_hidden_states = None
    metadata.__post_init__()
    assert metadata.create_cuda_graph_metadata(1).captured_hidden_states is None
    assert not metadata.is_layer_capture(1)
