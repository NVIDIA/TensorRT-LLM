# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm  # noqa: F401


def _pad_nvfp4_activation_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "pad_nvfp4_activation")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _pad_nvfp4_activation_available(),
    reason="Requires CUDA and the trtllm pad_nvfp4_activation op",
)


@pytest.mark.parametrize("num_tokens", [0, 1, 7, 228])
def test_pad_nvfp4_activation_copy_and_zero_tail(num_tokens: int) -> None:
    logical_hidden_size = 2880
    padded_hidden_size = 3072
    fp4_input = torch.randint(
        0,
        256,
        (num_tokens, logical_hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    sf_input = torch.randint(
        0,
        256,
        (num_tokens, logical_hidden_size // 16),
        dtype=torch.uint8,
        device="cuda",
    )

    fp4_output, sf_output = torch.ops.trtllm.pad_nvfp4_activation(
        fp4_input, sf_input, padded_hidden_size
    )

    assert fp4_output.shape == (num_tokens, padded_hidden_size // 2)
    assert sf_output.shape == (num_tokens, padded_hidden_size // 16)
    assert fp4_output.dtype == fp4_input.dtype
    assert sf_output.dtype == sf_input.dtype
    torch.testing.assert_close(fp4_output[:, : fp4_input.shape[1]], fp4_input)
    torch.testing.assert_close(sf_output[:, : sf_input.shape[1]], sf_input)
    assert torch.count_nonzero(fp4_output[:, fp4_input.shape[1] :]) == 0
    assert torch.count_nonzero(sf_output[:, sf_input.shape[1] :]) == 0


def test_pad_nvfp4_activation_cuda_graph() -> None:
    logical_hidden_size = 2880
    padded_hidden_size = 3072
    fp4_input = torch.randint(
        0,
        256,
        (32, logical_hidden_size // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    sf_input = torch.randint(
        0,
        256,
        (32, logical_hidden_size // 16),
        dtype=torch.uint8,
        device="cuda",
    )

    expected = torch.ops.trtllm.pad_nvfp4_activation(fp4_input, sf_input, padded_hidden_size)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = torch.ops.trtllm.pad_nvfp4_activation(fp4_input, sf_input, padded_hidden_size)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_pad_nvfp4_activation_rejects_mismatched_layout() -> None:
    fp4_input = torch.empty((2, 1440), dtype=torch.uint8, device="cuda")
    sf_input = torch.empty((2, 179), dtype=torch.uint8, device="cuda")

    with pytest.raises(RuntimeError, match="same logical hidden size"):
        torch.ops.trtllm.pad_nvfp4_activation(fp4_input, sf_input, 3072)


def test_pad_nvfp4_activation_fake_shapes() -> None:
    with FakeTensorMode():
        fp4_input = torch.empty((7, 1440), dtype=torch.uint8, device="cuda")
        sf_input = torch.empty((7, 180), dtype=torch.uint8, device="cuda")
        fp4_output, sf_output = torch.ops.trtllm.pad_nvfp4_activation(fp4_input, sf_input, 3072)

    assert fp4_output.shape == (7, 1536)
    assert sf_output.shape == (7, 192)
    assert fp4_output.dtype == fp4_input.dtype
    assert sf_output.dtype == sf_input.dtype
