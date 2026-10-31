# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eligibility, layout and stream integration of tiled CuteDSL input quantization."""

from functools import partial
from unittest.mock import Mock

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.moe.fused_moe.cutedsl import kernel_inputs
from tensorrt_llm._torch.moe.fused_moe.cutedsl.grouped_gemm import nvfp4
from tensorrt_llm._torch.utils import AuxStreamType, EventType, Fp4QuantizedTensor


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
def test_triton_input_quant_boolean_forwarding(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    quantizer = Mock(return_value=(None, None))
    monkeypatch.setattr(nvfp4, "quantize_input_nvfp4", quantizer)
    backend = object()
    x = torch.empty(0, 1024, dtype=torch.bfloat16)
    nvfp4.TrtllmCutedslGroupedGemmNvfp4Impl.quantize_input(
        backend, x, post_quant_comm=False, enable_triton=enabled
    )
    quantizer.assert_called_once_with(backend, x, False, enable_triton=enabled)


def _backend() -> nvfp4.TrtllmCutedslGroupedGemmNvfp4Impl:
    cls = nvfp4.TrtllmCutedslGroupedGemmNvfp4Impl
    backend = cls.__new__(cls)
    torch.nn.Module.__init__(backend)
    backend.hidden_size = 4096
    backend.scaling_vector_size = 16
    backend.fc31_input_scale = torch.ones(1, device="cuda")
    backend.fc31_act_scale = None
    backend.aux_stream_dict = {AuxStreamType.MoeOutputMemset: torch.cuda.Stream()}
    backend.event_dict = {
        key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeOutputMemset]
    }
    backend._init_format_state()
    return backend


@skip_pre_blackwell
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "case",
    ["normal", "empty", "fp16", "unaligned", "strided", "awq", "prequantized"],
)
def test_triton_input_quant_dispatch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, case: str
) -> None:
    backend = _backend()
    hidden = 512 if case == "unaligned" else 4096
    backend.hidden_size = hidden
    x = torch.randn(5, hidden, device="cuda", dtype=torch.bfloat16)
    if case == "empty":
        x = x[:0]
    elif case == "fp16":
        x = x.half()
    elif case == "strided":
        x = torch.randn(5, hidden * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif case == "awq":
        backend.fc31_act_scale = torch.linspace(
            0.25, 2, hidden, device="cuda", dtype=torch.bfloat16
        )
    if case == "strided":
        tiled = Mock(wraps=kernel_inputs.nvfp4_quant)
        monkeypatch.setattr(kernel_inputs, "nvfp4_quant", tiled)
        with pytest.raises(RuntimeError, match="must be contiguous"):
            backend.quantize_input(x, enable_triton=enabled)
        tiled.assert_not_called()
        return
    reference_x = x if backend.fc31_act_scale is None else x * backend.fc31_act_scale
    expected = torch.ops.trtllm.fp4_quantize(
        reference_x, backend.fc31_input_scale, 16, False, False
    )
    if case == "prequantized":
        x = Fp4QuantizedTensor(expected[0], expected[1], is_sf_swizzled=False)
    tiled = Mock(wraps=kernel_inputs.nvfp4_quant)
    monkeypatch.setattr(kernel_inputs, "nvfp4_quant", tiled)
    actual = backend.quantize_input(x, enable_triton=enabled)
    should_tile = enabled and case in {"normal", "awq"}
    assert tiled.call_count == int(should_tile)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.flatten(), reference.flatten(), rtol=0, atol=0)
    assert actual[1].shape == (x.shape[0], hidden // 16)


@skip_pre_blackwell
@pytest.mark.parametrize("rows", [5, 1024, 1025])
def test_triton_input_quant_async_and_graph(rows: int) -> None:
    backend = _backend()
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    backend.quantize_input = partial(backend.quantize_input, enable_triton=True)
    # Warm the exact path before capture, then verify replay with changing data.
    backend.quantize_input_async(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, sf, ready = backend.quantize_input_async(x)
        if ready is not None:
            ready.wait()
        output = q.clone(), sf.clone()
    assert (ready is not None) == (
        nvfp4.ASYNC_INPUT_QUANT and rows <= nvfp4.ASYNC_INPUT_QUANT_MAX_TOKENS
    )
    for value in [0.0, -1.0, 2.5]:
        x.fill_(value)
        graph.replay()
        expected = torch.ops.trtllm.fp4_quantize(x, backend.fc31_input_scale, 16, False, False)
        for result, reference in zip(output, expected):
            torch.testing.assert_close(result.flatten(), reference.flatten(), rtol=0, atol=0)


@skip_pre_blackwell
def test_triton_input_quant_compile() -> None:
    backend = _backend()
    x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(backend.quantize_input, fullgraph=True)
    actual = compiled(x, enable_triton=True)
    expected = torch.ops.trtllm.fp4_quantize(x, backend.fc31_input_scale, 16, False, False)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.flatten(), reference.flatten(), rtol=0, atol=0)
