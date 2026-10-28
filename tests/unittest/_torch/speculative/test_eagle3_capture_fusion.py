# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact CUDA parity for bounded EAGLE3 capture and the following RMSNorm."""

import weakref
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative import eagle3
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.utils import piecewise_cuda_graph, torch_compiling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_CONTROL = "TLLM_GPT_OSS_ENABLE_FUSED_EAGLE_CAPTURE"


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # Include BF16 rounding, signed zero and NaN payloads in the comparison.
    assert torch.equal(
        actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)
    )


def _metadata(
    width: int, capacity: int, enabled: bool, layers: set[int], monkeypatch: pytest.MonkeyPatch
) -> eagle3.Eagle3OneModelSpecMetadata:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    return eagle3.Eagle3OneModelSpecMetadata(
        max_num_requests=1,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        num_layers=36,
        hidden_size=width,
        max_num_tokens=capacity,
        dtype=torch.bfloat16,
        layers_to_capture=layers,
    )


@pytest.mark.parametrize("width", [7, 2880])
@pytest.mark.parametrize("num_tokens", [0, 1, 127, 128, 129, 8192])
@torch.inference_mode()
def test_fused_capture_exact_values_strides_and_untouched_storage(
    width: int, num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(42)
    metadata = _metadata(width, num_tokens + 3, True, {32, 1, 17}, monkeypatch)
    metadata.num_capture_tokens = num_tokens
    metadata.num_tokens = max(0, num_tokens - 3)
    dest = metadata.hidden_states
    pointer, stride = dest.data_ptr(), dest.stride()
    dest.fill_(-64)
    expected = dest.clone()
    hidden = torch.randn((num_tokens + 5, width), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    original_residual = residual.clone()
    copy = Mock(wraps=eagle3.inplace_slice_copy)
    monkeypatch.setattr(eagle3, "inplace_slice_copy", copy)

    for index, layer_id in enumerate(metadata.layers_to_capture):
        source = hidden + layer_id
        original_source = source.clone()
        expected[:num_tokens, index * width : (index + 1) * width] = (
            source[:num_tokens] + residual[:num_tokens]
        )
        metadata.maybe_capture_hidden_states(layer_id, source, residual)
        _assert_bits(dest, expected)
        _assert_bits(source, original_source)
        _assert_bits(residual, original_residual)
    copy.assert_not_called()
    assert metadata.hidden_states is dest
    assert dest.data_ptr() == pointer and dest.stride() == stride and dest.is_contiguous()


@torch.inference_mode()
def test_fused_capture_bf16_rounding_and_special_values(monkeypatch: pytest.MonkeyPatch) -> None:
    width = 2880
    left = torch.tensor(
        [1, 1.0078125, -1, -1.0078125, 2**-126, 2**-133, 0.0, -0.0, torch.inf, -torch.inf],
        device="cuda",
        dtype=torch.bfloat16,
    ).repeat(width // 10)
    right = torch.tensor(
        [2**-8, 2**-8, -(2**-8), -(2**-8), -(2**-126), 2**-133, -0.0, -0.0, -torch.inf, torch.inf],
        device="cuda",
        dtype=torch.bfloat16,
    ).repeat(width // 10)
    hidden, residual = left.repeat(3, 1), right.repeat(3, 1)
    dest = torch.full((5, width * 3), -64, device="cuda", dtype=torch.bfloat16)
    expected = dest.clone()
    expected[:3, width : width * 2] = hidden + residual
    copy = Mock(wraps=eagle3.inplace_slice_copy)
    monkeypatch.setattr(eagle3, "inplace_slice_copy", copy)
    eagle3._fused_eagle3_capture(dest, hidden, residual, 3, width, width * 2)
    copy.assert_not_called()
    _assert_bits(dest, expected)


@pytest.mark.parametrize(
    "case", ["fp16", "fp32", "broadcast", "strided", "alias_hidden", "alias_residual"]
)
@torch.inference_mode()
def test_unsupported_cuda_capture_retains_incumbent_copy(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(42)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
    dest = torch.full((6, 2880), -64, device="cuda", dtype=dtype)
    hidden = torch.randn((8, 2880), device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden)
    if case == "broadcast":
        residual = residual[:1]
    elif case == "strided":
        hidden = torch.randn((8, 5760), device="cuda", dtype=dtype)[:, ::2]
    elif case == "alias_hidden":
        hidden = dest
    elif case == "alias_residual":
        residual = dest
    expected = dest.clone()
    expected[:4] = hidden[:4] + residual[:4]
    copy = Mock(wraps=eagle3.inplace_slice_copy)
    monkeypatch.setattr(eagle3, "inplace_slice_copy", copy)
    eagle3._fused_eagle3_capture(dest, hidden, residual, 4, 0, 2880)
    copy.assert_called_once()
    torch.testing.assert_close(dest, expected, rtol=0, atol=0)


class _CaptureThenNorm(torch.nn.Module):
    def __init__(self, metadata: eagle3.Eagle3OneModelSpecMetadata) -> None:
        super().__init__()
        self.metadata = metadata
        self.norm = RMSNorm(hidden_size=metadata.hidden_size, eps=1e-5, dtype=torch.bfloat16).cuda()

    def forward(
        self, inputs_embeds: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.metadata.maybe_capture_hidden_states(17, inputs_embeds, residual)
        normalized, residual = self.norm(inputs_embeds, residual)
        # Expose downstream users as the real model does. Returning only input
        # aliases lets Dynamo reconstruct them outside the graph, hiding the
        # norm outputs from the backend's recovery/fusion passes. Views add no
        # GPU copy and preserve values, shape, strides and storage.
        return (
            normalized.view_as(normalized),
            residual.view_as(residual),
            self.metadata.hidden_states,
        )


@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_capture_then_norm_eager_and_residual_free_copy(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_meta = _metadata(2880, 132, enabled, {1, 17, 32}, monkeypatch)
    incumbent_meta = _metadata(2880, 132, False, {1, 17, 32}, monkeypatch)
    candidate_meta.num_capture_tokens = incumbent_meta.num_capture_tokens = 129
    candidate = _CaptureThenNorm(candidate_meta)
    incumbent = _CaptureThenNorm(incumbent_meta)
    incumbent.load_state_dict(candidate.state_dict())
    torch.manual_seed(42)
    for _ in range(3):
        source = torch.randn((256, 2880), device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(source)
        candidate_meta.hidden_states.fill_(-64)
        incumbent_meta.hidden_states.fill_(-64)
        expected = incumbent(source.clone(), residual.clone())
        actual = candidate(source.clone(), residual.clone())
        for result, reference in zip(actual, expected):
            _assert_bits(result, reference)
        candidate_meta.hidden_states.fill_(-64)
        incumbent_meta.hidden_states.fill_(-64)
        candidate_meta.maybe_capture_hidden_states(17, source, None)
        incumbent_meta.maybe_capture_hidden_states(17, source, None)
        _assert_bits(candidate_meta.hidden_states, incumbent_meta.hidden_states)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("live,physical", [(1, 8), (127, 128), (129, 256), (7700, 8192)])
@torch.inference_mode()
def test_capture_then_unchanged_norm_compiled_and_changed_input_replay(
    enabled: bool,
    piecewise: bool,
    live: int,
    physical: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.utils import _model_extra_attrs

    compiler_streams = Backend.Streams()
    monkeypatch.setattr(
        _model_extra_attrs,
        "attrs",
        {
            "global_stream": torch.cuda.current_stream(),
            "aux_streams": weakref.ref(compiler_streams),
        },
        raising=False,
    )
    # Match the serving configuration: fullgraph, Inductor disabled.
    backend = Backend(
        enable_inductor=False,
        enable_piecewise_cuda_graph=piecewise,
        capture_num_tokens=[8, 128, 256, 8192],
    )
    optimized = []
    original_optimize = backend.optimize

    def record_optimize(gm: torch.fx.GraphModule, examples: list[torch.Tensor]):
        result = original_optimize(gm, examples)
        optimized.append(gm)
        return result

    monkeypatch.setattr(backend, "optimize", record_optimize)
    candidate_meta = _metadata(2880, live + 3, enabled, {1, 17, 32}, monkeypatch)
    incumbent_meta = _metadata(2880, live + 3, False, {1, 17, 32}, monkeypatch)
    candidate_meta.num_capture_tokens = incumbent_meta.num_capture_tokens = live
    candidate_meta.num_tokens = incumbent_meta.num_tokens = max(0, live - 3)
    candidate = _CaptureThenNorm(candidate_meta)
    incumbent = _CaptureThenNorm(incumbent_meta)
    incumbent.load_state_dict(candidate.state_dict())
    compiled = torch.compile(candidate, backend=backend, fullgraph=True)
    inputs = torch.empty((physical, 2880), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty_like(inputs)
    pointer = candidate_meta.hidden_states.data_ptr()
    torch.manual_seed(42)

    try:
        with (
            capture_piecewise_cuda_graph(piecewise),
            torch_compiling(True),
            piecewise_cuda_graph(piecewise),
        ):
            for _ in range(5):
                inputs.normal_()
                residual.normal_()
                compiled(inputs, residual)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # PiecewiseRunner performs its own capture/replay; fullgraph uses
            # the executor's outer graph boundary, reproduced here.
            if not piecewise:
                with torch.cuda.graph(graph):
                    outputs = compiled(inputs, residual)
            for _ in range(3):
                source = torch.randn_like(inputs)
                source_residual = torch.randn_like(residual)
                incumbent_meta.hidden_states.fill_(-64)
                expected = incumbent(source.clone(), source_residual.clone())
                candidate_meta.hidden_states.fill_(-64)
                inputs.copy_(source)
                residual.copy_(source_residual)
                if piecewise:
                    outputs = compiled(inputs, residual)
                else:
                    graph.replay()
                for actual, reference in zip(outputs, expected):
                    _assert_bits(actual, reference)
                assert candidate_meta.hidden_states.data_ptr() == pointer
        if not piecewise:
            targets = [node.target for gm in optimized for node in gm.graph.nodes]
            capture_op = (
                torch.ops.trtllm.fused_eagle3_capture.default
                if enabled
                else torch.ops.trtllm.inplace_slice_copy.default
            )
            assert capture_op in targets
            assert torch.ops.aten.clone.default not in targets
            assert targets.index(capture_op) < targets.index(
                torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default
            )
    finally:
        backend.clear_piecewise_cuda_graphs()
        torch._dynamo.reset()
