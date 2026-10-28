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
"""CPU regressions for EAGLE3 capture policy and the incumbent fallback.

Only unrelated CUDA metadata allocations and the CUDA slice-copy kernel are
replaced. These tests exercise CPU fallback semantics; CUDA fusion parity is
covered separately by GPU tests.
"""

import weakref
from collections.abc import Callable
from operator import getitem
from unittest.mock import Mock, patch

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized, auto_functionalized_v2
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph

from tensorrt_llm._torch.compilation.remove_copy_pass import remove_copy_for_mutates_args
from tensorrt_llm._torch.speculative import eagle3
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.cpu_only

_CONTROL = "TLLM_GPT_OSS_ENABLE_FUSED_EAGLE_CAPTURE"


@pytest.fixture
def cpu_metadata_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    def to_cpu(factory: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def allocate(*args, **kwargs) -> torch.Tensor:
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return allocate

    for name in ("empty", "full"):
        monkeypatch.setattr(torch, name, to_cpu(getattr(torch, name)))


@pytest.fixture
def cpu_slice_copy(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def copy_slice(dest: torch.Tensor, src: torch.Tensor, dim1_start: int, dim1_end: int) -> None:
        # Match inplaceSliceCopyOp.cpp, allowing CPU tensors for this fixture.
        if dest.device != src.device:
            raise RuntimeError("dest and src must be on the same device")
        if not dest.is_contiguous():
            raise RuntimeError("dest must be contiguous")
        if not src.is_contiguous():
            raise RuntimeError("src must be contiguous")
        if dest.ndim != 2:
            raise RuntimeError("dest must be 2-D")
        if src.ndim != 2:
            raise RuntimeError("src must be 2-D")
        if dest.dtype != src.dtype:
            raise RuntimeError("dest and src must have the same dtype")
        if dim1_start < 0:
            raise RuntimeError("dim1Start must be non-negative")
        if dim1_end <= dim1_start:
            raise RuntimeError("dim1End must be greater than dim1Start")
        if src.shape[0] > dest.shape[0]:
            raise RuntimeError("numTokens exceeds dest row count")
        if dim1_end > dest.shape[1]:
            raise RuntimeError("dim1End exceeds dest column count")
        if src.shape[1] != dim1_end - dim1_start:
            raise RuntimeError("src column count must equal dim1End - dim1Start")
        dest[: src.shape[0], dim1_start:dim1_end].copy_(src)

    copy = Mock(side_effect=copy_slice)
    monkeypatch.setattr(eagle3, "inplace_slice_copy", copy)
    return copy


def _metadata(
    layers: set[int] | None = None,
    manager: eagle3.Eagle3ResourceManager | None = None,
) -> eagle3.Eagle3OneModelSpecMetadata:
    return eagle3.Eagle3OneModelSpecMetadata(
        max_num_requests=2,
        max_draft_len=1,
        max_total_draft_tokens=1,
        spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        num_layers=12,
        layers_to_capture=layers,
        hidden_size=4,
        max_num_tokens=6,
        dtype=torch.bfloat16,
        spec_resource_manager=manager,
    )


def _sources(rows: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    return (values * 0.1).to(torch.bfloat16), (values * -0.03).to(torch.bfloat16)


@pytest.mark.parametrize("control", [None, "", "0", "false", "true", "01", "1"])
def test_capture_control_is_default_off_strict_and_cached(
    control: str | None,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    if control is None:
        monkeypatch.delenv(_CONTROL, raising=False)
    else:
        monkeypatch.setenv(_CONTROL, control)
    metadata = _metadata({8, 2, 5})
    enabled = control == "1"
    assert metadata._enable_fused_eagle_capture is enabled
    assert metadata.layers_to_capture == [2, 5, 8]
    monkeypatch.setenv(_CONTROL, "0" if enabled else "1")
    metadata.num_capture_tokens = 3
    metadata.hidden_states.fill_(-64)
    hidden, residual = _sources()
    with patch.object(eagle3, "_fused_eagle3_capture", wraps=eagle3._fused_eagle3_capture) as fused:
        metadata.maybe_capture_hidden_states(5, hidden, residual)
    assert fused.call_count == int(enabled)
    cpu_slice_copy.assert_called_once()
    torch.testing.assert_close(
        metadata.hidden_states[:3, 4:8], hidden[:3] + residual[:3], rtol=0, atol=0
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_capture_tokens", [0, 1, 4, 6])
@pytest.mark.parametrize("custom_layers", [False, True])
def test_capture_preserves_layer_order_real_rows_and_input_values(
    enabled: bool,
    num_capture_tokens: int,
    custom_layers: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    metadata = _metadata({8, 2, 5} if custom_layers else None)
    expected_layers = [2, 5, 8] if custom_layers else [1, 5, 8]
    assert list(metadata.layers_to_capture) == expected_layers
    metadata.num_capture_tokens = num_capture_tokens
    # prepare() may reduce num_tokens to the attention-DP subsequence shape.
    metadata.num_tokens = max(0, num_capture_tokens - 2)
    capture_buffer = metadata.hidden_states
    capture_buffer.fill_(-64)
    expected = capture_buffer.clone()
    hidden, residual = _sources()
    hidden_before, residual_before = hidden.clone(), residual.clone()

    for index, layer_id in enumerate(expected_layers):
        layer_hidden = hidden + layer_id
        layer_hidden_before = layer_hidden.clone()
        expected[:num_capture_tokens, index * 4 : (index + 1) * 4] = (
            layer_hidden[:num_capture_tokens] + residual[:num_capture_tokens]
        )
        metadata.maybe_capture_hidden_states(layer_id, layer_hidden, residual)
        torch.testing.assert_close(layer_hidden, layer_hidden_before, rtol=0, atol=0)

    assert metadata.hidden_states is capture_buffer
    assert cpu_slice_copy.call_count == len(expected_layers)
    torch.testing.assert_close(capture_buffer, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_without_residual_preserves_legacy_copy(
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    metadata = _metadata({8, 2, 5})
    metadata.num_capture_tokens = 4
    metadata.hidden_states.fill_(-64)
    expected = metadata.hidden_states.clone()
    hidden, _ = _sources()
    expected[:4, 8:12] = hidden[:4]
    with patch.object(eagle3, "_fused_eagle3_capture") as fused:
        metadata.maybe_capture_hidden_states(8, hidden)
    fused.assert_not_called()
    cpu_slice_copy.assert_called_once()
    src = cpu_slice_copy.call_args.args[1]
    assert src.data_ptr() == hidden.data_ptr()
    torch.testing.assert_close(metadata.hidden_states, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["capacity", "hidden_source", "residual_source"])
def test_capture_errors_preserve_destination(
    enabled: bool,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    metadata = _metadata({2})
    metadata.num_capture_tokens = 7 if failure == "capacity" else 4
    metadata.hidden_states.fill_(-64)
    before = metadata.hidden_states.clone()
    hidden, residual = _sources(3 if failure == "hidden_source" else 8)
    if failure == "residual_source":
        residual = residual[:2]
    match = {"capacity": "buffer capacity", "hidden_source": "available hidden states"}.get(failure)
    with patch.object(eagle3, "_fused_eagle3_capture", wraps=eagle3._fused_eagle3_capture) as fused:
        with pytest.raises(RuntimeError, match=match):
            metadata.maybe_capture_hidden_states(2, hidden, residual)
    assert fused.call_count == int(enabled and failure == "residual_source")
    cpu_slice_copy.assert_not_called()
    torch.testing.assert_close(metadata.hidden_states, before, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_capture_layer_does_not_validate_or_write(
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    metadata = _metadata({2, 5, 8})
    metadata.num_capture_tokens = 9
    metadata.hidden_states.fill_(-64)
    before = metadata.hidden_states.clone()
    hidden, residual = _sources(0)
    with patch.object(eagle3, "_fused_eagle3_capture") as fused:
        metadata.maybe_capture_hidden_states(9, hidden, residual)
    fused.assert_not_called()
    cpu_slice_copy.assert_not_called()
    torch.testing.assert_close(metadata.hidden_states, before, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_empty_capture_layers_need_no_capture_buffer(
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, str(int(enabled)))
    metadata = _metadata(set())
    assert metadata.hidden_states is None
    metadata.num_capture_tokens = 9
    hidden, residual = _sources(0)
    with patch.object(eagle3, "_fused_eagle3_capture") as fused:
        metadata.maybe_capture_hidden_states(2, hidden, residual)
    fused.assert_not_called()
    cpu_slice_copy.assert_not_called()


def test_capture_preserves_borrowed_manager_buffer(
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    monkeypatch.setenv(_CONTROL, "1")
    manager = object.__new__(eagle3.Eagle3ResourceManager)
    manager.hidden_states = torch.full((6, 12), -64, dtype=torch.bfloat16)
    manager.batch_indices_cuda = torch.empty(2, dtype=torch.int)
    metadata = _metadata({8, 2, 5}, manager)
    assert metadata.hidden_states is manager.hidden_states
    assert metadata.batch_indices_cuda is manager.batch_indices_cuda
    pointer = manager.hidden_states.data_ptr()
    metadata.num_capture_tokens = 4
    hidden, residual = _sources()
    expected = manager.hidden_states.clone()
    expected[:4, 4:8] = hidden[:4] + residual[:4]
    metadata.maybe_capture_hidden_states(5, hidden, residual)
    assert metadata.hidden_states is manager.hidden_states
    assert metadata.hidden_states.data_ptr() == pointer
    cpu_slice_copy.assert_called_once()
    torch.testing.assert_close(manager.hidden_states, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stream_state", ["missing", "empty", "nonempty", "dead"])
def test_capture_fusion_declines_active_or_expired_aux_streams(
    stream_state: str,
    monkeypatch: pytest.MonkeyPatch,
    cpu_metadata_allocation: None,
    cpu_slice_copy: Mock,
) -> None:
    class Streams(list[object]):
        pass

    monkeypatch.setenv(_CONTROL, "1")
    metadata = _metadata({8, 2, 5})
    metadata.num_capture_tokens = 4
    metadata.hidden_states.fill_(-64)
    hidden, residual = _sources()
    expected = metadata.hidden_states.clone()
    expected[:4, 4:8] = hidden[:4] + residual[:4]
    attrs = None
    if stream_state != "missing":
        streams = Streams([object()] if stream_state == "nonempty" else [])
        stream_ref = weakref.ref(streams)
        attrs = {"aux_streams": stream_ref}
        if stream_state == "dead":
            del streams
            assert stream_ref() is None

    with (
        patch.object(eagle3, "get_model_extra_attrs", return_value=attrs) as extra_attrs,
        patch.object(eagle3, "_fused_eagle3_capture", wraps=eagle3._fused_eagle3_capture) as fused,
    ):
        metadata.maybe_capture_hidden_states(5, hidden, residual)

    extra_attrs.assert_called_once_with()
    assert fused.call_count == int(stream_state in {"missing", "empty"})
    cpu_slice_copy.assert_called_once()
    args = cpu_slice_copy.call_args.args
    assert args[0] is metadata.hidden_states and args[2:] == (4, 8)
    torch.testing.assert_close(metadata.hidden_states, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "cpu_bf16",
        "float32",
        "int32",
        "mixed_dtype",
        "broadcast_rows",
        "broadcast_columns",
        "strided_source",
        "strided_destination",
        "source_alias",
        "requires_grad",
    ],
)
def test_fused_capture_cpu_fallback_matches_incumbent(case: str, cpu_slice_copy: Mock) -> None:
    hidden, residual = _sources()
    dest = torch.full((6, 12), -64, dtype=torch.bfloat16)
    if case == "float32":
        dest, hidden, residual = dest.float(), hidden.float(), residual.float()
    elif case == "int32":
        dest, hidden, residual = dest.int(), hidden.int(), residual.int()
    elif case == "mixed_dtype":
        residual = residual.float()
    elif case == "broadcast_rows":
        residual = residual[:1]
    elif case == "broadcast_columns":
        residual = residual[:, :1]
    elif case == "strided_source":
        hidden = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)[:, ::2]
    elif case == "strided_destination":
        dest = torch.full((6, 24), -64, dtype=torch.bfloat16)[:, ::2]
    elif case == "source_alias":
        hidden = dest[:, :4]
    elif case == "requires_grad":
        hidden.requires_grad_(True)

    expected = dest.clone()
    if case in {"mixed_dtype", "strided_destination"}:
        message = "same dtype" if case == "mixed_dtype" else "dest must be contiguous"
        with torch.no_grad(), pytest.raises(RuntimeError, match=message):
            eagle3._fused_eagle3_capture(dest, hidden, residual, 4, 4, 8)
        cpu_slice_copy.assert_called_once()
        torch.testing.assert_close(dest, expected, rtol=0, atol=0)
        return

    expected[:4, 4:8] = hidden[:4] + residual[:4]
    with torch.no_grad():
        result = eagle3._fused_eagle3_capture(dest, hidden, residual, 4, 4, 8)
    assert result is None
    cpu_slice_copy.assert_called_once()
    args = cpu_slice_copy.call_args.args
    assert args[0] is dest and args[2:] == (4, 8)
    if case == "strided_source":
        assert not hidden.is_contiguous() and args[1].is_contiguous()
    torch.testing.assert_close(dest, expected, rtol=0, atol=0)


def test_fused_capture_fake_returns_none_and_declares_destination_mutation(
    cpu_slice_copy: Mock,
) -> None:
    schema = torch.ops.trtllm.fused_eagle3_capture.default._schema
    assert len(schema.returns) == 0
    writes = [arg.name for arg in schema.arguments if arg.alias_info and arg.alias_info.is_write]
    assert writes == ["dest"]
    dest = torch.full((6, 12), -64, dtype=torch.bfloat16)
    before = dest.clone()
    hidden, residual = _sources()
    with FakeTensorMode(allow_non_fake_inputs=True):
        result = eagle3._fused_eagle3_capture(dest, hidden, residual, 4, 4, 8)
    assert result is None
    cpu_slice_copy.assert_not_called()
    torch.testing.assert_close(dest, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "functionalize", [auto_functionalized, auto_functionalized_v2], ids=["v1", "v2"]
)
def test_fused_capture_compiler_preserves_persistent_destination(
    functionalize: Callable[..., object],
) -> None:
    graph = Graph()
    dest = graph.placeholder("dest")
    hidden = graph.placeholder("hidden_states")
    residual = graph.placeholder("residual")
    capture_op = torch.ops.trtllm.fused_eagle3_capture.default
    kwargs = {
        "hidden_states": hidden,
        "residual": residual,
        "num_tokens": 4,
        "dim1_start": 4,
        "dim1_end": 8,
    }
    if functionalize is auto_functionalized_v2:
        kwargs.update(_all_bases=(dest,), _dest_base_index=0)
    else:
        kwargs["dest"] = dest
    functionalized = graph.call_function(functionalize, args=(capture_op,), kwargs=kwargs)
    mutated_dest = graph.call_function(getitem, args=(functionalized, 1))
    output = graph.output(mutated_dest)

    # Use the real inplace_info registration: omitting the capture op must fail.
    remove_copy_for_mutates_args(graph)

    calls = [node for node in graph.nodes if node.op == "call_function"]
    assert len(calls) == 1 and calls[0].target is capture_op
    assert calls[0].kwargs == {
        "dest": dest,
        "hidden_states": hidden,
        "residual": residual,
        "num_tokens": 4,
        "dim1_start": 4,
        "dim1_end": 8,
    }
    assert output.args[0] is dest
    assert all(
        node.target
        not in (auto_functionalized, auto_functionalized_v2, torch.ops.aten.clone.default)
        for node in graph.nodes
    )
    graph.lint()
