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
"""Small GPU regressions for automatic MXFP8 packing, not performance tests."""

from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.utils import counters

from tensorrt_llm._torch.visual_gen.attention_backend import cudnn, mxfp8_pack
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

# Words are reinterpreted, not converted: preserve BF16 sNaNs and signed zeros.
EDGE_WORDS = {
    "signed_zero": [0, 0x8000],
    "tiny": [0, 1, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x0081],
    "rounding": [0x3FDF, 0x3FE0, 0x3FE1, 0x43DF, 0x43E0, 0x43E1, 0x7F7F, 0xFF7F],
    "fp8_ties": [0x3F80, 0x3F88, 0x3F89, 0x3F90, 0xBF88, 0xBF89, 0x3B80, 0x3C00],
    "all_nan": [0x7FC1, 0xFFC1, 0x7F81, 0xFF81],
    "nan_positions": [0x3F80] * 7 + [0x7FC1] + [0x3F80] * 23 + [0x7F81],
    "infinity": [0x7F80, 0xFF80, 0, 0x8000, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F],
}


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("MXFP8 pack kernels require CUDA")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Vector MXFP8 pack kernels target SM100")


def _inputs(
    sequence: int,
    layout: str = "packed",
    offset: int = 0,
    head_dim: int = 128,
    pattern: str | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    batch, heads = 2, 2
    width = heads * head_dim * 3 if layout == "packed" else head_dim + 16
    count = batch * sequence * width if layout == "packed" else batch * heads * sequence * width
    storage = torch.empty(count + offset, device="cuda", dtype=torch.bfloat16)
    if pattern is None:
        generator = torch.Generator(device="cuda").manual_seed(17 + sequence)
        storage.normal_(generator=generator)
    else:
        words = EDGE_WORDS[pattern]
        signed = [word if word < 32768 else word - 65536 for word in words]
        tile = torch.tensor(signed, device="cuda", dtype=torch.int16)
        target = storage.view(torch.int16)
        complete = target.numel() // tile.numel() * tile.numel()
        target[:complete].view(-1, tile.numel()).copy_(tile)
        target[complete:].copy_(tile[: target.numel() - complete])
    if layout == "packed":
        packed = storage[offset:].view(batch, sequence, heads * head_dim * 3)
        tensors = tuple(
            plane.unflatten(-1, (heads, head_dim)).transpose(1, 2)
            for plane in packed.split(heads * head_dim, dim=-1)
        )
    else:
        padded = storage[offset:].view(batch, heads, sequence, head_dim + 16)
        tensors = (padded[..., :head_dim],) * 3
    return storage, tensors


def _full_payload(pair: tuple[torch.Tensor, torch.Tensor], kind: str) -> torch.Tensor:
    payload, scales = pair
    if kind == "qk":
        batch, heads, _, head_dim = payload.shape
        padded_s = scales.shape[2]
        shape = (batch, heads, padded_s, head_dim)
        stride = (heads * padded_s * head_dim, padded_s * head_dim, head_dim, 1)
        assert payload.storage_offset() == 0
        assert payload.stride() == stride
        assert payload.untyped_storage().nbytes() == batch * heads * padded_s * head_dim
        payload = payload.as_strided(shape, stride)
    return payload.view(torch.uint8)


def _assert_equal(
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor],
    kind: str,
) -> None:
    for reference, result in zip(expected, actual, strict=True):
        assert result.shape == reference.shape
        assert result.stride() == reference.stride()
        assert result.dtype == reference.dtype
        assert result.storage_offset() == reference.storage_offset()
        assert result.untyped_storage().nbytes() == reference.untyped_storage().nbytes()
    # Compare every backing payload byte (including Q/K tail rows) and every SF.
    assert torch.equal(_full_payload(expected, kind), _full_payload(actual, kind))
    assert torch.equal(expected[1], actual[1])


def _native(x: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    function = cudnn._quantize_mxfp8_qk_native if kind == "qk" else cudnn._quantize_mxfp8_v_native
    return function(x)


def _automatic(x: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    function = cudnn._quantize_mxfp8_qk if kind == "qk" else cudnn._quantize_mxfp8_v
    return function(x)


@pytest.mark.parametrize("pattern", list(EDGE_WORDS))
@torch.no_grad()
def test_mxfp8_pack_special_value_bytes(pattern: str) -> None:
    _require_sm100()
    storage, (x, _, _) = _inputs(4097, pattern=pattern)
    original = storage.view(torch.uint8).clone()
    for kind in ("qk", "v"):
        expected = _native(x, kind)
        with patch.object(mxfp8_pack, "_pack", wraps=mxfp8_pack._pack) as call:
            actual = _automatic(x, kind)
            call.assert_called_once()
        _assert_equal(expected, actual, kind)
    assert torch.equal(storage.view(torch.uint8), original)


@pytest.mark.parametrize(
    "sequence,layout,offset", [(4096, "packed", 0), (4097, "outer", 8), (4224, "packed", 8)]
)
@torch.no_grad()
def test_mxfp8_pack_layout_bytes(sequence: int, layout: str, offset: int) -> None:
    _require_sm100()
    storage, tensors = _inputs(sequence, layout=layout, offset=offset)
    original = storage.view(torch.uint8).clone()
    assert all(mxfp8_pack.metadata_eligible(x) for x in tensors)
    for x, kind in zip(tensors, ("qk", "qk", "v"), strict=True):
        expected = _native(x, kind)
        with patch.object(mxfp8_pack, "_pack", wraps=mxfp8_pack._pack) as call:
            actual = _automatic(x, kind)
            call.assert_called_once()
        _assert_equal(expected, actual, kind)
    assert torch.equal(storage.view(torch.uint8), original)


@pytest.mark.parametrize("sequence", [1, 31, 32, 33, 127, 128, 129])
@torch.no_grad()
def test_mxfp8_pack_kernel_tail_bytes(sequence: int) -> None:
    """Exercise private kernels below dispatch threshold; no auto-routing claim."""
    _require_sm100()
    _, (x, _, _) = _inputs(sequence, offset=8)
    assert not mxfp8_pack.metadata_eligible(x)
    for kind in ("qk", "v"):
        _assert_equal(_native(x, kind), mxfp8_pack._pack(x, kind), kind)


@pytest.mark.parametrize("case", ["short", "unaligned", "wrong_d", "inner_stride", "requires_grad"])
@torch.no_grad()
def test_mxfp8_pack_native_fallback(case: str) -> None:
    _require_sm100()
    sequence = 4095 if case == "short" else 4097
    head_dim = 64 if case == "wrong_d" else 256 if case == "inner_stride" else 128
    _, (x, _, _) = _inputs(sequence, offset=1 if case == "unaligned" else 0, head_dim=head_dim)
    if case == "inner_stride":
        x = x[..., ::2]
    if case == "requires_grad":
        x.requires_grad_(True)
    for kind in ("qk", "v"):
        expected = _native(x, kind)
        with patch.object(mxfp8_pack, "_pack", side_effect=AssertionError("fallback fused")):
            _assert_equal(expected, _automatic(x, kind), kind)


def _dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
    return cudnn._quantize_mxfp8_qk(q), cudnn._quantize_mxfp8_qk(k), cudnn._quantize_mxfp8_v(v)


@torch.no_grad()
def test_mxfp8_pack_fullgraph_cache_and_unaligned_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_sm100()
    torch._dynamo.reset()
    monkeypatch.setattr(mxfp8_pack, "_COMPILED", {})
    frames: list[torch.fx.GraphModule] = []
    inductor = lookup_backend("inductor")

    def backend(graph: torch.fx.GraphModule, args: list[torch.Tensor]) -> Callable[..., object]:
        frames.append(graph)
        return inductor(graph, args)

    _, inputs = _inputs(4096, offset=8)
    expected = tuple(_native(x, kind) for x, kind in zip(inputs, ("qk", "qk", "v"), strict=True))
    before_graphs = counters["stats"]["unique_graphs"]
    before_calls = counters["stats"]["calls_captured"]
    compiled = torch.compile(_dispatch, backend=backend, fullgraph=True, dynamic=False)
    actual = compiled(*inputs)
    torch.cuda.synchronize()
    assert frames and counters["stats"]["unique_graphs"] > before_graphs
    assert counters["stats"]["calls_captured"] > before_calls
    nodes = [str(node.target) for frame in frames for node in frame.graph.nodes]
    assert any("visual_gen_mxfp8_qk" in node for node in nodes)
    assert any("visual_gen_mxfp8_v" in node for node in nodes)
    for reference, result, kind in zip(expected, actual, ("qk", "qk", "v"), strict=True):
        _assert_equal(reference, result, kind)
    assert len(mxfp8_pack._COMPILED) == 2
    cache = {key: id(value) for key, value in mxfp8_pack._COMPILED.items()}
    frame_count = len(frames)
    for _ in range(3):
        results = compiled(*inputs)
        for reference, result, kind in zip(expected, results, ("qk", "qk", "v"), strict=True):
            _assert_equal(reference, result, kind)
    assert len(frames) == frame_count
    assert {key: id(value) for key, value in mxfp8_pack._COMPILED.items()} == cache

    # Identical shape/stride but foreign alignment: same fake contract, native runtime arm.
    _, unaligned = _inputs(4096, offset=1)
    expected = tuple(_native(x, kind) for x, kind in zip(unaligned, ("qk", "qk", "v"), strict=True))
    with patch.object(mxfp8_pack, "_pack", side_effect=AssertionError("unaligned input fused")):
        actual = compiled(*unaligned)
    for reference, result, kind in zip(expected, actual, ("qk", "qk", "v"), strict=True):
        _assert_equal(reference, result, kind)
    assert {key: id(value) for key, value in mxfp8_pack._COMPILED.items()} == cache


@torch.no_grad()
def test_mxfp8_pack_sm103_compiler_visible_fallback() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Native SM103 fallback requires actual SM103 hardware")
    torch._dynamo.reset()
    _, inputs = _inputs(4096)
    assert all(not mxfp8_pack.metadata_eligible(x) for x in inputs)
    frames: list[torch.fx.GraphModule] = []
    inductor = lookup_backend("inductor")

    def backend(graph: torch.fx.GraphModule, args: list[torch.Tensor]) -> Callable[..., object]:
        frames.append(graph)
        return inductor(graph, args)

    expected = tuple(_native(x, kind) for x, kind in zip(inputs, ("qk", "qk", "v"), strict=True))
    compiled = torch.compile(_dispatch, backend=backend, fullgraph=True, dynamic=False)
    with patch.object(mxfp8_pack, "_pack", side_effect=AssertionError("SM103 used SM100 kernel")):
        actual = compiled(*inputs)
    assert frames
    assert not any("visual_gen_mxfp8_" in str(n.target) for f in frames for n in f.graph.nodes)
    for reference, result, kind in zip(expected, actual, ("qk", "qk", "v"), strict=True):
        _assert_equal(reference, result, kind)


@pytest.mark.parametrize("kind", ["qk", "v"])
@torch.no_grad()
def test_mxfp8_pack_fake_and_schema(kind: str) -> None:
    _require_sm100()
    _, (x, _, _) = _inputs(4097, offset=8)
    operation = (
        torch.ops.trtllm.visual_gen_mxfp8_qk.default
        if kind == "qk"
        else torch.ops.trtllm.visual_gen_mxfp8_v.default
    )
    torch.library.opcheck(operation, (x,), test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("kv_sequence", [512, 4096], ids=["cross_attention", "self_attention"])
@torch.no_grad()
def test_mxfp8_pack_cudnn_consumer(kv_sequence: int) -> None:
    _require_sm100()
    _, (q, _, _) = _inputs(4096)
    _, (_, k, v) = _inputs(kv_sequence)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    attention = cudnn.CuDNNAttention(
        num_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        quant_attention_config=QuantAttentionConfig(qk_dtype="mxfp8", v_dtype="mxfp8"),
    )
    # Reference and candidate both use the real production consumer and every output sink.
    with (
        patch.object(cudnn, "_quantize_mxfp8_qk", cudnn._quantize_mxfp8_qk_native),
        patch.object(cudnn, "_quantize_mxfp8_v", cudnn._quantize_mxfp8_v_native),
    ):
        expected = attention.forward_with_lse(q, k, v)
    with patch.object(mxfp8_pack, "_pack", wraps=mxfp8_pack._pack) as call:
        actual = attention.forward_with_lse(q, k, v)
        assert call.call_count == (1 if kv_sequence == 512 else 3)
    for reference, result in zip(expected, actual, strict=True):
        assert torch.isfinite(reference).all().item() and torch.isfinite(result).all().item()
        assert torch.equal(reference.view(torch.uint8), result.view(torch.uint8))
