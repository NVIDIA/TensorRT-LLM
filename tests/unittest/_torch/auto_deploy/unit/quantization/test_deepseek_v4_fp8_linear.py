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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FineGrainedFP8LinearQuantization,
    FP8LinearQuantizationFromConfig,
)
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_fp32
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")
_requires_e8m0 = pytest.mark.skipif(
    not _HAS_E8M0,
    reason="torch.float8_e8m0fnu is not available in this PyTorch build",
)


class _DummyFactory:
    def __init__(self, quant_config: dict[str, object]) -> None:
        self._quant_config = quant_config

    def get_quant_config(self) -> dict[str, object]:
        return self._quant_config


class _DeepSeekV4LinearFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = nn.Module()
        self.layers[0].attn.wq_a = nn.Linear(16, 8, bias=False)
        self.layers[0].attn.compressor = nn.Module()
        self.layers[0].attn.compressor.wkv = nn.Linear(16, 8, bias=False)
        self.layers[0].ffn = nn.Module()
        self.layers[0].ffn.gate = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].attn.wq_a.weight, None
            ),
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].attn.compressor.wkv.weight, None
            ),
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].ffn.gate.weight, None
            ),
        )


class _GenericLinearFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_linear_simple.default(x, self.proj.weight, None)


def _trace_with_weight_meta(module: nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
    return gm


def _deepseek_v4_quant_config() -> dict[str, object]:
    return {
        "quant_method": "deepseek_v4_fp8",
        "linear_quant_method": "finegrained_fp8",
        "weight_block_size": [128, 128],
        "exclude_modules": [
            "embed",
            "head",
            "*.ffn.gate",
            "*.attn.compressor",
            "*.attn.indexer.compressor",
            "*.attn.indexer.weights_proj",
            "*.norm",
            "*.hc_*",
            "*.attn_sink",
            "mtp.*",
        ],
    }


def _run_finegrained_transform(
    gm: torch.fx.GraphModule,
    quant_config: dict[str, object],
) -> tuple[torch.fx.GraphModule, object]:
    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    return transform._apply(gm, None, _DummyFactory(quant_config), SharedConfig())


def _load_finegrained_state_dict(state_dict: dict[str, torch.Tensor], weight_name: str) -> None:
    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    transform.load_hook(state_dict, "", weight_name=weight_name)


def _fp8_weight(shape: tuple[int, int]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float8_e4m3fn, device="cpu")


@_requires_e8m0
def _e8m0_from_bytes(raw_bytes: list[int], shape: tuple[int, ...]) -> torch.Tensor:
    return (
        torch.tensor(raw_bytes, dtype=torch.uint8, device="cpu")
        .view(torch.float8_e8m0fnu)
        .view(shape)
    )


def test_deepseek_v4_scale_alias_loads_into_weight_scale_inv() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_alias = "layers.0.attn.wq_a.scale"
    scale_buffer = "layers.0.attn.wq_a.weight_scale_inv"
    scale = torch.ones((8, 32), dtype=torch.bfloat16, device="cpu")
    state_dict = {
        weight_name: _fp8_weight((1024, 4096)),
        scale_alias: scale,
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    assert scale_alias not in state_dict
    assert state_dict[scale_buffer] is scale
    assert state_dict[scale_buffer].shape == (8, 32)


@_requires_e8m0
def test_deepseek_v4_e8m0_scale_alias_decodes_to_fp32() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_alias = "layers.0.attn.wq_a.scale"
    scale = _e8m0_from_bytes([126, 127, 128, 129, 130, 131], (2, 3))
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_alias: scale,
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    actual = state_dict["layers.0.attn.wq_a.weight_scale_inv"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, e8m0_to_fp32(scale), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("weight_shape", "scale_shape"),
    [
        ((1024, 4096), (8, 32)),
        ((129, 257), (2, 3)),
    ],
)
def test_deepseek_v4_scale_shape_validation_accepts_ceil_shapes(
    weight_shape: tuple[int, int],
    scale_shape: tuple[int, int],
) -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    state_dict = {
        weight_name: _fp8_weight(weight_shape),
        "layers.0.attn.wq_a.scale": torch.ones(scale_shape, dtype=torch.float32),
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    assert state_dict["layers.0.attn.wq_a.weight_scale_inv"].shape == scale_shape


def test_deepseek_v4_scale_shape_validation_rejects_mismatches() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        "layers.0.attn.wq_a.scale": torch.ones((1, 3), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=r"expected shape \(2, 3\)"):
        _load_finegrained_state_dict(state_dict, weight_name)


def test_deepseek_v4_transform_quantizes_only_dense_fp8_linear_paths() -> None:
    model = _DeepSeekV4LinearFixture().to(torch.bfloat16)
    gm = _trace_with_weight_meta(model)

    transformed, info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())

    assert info.num_matches == 1
    assert transformed.get_submodule("layers.0.attn.wq_a").weight.dtype == torch.float8_e4m3fn
    assert transformed.get_submodule("layers.0.attn.wq_a").weight_scale_inv.shape == (1, 1)
    assert transformed.get_submodule("layers.0.ffn.gate").weight.dtype == torch.bfloat16
    assert transformed.get_submodule("layers.0.attn.compressor.wkv").weight.dtype == torch.bfloat16

    quantized_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
    ]
    plain_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_linear_simple)
    ]
    assert len(quantized_nodes) == 1
    assert len(plain_nodes) == 2


def test_non_deepseek_finegrained_fp8_still_uses_weight_scale_inv() -> None:
    weight_name = "proj.weight"
    state_dict = {
        weight_name: _fp8_weight((8, 16)),
        "proj.weight_scale_inv": torch.ones((1, 1), dtype=torch.bfloat16),
        "proj.scale": torch.full((1, 1), 2.0, dtype=torch.bfloat16),
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    torch.testing.assert_close(
        state_dict["proj.weight_scale_inv"],
        torch.ones((1, 1), dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert "proj.scale" in state_dict

    gm = _trace_with_weight_meta(_GenericLinearFixture().to(torch.bfloat16))
    transformed, info = _run_finegrained_transform(
        gm, {"quant_method": "fp8", "weight_block_size": [128, 128]}
    )

    assert info.num_matches == 1
    assert transformed.proj.weight.dtype == torch.float8_e4m3fn


def test_generic_fp8_transform_skips_weight_block_size_configs() -> None:
    gm = _trace_with_weight_meta(_GenericLinearFixture().to(torch.bfloat16))
    transform = FP8LinearQuantizationFromConfig.from_kwargs(stage=Stages.PATTERN_MATCHER)

    transformed, info = transform._apply(
        gm,
        None,
        _DummyFactory({"quant_method": "fp8", "weight_block_size": [128, 128]}),
        SharedConfig(),
    )

    assert info.skipped
    assert transformed.proj.weight.dtype == torch.bfloat16


@_requires_e8m0
def test_trtllm_finegrained_fp8_linear_decodes_e8m0_for_fallback() -> None:
    input_tensor = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [[1.0, 2.0, -1.0, -2.0], [0.5, -0.5, 1.0, -1.0]],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    bias = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    weight_scale = _e8m0_from_bytes([127], (1, 1))

    actual = torch.ops.auto_deploy.trtllm_finegrained_fp8_linear.default(
        input_tensor,
        weight,
        bias,
        weight_scale,
    )

    expected_weight = weight.to(torch.bfloat16) * e8m0_to_fp32(weight_scale).to(torch.bfloat16)
    expected = F.linear(input_tensor, expected_weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
