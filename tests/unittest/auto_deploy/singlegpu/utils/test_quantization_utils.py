# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.quant import FP8_MAX
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FP8LinearQuantizationFromConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import _shard_fp4_weight_scale
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
    ceil_pow2_scale,
    fake_fp4_act_quant,
    fake_fp8_act_quant,
    fp4_global_scale,
    hadamard_rotate,
    modelopt_fp4_scale_to_cutlass_fp4_scale,
)


@pytest.mark.parametrize("dim", [0, 1])
def test_fp4_scale_sharding(dim):
    weight = torch.rand(130, 64, dtype=torch.half, device="cuda")
    weight_scale_2 = fp4_global_scale(weight)

    weight_scale_modelopt = (
        torch.max(weight.reshape(weight.shape[0], -1, 16), dim=-1).values.to(torch.float)
        / (6.0 * weight_scale_2)
    ).to(torch.float8_e4m3fn)

    weight_scale_cutlass = modelopt_fp4_scale_to_cutlass_fp4_scale(weight_scale_modelopt)

    # Original uint8 weight shape (FP4 packs 2 elements per byte, so last dim is halved)
    original_uint8_weight_shape = (130, 32)

    if dim == 0:
        expected_sharded_weight_scale_shape = 128 * 4
    elif dim == 1:
        expected_sharded_weight_scale_shape = 256 * 4

    fp4_scale_rank_0 = _shard_fp4_weight_scale(
        weight_scale_cutlass,
        original_uint8_weight_shape,
        dim,
        0,
        world_size=2,
    )
    fp4_scale_rank_1 = _shard_fp4_weight_scale(
        weight_scale_cutlass,
        original_uint8_weight_shape,
        dim,
        1,
        world_size=2,
    )
    assert (
        tuple(fp4_scale_rank_0.shape)
        == tuple(fp4_scale_rank_1.shape)
        == (expected_sharded_weight_scale_shape,)
    )


def test_fp4_global_scale():
    input = torch.rand(3, 64, dtype=torch.half, device="cuda")
    input[-1][-1] = 448 * 6
    input_scale = fp4_global_scale(input)
    assert input_scale.dtype == torch.float
    assert input_scale == torch.tensor(1.0, dtype=torch.float)


@pytest.mark.parametrize("amax, expected_scale", [(FP8_MAX, 1.0), (FP8_MAX / 2.0, 0.5)])
def test_fp8_convert_amax_hook(amax, expected_scale):
    config = TransformConfig(stage="pattern_matcher")
    fp8_imp = FP8LinearQuantizationFromConfig(config)

    mock_state_dict = {"amax": amax}

    fp8_imp.convert_amax_hook(mock_state_dict, None, None, scale_name="scale", amax_name="amax")

    assert "scale" in mock_state_dict
    assert mock_state_dict["scale"] == expected_scale


def test_fp8_load_hook_maps_prequantized_scales():
    config = TransformConfig(stage="pattern_matcher")
    fp8_imp = FP8LinearQuantizationFromConfig(config)

    weight_name = "layer.proj.weight"
    mock_state_dict = {
        weight_name: torch.ones(4, 4, dtype=torch.float8_e4m3fn),
        "layer.proj.activation_scale": torch.tensor(0.125, dtype=torch.float32),
        "layer.proj.weight_scale_inv": torch.tensor(0.25, dtype=torch.float32),
    }

    fp8_imp.load_hook(mock_state_dict, None, None, weight_name=weight_name)

    assert mock_state_dict["layer.proj.input_scale"] == torch.tensor(0.125, dtype=torch.float32)
    assert mock_state_dict["layer.proj.weight_scale"] == torch.tensor(0.25, dtype=torch.float32)
    assert "layer.proj.activation_scale" not in mock_state_dict
    assert "layer.proj.weight_scale_inv" not in mock_state_dict


def test_fp8_load_hook_maps_prequantized_scales_with_prefix():
    config = TransformConfig(stage="pattern_matcher")
    fp8_imp = FP8LinearQuantizationFromConfig(config)

    weight_name = "layer.proj.weight"
    prefix = "nested."
    mock_state_dict = {
        prefix + weight_name: torch.ones(4, 4, dtype=torch.float8_e4m3fn),
        prefix + "layer.proj.activation_scale": torch.tensor(0.125, dtype=torch.float32),
        prefix + "layer.proj.weight_scale_inv": torch.tensor(0.25, dtype=torch.float32),
    }

    fp8_imp.load_hook(mock_state_dict, prefix, None, weight_name=weight_name)

    assert mock_state_dict[prefix + "layer.proj.input_scale"] == torch.tensor(
        0.125, dtype=torch.float32
    )
    assert mock_state_dict[prefix + "layer.proj.weight_scale"] == torch.tensor(
        0.25, dtype=torch.float32
    )
    assert prefix + "layer.proj.activation_scale" not in mock_state_dict
    assert prefix + "layer.proj.weight_scale_inv" not in mock_state_dict


def _ref_fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_fake_fp4_act_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    levels = normalized.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    level_idx = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    quant = levels[level_idx] * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim <= 1:
        return x
    if dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {dim}.")

    original_shape = x.shape
    out = x.float()
    width = 1
    while width < dim:
        out = out.reshape(*out.shape[:-1], dim // (2 * width), 2, width)
        left = out[..., 0, :]
        right = out[..., 1, :]
        out = torch.cat((left + right, left - right), dim=-1).reshape(original_shape)
        width *= 2
    return (out * (dim**-0.5)).to(x.dtype)


def test_fake_fp4_act_quant_exports_with_block_aligned_dim() -> None:
    class Wrapper(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return fake_fp4_act_quant(x, block_size=32)

    x = torch.tensor(
        [
            0.0,
            0.25,
            0.2501,
            0.75,
            0.7501,
            1.25,
            1.2501,
            1.75,
            1.7501,
            2.5,
            2.5001,
            3.5,
            3.5001,
            5.0,
            5.0001,
            6.0,
            -0.25,
            -0.2501,
            -0.75,
            -0.7501,
            -1.25,
            -1.2501,
            -1.75,
            -1.7501,
            -2.5,
            -2.5001,
            -3.5,
            -3.5001,
            -5.0,
            -5.0001,
            -6.0,
            4.0,
        ],
        dtype=torch.float32,
    ).reshape(1, 32)
    wrapper = Wrapper().eval()

    expected = wrapper(x)
    exported = torch.export.export(wrapper, (x,), strict=False).module()
    actual = exported(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(expected, _ref_fake_fp4_act_quant(x), rtol=0, atol=0)
    assert expected[0, 1] == 0.0
    assert expected[0, 3] == 0.5
    assert expected[0, 13] == 4.0
    assert expected[0, 17] == -0.5


@pytest.mark.parametrize(
    ("quant_fn", "block_size", "groups"),
    [
        pytest.param(fake_fp8_act_quant, 64, 2, id="fp8"),
        pytest.param(fake_fp4_act_quant, 32, 4, id="fp4"),
    ],
)
def test_fake_act_quant_export_keeps_prefix_inferred_for_sharding(
    quant_fn, block_size: int, groups: int
) -> None:
    class Wrapper(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return quant_fn(x, block_size=block_size)

    x = torch.randn(2, 3, 64, 128)
    gm = torch.export.export(Wrapper().eval(), (x,), strict=False).module()
    grouping_shapes = []
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.reshape.default:
            continue
        shape = node.args[1]
        if isinstance(shape, (list, tuple)) and list(shape[-2:]) == [groups, block_size]:
            grouping_shapes.append(shape)

    assert grouping_shapes
    assert all(list(shape) == [-1, groups, block_size] for shape in grouping_shapes)


@pytest.mark.parametrize("head_count", [1, 8, 64])
def test_hadamard_rotate_matches_reference_for_sharded_prefix_shapes(head_count: int) -> None:
    x = torch.randn(2, 3, head_count, 1024, dtype=torch.bfloat16)

    actual = hadamard_rotate(x)
    expected = _ref_hadamard_rotate(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hadamard_rotate_export_keeps_prefix_inferred_for_sharding() -> None:
    class Wrapper(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return hadamard_rotate(x)

    x = torch.randn(2, 3, 64, 128)
    gm = torch.export.export(Wrapper().eval(), (x,), strict=False).module()
    hadamard_reshape_shapes = []
    for node in gm.graph.nodes:
        if node.target != torch.ops.aten.reshape.default:
            continue
        shape = node.args[1]
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            hadamard_reshape_shapes.append(shape)

    assert hadamard_reshape_shapes
    assert all(shape[0] == -1 for shape in hadamard_reshape_shapes)
