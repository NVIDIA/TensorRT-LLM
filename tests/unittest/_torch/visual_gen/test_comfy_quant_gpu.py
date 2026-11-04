# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_quant import convert_comfy_nvfp4
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3.adaln_curves import interpolate_adaln_curve
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@skip_pre_blackwell
@pytest.mark.parametrize("rows,cols", [(128, 128), (192, 320)])
def test_comfy_nvfp4_linear_matches_independent_decode(rows: int, cols: int) -> None:
    """Exercise import, runtime scale interleave, dynamic activations, and GPU GEMM."""
    generator = torch.Generator().manual_seed(42)
    packed = torch.randint(0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8)
    # Powers of two keep source decoding exact, isolating layout from rounding.
    scales = (2.0 ** (torch.arange(rows * (cols // 16)) % 5 - 3)).reshape(rows, cols // 16)
    scale_bytes = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols // 16 + 3) // 4 * 4
    tiled = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8)
    # Scalar cuBLAS tile addresses are independent of the importer's reshape/permute.
    for row in range(rows):
        for col in range(cols // 16):
            tile = (row // 128) * (padded_cols // 4) + col // 4
            address = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
            tiled[address] = scale_bytes[row, col]
    source = {
        "proj.weight": packed,
        "proj.weight_scale": tiled.reshape(padded_rows, padded_cols).view(torch.float8_e4m3fn),
        "proj.weight_scale_2": torch.tensor(0.125),
        "proj.comfy_quant": torch.tensor(
            list(json.dumps({"format": "nvfp4"}).encode()), dtype=torch.uint8
        ),
    }
    normalized = convert_comfy_nvfp4(source)
    linear = Linear(
        in_features=cols,
        out_features=rows,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        force_dynamic_quantization=True,
        nvfp4_allowed_backends=["cutlass"],
    ).cuda()
    linear.load_weights([{key.removeprefix("proj."): value for key, value in normalized.items()}])
    linear.post_load_weights()

    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = packed.long()
    decoded = torch.stack((lut[codes // 16], lut[codes % 16]), dim=-1).reshape(rows, cols)
    decoded *= scales.repeat_interleave(16, dim=1) * 0.125
    # Probe every input column, plus reductions. 0 and +/-1 quantize exactly.
    dense = torch.randint(0, 2, (16, cols), generator=generator).float() * 2 - 1
    inputs = torch.cat((torch.eye(cols), dense)).to(device="cuda", dtype=torch.bfloat16)
    reference = (inputs.float() @ decoded.cuda().T).to(torch.bfloat16)
    with torch.inference_mode():
        actual = linear(inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.016, atol=0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H3 projection uses CUDA GEMM")
def test_h3_pruned_adaln_projection_preserves_curve_coordinates() -> None:
    """Pruned AdaLN uses FP32 linear coordinates, without the dense model's SiLU."""
    hidden_size, curve_dim = 32, 8
    config = DiffusionModelConfig(extra_attrs={"adaln_curve_grid": 3})
    modulation = MiniMaxH3AdaLayerNormModulation(curve_dim, hidden_size, config).cuda()
    weight = (torch.arange(18 * hidden_size * curve_dim, device="cuda").float() % 17 - 8).reshape(
        18 * hidden_size, curve_dim
    ) / 32
    bias = torch.arange(18 * hidden_size, device="cuda").float() % 7 / 32
    modulation.linear.load_weights([{"weight": weight, "bias": bias}])
    modulation.linear.post_load_weights()
    curve = torch.arange(3 * curve_dim, device="cuda").float().reshape(3, curve_dim) / 8 - 2
    coordinates = interpolate_adaln_curve(curve, torch.tensor([0.0, 0.25, 1.0], device="cuda"))
    reference = (coordinates @ weight.T + bias).reshape(-1, 6 * hidden_size).to(torch.bfloat16)
    with torch.inference_mode():
        actual = modulation.table(coordinates)
    assert modulation.linear.weight.dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="H3 projection uses CUDA GEMM")
def test_h3_pruned_final_modulation_preserves_fp32() -> None:
    """Comfy's pruned final shift/scale promote normalized hidden states to FP32."""
    hidden_size, curve_dim = 32, 8
    config = DiffusionModelConfig(extra_attrs={"adaln_curve_grid": 3})
    modulation = MiniMaxH3AdaLayerNormOut(hidden_size, curve_dim, 1e-5, config).cuda()
    weight = torch.zeros(2 * hidden_size, curve_dim, device="cuda")
    # These values lose information if the FP32 projection is cast to BF16.
    bias = (torch.arange(2 * hidden_size, device="cuda").float() + 1) / 1000
    modulation.linear.load_weights([{"weight": weight, "bias": bias}])
    modulation.linear.post_load_weights()
    hidden = torch.ones(1, 3, hidden_size, device="cuda", dtype=torch.bfloat16)
    coordinates = torch.zeros(2, curve_dim, device="cuda")
    indices = torch.tensor([0, 1, 0], device="cuda")
    with torch.inference_mode():
        normalized = modulation.norm(hidden.reshape(-1, hidden_size)).reshape_as(hidden)
        shift, scale = bias.chunk(2)
        reference = normalized.float() * (1 + scale) + shift
        actual = modulation(hidden, coordinates, indices)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
