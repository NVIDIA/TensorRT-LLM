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

import sys
from pathlib import Path

import pytest
import torch

_UTILS_DIR = Path(__file__).resolve().parents[3] / "_utils_test"
sys.path.append(str(_UTILS_DIR))
from _deepseek_v4_checkpoint_utils import (  # noqa: E402
    deepseek_v4_flash_checkpoint_or_skip,
    load_safetensors_tensors_or_skip,
)
from _torch_test_utils import fp8_compatible  # noqa: E402

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: E402, F401
import tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4  # noqa: E402, F401
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (  # noqa: E402
    FineGrainedFP8CheckpointLayout,
    QuantCheckpointLayoutRegistry,
)


def _finegrained_fp8_dense_dequant_ref(
    input_tensor,
    weight_fp8,
    weight_scale_inv,
    block_size,
    input_scale_fmt="",
):
    block_n, block_k = block_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    x_blocks = input_tensor.contiguous().view(-1, input_tensor.shape[-1] // block_k, block_k)
    input_amax = x_blocks.abs().float().amax(dim=-1)
    if input_scale_fmt.lower() == "ue8m0":
        input_scale = torch.clamp(input_amax, min=1e-4) / fp8_max
        input_scale = torch.pow(2.0, torch.ceil(torch.log2(input_scale)))
    else:
        input_scale = torch.clamp(input_amax / fp8_max, min=1e-12)
    qinput = (x_blocks.float() / input_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    input_dequant = (qinput.float() * input_scale.unsqueeze(-1)).view_as(input_tensor)

    weight_scale = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )
    weight_dequant = weight_fp8.float() * weight_scale
    return torch.nn.functional.linear(
        input_dequant.to(input_tensor.dtype),
        weight_dequant.to(input_tensor.dtype),
    )


def _deepseek_v4_finegrained_fp8_layout() -> FineGrainedFP8CheckpointLayout:
    checkpoint_layout = QuantCheckpointLayoutRegistry.build_from_config(
        {
            "model_type": "deepseek_v4",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        }
    )
    assert checkpoint_layout is not None
    assert isinstance(checkpoint_layout.finegrained_fp8, FineGrainedFP8CheckpointLayout)
    return checkpoint_layout.finegrained_fp8


def _finegrained_fp8_linear_op_or_skip():
    try:
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
    except (AttributeError, RuntimeError) as error:
        pytest.skip(f"torch_fake_quant_finegrained_fp8_linear is unavailable: {error}")


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 CUDA support")
def test_deepseek_v4_flash_real_wq_a_fp8_linear_matches_dense_dequant_ue8m0_reference():
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("Requires torch.float8_e4m3fn")
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    checkpoint_dir = deepseek_v4_flash_checkpoint_or_skip()
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    tensors = load_safetensors_tensors_or_skip(checkpoint_dir, (weight_name, scale_name))
    weight_fp8_cpu = tensors[weight_name]
    scale_cpu = tensors[scale_name]

    layout = _deepseek_v4_finegrained_fp8_layout()
    layout.validate_scale_shape(weight_name, weight_fp8_cpu.shape, scale_name, scale_cpu.shape)
    weight_scale_inv_cpu = layout.decode_scale(scale_cpu)

    block_size = layout.weight_block_size
    block_n, block_k = block_size
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    assert weight_fp8_cpu.dtype == fp8_dtype
    assert scale_cpu.dtype == torch.uint8 or (
        e8m0_dtype is not None and scale_cpu.dtype == e8m0_dtype
    )
    assert weight_fp8_cpu.shape[0] >= block_n
    assert weight_fp8_cpu.shape[1] >= block_k
    assert weight_scale_inv_cpu.shape[0] >= 1
    assert weight_scale_inv_cpu.shape[1] >= 1

    weight_fp8 = weight_fp8_cpu[:block_n, :block_k].contiguous().to("cuda")
    weight_scale_inv = weight_scale_inv_cpu[:1, :1].contiguous().to("cuda")
    input_tensor = torch.linspace(
        -0.75,
        0.95,
        steps=4 * block_k,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, block_k)
    input_tensor = input_tensor.to(torch.float16)

    output = _finegrained_fp8_linear_op_or_skip()(
        input_tensor,
        weight_fp8,
        None,
        [],
        [weight_scale_inv],
        [],
        [],
        input_scale_fmt="ue8m0",
    )
    ref = _finegrained_fp8_dense_dequant_ref(
        input_tensor,
        weight_fp8,
        weight_scale_inv,
        block_size,
        input_scale_fmt="ue8m0",
    )

    assert output.shape == ref.shape
    torch.testing.assert_close(output, ref, rtol=0.02, atol=1.0)
