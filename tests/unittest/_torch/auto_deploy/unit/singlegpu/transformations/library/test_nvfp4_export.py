# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for NVFP4 linear ONNX export.

Tests the _translate_fake_quant_nvfp4_linear_op path used during export_to_onnx:
torch_fake_quant_nvfp4_linear is expanded to ONNX subgraph (TRT_FP4DynamicQuantize,
two-stage DequantizeLinear, MatMul, optional Add). Includes the same
_convert_nvfp4_weight_initializers_to_float4e2m1 step as production.
"""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp4_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.library import _onnx_schemas
from tensorrt_llm._torch.auto_deploy.transform.library.export_to_onnx import (
    _convert_nvfp4_weight_initializers_to_float4e2m1,
    _translate_fake_quant_nvfp4_linear_op,
)
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

# Register ONNX custom ops once so parametrized tests do not re-register.
_onnx_schemas.register_onnx_schemas()

torch.manual_seed(0)


class TinyFP4Ref(nn.Module):
    """Minimal module that uses torch_fake_quant_nvfp4_linear (same as test_quant_fusion.TinyFP4Ref)."""

    def __init__(self, in_features=64, out_features=32, use_bias=True):
        super().__init__()
        assert in_features % 16 == 0
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.half, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.half, device=device))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            s_in2 = fp4_global_scale(torch.rand(1, in_features, dtype=torch.half, device=device))
            s_w2 = fp4_global_scale(self.weight)
            w_fp4, cutlass_vec = torch.ops.trtllm.fp4_quantize(self.weight, s_w2, 16, False)
            alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)
        self.register_buffer("input_scale_2", s_in2.to(torch.float32))
        self.register_buffer("weight_scale_cutlass", cutlass_vec)
        self.register_buffer("alpha", alpha.to(torch.float32))

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.weight_fp4,
            bias,
            [self.input_scale_2],
            [self.weight_scale_cutlass, self.alpha],
            [],
            [],
        )


def _export_nvfp4_to_onnx_and_verify_structure(
    model: nn.Module,
    x: torch.Tensor,
    use_bias: bool,
) -> None:
    """Export model to ONNX (with converter), then verify graph structure."""
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "model.onnx"
        external_location = "model.onnx.data"

        custom_translation_table = {
            torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear.default: _translate_fake_quant_nvfp4_linear_op,
        }

        onnx_program = torch.onnx.export(
            model,
            (x,),
            None,
            opset_version=21,
            dynamo=True,
            custom_translation_table=custom_translation_table,
        )
        model_proto = onnx_program.model_proto
        _convert_nvfp4_weight_initializers_to_float4e2m1(model_proto)

        onnx.save_model(
            model_proto,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_location,
            size_threshold=1024,
        )

        assert output_path.exists(), f"ONNX file should exist at {output_path}"
        onnx_model = onnx.load(str(output_path))

        op_types = [node.op_type for node in onnx_model.graph.node]

        assert "TRT_FP4DynamicQuantize" in op_types, (
            f"TRT_FP4DynamicQuantize should be in graph, got ops: {op_types}"
        )
        assert "DequantizeLinear" in op_types, (
            f"DequantizeLinear should be in graph, got ops: {op_types}"
        )
        assert "MatMul" in op_types, (
            f"MatMul should be in graph for linear computation, got ops: {op_types}"
        )
        if use_bias:
            assert "Add" in op_types, (
                f"Add should be in graph for bias addition, got ops: {op_types}"
            )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
@torch.inference_mode()
def test_nvfp4_linear_export_to_onnx_structure(use_bias):
    """NVFP4 linear ONNX export expands to TRT_FP4DynamicQuantize, DequantizeLinear, MatMul, optional Add."""
    model = TinyFP4Ref(in_features=64, out_features=32, use_bias=use_bias).to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    _export_nvfp4_to_onnx_and_verify_structure(model, x, use_bias)
