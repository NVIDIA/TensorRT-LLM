# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""
Tests for FP8 fake quantized linear ONNX export.

This tests the _translate_fake_quant_fp8_linear_op function which expands
torch_fake_quant_fp8_linear into standard ONNX ops:
- QuantizeLinear + DequantizeLinear (input fake quantization)
- DequantizeLinear (weight dequantization)
- Transpose + MatMul + Add (linear computation)
"""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch

# Import to register the custom op
from tensorrt_llm._torch.auto_deploy.transform.library.export_to_onnx import (
    _translate_fake_quant_fp8_linear_op,
)

torch.manual_seed(0)


class FP8LinearModel(torch.nn.Module):
    """
    Simple model that uses torch_fake_quant_fp8_linear custom op.

    This model is designed to test the ONNX export of FP8 fake quantized linear.
    """

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        """Initialize FP8LinearModel with quantized weights and scales."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # FP8 quantized weight [out_features, in_features]
        # Initialize with randn, then convert to FP8
        weight_fp32 = torch.randn(out_features, in_features)
        self.register_buffer("weight_fp8", weight_fp32.to(torch.float8_e4m3fn))

        # Bias (optional)
        if use_bias:
            self.register_buffer("bias", torch.randn(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Scales for quantization (per-tensor)
        self.register_buffer("input_scale", torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("weight_scale", torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FP8 fake quantized linear operation."""
        return torch.ops.auto_deploy.torch_fake_quant_fp8_linear(
            x,
            self.weight_fp8,
            self.bias,
            input_scale=[self.input_scale],
            weight_scale=[self.weight_scale],
            input_zp=[],
            weight_zp=[],
        )


def _export_and_verify_onnx(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    use_bias: bool,
) -> None:
    """Helper function to export model to ONNX and verify the graph structure."""
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "model.onnx"

        # Build custom translation table
        custom_translation_table = {
            torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default: _translate_fake_quant_fp8_linear_op,
        }

        # Export to ONNX
        torch.onnx.export(
            model,
            (input_tensor,),
            str(output_path),
            opset_version=20,
            dynamo=True,
            custom_translation_table=custom_translation_table,
        )

        # Load and verify ONNX model
        assert output_path.exists(), f"ONNX file should exist at {output_path}"
        onnx_model = onnx.load(str(output_path))

        # Collect op types in the graph
        op_types = [node.op_type for node in onnx_model.graph.node]

        # Verify expected ops are present
        # 1. QuantizeLinear for input quantization
        assert "QuantizeLinear" in op_types, (
            f"QuantizeLinear should be in graph for input quantization, got ops: {op_types}"
        )

        # 2. DequantizeLinear for input and weight dequantization (at least 2)
        dequantize_count = op_types.count("DequantizeLinear")
        assert dequantize_count >= 2, (
            f"Expected at least 2 DequantizeLinear ops (input + weight), got {dequantize_count}"
        )

        # 3. Transpose for weight transposition
        assert "Transpose" in op_types, (
            f"Transpose should be in graph for weight transposition, got ops: {op_types}"
        )

        # 4. MatMul for linear computation
        assert "MatMul" in op_types, (
            f"MatMul should be in graph for linear computation, got ops: {op_types}"
        )

        # 5. Add for bias (only if use_bias is True)
        if use_bias:
            assert "Add" in op_types, (
                f"Add should be in graph for bias addition, got ops: {op_types}"
            )


@pytest.mark.parametrize(
    "in_features,out_features,batch_size,seq_len",
    [
        pytest.param(64, 128, 2, 16, id="small"),
        pytest.param(256, 512, 4, 32, id="medium"),
    ],
)
@torch.inference_mode()
def test_fp8_linear_with_bias(
    in_features: int,
    out_features: int,
    batch_size: int,
    seq_len: int,
):
    """Test FP8 linear ONNX export with bias."""
    model = FP8LinearModel(
        in_features=in_features,
        out_features=out_features,
        use_bias=True,
    ).to("cuda")

    input_tensor = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.float16)

    _export_and_verify_onnx(model, input_tensor, use_bias=True)


@pytest.mark.parametrize(
    "in_features,out_features,batch_size,seq_len",
    [
        pytest.param(64, 128, 2, 16, id="small"),
    ],
)
@torch.inference_mode()
def test_fp8_linear_without_bias(
    in_features: int,
    out_features: int,
    batch_size: int,
    seq_len: int,
):
    """Test FP8 linear ONNX export without bias."""
    model = FP8LinearModel(
        in_features=in_features,
        out_features=out_features,
        use_bias=False,
    ).to("cuda")

    input_tensor = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.float16)

    _export_and_verify_onnx(model, input_tensor, use_bias=False)
