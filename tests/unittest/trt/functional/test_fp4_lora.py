# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for LoRA support in FP4Linear and FP4RowLinear.

These tests verify:
  - FP4Linear + LoRA produces W*x + B*A*x (OOTB fallback path).
  - FP4RowLinear + LoRA produces W*x + B*A*x (OOTB fallback path).
  - FP4Linear raises RuntimeError when the input is a pre-quantized tuple and
    lora_runtime_params is not None.
  - FP4RowLinear raises RuntimeError when gemm_allreduce_plugin is enabled and
    lora_runtime_params is not None.
"""

import unittest

import tensorrt as trt
import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from utils.util import create_session, run_session, skip_pre_blackwell_unittest

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.layers.lora import Lora, LoraRuntimeParams
from tensorrt_llm.quantization.layers import FP4Linear, FP4RowLinear

BLOCK_SIZE = 16


def _quantize_fp4(tensor: torch.Tensor):
    """Return (NVFP4QTensor, block_sf, global_sf) for *tensor*."""
    quantized, block_sf, global_sf = NVFP4QTensor.quantize(tensor, block_size=BLOCK_SIZE)
    return quantized, block_sf, global_sf


def _build_lora_inputs(batch_size: int, in_size: int, out_size: int, lora_rank: int, dtype: str):
    """Return (lora_A, lora_B, weights_ptrs tensor, ranks tensor)."""
    torch_dtype = tensorrt_llm.str_dtype_to_torch(dtype)
    lora_A = torch.randn(lora_rank, in_size, dtype=torch_dtype, device="cuda") * 0.01
    lora_B = torch.randn(out_size, lora_rank, dtype=torch_dtype, device="cuda") * 0.01
    # TRT-LLM LoRA plugin expects [in_ptr, out_ptr, dora_scale_ptr] per request.
    lora_weights_pointers = torch.tensor(
        [[lora_A.data_ptr(), lora_B.data_ptr(), 0] for _ in range(batch_size)], dtype=torch.int64
    )
    lora_ranks = torch.tensor([lora_rank] * batch_size, dtype=torch.int32)
    return lora_A, lora_B, lora_weights_pointers, lora_ranks


class TestFP4LinearLora(unittest.TestCase):
    """Tests for LoRA support in FP4Linear and FP4RowLinear."""

    def setUp(self):
        """Suppress TRT-LLM log noise during tests."""
        tensorrt_llm.logger.set_level("error")

    @skip_pre_blackwell_unittest
    def test_fp4_linear_lora_forward(self):
        """FP4Linear OOTB path: LoRA delta must equal lora_B * lora_A * x.

        FP4 double-quantization (weights + activations) introduces noise that
        makes it difficult to compare the full output against a pure FP16
        reference.  Instead we verify the *LoRA delta*: the difference between
        the output with LoRA active and the output with LoRA disabled
        (lora_rank=0).  The delta is computed entirely in FP16 so it is
        accurate to within floating-point rounding.
        """
        dtype = "float16"
        torch_dtype = torch.float16
        batch_size, seq_len, in_size, out_size, lora_rank = 2, 16, 512, 256, 8

        x = torch.randn(batch_size, seq_len, in_size, dtype=torch_dtype, device="cuda") * 0.1
        weight_raw = torch.randn(out_size, in_size, dtype=torch_dtype, device="cuda") * 0.1
        bias_raw = torch.randn(out_size, dtype=torch_dtype, device="cpu")

        weight_q, weight_block_sf, weight_global_sf = _quantize_fp4(weight_raw)
        _, _act_block_sf, act_global_sf = _quantize_fp4(x)

        lora_A, lora_B, lora_weights_ptrs, lora_ranks = _build_lora_inputs(
            batch_size, in_size, out_size, lora_rank, dtype
        )

        # Build TRT network.
        linear = FP4Linear(in_size, out_size, dtype=dtype, bias=True)
        linear.weight.value = weight_q._quantized_data
        linear.weights_block_scaling_factor.value = weight_block_sf
        linear.weights_global_scaling_factor.value = weight_global_sf
        linear.activation_global_scaling_factor.value = act_global_sf
        linear.bias.value = bias_raw.numpy()
        linear.lora = Lora(
            in_hidden_size=in_size, out_hidden_sizes=[out_size], max_low_rank=lora_rank
        )

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.lora_plugin = dtype
        # Use padded 3-D input layout (batch, seq, hidden); remove_input_padding
        # requires host_context_lengths which is not needed for this test.
        network.plugin_config.remove_input_padding = False

        with tensorrt_llm.net_guard(network):
            inp = Tensor(
                name="input",
                shape=(batch_size, seq_len, in_size),
                dtype=tensorrt_llm.str_dtype_to_trt(dtype),
            )
            host_request_types = Tensor(
                name="host_request_types", shape=(batch_size,), dtype=trt.int32
            )
            lora_weights_pointers_tensor = Tensor(
                name="lora_weights_pointers", shape=(batch_size, 3), dtype=trt.int64
            )
            lora_ranks_tensor = Tensor(name="lora_ranks", shape=(batch_size,), dtype=trt.int32)
            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks_tensor],
                lora_weights_pointers=[lora_weights_pointers_tensor],
                host_request_types=host_request_types,
                weight_index=0,
            )

            out = linear(inp, lora_runtime_params=lora_params)
            out.mark_output("output", dtype)

        session = create_session(builder, network, precision=dtype)
        base_inputs = {
            "input": x,
            "host_request_types": torch.zeros(batch_size, dtype=torch.int32),
            "lora_weights_pointers": lora_weights_ptrs,
            "lora_ranks": torch.zeros(batch_size, dtype=torch.int32),  # rank=0 → no LoRA
        }
        lora_inputs = {**base_inputs, "lora_ranks": lora_ranks}

        out_base = run_session(session, base_inputs)["output"]
        out_lora = run_session(session, lora_inputs)["output"]
        torch.cuda.synchronize()

        # The LoRA delta is computed in FP16 (original x, before FP4 quantization),
        # so it should be accurate to within FP16 rounding.
        trt_delta = (out_lora - out_base).float()
        expected_delta = (x @ lora_A.T @ lora_B.T).float()
        torch.testing.assert_close(trt_delta, expected_delta, atol=2e-3, rtol=1e-2)

    @skip_pre_blackwell_unittest
    def test_fp4_row_linear_lora_forward(self):
        """FP4RowLinear OOTB path (tp_size=1): LoRA delta must equal lora_B * lora_A * x.

        Same delta-based verification strategy as test_fp4_linear_lora_forward:
        compare (output with LoRA) - (output without LoRA) against the expected
        FP16 LoRA contribution to avoid conflating FP4 quantization noise with
        LoRA injection correctness.
        """
        dtype = "float16"
        torch_dtype = torch.float16
        batch_size, seq_len, in_size, out_size, lora_rank = 2, 16, 512, 256, 8

        x = torch.randn(batch_size, seq_len, in_size, dtype=torch_dtype, device="cuda") * 0.1
        weight_raw = torch.randn(out_size, in_size, dtype=torch_dtype, device="cuda") * 0.1
        bias_raw = torch.randn(out_size, dtype=torch_dtype, device="cpu")

        weight_q, weight_block_sf, weight_global_sf = _quantize_fp4(weight_raw)
        _, _, act_global_sf = _quantize_fp4(x)

        lora_A, lora_B, lora_weights_ptrs, lora_ranks = _build_lora_inputs(
            batch_size, in_size, out_size, lora_rank, dtype
        )

        linear = FP4RowLinear(in_size, out_size, dtype=dtype, bias=True)
        linear.weight.value = weight_q._quantized_data
        linear.weights_block_scaling_factor.value = weight_block_sf
        linear.weights_global_scaling_factor.value = weight_global_sf
        linear.activation_global_scaling_factor.value = act_global_sf
        linear.bias.value = bias_raw.numpy()
        linear.lora = Lora(
            in_hidden_size=in_size, out_hidden_sizes=[out_size], max_low_rank=lora_rank
        )

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.lora_plugin = dtype
        # Use padded 3-D input layout; remove_input_padding requires
        # host_context_lengths which is not needed for this test.
        network.plugin_config.remove_input_padding = False

        with tensorrt_llm.net_guard(network):
            inp = Tensor(
                name="input",
                shape=(batch_size, seq_len, in_size),
                dtype=tensorrt_llm.str_dtype_to_trt(dtype),
            )
            host_request_types = Tensor(
                name="host_request_types", shape=(batch_size,), dtype=trt.int32
            )
            lora_weights_pointers_tensor = Tensor(
                name="lora_weights_pointers", shape=(batch_size, 3), dtype=trt.int64
            )
            lora_ranks_tensor = Tensor(name="lora_ranks", shape=(batch_size,), dtype=trt.int32)
            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks_tensor],
                lora_weights_pointers=[lora_weights_pointers_tensor],
                host_request_types=host_request_types,
                weight_index=0,
            )

            out = linear(inp, lora_runtime_params=lora_params)
            out.mark_output("output", dtype)

        session = create_session(builder, network, precision=dtype)
        base_inputs = {
            "input": x,
            "host_request_types": torch.zeros(batch_size, dtype=torch.int32),
            "lora_weights_pointers": lora_weights_ptrs,
            "lora_ranks": torch.zeros(batch_size, dtype=torch.int32),  # rank=0 → no LoRA
        }
        lora_inputs = {**base_inputs, "lora_ranks": lora_ranks}

        out_base = run_session(session, base_inputs)["output"]
        out_lora = run_session(session, lora_inputs)["output"]
        torch.cuda.synchronize()

        trt_delta = (out_lora - out_base).float()
        expected_delta = (x @ lora_A.T @ lora_B.T).float()
        torch.testing.assert_close(trt_delta, expected_delta, atol=2e-3, rtol=1e-2)

    def test_fp4_linear_lora_tuple_input_raises(self):
        """FP4Linear must raise RuntimeError when input is a pre-quantized tuple.

        Specifically when lora_runtime_params is also provided.
        """
        dtype = "float16"
        in_size, out_size = 512, 256

        linear = FP4Linear(in_size, out_size, dtype=dtype, bias=False)

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.lora_plugin = dtype

        with tensorrt_llm.net_guard(network):
            fp4_x = Tensor(name="fp4_x", shape=(1, in_size // 2), dtype=trt.int8)  # packed fp4
            scale = Tensor(name="scale", shape=(1, in_size // BLOCK_SIZE), dtype=trt.fp8)
            host_request_types = Tensor(name="host_request_types", shape=(1,), dtype=trt.int32)
            lora_weights_pointers_tensor = Tensor(
                name="lora_weights_pointers", shape=(1, 3), dtype=trt.int64
            )
            lora_ranks_tensor = Tensor(name="lora_ranks", shape=(1,), dtype=trt.int32)
            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks_tensor],
                lora_weights_pointers=[lora_weights_pointers_tensor],
                host_request_types=host_request_types,
                weight_index=0,
            )

            with self.assertRaises(RuntimeError) as ctx:
                linear((fp4_x, scale), lora_runtime_params=lora_params)

        self.assertIn("pre-quantized", str(ctx.exception))

    def test_fp4_row_linear_lora_gemm_allreduce_raises(self):
        """FP4RowLinear must raise RuntimeError when gemm_allreduce_plugin is enabled.

        Specifically when lora_runtime_params is also provided.
        """
        dtype = "float16"
        in_size, out_size = 512, 256

        linear = FP4RowLinear(in_size, out_size, dtype=dtype, bias=False)

        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        network.plugin_config.lora_plugin = dtype
        network.plugin_config.gemm_allreduce_plugin = dtype  # enable fused path

        with tensorrt_llm.net_guard(network):
            inp = Tensor(
                name="input", shape=(1, in_size), dtype=tensorrt_llm.str_dtype_to_trt(dtype)
            )
            host_request_types = Tensor(name="host_request_types", shape=(1,), dtype=trt.int32)
            lora_weights_pointers_tensor = Tensor(
                name="lora_weights_pointers", shape=(1, 3), dtype=trt.int64
            )
            lora_ranks_tensor = Tensor(name="lora_ranks", shape=(1,), dtype=trt.int32)
            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks_tensor],
                lora_weights_pointers=[lora_weights_pointers_tensor],
                host_request_types=host_request_types,
                weight_index=0,
            )

            with self.assertRaises(RuntimeError) as ctx:
                linear(inp, lora_runtime_params=lora_params)

        self.assertIn("gemm_allreduce_plugin", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
