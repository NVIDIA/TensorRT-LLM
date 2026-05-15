# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm._torch.modules.rms_norm import RMSNorm, group_rms_norm


def _prepare_rms_test_data(batch_size, hidden_dims, eps, dtype, enable_weights):
    """Common setup for RMSNorm tests."""
    assert torch.cuda.is_available(), "This test requires CUDA"
    device = "cuda"

    # Create input tensors
    inputs = [
        torch.randn((batch_size, dim), dtype=dtype, device=device)
        for dim in hidden_dims
    ]

    # Create weights
    if enable_weights:
        weights = [
            torch.randn((dim), dtype=dtype, device=device)
            for dim in hidden_dims
        ]
    else:
        weights = [
            torch.ones((dim), dtype=dtype, device=device) for dim in hidden_dims
        ]

    # Generate reference outputs
    ref_outputs = []
    if enable_weights:
        for i, dim in enumerate(hidden_dims):
            ref_outputs.append(
                torch.ops.trtllm.flashinfer_rmsnorm(inputs[i], weights[i], eps))
    else:
        for i, dim in enumerate(hidden_dims):
            norm = RMSNorm(hidden_size=dim, eps=eps, dtype=dtype, device=device)
            ref_outputs.append(norm(inputs[i]))

    return inputs, weights, ref_outputs


def _verify_outputs(test_outputs, ref_outputs):
    """Verify that test outputs match reference outputs."""
    assert len(test_outputs) == len(ref_outputs), \
        f"Expected {len(ref_outputs)} outputs, got {len(test_outputs)}"

    # Verify each output matches reference
    for i, (test_out, ref_out) in enumerate(zip(test_outputs, ref_outputs)):
        torch.testing.assert_close(test_out, ref_out, rtol=1e-2, atol=1e-2)


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize("hidden_dims",
                         [[256], [8448], [256, 512], [8448, 1024]],
                         ids=lambda x: f"dims:{'-'.join(str(d) for d in x)}")
@pytest.mark.parametrize("eps", [1e-6, 1e-5], ids=lambda x: f"eps:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("enable_weights", [True, False],
                         ids=lambda x: f"enable_weights:{x}")
def test_group_rms_norm_heuristic(batch_size, hidden_dims, eps, dtype,
                                  enable_weights):
    """Compare group_rms_norm with RMSNorm."""
    inputs, weights, ref_outputs = _prepare_rms_test_data(
        batch_size, hidden_dims, eps, dtype, enable_weights)

    # Test tensorrt_llm._torch.modules.rms_norm.group_rms_norm
    if enable_weights:
        group_outputs_heuristic = group_rms_norm(inputs,
                                                 weights=weights,
                                                 eps=eps)
    else:
        group_outputs_heuristic = group_rms_norm(inputs, eps=eps)

    _verify_outputs(group_outputs_heuristic, ref_outputs)


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize("hidden_dims",
                         [[256], [8448], [256, 512], [8448, 1024]],
                         ids=lambda x: f"dims:{'-'.join(str(d) for d in x)}")
@pytest.mark.parametrize("eps", [1e-6, 1e-5], ids=lambda x: f"eps:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("enable_weights", [True, False],
                         ids=lambda x: f"enable_weights:{x}")
def test_group_rms_norm_base(batch_size, hidden_dims, eps, dtype,
                             enable_weights):
    """Compare group_rms_norm_base with RMSNorm."""
    inputs, weights, ref_outputs = _prepare_rms_test_data(
        batch_size, hidden_dims, eps, dtype, enable_weights)

    # Create output tensors
    group_outputs_base = [torch.empty_like(input) for input in inputs]

    # Test base implementation
    if enable_weights:
        torch.ops.trtllm.group_rms_norm_base(inputs,
                                             group_outputs_base,
                                             weights,
                                             eps=eps,
                                             weight_bias=0.0)
    else:
        torch.ops.trtllm.group_rms_norm_base(inputs,
                                             group_outputs_base, [],
                                             eps=eps,
                                             weight_bias=0.0)

    _verify_outputs(group_outputs_base, ref_outputs)


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize("hidden_dims", [[256, 512], [8448, 1024]],
                         ids=lambda x: f"dims:{'-'.join(str(d) for d in x)}")
@pytest.mark.parametrize("eps", [1e-6, 1e-5], ids=lambda x: f"eps:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("enable_weights", [True, False],
                         ids=lambda x: f"enable_weights:{x}")
def test_group_rms_norm_large_batch(batch_size, hidden_dims, eps, dtype,
                                    enable_weights):
    """Compare group_rms_norm_large_batch with RMSNorm."""
    inputs, weights, ref_outputs = _prepare_rms_test_data(
        batch_size, hidden_dims, eps, dtype, enable_weights)

    # Create output tensors
    group_outputs_large_batch = [torch.empty_like(input) for input in inputs]

    # Test large batch implementation
    if enable_weights:
        torch.ops.trtllm.group_rms_norm_large_batch(inputs,
                                                    group_outputs_large_batch,
                                                    weights,
                                                    eps=eps,
                                                    weight_bias=0.0)
    else:
        torch.ops.trtllm.group_rms_norm_large_batch(inputs,
                                                    group_outputs_large_batch,
                                                    [],
                                                    eps=eps,
                                                    weight_bias=0.0)

    _verify_outputs(group_outputs_large_batch, ref_outputs)
