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
import time

import pytest
import torch

from tensorrt_llm._torch.modules.rms_norm import GroupRMSNorm, RMSNorm


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize("hidden_dims",
                         [[256], [8448], [256, 512], [256, 512, 7680]],
                         ids=lambda x: f"dims:{'-'.join(str(d) for d in x)}")
@pytest.mark.parametrize("eps", [1e-6, 1e-5], ids=lambda x: f"eps:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
@pytest.mark.parametrize("enable_weights", [False, True],
                         ids=lambda x: f"enable_weights:{x}")
def test_group_rms_norm(batch_size, hidden_dims, eps, dtype, enable_weights):
    """Test group_rms_norm against individual RMSNorm modules."""
    assert torch.cuda.is_available(), "This test requires CUDA"
    device = "cuda"

    # Create input tensors
    inputs = [
        torch.randn((batch_size, dim), dtype=dtype, device=device)
        for dim in hidden_dims
    ]
    weights = [
        torch.ones((dim), dtype=dtype, device=device) for dim in hidden_dims
    ]

    # Individual RMSNorm modules for reference
    ref_outputs = []
    for i, dim in enumerate(hidden_dims):
        norm = RMSNorm(hidden_size=dim, eps=eps, dtype=dtype, device=device)
        ref_outputs.append(norm(inputs[i]))

    weight_bias = 0.0
    # Test torch.ops.trtllm.group_rms_norm
    group_outputs = torch.ops.trtllm.group_rms_norm(inputs, weights, eps,
                                                    weight_bias, enable_weights)

    # Test tensorrt_llm._torch.modules.rms_norm.GroupRMSNorm
    group_rms_norm_op = GroupRMSNorm(hidden_sizes=hidden_dims,
                                     eps=eps,
                                     dtype=dtype,
                                     device=device,
                                     enable_weights=enable_weights)
    if not enable_weights:
        group_outputs_op = group_rms_norm_op(inputs)
    else:
        group_outputs_op = group_rms_norm_op(inputs, weights)

    # Verify same number of outputs
    assert len(group_outputs) == len(ref_outputs), \
        f"Expected {len(ref_outputs)} outputs, got {len(group_outputs)}"
    assert len(group_outputs_op) == len(ref_outputs), \
        f"Expected {len(ref_outputs)} outputs, got {len(group_outputs_op)}"

    # Verify each output matches reference
    for i, (group_out, ref_out) in enumerate(zip(group_outputs, ref_outputs)):
        print(f"Checking output{i}")
        torch.testing.assert_close(group_out, ref_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(group_outputs_op[i],
                                   ref_out,
                                   rtol=1e-3,
                                   atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 128], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize(
    "hidden_dims",
    [
        # Default hidden dims for DSV3
        [1536, 512],
        # Default hidden dims for LLAMA4
        [5120, 1024]
    ],
    ids=lambda x: f"dims:{'-'.join(str(d) for d in x)}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
def test_group_rms_norm_benchmark(batch_size,
                                  hidden_dims,
                                  dtype,
                                  eps=1e-5,
                                  enable_weights=False,
                                  warmup_runs=10,
                                  timing_runs=50):
    """Benchmark group_rms_norm against individual RMSNorm modules."""
    assert torch.cuda.is_available(), "This test requires CUDA"
    device = "cuda"

    # Create input tensors
    inputs = [
        torch.randn((batch_size, dim), dtype=dtype, device=device)
        for dim in hidden_dims
    ]

    ref_norms = [
        RMSNorm(hidden_size=dim, eps=eps, dtype=dtype, device=device)
        for dim in hidden_dims
    ]
    group_norms = GroupRMSNorm(hidden_sizes=hidden_dims,
                               eps=eps,
                               dtype=dtype,
                               device=device,
                               enable_weights=enable_weights)

    # Benchmark RMSNorm as reference
    # Warmup
    for _ in range(warmup_runs):
        ref_outputs = [norm(inputs[i]) for i, norm in enumerate(ref_norms)]

    # Timing
    start_time = time.time()
    for _ in range(timing_runs):
        ref_outputs = [norm(inputs[i]) for i, norm in enumerate(ref_norms)]
    end_time = time.time()
    ref_time = end_time - start_time
    print(f"RMSNorm time: {ref_time / timing_runs * 1000} ms")

    # Benchmark GroupRMSNorm
    # Warmup
    for _ in range(warmup_runs):
        group_outputs = group_norms(inputs)
    # Timing
    start_time = time.time()
    for _ in range(timing_runs):
        group_outputs = group_norms(inputs)
    end_time = time.time()
    group_time = end_time - start_time
    print(f"GroupRMSNorm time: {group_time / timing_runs * 1000} ms")

    for i, (group_out, ref_out) in enumerate(zip(group_outputs, ref_outputs)):
        print(f"Checking output{i}")
        torch.testing.assert_close(group_out, ref_out, rtol=1e-3, atol=1e-3)
