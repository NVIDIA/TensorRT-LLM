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

from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@torch.inference_mode()
@pytest.mark.parametrize("batch_size", [1, 4, 16], ids=lambda x: f"batch:{x}")
@pytest.mark.parametrize("hidden_dims", [[128], [128, 256]],
                         ids=lambda x: f"dims:{len(x)}")
@pytest.mark.parametrize("eps", [1e-6, 1e-5], ids=lambda x: f"eps:{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=["fp16", "bf16"])
def test_group_rms_norm(batch_size, hidden_dims, eps, dtype):
    """Test group_rms_norm against individual RMSNorm modules."""
    assert torch.cuda.is_available(), "This test requires CUDA"
    device = "cuda"

    # Create input tensors
    inputs = [
        torch.randn((batch_size, dim), dtype=dtype, device=device)
        for dim in hidden_dims
    ]

    # Individual RMSNorm modules for reference
    ref_outputs = []
    for i, dim in enumerate(hidden_dims):
        norm = RMSNorm(hidden_size=dim, eps=eps, dtype=dtype, device=device)
        ref_outputs.append(norm(inputs[i].clone()))

    # Test group_rms_norm
    # Apply GroupRMSNorm to all inputs
    group_outputs = torch.ops.trtllm.group_rms_norm(inputs, None, eps, False)

    # Verify same number of outputs
    assert len(group_outputs) == len(ref_outputs), \
        f"Expected {len(ref_outputs)} outputs, got {len(group_outputs)}"

    # Verify each output matches reference
    for i, (group_out, ref_out) in enumerate(zip(group_outputs, ref_outputs)):
        torch.testing.assert_close(group_out,
                                   ref_out,
                                   rtol=1e-4,
                                   atol=1e-4,
                                   msg=f"Output mismatch at index {i}")
