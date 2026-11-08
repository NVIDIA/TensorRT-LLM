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
"""GPU vs CPU golden tests for trtllm weight-only quant utility ops.

These ops used to be CPU-only; the CUDA branch must produce bit-identical
results to the original CPU reference implementation in weightOnlyQuantOp.cpp.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm torch ops)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU vs CPU comparison"
)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 16),
        (4, 32),
        (1, 1),
        (3, 5, 7),
        (2, 3, 4, 9),
    ],
)
def test_unpack_int4_packed_tensor_to_int8_gpu_matches_cpu(shape):
    op = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8

    packed = torch.randint(-128, 128, shape, dtype=torch.int8)

    cpu_out = op(packed)
    gpu_out = op(packed.cuda())

    assert gpu_out.is_cuda
    assert gpu_out.dtype == torch.int8
    # Output last dim is doubled, everything else identical.
    assert list(gpu_out.shape[:-1]) == list(packed.shape[:-1])
    assert gpu_out.shape[-1] == packed.shape[-1] * 2

    # Integer unpack must be exactly equal.
    assert torch.equal(gpu_out.cpu(), cpu_out)


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize(
    "n, k",
    [
        (8, 256),
        (4, 512),
        (1, 128),
        (16, 1024),
    ],
)
def test_mxfp4_dequantize_unswizzled_gpu_matches_cpu(n, k, group_size):
    op = torch.ops.trtllm.mxfp4_dequantize_unswizzled

    # weight: (n, k/2), two packed e2m1 codes per byte -> any uint8 value is valid.
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)

    # scale: (n, k/group_size), interpreted as e8m0 (2^(e-127)).
    # Restrict the exponent to a stable finite range so neither CPU nor GPU
    # produces NaN (0xFF) or Inf (very large exponents); see plan doc.
    scale = torch.randint(118, 135, (n, k // group_size), dtype=torch.uint8)

    cpu_out = op(weight, scale, group_size)
    gpu_out = op(weight.cuda(), scale.cuda(), group_size)

    assert gpu_out.is_cuda
    assert gpu_out.dtype == torch.float32
    assert list(gpu_out.shape) == [n, k]

    # LUT value * power-of-two scale, same decode on both sides -> bit exact.
    torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=0, atol=0)


def test_unpack_int4_non_contiguous_raises():
    op = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    packed = torch.randint(-128, 128, (8, 16), dtype=torch.int8).cuda()
    non_contiguous = packed.t()
    assert not non_contiguous.is_contiguous()
    with pytest.raises(RuntimeError):
        op(non_contiguous)


def test_mxfp4_dequantize_device_mismatch_raises():
    op = torch.ops.trtllm.mxfp4_dequantize_unswizzled
    n, k, group_size = 4, 256, 32
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).cuda()
    scale = torch.randint(118, 135, (n, k // group_size), dtype=torch.uint8)  # CPU
    with pytest.raises(RuntimeError):
        op(weight, scale, group_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
