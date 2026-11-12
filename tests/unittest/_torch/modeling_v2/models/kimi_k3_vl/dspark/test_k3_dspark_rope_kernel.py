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
"""The family copy of the DSpark RMSNorm + RoPE kernel (``dspark/kernels/dspark_rmsnorm_rope.py``) is
the shared kernel's source, and its operand contract answers as the shared custom op's does."""

import inspect

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.kernels import (
    dspark_rmsnorm_rope as family,
)
from tensorrt_llm._torch.attention.kernels.blackwell import dspark_rmsnorm_rope as shared_kernel
from tensorrt_llm._torch.custom_ops import dspark_rmsnorm_rope_custom_op as shared_op


def test_the_kernel_is_the_shared_kernel():
    assert inspect.getsource(family.DSparkRMSNormRoPEKernel) == inspect.getsource(
        shared_kernel.DSparkRMSNormRoPEKernel
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the contract takes CUDA tensors")
def test_the_operand_contract_is_the_shared_ops():
    dev = "cuda"
    freqs = torch.zeros(4, 32, 2, dtype=torch.float32, device=dev)
    rows = torch.zeros(4, 64, 192, dtype=torch.bfloat16, device=dev)
    latent = torch.zeros(4, 576, dtype=torch.bfloat16, device=dev)
    cases = [
        (rows, None, freqs, 64, 64, None),
        (rows[..., 128:], None, freqs, 64, 64, None),
        (latent, torch.ones(512, dtype=torch.bfloat16, device=dev), freqs, 1, 64, 512),
        (latent, torch.ones(576, dtype=torch.bfloat16, device=dev), freqs, 1, 64, None),
        (latent, torch.ones(500, dtype=torch.bfloat16, device=dev), freqs, 1, 64, 512),
        (latent.float(), None, freqs, 1, 64, None),
        (latent, None, freqs[:2], 1, 64, None),
        (latent, None, freqs, 3, 64, None),
        (latent[:, :570], None, freqs, 1, 64, None),
    ]
    for x, w, f, heads, rope, norm in cases:
        assert family.is_fused_dspark_rmsnorm_rope_supported(
            x, w, f, heads, rope, norm
        ) == shared_op.is_fused_dspark_rmsnorm_rope_supported(x, w, f, heads, rope, norm)
