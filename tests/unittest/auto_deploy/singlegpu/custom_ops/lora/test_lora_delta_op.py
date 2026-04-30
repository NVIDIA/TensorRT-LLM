# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for auto_deploy::lora_delta custom op.

Tests the op's behavior by checking:
- No LoRA → zero output
- With LoRA → non-zero output
- Different weights → different outputs
- Fake implementation returns correct shape
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.lora.lora_delta import (
    LoraModuleParams,
    _GlobalLoraPlanner,
)


@pytest.fixture(autouse=True)
def reset_planner():
    """Reset the global planner between tests."""
    _GlobalLoraPlanner.reset()
    yield
    _GlobalLoraPlanner.reset()


def _setup_planner_with_weights(num_seqs, layer_id, module_id, lora_A, lora_B, rank, max_rank):
    """Populate _GlobalLoraPlanner with given weights for all requests."""
    planner = _GlobalLoraPlanner.get()
    ranks = torch.full((num_seqs,), rank, dtype=torch.int32)
    pointers = torch.zeros(num_seqs, 3, dtype=torch.int64)
    for i in range(num_seqs):
        pointers[i, 0] = lora_A.data_ptr()
        pointers[i, 1] = lora_B.data_ptr()

    planner._params[(layer_id, module_id)] = LoraModuleParams(
        lora_ranks=ranks, lora_weight_pointers=pointers
    )
    planner._host_request_types = torch.ones(num_seqs, dtype=torch.int32)  # generation
    planner._prompt_lens_cpu = torch.ones(num_seqs, dtype=torch.int32)
    planner._max_rank = max_rank
    planner._num_seqs = num_seqs
    planner._active = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_lora_delta_no_lora_returns_zeros():
    """When planner has no LoRA for the target, output is zeros."""
    x = torch.randn(4, 64, dtype=torch.float16, device="cuda")
    linear_out = torch.randn(4, 64, dtype=torch.float16, device="cuda")
    result = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)
    assert result.shape == (4, 64)
    assert torch.all(result == 0), "Should return zeros when no LoRA is active"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_lora_delta_with_lora_returns_nonzero():
    """With LoRA weights set, output should be non-zero."""
    hidden, output_size, rank = 64, 64, 8
    x = torch.randn(1, hidden, dtype=torch.float16, device="cuda") * 0.1
    linear_out = torch.randn(1, output_size, dtype=torch.float16, device="cuda")

    lora_A = (torch.randn(hidden, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    lora_B = (torch.randn(output_size, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()

    _setup_planner_with_weights(
        1, layer_id=0, module_id=0, lora_A=lora_A, lora_B=lora_B, rank=rank, max_rank=rank
    )

    result = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)
    assert result.shape == (1, output_size)
    assert result.norm() > 0, "Should return non-zero when LoRA weights are set"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_lora_delta_different_weights_different_output():
    """Two different adapters should produce different outputs for the same input."""
    hidden, output_size, rank = 64, 64, 8
    x = torch.randn(1, hidden, dtype=torch.float16, device="cuda") * 0.1
    linear_out = torch.randn(1, output_size, dtype=torch.float16, device="cuda")

    A1 = (torch.randn(hidden, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    B1 = (torch.randn(output_size, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()

    _setup_planner_with_weights(
        1, layer_id=0, module_id=0, lora_A=A1, lora_B=B1, rank=rank, max_rank=rank
    )
    result1 = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)

    A2 = (torch.randn(hidden, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    B2 = (torch.randn(output_size, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()

    _GlobalLoraPlanner.reset()
    _setup_planner_with_weights(
        1, layer_id=0, module_id=0, lora_A=A2, lora_B=B2, rank=rank, max_rank=rank
    )
    result2 = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)

    assert not torch.allclose(result1, result2, atol=1e-3), (
        "Different weights should produce different outputs"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_lora_delta_deterministic():
    """Same input and weights should produce same output (deterministic)."""
    hidden, output_size, rank = 64, 64, 8
    x = torch.randn(1, hidden, dtype=torch.float16, device="cuda") * 0.1
    linear_out = torch.randn(1, output_size, dtype=torch.float16, device="cuda")

    lora_A = (torch.randn(hidden, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()
    lora_B = (torch.randn(output_size, rank, dtype=torch.float16, device="cuda") * 0.1).contiguous()

    _setup_planner_with_weights(
        1, layer_id=0, module_id=0, lora_A=lora_A, lora_B=lora_B, rank=rank, max_rank=rank
    )
    result1 = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)

    _GlobalLoraPlanner.reset()
    _setup_planner_with_weights(
        1, layer_id=0, module_id=0, lora_A=lora_A, lora_B=lora_B, rank=rank, max_rank=rank
    )
    result2 = torch.ops.auto_deploy.lora_delta(x, linear_out, 0, 0)

    torch.testing.assert_close(result1, result2)


def test_lora_delta_fake_returns_correct_shape():
    """register_fake returns correct shape matching linear_out."""
    x = torch.randn(4, 64, dtype=torch.float16, device="meta")
    linear_out = torch.randn(4, 128, dtype=torch.float16, device="meta")
    result = torch.ops.auto_deploy.lora_delta.default(x, linear_out, 0, 1)
    assert result.shape == (4, 128)
    assert result.dtype == torch.float16
