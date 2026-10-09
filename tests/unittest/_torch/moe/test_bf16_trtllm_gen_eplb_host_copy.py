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
"""Online-EPLB host copies under BF16TRTLLMGenFusedMoEMethod.

The device slots are rewritten into shuffled BlockMajorK after loading, so the
host copies that EPLB migrates into those slots must be rewritten the same way.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
BLOCK_K = BF16TRTLLMGenFusedMoEMethod.block_k


def _expert_stacks(num_experts: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    w3_w1 = torch.randn(num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator).to(
        torch.bfloat16
    )
    w2 = torch.randn(num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator).to(
        torch.bfloat16
    )
    return w3_w1, w2


def _make_module(w3_w1: torch.Tensor, w2: torch.Tensor) -> nn.Module:
    """A MoE module with device slots and the host copies EPLB stages."""
    module = nn.Module()
    module.w3_w1_weight = nn.Parameter(w3_w1.clone(), requires_grad=False)
    module.w2_weight = nn.Parameter(w2.clone(), requires_grad=False)
    module.rebuild_tensor_metadata = {}
    module.is_gated_activation = True
    # Host copies are staged from the checkpoint in MajorK, like the slots.
    module.local_shared_w3_w1_tensors = w3_w1.clone()
    module.local_shared_w2_tensors = w2.clone()
    return module


class _NoAttributeAccess:
    """Fails if the helper reads the module before its empty-stack return."""

    def __getattr__(self, name):
        raise AssertionError(f"module.{name} accessed for an empty stack")


@pytest.mark.parametrize("empty", ["w3_w1", "w2", "both"])
def test_transform_returns_empty_stacks_unchanged(empty):
    w3_w1, w2 = _expert_stacks(NUM_EXPERTS)
    if empty in ("w3_w1", "both"):
        w3_w1 = w3_w1[:0]
    if empty in ("w2", "both"):
        w2 = w2[:0]

    out_w3_w1, out_w2 = BF16TRTLLMGenFusedMoEMethod()._transform_expert_stacks_for_trtllm_gen(
        _NoAttributeAccess(), w3_w1, w2
    )

    assert out_w3_w1 is w3_w1
    assert out_w2 is w2


def test_host_copies_match_transformed_slots():
    w3_w1, w2 = _expert_stacks(NUM_EXPERTS)
    module = _make_module(w3_w1, w2)
    method = BF16TRTLLMGenFusedMoEMethod()

    method.process_weights_after_loading(module)
    method._prepare_shared_weights_for_finalization(module)

    host_w3_w1 = module.local_shared_w3_w1_tensors
    host_w2 = module.local_shared_w2_tensors
    # Per expert: [K / block_k, M, block_k].
    assert host_w3_w1.shape == (NUM_EXPERTS, HIDDEN_SIZE // BLOCK_K, 2 * INTERMEDIATE_SIZE, BLOCK_K)
    assert host_w2.shape == (NUM_EXPERTS, INTERMEDIATE_SIZE // BLOCK_K, HIDDEN_SIZE, BLOCK_K)
    assert host_w3_w1.device == w3_w1.device
    assert host_w2.device == w2.device
    # A migration copies a host expert into a slot byte for byte, so the two
    # must hold the same layout.
    assert torch.equal(host_w3_w1, module.w3_w1_weight)
    assert torch.equal(host_w2, module.w2_weight)
    # And the layout did change from the staged MajorK copy.
    assert not torch.equal(host_w3_w1.flatten(), w3_w1.flatten())
    assert not torch.equal(host_w2.flatten(), w2.flatten())


@pytest.mark.parametrize("empty_slot", ["w3_w1", "w2"])
def test_host_copies_unchanged_when_a_slot_stack_is_empty(empty_slot):
    w3_w1, w2 = _expert_stacks(NUM_EXPERTS)
    module = _make_module(w3_w1, w2)
    empty_name = f"{empty_slot}_weight"
    setattr(
        module,
        empty_name,
        nn.Parameter(getattr(module, empty_name)[:0].clone(), requires_grad=False),
    )
    host_w3_w1 = module.local_shared_w3_w1_tensors
    host_w2 = module.local_shared_w2_tensors

    BF16TRTLLMGenFusedMoEMethod()._prepare_shared_weights_for_finalization(module)

    assert module.local_shared_w3_w1_tensors is host_w3_w1
    assert module.local_shared_w2_tensors is host_w2
    assert torch.equal(host_w3_w1, w3_w1)
    assert torch.equal(host_w2, w2)
