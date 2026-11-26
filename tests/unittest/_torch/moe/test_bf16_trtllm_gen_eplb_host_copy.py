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
The row order depends on the activation: a gated one reorders the w3/w1 rows for
its fused GEMM, a non-gated one (e.g. Relu2) does not.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.moe.fused_moe.quantization import (
    BF16TRTLLMGenFusedMoEMethod,
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 2
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
BLOCK_K = BF16TRTLLMGenFusedMoEMethod.block_k
# The epilogue tile BF16TRTLLMGenFusedMoEMethod derives its row shuffle with.
EPILOGUE_TILE_M = 128


def _expert_stacks(num_experts: int, seed: int = 0, gated: bool = True):
    """Expert stacks in the checkpoint's MajorK: w3/w1 ``[E, (2 if gated else 1) * I, H]``
    and w2 ``[E, H, I]``."""
    generator = torch.Generator().manual_seed(seed)
    fc1_rows = (2 if gated else 1) * INTERMEDIATE_SIZE
    w3_w1 = torch.randn(num_experts, fc1_rows, HIDDEN_SIZE, generator=generator).to(torch.bfloat16)
    w2 = torch.randn(num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator).to(
        torch.bfloat16
    )
    return w3_w1, w2


def _make_module(w3_w1: torch.Tensor, w2: torch.Tensor, gated: bool = True) -> nn.Module:
    """A MoE module with device slots and the host copies EPLB stages."""
    module = nn.Module()
    module.w3_w1_weight = nn.Parameter(w3_w1.clone(), requires_grad=False)
    module.w2_weight = nn.Parameter(w2.clone(), requires_grad=False)
    module.rebuild_tensor_metadata = {}
    module.is_gated_activation = gated
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


def test_slots_and_host_copies_are_converted_once():
    w3_w1, w2 = _expert_stacks(NUM_EXPERTS)
    module = _make_module(w3_w1, w2)
    method = BF16TRTLLMGenFusedMoEMethod()

    method.process_weights_after_loading(module)
    slots_w3_w1, slots_w2 = module.w3_w1_weight, module.w2_weight
    snapshot_w3_w1, snapshot_w2 = slots_w3_w1.detach().clone(), slots_w2.detach().clone()
    # The slots are BlockMajorK now, so a second call leaves them alone.
    method.process_weights_after_loading(module)
    assert module.w3_w1_weight is slots_w3_w1
    assert module.w2_weight is slots_w2
    assert torch.equal(module.w3_w1_weight, snapshot_w3_w1)
    assert torch.equal(module.w2_weight, snapshot_w2)

    method._prepare_shared_weights_for_finalization(module)
    host_w3_w1 = module.local_shared_w3_w1_tensors
    host_w2 = module.local_shared_w2_tensors
    assert torch.equal(host_w3_w1, slots_w3_w1)
    assert torch.equal(host_w2, slots_w2)
    # Host copies that are BlockMajorK already are kept as they are.
    method._prepare_shared_weights_for_finalization(module)
    assert module.local_shared_w3_w1_tensors is host_w3_w1
    assert module.local_shared_w2_tensors is host_w2


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
    method = BF16TRTLLMGenFusedMoEMethod()

    # The device transform skips the empty layer, so its slots stay MajorK.
    method.process_weights_after_loading(module)
    method._prepare_shared_weights_for_finalization(module)

    assert module.local_shared_w3_w1_tensors is host_w3_w1
    assert module.local_shared_w2_tensors is host_w2
    assert torch.equal(host_w3_w1, w3_w1)
    assert torch.equal(host_w2, w2)


def _reference_kernel_layout(weight: torch.Tensor, permute_indices: torch.Tensor) -> torch.Tensor:
    """One expert ``[M, K]`` in shuffled BlockMajorK ``[K / block_k, M, block_k]``.

    Written out here rather than taken from BF16TRTLLMGenFusedMoEMethod, so a
    wrong row order or block split in the method shows up against it.
    """
    m, k = weight.shape
    return weight[permute_indices].reshape(m, k // BLOCK_K, BLOCK_K).permute(1, 0, 2).contiguous()


def _reference_w3_w1(expert: torch.Tensor, gated: bool) -> torch.Tensor:
    # A fresh cache: the method's cache is shared by its whole class, so an entry
    # that another layer or test left there must not decide the reference.
    permute_indices = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        expert.view(torch.uint8), {}, epilogue_tile_m=EPILOGUE_TILE_M, is_gated_act_gemm=gated
    )
    return _reference_kernel_layout(expert, permute_indices)


def _reference_w2(expert: torch.Tensor) -> torch.Tensor:
    permute_indices = trtllmgen_maybe_get_cached_w2_permute_indices(
        expert.view(torch.uint8), {}, epilogue_tile_m=EPILOGUE_TILE_M
    )
    return _reference_kernel_layout(expert, permute_indices)


@pytest.mark.parametrize("gated", [True, False], ids=["gated", "non_gated"])
def test_slots_and_host_copies_match_a_per_expert_reference(gated: bool) -> None:
    """Slots and host copies hold every expert in the kernel layout of its activation.

    Equality between slots and host copies cannot show this on its own, because
    both go through the same conversion.
    """
    w3_w1, w2 = _expert_stacks(NUM_EXPERTS, gated=gated)
    module = _make_module(w3_w1, w2, gated=gated)
    method = BF16TRTLLMGenFusedMoEMethod()

    method.process_weights_after_loading(module)
    method._prepare_shared_weights_for_finalization(module)

    for expert_id in range(NUM_EXPERTS):
        expected_w3_w1 = _reference_w3_w1(w3_w1[expert_id], gated)
        expected_w2 = _reference_w2(w2[expert_id])
        assert torch.equal(module.w3_w1_weight[expert_id], expected_w3_w1), expert_id
        assert torch.equal(module.local_shared_w3_w1_tensors[expert_id], expected_w3_w1), expert_id
        assert torch.equal(module.w2_weight[expert_id], expected_w2), expert_id
        assert torch.equal(module.local_shared_w2_tensors[expert_id], expected_w2), expert_id


def test_w3_w1_row_order_is_cached_per_gated_flag() -> None:
    """A gated and a non-gated layer with one w3/w1 shape each get their own row order.

    The permute cache is one dict for the whole class, so its key must carry the
    gated flag; without it the second layer would get the rows of the first.
    """
    method = BF16TRTLLMGenFusedMoEMethod()
    expert = torch.zeros(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)

    gated = method._get_w3_w1_permute_indices(expert, is_gated_act_gemm=True)
    non_gated = method._get_w3_w1_permute_indices(expert, is_gated_act_gemm=False)

    for is_gated, got in ((True, gated), (False, non_gated)):
        expected = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            expert.view(torch.uint8),
            {},
            epilogue_tile_m=EPILOGUE_TILE_M,
            is_gated_act_gemm=is_gated,
        )
        assert torch.equal(got.cpu(), expected.cpu()), is_gated
    assert not torch.equal(gated.cpu(), non_gated.cpu())
