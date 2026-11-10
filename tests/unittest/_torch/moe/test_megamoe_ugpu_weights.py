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
"""CPU byte-layout, VMM transaction, and lifetime contracts for MegaMoE."""

import gc

import pytest
import torch
from _torch.moe.megamoe_ugpu_test_utils import MockVmm, packed_weights

from tensorrt_llm._torch.moe.fused_moe.mega_moe import ugpu_weights as _MODULE


@pytest.mark.parametrize("shape", [(3, 512, 64), (5, 65536), (4, 1024, 512), (12, 40960)])
@pytest.mark.parametrize("granularity", [4096, 2 * 1024 * 1024])
def test_die_major_halves_have_exact_mapping_boundary_and_roundtrip_bytes(shape, granularity):
    source = torch.randint(
        0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(6)
    )
    plan = _MODULE.plan_packed_weight_layout(shape, mode="n_split", granularity=granularity)
    destination = torch.empty(plan.target_shape, dtype=torch.uint8)
    _MODULE.reorder_packed_weight(source, destination, plan)
    experts, rows = shape[:2]
    lower, upper = destination[:experts], destination[experts:]
    torch.testing.assert_close(lower, source[:, : rows // 2], rtol=0, atol=0)
    torch.testing.assert_close(upper, source[:, rows // 2 :], rtol=0, atol=0)
    torch.testing.assert_close(torch.cat((lower, upper), dim=1), source, rtol=0, atol=0)
    assert plan.tensor_offset + plan.payload_bytes // 2 == plan.die1_offset
    assert plan.die1_offset % granularity == 0
    assert plan.tensor_offset % 128 == 0
    assert plan.boundary_spill_bytes == 0
    assert plan.reserved_bytes - plan.payload_bytes < 2 * granularity
    if source.ndim == 3:
        transposed = destination.transpose(1, 2)
        assert transposed.stride() == (shape[1] // 2 * shape[2], 1, shape[2])


def test_expert_half_preserves_standard_layout_and_reports_partial_boundary_page():
    shape = (6, 256, 48)
    plan = _MODULE.plan_packed_weight_layout(shape, mode="expert_half", granularity=16384)
    source = torch.randint(0, 256, shape, dtype=torch.uint8)
    destination = torch.empty_like(source)
    _MODULE.reorder_packed_weight(source, destination, plan)
    torch.testing.assert_close(source, destination, rtol=0, atol=0)
    assert plan.target_shape == shape
    assert plan.tensor_offset == 0
    assert plan.boundary_spill_bytes == 4096
    assert plan.die1_offset == 32768
    assert plan.spans == ((0, 0, 32768), (1, 32768, 49152))


@pytest.mark.parametrize(
    "shape,mode",
    [((3, 256, 64), "expert_half"), ((4, 255, 64), "n_split"), ((2, 512, 64), "bad")],
)
def test_layout_rejects_unsafe_shapes_before_cuda(shape, mode):
    with pytest.raises(ValueError):
        _MODULE.plan_packed_weight_layout(shape, mode=mode, granularity=4096)


@pytest.mark.parametrize("mode", ["expert_half", "n_split"])
def test_four_tensor_placement_preserves_bytes_and_explicit_close_is_idempotent(mode):
    source = packed_weights()
    before = tuple(t.clone() for t in source)
    driver = MockVmm()
    owner = _MODULE._place_weights(source, mode=mode, driver=driver)
    tensors = (owner.fc1_weight, owner.fc1_weight_sf, owner.fc2_weight, owner.fc2_weight_sf)
    plans = [
        _MODULE.plan_packed_weight_layout(tuple(t.shape), mode=mode, granularity=driver.gran)
        for t in source
    ]
    for original, old_copy, placed, plan in zip(source, before, tensors, plans):
        expected = torch.empty(plan.target_shape, dtype=torch.uint8)
        _MODULE.reorder_packed_weight(original, expected, plan)
        torch.testing.assert_close(placed, expected, rtol=0, atol=0)
        torch.testing.assert_close(original, old_copy, rtol=0, atol=0)
        assert original.untyped_storage().data_ptr() != placed.untyped_storage().data_ptr()
    assert driver.counts["locality"] == sum(p.reserved_bytes // p.granularity for p in plans)
    owner.close()
    assert owner.closed
    driver.assert_empty()
    assert driver.calls.index("synchronize") < driver.calls.index("unmap")
    previous_calls = driver.calls[:]
    owner.close()
    assert driver.calls == previous_calls
    with pytest.raises(RuntimeError, match="not live"):
        _ = owner.fc1_weight


@pytest.mark.parametrize("operation,index", [("create", 2), ("alias", 4)])
def test_failed_allocation_unwinds_this_and_all_prior_tensors(operation, index):
    driver = MockVmm(fail_operation=operation, fail_index=index)
    with pytest.raises(RuntimeError, match=f"injected {operation}"):
        _MODULE._place_weights(packed_weights(), mode="expert_half", driver=driver)
    driver.assert_empty()
    if operation == "alias":
        # Earlier tensors were already copied: finish that work before unmapping.
        assert driver.calls.index("synchronize") < driver.calls.index("unmap")


def test_wrong_locality_is_a_failure_not_an_unverified_success():
    driver = MockVmm()
    driver.locality = lambda address: 1
    with pytest.raises(RuntimeError, match="expected die 0, got 1"):
        _MODULE._place_weights(packed_weights(), mode="expert_half", driver=driver)
    driver.assert_empty()
    assert "alias" not in driver.calls


def test_discarding_owner_cannot_unmap_a_tensor_alias_or_captured_address():
    driver = MockVmm()
    owner = _MODULE._place_weights(packed_weights(), mode="expert_half", driver=driver)
    alias = owner.fc1_weight.view(-1)
    expected = alias.clone()
    with pytest.warns(ResourceWarning, match="graph/alias lifetime is unknown"):
        del owner
        gc.collect()
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)
    assert "unmap" not in driver.calls
    assert driver.addresses and driver.handles and driver.mappings


def test_reorder_rejects_aliases_noncontiguous_and_dtype_mismatches():
    source = torch.zeros((4, 512, 64), dtype=torch.uint8)
    plan = _MODULE.plan_packed_weight_layout(source.shape, mode="n_split", granularity=4096)
    with pytest.raises(ValueError, match="distinct destination"):
        _MODULE.reorder_packed_weight(source, source.view(plan.target_shape), plan)
    with pytest.raises(ValueError, match="source"):
        _MODULE.reorder_packed_weight(
            source.float(), torch.empty(plan.target_shape, dtype=torch.uint8), plan
        )
    with pytest.raises(ValueError, match="destination"):
        _MODULE.reorder_packed_weight(
            source, torch.empty((8, 64, 256), dtype=torch.uint8).transpose(1, 2), plan
        )
