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
"""CPU assembly tests use canonical lender records and the backend contract."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation._assembly import ProfileSpec, RegionNamespace, project_view
from tensorrt_llm._torch.disaggregation.base.backend import CacheKind, Outcome, SlotRegion, View
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LayerGroupBlocks,
    LeaseView,
    MemoryRegion,
    StagingOptions,
)

pytestmark = pytest.mark.cpu_only


def _profile() -> ProfileSpec:
    """Build the initial compatibility profile without activating any provider."""
    return ProfileSpec(
        id="reference",
        lifecycle_profile_id="test-profile",
        scope=b"model-revision-and-all-byte-affecting-options",
        model_layout="mha-trtllm-hnd",
        dtype="bfloat16",
        topology=(1, 1, 1, 1),
        state_kinds=(CacheKind.PAGED,),
        full_attention=True,
        pool_count=1,
        speculative_mode="none",
        model_role="target",
        joint_reuse=False,
        staging=StagingOptions(96),
    )


def _blocks(group: int, region: int, addresses: list[int]) -> LayerGroupBlocks:
    """Create real lender rows with distinct opaque keys and nonconsecutive ordinals.

    Args:
        group: Original layer-group identifier.
        region: Original region index.
        addresses: Addresses in the requested row order.

    Returns:
        Canonical immutable lender rows.
    """
    count = len(addresses)
    return LayerGroupBlocks(
        layer_group_index=group,
        block_ordinals=np.arange(count, dtype=np.int64) * 3 + 2,
        object_keys=np.arange(count * 54, dtype=np.uint8).reshape(count, 54),
        addresses=np.asarray(addresses, dtype=np.int64),
        region_index=region,
    )


def test_projection_preserves_region_group_and_row_identity() -> None:
    """Projection preserves ordering and shares opaque arrays without flattening."""
    regions = (
        MemoryRegion("layout-a", 0x1000, 256, 64, 4, (9,)),
        MemoryRegion("layout-b", 0x2000, 512, 128, 4, (3,)),
    )
    lease_view = LeaseView((_blocks(3, 1, [0x2180, 0x2000]), _blocks(9, 0, [0x1080])))
    view = project_view(lease_view, RegionNamespace(regions, "HOST"))
    assert view is lease_view
    assert isinstance(view, View)
    assert all(isinstance(region, SlotRegion) for region in regions)
    assert [group.layer_group_index for group in view.groups] == [3, 9]
    assert [group.region_index for group in view.groups] == [1, 0]
    for group, original in zip(view.groups, lease_view.groups):
        assert group.object_keys is original.object_keys
        assert group.addresses is original.addresses
        assert group is original
        assert not group.object_keys.flags.writeable
        assert not group.addresses.flags.writeable
    masks = view.row_masks(True)
    assert [mask.tolist() for mask in masks] == [[True, True], [True]]
    assert all(mask.flags.writeable and mask.dtype == np.bool_ for mask in masks)
    masks[0][0] = False
    assert view.row_masks(True)[0].all()


def test_outcome_masks_are_already_in_lender_row_order() -> None:
    """Canonical masks map directly, including partial delivery alongside a fault."""
    regions = (MemoryRegion("layout", 0x1000, 256, 64, 4, (7, 12)),)
    lease_view = LeaseView((_blocks(12, 0, [0x1080, 0x1000]), _blocks(7, 0, [])))
    view = project_view(lease_view, RegionNamespace(regions, "HOST"))
    assert view is lease_view
    masks = (np.array([False, True]), np.array([], dtype=bool))
    outcome = Outcome.for_view(view, masks, fault="one source row failed")
    outcome.check_mask_shapes(view)
    assert [mask.tolist() for mask in outcome.delivered_masks] == [[False, True], []]
    assert outcome.fault == "one source row failed"
    assert [mask.shape for mask in outcome.delivered_masks] == [
        mask.shape for mask in lease_view.row_masks()
    ]
    masks[0][:] = True
    assert outcome.delivered_masks[0].tolist() == [False, True]


@pytest.mark.parametrize(
    "masks",
    [(), (np.array([True]),), (np.array([[True, False]]),)],
    ids=["missing-group", "wrong-row-count", "wrong-dimension"],
)
def test_canonical_outcome_rejects_invalid_mask_shapes(masks: tuple[np.ndarray, ...]) -> None:
    """The existing Outcome checker validates the exact submitted projection."""
    region = MemoryRegion("layout", 0x1000, 128, 64, 2, (7,))
    view = project_view(
        LeaseView((_blocks(7, 0, [0x1000, 0x1040]),)), RegionNamespace((region,), "HOST")
    )
    with pytest.raises(ValueError):
        Outcome(masks).check_mask_shapes(view)


def test_empty_view_preserves_empty_masks() -> None:
    """A ready lease with no rows remains a valid empty canonical view."""
    view = project_view(LeaseView(()), RegionNamespace((), "HOST"))
    assert view.groups == ()
    assert view.row_masks() == ()
    Outcome.for_view(view).check_mask_shapes(view)


@pytest.mark.parametrize(
    "address, message",
    [(0x0FC0, "outside"), (0x1100, "outside"), (0x1001, "aligned")],
    ids=["below-region", "padding-is-not-a-slot", "misaligned-slot"],
)
def test_projection_rejects_invalid_slot_addresses(address: int, message: str) -> None:
    """Padding bytes never increase the number of valid slots."""
    region = MemoryRegion("layout", 0x1000, 320, 64, 4, (7,))
    with pytest.raises(ValueError, match=message):
        project_view(LeaseView((_blocks(7, 0, [address]),)), RegionNamespace((region,), "HOST"))


@pytest.mark.parametrize(
    "region, message",
    [
        (MemoryRegion("layout", 0x1000, 63, 64, 1, (7,)), "contain"),
        (MemoryRegion("layout", 0x1000, 64, 0, 1, (7,)), "positive"),
        (MemoryRegion("layout", True, 64, 64, 1, (7,)), "positive"),
        (MemoryRegion("layout", 0x1000, 64, 64, 0, (7,)), "positive"),
        (MemoryRegion("layout", 2**63 - 1, 64, 64, 1, (7,)), "int64"),
    ],
    ids=["truncated-slot", "zero-slot-bytes", "boolean-address", "zero-slots", "overflow"],
)
def test_projection_rejects_invalid_region_geometry(region: MemoryRegion, message: str) -> None:
    """Bad geometry is refused before a backend can access a projected view."""
    with pytest.raises(ValueError, match=message):
        RegionNamespace((region,), "HOST")


def test_projection_rejects_overlapping_regions() -> None:
    """Two region indices must not alias the same staging storage."""
    regions = (
        MemoryRegion("layout-a", 0x1000, 128, 64, 2, (3,)),
        MemoryRegion("layout-b", 0x1040, 128, 64, 2, (9,)),
    )
    with pytest.raises(ValueError, match="overlap"):
        RegionNamespace(regions, "HOST")


@pytest.mark.parametrize(
    "group, index, message",
    [(7, 1, "namespace"), (9, 0, "layer group")],
    ids=["out-of-range-region", "wrong-layer-group"],
)
def test_projection_rejects_wrong_region_namespace(group: int, index: int, message: str) -> None:
    """Region indices and group membership remain lender-defined."""
    region = MemoryRegion("layout", 0x1000, 128, 64, 2, (7,))
    with pytest.raises(ValueError, match=message):
        project_view(
            LeaseView((_blocks(group, index, [0x1000]),)), RegionNamespace((region,), "HOST")
        )


def test_projection_rejects_unplaced_rows() -> None:
    """Keyless/addressless views require a future capability agreement."""
    view = LeaseView((LayerGroupBlocks(7, np.array([0])),))
    with pytest.raises(ValueError, match="requires keys"):
        project_view(view, RegionNamespace((), "HOST"))


def test_projection_rejects_writable_lender_arrays() -> None:
    """A provider cannot receive arrays that violate the lender's immutability contract."""
    blocks = _blocks(7, 0, [0x1000])
    blocks.object_keys.flags.writeable = True
    region = MemoryRegion("layout", 0x1000, 64, 64, 1, (7,))
    with pytest.raises(ValueError, match="read-only"):
        project_view(LeaseView((blocks,)), RegionNamespace((region,), "HOST"))


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("id", "", ValueError),
        ("scope", "model", TypeError),
        ("scope", b"", ValueError),
        ("scope", bytes(65536), ValueError),
        ("staging", object(), TypeError),
        ("lifecycle_profile_id", "", ValueError),
        ("model_layout", "mha-token-major", ValueError),
        ("topology", (2, 1, 1, 1), ValueError),
        ("topology", (1, 2, 1, 1), ValueError),
        ("topology", (1, 1, 2, 1), ValueError),
        ("topology", (1, 1, 1, 2), ValueError),
        ("topology", (True, 1, 1, 1), TypeError),
        ("state_kinds", (CacheKind.PAGED, CacheKind.STATE), ValueError),
        ("state_kinds", ("attention",), TypeError),
        ("dtype", "float16", ValueError),
        ("full_attention", False, ValueError),
        ("pool_count", 2, ValueError),
        ("pool_count", True, TypeError),
        ("speculative_mode", "eagle", ValueError),
        ("model_role", "draft", ValueError),
        ("joint_reuse", True, ValueError),
    ],
    ids=[
        "missing-id",
        "scope-type",
        "empty-scope",
        "oversize-scope",
        "staging-type",
        "missing-lifecycle-profile",
        "writer-layout",
        "tp",
        "dp",
        "pp",
        "cp",
        "boolean-rank",
        "hybrid-state",
        "noncanonical-kind",
        "dtype",
        "window",
        "multiple-pools",
        "boolean-pool-count",
        "speculative-mode",
        "draft-role",
        "joint-reuse",
    ],
)
def test_profile_rejects_unsupported_inputs(
    field: str, value: object, error: type[Exception]
) -> None:
    """No backend name can bypass the initial compatibility and containment gates."""
    with pytest.raises(error):
        replace(_profile(), **{field: value})


def test_profile_preserves_scope_and_opaque_provider_config() -> None:
    """Byte identity is immutable and provider configuration remains opaque to RI."""
    config = object()
    profile = replace(_profile(), backend_config=config)
    assert profile.backend_config is config
    assert profile.scope == b"model-revision-and-all-byte-affecting-options"
    with pytest.raises(FrozenInstanceError):
        profile.scope = b"changed-weights"


@pytest.mark.parametrize("profile_id", ["qualified-lease", "qualified-binding", "executor-fatal"])
def test_profile_does_not_infer_containment_from_host_memory(profile_id: str) -> None:
    """Lifecycle selection remains an explicit assembly fact, not a HOST heuristic."""
    assert replace(_profile(), lifecycle_profile_id=profile_id).lifecycle_profile_id == profile_id


def test_namespace_requires_explicit_known_memory_kind() -> None:
    """The baseline refuses unknown/device metadata rather than registering by guess."""
    with pytest.raises(ValueError, match="memory-kind"):
        RegionNamespace((), "DEVICE")


def test_namespace_rejects_mutable_group_membership() -> None:
    """Descriptor facts cannot change after static assembly checks."""
    region = MemoryRegion("layout", 0x1000, 64, 64, 1, [7])
    with pytest.raises(ValueError, match="immutable tuple"):
        RegionNamespace((region,), "HOST")
