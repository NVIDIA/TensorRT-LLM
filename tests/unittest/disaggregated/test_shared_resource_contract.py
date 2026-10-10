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
"""CPU contracts for actual lender-to-backend mapping and profile checks."""

from dataclasses import replace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.resource.shared import (
    SHARED_CONTRACT_REVISION,
    STAGING_EXTENT_NAMESPACE,
    SharedRuntimeProfile,
    build_extent,
    served_masks,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import GroupRun, Part, RegionView

pytestmark = pytest.mark.cpu_only

NAME_A = bytes(range(54))
NAME_B = b"\xff\x00" * 27
NAME_C = b"\x80" * 54
PART = Part(name="same-layout", address=4096, nbytes=1024, slot_bytes=128, slots=8)


def _view(
    *,
    names: tuple[bytes, ...] = (NAME_A, NAME_B),
    addresses: tuple[int, ...] = (4480, 4736),
    ordinals: tuple[int, ...] = (40, 41),
    part: int = 0,
    layer_group: int = 7,
) -> RegionView:
    """Construct real public lender values with independently chosen coordinates.

    Args:
        names: Opaque content names, including zero and non-ASCII bytes.
        addresses: Host addresses of staging slots.
        ordinals: Logical block positions, deliberately unlike slot positions.
        part: Index of the containing registered region.
        layer_group: Lender-local layer group index.

    Returns:
        A public staging view for the mapping functions under test.
    """
    return RegionView(
        (
            GroupRun(
                layer_group=layer_group,
                ordinals=np.array(ordinals),
                names=np.array([list(name) for name in names], dtype=np.uint8),
                addresses=np.array(addresses),
                part=part,
            ),
        )
    )


def _profile() -> SharedRuntimeProfile:
    """Return explicit first-profile assembly facts with a distinct backend revision.

    Returns:
        A compatibility profile, which is not a deployment qualification record.
    """
    return SharedRuntimeProfile(
        contract_revision=SHARED_CONTRACT_REVISION,
        backend_revision="native-nixl-test-build-42",
        backend_kind="native_nixl",
        manager_kind="KVCacheManagerV2",
        cache_dtype="bfloat16",
        attention_backend="TRTLLM",
        attention_kind="mha",
        layout="HND",
        parallelism=(1, 1, 1, 1),
        staging="manager_host",
        committed_whole_blocks=True,
        extra_features=frozenset(),
    )


def test_golden_names_and_slot_coordinates() -> None:
    """Preserve opaque bytes and map addresses, not logical ordinals, into slots."""
    extent = build_extent(_view(), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True)

    assert extent.name == b"trtllm:shared-kv:staging:1"
    assert extent.is_last is True
    assert [(unit.name, unit.local_group, unit.local) for unit in extent.units] == [
        (bytes(range(54)), 7, 3),
        (b"\xff\x00" * 27, 7, 5),
    ]


def test_relocation_changes_slots_without_renaming_content() -> None:
    """Content identity is independent of allocation address and staging position."""
    before = build_extent(_view(), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=False)
    relocated = replace(PART, address=8192)
    after = build_extent(
        _view(addresses=(8704, 8960)),
        (relocated,),
        name=STAGING_EXTENT_NAMESPACE,
        is_last=False,
    )

    assert [unit.name for unit in before.units] == [unit.name for unit in after.units]
    assert [unit.local for unit in after.units] == [4, 6]
    assert before.name == after.name
    assert PART.name == relocated.name


def test_multiple_groups_keep_local_group_identity() -> None:
    """Local group numbers survive mapping even when groups share a part."""
    first = _view()
    second = _view(names=(NAME_C,), addresses=(4992,), ordinals=(9,), layer_group=2)
    extent = build_extent(
        RegionView(first.runs + second.runs), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True
    )

    assert [(unit.local_group, unit.local) for unit in extent.units] == [(7, 3), (7, 5), (2, 7)]


@pytest.mark.parametrize(
    ("address", "error"),
    [(3968, "outside"), (4481, "misaligned"), (5120, "capacity")],
    ids=["before-part", "unaligned", "past-final-slot"],
)
def test_invalid_row_address_is_rejected(address: int, error: str) -> None:
    """Do not expose a row whose complete bytes are outside a staging slot.

    Args:
        address: Invalid host row address.
        error: Expected diagnostic category.
    """
    with pytest.raises(ValueError, match=error):
        build_extent(
            _view(addresses=(address, 4736)),
            (PART,),
            name=STAGING_EXTENT_NAMESPACE,
            is_last=True,
        )


@pytest.mark.parametrize(
    "part",
    [
        replace(PART, address=0),
        replace(PART, slot_bytes=0),
        replace(PART, slots=True),
        replace(PART, nbytes=1023),
    ],
    ids=["null-region", "zero-width", "boolean-capacity", "incomplete-last-slot"],
)
def test_malformed_part_is_rejected(part: Part) -> None:
    """Reject malformed public allocation descriptions before using addresses.

    Args:
        part: Invalid public region metadata.
    """
    with pytest.raises(ValueError, match="staging part"):
        build_extent(_view(), (part,), name=STAGING_EXTENT_NAMESPACE, is_last=True)


def test_overlapping_parts_are_rejected() -> None:
    """Different region indices cannot disguise aliased physical storage."""
    with pytest.raises(ValueError, match="overlap"):
        build_extent(
            _view(),
            (PART, replace(PART, address=4608)),
            name=STAGING_EXTENT_NAMESPACE,
            is_last=True,
        )


def test_unknown_part_is_rejected() -> None:
    """A run must address a region belonging to the same lender."""
    with pytest.raises(ValueError, match="unknown part"):
        build_extent(_view(part=1), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True)


def test_in_place_view_is_rejected() -> None:
    """Device-page views without staging metadata are outside this adapter."""
    view = RegionView((GroupRun(layer_group=0, ordinals=np.array([1])),))
    with pytest.raises(ValueError, match="staging names"):
        build_extent(view, (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True)


def test_negative_ordinal_is_rejected() -> None:
    """Only actual block rows can be mapped to whole-unit transfers."""
    with pytest.raises(ValueError, match="nonnegative"):
        build_extent(_view(ordinals=(-1, 41)), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True)


def test_duplicate_physical_coordinate_across_groups_is_rejected() -> None:
    """Distinct content names must not alias one physical destination slot."""
    view = RegionView(
        _view().runs + _view(names=(NAME_C,), addresses=(4480,), ordinals=(9,), layer_group=2).runs
    )
    with pytest.raises(ValueError, match="duplicate physical coordinate"):
        build_extent(view, (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True)


def test_duplicate_content_name_uses_canonical_extent_validation() -> None:
    """One backend extent cannot ambiguously address the same content twice."""
    with pytest.raises(ValueError, match="share a name"):
        build_extent(
            _view(names=(NAME_A, NAME_A)), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=True
        )


@pytest.mark.parametrize(
    ("served", "expected"),
    [
        (frozenset(), [False, False]),
        (frozenset({NAME_B}), [False, True]),
        (frozenset({NAME_A, NAME_B}), [True, True]),
    ],
    ids=["miss", "partial-with-prefix-hole", "all-units"],
)
def test_served_masks_preserve_whole_rows(served: frozenset[bytes], expected: list[bool]) -> None:
    """Map delivery exactly, without treating the highest delivered block as ready.

    Args:
        served: Whole-unit backend result.
        expected: Expected arrival mask in original lender row order.
    """
    view = _view()
    masks = served_masks(view, served)

    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], expected)
    assert masks[0].dtype == np.bool_
    assert masks[0].flags.writeable
    np.testing.assert_array_equal(view.runs[0].ordinals, [40, 41])


def test_unknown_served_name_is_rejected() -> None:
    """A backend cannot claim delivery of a unit absent from the submitted view."""
    with pytest.raises(ValueError, match="subset"):
        served_masks(_view(), frozenset({NAME_C}))


def test_duplicate_view_names_cannot_mark_multiple_rows() -> None:
    """Reject ambiguous arrival mapping even when called without extent construction."""
    with pytest.raises(ValueError, match="duplicate names"):
        served_masks(_view(names=(NAME_A, NAME_A)), frozenset({NAME_A}))


def test_empty_view_and_miss_are_valid() -> None:
    """Empty extents have no addresses to expose and no delivered rows."""
    view = RegionView(())
    assert build_extent(view, (), name=STAGING_EXTENT_NAMESPACE, is_last=True).units == ()
    assert served_masks(view, frozenset()) == ()


def test_supported_profile_and_backend_revision_are_distinct() -> None:
    """Accept explicit compatible facts without equating code and contract revisions."""
    profile = _profile()
    profile.validate()
    assert profile.backend_revision != profile.contract_revision


@pytest.mark.parametrize(
    "changes",
    [
        {"contract_revision": "unreviewed-contract"},
        {"backend_revision": " "},
        {"backend_kind": "mooncake"},
        {"manager_kind": "KVCacheManager"},
        {"cache_dtype": "float16"},
        {"attention_backend": "FLASHINFER"},
        {"attention_kind": "mla"},
        {"layout": "NHD"},
        {"parallelism": (2, 1, 1, 1)},
        {"parallelism": (1, 2, 1, 1)},
        {"parallelism": (1, 1, 2, 1)},
        {"parallelism": (1, 1, 1, 2)},
        {"staging": "python_bounce"},
        {"committed_whole_blocks": False},
        {"extra_features": frozenset({"compression"})},
        {"extra_features": frozenset({"retry"})},
        {"extra_features": frozenset({"unknown-feature"})},
    ],
    ids=[
        "contract-revision",
        "backend-revision",
        "backend",
        "manager",
        "dtype",
        "writer",
        "cache-kind",
        "layout",
        "tp",
        "dp",
        "pp",
        "cp",
        "staging",
        "partial-blocks",
        "compression",
        "retry",
        "unknown-feature",
    ],
)
def test_unsupported_profile_is_rejected(changes: dict[str, object]) -> None:
    """Fail closed for profiles requiring separate compatibility and lifecycle work.

    Args:
        changes: Unsupported facts replacing an otherwise supported profile.
    """
    with pytest.raises(ValueError):
        replace(_profile(), **changes).validate()


def test_request_id_is_not_an_extent_namespace() -> None:
    """Reject a request-shaped integer where a shared namespace is required."""
    with pytest.raises(TypeError, match="namespace must be bytes"):
        build_extent(_view(), (PART,), name=123, is_last=True)


def test_nonboolean_final_extent_flag_is_rejected() -> None:
    """Do not silently reinterpret truthy objects as the final-extent flag."""
    with pytest.raises(TypeError, match="is_last must be bool"):
        build_extent(_view(), (PART,), name=STAGING_EXTENT_NAMESPACE, is_last=1)
