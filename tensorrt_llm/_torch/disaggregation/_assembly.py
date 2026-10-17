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
"""Private runtime assembly shared by transfer and Store integrations.

Only the current homogeneous HOST lender is covered. Public memory-kind and row
selection, immutable View/hint metadata, and canonical model-scope derivation are
prerequisites for general assembly. This module activates no provider. Profile
facts come from the serving model's assembly; neither a provider name nor a
caller-supplied hint can establish memory compatibility or isolation.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    LeaseView,
    Lender,
    MemoryRegion,
    StagingOptions,
    attach_lender,
)

from .base.backend import Backend, CacheKind, SlotRegion, WakeFn

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

OwnerCleanupFn = Callable[[Callable[[], None]], bool]
"""Thread-safe, nonblocking posting of bounded manager-thread finalizers.

True accepts exactly-once execution; False retains provider ownership for retry.
A callback is posted only after required deregistration/access-end evidence.
Acceptance or queue drain does not prove cleanup. Failed finalizers retain their
fault and resources. Blocking SDK work must run outside the manager thread.
"""

ProviderCleanupCompleteFn = Callable[[], bool]
"""Private, nonblocking and thread-safe closure of provider cleanup production.

True is monotonic: no further finalizers will be posted and no owner-cleanup
obligation remains outside the accepted dispatcher queue. False keeps provider
ownership, including finalizers whose posting was rejected, for later retry.
This proves neither ended access nor completion of accepted callbacks. Retirement
also requires empty public shutdown labels, a fault-free drained callback queue,
and all lease/result/Lifecycle obligations. Uncertain construction without a
Backend still requires independent access-end evidence. Public shutdown labels
retain their exact access/registration meaning.
"""

BackendFactory = Callable[[tuple[SlotRegion, ...], WakeFn, object, OwnerCleanupFn], Backend]
"""Private provider construction hook receiving the complete region namespace.

The runtime roots this callable, its ProviderCleanupCompleteFn and its cleanup
dispatcher before invocation. A stateful factory roots partial registration and
RegionHold state before exposure, including failures before a Backend returns.
It captures lender access privately; hold acquisition runs on the manager thread.
It registers only supported/accessed regions without renumbering them, and posts
bounded hold finalizers through the dispatcher. Rejected posts retain provider
ownership for retry through provider progress or repeated shutdown. Public labels,
private cleanup closure and accepted callback completion are separate retirement
conditions. No public Backend cleanup API is added.
"""


class FactoryRejected(RuntimeError):
    """Construction certifies no remaining access, registration or region hold."""


@dataclass(frozen=True)
class ProfileSpec:
    """Trusted immutable facts for the initial BF16 MHA compatibility cell.

    The model/executor assembly supplies facts for the actual writer and serving
    manager. This is not user-facing configuration and does not independently
    qualify a provider. Scope must come from the agreed canonical identity builder,
    covering weights, adapters, writer layout, dtype and speculative/target role;
    that builder remains a prerequisite rather than a second RI hash algorithm.

    Attributes:
        id: Local profile identifier.
        lifecycle_profile_id: Selected, separately qualified Lifecycle profile.
        scope: Stable canonical byte identity for this manager's entire life.
        model_layout: Actual attention-writer layout identifier.
        dtype: Actual cache dtype supplied by model assembly.
        topology: Tensor, data, pipeline and context parallel sizes.
        state_kinds: Canonical cache kinds from the layout's layer groups.
        full_attention: Whether every group uses full attention.
        pool_count: Number of physical pool groups in the selected layout.
        speculative_mode: Explicit speculative mode; only none is supported here.
        model_role: Target or draft role; this cell accepts only the target.
        joint_reuse: Whether reuse requires another manager's draft state.
        staging: Canonical capacity options for the pinned lender baseline.
        backend_config: Opaque provider-owned factory configuration.

    Raises:
        TypeError: A fact has an invalid type.
        ValueError: A required identity is empty or the cell is unsupported.
    """

    id: str
    lifecycle_profile_id: str
    scope: bytes
    model_layout: str
    dtype: str
    topology: tuple[int, int, int, int]
    state_kinds: tuple[CacheKind, ...]
    full_attention: bool
    pool_count: int
    speculative_mode: str
    model_role: Literal["target", "draft"]
    joint_reuse: bool
    staging: StagingOptions
    backend_config: object = None

    def __post_init__(self) -> None:
        """Reject unsupported facts before any memory or backend exposure."""
        for name, value in (("id", self.id), ("lifecycle_profile_id", self.lifecycle_profile_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.scope, bytes):
            raise TypeError("profile scope must be bytes")
        if not self.scope or len(self.scope) > 65535:
            raise ValueError("profile scope must contain 1 to 65535 bytes")
        if not isinstance(self.staging, StagingOptions):
            raise TypeError("profile staging must be StagingOptions")
        if self.model_layout != "mha-trtllm-hnd" or self.dtype != "bfloat16":
            raise ValueError("initial profile requires BF16 MHA with the HND writer layout")
        topology = tuple(self.topology)
        if any(
            isinstance(size, bool) or not isinstance(size, numbers.Integral) for size in topology
        ):
            raise TypeError("topology sizes must be integers")
        if topology != (1, 1, 1, 1):
            raise ValueError("initial profile requires TP1/DP1/PP1/CP1")
        kinds = tuple(self.state_kinds)
        if not kinds or any(not isinstance(kind, CacheKind) for kind in kinds):
            raise TypeError("state kinds must contain canonical CacheKind values")
        if any(kind != CacheKind.PAGED for kind in kinds):
            raise ValueError("initial profile supports paged attention state only")
        if self.full_attention is not True:
            raise ValueError("initial profile requires full attention")
        if isinstance(self.pool_count, bool) or not isinstance(self.pool_count, numbers.Integral):
            raise TypeError("pool_count must be an integer")
        if self.pool_count != 1:
            raise ValueError("initial profile requires one pool group")
        if (
            self.speculative_mode != "none"
            or self.model_role != "target"
            or self.joint_reuse is not False
        ):
            raise ValueError("initial profile excludes speculative and joint draft reuse")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "state_kinds", kinds)


@dataclass(frozen=True)
class RegionNamespace:
    """Validated whole namespace and assembly-established memory kind.

    The current attach_lender contract returns HOST regions only. Callers using
    fixtures must explicitly supply that provenance. This does not infer future
    heterogeneous region kinds; their canonical metadata is a merge prerequisite.
    Region descriptors and group membership are immutable canonical records, with
    indices unchanged. A provider owns any registration hold, not this descriptor.

    Attributes:
        regions: Complete namespace in the lender's original index order.
        memory_kind: Verified homogeneous HOST baseline, never inferred from names.

    Raises:
        ValueError: The memory kind or region geometry is unsupported.
        TypeError: A region is not the canonical immutable MemoryRegion.
    """

    regions: tuple[MemoryRegion, ...]
    memory_kind: Literal["HOST"]

    def __post_init__(self) -> None:
        """Validate static geometry once before any backend factory is called."""
        if self.memory_kind != "HOST":
            raise ValueError("general memory-kind metadata is not available in this baseline")
        regions = tuple(self.regions)
        intervals: list[tuple[int, int]] = []
        for region in regions:
            if not isinstance(region, MemoryRegion):
                raise TypeError("region namespace requires canonical MemoryRegion records")
            dimensions = (region.address, region.nbytes, region.slot_bytes, region.slot_count)
            if any(
                isinstance(v, bool) or not isinstance(v, numbers.Integral) or v <= 0
                for v in dimensions
            ):
                raise ValueError("region geometry must use positive integers")
            if int(region.nbytes) < int(region.slot_bytes) * int(region.slot_count):
                raise ValueError("region does not contain all declared slots")
            end = int(region.address) + int(region.nbytes)
            if end > np.iinfo(np.int64).max:
                raise ValueError("region addresses must fit int64")
            if not isinstance(region.layer_group_indices, tuple) or any(
                isinstance(index, bool) or not isinstance(index, numbers.Integral) or index < 0
                for index in region.layer_group_indices
            ):
                raise ValueError(
                    "region layer groups must be an immutable tuple of nonnegative indices"
                )
            intervals.append((int(region.address), end))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("regions overlap")
        object.__setattr__(self, "regions", regions)


@dataclass(frozen=True)
class DomainBinding:
    """Orchestration's strong lifetime root for one serving manager and lender.

    The holder receives only ``namespace``, never this association. The executor
    shuts the manager down last, after leases, provider registrations/holds and
    accepted owner-thread finalizers retire. Strong ownership prevents premature
    manager destruction; the lender remains the only memory-access authority.

    Attributes:
        manager: Serving manager, retained by orchestration.
        lender: Canonical access-granting interface.
        profile: Trusted immutable compatibility/identity facts.
        namespace: Complete validated region descriptors for holder/provider use.
    """

    manager: KVCacheManagerV2
    lender: Lender
    profile: ProfileSpec
    namespace: RegionNamespace


def attach_domain(manager: KVCacheManagerV2, profile: ProfileSpec) -> DomainBinding:
    """Attach the canonical lender without duplicating manager policy checks.

    The caller supplies facts derived from its actual writer/serving-manager
    assembly. The lender validates its own attachment restrictions, including
    lossy codecs. Backend-specific pinning requirements belong to its factory.
    This helper neither activates a backend nor certifies provider qualification.
    Unexpected attach failures retain the canonical lender's safety meaning.

    Args:
        manager: Serving manager, never a sizing probe.
        profile: Trusted compatibility facts and canonical byte scope.

    Returns:
        An orchestration-owned manager association and HOST region namespace.

    Raises:
        TypeError: Profile is not ProfileSpec.
        ValueError: The canonical lender refuses attachment or region validation fails.
        MemoryError: The lender cannot allocate its staging capacity.
    """
    if not isinstance(profile, ProfileSpec):
        raise TypeError("profile must be ProfileSpec")
    lender = attach_lender(manager, scope=profile.scope, staging=profile.staging)
    namespace = RegionNamespace(lender.regions, memory_kind="HOST")
    return DomainBinding(manager, lender, profile, namespace)


def project_view(lease_view: LeaseView, namespace: RegionNamespace) -> LeaseView:
    """Validate full ready rows and return the original structural backend View.

    Canonical LeaseView already implements View; no wrapper, row reordering or
    address copying is required. Retain its metadata through the final Outcome,
    including when access ends first. The lease still grants the actual access.

    Args:
        lease_view: Canonical ready view with immutable keyed/addressed rows.
        namespace: Assembly-validated complete namespace for this manager.

    Returns:
        The same LeaseView, with the same keys, row masks and addresses.

    Raises:
        ValueError: Rows are unplaced, writable or outside their declared slots.
    """
    for blocks in lease_view.groups:
        keys, addresses, region_index = blocks.object_keys, blocks.addresses, blocks.region_index
        if keys is None or addresses is None or region_index is None:
            raise ValueError("initial profile requires keys, addresses and a region")
        if not 0 <= region_index < len(namespace.regions):
            raise ValueError("row region index is outside the lender namespace")
        if (
            keys.dtype != np.uint8
            or keys.ndim != 2
            or keys.shape[0] != len(blocks)
            or keys.shape[1] == 0
            or addresses.dtype != np.int64
            or addresses.shape != (len(blocks),)
            or keys.flags.writeable
            or addresses.flags.writeable
        ):
            raise ValueError("lender row arrays must remain aligned, typed and read-only")
        region = namespace.regions[region_index]
        if blocks.layer_group_index not in region.layer_group_indices:
            raise ValueError("row layer group is not declared by its region")
        last_slot = int(region.address) + (int(region.slot_count) - 1) * int(region.slot_bytes)
        if np.any(addresses < region.address) or np.any(addresses > last_slot):
            raise ValueError("row address is outside the region's complete slots")
        if np.any((addresses - region.address) % region.slot_bytes):
            raise ValueError("row address is not aligned to a complete slot")
    return lease_view
