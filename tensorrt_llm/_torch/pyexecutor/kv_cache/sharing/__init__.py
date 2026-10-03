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
"""Lending a KV cache manager v2's blocks to transfer backends. This module is the whole API;
every other module in the package is private, and the attach functions load the implementation."""

import typing as _typing

from ._types import (
    GroupRun,
    InPlaceLender,
    Lease,
    Part,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
)

if _typing.TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "GroupRun",
    "InPlaceLender",
    "Lease",
    "Part",
    "PartsHold",
    "Readiness",
    "RegionView",
    "StagingLender",
    "StagingOptions",
    "attach_in_place",
    "attach_staging",
]


def attach_staging(
    manager: "KVCacheManagerV2", *, scope: bytes, staging: StagingOptions
) -> StagingLender:
    """Attach the manager's one lender, which relays whole blocks through host staging.

    Args:
        manager: A ``KVCacheManagerV2`` that commits blocks to its prefix-reuse tree: block reuse
            on, and joint reuse for a draft manager, since a publish lends only committed blocks.
        scope: Equal exactly where KV bytes mean the same: model, weights, numerics, attention;
            at most 65535 bytes.
        staging: The staging size, in whole fetches.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping, ``scope`` is not
            ``bytes``, or ``staging`` is not ``StagingOptions``.
        ValueError: A lender is attached already; the manager is context-parallel, holds recurrent
            state or commits no blocks (block reuse off); ``scope`` is longer than 65535 bytes;
            ``staging.max_bytes`` is below one fetch.
    """
    from ._lender import attach_staging as _attach

    return _attach(manager, scope=scope, staging=staging)


def attach_in_place(manager: "KVCacheManagerV2") -> InPlaceLender:
    """Attach the manager's one lender, which lends requests' own device pages in place.

    Args:
        manager: A ``KVCacheManagerV2``. Caches on loan at the manager's shutdown, with its device
            pools, stay until the process exits.

    Returns:
        The lender, installed on ``manager`` for the manager's life.

    Raises:
        TypeError: ``manager`` is not a ``KVCacheManagerV2`` or has no mapping.
        ValueError: A lender is attached already, or the manager is context-parallel or holds
            recurrent state.
    """
    from ._lender import attach_in_place as _attach

    return _attach(manager)
