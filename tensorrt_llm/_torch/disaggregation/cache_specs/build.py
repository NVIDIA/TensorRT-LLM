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
"""Describe a cache manager as a page table: the spec claiming the manager does it."""

from __future__ import annotations

from typing import Tuple

from tensorrt_llm._torch.disaggregation.base import CacheKind

from .correspondence import check_page_table
from .page_table import BufferDecl, PageTable
from .registry import claim


def build_page_table(manager) -> PageTable:
    """The page table of a cache manager, described by the spec that claims it."""
    page_table = claim(manager).describe(manager)
    check_page_table(page_table)
    return page_table


def role_layout(
    manager, kind: CacheKind, role: str, nbytes: int, *, heads: int, tokens_per_block: int
) -> Tuple[BufferDecl, ...]:
    """How the spec describing ``manager`` lays out one ``(layer, role)`` buffer, for readers of the
    cache layout other than the transfer (named reuse)."""
    spec = claim(manager)
    return spec.role_buffers(
        manager,
        kind,
        role,
        nbytes,
        heads=heads,
        tokens_per_block=tokens_per_block,
        context=f"{spec.name} role {role!r}",
    )


def requires_physical_ownership(manager, mapping, transceiver_config) -> bool:
    """Whether the spec describing ``manager`` needs the transfer to enforce physical ownership."""
    return claim(manager).requires_physical_ownership(manager, mapping, transceiver_config)
