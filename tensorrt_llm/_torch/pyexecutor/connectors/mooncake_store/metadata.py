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
"""The per-iteration work list the scheduler hands the workers.

Instances are broadcast from rank 0 to every worker, so these carry only plain
data: a page's identity (its block hash) and where that page currently lives on
this rank (a layer group and a page slot index). Deliberately no store keys --
each worker prefixes its own rank namespace, so one broadcast serves all shards.
"""

from dataclasses import dataclass, field
from typing import List

__all__ = ["MooncakeStoreMetadata", "PageTransfer", "RequestTransfers"]


@dataclass
class PageTransfer:
    """One page of one layer group, to move in either direction."""

    #: Content identity from ``BlockHashChain``; names the key, not the location.
    block_hash: bytes
    layer_group_id: int
    #: Page slot index within ``layer_group_id``, as reported by
    #: ``RequestData.new_block_ids_by_layer_group``.
    page_index: int


@dataclass
class RequestTransfers:
    """Pages belonging to one request, kept together for save bookkeeping.

    The worker owes ``get_finished`` an answer per request, so a save's owner has
    to survive the trip from scheduler to worker.
    """

    request_id: int
    pages: List[PageTransfer] = field(default_factory=list)


@dataclass
class MooncakeStoreMetadata:
    """Loads to perform before the next forward pass, saves to start after it."""

    loads: List[RequestTransfers] = field(default_factory=list)
    saves: List[RequestTransfers] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Whether there is any work at all this iteration."""
        return bool(self.loads or self.saves)
