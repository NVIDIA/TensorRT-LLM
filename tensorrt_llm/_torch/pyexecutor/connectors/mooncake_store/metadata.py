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
data: a page's identity (its block hash), its layer group, its block ordinal
in the request, and the page slot index rank 0 allocated for it. Deliberately
no store keys: each worker prefixes its own rank namespace, so one broadcast
serves all shards.

The slot index is rank 0's. Tensor-parallel ranks make the same allocation
decisions, but not always on the same slots: KV cache manager V2 hands out a
recycled slot whose last GPU use has finished before a never-used one, and
whether that use has finished depends on each rank's GPU progress. So every
worker maps the ordinal back to its own slot before it touches a page (see
`MooncakeStoreConnectorWorker._localize`); the broadcast index is only a
fallback for a transfer built without an ordinal.
"""

from dataclasses import dataclass, field
from typing import List

__all__ = [
    "MooncakeStoreMetadata",
    "PageTransfer",
    "PrefixLoadTransfers",
    "RequestTransfers",
    "WindowRetire",
]


@dataclass
class PageTransfer:
    """One page of one layer group, to move in either direction."""

    #: Content identity from `BlockHashChain`; names the key, not the location.
    block_hash: bytes
    layer_group_id: int
    #: Page slot index within `layer_group_id` on the rank that built the
    #: metadata (rank 0), as reported by `RequestData.new_block_ids_by_layer_group`.
    #: A worker replaces it with its own rank's slot for `block_ordinal`.
    page_index: int
    #: The block's ordinal in its request (its position in the request's page
    #: list for the group). -1 when unknown, in which case `page_index` is used as is.
    block_ordinal: int = -1


@dataclass
class RequestTransfers:
    """Pages belonging to one request, kept together for save bookkeeping.

    The worker owes `get_finished` an answer per request, so a save's owner has
    to survive the trip from scheduler to worker.
    """

    request_id: int
    pages: List[PageTransfer] = field(default_factory=list)


@dataclass
class PrefixLoadTransfers:
    """The pages of one confirmed prefix reservation.

    Reported back through `get_finished_prefix_loads` once they are on the
    device. An asynchronous one is moved off the executor thread while its
    request waits outside the batch.
    """

    reservation_id: int
    is_async: bool
    transfers: RequestTransfers


@dataclass
class WindowRetire:
    """A prompt end whose window was just handed to the save thread, and the prompt end it resumed from.

    The worker records the link (end -> resume) in the pool and removes the window that is `keep` links back
    (`TRTLLM_MOONCAKE_STORE_WINDOW_KEEP`).
    """

    request_id: int
    #: Block ordinal of the new prompt end, and the chain hash of its last block (names the link).
    end_block: int
    end_hash: bytes
    #: Block ordinal the request resumed from (a stored prompt end), and the chain hashes of blocks [0, resume_block).
    resume_block: int
    hashes: List[bytes] = field(default_factory=list)


@dataclass
class MooncakeStoreMetadata:
    """Loads to perform before the next forward pass, saves to start after it."""

    loads: List[RequestTransfers] = field(default_factory=list)
    saves: List[RequestTransfers] = field(default_factory=list)
    prefix_loads: List[PrefixLoadTransfers] = field(default_factory=list)
    #: Prompt ends saved this pass whose older windows may be retired (TRTLLM_MOONCAKE_STORE_WINDOW_KEEP; empty
    #: when it is off).
    retires: List[WindowRetire] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Whether there is any work at all this iteration."""
        return bool(self.loads or self.saves or self.prefix_loads or self.retires)
