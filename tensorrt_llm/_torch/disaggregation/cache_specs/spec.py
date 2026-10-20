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
"""What one cache spec tells the P/D transfer about itself.

A spec is a plugin. It claims the cache-manager classes it understands and answers three sorts of
questions:

* **Description**: what memory a manager holds (:meth:`CacheSpec.describe`), as unit spaces of
  regions with their layout declarations. How two sides' bytes correspond follows from those
  declarations alone (``correspondence``); a spec never writes a mapper.
* **Selection**: which units of a request move (:meth:`CacheSpec.local_units`,
  :meth:`CacheSpec.chunk_units`, :meth:`CacheSpec.pair_units`) and the request-wide rules
  of the spec (how much history moves, what is excluded, where a received request resumes).
* **Checks and memory**: peer compatibility beyond the declarations, and what to register.

Everything outside ``cache_specs`` asks the spec of a unit space instead of testing its spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind

from .page_table import BufferDecl, PageTable, Region, UnitSpace

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


@dataclass(frozen=True)
class LocalUnitsContext:
    """What :meth:`CacheSpec.local_units` may consult to list a request's units on this rank."""

    request: Any
    space_index: int
    space: UnitSpace
    cache_manager: Any
    reuse_adapter: Any
    transfer_len: int
    """Prompt tokens the request moves (see :meth:`CacheSpec.transfer_length`)."""
    cached_tokens: int
    """Block-aligned prefix the receiver already holds for this space (0 on the sender)."""


@dataclass(frozen=True)
class ChunkContext:
    """One prompt chunk as :meth:`CacheSpec.chunk_units` sees it, in block ordinals."""

    start_block: int
    end_block: int
    is_last: bool
    prompt_len: int


@dataclass(frozen=True)
class PairContext:
    """What :meth:`CacheSpec.pair_units` may consult when pairing one request's units."""

    self_ri: "RankInfo"
    peer_ri: "RankInfo"
    space: UnitSpace
    """The sender's space being paired."""
    dst_start_token: Optional[int]


@dataclass(frozen=True)
class RegistrationRegion:
    """One device allocation to register with the transfer agent.

    ``within_known`` lets a region be skipped when it lies inside one already registered (pools
    that share a single allocation, registered lowest base first).
    """

    base_address: int
    size: int
    within_known: bool = False


MatchedPair = Tuple[UnitSpace, Region, UnitSpace, Region]
"""``(local space, local region, peer space, peer region)`` of one matched region."""


class CacheSpec(ABC):
    """The one place a cache spec describes itself to the P/D transfer.

    Subclass it in ``cache_specs/specs/variants.py`` and decorate the class with
    :func:`~tensorrt_llm._torch.disaggregation.cache_specs.registry.register`. Most specs subclass
    ``specs._base.StandardSpec``, which implements paged and state units for managers built on
    ``KVCacheManagerV2``, and override what their cache does differently.
    """

    name: ClassVar[str]
    """Recorded as ``UnitSpace.spec`` on every space the spec describes; peers match on it."""

    managers: ClassVar[Tuple[str, ...]] = ()
    """Dotted paths of the cache-manager classes this spec describes. A manager is claimed by the
    spec naming the class nearest to it in its MRO."""

    # ---- description (asked of the spec that claims the manager) ----

    @abstractmethod
    def describe(self, manager) -> PageTable:
        """Every region of the manager's memory, as unit spaces of this or other specs."""

    def role_buffers(
        self,
        manager,
        unit: CacheKind,
        role: str,
        nbytes: int,
        *,
        heads: int,
        tokens_per_block: int,
        context: str,
    ) -> Tuple[BufferDecl, ...]:
        """How one ``(layer, role)`` buffer of ``nbytes`` in a ``unit`` space is laid out and split
        across ranks (``heads``: the rank's KV heads)."""
        raise NotImplementedError(f"{self.name} does not describe per-role layouts")

    def requires_physical_ownership(self, manager, mapping, transceiver_config) -> bool:
        """Whether the transfer must run with physical KV ownership (retire only proven writes).

        A spec that needs it validates the deployment here and raises when it cannot be honoured.
        """
        return False

    # ---- request-wide rules (asked of every spec in the page table) ----

    def transfer_length(self, request) -> int:
        """Prompt tokens whose history the request moves (and the receiver must have declared)."""
        return request.prompt_len

    def resume_after_receive(self, request) -> int:
        """Token position the request continues from once its whole prompt has arrived."""
        return request.prompt_len

    def sends_first_tokens(self, request) -> bool:
        """Whether the context side's first generated tokens travel with the request."""
        return True

    def excluded_regions(
        self, manager, page_table: PageTable, space_index: int, request
    ) -> FrozenSet[int]:
        """Indices of regions of one space this request does not move."""
        return frozenset()

    # ---- selection: which units of a request move ----

    @abstractmethod
    def local_units(self, ctx: LocalUnitsContext) -> np.ndarray:
        """This rank's unit ids (pool slots) for one request in one space; ``-1`` = none."""

    @abstractmethod
    def chunk_units(self, space: UnitSpace, units: np.ndarray, ctx: ChunkContext) -> np.ndarray:
        """The part of :meth:`local_units` one prompt chunk sends; ``-1`` blanks the rest."""

    @abstractmethod
    def pair_units(
        self, src_units: np.ndarray, dst_units: np.ndarray, ctx: PairContext
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equal-length sender / receiver unit ids that move; both empty when nothing does."""

    @abstractmethod
    def transfer_bytes(self, space: UnitSpace, region_bytes: int, units: np.ndarray) -> int:
        """Bytes ``region_bytes`` per unit of a space move for ``units``."""

    @abstractmethod
    def sends_incrementally(self, space: UnitSpace) -> bool:
        """Whether the space's units can move in token chunks while the prompt is still running."""

    # ---- compatibility ----

    def check_rank_compatible(self, self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
        """Topology-level gate at peer registration; logs why and returns False on mismatch."""
        return True

    def validate_peer(
        self, self_ri: "RankInfo", peer_ri: "RankInfo", pairs: Sequence[MatchedPair]
    ) -> None:
        """Checks on this spec's matched regions beyond what their declarations make generic;
        raises ``ValueError`` naming the mismatch."""

    def on_peer_registered(self, self_ri: "RankInfo", peer_ri: "RankInfo", pairs) -> None:
        """Hook for diagnostics once a peer is registered (e.g. warn about fragment counts)."""

    # ---- memory ----

    @abstractmethod
    def registration_regions(
        self, page_table: PageTable, space_index: int
    ) -> List[RegistrationRegion]:
        """Device allocations backing one space, to register with the transfer agent."""

    def bounce_unit_bytes(self, page_table: PageTable, space_index: int) -> Optional[int]:
        """Bytes per unit the bounce buffer reserves per ordinal, or None when it sizes by
        :meth:`receiver_extra_bytes` instead."""
        return None

    def receiver_extra_bytes(
        self,
        sender_page_table: PageTable,
        receiver_page_table: PageTable,
        space_index: int,
        dst_units: np.ndarray,
    ) -> int:
        """Bytes landing in this space's units that the per-ordinal bounce sizing does not cover."""
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"
