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
"""Schema for the GPU memory ledger.

Three orthogonal axes describe every entry, and they must stay orthogonal:

``category``
    What the memory is *for*. A business purpose, never an allocation
    mechanism.
``placement``
    Which pool physically holds it. Kept separate from ``category`` because
    CUDA-graph pools hold captured activations -- modelling "CUDA graph" as a
    category would double count against ``ACTIVATION``.
``basis``
    What the number actually measures. Being able to read a number does not
    make it a physical-occupancy number: a page pool can report its
    *configured* capacity long after the pages were released.

A fourth axis, ``source``, records how the number was obtained, so a reader can
tell a measured value from a subtracted one.

Unknown is represented by ``None`` and never by ``0``. A provider that cannot
find its host object reports ``available() is False``; it does not report zero
bytes.
"""

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Basis",
    "Category",
    "Entry",
    "Placement",
    "Source",
]


class StrEnum(str, Enum):
    """``str``-valued enum (``enum.StrEnum`` needs Python 3.11+)."""

    def __str__(self) -> str:
        return str(self.value)


class Category(StrEnum):
    """Business purpose of a block of device memory.

    Deliberately absent:

    ``cuda_graph``
        Expressed as :attr:`Placement.TORCH_GRAPH_POOL`. Graph pools hold the
        captured region's activations and outputs, so a category would double
        count against :attr:`ACTIVATION`.
    ``locality_domain``
        Its bytes are torch-visible (a ``CUDAPluggableAllocator`` installed as
        a ``MemPool``) and are the model weights, merely placed on a specific
        NUMA partition. A category would double count against :attr:`WEIGHTS`.
    """

    WEIGHTS = "weights"
    KV_CACHE_POOL = "kv_cache.pool"
    KV_CACHE_AUX = "kv_cache.aux"
    ATTENTION_WS = "attention.workspace"
    MOE_WS = "moe.workspace"
    MOE_PROFILER = "moe.profiler"
    SAMPLER = "sampler"
    LOGITS = "logits"
    SPECULATIVE = "speculative"
    BUFFER_POOL = "buffer_pool"
    COMM_MNNVL = "comm.mnnvl"
    COMM_SYMM_MEM = "comm.symm_mem"
    COMM_DEEPEP = "comm.deepep"
    COMM_NCCL_WINDOW = "comm.nccl_window"
    COMM_NVLS = "comm.nvls"
    COMM_ULYSSES = "comm.ulysses"
    DWDP_PAGES = "dwdp.pages"
    KV_TRANSFER = "kv_transfer"
    CPP_BUFFERS = "cpp_buffers"

    #: Only for allocations with positive evidence of being activations.
    #: The leftover of ``torch_live - sum(tagged)`` is *not* activation -- it
    #: also holds untagged weights, workspaces and allocations whose tag was
    #: lost to trace-ring eviction. That leftover belongs in
    #: :attr:`UNATTRIBUTED_LIVE`.
    ACTIVATION = "activation"

    #: A tag the category table does not recognise. Reported under its own
    #: name via the entry's ``owner`` rather than folded into a neighbouring
    #: category, so a missing mapping surfaces as a gap to fill instead of
    #: silently inflating whichever category it was guessed into.
    TAGGED_OTHER = "tagged.other"

    #: Default home for ``torch_live - sum(tagged)``.
    UNATTRIBUTED_LIVE = "unattributed_live"

    #: Everything held by this process that no provider explains: the CUDA
    #: context, NCCL comm internals, cuBLAS handles and lazily loaded module
    #: code. Lazy module loading in particular grows during warmup with no
    #: ``cudaMalloc`` and must not be reported as a leak.
    NONTORCH_RESIDUAL = "nontorch_residual"


class Source(StrEnum):
    """How a number was obtained -- i.e. how much to trust it."""

    #: Per-allocation attribution from allocator metadata.
    TAG = "tag"
    #: Read from a component that keeps its own accounting.
    LEDGER = "ledger"
    #: Computed by subtracting other quantities.
    DERIVED = "derived"
    #: Top-level catch-all remainder.
    RESIDUAL = "residual"


class Placement(StrEnum):
    """Which pool physically holds the memory."""

    TORCH_DEFAULT = "torch_default"
    TORCH_GRAPH_POOL = "torch_graph_pool"
    #: A ``torch.cuda.MemPool``, including pools backed by a pluggable
    #: allocator. Measured on torch 2.14 to count toward both
    #: ``memory_reserved()`` and ``memory_allocated()``, and to be separable
    #: via ``segment_pool_id``. Its *physical* occupancy may still differ from
    #: its logical size -- see :attr:`Basis.LOGICAL`.
    TORCH_PRIVATE_POOL = "torch_private_pool"
    NON_TORCH = "non_torch"


class Basis(StrEnum):
    """What the number measures.

    The distinction is load bearing. ``PagePool`` can report a configured
    capacity for pages it has already released, and symmetric-memory wrappers
    report the size they asked for rather than the heap actually reserved.
    Mixing these with physical backing produces a plausible, wrong total.
    """

    #: Device memory actually backed by physical pages right now.
    PHYSICAL_BACKING = "physical_backing"
    #: Bytes the component asked for; the real reservation may be larger.
    REQUESTED = "requested"
    #: Capacity from configuration, which may outlive the pages themselves.
    CONFIGURED_CAPACITY = "configured_capacity"
    #: An allocator pool's reserved/used counters.
    POOL_RESERVED = "pool_reserved"
    #: A logical/virtual-address view that can exceed physical backing, e.g.
    #: torch's accounting for a VMM-backed pool whose pages were released.
    LOGICAL = "logical"


@dataclass(frozen=True)
class Entry:
    """One line of the ledger.

    ``size_bytes`` is ``None`` when the value is genuinely unknown. Providers
    must not substitute ``0``: a reader cannot distinguish "this component
    holds nothing" from "we could not measure this component" once that
    happens.
    """

    category: Category
    size_bytes: int | None
    source: Source
    placement: Placement
    basis: Basis
    owner: str | None = None
    executor_stage: str | None = None
    note: str | None = None

    #: ``False`` for virtual-address reservations and remote imported
    #: mappings. They consume no local device memory, and summing them
    #: inflates the total by the number of peers.
    counts_toward_total: bool = True

    def __post_init__(self) -> None:
        if (
            self.size_bytes is not None
            and self.size_bytes < 0
            and self.source is not Source.RESIDUAL
        ):
            raise ValueError(
                f"Entry {self.category} has negative size {self.size_bytes}; use None for unknown."
            )

    @property
    def is_unknown(self) -> bool:
        return self.size_bytes is None
