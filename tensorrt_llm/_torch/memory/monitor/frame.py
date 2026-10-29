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
"""Top-level accounting frame.

The frame decomposes device memory downward, giving every difference an
explicit home::

    device_total
    |-- free
    |-- other_process                 NVML; unknown, never 0, when unavailable
    `-- process_used
        |-- torch_reserved
        |   |-- torch_live
        |   |   |-- <tagged categories>
        |   |   `-- unattributed_live      subtraction (1)
        |   `-- torch_reserved_unallocated
        |-- nontorch_known                 sum of component counters
        `-- nontorch_residual              subtraction (2)

The two subtractions mean different things and are never merged:

1. ``torch_live - sum(tagged)`` is memory we hold and could in principle name,
   but have not tagged -- untagged weights and workspaces, logits, and
   allocations whose tag was evicted from the trace ring. Calling it
   "activation" would be a systematic misattribution, especially before
   tagging lands.
2. ``process_used - torch_reserved - nontorch_known`` is memory nothing can
   attribute: CUDA context, NCCL internals, cuBLAS handles, lazily loaded
   module code.

Everything here is a pure function over injected readings, so the arithmetic
is testable without a GPU.
"""

from dataclasses import dataclass, field

from .schema import Basis, Category, Entry, Placement, Source

__all__ = [
    "DeviceReading",
    "Frame",
    "TorchReading",
    "build_frame",
]


@dataclass(frozen=True)
class DeviceReading:
    """Device-level readings.

    ``process_used`` is this process's own consumption, from NVML. It is
    ``None`` when NVML is unavailable; callers then fall back to
    ``total - free``, which is a device-wide figure and therefore only an
    upper bound. TensorRT-LLM supports several colocated-process
    configurations (Ray fractional GPUs, GMS weight sharing, RLHF alongside a
    trainer, disaggregated ctx+gen on one device), so this distinction is not
    hypothetical.
    """

    total: int
    free: int
    process_used: int | None = None

    @property
    def device_used(self) -> int:
        return self.total - self.free

    @property
    def effective_process_used(self) -> int:
        return self.device_used if self.process_used is None else self.process_used

    @property
    def process_used_is_upper_bound(self) -> bool:
        return self.process_used is None

    @property
    def other_process(self) -> int | None:
        """Memory held by other processes, or ``None`` if unknowable."""
        if self.process_used is None:
            return None
        return max(self.device_used - self.process_used, 0)


@dataclass(frozen=True)
class TorchReading:
    """Caching-allocator readings.

    ``reserved`` and ``live`` come from ``memory_reserved()`` and
    ``memory_stats()["allocated_bytes.all.current"]``. ``reserved`` is
    ``None`` when the allocator does not report it.
    """

    reserved: int | None
    live: int
    #: Usage by pool (``pools.PoolUsage``), kind ``default``, ``graph`` or
    #: ``mempool``. ``None`` when pools could not be read; free space is then
    #: reported as one figure.
    pools: tuple | None = None

    @property
    def released(self) -> int:
        """Reserved bytes of sleep-mode pools whose memory is released."""
        return sum(p.reserved for p in self.pools or () if p.released)

    @property
    def footprint(self) -> int:
        """What torch holds from the driver, or ``live`` as a lower bound.

        Released sleep-mode pools are left out: torch still counts them,
        the driver no longer backs them.
        """
        return (self.live if self.reserved is None else self.reserved) - self.released

    @property
    def reserved_unallocated(self) -> int | None:
        """Reserved but not handed out.

        Deliberately *not* named "fragmentation". Besides genuinely
        fragmented space it also covers blocks already requested for release
        that are waiting on other streams; torch's snapshot distinguishes
        ``active_allocated``, ``active_awaiting_free`` and ``inactive``.
        Calling the whole gap fragmentation would suggest it is immediately
        reusable.
        """
        if self.reserved is None:
            return None
        return max(self.reserved - self.live, 0)


@dataclass
class Frame:
    """Resolved accounting frame."""

    device: DeviceReading
    torch: TorchReading
    entries: list[Entry] = field(default_factory=list)

    tagged_total: int = 0
    unattributed_live: int = 0
    nontorch_known: int = 0
    nontorch_residual: int = 0

    #: Entries whose size is unknown, by category. These are excluded from
    #: every sum; they exist so the report can say "we could not measure
    #: this" instead of silently contributing zero.
    unknown_categories: list[Category] = field(default_factory=list)

    #: Entries excluded from totals by construction (VA reservations, remote
    #: mappings). Shown, never summed.
    excluded_total: int = 0

    @property
    def residual_is_negative(self) -> bool:
        return self.nontorch_residual < 0

    @property
    def residual_is_upper_bound(self) -> bool:
        """The residual also holds torch's pool slack, which went unreported."""
        return self.torch.reserved is None


def build_frame(
    device: DeviceReading,
    torch: TorchReading,
    entries: list[Entry],
) -> Frame:
    """Resolve the frame from readings and provider entries.

    A negative ``nontorch_residual`` is preserved rather than clamped. It
    signals that two readings disagree -- different vantage points, a sampling
    skew, or another process releasing memory mid-measurement -- and clamping
    it to zero hides exactly the case worth investigating.
    """
    frame = Frame(device=device, torch=torch, entries=list(entries))

    for entry in entries:
        if entry.size_bytes is None:
            frame.unknown_categories.append(entry.category)
            continue
        if not entry.counts_toward_total:
            frame.excluded_total += entry.size_bytes
            continue
        if entry.placement is Placement.NON_TORCH:
            frame.nontorch_known += entry.size_bytes
        else:
            frame.tagged_total += entry.size_bytes

    # Subtraction (1): torch-visible memory we have not attributed.
    frame.unattributed_live = max(torch.live - frame.tagged_total, 0)

    # Subtraction (2): held by this process, explained by nobody.
    # With reserved unreported, live stands in for torch's footprint, so the
    # residual also absorbs the pool's slack and is only an upper bound.
    frame.nontorch_residual = device.effective_process_used - torch.footprint - frame.nontorch_known

    return frame


def derived_entries(frame: Frame) -> list[Entry]:
    """The subtraction results, as monitor entries."""
    released = []
    if frame.torch.released:
        released.append(
            Entry(
                category=Category.SLEEP_RELEASED,
                size_bytes=-frame.torch.released,
                source=Source.DERIVED,
                placement=Placement.TORCH_DEFAULT,
                basis=Basis.PHYSICAL_BACKING,
                note="counted by torch, physically released by sleep mode",
            )
        )
    return released + [
        Entry(
            category=Category.UNATTRIBUTED_LIVE,
            size_bytes=frame.unattributed_live,
            source=Source.DERIVED,
            placement=Placement.TORCH_DEFAULT,
            basis=Basis.LOGICAL,
            note="torch_live minus tagged; NOT activation",
        ),
        # The residual keeps its sign. Reporting a negative value as
        # "unknown" would hide the disagreement just as effectively as
        # clamping it to zero.
        Entry(
            category=Category.NONTORCH_RESIDUAL,
            size_bytes=frame.nontorch_residual,
            source=Source.RESIDUAL,
            placement=Placement.NON_TORCH,
            basis=Basis.PHYSICAL_BACKING,
            note=(
                "NEGATIVE: readings disagree (vantage point, sampling skew, "
                "or another process released memory mid-measurement)"
                if frame.residual_is_negative
                else "UPPER BOUND: also holds torch pool slack the allocator did not report"
                if frame.residual_is_upper_bound
                else "CUDA context, NCCL internals, cuBLAS, lazy module loading"
            ),
        ),
    ]
