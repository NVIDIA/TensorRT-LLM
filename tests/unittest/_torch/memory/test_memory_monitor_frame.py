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
"""Frame arithmetic and reporting rules for the GPU memory monitor.

Several tests here are deliberately adversarial: they encode a rule by
constructing the input that violates it. If someone reintroduces the
behaviour the rule forbids -- clamping a negative residual, substituting 0
for unknown, relabelling the unattributed remainder as activation -- the
corresponding test fails. A test that cannot fail verifies nothing.

CPU only: the frame is a pure function over injected readings.
"""

import pytest

from tensorrt_llm._torch.memory.monitor.capability import Capability
from tensorrt_llm._torch.memory.monitor.frame import (
    DeviceReading,
    Frame,
    TorchReading,
    build_frame,
    derived_entries,
)
from tensorrt_llm._torch.memory.monitor.report import UNAVAILABLE, render, summarize_by_category
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source

GIB = 1 << 30


def _capability(**overrides) -> Capability:
    defaults = dict(
        allocator_backend="native",
        alloc_conf="",
        alloc_conf_var=None,
        expandable_segments=False,
        metadata_supported=True,
        history_api_present=True,
        snapshot_api_present=True,
        oom_observer_present=True,
        nvml_available=True,
        unsupported_stats=(),
    )
    defaults.update(overrides)
    return Capability(**defaults)


def _entry(
    category: Category, size: int | None, placement: Placement = Placement.TORCH_DEFAULT, **kwargs
) -> Entry:
    kwargs.setdefault("source", Source.TAG)
    kwargs.setdefault("basis", Basis.PHYSICAL_BACKING)
    return Entry(category=category, size_bytes=size, placement=placement, **kwargs)


class TestFrameArithmetic:
    def test_frame_balances(self):
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=150 * GIB),
            TorchReading(reserved=96 * GIB, live=78 * GIB),
            [
                _entry(Category.WEIGHTS, 52 * GIB),
                _entry(
                    Category.KV_CACHE_POOL, 40 * GIB, Placement.NON_TORCH, source=Source.COMPONENT
                ),
            ],
        )
        assert frame.tagged_total == 52 * GIB
        assert frame.nontorch_known == 40 * GIB
        assert frame.unattributed_live == 26 * GIB
        assert frame.nontorch_residual == 150 * GIB - 96 * GIB - 40 * GIB

        total = frame.torch.reserved + frame.nontorch_known + frame.nontorch_residual
        assert total == frame.device.effective_process_used

    def test_reserved_unallocated_is_the_gap(self):
        torch_r = TorchReading(reserved=96 * GIB, live=78 * GIB)
        assert torch_r.reserved_unallocated == 18 * GIB


class TestNegativeResidual:
    """A negative residual is a signal, not an error to be smoothed away."""

    def _negative_frame(self) -> Frame:
        # torch claims more than the process is measured to hold.
        return build_frame(
            DeviceReading(total=180 * GIB, free=0, process_used=50 * GIB),
            TorchReading(reserved=60 * GIB, live=40 * GIB),
            [],
        )

    def test_negative_residual_is_not_clamped(self):
        frame = self._negative_frame()
        assert frame.nontorch_residual == -10 * GIB
        assert frame.residual_is_negative

    def test_negative_residual_survives_into_entries(self):
        # Reporting a negative value as "unknown" would hide the
        # disagreement just as effectively as clamping it to zero.
        residual = derived_entries(self._negative_frame())[1]
        assert residual.size_bytes == -10 * GIB
        assert residual.size_bytes is not None
        assert "NEGATIVE" in (residual.note or "")

    def test_report_flags_negative_residual(self):
        text = render(self._negative_frame(), _capability())
        assert "-" in text
        assert "negative" in text.lower()


class TestUnknownIsNotZero:
    """``None`` and ``0`` must stay distinguishable all the way to the page."""

    def test_unknown_entry_excluded_from_sums(self):
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
            TorchReading(reserved=50 * GIB, live=40 * GIB),
            [
                _entry(Category.DWDP_PAGES, None, Placement.NON_TORCH, source=Source.COMPONENT),
            ],
        )
        assert frame.nontorch_known == 0
        assert Category.DWDP_PAGES in frame.unknown_categories

    def test_unknown_renders_as_unavailable_not_zero(self):
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
            TorchReading(reserved=50 * GIB, live=40 * GIB),
            [
                _entry(Category.DWDP_PAGES, None, Placement.NON_TORCH, source=Source.COMPONENT),
            ],
        )
        text = render(frame, _capability())
        assert UNAVAILABLE.strip() in text
        dwdp_lines = [line for line in text.splitlines() if str(Category.DWDP_PAGES) in line]
        assert dwdp_lines
        assert not any("0.00 GiB" in line for line in dwdp_lines)

    def test_unknown_survives_category_aggregation(self):
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
            TorchReading(reserved=50 * GIB, live=40 * GIB),
            [
                _entry(Category.DWDP_PAGES, None, Placement.NON_TORCH, source=Source.COMPONENT),
            ],
        )
        assert summarize_by_category(frame)[Category.DWDP_PAGES] is None

    def test_missing_nvml_yields_unknown_other_process(self):
        device = DeviceReading(total=180 * GIB, free=10 * GIB, process_used=None)
        assert device.other_process is None
        assert device.process_used_is_upper_bound
        text = render(
            build_frame(device, TorchReading(reserved=0, live=0), []),
            _capability(nvml_available=False),
        )
        assert "UPPER BOUND" in text


class TestUnsupportedBackend:
    """Zeros from a backend that never populates a field are not measurements."""

    def test_unsupported_stats_are_announced(self):
        capability = _capability(
            allocator_backend="cudaMallocAsync",
            unsupported_stats=("reserved_bytes.all.current",),
        )
        assert not capability.stats_trustworthy
        text = render(
            build_frame(
                DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
                TorchReading(reserved=50 * GIB, live=40 * GIB),
                [],
            ),
            capability,
        )
        assert "unsupported" in text
        assert "not as zero" in text


class TestNamingRules:
    def test_reserved_gap_is_not_called_fragmentation(self):
        # The gap also holds blocks awaiting release on other streams;
        # torch's snapshot distinguishes active_allocated /
        # active_awaiting_free / inactive. Calling it fragmentation implies
        # it is immediately reusable.
        text = render(
            build_frame(
                DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
                TorchReading(reserved=50 * GIB, live=40 * GIB),
                [],
            ),
            _capability(),
        )
        assert "fragmentation" not in text.lower()
        assert "not handed out" in text

    def test_remainder_is_unattributed_not_activation(self):
        # Before tagging exists nearly all torch-live memory lands here.
        # Labelling it activation would be a systematic misattribution.
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
            TorchReading(reserved=50 * GIB, live=40 * GIB),
            [],
        )
        remainder = derived_entries(frame)[0]
        assert remainder.category is Category.UNATTRIBUTED_LIVE
        assert remainder.category is not Category.ACTIVATION
        assert remainder.source is Source.DERIVED
        totals = summarize_by_category(frame)
        assert totals.get(Category.ACTIVATION) is None
        assert totals[Category.UNATTRIBUTED_LIVE] == 40 * GIB


class TestExcludedFromTotals:
    """VA reservations and remote mappings consume no local memory."""

    def test_remote_mapping_not_summed(self):
        local, peers = 2 * GIB, 6 * GIB
        frame = build_frame(
            DeviceReading(total=180 * GIB, free=10 * GIB, process_used=100 * GIB),
            TorchReading(reserved=50 * GIB, live=40 * GIB),
            [
                _entry(Category.COMM_MNNVL, local, Placement.NON_TORCH, source=Source.COMPONENT),
                _entry(
                    Category.COMM_MNNVL,
                    peers,
                    Placement.NON_TORCH,
                    source=Source.COMPONENT,
                    basis=Basis.LOGICAL,
                    note="remote mapped",
                    counts_toward_total=False,
                ),
            ],
        )
        assert frame.nontorch_known == local
        assert frame.excluded_total == peers


class TestSchemaGuards:
    def test_negative_size_rejected_except_for_residual(self):
        with pytest.raises(ValueError):
            Entry(
                category=Category.WEIGHTS,
                size_bytes=-1,
                source=Source.TAG,
                placement=Placement.TORCH_DEFAULT,
                basis=Basis.PHYSICAL_BACKING,
            )
        Entry(
            category=Category.NONTORCH_RESIDUAL,
            size_bytes=-1,
            source=Source.RESIDUAL,
            placement=Placement.NON_TORCH,
            basis=Basis.PHYSICAL_BACKING,
        )

    def test_basis_is_mandatory(self):
        # Being able to read a number does not make it physical occupancy:
        # a page pool reports configured capacity for released pages.
        with pytest.raises(TypeError):
            Entry(
                category=Category.DWDP_PAGES,
                size_bytes=GIB,
                source=Source.COMPONENT,
                placement=Placement.NON_TORCH,
            )

    def test_no_cuda_graph_or_locality_domain_category(self):
        # Both are placements, not purposes; as categories they would double
        # count against ACTIVATION and WEIGHTS respectively.
        names = {c.value for c in Category}
        assert "cuda_graph" not in names
        assert "locality_domain" not in names
        assert Placement.TORCH_GRAPH_POOL.value == "torch_graph_pool"
