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
import torch
from memory_monitor_test_utils import GIB, capability

from tensorrt_llm._torch.memory.monitor.capture import read_torch
from tensorrt_llm._torch.memory.monitor.frame import (
    DeviceReading,
    TorchReading,
    build_frame,
    derived_entries,
)
from tensorrt_llm._torch.memory.monitor.report import UNAVAILABLE, render
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source
from tensorrt_llm._torch.memory.monitor.summary import group_totals

pytestmark = pytest.mark.cpu_only


def _entry(
    category: Category, size: int | None, placement: Placement = Placement.TORCH_DEFAULT, **kwargs
) -> Entry:
    kwargs.setdefault("source", Source.TAG)
    kwargs.setdefault("basis", Basis.PHYSICAL_BACKING)
    return Entry(category=category, size_bytes=size, placement=placement, **kwargs)


def _frame(process_used=100 * GIB, reserved=50 * GIB, live=40 * GIB, entries=(), free=10 * GIB):
    return build_frame(
        DeviceReading(total=180 * GIB, free=free, process_used=process_used),
        TorchReading(reserved=reserved, live=live),
        list(entries),
    )


def test_frame_balances_and_keeps_a_negative_residual():
    frame = _frame(
        process_used=150 * GIB,
        reserved=96 * GIB,
        live=78 * GIB,
        entries=[
            _entry(Category.WEIGHTS, 52 * GIB),
            _entry(Category.KV_CACHE_POOL, 40 * GIB, Placement.NON_TORCH, source=Source.COMPONENT),
        ],
    )
    assert frame.tagged_total == 52 * GIB
    assert frame.nontorch_known == 40 * GIB
    assert frame.unattributed_live == 26 * GIB
    assert frame.nontorch_residual == 150 * GIB - 96 * GIB - 40 * GIB
    total = frame.torch.reserved + frame.nontorch_known + frame.nontorch_residual
    assert total == frame.device.effective_process_used
    assert frame.torch.reserved_unallocated == 18 * GIB, "the reserved gap"

    # VA reservations and remote mappings consume no local memory.
    local, peers = 2 * GIB, 6 * GIB
    frame = _frame(
        entries=[
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
        ]
    )
    assert frame.nontorch_known == local, "a remote mapping is not summed"
    assert frame.excluded_total == peers

    # A negative residual is a signal, not an error to be smoothed away:
    # torch claims more than the process is measured to hold.
    negative = _frame(process_used=50 * GIB, reserved=60 * GIB, live=40 * GIB, free=0)
    assert negative.nontorch_residual == -10 * GIB, "a negative residual is not clamped"
    assert negative.residual_is_negative
    # Reporting a negative value as "unknown" would hide the
    # disagreement just as effectively as clamping it to zero.
    residual = derived_entries(negative)[1]
    assert residual.size_bytes == -10 * GIB
    assert "NEGATIVE" in (residual.note or "")
    text = render(negative, capability())
    assert "-" in text and "negative" in text.lower(), "the report flags a negative residual"


def test_unknown_is_not_zero(monkeypatch):
    """``None`` and ``0`` must stay distinguishable all the way to the page."""

    frame = _frame(
        entries=[_entry(Category.DWDP_PAGES, None, Placement.NON_TORCH, source=Source.COMPONENT)]
    )
    assert frame.nontorch_known == 0, "an unknown entry is excluded from sums"
    assert Category.DWDP_PAGES in frame.unknown_categories
    assert group_totals(frame).unmeasured, "survives aggregation as unmeasured"
    text = render(frame, capability())
    assert UNAVAILABLE.strip() in text
    # The detail row for the category itself, not the summary row that
    # names it among the unmeasured (whose "50.00 GiB" contains "0.00 GiB").
    dwdp_lines = [
        line for line in text.splitlines() if line.split()[:1] == [str(Category.DWDP_PAGES)]
    ]
    assert dwdp_lines
    assert not any(" 0.00 GiB" in line for line in dwdp_lines)

    device = DeviceReading(total=180 * GIB, free=10 * GIB, process_used=None)
    assert device.other_process is None, "missing NVML: other processes unknown"
    assert device.process_used_is_upper_bound
    text = render(
        build_frame(device, TorchReading(reserved=0, live=0), []),
        capability(nvml_available=False),
    )
    assert "UPPER BOUND" in text

    # A reserved figure the allocator did not report is not a measurement.
    # Whether it is reported depends on the backend and the torch version --
    # torch 2.14 populates it under cudaMallocAsync, older versions leave it
    # at zero -- so it is judged from the data, not from the backend name.
    frame = _frame(reserved=None)
    text = render(frame, capability(allocator_backend="cudaMallocAsync"))
    assert (
        "torch reserved" in text
        and "0.00 GiB" not in text.split("torch reserved")[1].splitlines()[0]
    ), "unreported reserved renders as unknown"
    assert "not reported" in text
    # Subtracting a zero reserved would have put all 40 GiB of torch
    # memory into "unexplained".
    assert frame.nontorch_residual == 60 * GIB
    assert frame.residual_is_upper_bound
    assert "UPPER BOUND" in render(frame, capability())

    frame = _frame()
    text = render(frame, capability(allocator_backend="cudaMallocAsync"))
    assert frame.nontorch_residual == 50 * GIB, "a populated reserved is trusted on any backend"
    assert not frame.residual_is_upper_bound
    assert "meaningless" not in text and "not reported" not in text

    monkeypatch.setattr(
        torch.cuda, "memory_stats", lambda i: {"allocated_bytes.all.current": 40 * GIB}
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 0)
    assert read_torch(0).reserved is None, "reserved below live is refused"
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 50 * GIB)
    assert read_torch(0).reserved == 50 * GIB


def test_report_wording_and_columns():
    frame = _frame()
    # The gap also holds blocks awaiting release on other streams;
    # torch's snapshot distinguishes active_allocated /
    # active_awaiting_free / inactive. Calling it fragmentation implies
    # it is immediately reusable.
    text = render(frame, capability())
    assert "fragmentation" not in text.lower()
    assert "not handed out" in text

    # Before tagging exists nearly all torch-live memory lands here.
    # Labelling it activation would be a systematic misattribution.
    remainder = derived_entries(frame)[0]
    assert remainder.category is Category.UNATTRIBUTED_LIVE
    assert "NOT activation" in remainder.note
    assert remainder.source is Source.DERIVED
    assert remainder.size_bytes == 40 * GIB

    # Renaming LEDGER to the longer COMPONENT once filled the source
    # column exactly and ran it into the basis column.
    from tensorrt_llm._torch.memory.monitor.report import _entry_line

    for source in Source:
        for basis in Basis:
            line = _entry_line(
                Entry(
                    category=Category.WEIGHTS,
                    size_bytes=GIB,
                    source=source,
                    placement=Placement.NON_TORCH,
                    basis=basis,
                )
            )
            assert f"{source} " in line and f"{basis} " in line, line


def test_schema_guards():
    with pytest.raises(ValueError):
        Entry(
            category=Category.WEIGHTS,
            size_bytes=-1,
            source=Source.TAG,
            placement=Placement.TORCH_DEFAULT,
            basis=Basis.PHYSICAL_BACKING,
        )
    # Only the residual may be negative.
    Entry(
        category=Category.NONTORCH_RESIDUAL,
        size_bytes=-1,
        source=Source.RESIDUAL,
        placement=Placement.NON_TORCH,
        basis=Basis.PHYSICAL_BACKING,
    )

    # Being able to read a number does not make it physical occupancy:
    # a page pool reports configured capacity for released pages.
    with pytest.raises(TypeError):
        Entry(
            category=Category.DWDP_PAGES,
            size_bytes=GIB,
            source=Source.COMPONENT,
            placement=Placement.NON_TORCH,
        )

    # Both are placements, not purposes; as categories they would double
    # count against ACTIVATION and WEIGHTS respectively.
    names = {c.value for c in Category}
    assert "cuda_graph" not in names
    assert "locality_domain" not in names
    assert Placement.TORCH_GRAPH_POOL.value == "torch_graph_pool"


def test_nvml_is_read_for_the_right_device_and_only_trusted_when_it_answers(monkeypatch):
    """torch and NVML number devices differently under CUDA_VISIBLE_DEVICES."""
    import contextlib
    import os
    import sys
    import types

    from tensorrt_llm import profiler
    from tensorrt_llm._torch.memory.monitor import capability
    from tensorrt_llm._torch.memory.monitor.capability import _nvml_available, nvml_process_usage

    monkeypatch.setattr(profiler, "pynvml_context", contextlib.nullcontext)

    class NVMLError(Exception):
        pass

    def broken():
        raise NVMLError("NVML_ERROR_UNINITIALIZED")

    def nvml(**functions):
        return types.SimpleNamespace(NVMLError=NVMLError, **functions)

    # Importable is not enough: the library has to answer.
    monkeypatch.setitem(sys.modules, "pynvml", nvml(nvmlDeviceGetCount=broken))
    assert _nvml_available() is False
    monkeypatch.setitem(sys.modules, "pynvml", nvml(nvmlDeviceGetCount=lambda: 1))
    assert _nvml_available() is True

    looked_up = []
    proc = types.SimpleNamespace(pid=os.getpid(), usedGpuMemory=7 * GIB)

    def by_index(i):
        raise AssertionError("looked up by index; that is the wrong GPU when remapped")

    fake = nvml(
        nvmlDeviceGetHandleByIndex=by_index,
        nvmlDeviceGetHandleByUUID=lambda u: looked_up.append(u) or "handle",
        nvmlDeviceGetComputeRunningProcesses_v3=lambda h: [proc],
        nvmlDeviceGetCount=lambda: 3,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(
        capability.torch.cuda,
        "get_device_properties",
        lambda i: types.SimpleNamespace(uuid="6b4fa2a7-2f22-4b34-1f72-7e6c4a551774"),
    )
    assert nvml_process_usage(0) == (7 * GIB, "")
    assert looked_up == ["GPU-6b4fa2a7-2f22-4b34-1f72-7e6c4a551774"]

    # A process NVML does not list (another PID namespace) is unknown, with the reason.
    proc.pid = -1
    used, reason = nvml_process_usage(0)
    assert used is None and "not in NVML's process list" in reason
    # So is NVML failing outright.
    fake.nvmlDeviceGetComputeRunningProcesses_v3 = lambda h: broken()
    assert nvml_process_usage(0) == (None, capability.NVML_UNAVAILABLE)


def test_an_unknown_diagnostic_entry_is_not_an_unmeasured_provider():
    """An inconsistent KV pool: capacity counted, its in-use split unknown."""
    entries = [
        _entry(Category.KV_CACHE_POOL, 16 * GIB, owner="kv_cache.main"),
        _entry(
            Category.KV_CACHE_POOL,
            None,
            owner="kv_cache.main",
            basis=Basis.LOGICAL,
            counts_toward_total=False,
        ),
    ]
    frame = build_frame(
        DeviceReading(total=80 * GIB, free=60 * GIB, process_used=20 * GIB),
        TorchReading(reserved=2 * GIB, live=GIB),
        entries,
    )
    assert frame.unknown_categories == []
    text = render(frame, capability())
    assert "[unmeasured]" not in text

    # A counted entry of unknown size is listed, by owner.
    entries.append(_entry(Category.COMM_MNNVL, None, owner="mnnvl"))
    frame = build_frame(frame.device, frame.torch, entries)
    text = render(frame, capability())
    assert "[unmeasured]" in text and "mnnvl (comm.mnnvl)" in text
