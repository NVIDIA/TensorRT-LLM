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
"""The summary a reader sees first.

It has to be readable without knowing the allocators, and it has to add up:
if the rows did not sum to the process total, a reader would either distrust
every number or, worse, not notice. CPU only.
"""

import pytest
from memory_monitor_test_utils import GIB, capability

from tensorrt_llm._torch.memory.monitor.frame import DeviceReading, TorchReading, build_frame
from tensorrt_llm._torch.memory.monitor.report import render
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source
from tensorrt_llm._torch.memory.monitor.summary import GROUP_OF, Group, group_totals

pytestmark = pytest.mark.cpu_only


def _entry(category, size, placement=Placement.NON_TORCH, **kw):
    kw.setdefault("source", Source.COMPONENT)
    kw.setdefault("basis", Basis.PHYSICAL_BACKING)
    return Entry(category=category, size_bytes=size, placement=placement, **kw)


def _frame(entries, *, process=20 * GIB, reserved=2 * GIB, live=1 * GIB):
    return build_frame(
        DeviceReading(total=180 * GIB, free=150 * GIB, process_used=process),
        TorchReading(reserved=reserved, live=live),
        entries,
    )


def _typical(**kw):
    """16 GiB KV cache, 0.5 GiB of tagged weights and 1 GiB of MNNVL."""
    return _frame(
        [
            _entry(Category.KV_CACHE_POOL, 16 * GIB),
            _entry(
                Category.WEIGHTS,
                GIB // 2,
                Placement.TORCH_DEFAULT,
                source=Source.TAG,
                basis=Basis.LOGICAL,
            ),
            _entry(Category.COMM_MNNVL, GIB),
        ],
        **kw,
    )


def _baseline():
    return dict(group_totals(_typical()).sizes)


def _rows_sum(totals):
    return sum(s for s in totals.sizes.values() if s is not None)


def test_groups_add_up_and_every_category_has_a_group():
    # A category without a group would vanish from the summary and the
    # rows would stop adding up to the process total.
    missing = [c for c in Category if c not in GROUP_OF]
    assert not missing, f"no summary group for {missing}"

    totals = group_totals(_typical())
    assert _rows_sum(totals) == totals.process_total
    sizes = totals.sizes
    assert sizes[Group.KV_CACHE] == 16 * GIB
    assert sizes[Group.WEIGHTS] == GIB // 2
    assert sizes[Group.COMMUNICATION] == GIB
    assert sizes[Group.TORCH_UNATTRIBUTED] == GIB // 2  # live 1 GiB - 0.5 tagged
    assert sizes[Group.TORCH_CACHE] == GIB  # reserved 2 - live 1
    assert sizes[Group.OVERHEAD] == 20 * GIB - 2 * GIB - 17 * GIB

    totals = group_totals(
        _frame(
            [
                _entry(Category.KV_CACHE_POOL, 16 * GIB),
                _entry(Category.CPP_BUFFERS, 5 * GIB, counts_toward_total=False),
            ]
        )
    )
    assert _rows_sum(totals) == totals.process_total, "diagnostic-only entries are not summed"


def test_summary_is_readable_without_allocator_terms():
    text = render(_typical(), capability())
    summary, _, details = text.partition("--- details by allocator ---")
    assert details, "allocator detail must still be there, below the summary"
    for plain in ("KV cache", "Model weights", "Communication buffers"):
        assert plain in summary
    # The detail vocabulary stays out of the summary.
    for jargon in ("nontorch_residual", "unattributed_live", "component counters", "basis"):
        assert jargon not in summary

    order = [g.value for g in Group if g.value in summary]
    assert sorted(order, key=summary.index)[0] == "KV cache", "largest first"

    # advice=None with a headline once raised TypeError on the OOM path.
    text = render(_typical(), capability(), headline="OUT OF MEMORY: example")
    assert text.index("OUT OF MEMORY") < text.index("KV cache"), "headline above the table"

    # Its bytes are not missing from the device -- they fall into the
    # residual. Without the note, an unmeasured KV pool would read as
    # CUDA overhead.
    frame = _frame([_entry(Category.KV_CACHE_POOL, None, owner="kv_cache.main (KVCacheManager)")])
    summary = render(frame, capability()).partition("--- details")[0]
    overhead = next(line for line in summary.splitlines() if Group.OVERHEAD.value in line)
    assert "could not be measured" in overhead and "kv_cache.main" in overhead

    frame = _frame([_entry(Category.KV_CACHE_POOL, 16 * GIB), _entry(Category.COMM_MNNVL, 1024)])
    assert Group.COMMUNICATION.value not in render(frame, capability()).partition("---")[0], (
        "rows that round to zero are left out"
    )


def test_conclusion_names_what_grew_since_warmup():
    text = render(_typical(), capability())
    assert "Largest: KV cache" in text, "without a baseline it names the largest"
    assert "since warmup" not in text

    text = render(_typical(reserved=6 * GIB, live=5 * GIB), capability(), baseline=_baseline())
    assert "since warmup" in text
    assert "PyTorch, unattributed grew the most (+4.00 GiB)" in text

    # Warmup can leave the allocator's free cache large or small; its
    # change says little about what the workload did.
    text = render(_typical(reserved=4 * GIB, live=1 * GIB), capability(), baseline=_baseline())
    assert "PyTorch cache (free) grew" not in text

    text = render(_typical(), capability(), baseline=_baseline())
    assert "nothing has grown since warmup" in text

    # Seen on a real run: "grew the most (+0.00 GiB)".
    baseline = _baseline()
    for group in baseline:
        if baseline[group] is not None and group is not Group.TORCH_CACHE:
            baseline[group] -= 1 << 20
    text = render(_typical(), capability(), baseline=baseline)
    assert "grew the most" not in text, "growth too small to print is not named"
    assert "nothing has grown since warmup" in text


def test_suggestions():
    from tensorrt_llm._torch.memory.monitor.summary import render_summary

    text = render(
        _typical(),
        capability(),
        advice={Group.KV_CACHE: "reduce X"},
        headline="OUT OF MEMORY: example",
    )
    assert "Suggestion: reduce X." in text, "advice follows the focus group on a failure"
    # A KV cache that is the largest part is the design, not a problem.
    text = render(_typical(), capability(), advice={Group.KV_CACHE: "reduce X"})
    assert "Suggestion" not in text, "no advice for a healthy report"

    # Growth after warmup, which already ran the largest batch, is specific
    # evidence; the suggestion should use it rather than generic row advice.
    def summary(frame, **kw):
        advice = {Group.TORCH_UNATTRIBUTED: "reduce max_num_tokens"}
        return "\n".join(
            render_summary(frame, group_totals(frame), baseline=_baseline(), advice=advice, **kw)
        )

    def grown(unattributed):
        # The typical frame with ``unattributed`` more PyTorch memory in use.
        return _typical(
            process=20 * GIB + unattributed,
            reserved=2 * GIB + unattributed,
            live=GIB + unattributed,
        )

    text = summary(grown(8 * GIB), headline="OUT OF MEMORY")
    assert "grew 8.00 GiB after warmup" in text
    assert "TLLM_MEM_MONITOR_STACKS=1" in text
    # The executor's own advice stays, for the part that is activations.
    assert "to reduce activation memory, reduce max_num_tokens" in text

    sites = {"my_app/processors.py:__call__": 6 * GIB, "modules/linear.py:forward": GIB}
    text = summary(grown(8 * GIB), headline="OUT OF MEMORY", unattributed_sites=sites)
    assert "allocated by my_app/processors.py:__call__ (6.00 GiB)" in text, "stacks name code"
    assert "TLLM_MEM_MONITOR_STACKS" not in text

    # A periodic report: no headline, but 2 GiB held since warmup.
    text = summary(grown(2 * GIB))
    assert "Suggestion: PyTorch memory with no known owner grew 2.00 GiB" in text
    assert "Suggestion" not in summary(grown(GIB // 2)), "small growth without a failure"

    text = summary(_typical(process=23 * GIB), headline="OUT OF MEMORY")
    assert "memory outside PyTorch grew 3.00 GiB after warmup" in text


def test_activations_of_the_failing_step_are_not_called_held_memory(monkeypatch):
    """The baseline is at rest; an OOM report is taken inside a forward pass."""
    from tensorrt_llm._torch.memory.monitor import summary as summary_module
    from tensorrt_llm._torch.memory.monitor.summary import render_summary

    advice = {Group.TORCH_UNATTRIBUTED: "reduce max_num_tokens"}

    def oom_summary(unattributed, activation):
        monkeypatch.setattr(summary_module, "_profiled_activation", lambda: activation)
        frame = _typical(
            process=20 * GIB + unattributed,
            reserved=2 * GIB + unattributed,
            live=GIB + unattributed,
        )
        return "\n".join(
            render_summary(
                frame,
                group_totals(frame),
                baseline=_baseline(),
                advice=advice,
                headline="OUT OF MEMORY",
            )
        )

    text = oom_summary(3 * GIB, activation=4 * GIB)
    assert "within the 4.00 GiB the profiled forward pass allocated" in text, "activations"
    assert "leak" not in text and "TLLM_MEM_MONITOR_STACKS" not in text
    assert "reduce max_num_tokens" in text

    text = oom_summary(10 * GIB, activation=4 * GIB)
    assert "6.00 GiB more than the profiled forward pass allocated (4.00 GiB)" in text, "excess"
    assert "TLLM_MEM_MONITOR_STACKS=1" in text

    # Without a failure, growth is named only when the part beyond the
    # activations is large.
    def periodic(unattributed, activation):
        monkeypatch.setattr(summary_module, "_profiled_activation", lambda: activation)
        frame = _typical(
            process=20 * GIB + unattributed,
            reserved=2 * GIB + unattributed,
            live=GIB + unattributed,
        )
        return "\n".join(
            render_summary(frame, group_totals(frame), baseline=_baseline(), advice=advice)
        )

    assert "Suggestion" not in periodic(2 * GIB, activation=4 * GIB), "within activations"
    assert "Suggestion" in periodic(6 * GIB, activation=4 * GIB), "2 GiB beyond activations"


def test_a_group_absent_at_warmup_counts_as_grown_from_zero():
    from tensorrt_llm._torch.memory.monitor.summary import render_summary

    frame = _typical(process=28 * GIB, reserved=10 * GIB, live=9 * GIB)
    baseline = _baseline()
    baseline.pop(Group.TORCH_UNATTRIBUTED, None)
    text = "\n".join(render_summary(frame, group_totals(frame), baseline=baseline))
    row = next(line for line in text.splitlines() if "PyTorch, unattributed" in line)
    grown = group_totals(frame).sizes[Group.TORCH_UNATTRIBUTED]
    assert f"{grown / GIB:+.2f} GiB" in row, "the table shows the whole row as growth"
    assert "PyTorch, unattributed grew the most" in text, "and so does the conclusion"

    # An unknown baseline value is not zero: no growth is claimed for it.
    baseline[Group.TORCH_UNATTRIBUTED] = None
    text = "\n".join(render_summary(frame, group_totals(frame), baseline=baseline))
    assert "PyTorch, unattributed grew" not in text
