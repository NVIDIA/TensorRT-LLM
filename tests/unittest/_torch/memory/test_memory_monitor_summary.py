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

from tensorrt_llm._torch.memory.monitor.capability import Capability
from tensorrt_llm._torch.memory.monitor.frame import DeviceReading, TorchReading, build_frame
from tensorrt_llm._torch.memory.monitor.report import render
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category, Entry, Placement, Source
from tensorrt_llm._torch.memory.monitor.summary import GROUP_OF, Group, group_totals

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30


def _capability():
    return Capability(
        allocator_backend="native",
        alloc_conf="",
        alloc_conf_var=None,
        expandable_segments=False,
        metadata_supported=True,
        history_api_present=True,
        snapshot_api_present=True,
        oom_observer_present=True,
        nvml_available=True,
    )


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


def _typical():
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
        ]
    )


class TestGroupsAddUp:
    def test_every_category_has_a_group(self):
        # A category without a group would vanish from the summary and the
        # rows would stop adding up to the process total.
        missing = [c for c in Category if c not in GROUP_OF]
        assert not missing, f"no summary group for {missing}"

    def test_rows_sum_to_the_process_total(self):
        totals = group_totals(_typical())
        assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total

    def test_each_part_lands_in_its_group(self):
        sizes = group_totals(_typical()).sizes
        assert sizes[Group.KV_CACHE] == 16 * GIB
        assert sizes[Group.WEIGHTS] == GIB // 2
        assert sizes[Group.COMMUNICATION] == GIB
        assert sizes[Group.TORCH_UNATTRIBUTED] == GIB // 2  # live 1 GiB - 0.5 tagged
        assert sizes[Group.TORCH_CACHE] == GIB  # reserved 2 - live 1
        assert sizes[Group.OVERHEAD] == 20 * GIB - 2 * GIB - 17 * GIB

    def test_diagnostic_only_entries_are_not_summed(self):
        frame = _frame(
            [
                _entry(Category.KV_CACHE_POOL, 16 * GIB),
                _entry(Category.CPP_BUFFERS, 5 * GIB, counts_toward_total=False),
            ]
        )
        totals = group_totals(frame)
        assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total


class TestReadableWithoutAllocatorTerms:
    def test_summary_comes_first_and_uses_plain_names(self):
        text = render(_typical(), _capability())
        summary, _, details = text.partition("--- details by allocator ---")
        assert details, "allocator detail must still be there, below the summary"
        for plain in ("KV cache", "Model weights", "Communication buffers"):
            assert plain in summary
        # The detail vocabulary stays out of the summary.
        for jargon in ("nontorch_residual", "unattributed_live", "component counters", "basis"):
            assert jargon not in summary

    def test_largest_first(self):
        text = render(_typical(), _capability()).partition("--- details")[0]
        order = [g.value for g in Group if g.value in text]
        positions = sorted(order, key=text.index)
        assert positions[0] == "KV cache"

    def test_unmeasured_memory_is_named_in_the_overhead_row(self):
        # Its bytes are not missing from the device -- they fall into the
        # residual. Without the note, an unmeasured KV pool would read as
        # CUDA overhead.
        frame = _frame(
            [
                _entry(Category.KV_CACHE_POOL, None, owner="kv_cache.main (KVCacheManager)"),
            ]
        )
        summary = render(frame, _capability()).partition("--- details")[0]
        overhead = next(line for line in summary.splitlines() if Group.OVERHEAD.value in line)
        assert "could not be measured" in overhead and "kv_cache.main" in overhead

    def test_headline_is_shown_above_the_table(self):
        text = render(_typical(), _capability(), headline="OUT OF MEMORY: example")
        assert text.index("OUT OF MEMORY") < text.index("KV cache")


class TestBaselineDelta:
    def _baseline(self):
        return dict(group_totals(_typical()).sizes)

    def test_without_a_baseline_it_names_the_largest(self):
        text = render(_typical(), _capability())
        assert "Largest: KV cache" in text
        assert "since warmup" not in text

    def test_names_the_group_that_grew(self):
        baseline = self._baseline()
        grown = _frame(
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
            reserved=6 * GIB,
            live=5 * GIB,
        )
        text = render(grown, _capability(), baseline=baseline)
        assert "since warmup" in text
        assert "PyTorch, unattributed grew the most (+4.00 GiB)" in text

    def test_cache_slack_does_not_drive_the_conclusion(self):
        # Warmup can leave the allocator's free cache large or small; its
        # change says little about what the workload did.
        baseline = self._baseline()
        frame = _frame(
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
            reserved=4 * GIB,
            live=1 * GIB,
        )
        text = render(frame, _capability(), baseline=baseline)
        assert "PyTorch cache (free) grew" not in text

    def test_nothing_grew(self):
        text = render(_typical(), _capability(), baseline=self._baseline())
        assert "nothing has grown since warmup" in text

    def test_growth_too_small_to_print_is_not_named(self):
        # Seen on a real run: "grew the most (+0.00 GiB)".
        baseline = self._baseline()
        for group in baseline:
            if baseline[group] is not None and group is not Group.TORCH_CACHE:
                baseline[group] -= 1 << 20
        text = render(_typical(), _capability(), baseline=baseline)
        assert "grew the most" not in text
        assert "nothing has grown since warmup" in text

    def test_advice_follows_the_focus_group_when_something_went_wrong(self):
        text = render(
            _typical(),
            _capability(),
            advice={Group.KV_CACHE: "reduce X"},
            headline="OUT OF MEMORY: example",
        )
        assert "Suggestion: reduce X." in text

    def test_no_advice_for_a_healthy_report(self):
        # A KV cache that is the largest part is the design, not a problem.
        text = render(_typical(), _capability(), advice={Group.KV_CACHE: "reduce X"})
        assert "Suggestion" not in text

    def test_headline_without_advice_does_not_crash(self):
        # advice=None with a headline once raised TypeError on the OOM path.
        text = render(_typical(), _capability(), headline="OUT OF MEMORY: example")
        assert "OUT OF MEMORY" in text

    def test_rounds_to_zero_rows_are_left_out(self):
        frame = _frame(
            [_entry(Category.KV_CACHE_POOL, 16 * GIB), _entry(Category.COMM_MNNVL, 1024)]
        )
        assert Group.COMMUNICATION.value not in render(frame, _capability()).partition("---")[0]


@pytest.mark.cpu_only
class TestGrowthSuggestion:
    """Growth after warmup, which already ran the largest batch, is specific
    evidence; the suggestion should use it rather than generic row advice."""

    def _grown(self, unattributed):
        # The typical frame with ``unattributed`` more PyTorch memory in use.
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
            process=20 * GIB + unattributed,
            reserved=2 * GIB + unattributed,
            live=GIB + unattributed,
        )

    def _summary(self, frame, **kw):
        from tensorrt_llm._torch.memory.monitor.summary import render_summary

        baseline = dict(group_totals(_typical()).sizes)
        advice = {Group.TORCH_UNATTRIBUTED: "reduce max_num_tokens"}
        return "\n".join(
            render_summary(frame, group_totals(frame), baseline=baseline, advice=advice, **kw)
        )

    def test_oom_after_growth_points_at_held_memory(self):
        text = self._summary(self._grown(8 * GIB), headline="OUT OF MEMORY")
        assert "grew 8.00 GiB after warmup" in text
        assert "TLLM_MEM_MONITOR_STACKS=1" in text
        # The executor's own advice stays, as the fallback.
        assert "if it is activation memory, reduce max_num_tokens" in text

    def test_recorded_stacks_name_the_code(self):
        sites = {"my_app/processors.py:__call__": 6 * GIB, "modules/linear.py:forward": GIB}
        text = self._summary(
            self._grown(8 * GIB), headline="OUT OF MEMORY", unattributed_sites=sites
        )
        assert "allocated by my_app/processors.py:__call__ (6.00 GiB)" in text
        assert "TLLM_MEM_MONITOR_STACKS" not in text

    def test_large_growth_is_named_before_anything_fails(self):
        # A periodic report: no headline, but 2 GiB held since warmup.
        text = self._summary(self._grown(2 * GIB))
        assert "Suggestion: PyTorch memory with no known owner grew 2.00 GiB" in text

    def test_small_growth_without_a_failure_is_left_alone(self):
        text = self._summary(self._grown(GIB // 2))
        assert "Suggestion" not in text

    def test_growth_outside_pytorch(self):
        frame = _frame(
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
            process=23 * GIB,
        )
        text = self._summary(frame, headline="OUT OF MEMORY")
        assert "memory outside PyTorch grew 3.00 GiB after warmup" in text
