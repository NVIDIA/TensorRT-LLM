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
"""Tag-based attribution of torch-visible memory.

The traps these pin down were all found by measurement, and each is silent:
the ring buffer drops the earliest tags (which are the weights), a non-native
allocator accepts a tag and discards it, and an unsupported setup produces a
breakdown of zeros that reads exactly like a process holding nothing.

Needs a GPU: the attribution is read back out of a real allocator snapshot.
"""

import pytest
import torch

from tensorrt_llm._torch.memory.ledger.checks import check_tag_coverage
from tensorrt_llm._torch.memory.ledger.tagging import (
    UNTAGGED,
    TagSnapshot,
    freeze_tag_map,
    live_bytes_by_tag,
    start_tagging_window,
    stop_tagging_window,
    tag,
    tagging_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reads a real allocator snapshot"
)

MB = 1 << 20


@pytest.fixture
def window():
    """A tagging window that is always closed again."""
    started = start_tagging_window()
    try:
        yield started
    finally:
        stop_tagging_window()


class TestAttribution:
    def test_tags_land_on_the_right_bytes(self, window):
        if not window:
            pytest.skip("tagging unsupported here")

        with tag("WEIGHTS"):
            weights = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
        with tag("SAMPLER"):
            sampler = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        plain = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")

        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            assert by_tag["WEIGHTS"] == 64 * MB
            assert by_tag["SAMPLER"] == 32 * MB
            assert by_tag[UNTAGGED] >= 16 * MB
        finally:
            del weights, sampler, plain

    def test_untagged_is_always_present(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("ONLY"):
            buf = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            # Reporting only the tags would read as a complete breakdown.
            assert UNTAGGED in by_tag
        finally:
            del buf

    def test_nesting_restores_the_outer_tag(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("OUTER"):
            a = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
            with tag("INNER"):
                b = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
            c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            assert by_tag.get("INNER", 0) == 16 * MB
            # Clearing instead of restoring would leave this at 16 MiB.
            assert by_tag.get("OUTER", 0) == 32 * MB
        finally:
            del a, b, c


class TestFrozenMapSurvivesTheRingBuffer:
    """Why the map is frozen rather than replayed on demand."""

    def test_attribution_holds_after_the_window_closes(self, window):
        if not window:
            pytest.skip("tagging unsupported here")

        with tag("WEIGHTS"):
            weights = torch.empty(128 * MB, dtype=torch.uint8, device="cuda")
        frozen = freeze_tag_map()
        stop_tagging_window()

        try:
            # Steady state: plenty of churn, no recording. A design that
            # replayed the trace at read time would have lost WEIGHTS by now.
            for _ in range(3000):
                scratch = torch.empty(4096, dtype=torch.uint8, device="cuda")
                del scratch
            activation = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")

            by_tag = live_bytes_by_tag(frozen)
            assert by_tag["WEIGHTS"] == 128 * MB, (
                "the frozen map must keep attributing weights once the trace has moved on"
            )
            assert by_tag[UNTAGGED] >= 64 * MB
            del activation
        finally:
            del weights


class TestUnsupportedIsNotZero:
    """An unsupported setup must not look like an empty one."""

    def test_unsupported_snapshot_yields_no_entries(self):
        empty = TagSnapshot(supported=False)
        assert live_bytes_by_tag(empty) == {}, "a dict of zeros would read as a measurement"
        assert not empty

    def test_tag_is_a_noop_when_unsupported(self, monkeypatch):
        import tensorrt_llm._torch.memory.ledger.tagging as mod

        monkeypatch.setattr(mod, "tagging_supported", lambda: False)
        # Call sites must not need a guard of their own.
        with mod.tag("WEIGHTS"):
            pass

    def test_freeze_reports_unsupported_rather_than_empty(self, monkeypatch):
        import tensorrt_llm._torch.memory.ledger.tagging as mod

        monkeypatch.setattr(mod, "tagging_supported", lambda: False)
        snapshot = mod.freeze_tag_map()
        assert snapshot.supported is False

    def test_non_native_backend_refuses(self, monkeypatch):
        # The torch docs are explicit that cudaMallocAsync and pluggable
        # allocators accept the call and drop the tag. Detecting that up
        # front is the difference between "unsupported" and a small,
        # plausible, wrong number.
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
        assert tagging_supported() is False

    def test_pluggable_allocator_raises_and_is_caught(self, monkeypatch):
        def boom():
            raise RuntimeError("pluggable allocator installed")

        monkeypatch.setattr(torch.cuda.memory, "_memory_metadata_supported", boom)
        assert tagging_supported() is False


class TestCheckB:
    def test_passes_when_tags_fit_inside_live(self):
        by_tag = {"WEIGHTS": 64 * MB, UNTAGGED: 16 * MB}
        results = check_tag_coverage(by_tag, torch_live_bytes=80 * MB)
        assert results[0].passed

    def test_fails_when_tags_exceed_live(self):
        # What a stale frozen tag looks like: an address reused by an
        # untagged allocation whose reuse event the ring buffer dropped.
        by_tag = {"WEIGHTS": 512 * MB, UNTAGGED: 0}
        results = check_tag_coverage(by_tag, torch_live_bytes=64 * MB)
        assert not results[0].passed
        assert "tagged=536870912" in results[0].detail

    def test_unsupported_is_not_a_failure(self):
        results = check_tag_coverage({}, torch_live_bytes=64 * MB)
        assert results[0].passed
        assert "not reported as zero" in results[0].detail

    def test_untagged_share_is_informational_not_a_failure(self):
        # Activations are never tagged, so a large untagged share is normal
        # and must not be raised as a problem.
        by_tag = {"WEIGHTS": 1 * MB, UNTAGGED: 99 * MB}
        results = check_tag_coverage(by_tag, torch_live_bytes=100 * MB)
        assert all(r.passed for r in results)
        assert "99.0%" in results[1].detail

    def test_real_reading_satisfies_check_b(self, window):
        if not window:
            pytest.skip("tagging unsupported here")
        with tag("WEIGHTS"):
            weights = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = live_bytes_by_tag(freeze_tag_map())
            results = check_tag_coverage(by_tag, torch.cuda.memory_allocated())
            assert all(r.passed for r in results)
        finally:
            del weights
