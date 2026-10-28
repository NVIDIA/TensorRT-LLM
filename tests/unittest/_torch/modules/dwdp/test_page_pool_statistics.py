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
"""Read-only statistics on :class:`PagePool`.

The distinction under test is between what a pool is *sized for* and what it
currently *holds*. ``configured_pages`` keeps its value for the pool's whole
lifetime; ``held_pages`` goes to zero the moment ``release()`` hands the
pages back. Conflating them reports a full pool's worth of memory that the
driver has already taken back.

Requires a GPU: the pages are real VMM allocations, and the device-side
accounting is the point.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp.page_pool import PagePool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PagePool allocates real VMM pages"
)

MB = 1 << 20


@pytest.fixture
def pool():
    p = PagePool.create(slot_sizes=[64 * MB, 64 * MB], device_id=0)
    yield p
    p.release()


def _device_used() -> int:
    free, total = torch.cuda.mem_get_info(0)
    return total - free


class TestConfiguredVersusHeld:
    def test_both_agree_before_release(self, pool):
        for slot in range(pool.num_slots):
            assert pool.held_pages(slot) == pool.configured_pages(slot)
            assert pool.held_pages(slot) > 0

    def test_held_drops_to_zero_and_configured_does_not(self, pool):
        configured = [pool.configured_pages(s) for s in range(pool.num_slots)]
        assert sum(configured) > 0

        pool.release()

        assert [pool.held_pages(s) for s in range(pool.num_slots)] == [0] * pool.num_slots, (
            "held_pages must track the handles actually owned; reporting the "
            "configured count here invents a full pool of memory that the "
            "driver has already taken back"
        )
        assert [pool.configured_pages(s) for s in range(pool.num_slots)] == configured

    def test_held_pages_matches_the_device(self, pool):
        before = _device_used()
        pool.release()
        torch.cuda.synchronize()
        returned = before - _device_used()

        # The pages were really handed back, so a non-zero occupancy report
        # after this point would contradict the device itself.
        expected = sum(pool.configured_pages(s) for s in range(pool.num_slots))
        assert returned >= expected * pool.page_size * 0.9
        assert sum(pool.held_pages(s) for s in range(pool.num_slots)) == 0

    def test_num_pages_still_means_configured(self, pool):
        # Kept for back-compat and deliberately *not* repurposed: callers
        # reading it expect a configuration value.
        pool.release()
        for slot in range(pool.num_slots):
            assert pool.num_pages(slot) == pool.configured_pages(slot)


class TestReleaseFailuresAreCounted:
    def test_zero_on_a_clean_release(self, pool):
        pool.release()
        assert pool.release_failed == 0

    def test_counts_handles_that_raised(self, monkeypatch, pool):
        # release() logs and continues so one bad handle cannot derail
        # teardown; the count is what keeps that from being invisible.
        import tensorrt_llm._torch.modules.dwdp.page_pool as mod

        calls = {"n": 0}
        real = mod.release_handle

        def flaky(handle):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("synthetic release failure")
            return real(handle)

        monkeypatch.setattr(mod, "release_handle", flaky)

        total = sum(pool.configured_pages(s) for s in range(pool.num_slots))
        pool.release()

        assert pool.release_failed == total // 2
        assert sum(pool.held_pages(s) for s in range(pool.num_slots)) == 0

    def test_release_stays_idempotent_after_failures(self, monkeypatch, pool):
        import tensorrt_llm._torch.modules.dwdp.page_pool as mod

        monkeypatch.setattr(
            mod, "release_handle", lambda h: (_ for _ in ()).throw(RuntimeError("x"))
        )

        pool.release()
        first = pool.release_failed
        pool.release()  # must not re-walk the handles
        assert pool.release_failed == first


class TestSlotCountSurvivesRelease:
    def test_num_slots_unchanged(self, pool):
        before = pool.num_slots
        pool.release()
        assert pool.num_slots == before

    def test_three_slots_do_not_collapse_to_two(self):
        # release() used to reset the handles to a fixed-length [[], []],
        # which silently reshaped any pool that was not double buffered.
        p = PagePool.create(slot_sizes=[16 * MB] * 3, device_id=0)
        try:
            assert p.num_slots == 3
            p.release()
            assert p.num_slots == 3
            assert [p.held_pages(s) for s in range(3)] == [0, 0, 0]
        finally:
            p.release()


class TestMonitorProviderUsesHeldPages:
    def test_reports_zero_after_release(self, monkeypatch, pool):
        from tensorrt_llm._torch.memory.monitor.providers.misc import DwdpPagePoolProvider
        from tensorrt_llm._torch.memory.monitor.schema import Basis

        provider = DwdpPagePoolProvider()
        monkeypatch.setattr(provider, "_pool", lambda: pool)

        entry = provider.collect()[0]
        assert entry.basis is Basis.PHYSICAL_BACKING
        assert entry.size_bytes > 0

        pool.release()
        assert provider.collect()[0].size_bytes == 0

    def test_release_failure_is_surfaced_as_a_lower_bound(self, monkeypatch, pool):
        import tensorrt_llm._torch.modules.dwdp.page_pool as mod
        from tensorrt_llm._torch.memory.monitor.providers.misc import DwdpPagePoolProvider

        monkeypatch.setattr(
            mod, "release_handle", lambda h: (_ for _ in ()).throw(RuntimeError("x"))
        )
        pool.release()

        provider = DwdpPagePoolProvider()
        monkeypatch.setattr(provider, "_pool", lambda: pool)
        entry = provider.collect()[0]

        # The pages are untracked but plausibly still mapped, so a bare 0
        # would read as a clean release.
        assert "at least" in (entry.note or ""), (
            "a lost handle must not be reported as a clean zero"
        )
