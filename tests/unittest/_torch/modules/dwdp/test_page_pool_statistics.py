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
currently *holds*. ``num_pages`` keeps its value for the pool's whole
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


@pytest.fixture
def unreleased(monkeypatch):
    """The process-level count of bytes that failed to release, starting at 0."""
    import tensorrt_llm._torch.modules.dwdp.page_pool as mod

    monkeypatch.setattr(mod, "_unreleased_bytes", 0)
    return mod.unreleased_page_bytes


def _fail_every(monkeypatch, nth: int) -> None:
    """Make every ``nth`` handle release raise."""
    import tensorrt_llm._torch.modules.dwdp.page_pool as mod

    calls = {"n": 0}
    real = mod.release_handle

    def flaky(handle):
        calls["n"] += 1
        if calls["n"] % nth == 0:
            raise RuntimeError("synthetic release failure")
        return real(handle)

    monkeypatch.setattr(mod, "release_handle", flaky)


def test_held_pages_follow_release_and_num_pages_does_not(pool):
    configured = [pool.num_pages(s) for s in range(pool.num_slots)]
    assert all(n > 0 for n in configured)
    assert [pool.held_pages(s) for s in range(pool.num_slots)] == configured

    before = _device_used()
    pool.release()
    torch.cuda.synchronize()
    returned = before - _device_used()

    assert [pool.held_pages(s) for s in range(pool.num_slots)] == [0] * pool.num_slots, (
        "held_pages counts the handles actually owned"
    )
    assert [pool.num_pages(s) for s in range(pool.num_slots)] == configured, (
        "num_pages is the configuration and survives release"
    )
    # The pages really went back to the driver, which is what held_pages says.
    assert returned >= sum(configured) * pool.page_size * 0.9


def test_release_failures_are_counted_per_pool_and_per_process(monkeypatch, pool, unreleased):
    clean = PagePool.create(slot_sizes=[16 * MB], device_id=0)
    clean.release()
    assert clean.num_release_failures == 0 and unreleased() == 0, "a clean release counts nothing"

    _fail_every(monkeypatch, 2)
    total = sum(pool.num_pages(s) for s in range(pool.num_slots))
    pool.release()
    assert pool.num_release_failures == total // 2
    assert unreleased() == total // 2 * pool.page_size
    assert sum(pool.held_pages(s) for s in range(pool.num_slots)) == 0

    pool.release()  # idempotent: the handles are not walked again
    assert pool.num_release_failures == total // 2
    assert unreleased() == total // 2 * pool.page_size


def test_slot_count_survives_release():
    # release() empties every slot it was created with, however many there are.
    p = PagePool.create(slot_sizes=[16 * MB] * 3, device_id=0)
    try:
        assert p.num_slots == 3
        p.release()
        assert p.num_slots == 3
        assert [p.held_pages(s) for s in range(3)] == [0, 0, 0]
    finally:
        p.release()


def test_monitor_reports_held_pages_and_lists_failed_releases_apart(monkeypatch, pool, unreleased):
    from tensorrt_llm._torch.memory.monitor.providers.misc import DwdpPagePoolProvider
    from tensorrt_llm._torch.memory.monitor.schema import Basis

    provider = DwdpPagePoolProvider()
    monkeypatch.setattr(provider, "_pool", lambda: pool)

    (entry,) = provider.collect()
    assert entry.basis is Basis.PHYSICAL_BACKING and entry.size_bytes > 0

    _fail_every(monkeypatch, 1)
    pool.release()
    held, failed = provider.collect()
    assert held.size_bytes == 0, "released pages are no longer held"
    # Whether the GPU still backs these is unknown, so they are listed, not summed.
    assert failed.size_bytes == unreleased() > 0
    assert failed.counts_toward_total is False

    # The pool is dropped after release in production; the failed bytes stay visible.
    monkeypatch.setattr(provider, "_pool", lambda: None)
    assert provider.available()
    assert [e.size_bytes for e in provider.collect()] == [unreleased()]
