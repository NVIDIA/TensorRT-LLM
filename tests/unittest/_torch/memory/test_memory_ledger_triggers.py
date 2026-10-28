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
"""Trigger points, and the provider registry they depend on.

Without these the ledger is unreachable in practice: the memory sits in the
executor's process while the obvious place to ask for a report is the parent,
where there is nothing to see. A report that never reaches anyone is the same
as no report.
"""

import gc
import types

import pytest
import torch

from tensorrt_llm._torch.memory.ledger.providers import (
    clear_registered_providers,
    collect_all,
    default_providers,
    register_provider,
)
from tensorrt_llm._torch.memory.ledger.providers.kv_cache import KvCacheProvider
from tensorrt_llm._torch.memory.ledger.schema import Basis, Category
from tensorrt_llm._torch.memory.ledger.triggers import (
    install_oom_reporter,
    report_now,
    uninstall_oom_reporter,
)

MB = 1 << 20


def _stats(total, free, evictable, slot_sizes=(MB,)):
    return types.SimpleNamespace(
        total=total, free=free, evictable=evictable, slot_sizes=tuple(slot_sizes)
    )


class _FakeManager:
    def __init__(self, *stats):
        self._stats = list(stats)

    def _get_storage_statistics(self, level):
        return self._stats


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registered_providers()
    uninstall_oom_reporter()
    yield
    clear_registered_providers()
    uninstall_oom_reporter()


class TestRegistry:
    def test_registered_provider_is_collected(self):
        manager = _FakeManager(_stats(100, 40, 10))
        register_provider(KvCacheProvider(manager))

        entries = collect_all()
        kv = [e for e in entries if e.category is Category.KV_CACHE_POOL]
        assert kv, "a registered provider must reach the report"
        physical = [e for e in kv if e.basis is Basis.PHYSICAL_BACKING]
        assert physical[0].size_bytes == 100 * MB

    def test_absent_without_registration(self):
        # The default set cannot find the manager, so the KV rows are simply
        # missing -- which is what made a real run attribute the pool to the
        # residual bucket.
        assert not any(isinstance(p, KvCacheProvider) for p in default_providers())
        assert not [e for e in collect_all() if e.category is Category.KV_CACHE_POOL]

    def test_reregistering_replaces_rather_than_duplicates(self):
        first = _FakeManager(_stats(100, 0, 0))
        second = _FakeManager(_stats(200, 0, 0))
        register_provider(KvCacheProvider(first))
        register_provider(KvCacheProvider(second))

        physical = [
            e
            for e in collect_all()
            if e.category is Category.KV_CACHE_POOL and e.basis is Basis.PHYSICAL_BACKING
        ]
        assert len(physical) == 1, "a rebuilt manager must supersede the old one"
        assert physical[0].size_bytes == 200 * MB


class TestProviderDoesNotPinThePool:
    """The pool is the largest allocation in the process."""

    def test_manager_is_held_weakly(self):
        manager = _FakeManager(_stats(100, 0, 0))
        provider = KvCacheProvider(manager)
        assert provider.available()

        del manager
        gc.collect()

        # Keeping it alive would stop tens of GiB from ever being released.
        assert not provider.available(), "the ledger must not keep the KV cache manager alive"
        assert provider.collect() == []

    def test_dead_manager_is_skipped_not_reported_as_zero(self):
        manager = _FakeManager(_stats(100, 0, 0))
        register_provider(KvCacheProvider(manager))
        del manager
        gc.collect()

        assert not [e for e in collect_all() if e.category is Category.KV_CACHE_POOL]


class TestReportNow:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("TLLM_MEM_LEDGER", raising=False)
        assert report_now(tag="t", rank=0) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_returns_text_when_enabled(self, monkeypatch):
        monkeypatch.setenv("TLLM_MEM_LEDGER", "1")
        text = report_now(tag="unit", rank=0)
        assert text is not None and "memory ledger" in text

    def test_a_broken_provider_does_not_break_the_report(self, monkeypatch):
        monkeypatch.setenv("TLLM_MEM_LEDGER", "1")
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU")

        class _Boom:
            name = "boom"
            category = Category.KV_CACHE_POOL

            def available(self):
                return True

            def collect(self):
                raise RuntimeError("provider exploded")

        register_provider(_Boom())
        assert report_now(tag="unit", rank=0) is not None


class TestOomReporter:
    def test_not_installed_when_disabled(self, monkeypatch):
        monkeypatch.delenv("TLLM_MEM_LEDGER", raising=False)
        assert install_oom_reporter(rank=0) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_installs_once_and_stays_installed(self, monkeypatch):
        monkeypatch.setenv("TLLM_MEM_LEDGER", "1")
        attaches = []
        monkeypatch.setattr(
            torch._C, "_cuda_attach_out_of_memory_observer", lambda cb: attaches.append(cb)
        )

        assert install_oom_reporter(rank=0) is True
        assert install_oom_reporter(rank=0) is True
        # torch keeps every observer it is handed, so a second attach would
        # print every report twice.
        assert len(attaches) == 1

    def test_reports_once_per_process_by_default(self, monkeypatch):
        # Exercise the callback directly: the attach flag deliberately never
        # resets, so a test cannot rely on capturing it from a second
        # install.
        import tensorrt_llm._torch.memory.ledger.triggers as mod

        monkeypatch.setenv("TLLM_MEM_LEDGER", "1")
        monkeypatch.setattr(mod, "_enabled_flag", True)
        monkeypatch.setattr(mod, "_reports_emitted", 0)
        emitted = []
        monkeypatch.setattr(mod, "report_now", lambda **kw: emitted.append(kw) or "report")

        for _ in range(5):
            mod._on_out_of_memory(0, 1 << 30, 1 << 30, 1 << 30)

        # A retry loop that OOMs repeatedly would otherwise bury the first
        # and most informative report under identical copies.
        assert len(emitted) == 1

    def test_limit_is_configurable(self, monkeypatch):
        import tensorrt_llm._torch.memory.ledger.triggers as mod

        monkeypatch.setenv("TLLM_MEM_LEDGER", "1")
        monkeypatch.setenv("TLLM_MEM_LEDGER_OOM_REPORTS", "3")
        monkeypatch.setattr(mod, "_enabled_flag", True)
        monkeypatch.setattr(mod, "_reports_emitted", 0)
        emitted = []
        monkeypatch.setattr(mod, "report_now", lambda **kw: emitted.append(kw) or "report")

        for _ in range(5):
            mod._on_out_of_memory(0, 1 << 30, 1 << 30, 1 << 30)
        assert len(emitted) == 3

    def test_inert_once_uninstalled(self, monkeypatch):
        import tensorrt_llm._torch.memory.ledger.triggers as mod

        monkeypatch.setattr(mod, "_enabled_flag", False)
        monkeypatch.setattr(mod, "_reports_emitted", 0)
        emitted = []
        monkeypatch.setattr(mod, "report_now", lambda **kw: emitted.append(kw) or "report")

        # torch cannot detach an observer, so the callback stays attached
        # and must simply do nothing.
        mod._on_out_of_memory(0, 1 << 30, 1 << 30, 1 << 30)
        assert emitted == []

    def test_a_failing_report_does_not_mask_the_oom(self, monkeypatch):
        import tensorrt_llm._torch.memory.ledger.triggers as mod

        monkeypatch.setattr(mod, "_enabled_flag", True)
        monkeypatch.setattr(mod, "_reports_emitted", 0)

        def explode(**kwargs):
            raise RuntimeError("reporting failed")

        monkeypatch.setattr(mod, "report_now", explode)
        # Raising here would replace an out-of-memory error, which the user
        # needs to see, with a diagnostic's own failure.
        mod._on_out_of_memory(0, 1 << 30, 1 << 30, 1 << 30)
