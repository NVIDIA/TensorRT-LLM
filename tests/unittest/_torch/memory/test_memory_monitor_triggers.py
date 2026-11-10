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

Without these the monitor is unreachable in practice: the memory sits in the
executor's process while the obvious place to ask for a report is the parent,
where there is nothing to see. A report that never reaches anyone is the same
as no report.
"""

import gc
import os
import types

import pytest
import torch
from memory_monitor_test_utils import MB, FakeKvManager, kv_stats

from tensorrt_llm._torch.memory.monitor import settings, startup
from tensorrt_llm._torch.memory.monitor.providers import (
    clear_registered_providers,
    collect_all,
    default_providers,
    register_provider,
)
from tensorrt_llm._torch.memory.monitor.providers.kv_cache import KvCacheProvider
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category
from tensorrt_llm._torch.memory.monitor.triggers import (
    install_oom_reporter,
    release_oom_reporter,
    report_now,
    suspend_reporting,
)


@pytest.mark.cpu_only
def test_registry():
    def physical():
        return [
            e
            for e in collect_all()
            if e.category is Category.KV_CACHE_POOL and e.basis is Basis.PHYSICAL_BACKING
        ]

    # The default set cannot find the manager; without a registration the
    # pool would land in the residual bucket.
    assert not any(isinstance(p, KvCacheProvider) for p in default_providers())
    assert not [e for e in collect_all() if e.category is Category.KV_CACHE_POOL]

    manager = FakeKvManager(kv_stats(100, 40, 10))
    register_provider(KvCacheProvider(manager))
    kv = [e for e in collect_all() if e.category is Category.KV_CACHE_POOL]
    assert kv, "a registered provider must reach the report"
    assert physical()[0].size_bytes == 100 * MB

    clear_registered_providers()
    first = FakeKvManager(kv_stats(100, 0, 0))
    second = FakeKvManager(kv_stats(200, 0, 0))
    register_provider(KvCacheProvider(first))
    register_provider(KvCacheProvider(second))
    assert len(physical()) == 1, "a rebuilt manager must supersede the old one"
    assert physical()[0].size_bytes == 200 * MB


@pytest.mark.cpu_only
def test_manager_is_held_weakly():
    """The pool is the largest allocation in the process."""
    manager = FakeKvManager(kv_stats(100, 0, 0))
    provider = KvCacheProvider(manager)
    register_provider(provider)
    assert provider.available()

    del manager
    gc.collect()

    # Keeping it alive would stop tens of GiB from ever being released.
    assert not provider.available(), "the monitor must not keep the KV cache manager alive"
    assert provider.collect() == []
    # A dead manager is skipped, not reported as zero.
    assert not [e for e in collect_all() if e.category is Category.KV_CACHE_POOL]


@pytest.mark.cpu_only
def test_reports_only_where_enabled(monkeypatch):
    from tensorrt_llm._torch.memory import monitor

    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    assert report_now(tag="t", rank=0) is None, "an unforced report needs TLLM_MEM_MONITOR=1"

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", "0")
    monkeypatch.setattr(settings, "current_rank", lambda: 3)
    # Every call site passes no rank; resolving it here is what makes the
    # filter do anything at all.
    assert monitor.full_mode() is False
    assert monitor.full_mode_requested() is True, "the rank-independent check"
    monkeypatch.setattr(settings, "current_rank", lambda: 0)
    assert monitor.full_mode() is True
    # An unknown rank does not silently skip.
    monkeypatch.setattr(settings, "current_rank", lambda: None)
    assert monitor.full_mode() is True


@pytest.mark.cpu_only
def test_rank_filter_is_parsed_strictly(monkeypatch):
    from tensorrt_llm._torch.memory import monitor

    warnings = []
    monkeypatch.setattr(settings.logger, "warning", warnings.append)
    monkeypatch.setattr(settings, "_warned", set())
    for raw, expected, warns in [
        ("0,4", frozenset({0, 4}), False),
        (" 1 , 2 ", frozenset({1, 2}), False),
        ("0-3,8", frozenset({0, 1, 2, 3, 8}), False),
        ("", None, False),
        (" ", None, True),
        (",", None, True),
        ("garbage", None, True),
        ("0;4", None, True),
        ("3-1", None, True),
    ]:
        warnings.clear()
        monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", raw)
        assert monitor.selected_ranks() == expected, raw
        assert bool(warnings) is warns, f"warning for {raw!r}"
    # A filter that cannot be read selects every rank rather than none, so
    # it never silences out-of-memory reports.
    monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", ",")
    assert monitor.oom_reports_enabled(rank=7)


@pytest.mark.cpu_only
def test_settings_are_validated(monkeypatch):
    from tensorrt_llm._torch.memory import monitor
    from tensorrt_llm._torch.memory.monitor import triggers

    warnings = []
    monkeypatch.setattr(settings.logger, "warning", warnings.append)
    monkeypatch.setattr(settings, "_warned", set())
    for raw, expected in [("abc", 60.0), ("-1", 60.0), ("nan", 60.0), ("inf", 60.0), ("5", 5.0)]:
        monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, raw)
        assert triggers._oom_interval_s() == expected, raw
    assert len(warnings) == 4, "one warning per invalid value"
    triggers._oom_interval_s()
    assert len(warnings) == 4, "warned once, not on every read"

    for raw, expected in [("", 0.0), ("0", 0.0), ("0.001", 1.0), ("30", 30.0), ("x", 0.0)]:
        monkeypatch.setenv(triggers.PERIODIC_INTERVAL_VAR, raw)
        assert triggers._periodic_interval_s() == expected, raw

    for raw, expected in [("1", True), ("true", True), ("0", False), ("", False), ("maybe", False)]:
        monkeypatch.setenv(monitor.STACKS_ENV_VAR, raw)
        assert monitor.stacks_requested() is expected, raw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_oom_observer_is_attached_once(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    import tensorrt_llm._torch.memory.monitor.triggers as trig

    monkeypatch.setattr(trig, "_attached", False)
    attaches = []
    monkeypatch.setattr(
        torch._C, "_cuda_attach_out_of_memory_observer", lambda cb: attaches.append(cb)
    )

    assert install_oom_reporter(rank=0) is True
    assert install_oom_reporter(rank=0) is True
    # torch keeps every observer it is handed, so a second attach would
    # print every report twice.
    assert len(attaches) == 1


@pytest.fixture
def armed(monkeypatch):
    """OOM reporting armed on device 3, with reports recorded instead of built."""
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mod, "_enabled_flag", True)
    monkeypatch.setattr(mod, "_device", 3)
    monkeypatch.setattr(mod, "_device_nearly_full", lambda: False)
    emitted = []
    monkeypatch.setattr(mod, "report_now", lambda **kw: emitted.append(kw) or "report")

    def reset():
        monkeypatch.setattr(mod, "_last_report_at", None)
        monkeypatch.setattr(mod, "_last_failed_allocation", None)
        emitted.clear()

    reset()
    return mod, clock, emitted, reset


_TORCH_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB"


def _oom(mod):
    mod._on_out_of_memory(0, 1 << 30, 80 << 30, 1 << 29)


@pytest.mark.cpu_only
def test_torch_observer_only_records(armed):
    mod, clock, emitted, reset = armed

    # A caller that catches the OOM and falls back must not cause a report.
    _oom(mod)
    assert emitted == [], "the observer must not build a report"

    # The error escaping to the executor reports it, with what the observer saw.
    mod.notify_error(_TORCH_OOM)
    assert len(emitted) == 1
    headline = emitted[0]["headline"]
    assert "PyTorch could not allocate 1.00 GiB on device 0" in headline
    assert "device free 0.50 GiB; PyTorch may use up to 80.00 GiB" in headline
    assert "CUDA out of memory" in headline
    assert emitted[0]["device_index"] == 3, "built on the executor's device"

    # A failed allocation long before the error is not linked to it.
    reset()
    _oom(mod)
    clock[0] += 3600.0
    mod.notify_error(_TORCH_OOM)
    assert "could not allocate" not in emitted[0]["headline"]


@pytest.mark.cpu_only
def test_oom_reports_are_rate_limited(armed, monkeypatch):
    mod, clock, emitted, reset = armed

    for _ in range(5):
        mod.notify_error(_TORCH_OOM)
        clock[0] += 1.0
    # A retry loop must not bury the first report under identical copies.
    assert len(emitted) == 1, "repeated OOMs within the interval"

    # A once-per-process budget would leave an OOM hours into serving
    # unreported because something earlier used it up.
    reset()
    mod.notify_error(_TORCH_OOM)
    clock[0] += 3600.0
    mod.notify_error(_TORCH_OOM)
    assert len(emitted) == 2, "a later OOM"

    reset()
    monkeypatch.setenv("TLLM_MEM_MONITOR_OOM_INTERVAL", "0")
    for _ in range(3):
        mod.notify_error(_TORCH_OOM)
    assert len(emitted) == 3, "a configured interval of 0"

    # A report that could not be built does not use up the interval.
    reset()
    monkeypatch.delenv("TLLM_MEM_MONITOR_OOM_INTERVAL")
    failed = []
    monkeypatch.setattr(mod, "report_now", lambda **kw: failed.append(kw) and None)
    mod.notify_error(_TORCH_OOM)
    monkeypatch.setattr(mod, "report_now", lambda **kw: emitted.append(kw) or "report")
    mod.notify_error(_TORCH_OOM)
    assert len(failed) == 1 and len(emitted) == 1, "the slot after a failed report"


@pytest.mark.cpu_only
def test_notify_error_reports_only_device_out_of_memory(armed, monkeypatch):
    mod, _, emitted, reset = armed
    nccl = (
        "Failed, NCCL error /src/cpp/tensorrt_llm/common/opUtils.cpp:212 "
        "'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'"
    )
    for message in (
        "CUDA error: out of memory",
        "[TensorRT-LLM][ERROR] CUDA runtime error in cudaMallocAsync: out of memory",
        "cudaErrorMemoryAllocation",
        "CUDA driver error: CUDA_ERROR_OUT_OF_MEMORY",
        _TORCH_OOM,
        "[TensorRT-LLM][ERROR] CUBLAS_STATUS_ALLOC_FAILED",
        "Failed to create cublas handle. Status: 3, Context: 0x5a1, Free Memory: 12 MB "
        "(0.0%), Total: 81559 MB. Consider reducing kv_cache_config.free_gpu_memory_fraction.",
    ):
        reset()
        mod.notify_error(message)
        assert len(emitted) == 1, f"not recognised: {message!r}"
        assert emitted[0]["headline"].startswith("OUT OF MEMORY: ")
    for message in (
        "Encountered an error in decode: index out of range",
        "error",
        "",
        None,
        # Host memory, not the GPU's.
        "cuMemHostRegister failed: out of memory",
        "HostOOMError: Host out of memory",
        "Failed to create cublas handle. Status: 1, Context: 0x5a1, Free Memory: 40000 MB",
        # NCCL's generic CUDA error, with plenty of memory free.
        nccl,
    ):
        reset()
        mod.notify_error(message)
        assert emitted == [], f"not a device OOM: {message!r}"

    # The same NCCL error on a device that is nearly full is an OOM.
    reset()
    monkeypatch.setattr(mod, "_device_nearly_full", lambda: True)
    mod.notify_error(nccl)
    assert len(emitted) == 1, "NCCL failure on a full device"


@pytest.mark.cpu_only
def test_oom_handlers_never_get_in_the_way(armed, monkeypatch):
    mod, _, emitted, _ = armed

    # torch cannot detach an observer, so the callback stays attached
    # and must simply do nothing once uninstalled.
    monkeypatch.setattr(mod, "_enabled_flag", False)
    _oom(mod)
    mod.notify_error(_TORCH_OOM)
    assert emitted == [] and mod._last_failed_allocation is None, "inert when not armed"

    monkeypatch.setattr(mod, "_enabled_flag", True)
    with suspend_reporting():
        _oom(mod)
        mod.notify_error(_TORCH_OOM)
    assert emitted == [] and mod._last_failed_allocation is None, "inert while suspended"

    def explode(**kwargs):
        raise RuntimeError("reporting failed")

    monkeypatch.setattr(mod, "report_now", explode)
    # Raising here would replace an out-of-memory error, which the user
    # needs to see, with a diagnostic's own failure.
    _oom(mod)
    mod.notify_error(_TORCH_OOM)


@pytest.mark.cpu_only
def test_oom_report_is_visible_at_the_default_log_level(armed, monkeypatch):
    import logging

    from tensorrt_llm.logger import logger

    mod, _, _, _ = armed
    monkeypatch.setattr(mod, "report_now", lambda **kw: "OOM REPORT BODY")
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect()
    level = logger.level
    logger._logger.addHandler(handler)
    logger.set_level("error")
    try:
        mod.notify_error(_TORCH_OOM)
    finally:
        logger.set_level(level)
        logger._logger.removeHandler(handler)
    assert any(
        r.levelno >= logging.ERROR and "OOM REPORT BODY" in r.getMessage() for r in records
    ), "the report must pass the default 'error' level"


@pytest.mark.cpu_only
def test_report_files_have_safe_names(monkeypatch, tmp_path):
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    monkeypatch.setenv(mod.DIR_VAR, str(tmp_path))
    path = mod._save("text", "warmup/before_warmup", rank=1)
    assert path is not None and os.path.dirname(path) == str(tmp_path)
    assert path.endswith("_warmup_before_warmup.txt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_report_now(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.delenv("TLLM_MEM_MONITOR_RANKS", raising=False)
    text = report_now(tag="unit", rank=0)
    assert text is not None and "memory monitor" in text

    monkeypatch.setattr(settings, "current_rank", lambda: 5)
    text = report_now(tag="unit")
    assert "rank=5" in text, "interleaved multi-rank logs must be separable"

    class _Boom:
        name = "boom"
        category = Category.KV_CACHE_POOL

        def available(self):
            return True

        def collect(self):
            raise RuntimeError("provider exploded")

    register_provider(_Boom())
    assert report_now(tag="unit", rank=0) is not None, "a broken provider broke the report"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_trace_log_hook(monkeypatch):
    """``log_mem_snapshot`` is the other way into the report: the real entry
    point must put the report in the log."""
    import tensorrt_llm._torch.pyexecutor.trace_log_utils as mod

    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    logged = []
    monkeypatch.setattr(mod.logger, "info", lambda msg, *a, **k: logged.append(msg))

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    mod.log_mem_snapshot("unit")
    assert any("GPU memory monitor" in str(m) for m in logged), (
        "the monitor report must reach the log through log_mem_snapshot"
    )

    logged.clear()
    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    mod.log_mem_snapshot("unit")
    assert logged, "the one-line snapshot is still logged"
    assert not any("GPU memory monitor" in str(m) for m in logged), "monitor off"


@pytest.mark.cpu_only
def test_failed_startup_is_labelled(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import tagging

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    reports = []
    monkeypatch.setattr(mod, "report_now", lambda **kw: reports.append(kw["tag"]))

    with pytest.raises(torch.OutOfMemoryError):
        with tagging.startup_window():
            raise torch.OutOfMemoryError("loading weights")
    # An unhandled OOM during load is the startup failure people most
    # need a breakdown for, so it is reported, and named for what it is.
    assert reports == ["startup_failed"]


@pytest.mark.cpu_only
def test_oom_reporter_is_armed_only_after_startup(monkeypatch):
    import contextlib

    import tensorrt_llm._torch.memory.monitor.triggers as trig
    import tensorrt_llm._torch.pyexecutor.py_executor_creator as creator

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(torch._C, "_cuda_attach_out_of_memory_observer", lambda cb: None)
    monkeypatch.setattr(trig, "_attached", False)
    monkeypatch.setattr(creator.gpu_memory_tagging, "startup_window", contextlib.nullcontext)
    monkeypatch.setattr(creator, "monitor_executor_initialization", contextlib.nullcontext)

    during_startup = []

    def fake_impl(**kwargs):
        # Warmup and KV sizing catch their own OOMs here; nothing may be
        # reported meanwhile, also for an executor already serving.
        during_startup.append((trig._enabled_flag, trig._suspended))
        return types.SimpleNamespace(device_id=None)

    monkeypatch.setattr(creator, "_create_py_executor_impl", fake_impl)
    first = creator.create_py_executor(llm_args=None)
    assert during_startup == [(False, 1)], "not armed while the first executor starts"
    assert first._oom_reporter_armed and trig._enabled_flag

    # A second executor in the same process: reporting is suspended while it
    # is built and stays armed until both have released it.
    second = creator.create_py_executor(llm_args=None)
    assert during_startup[1] == (True, 1), "suspended while the second executor starts"
    assert second._oom_reporter_armed and trig._suspended == 0
    release_oom_reporter()
    assert trig._enabled_flag, "still armed for the executor that is serving"
    release_oom_reporter()
    assert not trig._enabled_flag, "the last release disarms"
    release_oom_reporter()  # an extra release is harmless


@pytest.mark.cpu_only
def test_periodic_reporter_follows_the_latest_executor(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as trig

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv(trig.PERIODIC_INTERVAL_VAR, "60")
    monkeypatch.setattr(trig, "_attached", True)
    assert install_oom_reporter(rank=0, device=1)
    first = trig._periodic
    assert first is not None and first._device == 1
    assert install_oom_reporter(rank=0, device=2)
    assert trig._periodic is not first and trig._periodic._device == 2
    assert not first._thread.is_alive(), "the previous reporter is stopped"
    release_oom_reporter()
    release_oom_reporter()
    assert trig._periodic is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_baseline_is_taken_when_startup_completes(monkeypatch):
    """The baseline is taken when startup, warmup included, completes."""
    from tensorrt_llm._torch.memory import monitor
    from tensorrt_llm._torch.memory.monitor import tagging, triggers

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
    monitor.clear_baseline()
    try:
        # A half-built executor is not a state to measure growth from.
        with pytest.raises(RuntimeError):
            with tagging.startup_window():
                raise RuntimeError("load failed")
        assert startup.baseline() is None
        assert "since warmup" not in report_now(tag="later"), "after a failed startup"

        texts = []
        real = triggers.report_now
        monkeypatch.setattr(
            triggers, "report_now", lambda **kw: texts.append(real(**kw)) or texts[-1]
        )
        with tagging.startup_window():
            pass
        assert texts and "since warmup" not in texts[0], "the startup report itself"

        held = torch.empty(512 * MB, dtype=torch.uint8, device="cuda")
        try:
            text = real(tag="later")
            assert "since warmup" in text
            assert "PyTorch, unattributed grew the most (+0.50 GiB)" in text
        finally:
            del held
    finally:
        monitor.clear_baseline()


@pytest.mark.cpu_only
def test_armed_even_without_a_torch_observer(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(mod, "_attached", False)
    monkeypatch.delattr(torch._C, "_cuda_attach_out_of_memory_observer", raising=False)
    assert mod.install_oom_reporter() is True
    assert mod._enabled_flag is True


@pytest.mark.cpu_only
def test_executor_error_handler_passes_its_message_on(monkeypatch):
    from unittest import mock

    import tensorrt_llm._torch.pyexecutor.py_executor as pe

    seen = []
    monkeypatch.setattr(pe.gpu_memory_triggers, "notify_error", seen.append)
    try:
        pe.PyExecutor._handle_errors(mock.MagicMock(), "CUDA error: out of memory")
    except Exception:  # only the first statement matters here
        pass
    assert seen == ["CUDA error: out of memory"]
