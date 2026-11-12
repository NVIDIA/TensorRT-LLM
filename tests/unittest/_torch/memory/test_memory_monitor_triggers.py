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
import stat
import threading
import types

import pytest
import torch
from memory_monitor_test_utils import MB, FakeKvManager, kv_stats

from tensorrt_llm._torch.memory.monitor import providers, settings, startup
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
from tensorrt_llm.logger import logger


def _capture_warnings(monkeypatch) -> list[str]:
    """Warnings logged from now on, each once-only warning included again."""
    warnings: list[str] = []
    monkeypatch.setattr(logger, "_appeared_keys", set())

    def log(severity, *msg):
        if severity == logger.WARNING:
            warnings.append(" ".join(str(m) for m in msg))

    monkeypatch.setattr(logger, "log", log)
    return warnings


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

    warnings = _capture_warnings(monkeypatch)
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
        # Bounded: a rank above the maximum, or a huge range, is refused, not expanded.
        (f"{settings.MAX_RANK + 1}", None, True),
        ("0-4294967295", None, True),
        (f"0-{settings.MAX_RANK}", frozenset(range(settings.MAX_RANK + 1)), False),
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
def test_rank_filter_is_parsed_once_per_value(monkeypatch):
    parsed = []
    real_parse = settings._parse_ranks
    monkeypatch.setattr(settings, "_parse_ranks", lambda raw: parsed.append(raw) or real_parse(raw))
    monkeypatch.setenv(settings.RANK_ENV_VAR, "0-3")
    for _ in range(100):
        settings.rank_selected(2)
    monkeypatch.setenv(settings.RANK_ENV_VAR, "5")
    settings.rank_selected(2)
    assert parsed == ["0-3", "5"]


@pytest.mark.cpu_only
def test_a_rank_filter_outside_the_job_reports_on_every_rank(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    monkeypatch.setattr(mod, "_attached", True)
    warnings = _capture_warnings(monkeypatch)
    # Rank 8 of an eight-rank job does not exist: every rank would be silent.
    monkeypatch.setenv(settings.RANK_ENV_VAR, "8")
    assert install_oom_reporter(rank=0, device=0, world_size=8) is True
    assert settings.rank_selected(3), "falls back to every rank"
    assert [w for w in warnings if "selects none of this job's ranks 0-7" in w]
    release_oom_reporter()

    # A filter naming one of the job's ranks still applies.
    monkeypatch.setenv(settings.RANK_ENV_VAR, "1,8")
    assert install_oom_reporter(rank=0, device=0, world_size=8) is False
    assert install_oom_reporter(rank=1, device=0, world_size=8) is True


@pytest.mark.cpu_only
def test_settings_are_validated(monkeypatch):
    from tensorrt_llm._torch.memory import monitor
    from tensorrt_llm._torch.memory.monitor import triggers

    warnings = _capture_warnings(monkeypatch)
    for raw, expected in [
        ("abc", 60.0),
        ("-1", 60.0),
        ("nan", 60.0),
        ("inf", 60.0),
        ("1e10", 60.0),  # threading.Event.wait would overflow
        ("5", 5.0),
    ]:
        monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, raw)
        assert triggers._oom_interval_s() == expected, raw
    assert len(warnings) == 5, "one warning per invalid value"
    triggers._oom_interval_s()
    assert len(warnings) == 5, "warned once, not on every read"

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


_TORCH_OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB. GPU 0 has a "
    "total capacity of 80.00 GiB of which 512.00 MiB is free."
)


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
def test_only_the_failed_allocation_the_error_names_is_in_the_headline(armed):
    mod, _, emitted, reset = armed

    # A caller recovered from a PyTorch OOM; then a native allocation fails.
    _oom(mod)
    mod.notify_error("[TensorRT-LLM][ERROR] CUBLAS_STATUS_ALLOC_FAILED")
    headline = emitted[0]["headline"]
    assert headline == "OUT OF MEMORY: [TensorRT-LLM][ERROR] CUBLAS_STATUS_ALLOC_FAILED"

    for other in (
        _TORCH_OOM.replace("1.00 GiB", "2.00 GiB"),  # another size
        _TORCH_OOM.replace("GPU 0", "GPU 1"),  # another device
    ):
        reset()
        _oom(mod)
        mod.notify_error(other)
        assert "could not allocate" not in emitted[0]["headline"], other

    # PyTorch rounds the size to two decimals of the unit it prints.
    reset()
    mod._on_out_of_memory(0, (1 << 30) + (4 << 20), 80 << 30, 1 << 29)
    mod.notify_error(_TORCH_OOM)
    assert "PyTorch could not allocate 1.00 GiB on device 0" in emitted[0]["headline"]
    reset()
    mod._on_out_of_memory(0, 20 << 20, 80 << 30, 1 << 29)
    mod.notify_error("CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total")
    assert "PyTorch could not allocate 0.02 GiB on device 0" in emitted[0]["headline"]


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


@pytest.mark.cpu_only
def test_report_files_are_private_and_periodic_ones_are_capped(monkeypatch, tmp_path):
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    directory = tmp_path / "reports"
    monkeypatch.setenv(mod.DIR_VAR, str(directory))
    monkeypatch.setattr(mod, "MAX_PERIODIC_FILES", 3)
    path = mod._save("text", "out_of_memory", rank=0)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700, "only the owner may list it"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, "only the owner may read it"

    periodic = [mod._save(f"periodic {i}", "periodic", rank=0) for i in range(5)]
    mod._save("other rank", "periodic", rank=1)
    kept = sorted(p.name for p in directory.glob("memory_rank0_*_periodic.txt"))
    assert kept == sorted(os.path.basename(p) for p in periodic[-3:]), "the newest three"
    assert len(list(directory.glob("memory_rank1_*"))) == 1, "another rank's are its own"
    assert os.path.exists(path), "reports other than periodic ones are kept"


@pytest.mark.cpu_only
def test_an_unwritable_report_directory_warns_once_and_never_raises(monkeypatch, tmp_path):
    import tensorrt_llm._torch.memory.monitor.triggers as mod
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    warnings = _capture_warnings(monkeypatch)
    # A NUL byte from a YAML or Python config raises ValueError, not OSError.
    settings.configure(MemoryMonitorConfig(report_dir=f"{tmp_path}/bad\0dir"))
    assert mod._save("text", "startup", rank=0) is None
    assert mod._save("text", "startup", rank=0) is None
    settings.configure(None)
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(mod.DIR_VAR, str(blocker / "sub"))
    assert mod._save("text", "startup", rank=0) is None
    assert len(warnings) == 2 and all("cannot be written to" in w for w in warnings)


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
    from tensorrt_llm._torch.memory.monitor import tagging, triggers

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(tagging, "tagging_supported", lambda: False)
    startup.clear()
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
        startup.clear()


@pytest.mark.cpu_only
def test_armed_even_without_a_torch_observer(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as mod

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(mod, "_attached", False)
    monkeypatch.delattr(torch._C, "_cuda_attach_out_of_memory_observer", raising=False)
    assert mod.install_oom_reporter() is True
    assert mod._enabled_flag is True


@pytest.mark.cpu_only
def test_a_failed_arming_is_undone_and_leaves_other_executors_armed(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import triggers

    monkeypatch.setattr(triggers, "_attached", True)  # leave torch's observer alone
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv(triggers.PERIODIC_INTERVAL_VAR, "60")
    hooks_before = triggers._previous_thread_hook

    def no_threads(*_args, **_kwargs):
        raise RuntimeError("can't start new thread")

    with monkeypatch.context() as failing:
        failing.setattr(triggers, "_start_periodic", no_threads)
        assert install_oom_reporter(rank=0) is False, "a failed arming reports reporting as off"
    assert triggers._armed == [] and triggers._enabled_flag is False
    assert triggers._previous_thread_hook is hooks_before, "its hooks are removed again"

    monkeypatch.setenv("TLLM_MEM_MONITOR", "oom")
    assert install_oom_reporter(rank=0) is True
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    with monkeypatch.context() as failing:
        failing.setattr(triggers, "_start_periodic", no_threads)
        assert install_oom_reporter(rank=0) is False
    assert len(triggers._armed) == 1 and triggers._enabled_flag, "the first executor stays armed"
    release_oom_reporter()


@pytest.mark.cpu_only
def test_releasing_one_executor_restores_the_other(monkeypatch):
    """Two executors in one process; the second runs with the default config."""
    import tensorrt_llm._torch.memory.monitor.triggers as trig
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    monkeypatch.setattr(trig, "_attached", True)
    settings.configure(MemoryMonitorConfig(mode="full", periodic_report_interval=60))
    first_token = providers.begin_executor()
    assert install_oom_reporter(rank=0, device=1, owner=first_token)
    settings.configure(None)  # the second LLM's default config is not applied
    second_token = providers.begin_executor()
    assert install_oom_reporter(rank=0, device=2, owner=second_token)
    assert trig._device == 2 and trig._periodic._device == 2

    # The second executor shuts down first.
    release_oom_reporter(owner=second_token)
    providers.end_executor(second_token)
    assert trig._device == 1, "reports describe the remaining executor's device"
    assert trig._periodic is not None and trig._periodic._device == 1, "and its periodic reports"
    assert trig._periodic._thread.is_alive()
    assert settings.mode() == "full", "still in the first executor's mode"

    release_oom_reporter(owner=first_token)
    providers.end_executor(first_token)
    assert trig._armed == [] and trig._periodic is None and trig._device is None


@pytest.mark.cpu_only
def test_releasing_an_older_executor_keeps_the_newest_one_reporting(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as trig

    monkeypatch.setattr(trig, "_attached", True)
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setenv(trig.PERIODIC_INTERVAL_VAR, "60")
    first, second = object(), object()
    assert install_oom_reporter(rank=0, device=1, owner=first)
    assert install_oom_reporter(rank=0, device=2, owner=second)
    newest = trig._periodic
    release_oom_reporter(owner=first)
    assert trig._device == 2 and trig._periodic is newest, "the newest executor is untouched"
    release_oom_reporter(owner=first)  # released twice: no effect
    assert len(trig._armed) == 1
    release_oom_reporter(owner=second)
    assert trig._armed == [] and trig._periodic is None


@pytest.mark.cpu_only
def test_settings_that_act_only_in_full_mode_are_reported(monkeypatch):
    import tensorrt_llm._torch.memory.monitor.triggers as trig

    monkeypatch.setattr(trig, "_attached", True)
    warnings = _capture_warnings(monkeypatch)
    monkeypatch.setenv(trig.PERIODIC_INTERVAL_VAR, "30")
    monkeypatch.setenv(settings.STACKS_ENV_VAR, "1")
    assert install_oom_reporter(rank=0, device=0)  # the default mode, oom
    assert any(
        "TLLM_MEM_MONITOR_INTERVAL='30' takes effect only in full mode" in w for w in warnings
    )
    assert any("TLLM_MEM_MONITOR_STACKS='1' takes effect only in full mode" in w for w in warnings)
    release_oom_reporter()

    warnings.clear()
    monkeypatch.delenv(trig.PERIODIC_INTERVAL_VAR)
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    monkeypatch.setattr(trig, "tagging_supported", lambda: False)
    assert install_oom_reporter(rank=0, device=0)
    assert warnings == [
        "memory monitor: TLLM_MEM_MONITOR_STACKS='1' needs an allocator that keeps "
        "allocation metadata, and this one does not; stacks are not recorded"
    ]
    release_oom_reporter()


@pytest.mark.cpu_only
def test_a_thread_hook_that_chains_back_does_not_recurse(monkeypatch):
    """A library hook saved ours as its previous; re-arming saved it as ours."""
    import tensorrt_llm._torch.memory.monitor.triggers as trig

    monkeypatch.setattr(trig, "_attached", True)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    printed = []
    monkeypatch.setattr(threading, "__excepthook__", lambda args: printed.append(args.exc_type))
    assert install_oom_reporter(rank=0, device=0)
    ours = threading.excepthook

    def library_hook(args):
        ours(args)  # chains to the hook it found: ours

    threading.excepthook = library_hook
    release_oom_reporter()  # leaves the library hook in place
    assert install_oom_reporter(rank=0, device=0)  # saves library_hook as previous

    def run():
        raise ValueError("boom")

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert printed == [ValueError], "printed once by the default hook, no RecursionError"
    release_oom_reporter()
