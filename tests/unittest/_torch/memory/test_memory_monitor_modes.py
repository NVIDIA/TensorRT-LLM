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
"""TLLM_MEM_MONITOR modes: 0 (off), oom (the default), 1 (full)."""

import threading

import pytest
import yaml

import tensorrt_llm._torch.memory.monitor as mon
from tensorrt_llm._torch.memory.monitor import providers, startup, tagging, triggers
from tensorrt_llm.logger import logger

pytestmark = pytest.mark.cpu_only


def _capture_warnings(monkeypatch) -> list[str]:
    """Warnings logged from now on, each once-only warning included again."""
    warnings: list[str] = []
    monkeypatch.setattr(logger, "_appeared_keys", set())

    def log(severity, *msg):
        if severity == logger.WARNING:
            warnings.append(" ".join(str(m) for m in msg))

    monkeypatch.setattr(logger, "log", log)
    return warnings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(triggers, "_attached", True)  # leave torch's observer alone
    triggers.uninstall_oom_reporter()
    yield
    triggers.uninstall_oom_reporter()


def test_values(monkeypatch):
    for value, expected in [
        (None, "oom"),
        ("", "oom"),
        ("oom", "oom"),
        ("OOM", "oom"),
        ("0", "off"),
        ("off", "off"),
        ("false", "off"),
        ("1", "full"),
        ("full", "full"),
        ("on", "full"),
    ]:
        if value is None:
            monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        else:
            monkeypatch.setenv("TLLM_MEM_MONITOR", value)
        assert mon.mode() == expected, f"TLLM_MEM_MONITOR={value!r}"

    # An unknown value falls back to the default, with a warning.
    monkeypatch.setenv("TLLM_MEM_MONITOR", "yes-please")
    warnings = _capture_warnings(monkeypatch)
    assert mon.mode() == "oom"
    assert mon.mode() == "oom"
    assert len(warnings) == 1 and "is not one of 0, oom, 1" in warnings[0], "warned once"


class TestDefaultReportsOnlyOutOfMemory:
    def test_reports_only_out_of_memory(self, monkeypatch):
        monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
        monkeypatch.setenv(triggers.PERIODIC_INTERVAL_VAR, "1")
        seen = []
        monkeypatch.setenv("TLLM_MEM_MONITOR_OOM_INTERVAL", "0")
        monkeypatch.setattr(
            triggers, "report_now", lambda tag, **kw: seen.append((tag, kw.get("force"))) or "r"
        )
        assert triggers.install_oom_reporter(rank=0) is True
        triggers.notify_error("CUDA error: out of memory")
        # The full monitor is off, so the report has to be forced.
        assert seen == [("out_of_memory", True)]

        # No periodic thread even though an interval is set ...
        assert not any(t.name == "memory-monitor-periodic" for t in threading.enumerate())
        # ... no startup recording or regions ...
        assert not startup._on()
        with tagging.startup_window() as recording:
            assert recording is False
        # ... and an unforced report (startup, periodic, warmup logging) is not
        # built; it returns before touching CUDA, so this holds without a GPU.
        assert mon.capture(tag="unit") is None

        triggers.uninstall_oom_reporter()
        monkeypatch.setenv("TLLM_MEM_MONITOR_RANKS", "3")
        assert triggers.install_oom_reporter(rank=0) is False, "the rank filter applies"


def test_off_reports_nothing(monkeypatch):
    seen = []
    monkeypatch.setenv("TLLM_MEM_MONITOR", "0")
    monkeypatch.setattr(triggers, "report_now", lambda tag, **kw: seen.append(tag) or "r")
    assert triggers.install_oom_reporter(rank=0) is False
    triggers.notify_error("CUDA error: out of memory")
    assert seen == []


def test_llm_args_configure_the_monitor_and_the_environment_overrides(monkeypatch, tmp_path):
    from pydantic import ValidationError

    from tensorrt_llm.llmapi import MemoryMonitorConfig, TorchLlmArgs

    for var in (
        mon.ENV_VAR,
        mon.RANK_ENV_VAR,
        mon.STACKS_ENV_VAR,
        triggers.PERIODIC_INTERVAL_VAR,
        triggers.OOM_INTERVAL_VAR,
        triggers.DIR_VAR,
    ):
        monkeypatch.delenv(var, raising=False)
    config = MemoryMonitorConfig(
        mode="full",
        ranks=[0, 2],
        record_stacks=True,
        periodic_report_interval=30,
        oom_report_interval=5,
        report_dir=str(tmp_path),
    )
    args = TorchLlmArgs(
        model="/tmp/dummy_model", tensor_parallel_size=4, memory_monitor_config=config
    )
    try:
        mon.configure(args.memory_monitor_config)
        assert mon.mode() == "full", "the configured mode"
        assert mon.selected_ranks() == frozenset({0, 2}), "the configured ranks"
        assert mon.stacks_requested(), "the configured stack recording"
        assert triggers._periodic_interval_s() == 30.0, "the configured periodic interval"
        assert triggers._oom_interval_s() == 5.0, "the configured OOM interval"
        assert mon.setting(triggers.DIR_VAR) == str(tmp_path), "the configured directory"

        monkeypatch.setenv(mon.ENV_VAR, "0")
        monkeypatch.setenv(mon.RANK_ENV_VAR, "1")
        assert mon.mode() == "off", "a set variable overrides the configured mode"
        assert mon.selected_ranks() == frozenset({1}), "and the configured ranks"
    finally:
        mon.configure(None)
    monkeypatch.delenv(mon.ENV_VAR)
    assert mon.mode() == "oom", "clearing the configuration restores the default"
    assert TorchLlmArgs(model="/tmp/dummy_model").memory_monitor_config is None

    for bad in (
        {"mode": "on"},
        {"ranks": []},
        {"ranks": [-1]},
        {"periodic_report_interval": 0.5},
        {"oom_report_interval": -1},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            MemoryMonitorConfig(**bad)

    for bad in (
        {"oom_report_interval": float("inf")},
        {"periodic_report_interval": float("inf")},
        {"periodic_report_interval": float("nan")},
        {"periodic_report_interval": 1e10},  # beyond threading.TIMEOUT_MAX on some hosts
        {"oom_report_interval": 1e10},
    ):
        with pytest.raises(ValidationError):
            MemoryMonitorConfig(**bad)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mode: off", "off"),
        ("mode: on", "full"),
        ("mode: 0", "off"),
        ("mode: 1", "full"),
        ("mode: '0'", "off"),
        ("mode: full", "full"),
        ("mode: oom", "oom"),
    ],
)
def test_yaml_mode_scalars_are_read_as_modes(text, expected):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    assert MemoryMonitorConfig(**yaml.safe_load(text)).mode == expected


@pytest.mark.parametrize(
    "text", ["oom_report_interval: .inf", "periodic_report_interval: .inf", "mode: 2"]
)
def test_yaml_values_the_runtime_would_replace_are_rejected(text):
    from pydantic import ValidationError

    from tensorrt_llm.llmapi import MemoryMonitorConfig

    with pytest.raises(ValidationError):
        MemoryMonitorConfig(**yaml.safe_load(text))


def test_ranks_outside_the_job_are_rejected():
    from pydantic import ValidationError

    from tensorrt_llm.llmapi import MemoryMonitorConfig, TorchLlmArgs

    TorchLlmArgs(
        model="/tmp/dummy_model",
        tensor_parallel_size=2,
        memory_monitor_config=MemoryMonitorConfig(ranks=[1]),
    )
    with pytest.raises(ValidationError, match=r"\[2, 8\] are not ranks of this job"):
        TorchLlmArgs(
            model="/tmp/dummy_model",
            tensor_parallel_size=2,
            memory_monitor_config=MemoryMonitorConfig(ranks=[0, 2, 8]),
        )


def test_an_empty_variable_does_not_override_the_configuration(monkeypatch):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    mon.configure(MemoryMonitorConfig(mode="full", ranks=[1], oom_report_interval=5))
    # Launchers export variables empty: docker -e VAR=, export VAR=${VAR:-}.
    monkeypatch.setenv(mon.ENV_VAR, "")
    monkeypatch.setenv(mon.RANK_ENV_VAR, "")
    monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, "")
    assert mon.mode() == "full"
    assert mon.selected_ranks() == frozenset({1})
    assert triggers._oom_interval_s() == 5.0


def test_an_overriding_variable_is_reported_once(monkeypatch):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    warnings = _capture_warnings(monkeypatch)
    mon.configure(MemoryMonitorConfig(mode="full", oom_report_interval=5))
    monkeypatch.setenv(mon.ENV_VAR, "1")  # the same mode, spelled differently
    monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, "5.0")
    assert mon.mode() == "full" and triggers._oom_interval_s() == 5.0
    assert warnings == [], "a variable that agrees with the configuration"

    monkeypatch.setenv(mon.ENV_VAR, "0")
    assert mon.mode() == "off" and mon.mode() == "off"
    assert warnings == [
        "memory monitor: TLLM_MEM_MONITOR='0' overrides memory_monitor_config.mode='full'"
    ]


def test_a_second_executor_keeps_the_first_executors_settings(monkeypatch):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    warnings = _capture_warnings(monkeypatch)
    first_config = MemoryMonitorConfig(mode="full", periodic_report_interval=30)
    mon.configure(first_config)
    first = providers.begin_executor()
    try:
        # The second LLM uses the default configuration.
        mon.configure(None)
        second = providers.begin_executor()
        assert mon.mode() == "full", "the first executor's mode stays on"
        assert triggers._periodic_interval_s() == 30.0
        assert len(warnings) == 1 and "is not applied" in warnings[0]
        mon.configure(None)
        assert len(warnings) == 1, "warned once"
        # The same configuration is not a conflict.
        mon.configure(first_config)
        assert len(warnings) == 1
        providers.end_executor(second)
    finally:
        providers.end_executor(first)
    mon.configure(None)
    assert mon.mode() == "oom", "applied once no executor is alive"
