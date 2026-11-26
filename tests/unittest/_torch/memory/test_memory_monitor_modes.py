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
from memory_monitor_test_utils import capture_warnings

import tensorrt_llm._torch.memory.monitor as mon
from tensorrt_llm._torch.memory.monitor import providers, settings, startup, tagging, triggers

pytestmark = pytest.mark.cpu_only


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
    warnings = capture_warnings(monkeypatch)
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
        triggers.notify_error("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0")
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


def test_llm_args_configure_the_monitor(monkeypatch, tmp_path):
    from pydantic import ValidationError

    from tensorrt_llm.llmapi import MemoryMonitorConfig, TorchLlmArgs

    config = MemoryMonitorConfig(
        mode="full",
        ranks=[0, 2],
        record_stacks=True,
        periodic_report_interval_s=30,
        oom_report_interval_s=5,
        report_dir=str(tmp_path),
    )
    args = TorchLlmArgs(
        model="/tmp/dummy_model", tensor_parallel_size=4, memory_monitor_config=config
    )
    mon.configure(args.memory_monitor_config, read_environment=False)
    assert mon.mode() == "full", "the configured mode"
    assert mon.selected_ranks() == frozenset({0, 2}), "the configured ranks"
    assert mon.stacks_requested(), "the configured stack recording"
    assert triggers._periodic_interval_s() == 30.0, "the configured periodic interval"
    assert triggers._oom_interval_s() == 5.0, "the configured OOM interval"
    assert mon.setting(triggers.DIR_VAR) == str(tmp_path), "the configured directory"

    mon.configure(None, read_environment=False)
    assert mon.mode() == "oom", "a new configuration replaces the previous one"
    assert TorchLlmArgs(model="/tmp/dummy_model").memory_monitor_config is None

    for bad in (
        {"mode": "on"},
        {"ranks": []},
        {"ranks": [-1]},
        {"periodic_report_interval_s": 0.5},
        {"oom_report_interval_s": -1},
        {"unknown": 1},
        {"oom_report_interval_s": float("inf")},
        {"periodic_report_interval_s": float("inf")},
        {"periodic_report_interval_s": float("nan")},
        {"periodic_report_interval_s": 1e10},  # beyond threading.TIMEOUT_MAX on some hosts
        {"oom_report_interval_s": 1e10},
    ):
        with pytest.raises(ValidationError):
            MemoryMonitorConfig(**bad)


def test_the_arguments_carry_the_environment_to_every_rank(monkeypatch):
    """The environment is read once, where the arguments are built."""
    from tensorrt_llm.llmapi import MemoryMonitorConfig, TorchLlmArgs

    monkeypatch.setenv(mon.ENV_VAR, "1")
    monkeypatch.setenv(mon.RANK_ENV_VAR, "0-1")
    monkeypatch.setenv(triggers.PERIODIC_INTERVAL_VAR, "30")
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        tensor_parallel_size=2,
        memory_monitor_config=MemoryMonitorConfig(oom_report_interval_s=5),
    )
    config = args.memory_monitor_config
    assert (config.mode, config.ranks, config.periodic_report_interval_s) == ("full", [0, 1], 30)
    assert config.oom_report_interval_s == 5, "a field no variable sets is kept"
    assert mon.full_mode_in(config)

    # A worker applies the arguments; its own environment no longer counts.
    monkeypatch.setenv(mon.ENV_VAR, "0")
    monkeypatch.setenv(mon.RANK_ENV_VAR, "1")
    mon.configure(config, read_environment=False)
    assert mon.mode() == "full" and mon.selected_ranks() == frozenset({0, 1})
    label = settings.setting_label
    assert label(triggers.PERIODIC_INTERVAL_VAR) == triggers.PERIODIC_INTERVAL_VAR, (
        "a value from the environment is named by its variable"
    )
    assert label(triggers.OOM_INTERVAL_VAR) == ("memory_monitor_config.oom_report_interval_s")


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
    "text", ["oom_report_interval_s: .inf", "periodic_report_interval_s: .inf", "mode: 2"]
)
def test_yaml_values_the_runtime_would_replace_are_rejected(text):
    from pydantic import ValidationError

    from tensorrt_llm.llmapi import MemoryMonitorConfig

    with pytest.raises(ValidationError):
        MemoryMonitorConfig(**yaml.safe_load(text))


def test_the_config_and_the_runtime_share_their_bounds_and_defaults():
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    fields = MemoryMonitorConfig.model_fields

    def bound(field, name):
        return next(getattr(m, name) for m in fields[field].metadata if hasattr(m, name))

    assert fields["mode"].default == settings.DEFAULT_MODE
    assert fields["oom_report_interval_s"].default == settings.DEFAULT_OOM_INTERVAL_S
    assert bound("periodic_report_interval_s", "ge") == settings.MIN_PERIODIC_INTERVAL_S
    assert bound("periodic_report_interval_s", "le") == settings.MAX_SECONDS
    assert bound("oom_report_interval_s", "le") == settings.MAX_SECONDS
    assert (
        f"newest {triggers.MAX_PERIODIC_FILES} periodic reports of each rank and process"
        in fields["report_dir"].description
    )


def test_ranks_outside_the_job_are_rejected(monkeypatch):
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

    # From the environment they are reported and dropped instead.
    warnings = capture_warnings(monkeypatch)
    monkeypatch.setenv(mon.RANK_ENV_VAR, "1,8")
    args = TorchLlmArgs(model="/tmp/dummy_model", tensor_parallel_size=2)
    assert args.memory_monitor_config.ranks == [1]
    monkeypatch.setenv(mon.RANK_ENV_VAR, "8")
    assert TorchLlmArgs(model="/tmp/dummy_model", tensor_parallel_size=2).memory_monitor_config is (
        None
    )
    assert len([w for w in warnings if mon.RANK_ENV_VAR in w]) == 2


def test_a_rank_filter_outside_the_job_is_ignored_before_startup_is_recorded(monkeypatch):
    """``TLLM_MEM_MONITOR_RANKS=8`` at TP=8 would leave every rank unrecorded."""
    monkeypatch.setenv(mon.ENV_VAR, "1")
    monkeypatch.setenv(mon.RANK_ENV_VAR, "8")
    monkeypatch.setattr(tagging, "_log_window_result", lambda *args, **kwargs: None)
    mon.configure(None, world_size=8)
    assert mon.full_mode(rank=3), "every rank records its startup"
    with tagging.startup_window(rank=3):
        assert startup._on(), "startup steps are recorded"


@pytest.mark.parametrize("blank", ["", " ", "\t"])
def test_a_blank_variable_does_not_override_the_configuration(monkeypatch, blank):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    warnings = capture_warnings(monkeypatch)
    config = MemoryMonitorConfig(mode="full", ranks=[1], oom_report_interval_s=5)
    # Launchers export variables empty: docker -e VAR=, export VAR=${VAR:-}.
    monkeypatch.setenv(mon.ENV_VAR, blank)
    monkeypatch.setenv(mon.RANK_ENV_VAR, blank)
    monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, blank)
    assert settings.resolve_environment(config) is config
    mon.configure(config)
    assert mon.mode() == "full"
    assert mon.selected_ranks() == frozenset({1})
    assert triggers._oom_interval_s() == 5.0
    settings.reset()
    assert mon.mode() == "oom", "unconfigured, a blank variable is unset too"
    assert warnings == [], "a blank variable is not an invalid value"


def test_only_overriding_a_field_the_configuration_set_is_reported(monkeypatch):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    warnings = capture_warnings(monkeypatch)
    config = MemoryMonitorConfig(mode="full", oom_report_interval_s=5)
    monkeypatch.setenv(mon.ENV_VAR, "1")  # the same mode, spelled differently
    monkeypatch.setenv(triggers.OOM_INTERVAL_VAR, "5.0")
    monkeypatch.setenv(mon.STACKS_ENV_VAR, "1")  # a field the configuration left unset
    resolved = settings.resolve_environment(config)
    assert resolved.mode == "full" and resolved.record_stacks
    assert warnings == [], "agreeing with the configuration, or filling a field it left unset"

    monkeypatch.setenv(mon.ENV_VAR, "0")
    assert settings.resolve_environment(config).mode == "off"
    settings.resolve_environment(config)
    assert warnings == [
        "memory monitor: TLLM_MEM_MONITOR='0' overrides memory_monitor_config.mode='full'"
    ]


def test_a_second_executor_does_not_change_the_settings(monkeypatch):
    from tensorrt_llm.llmapi import MemoryMonitorConfig

    first_config = MemoryMonitorConfig(mode="full", periodic_report_interval_s=30)
    mon.configure(first_config)
    first = providers.begin_executor()
    try:
        # The second LLM uses the default configuration.
        mon.configure(None)
        second = providers.begin_executor()
        assert mon.mode() == "full", "the monitored executor's mode stays on"
        assert triggers._periodic_interval_s() == 30.0
        providers.end_executor(second)
    finally:
        providers.end_executor(first)
    mon.configure(None)
    assert mon.mode() == "oom", "applied once no monitored executor is alive"
