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
"""Tests for the shared Click process-exit telemetry boundary."""

import os
import threading
from unittest.mock import patch

import click
import pytest

from tensorrt_llm.commands import _telemetry
from tensorrt_llm.usage import usage_lib
from tensorrt_llm.usage.config import UsageContext


def _make_cli(callback=None):
    @click.group(
        cls=_telemetry.TelemetryGroup,
        telemetry_usage_context=UsageContext.CLI_SERVE,
        telemetry_component="server",
    )
    def cli():
        pass

    @cli.command()
    def run():
        if callback is not None:
            callback()

    return cli


@pytest.fixture
def captured_exit_payloads(monkeypatch, enable_telemetry):
    """Capture real terminal payloads while isolating process-global state."""
    usage_lib._SESSION = None
    usage_lib._SESSION_DISABLED = False
    usage_lib._SESSION_LOCK = threading.Lock()
    usage_lib._REPORTER_STARTED = False
    usage_lib._REPORTER_ACTIVE = False
    usage_lib._REPORTER_LOCK = threading.Lock()
    usage_lib._REPORTER_STOP = threading.Event()
    usage_lib._PENDING_TERMINAL = None
    usage_lib._PROCESS_PID = os.getpid()

    payloads = []
    monkeypatch.setattr(usage_lib, "_send_to_gxt", payloads.append)
    monkeypatch.setattr(usage_lib, "_is_reporting_rank", lambda: True)
    yield payloads

    usage_lib._REPORTER_STOP.set()
    usage_lib._SESSION = None
    usage_lib._SESSION_DISABLED = False
    usage_lib._REPORTER_STARTED = False
    usage_lib._REPORTER_ACTIVE = False
    usage_lib._PENDING_TERMINAL = None


def _captured_terminal_parameters(payloads):
    """Return the parameters from one fully serialized terminal event."""
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["eventSchemaVer"] == "0.3"
    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["name"] == "trtllm_exit_report"
    return event["parameters"]


class TestTelemetryGroup:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (None, (True, 0)),
            (2, (True, 2)),
            (-1, (False, 0)),
            ("failure", (True, 1)),
        ],
    )
    def test_system_exit_code_normalization(self, code, expected):
        """Out-of-contract integers are unknown instead of guessed."""
        assert _telemetry._system_exit_code(SystemExit(code)) == expected

    @pytest.mark.parametrize(
        ("args", "disabled"),
        [
            (["--no-telemetry"], True),
            (["--no-telemetry", "--telemetry"], False),
            (["--telemetry", "--no-telemetry"], True),
            (["--no-telemetry", "--", "--telemetry"], True),
        ],
    )
    def test_early_flags_use_click_last_value(self, args, disabled):
        """Early opt-out scanning matches Click's repeated-flag behavior."""
        assert _telemetry._telemetry_disabled_from_args(args) is disabled

    def test_raw_yaml_explicit_opt_out_is_honored_before_validation(self):
        """Only the unambiguous disabled=true value is read before validation."""
        config = {
            "telemetry_config": {"disabled": True},
            "another_option": object(),
        }
        with patch.object(_telemetry.usage, "start_usage_session") as start_session:
            _telemetry.apply_raw_config_telemetry_opt_out(
                config,
                usage_context=UsageContext.CLI_SERVE,
                component="server",
            )

        assert start_session.call_args.args[0] == {"disabled": True}
        assert start_session.call_args.kwargs["lifecycle_phase"] == "config_validation"

    def test_explicit_cli_telemetry_overrides_yaml_opt_out(self):
        """A typed --telemetry flag retains Click/YAML precedence."""
        with patch.object(_telemetry.usage, "start_usage_session") as start_session:
            _telemetry.apply_raw_config_telemetry_opt_out(
                {"telemetry_config": {"disabled": True}},
                usage_context=UsageContext.CLI_SERVE,
                component="server",
                explicit_cli_telemetry=True,
            )

        start_session.assert_not_called()

    def test_clean_exit_reports_zero(self):
        """A normally completed CLI emits one clean terminal classification."""
        cli = _make_cli()
        with (
            patch.object(_telemetry.usage, "start_usage_session", return_value=True),
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value={
                    "termination_kind": None,
                    "component": None,
                    "reporting_source": None,
                },
            ),
            patch.object(_telemetry.usage, "report_exit") as report_exit,
            pytest.raises(SystemExit) as raised,
        ):
            cli.main(args=["run"], prog_name="test-cli")

        assert raised.value.code == 0
        assert report_exit.call_count == 1
        kwargs = report_exit.call_args.kwargs
        assert kwargs["exit_code_known"] is True
        assert kwargs["exit_code"] == 0
        assert kwargs["termination_kind"] == "clean"

    def test_parse_error_reports_cli_failure(self):
        """Click parse failures are observed before command invocation."""
        cli = _make_cli()
        with (
            patch.object(_telemetry.usage, "start_usage_session", return_value=True),
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value={
                    "termination_kind": None,
                    "component": None,
                    "reporting_source": None,
                },
            ),
            patch.object(_telemetry.usage, "report_exit") as report_exit,
            pytest.raises(SystemExit) as raised,
        ):
            cli.main(args=["--invalid-option"], prog_name="test-cli")

        assert raised.value.code == 2
        kwargs = report_exit.call_args.kwargs
        assert kwargs["exit_code"] == 2
        assert kwargs["termination_kind"] == "exception"

    def test_no_telemetry_is_honored_before_parsing(self):
        """The early local session sees the CLI opt-out flag."""
        cli = _make_cli()
        with (
            patch.object(_telemetry.usage, "start_usage_session") as start_session,
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value={
                    "termination_kind": None,
                    "component": None,
                    "reporting_source": None,
                },
            ),
            patch.object(_telemetry.usage, "report_exit"),
            pytest.raises(SystemExit),
        ):
            cli.main(args=["--no-telemetry", "run"], prog_name="test-cli")

        config = start_session.call_args.args[0]
        assert config.disabled is True

    def test_keyboard_interrupt_reports_signal(self):
        """Click's wrapped KeyboardInterrupt retains a SIGINT classification."""

        def interrupt():
            raise KeyboardInterrupt

        cli = _make_cli(interrupt)
        with (
            patch.object(_telemetry.usage, "start_usage_session", return_value=True),
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value={
                    "termination_kind": None,
                    "component": None,
                    "reporting_source": None,
                },
            ),
            patch.object(_telemetry.usage, "record_observed_signal"),
            patch.object(_telemetry.usage, "report_exit") as report_exit,
            pytest.raises(SystemExit),
        ):
            cli.main(args=["run"], prog_name="test-cli")

        kwargs = report_exit.call_args.kwargs
        assert kwargs["signal_number"] == 2
        assert kwargs["termination_kind"] == "signal"

    def test_pending_worker_failure_preserves_authoritative_child_status(self):
        """A supervised child status overrides a later generic parent error."""

        def fail():
            raise RuntimeError("worker unavailable")

        cli = _make_cli(fail)
        observation = {
            "termination_kind": "worker_failure",
            "component": "engine_worker",
            "reporting_source": "supervisor",
            "exit_code_known": True,
            "exit_code": 137,
            "signal_number": 9,
        }
        with (
            patch.object(_telemetry.usage, "start_usage_session", return_value=True),
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value=observation,
            ),
            patch.object(_telemetry.usage, "report_exit") as report_exit,
            pytest.raises(RuntimeError),
        ):
            cli.main(args=["run"], prog_name="test-cli")

        kwargs = report_exit.call_args.kwargs
        assert kwargs["exit_code_known"] is True
        assert kwargs["exit_code"] == 137
        assert kwargs["signal_number"] == 9
        assert kwargs["termination_kind"] == "worker_failure"
        assert kwargs["component"] == "engine_worker"
        assert kwargs["reporting_source"] == "supervisor"

    def test_liveness_only_worker_failure_does_not_invent_status(self):
        """A proxy observation suppresses the synthetic shutdown signal/code."""

        def fail():
            raise RuntimeError("worker unavailable")

        cli = _make_cli(fail)
        observation = {
            "termination_kind": "worker_failure",
            "component": "engine_worker",
            "reporting_source": "executor_proxy",
            "exit_code_known": False,
            "exit_code": 0,
            "signal_number": 0,
        }
        with (
            patch.object(_telemetry.usage, "start_usage_session", return_value=True),
            patch.object(_telemetry.usage, "set_lifecycle_phase"),
            patch.object(_telemetry.usage, "get_observed_signal", return_value=2),
            patch.object(
                _telemetry.usage,
                "get_termination_observation",
                return_value=observation,
            ),
            patch.object(_telemetry.usage, "report_exit") as report_exit,
            pytest.raises(RuntimeError),
        ):
            cli.main(args=["run"], prog_name="test-cli")

        kwargs = report_exit.call_args.kwargs
        assert kwargs["exit_code_known"] is False
        assert kwargs["exit_code"] == 0
        assert kwargs["signal_number"] == 0
        assert kwargs["termination_kind"] == "worker_failure"


class TestSerializedTerminationKinds:
    """Exercise non-signal terminal kinds through the real payload builder."""

    def test_clean_callback_return_serializes_clean(self, captured_exit_payloads):
        """A normally returning owned CLI boundary is an authoritative clean exit."""
        cli = _make_cli()

        with pytest.raises(SystemExit) as raised:
            cli.main(args=["run"], prog_name="test-cli")

        assert raised.value.code == 0
        params = _captured_terminal_parameters(captured_exit_payloads)
        assert params["terminationKind"] == "clean"
        assert params["exitCodeKnown"] is True
        assert params["exitCode"] == 0
        assert params["signalNumber"] == 0
        assert params["lifecyclePhase"] == "shutdown"

    def test_click_parse_failure_serializes_exception(self, captured_exit_payloads):
        """A CLI parse failure is an authoritative nonzero exception exit."""
        cli = _make_cli()

        with pytest.raises(SystemExit) as raised:
            cli.main(args=["--invalid-option"], prog_name="test-cli")

        assert raised.value.code == 2
        params = _captured_terminal_parameters(captured_exit_payloads)
        assert params["terminationKind"] == "exception"
        assert params["exitCodeKnown"] is True
        assert params["exitCode"] == 2
        assert params["signalNumber"] == 0
        assert params["lifecyclePhase"] == "cli_parsing"

    def test_supervised_child_failure_serializes_worker_failure(
        self, captured_exit_payloads
    ):
        """A causal child status overrides the parent's generic exception."""

        def fail_after_child_observation():
            _telemetry.usage.record_termination_observation(
                termination_kind="worker_failure",
                component="engine_worker",
                reporting_source="supervisor",
                exit_code_known=True,
                exit_code=137,
                signal_number=9,
            )
            raise RuntimeError("supervised worker exited")

        cli = _make_cli(fail_after_child_observation)

        with pytest.raises(RuntimeError, match="supervised worker exited"):
            cli.main(args=["run"], prog_name="test-cli")

        params = _captured_terminal_parameters(captured_exit_payloads)
        assert params["terminationKind"] == "worker_failure"
        assert params["exitCodeKnown"] is True
        assert params["exitCode"] == 137
        assert params["signalNumber"] == 9
        assert params["component"] == "engine_worker"
        assert params["reportingSource"] == "supervisor"
