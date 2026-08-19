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
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
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
def isolated_telemetry_opt_out_env():
    """Restore the process-wide disaggregated opt-out after each test."""
    previous = os.environ.get(_telemetry._TELEMETRY_OPT_OUT_ENV)
    os.environ.pop(_telemetry._TELEMETRY_OPT_OUT_ENV, None)
    yield
    os.environ.pop(_telemetry._TELEMETRY_OPT_OUT_ENV, None)
    if previous is not None:
        os.environ[_telemetry._TELEMETRY_OPT_OUT_ENV] = previous


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


@pytest.fixture
def terminal_mocks():
    """Provide the standard fail-silent session boundary dependencies."""
    with (
        patch.object(
            _telemetry.usage, "apply_usage_session_config", return_value=True
        ) as apply_config,
        patch.object(_telemetry.usage, "set_lifecycle_phase") as set_phase,
        patch.object(_telemetry.usage, "get_observed_signal", return_value=0) as signal,
        patch.object(_telemetry.usage, "record_observed_signal") as record_signal,
        patch.object(_telemetry.usage, "report_exit") as report_exit,
    ):
        yield SimpleNamespace(
            apply_config=apply_config,
            set_phase=set_phase,
            signal=signal,
            record_signal=record_signal,
            report_exit=report_exit,
        )


def _captured_terminal_parameters(payloads):
    """Return the parameters from one fully serialized terminal event."""
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["eventSchemaVer"] == "0.7"
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
        with patch.object(_telemetry.usage, "apply_usage_session_config") as apply_config:
            _telemetry.apply_raw_config_telemetry_opt_out(
                config,
                usage_context=UsageContext.CLI_SERVE,
                component="server",
            )

        assert apply_config.call_args.args[0] == {"disabled": True}
        assert apply_config.call_args.kwargs["lifecycle_phase"] == "config_validation"

    def test_explicit_cli_telemetry_overrides_yaml_opt_out(self):
        """A typed --telemetry flag retains Click/YAML precedence."""
        with patch.object(_telemetry.usage, "apply_usage_session_config") as apply_config:
            _telemetry.apply_raw_config_telemetry_opt_out(
                {"telemetry_config": {"disabled": True}},
                usage_context=UsageContext.CLI_SERVE,
                component="server",
                explicit_cli_telemetry=True,
            )

        apply_config.assert_not_called()

    @pytest.mark.parametrize(
        ("yaml_config", "telemetry", "expected_disabled"),
        [
            ("telemetry_config:\n  disabled: true\n", True, True),
            (
                "context_servers:\n  telemetry_config:\n    disabled: true\n",
                True,
                True,
            ),
            (
                "generation_servers:\n  telemetry_config:\n    disabled: true\n",
                True,
                True,
            ),
            ("telemetry_config:\n  disabled: false\n", False, True),
            ("telemetry_config:\n  disabled: false\n", True, False),
        ],
    )
    def test_disaggregated_opt_out_precedence(
        self,
        tmp_path,
        yaml_config,
        telemetry,
        expected_disabled,
        isolated_telemetry_opt_out_env,
    ):
        """An opt-out from either supported ingress disables the process tree."""
        del isolated_telemetry_opt_out_env
        config_path = tmp_path / "disagg.yaml"
        config_path.write_text(yaml_config, encoding="utf-8")

        with patch.object(_telemetry.usage, "apply_usage_session_config") as apply_config:
            disabled = _telemetry.apply_disaggregated_telemetry_config(
                str(config_path),
                telemetry=telemetry,
            )

        assert disabled is expected_disabled
        if expected_disabled:
            assert os.environ[_telemetry._TELEMETRY_OPT_OUT_ENV] == "1"
            assert apply_config.call_args.args == ({"disabled": True},)
        else:
            assert _telemetry._TELEMETRY_OPT_OUT_ENV not in os.environ
            apply_config.assert_not_called()

    @pytest.mark.parametrize("config_contents", [None, "telemetry_config: ["])
    def test_disaggregated_opt_out_precheck_defers_file_errors(
        self,
        tmp_path,
        config_contents,
        isolated_telemetry_opt_out_env,
    ):
        """The authoritative parser, not telemetry, reports invalid files."""
        del isolated_telemetry_opt_out_env
        config_path = tmp_path / "disagg.yaml"
        if config_contents is not None:
            config_path.write_text(config_contents, encoding="utf-8")

        with patch.object(_telemetry.usage, "apply_usage_session_config") as apply_config:
            disabled = _telemetry.apply_disaggregated_telemetry_config(
                str(config_path),
                telemetry=True,
            )

        assert disabled is False
        assert _telemetry._TELEMETRY_OPT_OUT_ENV not in os.environ
        apply_config.assert_not_called()

    def test_disaggregated_opt_out_deactivates_and_propagates(
        self,
        tmp_path,
        captured_exit_payloads,
        isolated_telemetry_opt_out_env,
    ):
        """A YAML opt-out stops the coordinator session and reaches children."""
        del captured_exit_payloads, isolated_telemetry_opt_out_env
        config_path = tmp_path / "disagg.yaml"
        config_path.write_text(
            "telemetry_config:\n  disabled: true\n",
            encoding="utf-8",
        )
        assert usage_lib.apply_usage_session_config(default_usage_context="cli_serve")
        assert usage_lib._get_session() is not None

        assert _telemetry.apply_disaggregated_telemetry_config(
            str(config_path),
            telemetry=True,
        )

        assert usage_lib._get_session() is None
        assert usage_lib._SESSION_DISABLED is True
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,sys; sys.exit(0 if "
                "os.environ.get('TRTLLM_NO_USAGE_STATS') == '1' else 1)",
            ],
            check=False,
            env=os.environ.copy(),
        )
        assert child.returncode == 0

    @pytest.mark.parametrize(
        ("args", "expected_code", "expected_kind"),
        [
            pytest.param(["run"], 0, "clean", id="clean"),
            pytest.param(["--invalid-option"], 2, "exception", id="parse-error"),
        ],
    )
    def test_system_exit_classification(
        self,
        terminal_mocks,
        args,
        expected_code,
        expected_kind,
    ):
        """Normal completion and Click failures retain their process status."""
        cli = _make_cli()
        with pytest.raises(SystemExit) as raised:
            cli.main(args=args, prog_name="test-cli")

        assert raised.value.code == expected_code
        terminal_mocks.report_exit.assert_called_once()
        outcome = terminal_mocks.report_exit.call_args.args[0]
        assert outcome.exit_code_known is True
        assert outcome.exit_code == expected_code
        assert outcome.termination_kind == expected_kind

    def test_no_telemetry_is_honored_before_parsing(self, terminal_mocks):
        """The early local session sees the CLI opt-out flag."""
        cli = _make_cli()
        with pytest.raises(SystemExit):
            cli.main(args=["--no-telemetry", "run"], prog_name="test-cli")

        config = terminal_mocks.apply_config.call_args.args[0]
        assert config.disabled is True

    def test_keyboard_interrupt_reports_signal(self, terminal_mocks):
        """Click's wrapped KeyboardInterrupt retains a SIGINT classification."""

        def interrupt():
            raise KeyboardInterrupt

        cli = _make_cli(interrupt)
        with pytest.raises(SystemExit):
            cli.main(args=["run"], prog_name="test-cli")

        outcome = terminal_mocks.report_exit.call_args.args[0]
        assert outcome.signal_number == 2
        assert outcome.termination_kind == "signal"

    def test_signal_exit_records_after_handler_unwinds(self, terminal_mocks):
        """A lock-free handler hands its signal to the outer boundary."""

        def terminate():
            _telemetry.raise_signal_exit(signal.SIGTERM, None)

        cli = _make_cli(terminate)
        with pytest.raises(_telemetry.SignalExit) as raised:
            cli.main(args=["run"], prog_name="test-cli")

        assert raised.value.signal_number == signal.SIGTERM
        assert raised.value.code == 128 + signal.SIGTERM
        terminal_mocks.record_signal.assert_called_once_with(signal.SIGTERM)
        outcome = terminal_mocks.report_exit.call_args.args[0]
        assert outcome.exit_code == 128 + signal.SIGTERM
        assert outcome.signal_number == signal.SIGTERM
        assert outcome.termination_kind == "signal"

    def test_optional_shell_exit_code_signal_inference(self, terminal_mocks):
        """Fleet compatibility can interpret 128 + signal exit codes."""

        def terminate():
            raise SystemExit(128 + signal.SIGTERM)

        with pytest.raises(SystemExit) as raised:
            _telemetry.run_with_terminal_reporting(
                terminate,
                infer_signal_from_exit_code=True,
            )

        assert raised.value.code == 128 + signal.SIGTERM
        terminal_mocks.record_signal.assert_called_once_with(signal.SIGTERM)
        outcome = terminal_mocks.report_exit.call_args.args[0]
        assert outcome.exit_code == 128 + signal.SIGTERM
        assert outcome.signal_number == signal.SIGTERM
        assert outcome.termination_kind == "signal"


class TestSerializedTerminationKinds:
    """Exercise non-signal terminal kinds through the real payload builder."""

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            pytest.param(
                ["run"],
                {
                    "terminationKind": "clean",
                    "exitCodeKnown": True,
                    "exitCode": 0,
                    "signalNumber": 0,
                    "lifecyclePhase": "shutdown",
                },
                id="clean",
            ),
            pytest.param(
                ["--invalid-option"],
                {
                    "terminationKind": "exception",
                    "exitCodeKnown": True,
                    "exitCode": 2,
                    "signalNumber": 0,
                    "lifecyclePhase": "cli_parsing",
                },
                id="parse-error",
            ),
        ],
    )
    def test_system_exit_serialization(self, captured_exit_payloads, args, expected):
        """Clean and Click-error paths serialize their complete status."""
        cli = _make_cli()

        with pytest.raises(SystemExit) as raised:
            cli.main(args=args, prog_name="test-cli")

        assert raised.value.code == expected["exitCode"]
        params = _captured_terminal_parameters(captured_exit_payloads)
        assert {field: params[field] for field in expected} == expected

    def test_supervised_child_failure_serializes_worker_failure(self, captured_exit_payloads):
        """A causal child status overrides the parent's generic exception."""

        def fail_after_child_observation():
            _telemetry.usage.record_termination_observation(
                _telemetry.usage.TerminalOutcome(
                    termination_kind="worker_failure",
                    component="engine_worker",
                    reporting_source="supervisor",
                    exit_code_known=True,
                    exit_code=137,
                    signal_number=9,
                )
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
