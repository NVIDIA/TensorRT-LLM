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
"""Shared process-exit telemetry boundary for TRT-LLM Click CLIs."""

import signal
import sys
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import click

import tensorrt_llm.usage as usage
from tensorrt_llm.usage.config import TelemetryConfig, UsageContext
from tensorrt_llm.usage.schema import TerminationComponent, TerminationKind


def _telemetry_disabled_from_args(args: Sequence[str]) -> bool:
    """Apply Click's last-flag-wins behavior before full argument parsing."""
    disabled = False
    for arg in args:
        if arg == "--":
            break
        if arg == "--no-telemetry":
            disabled = True
        elif arg == "--telemetry":
            disabled = False
    return disabled


def _system_exit_code(exc: SystemExit) -> tuple[bool, int]:
    """Translate ``SystemExit.code`` into the process-style uint32 contract."""
    if exc.code is None:
        return True, 0
    if isinstance(exc.code, int):
        if 0 <= exc.code <= 4_294_967_295:
            return True, exc.code
        return False, 0
    # Python converts a non-integer SystemExit payload into process exit 1.
    return True, 1


def apply_raw_config_telemetry_opt_out(
    config: Any,
    *,
    usage_context: UsageContext,
    component: TerminationComponent,
    explicit_cli_telemetry: bool = False,
) -> None:
    """Honor only an unambiguous YAML opt-out before full config validation."""
    try:
        if explicit_cli_telemetry or not isinstance(config, Mapping):
            return
        telemetry_config = config.get("telemetry_config")
        if isinstance(telemetry_config, Mapping) and telemetry_config.get("disabled") is True:
            usage.start_usage_session(
                {"disabled": True},
                default_usage_context=usage_context.value,
                component=component,
                lifecycle_phase="config_validation",
            )
    except Exception:
        pass


def _contains_keyboard_interrupt(exc: BaseException) -> bool:
    """Detect Click's wrapped ``KeyboardInterrupt`` without reading messages."""
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, KeyboardInterrupt):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


class TelemetryGroup(click.Group):
    """Click group that reports one authoritative terminal process outcome."""

    def __init__(
        self,
        *args: Any,
        telemetry_usage_context: UsageContext,
        telemetry_component: TerminationComponent,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._telemetry_usage_context = telemetry_usage_context
        self._telemetry_component = telemetry_component
        self._telemetry_config: Optional[TelemetryConfig] = None

    def invoke(self, ctx: click.Context) -> Any:
        usage.set_lifecycle_phase("config_validation")
        return super().invoke(ctx)

    def _start_telemetry(self, args: Sequence[str]) -> None:
        try:
            self._telemetry_config = TelemetryConfig(
                disabled=_telemetry_disabled_from_args(args),
                usage_context=self._telemetry_usage_context,
            )
            usage.start_usage_session(
                self._telemetry_config,
                component=self._telemetry_component,
                lifecycle_phase="cli_parsing",
            )
        except Exception:
            self._telemetry_config = None

    def _report_terminal(
        self,
        *,
        exit_code_known: bool,
        exit_code: int,
        signal_number: int,
        termination_kind: TerminationKind,
    ) -> None:
        try:
            observation = usage.get_termination_observation()
            if observation["termination_kind"] is not None:
                signal_number = observation.get("signal_number", 0)
            if observation.get("exit_code_known") is not None:
                exit_code_known = observation["exit_code_known"]
                exit_code = observation["exit_code"]
            usage.report_exit(
                exit_code_known=exit_code_known,
                exit_code=exit_code,
                signal_number=signal_number,
                termination_kind=observation["termination_kind"] or termination_kind,
                lifecycle_phase=None,
                component=observation["component"],
                reporting_source=observation["reporting_source"] or "self",
                telemetry_config=self._telemetry_config,
                default_usage_context=self._telemetry_usage_context.value,
            )
        except Exception:
            pass

    def main(self, *args: Any, **kwargs: Any) -> Any:
        """Run Click and report the outcome before returning or re-raising."""
        cli_args = kwargs.get("args")
        if cli_args is None and args:
            cli_args = args[0]
        if cli_args is None:
            cli_args = sys.argv[1:]
        self._start_telemetry(cli_args)

        try:
            result = super().main(*args, **kwargs)
        except SystemExit as exc:
            exit_code_known, exit_code = _system_exit_code(exc)
            signal_number = usage.get_observed_signal()
            if signal_number == 0 and _contains_keyboard_interrupt(exc):
                signal_number = signal.SIGINT
                usage.record_observed_signal(signal_number)

            if signal_number:
                termination_kind = "signal"
            elif exit_code == 0:
                termination_kind = "clean"
                usage.set_lifecycle_phase("shutdown")
            else:
                termination_kind = "exception"
            self._report_terminal(
                exit_code_known=exit_code_known,
                exit_code=exit_code,
                signal_number=signal_number,
                termination_kind=termination_kind,
            )
            raise
        except KeyboardInterrupt:
            usage.record_observed_signal(signal.SIGINT)
            self._report_terminal(
                exit_code_known=True,
                exit_code=128 + signal.SIGINT,
                signal_number=signal.SIGINT,
                termination_kind="signal",
            )
            raise
        except Exception:
            signal_number = usage.get_observed_signal()
            self._report_terminal(
                exit_code_known=True,
                exit_code=1,
                signal_number=signal_number,
                termination_kind="signal" if signal_number else "exception",
            )
            raise
        else:
            signal_number = usage.get_observed_signal()
            if signal_number:
                termination_kind = "signal"
            else:
                termination_kind = "clean"
                usage.set_lifecycle_phase("shutdown")
            self._report_terminal(
                exit_code_known=True,
                exit_code=0,
                signal_number=signal_number,
                termination_kind=termination_kind,
            )
            return result
