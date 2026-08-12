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
"""Serving-boundary telemetry helpers."""

import inspect
import socket

import uvicorn

import tensorrt_llm.usage as usage


def _ensure_uvicorn_signal_api(server: uvicorn.Server) -> None:
    """Fail clearly if Uvicorn changes the private signal lifecycle we use."""
    private_serve = getattr(uvicorn.Server, "_serve", None)
    try:
        parameters = tuple(inspect.signature(private_serve).parameters)
    except (TypeError, ValueError):
        parameters = ()
    if (
        not inspect.iscoroutinefunction(private_serve)
        or parameters != ("self", "sockets")
        or not isinstance(getattr(server, "_captured_signals", None), list)
    ):
        version = getattr(uvicorn, "__version__", "unknown")
        raise RuntimeError(
            "Unsupported Uvicorn signal API in version "
            f"{version}: TRT-LLM requires Server._serve(self, sockets) and "
            "Server._captured_signals for terminal telemetry"
        )


class TelemetryUvicornServer(uvicorn.Server):
    """Uvicorn server that reports captured signals after graceful shutdown."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        _ensure_uvicorn_signal_api(self)

    async def _serve(self, sockets: list[socket.socket] | None = None) -> None:
        await super()._serve(sockets)
        if not self._captured_signals:
            return

        # Uvicorn restores the original handlers and re-raises captured
        # signals after _serve() returns. Report while normal coroutine
        # control is still available, without taking locks in handle_exit().
        signal_number = self._captured_signals[-1]
        usage.record_observed_signal(signal_number)
        usage.report_exit(
            usage.TerminalOutcome(
                termination_kind="signal",
                exit_code_known=True,
                exit_code=128 + signal_number,
                signal_number=signal_number,
            )
        )
