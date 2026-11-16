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
"""Unit tests for the disagg server's steady-clock handshake.

`OpenAIDisaggServer._sync_server_clock` estimates each ctx/gen server's clock
offset with the NTP algorithm over an HTTP round trip. That estimate is only
accurate to +/- delay/2, because an asymmetric round trip is indistinguishable
from a genuine clock offset. Whatever it computes is then added to every
perf-metric timestamp the ctx/gen server reports, so a bad estimate silently
corrupts those timestamps.

These tests drive the handshake against a stub `/steady_clock_offset` endpoint
whose per-leg latency we control, and assert that a conclusive measurement is
applied while an inconclusive one is discarded.
"""

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest

from tensorrt_llm.serve.openai_disagg_server import (
    _CLOCK_SYNC_GOOD_DELAY_SECONDS,
    _CLOCK_SYNC_MAX_DELAY_SECONDS,
    _CLOCK_SYNC_PROBES,
    OpenAIDisaggServer,
)

# The stub sleeps this long between its own receive/transmit stamps, mirroring
# the real `get_steady_clock_offset` handler. It cancels out of the NTP math.
_SERVER_PROCESSING_S = 0.01


class _StubWorkerServer:
    """Minimal `/steady_clock_offset` endpoint with an injectable stall.

    `stall_before_receive` models the ctx/gen server's event loop being busy
    when the request lands: the request has already arrived, but the handler --
    and therefore `receive_ts` -- is delayed. That inflates the outbound leg
    only, which is exactly the one-sided asymmetry seen in CI when the
    handshake races server startup.
    """

    def __init__(self, stall_before_receive: float = 0.0, fail: bool = False):
        self.stall_before_receive = stall_before_receive
        self.fail = fail
        self.get_count = 0
        self.posted_offsets: list[float] = []
        self.paths: list[str] = []
        self._server = None
        self.url = None

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        try:
            while True:
                try:
                    header = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    return
                method = header.split(b" ", 1)[0]
                self.paths.append(header.split(b" ", 2)[1].decode())
                if method == b"POST":
                    length = 0
                    for line in header.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            length = int(line.split(b":", 1)[1])
                    payload = json.loads(await reader.readexactly(length))
                    self.posted_offsets.append(payload["offset"])
                    body = b"{}"
                elif self.fail:
                    self.get_count += 1
                    writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                    await writer.drain()
                    continue
                else:
                    self.get_count += 1
                    if self.stall_before_receive:
                        await asyncio.sleep(self.stall_before_receive)
                    # Both ends read CLOCK_MONOTONIC, so the true offset is 0
                    # and any offset the handshake computes is pure error.
                    receive_ts = asyncio.get_running_loop().time()
                    await asyncio.sleep(_SERVER_PROCESSING_S)
                    transmit_ts = asyncio.get_running_loop().time()
                    body = json.dumps(
                        {
                            "receive_ts": receive_ts,
                            "transmit_ts": transmit_ts,
                        }
                    ).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: %d\r\n\r\n" % len(body) + body
                )
                await writer.drain()
        finally:
            writer.close()


async def _run_handshake(stub: _StubWorkerServer) -> None:
    # `_sync_server_clock` only reads `_req_timeout_secs` off `self`, so a stub
    # avoids standing up a whole disagg server for a clock-arithmetic test.
    fake_self = SimpleNamespace(_req_timeout_secs=30)
    await OpenAIDisaggServer._sync_server_clock(fake_self, stub.url)


@pytest.mark.asyncio
async def test_fast_server_offset_is_applied_and_small():
    """A responsive server yields a conclusive, near-zero offset."""
    async with _StubWorkerServer() as stub:
        await _run_handshake(stub)

    assert len(stub.posted_offsets) == 1, (
        "a conclusive measurement should be pushed to the worker server"
    )
    # The stub shares this process's clock, so the true offset is exactly 0.
    assert abs(stub.posted_offsets[0]) <= _CLOCK_SYNC_GOOD_DELAY_SECONDS, (
        f"applied offset {stub.posted_offsets[0]} is larger than the measurement is accurate to"
    )


@pytest.mark.asyncio
async def test_fast_server_stops_after_one_probe():
    """Probing stops as soon as a sample is conclusive.

    Each probe costs a full round trip and servers are prepared sequentially,
    so a healthy server must not spend the whole probe budget.
    """
    async with _StubWorkerServer() as stub:
        await _run_handshake(stub)

    # One warm-up probe (discarded) plus a single measured probe.
    assert stub.get_count == 2, (
        f"expected warm-up + 1 measured probe, got {stub.get_count} requests"
    )


@pytest.mark.asyncio
async def test_stalled_server_offset_is_discarded():
    """An inconclusive measurement must not be applied.

    With a one-sided stall the NTP estimate is ~stall/2 of pure error. Applying
    it would skew every perf-metric timestamp the worker reports by that much,
    which is strictly worse than leaving the clocks alone.
    """
    stall = 10 * _CLOCK_SYNC_MAX_DELAY_SECONDS
    async with _StubWorkerServer(stall_before_receive=stall) as stub:
        await _run_handshake(stub)

    assert stub.posted_offsets == [], (
        "an offset estimated from a stalled round trip must be discarded"
    )
    # Warm-up plus the full budget, since no sample ever became conclusive.
    assert stub.get_count == 1 + _CLOCK_SYNC_PROBES


@pytest.mark.asyncio
async def test_erroring_server_is_not_retried():
    """A server that answers with an error is not probed again.

    Servers are prepared one at a time, so retrying a broken one delays the
    readiness of every server behind it in the queue.
    """
    async with _StubWorkerServer(fail=True) as stub:
        await _run_handshake(stub)

    assert stub.posted_offsets == []
    # Warm-up plus a single measured probe that failed; no further retries.
    assert stub.get_count == 2, (
        f"expected the probe loop to stop at the first failure, got {stub.get_count} requests"
    )


@pytest.mark.asyncio
async def test_unreachable_server_is_skipped_quietly():
    """A server that never answers is skipped without raising."""
    async with _StubWorkerServer() as stub:
        dead_url = stub.url
    # Stub is now closed; the port should refuse connections.
    fake_self = SimpleNamespace(_req_timeout_secs=5)
    await OpenAIDisaggServer._sync_server_clock(fake_self, dead_url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme,suffix",
    [("", ""), ("http://", ""), ("http://", "/")],
    ids=["bare", "http_scheme", "http_scheme_trailing_slash"],
)
async def test_server_entry_forms_reach_the_endpoint(scheme, suffix):
    """A server entry with or without a scheme or trailing slash is dialed as is.

    Only the requested paths are checked: whether the measured offset is applied
    depends on the round-trip time of the machine running the test.
    """
    async with _StubWorkerServer() as stub:
        fake_self = SimpleNamespace(_req_timeout_secs=30)
        await OpenAIDisaggServer._sync_server_clock(fake_self, f"{scheme}{stub.url}{suffix}")

    assert stub.paths, "no request reached the stub server"
    assert set(stub.paths) == {"/steady_clock_offset"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    ["https://worker:8443", "https://worker:8443/"],
    ids=["https_scheme", "https_scheme_trailing_slash"],
)
async def test_https_server_entry_is_dialed_as_https(monkeypatch, entry):
    """An https:// entry keeps its scheme when the handshake URL is built."""
    urls = []

    class _Response:
        status = 503

    class _Request:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *exc):
            return False

    class _RecordingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, **kwargs):
            urls.append(url)
            return _Request()

        def post(self, url, **kwargs):
            urls.append(url)
            return _Request()

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: _RecordingSession())
    monkeypatch.setattr(aiohttp, "TCPConnector", lambda *args, **kwargs: None)

    fake_self = SimpleNamespace(_req_timeout_secs=30)
    await OpenAIDisaggServer._sync_server_clock(fake_self, entry)

    assert urls, "the handshake did not request any URL"
    assert set(urls) == {"https://worker:8443/steady_clock_offset"}
