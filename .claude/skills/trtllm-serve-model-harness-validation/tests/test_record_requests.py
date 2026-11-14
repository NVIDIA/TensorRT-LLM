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
"""The recorder keeps a record of every stream, however it ended (CPU only)."""

import asyncio
import json
import os
import sys

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

import record_requests  # noqa: E402

CHUNKS = [b'event: ping\ndata: {"type": "ping"}\n\n'] * 3


async def _upstream(request):
    resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    for chunk in CHUNKS:
        await resp.write(chunk)
        await asyncio.sleep(0.01)
    if request.query.get("break"):
        # The upstream dies mid-stream: no terminating chunk, connection gone.
        request.transport.close()
        return resp
    await resp.write_eof()
    return resp


async def _run(out_dir, path="/v1/messages", build_id=None, which="responses"):
    upstream_app = web.Application()
    upstream_app.router.add_route("POST", "/v1/messages", _upstream)
    async with TestServer(upstream_app) as upstream:
        recorder = record_requests.build_app(
            str(out_dir), forward=str(upstream.make_url("")), build_id=build_id
        )
        async with TestServer(recorder) as server:
            async with aiohttp.ClientSession() as client:
                try:
                    async with client.post(server.make_url(path), json={"stream": True}) as reply:
                        await reply.read()
                except aiohttp.ClientError:
                    pass
    with open(os.path.join(out_dir, f"{which}-capture.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def test_a_completed_stream_is_recorded(tmp_path):
    (record,) = asyncio.run(_run(tmp_path))

    assert record["status"] == "completed"
    assert record["response"]["body"] == b"".join(CHUNKS).decode()
    assert "first_byte_s" in record["timing"]


def test_each_request_reaches_the_upstream_on_a_new_connection(tmp_path):
    """An upstream that closes idle connections never gets a request on a closed one.

    trtllm-serve closes a keep-alive connection after 5 s idle. A recorder that reuses a pooled
    connection as it closes fails the request before the server reads it, and R-06 then judges
    the server for the recorder's loss.
    """
    peers = set()

    async def upstream(request):
        peer = request.transport.get_extra_info("peername")
        if peer in peers:
            # A reused connection: answer the way a server that closed it while idle does.
            request.transport.close()
            return web.Response()
        peers.add(peer)
        return await _upstream(request)

    async def run():
        upstream_app = web.Application()
        upstream_app.router.add_route("POST", "/v1/messages", upstream)
        async with TestServer(upstream_app) as up:
            recorder = record_requests.build_app(str(tmp_path), forward=str(up.make_url("")))
            async with TestServer(recorder) as server:
                async with aiohttp.ClientSession() as client:
                    for _ in range(3):
                        url = server.make_url("/v1/messages")
                        async with client.post(url, json={"stream": True}) as reply:
                            await reply.read()
        with open(os.path.join(tmp_path, "responses-capture.jsonl")) as fh:
            return [json.loads(line) for line in fh]

    records = asyncio.run(run())

    assert [record["status"] for record in records] == ["completed"] * 3


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Connection lost"),
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError(110, "Connection timed out"),
        OSError(113, "No route to host"),
    ],
    ids=["lost", "reset", "timed_out", "no_route"],
)
def test_a_client_that_went_away_is_recorded(tmp_path, monkeypatch, error):
    """Each way a write to a departed client can fail still leaves a record."""
    real_write = web.StreamResponse.write
    calls = {"n": 0}

    async def write(self, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise error
        return await real_write(self, data)

    monkeypatch.setattr(web.StreamResponse, "write", write)

    (record,) = asyncio.run(_run(tmp_path))

    assert record["status"] == "client_disconnected"
    assert record["timing"]["client_error"] == type(error).__name__
    assert record["response"]["body"].startswith(CHUNKS[0].decode())


async def _slow_upstream(request):
    await asyncio.sleep(1.0)  # no headers for a second, like a server under a prefill stall
    resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(CHUNKS[0])
    await resp.write_eof()
    return resp


async def _run_slow(out_dir, client_timeout, stop_recorder_after=None):
    """A client that waits ``client_timeout`` s for a reply that takes a second to start.

    With ``stop_recorder_after``, the recorder is stopped that many seconds in, the request still
    in flight; its graceful-shutdown wait is shorter than the reply's delay, so the handler is
    cancelled.
    """
    upstream_app = web.Application()
    upstream_app.router.add_route("POST", "/v1/messages", _slow_upstream)
    async with TestServer(upstream_app) as upstream:
        recorder = record_requests.build_app(str(out_dir), forward=str(upstream.make_url("")))
        server = TestServer(recorder)
        await server.start_server(shutdown_timeout=0.1)
        timeout = aiohttp.ClientTimeout(total=client_timeout)

        async def call():
            try:
                async with aiohttp.ClientSession(timeout=timeout) as client:
                    async with client.post(
                        server.make_url("/v1/messages"), json={"stream": True}
                    ) as r:
                        await r.read()
            except (asyncio.TimeoutError, aiohttp.ClientError):
                pass

        task = asyncio.ensure_future(call())
        if stop_recorder_after is not None:
            await asyncio.sleep(stop_recorder_after)
            await server.close()
        await task
        await asyncio.sleep(1.5)  # long enough for the upstream to have answered
        if stop_recorder_after is None:
            await server.close()
    with open(os.path.join(out_dir, "responses-capture.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def test_a_client_that_left_before_the_reply_is_recorded(tmp_path):
    """A request whose client gives up before the first byte still leaves a record."""
    records = asyncio.run(_run_slow(tmp_path, client_timeout=0.3))

    assert len(records) == 1, "the request has no response record"
    (record,) = records
    assert record["status"] == "client_disconnected"
    assert record["timing"]["before_reply"] is True
    assert record["response"] == {"kind": "none", "body": None}


def test_a_request_in_flight_when_the_recorder_stops_is_recorded(tmp_path):
    records = asyncio.run(_run_slow(tmp_path, client_timeout=5.0, stop_recorder_after=0.3))

    assert len(records) == 1, "the request has no response record"
    (record,) = records
    assert record["status"] == "recorder_shutdown"
    assert record["timing"]["before_reply"] is True


def test_an_upstream_that_breaks_is_not_the_client(tmp_path):
    (record,) = asyncio.run(_run(tmp_path, "/v1/messages?break=1"))

    assert record["status"] in ("upstream_error", "completed")
    if record["status"] == "upstream_error":
        assert "upstream_error" in record["timing"]
    assert "client_error" not in record["timing"]


def test_every_record_carries_the_build_id(tmp_path):
    """The validator proves which build served from the records (I-05, R-05)."""
    (response,) = asyncio.run(_run(tmp_path / "r", build_id="1.4.0rc2+abc"))
    (request,) = asyncio.run(_run(tmp_path / "q", build_id="1.4.0rc2+abc", which="requests"))

    assert response["build_id"] == request["build_id"] == "1.4.0rc2+abc"


def test_a_running_recorder_records_a_cut_when_it_happens(tmp_path):
    """The recorder as deployed records a client that leaves when it leaves.

    It runs its own main, in a subprocess, and the client leaves while the server is silent: the
    record must not wait for the server's next write.
    """
    import socket
    import subprocess

    async def scenario():
        upstream_app = web.Application()
        upstream_app.router.add_route("POST", "/v1/messages", _slow_upstream)  # silent for 1 s
        async with TestServer(upstream_app) as upstream:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            proc = subprocess.Popen(
                [sys.executable, os.path.join(SCRIPTS, "record_requests.py"), "--out-dir",
                 str(tmp_path), "--port", str(port), "--forward", str(upstream.make_url(""))],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
            try:
                for _ in range(100):
                    with socket.socket() as s:
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    await asyncio.sleep(0.05)
                timeout = aiohttp.ClientTimeout(total=0.3)
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as client:
                        async with client.post(f"http://127.0.0.1:{port}/v1/messages",
                                               json={"stream": True}) as reply:  # fmt: skip
                            await reply.read()
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    pass
                await asyncio.sleep(0.5)  # the upstream is still silent for another 0.2 s
                path = os.path.join(tmp_path, "responses-capture.jsonl")
                with open(path) as fh:
                    return [json.loads(line) for line in fh]
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # a handler still waiting on the upstream
                    proc.kill()
                    proc.wait()

    records = asyncio.run(scenario())

    assert len(records) == 1, "no record while the server was still silent"
    assert records[0]["status"] == "client_disconnected"
    assert records[0]["timing"]["total_s"] < 0.6


def test_the_capture_is_readable_by_its_owner_only(tmp_path):
    """The capture can hold credentials and user code: 0600 files, in a 0700 directory it made."""
    import stat

    out = tmp_path / "capture"
    out.mkdir(mode=0o755)
    (out / "requests-capture.jsonl").write_text("")  # left by an earlier recorder, 0644
    os.chmod(out / "requests-capture.jsonl", 0o644)
    made = tmp_path / "made"

    for out_dir in (out, made):
        app = record_requests.build_app(str(out_dir))
        app["req_out"].close()
        app["resp_out"].close()
        for name in ("requests-capture.jsonl", "responses-capture.jsonl"):
            assert stat.S_IMODE(os.stat(out_dir / name).st_mode) == 0o600, (out_dir, name)
    assert stat.S_IMODE(os.stat(made).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o755  # an existing directory is left as it was


def test_an_override_sets_only_fields_no_check_reads():
    """A reply is judged against the body as recorded: --override takes sampling fields only."""
    assert record_requests.parse_overrides(["temperature=0", "seed=7"]) == {"temperature": 0,
                                                                            "seed": 7}  # fmt: skip
    for item in ("stream=true", "tools=[]", "max_tokens=16", 'reasoning_effort="low"'):
        with pytest.raises(ValueError, match="only sampling fields"):
            record_requests.parse_overrides([item])
    with pytest.raises(ValueError):
        record_requests.parse_overrides(["temperature=zero"])
