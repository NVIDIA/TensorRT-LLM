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
"""Offline tests for the Anthropic gateway example.

No sockets and no scheduler: the relay is driven with byte strings and the
supervisor with a stubbed launcher, so the rules below can be pinned on CPU.

The gateway is an example, not library code, so this file deliberately covers
only the properties whose failure would be silent or destructive: a corrupted
response body, a request that never reaches the backend it was meant for, an
identity the client got to choose, a fleet that expires with no successor, and
an allocation reclaimed out from under a live stream.
"""

import asyncio
import collections
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[4] / "examples" / "serve" / "anthropic_compatibility" / "gateway.py"
SPEC = importlib.util.spec_from_file_location("anthropic_gateway", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)

# The whole Anthropic surface the server registers. The gateway is expected to
# carry all of it without naming any of it, so these are the paths the
# forwarding and auth tests below sweep over.
ANTHROPIC_PATHS = [
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/messages/batches",
    "/v1/messages/batches/msgbatch_abc123",
    "/v1/messages/batches/msgbatch_abc123/cancel",
    "/v1/messages/batches/msgbatch_abc123/results",
]


class FragmentedReader:
    def __init__(self, fragments):
        self.fragments = collections.deque(bytearray(f) for f in fragments)

    async def read(self, size):
        if not self.fragments:
            return b""
        fragment = self.fragments[0]
        data = bytes(fragment[:size])
        del fragment[:size]
        if not fragment:
            self.fragments.popleft()
        return data


class RecordingWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        pass


class ClientWriter(RecordingWriter):
    """A RecordingWriter that also satisfies what `Gateway.handle` calls."""

    def __init__(self):
        super().__init__()
        self.closed = False

    def get_extra_info(self, _name):
        return ("127.0.0.1", 40000)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


def relay_response(*fragments):
    reader = FragmentedReader(fragments)
    writer = RecordingWriter()
    status = asyncio.run(gateway.Gateway(None).relay_response(reader, writer))
    return status, bytes(writer.data)


def request_head(method, path, headers=()):
    lines = ["%s %s HTTP/1.1" % (method, path), "Host: gateway:8333"]
    lines.extend("%s: %s" % pair for pair in headers)
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def run_handle(fleet, method, path, headers=()):
    writer = ClientWriter()
    asyncio.run(
        gateway.Gateway(fleet).handle(
            FragmentedReader([request_head(method, path, headers)]), writer
        )
    )
    return bytes(writer.data)


def make_args():
    return SimpleNamespace(
        lead_time=2700,
        no_relay=False,
        promote_after=180,
        serve_sh="/unused/serve.sh",
        yaml="/unused/deployment.yaml",
    )


def make_backend(job_id, end_time, healthy=True, state="running attempt 1"):
    backend = gateway.Backend(
        {
            "job_id": job_id,
            "url": "http://localhost:8333",
            "run_dir": "/runs/%s" % job_id,
            "state": state,
            "end_time": end_time,
            "heartbeat": time.time(),
        }
    )
    backend.healthy = healthy
    backend.healthy_since = time.time() - 1000 if healthy else 0
    return backend


def test_batch_results_ndjson_is_relayed_without_an_injected_event():
    # Batch results are application/x-ndjson, not SSE. The terminal-event
    # machinery must stay out of them: appending an SSE error to a .jsonl body
    # would corrupt every line-delimited parser reading it, and nothing about
    # the corruption is visible until a client tries to parse the last line.
    body = b'{"custom_id":"a","result":{"type":"succeeded"}}\n'
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/x-ndjson\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body) + body
    )

    status, relayed = relay_response(response)

    assert status == "200"
    assert relayed == response
    assert gateway.SSE_ROTATED not in relayed


@pytest.mark.parametrize("path", ANTHROPIC_PATHS)
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_anthropic_routes_are_forwarded_verbatim(path, method):
    # The gateway keeps no route table, which is why count_tokens and the
    # Message Batches endpoints need no special casing: the request line goes
    # upstream exactly as it arrived.
    backend = make_backend("job", time.time() + 3600)
    head = gateway.Gateway(None).upstream_head(
        backend,
        method,
        path,
        [
            ("x-api-key", "someone"),
            ("anthropic-version", "2023-06-01"),
            # A client that names itself: the gateway owns this header, so the
            # forged copy must not survive alongside the resolved identity.
            ("X-Gateway-User", "somebody-else"),
        ],
        "someone",
    )
    lines = head.decode("latin-1").split("\r\n")

    assert lines[0] == "%s %s HTTP/1.1" % (method, path)
    # Client-supplied headers survive, but the gateway owns identity: the key
    # is stripped and replaced by the resolved user.
    assert "anthropic-version: 2023-06-01" in lines
    assert not any(line.lower().startswith("x-api-key") for line in lines)
    # Exactly one identity reaches the backend, and it is the resolved one.
    assert [line for line in lines if line.lower().startswith("x-gateway-user")] == [
        "X-Gateway-User: someone"
    ]


@pytest.mark.parametrize("path", ANTHROPIC_PATHS)
def test_unknown_api_key_is_rejected_on_every_route(path):
    fleet = gateway.Fleet(make_args())
    fleet.users = {"someone"}
    fleet.backends = {"job": make_backend("job", time.time() + 3600)}
    fleet.active = "job"

    response = run_handle(fleet, "POST", path, [("x-api-key", "somebody-else")])

    assert response.startswith(b"HTTP/1.1 401 Unauthorized\r\n")
    assert b'"authentication_error"' in response


def test_known_api_key_is_accepted_and_no_backend_answers_with_a_retryable_503():
    """The accepting half of auth, and the shape of the 'come back later'.

    A key on the allowlist has to get past authentication - with no backend
    reachable the request cannot be proxied, but reaching the 503 at all is
    what proves the key was accepted rather than rejected. The 503 itself
    matters too: an overloaded_error with Retry-After is what makes a client
    wait and retry instead of surfacing a hard failure to the user during a
    handover.
    """
    fleet = gateway.Fleet(make_args())
    fleet.users = {"someone"}

    response = run_handle(fleet, "POST", "/v1/messages", [("x-api-key", "someone")])

    assert response.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
    assert b"Retry-After: 20" in response
    assert b'"overloaded_error"' in response


def test_unknown_gateway_path_is_a_404():
    """A path the gateway claims for itself must not be blamed on a backend.

    Everything under /_gateway/ is answered locally, so an unrecognized one is
    a 404 from the gateway - not a 502 naming a backend that was never
    contacted, which sends whoever is debugging it to the wrong machine.
    """
    fleet = gateway.Fleet(make_args())
    fleet.users = {"someone"}

    response = run_handle(fleet, "GET", "/_gateway/nonsense")

    assert response.startswith(b"HTTP/1.1 404 Not Found\r\n")


def test_forward_handover_supersedes_older_backend():
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", 100),
        "new": make_backend("new", 200),
    }
    fleet.active = "old"

    fleet.elect()

    assert fleet.active == "new"
    assert fleet.superseded == {"old"}


def test_failback_keeps_newer_backend_available_for_recovery():
    """Election picks the newest *healthy* backend, not simply the newest.

    Without the health filter the fleet hands routing to a successor that
    cannot serve, and the predecessor that still can is marked superseded and
    released. The successor stays out of `superseded` so it can take over once
    it recovers.
    """
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", 100),
        "new": make_backend("new", 200, healthy=False),
    }
    fleet.active = "new"

    fleet.elect()

    assert fleet.active == "old"
    assert "new" not in fleet.superseded


def test_backend_refresh_takes_the_late_url_and_end_time():
    """A registration file is rewritten as the job learns about itself.

    Both fields below arrive after the first read, so refresh() has to take
    them rather than keep the value it first saw. `end_time` is absent until
    the job resolves its wall clock, and
    `end_time == 0` makes the supervisor skip relay entirely - the fleet then
    expires with no successor queued, which is the one failure the relay exists
    to prevent. `url` changes when a failed successor is restarted in place and
    binds a different port, leaving the gateway proxying a dead address while a
    fresh heartbeat keeps the entry from ever being retired.
    """
    backend = gateway.Backend(
        {
            "job_id": "job",
            "url": "http://localhost:8333",
            "run_dir": "/runs/job",
            "state": "starting",
            "end_time": 0,
            "heartbeat": 1.0,
        }
    )
    assert backend.end_time == 0

    backend.refresh(
        {
            "job_id": "job",
            "url": "http://localhost:9001",
            "state": "running attempt 1",
            "end_time": 4242.0,
            "heartbeat": 2.0,
        }
    )

    assert backend.end_time == 4242.0
    assert backend.url == "http://localhost:9001"
    # The host/port used for every proxied request are re-resolved, not just
    # the display string.
    assert (backend.host, backend.port) == ("localhost", 9001)

    # A later record that simply omits the field must not reset it: dropping
    # back to 0 would disable relay again.
    backend.refresh({"job_id": "job", "state": "running attempt 1", "heartbeat": 3.0})
    assert backend.end_time == 4242.0
    assert backend.url == "http://localhost:9001"


def test_reclaim_waits_for_inflight_when_no_end_time_is_known(monkeypatch):
    """A backend with requests in flight must not be quit out from under them.

    A backend that never published an end_time drains with a deadline in the
    past, so a bare `now <= deadline` is false immediately and the quit kills
    whatever is still streaming. With no real deadline there is nothing to time
    out against, so the drain has to be waited for instead.
    """
    fleet = gateway.Fleet(make_args())
    fleet.backends = {"old": make_backend("old", 0)}
    # end_time 0 - 60: what supervise() computes when the wall clock is unknown.
    fleet.draining["old"] = -60.0
    fleet.inflight["old"] = 1
    calls = []

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)

    asyncio.run(gateway.supervise(fleet))

    assert not calls, "quit a backend that still had a request in flight"
    assert "old" in fleet.draining, "the backend must stay queued for reclaim"

    # Not vacuous: once the last request finishes, the allocation is released.
    fleet.inflight["old"] = 0
    asyncio.run(gateway.supervise(fleet))

    assert calls == [("quit", "/runs/old")]
    assert "old" not in fleet.draining


def test_supervisor_never_reclaims_active_backend(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "active": make_backend("active", time.time() + 10_000),
    }
    fleet.active = "active"
    fleet.ever_active = True
    fleet.draining["active"] = time.time() - 1
    calls = []

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)
    asyncio.run(gateway.supervise(fleet))

    assert not calls
    assert "active" not in fleet.draining


def test_keyless_request_is_rejected_unless_anonymous_is_allowlisted():
    fleet = gateway.Fleet(make_args())
    fleet.users = {"someone"}
    fleet.backends = {"job": make_backend("job", time.time() + 3600)}
    fleet.active = "job"

    assert run_handle(fleet, "POST", "/v1/messages").startswith(b"HTTP/1.1 401 ")

    fleet.users = {"someone", gateway.ANONYMOUS_USER}
    # With no backend reachable the request cannot be proxied, but it does get
    # past the allowlist, which is what this asserts.
    fleet.active = None
    assert run_handle(fleet, "POST", "/v1/messages").startswith(b"HTTP/1.1 503 ")


# ---------------------------------------------------------------------------
# Discovery resilience
# ---------------------------------------------------------------------------


# Every one of these is valid JSON that a half-written or hand-edited
# registration file can plausibly contain. The bare containers are the
# dangerous ones: they reach .get()/.rstrip() and raise AttributeError, which
# is not a coercion error, so an except tuple listing only ValueError/TypeError
# lets them escape discover() entirely.
def _malformed(body):
    """A record-shaped registration carrying a current heartbeat.

    The heartbeat is load-bearing: discover() drops anything staler than
    --stale-after before it constructs a Backend, so a record with a zero
    heartbeat is skipped at the staleness check and never reaches the guard the
    parameter is meant to exercise.
    """
    return json.dumps({"heartbeat": time.time(), **body})


MALFORMED_REGISTRATIONS = [
    # Bare containers reach .get() and raise AttributeError, which is not a
    # coercion error -- an except tuple of ValueError/TypeError alone lets
    # them escape discover() entirely.
    pytest.param("[1, 2, 3]", id="bare_array"),
    pytest.param('"just-a-string"', id="bare_string"),
    pytest.param("5", id="bare_number"),
    pytest.param("null", id="bare_null"),
    # Record-shaped and fresh, so they reach Backend construction: a non-string
    # url hits .rstrip() (AttributeError), a missing one hits record["url"]
    # (KeyError).
    pytest.param(_malformed({"job_id": "bad", "url": 1234}), id="non_string_url"),
    pytest.param(_malformed({"job_id": "bad"}), id="missing_url"),
    # Caught earlier, in the coercion guard rather than at construction.
    pytest.param(
        '{"job_id": "bad", "url": "http://h:1", "heartbeat": "nan-x"}', id="non_numeric_heartbeat"
    ),
    # Non-finite timestamps. NaN is the dangerous one: every comparison
    # against it is False, so `now - nan > stale_after` never fires and a dead
    # backend would sit in the table forever without ever being retired.
    pytest.param(
        _malformed({"job_id": "bad", "url": "http://h:1", "heartbeat": float("nan")}),
        id="nan_heartbeat",
    ),
    pytest.param(
        json.dumps({"job_id": "bad", "url": "http://h:1", "heartbeat": float("inf")}),
        id="inf_heartbeat",
    ),
    pytest.param(
        _malformed({"job_id": "bad", "url": "http://h:1", "end_time": float("nan")}),
        id="nan_end_time",
    ),
    # Not JSON at all.
    pytest.param("{", id="truncated"),
    pytest.param("", id="empty"),
]


@pytest.mark.parametrize("content", MALFORMED_REGISTRATIONS)
def test_one_bad_registration_does_not_abort_the_sweep(tmp_path, content):
    """A single unusable file must cost only itself, never the whole sweep.

    discover() rebuilds the entire table in one pass, so an exception escaping
    the per-file guard does not merely skip that backend: it aborts before the
    retirement loop and the active-pointer check at the end, so stale backends
    are never dropped and `active` can keep naming a job that is already gone.
    The offending file stays in the directory, so the failure repeats on every
    sweep.
    """
    args = make_args()
    args.fleet_dir = str(tmp_path)
    args.stale_after = 30
    fleet = gateway.Fleet(args)

    good = {
        "job_id": "good",
        "url": "http://localhost:8333",
        "run_dir": "/runs/good",
        "state": "running attempt 1",
        "end_time": time.time() + 3600,
        "heartbeat": time.time(),
    }
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text(content)

    fleet.discover()

    # The healthy neighbour is discovered despite sharing the directory.
    assert "good" in fleet.backends
    assert fleet.backends["good"].url == "http://localhost:8333"
    assert "bad" not in fleet.backends


def test_retirement_still_runs_when_a_bad_registration_is_present(tmp_path):
    """The retirement pass is the half a mid-sweep exception silently skips."""
    args = make_args()
    args.fleet_dir = str(tmp_path)
    args.stale_after = 30
    fleet = gateway.Fleet(args)
    fleet.backends = {"gone": make_backend("gone", time.time() + 3600)}
    fleet.active = "gone"
    fleet.inflight = {"gone": 0}

    # No file for "gone", so a complete sweep must retire it -- and it has to
    # get there past the unusable file.
    (tmp_path / "bad.json").write_text("[1, 2, 3]")

    fleet.discover()

    assert "gone" not in fleet.backends
    assert fleet.active is None


def test_registration_url_with_a_trailing_slash_survives_refresh(tmp_path):
    """refresh() must normalise the url the same way __init__ does.

    Without the rstrip, a rewrite that only adds a trailing slash reads as a
    change, fails the regex, and warns on every sweep while the url never moves.
    """
    backend = make_backend("job", time.time() + 3600)
    backend.refresh(
        {
            "job_id": "job",
            "url": "http://localhost:8333/",
            "heartbeat": time.time(),
            "end_time": time.time() + 3600,
        }
    )

    assert backend.url == "http://localhost:8333"
    assert backend.host == "localhost"
    assert backend.port == 8333


def test_nan_heartbeat_cannot_keep_a_dead_backend_alive(tmp_path):
    """NaN must be rejected, not merely tolerated.

    float("nan") is accepted by float(), and then `now - nan > stale_after` is
    False because every NaN comparison is False. A backend whose registration
    carries a NaN heartbeat would therefore never look stale and never be
    retired -- the gateway would keep electing a job that is long gone.
    """
    args = make_args()
    args.fleet_dir = str(tmp_path)
    args.stale_after = 30
    fleet = gateway.Fleet(args)

    (tmp_path / "dead.json").write_text(
        json.dumps(
            {
                "job_id": "dead",
                "url": "http://localhost:8333",
                "end_time": time.time() + 3600,
                "heartbeat": float("nan"),
            }
        )
    )

    fleet.discover()

    assert "dead" not in fleet.backends
