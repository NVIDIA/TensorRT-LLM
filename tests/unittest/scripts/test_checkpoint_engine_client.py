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
"""Loopback tests for the real CPU engine client using a simulated worker, not a host driver."""

import copy
import importlib.util
import json
import runpy
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / "examples/serve/checkpoint/engine_client.py"
spec = importlib.util.spec_from_file_location("checkpoint_engine_client", CLIENT)
assert spec is not None and spec.loader is not None
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
auth = runpy.run_path(str(ROOT / "tensorrt_llm/serve/runtime_control_auth.py"))

KEY = "test-only-key"
IDENTITY = {
    "template_id": "template",
    "source_session_id": "source",
    "profile_digest": "a" * 64,
    "cohort_digest": "b" * 64,
}
PROBE = {"prompt_token_ids": [1, 2], "expected_token_ids": [3, 4]}
READY = {
    "capture_safe": False,
    "memory_ready": False,
    "runtime_ready": False,
    "validation_passed": False,
    "serving_ready": False,
    "admission_open": True,
}
CHECKS = {
    "prepare_capture": [
        "admission_closed",
        "scheduler_drained",
        "no_external_exposure",
        "resident_owners_recorded",
    ],
    "validate_runtime": [
        "validation_retired",
        "scheduler_drained",
        "warmed_graph_replayed",
        "resident_state_matches",
    ],
    "activate": ["activation_committed"],
}


class Engine:
    """Simulate journal transitions and worker health; HTTP and checkpoint HMAC are real."""

    def __init__(self) -> None:
        self.status = {
            "stage": "running",
            "state_version": 0,
            "remaining_seconds": None,
            "source_session_id": "source",
            "restore_session_id": None,
            "operation": None,
            "readiness": READY.copy(),
            "profile": {
                "profile_digest": "a" * 64,
                "cohort_digest": "b" * 64,
                "ranks": [0],
                "capture_policies": ["resident"],
                "drain_policies": ["natural"],
                "probe_limits": {
                    "vocab_size": 32,
                    "max_input_len": 12,
                    "max_seq_len": 16,
                    "max_num_tokens": 4,
                    "chunked_prefill": False,
                },
            },
        }
        self.calls: list[tuple[str, dict]] = []
        self.healthy = True
        self.drop_prepare_reply = False
        self.response: Callable[[str, dict], tuple[int, dict]] = lambda _, data: (200, data)

    def reply(self, route: str, request: dict) -> tuple[int, dict]:
        self.calls.append((route, request))
        s = self.status
        if route == "status":
            return self.response(route, copy.deepcopy(s))
        ctx = request["context"]
        assert ctx["expected_state_version"] == s["state_version"]
        assert ctx["protocol_version"] == 1 and 0 < ctx["time_budget_seconds"] <= 3600
        assert ctx["session_kind"] == ("source" if route == "prepare_capture" else "restore")
        session = "source" if route == "prepare_capture" else "restore"
        artifact = {k: IDENTITY[k] for k in ("template_id", "profile_digest", "cohort_digest")}
        binding = {**artifact, "session_id": session}
        assert all(ctx[k] == v for k, v in binding.items())
        ready = s["readiness"]
        if route == "prepare_capture":
            assert s["stage"] == "running"
            assert request["drain_policy"] == "natural" and request["capture_policy"] == "resident"
            s.update(stage="capture_safe", remaining_seconds=3.0)
            ready.update(capture_safe=True, admission_open=False)
        elif route == "validate_runtime":
            assert s["stage"] == "capture_safe" and not ready["admission_open"]
            assert request["source_session_id"] == "source" and request["validation"] == PROBE
            s.update(stage="runtime_ready", restore_session_id=session, remaining_seconds=2.0)
            ready.update(
                capture_safe=False, memory_ready=True, runtime_ready=True, validation_passed=True
            )
        else:
            assert (
                route == "activate"
                and s["stage"] == "runtime_ready"
                and not ready["admission_open"]
            )
            assert request["readiness_id"] == s["operation"]["readiness_id"]
            assert request["host_restore"] == {
                **binding,
                "status": "succeeded",
                "result_id": "final-host-result",
            }
            s.update(stage="serving", remaining_seconds=None)
            ready.update(serving_ready=True, admission_open=True)
        # The protocol exposes a monotone version, not a fixed increment.
        s["state_version"] += {"prepare_capture": 3, "validate_runtime": 1, "activate": 4}[route]
        rank = {
            "rank": 0,
            "session_id": session,
            "profile_digest": artifact["profile_digest"],
            "cohort_digest": artifact["cohort_digest"],
            "checks": CHECKS[route],
            "state_digest": "c" * 64,
        }
        result = {
            **binding,
            "operation_id": ctx["operation_id"],
            "state_version": s["state_version"],
            "status": "completed",
            "readiness_id": None if route == "prepare_capture" else "ready",
            "cleanup_status": "not_requested",
            "native_prepare_not_dispatched": False,
            "error": None,
            "rank_results": [rank],
        }
        s["operation"] = result
        return self.response(route, copy.deepcopy(result))


@pytest.fixture
def endpoint() -> Iterator[tuple[int, Engine]]:
    """Verify signed checkpoint requests and serve the separate public, empty-body /ready."""
    engine = Engine()
    replay = auth["RuntimeControlReplayCache"]()

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/ready":
                assert self.command == "GET" and not body
                engine.calls.append(("ready", {}))
                admitted = engine.status["readiness"]["admission_open"]
                self.send_response(200 if engine.healthy and admitted else 503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            try:
                auth["validate_runtime_control_request"](
                    KEY, self.command, self.path, body, self.headers, replay
                )
            except ValueError:
                code, data = 403, {"detail": KEY}
            else:
                route = self.path.removeprefix("/checkpoint/")
                assert self.path == "/checkpoint/" + route
                assert self.command == ("GET" if route == "status" else "POST")
                code, data = engine.reply(route, json.loads(body) if body else {})
                if route == "prepare_capture" and engine.drop_prepare_reply:
                    self.close_connection = True
                    return
            encoded = json.dumps(data).encode()
            self.send_response(code)
            if code == 307:
                self.send_header("Location", "/must-not-follow")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = handle_request

        def log_message(self, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield server.server_port, engine
        finally:
            server.shutdown()
            thread.join()


def test_import_without_gpu_package() -> None:
    """Sign and verify with no site packages, CUDA, or TensorRT package initialization."""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys,runpy; client=runpy.run_path(sys.argv[1]); "
            "auth=runpy.run_path(str(client['_AUTH'])); "
            "args=('test','GET','/checkpoint/status',b''); headers=client['_sign'](*args); "
            "auth['validate_runtime_control_request'](*args,headers,auth['RuntimeControlReplayCache']()); "
            "assert not {'torch','cuda','tensorrt_llm'} & sys.modules.keys()",
            str(CLIENT),
        ],
        check=True,
        timeout=10,
    )


def test_signed_lifecycle(endpoint: tuple[int, Engine], monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the live source bound, independent restore clock, and shared activation budget."""
    port, engine = endpoint
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    now = [100.0]
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def transit(route: str, data: dict) -> tuple[int, dict]:
        # Charge the entire observation round trip, even if the deadline is mostly remote.
        now[0] += 0.25
        return 200, data

    engine.response = transit
    engine_client = client.EngineClient(port, KEY)
    capture = engine_client.prepare("template", 10, PROBE, before_dispatch=lambda: None)
    assert capture == {**IDENTITY, "remaining_seconds": 2.75, "capture_deadline": 103.5}
    now[0] += 3  # The caller was descheduled at return; the deadline did not move.
    assert now[0] > capture["capture_deadline"]
    engine.status["remaining_seconds"] = 0.0
    engine.status["readiness"]["capture_safe"] = False
    assert (
        engine_client.validate_and_activate(IDENTITY, "restore", "final-host-result", PROBE, 10)
        is None
    )
    assert [route for route, _ in engine.calls] == [
        "status",
        "prepare_capture",
        "status",
        "status",
        "validate_runtime",
        "status",
        "activate",
        "status",
        "ready",
    ]
    contexts = [request["context"] for _, request in engine.calls if "context" in request]
    assert len({ctx["operation_id"] for ctx in contexts}) == 3
    assert [ctx["expected_state_version"] for ctx in contexts] == [0, 3, 4]
    assert [ctx["time_budget_seconds"] for ctx in contexts] == [9.75, 9.75, 1.75]


@pytest.mark.parametrize(
    ("probe", "expected_calls"),
    [
        ({**PROBE, "expected_token_ids": [3]}, []),
        ({**PROBE, "expected_token_ids": [3, 32]}, ["status"]),
    ],
)
def test_source_probe_preflight(
    endpoint: tuple[int, Engine], probe: dict, expected_calls: list[str]
) -> None:
    """Neither a malformed probe nor one outside this model's vocabulary claims the source."""
    port, engine = endpoint
    with pytest.raises(ValueError):
        client.EngineClient(port, KEY).prepare(
            "template", 10, probe, before_dispatch=lambda: pytest.fail("unexpected dispatch")
        )
    assert [route for route, _ in engine.calls] == expected_calls


@pytest.mark.parametrize("lost_reply", [False, True], ids=["journal-failure", "lost-post-reply"])
def test_prepare_dispatch_boundary(endpoint: tuple[int, Engine], lost_reply: bool) -> None:
    """A failed callback prevents the POST; a lost reply still has exactly one prior callback."""
    port, engine = endpoint
    observations = []

    def mark() -> None:
        observations.append([route for route, _ in engine.calls])
        if not lost_reply:
            raise FileExistsError("journal already claimed")

    engine.drop_prepare_reply = lost_reply
    sends = []
    error = RuntimeError if lost_reply else FileExistsError
    with pytest.raises(error, match="outcome unknown" if lost_reply else "journal already claimed"):
        client.EngineClient(port, KEY).prepare(
            "template", 10, PROBE, before_dispatch=mark, on_send=lambda: sends.append(1)
        )
    assert observations == [["status"]]
    assert len(sends) == int(lost_reply)
    assert [route for route, _ in engine.calls] == (
        ["status", "prepare_capture"] if lost_reply else ["status"]
    )


def test_prepare_journal_exhausts_deadline(
    endpoint: tuple[int, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    port, engine = endpoint
    now = [100.0]
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def save_intent() -> None:
        now[0] += 11

    with pytest.raises(TimeoutError, match="deadline expired"):
        client.EngineClient(port, KEY).prepare(
            "template",
            10,
            PROBE,
            before_dispatch=save_intent,
            on_send=lambda: pytest.fail("prepare was not sent"),
        )
    assert [route for route, _ in engine.calls] == ["status"]


@pytest.mark.parametrize(
    "fault", ["tp2", "expired", "wrong_source", "in_progress", "wrong_op", "wrong_rank"]
)
def test_no_capture_from_stale_evidence(endpoint: tuple[int, Engine], fault: str) -> None:
    """Reject stale source state, ambiguous preparation, and non-TP1 evidence."""
    port, engine = endpoint

    def corrupt(route: str, data: dict) -> tuple[int, dict]:
        if fault == "tp2" and route == "status":
            data["profile"]["ranks"] = [0, 1]
        if route == "status" and data["stage"] == "capture_safe":
            if fault == "expired":
                data["remaining_seconds"] = 0
            elif fault == "wrong_source":
                data["source_session_id"] = "other"
            elif fault == "wrong_op":
                data["operation"]["operation_id"] = "previous"
        if route == "prepare_capture":
            if fault == "in_progress":
                data["status"] = "in_progress"
            elif fault == "wrong_rank":
                data["rank_results"][0]["rank"] = False
        return 200, data

    engine.response = corrupt
    with pytest.raises(RuntimeError):
        client.EngineClient(port, KEY).prepare("template", 10, PROBE, before_dispatch=lambda: None)
    assert not any(route == "activate" for route, _ in engine.calls)
    if fault == "tp2":
        assert [route for route, _ in engine.calls] == ["status"]


@pytest.mark.parametrize(
    "fault",
    [
        "source_binding",
        "already_bound",
        "version",
        "unchanged_version",
        "profile",
        "rank_checks",
        "early_open",
        "implicit_pass",
        "expired",
        "stale_ready",
        "inventory",
        "missing_commit",
        "closed_after_commit",
    ],
)
def test_no_success_from_unbound_evidence(endpoint: tuple[int, Engine], fault: str) -> None:
    """Never dispatch activation on bad validation or report success on a bad commit."""
    port, engine = endpoint
    engine_client = client.EngineClient(port, KEY)
    engine_client.prepare("template", 10, PROBE, before_dispatch=lambda: None)

    def corrupt(route: str, data: dict) -> tuple[int, dict]:
        stage = data.get("stage")
        if stage == "capture_safe":
            if fault == "source_binding":
                data["operation"]["template_id"] = "other"
            elif fault == "already_bound":
                data["restore_session_id"] = "prior"
        if stage == "runtime_ready":
            if fault == "version":
                data["state_version"] += 1
            elif fault == "profile":
                data["profile"]["profile_digest"] = "f" * 64
            elif fault == "early_open":
                data["readiness"]["admission_open"] = True
            elif fault == "implicit_pass":
                del data["readiness"]["validation_passed"]
            elif fault == "expired":
                data["remaining_seconds"] = 0
            elif fault == "stale_ready":
                data["operation"]["readiness_id"] = "earlier"
        if route == "validate_runtime" and fault == "rank_checks":
            data["rank_results"][0]["checks"] = []
        if fault == "unchanged_version" and route == "validate_runtime":
            data["state_version"] = engine.calls[-1][1]["context"]["expected_state_version"]
        if fault == "inventory" and (route == "validate_runtime" or stage == "runtime_ready"):
            result = data.get("operation", data)
            result["rank_results"][0]["state_digest"] = "f" * 64
        if stage == "serving" and fault == "closed_after_commit":
            data["readiness"]["admission_open"] = False
        if route == "activate" and fault == "missing_commit":
            data["rank_results"] = []
        return 200, data

    engine.response = corrupt
    with pytest.raises(RuntimeError):
        engine_client.validate_and_activate(IDENTITY, "restore", "final-host-result", PROBE, 10)
    assert not any(route == "ready" for route, _ in engine.calls)
    mutations = [route for route, _ in engine.calls if route != "status"]
    if fault in {"source_binding", "already_bound"}:
        assert mutations == ["prepare_capture"]
    elif fault in {"missing_commit", "closed_after_commit"}:
        assert mutations == ["prepare_capture", "validate_runtime", "activate"]
    else:
        assert mutations == ["prepare_capture", "validate_runtime"]


def test_unhealthy_after_activation(endpoint: tuple[int, Engine]) -> None:
    """A confirmed activation does not override a real HTTP 503 from the readiness double."""
    port, engine = endpoint
    engine_client = client.EngineClient(port, KEY)
    engine_client.prepare("template", 10, PROBE, before_dispatch=lambda: None)
    engine.healthy = False
    with pytest.raises(RuntimeError, match=r"^Checkpoint HTTP 503\.$"):
        engine_client.validate_and_activate(IDENTITY, "restore", "final-host-result", PROBE, 10)
    assert engine.status["stage"] == "serving"
    assert [route for route, _ in engine.calls[-3:]] == ["activate", "status", "ready"]


@pytest.mark.parametrize(
    ("session", "result", "probe"),
    [
        ("source", "host", PROBE),
        ("restore", "", PROBE),
        ("restore", "host", {**PROBE, "expected_token_ids": [1]}),
        ("restore", "host", {**PROBE, "prompt_token_ids": [False]}),
        ("restore", "host", {**PROBE, "expected_token_ids": [1, 32]}),
        ("restore", "host", {**PROBE, "prompt_token_ids": [1] * 5}),
    ],
)
def test_bad_inputs_never_bind_restore(
    endpoint: tuple[int, Engine], session: str, result: str, probe: dict
) -> None:
    """Malformed references and reused sessions never dispatch a restore mutation."""
    port, engine = endpoint
    engine_client = client.EngineClient(port, KEY)
    engine_client.prepare("template", 10, PROBE, before_dispatch=lambda: None)
    with pytest.raises(ValueError):
        engine_client.validate_and_activate(IDENTITY, session, result, probe, 10)
    assert [route for route, _ in engine.calls if route != "status"] == ["prepare_capture"]


def test_http_failures_are_private(
    endpoint: tuple[int, Engine], capsys: pytest.CaptureFixture
) -> None:
    """Neither rejected authentication nor redirects expose server-controlled text or credentials."""
    port, engine = endpoint
    with pytest.raises(RuntimeError, match=r"^Checkpoint HTTP 403\.$"):
        client.EngineClient(port, "incorrect-key").prepare(
            "template", 10, PROBE, before_dispatch=lambda: None
        )
    assert not engine.calls
    engine.response = lambda _, data: (307, {"detail": KEY})
    with pytest.raises(RuntimeError, match=r"^Checkpoint HTTP 307\.$"):
        client.EngineClient(port, KEY).prepare("template", 10, PROBE, before_dispatch=lambda: None)
    assert [route for route, _ in engine.calls] == ["status"]
    out, err = capsys.readouterr()
    assert not out and not err
