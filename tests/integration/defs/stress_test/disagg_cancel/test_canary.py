# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for `DisaggCancellationStressHarness._canary_thread_body`.

Loader and token-equivalence helpers run directly. Transport and
thread behavior run against an in-process HTTP server that answers
`POST /v1/completions` with configurable payloads.
"""

from __future__ import annotations

import json
import socket
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from .harness import (
    DisaggCancellationStressHarness,
    _load_canary_prompts,
    _send_canary_request,
    _tokens_equivalent,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _pick_port() -> int:
    """Bind-and-release to return an OS-allocated free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CompletionsHandler(BaseHTTPRequestHandler):
    """Answers `POST /v1/completions`.

    Tests mutate `server.token_ids` / `text` / `status` / `raw_body`
    to drive outcomes and read `server.request_bodies` for
    wire-format assertions.
    """

    def do_POST(self) -> None:  # noqa: N802 — fixed by BaseHTTPRequestHandler
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            self.server.request_bodies.append(json.loads(body))  # type: ignore[attr-defined]
        except (json.JSONDecodeError, AttributeError):
            pass

        status = getattr(self.server, "status", 200)
        if status != 200:
            self.send_response(status)
            self.end_headers()
            return

        raw_body = getattr(self.server, "raw_body", None)
        if raw_body is not None:
            encoded = raw_body.encode("utf-8")
        else:
            payload = {
                "choices": [
                    {
                        "token_ids": getattr(self.server, "token_ids", None),
                        "text": getattr(self.server, "text", ""),
                    }
                ]
            }
            encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args, **_kwargs) -> None:  # silence test noise
        pass


@pytest.fixture
def completions_server():
    """In-process HTTP server answering `POST /v1/completions`; yields `(server, port)`."""
    port = _pick_port()
    server = HTTPServer(("127.0.0.1", port), _CompletionsHandler)
    server.token_ids = [1, 2, 3]  # type: ignore[attr-defined]
    server.text = "ok"  # type: ignore[attr-defined]
    server.status = 200  # type: ignore[attr-defined]
    server.raw_body = None  # type: ignore[attr-defined]
    server.request_bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _write_prompts(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "stress_canary_prompts.json"
    path.write_text(json.dumps({"prompts": entries}), encoding="utf-8")
    return path


_CANARY_YAML = textwrap.dedent(
    """\
    hostname: localhost
    model: dummy
    backend: pytorch
    context_servers: {{}}
    generation_servers: {{}}
    stress_config:
      duration_min: 1
      kv_cache_manager: v1
      transceiver: cpp
      canary:
        prompts_file: {prompts_file}
        rate_per_min: 5
        max_tokens: 8
        seed: 42
        check_token_equivalent: {check_equiv}
    """
)


def _make_harness(
    tmp_path: Path,
    *,
    prompts_file: str = "stress_canary_prompts.json",
    check_equiv: bool = True,
    server_url: str | None = "http://127.0.0.1:1",
) -> DisaggCancellationStressHarness:
    """Construct a canary harness with a small interval and bound endpoint."""
    yaml_path = tmp_path / "marathon.yaml"
    yaml_path.write_text(
        _CANARY_YAML.format(
            prompts_file=prompts_file,
            check_equiv="true" if check_equiv else "false",
        )
    )
    h = DisaggCancellationStressHarness(
        yaml_path,
        canary_interval_s=0.02,
        canary_request_timeout_s=1.0,
    )
    if server_url is not None:
        h.bind_server_endpoint(server_url, "test-model")
    return h


def _run_canary_thread_briefly(h: DisaggCancellationStressHarness, duration_s: float) -> None:
    """Drive `_canary_thread_body` for `duration_s` then stop."""
    thread = threading.Thread(target=h._canary_thread_body, daemon=True)
    thread.start()
    time.sleep(duration_s)
    h.stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "canary thread failed to exit after stop_event"


def _run_until_self_exit(h: DisaggCancellationStressHarness) -> None:
    """Spawn the thread and wait for it to exit on its own (warn-and-return paths)."""
    thread = threading.Thread(target=h._canary_thread_body, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def _wait_until(predicate, *, timeout_s: float, poll_s: float = 0.01) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise AssertionError(f"predicate did not become true within {timeout_s}s")


# ---------------------------------------------------------------------------
# Prompt-loader tests
# ---------------------------------------------------------------------------


def test_load_prompts_valid(tmp_path: Path) -> None:
    path = _write_prompts(
        tmp_path,
        [
            {"prompt": "hello", "reference_token_ids": [1, 2]},
            {"prompt": "world", "reference_token_ids": [3, 4], "reference_text": "w"},
        ],
    )
    prompts = _load_canary_prompts(path)
    assert len(prompts) == 2
    assert prompts[0]["prompt"] == "hello"
    assert prompts[1]["reference_token_ids"] == [3, 4]


def test_load_prompts_reference_token_ids_omitted_is_allowed(tmp_path: Path) -> None:
    path = _write_prompts(tmp_path, [{"prompt": "p"}])
    assert _load_canary_prompts(path) == [{"prompt": "p"}]


@pytest.mark.parametrize(
    "raw_content,match",
    [
        pytest.param("{not valid json", "not valid JSON", id="malformed-json"),
        pytest.param(json.dumps({"items": []}), "'prompts' list", id="missing-prompts-key"),
        pytest.param(
            json.dumps({"prompts": {"prompt": "x"}}), "must be a list", id="prompts-not-list"
        ),
        pytest.param(
            json.dumps({"prompts": [{"reference_token_ids": [1]}]}),
            "string 'prompt'",
            id="entry-missing-prompt",
        ),
        pytest.param(
            json.dumps({"prompts": [{"prompt": "p", "reference_token_ids": 123}]}),
            "reference_token_ids",
            id="reftokens-not-list",
        ),
        pytest.param(
            json.dumps({"prompts": [{"prompt": "p", "reference_token_ids": [1, "two"]}]}),
            "reference_token_ids",
            id="reftokens-non-int-element",
        ),
    ],
)
def test_load_prompts_invalid_raises(tmp_path: Path, raw_content: str, match: str) -> None:
    path = tmp_path / "p.json"
    path.write_text(raw_content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _load_canary_prompts(path)


# ---------------------------------------------------------------------------
# Token-equivalence helper tests
# ---------------------------------------------------------------------------


def test_tokens_equivalent_exact_match() -> None:
    assert _tokens_equivalent([1, 2, 3], [1, 2, 3]) is True


def test_tokens_equivalent_mismatch() -> None:
    assert _tokens_equivalent([1, 2, 3], [1, 2, 4]) is False
    assert _tokens_equivalent([1, 2], [1, 2, 3]) is False


def test_tokens_equivalent_none_is_false() -> None:
    assert _tokens_equivalent(None, [1, 2]) is False
    assert _tokens_equivalent([1, 2], None) is False
    assert _tokens_equivalent(None, None) is False


# ---------------------------------------------------------------------------
# Send-request tests (in-process HTTP server)
# ---------------------------------------------------------------------------


def test_send_success_returns_token_ids(completions_server) -> None:
    server, port = completions_server
    server.token_ids = [10, 20, 30]
    server.text = "hi"
    token_ids, text, err = _send_canary_request(
        f"http://127.0.0.1:{port}", "m", "prompt", max_tokens=8, seed=42, timeout_s=1.0
    )
    assert err is None
    assert token_ids == [10, 20, 30]
    assert text == "hi"


def test_send_http_503_returns_http_error(completions_server) -> None:
    server, port = completions_server
    server.status = 503
    token_ids, text, err = _send_canary_request(
        f"http://127.0.0.1:{port}", "m", "prompt", max_tokens=8, seed=42, timeout_s=1.0
    )
    assert token_ids is None and text is None
    assert err is not None and err.startswith("http_error: 503")


def test_send_connection_refused_returns_url_error() -> None:
    port = _pick_port()  # nobody listening
    token_ids, _text, err = _send_canary_request(
        f"http://127.0.0.1:{port}", "m", "prompt", max_tokens=8, seed=42, timeout_s=0.5
    )
    assert token_ids is None
    assert err is not None and err.startswith("url_error")


def test_send_malformed_body_returns_parse_error(completions_server) -> None:
    server, port = completions_server
    server.raw_body = "{not json"
    token_ids, _text, err = _send_canary_request(
        f"http://127.0.0.1:{port}", "m", "prompt", max_tokens=8, seed=42, timeout_s=1.0
    )
    assert token_ids is None
    assert err is not None and err.startswith("parse_error")


def test_send_missing_token_ids_returns_error(completions_server) -> None:
    # 200 with `choices[0]` lacking `token_ids` is a server problem,
    # not a token-equivalence mismatch downstream.
    server, port = completions_server
    server.token_ids = None
    server.text = "hi"
    token_ids, text, err = _send_canary_request(
        f"http://127.0.0.1:{port}", "m", "prompt", max_tokens=8, seed=42, timeout_s=1.0
    )
    assert token_ids is None
    assert text == "hi"
    assert err == "missing_token_ids"


def test_send_wire_format_includes_greedy_determinism_knobs(completions_server) -> None:
    # Pin the wire shape so a future change can't silently drop
    # `temperature=0.0` / `seed` / `detokenize=False`.
    server, port = completions_server
    _send_canary_request(
        f"http://127.0.0.1:{port}",
        "test-model",
        "the prompt",
        max_tokens=11,
        seed=7,
        timeout_s=1.0,
    )
    assert len(server.request_bodies) == 1
    body = server.request_bodies[0]
    assert body["model"] == "test-model"
    assert body["prompt"] == "the prompt"
    assert body["max_tokens"] == 11
    assert body["temperature"] == 0.0
    assert body["seed"] == 7
    assert body["stream"] is False
    assert body["detokenize"] is False


# ---------------------------------------------------------------------------
# Thread-body integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server_tokens,ref_tokens,check_equiv,expected",
    [
        pytest.param([1, 2, 3], [1, 2, 3], True, True, id="match"),
        pytest.param([9, 9, 9], [1, 2, 3], True, False, id="mismatch"),
        pytest.param([1, 2, 3], None, True, None, id="no-reference"),
        pytest.param([1, 2, 3], [1, 2, 3], False, None, id="check-disabled"),
    ],
)
def test_thread_records_token_equivalent(
    tmp_path, completions_server, server_tokens, ref_tokens, check_equiv, expected
) -> None:
    server, port = completions_server
    server.token_ids = server_tokens
    entry: dict = {"prompt": "p"}
    if ref_tokens is not None:
        entry["reference_token_ids"] = ref_tokens
    _write_prompts(tmp_path, [entry])
    h = _make_harness(tmp_path, check_equiv=check_equiv, server_url=f"http://127.0.0.1:{port}")

    _run_canary_thread_briefly(h, duration_s=0.15)

    assert len(h._canary_records) >= 1
    for rec in h._canary_records:
        assert rec["success"] is True
        assert rec["error"] is None
        assert rec["token_equivalent"] is expected


def test_thread_records_error_when_server_down(tmp_path) -> None:
    port = _pick_port()  # nobody listening
    _write_prompts(tmp_path, [{"prompt": "p", "reference_token_ids": [1]}])
    h = _make_harness(tmp_path, server_url=f"http://127.0.0.1:{port}")

    _run_canary_thread_briefly(h, duration_s=0.15)

    assert len(h._canary_records) >= 1
    for rec in h._canary_records:
        assert rec["success"] is False
        assert rec["token_equivalent"] is None
        assert rec["error"] is not None


def test_thread_round_robins_prompts(tmp_path, completions_server) -> None:
    server, port = completions_server
    server.token_ids = [1]
    _write_prompts(
        tmp_path,
        [
            {"prompt": "p0", "reference_token_ids": [1]},
            {"prompt": "p1", "reference_token_ids": [1]},
            {"prompt": "p2", "reference_token_ids": [1]},
        ],
    )
    h = _make_harness(tmp_path, server_url=f"http://127.0.0.1:{port}")

    thread = threading.Thread(target=h._canary_thread_body, daemon=True)
    thread.start()
    try:
        _wait_until(lambda: len(h._canary_records) >= 4, timeout_s=2.0)
    finally:
        h.stop_event.set()
        thread.join(timeout=2.0)

    seen = {rec["prompt_index"] for rec in h._canary_records}
    assert {0, 1, 2}.issubset(seen)


def test_thread_exits_when_no_server_url(tmp_path) -> None:
    _write_prompts(tmp_path, [{"prompt": "p", "reference_token_ids": [1]}])
    h = _make_harness(tmp_path, server_url=None)
    _run_until_self_exit(h)
    assert h._canary_records == []


def test_thread_exits_when_prompts_file_missing(tmp_path, completions_server) -> None:
    _, port = completions_server
    h = _make_harness(
        tmp_path, prompts_file="does_not_exist.json", server_url=f"http://127.0.0.1:{port}"
    )
    _run_until_self_exit(h)
    assert h._canary_records == []


def test_thread_exits_when_prompts_list_empty(tmp_path, completions_server) -> None:
    _, port = completions_server
    _write_prompts(tmp_path, [])
    h = _make_harness(tmp_path, server_url=f"http://127.0.0.1:{port}")
    _run_until_self_exit(h)
    assert h._canary_records == []


def test_thread_exits_promptly_on_failed_event(tmp_path, completions_server) -> None:
    # The between-request wait only observes `stop_event`; verify
    # `failed_event` is acted on within one canary interval (records
    # grow by at most one after the event fires).
    _, port = completions_server
    _write_prompts(tmp_path, [{"prompt": "p", "reference_token_ids": [1, 2, 3]}])
    h = _make_harness(tmp_path, server_url=f"http://127.0.0.1:{port}")

    thread = threading.Thread(target=h._canary_thread_body, daemon=True)
    thread.start()
    _wait_until(lambda: len(h._canary_records) >= 1, timeout_s=2.0)
    pre = len(h._canary_records)
    h.failed_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(h._canary_records) - pre <= 1


def test_thread_resolves_absolute_prompts_path(tmp_path, completions_server) -> None:
    # Absolute `prompts_file` must NOT be joined against `yaml_path.parent`.
    server, port = completions_server
    server.token_ids = [1, 2, 3]
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    prompts_path = outside_dir / "abs_prompts.json"
    prompts_path.write_text(
        json.dumps({"prompts": [{"prompt": "p", "reference_token_ids": [1, 2, 3]}]}),
        encoding="utf-8",
    )
    h = _make_harness(
        tmp_path, prompts_file=str(prompts_path), server_url=f"http://127.0.0.1:{port}"
    )

    _run_canary_thread_briefly(h, duration_s=0.15)

    assert len(h._canary_records) >= 1
    assert all(rec["success"] is True for rec in h._canary_records)
    assert all(rec["token_equivalent"] is True for rec in h._canary_records)
