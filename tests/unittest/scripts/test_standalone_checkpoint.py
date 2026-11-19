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
"""Standalone checkpoint transactions against a simulated host adapter and engine."""

import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from unittest.mock import ANY, Mock, call

import pytest

pytestmark = pytest.mark.cpu_only

TEMPLATE = "a" * 32
SOURCE = "b" * 64
TARGETS = ("1" * 64, "2" * 64)
DIGEST = "c" * 64
PROBE = {"prompt_token_ids": [17], "expected_token_ids": [23, 29]}
IDENTITY = {
    "template_id": TEMPLATE,
    "source_session_id": "source-session",
    "profile_digest": "d" * 64,
    "cohort_digest": "e" * 64,
}
ADAPTER = Path("/operator/checkpoint-adapter")
KEY = "test-engine-credential"
BUDGET = 120.0


@pytest.fixture
def coordinator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    directory = Path(__file__).resolve().parents[3] / "examples/serve/checkpoint"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("coordinator")


@pytest.fixture
def peers(coordinator: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Mock:
    peers = Mock()

    def prepare(
        template: str,
        budget: float,
        probe: dict,
        *,
        before_dispatch: Callable[[], None],
        on_send: Callable[[], None],
    ) -> dict:
        before_dispatch()
        on_send()
        return peers.engine.return_value.prepare.return_value

    peers.engine.return_value.prepare.side_effect = prepare
    peers.engine.return_value.prepare.return_value = {
        **IDENTITY,
        "remaining_seconds": 60.0,
        "capture_deadline": time.monotonic() + 60,
    }
    peers.engine.return_value.validate_and_activate.return_value = None
    monkeypatch.setattr(coordinator, "EngineClient", peers.engine)
    peers.bind_source.return_value = {"source": SOURCE, "control_port": 1234}
    peers.discard.return_value = {"contained": True}
    peers.publish.side_effect = lambda attempt, target: {"target": target, "published": True}

    def snapshot(action: str, attempt: str, container: str, **facts: object) -> dict:
        return {
            "snapshot_revision": coordinator.SNAPSHOT_REVISION,
            "operation": action,
            "attempt": attempt,
            "artifact": TEMPLATE,
            "container": container,
            "status": "PASS",
            **facts,
        }

    def captured(attempt: str, **fields: object) -> dict:
        return {
            "source": SOURCE,
            "identity": IDENTITY,
            "artifact_digest": DIGEST,
            "source_terminated": True,
            "snapshot": snapshot("capture", attempt, SOURCE, source_terminated=True),
        }

    targets = iter(TARGETS)

    def restored(attempt: str, **fields: object) -> dict:
        target = next(targets)
        return {
            "source": SOURCE,
            "identity": IDENTITY,
            "artifact_digest": DIGEST,
            "target": target,
            "control_port": 4321,
            "isolated": True,
            "authority_established": True,
            "host_result_id": attempt,
            "snapshot": snapshot("restore", attempt, target, memory_ready=True, restored_pid=123),
        }

    peers.capture.side_effect = captured
    peers.restore.side_effect = restored

    def run(
        argv: list[Path], *, input: bytes, stdout: int, env: dict, timeout: float, check: bool
    ) -> subprocess.CompletedProcess:
        request = json.loads(input)
        assert argv == [ADAPTER] and stdout == subprocess.PIPE and check
        assert env == {"PATH": os.defpath}
        assert KEY.encode() not in input
        assert set(request) == {
            "version",
            "action",
            "attempt",
            "template",
            "deadline_monotonic",
            "input",
        }
        assert request["version"] == 1 and request["template"] == TEMPLATE
        assert 0 < timeout <= BUDGET and request["deadline_monotonic"] > 0
        action = request["action"]
        assert action in {"bind_source", "capture", "restore", "publish", "discard"}
        result = getattr(peers, action)(request["attempt"], **request["input"])
        receipt = {k: request[k] for k in ("version", "action", "attempt", "template")}
        receipt.update(status="succeeded", result=result)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(receipt).encode())

    monkeypatch.setattr(coordinator.subprocess, "run", run)
    return peers


@pytest.fixture
def capture(coordinator: ModuleType, tmp_path: Path, peers: Mock) -> Callable[[], None]:
    return partial(
        coordinator.capture,
        tmp_path,
        TEMPLATE,
        SOURCE,
        PROBE,
        coordinator.Host(ADAPTER),
        KEY,
        BUDGET,
    )


@pytest.fixture
def restore(coordinator: ModuleType, tmp_path: Path, peers: Mock) -> Callable[[], str]:
    return partial(coordinator.restore, tmp_path, TEMPLATE, coordinator.Host(ADAPTER), KEY, BUDGET)


def test_capture_then_two_distinct_restores(capture, restore, peers: Mock, tmp_path: Path) -> None:
    capture()
    manifest = tmp_path / TEMPLATE / "manifest.json"
    assert not (manifest.parent / "pending-capture.json").exists()
    assert json.loads(manifest.read_bytes()) == {
        "version": 1,
        "source": SOURCE,
        "identity": IDENTITY,
        "artifact_digest": DIGEST,
        "probe": PROBE,
    }
    assert peers.mock_calls == [
        call.bind_source(TEMPLATE, source=SOURCE),
        call.engine(1234, KEY),
        call.engine().prepare(TEMPLATE, ANY, PROBE, before_dispatch=ANY, on_send=ANY),
        call.capture(TEMPLATE, source=SOURCE, identity=IDENTITY),
    ]
    sessions = {TEMPLATE, IDENTITY["source_session_id"]}
    for expected_target in TARGETS:
        peers.reset_mock()
        target = restore()
        session = peers.restore.call_args.args[0]
        assert session not in sessions
        sessions.add(session)
        assert target == expected_target
        assert peers.mock_calls == [
            call.restore(session, source=SOURCE, identity=IDENTITY, artifact_digest=DIGEST),
            call.engine(4321, KEY),
            call.engine().validate_and_activate(IDENTITY, session, session, PROBE, ANY),
            call.publish(session, target=target),
        ]


@pytest.mark.parametrize(
    "fault", ["stale-snapshot", "driver-not-terminated", "source-not-terminated"]
)
def test_failed_capture_never_commits_manifest(
    capture, restore, peers: Mock, tmp_path: Path, fault: str
) -> None:
    result = peers.capture.side_effect(TEMPLATE)
    if fault == "stale-snapshot":
        result["snapshot"]["attempt"] = "0" * 32
    elif fault == "driver-not-terminated":
        result["snapshot"]["source_terminated"] = False
    else:
        result["source_terminated"] = False
    peers.capture.side_effect = None
    peers.capture.return_value = result
    with pytest.raises(RuntimeError):
        capture()
    peers.discard.assert_called_once_with(TEMPLATE)
    assert not (tmp_path / TEMPLATE / "manifest.json").exists()
    peers.reset_mock()
    with pytest.raises(FileNotFoundError):
        restore()
    with pytest.raises(FileExistsError):
        capture()
    assert not peers.mock_calls


def test_duplicate_capture_and_stale_manifest(
    capture, restore, peers: Mock, tmp_path: Path
) -> None:
    capture()
    peers.reset_mock()
    path = tmp_path / TEMPLATE / "manifest.json"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        capture()
    assert path.read_bytes() == original
    manifest = json.loads(original)
    manifest["identity"]["template_id"] = "0" * 32
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        restore()
    assert not peers.mock_calls


def test_elapsed_capture_deadline_prevents_host_capture(
    capture, peers: Mock, tmp_path: Path
) -> None:
    prepare = peers.engine.return_value.prepare
    prepare.return_value["capture_deadline"] = time.monotonic() - 1
    assert prepare.return_value["remaining_seconds"] > 0
    with pytest.raises(TimeoutError):
        capture()
    prepare.assert_called_once_with(TEMPLATE, ANY, PROBE, before_dispatch=ANY, on_send=ANY)
    peers.capture.assert_not_called()
    peers.discard.assert_called_once_with(TEMPLATE)
    assert not (tmp_path / TEMPLATE / "manifest.json").exists()


@pytest.mark.parametrize(
    "fault", ["preflight-get-failed", "deadline-after-journal", "prepare-reply-lost"]
)
def test_prepare_failure_discards_only_after_dispatch(
    capture, peers: Mock, tmp_path: Path, fault: str
) -> None:
    directory = tmp_path / TEMPLATE
    pending = directory / "pending-capture.json"

    def fail_prepare(
        template: str,
        budget: float,
        probe: dict,
        *,
        before_dispatch: Callable[[], None],
        on_send: Callable[[], None],
    ) -> None:
        assert not directory.exists()
        if fault == "preflight-get-failed":
            raise ConnectionError("preflight GET failed")
        before_dispatch()
        assert pending.is_file()
        if fault == "deadline-after-journal":
            raise TimeoutError("preparation deadline expired before send")
        on_send()
        raise TimeoutError("prepare reply lost")

    peers.engine.return_value.prepare.side_effect = fail_prepare
    with pytest.raises(ConnectionError if fault == "preflight-get-failed" else TimeoutError):
        capture()
    expected = [
        call.bind_source(TEMPLATE, source=SOURCE),
        call.engine(1234, KEY),
        call.engine().prepare(TEMPLATE, ANY, PROBE, before_dispatch=ANY, on_send=ANY),
    ]
    if fault == "prepare-reply-lost":
        expected.append(call.discard(TEMPLATE))
    if fault != "preflight-get-failed":
        assert pending.is_file()
        assert not (directory / "manifest.json").exists()
    else:
        assert not directory.exists()
    assert peers.mock_calls == expected


def test_uncertain_manifest_save_blocks_restore(
    coordinator: ModuleType,
    capture,
    restore,
    peers: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / TEMPLATE
    manifest = directory / "manifest.json"
    pending = directory / "pending-capture.json"
    save = coordinator._save

    def save_then_fail(path: Path, record: dict) -> None:
        save(path, record)
        if path == manifest:
            assert pending.is_file()
            raise OSError("manifest save unconfirmed")

    with monkeypatch.context() as patch:
        patch.setattr(coordinator, "_save", save_then_fail)
        with pytest.raises(OSError, match="manifest save unconfirmed"):
            capture()
    assert json.loads(manifest.read_bytes())["artifact_digest"] == DIGEST
    assert pending.is_file()
    peers.reset_mock()
    with pytest.raises(RuntimeError, match="publication is unconfirmed"):
        restore()
    assert not peers.mock_calls


@pytest.mark.parametrize(
    "fault",
    [
        "stale-snapshot",
        "bare-snapshot",
        "wrong-host-result",
        "isolated",
        "authority_established",
        "timeout",
    ],
)
def test_host_failure_never_activates(capture, restore, peers: Mock, fault: str) -> None:
    capture()
    peers.reset_mock()
    restored = peers.restore.side_effect

    def faulty(attempt: str, **fields: object) -> dict:
        result = restored(attempt, **fields)
        if fault == "bare-snapshot":
            return result["snapshot"]
        if fault == "stale-snapshot":
            result["snapshot"]["attempt"] = TEMPLATE
        elif fault == "wrong-host-result":
            result["host_result_id"] = TEMPLATE
        elif fault in ("isolated", "authority_established"):
            result[fault] = False
        else:
            raise TimeoutError("restore reply lost")
        return result

    peers.restore.side_effect = faulty
    with pytest.raises(TimeoutError if fault == "timeout" else RuntimeError):
        restore()
    attempt = peers.restore.call_args.args[0]
    assert peers.mock_calls == [
        call.restore(attempt, source=SOURCE, identity=IDENTITY, artifact_digest=DIGEST),
        call.discard(attempt),
    ]


@pytest.mark.parametrize("contained", [True, False])
def test_validation_failure_never_publishes(capture, restore, peers: Mock, contained: bool) -> None:
    capture()
    peers.reset_mock()
    validate = peers.engine.return_value.validate_and_activate
    validate.side_effect = RuntimeError("token mismatch")
    peers.discard.return_value = {"contained": contained}
    with pytest.raises(RuntimeError, match="token mismatch"):
        restore()
    attempt = peers.restore.call_args.args[0]
    validate.assert_called_once_with(IDENTITY, attempt, attempt, PROBE, ANY)
    peers.discard.assert_called_once_with(attempt)
    peers.publish.assert_not_called()
    peers.reset_mock()
    if not contained:
        with pytest.raises(FileExistsError):
            restore()
        assert not peers.mock_calls
    else:
        validate.side_effect = None
        assert restore() == TARGETS[1]


def test_lost_publish_reply_blocks_another_attempt(
    capture, restore, peers: Mock, tmp_path: Path
) -> None:
    capture()
    peers.reset_mock()
    peers.publish.side_effect = TimeoutError("publish reply lost")
    with pytest.raises(TimeoutError, match="publish reply lost"):
        restore()
    attempt = peers.restore.call_args.args[0]
    peers.publish.assert_called_once_with(attempt, target=TARGETS[0])
    peers.discard.assert_not_called()
    assert (
        json.loads((tmp_path / TEMPLATE / "active-restore.json").read_bytes())["attempt"] == attempt
    )
    peers.reset_mock()
    with pytest.raises(FileExistsError):
        restore()
    assert not peers.mock_calls


@pytest.mark.parametrize("reply", ["succeeded", "stale-attempt", "pending"])
def test_host_subprocess_receipt(
    coordinator: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str
) -> None:
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", KEY)
    attempt = "f" * 32
    deadline = time.monotonic() + BUDGET
    envelope = {"version": 1, "action": "bind_source", "template": TEMPLATE, "attempt": attempt}
    request = {**envelope, "deadline_monotonic": deadline, "input": {"source": SOURCE}}
    adapter = tmp_path / "host adapter"
    adapter.write_text(
        f"#!{sys.executable} -I\n"
        + dedent(f"""\
            import json
            import os
            import sys

            assert "TRTLLM_RUNTIME_CONTROL_API_KEY" not in os.environ
            request = json.load(sys.stdin)
            assert request == {request!r}
            receipt = {{k: request[k] for k in ("version", "action", "template", "attempt")}}
            if {reply!r} == "stale-attempt":
                receipt["attempt"] = "0" * 32
            receipt["status"] = {"pending" if reply == "pending" else "succeeded"!r}
            receipt["result"] = {{"source": request["input"]["source"], "control_port": 1234}}
            json.dump(receipt, sys.stdout)
            """),
        encoding="utf-8",
    )
    adapter.chmod(0o700)
    host = coordinator.Host(adapter)
    if reply == "succeeded":
        assert host.call("bind_source", attempt, TEMPLATE, deadline, source=SOURCE) == {
            "source": SOURCE,
            "control_port": 1234,
        }
    else:
        with pytest.raises(RuntimeError, match="receipt"):
            host.call("bind_source", attempt, TEMPLATE, deadline, source=SOURCE)
