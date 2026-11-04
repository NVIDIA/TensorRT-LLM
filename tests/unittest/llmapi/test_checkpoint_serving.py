# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route authentication, admission and CLI wiring for checkpoint operations."""

import json
import os
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import click
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from test_common.serve.app_factory import AggConfig, isolated_serve_env

from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.executor.checkpoint import (
    CheckpointBackend,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointPrepareNotDispatchedError,
    CheckpointProbeLimits,
    CheckpointProfile,
    CheckpointStage,
    CheckpointStep,
    GenerationValidation,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
)
from tensorrt_llm.executor.rpc.rpc_common import RPCCancelled
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.serve.checkpoint import CheckpointIngress, ExecutorCheckpointBackend
from tensorrt_llm.serve.runtime_control_auth import build_runtime_control_auth_headers

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def checkpoint_server(tmp_path) -> Iterator[tuple]:
    """Use production routes, liveness and lifecycle with a mocked native boundary.

    Yields:
        Test client, lifecycle, native backend and liveness owner.
    """
    backend = create_autospec(CheckpointBackend, instance=True, spec_set=True)
    backend.describe.return_value = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]
    )
    ingress = CheckpointIngress()
    # Only the native bridge is mocked; constructor, auth and middleware are real.
    boundary = SimpleNamespace(
        ingress=ingress,
        describe=backend.describe,
        step=backend.step,
        generate=backend.generate,
        validate_probe=backend.validate_probe,
        fail_closed=backend.fail_closed,
    )
    with (
        isolated_serve_env(tmp_path) as env,
        patch("tensorrt_llm.serve.checkpoint.ExecutorCheckpointBackend", return_value=boundary),
    ):
        server = env.build_openai_server(AggConfig(checkpoint=True))
        server.checkpoint_ingress = ingress
        yield env.client(server.app), server.checkpoint_lifecycle, backend, server


def post_checkpoint(client: TestClient, path: str, payload: dict) -> Response:
    """Sign a control mutation with a new transport nonce.

    Args:
        client: In-process HTTP client.
        path: Checkpoint control route.
        payload: JSON-compatible operation envelope.

    Returns:
        The actual route response.
    """
    body = json.dumps(payload).encode()
    headers = build_runtime_control_auth_headers("test-key", "POST", path, body)
    return client.post(path, content=body, headers={**headers, "content-type": "application/json"})


@pytest.mark.parametrize("failure", [None, RuntimeError, TimeoutError, OSError])
def test_authenticated_prepare_keeps_admin_reachable_and_generation_closed(
    checkpoint_server, failure
) -> None:
    """Exercise real route/auth/gate behavior with only native work mocked."""
    client, _, backend, _ = checkpoint_server
    assert client.get("/checkpoint/status").status_code == 401
    path = "/checkpoint/status"
    response = client.get(
        path, headers=build_runtime_control_auth_headers("test-key", "GET", path, b"")
    )
    state = response.json()
    payload = {
        "context": {
            "protocol_version": 1,
            "operation_id": "prepare",
            "session_id": state["source_session_id"],
            "session_kind": "source",
            "template_id": "template",
            "profile_digest": "a" * 64,
            "cohort_digest": "b" * 64,
            "expected_state_version": 0,
            "time_budget_seconds": 30,
        }
    }
    backend.step.return_value = [
        RankCheckpointEvidence(
            rank=0,
            session_id=state["source_session_id"],
            profile_digest="a" * 64,
            cohort_digest="b" * 64,
            state_digest="d" * 64,
            checks=["admission_closed", "scheduler_drained", "no_external_exposure"],
        )
    ]
    backend.step.side_effect = None
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    stale = {"context": {**payload["context"], "expected_state_version": 99}}
    assert post_checkpoint(client, "/checkpoint/prepare_capture", stale).status_code == 409
    backend.step.assert_not_called()
    if failure is not None:
        backend.step.side_effect = failure("native operation failed")
    body = json.dumps(payload).encode()
    path = "/checkpoint/prepare_capture"
    headers = build_runtime_control_auth_headers("test-key", "POST", path, body)
    response = client.post(
        path, content=body, headers={**headers, "content-type": "application/json"}
    )
    assert response.status_code == (200 if failure is None else 500)
    assert response.json()["status"] == ("completed" if failure is None else "failed")
    retry = post_checkpoint(client, path, payload)
    assert retry.status_code == response.status_code
    assert retry.json() == response.json()
    assert backend.step.call_count == 1
    conflicting = {"context": {**payload["context"], "operation_id": "other"}}
    assert post_checkpoint(client, path, conflicting).status_code == (
        409 if failure is None else 500
    )
    assert backend.step.call_count == 1
    assert client.post("/v1/completions").status_code == 503
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post(path, content=body, headers=headers).status_code == 401
    path = "/checkpoint/status"
    assert (
        client.get(
            path, headers=build_runtime_control_auth_headers("test-key", "GET", path, b"")
        ).status_code
        == 200
    )


@pytest.mark.parametrize("stage", list(CheckpointStage))
@pytest.mark.parametrize("healthy", [False, True])
def test_liveness_and_readiness_are_independent(checkpoint_server, stage, healthy) -> None:
    """Admission never hides the existing health handler or a dead engine."""
    client, lifecycle, _, server = checkpoint_server
    # Isolate probe routing from transition execution, which has its own coverage.
    lifecycle._stage = stage
    lifecycle._readiness.admission_open = stage in (
        CheckpointStage.RUNNING,
        CheckpointStage.SERVING,
    )
    server.generator._check_health = lambda: healthy
    assert client.get("/health").status_code == (200 if healthy else 503)
    ready = healthy and lifecycle._readiness.admission_open
    assert client.get("/ready").status_code == (200 if ready else 503)


@pytest.mark.parametrize(
    "error,code",
    [
        (ValueError("stale"), 409),
        (RuntimeError("unavailable"), 500),
        (TimeoutError("expired"), 504),
    ],
)
@pytest.mark.parametrize(
    "route", ["prepare_capture", "validate_runtime", "activate", "abort_operation"]
)
def test_control_error_status_codes(checkpoint_server, error, code, route) -> None:
    """Each control route distinguishes rejected input from operational failure."""
    client, lifecycle, _, _ = checkpoint_server
    ctx = CheckpointContext(
        protocol_version=1,
        operation_id="test",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=30,
    ).model_dump(mode="json")
    payload = {"context": ctx}
    if route == "validate_runtime":
        payload.update(
            source_session_id="source",
            validation={"prompt_token_ids": [1], "expected_token_ids": [2, 3]},
        )
    elif route == "activate":
        payload.update(
            readiness_id="ready",
            host_restore={
                "result_id": "host",
                "status": "succeeded",
                "session_id": "source",
                "template_id": "template",
                "profile_digest": "a" * 64,
                "cohort_digest": "b" * 64,
            },
        )
    elif route == "abort_operation":
        payload["reason"] = "cancel"
    with patch.object(lifecycle, route, side_effect=error):
        assert post_checkpoint(client, f"/checkpoint/{route}", payload).status_code == code


@pytest.mark.parametrize(
    "extra,key,message",
    [([], None, "TRTLLM_RUNTIME_CONTROL_API_KEY"), (["--grpc"], "test-key", "PyTorch HTTP server")],
)
def test_checkpoint_cli_requires_authenticated_http(monkeypatch, extra, key, message) -> None:
    """Reject missing credentials and unsupported transports before model loading."""
    monkeypatch.delenv("TRTLLM_RUNTIME_CONTROL_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", key)
    with pytest.raises(click.UsageError, match=message):
        serve_main(args=["dummy/model", "--enable_checkpoint_mode", *extra], standalone_mode=False)


@pytest.mark.parametrize(
    "extra",
    [
        ["--server_role", "CONTEXT"],
        ["--disagg_cluster_uri", "etcd://localhost:2379"],
        ["--metadata_server_config_file", "unused.yaml"],
    ],
)
def test_checkpoint_cli_rejects_disaggregation_before_loading(monkeypatch, extra) -> None:
    """Reject unsupported deployment owners before config parsing or model startup."""
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "test-key")
    with patch("tensorrt_llm.commands.serve.launch_server") as launch:
        with pytest.raises(click.UsageError, match="Checkpoint mode does not support"):
            serve_main(
                args=["dummy/model", "--enable_checkpoint_mode", *extra], standalone_mode=False
            )
        launch.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_cli_does_not_require_sleep_config(monkeypatch, enabled) -> None:
    """Resident capture enables native lifecycle without enabling memory release."""
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "test-key")
    profile_env = (
        "TLLM_NCCL_SYMMETRIC_ZERO_COPY",
        "TRTLLM_PROXY_BATCH_REQUESTS",
        "TRTLLM_GPU_KEEPALIVE",
    )
    for key in profile_env:
        monkeypatch.setenv(key, "1")
    with (
        patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        serve_main(
            args=["dummy/model", *(["--enable_checkpoint_mode"] if enabled else [])],
            standalone_mode=False,
        )
    args = launch.call_args.args[2]
    assert args.get("sleep_config") is None
    if enabled:
        assert args["num_serve_frontends"] == 1
    overrides = args.get("env_overrides") or {}
    assert overrides.get("TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE") == ("1" if enabled else None)
    for key in profile_env:
        assert overrides.get(key) == ("0" if enabled else None)
        assert os.environ[key] == ("0" if enabled else "1")
    assert launch.call_args.kwargs["enable_checkpoint_mode"] == enabled
    assert not launch.call_args.kwargs["enable_runtime_control_endpoints"]


def test_admission_does_not_copy_operation_evidence(checkpoint_server) -> None:
    """Ordinary traffic and readiness probes do not build the full status response."""
    client, lifecycle, _, _ = checkpoint_server
    with patch.object(lifecycle, "get_lifecycle_status", side_effect=AssertionError("status copy")):
        assert (
            client.post("/v1/completions", json={"model": "fake-model", "prompt": "hi"}).status_code
            == 200
        )
        assert client.get("/ready").status_code == 200
        lifecycle._cancelled.set()
        assert client.post("/v1/completions").status_code == 503
        assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("drain", [True, False], ids=["complete", "timeout"])
def test_prepare_drains_http_before_closing_native_queue(drain: bool) -> None:
    """Accepted HTTP preprocessing must finish before native admission closes."""
    llm = create_autospec(LLM, instance=True, spec_set=True)
    future = Future()
    future.set_result([])
    llm._collective_rpc.return_value = (future,)
    backend = ExecutorCheckpointBackend(llm)
    backend.ingress.active = 1
    context = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    entered = threading.Event()
    wait_drained = backend.ingress.wait_drained

    def observe_wait(timeout: float) -> None:
        """Observe the real condition wait without replacing its drain behavior."""
        entered.set()
        wait_drained(timeout)

    with (
        patch.object(backend.ingress, "wait_drained", side_effect=observe_wait),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        task = pool.submit(backend.step, CheckpointStep.PREPARE, context, 1 if drain else 0.05)
        assert entered.wait(2)
        llm._collective_rpc.assert_not_called()
        if drain:
            with backend.ingress.condition:
                backend.ingress.active = 0
                backend.ingress.condition.notify_all()
            assert task.result(timeout=2) == []
            llm._collective_rpc.assert_called_once()
            assert 0 < llm._collective_rpc.call_args.kwargs["args"][2] <= 1
        else:
            with pytest.raises(CheckpointPrepareNotDispatchedError, match="have not drained"):
                task.result(timeout=2)
            llm._collective_rpc.assert_not_called()


@pytest.mark.parametrize("dispatched", [False, True], ids=["http-drain", "native-rpc"])
def test_prepare_timeout_keeps_admission_closed_without_inferred_native_failure(
    checkpoint_server, dispatched: bool
) -> None:
    """Only proven pre-dispatch failure skips revocation; neither timeout permits serving."""
    client, lifecycle, boundary, server = checkpoint_server
    llm = create_autospec(LLM, instance=True, spec_set=True)
    backend = ExecutorCheckpointBackend(llm)
    backend.ingress = server.checkpoint_ingress
    boundary.step.side_effect = backend.step
    boundary.fail_closed.side_effect = backend.fail_closed
    future = Future()
    llm._collective_rpc.return_value = (future,)
    backend.ingress.active = 0 if dispatched else 1
    ctx = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id=lifecycle.get_lifecycle_status().source_session_id,
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=0.5,
    )
    payload = PrepareCaptureRequest(context=ctx).model_dump(mode="json")
    response = post_checkpoint(client, "/checkpoint/prepare_capture", payload)
    assert response.status_code == 500
    result = response.json()
    assert result["status"] == "failed"
    assert result["native_prepare_not_dispatched"] is not dispatched
    assert result["cleanup_status"] == (
        "host_confirmation_required" if dispatched else "not_requested"
    )
    if dispatched:
        assert [call.args[0] for call in llm._collective_rpc.call_args_list] == [
            "checkpoint_step",
            "checkpoint_fail_closed",
        ]
        future.set_result([])
    else:
        llm._collective_rpc.assert_not_called()
        with backend.ingress.condition:
            backend.ingress.active = 0
            backend.ingress.condition.notify_all()
    assert post_checkpoint(client, "/checkpoint/prepare_capture", payload).json() == result
    next_attempt = {"context": {**payload["context"], "operation_id": "new"}}
    assert post_checkpoint(client, "/checkpoint/prepare_capture", next_attempt).status_code == 500
    boundary.step.assert_called_once()
    assert boundary.fail_closed.call_count == int(dispatched)
    assert llm._collective_rpc.call_count == (2 if dispatched else 0)
    status = lifecycle.get_lifecycle_status()
    assert not status.readiness.capture_safe
    assert not status.readiness.admission_open
    assert client.post("/v1/completions").status_code == 503
    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 200


def test_closed_rpc_does_not_mask_terminal_operation_failure() -> None:
    """A dead worker cannot leave the operation in progress through cleanup errors."""
    llm = create_autospec(LLM, instance=True, spec_set=True)
    backend = ExecutorCheckpointBackend(llm)
    with patch.object(
        backend,
        "describe",
        return_value=CheckpointProfile(profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]),
    ):
        lifecycle = CheckpointLifecycle(backend)
    llm._collective_rpc.side_effect = RPCCancelled("RPC client closed")
    context = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id=lifecycle.get_lifecycle_status().source_session_id,
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    request = PrepareCaptureRequest(context=context)
    result = lifecycle.prepare_capture(request)
    assert result.status == "failed"
    assert result.cleanup_status == "host_confirmation_required"
    assert not lifecycle.get_lifecycle_status().readiness.admission_open
    assert lifecycle.prepare_capture(request) == result
    assert llm._collective_rpc.call_count == 2


@pytest.mark.parametrize("active", [0, 1])
def test_abort_prevents_late_prepare_dispatch(active) -> None:
    """Abort wakes an HTTP drain and excludes submission even after drain finishes."""
    llm = create_autospec(LLM, instance=True, spec_set=True)
    done = Future()
    done.set_result(None)
    llm._collective_rpc.return_value = (done,)
    backend = ExecutorCheckpointBackend(llm)
    backend.ingress.active = active
    context = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=5,
    )
    entered = threading.Event()
    wait = backend.ingress.wait_drained

    def drain(timeout):
        entered.set()
        if active:
            wait(timeout)
        else:
            wait(timeout)
            backend.fail_closed()

    with (
        patch.object(backend.ingress, "wait_drained", side_effect=drain),
        ThreadPoolExecutor(1) as pool,
    ):
        task = pool.submit(backend.step, CheckpointStep.PREPARE, context, 5)
        assert entered.wait(1)
        if active:
            backend.fail_closed()
        with pytest.raises(CheckpointPrepareNotDispatchedError, match="stopped"):
            task.result(timeout=1)
    assert [call.args[0] for call in llm._collective_rpc.call_args_list] == [
        "checkpoint_fail_closed"
    ]


@pytest.mark.parametrize("path", ["/start_profile", "/stop_profile", "/v1/messages/batches"])
def test_unqualified_background_work_is_rejected(checkpoint_server, path) -> None:
    client, _, backend, server = checkpoint_server
    response = client.post(path, json={})
    assert response.status_code == 400
    backend.step.assert_not_called()
    assert server.checkpoint_ingress.active == 0


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/messages"])
def test_closed_admission_preserves_error_envelope(checkpoint_server, path) -> None:
    client, lifecycle, _, _ = checkpoint_server
    lifecycle._cancelled.set()
    response = client.post(path, json={})
    assert response.status_code == 503
    payload = response.json()
    if path == "/v1/messages":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "overloaded_error"
    else:
        assert payload["object"] == "error" and payload["code"] == 503
    assert client.get("/version").status_code == 200
    assert client.get("/v1/models").status_code == 200


@pytest.mark.parametrize(
    "prompt,expected,message",
    [
        ([100], [1, 2], "vocabulary"),
        ([1], [2, 100], "vocabulary"),
        ([1] * 9, [2, 3], "max_seq_len"),
        ([1] * 11, [2, 3], "max_input_len"),
        ([1] * 5, [2, 3], "max_num_tokens"),
    ],
)
def test_probe_rejected_before_native_work(prompt, expected, message) -> None:
    backend = ExecutorCheckpointBackend(None)
    backend._profile = CheckpointProfile(
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        ranks=[0],
        probe_limits=CheckpointProbeLimits(
            vocab_size=100,
            max_seq_len=10,
            max_input_len=10,
            max_num_tokens=4,
            chunked_prefill=False,
        ),
    )
    with pytest.raises(ValueError, match=message):
        backend.validate_probe(
            GenerationValidation(prompt_token_ids=prompt, expected_token_ids=expected)
        )


@pytest.mark.parametrize(
    "method", ["checkpoint_profile", "checkpoint_step", "checkpoint_fail_closed"]
)
def test_checkpoint_rpc_keeps_rank_zero_routing(method) -> None:
    from tensorrt_llm.executor.proxy import _check_collective_rpc_guard

    _check_collective_rpc_guard(2, None, None, method)
    with pytest.raises(NotImplementedError, match="rank-0"):
        _check_collective_rpc_guard(2, None, [1], method)
    with pytest.raises(NotImplementedError, match="rank-0"):
        _check_collective_rpc_guard(2, 1, None, method)
