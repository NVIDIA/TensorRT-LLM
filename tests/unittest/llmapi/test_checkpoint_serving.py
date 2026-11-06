# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route authentication, admission and CLI wiring for checkpoint operations."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import create_autospec, patch

import click
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.executor.checkpoint import (
    CheckpointBackend,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointProfile,
    CheckpointStep,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
)
from tensorrt_llm.executor.rpc.rpc_common import RPCCancelled
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.serve.checkpoint import (
    CheckpointIngress,
    ExecutorCheckpointBackend,
    register_checkpoint_routes,
)
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache,
    build_runtime_control_auth_headers,
    validate_runtime_control_request,
)

pytestmark = pytest.mark.cpu_only


def test_authenticated_prepare_keeps_admin_reachable_and_generation_closed() -> None:
    """Exercise real route/auth/gate behavior with only native work mocked."""
    app = FastAPI()
    backend = create_autospec(CheckpointBackend, instance=True, spec_set=True)
    backend.describe.return_value = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]
    )
    lifecycle = CheckpointLifecycle(backend)
    cache = RuntimeControlReplayCache()

    async def authenticate(request: Request) -> None:
        try:
            validate_runtime_control_request(
                "test-key",
                request.method,
                request.scope["route"].path,
                await request.body(),
                request.headers,
                cache,
            )
        except ValueError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error

    @app.post("/v1/completions")
    async def generate() -> dict:
        return {"generated": True}

    register_checkpoint_routes(app, lifecycle, authenticate, CheckpointIngress())
    with TestClient(app) as client:
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
        body = json.dumps(payload).encode()
        path = "/checkpoint/prepare_capture"
        headers = build_runtime_control_auth_headers("test-key", "POST", path, body)
        response = client.post(
            path, content=body, headers={**headers, "content-type": "application/json"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert client.post("/v1/completions").status_code == 503
        assert client.get("/health").status_code == 503
        assert client.post(path, content=body, headers=headers).status_code == 401
        path = "/checkpoint/status"
        assert (
            client.get(
                path, headers=build_runtime_control_auth_headers("test-key", "GET", path, b"")
            ).status_code
            == 200
        )


@pytest.mark.parametrize("extra", [[], ["--grpc"]])
def test_checkpoint_cli_requires_authenticated_http(monkeypatch, extra) -> None:
    """Reject missing credentials and unsupported transports before model loading."""
    monkeypatch.delenv("TRTLLM_RUNTIME_CONTROL_API_KEY", raising=False)
    with pytest.raises(click.UsageError):
        serve_main(args=["dummy/model", "--enable_checkpoint_mode", *extra], standalone_mode=False)


def test_checkpoint_cli_does_not_require_sleep_config(monkeypatch) -> None:
    """Resident capture enables native lifecycle without enabling memory release."""
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "test-key")
    with (
        patch("tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False),
        patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        serve_main(args=["dummy/model", "--enable_checkpoint_mode"], standalone_mode=False)
    args = launch.call_args.args[2]
    assert args.get("sleep_config") is None
    assert args["num_serve_frontends"] == 1
    assert args["env_overrides"]["TRTLLM_ENABLE_CHECKPOINT_LIFECYCLE"] == "1"
    assert launch.call_args.kwargs["enable_checkpoint_mode"]
    assert not launch.call_args.kwargs["enable_runtime_control_endpoints"]


@pytest.mark.parametrize("drain", [True, False], ids=["complete", "timeout"])
def test_prepare_drains_http_before_closing_native_queue(drain: bool) -> None:
    """Accepted HTTP preprocessing must finish before native admission closes."""
    backend = ExecutorCheckpointBackend(None)
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
        patch.object(backend, "_rpc", autospec=True, return_value=[]) as rpc,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        task = pool.submit(backend.step, CheckpointStep.PREPARE, context, 1 if drain else 0.05)
        assert entered.wait(2)
        rpc.assert_not_called()
        if drain:
            with backend.ingress.condition:
                backend.ingress.active = 0
                backend.ingress.condition.notify_all()
            assert task.result(timeout=2) == []
            rpc.assert_called_once()
            assert 0 < rpc.call_args.args[2] <= 1
        else:
            with pytest.raises(TimeoutError, match="have not drained"):
                task.result(timeout=2)
            rpc.assert_not_called()


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
