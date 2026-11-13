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
"""Real serving cancellation checks for opt-in BF16 context-first ownership.

These tests exercise HTTP, real TP executors, KVCMv2 and NIXL. Bounded test
gates delay handoff or observation after genuine submission; they do not
establish outstanding-DMA fencing, executor replacement or default-on rollout.
"""

import http.client
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pytest
import requests
from defs.common import get_ucx_tls, resolve_llm_model_path
from disagg_test_utils import terminate
from test_disaggregated import (
    archive_process_logs,
    get_test_config,
    get_ucx_net_devices,
    setup_disagg_cluster,
)

from tensorrt_llm.serve.disagg_auth import (
    build_context_cancellation_headers,
    build_internal_disagg_auth_headers,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest

_PROBE_MODULE = "defs.disaggregated.context_cancellation_probe"
_PROMPT = "Explain why the sky is blue in one short sentence."
_TRANSFER_TIMEOUT_SECONDS = 120
_PROMPT_RETIREMENT_SECONDS = 15


class _Observations:
    """Read complete append-only records from each real rank."""

    def __init__(self, directory: Path, tp: int) -> None:
        """Retain the artifact directory and required TP participant count."""
        self.directory = directory
        self.tp = tp

    def select(
        self, event: str, pair_id: int | None = None, role: str | None = None
    ) -> list[dict[str, Any]]:
        """Read matching complete observations without losing duplicates.

        Args:
            event: Production boundary to inspect.
            pair_id: Exact request identity, or None to include every request.
            role: Endpoint role, or None to include both endpoints.

        Returns:
            All matching records, including repeated emissions by one rank.
        """
        records = []
        for path in sorted(self.directory.glob("*.jsonl")):
            text = path.read_text()
            # The last write can be in progress; never mistake it for a record.
            for line in text[: text.rfind("\n") + 1].splitlines():
                record = json.loads(line)
                if (
                    record["event"] == event
                    and (pair_id is None or record["pair_id"] == pair_id)
                    and (role is None or record["role"] == role)
                ):
                    records.append(record)
        return records

    def wait(
        self,
        event: str,
        pair_id: int | None = None,
        role: str | None = None,
        *,
        all_ranks: bool = False,
        timeout: float = _PROMPT_RETIREMENT_SECONDS,
    ) -> list[dict[str, Any]]:
        """Wait within a real-time bound for the required owner observations.

        Args:
            event: Production boundary to inspect.
            pair_id: Exact request identity, or None to include every request.
            role: Endpoint role, or None to include both endpoints.
            all_ranks: Require evidence from every TP rank.
            timeout: Maximum observation wait in seconds.

        Returns:
            Matching records, without treating missing or skipped work as pass.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            records = self.select(event, pair_id, role)
            if records and (not all_ranks or {r["rank"] for r in records} == set(range(self.tp))):
                return records
            assert not self.select("gate_expired"), (
                "Test gate expired before the scenario completed"
            )
            time.sleep(0.02)
        pytest.fail(f"Missing {event}: pair={pair_id}, role={role}; observed={records}")

    def terminal(self, pair_id: int, role: str) -> list[dict[str, Any]]:
        """Require one terminal result from each real TP participant.

        Args:
            pair_id: Exact request whose result must be stable.
            role: CTX or GEN owner of the result.

        Returns:
            The original per-rank results, for later immutability checks.
        """
        records = self.wait("response", pair_id, role, all_ranks=True)
        assert len(records) == self.tp and all(r["terminal"] for r in records), records
        return records

    def retired(self, pair_id: int, role: str, after: float) -> None:
        """Require prompt, exactly-once release by each actual resource owner.

        Args:
            pair_id: Exact request whose resource claims must be gone.
            role: CTX or GEN resource owner.
            after: Monotonic cancellation or gate-release time in this host.
        """
        records = self.wait("resources_released", pair_id, role, all_ranks=True)
        assert len(records) == self.tp, records
        assert all(
            r["was_owned"] and not r["kv_owned"] and not r["transfer_owned"] for r in records
        )
        assert all(0 <= r["time"] - after < _PROMPT_RETIREMENT_SECONDS for r in records)


def _post(
    url: str, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> requests.Response:
    """Send one bounded HTTP request without automatic retry."""
    return requests.post(f"{url}/{path}", json=payload, headers=headers, timeout=90)


def _control(url: str, key: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Use the real authenticated cancellation API, preserving receipt semantics.

    Args:
        url: Worker HTTP address.
        key: This test cluster's internal authentication key.
        action: Reservation or cancellation operation.
        payload: Original control request payload.

    Returns:
        The response body; a queued receipt is not physical-retirement evidence.
    """
    response = _post(
        url,
        f"internal/context-cancellation/{action}",
        payload,
        build_context_cancellation_headers(key, action, payload),
    )
    assert response.status_code == (200 if action == "reserve" else 202), response.text
    return response.json()


def _completion(model: str, **extra: Any) -> dict[str, Any]:
    """Build a short deterministic completion that requires a GEN handoff."""
    return dict(
        model=model,
        prompt=_PROMPT,
        max_tokens=8,
        min_tokens=8,
        ignore_eos=True,
        temperature=0,
        **extra,
    )


def _context(url: str, model: str, key: str, pair_id: int) -> tuple[str, dict[str, Any]]:
    """Complete real prefill and retain its authenticated cancellation identity.

    Args:
        url: CTX worker HTTP address.
        model: Served model identifier.
        key: This test cluster's internal authentication key.
        pair_id: Unreused pairing identity for this scenario.

    Returns:
        The reservation token and unmodified CTX handoff response.
    """
    token = _control(url, key, "reserve", {"disagg_request_id": pair_id})["token"]
    payload = _completion(
        model,
        disaggregated_params={
            "request_type": "context_only",
            "disagg_request_id": pair_id,
            "schedule_style": "context_first",
            "context_cancellation_token": token,
        },
    )
    headers = build_context_cancellation_headers(
        key,
        "submit",
        {
            "token": token,
            "model": model,
            "disagg_request_id": pair_id,
        },
    )
    response = _post(url, "v1/completions", payload, headers)
    response.raise_for_status()
    handoff = response.json()
    assert handoff["prompt_token_ids"]
    assert handoff["choices"][0]["disaggregated_params"]["disagg_request_id"] == pair_id
    return token, handoff


def _start_request(
    url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> http.client.HTTPConnection:
    """Send a real request whose caller can disconnect before its HTTP result.

    Args:
        url: Coordinator or worker address.
        payload: Real completion payload.
        headers: Internal authentication headers when forwarding a handoff.

    Returns:
        A caller-owned connection, closed explicitly at the tested boundary.
    """
    parsed = urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=15)
    connection.request(
        "POST",
        "/v1/completions",
        json.dumps(payload),
        {"Content-Type": "application/json", **(headers or {})},
    )
    return connection


def _healthy(url: str, model: str) -> None:
    """Require an unrelated completion to progress through real execution."""
    response = _post(url, "v1/completions", _completion(model))
    response.raise_for_status()
    assert response.json()["usage"]["completion_tokens"] == 8, response.text


def _launcher(
    popen: Callable[..., subprocess.Popen], python: str
) -> Callable[..., subprocess.Popen]:
    """Change only serving bootstrap; retain the shared launch/cleanup helpers.

    Args:
        popen: Original subprocess launcher.
        python: Interpreter of the test's actual TRT-LLM runtime.

    Returns:
        A launcher that delegates the official CLI through test instrumentation.
    """

    def launch(command: list[str], *args: Any, **kwargs: Any) -> subprocess.Popen:
        """Preserve non-serving commands and all process-launch arguments."""
        if command[0] == "trtllm-serve":
            command = [python, "-c", f"from {_PROBE_MODULE} import main; main()", *command[1:]]
        return popen(command, *args, **kwargs)

    return launch


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("tp", [1, 2], ids=["tp1", "tp2"])
def test_context_first_cancellation_serving(
    llm_venv: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tp: int,
    output_dir: str | None,
) -> None:
    """Observe request cancellation and actual owner retirement on both endpoints."""
    model = resolve_llm_model_path("Qwen3/Qwen3-0.6B")
    key = secrets.token_hex(32)
    observations = _Observations(tmp_path, tp)
    env = llm_venv._new_env.copy()
    env.update(
        {
            "TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT": "1",
            "TRTLLM_DISAGG_NO_RETRY": "1",
            "TRTLLM_USE_PY_NIXL_KVCACHE": "0",
            "TRTLLM_NIXL_KVCACHE_BACKEND": "UCX",
            "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "0",
            "TRTLLM_DISAGG_LAYERWISE": "0",
            "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL": "0",
            "TRTLLM_ENABLE_FP4_MLA_KV_OWNERSHIP_BRIDGE": "0",
            "TLLM_CONTEXT_CANCELLATION_PROBE_DIR": str(tmp_path),
            "TLLM_CONTEXT_CANCELLATION_PROBE_ROLE": "coordinator",
            "UCX_TLS": get_ucx_tls(),
            "UCX_NET_DEVICES": get_ucx_net_devices(),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH")])
            ),
        }
    )
    worker = {
        "dtype": "bfloat16",
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": False,
        "disable_overlap_scheduler": False,
        "kv_cache_config": {
            "use_kv_cache_manager_v2": True,
            "free_gpu_memory_fraction": 0.2,
            "enable_partial_reuse": False,
        },
        "cache_transceiver_config": {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "kv_transfer_timeout_ms": _TRANSFER_TIMEOUT_SECONDS * 1000,
            "kv_cache_bounce_size_mb": 0,
            "agent_bounce_buffer_enable": False,
            "enable_pipelined_transfer": False,
        },
    }
    monkeypatch.setattr(subprocess, "Popen", _launcher(subprocess.Popen, llm_venv._venv_bin))
    config_file = get_test_config(
        "overlap_transceiver_runtime_python", "", str(Path(__file__).parent)
    )
    _, ctx_workers, gen_workers, server, port, work_dir = setup_disagg_cluster(
        config_file,
        model_name=model,
        env=env,
        cwd=llm_venv.get_working_directory(),
        schedule_style="context_first",
        save_log=True,
        log_output_dir=output_dir,
        ctx_env={**env, "TLLM_CONTEXT_CANCELLATION_PROBE_ROLE": "ctx"},
        gen_env={**env, "TLLM_CONTEXT_CANCELLATION_PROBE_ROLE": "gen"},
        config_overrides={
            "context_servers": worker,
            "generation_servers": worker,
            "internal_request_auth_key": key,
        },
    )
    coordinator = f"http://localhost:{port}"
    try:
        for role in ("ctx", "gen"):
            observations.wait("worker_ready", role=role, all_ranks=True)
        cluster = requests.get(f"{coordinator}/cluster_info", timeout=5).json()["current_workers"]
        ctx_info = cluster["context_servers"][0]
        gen_info = cluster["generation_servers"][0]
        ctx = f"http://{ctx_info['host']}:{ctx_info['port']}"
        gen = f"http://{gen_info['host']}:{gen_info['port']}"
        _healthy(coordinator, model)

        # Cancellation closes a real authenticated reservation before submission.
        pair = 910001
        token = _control(ctx, key, "reserve", {"disagg_request_id": pair})["token"]
        assert _control(ctx, key, "cancel", {"token": token})["status"] == "not_submitted"
        payload = _completion(
            model,
            disaggregated_params={
                "request_type": "context_only",
                "disagg_request_id": pair,
                "schedule_style": "context_first",
                "context_cancellation_token": token,
            },
        )
        headers = build_context_cancellation_headers(
            key,
            "submit",
            {
                "token": token,
                "model": model,
                "disagg_request_id": pair,
            },
        )
        rejected = _post(ctx, "v1/completions", payload, headers)
        assert rejected.status_code >= 400 and "admission is closed" in rejected.text
        assert not observations.select("model_batch", pair)
        assert not observations.select("source_retained", pair)

        # A returned CTX result retains KV, but needs no cleanup-only GEN.
        pair = 910002
        token, handoff = _context(ctx, model, key, pair)
        retained = observations.wait("source_retained", pair, "ctx", all_ranks=True)
        assert all(r["kv_owned"] for r in retained)
        assert not observations.select("resources_released", pair)
        before = observations.terminal(pair, "ctx")
        assert all(not r["error"] and "CANCELLED" not in r["finish_reasons"] for r in before)
        cancelled_at = time.monotonic()
        for _ in range(2):
            assert _control(ctx, key, "cancel", {"token": token})["status"] == "queued"
        observations.retired(pair, "ctx", cancelled_at)
        assert observations.select("response", pair) == before
        assert not observations.select("receive_published", pair, "gen")
        _healthy(coordinator, model)

        # Disconnect the real coordinator caller after CTX replied, before GEN.
        gate = tmp_path / "hold_handoff"
        gate.touch()
        connection = _start_request(coordinator, _completion(model))
        try:
            held = observations.wait("handoff_held", role="coordinator")[-1]
            pair = held["pair_id"]
            observations.wait("source_retained", pair, "ctx", all_ranks=True)
            cancelled_at = time.monotonic()
            connection.close()
            observations.wait("handoff_cancelled", pair, "coordinator")
            observations.retired(pair, "ctx", cancelled_at)
            assert not observations.select("gen_dispatched", pair)
            assert not observations.select("receive_published", pair, "gen")
        finally:
            connection.close()
            gate.unlink(missing_ok=True)
        _healthy(coordinator, model)

        # An existing GEN must return logically while its exact native send is held.
        pair = 910003
        token, handoff = _context(ctx, model, key, pair)
        gate = tmp_path / f"native_submitted-{pair}"
        gate.touch()
        params = handoff["choices"][0]["disaggregated_params"]
        params.update(request_type="generation_only", context_cancellation_token=None)
        payload = _completion(model, disaggregated_params=params)
        payload["prompt"] = handoff["prompt_token_ids"]
        connection = _start_request(
            gen, payload, build_internal_disagg_auth_headers(key, CompletionRequest(**payload))
        )
        try:
            observations.wait("native_submitted", pair, "ctx", all_ranks=True)
            observations.wait("receive_published", pair, "gen", all_ranks=True)
            connection.close()
            terminal = observations.wait("termination_attempt", pair, "gen", all_ranks=True)
            assert len(terminal) == tp, terminal
            assert all(
                r["state"] == "GENERATION_COMPLETE"
                and r["cancelled"]
                and r["kv_owned"]
                and r["retained"]
                for r in terminal
            )
            responses = observations.terminal(pair, "gen")
            assert all(not r["error"] and r["finish_reasons"] == ["CANCELLED"] for r in responses)
            assert not observations.select("resources_released", pair)
            # The held sender thread must not stall unrelated GEN computation.
            _healthy(gen, model)
            assert not observations.select("resources_released", pair)
            assert not observations.select("model_batch", pair, "gen")
            released_at = time.monotonic()
            gate.unlink()
            observations.retired(pair, "gen", released_at)
            observations.retired(pair, "ctx", released_at)
            assert observations.select("response", pair, "gen") == responses
        finally:
            connection.close()
            gate.unlink(missing_ok=True)
        _healthy(coordinator, model)
        assert not observations.select("model_batch", 910001)
        assert not observations.select("source_retained", 910001)
        assert not observations.select("model_batch", 910003, "gen")
        assert observations.select("response", 910002, "ctx") == before
        assert observations.select("response", 910003, "gen") == responses
        for retired_pair, role in (
            (910002, "ctx"),
            (held["pair_id"], "ctx"),
            (910003, "ctx"),
            (910003, "gen"),
        ):
            releases = observations.select("resources_released", retired_pair, role)
            assert len(releases) == tp and {r["rank"] for r in releases} == set(range(tp)), releases
        assert not observations.select("gate_expired")
        assert all(worker.process.poll() is None for worker in [*ctx_workers, *gen_workers])
    finally:
        for gate in tmp_path.glob("native_submitted-*"):
            gate.unlink(missing_ok=True)
        (tmp_path / "hold_handoff").unlink(missing_ok=True)
        terminate(*ctx_workers, *gen_workers, server)
        archived = archive_process_logs(work_dir, output_dir)
        if archived is not None:
            for report in tmp_path.glob("*.jsonl"):
                shutil.copy2(report, archived)
        print(f"PACT rank evidence: {tmp_path}; serving logs: {work_dir}")
