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
import uuid
from contextlib import suppress
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import psutil
import pytest
import requests
from defs.common import get_ucx_tls, resolve_llm_model_path
from disagg_test_utils import terminate
from test_common.serve.context_cancellation import with_context_reservation
from test_disaggregated import (
    archive_process_logs,
    get_test_config,
    get_ucx_net_devices,
    setup_disagg_cluster,
)

from tensorrt_llm.serve.disagg_auth import (
    build_context_cancellation_headers,
    build_internal_disagg_auth_headers,
    context_cancellation_reservation_payload,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest

_PROBE_MODULE = "defs.disaggregated.context_cancellation_probe"
_PROMPT = "Explain why the sky is blue in one short sentence."
_TRANSFER_TIMEOUT_SECONDS = 120
_OBSERVATION_HANG_GUARD_SECONDS = 2 * _TRANSFER_TIMEOUT_SECONDS


class _Observations:
    """Read complete append-only records from each real rank."""

    def __init__(self, directory: Path, tp: int) -> None:
        """Retain the artifact directory and required TP participant count."""
        self.directory = directory
        self.tp = tp
        self.participants: set[tuple[str, int, int]] = set()

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
            role_name = path.name.split("-", 1)[0]
            process_records = []
            identities = {}
            # The last write can be in progress; never mistake it for a record.
            for sequence, line in enumerate(text[: text.rfind("\n") + 1].splitlines(), 1):
                record = json.loads(line)
                assert record["capture_sequence"] == sequence, (path, record)
                assert record["capture_errors"] == 0, (path, record)
                local_id, pair = record.get("local_request_id"), record["request_id"]
                if local_id is not None and pair is not None:
                    assert local_id not in identities or identities[local_id] == pair
                    identities[local_id] = pair
                record["role"] = role_name
                record["time"] = record["monotonic_ns"] / 1_000_000_000
                process_records.append(record)
            for record in process_records:
                record["pair_id"] = record["request_id"]
                if record["pair_id"] is None:
                    record["pair_id"] = identities.get(record.get("local_request_id"))
                if record["event"] == "request_response_enqueued":
                    record["finish_reasons"] = (
                        record["finish_reasons"].split("|") if record["finish_reasons"] else []
                    )
                if (
                    record["event"] == event
                    and (pair_id is None or record["pair_id"] == pair_id)
                    and (role is None or record["role"] == role)
                ):
                    records.append(record)
        return records

    def checkpoint(self) -> None:
        """Require complete, zero-loss capture on every original process.

        A missing process, failed write, suppressed diagnostic preparation error
        or sequence gap cannot turn a negative assertion into a passing test.
        This barrier does not synchronize model execution or establish quiescence.
        """
        assert len(self.participants) == 2 * self.tp + 1, self.participants
        token = uuid.uuid4().hex
        temporary = self.directory / "capture-checkpoint.next"
        temporary.write_text(token)
        temporary.replace(self.directory / "capture-checkpoint")
        end = time.monotonic() + _OBSERVATION_HANG_GUARD_SECONDS
        while time.monotonic() < end:
            records = self.select("qualification_checkpoint")
            seen = {(r["role"], r["rank"], r["pid"]) for r in records if r["checkpoint"] == token}
            if self.participants <= seen:
                return
            time.sleep(0.02)
        pytest.fail(f"Incomplete diagnostic checkpoint: missing {self.participants - seen}")

    def wait(
        self,
        event: str,
        pair_id: int | None = None,
        role: str | None = None,
        *,
        all_ranks: bool = False,
        timeout: float = _OBSERVATION_HANG_GUARD_SECONDS,
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
        records = self.wait("request_response_enqueued", pair_id, role, all_ranks=True)
        assert len(records) == self.tp and all(r["terminal"] for r in records), records
        return records

    def retired(self, pair_id: int, role: str, after: float, *, held_send: bool = False) -> None:
        """Qualify early cancellation within this transfer's original time budget.

        These scenarios make access-end evidence available before the original
        deadline. They must retire without waiting for it. This is not a bound
        on arbitrary cancellation near expiry or on still-unproven access.

        Args:
            pair_id: Exact request whose resource claims must be gone.
            role: CTX or GEN resource owner.
            after: Monotonic cancellation or gate-release time in this host.
            held_send: All senders had to leave the submission gate before settlement.
        """
        records = self.wait("request_resources_released", pair_id, role, all_ranks=True)
        assert len(records) == self.tp, records
        started = self.wait("request_resource_release_started", pair_id, role, all_ranks=True)
        assert len(started) == self.tp and all(after <= r["time"] for r in started), started
        assert all(r["native_transfer_owned"] is False for r in started), started
        assert all(
            r["was_owned"] is True and r["kv_owned"] is False and r["transfer_owned"] is False
            for r in records
        )
        assert all(after <= r["time"] and r["transfer_timed_out"] is False for r in records), (
            records
        )
        settled = self.wait(f"{role}_transfer_settled", pair_id, role, all_ranks=True)
        assert len(settled) == self.tp, settled
        assert all(
            r["retirement_owner_closed"] is True and r["retirement_fatal"] is False for r in settled
        )
        assert all(r["retirement_reason"] in ("cancellation", "peer cancellation") for r in settled)
        deadline_event = (
            "ctx_native_deadline_started" if role == "ctx" else "transfer_timeout_started"
        )
        deadline_starts = self.wait(deadline_event, pair_id, role, all_ranks=True)
        assert len(deadline_starts) == self.tp, deadline_starts
        for rank in range(self.tp):
            settled_at = next(r["time"] for r in settled if r["rank"] == rank)
            started_at = next(r["time"] for r in started if r["rank"] == rank)
            released = next(r for r in records if r["rank"] == rank)
            deadline_start = next(r for r in deadline_starts if r["rank"] == rank)
            assert deadline_start["pid"] == released["pid"]
            assert deadline_start["timeout_ms"] > 0
            assert deadline_start["timer_start_monotonic_ns"] is not None
            if role == "ctx":
                assert deadline_start["timeout_owner"] == "native_tx_session"
                deadline_ns = deadline_start["deadline_monotonic_ns"]
            else:
                assert deadline_start["timeout_owner"] == "pyexecutor"
                deadline_ns = deadline_start["timer_start_monotonic_ns"] + int(
                    deadline_start["timeout_ms"] * 1_000_000
                )
            assert deadline_start["timer_start_monotonic_ns"] < deadline_ns
            if held_send:
                assert after <= settled_at, (role, rank, after, settled_at)
            assert settled_at <= started_at <= released["time"]
            assert released["monotonic_ns"] < deadline_ns, (
                "Cancellation did not release before its original transfer deadline",
                deadline_start,
                released,
            )
        self.checkpoint()
        self.assert_no_timeout_retirement(pair_id)

    def assert_no_timeout_retirement(self, pair_id: int) -> None:
        """Reject timeout/fatal cleanup even when CANCELLED remains the outcome."""
        for event in (
            "transfer_timeout_observed",
            "ctx_native_deadline_observed",
            "retirement_fatal",
        ):
            assert not self.select(event, pair_id), (event, pair_id)


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
    if action == "reserve":
        payload = context_cancellation_reservation_payload(payload["disagg_request_id"])
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
    payload = with_context_reservation(_completion(model), pair_id=pair_id, token=token)
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


def _stop_recorded_ranks(directory: Path) -> None:
    """Stop only surviving executor processes created by this test, after launcher cleanup."""
    ranks = []
    for path in directory.glob("rank-*.identity"):
        pid, created = json.loads(path.read_text())
        with suppress(psutil.NoSuchProcess):
            rank = psutil.Process(pid)
            if rank.create_time() == created:
                ranks.append(rank)
                rank.terminate()
    _, alive = psutil.wait_procs(ranks, timeout=5)
    for rank in alive:
        with suppress(psutil.NoSuchProcess):
            rank.kill()
    _, alive = psutil.wait_procs(alive, timeout=5)
    survivors = []
    for rank in alive:
        with suppress(psutil.NoSuchProcess):
            if rank.is_running() and rank.status() != psutil.STATUS_ZOMBIE:
                survivors.append(rank.pid)
    assert not survivors, f"Test executor ranks survived cleanup: {survivors}"


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


@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(1, marks=pytest.mark.skip_less_device(2), id="tp1"),
        pytest.param(2, marks=pytest.mark.skip_less_device(4), id="tp2"),
    ],
)
def test_context_first_cancellation_serving(
    llm_venv: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tp: int,
    output_dir: str | None,
    request: pytest.FixtureRequest,
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
            "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS": "1",
            "TRTLLM_DISAGG_TRANSFER_DIAGNOSTICS_RUN_ID": str(uuid.uuid4()),
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
    request.addfinalizer(lambda: _stop_recorded_ranks(tmp_path))
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
            ready = observations.wait("worker_ready", role=role, all_ranks=True)
            assert len(ready) == tp, ready
            observations.participants.update((r["role"], r["rank"], r["pid"]) for r in ready)
            profile = observations.wait("diagnostic_profile", role=role, all_ranks=True)
            assert all(r["profile"] == "native_v1" for r in profile)
        coordinator_ready = observations.wait("qualification_capture_ready", role="coordinator")
        assert len(coordinator_ready) == 1, coordinator_ready
        observations.participants.update(
            (r["role"], r["rank"], r["pid"]) for r in coordinator_ready
        )
        observations.checkpoint()
        cluster = requests.get(f"{coordinator}/cluster_info", timeout=5).json()["current_workers"]
        ctx_info = cluster["context_servers"][0]
        gen_info = cluster["generation_servers"][0]
        ctx = f"http://{ctx_info['host']}:{ctx_info['port']}"
        gen = f"http://{gen_info['host']}:{gen_info['port']}"
        # The later absence checks require working dispatch probes on this serving path.
        healthy_started = time.monotonic_ns()
        _healthy(coordinator, model)
        dispatched = [
            r
            for r in observations.select("gen_dispatched", role="coordinator")
            if r["monotonic_ns"] >= healthy_started
        ]
        assert len(dispatched) == 1, dispatched
        healthy_pair = dispatched[0]["pair_id"]
        assert healthy_pair is not None, dispatched
        for role in ("ctx", "gen"):
            observations.wait("request_model_dispatch", healthy_pair, role, all_ranks=True)

        # Cancellation closes a real authenticated reservation before submission.
        pair = 910001
        token = _control(ctx, key, "reserve", {"disagg_request_id": pair})["token"]
        assert _control(ctx, key, "cancel", {"token": token})["status"] == "not_submitted"
        payload = with_context_reservation(_completion(model), pair_id=pair, token=token)
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
        assert not observations.select("request_model_dispatch", pair)
        assert not observations.select("ctx_send_ready", pair)

        # A returned CTX result retains KV, but needs no cleanup-only GEN.
        pair = 910002
        token, handoff = _context(ctx, model, key, pair)
        retained = observations.wait("ctx_send_ready", pair, "ctx", all_ranks=True)
        assert all(r["source_kv_request_owned"] for r in retained)
        assert not observations.select("request_resources_released", pair)
        before = observations.terminal(pair, "ctx")
        assert all(not r["has_error"] and "CANCELLED" not in r["finish_reasons"] for r in before)
        cancelled_at = time.monotonic()
        for _ in range(2):
            assert _control(ctx, key, "cancel", {"token": token})["status"] == "queued"
        observations.wait("logical_cancellation_accepted", pair, "ctx", all_ranks=True)
        observations.retired(pair, "ctx", cancelled_at)
        assert observations.select("request_response_enqueued", pair) == before
        assert not observations.select("gen_request_data_sent", pair, "gen")
        _healthy(coordinator, model)

        # Disconnect the real coordinator caller after CTX replied, before GEN.
        gate = tmp_path / "hold_handoff"
        gate.touch()
        connection = _start_request(coordinator, _completion(model))
        try:
            held = observations.wait("handoff_held", role="coordinator")[-1]
            pair = held["pair_id"]
            observations.wait("ctx_send_ready", pair, "ctx", all_ranks=True)
            cancelled_at = time.monotonic()
            connection.close()
            observations.wait("handoff_cancelled", pair, "coordinator")
            observations.wait("logical_cancellation_accepted", pair, "ctx", all_ranks=True)
            observations.retired(pair, "ctx", cancelled_at)
            assert not observations.select("gen_dispatched", pair)
            assert not observations.select("gen_request_data_sent", pair, "gen")
        finally:
            connection.close()
            gate.unlink(missing_ok=True)
        _healthy(coordinator, model)

        # An existing GEN must return logically while its exact native send is held.
        pair = 910003
        token, handoff = _context(ctx, model, key, pair)
        ctx_responses = observations.terminal(pair, "ctx")
        assert all(
            not r["has_error"] and "CANCELLED" not in r["finish_reasons"] for r in ctx_responses
        )
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
            observations.wait("gen_request_data_sent", pair, "gen", all_ranks=True)
            connection.close()
            responses = observations.terminal(pair, "gen")
            accepted = observations.wait(
                "logical_cancellation_accepted", pair, "gen", all_ranks=True
            )
            assert all(r["native_transfer_owned"] is True for r in accepted), accepted
            assert all(
                not r["has_error"] and r["finish_reasons"] == ["CANCELLED"] for r in responses
            )
            assert not observations.select("request_resources_released", pair)
            assert not observations.select("request_resource_release_started", pair)
            # The held sender thread must not stall unrelated GEN computation.
            _healthy(gen, model)
            observations.checkpoint()
            assert not observations.select("request_resources_released", pair)
            assert not observations.select("request_resource_release_started", pair)
            assert not observations.select("request_model_dispatch", pair, "gen")
            gate.unlink()
            # CTX settlement seals the set of submissions. One gate exit per rank
            # alone cannot account for multiple concurrent callbacks on the same rank.
            observations.wait("ctx_transfer_settled", pair, "ctx", all_ranks=True)
            gate_released = observations.select("gate_released", pair, "ctx")
            held_submissions = observations.select("native_submitted", pair, "ctx")
            hold_identity = itemgetter("rank", "peer_rank", "piece_id", "hold_id")
            assert sorted(map(hold_identity, gate_released)) == sorted(
                map(hold_identity, held_submissions)
            ), (held_submissions, gate_released)
            native = observations.select("ctx_backend_submitted", pair, "ctx")
            assert set(map(itemgetter("rank", "peer_rank", "piece_id"), held_submissions)) <= set(
                map(itemgetter("rank", "peer_rank", "slice_id"), native)
            ), (held_submissions, native)
            # Every held sender must observe the open gate before group
            # retirement. A pre-unlink parent timestamp is not that evidence.
            released_at = max(r["time"] for r in gate_released)
            observations.retired(pair, "gen", released_at, held_send=True)
            observations.retired(pair, "ctx", released_at, held_send=True)
            assert observations.select("request_response_enqueued", pair, "gen") == responses
        finally:
            connection.close()
            gate.unlink(missing_ok=True)
        _healthy(coordinator, model)
        observations.checkpoint()
        assert not observations.select("gen_dispatched", held["pair_id"])
        assert not observations.select("gen_request_data_sent", held["pair_id"], "gen")
        assert not observations.select("request_model_dispatch", 910001)
        assert not observations.select("ctx_send_ready", 910001)
        assert not observations.select("request_model_dispatch", 910003, "gen")
        assert observations.select("request_response_enqueued", 910002, "ctx") == before
        assert observations.select("request_response_enqueued", 910003, "gen") == responses
        assert observations.select("request_response_enqueued", 910003, "ctx") == ctx_responses
        for retired_pair, role in (
            (910002, "ctx"),
            (held["pair_id"], "ctx"),
            (910003, "ctx"),
            (910003, "gen"),
        ):
            releases = observations.select("request_resources_released", retired_pair, role)
            assert len(releases) == tp and {r["rank"] for r in releases} == set(range(tp)), releases
            starts = observations.select("request_resource_release_started", retired_pair, role)
            assert len(starts) == tp and {r["rank"] for r in starts} == set(range(tp)), starts
            observations.assert_no_timeout_retirement(retired_pair)
        assert not observations.select("gate_expired")
        # Fatal records can precede the local-to-pair ID mapping.
        assert not observations.select("retirement_fatal")
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
