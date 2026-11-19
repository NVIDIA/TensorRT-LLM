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
"""CPU-only controller for the single-rank, resident checkpoint HTTP contract.

HMAC authenticates requests, not the listener or its responses. The caller owns
endpoint isolation, host/artifact verification, and cleanup after *any* error.
A lost activation reply does not prove that the engine kept admission closed.
"""

import http.client
import json
import re
import runpy
import secrets
import socket
import threading
import time
from collections.abc import Callable
from contextlib import closing, suppress
from pathlib import Path

# Load the checkout's stdlib-only signer without executing tensorrt_llm/__init__.py.
_AUTH = Path(__file__).resolve().parents[3] / "tensorrt_llm/serve/runtime_control_auth.py"
_sign = runpy.run_path(str(_AUTH))["build_runtime_control_auth_headers"]
_DIGEST = re.compile("[0-9a-f]{64}")
_BINDING = ("template_id", "profile_digest", "cohort_digest")
_FLAGS = {
    "capture_safe",
    "memory_ready",
    "runtime_ready",
    "validation_passed",
    "serving_ready",
    "admission_open",
}
_READY = {
    "running": {"admission_open"},
    "capture_safe": {"capture_safe"},
    "runtime_ready": {"memory_ready", "runtime_ready", "validation_passed"},
    "serving": {
        "memory_ready",
        "runtime_ready",
        "validation_passed",
        "serving_ready",
        "admission_open",
    },
}
_CHECKS = {
    "capture_safe": {"admission_closed", "scheduler_drained", "no_external_exposure"},
    "runtime_ready": {
        "validation_retired",
        "scheduler_drained",
        "warmed_graph_replayed",
        "resident_state_matches",
    },
    "serving": {"activation_committed"},
}
_MAX_BODY = 128 * 1024


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128 and bool(value.strip())


def _digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _seconds(value: float) -> bool:
    return type(value) in (int, float) and 0 < value <= 3600


def _deadline(budget: float) -> float:
    if not _seconds(budget):
        raise ValueError("Checkpoint budget must be positive and at most 3600 seconds.")
    return time.monotonic() + budget


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Checkpoint deadline expired.")
    return remaining


def _require(valid: bool, message: str) -> None:
    if not valid:
        raise RuntimeError(message)


def _object(pairs: list[tuple[str, object]]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("Duplicate JSON key.")
    return obj


def _check_result(result: dict, binding: dict, stage: str) -> dict:
    """Require the completed mutation and its stage-specific rank-zero evidence."""
    _require(
        isinstance(result, dict)
        and all(result.get(k) == v for k, v in binding.items())
        and result.get("status") == "completed"
        and type(result.get("state_version")) is int
        and result["state_version"] >= 0
        and result.get("native_prepare_not_dispatched") is False
        and result.get("cleanup_status") == "not_requested"
        and "error" in result
        and result["error"] is None
        and "readiness_id" in result,
        "Incomplete or stale checkpoint result.",
    )
    readiness_id = result["readiness_id"]
    _require(
        readiness_id is None if stage == "capture_safe" else _identifier(readiness_id),
        "Invalid engine readiness reference.",
    )
    ranks = result.get("rank_results")
    _require(
        isinstance(ranks, list) and len(ranks) == 1 and isinstance(ranks[0], dict),
        "Expected exactly one rank acknowledgement.",
    )
    rank = ranks[0]
    checks = rank.get("checks")
    _require(
        type(rank.get("rank")) is int
        and rank["rank"] == 0
        and all(rank.get(k) == binding[k] for k in ("session_id", *_BINDING[1:]))
        and _digest(rank.get("state_digest"))
        and isinstance(checks, list)
        and all(isinstance(c, str) for c in checks)
        and _CHECKS[stage].issubset(checks),
        "Invalid rank-zero checkpoint evidence.",
    )
    return rank


def validate_probe(probe: dict) -> None:
    """Check the artifact's token schema before capture or restore.

    Args:
        probe: Deterministic prompt and exact reference completion token IDs.

    Raises:
        ValueError: If fields, lengths, or token IDs violate the wire contract.
    """
    if not isinstance(probe, dict) or set(probe) != {"prompt_token_ids", "expected_token_ids"}:
        raise ValueError("Provide exactly prompt_token_ids and expected_token_ids.")
    for field, low, high in (("prompt_token_ids", 1, 4096), ("expected_token_ids", 2, 64)):
        ids = probe[field]
        if (
            not isinstance(ids, list)
            or not low <= len(ids) <= high
            or not all(type(token) is int and token >= 0 for token in ids)
        ):
            raise ValueError(f"Invalid probe: {field}")


def _check_probe_limits(probe: dict, limits: dict) -> None:
    """Check the schema-validated probe against this worker's resolved model."""
    _require(
        isinstance(limits, dict)
        and type(limits.get("chunked_prefill")) is bool
        and all(
            type(limits.get(k)) is int and limits[k] > 0
            for k in ("vocab_size", "max_input_len", "max_seq_len", "max_num_tokens")
        ),
        "Missing or invalid engine probe limits.",
    )
    prompt, expected = probe["prompt_token_ids"], probe["expected_token_ids"]
    if not (
        len(prompt) <= limits["max_input_len"]
        and len(prompt) + len(expected) <= limits["max_seq_len"]
        and (limits["chunked_prefill"] or len(prompt) <= limits["max_num_tokens"])
        and max(max(prompt), max(expected)) < limits["vocab_size"]
    ):
        raise ValueError("Checkpoint probe exceeds the model limits.")


class EngineClient:
    """Signed, direct IPv4 loopback client for a single fixed-cohort TP1 worker."""

    def __init__(self, port: int, key: str) -> None:
        """Bind an independently secured endpoint.

        Args:
            port: Local listener port, never a URL or proxy address.
            key: Current runtime-control HMAC key.
        """
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Expected a loopback TCP port.")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Missing runtime-control key.")
        self._port = port
        self._key = key

    def _exchange(
        self,
        path: str,
        deadline: float,
        post: Callable[[], dict] | None = None,
        on_send: Callable[[], None] | None = None,
    ) -> bytes:
        """Connect first, then build and send at most one request within the deadline."""
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=_remaining(deadline))
        conn.set_debuglevel(0)
        with closing(conn):
            try:
                conn.connect()
            except OSError:
                raise RuntimeError("Could not connect to the checkpoint listener.") from None
            sock = conn.sock
            assert sock is not None

            def expire() -> None:
                # A per-read socket timeout alone can be extended by trickling bytes.
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

            _remaining(deadline)
            method = "GET" if post is None else "POST"
            body = b"" if post is None else json.dumps(post(), allow_nan=False).encode()
            # /ready is a public health and admission check, not an authenticated lifecycle route.
            headers = {} if path == "/ready" else _sign(self._key, method, path, body)
            if post is not None:
                headers["Content-Type"] = "application/json"
            timer = threading.Timer(_remaining(deadline), expire)
            timer.start()
            try:
                _remaining(deadline)
                if on_send is not None:
                    on_send()
                conn.request(method, path, body=body or None, headers=headers)
                with conn.getresponse() as response:
                    _require(response.status == 200, f"Checkpoint HTTP {response.status}.")
                    raw = response.read(_MAX_BODY + 1)
                    _require(response.length in (None, 0), "Incomplete or oversized HTTP body.")
            except (OSError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Checkpoint deadline expired.") from None
                raise RuntimeError("Checkpoint HTTP transport failed; outcome unknown.") from None
            finally:
                timer.cancel()
                timer.join()
        _remaining(deadline)
        _require(len(raw) <= _MAX_BODY, "Oversized checkpoint response.")
        return raw

    def _request(
        self,
        path: str,
        deadline: float,
        post: Callable[[], dict] | None = None,
        on_send: Callable[[], None] | None = None,
    ) -> dict:
        raw = self._exchange(path, deadline, post, on_send)
        try:
            obj = json.loads(raw, object_pairs_hook=_object)
        except (ValueError, RecursionError):
            raise RuntimeError("Invalid checkpoint JSON response.") from None
        _require(isinstance(obj, dict), "Expected a checkpoint JSON object.")
        return obj

    def _status(self, stage: str, deadline: float, *, restored: bool = False) -> tuple[dict, float]:
        start = time.monotonic()
        status = self._request("/checkpoint/status", deadline)
        profile = status.get("profile")
        _require(
            isinstance(profile, dict)
            and profile.get("ranks") == [0]
            and type(profile["ranks"][0]) is int
            and all(_digest(profile.get(k)) for k in _BINDING[1:])
            and profile.get("capture_policies") == ["resident"]
            and profile.get("drain_policies") == ["natural"],
            "Expected the resident, naturally drained TP1 checkpoint profile.",
        )
        _require(
            status.get("stage") == stage
            and _identifier(status.get("source_session_id"))
            and type(status.get("state_version")) is int
            and status["state_version"] >= 0
            and {"remaining_seconds", "operation", "restore_session_id"} <= status.keys(),
            "Unexpected checkpoint stage, source, or version.",
        )
        ready, flags = status.get("readiness"), _READY[stage]
        remaining = status["remaining_seconds"]
        if stage in ("capture_safe", "runtime_ready"):
            # A captured source timer can expire in storage. It is not the restore deadline.
            if restored and type(remaining) in (int, float) and remaining == 0:
                flags = set()
            else:
                _require(_seconds(remaining), "Checkpoint readiness budget expired or invalid.")
                if not restored:
                    deadline = min(deadline, start + remaining)
        else:
            _require(remaining is None, "Unexpected active checkpoint deadline.")
        _require(
            isinstance(ready, dict) and all(ready.get(k) is (k in flags) for k in _FLAGS),
            "Inconsistent current checkpoint readiness or admission.",
        )
        _remaining(deadline)
        return status, deadline

    def _mutate(
        self,
        route: str,
        stage: str,
        status: dict,
        binding: dict,
        kind: str,
        deadline: float,
        *,
        before_dispatch: Callable[[], None] | None = None,
        on_send: Callable[[], None] | None = None,
        **fields: object,
    ) -> tuple[dict, float]:
        version = status["state_version"]
        binding = {**binding, "operation_id": secrets.token_hex(16)}

        def post() -> dict:
            if before_dispatch is not None:
                before_dispatch()
            context = {
                **binding,
                "protocol_version": 1,
                "session_kind": kind,
                "expected_state_version": version,
                "time_budget_seconds": _remaining(deadline),
            }
            return {"context": context, **fields}

        result = self._request("/checkpoint/" + route, deadline, post, on_send)
        # Fresh operation IDs need no journal retries. An in-progress/uncertain reply fails closed.
        _check_result(result, binding, stage)
        _require(result["state_version"] > version, "Checkpoint version did not advance.")
        current, deadline = self._status(stage, deadline)
        _require(
            current["source_session_id"] == status["source_session_id"]
            and current["restore_session_id"]
            == (None if kind == "source" else binding["session_id"])
            and current["profile"] == status["profile"]
            and current["state_version"] == result["state_version"]
            and json.dumps(current.get("operation"), sort_keys=True)
            == json.dumps(result, sort_keys=True),
            "Current status does not corroborate the checkpoint result.",
        )
        return current, deadline

    def prepare(
        self,
        template_id: str,
        budget: float,
        probe: dict,
        *,
        before_dispatch: Callable[[], None],
        on_send: Callable[[], None] | None = None,
    ) -> dict:
        """Prepare a running source and read back live capture permission.

        Args:
            template_id: Immutable artifact identifier.
            budget: Shared seconds for preparation and host capture, at most 3600.
            probe: Tokenized prompt and reference to preflight against the running source.
            before_dispatch: Required callback to record intent before sending the prepare POST.
                Recording intent alone does not indicate that the POST was attempted.
            on_send: Optional nonblocking in-memory marker, called immediately before sending.
                After this point a lost reply does not establish whether the engine mutated.

        Returns:
            Four source identity fields, remaining_seconds at return, and
            capture_deadline on this CPU host's monotonic clock. Use the absolute
            deadline for capture; do not restart the duration or persist it for restore.
        """
        deadline = _deadline(budget)
        if not _identifier(template_id):
            raise ValueError("Invalid template ID.")
        if not callable(before_dispatch):
            raise TypeError("Expected a prepare dispatch callback.")
        validate_probe(probe)
        status, deadline = self._status("running", deadline)
        _require(
            status["restore_session_id"] is None and status["operation"] is None,
            "Source is not a fresh engine.",
        )
        _check_probe_limits(probe, status["profile"].get("probe_limits"))
        source = status["source_session_id"]
        artifact = {"template_id": template_id, **{k: status["profile"][k] for k in _BINDING[1:]}}
        _, deadline = self._mutate(
            "prepare_capture",
            "capture_safe",
            status,
            {**artifact, "session_id": source},
            "source",
            deadline,
            before_dispatch=before_dispatch,
            on_send=on_send,
            capture_policy="resident",
            drain_policy="natural",
        )
        return {
            **artifact,
            "source_session_id": source,
            "capture_deadline": deadline,
            "remaining_seconds": _remaining(deadline),
        }

    def validate_and_activate(
        self, identity: dict, session_id: str, result_id: str, probe: dict, budget: float
    ) -> None:
        """Validate an isolated restored worker, confirm activation, then check traffic readiness.

        Args:
            identity: Independently verified captured source and artifact identity.
                Any source remaining_seconds is *not* a destination deadline.
            session_id: Fresh destination attempt bound by the host.
            result_id: Independently verified, final successful host result for
                this destination and artifact. Not a marker or an engine result.
            probe: Artifact-bound prompt_token_ids and exact expected_token_ids.
            budget: One relative deadline covering validation through activation.
        """
        deadline = _deadline(budget)
        if not (
            isinstance(identity, dict)
            and _identifier(identity.get("source_session_id"))
            and _identifier(identity.get("template_id"))
            and all(_digest(identity.get(k)) for k in _BINDING[1:])
            and _identifier(session_id)
            and session_id != identity["source_session_id"]
            and _identifier(result_id)
        ):
            raise ValueError(
                "Invalid source, artifact, host result, or fresh destination identity."
            )
        artifact = {k: identity[k] for k in _BINDING}
        source = identity["source_session_id"]
        validate_probe(probe)
        status, deadline = self._status("capture_safe", deadline, restored=True)
        _require(
            status["source_session_id"] == source
            and status["restore_session_id"] is None
            and all(status["profile"][k] == artifact[k] for k in _BINDING[1:]),
            "Restored engine does not identify the expected unbound source.",
        )
        op = status.get("operation")
        _require(
            isinstance(op, dict)
            and _identifier(op.get("operation_id"))
            and op.get("state_version") == status["state_version"],
            "Missing or stale captured preparation result.",
        )
        rank = _check_result(
            op,
            {**artifact, "session_id": source, "operation_id": op["operation_id"]},
            "capture_safe",
        )
        _check_probe_limits(probe, status["profile"].get("probe_limits"))
        binding = {**artifact, "session_id": session_id}
        current, deadline = self._mutate(
            "validate_runtime",
            "runtime_ready",
            status,
            binding,
            "restore",
            deadline,
            source_session_id=source,
            validation=probe,
        )
        validated = current["operation"]
        _require(
            validated["rank_results"][0]["state_digest"] == rank["state_digest"],
            "Validated inventory differs from the captured rank.",
        )
        serving, deadline = self._mutate(
            "activate",
            "serving",
            current,
            binding,
            "restore",
            deadline,
            readiness_id=validated["readiness_id"],
            host_restore={**binding, "result_id": result_id, "status": "succeeded"},
        )
        committed = serving["operation"]
        _require(
            committed["readiness_id"] == validated["readiness_id"]
            and committed["rank_results"][0]["state_digest"] == rank["state_digest"],
            "Activation does not corroborate the validated runtime.",
        )
        self._exchange("/ready", deadline)
