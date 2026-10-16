# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run bounded late-receiver trials against an already-running dedicated server.

This driver does not allocate compute, start servers, or retry requests.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_TIMEOUT_MS = 60_000


def _finite_number(value: object) -> bool:
    """Accept finite clock values, excluding booleans masquerading as numbers."""
    return type(value) in (int, float) and math.isfinite(value)


def _owner(event: dict) -> tuple:
    """Identify one rank in one worker process on the shared-clock host."""
    return event["host"], event["pid"], event.get("rank")


def _helper() -> ModuleType:
    """Load the stdlib controller without importing the native runtime."""
    source = Path(__file__).resolve().parents[4] / "tensorrt_llm"
    spec = importlib.util.spec_from_file_location(
        "_late_receiver_controller", source / "_disagg_late_receiver_diagnostic.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Diagnostic controller cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify(mode: str, response: dict, events: list[dict]) -> str:
    """Require correlated lifecycle evidence, not an HTTP timeout alone.

    Args:
        mode: Control or delayed-receiver arm.
        response: Captured HTTP status and response body.
        events: Immutable events for one trial.

    Returns:
        CONTROL_OK, MECHANISM_REPRODUCED, or INCONCLUSIVE.
    """
    if mode not in {"control", "after_retirement"} or not events:
        return "INCONCLUSIVE"
    grouped: dict[str, list[dict]] = {}
    for event in events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("event"), str)
            or not isinstance(event.get("host"), str)
            or not event["host"]
            or not isinstance(event.get("trial_id"), str)
            or not event["trial_id"]
            or event.get("mode") != mode
            or type(event.get("disagg_request_id")) is not int
            or type(event.get("monotonic_ns")) is not int
            or event["monotonic_ns"] < 0
            or type(event.get("pid")) is not int
            or event.get("no_retry") is not True
        ):
            return "INCONCLUSIVE"
        grouped.setdefault(event["event"], []).append(event)
    # Do not discard other IDs/hosts: they make attribution ambiguous. Every
    # monotonic timestamp below must come from the same single-node experiment.
    if (
        len({(event["trial_id"], event["host"], event["disagg_request_id"]) for event in events})
        != 1
    ):
        return "INCONCLUSIVE"
    invalid_events = {
        "gate_invalid",
        "gate_timeout",
        "router_cancelled",
        "router_no_generation",
        "sender_terminal_reply_attempt",
    }
    if invalid_events.intersection(grouped) or any(len(group) != 1 for group in grouped.values()):
        return "INCONCLUSIVE"
    required = {
        "ctx_send_published",
        "router_gate_released",
        "gen_receive_published",
        "gen_request_data_sent",
        "gen_timeout_started",
        "sender_request_data",
    }
    if not required.issubset(grouped):
        return "INCONCLUSIVE"
    records = {name: group[0] for name, group in grouped.items()}
    release = records["router_gate_released"]
    sender = records["ctx_send_published"]
    publication = records["gen_receive_published"]
    wire = records["gen_request_data_sent"]
    start = records["gen_timeout_started"]
    message = records["sender_request_data"]
    started_at = start.get("timeout_start_monotonic")
    if (
        type(sender.get("rank")) is not int
        or type(sender.get("ctx_dp_rank")) is not int
        or type(publication.get("rank")) is not int
        or type(start.get("request_id")) is not int
        or _owner(sender) != _owner(message)
        or _owner(wire) != _owner(publication)
        or _owner(publication) != _owner(start)
        or release.get("rank") != sender["rank"]
        or message.get("peer_rank") != publication["rank"]
        or release.get("ctx_dp_rank") != sender.get("ctx_dp_rank")
        or sender.get("pp_size") != 1
        or sender.get("cp_size") != 1
        or sender.get("attention_dp") is not True
        or sender.get("bridge_enabled") is not False
        or sender.get("pipeline_transfer") is not False
        or sender.get("kv_transfer_timeout_ms") != _TIMEOUT_MS
        or start.get("kv_transfer_timeout_ms") != _TIMEOUT_MS
        or not _finite_number(started_at)
        or not (
            sender["monotonic_ns"]
            <= release["monotonic_ns"]
            <= wire["monotonic_ns"]
            <= publication["monotonic_ns"]
            <= started_at * 1_000_000_000
            <= start["monotonic_ns"]
        )
        or message["monotonic_ns"] < release["monotonic_ns"]
        or message.get("fenced") is not False
    ):
        return "INCONCLUSIVE"
    # The CTX listener may handle the request before GEN records messenger.send's
    # return. Do not order sender_request_data against gen_request_data_sent.
    if mode == "control":
        valid = (
            release.get("reason") == "control"
            and response.get("status") == 200
            and message.get("session_present") is True
            and not {"ctx_retired", "gen_timeout_expired"}.intersection(records)
        )
        return "CONTROL_OK" if valid else "INCONCLUSIVE"
    if not {
        "ctx_timeout_started",
        "ctx_native_closed",
        "ctx_retired",
        "gen_timeout_expired",
    }.issubset(records):
        return "INCONCLUSIVE"
    ctx_start = records["ctx_timeout_started"]
    closed = records["ctx_native_closed"]
    retired = records["ctx_retired"]
    expired = records["gen_timeout_expired"]
    elapsed_ms = expired.get("observed_elapsed_ms")
    if (
        release.get("reason") != "confirmed_retirement"
        or any(_owner(event) != _owner(sender) for event in (ctx_start, closed, retired))
        or retired.get("ctx_dp_rank") != release.get("ctx_dp_rank")
        or not all(
            retired.get(field) is True
            for field in ("timed_out", "close_confirmed", "session_absent", "retired_flag")
        )
        or not (
            sender["monotonic_ns"]
            <= ctx_start["monotonic_ns"]
            <= closed["monotonic_ns"]
            <= retired["monotonic_ns"]
            < release["monotonic_ns"]
        )
        or _owner(expired) != _owner(start)
        or expired.get("request_id") != start.get("request_id")
        or expired.get("timeout_start_monotonic") != started_at
        or expired.get("kv_transfer_timeout_ms") != _TIMEOUT_MS
        or not _finite_number(elapsed_ms)
        or elapsed_ms <= _TIMEOUT_MS
        or expired["monotonic_ns"] <= (started_at + _TIMEOUT_MS / 1000) * 1_000_000_000
        or expired["monotonic_ns"] < start["monotonic_ns"]
        or expired["monotonic_ns"] < message["monotonic_ns"]
    ):
        return "INCONCLUSIVE"
    valid = (
        message.get("session_present") is False
        and response.get("status") == 400
        and isinstance(response.get("body"), str)
        and "Request timed out (KV transfer)" in response["body"]
    )
    return "MECHANISM_REPRODUCED" if valid else "INCONCLUSIVE"


def _request(url: str, payload: bytes, timeout: float) -> dict:
    """Send one non-streaming request and capture HTTP errors without retrying."""
    started = time.monotonic_ns()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status, body = response.status, response.read().decode("utf-8")
    except HTTPError as error:
        status, body = error.code, error.read().decode("utf-8")
    except (TimeoutError, URLError) as error:
        status, body = None, str(error)
    return {"status": status, "body": body, "elapsed_ns": time.monotonic_ns() - started}


def main() -> int:
    """Run the armed experiment and retain one result per fresh trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--wait-seconds", type=float, default=90)
    parser.add_argument("--http-timeout", type=float, default=240)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10:
        parser.error("--repetitions must be between 1 and 10")
    if not 0 < args.wait_seconds <= 180:
        parser.error("--wait-seconds must be positive and at most 180")
    if args.http_timeout < args.wait_seconds + 90 or args.http_timeout > 600:
        parser.error("--http-timeout must allow the gate plus 90 seconds, and be at most 600")
    endpoint = urlsplit(args.url)
    if endpoint.scheme not in {"http", "https"} or endpoint.path not in {
        "/v1/completions",
        "/v1/chat/completions",
    }:
        parser.error("--url must name the dedicated completions or chat/completions endpoint")
    body = json.loads(args.payload.read_text())
    if not isinstance(body, dict) or body.get("stream", False):
        parser.error("--payload must contain one non-streaming JSON object")
    if "disaggregated_params" in body:
        parser.error("The diagnostic server must assign fresh request IDs")
    payload = json.dumps(body).encode("utf-8")
    helper = _helper()
    outcomes = []
    request_ids = set()
    for mode in ["control", *(["after_retirement"] * args.repetitions)]:
        trial = helper.arm(args.diagnostic_dir, mode, args.wait_seconds)
        trial_id = trial["trial_id"]
        try:
            response = _request(args.url, payload, args.http_timeout)
            events = helper.read_events(args.diagnostic_dir, trial_id)
            outcome = classify(mode, response, events)
            releases = [event for event in events if event["event"] == "router_gate_released"]
            if len(releases) == 1:
                request_id = releases[0]["disagg_request_id"]
                if request_id in request_ids:
                    outcome = "INCONCLUSIVE"
                request_ids.add(request_id)
            record = {
                "trial": trial,
                "classification": outcome,
                "response": response,
                "events": events,
                "meaning": "Characterization only; not a product fix or full stress-test pass.",
            }
            result_path = args.diagnostic_dir / "trials" / trial_id / "client-result.json"
            with result_path.open("x") as result_file:
                json.dump(record, result_file, indent=2)
                result_file.write("\n")
            outcomes.append(outcome)
            print(json.dumps({"trial_id": trial_id, "classification": outcome}), flush=True)
        finally:
            helper.disarm(args.diagnostic_dir, trial_id)
        if outcome == "INCONCLUSIVE":
            return 2
    return 0 if outcomes == ["CONTROL_OK", *(["MECHANISM_REPRODUCED"] * args.repetitions)] else 2


if __name__ == "__main__":
    raise SystemExit(main())
