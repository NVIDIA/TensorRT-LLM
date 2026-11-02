# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe isolated native serving candidates; do not certify Snapshot restore."""

import argparse
import hashlib
import http.client
import ipaddress
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_SCHEMA_VERSION = 1
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_UNTESTED = {
    "capture_quiescence": "UNTESTED",
    "source_termination": "UNTESTED",
    "process_and_cuda_restore": "UNTESTED",
    "weight_and_graph_reuse": "UNTESTED",
    "kv_scheduler_validity": "UNTESTED",
    "rank_and_peer_agreement": "UNTESTED",
    "serving_admission": "UNTESTED",
}


def _digest(value: Any) -> str:
    """Hash JSON data with stable key ordering.

    Args:
        value: JSON-serializable input.

    Returns:
        The SHA-256 digest of the canonical JSON representation.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_json(path: Path) -> Any:
    """Read JSON data from a local file.

    Args:
        path: Input file.

    Returns:
        The decoded JSON data.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically replace a report inside the private run directory.

    Args:
        path: Destination file.
        report: JSON-serializable observations.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remaining(deadline: float) -> float:
    """Return the remaining attempt budget or raise on expiry.

    Args:
        deadline: Absolute monotonic deadline.

    Returns:
        Remaining seconds.

    Raises:
        TimeoutError: If the attempt deadline has expired.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Startup probe deadline expired")
    return remaining


def _check_process(process: subprocess.Popen) -> None:
    """Require the foreground candidate supervisor to remain alive.

    Args:
        process: Supervisor owned by this attempt.

    Raises:
        RuntimeError: If the supervisor has exited, including with status zero.
    """
    code = process.poll()
    if code is not None:
        raise RuntimeError(f"Candidate supervisor exited with status {code}")


def _read_address(path: Path) -> tuple[str, int]:
    """Read a native --report_addr file and require an isolated loopback target.

    Args:
        path: Address file from the candidate supervisor.

    Returns:
        A numeric loopback host and TCP port.

    Raises:
        ValueError: If the address is malformed or not numeric loopback.
    """
    address = path.read_text(encoding="utf-8").strip()
    parsed = urlsplit("http://" + address)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or not parsed.port
        or not ipaddress.ip_address(parsed.hostname).is_loopback
    ):
        raise ValueError("Candidate must report a numeric loopback host and nonzero port")
    return parsed.hostname, parsed.port


def _request(
    address: tuple[str, int],
    path: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Send one bounded HTTP request without proxies or redirect following.

    Args:
        address: Numeric loopback host and port.
        path: HTTP request path.
        timeout: Total HTTP request budget in seconds.
        payload: Optional JSON POST body; otherwise send GET.
        headers: Optional authentication headers for an isolated candidate.

    Returns:
        HTTP status and response bytes.

    Raises:
        ValueError: If the response exceeds the probe's size limit.
    """
    deadline = time.monotonic() + timeout
    connection = http.client.HTTPConnection(*address, timeout=timeout)
    watchdog = None
    expired = threading.Event()
    try:
        connection.connect()
        request_socket = connection.sock

        def interrupt() -> None:
            """Interrupt a response that keeps the socket alive beyond its budget."""
            expired.set()
            try:
                request_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        watchdog = threading.Timer(_remaining(deadline), interrupt)
        watchdog.start()
        connection.request(
            "POST" if payload is not None else "GET",
            path,
            body=json.dumps(payload) if payload is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ValueError("HTTP response exceeds probe size limit")
        return response.status, body
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
        connection.close()
        if expired.is_set():
            raise TimeoutError("HTTP request deadline expired")


def _wait_for_health(
    process: subprocess.Popen, address_file: Path, deadline: float
) -> tuple[str, int]:
    """Wait for HTTP health, without treating it as serving validity.

    Args:
        process: Foreground candidate supervisor.
        address_file: Fresh native address report for this attempt.
        deadline: Absolute monotonic attempt deadline.

    Returns:
        The candidate's numeric loopback address.
    """
    address = None
    while True:
        _check_process(process)
        remaining = _remaining(deadline)
        if address is None:
            try:
                address = _read_address(address_file)
            except FileNotFoundError:
                pass
        if address is not None:
            try:
                status, _ = _request(address, "/health", min(1.0, remaining))
                if status == 200:
                    _remaining(deadline)
                    _check_process(process)
                    return address
                if status != 503:
                    raise ValueError(f"Unexpected health HTTP status {status}")
            except (OSError, http.client.HTTPException):
                pass
        time.sleep(min(0.1, _remaining(deadline)))


def _completion(body: bytes) -> dict[str, Any]:
    """Validate a non-streaming, single-choice completion for exact comparison.

    Args:
        body: JSON completion response.

    Returns:
        Output text, finish reason and completion token count.

    Raises:
        ValueError: If the completion is empty, incomplete or malformed.
    """
    result = json.loads(body)
    if not isinstance(result, dict):
        raise ValueError("Completion response must be an object")
    choices = result.get("choices")
    usage = result.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError("Completion must have one choice and usage")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Completion choice must be an object")
    text = choice.get("text")
    reason = choice.get("finish_reason")
    tokens = usage.get("completion_tokens")
    if (
        not isinstance(text, str)
        or not text.strip()
        or reason not in ("stop", "length")
        or type(tokens) is not int
        or tokens <= 0
    ):
        raise ValueError("Completion must contain text, a finish reason and positive token usage")
    return {"text": text, "finish_reason": reason, "completion_tokens": tokens}


def _stop_process_group(process: subprocess.Popen) -> None:
    """Stop this attempt's POSIX process group, including surviving children.

    Args:
        process: Supervisor launched with start_new_session=True.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The supervisor can exit before its workers. Reap the entire owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_probe(
    command: list[str],
    run_dir: Path,
    profile: dict[str, Any],
    requests: list[dict[str, Any]],
    timeout: float,
    mode: str,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Launch and probe one isolated candidate, preserving failed-attempt evidence.

    Args:
        command: Foreground argv with one literal {address} argument placeholder.
        run_dir: New private directory; existing directories are rejected.
        profile: Operator-declared model/build/topology identity, not attestation.
        requests: Deterministic completion request bodies.
        timeout: Total health and generation deadline in seconds.
        mode: cold or restore; the latter is an operator label, not restore proof.
        baseline: Successful cold report required for restore comparisons.

    Returns:
        Report with generation-probe PASS/FAIL and separately untested restore gates.

    Raises:
        ValueError: If inputs or the baseline are invalid.
        FileExistsError: If run_dir already exists.
    """
    if os.name != "posix":
        raise ValueError("The probe requires POSIX process-group cleanup")
    if not command or command.count("{address}") != 1:
        raise ValueError("Command must contain one standalone {address} argument")
    if not math.isfinite(timeout) or timeout <= 0 or mode not in ("cold", "restore"):
        raise ValueError("Use a positive finite timeout and cold or restore mode")
    if (
        not isinstance(profile, dict)
        or not profile
        or not isinstance(requests, list)
        or not requests
    ):
        raise ValueError("A nonempty profile and request list are required")
    for request in requests:
        if (
            not isinstance(request, dict)
            or not isinstance(request.get("model"), str)
            or not request["model"].strip()
            or not isinstance(request.get("prompt"), str)
            or not request["prompt"].strip()
            or type(request.get("max_tokens")) is not int
            or request["max_tokens"] <= 0
            or type(request.get("temperature")) not in (int, float)
            or request["temperature"] != 0
            or request.get("stream") is not False
            or type(request.get("n", 1)) is not int
            or request.get("n", 1) != 1
        ):
            raise ValueError(
                "Use nonempty model/prompt, max_tokens > 0, temperature=0, stream=false, n=1"
            )
    identity = {"profile_digest": _digest(profile), "requests_digest": _digest(requests)}
    if mode == "restore":
        if (
            not isinstance(baseline, dict)
            or baseline.get("schema_version") != _SCHEMA_VERSION
            or baseline.get("mode") != "cold"
            or baseline.get("generation_probe_status") != "PASS"
            or baseline.get("cleanup_status") != "PASS"
            or any(baseline.get(key) != value for key, value in identity.items())
            or not isinstance(baseline.get("outputs"), list)
            or len(baseline["outputs"]) != len(requests)
        ):
            raise ValueError(
                "Restore needs a successful cold baseline with matching profile and requests"
            )
    elif baseline is not None:
        raise ValueError("Cold mode does not accept a baseline")

    run_dir = run_dir.absolute()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    address_file = run_dir / "address"
    argv = [str(address_file) if arg == "{address}" else arg for arg in command]
    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "attempt_id": uuid.uuid4().hex,
        "mode": mode,
        **identity,
        "profile": profile,
        "command": argv,
        "generation_probe_status": "FAIL",
        "snapshot_qualification": "UNTESTED",
        "gates": dict(_UNTESTED),
        "outputs": [],
        "timings_seconds": {},
    }
    process = None
    started = time.monotonic()
    deadline = started + timeout
    _write_report(run_dir / "report.json", report)
    try:
        with (run_dir / "candidate.log").open("wb") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)
            report["candidate_pid"] = process.pid
            address = _wait_for_health(process, address_file, deadline)
            report["timings_seconds"]["http_health"] = time.monotonic() - started
            for request in requests:
                _check_process(process)
                status, body = _request(address, "/v1/completions", _remaining(deadline), request)
                _remaining(deadline)
                if status != 200:
                    raise ValueError(f"Generation returned HTTP {status}")
                output = _completion(body)
                report["outputs"].append(output)
                report["timings_seconds"].setdefault("first_completion", time.monotonic() - started)
                _check_process(process)
            if baseline is not None and report["outputs"] != baseline["outputs"]:
                raise ValueError("Candidate completions do not exactly match the cold baseline")
            report["generation_probe_status"] = "PASS"
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        report["generation_probe_status"] = "FAIL"
        report["error"] = "KeyboardInterrupt: probe interrupted"
        raise
    finally:
        report["timings_seconds"]["probe"] = time.monotonic() - started
        try:
            if process is not None:
                _stop_process_group(process)
            report["cleanup_status"] = "PASS"
        except (OSError, subprocess.TimeoutExpired) as error:
            report["generation_probe_status"] = "FAIL"
            report["cleanup_status"] = "FAIL"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
        finally:
            _write_report(run_dir / "report.json", report)
    return report


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize probe observations without promoting them to restore evidence.

    Args:
        reports: Reports for one identical profile and deterministic request set.

    Returns:
        Attempt counts and passing-probe p50/p95 times, grouped by mode.

    Raises:
        ValueError: If reports are malformed, incompatible or disagree on output.
    """
    if not reports:
        raise ValueError("At least one probe report is required")
    identity = None
    expected_outputs = None
    groups: dict[str, dict[str, Any]] = {
        mode: {"attempted": 0, "passed": 0, "failed": 0} for mode in ("cold", "restore")
    }
    samples: dict[str, list[float]] = {mode: [] for mode in groups}
    attempts = set()
    for report in reports:
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != _SCHEMA_VERSION
            or report.get("mode") not in groups
            or report.get("generation_probe_status") not in ("PASS", "FAIL")
            or report.get("snapshot_qualification") != "UNTESTED"
            or not isinstance(report.get("profile"), dict)
            or report.get("profile_digest") != _digest(report["profile"])
            or not isinstance(report.get("requests_digest"), str)
            or not report["requests_digest"]
            or not isinstance(report.get("attempt_id"), str)
            or not report["attempt_id"]
        ):
            raise ValueError("Malformed or unsupported probe report")
        if report["attempt_id"] in attempts:
            raise ValueError("Duplicate attempt IDs do not represent independent attempts")
        attempts.add(report["attempt_id"])
        key = (report["profile_digest"], report["requests_digest"])
        if identity is not None and key != identity:
            raise ValueError("Reports must have identical profile and request digests")
        identity = key
        group = groups[report["mode"]]
        group["attempted"] += 1
        if report["generation_probe_status"] == "FAIL":
            group["failed"] += 1
            continue
        timing = report.get("timings_seconds", {})
        elapsed = timing.get("first_completion") if isinstance(timing, dict) else None
        outputs = report.get("outputs")
        if (
            report.get("cleanup_status") != "PASS"
            or type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or not isinstance(outputs, list)
            or not outputs
        ):
            raise ValueError("Passing probes require cleanup, output and a finite completion time")
        if expected_outputs is not None and outputs != expected_outputs:
            raise ValueError("Passing probe outputs differ")
        expected_outputs = outputs
        group["passed"] += 1
        samples[report["mode"]].append(elapsed)
    for mode, values in samples.items():
        groups[mode]["launch_to_first_completion_seconds"] = {
            "samples": len(values),
            "p50": statistics.median(values) if values else None,
            "p95": (
                statistics.quantiles(values, n=100, method="inclusive")[94]
                if len(values) > 1
                else values[0]
                if values
                else None
            ),
        }
    assert identity is not None
    return {
        "schema_version": _SCHEMA_VERSION,
        "profile_digest": identity[0],
        "requests_digest": identity[1],
        "snapshot_qualification": "UNTESTED",
        "modes": groups,
        "percentile_method": "inclusive interpolation; passing probes only; one sample is itself",
        "limitations": "Operator labels are not process/CUDA restore proof or measured restore speedup.",
    }


def main() -> int:
    """Run a CLI probe or summarize compatible observations.

    Returns:
        Zero for a passed probe or a written summary; one for a failed probe.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    probe = commands.add_parser("probe", help="Launch and measure one isolated candidate")
    probe.add_argument("--mode", choices=("cold", "restore"), required=True)
    probe.add_argument("--profile", type=Path, required=True)
    probe.add_argument("--requests", type=Path, required=True)
    probe.add_argument("--run-dir", type=Path, required=True)
    probe.add_argument("--baseline", type=Path)
    probe.add_argument("--timeout", type=float, default=600)
    probe.add_argument("command", nargs=argparse.REMAINDER)
    summary = commands.add_parser("summarize", help="Summarize compatible probe reports")
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    if args.action == "summarize":
        try:
            if len({path.resolve() for path in args.reports}) != len(args.reports):
                raise ValueError("Duplicate report paths do not represent independent attempts")
            result = summarize_reports([_read_json(path) for path in args.reports])
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(args.output)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = run_probe(
            command,
            args.run_dir,
            _read_json(args.profile),
            _read_json(args.requests),
            args.timeout,
            args.mode,
            _read_json(args.baseline) if args.baseline else None,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Generation probe: {report['generation_probe_status']}; Snapshot qualification: UNTESTED"
    )
    print(args.run_dir / "report.json")
    return 0 if report["generation_probe_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
