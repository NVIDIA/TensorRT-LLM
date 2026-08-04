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
"""Pure coordination helpers for paired perf-sanity agreement experiments."""

import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Iterable, Optional, Set, Tuple

BENCHMARK_STATUS_FAILED = "FAILED"
BENCHMARK_STATUS_SUCCESS = "SUCCESS"
CONTEXT_ACTIVATION_DIGEST_ALGORITHM = "sha256-length-prefixed-prompt-sha256-v1"

_AGREEMENT_ARM_MARKER_RE = re.compile(
    r"PYTHON_AGREEMENT_AB_ARM_(START|END) "
    r"server_idx=(\d+) role=(CTX_\d+) process_id=(\d+)"
)
_BACKEND_EVENT_LOOP_FATAL_MARKERS = (
    "Event loop terminated with error:",
    "Error in event loop:",
    "Broadcasting event-loop error",
)
_BACKEND_EVENT_LOOP_FATAL_OVERLAP = max(map(len, _BACKEND_EVENT_LOOP_FATAL_MARKERS))
_CONTEXT_MODE_RE = re.compile(
    r"PYTHON_ASYNC_CONSENSUS transition=mode_active "
    r"[^\n]*?rank=(\d+) terminal=(\d+) peer_ready=(\d+)"
)
_CONTEXT_SHUTDOWN_RE = re.compile(
    r"PYTHON_ASYNC_CONSENSUS transition=shutdown_summary "
    r"rank=(\d+) counters=\{[^\n]*\}"
)
_CONTEXT_ACTIVATION_SEQUENCE_RE = re.compile(
    r"PYTHON_CONTEXT_ACTIVATION_SEQUENCE "
    r"rank=(\d+) count=(\d+) digest=([0-9a-f]{64}) "
    rf"algorithm={re.escape(CONTEXT_ACTIVATION_DIGEST_ALGORITHM)}"
)


def write_atomic_coordination_text(path: str, text: str) -> None:
    """Atomically publish one small cross-role coordination marker."""
    directory = os.path.dirname(path) or "."
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as marker:
            temporary_path = marker.name
            marker.write(text)
            marker.flush()
            os.fsync(marker.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def read_coordination_text(path: str) -> Optional[str]:
    """Return a coordination marker's contents, or ``None`` before publication."""
    try:
        with open(path) as marker:
            return marker.read().strip()
    except FileNotFoundError:
        return None


def extract_log_after_offset(log_bytes: bytes, offset_text: str) -> bytes:
    """Return cumulative log bytes after one published byte offset."""
    try:
        offset = int(offset_text)
    except ValueError as error:
        raise ValueError(f"Invalid log offset: {offset_text!r}") from error
    if offset < 0 or offset > len(log_bytes):
        raise ValueError(
            f"Log offset is outside the cumulative log: offset={offset}, log_size={len(log_bytes)}"
        )
    return log_bytes[offset:]


def extract_measured_context_lifecycle_log(
    log_text: str,
    expected_ranks: Set[int],
) -> Optional[str]:
    """Exclude memory-profiling evidence from one measured CTX lifecycle.

    PyTorch serving constructs and shuts down a profiling executor before the
    final runtime executor. After the first complete profiling-shutdown set,
    the first complete PP mode set identifies the final runtime's exact start.
    """
    first_shutdown_end_by_rank = {}
    for match in _CONTEXT_SHUTDOWN_RE.finditer(log_text):
        rank = int(match.group(1))
        if rank in expected_ranks and rank not in first_shutdown_end_by_rank:
            first_shutdown_end_by_rank[rank] = match.end()
        if set(first_shutdown_end_by_rank) == expected_ranks:
            profiling_boundary = max(first_shutdown_end_by_rank.values())
            break
    else:
        return None

    final_runtime_tail = log_text[profiling_boundary:]
    first_mode_start_by_rank = {}
    for match in _CONTEXT_MODE_RE.finditer(final_runtime_tail):
        rank = int(match.group(1))
        if rank in expected_ranks and rank not in first_mode_start_by_rank:
            first_mode_start_by_rank[rank] = match.start()
        if set(first_mode_start_by_rank) == expected_ranks:
            measured_start = min(first_mode_start_by_rank.values())
            return final_runtime_tail[measured_start:]
    return None


def validate_context_agreement_mode(
    log_text: str,
    expected_ranks: Set[int],
    expected_mode: Tuple[int, int],
) -> bool:
    """Validate a complete final-runtime mode set, or report it as pending."""
    mode_by_rank = {
        int(rank_text): (int(terminal_text), int(peer_ready_text))
        for rank_text, terminal_text, peer_ready_text in _CONTEXT_MODE_RE.findall(log_text)
        if int(rank_text) in expected_ranks
    }
    if set(mode_by_rank) != expected_ranks:
        return False
    observed_modes = set(mode_by_rank.values())
    if observed_modes != {expected_mode}:
        raise ValueError(
            "Agreement mode does not match the configured A/B arm: "
            f"observed={sorted(observed_modes)}, expected={expected_mode}"
        )
    return True


def parse_context_activation_sequences(log_text: str):
    """Parse only prompt-content activation digests from CTX rank evidence."""
    return {
        int(rank_text): (int(count_text), digest)
        for rank_text, count_text, digest in _CONTEXT_ACTIVATION_SEQUENCE_RE.findall(log_text)
    }


def find_backend_event_loop_fatal(
    log_path: str,
    offset: int = 0,
) -> Tuple[Optional[str], int]:
    """Scan newly appended log text for a fatal backend event-loop line."""
    try:
        with open(log_path, "rb") as server_log:
            server_log.seek(0, os.SEEK_END)
            log_size = server_log.tell()
            scan_start = max(0, min(offset, log_size) - _BACKEND_EVENT_LOOP_FATAL_OVERLAP)
            server_log.seek(scan_start)
            new_text = server_log.read().decode(errors="replace")
            new_offset = server_log.tell()
            for line in new_text.splitlines():
                if any(marker in line for marker in _BACKEND_EVENT_LOOP_FATAL_MARKERS):
                    return line.strip(), new_offset
            return None, new_offset
    except FileNotFoundError:
        return None, 0


def terminate_subprocess(
    process: subprocess.Popen | None,
    description: str,
    timeout_seconds: float,
) -> Optional[RuntimeError]:
    """Reap a subprocess, returning an error when force-kill was required."""
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
        return None
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return RuntimeError(f"{description} could not be reaped after terminate and kill")
        return RuntimeError(
            f"{description} did not terminate within {timeout_seconds:g}s and was killed"
        )


def run_polled_command(
    command: Sequence[str],
    *,
    env: Optional[Mapping[str, str]],
    log_path: str,
    timeout_seconds: Optional[float],
    abort_check: Callable[[], None],
    poll_interval_seconds: float = 1.0,
    terminate_timeout_seconds: float = 60.0,
) -> str:
    """Run a logged command while polling a shared abort and a deadline."""
    process = None
    with open(log_path, "w") as output_file:
        try:
            abort_check()
            process = subprocess.Popen(
                command,
                env=env,
                stdout=output_file,
                stderr=subprocess.STDOUT,
            )
            start_time = time.monotonic()
            while True:
                abort_check()
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, command)
                    break
                elapsed = time.monotonic() - start_time
                if timeout_seconds is not None and elapsed > timeout_seconds:
                    raise TimeoutError(f"Benchmark client timed out after {timeout_seconds:g}s")
                time.sleep(poll_interval_seconds)
        except BaseException as error:
            teardown_error = terminate_subprocess(
                process,
                "Benchmark client",
                terminate_timeout_seconds,
            )
            if teardown_error is not None:
                if hasattr(error, "add_note"):
                    error.add_note(str(teardown_error))
            raise
        else:
            teardown_error = terminate_subprocess(
                process,
                "Benchmark client",
                terminate_timeout_seconds,
            )
            if teardown_error is not None:
                raise teardown_error

    with open(log_path, errors="replace") as output_file:
        return output_file.read()


def validate_completed_benchmark_output(
    output: str,
    *,
    expected_requests: int,
    required_metric_patterns: Mapping[str, re.Pattern],
) -> None:
    """Require a complete successful benchmark before releasing server roles."""
    successful_match = re.search(r"Successful requests:\s+(\d+)", output)
    if successful_match is None:
        raise RuntimeError("Benchmark output is missing the successful-request count")
    successful_requests = int(successful_match.group(1))
    if successful_requests != expected_requests:
        raise RuntimeError(
            "Benchmark output has an unexpected successful-request count: "
            f"observed={successful_requests}, expected={expected_requests}"
        )

    failed_match = re.search(r"Failed requests:\s+(\d+)", output)
    if failed_match is None:
        raise RuntimeError("Benchmark output is missing the failed-request count")
    failed_requests = int(failed_match.group(1))
    if failed_requests != 0:
        raise RuntimeError(f"Benchmark output contains {failed_requests} failed requests")

    if "!FAILED REQUESTS!" in output or "!CHECK LOG FOR ERRORS!" in output:
        raise RuntimeError("Benchmark output contains failure markers")

    missing_metrics = [
        metric_name
        for metric_name, pattern in required_metric_patterns.items()
        if pattern.search(output) is None
    ]
    if missing_metrics:
        raise RuntimeError(
            "Benchmark output is missing required metrics before service teardown: "
            f"{missing_metrics}"
        )


def format_agreement_arm_marker(
    transition: str,
    server_idx: int,
    role: str,
    process_id: str,
) -> str:
    """Return a stable marker that scopes evidence to one CTX service lifetime."""
    if transition not in ("START", "END"):
        raise ValueError(f"Unsupported agreement-arm transition: {transition}")
    return (
        f"PYTHON_AGREEMENT_AB_ARM_{transition} "
        f"server_idx={server_idx} role={role} process_id={process_id}"
    )


def expected_disagg_lifecycle_roles(
    num_ctx_servers: int,
    num_gen_servers: int,
) -> Set[str]:
    """Return every outer pytest controller expected at a lifecycle barrier."""
    roles = {f"CTX_{server_idx}.0" for server_idx in range(num_ctx_servers)}
    roles.update(f"GEN_{server_idx}.0" for server_idx in range(num_gen_servers))
    roles.update(("DISAGG_SERVER.0", "BENCHMARK.0"))
    return roles


def is_paired_agreement_configuration(
    modes: Iterable[Tuple[str, Optional[int], Optional[int]]],
) -> bool:
    """Return whether modes describe arm(s) of a fully specified paired e2e A/B."""
    modes = list(modes)
    return bool(modes) and all(
        benchmark_mode == "e2e" and terminal_mode is not None and peer_ready_mode is not None
        for benchmark_mode, terminal_mode, peer_ready_mode in modes
    )


def extract_agreement_arm_log(
    log_text: str,
    server_idx: int,
    expected_ctx_roles: int,
    role: str = "CTX_0",
) -> Optional[str]:
    """Extract one exact arm from a cumulative outer CTX log.

    Returns ``None`` while the complete marker set has not yet become visible.
    Raises ``ValueError`` for duplicate or inconsistent marker sets.
    """
    all_markers = list(_AGREEMENT_ARM_MARKER_RE.finditer(log_text))
    start_markers = {}
    end_markers = {}
    for marker in all_markers:
        transition, marker_server_idx, marker_role, process_id = marker.groups()
        if int(marker_server_idx) != server_idx or marker_role != role:
            continue
        markers = start_markers if transition == "START" else end_markers
        if process_id in markers:
            raise ValueError(
                "Duplicate agreement-arm marker: "
                f"transition={transition}, server_idx={server_idx}, "
                f"role={role}, process_id={process_id}"
            )
        markers[process_id] = marker

    expected_process_ids = {str(process_id) for process_id in range(expected_ctx_roles)}
    observed_process_ids = set(start_markers).union(end_markers)
    if not observed_process_ids.issubset(expected_process_ids):
        raise ValueError(
            "Agreement-arm markers contain unexpected CTX process IDs: "
            f"observed={sorted(observed_process_ids)}, "
            f"expected={sorted(expected_process_ids)}"
        )
    if set(start_markers) != expected_process_ids or set(end_markers) != expected_process_ids:
        return None

    for process_id in expected_process_ids:
        if end_markers[process_id].start() <= start_markers[process_id].end():
            raise ValueError(
                "Agreement-arm END marker precedes its START marker: "
                f"server_idx={server_idx}, role={role}, process_id={process_id}"
            )
    start_offset = min(marker.end() for marker in start_markers.values())
    next_arm_offsets = [
        marker.start()
        for marker in all_markers
        if marker.group(1) == "START"
        and int(marker.group(2)) > server_idx
        and marker.group(3) == role
    ]
    # Include the cumulative tail so evidence forwarded after END remains in
    # scope. The caller keeps every role at a post-verification barrier, so a
    # later arm cannot start while this tail is being polled.
    end_offset = min(next_arm_offsets) if next_arm_offsets else len(log_text)
    return log_text[start_offset:end_offset]
