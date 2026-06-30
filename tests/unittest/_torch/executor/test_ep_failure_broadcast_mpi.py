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
"""Launcher for logical WideEP failure injection over a real MPI transport."""

import importlib.util
import math
import os
import re
import shutil
import signal
import subprocess  # nosec B404
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import pytest

_SKIP_MARKER = "WIDEEP_MPI_SMOKE_SKIP:"
_PROPAGATION_MARKER = "WIDEEP_MPI_PROPAGATION"
_INVALID_TOPOLOGY_MARKER = "WIDEEP_MPI_INVALID_TOPOLOGY_OK"
_HEALTHY_LIFECYCLE_MARKER = "WIDEEP_MPI_HEALTHY_LIFECYCLE_OK"
_ABORT_WORLD_MARKER = "WIDEEP_MPI_ABORT_WORLD_OK"
_TERMINAL_READY_MARKER = "WIDEEP_MPI_TERMINAL_READY"
_TERMINAL_COMPLETE_MARKER = "WIDEEP_MPI_TERMINAL_COMPLETE"
_TERMINAL_DONE_MARKER = "WIDEEP_MPI_TERMINAL_DONE"
_TERMINAL_ERROR_MARKER = "WIDEEP_MPI_TERMINAL_ERROR"
_TERMINAL_ATEXIT_MARKER = "WIDEEP_MPI_TERMINAL_ATEXIT_RAN"
_WORKER_TIMEOUT_SEC = 30.0
_REAP_TIMEOUT_SEC = 5.0
_ALLOW_SKIP_ENV = "TLLM_ALLOW_MPI_FT_SMOKE_SKIP"
_HEALTHY_MODE = "healthy"
_TERMINAL_MODE = "terminal"
_PROPAGATION_PATTERN = re.compile(
    rf"^{_PROPAGATION_MARKER} world_size=(\d+) "
    r"elapsed_ms=(\S+) target_ms=(\S+) target_met=(True|False)$",
    re.MULTILINE,
)
_TERMINAL_READY_PATTERN = re.compile(rf"^{_TERMINAL_READY_MARKER} rank=(\d+) world_size=(\d+)$")
_TERMINAL_COMPLETE_PATTERN = re.compile(rf"^{_TERMINAL_COMPLETE_MARKER} world_size=(\d+)$")
_TERMINAL_DONE_PATTERN = re.compile(rf"^{_TERMINAL_DONE_MARKER} rank=(\d+) world_size=(\d+)$")


class _ReapableProcess(Protocol):
    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def _mpi_launcher() -> str | None:
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _smoke_is_required(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether missing MPI prerequisites must fail this invocation."""
    environment = os.environ if environment is None else environment
    return environment.get(_ALLOW_SKIP_ENV) != "1"


def _handle_missing_prerequisite(reason: str, *, required: bool) -> None:
    if required:
        pytest.fail(
            f"WideEP MPI FT smoke is required but cannot run: {reason}",
            pytrace=False,
        )
    pytest.skip(reason)


def _subprocess_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _merge_subprocess_output(*outputs: str | bytes | None) -> str:
    """Preserve timeout output without duplicating communicate's cumulative data."""
    merged = ""
    for output in outputs:
        text = _subprocess_output_text(output)
        if not text or text in merged:
            continue
        if merged and merged in text:
            merged = text
        else:
            merged += text
    return merged


def _validate_rank_marker_set(
    output: str,
    expected_world_size: int,
    marker: str,
    pattern: re.Pattern[str],
) -> None:
    """Require one exact structured marker from every expected rank."""
    marker_lines = [line for line in output.splitlines() if marker in line]
    # MPI launchers may prefix a forwarded rank-0 stdout line. The worker emits
    # all success evidence from one canonical stream, so strip only the prefix
    # and retain strict validation of the marker payload itself.
    marker_payloads = [line[line.index(marker) :] for line in marker_lines]
    matches = [pattern.fullmatch(payload) for payload in marker_payloads]
    malformed = [line for line, match in zip(marker_lines, matches) if match is None]
    reported = [
        (int(match.group(1)), int(match.group(2))) for match in matches if match is not None
    ]
    expected_ranks = list(range(expected_world_size))
    reported_ranks = sorted(rank for rank, _world_size in reported)
    invalid_world_sizes = sorted(
        {world_size for _rank, world_size in reported if world_size != expected_world_size}
    )
    if malformed or invalid_world_sizes or reported_ranks != expected_ranks:
        pytest.fail(
            f"expected exactly one {marker} from ranks "
            f"{expected_ranks} with world_size={expected_world_size}, got "
            f"markers={reported}, malformed={malformed}, "
            f"invalid_world_sizes={invalid_world_sizes}:\n{output}",
            pytrace=False,
        )


def _validate_terminal_completion(output: str, expected_world_size: int) -> None:
    """Require exact READY/DONE sets around one global completion marker."""
    if _TERMINAL_ERROR_MARKER in output:
        pytest.fail(f"terminal MPI worker reported an error:\n{output}", pytrace=False)
    if _TERMINAL_ATEXIT_MARKER in output:
        pytest.fail(
            "terminal MPI worker entered Python atexit/mpi4py Finalize instead "
            f"of the intentional os._exit path:\n{output}",
            pytrace=False,
        )

    _validate_rank_marker_set(
        output,
        expected_world_size,
        _TERMINAL_READY_MARKER,
        _TERMINAL_READY_PATTERN,
    )

    completion_lines = [line for line in output.splitlines() if _TERMINAL_COMPLETE_MARKER in line]
    completion_payloads = [
        line[line.index(_TERMINAL_COMPLETE_MARKER) :] for line in completion_lines
    ]
    completion_matches = [
        _TERMINAL_COMPLETE_PATTERN.fullmatch(payload) for payload in completion_payloads
    ]
    if (
        len(completion_matches) != 1
        or completion_matches[0] is None
        or int(completion_matches[0].group(1)) != expected_world_size
    ):
        pytest.fail(
            f"expected exactly one valid {_TERMINAL_COMPLETE_MARKER} with "
            f"world_size={expected_world_size}, got {completion_lines}:\n{output}",
            pytrace=False,
        )
    _validate_rank_marker_set(
        output,
        expected_world_size,
        _TERMINAL_DONE_MARKER,
        _TERMINAL_DONE_PATTERN,
    )


def _validate_worker_result(
    returncode: int,
    output: str,
    *,
    required: bool,
    mode: str = _HEALTHY_MODE,
    expected_world_size: int | None = None,
) -> None:
    """Validate normal-finalize and intentional no-finalize launcher results."""
    # A skip marker never excuses a launcher failure. Required workers return a
    # nonzero status instead of printing this marker, but retain this ordering
    # so a malformed optional worker cannot hide a launch error.
    if returncode != 0 and (mode != _TERMINAL_MODE or _SKIP_MARKER in output):
        pytest.fail(output or f"MPI smoke launcher exited with status {returncode}", pytrace=False)
    if _SKIP_MARKER not in output:
        if mode == _TERMINAL_MODE:
            if expected_world_size is None:
                raise ValueError("terminal MPI smoke validation requires expected_world_size")
            # Open MPI reports a nonzero launcher status when workers
            # intentionally bypass MPI_Finalize via os._exit(). Structured
            # rank readiness plus global completion distinguishes that expected
            # status from a real worker failure or MPI_Abort.
            _validate_terminal_completion(output, expected_world_size)
        return
    if required:
        pytest.fail(
            f"WideEP MPI FT smoke is required but the worker skipped:\n{output.strip()}",
            pytrace=False,
        )
    pytest.skip(output.strip())


def _parse_propagation_metric(output: str, expected_world_size: int) -> tuple[float, bool]:
    """Validate and return the functional smoke's structured latency metric."""
    metric_lines = [line for line in output.splitlines() if _PROPAGATION_MARKER in line]
    metric_payloads = [line[line.index(_PROPAGATION_MARKER) :] for line in metric_lines]
    matches = [_PROPAGATION_PATTERN.fullmatch(payload) for payload in metric_payloads]
    if len(matches) != 1 or matches[0] is None:
        pytest.fail(
            f"expected exactly one valid {_PROPAGATION_MARKER} line, got {metric_lines}:\n{output}",
            pytrace=False,
        )
    match = matches[0]
    assert match is not None
    world_size = int(match.group(1))
    try:
        elapsed_ms = float(match.group(2))
        target_ms = float(match.group(3))
    except ValueError:
        pytest.fail(f"invalid propagation metric:\n{match.group(0)}", pytrace=False)
    target_met = match.group(4) == "True"
    if world_size != expected_world_size:
        pytest.fail(
            f"propagation metric reported world_size={world_size}, expected {expected_world_size}",
            pytrace=False,
        )
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        pytest.fail(f"invalid elapsed_ms={elapsed_ms}", pytrace=False)
    if not math.isfinite(target_ms) or target_ms != 100.0:
        pytest.fail(f"unexpected target_ms={target_ms}, expected 100", pytrace=False)
    if target_met is not (elapsed_ms < target_ms):
        pytest.fail(
            f"target_met={target_met} is inconsistent with "
            f"elapsed_ms={elapsed_ms} and target_ms={target_ms}",
            pytrace=False,
        )
    return elapsed_ms, target_met


def _validate_propagation_budget(output: str, expected_world_size: int) -> float:
    """Require the real-MPI logical-failure fanout to meet its 100 ms budget."""
    elapsed_ms, target_met = _parse_propagation_metric(output, expected_world_size)
    if not target_met:
        pytest.fail(
            f"WideEP MPI propagation exceeded the <100 ms budget: elapsed_ms={elapsed_ms}",
            pytrace=False,
        )
    return elapsed_ms


def _kill_and_reap_launcher(process: _ReapableProcess, timeout: float) -> bool:
    """Kill the launcher and make one bounded attempt to reap it."""
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, True),
        ({_ALLOW_SKIP_ENV: "0"}, True),
        ({_ALLOW_SKIP_ENV: "1"}, False),
    ],
)
def test_mpi_ft_smoke_required_mode(environment: dict[str, str], expected: bool) -> None:
    assert _smoke_is_required(environment) is expected


def test_mpi_ft_smoke_missing_prerequisite_is_optional_locally() -> None:
    with pytest.raises(pytest.skip.Exception, match="missing launcher"):
        _handle_missing_prerequisite("missing launcher", required=False)


def test_mpi_ft_smoke_missing_prerequisite_fails_when_required() -> None:
    with pytest.raises(pytest.fail.Exception, match="required.*missing launcher"):
        _handle_missing_prerequisite("missing launcher", required=True)


def test_mpi_ft_smoke_launcher_failure_precedes_skip_marker() -> None:
    with pytest.raises(pytest.fail.Exception, match="launcher failed"):
        _validate_worker_result(
            1,
            f"launcher failed\n{_SKIP_MARKER} unsupported thread level",
            required=False,
        )


def test_mpi_ft_smoke_worker_skip_fails_when_required() -> None:
    with pytest.raises(pytest.fail.Exception, match="worker skipped"):
        _validate_worker_result(
            0,
            f"{_SKIP_MARKER} unsupported thread level",
            required=True,
        )


def test_mpi_ft_smoke_worker_skip_is_optional_when_opted_out() -> None:
    with pytest.raises(pytest.skip.Exception, match="unsupported thread level"):
        _validate_worker_result(
            0,
            f"{_SKIP_MARKER} unsupported thread level",
            required=False,
        )


def _terminal_completion_output(world_size: int) -> str:
    ready = [
        f"{_TERMINAL_READY_MARKER} rank={rank} world_size={world_size}"
        for rank in range(world_size)
    ]
    done = [
        f"{_TERMINAL_DONE_MARKER} rank={rank} world_size={world_size}" for rank in range(world_size)
    ]
    return "\n".join([*ready, f"{_TERMINAL_COMPLETE_MARKER} world_size={world_size}", *done])


@pytest.mark.parametrize("returncode", [0, 1, 137], ids=["zero", "open-mpi", "signal-style"])
def test_mpi_ft_terminal_result_accepts_complete_evidence(returncode: int) -> None:
    _validate_worker_result(
        returncode,
        _terminal_completion_output(4),
        required=True,
        mode=_TERMINAL_MODE,
        expected_world_size=4,
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "\n".join(
            [
                f"{_TERMINAL_READY_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_COMPLETE_MARKER} world_size=2",
            ]
        ),
        "\n".join(
            [
                f"{_TERMINAL_READY_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_READY_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_READY_MARKER} rank=1 world_size=2",
                f"{_TERMINAL_COMPLETE_MARKER} world_size=2",
            ]
        ),
        "\n".join(
            [
                f"{_TERMINAL_READY_MARKER} rank=0 world_size=3",
                f"{_TERMINAL_READY_MARKER} rank=1 world_size=3",
                f"{_TERMINAL_COMPLETE_MARKER} world_size=3",
            ]
        ),
    ],
    ids=["missing", "incomplete", "duplicate", "wrong-world-size"],
)
def test_mpi_ft_terminal_result_rejects_incomplete_rank_set(output: str) -> None:
    with pytest.raises(pytest.fail.Exception, match="expected exactly one"):
        _validate_worker_result(
            1,
            output,
            required=True,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


@pytest.mark.parametrize(
    "completion_suffix",
    [
        "",
        "\n".join(
            [
                f"{_TERMINAL_COMPLETE_MARKER} world_size=2",
                f"{_TERMINAL_COMPLETE_MARKER} world_size=2",
            ]
        ),
        f"{_TERMINAL_COMPLETE_MARKER} world_size=3",
        f"{_TERMINAL_COMPLETE_MARKER} world_size=2 trailing-garbage",
        f"{_TERMINAL_COMPLETE_MARKER} world_size=2\n{_TERMINAL_ERROR_MARKER} rank=0",
    ],
    ids=["missing", "duplicate", "wrong-world-size", "malformed", "worker-error"],
)
def test_mpi_ft_terminal_result_rejects_invalid_global_completion(
    completion_suffix: str,
) -> None:
    ready = "\n".join(f"{_TERMINAL_READY_MARKER} rank={rank} world_size=2" for rank in range(2))
    done = "\n".join(f"{_TERMINAL_DONE_MARKER} rank={rank} world_size=2" for rank in range(2))
    output = "\n".join(part for part in (ready, completion_suffix, done) if part)
    with pytest.raises(pytest.fail.Exception):
        _validate_worker_result(
            1,
            output,
            required=True,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


def test_mpi_ft_terminal_result_rejects_crash_before_done() -> None:
    ready_and_complete = "\n".join(
        [
            *(f"{_TERMINAL_READY_MARKER} rank={rank} world_size=2" for rank in range(2)),
            f"{_TERMINAL_COMPLETE_MARKER} world_size=2",
        ]
    )
    with pytest.raises(pytest.fail.Exception, match=_TERMINAL_DONE_MARKER):
        _validate_worker_result(
            137,
            ready_and_complete,
            required=True,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


@pytest.mark.parametrize(
    "done_suffix",
    [
        f"{_TERMINAL_DONE_MARKER} rank=0 world_size=2",
        "\n".join(
            [
                f"{_TERMINAL_DONE_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_DONE_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_DONE_MARKER} rank=1 world_size=2",
            ]
        ),
        "\n".join(
            [
                f"{_TERMINAL_DONE_MARKER} rank=0 world_size=3",
                f"{_TERMINAL_DONE_MARKER} rank=1 world_size=3",
            ]
        ),
        "\n".join(
            [
                f"{_TERMINAL_DONE_MARKER} rank=0 world_size=2",
                f"{_TERMINAL_DONE_MARKER} rank=1 world_size=2 trailing-garbage",
            ]
        ),
    ],
    ids=["incomplete", "duplicate", "wrong-world-size", "malformed"],
)
def test_mpi_ft_terminal_result_rejects_invalid_done_set(done_suffix: str) -> None:
    ready = "\n".join(f"{_TERMINAL_READY_MARKER} rank={rank} world_size=2" for rank in range(2))
    output = "\n".join([ready, f"{_TERMINAL_COMPLETE_MARKER} world_size=2", done_suffix])
    with pytest.raises(pytest.fail.Exception, match=_TERMINAL_DONE_MARKER):
        _validate_worker_result(
            1,
            output,
            required=True,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


def test_mpi_ft_terminal_result_accepts_launcher_prefixes_and_noise() -> None:
    output = "\n".join(
        [
            "unrelated launcher diagnostic",
            *(f"[rank0]<stdout>: {line}" for line in _terminal_completion_output(2).splitlines()),
            "another unrelated diagnostic",
        ]
    )

    _validate_worker_result(
        1,
        output,
        required=True,
        mode=_TERMINAL_MODE,
        expected_world_size=2,
    )


def test_mpi_ft_terminal_result_rejects_python_atexit_evidence() -> None:
    output = "\n".join(
        [
            _terminal_completion_output(2),
            f"{_TERMINAL_ATEXIT_MARKER} pid=123",
        ]
    )

    with pytest.raises(pytest.fail.Exception, match="atexit/mpi4py Finalize"):
        _validate_worker_result(
            0,
            output,
            required=True,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


def test_mpi_ft_terminal_launcher_failure_precedes_skip_marker() -> None:
    with pytest.raises(pytest.fail.Exception, match="launcher failed"):
        _validate_worker_result(
            1,
            f"launcher failed\n{_SKIP_MARKER} unsupported thread level",
            required=False,
            mode=_TERMINAL_MODE,
            expected_world_size=2,
        )


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ((None, "complete"), "complete"),
        ((b"partial ", b"partial complete"), "partial complete"),
        (("first\n", "second\n"), "first\nsecond\n"),
        ((b"bad-\xff",), "bad-\ufffd"),
    ],
    ids=["none", "cumulative", "incremental", "bytes-replacement"],
)
def test_merge_subprocess_output_preserves_timeout_diagnostics(
    outputs: tuple[str | bytes | None, ...], expected: str
) -> None:
    assert _merge_subprocess_output(*outputs) == expected


def test_parse_propagation_metric_validates_target_without_requiring_it() -> None:
    elapsed_ms, target_met = _parse_propagation_metric(
        f"{_PROPAGATION_MARKER} world_size=4 elapsed_ms=125.5 target_ms=100.0 target_met=False",
        4,
    )

    assert elapsed_ms == 125.5
    assert target_met is False


def test_parse_propagation_metric_accepts_launcher_prefix_and_noise() -> None:
    metric = f"{_PROPAGATION_MARKER} world_size=4 elapsed_ms=25.5 target_ms=100.0 target_met=True"
    output = f"launcher diagnostic\n[rank0]<stdout>: {metric}\nmore noise"

    assert _parse_propagation_metric(output, 4) == (25.5, True)


def test_validate_propagation_budget_rejects_semantically_valid_miss() -> None:
    metric = f"{_PROPAGATION_MARKER} world_size=4 elapsed_ms=125.5 target_ms=100.0 target_met=False"
    assert _parse_propagation_metric(metric, 4) == (125.5, False)

    with pytest.raises(pytest.fail.Exception, match="exceeded the <100 ms budget"):
        _validate_propagation_budget(metric, 4)


@pytest.mark.parametrize("wait_times_out", [False, True], ids=["reaped", "still-running"])
def test_kill_and_reap_launcher_is_bounded(wait_times_out: bool) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.kill_calls = 0
            self.wait_timeouts: list[float] = []

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            self.wait_timeouts.append(timeout)
            if wait_times_out:
                raise subprocess.TimeoutExpired("mpiexec", timeout)
            return -signal.SIGKILL

    process = FakeProcess()
    assert _kill_and_reap_launcher(process, 0.25) is (not wait_times_out)
    assert process.kill_calls == 1
    assert process.wait_timeouts == [0.25]


@pytest.mark.parametrize(
    "metric",
    [
        "",
        f"{_PROPAGATION_MARKER} world_size=2 elapsed_ms=nan target_ms=100 target_met=False",
        f"{_PROPAGATION_MARKER} world_size=2 elapsed_ms=-1 target_ms=100 target_met=True",
        f"{_PROPAGATION_MARKER} world_size=2 elapsed_ms=1 target_ms=99 target_met=True",
        f"{_PROPAGATION_MARKER} world_size=2 elapsed_ms=101 target_ms=100 target_met=True",
        f"{_PROPAGATION_MARKER} world_size=2 elapsed_ms=1 target_ms=100 target_met=True trailing",
    ],
)
def test_parse_propagation_metric_rejects_malformed_semantics(metric: str) -> None:
    with pytest.raises(pytest.fail.Exception):
        _parse_propagation_metric(metric, 2)


@pytest.mark.parametrize("world_size", [2, 4], ids=["2-ranks", "4-ranks"])
def test_ep_failure_broadcast_real_mpi(world_size: int) -> None:
    """Exercise logical failure fanout and shutdown over a real MPI transport.

    The victim remains alive for portable ``COMM_WORLD`` result coordination;
    actual process death and MPI error classification are outside this smoke.
    """
    required = _smoke_is_required()
    if importlib.util.find_spec("mpi4py") is None:
        _handle_missing_prerequisite("mpi4py is not installed", required=required)
    launcher = _mpi_launcher()
    if launcher is None:
        _handle_missing_prerequisite("mpiexec/mpirun is not installed", required=required)

    worker = Path(__file__).with_name("_ep_failure_broadcast_mpi_worker.py")
    environment = os.environ.copy()
    # Open MPI requires explicit opt-in when CI runs inside a root container.
    # Other MPI implementations ignore these variables.
    environment["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    environment["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    environment["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if not required:
        environment[_ALLOW_SKIP_ENV] = "1"

    outputs: dict[str, str] = {}
    for mode in (_HEALTHY_MODE, _TERMINAL_MODE):
        command = [
            launcher,
            "-n",
            str(world_size),
            sys.executable,
            str(worker),
            str(world_size),
            mode,
        ]
        process = subprocess.Popen(  # nosec B603
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            output, _ = process.communicate(timeout=_WORKER_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as timeout_error:
            partial_output = _subprocess_output_text(timeout_error.output)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # mpiexec may exit between communicate() timing out and cleanup.
                pass
            try:
                output, _ = process.communicate(timeout=_REAP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired as reap_error:
                output = _merge_subprocess_output(partial_output, reap_error.output)
                reaped_after_kill = _kill_and_reap_launcher(process, _REAP_TIMEOUT_SEC)
                if process.stdout is not None:
                    try:
                        process.stdout.close()
                    except OSError:
                        pass
                reap_status = (
                    "launcher was reaped after direct SIGKILL"
                    if reaped_after_kill
                    else "launcher still could not be reaped after direct SIGKILL"
                )
                pytest.fail(
                    f"{world_size}-rank WideEP MPI {mode} smoke timed out after "
                    f"{_WORKER_TIMEOUT_SEC:.0f}s and could not be reaped within "
                    f"{_REAP_TIMEOUT_SEC:.0f}s; {reap_status}\n{output}"
                )
            output = _merge_subprocess_output(partial_output, output)
            pytest.fail(
                f"{world_size}-rank WideEP MPI {mode} smoke timed out after "
                f"{_WORKER_TIMEOUT_SEC:.0f}s\n{output}"
            )

        _validate_worker_result(
            process.returncode,
            output,
            required=required,
            mode=mode,
            expected_world_size=world_size,
        )
        outputs[mode] = output

    _validate_propagation_budget(outputs[_TERMINAL_MODE], world_size)
    assert _INVALID_TOPOLOGY_MARKER in outputs[_HEALTHY_MODE]
    assert _HEALTHY_LIFECYCLE_MARKER in outputs[_HEALTHY_MODE]
    assert _ABORT_WORLD_MARKER in outputs[_TERMINAL_MODE]
