# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Integration test for the KV cache transceiver harness.

Drives ``run_cache_transceiver_test.py`` on a single node by launching two
independent ``mpirun`` subprocesses (ctx and gen), each forming its own MPI
world of size 2 (TP=2). All processes share the same physical GPU.

Requires: 1 GPU, mpirun, mpi4py, tensorrt_llm.
"""

import ast
import json
import os
import pickle
import shutil
import signal
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from unittest.mock import MagicMock, call

import pytest

CTT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "examples",
        "disaggregated",
        "slurm",
        "cache_transceiver_test",
    )
)

DRIVER_SCRIPT = os.path.join(CTT_DIR, "run_cache_transceiver_test.py")
REPORT_SCRIPT = os.path.join(CTT_DIR, "report.py")
_LOG_TAIL_CHARS = 16 * 1024
_PROCESS_TIMEOUT_SECONDS = 120
_TERMINATE_GRACE_SECONDS = 5


def _load_driver_subset(
    module_name: str,
    selected_names: set[str],
    stubs: dict[str, Any],
) -> types.ModuleType:
    """Execute selected driver definitions with injected runtime stand-ins."""
    source = Path(DRIVER_SCRIPT).read_text()
    tree = ast.parse(source, filename=DRIVER_SCRIPT)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected_names
    ]
    module = types.ModuleType(module_name)
    module.__dict__.update(stubs)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), DRIVER_SCRIPT, "exec"),
        module.__dict__,
    )
    missing = selected_names - set(module.__dict__)
    assert not missing, f"{module_name}: driver definitions not loaded: {sorted(missing)}"
    return module


def _load_driver_ownership_helpers() -> types.ModuleType:
    """Load pure ownership helpers without importing GPU/MPI runtime modules."""
    selected_names = {
        "_Timeout",
        "_TransferError",
        "_FatalTransferError",
        "_request_ids",
        "_context_completion_error",
        "_gen_completion_error",
        "_can_release_sequence",
        "_release_sequence_if_safe",
        "_validate_context_completion",
        "_validate_python_gen_completion",
        "_first_reason",
        "_exchange_release_decision",
        "_hard_abort_process",
    }
    return _load_driver_subset(
        "cache_transceiver_harness_ownership",
        selected_names,
        {
            "Any": Any,
            "Iterable": Iterable,
            "Optional": Optional,
            "Sequence": Sequence,
            "MPI": types.SimpleNamespace(MIN="MIN"),
            "LlmRequestState": types.SimpleNamespace(
                DISAGG_GENERATION_TRANS_COMPLETE="gen_complete",
                DISAGG_TRANS_ERROR="error",
            ),
            "free_sequence": MagicMock(),
            "os": os,
            "pickle": pickle,
            "signal": signal,
        },
    )


OWNERSHIP = _load_driver_ownership_helpers()


def _load_driver_request_flow() -> types.ModuleType:
    """Load the request flow with local stand-ins for GPU/MPI dependencies."""
    selected_names = {
        "_Timeout",
        "_TransferError",
        "_FatalTransferError",
        "_request_ids",
        "_context_completion_error",
        "_gen_completion_error",
        "_can_release_sequence",
        "_release_sequence_if_safe",
        "_validate_context_completion",
        "_validate_python_gen_completion",
        "_first_reason",
        "_exchange_release_decision",
        "_wait_gen_complete",
        "run_one_request",
    }
    return _load_driver_subset(
        "cache_transceiver_harness_request_flow",
        selected_names,
        {
            "Any": Any,
            "Iterable": Iterable,
            "Optional": Optional,
            "Sequence": Sequence,
            "MPI": types.SimpleNamespace(MAX="MAX", MIN="MIN"),
            "LlmRequest": Any,
            "LlmRequestState": types.SimpleNamespace(
                DISAGG_GENERATION_TRANS_COMPLETE="gen_complete",
                DISAGG_TRANS_ERROR="error",
            ),
            "add_sequence": MagicMock(),
            "fill_request": MagicMock(),
            "free_sequence": MagicMock(),
            "make_request": MagicMock(),
            "pickle": pickle,
            "tensorrt_llm": types.SimpleNamespace(logger=MagicMock()),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=MagicMock())),
            "verify_request": MagicMock(),
        },
    )


REQUEST_FLOW = _load_driver_request_flow()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _find_mpirun():
    path = shutil.which("mpirun")
    if path is None:
        pytest.skip("mpirun not found on PATH")
    return path


def _log_tail(log: str) -> str:
    if len(log) <= _LOG_TAIL_CHARS:
        return log
    omitted = len(log) - _LOG_TAIL_CHARS
    return f"... {omitted} earlier characters omitted ...\n{log[-_LOG_TAIL_CHARS:]}"


def _wait_for_processes(processes: list[subprocess.Popen]) -> None:
    deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.wait(timeout=remaining)


def _terminate_process_groups(processes: list[subprocess.Popen], group_ids: list[int]) -> None:
    for group_id in group_ids:
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    for group_id in group_ids:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass


class TestProcessHelpers:
    def test_wait_for_processes_uses_shared_deadline(self, monkeypatch):
        first = MagicMock()
        second = MagicMock()
        monotonic = MagicMock(side_effect=[100.0, 101.0, 111.0])
        monkeypatch.setattr(time, "monotonic", monotonic)

        _wait_for_processes([first, second])

        first.wait.assert_called_once_with(timeout=119.0)
        second.wait.assert_called_once_with(timeout=109.0)

    def test_terminate_process_groups_bounds_post_kill_wait(self, monkeypatch):
        proc = MagicMock(pid=1234)
        proc.poll.return_value = None
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("mpirun", 4.0),
            subprocess.TimeoutExpired("mpirun", 4.0),
        ]
        killpg = MagicMock()
        monotonic = MagicMock(side_effect=[100.0, 101.0, 200.0, 201.0])
        monkeypatch.setattr(os, "killpg", killpg)
        monkeypatch.setattr(time, "monotonic", monotonic)

        _terminate_process_groups([proc], [proc.pid])

        assert proc.wait.call_args_list == [call(timeout=4.0), call(timeout=4.0)]
        assert killpg.call_args_list == [
            call(proc.pid, signal.SIGTERM),
            call(proc.pid, signal.SIGKILL),
        ]

    def test_terminate_process_groups_signals_group_after_leader_exit(self, monkeypatch):
        proc = MagicMock(pid=1234)
        proc.poll.return_value = 1
        killpg = MagicMock()
        monkeypatch.setattr(os, "killpg", killpg)

        _terminate_process_groups([proc], [proc.pid])

        assert killpg.call_args_list == [
            call(proc.pid, signal.SIGTERM),
            call(proc.pid, signal.SIGKILL),
        ]
        proc.wait.assert_not_called()


class _FakeDecisionComm:
    def __init__(self, *, reduced: Optional[int] = None) -> None:
        self.reduced = reduced
        self.abort_calls = []

    def allreduce(self, value, op=None):
        del op
        return value if self.reduced is None else self.reduced

    def gather(self, value, root=0):
        del root
        return [value]

    def bcast(self, value, root=0):
        del root
        return value

    def Abort(self, code):
        self.abort_calls.append(code)


class _FakeDecisionSocket:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.sent = []

    def send(self, payload) -> None:
        self.sent.append(pickle.loads(payload))

    def recv(self):
        return pickle.dumps(self.responses.pop(0))


class _FakeRunSocket:
    def __init__(self, responses, events) -> None:
        self.responses = list(responses)
        self.events = events
        self.sent = []
        self.recv_count = 0

    def send(self, payload) -> None:
        self.sent.append(payload if payload == b"go" else pickle.loads(payload))

    def recv(self):
        self.recv_count += 1
        if self.recv_count == 2:
            self.events.append("peer_ack")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, bytes):
            return response
        return pickle.dumps(response)


class _FakeGenTransceiver:
    def __init__(self, status, events, receive_error=None) -> None:
        self.status = status
        self.events = events
        self.receive_error = receive_error

    def request_and_receive_async(self, req) -> None:
        del req
        if self.receive_error is not None:
            raise self.receive_error

    def check_gen_transfer_status(self, timeout):
        assert timeout is None
        self.events.append("block_all")
        if isinstance(self.status, BaseException):
            raise self.status
        return self.status


class _FakeContextTransceiver:
    def __init__(self, send_error) -> None:
        self.send_error = send_error

    def respond_and_send_async(self, req) -> None:
        del req
        raise self.send_error


def _run_gen_request(
    status,
    state,
    *,
    final_response=("COMPLETE", ""),
    receive_error=None,
    synchronize_error=None,
):
    events = []
    req = types.SimpleNamespace(py_request_id=7, state=state)
    comm = _FakeDecisionComm()
    sock = _FakeRunSocket(
        [("OK", types.SimpleNamespace()), final_response],
        events,
    )
    xcvr = _FakeGenTransceiver(status, events, receive_error=receive_error)

    REQUEST_FLOW.make_request.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.make_request.return_value = req
    REQUEST_FLOW.add_sequence.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.add_sequence.return_value = "handle"
    REQUEST_FLOW.fill_request.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.free_sequence.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.free_sequence.side_effect = lambda *args: events.append("free")
    REQUEST_FLOW.verify_request.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.verify_request.return_value = True
    REQUEST_FLOW.torch.cuda.synchronize.reset_mock(return_value=True, side_effect=True)

    def synchronize():
        events.append("cuda_sync")
        if synchronize_error is not None:
            raise synchronize_error

    REQUEST_FLOW.torch.cuda.synchronize.side_effect = synchronize
    result = REQUEST_FLOW.run_one_request(
        "gen",
        comm,
        "manager",
        xcvr,
        "PYTHON",
        True,
        2,
        7,
        16,
        0,
        sock,
    )
    return result, events, sock


def _run_ctx_request(send_error):
    events = []
    req = types.SimpleNamespace(
        py_request_id=7,
        state="in_progress",
        context_phase_params=types.SimpleNamespace(),
    )
    comm = _FakeDecisionComm()
    sock = _FakeRunSocket([b"go"], events)
    xcvr = _FakeContextTransceiver(send_error)

    REQUEST_FLOW.make_request.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.make_request.return_value = req
    REQUEST_FLOW.add_sequence.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.add_sequence.return_value = "handle"
    REQUEST_FLOW.fill_request.reset_mock(return_value=True, side_effect=True)
    REQUEST_FLOW.free_sequence.reset_mock(return_value=True, side_effect=True)
    return REQUEST_FLOW.run_one_request(
        "ctx",
        comm,
        "manager",
        xcvr,
        "PYTHON",
        True,
        2,
        7,
        16,
        0,
        sock,
    )


class TestRequestTransferOwnershipFlow:
    @pytest.mark.parametrize(
        ("status", "state"),
        [
            (([], [], []), "gen_complete"),
            (([], [7], []), "error"),
            (([], [], [7]), "in_progress"),
            (([7], [], []), "in_progress"),
        ],
        ids=("missing", "failed", "cancelled", "nonterminal"),
    )
    def test_python_gen_unproven_completion_never_frees(self, status, state):
        with pytest.raises(REQUEST_FLOW._FatalTransferError):
            _run_gen_request(status, state, final_response=("FATAL", "gen unsafe"))

        REQUEST_FLOW.free_sequence.assert_not_called()

    def test_cuda_synchronize_error_before_release_handshake_never_frees(self):
        with pytest.raises(REQUEST_FLOW._FatalTransferError):
            _run_gen_request(
                ([7], [], []),
                "gen_complete",
                final_response=("FATAL", "gen cuda synchronize failed"),
                synchronize_error=RuntimeError("CUDA synchronize failed"),
            )

        REQUEST_FLOW.free_sequence.assert_not_called()

    def test_gen_partial_dispatch_setup_error_is_fatal_and_never_frees(self):
        with pytest.raises(REQUEST_FLOW._FatalTransferError, match="partial gen dispatch"):
            _run_gen_request(
                ([7], [], []),
                "gen_complete",
                receive_error=RuntimeError("partial gen dispatch"),
            )

        REQUEST_FLOW.free_sequence.assert_not_called()

    def test_ctx_partial_dispatch_setup_error_is_fatal_and_never_frees(self):
        with pytest.raises(REQUEST_FLOW._FatalTransferError, match="partial ctx dispatch"):
            _run_ctx_request(RuntimeError("partial ctx dispatch"))

        REQUEST_FLOW.free_sequence.assert_not_called()

    @pytest.mark.parametrize("operation", ("receive", "block_all", "cuda_sync"))
    def test_timeout_from_caught_driver_operation_propagates(self, operation):
        timeout = REQUEST_FLOW._Timeout("cell deadline expired")
        kwargs = {}
        status = ([7], [], [])
        if operation == "receive":
            kwargs["receive_error"] = timeout
        elif operation == "block_all":
            status = timeout
        else:
            kwargs["synchronize_error"] = timeout

        with pytest.raises(REQUEST_FLOW._Timeout, match="cell deadline expired"):
            _run_gen_request(status, "gen_complete", **kwargs)

        REQUEST_FLOW.free_sequence.assert_not_called()

    def test_gen_frees_only_after_completion_sync_and_both_role_ack(self):
        result, events, sock = _run_gen_request(([7], [], []), "gen_complete")

        assert result is True
        assert events == ["block_all", "cuda_sync", "peer_ack", "free"]
        assert sock.sent == [b"go", ("COMPLETE", "")]
        REQUEST_FLOW.free_sequence.assert_called_once_with(
            "manager",
            REQUEST_FLOW.make_request.return_value,
            "handle",
            True,
        )

    def test_context_fatal_ack_never_frees_completed_gen_sequence(self):
        with pytest.raises(REQUEST_FLOW._FatalTransferError, match="ctx unsafe"):
            _run_gen_request(
                ([7], [], []),
                "gen_complete",
                final_response=("FATAL", "ctx unsafe"),
            )

        REQUEST_FLOW.free_sequence.assert_not_called()


class TestTransferOwnershipHelpers:
    def test_context_requires_exact_completed_request(self):
        assert OWNERSHIP._context_completion_error(7, [7], [], "in_progress", "error") is None
        assert "failed" in OWNERSHIP._context_completion_error(7, [], [7], "error", "error")
        assert "without completing" in OWNERSHIP._context_completion_error(
            7, [8], [], "in_progress", "error"
        )

    @pytest.mark.parametrize(
        ("completed", "failed", "cancelled", "state", "expected"),
        [
            ([7], [], [], "gen_complete", None),
            ([], [7], [], "error", "failed"),
            ([], [], [types.SimpleNamespace(py_request_id=7)], "in_progress", "cancelled"),
            ([], [], [], "in_progress", "without completing"),
            ([7], [], [], "in_progress", "nonterminal"),
        ],
    )
    def test_generation_requires_completed_terminal_request(
        self, completed, failed, cancelled, state, expected
    ):
        error = OWNERSHIP._gen_completion_error(
            7,
            completed,
            failed,
            cancelled,
            state,
            "gen_complete",
            "error",
        )
        if expected is None:
            assert error is None
        else:
            assert expected in error

    @pytest.mark.parametrize(
        ("started", "completed", "should_release"),
        [
            (False, False, True),
            (True, False, False),
            (True, True, True),
        ],
    )
    def test_release_sequence_requires_quiescence(self, started, completed, should_release):
        OWNERSHIP.free_sequence.reset_mock()
        released = OWNERSHIP._release_sequence_if_safe(
            "manager",
            types.SimpleNamespace(py_request_id=7),
            "handle",
            True,
            transfer_may_have_started=started,
            transfer_completed=completed,
        )

        assert released is should_release
        assert OWNERSHIP.free_sequence.call_count == int(should_release)

    def test_gen_release_handshake_requires_context_acknowledgement(self):
        socket = _FakeDecisionSocket(("COMPLETE", ""))

        safe, reason = OWNERSHIP._exchange_release_decision(
            "gen", _FakeDecisionComm(), True, socket, True
        )

        assert safe and reason == ""
        assert socket.sent == [("COMPLETE", "")]

    def test_context_release_handshake_propagates_local_timeout(self):
        socket = _FakeDecisionSocket(("COMPLETE", ""))

        safe, reason = OWNERSHIP._exchange_release_decision(
            "ctx",
            _FakeDecisionComm(reduced=0),
            True,
            socket,
            False,
            "sender deadline expired",
        )

        assert not safe
        assert reason == "sender deadline expired"
        assert socket.sent == [("FATAL", "sender deadline expired")]

    def test_release_handshake_rejects_invalid_peer_status(self):
        socket = _FakeDecisionSocket(("UNKNOWN", ""))

        safe, reason = OWNERSHIP._exchange_release_decision(
            "gen", _FakeDecisionComm(), True, socket, True
        )

        assert not safe
        assert "invalid" in reason

    def test_context_release_handshake_sends_invalid_peer_verdict(self):
        socket = _FakeDecisionSocket(("UNKNOWN", ""))

        safe, reason = OWNERSHIP._exchange_release_decision(
            "ctx", _FakeDecisionComm(), True, socket, True
        )

        expected = "invalid gen peer release status: 'UNKNOWN'"
        assert not safe
        assert reason == expected
        assert socket.sent == [("FATAL", expected)]

    def test_release_handshake_preserves_timeout(self):
        socket = _FakeDecisionSocket()
        socket.recv = MagicMock(side_effect=OWNERSHIP._Timeout("release deadline expired"))

        with pytest.raises(OWNERSHIP._Timeout, match="release deadline expired"):
            OWNERSHIP._exchange_release_decision("gen", _FakeDecisionComm(), True, socket, True)

    def test_hard_abort_falls_back_to_sigkill(self, monkeypatch):
        comm = _FakeDecisionComm()
        kill = MagicMock()
        monkeypatch.setattr(OWNERSHIP.os, "kill", kill)

        with pytest.raises(RuntimeError, match="SIGKILL unexpectedly returned"):
            OWNERSHIP._hard_abort_process(comm)

        assert comm.abort_calls == [137]
        kill.assert_called_once_with(os.getpid(), signal.SIGKILL)


def _build_config(work_dir: str) -> dict:
    return {
        "hardware": {"gpus_per_node": 2},
        "environment": {
            "container_image": "",
            "work_dir": work_dir,
        },
        "test_matrix": {
            "combinations": [
                {"backend": "NIXL", "runtime": "PYTHON"},
            ],
            "cache_manager_versions": ["V2"],
            "request_lengths": [100],
            "num_requests_per_length": 8,
            "warmup_requests": 1,
        },
        "kv_cache": {
            "num_layers": 4,
            "num_kv_heads": 2,
            "head_dim": 128,
            "tokens_per_block": 8,
            "dtype": "HALF",
            "max_tokens_in_buffer": 256,
        },
        "parallel": {"ctx_tp": 2, "ctx_pp": 1, "gen_tp": 2, "gen_pp": 1},
        "ucx_env_sweep": [
            {"name": "default", "env": {"UCX_TLS": "all"}},
        ],
        "run": {
            "timeout_per_cell_s": 30,
            "max_sweep_s": 60,
        },
    }


@pytest.mark.timeout(180)
def test_single_node_transfer(tmp_path):
    """Launch ctx and gen via mpirun on a single node, verify transfer passes."""
    pytest.importorskip("mpi4py")
    mpirun = _find_mpirun()

    work_dir = str(tmp_path / "work")
    cfg = _build_config(work_dir)
    config_path = str(tmp_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    zmq_port = _find_free_port()

    env = os.environ.copy()
    env.update(
        {
            "CTT_CONFIG": config_path,
            "CTX_NODE": "127.0.0.1",
            "ZMQ_PORT": str(zmq_port),
            "CTT_SWEEP": "0",
            "CTT_SWEEP_NAME": "default",
            "CUDA_VISIBLE_DEVICES": "0",
            # Emit protocol-selection tables while transfers are active so the
            # report can deterministically identify the CUDA KV-data transport.
            "UCX_PROTO_INFO": "y",
        }
    )
    sweep_env = cfg["ucx_env_sweep"][0].get("env") or {}
    env.update({k: str(v) for k, v in sweep_env.items()})

    mpi_args = [
        mpirun,
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "2",
        sys.executable,
        DRIVER_SCRIPT,
    ]

    # Write directly to the filenames report.py consumes. Full UCX protocol
    # tables are verbose, so files also prevent either child from blocking on a
    # full stdout pipe while the other child is being drained.
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ctx_log_path = os.path.join(log_dir, "sweep0_ctx_rank0.log")
    gen_log_path = os.path.join(log_dir, "sweep0_gen_rank0.log")
    timeout_error = None
    processes = []
    process_group_ids = []
    processes_succeeded = False
    with open(ctx_log_path, "wb") as ctx_log_file, open(gen_log_path, "wb") as gen_log_file:
        try:
            ctx_proc = subprocess.Popen(
                mpi_args + ["--role", "ctx"],
                env=env,
                stdout=ctx_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(ctx_proc)
            process_group_ids.append(ctx_proc.pid)
            gen_proc = subprocess.Popen(
                mpi_args + ["--role", "gen"],
                env=env,
                stdout=gen_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(gen_proc)
            process_group_ids.append(gen_proc.pid)
            _wait_for_processes(processes)
            processes_succeeded = all(proc.returncode == 0 for proc in processes)
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
        finally:
            if not processes_succeeded:
                _terminate_process_groups(processes, process_group_ids)

    with open(ctx_log_path, errors="replace") as f:
        ctx_log = f.read()
    with open(gen_log_path, errors="replace") as f:
        gen_log = f.read()

    if timeout_error is not None:
        pytest.fail(
            f"mpirun timed out: {timeout_error}\n"
            f"ctx log tail:\n{_log_tail(ctx_log)}\n"
            f"gen log tail:\n{_log_tail(gen_log)}"
        )

    if ctx_proc.returncode != 0:
        pytest.fail(f"ctx mpirun failed (rc={ctx_proc.returncode}):\n{_log_tail(ctx_log)}")
    if gen_proc.returncode != 0:
        pytest.fail(f"gen mpirun failed (rc={gen_proc.returncode}):\n{_log_tail(gen_log)}")

    # Parse gen status JSONL and verify all entries are PASS.
    status_path = os.path.join(work_dir, "status", "sweep0_gen.jsonl")
    assert os.path.exists(status_path), (
        f"gen status file not found at {status_path}\ngen log tail:\n{_log_tail(gen_log)}"
    )

    with open(status_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0, "No status records found"
    for rec in records:
        assert rec["status"] == "PASS", (
            f"Expected PASS, got {rec['status']} "
            f"(combination_idx={rec.get('combination_idx')}, "
            f"reqlen_idx={rec.get('reqlen_idx')}, "
            f"reason={rec.get('reason', '')})\n"
            f"gen log tail:\n{_log_tail(gen_log)}"
        )

    # Verify report aggregation produces valid results.
    results_path = os.path.join(work_dir, "results.json")
    agg_result = subprocess.run(
        [
            sys.executable,
            REPORT_SCRIPT,
            config_path,
            "--aggregate",
            "--require-kv-transport",
            "--out",
            results_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert agg_result.returncode == 0, (
        f"report.py --aggregate failed (rc={agg_result.returncode}):\n"
        f"stdout: {agg_result.stdout}\nstderr: {agg_result.stderr}"
    )

    assert os.path.exists(results_path), "results.json was not created"
    with open(results_path) as f:
        results = json.load(f)

    assert "by_combination" in results
    assert len(results["by_combination"]) > 0
    for combo in results["by_combination"]:
        assert "combination" in combo
        assert "sweeps" in combo
        for sweep in combo["sweeps"]:
            assert sweep["status"] == "PASS", (
                f"Aggregated status not PASS for {combo['combination']} "
                f"sweep={sweep['sweep']}: {sweep['status']} "
                f"{sweep.get('error_detail', '')}"
            )
            assert sweep["selected_transport"], (
                f"selected_transport is empty for {combo['combination']} "
                f"sweep={sweep['sweep']} — UCX did not emit a parseable protocol table\n"
                f"ctx log tail:\n{_log_tail(ctx_log)}\n"
                f"gen log tail:\n{_log_tail(gen_log)}"
            )
            # Bandwidth must actually be parsed from the perf CSVs, not just
            # the transfer verified. The PYTHON+NIXL combination writes its
            # perf CSVs via PerfLogManager into TRTLLM_KVCACHE_TIME_OUTPUT_PATH
            # (set by the driver) as "<instanceUuid>_<rank>.csv"; a naming or
            # glob mismatch between perf_logger.py and report.py leaves
            # per_gpu_BW_GBps None while status stays PASS, so assert on it
            # explicitly.
            ctx_csv_dir = os.path.join(work_dir, "csv", "0", "ctx")
            csv_listing = os.listdir(ctx_csv_dir) if os.path.isdir(ctx_csv_dir) else []
            assert sweep["per_gpu_BW_GBps"] is not None and sweep["per_gpu_BW_GBps"] > 0, (
                f"per_gpu_BW_GBps missing for {combo['combination']} "
                f"sweep={sweep['sweep']} — perf CSVs were not parsed "
                f"(ctx csv dir contents: {csv_listing})"
            )
            assert sweep["num_samples"] > 0, (
                f"num_samples is 0 for {combo['combination']} sweep={sweep['sweep']} "
                f"(ctx csv dir contents: {csv_listing})"
            )

    best_path = os.path.splitext(results_path)[0] + ".best.json"
    assert os.path.exists(best_path), "results.best.json was not created"
