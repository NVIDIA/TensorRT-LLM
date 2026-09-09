# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MGMN launch mode (trtllm-llmapi-launch): detection, worker
task fan-out, env hygiene, fail-fast watchdog, and rendezvous agreement.

All tests run without GPU or a real MPI world — mpi4py, torch.distributed,
and the MPI session are mocked.
"""

import asyncio
import os
import sys
import threading
import types
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

import tensorrt_llm._torch.visual_gen.executor as executor_mod
import tensorrt_llm._torch.visual_gen.launch as launch_mod
from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionRemoteClient,
    DiffusionRequest,
    run_diffusion_worker,
)
from tensorrt_llm._torch.visual_gen.launch import LaunchMode, run_diffusion_worker_mgmn
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.visual_gen import VisualGen

_ENV_VARS = [
    "TLLM_SPAWN_PROXY_PROCESS",
    "tllm_mpi_size",
    "TLLM_DISABLE_MPI",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "GROUP_RANK",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TLLM_VG_MGMN_PG_TIMEOUT_SEC",
]


@pytest.fixture(autouse=True)
def _isolated_env():
    """Snapshot os.environ and drop launcher/rendezvous vars for each test.

    ``patch.dict`` restores the full environment afterwards, including any
    writes made by the code under test (MASTER_*, GROUP_RANK, ...).
    """
    with patch.dict(os.environ):
        for var in _ENV_VARS:
            os.environ.pop(var, None)
        yield


# =============================================================================
# mpi4py mocks — one simulated rank per call, MPI semantics over a shared wire
# =============================================================================


class _FakeLocalComm:
    """Result of COMM_WORLD.Split_type(COMM_TYPE_SHARED) for one rank."""

    def __init__(self, local_rank):
        self._local_rank = local_rank

    def Get_rank(self):
        return self._local_rank


class _FakeComm:
    """COMM_WORLD stand-in for one simulated rank.

    ``bcast`` emulates MPI semantics over a shared ``wire`` dict keyed by
    call index: the root's value is recorded (so the root rank must be
    simulated first, or the wire pre-populated) and non-root ranks read the
    recorded value, discarding their own input.
    """

    def __init__(self, rank, size, names, local_rank, wire):
        self._rank = rank
        self._size = size
        self._names = names
        self._local_rank = local_rank
        self._wire = wire
        self.bcast_calls = 0

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def Split_type(self, split_type):
        return _FakeLocalComm(self._local_rank)

    def allgather(self, value):
        return list(self._names)

    def bcast(self, value, root=0):
        idx = self.bcast_calls
        self.bcast_calls += 1
        if self._rank == root:
            self._wire[idx] = value
            return value
        assert idx in self._wire, "non-root bcast without a root value — would hang in real MPI"
        return self._wire[idx]


def _fake_mpi4py(comm):
    mpi = types.SimpleNamespace(
        COMM_WORLD=comm,
        COMM_TYPE_SHARED=object(),
        Get_processor_name=lambda: comm._names[comm._rank],
    )
    module = types.ModuleType("mpi4py")
    module.MPI = mpi
    return module


REQ_ADDR = "tcp://10.0.0.1:5555"
RESP_ADDR = "tcp://10.0.0.1:5556"


def _run_mgmn_rank(rank, world_size, names, local_rank, wire, worker_mock, task_world_size=None):
    """Run run_diffusion_worker_mgmn as one simulated rank with mocked mpi4py."""
    comm = _FakeComm(rank, world_size, names, local_rank, wire)
    with (
        patch.dict(sys.modules, {"mpi4py": _fake_mpi4py(comm)}),
        patch.object(executor_mod, "run_diffusion_worker", worker_mock),
    ):
        run_diffusion_worker_mgmn(
            world_size=task_world_size if task_world_size is not None else world_size,
            request_queue_addr=REQ_ADDR,
            response_queue_addr=RESP_ADDR,
            visual_gen_args=None,
            req_hmac_key=b"req-key",
            resp_hmac_key=b"resp-key",
        )
    return comm


# =============================================================================
# MGMN detection in DiffusionRemoteClient.__init__
# =============================================================================


class TestMgmnDetection:
    """Launch-mode selection: MGMN before external launch before spawn."""

    def _make_args(self):
        # cfg_size * ulysses_size = 4 workers
        return VisualGenArgs(
            model="/tmp/model",
            parallel_config={"cfg_size": 2, "ulysses_size": 2},
        )

    def _construct(self, args, mock_session):
        """Construct a client with spawn ctx and readiness mocked.

        The client's own background event-loop thread runs for real (with a
        stub serve loop, no IPC) because the constructor marshals the MGMN
        session dispatch onto that loop and blocks on it. Only the launch
        layer's seams are patched, so the client's thread is untouched.
        """
        mock_ctx = MagicMock()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_proc

        real_thread = threading.Thread

        def selective_thread(*t_args, **t_kwargs):
            # The spawner thread must really run so wait_for_spawn() completes;
            # the external-launch rank-0 worker must not start for real.
            if t_kwargs.get("name") == "visualgen-worker-process-spawner":
                return real_thread(*t_args, **t_kwargs)
            return MagicMock()

        async def stub_serve(client_self):
            while not client_self.shutdown_event.is_set():
                await asyncio.sleep(0.001)

        with (
            patch.object(
                launch_mod, "create_mpi_comm_session", return_value=mock_session
            ) as mock_create,
            patch.object(launch_mod, "_get_mp_context", return_value=mock_ctx),
            patch.object(launch_mod, "_Thread", side_effect=selective_thread) as mock_thread_cls,
            patch.object(DiffusionRemoteClient, "_serve_forever", stub_serve),
            patch.object(DiffusionRemoteClient, "_wait_ready"),
        ):
            client = DiffusionRemoteClient(args=args)
            # Stop the real event-loop thread before the patches lift.
            client.shutdown_event.set()
            client.background_thread.join(timeout=5.0)

        return client, {"create": mock_create, "ctx": mock_ctx, "threads": mock_thread_cls}

    def test_mgmn_branch_chosen(self):
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["tllm_mpi_size"] = "4"
        session = MagicMock()
        session.submit.return_value = []

        client, cap = self._construct(self._make_args(), session)

        cap["create"].assert_called_once_with(4)
        assert client.plan.mode is LaunchMode.MGMN
        assert client._workers.session is session
        cap["ctx"].Process.assert_not_called()

        session.submit.assert_called_once()
        submit_args, submit_kwargs = session.submit.call_args
        assert submit_args[0] is run_diffusion_worker_mgmn
        assert submit_kwargs["world_size"] == 4
        assert submit_kwargs["request_queue_addr"] == client.req_addr_connect
        assert submit_kwargs["response_queue_addr"] == client.resp_addr_connect
        assert submit_kwargs["req_hmac_key"] == client.req_hmac_key
        assert submit_kwargs["resp_hmac_key"] == client.resp_hmac_key

    def test_mgmn_submit_runs_on_event_loop_thread(self):
        # The session's ZMQ PAIR socket is single-thread-owned: the dispatch
        # submit must run on the same background event-loop thread that later
        # polls check_worker_error, not on the constructing thread.
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["tllm_mpi_size"] = "4"
        session = MagicMock()
        submit_threads = []

        def record_submit(*args, **kwargs):
            submit_threads.append(threading.get_ident())
            return []

        session.submit.side_effect = record_submit

        client, _ = self._construct(self._make_args(), session)

        assert submit_threads == [client.background_thread.ident]
        assert client.background_thread.ident != threading.get_ident()

    def test_mgmn_wins_over_stale_external_launch_env(self):
        # Leaked torchrun env without MASTER_ADDR makes _detect_external_launch
        # raise, so a successful construction proves the MGMN check runs first
        # and the external-launch detector is never consulted.
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["tllm_mpi_size"] = "4"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"
        session = MagicMock()
        session.submit.return_value = []

        client, cap = self._construct(self._make_args(), session)

        assert client.plan.mode is LaunchMode.MGMN
        assert client._workers.session is session
        cap["ctx"].Process.assert_not_called()

    def test_world_size_mismatch_raises_actionable(self):
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["tllm_mpi_size"] = "3"

        with pytest.raises(ValueError, match=r"world size \(3\) does not match") as excinfo:
            self._construct(self._make_args(), MagicMock())
        assert "trtllm-llmapi-launch" in str(excinfo.value)

    def test_no_launcher_env_falls_through_to_external_launch(self):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["MASTER_ADDR"] = "node0"
        os.environ["MASTER_PORT"] = "29500"

        client, cap = self._construct(self._make_args(), MagicMock())

        assert client.plan.mode is LaunchMode.EXTERNAL
        cap["create"].assert_not_called()
        cap["ctx"].Process.assert_not_called()
        # The rank-0 worker runs in a background thread in external mode.
        targets = [c.kwargs.get("target") for c in cap["threads"].call_args_list]
        assert run_diffusion_worker in targets

    def test_no_env_falls_through_to_spawn(self):
        client, cap = self._construct(self._make_args(), MagicMock())

        assert client.plan.mode is LaunchMode.SPAWN
        cap["create"].assert_not_called()
        assert cap["ctx"].Process.call_count == 4


# =============================================================================
# run_diffusion_worker_mgmn — per-rank argument fan-out
# =============================================================================


class TestMgmnWorkerFanout:
    """Rank/topology fan-out with mocked mpi4py — no real MPI."""

    def test_rank0_gets_ipc_endpoints_others_none(self):
        world = 4
        names = ["nodeA"] * world
        wire = {}
        calls = []
        for rank in range(world):
            worker = MagicMock()
            _run_mgmn_rank(rank, world, names, local_rank=rank, wire=wire, worker_mock=worker)
            calls.append(worker.call_args.kwargs)

        rank0 = calls[0]
        assert rank0["request_queue_addr"] == REQ_ADDR
        assert rank0["response_queue_addr"] == RESP_ADDR
        assert rank0["req_hmac_key"] == b"req-key"
        assert rank0["resp_hmac_key"] == b"resp-key"
        for kwargs in calls[1:]:
            assert kwargs["request_queue_addr"] is None
            assert kwargs["response_queue_addr"] is None
            assert kwargs["req_hmac_key"] is None
            assert kwargs["resp_hmac_key"] is None
        for rank, kwargs in enumerate(calls):
            assert kwargs["rank"] == rank
            assert kwargs["world_size"] == world
            assert kwargs["disable_mpi_env"] is False

    def test_local_rank_from_shared_comm_split(self):
        # Pre-populated wire stands in for rank 0's earlier broadcasts.
        wire = {0: "node0", 1: 29500}
        worker = MagicMock()

        _run_mgmn_rank(2, 4, ["a", "a", "b", "b"], local_rank=7, wire=wire, worker_mock=worker)

        assert worker.call_args.kwargs["local_rank"] == 7

    def test_group_rank_from_interleaved_processor_names(self):
        # First-occurrence ordering of unique names: nodeB -> 0, nodeA -> 1,
        # regardless of rank interleaving across the nodes.
        names = ["nodeB", "nodeA", "nodeB", "nodeA"]
        expected_host_ids = ["0", "1", "0", "1"]
        wire = {}
        for rank in range(4):
            worker = MagicMock()
            os.environ["GROUP_RANK"] = "stale"  # must be overwritten per rank
            _run_mgmn_rank(rank, 4, names, local_rank=rank // 2, wire=wire, worker_mock=worker)
            assert os.environ["GROUP_RANK"] == expected_host_ids[rank]

    def test_world_size_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="does not match"):
            _run_mgmn_rank(
                0, 4, ["a"] * 4, local_rank=0, wire={}, worker_mock=MagicMock(), task_world_size=8
            )


# =============================================================================
# Env hygiene — TLLM_DISABLE_MPI scoping
# =============================================================================


class TestEnvHygiene:
    def _run_worker(self, **kwargs):
        """Invoke run_diffusion_worker with all CUDA/dist calls mocked."""
        mock_exec = MagicMock()
        mock_exec.pipeline = None

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.destroy_process_group"),
            patch.object(executor_mod.DiffusionExecutor, "__new__", return_value=mock_exec),
            patch.object(executor_mod.DiffusionExecutor, "__init__", return_value=None),
        ):
            run_diffusion_worker(
                rank=0,
                world_size=1,
                master_addr="127.0.0.1",
                master_port=29500,
                request_queue_addr=None,
                response_queue_addr=None,
                visual_gen_args=None,
                **kwargs,
            )

    def test_mgmn_task_pops_tllm_disable_mpi_at_entry(self):
        os.environ["TLLM_DISABLE_MPI"] = "1"  # leaked from user env / prior task

        _run_mgmn_rank(0, 2, ["a", "a"], local_rank=0, wire={}, worker_mock=MagicMock())

        assert "TLLM_DISABLE_MPI" not in os.environ

    def test_disable_mpi_env_false_leaves_env_unset(self):
        self._run_worker(disable_mpi_env=False)

        assert "TLLM_DISABLE_MPI" not in os.environ

    def test_default_still_sets_tllm_disable_mpi(self):
        # Regression for spawn/external-launch modes.
        self._run_worker()

        assert os.environ.get("TLLM_DISABLE_MPI") == "1"


# =============================================================================
# Fail-fast — the check_worker_error watchdog
# =============================================================================


class TestFailFast:
    def _mgmn_client(self, session):
        """Construct a real client (live threads + ZMQ) in MGMN mode."""
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["tllm_mpi_size"] = "2"
        args = VisualGenArgs(model="/tmp/model", parallel_config={"cfg_size": 2})
        with patch.object(launch_mod, "create_mpi_comm_session", return_value=session):
            return DiffusionRemoteClient(args=args)

    def test_wait_ready_raises_on_injected_worker_error(self):
        session = MagicMock()
        session.submit.return_value = []
        session.check_worker_error.return_value = RuntimeError("rank 1 exploded")

        # The coordinator loop and the readiness wait are coroutines on one
        # event loop and both poll liveness, so either may observe the death
        # first. Both abort construction; only their wording differs.
        with pytest.raises(
            RuntimeError, match=r"Worker died during initialization|MGMN worker death reported"
        ):
            self._mgmn_client(session)

    def test_steady_state_monitor_fails_pending_requests(self):
        import zmq

        session = MagicMock()
        session.submit.return_value = []
        session.check_worker_error.return_value = None

        with (
            patch.object(launch_mod, "MGMN_ERROR_POLL_INTERVAL", 0.01),
            patch.object(DiffusionRemoteClient, "_wait_ready"),
        ):
            client = self._mgmn_client(session)
            # Stand-in for the rank-0 worker's request PULL socket: a PUSH
            # send is not writable with no connected peer, so give it one
            # (messages are never read — the "workers" are dead).
            pull = zmq.Context.instance().socket(zmq.PULL)
            pull.connect(client.req_addr_connect)
            try:
                client.enqueue_requests([DiffusionRequest(request_id=0, prompt=["hi"])])
                session.check_worker_error.return_value = RuntimeError("rank 1 exploded")

                response = client.await_responses_sync(0, timeout=5.0)
                assert response is not None
                assert response.error_msg is not None
                assert "rank 1 exploded" in response.error_msg

                with pytest.raises(RuntimeError, match="rank 1 exploded"):
                    client.enqueue_requests([DiffusionRequest(request_id=1, prompt=["hi"])])
            finally:
                client.shutdown()
                pull.close(0)

    def test_abort_signals_a_worker_group_it_cannot_kill(self):
        # MGMN workers live in launcher-owned processes with no handle to kill,
        # and the rank-0 worker blocks in a request recv that has no timeout,
        # so the shutdown sentinel is the only thing that ends it. Without it
        # that worker holds its GPU for the lifetime of the job.
        import pickle
        import time

        import zmq

        session = MagicMock()
        session.submit.return_value = []
        session.check_worker_error.return_value = None

        with (
            patch.object(launch_mod, "MGMN_ERROR_POLL_INTERVAL", 0.01),
            patch.object(DiffusionRemoteClient, "_wait_ready"),
        ):
            client = self._mgmn_client(session)
            # Stands in for the rank-0 worker's request PULL socket.
            pull = zmq.Context.instance().socket(zmq.PULL)
            pull.connect(client.req_addr_connect)
            try:
                deadline = time.time() + 5.0
                while client.requests_ipc is None and time.time() < deadline:
                    time.sleep(0.01)
                assert client.requests_ipc is not None

                session.check_worker_error.return_value = RuntimeError("rank 1 exploded")

                assert pull.poll(timeout=10_000), (
                    "the rank-0 worker was never told to exit; its request recv "
                    "has no timeout, so it would block forever holding its GPU"
                )
                # ZeroMqQueue frames the pickled payload followed by its HMAC;
                # pickle stops at the STOP opcode and ignores the suffix.
                assert pickle.loads(pull.recv()) is None
            finally:
                client.shutdown()
                pull.close(0)

    def test_worker_death_is_sticky_after_first_report(self):
        # check_worker_error is edge-triggered: it reports the death once and
        # returns None afterwards. The handle must latch it so the client's
        # every-pass liveness check keeps seeing the failure. It *returns* the
        # exception rather than raising, so return_value is the faithful stub.
        session = MagicMock()
        session.check_worker_error.return_value = RuntimeError("rank 1 exploded")
        handle = launch_mod._MgmnWorkers(session)

        first = handle.liveness_failure()
        assert first is not None and "rank 1 exploded" in first

        session.check_worker_error.return_value = None
        assert handle.liveness_failure() == first

    def test_worker_error_poll_failure_is_swallowed(self):
        # A session that raises must not kill the coordinator loop.
        session = MagicMock()
        session.check_worker_error.side_effect = RuntimeError("socket is gone")

        assert launch_mod._MgmnWorkers(session).liveness_failure() is None


# =============================================================================
# Rendezvous — unconditional, rank-0-authoritative bcast
# =============================================================================


class TestMgmnRendezvous:
    def test_bcast_unconditional_and_agreed(self):
        world = 4
        names = ["nodeA", "nodeA", "nodeB", "nodeB"]
        wire = {}
        addrs, ports, comms = [], [], []
        for rank in range(world):
            worker = MagicMock()
            comm = _run_mgmn_rank(
                rank, world, names, local_rank=rank % 2, wire=wire, worker_mock=worker
            )
            comms.append(comm)
            addrs.append(worker.call_args.kwargs["master_addr"])
            ports.append(worker.call_args.kwargs["master_port"])

        # Every rank participates in exactly two bcasts (addr + port), so the
        # collective can neither hang nor fork, and all ranks agree.
        assert [c.bcast_calls for c in comms] == [2] * world
        assert len(set(addrs)) == 1
        assert len(set(ports)) == 1

    def test_rank0_env_override_wins(self):
        os.environ["MASTER_ADDR"] = "override-host"
        os.environ["MASTER_PORT"] = "12345"
        worker = MagicMock()
        wire = {}

        _run_mgmn_rank(0, 2, ["a", "a"], local_rank=0, wire=wire, worker_mock=worker)

        assert worker.call_args.kwargs["master_addr"] == "override-host"
        assert worker.call_args.kwargs["master_port"] == 12345
        # The env override is what went over the wire to the other ranks.
        assert wire == {0: "override-host", 1: 12345}

    def test_non_root_env_is_ignored(self):
        # Rank 0 runs without MASTER_* env and resolves its own rendezvous.
        wire = {}
        worker0 = MagicMock()
        _run_mgmn_rank(0, 2, ["a", "a"], local_rank=0, wire=wire, worker_mock=worker0)
        root_addr = worker0.call_args.kwargs["master_addr"]
        root_port = worker0.call_args.kwargs["master_port"]

        # Rank 1 has conflicting env; the broadcast value must win.
        os.environ["MASTER_ADDR"] = "wrong-host"
        os.environ["MASTER_PORT"] = "1"
        worker1 = MagicMock()
        comm1 = _run_mgmn_rank(1, 2, ["a", "a"], local_rank=1, wire=wire, worker_mock=worker1)

        assert worker1.call_args.kwargs["master_addr"] == root_addr
        assert worker1.call_args.kwargs["master_port"] == root_port
        assert comm1.bcast_calls == 2


# =============================================================================
# VisualGen entry point — MGMN wins over stale external-launch env
# =============================================================================


class TestVisualGenEntryMgmn:
    def test_stale_rank_env_ignored_under_mgmn(self):
        # trtllm-llmapi-launch strips SLURM_*/OMPI_* from the wrapped program
        # but not a plain RANK/WORLD_SIZE; without MASTER_ADDR the external-
        # launch detector raises, so reaching DiffusionRemoteClient proves
        # the MGMN check short-circuits it.
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"

        with patch("tensorrt_llm.visual_gen.visual_gen.DiffusionRemoteClient") as mock_client:
            VisualGen(model="/tmp/model")

        mock_client.assert_called_once()

    def test_serve_rank0_gate_ignores_stale_rank_under_mgmn(self):
        # The serve entry has its own rank-0 gate: a non-leader external rank
        # must become a worker instead of binding the HTTP port. Under MGMN a
        # stale non-zero RANK must not trigger it, or rank 0 either raises on
        # the missing MASTER_ADDR or silently never serves.
        os.environ["TLLM_SPAWN_PROXY_PROCESS"] = "1"
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"

        assert launch_mod.is_external_worker_rank() is False

    def test_serve_rank0_gate_still_holds_for_external_launch(self):
        # Regression guard for the external-launch mode the gate exists for.
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["MASTER_ADDR"] = "node0"

        assert launch_mod.is_external_worker_rank() is True


# =============================================================================
# MGMN process-group timeout — TLLM_VG_MGMN_PG_TIMEOUT_SEC override
# =============================================================================


class TestMgmnPgTimeout:
    def _init_pg_timeout(self):
        """Run the MGMN worker path end-to-end with dist + executor mocked,
        returning the timeout handed to torch.distributed.init_process_group."""
        mock_exec = MagicMock()
        mock_exec.pipeline = None
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.distributed.init_process_group") as mock_init,
            patch("torch.distributed.destroy_process_group"),
            patch.object(executor_mod.DiffusionExecutor, "__new__", return_value=mock_exec),
            patch.object(executor_mod.DiffusionExecutor, "__init__", return_value=None),
        ):
            _run_mgmn_rank(
                0,
                1,
                ["a"],
                local_rank=0,
                wire={},
                worker_mock=executor_mod.run_diffusion_worker,
            )
        mock_init.assert_called_once()
        return mock_init.call_args.kwargs["timeout"]

    def test_env_override_reaches_init_process_group(self):
        os.environ["TLLM_VG_MGMN_PG_TIMEOUT_SEC"] = "7"

        assert self._init_pg_timeout() == timedelta(seconds=7)

    def test_default_without_env(self):
        assert self._init_pg_timeout() == timedelta(seconds=launch_mod.MGMN_PG_TIMEOUT_SEC)
