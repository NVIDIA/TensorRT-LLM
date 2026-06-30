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
"""Logical WideEP failure-injection worker using a real MPI transport.

The smoke keeps every process alive so ``COMM_WORLD`` can coordinate portable
results. Actual peer death and MPI runtime error classification are outside its
scope because common launchers terminate the whole job when a worker exits
without ``MPI_Finalize``.
"""

import atexit
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.pyexecutor.ep_failure_broadcast import MpiFtSubcomm, MpiFtSubcommConfig
from tensorrt_llm.mapping import MpiTopology

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
_PROPAGATION_TARGET_SEC = 0.1
_CONVERGENCE_TIMEOUT_SEC = 2.0
_ALLOW_SKIP_ENV = "TLLM_ALLOW_MPI_FT_SMOKE_SKIP"
_HEALTHY_MODE = "healthy"
_TERMINAL_MODE = "terminal"
_WORKER_SKIPPED = False


class _NonUlfmComm:
    """Typed MPI communicator view that makes smoke behavior independent of ULFM."""

    def __init__(self, comm: MPI.Intracomm) -> None:
        self._comm = comm
        self.abort_calls = 0

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def Set_errhandler(self, errhandler: MPI.Errhandler) -> None:
        self._comm.Set_errhandler(errhandler)

    def Irecv(self, buffer: np.ndarray, source: int, tag: int) -> MPI.Request:
        return self._comm.Irecv(buffer, source=source, tag=tag)

    def Isend(self, buffer: np.ndarray, dest: int, tag: int) -> MPI.Request:
        return self._comm.Isend(buffer, dest=dest, tag=tag)

    def Abort(self, errorcode: int) -> None:
        self.abort_calls += 1
        self._comm.Abort(errorcode)


def _mapping(world_size: int, rank: int) -> MpiTopology:
    return MpiTopology(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        moe_tp_size=1,
        moe_ep_size=world_size,
    )


def _unsupported_environment(message: str, *, rank: int) -> int:
    """Skip optional local runs, but fail a designated required smoke run."""
    global _WORKER_SKIPPED
    _WORKER_SKIPPED = True
    required = os.environ.get(_ALLOW_SKIP_ENV) != "1"
    if rank == 0:
        if required:
            print(f"WideEP MPI FT smoke requirement failed: {message}", file=sys.stderr)
        else:
            print(f"{_SKIP_MARKER} {message}")
    return 2 if required else 0


def _synchronize_phase_error(
    world: MPI.Intracomm,
    phase: str,
    local_error: str | None,
) -> str | None:
    """Return the same rank-attributed phase failure before the next collective."""
    errors = world.allgather(local_error)
    failures = [(rank, error) for rank, error in enumerate(errors) if error is not None]
    if not failures:
        return None
    details = "\n".join(f"rank {rank}:\n{error}" for rank, error in failures)
    return f"{phase} failed before the next MPI phase:\n{details}"


def _run_invalid_topology_smoke(world: MPI.Intracomm) -> str | None:
    """Prove a rank-local topology error is reconciled before ``Split``."""
    rank = world.Get_rank()
    world_size = world.Get_size()
    ep_group = tuple(range(world_size))
    if rank == 0:
        ep_group = ep_group[:-1]
    mapping = SimpleNamespace(
        world_size=world_size,
        rank=rank,
        moe_ep_size=world_size,
        moe_ep_rank=rank,
        moe_ep_group=ep_group,
    )
    caught_error: str | None = None
    unexpected_broadcaster: MpiFtSubcomm | None = None
    try:
        unexpected_broadcaster = MpiFtSubcomm(
            mapping,
            EPGroupHealth(world_size),
            MpiFtSubcommConfig(startup_timeout_sec=5.0, stop_timeout_sec=5.0),
        )
    except Exception as error:
        caught_error = f"{type(error).__name__}: {error}"

    caught_errors = world.allgather(caught_error)
    expected_fragment = "startup validation failed on parent rank 0"
    if any(error is None for error in caught_errors):
        return f"invalid topology unexpectedly constructed on some ranks: {caught_errors}"
    if len(set(caught_errors)) != 1 or expected_fragment not in caught_errors[0]:
        return f"invalid topology errors were not identical across ranks: {caught_errors}"

    # This barrier is intentionally after the failed constructor. It proves the
    # parent communicator did not strand any rank in MPI_Comm_split.
    world.Barrier()
    if rank == 0:
        print(_INVALID_TOPOLOGY_MARKER, flush=True)
    if unexpected_broadcaster is not None:
        return "invalid topology unexpectedly returned a broadcaster"
    return None


def _run_healthy_lifecycle_smoke(world: MPI.Intracomm) -> str | None:
    """Exercise production construction, progress startup, and clean shutdown."""
    rank = world.Get_rank()
    world_size = world.Get_size()
    mapping = _mapping(world_size, rank)
    broadcaster: MpiFtSubcomm | None = None
    error: str | None = None

    try:
        config = MpiFtSubcommConfig(
            startup_timeout_sec=5.0,
            stop_timeout_sec=5.0,
            reconcile_timeout_sec=5.0,
        )
        # This path obtains TRT-LLM's pkl5 parent through mpi_comm(), performs
        # collective validation and Split, and owns MPI progress directly.
        broadcaster = MpiFtSubcomm(
            mapping,
            EPGroupHealth(world_size),
            config,
        )
        ft_comm = broadcaster._comm
        if ft_comm.Get_rank() != rank or ft_comm.Get_size() != world_size:
            raise AssertionError(
                "MPI_Comm_split returned an unexpected FT communicator: "
                f"rank={ft_comm.Get_rank()}, size={ft_comm.Get_size()}"
            )
        if ft_comm.Get_errhandler() != MPI.ERRORS_RETURN:
            raise AssertionError("FT communicator does not use MPI.ERRORS_RETURN")
        broadcaster.start()
        broadcaster.stop()
        if broadcaster.last_error is not None:
            raise AssertionError(
                "Production-construction progress failed during healthy shutdown: "
                f"{broadcaster.last_error}"
            )
    except Exception:
        error = traceback.format_exc()
    finally:
        if broadcaster is not None:
            try:
                broadcaster.stop(timeout=5.0)
                if broadcaster.last_error is not None:
                    raise AssertionError(
                        "MPI progress failed during healthy lifecycle smoke: "
                        f"{broadcaster.last_error}"
                    )
            except Exception:
                cleanup_error = traceback.format_exc()
                error = f"{error or ''}\ncleanup failure:\n{cleanup_error}"
        # Production-created FT communicators are intentionally retained for
        # process lifetime. Freeing one here would violate the same ownership
        # contract this smoke is meant to exercise.

    return error


def _run_failure_propagation_smoke(world: MPI.Intracomm) -> str | None:
    """Exercise logical failure fanout and sticky state over real MPI traffic."""
    rank = world.Get_rank()
    world_size = world.Get_size()
    mapping = _mapping(world_size, rank)
    health = EPGroupHealth(world_size)
    owner: MpiFtSubcomm | None = None
    broadcaster: MpiFtSubcomm | None = None
    detector_rank = 0
    failure_rank = world_size - 1
    error: str | None = None
    propagation_converged = False

    setup_error: str | None = None
    try:
        config = MpiFtSubcommConfig(
            startup_timeout_sec=5.0,
            stop_timeout_sec=5.0,
            reconcile_timeout_sec=5.0,
        )
        # Create the real FT communicator collectively, then inject a non-ULFM
        # view so this smoke deterministically exercises typed Isend/Irecv+Test.
        owner = MpiFtSubcomm(
            mapping,
            EPGroupHealth(world_size),
            config,
        )
        ft_comm = owner._comm
        owner.stop()
        broadcaster = MpiFtSubcomm(
            mapping,
            health,
            config,
            comm=_NonUlfmComm(ft_comm),
        )
        broadcaster.start()
    except Exception:
        setup_error = traceback.format_exc()
    error = _synchronize_phase_error(world, "failure-propagation setup", setup_error)

    detector_elapsed_sec: float | None = None
    local_converged = False
    if error is None:
        phase_errors: list[str] = []
        try:
            # Every broadcaster is running before rank 0 starts the latency
            # clock. This synchronization is deliberately outside the measured
            # interval.
            world.Barrier()
        except Exception:
            phase_errors.append(f"readiness:\n{traceback.format_exc()}")

        trigger_error: str | None = None
        propagation_start: float | None = None
        if not phase_errors and rank == detector_rank:
            propagation_start = time.monotonic()
            try:
                assert broadcaster is not None
                broadcaster.pre_failover(failure_rank)
            except Exception:
                trigger_error = traceback.format_exc()

        observation_error: str | None = None
        try:
            # The victim process remains alive only so the smoke can coordinate
            # its result on COMM_WORLD. Survivors exclude its logical rank and
            # exercise the same failure/reconciliation fanout as production.
            assert broadcaster is not None
            deadline = time.monotonic() + _CONVERGENCE_TIMEOUT_SEC
            while not phase_errors and rank != failure_rank and time.monotonic() < deadline:
                failure_observed = not health.is_active(failure_rank)
                failure_reconciled = broadcaster.failure_is_reconciled(failure_rank)
                if failure_observed and failure_reconciled:
                    if rank == detector_rank:
                        assert propagation_start is not None
                        detector_elapsed_sec = time.monotonic() - propagation_start
                    break
                time.sleep(0.005)
            local_converged = rank == failure_rank or (
                not health.is_active(failure_rank)
                and broadcaster.failure_is_reconciled(failure_rank)
                and broadcaster.health_is_reconciled()
                and broadcaster.world_is_poisoned()
            )
        except Exception:
            observation_error = traceback.format_exc()

        if trigger_error is not None:
            phase_errors.append(f"trigger:\n{trigger_error}")
        if observation_error is not None:
            phase_errors.append(f"observation:\n{observation_error}")
        error = _synchronize_phase_error(
            world,
            "failure-propagation trigger/observation",
            "\n".join(phase_errors) or None,
        )

    if error is None:
        all_converged = world.allreduce(local_converged, op=MPI.LAND)
        # Share only rank 0's completed duration after the timed interval; raw
        # monotonic timestamps never cross rank or host boundaries.
        detector_elapsed_sec = world.bcast(detector_elapsed_sec, root=detector_rank)
        if all_converged and detector_elapsed_sec is not None:
            propagation_converged = True
        else:
            error = (
                f"failure rank {failure_rank} did not converge within "
                f"{_CONVERGENCE_TIMEOUT_SEC:.0f}s"
            )
        if rank == detector_rank and detector_elapsed_sec is not None:
            elapsed_ms = detector_elapsed_sec * 1000.0
            target_ms = _PROPAGATION_TARGET_SEC * 1000.0
            print(
                f"{_PROPAGATION_MARKER} world_size={world_size} "
                f"elapsed_ms={elapsed_ms!r} target_ms={target_ms!r} "
                f"target_met={elapsed_ms < target_ms}",
                flush=True,
            )

    stop_succeeded = False
    if broadcaster is not None:
        # Deliberately do not reactivate failure_rank. Survivors must take the
        # poisoned shutdown path and retain their unmatched receive requests.
        try:
            broadcaster.stop(timeout=5.0)
            if broadcaster.last_error is not None:
                raise AssertionError(
                    f"MPI progress failed during propagation smoke: {broadcaster.last_error}"
                )
            stop_succeeded = True
        except Exception:
            cleanup_error = traceback.format_exc()
            error = f"{error or ''}\ncleanup failure:\n{cleanup_error}"
    if propagation_converged and stop_succeeded:
        try:
            if rank == failure_rank:
                if broadcaster.world_is_poisoned():
                    raise AssertionError("logical victim unexpectedly entered poisoned state")
                if broadcaster._retained_requests:
                    raise AssertionError("logical victim retained healthy MPI requests")
            else:
                if not broadcaster.world_is_poisoned():
                    raise AssertionError("survivor lost sticky poisoned state during stop")
                if not broadcaster._retained_requests:
                    raise AssertionError("survivor did not retain poisoned MPI requests")
        except Exception:
            sticky_error = traceback.format_exc()
            error = f"{error or ''}\nsticky-poison failure:\n{sticky_error}"
    if owner is not None:
        try:
            owner.stop(timeout=5.0)
        except Exception:
            cleanup_error = traceback.format_exc()
            error = f"{error or ''}\nowner cleanup failure:\n{cleanup_error}"

    return error


def _write_terminal_manifest(lines: list[str]) -> None:
    """Emit canonical rank evidence from the single rank-0 output stream."""
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    sys.stdout.flush()


def _terminal_atexit_sentinel() -> None:
    """Negative evidence: this must not run in intentional no-Finalize mode."""
    marker = f"{_TERMINAL_ATEXIT_MARKER} pid={os.getpid()}\n".encode()
    try:
        os.write(sys.stderr.fileno(), marker)
    except BaseException:
        pass


def _run_terminal_abort_smoke(world: MPI.Intracomm) -> str | None:
    """Exercise terminal ABORT relay while keeping ``COMM_WORLD`` healthy."""
    rank = world.Get_rank()
    world_size = world.Get_size()
    mapping = _mapping(world_size, rank)
    broadcaster: MpiFtSubcomm | None = None
    wrapped_comm: _NonUlfmComm | None = None
    error: str | None = None

    setup_error: str | None = None
    try:
        # Construct a fresh production communicator, then hide ULFM so this
        # scenario specifically validates the bounded ABORT echo fallback.
        owner = MpiFtSubcomm(
            mapping,
            EPGroupHealth(world_size),
            MpiFtSubcommConfig(startup_timeout_sec=5.0, stop_timeout_sec=5.0),
        )
        ft_comm = owner._comm
        owner.stop()
        wrapped_comm = _NonUlfmComm(ft_comm)
        broadcaster = MpiFtSubcomm(
            mapping,
            EPGroupHealth(world_size),
            MpiFtSubcommConfig(
                startup_timeout_sec=5.0,
                stop_timeout_sec=5.0,
                reconcile_timeout_sec=5.0,
                abort_timeout_sec=3.0,
            ),
            comm=wrapped_comm,
        )
        broadcaster.start()
    except Exception:
        setup_error = traceback.format_exc()
    error = _synchronize_phase_error(world, "terminal-ABORT setup", setup_error)

    if error is None:
        trigger_error: str | None = None
        try:
            world.Barrier()
            if rank == 0:
                # A transport-independent terminal request is the protocol seam
                # used by local second-failure and timeout paths.
                assert broadcaster is not None
                broadcaster._request_terminal_abort(
                    RuntimeError("real-MPI terminal smoke"), -1, rank
                )
        except Exception:
            trigger_error = traceback.format_exc()
        error = _synchronize_phase_error(
            world,
            "terminal-ABORT trigger",
            trigger_error,
        )

    local_converged = False
    if error is None:
        observation_error: str | None = None
        try:
            assert broadcaster is not None
            assert wrapped_comm is not None
            deadline = time.monotonic() + _CONVERGENCE_TIMEOUT_SEC
            while time.monotonic() < deadline:
                if broadcaster._progress_failed.is_set():
                    break
                time.sleep(0.005)
            local_converged = (
                broadcaster._progress_failed.is_set()
                and broadcaster.last_error is not None
                and broadcaster.world_is_poisoned()
                and wrapped_comm.abort_calls == 0
            )
        except Exception:
            observation_error = traceback.format_exc()
        error = _synchronize_phase_error(
            world,
            "terminal-ABORT observation",
            observation_error,
        )

    if error is None:
        if not world.allreduce(local_converged, op=MPI.LAND):
            error = "terminal ABORT did not converge on every MPI rank"
        else:
            # The ABORT protocol is confined to the FT subcommunicator. A
            # parent barrier after convergence proves COMM_WORLD remains usable.
            world.Barrier()
            if rank == 0:
                print(_ABORT_WORLD_MARKER, flush=True)
    if broadcaster is not None:
        try:
            broadcaster.stop(timeout=5.0)
            if not broadcaster.world_is_poisoned():
                raise AssertionError("terminal FT state lost its poisoned-world latch")
        except Exception:
            cleanup_error = traceback.format_exc()
            error = f"{error or ''}\nterminal cleanup failure:\n{cleanup_error}"

    # Do not free the terminal communicator. Its component intentionally
    # retains potentially active requests until process teardown.
    return error


def _report_scenario_errors(
    world: MPI.Intracomm,
    scenario: str,
    local_error: str | None,
) -> bool:
    rank = world.Get_rank()
    errors = world.gather(local_error, root=0)
    failed = None
    if rank == 0:
        failed = any(error is not None for error in errors)
        if failed:
            for error_rank, error in enumerate(errors):
                if error is not None:
                    print(
                        f"{scenario} failed on rank {error_rank}:\n{error}",
                        file=sys.stderr,
                    )
    return world.bcast(failed, root=0)


def main() -> int:
    world = MPI.COMM_WORLD
    expected_world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    world_size = world.Get_size()
    rank = world.Get_rank()
    if world_size < 2 or world_size != expected_world_size:
        return _unsupported_environment(
            f"expected {expected_world_size} MPI ranks, got {world_size}",
            rank=rank,
        )
    if MPI.Query_thread() < MPI.THREAD_MULTIPLE:
        return _unsupported_environment(
            f"MPI.THREAD_MULTIPLE is unavailable (provided={MPI.Query_thread()})",
            rank=rank,
        )

    if mode == _HEALTHY_MODE:
        scenarios = (
            ("invalid-topology", _run_invalid_topology_smoke),
            ("healthy-lifecycle", _run_healthy_lifecycle_smoke),
        )
    elif mode == _TERMINAL_MODE:
        scenarios = (
            ("failure-broadcast", _run_failure_propagation_smoke),
            ("terminal-abort", _run_terminal_abort_smoke),
        )
    else:
        if rank == 0:
            print(f"unknown WideEP MPI smoke mode: {mode!r}", file=sys.stderr)
        return 2
    for scenario, run_scenario in scenarios:
        if _report_scenario_errors(world, scenario, run_scenario(world)):
            return 1
    if mode == _HEALTHY_MODE and rank == 0:
        print(_HEALTHY_LIFECYCLE_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    worker_mode = sys.argv[2] if len(sys.argv) > 2 else ""
    if worker_mode == _TERMINAL_MODE:
        # A normal interpreter shutdown would run mpi4py's Finalize hook. The
        # launcher rejects this sentinel, so the smoke independently proves that
        # the terminal path below bypassed Python atexit via os._exit().
        atexit.register(_terminal_atexit_sentinel)
    exit_code = main()
    if worker_mode == _TERMINAL_MODE and not _WORKER_SKIPPED:
        # This mode deliberately leaves the FT control plane terminal and
        # retains its requests. Skip mpi4py's collective MPI_Finalize just as
        # the production poisoned-world shutdown hook will do.
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code == 0:
            world = MPI.COMM_WORLD
            rank = world.Get_rank()
            world_size = world.Get_size()
            try:
                # Gather evidence first, then let rank 0 emit a canonical
                # manifest. MPI launchers forward separate per-rank pipes and do
                # not preserve cross-rank write atomicity in the merged stream.
                ready_ranks = world.gather(rank, root=0)
                if rank == 0:
                    if sorted(ready_ranks) != list(range(world_size)):
                        raise AssertionError(f"invalid terminal READY ranks: {ready_ranks}")
                    _write_terminal_manifest(
                        [
                            f"{_TERMINAL_READY_MARKER} rank={ready_rank} world_size={world_size}"
                            for ready_rank in ready_ranks
                        ]
                    )
                world.Barrier()
                done_ranks = world.gather(rank, root=0)
                if rank == 0:
                    if sorted(done_ranks) != list(range(world_size)):
                        raise AssertionError(f"invalid terminal DONE ranks: {done_ranks}")
                    _write_terminal_manifest(
                        [
                            f"{_TERMINAL_COMPLETE_MARKER} world_size={world_size}",
                            *(
                                f"{_TERMINAL_DONE_MARKER} rank={done_rank} world_size={world_size}"
                                for done_rank in done_ranks
                            ),
                        ]
                    )
                # No rank can trigger launcher fail-fast until rank 0 has copied
                # the complete gathered manifest to stdout.
                world.Barrier()
            except BaseException:
                try:
                    _write_terminal_manifest([f"{_TERMINAL_ERROR_MARKER} rank={rank} exit_code=1"])
                except BaseException:
                    pass
                traceback.print_exc()
                sys.stderr.flush()
                exit_code = 1
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            try:
                _write_terminal_manifest(
                    [f"{_TERMINAL_ERROR_MARKER} rank={rank} exit_code={exit_code}"]
                )
            except BaseException:
                pass
        os._exit(exit_code)
    # Healthy mode returns normally so mpi4py's MPI_Finalize path remains part
    # of the smoke test.
    sys.exit(exit_code)
