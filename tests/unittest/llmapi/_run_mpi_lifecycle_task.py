# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable MPI tasks and the non-MPI engine used by lifecycle regressions."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import psutil


def _mark(directory: Path, name: str, value: object = True) -> None:
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value))
    temporary.replace(directory / f"{name}.json")


def _record_identity(directory: Path, name: str) -> None:
    process = psutil.Process()
    _mark(directory, f"identity-{name}", {"pid": process.pid, "created": process.create_time()})


def _wait_for_markers(directory: Path, names: list[str], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not all((directory / f"{name}.json").exists() for name in names):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing lifecycle readiness markers: {names}")
        time.sleep(0.01)


def worker_task(directory_name: str, scenario: str, batch: int, size: int) -> tuple[int, int]:
    """Record per-rank execution, then finish, fail, or wait for world teardown."""
    from mpi4py import MPI

    directory = Path(directory_name)
    rank = MPI.COMM_WORLD.Get_rank()
    assert MPI.COMM_WORLD.Get_size() == size
    _record_identity(directory, f"rank-{rank}")

    if batch:
        assert all(
            (directory / f"finished-{batch - 1}-{peer}.json").exists() for peer in range(size)
        ), "The next task started before every rank finished the previous task"
    event_file = directory / f"events-{rank}.jsonl"
    with event_file.open("a") as events:
        events.write(json.dumps({"batch": batch, "scenario": scenario}) + "\n")
    _mark(directory, f"started-{batch}-{rank}")

    if scenario == "async_drain" and batch == 0:
        _wait_for_markers(directory, ["engine-exiting"])

    if scenario in ("all_failure", "worker_system_exit"):
        _wait_for_markers(directory, [f"started-{batch}-{peer}" for peer in range(size)])
        if scenario == "all_failure":
            _mark(directory, f"failed-{batch}-{rank}")
            raise RuntimeError("injected MPI lifecycle failure")
        if rank == size - 1:
            # The failure must traverse MPI, rather than rank0's thread pool.
            _mark(directory, f"failed-{batch}-{rank}")
            sys.exit("injected MPI lifecycle exit")

    if scenario in ("mixed_failure", "mixed_collective", "all_hang"):
        _wait_for_markers(directory, [f"started-{batch}-{peer}" for peer in range(size)])
        if scenario == "mixed_failure" and rank % 2 == 0:
            raise RuntimeError("injected MPI lifecycle failure")
        if scenario == "mixed_collective":
            if rank == size - 1:
                raise RuntimeError("injected MPI lifecycle failure")
            MPI.COMM_WORLD.Barrier()
            raise AssertionError("The collective completed without the failed rank")
        time.sleep(3600)
        raise AssertionError("The owner did not terminate the stuck worker world")

    # Different completion times exercise ordering between queued batches.
    time.sleep(0.01 * (rank + 1))
    _mark(directory, f"finished-{batch}-{rank}")
    if scenario == "sync_failure" and rank == 0:
        raise RuntimeError("injected recoverable sync failure")
    return rank, batch


def main() -> int:
    """Run one engine scenario; readiness is signalled through persistent files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    directory = args.directory
    _record_identity(directory, "engine")
    _mark(directory, "engine-started")
    if args.scenario == "no_submission":
        return 3

    # Import through the module name so MPI receives an importable callable,
    # rather than a function belonging to the engine's __main__ module.
    from _run_mpi_lifecycle_task import worker_task as remote_task

    from tensorrt_llm.executor.utils import (
        get_spawn_proxy_process_ipc_addr_env,
        get_spawn_proxy_process_ipc_hmac_key_env,
    )
    from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionClient

    address = get_spawn_proxy_process_ipc_addr_env()
    key = get_spawn_proxy_process_ipc_hmac_key_env()
    client = RemoteMpiCommSessionClient(address, hmac_key=key)

    def run_sync(scenario: str, batch: int) -> object:
        return client.submit_sync(remote_task, str(directory), scenario, batch, args.ranks)

    def assert_results(response: object, batch: int) -> None:
        assert isinstance(response, list), response
        assert sorted(response) == [(rank, batch) for rank in range(args.ranks)], response

    if args.scenario in ("all_failure", "worker_system_exit"):
        system_exit = args.scenario == "worker_system_exit"
        failed_ranks = [args.ranks - 1] if system_exit else range(args.ranks)
        expected_error = (
            "Remote MPI worker died: SystemExit: injected MPI lifecycle exit"
            if system_exit
            else "injected MPI lifecycle failure"
        )
        exit_status = 24 if system_exit else 23
        try:
            client.submit(remote_task, str(directory), args.scenario, 0, args.ranks)
            _wait_for_markers(directory, [f"failed-0-{rank}" for rank in failed_ranks])
            delayed_at = time.monotonic()
            _mark(directory, "client-poll-delayed", {"monotonic": delayed_at})
            # A remote proxy may wait five seconds before checking worker death.
            time.sleep(5.1)
            _mark(directory, "client-first-poll", {"monotonic": time.monotonic()})
            while time.monotonic() - delayed_at < 30:
                error = client.check_worker_error()
                if error is not None:
                    assert expected_error in str(error), error
                    try:
                        raise error
                    except RuntimeError:
                        error_trace = traceback.format_exc()
                        traceback.print_exc()
                        _mark(
                            directory,
                            "error-observed",
                            {"error": str(error), "traceback": error_trace},
                        )
                    _mark(directory, "engine-exiting", {"status": exit_status})
                    return exit_status
                time.sleep(5.1)
            raise TimeoutError("No worker failure reached the delayed client")
        finally:
            # Release the PAIR connection before the launcher's stop helper connects.
            client.queue.close()

    if args.scenario in ("mixed_failure", "mixed_collective", "all_hang"):
        client.submit(remote_task, str(directory), args.scenario, 0, args.ranks)
        _wait_for_markers(directory, [f"started-0-{rank}" for rank in range(args.ranks)])
        _mark(directory, "workers-started")
        if args.scenario == "all_hang":
            _mark(directory, "engine-exiting", {"status": 1, "monotonic": time.monotonic()})
            return 1
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            error = client.check_worker_error()
            if error is not None:
                assert "injected MPI lifecycle failure" in str(error), error
                _mark(
                    directory,
                    "error-observed",
                    {"error": str(error), "monotonic": time.monotonic()},
                )
                # The owner must end this run even while the engine remains
                # alive; this is not the engine-exit shutdown path.
                time.sleep(3600)
                raise AssertionError("The engine survived fatal worker-world teardown")
            time.sleep(0.01)
        raise TimeoutError("No worker failure reached the live engine")

    if args.scenario == "all_return":
        assert_results(run_sync("return", 0), 0)
    elif args.scenario == "sync_recovery":
        response = run_sync("sync_failure", 0)
        assert isinstance(response, Exception), response
        assert "injected recoverable sync failure" in str(response), response
        _mark(directory, "sync-error-observed")
        assert_results(run_sync("return", 1), 1)
    elif args.scenario == "reuse":
        for batch in range(3):
            client.submit(remote_task, str(directory), "return", batch, args.ranks)
        assert_results(run_sync("return", 3), 3)
        for batch in range(4, 7):
            assert_results(run_sync("return", batch), batch)
        client.shutdown()
        reused = RemoteMpiCommSessionClient(address, hmac_key=key)
        assert reused is client
        assert_results(run_sync("return", 7), 7)
    elif args.scenario == "async_drain":
        for batch in range(3):
            client.submit(remote_task, str(directory), "async_drain", batch, args.ranks)
        _wait_for_markers(directory, [f"started-0-{rank}" for rank in range(args.ranks)])
        # The first batch cannot finish until every request is queued and the
        # engine exits. There is deliberately no synchronous flush request.
        _mark(directory, "engine-exiting", {"status": 0, "monotonic": time.monotonic()})
        return 0
    else:
        raise ValueError(f"Unknown lifecycle scenario: {args.scenario}")

    client.shutdown()
    _mark(directory, "engine-completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
