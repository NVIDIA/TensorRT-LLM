# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks that MpiCommSession runs every submission once per rank.

Run on every rank with ``mpirun -n <N>``: rank 0 drives the session, the other
ranks serve its tasks.
"""

import threading

from _mpi_session_test_tasks import allreduce_ranks, occupy_rank, session_rank
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from tensorrt_llm.llmapi.mpi_session import MpiCommSession, RemoteMpiCommSessionServer


def main() -> None:
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        with MPICommExecutor(comm):
            pass
        return

    size = comm.Get_size()
    ranks = list(range(size))
    session = MpiCommSession(n_workers=size)
    try:
        assert sorted(session.submit_sync(session_rank)) == ranks

        # With one worker rank busy, the idle ones must not drain its copy.
        busy = session.mpi_pool.submit(occupy_rank, 3)
        results = session.submit_sync(session_rank)
        busy.result()
        assert sorted(results) == ranks, results

        # Concurrent submissions of a collective must not cross-match.
        expected = [sum(ranks)] * size
        errors = []

        def submit_many():
            try:
                for _ in range(10):
                    results = session.submit_sync(allreduce_ranks)
                    assert results == expected, results
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=submit_many) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
    finally:
        session.shutdown()
        RemoteMpiCommSessionServer._close_global_comm_executor()
    print("MpiCommSession pinning checks passed", flush=True)


if __name__ == "__main__":
    main()
