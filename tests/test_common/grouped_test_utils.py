# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side helpers for tests that reuse one ``MpiPoolSession``.

Used by the automatic session-reuse layer (``test_common/session_reuse.py``).
This is test infrastructure only and must not be used on production inference
paths. Heavy imports are done lazily so importing this module never triggers a
circular import during package initialization.
"""


def _barrier_run(fn):
    """Runs inside a worker; module-level so it is picklable.

    The leading barrier is what makes ``submit_sync_per_worker`` exactly-once:
    a worker that picked up one of the ``n_workers`` submitted tasks blocks
    here until every other worker holds a task of its own, so no worker can
    drain a second one (pigeonhole). The workers' ``MPI_COMM_WORLD`` is the
    spawned worker world (the parent process is not a member), so the barrier
    spans exactly the pool's ``n_workers`` ranks.
    """
    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    return (MPI.COMM_WORLD.Get_rank(), fn())


def submit_sync_per_worker(mpi_session, fn, timeout: float = 60.0) -> list:
    """Run ``fn`` exactly ONCE on every worker; return results ordered by rank.

    ``MpiPoolSession.submit_sync`` enqueues ``n_workers`` tasks, but
    ``MPIPoolExecutor`` gives no one-task-per-worker guarantee: a fast task can
    be drained twice by one idle worker, leaving another untouched. Each task
    therefore opens with a barrier across the worker world (see
    ``_barrier_run``), which pins the tasks one-per-worker.

    This doubles as the pool health probe: completing the barrier proves every
    worker is alive AND that the worker world's collectives still function —
    the thing tests actually depend on. The wait is bounded by ``timeout``: a
    wedged worker (e.g. stuck in a collective after the previous test's
    executor died mid-shutdown) never completes the barrier, so an unbounded
    ``result()`` would hang the whole session. A timeout is reported as probe
    failure, making the caller retire the pool and spawn a fresh one.
    """
    import concurrent.futures

    futures = mpi_session.submit(_barrier_run, fn)
    done, not_done = concurrent.futures.wait(futures, timeout=timeout)
    if not_done:
        raise RuntimeError(
            f"health probe timed out after {timeout}s: "
            f"{len(not_done)}/{len(futures)} worker tasks never returned"
        )
    by_rank = dict(f.result() for f in done)  # .result() re-raises worker errors
    missing = set(range(mpi_session.n_workers)) - set(by_rank)
    if missing:  # unreachable given the barrier; guards scheduler surprises
        raise RuntimeError(f"no task ran on worker ranks {sorted(missing)}")
    return [result for _, result in sorted(by_rank.items())]


def reset_worker_torch_compile_state() -> None:
    """Reset per-worker torch.compile / Dynamo state (runs inside each worker).

    Dynamo's recompile counter is process-global and per-code-object. When
    worker processes are reused across LLMs (shared ``MpiPoolSession``), each
    ``torch_compile`` case recompiles the same ``model.forward`` code object
    under new guards; the count accumulates and eventually trips
    ``recompile_limit`` (16), which is a HARD failure under ``fullgraph=True``
    (``FailOnRecompileLimitHit``) and aborts the whole MPI job. Resetting
    between cases makes each LLM start from a clean compile cache, like a fresh
    process. Run on every worker via ``submit_sync_per_worker``.
    """
    import torch

    torch._dynamo.reset()
