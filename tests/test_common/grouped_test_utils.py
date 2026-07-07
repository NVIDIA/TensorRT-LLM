# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side helpers for tests that reuse one ``MpiPoolSession``.

Used by the automatic session-reuse layer (``test_common/session_reuse.py``).
This is test infrastructure only and must not be used on production inference
paths. Heavy imports are done lazily so importing this module never triggers a
circular import during package initialization.
"""


def _run_and_get_worker_rank(fn) -> int:
    fn()
    from tensorrt_llm._utils import mpi_rank

    return mpi_rank()


def submit_sync_per_worker(mpi_session, fn, max_rounds: int = 8) -> None:
    """Run ``fn`` at least once on EVERY worker of the pool.

    ``MpiPoolSession.submit_sync`` enqueues ``n_workers`` tasks, but
    ``MPIPoolExecutor`` gives no one-task-per-worker guarantee: a fast task can
    be drained twice by one idle worker, leaving another untouched. Verify
    coverage by collecting the worker ranks that actually ran ``fn`` and
    resubmitting until all workers are covered.
    """
    expected = mpi_session.n_workers
    seen: set = set()
    for _ in range(max_rounds):
        seen.update(mpi_session.submit_sync(_run_and_get_worker_rank, fn))
        if len(seen) >= expected:
            return
    raise RuntimeError(
        f"task only reached worker ranks {sorted(seen)} after {max_rounds} "
        f"rounds; expected {expected} distinct workers"
    )


def reset_worker_torch_compile_state() -> None:
    """Reset per-worker torch.compile / Dynamo state (runs inside each worker).

    Dynamo's recompile counter is process-global and per-code-object. When
    worker processes are reused across LLMs (shared ``MpiPoolSession``), each
    ``torch_compile`` case recompiles the same ``model.forward`` code object
    under new guards; the count accumulates and eventually trips
    ``recompile_limit`` (16), which is a HARD failure under ``fullgraph=True``
    (``FailOnRecompileLimitHit``) and aborts the whole MPI job. Resetting
    between cases makes each LLM start from a clean compile cache, like a fresh
    process. Submit this to each worker via ``mpi_session.submit_sync(...)``.
    """
    import torch

    torch._dynamo.reset()
