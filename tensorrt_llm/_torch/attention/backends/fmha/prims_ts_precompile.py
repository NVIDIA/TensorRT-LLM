# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile representative PrimTS variants in a fresh, per-rank worker pool."""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404 - fixed interpreter and worker script
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from tensorrt_llm.logger import logger

_MAX_DEFAULT_WORKERS = 4
_HOST_COMPILE_BUDGET = 16  # default workers of all ranks on one host, together
_LOCAL_RANK_ENV = ("OMPI_COMM_WORLD_LOCAL_SIZE", "RAY_LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE")


def local_rank_count(mapping) -> int:
    """Ranks of this job on this host, without any MPI call.

    From the model's ``Mapping`` (``min(world_size, gpus_per_node)``) when the
    caller has one, else from the launcher's environment, else 1. No
    communicator query: worker creation does not need a collective.
    """
    if mapping is not None:
        world = int(getattr(mapping, "world_size", 0) or 0)
        per_node = int(getattr(mapping, "gpus_per_node", 0) or 0)
        if world > 0 and per_node > 0:
            return max(1, min(world, per_node))
        if world > 0:
            return world
    for name in _LOCAL_RANK_ENV:
        raw = os.environ.get(name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return 1


def default_worker_count(local_ranks: int = 1) -> int:
    """Per-rank default: 4, lowered so the ranks on the host start at most 16 workers."""
    return max(1, min(_MAX_DEFAULT_WORKERS, _HOST_COMPILE_BUDGET // max(1, int(local_ranks))))


def worker_count(num_payloads: int, *, local_ranks: int = 1) -> int:
    """Subprocess count from ``TLLM_PRIMTS_WARM_WORKERS``; ``0``/``1`` disables.

    Without the variable, divide the host compile budget among its local ranks.
    Zero when the on-disk cache or JIT compilation is disabled: a worker could
    then compile but keep nothing.
    """
    if os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") == "1" or os.environ.get(
        "FLASHINFER_DISABLE_JIT"
    ):
        return 0
    raw = os.environ.get("TLLM_PRIMTS_WARM_WORKERS")
    if raw is None:
        workers = default_worker_count(local_ranks)
    else:
        try:
            workers = int(raw)
        except ValueError:
            workers = default_worker_count(local_ranks)
            logger.warning(
                f"Ignoring TLLM_PRIMTS_WARM_WORKERS={raw!r}; expected an integer "
                f"(using the default of {workers})."
            )
    return max(0, min(workers, num_payloads))


def strict() -> bool:
    return os.environ.get("TLLM_PRIMTS_WARM_STRICT", "0") == "1"


def encode_payload(device_index: int, rows: int, heads: int, plan_kwargs: dict) -> dict[str, Any]:
    """JSON-safe description of one ``BatchSparseMLADecodePagedTSWrapper.plan`` call."""
    kwargs = dict(plan_kwargs)
    dtype = kwargs.pop("q_dtype")
    kwargs["q_dtype"] = str(dtype).removeprefix("torch.")
    return {"device_index": device_index, "rows": rows, "heads": heads, "plan_kwargs": kwargs}


def run_precompile_workers(payloads: Sequence[dict[str, Any]], workers: int) -> None:
    """Join a fresh pool before the rank measures free memory or loads its plans."""
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    env, _ = split_mpi_env()
    # Spawn re-imports __main__. Give the pool its own import-clean entry point
    # even when the application script imports TensorRT-LLM at module scope.
    script = Path(__file__).with_name("_prims_ts_worker.py")
    timeout = float(os.environ.get("TLLM_PRIMTS_WARM_TIMEOUT", "1800"))
    request = json.dumps(
        {"parent_pid": os.getpid(), "sys_path": sys.path, "payloads": list(payloads)}
    ).encode()
    with subprocess.Popen(  # nosec B603 - fixed interpreter and script, no shell
        [sys.executable, str(script), str(workers)],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as proc:
        try:
            output, _ = proc.communicate(request, timeout=timeout)
        except BaseException:
            # A Future timeout leaves running compiles alive. Stop the entire
            # pool before the caller falls back to local planning.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        if proc.returncode:
            raise RuntimeError(
                f"PrimTS worker pool exited with {proc.returncode}: "
                f"{output[-4096:].decode(errors='replace')}"
            )


def precompile_payloads(payloads: Sequence[dict[str, Any]], *, local_ranks: int = 1) -> None:
    workers = worker_count(len(payloads), local_ranks=local_ranks)
    if workers <= 1:
        return
    start = time.perf_counter()
    try:
        run_precompile_workers(payloads, workers)
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        if strict():
            raise
        logger.warning(f"PrimTS sparse MLA parallel precompile skipped: {error!r}")
        return
    logger.info(
        f"PrimTS sparse MLA precompiled {len(payloads)} kernel variants with {workers} workers "
        f"in {time.perf_counter() - start:.1f} s"
    )
