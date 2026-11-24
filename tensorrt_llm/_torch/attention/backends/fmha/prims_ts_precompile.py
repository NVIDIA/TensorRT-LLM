# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile representative PrimTS variants in a fresh, per-rank worker pool.

`TLLM_PRIMTS_WARM_WORKERS` defaults to at most four workers per rank and
sixteen per host. Zero or one selects local planning.
`TLLM_PRIMTS_WARM_TIMEOUT` is the pool deadline in seconds (1800 by default,
+inf for an unlimited wait). `TLLM_PRIMTS_WARM_STRICT=1` propagates worker
failures. By default the rank retries missing kernels during local planning.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess  # nosec B404 - fixed interpreter and worker script
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

_MIN_POOL_WORKERS = 2
_MAX_DEFAULT_WORKERS = 4
_DEFAULT_TIMEOUT = 1800
_HOST_COMPILE_BUDGET = 16  # default workers of all ranks on one host, together
_LOCAL_RANK_ENV = ("OMPI_COMM_WORLD_LOCAL_SIZE", "RAY_LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE")


def local_rank_count(mapping: Mapping | None) -> int:
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


def can_publish() -> bool:
    """Whether a kernel compiled here reaches the disk cache other processes load from.

    False when the on-disk cache or JIT compilation is disabled.
    """
    return os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") != "1" and not os.environ.get(
        "FLASHINFER_DISABLE_JIT"
    )


def worker_count(num_payloads: int, *, local_ranks: int = 1) -> int:
    """Subprocess count from ``TLLM_PRIMTS_WARM_WORKERS``; ``0``/``1`` disables.

    Without the variable, divide the host compile budget among its local ranks.
    Zero when a worker could compile but keep nothing (:func:`can_publish`).
    """
    if not can_publish():
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


def pool_enabled(*, local_ranks: int) -> bool:
    """Whether this rank can run a pool if enough cache misses are found."""
    return worker_count(_MIN_POOL_WORKERS, local_ranks=local_ranks) >= _MIN_POOL_WORKERS


def _worker_timeout() -> float | None:
    raw = os.environ.get("TLLM_PRIMTS_WARM_TIMEOUT")
    if raw is None:
        return _DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
        if timeout > 0:
            return None if math.isinf(timeout) else timeout
    except ValueError:
        pass
    logger.warning(
        f"Ignoring TLLM_PRIMTS_WARM_TIMEOUT={raw!r}; expected positive seconds or inf "
        f"(using {_DEFAULT_TIMEOUT} seconds)."
    )
    return _DEFAULT_TIMEOUT


def _output_tail(log: BinaryIO) -> bytes:
    size = log.seek(0, os.SEEK_END)
    log.seek(max(0, size - 4096))
    return log.read()


def _wait_for_pool(proc: subprocess.Popen, timeout: float | None) -> int:
    """Keep the supervisor's PID reserved until its process group has been cleaned up."""
    from tensorrt_llm.llmapi._llmapi_process_guard import _child_finished

    if timeout is not None:
        deadline = time.monotonic() + timeout
        while not _child_finished(proc.pid):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            time.sleep(min(0.05, remaining))
    status = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
    assert status is not None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


def encode_payload(device_index: int, rows: int, heads: int, plan_kwargs: dict) -> dict[str, Any]:
    """JSON-safe description of one ``BatchSparseMLADecodePagedTSWrapper.plan`` call."""
    kwargs = dict(plan_kwargs)
    dtype = kwargs.pop("q_dtype")
    kwargs["q_dtype"] = str(dtype).removeprefix("torch.")
    return {"device_index": device_index, "rows": rows, "heads": heads, "plan_kwargs": kwargs}


DENSE_MLA_DECODE = "dense_mla_decode"


def encode_dense_payload(device_index: int, plan_kwargs: dict) -> dict[str, Any]:
    """JSON-safe description of one ``BatchMLADecodePagedTSWrapper.plan`` call."""
    kwargs = dict(plan_kwargs)
    for name in ("q_dtype", "kv_dtype", "out_dtype"):
        kwargs[name] = str(kwargs[name]).removeprefix("torch.")
    return {"kind": DENSE_MLA_DECODE, "device_index": device_index, "plan_kwargs": kwargs}


def payload_key(payload: dict[str, Any]) -> str:
    """Identify an encoded plan the same way on every rank, whichever GPU it uses."""
    return json.dumps({k: v for k, v in payload.items() if k != "device_index"}, sort_keys=True)


def compile_payloads_in_process(payloads: Sequence[dict[str, Any]]) -> None:
    """Compile encoded plans one after another in this process, as a worker would."""
    from ._prims_ts_worker import compile_payload

    start = time.perf_counter()
    try:
        for payload in payloads:
            compile_payload(payload)
    except RuntimeError as error:
        if strict():
            raise
        logger.warning(f"PrimTS precompile failed. Planning missing kernels locally.\n{error}")
        return
    logger.info(
        f"PrimTS MLA precompiled {len(payloads)} kernel variants in process "
        f"in {time.perf_counter() - start:.1f} s"
    )


def run_precompile_workers(payloads: Sequence[dict[str, Any]], workers: int) -> None:
    """Run the encoded plans with up to ``workers`` compiler processes and join them."""
    from tensorrt_llm.llmapi._llmapi_process_guard import _cleanup_group
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    env, _ = split_mpi_env()
    # Spawn re-imports __main__. Give the pool its own import-clean entry point
    # even when the application script imports TensorRT-LLM at module scope.
    script = Path(__file__).with_name("_prims_ts_worker.py")
    timeout = _worker_timeout()
    # File-backed I/O lets us enforce the full deadline without a pipe
    # selector's signed 32-bit millisecond limit.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as request,
        tempfile.TemporaryFile() as log,
    ):
        json.dump(
            {"parent_pid": os.getpid(), "sys_path": sys.path, "payloads": list(payloads)}, request
        )
        request.seek(0)
        with subprocess.Popen(  # nosec B603 - fixed interpreter and script, no shell
            [sys.executable, str(script), str(workers)],
            env=env,
            stdin=request,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc:
            try:
                returncode = _wait_for_pool(proc, timeout)
                if returncode:
                    raise RuntimeError(
                        f"PrimTS worker pool exited with {returncode}:\n"
                        f"{_output_tail(log).decode(errors='replace')}"
                    )
            except BaseException as error:
                # Terminate all compilers and let the resource tracker unlink
                # spawn semaphores before force-killing any remaining processes.
                _cleanup_group(proc.pid, grace=1, signum=signal.SIGTERM)
                if isinstance(error, subprocess.TimeoutExpired):
                    error.output = _output_tail(log)
                    if error.output:
                        logger.warning(
                            f"PrimTS worker output before timeout:\n{error.output.decode(errors='replace')}"
                        )
                raise


def precompile_payloads(payloads: Sequence[dict[str, Any]], *, local_ranks: int = 1) -> None:
    """Precompile missing plans, sharing the host worker budget among ``local_ranks``."""
    workers = worker_count(len(payloads), local_ranks=local_ranks)
    if workers < _MIN_POOL_WORKERS:
        return
    start = time.perf_counter()
    try:
        run_precompile_workers(payloads, workers)
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        if strict():
            raise
        logger.warning(
            f"PrimTS worker precompile failed. Planning missing kernels locally.\n{error}"
        )
        return
    logger.info(
        f"PrimTS MLA precompiled {len(payloads)} kernel variants with {workers} workers "
        f"in {time.perf_counter() - start:.1f} s"
    )
