# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel precompilation of PrimTS sparse MLA kernel variants.

The decoder's explicit warmup collects the kernel variants of every enabled
layer profile. This module fans their combined payloads out to one group of
worker subprocesses that call the public wrapper's ``plan()`` method. FlashInfer
compiles different kernels concurrently and serializes artifact publication;
the serving process loads every variant from disk before its first forward.
The workers serve all profiles and exit before model execution and graph capture.

Workers are plain interpreter subprocesses rather than ``multiprocessing``
children: importing ``tensorrt_llm`` initializes MPI, so a worker must not
inherit the launcher's MPI environment, and a subprocess gets a fully
controlled one. Worker ``i`` of ``N`` handles only payload slots ``i, i+N, ...``.
The parent waits for all workers, so their combined shards populate the cache.
FlashInfer's per-kernel lock handles overlapping compile signatures or other
ranks sharing the workspace. Failures are best-effort: the caller falls back
to compiling missing kernels in-process.

Host load. Every worker is an interpreter that imports torch and the CuTe DSL,
opens a CUDA context on its rank's GPU and runs ``cute.compile`` (one to a few
GB of host memory and one busy core each). Under the default launcher each
rank owns a private FlashInfer workspace, so the ranks of one host do not
share compiles and every rank fans out on its own. The default worker count
therefore follows the number of ranks on the host (from the model's
``Mapping``, or the launcher's environment): at most 4 per rank and at most
``_HOST_COMPILE_BUDGET`` (16) workers per host in total, so a TP=8 node runs 2
workers per rank, not 32 subprocesses at once. Set
``TLLM_PRIMTS_WARM_WORKERS`` explicitly to trade startup time against host
memory and CPU on a given node.

Environment:
    TLLM_PRIMTS_WARM_WORKERS   subprocess count per rank; default 4 capped so
                               that all ranks on the host start at most 16;
                               0 or 1 disables.
    TLLM_PRIMTS_WARM_TIMEOUT   seconds to wait for the workers, default 1800.
    TLLM_PRIMTS_WARM_STRICT    1 turns a worker failure into an error (tests).
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - runs sys.executable workers with a fixed argv, no shell
import sys
import tempfile
import time
from typing import Any, Sequence

from tensorrt_llm.logger import logger

_WORKER_BOOTSTRAP = """
import json, os, sys
_path = os.environ.get("TLLM_PRIMTS_PRECOMPILE_SYS_PATH")
if _path:
    sys.path[:] = json.loads(_path)
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile import _main
sys.exit(_main(sys.argv))
"""


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

    Without the variable the default comes from ``default_worker_count()``
    for ``local_ranks`` ranks on this host (see the module note on host load).
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


def precompile_payloads(payloads: Sequence[dict[str, Any]], *, local_ranks: int = 1) -> None:
    """Populate all profiles with one worker group through normal build/load calls.

    The caller subsequently loads every plan in-process, also compiling any
    artifacts missing after worker failure or when workers are disabled.
    """
    workers = worker_count(len(payloads), local_ranks=local_ranks)
    if workers <= 1:
        return
    start = time.perf_counter()
    try:
        ok = run_precompile_workers(payloads, workers)
    except Exception as error:  # noqa: BLE001 -- helper failures fall back to local planning
        if strict():
            raise
        logger.warning(f"PrimTS sparse MLA parallel precompile skipped: {error!r}")
        return
    if not ok and strict():
        raise RuntimeError(
            "PrimTS sparse MLA precompile workers failed (TLLM_PRIMTS_WARM_STRICT=1)"
        )
    logger.info(
        f"PrimTS sparse MLA precompiled {len(payloads)} kernel variants with {workers} workers "
        f"in {time.perf_counter() - start:.0f} s{'' if ok else ' (some workers failed)'}"
    )


def _worker_env() -> dict[str, str]:
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    env, _ = split_mpi_env()
    env["TLLM_PRIMTS_WARM_WORKERS"] = "0"  # workers never fan out again
    env.setdefault("TLLM_LOG_LEVEL", "warning")
    # The on-disk cache key includes the installed CuTe DSL distributions as
    # enumerated along sys.path; a worker must see the parent's sys.path
    # exactly (entries, order and duplicates) to write the same meta.json.
    env["TLLM_PRIMTS_PRECOMPILE_SYS_PATH"] = json.dumps(sys.path)
    return env


def run_precompile_workers(payloads: Sequence[dict[str, Any]], workers: int) -> bool:
    """Compile ``payloads`` with ``workers`` subprocesses; True iff every worker succeeded.

    Worker output goes to temporary files (a pipe could fill while the worker
    holds a kernel lock another worker waits on). The wait is bounded by
    ``TLLM_PRIMTS_WARM_TIMEOUT``; on timeout or any error every worker is
    killed and the caller compiles in-process.
    """
    if workers <= 1 or not payloads:
        return False
    timeout_s = float(os.environ.get("TLLM_PRIMTS_WARM_TIMEOUT", "1800"))
    env = _worker_env()
    payload_json = json.dumps(list(payloads))
    procs: list[tuple[int, subprocess.Popen, Any]] = []
    ok = True
    try:
        for slot in range(workers):
            out = tempfile.TemporaryFile(mode="w+", prefix="prims_ts_precompile_", suffix=".log")
            proc = subprocess.Popen(
                [sys.executable, "-c", _WORKER_BOOTSTRAP, payload_json, str(slot), str(workers)],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append((slot, proc, out))
        deadline = time.monotonic() + timeout_s
        for slot, proc, out in procs:
            remaining = deadline - time.monotonic()
            try:
                proc.wait(timeout=max(remaining, 0.0))
            except subprocess.TimeoutExpired:
                ok = False
                logger.warning(
                    f"PrimTS precompile worker (slot {slot}) exceeded TLLM_PRIMTS_WARM_TIMEOUT="
                    f"{timeout_s:g}s; the serving process compiles what is missing."
                )
                continue
            out.seek(0)
            text = out.read()
            if proc.returncode != 0:
                ok = False
                tail = "\n".join(text.strip().splitlines()[-15:])
                logger.warning(
                    f"PrimTS precompile worker (slot {slot}) exited with {proc.returncode}; "
                    f"the serving process compiles what is missing.\n{tail}"
                )
                continue
            logger.debug(f"PrimTS precompile worker (slot {slot}) finished")
    finally:
        for _, proc, out in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            out.close()
    return ok


def _plan(mla_decode, payload: dict[str, Any], torch) -> None:
    device = torch.device("cuda", int(payload["device_index"]))
    torch.cuda.set_device(device)
    kwargs = dict(payload["plan_kwargs"])
    kwargs["q_dtype"] = getattr(torch, kwargs["q_dtype"])
    mla_decode.BatchSparseMLADecodePagedTSWrapper().plan(
        device, int(payload["rows"]), int(payload["heads"]), **kwargs
    )


def _main(argv: list[str]) -> int:
    """Worker entry: compile only ``payloads[slot::slots]``."""
    payloads = json.loads(argv[1])
    slot, slots = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (0, 1)
    import torch

    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

    for payload in payloads[slot::slots]:
        _plan(mla_decode, payload, torch)
        print(
            f"[prims-ts precompile] rows={payload['rows']} extra={payload['plan_kwargs'].get('max_extra_topk')} done",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
