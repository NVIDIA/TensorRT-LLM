# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel precompilation of PrimTS sparse MLA kernel variants.

The decoder's explicit warmup collects the kernel variants of every enabled
layer profile. This module fans their combined payloads out to one group of
worker subprocesses that call the public wrapper's ``plan()`` method. FlashInfer
compiles different kernels concurrently and serializes artifact publication;
the serving process loads every variant from disk before its first forward.
During executor construction, one group preloads PyTorch while weights load.
It stays CPU-only until its first warmup batch and exits when that batch finishes.
Unused workers exit when construction ends. After a batch, later warmup passes
load all plans locally, including new variants, without launching another group.

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
import select
import subprocess  # nosec B404 - runs sys.executable workers with a fixed argv, no shell
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

_WORKER_BOOTSTRAP = """
import json, os, runpy, sys
_path = os.environ.get("TLLM_PRIMTS_PRECOMPILE_SYS_PATH")
if _path:
    sys.path[:] = json.loads(_path)
# Persistent bootstrap settings and sys.path are inherited from the parent.
# Reuse the temporary allocator policy without importing the product tree.
_bootstrap = runpy.run_path(os.environ.pop("TLLM_PRIMTS_PRECOMPILE_BOOTSTRAP"))
with _bootstrap["_mimalloc_arenas_off_thp"]():
    import torch
# Signal CPU-preload completion; backend imports wait for the payload.
_ready_fd = int(os.environ.pop("TLLM_PRIMTS_PRECOMPILE_READY_FD"))
os.write(_ready_fd, b"1")
os.close(_ready_fd)
if not sys.stdin.buffer.read(1):
    sys.exit(0)
with open(os.environ.pop("TLLM_PRIMTS_PRECOMPILE_PAYLOAD")) as _payload_file:
    sys.argv[1] = _payload_file.read()
_payloads = json.loads(sys.argv[1])
_slot, _slots = int(sys.argv[2]), int(sys.argv[3])
if not _payloads[_slot::_slots]:
    sys.exit(0)
torch.cuda.set_device(int(_payloads[_slot]["device_index"]))
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_precompile import _main
sys.exit(_main(sys.argv))
"""


_MAX_DEFAULT_WORKERS = 4
_HOST_COMPILE_BUDGET = 16  # default workers of all ranks on one host, together
_LOCAL_RANK_ENV = ("OMPI_COMM_WORLD_LOCAL_SIZE", "RAY_LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE")
_session: ContextVar["PrecompileSession | None"] = ContextVar("prims_ts_precompile", default=None)


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
        session = _session.get()
        if session is not None:
            if session.consumed:
                return
            if session.group is not None:
                workers = len(session.group.procs)
            ok = session.run(payloads, workers)
        else:
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
    from tensorrt_llm import _bootstrap
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    env, _ = split_mpi_env()
    env["TLLM_PRIMTS_WARM_WORKERS"] = "0"  # workers never fan out again
    env.setdefault("TLLM_LOG_LEVEL", "warning")
    # The on-disk cache key includes the installed CuTe DSL distributions as
    # enumerated along sys.path; a worker must see the parent's sys.path
    # exactly (entries, order and duplicates) to write the same meta.json.
    env["TLLM_PRIMTS_PRECOMPILE_SYS_PATH"] = json.dumps(sys.path)
    env["TLLM_PRIMTS_PRECOMPILE_BOOTSTRAP"] = _bootstrap.__file__
    return env


class _WorkerGroup:
    """One early import phase followed by one bounded compile batch."""

    def __init__(self, workers: int) -> None:
        self.procs: list[tuple[int, subprocess.Popen, Any, int]] = []
        self.timeout_s = float(os.environ.get("TLLM_PRIMTS_WARM_TIMEOUT", "1800"))
        env = _worker_env()
        self.payload_file = tempfile.NamedTemporaryFile(
            mode="w+", prefix="prims_ts_precompile_", suffix=".json"
        )
        env["TLLM_PRIMTS_PRECOMPILE_PAYLOAD"] = self.payload_file.name
        try:
            for slot in range(workers):
                out = tempfile.TemporaryFile(
                    mode="w+", prefix="prims_ts_precompile_", suffix=".log"
                )
                ready_read, ready_write = os.pipe()
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-c", _WORKER_BOOTSTRAP, "-", str(slot), str(workers)],
                        env={**env, "TLLM_PRIMTS_PRECOMPILE_READY_FD": str(ready_write)},
                        stdin=subprocess.PIPE,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        text=False,
                        bufsize=0,
                        pass_fds=(ready_write,),
                    )
                except BaseException:
                    os.close(ready_read)
                    out.close()
                    raise
                finally:
                    os.close(ready_write)
                self.procs.append((slot, proc, out, ready_read))
        except BaseException:
            self.close()
            raise

    def run(self, payloads: Sequence[dict[str, Any]]) -> bool:
        timeout_s = self.timeout_s
        ok = True
        try:
            deadline = time.monotonic() + timeout_s
            json.dump(list(payloads), self.payload_file)
            self.payload_file.flush()
            for slot, proc, out, ready in self.procs:
                readable, _, _ = select.select(
                    [ready], [], [], max(0.0, deadline - time.monotonic())
                )
                if not readable or os.read(ready, 1) != b"1":
                    ok = False
                    logger.warning(
                        f"PrimTS precompile worker (slot {slot}) did not finish its CPU preload."
                    )
                    continue
                assert proc.stdin is not None
                # This is the pipe's only write; one byte fits even if the
                # worker stalls immediately after signaling preload completion.
                proc.stdin.write(b"\n")
            for slot, proc, out, _ in self.procs:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    ok = False
                    logger.warning(
                        f"PrimTS precompile worker (slot {slot}) exceeded TLLM_PRIMTS_WARM_TIMEOUT="
                        f"{timeout_s:g}s; the serving process compiles what is missing."
                    )
                    continue
                if proc.returncode != 0:
                    ok = False
                    out.seek(0)
                    tail = "\n".join(out.read().strip().splitlines()[-15:])
                    logger.warning(
                        f"PrimTS precompile worker (slot {slot}) exited with {proc.returncode}; "
                        f"the serving process compiles what is missing.\n{tail}"
                    )
                else:
                    logger.debug(f"PrimTS precompile worker (slot {slot}) finished")
        finally:
            self.close()
        return ok

    def close(self) -> None:
        for _, proc, _, _ in self.procs:
            if proc.poll() is None:
                with suppress(ProcessLookupError):
                    proc.kill()
        for _, proc, _, _ in self.procs:
            proc.wait()
        for _, proc, out, ready in self.procs:
            if proc.stdin is not None:
                with suppress(OSError):
                    proc.stdin.close()
            with suppress(OSError):
                os.close(ready)
            with suppress(OSError):
                out.close()
        self.procs.clear()
        self.payload_file.close()


def run_precompile_workers(payloads: Sequence[dict[str, Any]], workers: int) -> bool:
    """Compile one batch, with file-backed logs and a hard subprocess timeout."""
    if workers <= 1 or not payloads:
        return False
    return _WorkerGroup(workers).run(payloads)


class PrecompileSession:
    """Own a single worker group across profiling and serving warmup passes.

    Workers start from the resolved model config, or from the first actual
    payload if no group was started early. They exit after their first batch.
    Subsequent passes use normal local planning, including compilation of any
    profiles first encountered there.
    No successful export is treated as a substitute for loading a parent plan.
    """

    def __init__(self) -> None:
        self.group: _WorkerGroup | None = None
        self.consumed = False

    def start(self, config: ModelConfig, mapping: Mapping) -> None:
        if self.group is not None or self.consumed:
            return
        sparse = config.sparse_attention_config
        if config.attn_backend != "TRTLLM" or getattr(sparse, "algorithm", None) != "deepseek_v4":
            return
        from tensorrt_llm._utils import get_sm_version

        from .prims_ts_sparse_mla import PrimsTSSparseMlaFmha
        from .registry import get_enabled_fmha_lib_classes

        if get_sm_version() != 100 or PrimsTSSparseMlaFmha not in get_enabled_fmha_lib_classes():
            return
        workers = worker_count(sys.maxsize, local_ranks=local_rank_count(mapping))
        if workers <= 1:
            return
        try:
            self.group = _WorkerGroup(workers)
        except Exception as error:  # noqa: BLE001 -- optional workers fall back to local planning
            if strict():
                raise
            logger.warning(f"PrimTS sparse MLA early precompile skipped: {error!r}")

    def run(self, payloads: Sequence[dict[str, Any]], workers: int) -> bool:
        self.consumed = True
        try:
            if self.group is None:
                return run_precompile_workers(payloads, workers)
            return self.group.run(payloads)
        finally:
            self.group = None

    def close(self) -> None:
        if self.group is not None:
            self.group.close()
            self.group = None


@contextmanager
def precompile_session() -> Iterator[PrecompileSession]:
    """Keep worker ownership scoped to one explicit executor construction."""
    session = PrecompileSession()
    token = _session.set(session)
    try:
        yield session
    finally:
        try:
            session.close()
        finally:
            _session.reset(token)


def start_precompile_workers(config: ModelConfig, mapping: Mapping) -> None:
    session = _session.get()
    if session is not None:
        session.start(config, mapping)


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
