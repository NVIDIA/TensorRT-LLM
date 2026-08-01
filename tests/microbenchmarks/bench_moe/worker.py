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

"""MPI worker loop, checkpointing, and launcher entrypoints for bench_moe."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._utils import local_mpi_rank, mpi_allgather, mpi_rank, mpi_world_size

from . import reporting as _reporting
from .case_runner import _run_one_candidate
from .cli import (
    _BenchmarkContext,
    _build_worker_header,
    _maybe_load_config_file,
    _resolve_benchmark_context,
    parse_args,
)
from .results import _make_skipped_run_result, _make_upstream_skipped_row, _runresult_to_row
from .search import expand_and_prune
from .specs import ConfigSpec, WorkloadSpec
from .timing import _CuptiContext, _try_init_cupti
from .utils import _InputCache, _maybe_print_rank0, _set_device_from_local_rank

_MICROBENCH_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _MICROBENCH_DIR.parent.parent


def _try_import(module_path: str, attr: Optional[str] = None, default: Any = None) -> Any:
    """Import module_path; if attr is given, return getattr(m, attr)."""
    try:
        m = importlib.import_module(module_path)
    except Exception:
        return default
    return m if attr is None else getattr(m, attr, default)


_cloudpickle = _try_import("cloudpickle")
_MPIPoolExecutor = _try_import("mpi4py.futures", "MPIPoolExecutor")

POISON_HERE_PREFIX = "cuda_context_poisoned_after_success"
POISON_UPSTREAM_PREFIX = "cuda_context_poisoned_upstream"
WATCHDOG_UPSTREAM_PREFIX = "watchdog_timeout_upstream"
# Terminal (status="failed") marker for a candidate the watchdog killed for
# exceeding its wall-clock budget. NOT suffixed "_upstream": is_completed_for_resume
# treats status="failed" as terminal, so --resume_from SKIPS it (does not re-attempt
# and re-hang) while still surfacing the hang as a result row with a clear reason.
WATCHDOG_TIMEOUT_PREFIX = "watchdog_timeout"
# Terminal (status="failed") placeholder pre-written for the in-flight candidate
# BEFORE it runs. If the process dies mid-candidate in a way nothing else can
# record -- a CUDA device-side assert that aborts the MPI step, OOM-kill,
# SIGSEGV, node loss -- this persisted row makes the candidate terminal so
# --resume_from skips it and advances. Replaced with the real result on normal
# completion. Like WATCHDOG_TIMEOUT_PREFIX, NOT suffixed "_upstream".
INCOMPLETE_PREFIX = "incomplete"
BENCH_MOE_POISON_EXIT_CODE = 75


def is_completed_for_resume(row: dict[str, Any]) -> bool:
    """Decide whether a previously-recorded row blocks a fresh attempt on resume."""
    status = (row.get("status") or "").lower()
    reason = (row.get("skip_reason") or "").lower()
    if status in {"success", "failed"}:
        return True
    if status == "skipped":
        if reason.endswith("_upstream") or reason.startswith(POISON_UPSTREAM_PREFIX):
            return False
        return True
    return False


def candidate_resume_key(*, workload: Any, config: Any) -> str:
    """Stable string key identifying one ``(workload, config)`` candidate."""

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    num_tokens = _get(workload, "num_tokens", 0) or 0
    backend = _get(config, "backend") or ""
    parallel_mode = _get(config, "parallel_mode") or ""
    comm_method = _get(config, "comm_method") or "AUTO"
    cuda_graph = _get(config, "cuda_graph", True)
    lpmc = _get(config, "use_low_precision_moe_combine", False)
    moe_ep_size = _get(config, "moe_ep_size")
    moe_tp_size = _get(config, "moe_tp_size")
    enable_attention_dp = _get(config, "enable_attention_dp")
    return "|".join(
        [
            f"nt={int(num_tokens)}",
            f"backend={str(backend).upper()}",
            f"pmode={str(parallel_mode).upper()}",
            f"comm={str(comm_method).upper()}",
            f"cg={int(bool(cuda_graph))}",
            f"lpmc={int(bool(lpmc))}",
            f"ep={moe_ep_size if moe_ep_size is not None else '-'}",
            f"tp={moe_tp_size if moe_tp_size is not None else '-'}",
            f"adp={'-' if enable_attention_dp is None else int(bool(enable_attention_dp))}",
        ]
    )


def load_resume_payload(
    path: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Load an existing JSON for resume.

    Returns ``(completed_by_key, rows_to_carry_forward)``. Only terminal rows
    are indexed and carried; upstream-skipped placeholders are dropped so they
    get re-attempted.
    """
    if not path or not os.path.exists(path):
        return {}, []
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as exc:
        if mpi_rank() == 0:
            print(
                f"[bench_moe] --resume_from={path}: failed to read existing JSON "
                f"({type(exc).__name__}: {exc}); starting from scratch.",
                flush=True,
            )
        return {}, []
    rows = payload.get("results") or []
    completed: Dict[str, Dict[str, Any]] = {}
    keep: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if is_completed_for_resume(row):
            key = candidate_resume_key(
                workload=row.get("workload") or {},
                config=row.get("requested_config") or {},
            )
            completed[key] = row
            keep.append(row)
    return completed, keep


def cuda_poison_self_check() -> Optional[str]:
    """Return a non-empty reason if the current CUDA context has a sticky error."""
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - belt and braces
        return f"{type(exc).__name__}: {exc}"
    return None


def allreduce_poison_reason(local_reason: Optional[str]) -> Optional[str]:
    """Gather poison reasons across MPI ranks and return a shared summary."""
    try:
        gathered: List[Optional[str]] = mpi_allgather(local_reason)
    except Exception as exc:
        return local_reason or f"mpi_allgather failed: {type(exc).__name__}: {exc}"
    bad: List[Tuple[int, str]] = [(idx, reason) for idx, reason in enumerate(gathered) if reason]
    if not bad:
        return None
    return "; ".join(f"rank{rank}={reason}" for rank, reason in bad)


class CandidateWatchdog:
    """Hard wall-clock guard around one candidate; SIGKILLs the process on timeout.

    On timeout the guard first invokes ``on_timeout`` (used to record the hung
    candidate as a terminal ``failed`` result + checkpoint so it is not silently
    lost and is skipped on ``--resume_from`` rather than re-attempted), then
    SIGKILLs to break the wedged CUDA/NCCL state. A genuine hang cannot be
    recovered in-process, so the kill is unavoidable; ``on_timeout`` makes it a
    recorded outcome instead of a vanished one.
    """

    def __init__(
        self,
        budget_s: float,
        label: str,
        on_timeout: Optional[Callable[[], None]] = None,
        rank0_persist_grace_s: float = 8.0,
    ):
        self._budget_s = float(budget_s)
        self._label = label
        self._on_timeout = on_timeout
        # Non-rank-0 ranks wait this long before SIGKILL so rank 0 can persist the
        # checkpoint before the first task exit tears down the whole srun step.
        self._rank0_persist_grace_s = float(rank0_persist_grace_s)
        self._cancelled = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "CandidateWatchdog":
        if self._budget_s <= 0:
            return self
        self._cancelled.clear()
        self._thread = threading.Thread(
            target=self._guard,
            name="bench_moe-watchdog",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cancelled.set()
        # Do not join: joining here could deadlock if the watchdog already
        # fired while CUDA/NCCL was still draining.
        return False

    def _guard(self) -> None:
        if self._cancelled.wait(self._budget_s):
            return
        rank = mpi_rank()
        try:
            sys.stderr.write(
                f"[bench_moe watchdog] candidate '{self._label}' exceeded "
                f"{self._budget_s:.1f}s budget on pid={os.getpid()} "
                f"rank={rank}; recording it as a failed (timeout) result, then "
                f"sending SIGKILL to break suspected NCCL deadlock or CUDA hang.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        # Record the hung candidate as a terminal failed row + checkpoint. Rank 0
        # writes; other ranks no-op (see _emit_checkpoint_report). The main thread
        # is blocked in a GIL-releasing CUDA call, so this watchdog thread can run.
        if self._on_timeout is not None:
            try:
                self._on_timeout()
            except Exception as exc:  # never let bookkeeping block the kill
                try:
                    sys.stderr.write(
                        f"[bench_moe watchdog] on_timeout callback failed "
                        f"({type(exc).__name__}: {exc}); killing anyway.\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
        # Let rank 0 flush its checkpoint before the first SIGKILL aborts the step.
        if rank != 0 and self._rank0_persist_grace_s > 0:
            time.sleep(self._rank0_persist_grace_s)
        os.kill(os.getpid(), signal.SIGKILL)


def _emit_checkpoint_report(
    *,
    args: argparse.Namespace,
    ctx: "_BenchmarkContext",
    rows: List[Dict[str, Any]],
    world_size: int,
) -> None:
    """Persist a JSON snapshot of all completed rows after a candidate finishes.

    Only rank 0 writes; other ranks no-op. Uses tmp + ``os.replace`` so a
    crash mid-dump cannot leave a half-written JSON. Called from the main
    candidate loop so a sticky-error / watchdog crash cannot lose prior work.
    """
    if mpi_rank() != 0:
        return
    out_path = getattr(args, "output_file", None)
    if not out_path:
        return
    payload = _reporting.build_report_payload(
        ctx=ctx,
        rows=rows,
        world_size=world_size,
        cuda_graph_default=bool(args.cuda_graph),
        repo_root=_REPO_ROOT,
    )
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".checkpoint.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


def _load_and_broadcast_resume_state(
    rank: int, resume_path: Optional[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Load ``--resume_from`` payload on rank 0 and broadcast to all ranks.

    Returns ``(resumed_by_key, resumed_rows)``. Returns empty containers when
    ``resume_path`` is unset or the bcast fails (with a logged warning so the
    operator can decide whether divergence matters).
    """
    resumed_by_key: Dict[str, Dict[str, Any]] = {}
    resumed_rows: List[Dict[str, Any]] = []
    if not resume_path:
        return resumed_by_key, resumed_rows
    if rank == 0:
        resumed_by_key, resumed_rows = load_resume_payload(resume_path)
        if resumed_rows:
            print(
                f"[bench_moe] --resume_from={resume_path}: loaded "
                f"{len(resumed_rows)} terminal row(s); they will be skipped.",
                flush=True,
            )
    try:
        resumed_by_key = MPI.COMM_WORLD.bcast(resumed_by_key, root=0)
        resumed_rows = MPI.COMM_WORLD.bcast(resumed_rows, root=0)
    except Exception as exc:
        _maybe_print_rank0(
            f"[bench_moe] resume bcast failed ({type(exc).__name__}: {exc}); "
            "ranks may diverge on which candidates to skip."
        )
    return resumed_by_key, resumed_rows


def _build_flat_plan_for_sweep(
    ctx: Any,
    world_size: int,
    max_configs: Optional[int],
) -> List[Tuple[WorkloadSpec, ConfigSpec, str]]:
    """Flatten the per-workload search expansion into a single linear plan.

    Items are ``(workload, config, kind)`` where ``kind`` is ``"run"`` for
    candidates to execute or ``"prune:<reason>"`` for compile-time rejects
    (e.g. ``backend.can_implement`` returned False). The flattened form lets
    the poison handler emit upstream-skipped placeholders for every
    not-yet-attempted candidate.
    """
    flat_plan: List[Tuple[WorkloadSpec, ConfigSpec, str]] = []
    for workload in ctx.workloads:
        candidates, skipped = expand_and_prune(
            base_config=ctx.base_config,
            search=ctx.search,
            model=ctx.model,
            world_size=world_size,
            act_dtype=ctx.act_dtype,
            max_configs=max_configs,
        )
        for cand, reason in skipped.items():
            flat_plan.append((workload, cand, f"prune:{reason}"))
        for cand in candidates:
            flat_plan.append((workload, cand, "run"))
    return flat_plan


def _handle_cuda_poison_and_exit(
    *,
    any_poison: str,
    args: argparse.Namespace,
    ctx: Any,
    rank: int,
    world_size: int,
    accumulated_rows: List[Dict[str, Any]],
    flat_plan: List[Tuple[WorkloadSpec, ConfigSpec, str]],
    resumed_by_key: Dict[str, Dict[str, Any]],
    next_idx: int,
) -> None:
    """Handle a poisoned CUDA context by checkpointing and exiting cleanly.

    Promotes the offending row to ``failed``, fills upstream placeholders for
    every not-yet-attempted candidate, writes a checkpoint, then exits with
    ``BENCH_MOE_POISON_EXIT_CODE``.

    Once a CUDA context is poisoned (sticky error from a prior candidate)
    every later kernel launch on the same context will fail. The outer
    driver wraps this worker and is responsible for restarting with
    ``--resume_from`` after we exit.

    NOTE: This function calls ``os._exit`` and never returns. Atexit hooks
    are skipped on purpose so NCCL / CUDA do not re-enter on a poisoned
    context and deadlock.
    """
    # Promote the just-finished row to "failed" so a future --resume_from
    # never picks it again. Preserve original instrumentation; record
    # the poison reason for the dashboard.
    if accumulated_rows:
        last = accumulated_rows[-1]
        instr = last.setdefault("instrumentation", {})
        instr["post_run_cuda_poison_reason"] = any_poison
        if (last.get("status") or "").lower() == "success":
            last["status"] = "failed"
            last["skip_reason"] = f"{POISON_HERE_PREFIX}: {any_poison}"

    # Emit upstream-skipped placeholders for every not-yet-attempted
    # candidate so dashboards see the full search space and a fresh
    # --resume_from run knows to re-attempt them.
    placeholder_reason = (
        f"{POISON_UPSTREAM_PREFIX}: prior candidate poisoned the CUDA context ({any_poison})"
    )
    for w2, c2, k2 in flat_plan[next_idx:]:
        key2 = candidate_resume_key(workload=w2, config=c2)
        if key2 in resumed_by_key:
            continue
        if k2.startswith("prune:"):
            # Pruned candidates do not touch CUDA; record their true
            # reason rather than the upstream placeholder.
            pruned = _make_skipped_run_result(
                model=ctx.model,
                workload=w2,
                config=c2,
                world_size=world_size,
                analysis=ctx.analysis,
                reason=k2[len("prune:") :],
            )
            accumulated_rows.append(_runresult_to_row(pruned))
            continue
        accumulated_rows.append(
            _make_upstream_skipped_row(
                model=ctx.model,
                workload=w2,
                config=c2,
                world_size=world_size,
                analysis=ctx.analysis,
                reason=placeholder_reason,
            )
        )

    _emit_checkpoint_report(args=args, ctx=ctx, rows=accumulated_rows, world_size=world_size)
    if rank == 0:
        sys.stderr.write(
            f"[bench_moe] CUDA context poisoned ({any_poison}); "
            f"checkpointed {len(accumulated_rows)} row(s) to "
            f"{args.output_file!r}; exiting with code "
            f"{BENCH_MOE_POISON_EXIT_CODE} so the outer driver can "
            "restart with --resume_from.\n"
        )
        sys.stderr.flush()
    # ``os._exit`` (not ``sys.exit``) so Python atexit hooks do not
    # re-enter NCCL / CUDA on a poisoned context and deadlock.
    os._exit(BENCH_MOE_POISON_EXIT_CODE)


def _write_final_report(
    *,
    args: argparse.Namespace,
    ctx: Any,
    rows: List[Dict[str, Any]],
    world_size: int,
) -> None:
    """Produce the final report payload and write it to disk + workbook.

    Two paths:
    - ``args.output_file`` set: atomic write (tmp + rename), plus an analysis
      workbook next to it.
    - No ``output_file``: echo a rankings-only summary on stdout for headless
      invocations, optionally writing a workbook if explicitly requested.

    Caller must ensure this only runs on rank 0 -- all other ranks should
    skip the final report step entirely.
    """
    _reporting.write_final_report(
        args=args,
        ctx=ctx,
        rows=rows,
        world_size=world_size,
        repo_root=_REPO_ROOT,
    )


def _run_benchmark_worker_under_current_mpi(args: argparse.Namespace, launcher: str) -> None:
    args = _maybe_load_config_file(args)
    ctx = _resolve_benchmark_context(args)

    # nsys capture and the CUPTI kernel breakdown both use CUPTI and cannot run
    # at the same time; --nsys wins and disables the kernel breakdown.
    if getattr(args, "nsys", False) and "kernels" in ctx.analysis:
        _maybe_print_rank0(
            "[bench_moe] --nsys disables the CUPTI kernel breakdown (--analysis kernels); "
            "the profiled region is captured by nsys instead."
        )
        ctx = dataclasses.replace(ctx, analysis=tuple(a for a in ctx.analysis if a != "kernels"))

    # CUPTI MUST be initialized before the CUDA context is created.
    _early_cupti_ctx: Optional[_CuptiContext] = None
    if args.cuda_graph and "kernels" in ctx.analysis:
        _cupti_init = _try_init_cupti()
        if _cupti_init.ok:
            _early_cupti_ctx = _cupti_init

    tllm.logger.set_level("error")

    world_size = mpi_world_size()
    rank = mpi_rank()
    _set_device_from_local_rank()
    device = torch.device("cuda")
    seed = int(args.random_seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Early header (rank 0) for stdout consumers.
    if rank == 0:
        print(json.dumps(_build_worker_header(ctx, launcher, world_size), indent=2), flush=True)

    resumed_by_key, resumed_rows = _load_and_broadcast_resume_state(
        rank=rank, resume_path=getattr(args, "resume_from", None)
    )
    flat_plan = _build_flat_plan_for_sweep(ctx, world_size, args.max_configs)

    # Accumulated rows include resumed rows (preserved as-is) plus rows
    # produced this run. ``_build_report_payload`` consumes this directly.
    accumulated_rows: List[Dict[str, Any]] = list(resumed_rows)
    input_cache: _InputCache = {}

    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 1) or 0))
    candidates_since_checkpoint = 0
    watchdog_budget_s = float(getattr(args, "per_candidate_timeout_s", 0.0) or 0.0)

    deadline: Optional[float] = None
    if args.time_budget_minutes is not None and args.time_budget_minutes > 0:
        deadline = time.monotonic() + args.time_budget_minutes * 60.0

    for idx, (workload, cand, kind) in enumerate(flat_plan):
        key = candidate_resume_key(workload=workload, config=cand)
        if key in resumed_by_key:
            # E: short-circuit. The resumed row is already in accumulated_rows.
            continue

        if kind.startswith("prune:"):
            reason = kind[len("prune:") :]
            r = _make_skipped_run_result(
                model=ctx.model,
                workload=workload,
                config=cand,
                world_size=world_size,
                analysis=ctx.analysis,
                reason=reason,
            )
            accumulated_rows.append(_runresult_to_row(r))
            continue

        if deadline is not None and time.monotonic() > deadline:
            _maybe_print_rank0(
                "[bench_moe] --time_budget_minutes exceeded; remaining candidates "
                "will be reported as skipped."
            )
            r = _make_skipped_run_result(
                model=ctx.model,
                workload=workload,
                config=cand,
                world_size=world_size,
                analysis=ctx.analysis,
                reason="time_budget_exceeded",
            )
            accumulated_rows.append(_runresult_to_row(r))
            continue

        case_label = (
            f"backend={cand.backend} parallel_mode={cand.parallel_mode} "
            f"comm={cand.comm_method} num_tokens={workload.num_tokens}"
        )
        _maybe_print_rank0(f"[bench_moe] running {case_label}")

        # Pre-write a terminal "failed" placeholder for THIS candidate and
        # checkpoint it BEFORE running. If the process then dies mid-candidate in
        # a way nothing else can catch -- a CUDA device-side assert that aborts
        # the MPI step, OOM-kill, SIGSEGV, node loss -- this persisted row keeps
        # the candidate terminal, so --resume_from skips it and advances to the
        # next one instead of re-attempting (and re-crashing on) the same
        # candidate forever. On normal completion it is replaced with the real
        # result below. Only the in-flight candidate gets a placeholder;
        # not-yet-run candidates have no row and are still attempted on resume.
        placeholder = _make_skipped_run_result(
            model=ctx.model,
            workload=workload,
            config=cand,
            world_size=world_size,
            analysis=ctx.analysis,
            reason=(
                f"{INCOMPLETE_PREFIX}: process died before this candidate "
                f"finished (crash/abort/OOM/kill) ({case_label})"
            ),
        )
        placeholder.status = "failed"
        placeholder.status_per_rank = {f"rank{i}": "incomplete" for i in range(world_size)}
        accumulated_rows.append(_runresult_to_row(placeholder))
        _emit_checkpoint_report(args=args, ctx=ctx, rows=accumulated_rows, world_size=world_size)

        # If the watchdog fires (suspected hang), overwrite the placeholder's
        # reason with the precise timeout text and re-checkpoint before SIGKILL,
        # so the hang is surfaced as a clear result (the row is already terminal).
        def _record_watchdog_timeout(
            _label: str = case_label,
            _budget_s: float = watchdog_budget_s,
        ) -> None:
            if accumulated_rows:
                accumulated_rows[-1]["skip_reason"] = (
                    f"{WATCHDOG_TIMEOUT_PREFIX}: exceeded {_budget_s:.0f}s; "
                    f"suspected NCCL/CUDA hang ({_label})"
                )
                accumulated_rows[-1]["status_per_rank"] = {
                    f"rank{i}": "timeout" for i in range(world_size)
                }
            _emit_checkpoint_report(
                args=args, ctx=ctx, rows=accumulated_rows, world_size=world_size
            )

        # Hard wall-clock guard around the actual candidate execution.
        with CandidateWatchdog(watchdog_budget_s, case_label, on_timeout=_record_watchdog_timeout):
            with torch.device(device):
                r = _run_one_candidate(
                    model=ctx.model,
                    workload=workload,
                    config=cand,
                    world_size=world_size,
                    rank=rank,
                    device=device,
                    act_dtype=ctx.act_dtype,
                    routing_logits_dtype=ctx.routing_logits_dtype,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    fast_autotune=bool(args.fast_autotune),
                    analysis=ctx.analysis,
                    cupti_ctx=_early_cupti_ctx,
                    random_seed=int(args.random_seed),
                    input_cache=input_cache,
                    enable_perfect_router_requested=bool(args.enable_perfect_router),
                    nsys=bool(getattr(args, "nsys", False)),
                )
        # Candidate finished normally: replace the pre-written placeholder (the
        # last row) with the real result.
        row = _runresult_to_row(r)
        accumulated_rows[-1] = row
        if rank == 0:
            print(json.dumps(row, indent=2), flush=True)

        # Sticky CUDA error detection + lockstep exit across all ranks.
        local_poison = cuda_poison_self_check()
        any_poison = allreduce_poison_reason(local_poison)
        if any_poison is not None:
            _handle_cuda_poison_and_exit(
                any_poison=any_poison,
                args=args,
                ctx=ctx,
                rank=rank,
                world_size=world_size,
                accumulated_rows=accumulated_rows,
                flat_plan=flat_plan,
                resumed_by_key=resumed_by_key,
                next_idx=idx + 1,
            )
            # _handle_cuda_poison_and_exit calls os._exit; control never returns.

        # Incremental checkpoint so a future watchdog SIGKILL never loses
        # already-completed rows.
        candidates_since_checkpoint += 1
        if checkpoint_every > 0 and candidates_since_checkpoint >= checkpoint_every:
            _emit_checkpoint_report(
                args=args, ctx=ctx, rows=accumulated_rows, world_size=world_size
            )
            candidates_since_checkpoint = 0

    if rank == 0:
        _write_final_report(args=args, ctx=ctx, rows=accumulated_rows, world_size=world_size)


# ---------------------------------------------------------------------------
# MPI launchers
# ---------------------------------------------------------------------------


_WORKER_ENV = {
    "TRTLLM_CAN_USE_DEEP_EP": "1",
    "TRTLLM_ENABLE_PDL": "0",
}


def _spawn_worker_main(args_blob: bytes) -> List[Dict[str, Any]]:
    args = pickle.loads(args_blob)
    try:
        _run_benchmark_worker_under_current_mpi(args, launcher="spawn")
    except Exception as exc:
        rank = mpi_rank()
        size = mpi_world_size()
        msg = (
            "[bench_moe worker] uncaught exception:\n"
            f"rank={rank}/{size} local_rank={local_mpi_rank()} pid={os.getpid()}\n"
            f"{traceback.format_exc()}"
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        raise RuntimeError(msg) from exc
    return []


def main() -> None:
    args = parse_args()

    external_world_size = mpi_world_size()
    if external_world_size > 1:
        if args.world_size is not None and int(args.world_size) != external_world_size:
            raise ValueError(
                f"--world_size ({args.world_size}) must match external MPI world size "
                f"({external_world_size}) under external mpirun/srun."
            )
        os.environ.update(_WORKER_ENV)
        _run_benchmark_worker_under_current_mpi(args, launcher="external_mpi")
        return

    world_size = int(args.world_size or 1)
    if world_size <= 0:
        raise ValueError("--world_size must be > 0")

    if world_size == 1:
        if mpi_rank() == 0:
            print(
                json.dumps(
                    {
                        "bench": "bench_moe",
                        "launcher": "inline_single_rank",
                        "world_size": 1,
                        "model": args.model,
                        "backend": args.backend,
                        "search": args.search,
                    },
                    indent=2,
                ),
                flush=True,
            )
        os.environ.update(_WORKER_ENV)
        args.world_size = 1
        _run_benchmark_worker_under_current_mpi(args, launcher="inline_single_rank")
        return

    if _cloudpickle is None or _MPIPoolExecutor is None:
        missing = [
            name
            for name, mod in (("cloudpickle", _cloudpickle), ("mpi4py.futures", _MPIPoolExecutor))
            if mod is None
        ]
        raise RuntimeError(
            f"--world_size > 1 self-spawn launcher requires {', '.join(missing)}; "
            "either install the missing package(s) or run the benchmark under mpirun/srun."
        )

    _cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(  # type: ignore[attr-defined]
        _cloudpickle.dumps,
        _cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )

    if mpi_rank() == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe",
                    "launcher": "spawn",
                    "world_size": world_size,
                    "model": args.model,
                    "backend": args.backend,
                    "search": args.search,
                },
                indent=2,
            ),
            flush=True,
        )

    args_blob = _cloudpickle.dumps(args)
    executor = _MPIPoolExecutor(max_workers=world_size, env=_WORKER_ENV)
    try:
        _ = list(executor.map(_spawn_worker_main, [args_blob] * world_size))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
