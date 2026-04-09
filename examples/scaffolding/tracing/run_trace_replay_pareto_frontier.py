r"""Replay a single compact trace concurrently while sweeping max_batch_size.

For each ladder step, uses n_sessions == max_batch_size (same value), restarts
``TRTLLMWorker``/``LLM`` with that ``max_batch_size``, and records throughput
and latency statistics suitable for Pareto plots (e.g. median tok/s per session
vs aggregate tok/s per GPU).

Default parallelism matches a typical ``trtllm-serve`` MoE deployment:
``tensor_parallel_size=4`` (``--tp_size 4``) and ``moe_expert_parallel_size=4``
(``--ep_size 4``). For dense models, pass ``--moe-expert-parallel-size 0`` to
omit expert parallel.

For MoE **DEP** (attention data parallel + expert parallel), use the same
``--tensor-parallel-size`` / ``--moe-expert-parallel-size`` as **TEP**, and add
``--enable-attention-dp`` (maps to ``LlmArgs.enable_attention_dp`` /
``parallel_config.yaml`` ``enable_attention_dp: true``).

Example::

    python examples/scaffolding/run_trace_replay_pareto_frontier.py \\
        --trace-dir /path/to/resolved_traces/django__django-14787 \\
        --model-dir $LLM_MODELS_ROOT/YourMoEModel

By default the JSON is written under the trace directory in a run subfolder
``<trace_dir>/<trace_dir.name>_<model_basename>_tp<TP>_ep<EP>_<YYYYMMDD_HHMMSS>/``
with basename matching that folder (same stem for ``.json`` and ``.log``).
Override with ``--output-json``. Console ``print`` output and stderr (including typical
``logging`` stream output) are also copied to a sibling ``.log`` file with the
same basename and directory as that JSON path.

During the ladder sweep, results are checkpointed so an interrupt or crash does not
lose completed steps: ``<stem>.partial.json`` is rewritten atomically after each
ladder step (full report object, same schema as the final ``.json``), and each
completed run row is appended as one line of JSON to ``<stem>.ladder_runs.jsonl``
(header line first). The final ``<stem>.json`` is still written when the run
finishes successfully.

Requires GPU(s) and a local model at ``--model-dir``. Set ``LLM_MODELS_ROOT``
when using standard model layouts.

After the JSON report is written, two PNGs are saved next to it (unless
``--no-pareto-png`` is set):

* ``<stem>_throughput_pareto.png`` — ``median_tps_per_user`` vs
  ``output_tps_per_gpu`` (throughput view; tokens from **replay-measured**
  counts); same plot as ``plot_trace_replay_token_pareto.py``.
* ``<stem>_agent_pareto.png`` — ``tasks/user/hour`` vs ``tasks/gpu/hour`` (same
  definitions as ``plot_trace_replay_agent_pareto.py``: mean session latency
  and wall-clock batch completion).

Default run folder and JSON stem (when ``--output-json`` is omitted) are
``<trace_dir.name>_<model_basename>_tp<TP>_ep<EP>_<timestamp>`` so reports and
PNG siblings encode model and parallelism without collisions.

If a ladder step fails with out-of-memory,
larger ladder values are not executed; those steps are recorded in ``runs`` with
``status=skipped_after_prior_oom`` and a summary is stored under
``pareto_ladder_oom``.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import json
import os
import platform
import shlex
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SchedulerConfig
from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace, TraceEvent
from tensorrt_llm.scaffolding.replay import ReplayEngine, ReplayGenerationStats
from tensorrt_llm.scaffolding.task import GenerationTask, TaskStatus
from tensorrt_llm.scaffolding.worker import TRTLLMWorker


def _load_plot_helper(module_stem: str, symbol_name: str) -> Optional[Any]:
    """Load optional plotting helper from a sibling script by filename."""
    module_path = Path(__file__).resolve().parent / f"{module_stem}.py"
    if not module_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_stem, str(module_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        symbol = getattr(module, symbol_name, None)
        return symbol if callable(symbol) else None
    except Exception:
        return None


class _TeeTextIO:
    """Duplicate writes to a primary stream (e.g. console) and a log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        self._secondary.write(s)
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


class ReplayTRTLLMWorker(TRTLLMWorker):
    """Subclass used only for trace replay: route token-id prompts into :class:`LLM`.

    Base :class:`TRTLLMWorker` always calls ``generate_async(task.input_str)``.
    :class:`ReplayEngine` fills ``GenerationTask.input_tokens`` (replay uses
    lengths from the trace, not original text) and leaves ``input_str`` unset,
    so the base worker would pass ``None`` into the runtime. The LLM API
    accepts ``List[int]`` as a token prompt; this subclass forwards
    ``input_tokens`` when present. Fixing this in the shared worker would
    change behavior for all call sites; keeping a narrow subclass avoids that.

    .. note::

        :meth:`Worker.run_task` dispatches via ``type(worker).task_handlers``.
        We must override ``task_handlers`` so ``GenerationTask`` maps to *this*
        class's ``generation_handler``; otherwise the inherited dict still
        references :class:`TRTLLMWorker`'s handler and passes ``None``.
    """

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        sampling_params = self.convert_task_params(task)
        if task.input_tokens is not None:
            inputs: Union[str, List[int]] = task.input_tokens
        elif task.input_str is not None:
            inputs = task.input_str
        else:
            return TaskStatus.WORKER_EXECEPTION

        if task.streaming_output_flag:
            result = self.llm.generate_async(
                inputs,
                sampling_params=sampling_params,
                streaming=True,
            )
            await self.streaming_generate_helper(result, None, task.streaming_output_list)
        else:
            result = await self.llm.generate_async(inputs, sampling_params=sampling_params)

        self.fill_task_with_result(task, result)
        return TaskStatus.SUCCESS

    task_handlers = {
        **TRTLLMWorker.task_handlers,
        GenerationTask: generation_handler,
    }


def parse_ladder(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def find_compact_trace_file(trace_dir: Path) -> Path:
    """Prefer ``*.trace.json`` over ``*.full.trace.json``."""
    all_traces = sorted(trace_dir.glob("*.trace.json"))
    if not all_traces:
        raise FileNotFoundError(f"No *.trace.json under {trace_dir}")
    for p in all_traces:
        if ".full.trace" not in p.name:
            return p
    return all_traces[0]


def count_assistant_completion_tokens(events: List[TraceEvent]) -> int:
    total = 0
    for ev in events:
        if ev.event_type == "message" and ev.role == "assistant":
            total += ev.completion_tokens or 0
    return total


def count_parallel_regions(events: List[TraceEvent]) -> Dict[str, int]:
    starts = sum(1 for e in events if e.event_type == "parallel_start")
    ends = sum(1 for e in events if e.event_type == "parallel_end")
    return {"parallel_start": starts, "parallel_end": ends}


def summarize_trace_events(events: List[TraceEvent]) -> Dict[str, Any]:
    """Aggregate trace structure for JSON (token budgets, tools, roles)."""
    event_type_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    assistant_turns = 0
    prompt_tokens_assistant_sum = 0
    completion_tokens_sum = 0
    reasoning_tokens_sum = 0
    tool_calls = 0
    tool_duration_ms_sum = 0.0
    tool_duration_ms_max = 0.0
    message_tokens_sum = 0
    drop_kv = 0

    for ev in events:
        et = ev.event_type or ""
        event_type_counts[et] = event_type_counts.get(et, 0) + 1
        if et == "message" and ev.role:
            role_counts[ev.role] = role_counts.get(ev.role, 0) + 1
            if ev.tokens:
                message_tokens_sum += ev.tokens
        if et == "message" and ev.role == "assistant":
            assistant_turns += 1
            prompt_tokens_assistant_sum += ev.prompt_tokens or 0
            completion_tokens_sum += ev.completion_tokens or 0
            reasoning_tokens_sum += ev.reasoning_tokens or 0
        if et == "tool_call":
            tool_calls += 1
            d = ev.duration_ms or 0.0
            tool_duration_ms_sum += d
            tool_duration_ms_max = max(tool_duration_ms_max, d)
        if et == "drop_kv_cache":
            drop_kv += 1

    return {
        "event_type_counts": event_type_counts,
        "message_role_counts": role_counts,
        "assistant_turns": assistant_turns,
        "prompt_tokens_assistant_sum": prompt_tokens_assistant_sum,
        "completion_tokens_sum": completion_tokens_sum,
        "reasoning_tokens_sum": reasoning_tokens_sum,
        "non_assistant_message_tokens_sum": message_tokens_sum,
        "tool_call_count": tool_calls,
        "tool_call_duration_ms_sum": tool_duration_ms_sum,
        "tool_call_duration_ms_mean": (tool_duration_ms_sum / tool_calls) if tool_calls else None,
        "tool_call_duration_ms_max": tool_duration_ms_max,
        "replay_tool_sleep_wall_s_estimated": tool_duration_ms_sum / 1000.0,
        "drop_kv_cache_events": drop_kv,
    }


def collect_host_info() -> Dict[str, Any]:
    """Runtime, CUDA, and optional GPU memory snapshot for reproducibility."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }
    # Select env vars that commonly affect GPU / TRT-LLM
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "LLM_MODELS_ROOT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "OMP_NUM_THREADS",
        "SCAFFOLDING_DETERMINISTIC",
    )
    info["env_subset"] = {k: os.environ.get(k) for k in env_keys if k in os.environ}

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = (
            str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        )
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            info["cuda_device_count"] = n
            devices = []
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                free_b, total_b = 0, 0
                try:
                    free_b, total_b = torch.cuda.mem_get_info(i)
                except Exception:
                    pass
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_bytes": props.total_memory,
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count,
                        "mem_get_info_free_bytes": free_b,
                        "mem_get_info_total_bytes": total_b,
                    }
                )
            info["cuda_devices"] = devices
    except Exception as exc:
        info["torch_error"] = repr(exc)

    try:
        import tensorrt_llm

        info["tensorrt_llm_version"] = getattr(tensorrt_llm, "__version__", "unknown")
    except Exception as exc:
        info["tensorrt_llm_import_error"] = repr(exc)

    return info


def collect_trace_file_stats(trace_path: Path) -> Dict[str, Any]:
    st = trace_path.stat()
    return {
        "trace_file_name": trace_path.name,
        "trace_file_size_bytes": st.st_size,
        "trace_file_mtime_iso": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
    }


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Full CLI snapshot (Path -> str)."""
    d = vars(args).copy()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


async def one_replay(
    worker: ReplayTRTLLMWorker,
    trace: ExecutionTrace,
    *,
    session_index: int,
    n_sessions: int,
    ladder_step: int,
) -> Tuple[float, Dict[str, Any]]:
    label = f"[ladder N=B={ladder_step}] session {session_index + 1}/{n_sessions}"
    print(f"{label}: replay start", flush=True)
    stats = ReplayGenerationStats()
    t0 = time.perf_counter()
    await ReplayEngine(worker, generation_stats=stats).launch_trace(trace)
    elapsed = time.perf_counter() - t0
    print(f"{label}: replay done in {elapsed:.3f}s", flush=True)
    return elapsed, {
        "replay_output_token_sum": stats.sum_replay_output_tokens(),
        "trace_completion_token_sum": stats.sum_trace_completion_tokens(),
        # Full per-turn detail only for session 0 to avoid duplicating in JSON.
        "per_assistant_generation_detail": stats.entries if session_index == 0 else [],
    }


def try_cuda_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def instantiate_replay_worker(
    model_dir: str, max_batch_size: int, args: argparse.Namespace
) -> ReplayTRTLLMWorker:
    """Build :class:`LLM` like ``TRTLLMWorker.init_with_new_llm`` plus TP/PP/MoE flags."""
    scheduler_config = SchedulerConfig()
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        legacy=False,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=False,
        use_fast=True,
    )

    llm_kw: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "backend": args.backend,
        "disable_overlap_scheduler": args.disable_overlap_scheduler,
        "kv_cache_config": kv_cache_config,
        "max_batch_size": max_batch_size,
        "max_num_tokens": args.max_num_tokens,
        "scheduler_config": scheduler_config,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.pipeline_parallel_size != 1:
        llm_kw["pipeline_parallel_size"] = args.pipeline_parallel_size
    # EP>0 only (dense models: --moe-expert-parallel-size 0)
    if args.moe_expert_parallel_size > 0:
        llm_kw["moe_expert_parallel_size"] = args.moe_expert_parallel_size
    if args.moe_tensor_parallel_size is not None and args.moe_tensor_parallel_size > 0:
        llm_kw["moe_tensor_parallel_size"] = args.moe_tensor_parallel_size
    if getattr(args, "enable_attention_dp", False):
        llm_kw["enable_attention_dp"] = True
    if getattr(args, "disable_autotuner", False):
        llm_kw["enable_autotuner"] = False

    llm = LLM(model_dir, **llm_kw)
    worker = ReplayTRTLLMWorker(llm, tokenizer)
    worker.own_llm = True
    return worker


def llm_config_record(args: argparse.Namespace) -> Dict[str, Any]:
    """Serializable LLM-related settings for the JSON report."""
    return {
        "backend": args.backend,
        "max_num_tokens": args.max_num_tokens,
        "kv_cache_free_gpu_memory_fraction": args.kv_cache_free_gpu_memory_fraction,
        "disable_overlap_scheduler": args.disable_overlap_scheduler,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "moe_expert_parallel_size": args.moe_expert_parallel_size,
        "moe_tensor_parallel_size": args.moe_tensor_parallel_size,
        "enable_attention_dp": getattr(args, "enable_attention_dp", False),
        "enable_autotuner": not getattr(args, "disable_autotuner", False),
        "scheduler_config_note": "SchedulerConfig() defaults (see TensorRT-LLM)",
        "kv_cache_config": {
            "free_gpu_memory_fraction": args.kv_cache_free_gpu_memory_fraction,
        },
    }


async def run_one_point(
    *,
    trace: ExecutionTrace,
    model_dir: str,
    n_sessions: int,
    max_batch_size: int,
    args: argparse.Namespace,
    ladder_index: int,
    ladder_step: int,
) -> Dict[str, Any]:
    """Run ``n_sessions`` concurrent replays; ``max_batch_size`` for new LLM."""
    worker: Optional[ReplayTRTLLMWorker] = None
    worker = instantiate_replay_worker(model_dir, max_batch_size, args)
    row: Dict[str, Any] = {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "n_sessions": n_sessions,
        "max_batch_size": max_batch_size,
        "status": "success",
        "error": None,
        "error_traceback": None,
    }
    try:
        wall_t0 = time.perf_counter()
        replay_results = await asyncio.gather(
            *[
                one_replay(
                    worker,
                    trace,
                    session_index=i,
                    n_sessions=n_sessions,
                    ladder_step=ladder_step,
                )
                for i in range(n_sessions)
            ]
        )
        wall_s = time.perf_counter() - wall_t0

        durations = [r[0] for r in replay_results]
        metas = [r[1] for r in replay_results]

        tokens_per_trace_trace_metadata = count_assistant_completion_tokens(trace.events)
        total_out_tokens_trace_metadata = float(n_sessions * tokens_per_trace_trace_metadata)

        per_session_replay_output = [m["replay_output_token_sum"] for m in metas]
        per_session_trace_completion = [m["trace_completion_token_sum"] for m in metas]
        total_out_tokens_replay_actual = float(sum(per_session_replay_output))

        detail_session0 = metas[0]["per_assistant_generation_detail"]

        tp_sizes = [
            per_session_replay_output[i] / durations[i]
            for i in range(len(durations))
            if durations[i] > 0
        ]
        row.update(
            {
                "wall_clock_s": wall_s,
                # From trace file ``completion_tokens`` (original recording).
                "assistant_output_tokens_per_trace": tokens_per_trace_trace_metadata,
                "total_output_tokens_trace_metadata": total_out_tokens_trace_metadata,
                # Kept for backward compatibility: same as total_output_tokens_trace_metadata.
                "total_output_tokens_estimated": total_out_tokens_trace_metadata,
                # Measured during this replay (decoder output token ids per session).
                "per_session_replay_output_token_sum": per_session_replay_output,
                # Same as per_session_replay_output_token_sum (explicit total per session).
                "per_session_total_output_tokens": list(per_session_replay_output),
                "per_session_trace_completion_token_sum": per_session_trace_completion,
                "assistant_output_tokens_per_trace_replay_actual_mean": (
                    statistics.mean(per_session_replay_output)
                    if per_session_replay_output
                    else None
                ),
                "total_output_tokens_replay_actual": total_out_tokens_replay_actual,
                "replay_assistant_generations_detail_session0": detail_session0,
                "session_duration_s": durations,
                "session_duration_min_s": min(durations) if durations else None,
                "session_duration_max_s": max(durations) if durations else None,
                "session_duration_sum_s": sum(durations) if durations else None,
                "session_duration_stdev_s": statistics.stdev(durations)
                if len(durations) > 1
                else 0.0,
                "session_duration_p50_s": statistics.median(durations) if durations else None,
                "session_duration_p90_s": _percentile(durations, 0.9) if durations else None,
                "session_duration_p99_s": _percentile(durations, 0.99) if durations else None,
                "session_duration_mean_s": statistics.mean(durations) if durations else None,
                "session_duration_cv": (statistics.stdev(durations) / statistics.mean(durations))
                if len(durations) > 1 and statistics.mean(durations) > 0
                else None,
                "aggregate_latency_person_s": sum(durations) if durations else None,
                "median_tps_per_user": statistics.median(tp_sizes) if tp_sizes else None,
                "mean_tps_per_user": statistics.mean(tp_sizes) if tp_sizes else None,
                "min_tps_per_user": min(tp_sizes) if tp_sizes else None,
                "max_tps_per_user": max(tp_sizes) if tp_sizes else None,
                "output_tps_aggregate": total_out_tokens_replay_actual / wall_s
                if wall_s > 0
                else None,
                "output_tokens_per_wall_s_per_session_mean": (
                    total_out_tokens_replay_actual / wall_s / n_sessions
                )
                if wall_s > 0 and n_sessions
                else None,
                "mean_tps_per_user_session_time": (total_out_tokens_replay_actual / sum(durations))
                if durations and sum(durations) > 0
                else None,
            }
        )
        tp = args.tensor_parallel_size
        row["output_tps_per_gpu"] = (
            row["output_tps_aggregate"] / tp
            if row["output_tps_aggregate"] is not None and tp > 0
            else None
        )
        row["pareto_x_median_tps_per_user"] = row.get("median_tps_per_user")
        row["pareto_y_output_tps_per_gpu"] = row.get("output_tps_per_gpu")
        # Alternate normalizations for plotting
        row["output_tps_per_aggregate_1gpu_equiv"] = row.get("output_tps_aggregate")
        try:
            import torch

            if torch.cuda.is_available():
                ng = torch.cuda.device_count()
                row["output_tps_per_cuda_device_count"] = (
                    row["output_tps_aggregate"] / ng
                    if row["output_tps_aggregate"] is not None and ng > 0
                    else None
                )
        except Exception:
            pass
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["error_traceback"] = traceback.format_exc()
        if _is_oom_exception(exc):
            row["error_kind"] = "out_of_memory"
    finally:
        if worker is not None:
            worker.shutdown()
        gc.collect()
        try_cuda_empty_cache()

    return row


def _percentile(data: List[float], q: float) -> float:
    if not data:
        raise ValueError("empty data")
    s = sorted(data)
    idx = int(round(q * (len(s) - 1)))
    return s[idx]


def _is_oom_exception(exc: BaseException) -> bool:
    """Heuristic: CUDA OOM, host MemoryError, or common runtime OOM strings."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        oom_cls2 = getattr(torch, "OutOfMemoryError", None)
        if oom_cls2 is not None and isinstance(exc, oom_cls2):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda out of memory" in msg:
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in msg:
        return True
    return False


def _skipped_row_after_prior_oom(
    *,
    ladder_index: int,
    ladder_step: int,
    prior_oom_error: Optional[str],
    prior_oom_step: int,
    prior_oom_index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Placeholder run row for ladder steps not executed after an earlier OOM."""
    return {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "n_sessions": ladder_step,
        "max_batch_size": ladder_step,
        "status": "skipped_after_prior_oom",
        "error": prior_oom_error,
        "error_kind": "out_of_memory_inherited",
        "error_traceback": None,
        "skipped_reason": (
            "Larger ladder values were not run after out-of-memory at "
            f"ladder_index={prior_oom_index}, ladder_step={prior_oom_step}."
        ),
        "prior_oom_ladder_step": prior_oom_step,
        "prior_oom_ladder_index": prior_oom_index,
        "llm_effective_config": {
            **llm_config_record(args),
            "max_batch_size": ladder_step,
            "n_sessions": ladder_step,
            "ladder_index": ladder_index,
            "ladder_step": ladder_step,
            "note": "not_executed_skipped_after_prior_oom",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pareto-style sweep: n_sessions == max_batch_size on ladder."
    )
    p.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory containing a *.trace.json (e.g. .../django__django-14787).",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("LLM_MODELS_ROOT", ""),
        help="HF model directory for TRTLLMWorker (or set LLM_MODELS_ROOT).",
    )
    p.add_argument(
        "--ladder",
        type=str,
        default="4, 8",
        help="Comma-separated ladder values; each run uses N=B=value.",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        help="LLM backend (e.g. pytorch).",
    )
    p.add_argument(
        "--max-num-tokens",
        type=int,
        default=131072,
        help="Passed to LLM (scheduler token budget).",
    )
    p.add_argument(
        "--kv-cache-free-gpu-memory-fraction",
        type=float,
        default=0.8,
        help="KvCacheConfig.free_gpu_memory_fraction for LLM.",
    )
    p.add_argument(
        "--disable-overlap-scheduler",
        action="store_true",
        default=False,
        help="Disable overlap scheduler on LLM.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size (aligns with trtllm-serve --tp_size).",
    )
    p.add_argument("--pipeline-parallel-size", type=int, default=1)
    p.add_argument(
        "--moe-expert-parallel-size",
        type=int,
        default=4,
        help="MoE expert parallel size (aligns with trtllm-serve --ep_size). "
        "Use 0 for dense models.",
    )
    p.add_argument(
        "--moe-tensor-parallel-size",
        type=int,
        default=None,
        help="MoE tensor parallel size (optional).",
    )
    p.add_argument(
        "--enable-attention-dp",
        action="store_true",
        default=False,
        help=(
            "Attention data parallel (LlmArgs.enable_attention_dp). "
            "With MoE EP, this selects DEP (same tp/ep as TEP). "
            "Aligns with trtllm-serve parallel_config.yaml enable_attention_dp: true.",
        ),
    )
    p.add_argument(
        "--disable-autotuner",
        action="store_true",
        default=False,
        help=(
            "Set LLM enable_autotuner=False. Use when MoE GEMM profile warmup fails "
            "with 'Can't allocate profile workspace' (VRAM pressure with large KV "
            "or attention-DP). Tuning is skipped; throughput may be suboptimal.",
        ),
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop ladder on first failed (N,B) (default: true).",
    )
    p.add_argument(
        "--no-stop-on-error",
        action="store_false",
        dest="stop_on_error",
        help="Continue ladder after failures (e.g. OOM).",
    )
    p.add_argument(
        "--pareto-curve-label",
        type=str,
        default="Pareto Frontier",
        help="Legend label for the PNG Pareto curve (successful runs only).",
    )
    p.add_argument(
        "--no-pareto-png",
        action="store_true",
        default=False,
        help=(
            "Do not write companion Pareto PNGs next to the JSON report "
            "(<stem>_throughput_pareto.png and <stem>_agent_pareto.png)."
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "JSON output path. Default: <trace_dir>/"
            "<trace_name>_<model_basename>_tp<TP>_ep<EP>_<YYYYMMDD_HHMMSS>/"
            "<same_stem>.json "
            "(model basename is the last segment of --model-dir)."
        ),
    )
    return p.parse_args()


def pareto_config_filename_suffix(args: argparse.Namespace) -> str:
    """Return ``<model_basename>_tp<TP>_ep<EP>`` for default artifact names."""
    raw = (getattr(args, "model_dir", None) or "").strip()
    if raw:
        name = Path(os.path.expanduser(raw)).resolve().name
    else:
        name = "model"
    if not name or name == ".":
        name = "model"
    safe = "".join((ch if ch.isalnum() or ch in "._-" else "_") for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_") or "model"
    tp = int(getattr(args, "tensor_parallel_size", 1) or 1)
    ep = int(getattr(args, "moe_expert_parallel_size", 0) or 0)
    adp = "_adp" if getattr(args, "enable_attention_dp", False) else ""
    return f"{safe}_tp{tp}_ep{ep}{adp}"


def resolve_output_json_path(args: argparse.Namespace, trace_dir: Path) -> Path:
    """Default path: ``<trace_dir>/<run_stem>/<run_stem>.json``."""
    if args.output_json is not None:
        return args.output_json.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = pareto_config_filename_suffix(args)
    run_stem = f"{trace_dir.name}_{slug}_{stamp}"
    run_dir = trace_dir / run_stem
    return run_dir / f"{run_stem}.json"


def pareto_checkpoint_paths(output_json: Path) -> Tuple[Path, Path]:
    """Return ``(partial_json, ladder_runs_jsonl)`` next to the main report path."""
    stem = output_json.stem
    parent = output_json.parent
    partial_json = parent / f"{stem}.partial.json"
    ladder_runs_jsonl = parent / f"{stem}.ladder_runs.jsonl"
    return partial_json, ladder_runs_jsonl


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + replace) and sync to reduce torn files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl_line(path: Path, obj: Any) -> None:
    """Append one JSON object per line; file must already exist."""
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _init_ladder_runs_jsonl(path: Path, header: Dict[str, Any]) -> None:
    """Create JSONL with a single header line (overwrites if present)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "kind": "trace_replay_pareto_frontier.ladder_runs_jsonl_header",
        **header,
    }
    with open(path, "w", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _pareto_checkpoint_write(
    *,
    partial_json: Path,
    ladder_runs_jsonl: Path,
    record: Dict[str, Any],
    append_run_row: Optional[Dict[str, Any]],
) -> None:
    """Set ``run_last_checkpoint_at_utc``, write partial JSON, optionally append JSONL run."""
    record["run_last_checkpoint_at_utc"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    _atomic_write_json(partial_json, record)
    if append_run_row is not None:
        _append_jsonl_line(ladder_runs_jsonl, append_run_row)


def print_experiment_startup_banner(
    *,
    args: argparse.Namespace,
    trace_dir: Path,
    trace_path: Path,
    output_json: Path,
    output_log: Path,
    ladder: List[int],
    partial_json: Optional[Path] = None,
    ladder_runs_jsonl: Optional[Path] = None,
) -> None:
    """Print a high-visibility summary of paths, ladder, and all CLI parameters."""
    width = 78
    bar = "#" * width
    rule = "=" * width
    cli = args_to_dict(args)
    print("", flush=True)
    print(bar, flush=True)
    print(
        "#" + " TRACE REPLAY PARETO FRONTIER — RUN CONFIGURATION ".center(width - 2) + "#",
        flush=True,
    )
    print(bar, flush=True)
    print(f"  hostname          : {socket.gethostname()}", flush=True)
    print(f"  cwd               : {os.getcwd()}", flush=True)
    print(f"  command           : {shlex.join(sys.argv)}", flush=True)
    print(rule, flush=True)
    print(f"  trace_dir         : {trace_dir}", flush=True)
    print(f"  trace_file        : {trace_path}", flush=True)
    print(f"  model_dir         : {os.path.abspath(args.model_dir)}", flush=True)
    print(f"  output_json       : {output_json}", flush=True)
    print(f"  output_log        : {output_log}", flush=True)
    if partial_json is not None:
        print(f"  checkpoint_json   : {partial_json}", flush=True)
    if ladder_runs_jsonl is not None:
        print(f"  ladder_runs_jsonl : {ladder_runs_jsonl}", flush=True)
    print(f"  ladder (raw)      : {args.ladder!r}", flush=True)
    print(f"  ladder (parsed)   : {ladder}", flush=True)
    print("  design            : n_sessions == max_batch_size == each ladder step", flush=True)
    print(rule, flush=True)
    for key in sorted(cli.keys()):
        print(f"  {key:32s} = {cli[key]!r}", flush=True)
    print(rule, flush=True)
    print(f"  LLM_MODELS_ROOT   : {os.environ.get('LLM_MODELS_ROOT', '')!r}", flush=True)
    print(f"  CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}", flush=True)
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"  torch.cuda.device_count() : {n}", flush=True)
            for i in range(n):
                print(
                    f"    [{i}] {torch.cuda.get_device_name(i)}",
                    flush=True,
                )
    except Exception as exc:
        print(f"  (torch GPU probe skipped: {exc!r})", flush=True)
    print(bar + "\n", flush=True)


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    output_json = resolve_output_json_path(args, trace_dir)
    partial_json, ladder_runs_jsonl = pareto_checkpoint_paths(output_json)
    output_log = output_json.with_suffix(".log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(output_log, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeTextIO(original_stdout, log_fp)
    sys.stderr = _TeeTextIO(original_stderr, log_fp)
    try:
        if not args.model_dir:
            print("ERROR: pass --model-dir or set LLM_MODELS_ROOT.", file=sys.stderr)
            sys.exit(2)

        trace_path = find_compact_trace_file(trace_dir)
        trace = ExecutionTrace.load(str(trace_path))
        ladder = parse_ladder(args.ladder)

        print_experiment_startup_banner(
            args=args,
            trace_dir=trace_dir,
            trace_path=trace_path,
            output_json=output_json,
            output_log=output_log,
            ladder=ladder,
            partial_json=partial_json,
            ladder_runs_jsonl=ladder_runs_jsonl,
        )

        host_info = collect_host_info()

        trace_meta = {
            "trace_id": trace.trace_id,
            "num_events": len(trace.events),
            "parallel_region_counts": count_parallel_regions(trace.events),
            "assistant_output_tokens_sum": count_assistant_completion_tokens(trace.events),
            **summarize_trace_events(trace.events),
            **collect_trace_file_stats(trace_path),
        }

        _md = (args.model_dir or "").strip()
        _artifact_model_name = Path(os.path.expanduser(_md)).name if _md else "model"
        if not _artifact_model_name or _artifact_model_name == ".":
            _artifact_model_name = "model"
        record: Dict[str, Any] = {
            "schema": "trace_replay_pareto_frontier.v3",
            "run_started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifact_naming": {
                "model_name": _artifact_model_name,
                "tensor_parallel_size": args.tensor_parallel_size,
                "moe_expert_parallel_size": args.moe_expert_parallel_size,
                "enable_attention_dp": getattr(args, "enable_attention_dp", False),
                "filename_suffix": pareto_config_filename_suffix(args),
            },
            "cli_argv": sys.argv,
            "cli_args": args_to_dict(args),
            "output_json": str(output_json),
            "output_log": str(output_log),
            "checkpoint_artifacts": {
                "partial_json": str(partial_json),
                "ladder_runs_jsonl": str(ladder_runs_jsonl),
            },
            "trace_dir": str(trace_dir),
            "trace_file": str(trace_path),
            "model_dir": os.path.abspath(args.model_dir),
            "host": host_info,
            "trace_meta": trace_meta,
            "llm_fixed_config": llm_config_record(args),
            "trtllm_serve_reference_cli": (
                f"trtllm-serve <model> --tp_size {args.tensor_parallel_size} "
                + (
                    f"--ep_size {args.moe_expert_parallel_size} [...]"
                    if args.moe_expert_parallel_size > 0
                    else "[...] (no --ep_size; dense model)"
                )
                + (
                    "; parallel_config: enable_attention_dp: true"
                    if getattr(args, "enable_attention_dp", False)
                    else ""
                )
            ),
            "experiment": {
                "ladder": ladder,
                "ladder_len": len(ladder),
                "design": "n_sessions == max_batch_size == ladder_step",
            },
            "metrics_notes": {
                "throughput_basis": "Pareto throughput (output_tps_*, median/mean_tps_per_user, etc.) uses "
                "total_output_tokens_replay_actual / measured wall time",
                "trace_token_fields": "assistant_output_tokens_per_trace, total_output_tokens_trace_metadata "
                "and completion_tokens_sum in trace_meta are from the trace file",
                "output_tps_aggregate": "total_output_tokens_replay_actual / wall_clock_s",
                "output_tps_per_gpu": "output_tps_aggregate / tensor_parallel_size",
                "output_tps_per_cuda_device_count": "output_tps_aggregate / torch.cuda.device_count()",
                "mean_tps_per_user_session_time": "total_output_tokens_replay_actual / sum(session durations)",
                "per_session_total_output_tokens": (
                    "Replay-measured decoder output token count per session "
                    "(order matches session index 0..n-1); "
                    "same values as per_session_replay_output_token_sum."
                ),
                "replay_tool_sleep_wall_s_estimated": "sum(tool_call.duration_ms)/1000 from trace (replay sleep)",
            },
            "runs": [],
        }

        _init_ladder_runs_jsonl(
            ladder_runs_jsonl,
            {
                "schema": record["schema"],
                "output_json": str(output_json),
                "partial_json": str(partial_json),
            },
        )
        _pareto_checkpoint_write(
            partial_json=partial_json,
            ladder_runs_jsonl=ladder_runs_jsonl,
            record=record,
            append_run_row=None,
        )
        print(
            f"Checkpoint files initialized (partial + JSONL): "
            f"{partial_json.name}, {ladder_runs_jsonl.name}",
            flush=True,
        )

        ladder_total = len(ladder)
        oom_stopped = False
        oom_error_text: Optional[str] = None
        oom_at_step: Optional[int] = None
        oom_at_index: Optional[int] = None

        for idx, step in enumerate(ladder):
            n_sessions = step
            max_batch_size = step

            if oom_stopped:
                row = _skipped_row_after_prior_oom(
                    ladder_index=idx,
                    ladder_step=step,
                    prior_oom_error=oom_error_text,
                    prior_oom_step=oom_at_step if oom_at_step is not None else step,
                    prior_oom_index=oom_at_index if oom_at_index is not None else idx,
                    args=args,
                )
                record["runs"].append(row)
                _pareto_checkpoint_write(
                    partial_json=partial_json,
                    ladder_runs_jsonl=ladder_runs_jsonl,
                    record=record,
                    append_run_row=row,
                )
                print(
                    f"[LADDER {idx + 1}/{ladder_total}] SKIPPED (prior OOM)  "
                    f"N=B={max_batch_size}  status={row.get('status')}",
                    flush=True,
                )
                continue

            banner = (
                "\n"
                + "=" * 72
                + "\n"
                + f"  LADDER [{idx + 1}/{ladder_total}]  START\n"
                + f"    max_batch_size (scheduler cap)  = {max_batch_size}\n"
                + f"    concurrent trace replays (N)    = {n_sessions}  "
                + "(one full trace per session)\n"
                + f"    design: N == max_batch_size == ladder step {step}\n"
                + "=" * 72
                + "\n"
            )
            print(banner, flush=True)
            try:
                row = asyncio.run(
                    run_one_point(
                        trace=trace,
                        model_dir=args.model_dir,
                        n_sessions=n_sessions,
                        max_batch_size=max_batch_size,
                        args=args,
                        ladder_index=idx,
                        ladder_step=step,
                    )
                )
            except Exception as exc:
                row = {
                    "ladder_index": idx,
                    "ladder_step": step,
                    "n_sessions": n_sessions,
                    "max_batch_size": max_batch_size,
                    "status": "failed",
                    "error": repr(exc),
                    "error_traceback": traceback.format_exc(),
                }
                if _is_oom_exception(exc):
                    row["error_kind"] = "out_of_memory"

            row["llm_effective_config"] = {
                **llm_config_record(args),
                "max_batch_size": max_batch_size,
                "n_sessions": n_sessions,
                "ladder_index": idx,
                "ladder_step": step,
            }
            record["runs"].append(row)
            _pareto_checkpoint_write(
                partial_json=partial_json,
                ladder_runs_jsonl=ladder_runs_jsonl,
                record=record,
                append_run_row=row,
            )

            status = row.get("status", "unknown")
            print(
                f"[LADDER {idx + 1}/{ladder_total}] DONE  N={n_sessions}  "
                f"max_batch_size={max_batch_size}  status={status}",
                flush=True,
            )

            if row.get("status") != "success":
                is_oom = row.get("error_kind") == "out_of_memory"
                if is_oom:
                    oom_stopped = True
                    oom_error_text = row.get("error")
                    oom_at_step = step
                    oom_at_index = idx
                    record["pareto_ladder_oom"] = {
                        "stopped_after_oom": True,
                        "first_oom_ladder_index": oom_at_index,
                        "first_oom_ladder_step": oom_at_step,
                        "error": oom_error_text,
                        "remaining_ladder_steps_marked_skipped": [s for s in ladder[idx + 1 :]],
                    }
                    print(
                        f"Out-of-memory at N=B={step}; remaining ladder values will "
                        f"be recorded as skipped (not executed).",
                        file=sys.stderr,
                    )
                    continue

                if args.stop_on_error:
                    print(
                        f"Stopped after failure at N=B={step}: {row.get('error')}",
                        file=sys.stderr,
                    )
                    break

        record["run_finished_at_utc"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_json, record)
        _atomic_write_json(partial_json, record)

        print(f"Wrote {output_json}", flush=True)
        print(f"Wrote {partial_json}", flush=True)
        print(f"Wrote {output_log}", flush=True)

        if not args.no_pareto_png:
            write_token_pareto_png_from_json_file = _load_plot_helper(
                "plot_trace_replay_token_pareto",
                "write_token_pareto_png_from_json_file",
            )
            if write_token_pareto_png_from_json_file is not None:
                tp_png = write_token_pareto_png_from_json_file(  # type: ignore[misc]
                    output_json,
                    curve_label=args.pareto_curve_label,
                    figure_caption=None,
                    png_path=None,
                )
                if tp_png is not None:
                    print(f"Wrote {tp_png}", flush=True)
            else:
                print(
                    "WARNING: could not import plot_trace_replay_token_pareto; "
                    "skip throughput Pareto PNG.",
                    file=sys.stderr,
                )
            write_agent_pareto_png_from_json_file = _load_plot_helper(
                "plot_trace_replay_agent_pareto",
                "write_agent_pareto_png_from_json_file",
            )
            if write_agent_pareto_png_from_json_file is not None:
                agent_png = write_agent_pareto_png_from_json_file(  # type: ignore[misc]
                    output_json,
                    figure_caption=None,
                    png_path=None,
                )
                if agent_png is not None:
                    print(f"Wrote {agent_png}", flush=True)
            else:
                print(
                    "WARNING: could not import plot_trace_replay_agent_pareto; "
                    "skip agent Pareto PNG.",
                    file=sys.stderr,
                )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()


if __name__ == "__main__":
    main()
