# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-only wall-clock timers for the executor loop (env-gated, default off).

``TRTLLM_HOST_TIMERS=lo-hi`` wraps a fixed list of executor-thread host
functions with ``time.perf_counter`` timers and accumulates them per executor
iteration. An iteration is the interval between two consecutive calls of the
loop's ``profile_step`` hook (one per loop pass); only passes that ran a forward
step count, and of those the ``lo``-th to ``hi``-th are kept. At ``hi`` (or when
the executor loop exits, whichever comes first) every rank writes
``hosttimers_r<rank>_<pid>.txt`` to ``TRTLLM_HOST_TIMERS_OUT`` (default
``/tmp``) and rank 0 logs the path.

Per target the summary lists the median / mean / p90 inclusive time per
iteration, the median exclusive time (inclusive minus wrapped callees) and the
mean call count. ``PERIOD`` is the loop period, ``SYNC`` the time blocked in
host-device synchronization (``torch.cuda.Event.synchronize``,
``torch.cuda.Stream.synchronize``, ``torch.cuda.synchronize`` and ``item`` /
``tolist`` / ``cpu`` on CUDA tensors), ``BUSY`` = ``PERIOD - SYNC``, the host
work of one loop pass, and ``COMM`` the time spent in host collectives
(``MPIDist`` / ``TorchDist`` broadcast / gather / reduce), which includes the
wait for the slowest rank. ``BUSY`` is the number to compare with the device
step: a loop whose BUSY exceeds the device step leaves the GPU idle.

``TRTLLM_HOST_TIMERS_BY_BATCH=1`` also keys every iteration by the batch its
forward ran (``gen<N>`` for N generation requests, ``ctx<C>+gen<N>`` with
context requests) and writes one summary per key, so a single run that steps
through several batch sizes reports each of them.

``TRTLLM_HOST_TIMERS_FUNCS`` adds comma-separated ``module:Qual.name`` targets
(``@path`` reads one per line from a file). A ``model:<name suffix>[#method]`` target wraps one
method (default ``forward``) of the first module of the executor's model whose qualified name ends
with the suffix, as an instance attribute: only that module instance is timed, e.g. the eager MTP
draft layer's submodules (``model:mtp_layers.0.mlp``), whose classes are also traced into the
torch.compile'd target model, where a class-level wrapper would invalidate the compiled frames.
``TRTLLM_HOST_TIMERS_ONLY=1`` drops
the default list (light mode: the loop period / SYNC / BUSY / COMM rows from the
forward-pass hook, the collectives and the sync points only).
``TRTLLM_HOST_TIMERS_LOG=1`` also logs every rank's summary
(each line prefixed ``[host_timers r<rank>]``) and, with
``TRTLLM_HOST_TIMERS_SERIES=BUSY,SYNC,COMM,<target>,...``, those rows'
per-iteration values in microseconds. Timers add about a microsecond per
wrapped call and nothing when the variable is unset.

The default targets cover the PyTorch executor, the model engine and the KV
cache transceiver (``PyExecutor``, ``PyTorchModelEngine``, ``KvCacheTransceiverV2``).
"""

import contextlib
import functools
import importlib
import os
import tempfile
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

_PX = "tensorrt_llm._torch.pyexecutor.py_executor:PyExecutor."
_ME = "tensorrt_llm._torch.pyexecutor.model_engine:PyTorchModelEngine."
_TX = "tensorrt_llm._torch.disaggregation.transceiver:KvCacheTransceiverV2."
_CG = "tensorrt_llm._torch.pyexecutor.cuda_graph_runner:CUDAGraphRunner."
_RB = "tensorrt_llm._torch.pyexecutor.request_utils:RequestBroadcaster."
_DIST = "tensorrt_llm._torch.distributed.communicator:"

DEFAULT_TARGETS: List[str] = [
    # executor loop
    _PX + "_prepare_and_schedule_batch",
    _PX + "_fetch_and_activate_new_requests",
    _PX + "_fetch_new_requests",
    _PX + "_handle_control_request",
    _PX + "_schedule",
    _PX + "_wait_for_model_engine_input_copy",
    _PX + "_can_queue",
    _PX + "_handle_disagg_cache_errors_synced",
    _PX + "_check_disagg_ctx_schedulable_status",
    _PX + "_check_disagg_gen_transfer_status",
    _PX + "_check_disagg_ctx_cache_transfer_status",
    _PX + "_check_kv_transfer_timeout",
    _PX + "_cancel_timed_out_gen_transfers",
    _PX + "_check_gen_cache_transfer_errors_consensus",
    _PX + "_apply_disagg_transfer_admission",
    _PX + "_prepare_disagg_gen_init",
    _PX + "_check_disagg_transfer_progress_when_idle",
    _PX + "_sync_disagg_gen_status_entry",
    _PX + "_sync_disagg_ctx_status_entry",
    _PX + "_sync_gen_only_benchmark_has_insufficient_kv",
    _PX + "_check_benchmark_disagg_gate",
    _PX + "_prepare_disagg_gen_transmission_complete",
    _PX + "_handle_dynamic_draft_len",
    _PX + "_handle_first_token_response",
    _PX + "_emit_first_token_responses",
    _PX + "_collect_scheduled_batch_stats",
    _PX + "_init_iter_stats_if_sampled",
    _PX + "_pad_attention_dp_dummy_request",
    _PX + "_finalize_adp_dummy_allocation",
    _PX + "_kv_connector_start_batch",
    _PX + "_kv_connector_terminate_requests",
    _PX + "_forward_step",
    _PX + "_update_requests",
    _PX + "_accumulate_spec_dec_stats",
    _PX + "_send_kv_async",
    _PX + "_flush_pending_transfer_responses",
    _PX + "_sample_async",
    _PX + "_update_request_states",
    _PX + "_process_previous_batch",
    _PX + "_handle_canceled_requests",
    _PX + "_handle_responses",
    _PX + "_enqueue_responses",
    _PX + "_process_iter_stats",
    _PX + "_commit_kv_cache_stats",
    _PX + "_handle_kv_transfer_timeouts_synced",
    _PX + "_flush_iter_stats_synced",
    _PX + "_update_generation_requests_that_will_complete_next_iteration",
    _PX + "_handle_control_request",
    _RB + "broadcast",
    # disaggregated transfer bookkeeping (PYTHON transceiver)
    _TX + "check_gen_transfer_status",
    _TX + "check_context_transfer_status",
    _TX + "prepare_context_requests",
    _TX + "request_and_receive_async",
    _TX + "request_and_receive_sync",
    _TX + "_gen_consensus",
    _TX + "_gen_consensus_outcome",
    "tensorrt_llm._torch.pyexecutor.resource_manager:ResourceManager.prepare_resources",
    "tensorrt_llm._torch.pyexecutor.resource_manager:ResourceManager.update_resources",
    "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2:KVCacheManagerV2.prepare_resources",
    "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2:KVCacheManagerV2.update_resources",
    "tensorrt_llm._torch.pyexecutor.mamba_cache_manager:MambaHybridCacheManagerV2.prepare_resources",
    "tensorrt_llm._torch.pyexecutor.seq_slot_manager:SeqSlotManager.prepare_resources",
    # forward
    _ME + "forward",
    _ME + "_set_up_attn_metadata",
    _ME + "_set_up_spec_metadata",
    _ME + "_sync_group_all_greedy_sample",
    _ME + "_prepare_inputs",
    _ME + "_prepare_tp_inputs",
    _ME + "_preprocess_inputs",
    _ME + "_execute_logit_post_processors",
    _CG + "maybe_get_cuda_graph",
    _CG + "needs_capture",
    _CG + "replay",
    _CG + "pad_batch",
    "tensorrt_llm._torch.speculative.interface:SpecMetadata.update_is_all_greedy_sample",
    "tensorrt_llm._torch.speculative.interface:SpecMetadata.populate_sampling_params_for_one_model",
    "tensorrt_llm._torch.speculative.eagle3:Eagle3OneModelSpecMetadata.prepare",
    "tensorrt_llm._torch.speculative.mtp:MTPSpecMetadata.prepare",
    "tensorrt_llm._torch.attention_backend.trtllm:TrtllmAttentionMetadata.prepare",
    "tensorrt_llm._torch.attention_backend.trtllm:TrtllmAttentionMetadata.update_spec_dec_param",
    "tensorrt_llm._torch.modules.mamba.mamba2_metadata:Mamba2Metadata.prepare",
    # sampling
    "tensorrt_llm._torch.speculative.spec_sampler_base:SpecSamplerBase.sample_async",
    "tensorrt_llm._torch.speculative.spec_sampler_base:SpecSamplerBase.update_requests",
    "tensorrt_llm._torch.pyexecutor.sampler.sampler:SamplerEvent.synchronize",
    # logging
    "tensorrt_llm.logger:Logger.info",
]

# Host collectives: leaf methods only (the tp_cp_* helpers call these).
_COMM_PREFIX = "comm:"
COMM_TARGETS: List[str] = [
    f"{_DIST}{cls}.{method}"
    for cls in ("MPIDist", "TorchDist")
    for method in (
        "broadcast",
        "allgather",
        "barrier",
        "allreduce",
        "tp_allgather",
        "tp_gather",
        "tp_broadcast",
        "tp_allreduce",
        "pp_allgather",
        "cp_allgather",
        "cp_broadcast",
    )
] + [
    # raw-buffer collectives of the TRTLLM_Q35_* host paths
    _RB + "_fast_count_broadcast",
    _PX + "_q35_merged_disagg_votes",
]

# Module-instance targets (install_module_target).
_MODEL_PREFIX = "model:"

# Leaf synchronization points; their sum per iteration is SYNC.
_SYNC_PREFIX = "sync:"
_FORWARD = "PyExecutor._forward_step"


def _parse_range(value: str):
    lo, _, hi = value.partition("-")
    lo_i = int(lo)
    hi_i = int(hi) if hi else lo_i + 200
    if hi_i <= lo_i:
        raise ValueError(f"TRTLLM_HOST_TIMERS={value!r}: expected lo-hi with hi > lo")
    return lo_i, hi_i


def _rank() -> int:
    for key in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        if key in os.environ:
            try:
                return int(os.environ[key])
            except ValueError:
                pass
    return 0


def _stats(values: List[float]):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0
    return v[n // 2], sum(v) / n, v[min(n - 1, (n * 9) // 10)]


def _batch_key(scheduled_requests) -> Optional[str]:
    try:
        num_ctx = int(scheduled_requests.num_context_requests)
        num_gen = int(scheduled_requests.num_generation_requests)
    except (AttributeError, TypeError, ValueError):
        return None
    return f"ctx{num_ctx}+gen{num_gen}" if num_ctx else f"gen{num_gen}"


class _Window:
    """Accumulated per-iteration samples of one key (all iterations, or one batch key)."""

    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.exc_samples: Dict[str, List[float]] = defaultdict(list)
        self.call_samples: Dict[str, List[int]] = defaultdict(list)
        self.periods: List[float] = []
        self.syncs: List[float] = []
        self.comms: List[float] = []

    def add(self, period: float, cur_inc, cur_exc, cur_calls) -> None:
        # Pad names first seen in this iteration so every series stays aligned.
        n_before = len(self.periods)
        self.periods.append(period)
        self.syncs.append(sum(v for k, v in cur_inc.items() if k.startswith(_SYNC_PREFIX)))
        self.comms.append(sum(v for k, v in cur_inc.items() if k.startswith(_COMM_PREFIX)))
        for name in set(self.samples) | set(cur_inc):
            if name not in self.samples and n_before:
                self.samples[name].extend([0.0] * n_before)
                self.exc_samples[name].extend([0.0] * n_before)
                self.call_samples[name].extend([0] * n_before)
            self.samples[name].append(cur_inc.get(name, 0.0))
            self.exc_samples[name].append(cur_exc.get(name, 0.0))
            self.call_samples[name].append(cur_calls.get(name, 0))

    def lines(self, title: str) -> List[str]:
        busy = [p - s for p, s in zip(self.periods, self.syncs)]
        local = [b - c for b, c in zip(busy, self.comms)]
        out = []
        for label, values in (
            ("PERIOD", self.periods),
            ("SYNC", self.syncs),
            ("BUSY", busy),
            ("COMM", self.comms),
            ("BUSY-COMM", local),
        ):
            med, mean, p90 = _stats(values)
            out.append(
                f"{1e3 * med:9.3f} med {1e3 * mean:9.3f} mean {1e3 * p90:9.3f} p90 ms/iter  "
                f"{label}{title}"
            )
        rows = []
        for name, values in self.samples.items():
            med, mean, p90 = _stats(values)
            exc_med = _stats(self.exc_samples[name])[0]
            calls = sum(self.call_samples[name]) / max(1, len(self.call_samples[name]))
            rows.append((mean, med, p90, exc_med, calls, name))
        rows.sort(reverse=True)
        out += self._spike_lines(busy, title)
        out.append(f"# {len(self.periods)} iterations{title}; ms/iter")
        out.append("#      med      mean       p90   excl-med  calls/iter  target")
        for mean, med, p90, exc_med, calls, name in rows:
            out.append(
                f"{1e3 * med:9.3f} {1e3 * mean:9.3f} {1e3 * p90:9.3f} {1e3 * exc_med:9.3f} "
                f"{calls:10.2f}  {name}"
            )
        return out

    def _spike_lines(self, busy: List[float], title: str) -> List[str]:
        """Iterations whose BUSY exceeds the median by max(0.5 ms, 25 %): which targets' exclusive time
        (and GC, and the unwrapped loop body) carries the excess."""
        if len(busy) < 20:
            return []
        med = _stats(busy)[0]
        limit = med + max(0.5e-3, 0.25 * med)
        spikes = [i for i, b in enumerate(busy) if b > limit]
        if not spikes:
            return [f"# spikes{title}: none above {1e3 * limit:.3f} ms"]
        excess_total = sum(busy[i] - med for i in spikes)
        per_target = []
        for name, values in self.exc_samples.items():
            if name.startswith(_SYNC_PREFIX):
                continue
            m = _stats(values)[0]
            extra = sum(max(0.0, values[i] - m) for i in spikes if i < len(values))
            if extra > 0:
                per_target.append((extra, name))
        per_target.sort(reverse=True)
        attributed = sum(e for e, _ in per_target)
        out = [
            f"# spikes{title}: {len(spikes)} of {len(busy)} iterations above {1e3 * limit:.3f} ms, "
            f"mean excess {1e3 * excess_total / len(spikes):.3f} ms; excess by target (exclusive, ms per spike):"
        ]
        for extra, name in per_target[:8]:
            out.append(f"#   {1e3 * extra / len(spikes):8.3f}  {name}")
        out.append(
            f"#   {1e3 * (excess_total - attributed) / len(spikes):8.3f}  (not in a wrapped target: loop body)"
        )
        return out

    def series(self, want: str) -> Optional[List[float]]:
        busy = [p - s for p, s in zip(self.periods, self.syncs)]
        named = {"PERIOD": self.periods, "SYNC": self.syncs, "BUSY": busy, "COMM": self.comms}
        if want in named:
            return named[want]
        key = next((k for k in self.samples if k == want or k.endswith(want)), None)
        return self.samples.get(key) if key is not None else None


class _HostTimers:
    """Per-executor accumulation state; owned by the executor thread."""

    def __init__(self, lo: int, hi: int, out_dir: str, rank: int, by_batch: bool) -> None:
        self.lo, self.hi, self.out_dir, self.rank = lo, hi, out_dir, rank
        self.by_batch = by_batch
        self.tid = threading.get_ident()
        self.stack: List[list] = []  # [name, t0, callee_time]
        self.cur_inc: Dict[str, float] = defaultdict(float)
        self.cur_exc: Dict[str, float] = defaultdict(float)
        self.cur_calls: Dict[str, int] = defaultdict(int)
        self.cur_forward = False
        self.cur_key: Optional[str] = None
        self.n_forward = 0
        self.last_boundary: Optional[float] = None
        self.all = _Window()
        self.keyed: Dict[str, _Window] = {}
        self.done = False

    def enter(self, name: str) -> list:
        frame = [name, time.perf_counter(), 0.0]
        self.stack.append(frame)
        return frame

    def leave(self, frame: list) -> None:
        dt = time.perf_counter() - frame[1]
        # Unwind to this frame even if a callee raised past its own leave.
        while self.stack and self.stack[-1] is not frame:
            self.stack.pop()
        if self.stack:
            self.stack.pop()
        name = frame[0]
        self.cur_inc[name] += dt
        self.cur_exc[name] += dt - frame[2]
        self.cur_calls[name] += 1
        if name == _FORWARD:
            self.cur_forward = True
        if self.stack:
            self.stack[-1][2] += dt

    def boundary(self) -> None:
        now = time.perf_counter()
        if self.last_boundary is not None and self.cur_forward:
            self.n_forward += 1
            if self.lo <= self.n_forward < self.hi:
                period = now - self.last_boundary
                self.all.add(period, self.cur_inc, self.cur_exc, self.cur_calls)
                if self.by_batch and self.cur_key is not None:
                    window = self.keyed.setdefault(self.cur_key, _Window())
                    window.add(period, self.cur_inc, self.cur_exc, self.cur_calls)
            elif self.n_forward >= self.hi and not self.done:
                self.finish()
        self.last_boundary = now
        self.stack = []
        self.cur_inc = defaultdict(float)
        self.cur_exc = defaultdict(float)
        self.cur_calls = defaultdict(int)
        self.cur_forward = False
        self.cur_key = None

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        if not self.all.periods:
            # E.g. the short-lived executor of the KV-cache estimation pass.
            return
        try:
            self.write()
        except Exception as error:  # measurement only: never break the executor
            from tensorrt_llm.logger import logger

            logger.warning(f"[host_timers] write failed: {error}")

    def write(self) -> None:
        lines = self.all.lines(f" (forward passes {self.lo}..{self.hi - 1}, all batches)")
        windows = [("", self.all)]
        if self.by_batch:
            for key in sorted(self.keyed, key=lambda k: (k.startswith("ctx"), len(k), k)):
                lines.append("")
                lines += self.keyed[key].lines(f" [{key}]")
                windows.append((key, self.keyed[key]))
        if _REGISTRY.missing:
            lines.append("# not installed: " + ", ".join(_REGISTRY.missing))
        path = os.path.join(self.out_dir, f"hosttimers_r{self.rank}_{os.getpid()}.txt")
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(path, "w") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError:
            path = None
        from tensorrt_llm.logger import logger

        if self.rank == 0:
            logger.info(f"[host_timers] wrote {path}; {lines[0].strip()}")
        if os.environ.get("TRTLLM_HOST_TIMERS_LOG") == "1":
            # The whole summary in the worker log as well, on every rank: for harnesses whose
            # container /tmp does not outlive the job and that run on several clusters.
            logger.info("".join(f"\n[host_timers r{self.rank}] {line}" for line in lines))
            # TRTLLM_HOST_TIMERS_SERIES=BUSY,SYNC,<target>,...: per-iteration values (us) of these
            # rows, to line host spikes up with the iteration log's device steps.
            for want in os.environ.get("TRTLLM_HOST_TIMERS_SERIES", "").split(","):
                want = want.strip()
                if not want:
                    continue
                for key, window in windows:
                    values = window.series(want)
                    if values is not None:
                        tag = f" [{key}]" if key else ""
                        logger.info(
                            f"[host_timers r{self.rank}] SERIES {want}{tag} first={self.lo} "
                            + ",".join(str(int(round(1e6 * v))) for v in values)
                        )


class _Registry:
    """Process-wide wrappers; they account into whichever state is current."""

    def __init__(self) -> None:
        self.current: Optional[_HostTimers] = None
        self.installed: List[str] = []
        self.missing: List[str] = []
        self.sync_installed = False

    def _active(self) -> Optional[_HostTimers]:
        state = self.current
        if state is None or state.done or threading.get_ident() != state.tid:
            return None
        return state

    def wrap(self, fn: Callable, name: str) -> Callable:
        registry = self
        is_forward = name == _FORWARD

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            state = registry._active()
            if state is None:
                return fn(*args, **kwargs)
            if is_forward and len(args) > 1:
                state.cur_key = _batch_key(args[1])
            frame = state.enter(name)
            try:
                return fn(*args, **kwargs)
            finally:
                state.leave(frame)

        wrapped.__host_timer_wrapped__ = True
        return wrapped

    def wrap_cuda_tensor_method(self, fn: Callable, name: str) -> Callable:
        registry = self

        @functools.wraps(fn)
        def wrapped(tensor, *args, **kwargs):
            state = registry._active()
            if state is None or not getattr(tensor, "is_cuda", False):
                return fn(tensor, *args, **kwargs)
            frame = state.enter(name)
            try:
                return fn(tensor, *args, **kwargs)
            finally:
                state.leave(frame)

        wrapped.__host_timer_wrapped__ = True
        return wrapped

    def install_target(self, target: str, prefix: str = "") -> None:
        if target in self.installed or target in self.missing:
            return
        module_name, _, qual = target.partition(":")
        try:
            obj = importlib.import_module(module_name)
            parts = qual.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            attr = parts[-1]
            raw = obj.__dict__.get(attr) if isinstance(obj, type) else getattr(obj, attr)
            if raw is None:
                raise AttributeError(attr)
        except (ImportError, AttributeError):
            self.missing.append(target)
            return
        name = prefix + qual
        if isinstance(raw, staticmethod):
            setattr(obj, attr, staticmethod(self.wrap(raw.__func__, name)))
        elif isinstance(raw, classmethod):
            setattr(obj, attr, classmethod(self.wrap(raw.__func__, name)))
        elif callable(raw) and not getattr(raw, "__host_timer_wrapped__", False):
            setattr(obj, attr, self.wrap(raw, name))
        else:
            self.missing.append(target)
            return
        self.installed.append(target)

    def install_module_target(self, executor, target: str) -> None:
        """``model:<name suffix>[#method]``: wrap that method of one module instance (see the module docstring)."""
        if target in self.installed or target in self.missing:
            return
        spec = target[len(_MODEL_PREFIX) :]
        spec, _, method = spec.partition("#")
        method = method or "forward"
        try:
            model = executor.model_engine.model
            module = next(
                m
                for name, m in model.named_modules(remove_duplicate=False)
                if name == spec or name.endswith("." + spec)
            )
            raw = getattr(module, method)
        except (AttributeError, StopIteration):
            self.missing.append(target)
            return
        if getattr(raw, "__host_timer_wrapped__", False):
            self.missing.append(target)
            return
        # Bound method stored on the instance: nn.Module.__call__ looks up self.forward there first.
        object.__setattr__(module, method, self.wrap(raw, target))
        self.installed.append(target)

    def install_gc_tracking(self) -> None:
        """Time every garbage collection run on the executor thread as target gc:gen<N>."""
        if getattr(self, "gc_installed", False):
            return
        import gc

        registry = self

        def on_gc(phase, info):
            state = registry._active()
            if state is None:
                return
            if phase == "start":
                state.gc_frame = state.enter(f"gc:gen{info.get('generation', -1)}")
            elif getattr(state, "gc_frame", None) is not None:
                frame, state.gc_frame = state.gc_frame, None
                state.leave(frame)

        gc.callbacks.append(on_gc)
        self.gc_installed = True

    def install_sync_points(self) -> None:
        if self.sync_installed:
            return
        import torch

        torch.cuda.Event.synchronize = self.wrap(
            torch.cuda.Event.synchronize, _SYNC_PREFIX + "Event.synchronize"
        )
        torch.cuda.Stream.synchronize = self.wrap(
            torch.cuda.Stream.synchronize, _SYNC_PREFIX + "Stream.synchronize"
        )
        torch.cuda.synchronize = self.wrap(
            torch.cuda.synchronize, _SYNC_PREFIX + "cuda.synchronize"
        )
        for method in ("item", "tolist", "cpu"):
            setattr(
                torch.Tensor,
                method,
                self.wrap_cuda_tensor_method(
                    getattr(torch.Tensor, method), _SYNC_PREFIX + "Tensor." + method
                ),
            )
        self.sync_installed = True


_REGISTRY = _Registry()


def install_host_timers(executor) -> None:
    """Start timing this executor's loop; call on the executor thread. No-op unless enabled."""
    value = os.environ.get("TRTLLM_HOST_TIMERS")
    if not value:
        return
    lo, hi = _parse_range(value)
    targets = [] if os.environ.get("TRTLLM_HOST_TIMERS_ONLY") == "1" else list(DEFAULT_TARGETS)
    extra = os.environ.get("TRTLLM_HOST_TIMERS_FUNCS", "")
    if extra.startswith("@"):
        # One target per line in a file (the launcher's env list is comma-separated).
        with open(extra[1:]) as handle:
            extra = ",".join(line.strip() for line in handle if line.strip())
    targets += [t.strip() for t in extra.split(",") if t.strip()]
    # The forward-pass hook defines the timed iterations and their batch keys, so it is installed in every mode
    # (TRTLLM_HOST_TIMERS_ONLY=1 without it would time nothing).
    if not any(t.split(":")[-1] == _FORWARD for t in targets):
        targets.append(_PX + "_forward_step")
    for target in targets:
        if target.startswith(_MODEL_PREFIX):
            _REGISTRY.install_module_target(executor, target)
        else:
            _REGISTRY.install_target(target)
    for target in COMM_TARGETS:
        _REGISTRY.install_target(target, prefix=_COMM_PREFIX)
    _REGISTRY.install_sync_points()
    _REGISTRY.install_gc_tracking()
    state = _HostTimers(
        lo,
        hi,
        os.environ.get("TRTLLM_HOST_TIMERS_OUT", tempfile.gettempdir()),
        _rank(),
        os.environ.get("TRTLLM_HOST_TIMERS_BY_BATCH") == "1",
    )
    _REGISTRY.current = state

    # Iteration boundary: the profile_step hook the loop calls once per pass.
    original_profiler = executor._profiler

    @contextlib.contextmanager
    def timed_profiler():
        with original_profiler() as step:

            def timed_step():
                state.boundary()
                frame = state.enter("profile_step")
                try:
                    step()
                finally:
                    state.leave(frame)

            yield timed_step

    executor._profiler = timed_profiler

    # Also write when the loop exits before ``hi`` forward passes (e.g. at shutdown).
    original_event_loop = executor.event_loop

    def timed_event_loop(*args, **kwargs):
        try:
            return original_event_loop(*args, **kwargs)
        finally:
            state.finish()

    executor.event_loop = timed_event_loop
    if state.rank == 0:
        from tensorrt_llm.logger import logger

        logger.info(
            f"[host_timers] {len(_REGISTRY.installed)} targets installed, "
            f"{len(_REGISTRY.missing)} missing; iterations {lo}-{hi}; "
            f"by_batch={state.by_batch}"
        )
