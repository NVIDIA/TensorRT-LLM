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
"""Compile Triton kernel variants ahead of the forward pass that needs them.

Experimental. Enabled by ``TLLM_JIT_PREFETCH=1``.

A kernel variant that warmup never touched compiles synchronously at its first
launch, on the executor thread, stalling the GPU. This module moves that work
earlier and off the executor thread:

1. Once a batch is scheduled, :func:`plan` asks each registered *variant
   provider* (one per model family; see ``modules/mamba/jit_prefetch.py``)
   which Triton kernels the batch will launch, and with what arguments. The
   provider builds those arguments as ``meta`` tensors: same shapes, strides,
   dtypes and view offsets as the real ones, no storage, no GPU work.
2. For each call, Triton's own binder turns the arguments into Triton's own
   cache key, and the key is looked up in the kernel's in-memory cache. This is
   the "is it compiled?" check, and it is exact by construction: it is the same
   computation ``JITFunction.run`` does at launch, minus the launch. Nothing is
   compiled or launched in the calling process.
3. A miss is serialized with Triton's specialization format (the one
   ``JITFunction.preload`` consumes) and sent to a CPU-only helper process. The
   helper calls ``triton.compile``, which writes the cubin to the on-disk
   Triton cache. The helper has no CUDA context, does not touch the GPU, and
   does not hold the executor's GIL.
4. When the real launch arrives, ``triton.compile`` in the executor finds the
   cubin on disk (milliseconds) instead of compiling it (seconds).

Autotuned kernels: the provider resolves the config the autotuner picked (from
its in-memory cache) and plans that one. If the autotuner has not tuned this
key yet, choosing a config means benchmarking on the GPU, which this module
does not do; it compiles every candidate config instead, so the benchmark at
first launch runs on cached cubins. The benchmark itself still runs at that
launch, unchanged, and is counted separately in the stats. Which config runs is
always the autotuner's own choice; this module never selects or reuses one.

Measurement hooks (always on when the module is enabled) log every compile and
every autotune benchmark that happens in the executor after warmup, with its
wall time, so a run reports how much JIT cost was left on the critical path.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger

_ENABLE_ENV = "TLLM_JIT_PREFETCH"
_STATS_ENV = "TLLM_JIT_STATS"
_WORKERS_ENV = "TLLM_JIT_PREFETCH_WORKERS"
_RECORD_ENV = "TLLM_JIT_RECORD_DIR"
_PROVIDERS_ENV = "TLLM_JIT_PREFETCH_PROVIDERS"

# Queue priorities, lowest first: the executor is blocked on it; the batch
# being scheduled needs it; a previous process compiled it (record replay);
# a provider can reach it (background enumeration).
_PRIO_URGENT, _PRIO_BATCH, _PRIO_REPLAY, _PRIO_ENUM = -1, 0, 1, 2


def _default_workers() -> int:
    """Helpers per rank: the CPUs this process may use, shared among the
    ranks on the node, capped. Compiling is single-threaded per variant."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 2
    local = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
        or 1
    )
    return max(2, min(16, cpus // max(1, local) - 2))


def prefetch_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "0") == "1"


def providers_enabled() -> bool:
    """Per-module variant providers (approach A). ``TLLM_JIT_PREFETCH_PROVIDERS=none``
    leaves only record/replay (approach B)."""
    return os.environ.get(_PROVIDERS_ENV, "all") != "none"


def stats_enabled() -> bool:
    return prefetch_enabled() or os.environ.get(_STATS_ENV, "0") == "1"


# ---------------------------------------------------------------------------
# Triton plumbing
# ---------------------------------------------------------------------------


@dataclass
class KernelCall:
    """One planned launch: a JITFunction or Autotuner plus its arguments.

    ``args``/``kwargs`` are exactly what the real call site passes, with tensor
    arguments replaced by meta tensors of the same layout. Autotuned
    meta-parameters are not passed; :func:`_expand` adds them.
    """

    fn: Any
    args: tuple
    kwargs: Dict[str, Any]
    label: str


def _unwrap_jit(fn):
    """Return (jit_function, autotuner_or_None, heuristics_chain)."""
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    autotuner = None
    heuristics = []
    cur = fn
    while not isinstance(cur, JITFunction):
        if isinstance(cur, Autotuner):
            autotuner = cur
        elif isinstance(cur, Heuristics):
            heuristics.append(cur)
        cur = cur.fn
    return cur, autotuner, heuristics


def _autotune_key(autotuner, jit_fn, args, kwargs) -> tuple:
    # Mirrors triton.runtime.autotuner.Autotuner.run (Triton 3.8).
    nargs = dict(zip(autotuner.arg_names, args))
    all_args = {**nargs, **kwargs}
    _args = {k: v for (k, v) in all_args.items() if k in autotuner.arg_names}
    key = [_args[k] for k in autotuner.keys if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


_shadow = threading.local()


class _ShadowLaunches:
    """Turn Triton launches into recorded KernelCalls on this thread.

    Inside the context, the outermost ``run`` of any JITFunction, Autotuner or
    Heuristics records ``(fn, args, kwargs)`` and returns None instead of
    compiling or launching. Launcher code (allocation, rearrange, grid math)
    runs unchanged on meta tensors, so the recorded arguments are exactly the
    ones the real call would pass.
    """

    _patched = False

    @classmethod
    def _patch(cls):
        if cls._patched:
            return
        from triton.runtime.autotuner import Autotuner, Heuristics
        from triton.runtime.jit import JITFunction

        def wrap(klass):
            orig = klass.run

            def run(self, *args, **kwargs):
                rec = getattr(_shadow, "calls", None)
                if rec is None:
                    return orig(self, *args, **kwargs)
                # `kernel[grid](...)` routes through KernelInterface.__getitem__,
                # which passes grid/warmup to whichever run() is outermost.
                kw = dict(kwargs)
                kw.pop("grid", None)
                kw.pop("warmup", None)
                rec.append(KernelCall(self, args, kw, _unwrap_jit(self)[0].fn.__name__))
                return None

            klass.run = run

        wrap(JITFunction)
        wrap(Autotuner)
        wrap(Heuristics)
        cls._patched = True

    def __enter__(self) -> List[KernelCall]:
        self._patch()
        _shadow.calls = []
        return _shadow.calls

    def __exit__(self, *exc):
        _shadow.calls = None
        return False


def shadow_launches() -> _ShadowLaunches:
    return _ShadowLaunches()


def _expand(call: KernelCall) -> Tuple[List[Dict[str, Any]], bool]:
    """Expand a planned call into concrete JITFunction kwargs per config.

    Returns (list_of_kwargs, tuned): ``tuned`` is False when the autotuner has
    not picked a config for this key yet, in which case every pruned config is
    returned (all of them will be benchmarked at the first real launch).
    """
    jit_fn, autotuner, heuristics = _unwrap_jit(call.fn)
    base = dict(call.kwargs)
    if autotuner is None:
        configs = [None]
        tuned = True
    else:
        key = _autotune_key(autotuner, jit_fn, call.args, base)
        cfg = autotuner.cache.get(key) if len(autotuner.configs) > 1 else autotuner.configs[0]
        if cfg is not None:
            configs = [cfg]
            tuned = True
        else:
            # Every candidate config: the superset of what the benchmark at
            # first launch will compile. Autotuner.prune_configs is not called
            # because it reads autotuner.nargs, which the real Autotuner.run
            # owns on the executor thread; this may run on another thread.
            configs = list(autotuner.configs)
            tuned = False
    out = []
    for cfg in configs:
        kw = dict(base)
        if cfg is not None:
            kw.update(cfg.all_kwargs())
        nargs = dict(zip(jit_fn.arg_names, call.args))
        for h in reversed(heuristics):
            for name, rule in h.values.items():
                kw[name] = rule({**nargs, **kw})
        out.append(kw)
    return out, tuned


def _on_disk(jit_fn, signature, constexprs, attrs, options, backend) -> bool:
    """True if triton.compile would load this variant from the on-disk cache.

    Same key and lookup as triton.compiler.compile (Triton 3.8): the metadata
    JSON named after the kernel, in the cache directory
    sha256(get_cache_key(src, backend, options, env_vars)). A variant already
    on disk loads in about a millisecond at launch, so a helper round trip
    would only add a wait. Costs about 0.1 ms per variant once Triton's
    install hash (``triton_key``, cached) has been computed.
    """
    import hashlib

    from triton import knobs
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.runtime.cache import get_cache_key, get_cache_manager

    if knobs.compilation.always_compile or knobs.runtime.add_stages_inspection_hook:
        return False
    src = jit_fn.ASTSource(jit_fn, signature, constexprs, attrs)
    opts = backend.parse_options(dict(options.__dict__, **src.parse_options()))
    key = get_cache_key(src, backend, opts, env_vars=get_cache_invalidating_env_vars())
    manager = get_cache_manager(hashlib.sha256(key.encode("utf-8")).hexdigest())
    name = f"{src.name[:150]}.json"
    return (manager.get_group(name) or {}).get(name) is not None


def _lookup_or_serialize(jit_fn, args, kwargs) -> Optional[Tuple[str, str]]:
    """Run Triton's binder; return None on a cache hit, else (key, spec JSON).

    Mirrors the first half of JITFunction.run (Triton 3.8): binder ->
    compute_cache_key -> kernel_cache lookup. On a miss, packs the arguments
    exactly like run() does and serializes them with Triton's own
    serialize_specialization_data, which JITFunction.preload consumes.
    """
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key, serialize_specialization_data

    kwargs = dict(kwargs)
    kwargs["debug"] = kwargs.get("debug", jit_fn.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, target, backend, binder = jit_fn.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    key = compute_cache_key(kernel_key_cache, specialization, options)
    if kernel_cache.get(key) is not None:
        return None
    options, signature, constexprs, attrs = jit_fn._pack_args(
        backend, kwargs, bound_args, specialization, options
    )
    if _on_disk(jit_fn, signature, constexprs, attrs, options, backend):
        return None
    from triton.runtime.jit import get_full_name

    return key, serialize_specialization_data(
        get_full_name(jit_fn.fn), signature, constexprs, attrs, options, key, target
    )


# ---------------------------------------------------------------------------
# Stats: what JIT cost remained on the executor's critical path
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    compiles: int = 0
    compile_s: float = 0.0
    disk_hits: int = 0
    disk_hit_s: float = 0.0
    tunes: int = 0
    tune_s: float = 0.0
    planned: int = 0
    already_compiled: int = 0
    submitted: int = 0
    helper_ok: int = 0
    helper_fail: int = 0
    helper_s: float = 0.0
    plan_s: float = 0.0
    wait_n: int = 0
    wait_s: float = 0.0
    bg_submitted: int = 0
    replay_submitted: int = 0
    recorded: int = 0
    urgent: int = 0
    events: List[str] = field(default_factory=list)


class JitPrefetcher:
    """Per-process singleton. Created after warmup by the model engine."""

    _instance: Optional["JitPrefetcher"] = None

    def __init__(self, rank: int):
        self.rank = rank
        self.prefetch = prefetch_enabled()
        self.stats = _Stats()
        self._providers: List[Callable[[Any], List[KernelCall]]] = []
        self._provider_names: set = set()
        self._spec_prio: Dict[str, int] = {}
        self._bg_started = False
        self._inflight: Dict[int, str] = {}
        self._tag_event: Dict[int, threading.Event] = {}
        self._key_event: Dict[str, threading.Event] = {}
        self._key_spec: Dict[str, Tuple[str, str]] = {}
        self._record_fh = None
        self._recorded: set = set()
        self._replay: List[Tuple[str, str, str]] = []
        self._tag = 0
        self._lock = threading.Lock()
        self._procs = []
        self._busy: Dict[int, threading.Event] = {}
        self._req_q = None

        self._executor_thread: Optional[int] = None
        self._ready: Dict[int, threading.Event] = {}
        self._install_hooks()
        self._open_record()
        if self.prefetch:
            self._start_helpers()
            # Hash the Triton install now (cached for the process lifetime),
            # so the first serving-time disk-cache check does not pay it.
            from triton.runtime.cache import triton_key

            triton_key()
            for key, spec, label in self._replay:
                self._submit(key, spec, label, _PRIO_REPLAY)
            if self._replay:
                self._event(f"replaying {len(self._replay)} recorded variants")

    @classmethod
    def get(cls) -> Optional["JitPrefetcher"]:
        return cls._instance

    @classmethod
    def init(cls, rank: int) -> Optional["JitPrefetcher"]:
        if cls._instance is None and stats_enabled():
            cls._instance = JitPrefetcher(rank)
        return cls._instance

    # -- helpers ----------------------------------------------------------
    def _start_helpers(self):
        import subprocess
        import sys

        from triton import knobs

        cache_dir = knobs.cache.dir
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jit_prefetch_helper.py")
        # A plain interpreter on a script, not multiprocessing: a "spawn"
        # child re-imports the parent's __main__, which under an MPI launcher
        # would initialize MPI and CUDA in the helper.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "UCX_", "MPI_"))
        }
        env.update(TRITON_CACHE_DIR=cache_dir, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        n = max(1, int(os.environ.get(_WORKERS_ENV, str(_default_workers()))))
        self._procs = []
        self._req_q = queue.PriorityQueue()
        for i in range(n):
            p = subprocess.Popen(
                [sys.executable, "-u", helper, pkg_root],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                env=env,
                cwd="/tmp",
                text=True,
                bufsize=1,
            )
            self._procs.append(p)
            self._ready[p.pid] = threading.Event()
            threading.Thread(target=self._feed_loop, args=(p,), daemon=True).start()
            threading.Thread(target=self._drain_loop, args=(p,), daemon=True).start()
        logger.info(
            f"[JIT prefetch] rank {self.rank}: {n} helper process(es) compiling into {cache_dir}"
        )

    def wait_ready(self, timeout_s: float = 60.0) -> float:
        """Block until every helper has imported Triton; return seconds waited.

        Called at the end of warmup, so a helper's start-up (about 2 s for
        the interpreter and the Triton import) is never paid by a request.
        """
        t0 = time.perf_counter()
        for ev in self._ready.values():
            ev.wait(timeout_s)
        dt = time.perf_counter() - t0
        n_ready = sum(ev.is_set() for ev in self._ready.values())
        self._event(f"helpers ready {n_ready}/{len(self._ready)} after {dt * 1e3:.0f} ms")
        return dt

    def _feed_loop(self, proc):
        # One request in flight per helper, so work spreads across helpers.
        while True:
            _prio, tag, spec = self._req_q.get()
            self._busy.setdefault(proc.pid, threading.Event()).clear()
            try:
                proc.stdin.write(json.dumps({"tag": tag, "spec": spec}) + "\n")
                proc.stdin.flush()
            except OSError:
                return
            self._busy[proc.pid].wait()

    def _drain_loop(self, proc):
        for line in proc.stdout:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ready"):
                self._ready[proc.pid].set()
                continue
            tag, ok, dt, err = r["tag"], r["ok"], r["s"], r["err"]
            ev_busy = self._busy.get(proc.pid)
            if ev_busy is not None:
                ev_busy.set()
            with self._lock:
                label = self._inflight.pop(tag, "?")
                ev = self._tag_event.pop(tag, None)
                if ev is not None:
                    ev.set()
                self.stats.helper_s += dt
                if ok:
                    self.stats.helper_ok += 1
                else:
                    self.stats.helper_fail += 1
                    logger.warning(f"[JIT prefetch] helper failed on {label}: {err}")
            self._event(f"helper {'done' if ok else 'FAIL'} {label} {dt * 1e3:.0f} ms")

    def register(self, name: str, provider: Callable[[Any], List[KernelCall]]):
        """Register a variant provider once per name (warmup may run twice)."""
        if name in self._provider_names:
            return False
        self._provider_names.add(name)
        self._providers.append(provider)
        return True

    # -- planning (executor thread) --------------------------------------
    def bind_executor_thread(self) -> None:
        """Mark the calling thread as the executor loop (idempotent)."""
        if self._executor_thread is None:
            self._executor_thread = threading.get_ident()
            self._event("executor thread bound; counting JIT from here")

    def plan(self, batch_ctx: Any) -> None:
        """Called once a batch is scheduled, before its inputs are prepared."""
        self.bind_executor_thread()
        if not self.prefetch or not self._providers:
            return
        if not self._bg_started:
            self._bg_started = True
            threading.Thread(
                target=self._enumerate_all, daemon=True, name="jit_prefetch_enumerate"
            ).start()
        t0 = time.perf_counter()
        for provider in self._providers:
            try:
                calls = provider(batch_ctx)
            except Exception as e:  # noqa: BLE001 - never break serving
                logger.warning_once(
                    f"[JIT prefetch] provider failed: {e}", key=f"jitp_provider_{id(provider)}"
                )
                continue
            for call in calls:
                self._plan_call(call)
        self.stats.plan_s += time.perf_counter() - t0

    def _plan_call(self, call: KernelCall, priority: int = 0):
        """Plan one call. ``priority`` 0 = the batch being scheduled now,
        1 = background enumeration; lower is compiled first."""
        jit_fn, _, _ = _unwrap_jit(call.fn)
        try:
            variants, tuned = _expand(call)
        except Exception as e:  # noqa: BLE001
            logger.warning_once(
                f"[JIT prefetch] cannot expand {call.label}: {e}", key=f"jitp_expand_{call.label}"
            )
            return
        for kw in variants:
            self.stats.planned += 1
            try:
                found = _lookup_or_serialize(jit_fn, call.args, kw)
            except Exception as e:  # noqa: BLE001
                logger.warning_once(
                    f"[JIT prefetch] binder failed on {call.label}: {e}",
                    key=f"jitp_bind_{call.label}",
                )
                continue
            if found is None:
                self.stats.already_compiled += 1
                continue
            key, spec = found
            if not self._submit(key, spec, call.label, priority):
                continue
            if os.environ.get("TLLM_JIT_PREFETCH_DEBUG_KEYS") == "1":
                print(f"[JIT keys] PLANNED {jit_fn.fn.__name__} key={key}", flush=True)
            if priority == _PRIO_BATCH:
                self._event(f"submit {call.label}{'' if tuned else ' (untuned: all configs)'}")

    def _submit(self, key: str, spec: str, label: str, priority: int) -> bool:
        """Queue one variant; False if it is already queued at this priority
        or a more urgent one.

        A variant queued at a lower priority and now needed sooner is queued
        again at the new priority, sharing the first request's completion
        event; whichever helper finishes first releases a waiting executor.
        The other request is a disk-cache hit in its helper.
        """
        with self._lock:
            prev = self._spec_prio.get(spec)
            if prev is not None and prev <= priority:
                return False
            self._spec_prio[spec] = priority
            self._tag += 1
            tag = self._tag
            self._inflight[tag] = label
            ev = self._key_event.get(key)
            if ev is None or ev.is_set():
                ev = threading.Event()
                self._key_event[key] = ev
            self._tag_event[tag] = ev
            self._key_spec[key] = (spec, label)
            self.stats.submitted += 1
            if priority == _PRIO_REPLAY:
                self.stats.replay_submitted += 1
            elif priority == _PRIO_ENUM:
                self.stats.bg_submitted += 1
            elif priority == _PRIO_URGENT:
                self.stats.urgent += 1
        self._req_q.put((priority, tag, spec))
        return True

    def _enumerate_all(self) -> None:
        """Queue every variant each provider can reach, at background priority.

        Runs on its own thread so the executor never waits for it. It plans
        with the same meta-tensor replay as ``plan``, only over the batch
        compositions a provider enumerates (``enumerate_batches``) instead of
        the one batch being scheduled. Variants the current batch needs are
        always queued ahead of these.
        """
        t0 = time.perf_counter()
        n_batches = 0
        for provider in self._providers:
            enum = getattr(provider, "enumerate_batches", None)
            if enum is None:
                continue
            seen: set = set()
            for batch_ctx in enum():
                n_batches += 1
                try:
                    calls = provider(batch_ctx, seen=seen)
                except Exception as e:  # noqa: BLE001
                    logger.warning_once(
                        f"[JIT prefetch] enumeration failed: {e}", key=f"jitp_enum_{id(provider)}"
                    )
                    break
                for call in calls:
                    self._plan_call(call, priority=_PRIO_ENUM)
        self._event(
            f"background enumeration: {n_batches} batch classes, "
            f"{self.stats.bg_submitted} variants queued in "
            f"{(time.perf_counter() - t0) * 1e3:.0f} ms"
        )

    # -- measurement hooks ------------------------------------------------
    def _install_hooks(self):
        from triton import knobs

        stats = self.stats
        me = self

        prev_compile_listener = knobs.compilation.listener

        def compile_listener(*, src, metadata, metadata_group, times, cache_hit):
            # Fires inside triton.compile in this process, for every in-memory
            # cache miss: cache_hit=True means the cubin was loaded from the
            # on-disk cache, False means a real compile ran here. Counted only
            # on the executor loop's thread, once it has been seen.
            if me._executor_thread is not None and threading.get_ident() == me._executor_thread:
                name = getattr(getattr(src, "fn", None), "__name__", "?")
                total_s = times.total / 1e6
                if cache_hit:
                    stats.disk_hits += 1
                    stats.disk_hit_s += total_s
                else:
                    stats.compiles += 1
                    stats.compile_s += total_s
                    me._event(f"COMPILE on executor {name} {total_s * 1e3:.0f} ms")
            if prev_compile_listener is not None:
                prev_compile_listener(
                    src=src,
                    metadata=metadata,
                    metadata_group=metadata_group,
                    times=times,
                    cache_hit=cache_hit,
                )

        knobs.compilation.listener = compile_listener

        prev_cache_hook = knobs.runtime.jit_cache_hook
        wait_timeout_s = float(os.environ.get("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "300"))

        def cache_hook(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
            # Called by JITFunction._do_compile just before it compiles a
            # variant missing from the in-memory cache. If a helper is
            # already compiling that exact variant, wait for its cubin rather
            # than compiling it again here; the compile that follows then
            # loads it from disk. Returning a falsy value lets the launch
            # proceed normally.
            ev = me._key_event.get(key)
            if (
                ev is None
                and me._executor_thread is not None
                and threading.get_ident() == me._executor_thread
                and os.environ.get("TLLM_JIT_PREFETCH_DEBUG_KEYS") == "1"
            ):
                print(f"[JIT keys] UNPLANNED {fn.name} key={key}", flush=True)
            if ev is not None and not ev.is_set():
                # Still queued behind other work: move it to the front so the
                # executor waits for one compile, not for the queue to drain.
                spec_label = me._key_spec.get(key)
                if spec_label is not None:
                    me._submit(key, spec_label[0], spec_label[1], _PRIO_URGENT)
                t0 = time.perf_counter()
                ev.wait(wait_timeout_s)
                dt = time.perf_counter() - t0
                stats.wait_n += 1
                stats.wait_s += dt
                me._event(f"WAIT for helper {fn.name} {dt * 1e3:.0f} ms")
            if prev_cache_hook is not None:
                return prev_cache_hook(
                    key=key,
                    repr=repr,
                    fn=fn,
                    compile=compile,
                    is_manual_warmup=is_manual_warmup,
                    already_compiled=already_compiled,
                )
            return None

        knobs.runtime.jit_cache_hook = cache_hook

        prev_post_hook = knobs.runtime.jit_post_compile_hook

        def post_compile_hook(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
            # JITFunction._do_compile calls this after every in-memory cache
            # miss in this process: a real compile or a disk-cache load. That
            # is exactly the set of variants this process launched, which is
            # what the next process should compile first (approach B).
            spec = compile.get("specialization_data") if isinstance(compile, dict) else None
            if me._record_fh is not None and spec and spec not in me._recorded:
                with me._lock:
                    if spec not in me._recorded:
                        me._recorded.add(spec)
                        me._record_fh.write(json.dumps({"name": fn.name, "spec": spec}) + "\n")
                        me._record_fh.flush()
                        stats.recorded += 1
            if prev_post_hook is not None:
                return prev_post_hook(
                    key=key,
                    repr=repr,
                    fn=fn,
                    compile=compile,
                    is_manual_warmup=is_manual_warmup,
                    already_compiled=already_compiled,
                )
            return None

        knobs.runtime.jit_post_compile_hook = post_compile_hook

        prev_listener = knobs.autotuning.listener

        def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
            if (
                duration is not None
                and me._executor_thread is not None
                and threading.get_ident() == me._executor_thread
            ):
                stats.tunes += 1
                stats.tune_s += duration
                me._event(
                    f"AUTOTUNE on executor {fn.fn.__name__} "
                    f"{len(configs_timings)} cfgs {duration * 1e3:.0f} ms"
                )
            if prev_listener is not None:
                prev_listener(
                    fn=fn,
                    key=key,
                    best_config=best_config,
                    configs_timings=configs_timings,
                    duration=duration,
                    cache_hit=cache_hit,
                )

        knobs.autotuning.listener = listener

    def _open_record(self) -> None:
        """Open this rank's record file and load the variants to replay.

        ``$TLLM_JIT_RECORD_DIR/jit_record.rank<r>.jsonl``: a header line
        (record format, Triton version, GPU target) followed by one line per
        variant, holding Triton's own specialization JSON. A record written
        by another Triton version or for another GPU target is discarded and
        rewritten. A stale entry can only cost a helper compile: the executor
        loads a cubin solely when its own cache-key lookup asks for it.
        """
        rec_dir = os.environ.get(_RECORD_ENV)
        if not rec_dir:
            return
        import triton
        from triton.runtime.driver import driver

        header = {
            "jit_record": 1,
            "triton": triton.__version__,
            "target": str(driver.active.get_current_target()),
        }
        os.makedirs(rec_dir, exist_ok=True)
        path = os.path.join(rec_dir, f"jit_record.rank{self.rank}.jsonl")
        mode = "w"
        if os.path.exists(path):
            with open(path) as f:
                lines = f.read().splitlines()
            try:
                valid = bool(lines) and json.loads(lines[0]) == header
            except json.JSONDecodeError:
                valid = False
            if valid:
                mode = "a"
                for line in lines[1:]:
                    try:
                        rec = json.loads(line)
                        spec = rec["spec"]
                        key = json.loads(spec)["key"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    if spec not in self._recorded:
                        self._recorded.add(spec)
                        self._replay.append((key, spec, str(rec.get("name", "?"))))
        self._record_fh = open(path, mode)
        if mode == "w":
            self._record_fh.write(json.dumps(header) + "\n")
            self._record_fh.flush()
        self._event(f"record {path}: {len(self._replay)} variants to replay")

    def _event(self, msg: str):
        line = f"[JIT stats] rank {self.rank} t={time.time():.3f} {msg}"
        self.stats.events.append(line)
        print(line, flush=True)

    def summary(self) -> str:
        s = self.stats
        return (
            f"[JIT stats] rank {self.rank} SUMMARY prefetch={int(self.prefetch)}"
            f" executor_compiles={s.compiles} executor_compile_s={s.compile_s:.3f}"
            f" executor_disk_hits={s.disk_hits} disk_hit_s={s.disk_hit_s:.3f}"
            f" executor_autotunes={s.tunes} executor_autotune_s={s.tune_s:.3f}"
            f" planned={s.planned} already_compiled={s.already_compiled}"
            f" submitted={s.submitted} helper_ok={s.helper_ok}"
            f" helper_fail={s.helper_fail} helper_s={s.helper_s:.3f}"
            f" plan_s={s.plan_s:.3f} wait_n={s.wait_n} wait_s={s.wait_s:.3f}"
            f" bg_submitted={s.bg_submitted} replay_submitted={s.replay_submitted}"
            f" recorded={s.recorded} urgent={s.urgent} workers={len(self._procs)}"
        )

    def shutdown(self):
        """Print the summary for the executor that is stopping.

        The singleton outlives an executor: the KV-cache estimation pass
        builds and stops one before the serving executor starts. Helpers are
        not stopped here; each exits when this process exits and its stdin
        closes.
        """
        # print(), not only the logger: the summary is the experiment's
        # measurement and must reach the log at any log level.
        print(self.summary(), flush=True)
