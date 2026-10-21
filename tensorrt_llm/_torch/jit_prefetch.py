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
"""Compile JIT kernel variants ahead of the forward pass that needs them.

Experimental. Enabled by ``TLLM_JIT_PREFETCH=1``.

A kernel variant that warmup never produced compiles synchronously at its
first launch, on the executor thread, stalling the GPU. This module compiles
such variants before the forward pass that launches them, in CPU-only helper
processes, without launching anything on the GPU and without adding work to
warmup.

Variant providers, one per module next to its kernels, say which variants to
compile. A provider replays the module's real Triton launchers on ``meta``
tensors under :func:`shadow_launches`, which records each ``run`` instead of
executing it, so the recorded arguments are exactly those of the real launch.
At the end of warmup a background thread plans one batch per kernel-signature
class each provider can reach within the engine limits, so a process that
has never run this model before is covered; a scheduled batch is planned too,
ahead of that. Some providers take the batch-independent part of a launch
(pool layouts, fixed scalars) from the launches warmup made; they are
enumerated again after each warmup. DeepGEMM and CuTe DSL providers do the
same through those libraries' compile-only entry points
(``jit_prefetch_deep_gemm.py``, ``jit_prefetch_cute_dsl.py``). MiniMax-M3's
MSA ``fmha_sm100`` kernels follow from the engine's geometry alone, so they
are built from the moment the model is constructed, while its weights load,
rather than at the end of warmup (:class:`FmhaSm100Builds`,
``start_model_builds``). So do FlashInfer JIT modules: a module a model's
layers build in their first forward (the TRTLLM-GEN attention launcher of
GLM-5.3-Flash and MiniMax-M3), and the MXFP8 GEMM the autotuner warmup tunes
for decode graphs (DeepSeek-V4.1's Engram WKV, MiniMax-M3;
``start_mxfp8_tune_builds``), from the moment the model is constructed; the
sampling module from the start of warmup. The DeepGEMM kernels and
FlashInfer's low-M BF16 GEMM variants follow from the weight shapes and the
engine limits; they start once the weights are materialized on the GPU, while
their values load (``start_weight_load_builds``).

For every planned Triton variant, Triton's own binder computes Triton's own
cache key, which is checked against the kernel's in-memory cache and then the
on-disk cache (the same lookup ``JITFunction.run`` and ``triton.compile`` do,
minus the launch). Misses go to the helpers, which call ``triton.compile``
into the on-disk cache, so the real launch loads the cubin in about a
millisecond. A priority queue orders the work: a variant the executor is
about to wait on, then the scheduled batch, then provider enumeration. If the
executor reaches a variant a helper is already compiling, it waits for that
compile instead of compiling it a second time; one still queued it compiles
itself.

Autotuned kernels: if the autotuner already picked a config for a key, only
that config is planned; otherwise every candidate config is compiled, so the
benchmark at first launch times cached cubins. The benchmark itself runs on
the GPU, unchanged. Which config runs is always the autotuner's own choice.

Tune ahead (``TLLM_JIT_TUNE_AHEAD``, independent of prefetch) uses the
providers that list ``tune_batches``: at the end of each warmup the candidate
configs of every autotune key those batches reach that is still untuned are
compiled, then the autotuner's own benchmark for that key runs on the GPU,
through the kernel's real launcher on scratch tensors. The first batch to
need the key then finds its config chosen. With prefetch on, the helpers
compile the candidates; without it, a thread pool in this process does
(``TLLM_JIT_TUNE_AHEAD_THREADS``), still inside warmup.

Measurement hooks (always on when the module is enabled, or alone with
``TLLM_JIT_STATS=1``) log every compile and every autotune benchmark that
still runs on the executor thread after warmup, with its wall time. A compile
listed there is a variant no provider covers.
"""

from __future__ import annotations

import importlib
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
_TUNE_ENV = "TLLM_JIT_TUNE_AHEAD"
_TUNE_THREADS_ENV = "TLLM_JIT_TUNE_AHEAD_THREADS"

# Queue priorities, lowest first: the executor is blocked on it; the batch
# being scheduled needs it; a provider can reach it (background enumeration).
_PRIO_URGENT, _PRIO_BATCH, _PRIO_ENUM = -1, 0, 1


def _local_world() -> int:
    """Ranks on this node (they share a node-local JIT cache)."""
    for var in ("OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE", "LOCAL_WORLD_SIZE"):
        try:
            return max(1, int(os.environ[var].split("(")[0]))
        except (KeyError, ValueError):
            continue
    return 1


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


def tune_ahead_enabled() -> bool:
    """Whether the end of warmup runs the autotune benchmark of the untuned
    keys the tune-ahead providers reach (see ``JitPrefetcher.tune``).

    Independent of prefetch. ``TLLM_JIT_TUNE_AHEAD`` decides; unset, it
    follows ``TLLM_JIT_PREFETCH``, so prefetch's default keeps tuning ahead.
    """
    v = os.environ.get(_TUNE_ENV)
    if v is not None:
        return v == "1"
    return prefetch_enabled()


def warmup_jit_coverage_enabled() -> bool:
    """Whether warmup runs the passes that exist only to trigger JIT compilation.

    Each such pass checks this (see the callers). With prefetch on, the
    helpers compile those variants instead, so the passes default to off; a
    pass whose kernels prefetch cannot compile on this device stays on
    regardless. ``TLLM_WARMUP_JIT_COVERAGE`` overrides the default either way.
    """
    v = os.environ.get("TLLM_WARMUP_JIT_COVERAGE")
    if v is not None:
        return v == "1"
    return not prefetch_enabled()


def disagg_context_only() -> bool:
    """Whether this process is a disaggregated context (prefill) worker, set
    by the disaggregated server for its executor ranks (``TRTLLM_DISAGG_ROLE``).

    Such a worker serves context-only requests and hands the KV cache to a
    generation worker; the only generation steps it runs are attention-DP
    padding dummies. When the JIT-coverage passes are off
    (``warmup_jit_coverage_enabled``), warmup then also skips the one that
    exists only for the kernels of generation requests (the TRTLLM-Gen
    generation FMHA's NVRTC grid)."""
    return os.environ.get("TRTLLM_DISAGG_ROLE") == "context"


def stats_enabled() -> bool:
    return prefetch_enabled() or os.environ.get(_STATS_ENV, "0") == "1"


def jit_engine_enabled() -> bool:
    """Whether the engine creates a ``JitPrefetcher``: for prefetch, for
    tune ahead or for the JIT stats, in any combination."""
    return stats_enabled() or tune_ahead_enabled()


# ---------------------------------------------------------------------------
# FlashInfer JIT modules built ahead of warmup
# ---------------------------------------------------------------------------

# The CPU-only builds started in this process, oldest first: the module names
# each one builds and its process. Warmup joins them (or stops them).
_fi_builds: List[Tuple[Tuple[str, ...], Any]] = []
# Every module a build has been started for, so none is started twice.
_fi_started: set = set()


def _flashinfer_is_compiled(name: str) -> bool:
    """Whether FlashInfer's JIT module ``name`` is on disk, its AOT library or
    a JIT build, as ``JitSpecNvcc.is_compiled`` checks it. Reads the paths
    without generating the spec: ``fmha_gen``'s generator downloads its
    kernel header."""
    from flashinfer.jit import env as jit_env

    return any(
        (root / name / f"{name}.so").exists()
        for root in (jit_env.FLASHINFER_AOT_DIR, jit_env.FLASHINFER_JIT_DIR)
    )


def start_flashinfer_build(modules: Tuple[str, ...] = ("sampling",)) -> bool:
    """Start building FlashInfer JIT ``modules`` in a CPU-only helper process.

    FlashInfer builds a JIT module with nvcc at its first use, about a minute
    with a cold cache, during which the GPU idles. Warmup builds the sampling
    module (``warmup_sampling_module``) before its first forward; a model's
    layers may build others in warmup's first forwards (see
    ``start_model_builds``). Started ahead of that use, the helper builds the
    modules, one after another in the given order, while the process does
    other work: the sampling module from the beginning of warmup, a layer's
    module while the weights load. ``wait_flashinfer_build`` joins the helpers
    before the first warmup phase that can launch the sampling kernels (CUDA
    graph capture); a use before that waits on FlashInfer's own file lock,
    which the helper holds while it builds the module. Either way the
    in-process ``build_and_load`` only loads the result. The helper has no
    visible GPU: ``FLASHINFER_CUDA_ARCH_LIST`` gives it this process's
    FlashInfer targets, which also name the cache directory, so the build
    lands where this process looks. Returns False if prefetch is off,
    FlashInfer is missing, or every module is built already or being built.
    """
    if not prefetch_enabled() or os.environ.get("FLASHINFER_DISABLE_JIT"):
        return False
    try:
        import torch

        from .flashinfer_utils import IS_FLASHINFER_AVAILABLE

        if not IS_FLASHINFER_AVAILABLE or not torch.cuda.is_available():
            return False
        from flashinfer.compilation_context import CompilationContext

        # Nothing to build: an AOT artifact or a JIT build already on disk, or
        # a helper started on it before.
        pending = [
            name
            for name in modules
            if name not in _fi_started and not _flashinfer_is_compiled(name)
        ]
        if not pending:
            return False
        # The targets as FlashInfer resolved them here, suffix included
        # ("10.3a"), which it takes verbatim from the environment.
        archs = " ".join(
            f"{major}.{minor}" for major, minor in sorted(CompilationContext().TARGET_CUDA_ARCHS)
        )
        if not archs:
            return False
    except Exception as e:  # noqa: BLE001 - the build then happens in place
        logger.warning(f"[JIT prefetch] FlashInfer build not started: {type(e).__name__}: {e}")
        return False
    import subprocess
    import sys

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "UCX_", "MPI_"))
    }
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", FLASHINFER_CUDA_ARCH_LIST=archs)
    env.pop("PYTHONPATH", None)
    helper = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "jit_prefetch_flashinfer_helper.py"
    )
    # In a process group of its own, with the ninja it runs, so that
    # cancel_flashinfer_build can signal both. Its stdout carries only the
    # result lines; its log goes to this process's stderr.
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", helper, str(os.getpid()), *pending],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=env,
            cwd="/tmp",
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        logger.warning(f"[JIT prefetch] FlashInfer build not started: {type(e).__name__}: {e}")
        return False
    try:
        os.setpriority(os.PRIO_PROCESS, proc.pid, 19)
    except OSError:
        pass
    _fi_builds.append((tuple(pending), proc))
    _fi_started.update(pending)
    logger.info(f"[JIT prefetch] building FlashInfer {', '.join(pending)} in a CPU-only helper")
    return True


def wait_flashinfer_build(timeout_s: Optional[float] = None) -> Optional[float]:
    """Join the helpers ``start_flashinfer_build`` started; return the seconds
    waited, or None if none was started. Called where warmup needs the
    modules: a failed or timed-out build only means the module is built in
    place. ``timeout_s`` bounds the whole join."""
    import subprocess

    if not _fi_builds:
        return None
    if timeout_s is None:
        timeout_s = float(os.environ.get("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "300"))
    t0 = time.perf_counter()
    while _fi_builds:
        modules, proc = _fi_builds[0]
        try:
            out, _ = proc.communicate(timeout=max(0.0, timeout_s - (time.perf_counter() - t0)))
        except subprocess.TimeoutExpired:
            unjoined = [name for names, _ in _fi_builds for name in names]
            cancel_flashinfer_build()
            logger.warning(
                f"[JIT prefetch] FlashInfer {', '.join(unjoined)} build(s) still running after"
                f" {timeout_s:.0f} s: stopped, built in place instead"
            )
            return time.perf_counter() - t0
        _fi_builds.pop(0)
        reported = set()
        for line in (out or "").splitlines():
            try:
                r = json.loads(line)
                name, ok, secs = r["module"], r["ok"], float(r["s"])
            except (ValueError, TypeError, KeyError):
                continue
            if name not in modules:
                continue
            reported.add(name)
            if not ok:
                logger.warning(f"[JIT prefetch] FlashInfer {name} build failed: {r.get('err')}")
            elif not _flashinfer_is_compiled(name):
                logger.warning(
                    f"[JIT prefetch] FlashInfer {name}: the helper's build is not where this"
                    " process loads it from; built in place instead"
                )
            else:
                logger.info(
                    f"[JIT prefetch] FlashInfer {name} built by the helper in {secs:.1f} s"
                    f" (waited {time.perf_counter() - t0:.1f} s)"
                )
        missing = [m for m in modules if m not in reported]
        if missing:
            logger.warning(
                f"[JIT prefetch] FlashInfer helper exited with code {proc.returncode} before"
                f" building {', '.join(missing)}"
            )
    return time.perf_counter() - t0


def cancel_flashinfer_build() -> None:
    """Stop the helpers ``start_flashinfer_build`` started that have not been
    joined, and return once they have exited. Their modules are then built in
    place."""
    import signal
    import subprocess

    while _fi_builds:
        _, proc = _fi_builds.pop(0)
        # SIGTERM to the helper's process group, sent again each second until
        # the helper exits, so a ninja started after the first one gets it as
        # well. ninja stops the compilers it runs (each in a process group of
        # its own) and removes their partial outputs; the helper waits for it.
        # A compiler left running would write into the directory the in-place
        # build uses.
        sig, deadline = signal.SIGTERM, time.monotonic() + 60.0
        while proc.poll() is None:
            if time.monotonic() > deadline:
                if sig == signal.SIGKILL:
                    logger.warning("[JIT prefetch] FlashInfer helper did not exit; left running")
                    break
                logger.warning(
                    "[JIT prefetch] FlashInfer helper did not stop on SIGTERM; killing it"
                )
                sig, deadline = signal.SIGKILL, time.monotonic() + 10.0
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        proc.stdout.close()


def flashinfer_builds_pending() -> bool:
    """Whether a helper ``start_flashinfer_build`` started is still to be
    joined (``wait_flashinfer_build``) or stopped (``cancel_flashinfer_build``)."""
    return bool(_fi_builds)


def start_model_builds(model, rank: int = 0, llm_args=None) -> bool:
    """Start building, in CPU-only helpers, the JIT kernels ``model``'s layers
    build in warmup's first forwards, which follow from the model's geometry
    alone; called once the model is constructed, so the builds run while the
    weights load instead of inside warmup:

    * FlashInfer JIT modules, one helper for all of them: each provider
      module that knows of such modules lists them (GLM-5.3-Flash's and
      MiniMax-M3's TRTLLM-GEN attention launcher ``fmha_gen``);
    * the FlashInfer modules of the autotuner warmup's FlashInfer MXFP8 pass,
      in a helper of their own, if the engine ``llm_args`` describes will run
      that pass (``tunes_flashinfer_mxfp8_at_load``: DeepSeek-V4.1's Engram
      WKV and MiniMax-M3's MXFP8 Linears with decode CUDA graphs). Warmup
      starts them otherwise (``start_mxfp8_tune_builds``);
    * the CuTe DSL FP8 blockwise GEMM tactics the autotuner benchmarks
      (``jit_prefetch_cute_dsl.Fp8GemmProvider``: block-FP8 Linear layers
      without DeepGEMM, and the batched GEMM of DeepSeek-V4's grouped FP8
      O-LoRA projection) and the KDA prefill kernel (``KdaPrefillProvider``);
    * MiniMax-M3's MSA ``fmha_sm100`` kernels (:class:`FmhaSm100Builds`),
      whose plan also needs the KV cache's page size and dtypes, taken from
      ``llm_args``. Without it they start at the beginning of warmup instead.

    The DeepGEMM and low-M GEMM kernels start a little later, once the weights
    are on the GPU (``start_weight_load_builds``).

    Returns False if prefetch is off or nothing was started."""
    if not prefetch_enabled():
        return False
    started = False
    modules: List[str] = []
    for provider_module in (
        "attention.backends.sparse.glm_kpool.jit_prefetch",
        "attention.backends.sparse.minimax_m3.jit_prefetch_fmha",
    ):
        try:
            planner = importlib.import_module(f".{provider_module}", __package__)
            modules += planner.flashinfer_modules(model)
        except Exception as e:  # noqa: BLE001 - its modules then build in place
            logger.warning(
                f"[JIT prefetch] FlashInfer builds of {provider_module} not planned: "
                f"{type(e).__name__}: {e}"
            )
    if modules:
        started = start_flashinfer_build(tuple(dict.fromkeys(modules)))
    if llm_args is not None:
        try:
            started = (
                start_mxfp8_tune_builds(tunes_flashinfer_mxfp8_at_load(model, llm_args)) or started
            )
        except Exception as e:  # noqa: BLE001 - warmup then starts them
            logger.warning(
                f"[JIT prefetch] FlashInfer MXFP8 builds not planned: {type(e).__name__}: {e}"
            )
    try:
        from .cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if IS_CUTLASS_DSL_AVAILABLE:
            from . import jit_prefetch_cute_dsl as jcd

            # Both follow from the layers' geometry and are compiled in
            # warmup's first forwards: the FP8 GEMM tactics by the
            # autotuner, the KDA prefill kernel by the first prefill.
            gemm_provider = jcd.Fp8GemmProvider(model)
            kda_provider = jcd.KdaPrefillProvider(model)
            if gemm_provider or kda_provider:
                prefetcher = JitPrefetcher.init(rank)
                if prefetcher is not None:
                    if gemm_provider:
                        started = prefetcher.enable_fp8_gemm(gemm_provider) or started
                    if kda_provider:
                        started = prefetcher.enable_cute_dsl(kda_provider) or started
    except Exception as e:  # noqa: BLE001 - the kernels then compile in place
        logger.warning(f"[JIT prefetch] CuTe DSL builds not planned: {type(e).__name__}: {e}")
    if llm_args is not None and FmhaSm100Builds.get() is None:
        try:
            from .attention.backends.sparse.minimax_m3.jit_prefetch_fmha import (
                MiniMaxM3FmhaProvider,
                kv_cache_geometry,
            )

            geometry = kv_cache_geometry(model, llm_args)
            if geometry is not None:
                started = FmhaSm100Builds.start(MiniMaxM3FmhaProvider(model, geometry)) or started
        except Exception as e:  # noqa: BLE001 - the kernels then build in place
            logger.warning(f"[JIT prefetch] fmha_sm100 builds not started: {type(e).__name__}: {e}")
    return started


def start_weight_load_builds(model, rank: int, deep_gemm_limits) -> bool:
    """Start, in CPU-only helpers, the GEMM kernels ``model``'s first warmup
    forwards and the autotuner warmup build, once its weights are
    materialized on the GPU and while their values load from the checkpoint:

    * the DeepGEMM FP8 Linear and MoE kernels (``start_deep_gemm_builds``);
    * FlashInfer's low-M BF16 direct GEMM variants the low-M dispatcher's
      AutoTuner race benchmarks (``jit_prefetch_cute_dsl.LowMDirectGemmProvider``).

    ``deep_gemm_limits`` are the engine limits the providers plan from
    (``max_num_tokens``, ``priority_ms``, ``first_ms``). Returns False if
    prefetch is off or nothing was started."""
    if not prefetch_enabled():
        return False
    started = start_deep_gemm_builds(model, rank, **deep_gemm_limits)
    try:
        from .cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if IS_CUTLASS_DSL_AVAILABLE:
            from . import jit_prefetch_cute_dsl as jcd

            lowm_provider = jcd.LowMDirectGemmProvider(model, deep_gemm_limits["max_num_tokens"])
            if lowm_provider:
                prefetcher = JitPrefetcher.init(rank)
                if prefetcher is not None:
                    started = prefetcher.enable_lowm_direct_gemm(lowm_provider) or started
    except Exception as e:  # noqa: BLE001 - the kernels then compile in place
        logger.warning(f"[JIT prefetch] low-M GEMM builds not planned: {type(e).__name__}: {e}")
    return started


def start_deep_gemm_builds(
    model, rank: int, max_num_tokens: int, priority_ms=(), first_ms=()
) -> bool:
    """Start the DeepGEMM helpers on the FP8 Linear and MoE kernels of
    ``model`` while its weights load.

    The variants follow from the weight shapes and the engine limits, which
    is everything ``FP8LinearDeepGemmProvider`` and ``DeepGemmMoEProvider``
    read. Started here, the ~10 s start-up of the helpers and the first
    layouts the first warmup forwards need (``first_ms``, then ``priority_ms``,
    then M = 1..16) are done by the time the weights are in; started at
    warmup, the first forward waits for them. Shapes created by weight loading itself
    come from the model's ``jit_prefetch_fp8_weight_read_shapes`` where it
    names them, and are otherwise added at warmup
    (``JitPrefetcher.refresh_deep_gemm``). Once per process; returns False if
    prefetch is off or nothing was started."""
    if not prefetch_enabled():
        return False
    try:
        from .._utils import get_sm_version
        from . import jit_prefetch_deep_gemm as jdg

        if get_sm_version() not in (100, 103) or not jdg.supported():
            return False
        prefetcher = JitPrefetcher.init(rank)
        if prefetcher is None or prefetcher.deep_gemm_enabled():
            return False
        dg_provider = jdg.FP8LinearDeepGemmProvider(
            model, max_num_tokens=max_num_tokens, priority_ms=priority_ms, first_ms=first_ms
        )
        dg_moe = jdg.DeepGemmMoEProvider(model)
        if not (dg_provider or dg_moe):
            return False
        return prefetcher.enable_deep_gemm(dg_provider, dg_moe)
    except Exception as e:  # noqa: BLE001 - the helpers then start at warmup
        logger.warning(
            f"[JIT prefetch] DeepGEMM helpers not started at model load: {type(e).__name__}: {e}"
        )
        return False


# FlashInfer modules the autotuner warmup's FlashInfer MXFP8 pass builds the
# first time it tunes ``mm_mxfp8`` on an SM100-family GPU, in the order it
# needs them: the delay kernel FlashInfer's autotuner times with (seconds),
# then the CUTLASS MXFP8 GEMM (over a minute).
FLASHINFER_MXFP8_TUNE_MODULES = ("trtllm_utils", "mxfp8_gemm_cutlass")


def start_mxfp8_tune_builds(tunes_flashinfer_mxfp8: bool) -> bool:
    """Start building, in a CPU-only helper, the FlashInfer modules the
    autotuner warmup's FlashInfer MXFP8 pass builds; called with whether this
    engine runs that pass, once the model is constructed
    (``start_model_builds``, which predicts it from the engine's arguments)
    and again at the start of warmup, where the engine decides it. The GEMM
    module is a ~75-100 s nvcc build, which then overlaps the weight load, or
    at least the warmup forwards before the autotuner; the pass waits on
    FlashInfer's file lock, held by the helper, instead of compiling it
    itself. Returns False if prefetch is off, the pass does not run, or there
    is nothing to build (the modules are built, or a helper is on them)."""
    if not prefetch_enabled() or not tunes_flashinfer_mxfp8:
        return False
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        return False
    return start_flashinfer_build(FLASHINFER_MXFP8_TUNE_MODULES)


def decode_batches_compiled(model, llm_args) -> bool:
    """Whether ``torch.compile`` runs the decode batches of ``model`` in the
    engine ``llm_args`` describes, as ``PyTorchModelEngine`` decides it: with
    a ``torch_compile_config``, unless piecewise prefill CUDA graphs confine
    a decoder model's compiled graph to prefill (the engine's
    ``_torch_compile_prefill_only``)."""
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

    from .models.modeling_utils import DecoderModelForCausalLM

    compile_cfg = llm_args.torch_compile_config
    if compile_cfg is None:
        return False
    return not (
        isinstance(model, DecoderModelForCausalLM)
        and llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
        and (compile_cfg.compile_only_piecewise_graphs or not model.use_fx_for_pcg_fallback)
    )


def tunes_flashinfer_mxfp8_at_load(model, llm_args) -> bool:
    """Whether the engine ``llm_args`` describes will run the autotuner
    warmup's FlashInfer MXFP8 pass on ``model``, known once the model is
    constructed: the decoder runner's own selection
    (``DecoderRunner.flashinfer_mxfp8_pass_selected``) on the settings
    ``PyTorchModelEngine`` derives from ``llm_args``. False if the engine
    runs another runner (encoder, pooling), which has no such pass."""
    from .pyexecutor.engine.runners import DecoderRunner, resolve_runner_type

    runner_cls = resolve_runner_type(model, llm_args)
    if not issubclass(runner_cls, DecoderRunner):
        return False
    return runner_cls.flashinfer_mxfp8_pass_selected(
        model,
        model.model_config.mapping,
        enable_autotuner=llm_args.enable_autotuner,
        cuda_graphs=llm_args.cuda_graph_config is not None,
        compile_all_batches=decode_batches_compiled(model, llm_args),
    )


# ---------------------------------------------------------------------------
# MSA fmha_sm100 kernels built from the start of warmup
# ---------------------------------------------------------------------------


class FmhaSm100Builds:
    """CPU-only helpers building MiniMax-M3's MSA ``fmha_sm100`` kernels.

    ``fmha_sm100`` (the MSA backend's context-phase kernels) builds each
    kernel with nvcc the first time a call needs it, 50-100 s each: a cold
    MiniMax-M3 context server builds about a dozen, most of them in warmup
    forwards and the rest in the first serving steps that reach a new
    context-row class. Every one of them is known from the engine's geometry
    (``minimax_m3/jit_prefetch_fmha.py``), so this pool starts building all of
    them once the model is constructed (``start_model_builds``; at the start
    of warmup if it was not started then), in ``jit_prefetch_fmha_sm100_helper.py``
    processes with no visible GPU, into the directories ``fmha_sm100`` loads
    from, while the weights load and the warmup forwards run. A forward that needs a
    kernel a helper is building waits on the per-kernel file lock that build
    holds, and then loads it; one no helper has started (more kernels than
    helpers) it builds itself, as without prefetch, and the helper then
    finds it built. A failed helper build leaves the kernel to the forward.

    Started once per process (the KV-cache estimation pass and the final
    executor share the kernels); helpers exit when their queue is drained.
    """

    _instance: Optional["FmhaSm100Builds"] = None

    def __init__(self, specs: List[str], pkg_dir: str, cache_base: str, n: int):
        import subprocess
        import sys

        self.specs = list(specs)
        self.t0 = time.perf_counter()
        self.done: Dict[str, Tuple[bool, bool, float, str]] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Tuple[int, str]]" = queue.Queue()
        for tag, spec in enumerate(self.specs):
            self._queue.put((tag, spec))
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "UCX_", "MPI_"))
        }
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        helper = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "jit_prefetch_fmha_sm100_helper.py"
        )
        self.procs = []
        for _ in range(n):
            proc = subprocess.Popen(
                [sys.executable, "-u", helper, str(os.getpid()), pkg_dir, cache_base],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                cwd="/tmp",
                text=True,
                bufsize=1,
            )
            try:
                os.setpriority(os.PRIO_PROCESS, proc.pid, 19)
            except OSError:
                pass
            self.procs.append(proc)
            threading.Thread(target=self._serve, args=(proc,), daemon=True).start()

    def _serve(self, proc) -> None:
        """Feed one helper a request at a time until the queue is empty. A
        helper that exits early leaves its kernels to be built in place."""
        if not proc.stdout.readline():
            logger.warning(f"[JIT prefetch] fmha_sm100 helper {proc.pid} exited before it started")
            return
        while True:
            try:
                tag, spec = self._queue.get_nowait()
            except queue.Empty:
                break
            r = None
            try:
                proc.stdin.write(json.dumps({"tag": tag, "spec": spec}) + "\n")
                proc.stdin.flush()
                r = json.loads(proc.stdout.readline() or "null")
            except (OSError, ValueError):
                pass
            if not isinstance(r, dict):
                logger.warning(
                    f"[JIT prefetch] fmha_sm100 helper {proc.pid} exited while building {spec}"
                )
                with self._lock:
                    self.done[spec] = (False, False, 0.0, "helper exited")
                break
            with self._lock:
                self.done[spec] = (bool(r["ok"]), bool(r["built"]), float(r["s"]), r["err"])
            if not r["ok"]:
                logger.warning(f"[JIT prefetch] fmha_sm100 helper failed on {spec}: {r['err']}")
        try:
            proc.stdin.close()
        except OSError:
            pass

    @classmethod
    def start(cls, provider) -> bool:
        """Start building ``provider.specs`` (once per process). Returns False
        if prefetch is off, there is nothing to build, or it already started."""
        if not prefetch_enabled() or cls._instance is not None or not provider:
            return False
        try:
            from .attention.backends.sparse.minimax_m3.jit_prefetch_fmha import fmha_jit_module

            jit = fmha_jit_module()
            pkg_dir = os.path.dirname(os.path.abspath(jit.__file__))
            cache_base = str(jit.CACHE_BASE)
        except Exception as e:  # noqa: BLE001 - the kernels then build in place
            logger.warning(f"[JIT prefetch] fmha_sm100 builds not started: {type(e).__name__}: {e}")
            return False
        n = max(
            1,
            min(
                len(provider.specs),
                int(os.environ.get("TLLM_JIT_PREFETCH_FMHA_WORKERS", str(_default_workers()))),
            ),
        )
        cls._instance = cls(provider.specs, pkg_dir, cache_base, n)
        # print(), as the [JIT stats] events: the experiment's record.
        print(
            f"[JIT prefetch] building {len(provider.specs)} MSA fmha_sm100 kernel(s) in {n} "
            f"CPU-only helper(s) into {cache_base}",
            flush=True,
        )
        return True

    @classmethod
    def get(cls) -> Optional["FmhaSm100Builds"]:
        return cls._instance

    def counts(self) -> Tuple[int, int, int, int]:
        """(requested, finished, built by a helper, failed)."""
        with self._lock:
            done = list(self.done.values())
        return (
            len(self.specs),
            len(done),
            sum(1 for ok, built, _, _ in done if ok and built),
            sum(1 for ok, _, _, _ in done if not ok),
        )


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


def _untuned(call: KernelCall) -> bool:
    """True if ``call`` is an autotuned launch whose key has no config yet,
    i.e. its next real launch would run the autotune benchmark."""
    _, autotuner, _ = _unwrap_jit(call.fn)
    if autotuner is None or len(autotuner.configs) <= 1:
        return False
    key = _autotune_key(autotuner, None, call.args, dict(call.kwargs))
    return key not in autotuner.cache


def _compile_spec(item: Tuple[Any, str]) -> bool:
    """Compile one variant in this process from Triton's own specialization
    JSON, with ``JITFunction.preload``: the same compile a launch would run,
    into the same in-memory and on-disk caches, launching nothing. Returns
    False if it failed (the launch that needs it then compiles it)."""
    jit_fn, spec = item
    try:
        jit_fn.preload(spec)
        return True
    except Exception as e:  # noqa: BLE001 - compiled at first launch instead
        logger.warning_once(
            f"[JIT prefetch] tune-ahead compile failed for {jit_fn.fn.__name__}: {e}",
            key=f"jitp_tune_compile_{jit_fn.fn.__name__}",
        )
        return False


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
    urgent: int = 0
    dg_planned: int = 0
    dg_unplanned: int = 0
    dg_helper_built: int = 0
    dg_executor_compiles: int = 0
    dg_executor_compile_s: float = 0.0
    # Every DeepGEMM kernel compiled in this process after warmup, from
    # DeepJIT's own record of disk-cache misses (all launchers, all ops).
    dg_process_compiles: int = 0
    cd_loaded: int = 0
    cd_executor_compiles: int = 0
    cd_executor_compile_s: float = 0.0
    # Tune ahead (JitPrefetcher.tune): batches run, autotune benchmarks run
    # and their time, variants the launches still compiled in process (none
    # when the candidates were delivered beforehand), time spent getting the
    # candidates compiled (waiting for the helpers, or the thread pool
    # without prefetch), total time.
    ta_batches: int = 0
    ta_tunes: int = 0
    ta_tune_s: float = 0.0
    ta_compiles: int = 0
    ta_compile_s: float = 0.0
    ta_wait_s: float = 0.0
    ta_s: float = 0.0
    # Tune ahead without prefetch: variants compiled by its thread pool
    # before the benchmark, and how many of those failed.
    ta_inproc_compiles: int = 0
    ta_inproc_failed: int = 0
    # MSA fmha_sm100 kernels (FmhaSm100Builds): built by the helpers, failed
    # there, and built in process on the executor thread after warmup.
    fmha_helper_built: int = 0
    fmha_helper_fail: int = 0
    fmha_executor_compiles: int = 0
    fmha_executor_compile_s: float = 0.0
    events: List[str] = field(default_factory=list)


class JitPrefetcher:
    """Per-process singleton. Created after warmup by the model engine."""

    _instance: Optional["JitPrefetcher"] = None

    def __init__(self, rank: int):
        self.rank = rank
        self.prefetch = prefetch_enabled()
        # The executor-side JIT stats (on with prefetch or TLLM_JIT_STATS);
        # tune ahead alone records only its own counters.
        self.stats_on = stats_enabled()
        self.stats = _Stats()
        self._providers: List[Callable[[Any], List[KernelCall]]] = []
        self._provider_names: set = set()
        self._spec_prio: Dict[str, int] = {}
        self._bg_started = False
        # ids of the providers a background enumeration has covered.
        self._enumerated: set = set()
        self._enum_lock = threading.Lock()
        self._inflight: Dict[int, str] = {}
        self._tag_event: Dict[int, threading.Event] = {}
        self._key_event: Dict[str, threading.Event] = {}
        self._key_spec: Dict[str, Tuple[str, str]] = {}
        # Triton and CuTe DSL keys a helper has been sent: the executor waits
        # only for those, and compiles a still-queued one itself.
        self._tag_key: Dict[int, str] = {}
        self._key_started: set = set()
        self._tag = 0
        self._lock = threading.Lock()
        self._procs = []
        self._procs_by_kind: Dict[str, List[Any]] = {}
        self._stdin_locks: Dict[int, threading.Lock] = {}
        self._busy: Dict[int, threading.Event] = {}
        self._req_q = None
        # kind -> queue; "triton" uses self._req_q
        self._queues: Dict[str, Any] = {}
        self._dg = None  # DeepGEMM provider (jit_prefetch_deep_gemm)
        self._dg_moe = None
        self._cd = None  # CuTe DSL KDA provider (jit_prefetch_cute_dsl)
        self._cd_dir = None
        self._fp8_gemm = False  # the FP8 blockwise GEMM loader is installed
        self._lowm = False  # the low-M direct GEMM loader is installed

        self._executor_thread: Optional[int] = None
        # The thread running JitPrefetcher.tune, while it runs.
        self._tune_thread: Optional[int] = None
        self._ready: Dict[int, threading.Event] = {}
        self._install_hooks()
        if self.prefetch:
            self._start_helpers()
            # Hash the Triton install now (cached for the process lifetime),
            # so the first serving-time disk-cache check does not pay it.
            from triton.runtime.cache import triton_key

            triton_key()

    @classmethod
    def get(cls) -> Optional["JitPrefetcher"]:
        return cls._instance

    @classmethod
    def init(cls, rank: int) -> Optional["JitPrefetcher"]:
        if cls._instance is None and jit_engine_enabled():
            cls._instance = JitPrefetcher(rank)
        return cls._instance

    # -- helpers ----------------------------------------------------------
    def _start_helpers(self):
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
        self._queues["triton"] = self._req_q
        self._spawn("triton", [sys.executable, "-u", helper, pkg_root], env, n)
        logger.info(
            f"[JIT prefetch] rank {self.rank}: {n} helper process(es) compiling into {cache_dir}"
        )

    def _spawn(self, kind: str, argv: List[str], env: Dict[str, str], n: int) -> None:
        import subprocess

        q = self._queues[kind]
        for _ in range(n):
            p = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                env=env,
                cwd="/tmp",
                text=True,
                bufsize=1,
            )
            # Lowest CPU priority, inherited by the compilers a helper forks:
            # a helper only uses CPU time the executor and its threads leave.
            try:
                os.setpriority(os.PRIO_PROCESS, p.pid, 19)
            except OSError:
                pass
            self._procs.append(p)
            self._procs_by_kind.setdefault(kind, []).append(p)
            self._ready[p.pid] = threading.Event()
            threading.Thread(target=self._feed_loop, args=(p, q), daemon=True).start()
            threading.Thread(target=self._drain_loop, args=(p,), daemon=True).start()

    def observe_deep_gemm(self) -> None:
        """Count DeepGEMM FP8 GEMM compiles on the executor thread after warmup.

        Independent of prefetch, so every arm of an experiment reports the
        same counter.
        """
        from . import jit_prefetch_deep_gemm as jdg

        # DeepGEMM keeps a kernel in memory once loaded, so only the first
        # launch of an (m, n, k) can compile; a later slow launch of it is a
        # host stall (e.g. the caching allocator releasing blocks near its
        # memory limit), not a compile.
        seen_shapes: set = set()
        # With the patched DeepJIT, its own record of disk-cache misses says
        # whether this launch compiled: a slow first launch that compiled
        # nothing (it waited on the GPU, or loaded a helper-built cubin) is
        # not counted. Without it, host time is the only signal.
        take = getattr(self._deep_gemm_module(), "take_compiled_kernel_names", None)

        def _done(m, n, k, dt):
            shape = (m, n, k)
            first = shape not in seen_shapes
            seen_shapes.add(shape)
            if not first or dt < jdg.COMPILE_HOST_S:
                return
            if threading.get_ident() != self._executor_thread:
                return
            names = ""
            if take is not None:
                compiled = take()
                if not compiled:
                    return
                self.stats.dg_process_compiles += len(compiled)
                names = " " + ",".join(sorted(set(compiled)))
            self.stats.dg_executor_compiles += 1
            self.stats.dg_executor_compile_s += dt
            # The MoE masked grouped GEMM reports m = -num_groups.
            what = f"moe g={-m}" if m < 0 else f"m={m}"
            self._event(f"DeepGEMM compile on executor {what} n={n} k={k} {dt:.3f}s{names}")

        jdg.done_observer = _done
        if jdg.launch_observer is None:
            # note_launch_done only runs while a launch observer is set.
            jdg.launch_observer = lambda m, n, k: None

    def add_deep_gemm_specs(self, specs, label: str = "deep_gemm") -> int:
        """Queue extra DeepGEMM requests (e.g. the DSA indexer kernels)."""
        from . import jit_prefetch_deep_gemm as jdg

        if self._dg is None and jdg.KIND not in self._queues:
            return 0
        n = 0
        for spec in specs:
            if self._submit(spec, spec, label, _PRIO_BATCH, kind=jdg.KIND):
                n += 1
        if n:
            self._event(f"DeepGEMM: {n} {label} request(s) queued")
        return n

    def deep_gemm_enabled(self) -> bool:
        """Whether ``enable_deep_gemm`` has started the DeepGEMM helpers."""
        return self._dg is not None

    def refresh_deep_gemm(self, provider, moe_provider=None) -> int:
        """Swap in providers built from the loaded model, keeping the helpers
        ``enable_deep_gemm`` started from the constructed one.

        Weight loading can add FP8 GEMM shapes and drops none. A model that
        creates GEMMs while its weights load names their shapes up front
        (Kimi K3's ``jit_prefetch_fp8_weight_read_shapes``), so the first
        providers already plan them; the variants of any shape added here are
        queued the way ``enable_deep_gemm`` queued the first ones. Returns the
        number of new shapes."""
        from . import jit_prefetch_deep_gemm as jdg

        if self._dg is None:
            return 0
        old = set(self._dg.shapes)
        old_moe = set(self._dg_moe.shapes) if self._dg_moe else set()
        new = [s for s in provider.shapes if s not in old]
        new_moe = [s for s in (moe_provider.shapes if moe_provider else ()) if s not in old_moe]
        provider.adopt_seen(self._dg)
        self._dg = provider
        if moe_provider:
            self._dg_moe = moe_provider
        if not (new or new_moe):
            return 0
        extra = jdg.FP8LinearDeepGemmProvider.with_shapes(provider, new)
        extra_moe = jdg.DeepGemmMoEProvider.with_shapes(moe_provider, new_moe) if new_moe else None

        # The enumeration of the constructed model's shapes is already
        # queued, so the new shapes' layouts the first warmup forwards reach
        # go ahead of it, in that order: the token counts of warmup's first
        # forwards (``first_ms``), the priority token counts, one token and a
        # full max_num_tokens batch.
        first_ms = list(dict.fromkeys(
            [*extra.first_ms, *extra.priority_ms, 1, extra.max_num_tokens]
        ))  # fmt: skip

        def _enumerate():
            n = 0
            if extra_moe:
                for spec in extra_moe.enumerate_specs():
                    if self._submit(spec, spec, "deep_gemm", _PRIO_BATCH, kind=jdg.KIND):
                        n += 1
            for m in first_ms:
                for spec in extra.specs_for_m(m):
                    if self._submit(spec, spec, "deep_gemm", _PRIO_BATCH, kind=jdg.KIND):
                        n += 1
            world = int(os.environ.get("TLLM_JIT_PREFETCH_DG_WORLD", "0")) or _local_world()
            for spec in extra.enumerate_specs(self.rank, world):
                if self._submit(spec, spec, "deep_gemm", _PRIO_ENUM, kind=jdg.KIND):
                    n += 1
            self._event(f"DeepGEMM enumeration of shapes added by weight loading: {n} requests")

        threading.Thread(target=_enumerate, daemon=True, name="jit_prefetch_dg_enum2").start()
        self._event(
            f"DeepGEMM prefetch: weight loading added {len(new)} FP8 Linear and "
            f"{len(new_moe)} MoE shape(s)"
        )
        return len(new) + len(new_moe)

    def enable_deep_gemm(self, provider, moe_provider=None) -> bool:
        """Start the DeepGEMM helpers and planning (needs the patched DeepGEMM).

        Helpers run ``jit_prefetch_dg_helper.py`` with no visible GPU; the
        target arch and SM count are this rank's, so they compile exactly the
        variants this rank's launches will look up in ``DG_JIT_CACHE_DIR``.
        """
        import sys

        from . import jit_prefetch_deep_gemm as jdg

        if not self.prefetch or self._dg is not None:
            return False
        if not jdg.supported():
            jdg.log_unsupported_once("DeepGEMM lacks compile_only_fp8_fp4_gemm_nt or not SM100")
            return False
        tgt = jdg.target()
        major, minor, sms = tgt[:3]
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "UCX_", "MPI_"))
        }
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        helper = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "jit_prefetch_dg_helper.py"
        )
        n = max(1, int(os.environ.get("TLLM_JIT_PREFETCH_DG_WORKERS", str(_default_workers()))))
        self._queues[jdg.KIND] = queue.PriorityQueue()
        self._spawn(
            jdg.KIND,
            [sys.executable, "-u", helper, jdg.package_dir(), *(str(x) for x in tgt)],
            env,
            n,
        )
        self._dg = provider
        self._dg_moe = moe_provider
        seen_mnk: set = set()

        def _observe(m, n, k, _seen=seen_mnk):
            if (m, n, k) in _seen:
                return
            _seen.add((m, n, k))
            spec = jdg.spec_for(m, n, k)
            if spec not in self._spec_prio:
                self.stats.dg_unplanned += 1
                self._event(f"DeepGEMM launch not planned: m={m} n={n} k={k}")
            ev = self._key_event.get(spec)
            if ev is not None and not ev.is_set():
                # A helper has it queued or in progress: move it to the front
                # and wait for its cubin instead of compiling it again here.
                self._submit(spec, spec, "deep_gemm", _PRIO_URGENT, kind=jdg.KIND)
                t0 = time.perf_counter()
                ev.wait(float(os.environ.get("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "300")))
                dt = time.perf_counter() - t0
                self.stats.wait_n += 1
                self.stats.wait_s += dt
                self._event(f"WAIT for helper deep_gemm m={m} n={n} k={k} {dt * 1e3:.0f} ms")

        jdg.launch_observer = _observe
        self.observe_deep_gemm()
        threading.Thread(
            target=self._enumerate_dg, daemon=True, name="jit_prefetch_dg_enum"
        ).start()
        self._event(
            f"DeepGEMM prefetch: {len(provider.shapes)} FP8 Linear shape(s), "
            f"{len(moe_provider.shapes) if moe_provider else 0} MoE shape(s), {n} helper(s), "
            f"target sm_{major}{minor} x{sms} SMs"
        )
        return True

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

    def _feed_loop(self, proc, q):
        # One request in flight per helper, so work spreads across helpers.
        while True:
            _prio, tag, spec = q.get()
            key = self._tag_key.get(tag)
            if key is not None:
                self._key_started.add(key)
            self._busy.setdefault(proc.pid, threading.Event()).clear()
            if not self._write_line(proc, json.dumps({"tag": tag, "spec": spec}) + "\n"):
                return
            self._busy[proc.pid].wait()

    def _write_line(self, proc, line: str) -> bool:
        """Write one request line to a helper; the feed thread and
        ``register`` (preloads) share its stdin."""
        with self._stdin_locks.setdefault(proc.pid, threading.Lock()):
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

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
                if r.get("built"):
                    self.stats.dg_helper_built += 1
                if ok:
                    self.stats.helper_ok += 1
                else:
                    self.stats.helper_fail += 1
                    logger.warning(f"[JIT prefetch] helper failed on {label}: {err}")
            if ok and label == "deep_gemm" and not r.get("built"):
                continue  # already on disk: most enumerated M share a layout
            self._event(f"helper {'done' if ok else 'FAIL'} {label} {dt * 1e3:.0f} ms")

    def register(self, name: str, provider: Callable[[Any], List[KernelCall]]):
        """Register a variant provider once per name (warmup may run twice).

        Each Triton helper is asked to import the provider's kernel modules
        now (``provider.kernel_modules``), during warmup, so the first
        variant a batch needs does not wait for those imports.
        """
        if name in self._provider_names:
            return False
        self._provider_names.add(name)
        self._providers.append(provider)
        modules = list(getattr(provider, "kernel_modules", ()) or ())
        if modules and self.prefetch:
            line = json.dumps({"preload": modules}) + "\n"
            for proc in self._procs_by_kind.get("triton", []):
                self._write_line(proc, line)
        return True

    # -- planning (executor thread) --------------------------------------
    def bind_executor_thread(self) -> None:
        """Mark the calling thread as the executor loop (idempotent).

        Also reports the DeepGEMM kernels this process compiled since the
        previous call; the first call drops those compiled during warmup.
        """
        if self._executor_thread is None:
            self._executor_thread = threading.get_ident()
            self._event("executor thread bound; counting JIT from here")
            self._take_deep_gemm_compiles(report=False)
            self.report_fmha_sm100_builds()
            return
        self._take_deep_gemm_compiles(report=True)

    def _take_deep_gemm_compiles(self, report: bool) -> None:
        """Count DeepGEMM compiles from DeepJIT's record of disk-cache misses.

        This sees every DeepGEMM kernel any op compiled in this process, also
        on paths without a launch observer; helpers compile in other
        processes and never appear here.
        """
        take = getattr(self._deep_gemm_module(), "take_compiled_kernel_names", None)
        if take is None:
            return
        names = take()
        if not report or not names:
            return
        self.stats.dg_process_compiles += len(names)
        counts: Dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        for name, n in counts.items():
            self._event(f"DEEP_GEMM compile in process {name} x{n}")

    @staticmethod
    def _deep_gemm_module():
        try:
            from tensorrt_llm import deep_gemm
        except ImportError:
            return None
        return deep_gemm

    def plan_deep_gemm(self, num_tokens: int) -> None:
        """Queue the DeepGEMM variants a batch of ``num_tokens`` will run."""
        if self._dg is None:
            return
        from . import jit_prefetch_deep_gemm as jdg

        t0 = time.perf_counter()
        for spec in self._dg.plan_tokens(num_tokens):
            self.stats.dg_planned += 1
            self._submit(spec, spec, "deep_gemm", _PRIO_BATCH, kind=jdg.KIND)
        self.stats.plan_s += time.perf_counter() - t0

    def _start_cute_dsl_helpers(self) -> bool:
        """Start the CuTe DSL helper pool once (shared by every CuTe DSL
        provider). Returns False if prefetch is off."""
        import json as _json
        import sys

        from . import jit_prefetch_cute_dsl as jcd

        if not self.prefetch:
            return False
        if self._cd_dir is not None:
            return True
        out_dir = os.environ.get("TLLM_JIT_CUTE_DSL_DIR") or os.path.join(
            os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.cache"),
            "..",
            "cute_dsl_aot",
            jcd.gpu_arch(),
        )
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self._cd_dir = out_dir
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "UCX_", "MPI_"))
        }
        # The helper targets the arch it is given (it sets CUTE_DSL_ARCH).
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1")
        helper = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "jit_prefetch_cute_dsl_helper.py"
        )
        # Each compile is 0.2 s to seconds and the variant set is small.
        n = max(1, int(os.environ.get("TLLM_JIT_PREFETCH_CD_WORKERS", "2")))
        self._queues[jcd.KIND] = queue.PriorityQueue()
        props = _json.dumps(jcd.device_props())
        self._cd_argv = [sys.executable, "-u", helper, jcd.gpu_arch(), out_dir, props]
        self._cd_env = env
        self._spawn(jcd.KIND, self._cd_argv, env, n)
        self._event(f"CuTe DSL helpers: {n}, exporting to {out_dir}")
        return True

    def _grow_cute_dsl_helpers(self, num_variants: int) -> None:
        """Add CuTe DSL helpers for a provider with many variants (one per
        ~32, up to the per-rank helper budget), unless
        ``TLLM_JIT_PREFETCH_CD_WORKERS`` fixes the pool size."""
        from . import jit_prefetch_cute_dsl as jcd

        if "TLLM_JIT_PREFETCH_CD_WORKERS" in os.environ or self._cd_dir is None:
            return
        have = len(self._procs_by_kind.get(jcd.KIND, []))
        want = min(_default_workers(), max(have, num_variants // 32))
        if want > have:
            self._spawn(jcd.KIND, self._cd_argv, self._cd_env, want - have)
            self._event(f"CuTe DSL helpers: {have} -> {want}")

    def _cute_dsl_load(self, spec: str):
        """Executor side: the helper-exported kernel for ``spec``; None to
        compile in place.

        A variant a helper is compiling is waited for: that compile is under
        way, so it usually finishes before one started here would. A variant
        still queued is compiled in place, as without prefetch, rather than
        waited for behind the helpers' other work; the Triton hook does the
        same.
        """
        from . import jit_prefetch_cute_dsl as jcd

        ev = self._key_event.get(spec)
        if ev is not None and not ev.is_set() and spec in self._key_started:
            wait_timeout_s = float(os.environ.get("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "300"))
            t0 = time.perf_counter()
            ev.wait(wait_timeout_s)
            dt = time.perf_counter() - t0
            self.stats.wait_n += 1
            self.stats.wait_s += dt
            self._event(f"WAIT for helper cute_dsl {dt * 1e3:.0f} ms")
        fn = jcd.load(spec, self._cd_dir)
        if fn is None:
            # The compile that follows is counted by the cute.compile wrapper.
            if threading.get_ident() == self._executor_thread or (
                os.environ.get("TLLM_JIT_PREFETCH_DEBUG_KEYS") == "1"
            ):
                self._event(f"CuTe DSL variant not prefetched: {spec}")
        else:
            self.stats.cd_loaded += 1
        return fn

    def enable_cute_dsl(self, kda_provider) -> bool:
        """Start CuTe DSL helpers and install the KDA K123 and K4 loaders.

        Helpers compile from fake tensors for this rank's arch and export
        ``<dir>/<key>.o``; at the point the KDA op would call ``cute.compile``
        the loader loads that file instead, waiting for a helper compile of
        the same variant already under way rather than compiling it again
        (see ``_cute_dsl_load``).
        """
        from . import jit_prefetch_cute_dsl as jcd
        from .custom_ops import cute_dsl_kimi_k3_custom_ops as kda_ops

        if self._cd is not None or not kda_provider or not self._start_cute_dsl_helpers():
            return False
        self._cd = kda_provider

        def _loader(cache_key):
            spec = jcd.k123_spec_from_cache_key(cache_key)
            return None if spec is None else self._cute_dsl_load(spec)

        def _k4_loader(cache_key):
            spec = jcd.k4_spec_from_cache_key(cache_key)
            return None if spec is None else self._cute_dsl_load(spec)

        kda_ops.k123_loader = _loader
        kda_ops.k4_loader = _k4_loader
        for spec in kda_provider.enumerate_specs():
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(
            f"CuTe DSL prefetch: {len(kda_provider.cfgs)} KDA config(s), "
            f"{len(kda_provider.k4_specs)} K4 variant(s)"
        )
        return True

    def enable_cute_dsl_topk(self, topk_provider, next_ns=(1,)) -> bool:
        """Compile the radix top-k variants the model's TopK modules can reach
        in the CuTe DSL helpers, and have the runners load them."""
        from . import jit_prefetch_cute_dsl as jcd

        if not topk_provider or not self._start_cute_dsl_helpers():
            return False
        if not jcd.install_topk_loader(self._cute_dsl_load):
            return False
        specs = topk_provider.enumerate_specs(next_ns)
        for spec in specs:
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(f"CuTe DSL top-k prefetch: {len(specs)} variant(s)")
        return True

    def enable_csa2_publish(self, publish_provider) -> bool:
        """Compile CSA2's candidate-publication variants in the CuTe DSL
        helpers and have CSA2's glue-kernel launcher load them."""
        from . import jit_prefetch_cute_dsl as jcd

        if not publish_provider or not self._start_cute_dsl_helpers():
            return False
        if not jcd.install_csa2_loader(self._cute_dsl_load):
            return False
        specs = publish_provider.enumerate_specs()
        for spec in specs:
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(f"CSA2 candidate publication prefetch: {len(specs)} variant(s)")
        return True

    def enable_fp8_gemm(self, gemm_provider) -> bool:
        """Compile the CuTe DSL FP8 blockwise GEMM tactics the model's layers
        reach in the CuTe DSL helpers, and have the GEMM and batched-GEMM
        runners load them.

        Called right after the model is constructed (``start_model_builds``),
        before warmup: the autotuner's first pass compiles every tactic it
        benchmarks, and the helpers compile them while the weights load. A
        tactic whose export is missing when the runner needs it compiles in
        place; one a helper is compiling is waited for (``_cute_dsl_load``).
        """
        from . import jit_prefetch_cute_dsl as jcd

        if self._fp8_gemm or not gemm_provider or not self._start_cute_dsl_helpers():
            return False
        if not jcd.install_fp8_gemm_loader(self._cute_dsl_load):
            return False
        self._fp8_gemm = True
        for spec in gemm_provider.enumerate_specs():
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(
            f"CuTe DSL FP8 GEMM prefetch: {len(gemm_provider.shapes)} GEMM shape(s), "
            f"{len(gemm_provider.specs)} tactic(s)"
        )
        return True

    def enable_lowm_direct_gemm(self, lowm_provider) -> bool:
        """Compile FlashInfer's low-M BF16 direct GEMM variants the low-M
        dispatcher's AutoTuner race benchmarks in the CuTe DSL helpers, and
        have FlashInfer's compiled-kernel getter load them.

        Called once the weights are materialized on the GPU
        (``start_weight_load_builds``), before warmup: the race's first pass
        compiles every (M bucket, Linear shape, direct tactic) on the executor
        thread, ~1 s each and hundreds of them on a low-latency generation
        worker. A variant whose export is missing when the race needs it
        compiles in place, as without prefetch.
        """
        from . import jit_prefetch_cute_dsl as jcd

        if self._lowm or not lowm_provider or not self._start_cute_dsl_helpers():
            return False
        if not jcd.install_lowm_direct_loader(self._cute_dsl_load):
            return False
        self._lowm = True
        self._grow_cute_dsl_helpers(len(lowm_provider.specs))
        for spec in lowm_provider.enumerate_specs():
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(
            f"Low-M direct GEMM prefetch: {len(lowm_provider.shapes)} Linear shape(s), "
            f"{len(lowm_provider.specs)} variant(s)"
        )
        return True

    def enable_flashinfer_norm(self, norm_provider) -> bool:
        """Compile FlashInfer's CuTe DSL norm variants in the CuTe DSL helpers
        and have FlashInfer's compiled-kernel getters load them."""
        from . import jit_prefetch_cute_dsl as jcd

        if not norm_provider or not self._start_cute_dsl_helpers():
            return False
        wrapped = jcd.install_flashinfer_norm_loader(self._cute_dsl_load)
        if not wrapped:
            return False
        for spec in norm_provider.enumerate_specs():
            self._submit(spec, spec, "cute_dsl", _PRIO_BATCH, kind=jcd.KIND)
        self._event(
            f"FlashInfer norm prefetch: {len(norm_provider.shapes)} norm shape(s), "
            f"{len(norm_provider.specs)} variant(s)"
        )
        return True

    def plan_cute_dsl(self, ctx_chunk_lens) -> None:
        """Move the K123 variant this batch needs to the front of the queue."""
        if self._cd is None:
            return
        from . import jit_prefetch_cute_dsl as jcd

        for spec in self._cd.plan(ctx_chunk_lens):
            ev = self._key_event.get(spec)
            if ev is not None and not ev.is_set():
                self._submit(spec, spec, "cute_dsl", _PRIO_URGENT, kind=jcd.KIND)

    def _enumerate_dg(self) -> None:
        """Queue every M window up to max_num_tokens, at background priority."""
        from . import jit_prefetch_deep_gemm as jdg

        n = 0
        # MoE first, at batch priority: one kernel per weight shape covers
        # every batch, and the first MoE layer runs in the first forward.
        if self._dg_moe:
            for spec in self._dg_moe.enumerate_specs():
                if self._submit(spec, spec, "deep_gemm", _PRIO_BATCH, kind=jdg.KIND):
                    n += 1
        world = int(os.environ.get("TLLM_JIT_PREFETCH_DG_WORLD", "0")) or _local_world()
        for spec in self._dg.enumerate_specs(self.rank, world):
            if self._submit(spec, spec, "deep_gemm", _PRIO_ENUM, kind=jdg.KIND):
                n += 1
        self._event(f"DeepGEMM enumeration: {n} requests queued")

    def start_background(self) -> None:
        """Queue every enumerable variant on a background thread.

        Called at the end of each warmup, once the providers are registered,
        so the helpers start on the background set before serving. Each
        provider is enumerated once, except those that plan from what warmup
        recorded (``plans_from_warmup``): they are enumerated again after
        every warmup and plan only the launch classes not planned before. The
        KV-cache estimation pass and the final executor warm up on different
        cache pools, and only the final executor's launches reach serving.
        """
        if not self.prefetch or not self._providers:
            return
        self._bg_started = True
        with self._lock:
            providers = [
                p
                for p in self._providers
                if id(p) not in self._enumerated or getattr(p, "plans_from_warmup", False)
            ]
            self._enumerated.update(id(p) for p in providers)
        if providers:
            threading.Thread(
                target=self._enumerate_all,
                args=(providers,),
                daemon=True,
                name="jit_prefetch_enumerate",
            ).start()

    def plan(self, batch_ctx: Any) -> None:
        """Called once a batch is scheduled, before its inputs are prepared."""
        self.bind_executor_thread()
        if not self.prefetch or not self._providers:
            return
        if not self._bg_started:
            self.start_background()
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

    def _plan_call(
        self, call: KernelCall, priority: int = 0, keys_out: Optional[List[str]] = None
    ) -> int:
        """Plan one call; return how many variants it queued. ``priority`` 0
        = the batch being scheduled now, 1 = background enumeration; lower is
        compiled first. ``keys_out`` collects the cache key of every variant
        not compiled yet, queued now or before."""
        jit_fn, _, _ = _unwrap_jit(call.fn)
        try:
            variants, tuned = _expand(call)
        except Exception as e:  # noqa: BLE001
            logger.warning_once(
                f"[JIT prefetch] cannot expand {call.label}: {e}", key=f"jitp_expand_{call.label}"
            )
            return 0
        queued = 0
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
            if keys_out is not None:
                keys_out.append(key)
            if not self._submit(key, spec, call.label, priority):
                continue
            queued += 1
            if os.environ.get("TLLM_JIT_PREFETCH_DEBUG_KEYS") == "1":
                print(f"[JIT keys] PLANNED {jit_fn.fn.__name__} key={key}", flush=True)
            if priority == _PRIO_BATCH:
                self._event(f"submit {call.label}{'' if tuned else ' (untuned: all configs)'}")
        return queued

    def _submit(self, key: str, spec: str, label: str, priority: int, kind: str = "triton") -> bool:
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
            self._tag_key[tag] = key
            self._key_spec[key] = (spec, label, kind)
            self.stats.submitted += 1
            if priority == _PRIO_ENUM:
                self.stats.bg_submitted += 1
            elif priority == _PRIO_URGENT:
                self.stats.urgent += 1
        self._queues[kind].put((priority, tag, spec))
        return True

    def tune(self) -> None:
        """Tune ahead: run the autotune benchmark of every untuned key the
        providers' ``tune_batches`` reach, on the GPU, before serving.

        Called at the end of each warmup, on the thread that ran it, once
        the helpers are up. For each batch a provider yields there, the
        provider first lists the batch's launches (``__call__``): every
        variant not compiled yet, all candidate configs of an untuned key
        included, is queued for the helpers at batch priority, and the
        executor waits for those to land on disk. The provider then runs the
        batch for real (``run_tune_batch``) on scratch tensors, through the
        same launchers a forward uses: each ``Autotuner.run`` whose key is
        untuned benchmarks its candidate configs on the GPU, as it would
        inside the first forward that needed them, and loads the
        helper-built cubins instead of compiling them. A batch whose launches
        are all tuned is skipped before anything is allocated.

        Without prefetch there are no helpers: the batch's missing variants
        are compiled by a thread pool in this process instead
        (``_compile_in_process``), still before the benchmark. Compiles
        neither delivered run here, inside warmup, as they would at that
        first forward. A batch that fails is logged and skipped: its keys
        are then tuned at first use, as without tune ahead.
        """
        if not tune_ahead_enabled():
            return
        jobs = [p for p in self._providers if getattr(p, "tune_batches", None) is not None]
        if not jobs:
            return
        import torch

        timeout_s = float(os.environ.get("TLLM_JIT_PREFETCH_WAIT_TIMEOUT_S", "300"))
        t0 = time.perf_counter()
        before = (self.stats.ta_tunes, self.stats.ta_compiles, self.stats.ta_inproc_compiles)
        n_batches = 0
        self._tune_thread = threading.get_ident()
        try:
            for provider in jobs:
                try:
                    n_batches += self._tune_provider(provider, timeout_s)
                except Exception as e:  # noqa: BLE001 - never break warmup
                    logger.warning(
                        f"[JIT prefetch] tune ahead failed, the remaining keys are "
                        f"tuned at first use: {type(e).__name__}: {e}"
                    )
        finally:
            self._tune_thread = None
            torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        self.stats.ta_batches += n_batches
        self.stats.ta_s += dt
        self._event(
            f"tune ahead: {n_batches} batch(es), "
            f"{self.stats.ta_tunes - before[0]} autotune(s), "
            f"{self.stats.ta_compiles - before[1]} in-process compile(s), "
            f"{dt * 1e3:.0f} ms"
            + (
                ""
                if self.prefetch
                else f" ({self.stats.ta_inproc_compiles - before[2]} candidate(s) "
                "compiled by its thread pool, no prefetch)"
            )
        )

    def _tune_provider(self, provider: Any, timeout_s: float) -> int:
        """Tune ahead for one provider (see ``tune``); return the number of
        batches run on the GPU."""
        import torch

        n_batches = 0
        seen: set = set()
        for batch_ctx in provider.tune_batches():
            calls = provider(batch_ctx, seen=seen)
            if not any(_untuned(call) for call in calls):
                continue
            tw = time.perf_counter()
            if self.prefetch:
                keys: List[str] = []
                for call in calls:
                    self._plan_call(call, priority=_PRIO_BATCH, keys_out=keys)
                self._wait_for_helpers(keys, tw + timeout_s)
            else:
                self._compile_in_process(calls)
            self.stats.ta_wait_s += time.perf_counter() - tw
            try:
                provider.run_tune_batch(batch_ctx)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 - never break warmup
                logger.warning(
                    f"[JIT prefetch] tune batch {batch_ctx} failed, its keys are "
                    f"tuned at first use: {type(e).__name__}: {e}"
                )
                continue
            n_batches += 1
        return n_batches

    def _compile_in_process(self, calls: List[KernelCall]) -> None:
        """Compile every variant of ``calls`` not compiled yet, all candidate
        configs of an untuned key included, into the on-disk Triton cache,
        with a pool of threads in this process (tune ahead without prefetch).

        Each variant goes through ``triton.compile`` from Triton's own
        specialization of the launch, as a helper compiles it, so the
        benchmark that follows loads the cubins from disk. The thread count
        is ``TLLM_JIT_TUNE_AHEAD_THREADS`` (default: as many as prefetch would
        start helpers). A variant that fails to compile here is compiled by
        the launch that needs it, as without tune ahead.
        """
        from concurrent.futures import ThreadPoolExecutor

        work: Dict[str, Tuple[Any, str]] = {}
        for call in calls:
            jit_fn, _, _ = _unwrap_jit(call.fn)
            try:
                variants, _ = _expand(call)
            except Exception:  # noqa: BLE001 - compiled at launch instead
                continue
            for kw in variants:
                try:
                    found = _lookup_or_serialize(jit_fn, call.args, kw)
                except Exception:  # noqa: BLE001
                    continue
                if found is not None:
                    work.setdefault(found[0], (jit_fn, found[1]))
        if not work:
            return
        import torch

        n = max(1, int(os.environ.get(_TUNE_THREADS_ENV, str(_default_workers()))))
        # The CUDA current device is per thread: each compile thread takes this
        # rank's, so the compiled kernels land in its in-memory caches.
        with ThreadPoolExecutor(
            max_workers=n,
            thread_name_prefix="jit_tune_compile",
            initializer=torch.cuda.set_device,
            initargs=(torch.cuda.current_device(),),
        ) as pool:
            for ok in pool.map(_compile_spec, work.values()):
                self.stats.ta_inproc_compiles += 1
                if not ok:
                    self.stats.ta_inproc_failed += 1

    def _wait_for_helpers(self, keys: List[str], deadline: float) -> None:
        """Wait until the Triton helpers have compiled every variant in
        ``keys``, the deadline passes, or no Triton helper is alive. A variant
        still missing then is compiled in process by the launch that needs
        it."""
        procs = self._procs_by_kind.get("triton", [])
        for key in keys:
            ev = self._key_event.get(key)
            while ev is not None and not ev.is_set():
                left = deadline - time.perf_counter()
                if left <= 0 or not any(p.poll() is None for p in procs):
                    return
                ev.wait(min(left, 1.0))

    def _enumerate_all(self, providers: List[Any]) -> None:
        """Queue every variant ``providers`` can reach, at background priority.

        Runs on its own thread so the executor never waits for it. It plans
        with the same meta-tensor replay as ``plan``, only over the batch
        compositions a provider enumerates (``enumerate_batches``) instead of
        the one batch being scheduled. Variants the current batch needs are
        always queued ahead of these. Enumerations run one at a time.
        """
        with self._enum_lock:
            t0 = time.perf_counter()
            n_batches = queued = 0
            for provider in providers:
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
                            f"[JIT prefetch] enumeration failed: {e}",
                            key=f"jitp_enum_{id(provider)}",
                        )
                        break
                    for call in calls:
                        queued += self._plan_call(call, priority=_PRIO_ENUM)
            self._event(
                f"background enumeration: {n_batches} batch classes, "
                f"{queued} variants queued in {(time.perf_counter() - t0) * 1e3:.0f} ms"
            )

    # -- measurement hooks ------------------------------------------------
    def install_fmha_sm100_counter(self) -> None:
        """Count and time every MSA ``fmha_sm100`` kernel built in this
        process on the executor thread (``FMHA_SM100 compile on executor``
        events; ``fmha_executor_compiles`` / ``fmha_executor_compile_s`` in the
        summary). Called by the MiniMax-M3 MSA setup; idempotent."""
        try:
            from .attention.backends.sparse.minimax_m3.jit_prefetch_fmha import fmha_jit_module

            jit = fmha_jit_module()
        except Exception:  # noqa: BLE001 - no fmha_sm100, nothing to count
            return
        cls = jit.FMHAVariantManager
        if getattr(cls._compile_only, "_jitp_wrapped", False):
            return
        me = self

        def counted(orig, what):
            # what(args) -> (name, path of the library the call would build)
            def wrapper(*args, **kwargs):
                if threading.get_ident() != me._executor_thread:
                    return orig(*args, **kwargs)
                name, lib = what(args)
                if lib.exists():
                    return orig(*args, **kwargs)
                t0 = time.perf_counter()
                try:
                    return orig(*args, **kwargs)
                finally:
                    dt = time.perf_counter() - t0
                    me.stats.fmha_executor_compiles += 1
                    me.stats.fmha_executor_compile_s += dt
                    me._event(f"FMHA_SM100 compile on executor {name} {dt * 1e3:.0f} ms")

            wrapper._jitp_wrapped = True
            return wrapper

        base = jit.CACHE_BASE
        cls._compile_only = counted(
            cls._compile_only, lambda a: (f"variant_{a[1]}", base / a[1] / f"{a[1]}.so")
        )
        for fn, sub, lib in (
            ("_do_compile_plan", "plan", "fmha_sm100_plan.so"),
            ("_do_compile_reduction", "reduction", "fmha_sm100_reduction.so"),
            ("_do_compile_sparse_topk", "sparse_topk", "sparse_topk_select.so"),
        ):
            orig = getattr(jit, fn, None)
            if orig is not None:
                setattr(jit, fn, counted(orig, lambda a, s=sub, f=lib: (s, base / s / f)))

    def report_fmha_sm100_builds(self) -> None:
        builds = FmhaSm100Builds.get()
        if builds is None:
            return
        requested, finished, built, failed = builds.counts()
        self.stats.fmha_helper_built = built
        self.stats.fmha_helper_fail = failed
        self._event(
            f"fmha_sm100 helpers: {finished}/{requested} kernel(s) done "
            f"({built} built, {failed} failed) {time.perf_counter() - builds.t0:.0f} s after start"
        )

    def _install_cute_dsl_counter(self) -> None:
        """Count and time every ``cute.compile`` on the executor thread.

        Wraps ``cutlass.cute.compile`` (and its ``compile[...]`` option form)
        so the JIT stats cover CuTe DSL as well as Triton and DeepGEMM: one
        ``CUTE_DSL compile on executor <fn> <s>`` event per compile, plus
        ``cd_executor_compiles`` / ``cd_executor_compile_s`` in the summary.
        """
        try:
            import cutlass.cute as cute
        except ImportError:
            return
        orig = cute.compile
        if getattr(orig, "_jitp_wrapped", False):
            return
        me = self

        def _time(fn, call):
            if threading.get_ident() != me._executor_thread:
                return call()
            t0 = time.perf_counter()
            try:
                return call()
            finally:
                dt = time.perf_counter() - t0
                me.stats.cd_executor_compiles += 1
                me.stats.cd_executor_compile_s += dt
                name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))
                me._event(f"CUTE_DSL compile on executor {name} {dt * 1e3:.0f} ms")

        class _Compile:
            _jitp_wrapped = True

            def __call__(self, fn, *a, **kw):
                return _time(fn, lambda: orig(fn, *a, **kw))

            def __getitem__(self, opts):
                inner = orig[opts]

                def call(fn, *a, **kw):
                    return _time(fn, lambda: inner(fn, *a, **kw))

                return call

        cute.compile = _Compile()

    def _install_hooks(self):
        self._install_cute_dsl_counter()
        from triton import knobs

        stats = self.stats
        me = self

        prev_compile_listener = knobs.compilation.listener

        def compile_listener(*, src, metadata, metadata_group, times, cache_hit):
            # Fires inside triton.compile in this process, for every in-memory
            # cache miss: cache_hit=True means the cubin was loaded from the
            # on-disk cache, False means a real compile ran here. Counted only
            # on the executor loop's thread, once it has been seen, and for
            # tune ahead (JitPrefetcher.tune) on the thread running it.
            if not cache_hit and me._tune_thread == threading.get_ident():
                stats.ta_compiles += 1
                stats.ta_compile_s += times.total / 1e6
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
            if ev is not None and not ev.is_set() and key not in me._key_started:
                # Not started by any helper yet: a Triton compile here takes
                # about 0.1 s, less than a helper's first compile (its
                # imports run then), so compile here and let the queued
                # request become a disk-cache hit.
                ev = None
            if ev is not None and not ev.is_set():
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

        prev_listener = knobs.autotuning.listener

        def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
            if duration is not None and me._tune_thread == threading.get_ident():
                stats.ta_tunes += 1
                stats.ta_tune_s += duration
                me._event(
                    f"AUTOTUNE ahead {fn.fn.__name__} key={key} "
                    f"{len(configs_timings)} cfgs {duration * 1e3:.0f} ms"
                )
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
            f" bg_submitted={s.bg_submitted} urgent={s.urgent} workers={len(self._procs)}"
            f" dg_planned={s.dg_planned} dg_unplanned={s.dg_unplanned} dg_helper_built={s.dg_helper_built}"
            f" dg_executor_compiles={s.dg_executor_compiles}"
            f" dg_executor_compile_s={s.dg_executor_compile_s:.3f}"
            f" dg_process_compiles={s.dg_process_compiles}"
            f" cd_loaded={s.cd_loaded} cd_executor_compiles={s.cd_executor_compiles}"
            f" cd_executor_compile_s={s.cd_executor_compile_s:.3f}"
            f" ta_batches={s.ta_batches} ta_tunes={s.ta_tunes} ta_tune_s={s.ta_tune_s:.3f}"
            f" ta_compiles={s.ta_compiles} ta_compile_s={s.ta_compile_s:.3f}"
            f" ta_wait_s={s.ta_wait_s:.3f} ta_s={s.ta_s:.3f}"
            f" ta_inproc_compiles={s.ta_inproc_compiles} ta_inproc_failed={s.ta_inproc_failed}"
            f" fmha_helper_built={s.fmha_helper_built} fmha_helper_fail={s.fmha_helper_fail}"
            f" fmha_executor_compiles={s.fmha_executor_compiles}"
            f" fmha_executor_compile_s={s.fmha_executor_compile_s:.3f}"
        )

    def shutdown(self):
        """Print the summary for the executor that is stopping.

        The singleton outlives an executor: the KV-cache estimation pass
        builds and stops one before the serving executor starts. Helpers are
        not stopped here; each exits when this process exits and its stdin
        closes.
        """
        if self._executor_thread is not None:
            self._take_deep_gemm_compiles(report=True)
        self.report_fmha_sm100_builds()
        # print(), not only the logger: the summary is the experiment's
        # measurement and must reach the log at any log level.
        print(self.summary(), flush=True)
