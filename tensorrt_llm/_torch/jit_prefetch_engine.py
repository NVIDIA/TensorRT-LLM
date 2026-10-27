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
"""JIT compile ahead and tune ahead for one ``PyTorchModelEngine``: what each
startup phase starts, and in which order.

With ``TLLM_JIT_AHEAD=1`` (see ``jit_prefetch.py``), CPU-only helpers build
the JIT kernel variants warmup and serving will launch before the forward
that needs them, and the end of warmup benchmarks the autotune keys that are
still untuned. Each kernel family's provider module, next to its kernels,
decides from the model and the engine limits what to build; this module only
calls those modules, at these points of the engine's startup:

=====================  =========================  ==========================================
phase                  called from                started there
=====================  =========================  ==========================================
model constructed      ``ModelLoader.load``       the FlashInfer modules the layers build and
                                                  the sampling module; CuTe DSL NVFP4 GEMM
                                                  tactics and FlashInfer norms
warmup starts          ``PyTorchModelEngine``     recording of the Mamba decode launches
first warmup forward   ``DecoderWarmup``          the Triton providers (Mamba2 SSD and gated
                                                  RMSNorm, relu2, the native decode step)
                                                  and tune ahead's batches
each warmup forward    ``DecoderRunner``          that batch's Triton variants
end of warmup          ``PyTorchModelEngine``     the providers that plan from what warmup
                                                  recorded; tune ahead; every variant left
first serving forward  ``PyTorchModelEngine``     the JIT stats' executor thread
=====================  =========================  ==========================================

The sampler's providers are registered by ``PyExecutor`` before warmup
(``pyexecutor/sampler/jit_prefetch.py``), and warmup itself joins the
FlashInfer helpers before the forwards that load their modules
(``DecoderWarmup``). Every phase is best effort: a failure is logged, and the
kernels concerned compile in place, as without compile ahead.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
from types import SimpleNamespace
from typing import Iterator, List

from tensorrt_llm.logger import logger

from . import jit_prefetch as jp

# Modules (relative to ``tensorrt_llm._torch``) whose ``flashinfer_modules(model)``
# names the FlashInfer JIT modules a model's layers build in warmup's first
# forwards: FlashInfer's TRTLLM-GEN attention launcher ``fmha_gen``, for the
# layers that run on its FMHA library, and the Mamba2 decode's
# ``selective_state_update``.
FLASHINFER_MODULE_PROVIDERS = (
    "attention.backends.fmha.jit_prefetch",
    "modules.mamba.jit_prefetch_flashinfer",
)


@contextlib.contextmanager
def _best_effort(what: str) -> Iterator[None]:
    """Log a failure of the block instead of raising it: JIT compile ahead
    never fails startup, and the kernels concerned compile in place."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - JIT compile ahead must never fail startup
        logger.warning(f"[JIT prefetch] {what}: {type(e).__name__}: {e}")


def warmup_builds_sampling_module(model, llm_args) -> bool:
    """Whether the engine ``llm_args`` describes builds FlashInfer's sampling
    module in warmup, known once ``model`` is constructed: every decoder
    runner's warmup does (``warmup_sampling_module``), other runners'
    (encoder, pooling) do not."""
    from .pyexecutor.engine.runners import DecoderRunner, resolve_runner_type

    return issubclass(resolve_runner_type(model, llm_args), DecoderRunner)


def start_model_builds(model, rank: int = 0, llm_args=None) -> bool:
    """Start building, in CPU-only helpers, the JIT kernels ``model``'s layers
    build in warmup's first forwards, which follow from the model's geometry
    alone; called once the model is constructed, so the builds run while the
    weights load instead of inside warmup:

    * FlashInfer JIT modules, one helper each: the modules each of
      ``FLASHINFER_MODULE_PROVIDERS`` names, which warmup joins before its
      first forward (``wait_flashinfer_layer_builds``), and the sampling
      module, if the engine ``llm_args`` describes builds it in warmup
      (``warmup_builds_sampling_module``; warmup starts it otherwise);
    * the CuTe DSL kernels the layers reach
      (``jit_prefetch_cute_dsl.start_model_builds``).

    Returns False if prefetch is off or nothing was started."""
    if not jp.prefetch_enabled():
        return False
    started = False
    modules: List[str] = []
    for provider_module in FLASHINFER_MODULE_PROVIDERS:
        try:
            planner = importlib.import_module(f".{provider_module}", __package__)
            modules += planner.flashinfer_modules(model)
        except Exception as e:  # noqa: BLE001 - its modules then build in place
            logger.warning(
                f"[JIT prefetch] FlashInfer builds of {provider_module} not planned: "
                f"{type(e).__name__}: {e}"
            )
    # One helper per module, so they build side by side. The layers' modules
    # are joined before warmup's first forward, which waits for them, so they
    # build at this process's priority.
    for name in dict.fromkeys(modules):
        started = jp.start_flashinfer_build((name,), nice=0, layer=True) or started
    if llm_args is not None:
        # Warmup loads the sampling module only before CUDA graph capture.
        with _best_effort("FlashInfer sampling build not planned"):
            if warmup_builds_sampling_module(model, llm_args):
                started = jp.start_flashinfer_build(("sampling",)) or started
    from . import jit_prefetch_cute_dsl

    return jit_prefetch_cute_dsl.start_model_builds(model, rank, llm_args) or started


def _register_triton_providers(prefetcher, engine) -> None:
    """Register the Triton providers that follow from ``engine``'s model and
    limits: the Mamba2 SSD prefill (which also tunes ahead) and the mixers'
    gated RMSNorm, then the relu2 MLP activation. Once per process (later
    calls change nothing)."""
    from .modules import jit_prefetch_mlp
    from .modules.mamba import jit_prefetch as mamba_ssd

    mamba_ssd.register_providers(prefetcher, engine)
    jit_prefetch_mlp.register_providers(prefetcher, engine)


class EngineJitAhead:
    """JIT compile ahead and tune ahead for one ``PyTorchModelEngine``.

    The engine hands it to its model loader (``model_constructed``) and runs
    its warmup inside ``warmup``, which also installs the engine's
    serving-forward hook for the JIT stats
    (``PyTorchModelEngine._jit_serving_forward``). The engine's model and
    limits are read when a phase runs: some are set only once the model is
    loaded.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        # Set at the end of warmup (_after_warmup).
        self._prefetcher = None

    def model_constructed(self, model) -> None:
        """``model`` is constructed and its weights not loaded yet: start the
        builds that follow from its geometry (``start_model_builds``)."""
        engine = self._engine
        start_model_builds(model, rank=engine.mapping.rank, llm_args=engine.llm_args)

    @contextlib.contextmanager
    def warmup(self, runner, resource_manager) -> Iterator[None]:
        """Run the engine's warmup on ``runner`` (the ``with`` body).

        With compile ahead on, the decode launches warmup makes are recorded
        for the Mamba decode provider, and the runner calls
        ``_before_warmup_forwards`` once its pre-JIT prewarms are done and
        ``_plan_warmup_batch`` before each warmup forward. Once warmup is
        done, ``_after_warmup`` runs: compile ahead off, it still sets up the
        JIT stats and tune ahead alone.
        """
        engine = self._engine
        record_decode = jp.prefetch_enabled()
        if record_decode:
            # The Mamba decode provider plans from the launches warmup makes.
            from .modules.mamba import jit_prefetch_decode

            jit_prefetch_decode.start_recording(engine.model)
        early_start = record_decode and hasattr(runner, "before_warmup_forwards_callable")
        if early_start:
            runner.before_warmup_forwards_callable = functools.partial(
                self._before_warmup_forwards, resource_manager
            )
            runner.warmup_batch_callable = self._plan_warmup_batch
        try:
            yield
        finally:
            if early_start:
                runner.before_warmup_forwards_callable = None
                runner.warmup_batch_callable = None
            if record_decode:
                jit_prefetch_decode.stop_recording()
        try:
            self._after_warmup()
        except Exception as e:  # noqa: BLE001 - JIT compile ahead must never fail startup
            logger.warning(
                f"[JIT prefetch] setup failed, serving without prefetch: {type(e).__name__}: {e}"
            )

    def _before_warmup_forwards(self, resource_manager) -> None:
        """Register the Triton providers that follow from the model and the
        engine limits alone before warmup's first forward, so they plan the
        warmup forwards' batches (``_plan_warmup_batch``), and queue the
        variants warmup's first forwards launch and the batches tune ahead
        will run at the end of warmup (``JitPrefetcher.queue_early_batches``).
        The Mamba decode provider starts from the native state-update launch
        each mixer's decode step will make."""
        engine = self._engine
        with _best_effort("Triton providers not started before warmup"):
            from .modules.mamba import jit_prefetch_decode
            from .pyexecutor.resource_manager import ResourceManagerType

            prefetcher = jp.JitPrefetcher.init(rank=engine.mapping.rank)
            if prefetcher is None:
                return
            _register_triton_providers(prefetcher, engine)
            kv_cache_manager = (
                resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)
                if resource_manager is not None
                else None
            )
            jit_prefetch_decode.start_warmup_builds(prefetcher, engine, kv_cache_manager)
            prefetcher.queue_early_batches()

    @staticmethod
    def _plan_warmup_batch(batch) -> None:
        """Queue the not yet compiled variants of a warmup forward's batch for
        the helpers before it runs (``JitPrefetcher.plan_warmup_batch``). A
        batch without context requests is skipped: warmup's generation
        batches reach no provider that plans warmup batches."""
        ctx = batch.context_requests
        if not ctx:
            return
        try:
            from .modules.mamba.mamba2_metadata import Mamba2Metadata

            prefetcher = jp.JitPrefetcher.get()
            if prefetcher is None:
                return
            # The facts the runner's inputs and Mamba2Metadata.prepare derive:
            # each request's chunk length, and whether a request has a cached
            # prefix (initial states), which Mamba's warmup can force.
            any_cached = Mamba2Metadata._warmup_force_initial_states or any(
                r.context_current_position - r.py_num_compressed_tokens > 0 for r in ctx
            )
            prefetcher.plan_warmup_batch(
                SimpleNamespace(
                    ctx_chunk_lens=[r.context_chunk_size for r in ctx], any_ctx_cached=any_cached
                )
            )
        except Exception as e:  # noqa: BLE001 - JIT compile ahead must never fail warmup
            logger.warning_once(
                f"[JIT prefetch] warmup batch not planned: {type(e).__name__}: {e}",
                key="jitp_engine_warmup_batch",
            )

    def _after_warmup(self) -> None:
        """Start compile ahead, tune ahead and the JIT stats at the end of
        warmup: register the providers that plan from the loaded model and
        from what warmup recorded, wait for the helpers to be up, tune ahead,
        then queue every variant left on a background thread."""
        if not jp.jit_engine_enabled():
            return
        engine = self._engine
        prefetcher = jp.JitPrefetcher.init(rank=engine.mapping.rank)
        self._prefetcher = prefetcher
        engine._jit_serving_forward = self._serving_forward if prefetcher is not None else None
        if prefetcher is None:
            return
        if not jp.prefetch_enabled():
            # Tune ahead alone (TLLM_JIT_TUNE_AHEAD=1 without TLLM_JIT_AHEAD):
            # no helpers, no planning; only the providers that tune ahead,
            # whose candidates compile in this process.
            if jp.tune_ahead_enabled():
                _register_triton_providers(prefetcher, engine)
                prefetcher.tune()
            return
        from . import jit_prefetch_cute_dsl

        # Started once the model was constructed; the loaded model's norms
        # add only the variants not queued then.
        jit_prefetch_cute_dsl.register_providers(prefetcher, engine)
        _register_triton_providers(prefetcher, engine)
        from .modules.mamba import jit_prefetch_decode

        jit_prefetch_decode.register_providers(prefetcher, engine)
        # Helpers were started when the prefetcher was created; make sure
        # their start-up is paid here, inside warmup, not by a request.
        prefetcher.wait_ready()
        # Benchmark the untuned autotune keys the providers reach on the GPU
        # now (their candidates compiled by the helpers), before the
        # background enumeration fills the helpers' queue. A no-op unless
        # tune ahead is on (TLLM_JIT_TUNE_AHEAD, default: TLLM_JIT_AHEAD).
        prefetcher.tune()
        prefetcher.start_background()

    def _serving_forward(self) -> None:
        """Run by the engine's first forward outside warmup once warmup is
        done, on the executor thread: the JIT stats count the compiles on
        that thread from here."""
        self._engine._jit_serving_forward = None
        self._prefetcher.bind_executor_thread()
