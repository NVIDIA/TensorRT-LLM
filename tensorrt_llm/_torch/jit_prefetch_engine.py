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
model constructed      ``ModelLoader.load``       the FlashInfer modules the layers build, the
                                                  sampling module and the FlashInfer MXFP8
                                                  pass's modules; CuTe DSL GEMM, norm and KDA
                                                  tactics; MiniMax-M3's MSA ``fmha_sm100``
weights on the GPU     ``ModelLoader.load``       DeepGEMM FP8 Linear and MoE; FlashInfer's
                                                  low-M GEMM
warmup starts          ``PyTorchModelEngine``     recording of the Mamba and MSA decode
                                                  launches; MSA ``fmha_sm100`` if not started
first warmup forward   ``DecoderWarmup``          DeepGEMM on the loaded model; the Triton
                                                  providers and tune ahead's batches; CuTe
                                                  DSL radix top-k and CSA2; MSA decode
each warmup forward    ``DecoderRunner``          that batch's Triton variants
end of warmup          ``PyTorchModelEngine``     the providers that plan from what warmup
                                                  recorded; tune ahead; every variant left
first serving forward  ``PyTorchModelEngine``     the JIT stats' executor thread
=====================  =========================  ==========================================

Warmup itself joins the FlashInfer helpers before the forwards that load their
modules (``DecoderWarmup``). Every phase is best effort: a failure is logged,
and the kernels concerned compile in place, as without compile ahead.
"""

from __future__ import annotations

import contextlib
import importlib
from types import SimpleNamespace
from typing import Iterator, List

from tensorrt_llm.logger import logger

from . import jit_prefetch as jp

# Modules listing the FlashInfer JIT modules a model's layers build in
# warmup's first forwards (``flashinfer_modules(model)``): FlashInfer's
# TRTLLM-GEN attention launcher ``fmha_gen``, for the layers of any model that
# run on its FMHA library, GLM-5.3-Flash's MLA decode and MiniMax-M3; the
# Mamba2 decode's ``selective_state_update``; the TRTLLM-Gen fused MoE
# launcher of MoE layers on FlashInfer's op backend; FlashInfer's RoPE.
FLASHINFER_MODULE_PROVIDERS = (
    "attention.backends.fmha.jit_prefetch",
    "attention.backends.sparse.glm_kpool.jit_prefetch",
    "attention.backends.sparse.minimax_m3.jit_prefetch_fmha",
    "modules.mamba.jit_prefetch_flashinfer",
    "moe.fused_moe.jit_prefetch_flashinfer",
    "attention.jit_prefetch_flashinfer_rope",
)


@contextlib.contextmanager
def _best_effort(what: str) -> Iterator[None]:
    """Log a failure of the block instead of raising it: JIT compile ahead
    never fails startup, and the kernels concerned compile in place."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - prefetch must never fail startup
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

    * the FlashInfer JIT modules the layers build, one helper each
      (``FLASHINFER_MODULE_PROVIDERS``), which warmup joins before its first
      forward (``wait_flashinfer_layer_builds``), and the sampling module, if
      the engine ``llm_args`` describes builds it in warmup
      (``warmup_builds_sampling_module``; warmup starts it otherwise);
    * the FlashInfer modules of the autotuner warmup's FlashInfer MXFP8 pass,
      in a helper of their own, if the engine ``llm_args`` describes will run
      that pass (``jit_prefetch_mxfp8``; warmup starts them otherwise);
    * the CuTe DSL GEMM tactics, KDA prefill kernel and FlashInfer norms
      (``jit_prefetch_cute_dsl.start_model_builds``);
    * MiniMax-M3's MSA ``fmha_sm100`` kernels, whose plan also needs the KV
      cache's page size and dtypes, taken from ``llm_args``
      (``jit_prefetch_fmha.start_model_builds``). Without it they start at the
      beginning of warmup instead.

    The DeepGEMM and low-M GEMM kernels start a little later, once the weights
    are on the GPU (``start_weight_load_builds``).

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
    mxfp8_pass = False
    if llm_args is not None:
        with _best_effort("FlashInfer MXFP8 builds not planned"):
            from .modules import jit_prefetch_mxfp8

            mxfp8_pass = jit_prefetch_mxfp8.tunes_flashinfer_mxfp8_at_load(model, llm_args)
            started = jit_prefetch_mxfp8.start_mxfp8_tune_builds(mxfp8_pass) or started
    from . import jit_prefetch_cute_dsl

    started = jit_prefetch_cute_dsl.start_model_builds(model, rank, llm_args, mxfp8_pass) or started
    if llm_args is not None and jp.FmhaSm100Builds.get() is None:
        with _best_effort("fmha_sm100 builds not started"):
            from .attention.backends.sparse.minimax_m3 import jit_prefetch_fmha

            started = jit_prefetch_fmha.start_model_builds(model, llm_args) or started
    return started


def start_weight_load_builds(model, rank: int, deep_gemm_limits) -> bool:
    """Start, in CPU-only helpers, the GEMM kernels ``model``'s first warmup
    forwards and the autotuner warmup build, once its weights are
    materialized on the GPU and while their values load from the checkpoint:
    the DeepGEMM FP8 Linear and MoE kernels
    (``jit_prefetch_deep_gemm.start_weight_load_builds``) and FlashInfer's
    low-M BF16 direct GEMM variants
    (``jit_prefetch_cute_dsl.start_weight_load_builds``).

    ``deep_gemm_limits`` are the engine limits the providers plan from
    (``max_num_tokens``, ``priority_ms``, ``first_ms``). Returns False if
    prefetch is off or nothing was started."""
    if not jp.prefetch_enabled():
        return False
    from . import jit_prefetch_cute_dsl, jit_prefetch_deep_gemm

    started = jit_prefetch_deep_gemm.start_weight_load_builds(model, rank, **deep_gemm_limits)
    max_num_tokens = deep_gemm_limits["max_num_tokens"]
    return jit_prefetch_cute_dsl.start_weight_load_builds(model, rank, max_num_tokens) or started


def _uses_minimax_m3_sparse(llm_args) -> bool:
    sparse_cfg = getattr(llm_args, "sparse_attention_config", None)
    return getattr(sparse_cfg, "algorithm", None) == "minimax_m3"


def _uses_minimax_m3_msa(llm_args) -> bool:
    sparse_cfg = getattr(llm_args, "sparse_attention_config", None)
    return (
        _uses_minimax_m3_sparse(llm_args) and getattr(sparse_cfg, "implementation", None) == "msa"
    )


class EngineJitAhead:
    """JIT compile ahead and tune ahead for one ``PyTorchModelEngine``.

    The engine hands it to its model loader (``model_constructed``,
    ``weights_allocated``) and runs its warmup inside ``warmup``, which also
    installs the engine's serving-forward hook for the JIT stats
    (``PyTorchModelEngine._jit_serving_forward``). The engine's model and
    limits are read when a phase runs: some are set only once the model is
    loaded.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        # Set at the end of warmup (_after_warmup).
        self._prefetcher = None
        self._bind_pending = False
        self._report_compiles = False

    # -- model loading ----------------------------------------------------

    def model_constructed(self, model) -> None:
        """``model`` is constructed and its weights not loaded yet: start the
        builds that follow from its geometry (``start_model_builds``)."""
        engine = self._engine
        start_model_builds(model, rank=engine.mapping.rank, llm_args=engine.llm_args)

    def weights_allocated(self, model) -> None:
        """``model``'s weights are materialized on the GPU and their values
        about to load: start the DeepGEMM and low-M GEMM builds, from the
        engine's arguments (``start_weight_load_builds``)."""
        from . import jit_prefetch_deep_gemm

        limits = jit_prefetch_deep_gemm.load_time_limits(self._engine)
        if limits is not None:
            start_weight_load_builds(model, self._engine.mapping.rank, limits)

    # -- warmup -----------------------------------------------------------

    @contextlib.contextmanager
    def warmup(self, runner, resource_manager) -> Iterator[None]:
        """Run the engine's warmup on ``runner`` (the ``with`` body).

        With compile ahead on, the decode launches warmup makes are recorded
        for the Mamba and MSA decode providers, MiniMax-M3's MSA ``fmha_sm100``
        builds start if the model loader did not start them, and the runner
        calls ``_before_warmup_forwards`` once its pre-JIT prewarms are done
        and ``_plan_warmup_batch`` before each warmup forward. Once warmup is
        done, ``_after_warmup`` runs: compile ahead off, it still sets up the
        JIT stats and tune ahead alone.
        """
        engine = self._engine
        record_decode = jp.prefetch_enabled()
        record_msa = record_decode and _uses_minimax_m3_sparse(engine.llm_args)
        if record_decode:
            # The Mamba and MSA decode providers plan from the launches
            # warmup makes.
            from .modules.mamba import jit_prefetch_decode

            jit_prefetch_decode.start_recording(engine.model)
        if record_msa:
            from .attention.backends.sparse.minimax_m3 import jit_prefetch_msa

            jit_prefetch_msa.start_recording()
            if _uses_minimax_m3_msa(engine.llm_args):
                with _best_effort("fmha_sm100 builds not started"):
                    from .attention.backends.sparse.minimax_m3 import jit_prefetch_fmha

                    jit_prefetch_fmha.start_warmup_builds(
                        engine.model,
                        resource_manager.get_resource_manager(engine.kv_cache_manager_key),
                    )
        early_start = record_decode and hasattr(runner, "before_warmup_forwards_callable")
        if early_start:
            runner.before_warmup_forwards_callable = self._before_warmup_forwards
            runner.warmup_batch_callable = self._plan_warmup_batch
        try:
            yield
        finally:
            if early_start:
                runner.before_warmup_forwards_callable = None
                runner.warmup_batch_callable = None
                from . import jit_prefetch_cute_dsl

                jit_prefetch_cute_dsl.stop_topk_builds()
            if record_decode:
                jit_prefetch_decode.stop_recording()
            if record_msa:
                jit_prefetch_msa.stop_recording()
        try:
            self._after_warmup(resource_manager)
        except Exception as e:  # noqa: BLE001 - prefetch must never fail startup
            logger.warning(
                f"[JIT prefetch] setup failed, serving without prefetch: {type(e).__name__}: {e}"
            )

    def _before_warmup_forwards(self) -> None:
        """Start what compile ahead builds before warmup's first forward; the
        runner calls this once its pre-JIT prewarms are done.

        The DeepGEMM helpers start here rather than earlier: started during
        the prewarms, their start-up would compete with those in-process nvcc
        builds for the CPU. Then the Triton providers that follow from the
        model and the engine limits alone are registered, so they plan each
        warmup forward's batch (``_plan_warmup_batch``), and the batches tune
        ahead runs at the end of warmup are queued
        (``JitPrefetcher.queue_early_batches``). The radix top-k and CSA2
        candidate-publication variants start here too, and MiniMax-M3's MSA
        decode provider plans each decode launch class as warmup records it.
        """
        engine = self._engine
        with _best_effort("DeepGEMM helpers not started early"):
            from . import jit_prefetch_deep_gemm

            prefetcher = jp.JitPrefetcher.init(rank=engine.mapping.rank)
            if prefetcher is not None:
                jit_prefetch_deep_gemm.start_warmup_builds(prefetcher, engine)
        with _best_effort("Triton providers not started before warmup"):
            prefetcher = jp.JitPrefetcher.get()
            if prefetcher is None:
                return
            from .modules.mamba import jit_prefetch as mamba_ssd

            mamba_ssd.register_providers(prefetcher, engine)
            prefetcher.queue_early_batches()
            from . import jit_prefetch_cute_dsl

            jit_prefetch_cute_dsl.start_topk_builds(prefetcher, engine)
            if _uses_minimax_m3_sparse(engine.llm_args):
                from .attention.backends.sparse.minimax_m3 import jit_prefetch_msa

                jit_prefetch_msa.start_warmup_builds(prefetcher, engine)

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
            gen_tokens = sum(
                1 + len(getattr(r, "py_draft_tokens", None) or [])
                for r in batch.generation_requests
            )
            prefetcher.plan_warmup_batch(
                SimpleNamespace(
                    ctx_chunk_lens=[r.context_chunk_size for r in ctx],
                    any_ctx_cached=any_cached,
                    gen_tokens=gen_tokens,
                )
            )
        except Exception as e:  # noqa: BLE001 - prefetch must never fail warmup
            logger.warning_once(
                f"[JIT prefetch] warmup batch not planned: {type(e).__name__}: {e}",
                key="jitp_engine_warmup_batch",
            )

    def _after_warmup(self, resource_manager) -> None:
        """Start compile ahead, tune ahead and the JIT stats at the end of
        warmup: register the providers that plan from the loaded model and
        from what warmup recorded, wait for the helpers to be up, tune ahead,
        then queue every variant left on a background thread."""
        engine = self._engine
        prefetcher = jp.JitPrefetcher.init(rank=engine.mapping.rank)
        self._prefetcher = prefetcher
        self._bind_pending = prefetcher is not None
        self._report_compiles = prefetcher is not None and prefetcher.stats_on
        engine._jit_serving_forward = self._serving_forward if prefetcher is not None else None
        if prefetcher is None:
            return
        from .._utils import get_sm_version

        if get_sm_version() in (100, 103):
            prefetcher.observe_deep_gemm()
            if _uses_minimax_m3_msa(engine.llm_args):
                prefetcher.install_fmha_sm100_counter()
        from .modules.mamba import jit_prefetch as mamba_ssd

        if not jp.prefetch_enabled():
            # Tune ahead alone (TLLM_JIT_TUNE_AHEAD=1 without TLLM_JIT_AHEAD):
            # no helpers, no planning; only the providers that tune ahead,
            # whose candidates compile in this process.
            if jp.tune_ahead_enabled():
                mamba_ssd.register_providers(prefetcher, engine)
                prefetcher.tune()
            return
        from . import jit_prefetch_deep_gemm

        jit_prefetch_deep_gemm.register_providers(prefetcher, engine)
        from . import jit_prefetch_cute_dsl

        jit_prefetch_cute_dsl.register_providers(prefetcher, engine)
        from .modules.mamba import jit_prefetch_linear_attn

        kv_cache_manager = (
            resource_manager.get_resource_manager(engine.kv_cache_manager_key)
            if resource_manager is not None
            else None
        )
        jit_prefetch_linear_attn.register_providers(prefetcher, engine, kv_cache_manager)
        from .attention.backends.sparse.dsa import jit_prefetch as dsa

        dsa.register_providers(prefetcher, engine)
        from .attention.backends.sparse.glm_kpool import jit_prefetch as glm_kpool

        glm_kpool.register_providers(prefetcher, engine)
        if _uses_minimax_m3_sparse(engine.llm_args):
            from .attention.backends.sparse.minimax_m3 import jit_prefetch as minimax_m3

            minimax_m3.register_providers(prefetcher, engine)
        mamba_ssd.register_providers(prefetcher, engine)
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

    # -- serving ----------------------------------------------------------

    def _serving_forward(self) -> None:
        """Run by each forward of the engine outside warmup once warmup is
        done. The first binds the JIT stats' executor thread: the compiles
        on that thread count from here. The later ones report the DeepGEMM
        compiles of paths no launch hook observes (the DSA indexer, MegaMoE),
        while the JIT stats are on."""
        if self._bind_pending:
            self._bind_pending = False
            self._prefetcher.bind_executor_thread()
            if not self._report_compiles:
                self._engine._jit_serving_forward = None
        elif self._report_compiles:
            self._prefetcher.report_process_compiles()
