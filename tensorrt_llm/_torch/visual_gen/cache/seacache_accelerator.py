# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SeaCache behind the unified :class:`CacheAccelerator` interface."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..config import SeaCacheConfig
from .base import CacheAccelerator
from .seacache import SeaCacheController

# Transformer forward arguments that add non-video tokens to the generation
# sequence; calls that carry any of them run uncached.
_UNSUPPORTED_STREAM_ARGS = ("control_latents", "audio_latents", "action_latents")


class SeaCacheAccelerator(CacheAccelerator):
    """Skips the transformer's generator stack (``_run_gen_stack``) on cacheable steps.

    The denoising step comes from the transformer's ``set_denoising_step`` hook,
    which the pipeline calls once per step of its denoising loop and clears with
    ``reset_denoising_step`` when the loop ends. Transformer calls made outside
    that loop, and calls the cache does not support (audio, action or control
    streams; single-frame latents), run the full stack. The noise level of a
    step is read from the pipeline scheduler's ``sigmas``.
    """

    def __init__(self, pipeline: Any, seacache_cfg: SeaCacheConfig):
        self._pipeline = pipeline
        self._cfg = seacache_cfg
        self._controller = SeaCacheController(
            seacache_cfg.threshold, seacache_cfg.max_consecutive_cached_steps
        )
        self._transformer: Optional[torch.nn.Module] = None
        self._hook_handles = []
        self._call_latents: Optional[torch.Tensor] = None
        # (step_index, num_steps) of the current denoising step, or None outside a loop.
        self._step: Optional[Tuple[int, int]] = None
        self._last_run_step: Optional[int] = None
        self._warned: set[str] = set()

    def _check_supported(self, transformer: torch.nn.Module) -> None:
        pipeline = self._pipeline
        name = type(pipeline).__name__
        required = ("_run_gen_stack", "set_denoising_step", "reset_denoising_step")
        if not all(callable(getattr(transformer, attr, None)) for attr in required):
            raise NotImplementedError(f"SeaCache is not supported for {name}.")
        sampling = getattr(pipeline, "sampling", None)
        if sampling is not None and sampling.is_distilled:
            raise NotImplementedError(
                f"SeaCache is not supported for distilled few-step checkpoints ({name})."
            )
        if getattr(transformer, "step_precision_controller", None) is not None:
            raise NotImplementedError(
                "SeaCache cannot be combined with a per-step precision policy "
                "(diffusion_step_policy) yet."
            )
        if not hasattr(getattr(pipeline, "scheduler", None), "sigmas"):
            raise NotImplementedError(
                f"SeaCache needs a scheduler exposing sigmas, got "
                f"{type(getattr(pipeline, 'scheduler', None)).__name__}."
            )
        if pipeline.world_size > 1:
            raise NotImplementedError("SeaCache currently supports single-GPU runs only.")
        if pipeline.pipeline_config.cuda_graph.enable:
            raise NotImplementedError("SeaCache cannot be combined with CUDA graphs.")
        if pipeline.offloader.stages():
            raise NotImplementedError("SeaCache cannot be combined with CPU offloading.")

    def wrap(self, *args: Any, **kwargs: Any) -> None:
        if self._transformer is not None:
            return
        transformer = getattr(self._pipeline, "transformer", None)
        if transformer is None:
            return
        self._check_supported(transformer)
        logger.info(
            f"SeaCache: Enabling for {type(transformer).__name__} "
            f"(threshold={self._cfg.threshold}, "
            f"max_consecutive_cached_steps={self._cfg.max_consecutive_cached_steps})"
        )
        original_run_gen_stack = transformer._run_gen_stack
        original_set_step = transformer.set_denoising_step
        original_reset_step = transformer.reset_denoising_step

        def cached_run_gen_stack(hidden_gen, freqs_gen, **kwargs):
            def run_stack(stack_input):
                return original_run_gen_stack(stack_input, freqs_gen, **kwargs)

            return self._run(hidden_gen, run_stack)

        def set_denoising_step(step_index: int, num_steps: int) -> None:
            self._step = (step_index, num_steps)
            original_set_step(step_index=step_index, num_steps=num_steps)

        def reset_denoising_step() -> None:
            self._step = None
            self._last_run_step = None
            self._controller.release()
            original_reset_step()

        self._hook_handles = [
            transformer.register_forward_pre_hook(self._capture_call, with_kwargs=True),
            transformer.register_forward_hook(self._release_call, always_call=True),
        ]
        transformer._run_gen_stack = cached_run_gen_stack
        transformer.set_denoising_step = set_denoising_step
        transformer.reset_denoising_step = reset_denoising_step
        self._transformer = transformer

    def unwrap(self) -> None:
        if self._transformer is None:
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for attr in ("_run_gen_stack", "set_denoising_step", "reset_denoising_step"):
            delattr(self._transformer, attr)
        self._transformer = None
        self._call_latents = None
        self._step = None
        self._last_run_step = None
        self._controller.release()

    def refresh(self, num_inference_steps: int) -> None:
        self._controller.reset(num_inference_steps)
        self._last_run_step = None

    def get_stats(self) -> Dict[str, Any]:
        return self._controller.get_stats() if self._transformer is not None else {}

    def is_enabled(self) -> bool:
        return self._transformer is not None

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            logger.warning(message)

    def _capture_call(self, module, args, kwargs) -> None:
        latents = kwargs.get("hidden_states", args[0] if args else None)
        unsupported = [name for name in _UNSUPPORTED_STREAM_ARGS if kwargs.get(name) is not None]
        if unsupported:
            self._warn_once(f"SeaCache: calls with {', '.join(unsupported)} run uncached.")
            latents = None
        elif latents is not None and (latents.ndim != 5 or latents.shape[2] < 2):
            self._warn_once("SeaCache: single-frame (image) generation runs uncached.")
            latents = None
        self._call_latents = latents

    def _release_call(self, module, args, output) -> None:
        self._call_latents = None

    def _run(self, hidden_gen: torch.Tensor, run_stack) -> torch.Tensor:
        latents = self._call_latents
        if latents is None or self._step is None:
            return run_stack(hidden_gen)
        step, num_steps = self._step
        if step == self._last_run_step:
            self._warn_once("SeaCache: repeated transformer calls within one step run uncached.")
            return run_stack(hidden_gen)
        sigmas = self._pipeline.scheduler.sigmas
        if num_steps != self._controller.num_inference_steps or step >= len(sigmas):
            self._warn_once("SeaCache: step outside the refreshed schedule runs uncached.")
            return run_stack(hidden_gen)
        sigma = float(sigmas[step])
        if not 0.0 <= sigma <= 1.0:
            self._warn_once(f"SeaCache: sigma {sigma} outside [0, 1]; running uncached.")
            return run_stack(hidden_gen)
        self._last_run_step = step
        return self._controller.step(step, sigma, latents, hidden_gen, run_stack)
