# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import atexit
import itertools
import secrets
import sys
import weakref
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from tensorrt_llm._torch.visual_gen import DiffusionRequest, DiffusionResponse
from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionRemoteClient,
    _detect_external_launch,
    run_diffusion_worker,
)
from tensorrt_llm._torch.visual_gen.output import split_visual_gen_output, to_visual_gen_output
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.output import VisualGenOutput
from tensorrt_llm.visual_gen.params import VisualGenParams, validate_visual_gen_params

__all__ = [
    "VisualGen",
    "VisualGenParams",
    "ExtraParamSchema",
    "VisualGenResult",
]
from tensorrt_llm.llmapi.utils import set_api_status
from tensorrt_llm.logger import logger


@set_api_status("prototype")
class VisualGenResult:
    """Future-like awaitable handle for a VisualGen request.

    A single instance backs both single-prompt and batch-prompt requests:

    - Single prompt: ``await handle`` resolves to a :class:`VisualGenOutput`.
      Underlying-request failure raises :class:`RuntimeError`.
    - Batch prompt: ``await handle`` resolves to ``List[VisualGenOutput]``.
      Per-item or whole-batch failure never raises; failed items carry
      ``error != None`` (Option B semantics).

    Three wait flavors are supported:

    - ``await handle`` — preferred async style; equivalent to
      ``await handle.aresult()``.
    - ``handle.aresult(timeout=None)`` — explicit async coroutine; usable
      from any async context (e.g., FastAPI handlers).
    - ``handle.result(timeout=None)`` — blocking call for non-async callers.
    """

    def __init__(
        self,
        request_id: int,
        executor: "DiffusionRemoteClient",
        batch_size: Optional[int] = None,
    ):
        self.request_id = request_id
        self.executor = executor
        # ``None`` means single-prompt; an int means batch and is the
        # number of per-item outputs to fan out from one wire response.
        self._batch_size = batch_size
        self._resolved = None
        self._finished = False

    @property
    def done(self) -> bool:
        """True once the underlying request has completed (success or error)."""
        return self._finished

    def __await__(self):
        return self.aresult().__await__()

    async def aresult(self, timeout: Optional[float] = None):
        """Wait for the underlying request and return the resolved value.

        For single-prompt requests, returns a :class:`VisualGenOutput`. Raises
        :class:`RuntimeError` on underlying-request failure.

        For batch-prompt requests, returns ``List[VisualGenOutput]``. Never
        raises; failed items carry ``error != None``.
        """
        if self._finished:
            return self._resolved_value()

        future = asyncio.run_coroutine_threadsafe(
            self.executor.await_responses(self.request_id, timeout=timeout),
            self.executor._event_loop,
        )
        response = await asyncio.wrap_future(future)

        if response is None:
            # Timeout before any response. Tell the executor to drop any
            # late-arriving response for this id so a full PipelineOutput
            # tensor does not leak into completed_responses for the process
            # lifetime, then persist the timeout as resolved error state so
            # subsequent aresult()/result() calls replay the same outcome
            # via the ``self._finished`` fast path instead of returning None.
            abandon_future = asyncio.run_coroutine_threadsafe(
                self.executor.abandon_request_id(self.request_id),
                self.executor._event_loop,
            )
            await asyncio.wrap_future(abandon_future)
            if self._batch_size is None:
                self._resolved = VisualGenOutput(
                    request_id=self.request_id, error="Generation timed out"
                )
            else:
                self._resolved = [
                    VisualGenOutput(request_id=self.request_id, error="Generation timed out")
                    for _ in range(self._batch_size)
                ]
            self._finished = True
            return self._resolved_value()

        self._resolved = self._build_resolved(response)
        self._finished = True
        return self._resolved_value()

    def result(self, timeout: Optional[float] = None):
        """Blocking variant of :meth:`aresult` for non-async callers.

        Internally dispatches to the executor's background event loop, so it
        works even when called from inside a different event loop's thread.
        """
        # Only the inner ``aresult`` carries a timeout; it owns the
        # ``abandon_request_id`` cleanup on the timeout branch. A second
        # ``timeout`` on ``future.result`` would let the cross-thread wait
        # raise before that cleanup runs, so a late-arriving response could
        # leak into ``completed_responses``.
        future = asyncio.run_coroutine_threadsafe(
            self.aresult(timeout=timeout),
            self.executor._event_loop,
        )
        return future.result()

    def cancel(self):
        raise NotImplementedError("Cancel request (not yet implemented).")

    # ----- internals -----

    def _build_resolved(self, response: "DiffusionResponse"):
        if self._batch_size is None:
            return to_visual_gen_output(response)
        return split_visual_gen_output(response, self._batch_size)

    def _resolved_value(self):
        # For single prompts, surface engine-side failure as
        # ``RuntimeError``. Request-parameter validation is enforced
        # synchronously at :meth:`VisualGen.generate_async` entry, so
        # anything reaching this point is by definition a runtime
        # failure from ``pipeline.infer()``. For batches, return the
        # list as-is so callers iterate per-item ``error``.
        if self._batch_size is None and isinstance(self._resolved, VisualGenOutput):
            if self._resolved.error is not None:
                raise RuntimeError(f"Generation failed: {self._resolved.error}")
        return self._resolved


class VisualGen:
    """High-level API for visual generation."""

    @classmethod
    @set_api_status("prototype")
    def supported_models(cls) -> List[str]:
        """Return canonical HuggingFace model IDs of every registered pipeline.

        The returned list is a *subset* of the variants each pipeline can
        actually run. It typically contains the original official upstream
        checkpoints and well-known optimized checkpoints (e.g. NVIDIA NVFP4 /
        FP8 quantizations published on HuggingFace) that have been tested.
        Other variants — community fine-tunes and quantizations not
        enumerated here while some of them may run if no model architecture
        changes.

        IDs are returned sorted alphabetically for stable.
        """
        return sorted(hf_id for entry in PIPELINE_REGISTRY.values() for hf_id in entry.hf_ids)

    @classmethod
    @set_api_status("prototype")
    def pipeline_config(cls, model: Union[str, Path]) -> Dict[str, Any]:
        """Return the default ``pipeline_config`` knobs for ``model``.

        ``model`` may be:

        * A canonical HuggingFace model id (looked up in each entry's
          ``hf_ids`` list — the common user-facing path).
        * A local checkpoint path (resolved to ``_class_name`` via the same
          logic ``PipelineLoader`` uses).
        * A registered Diffusers ``_class_name`` (e.g. ``"WanPipeline"``)
          for callers that already know the family.

        Raises ``KeyError`` when no entry matches. The returned dict is a
        copy — mutating it does not affect the registry.
        """
        key = str(model)
        # 1. HF id match — most common user path.
        for entry in PIPELINE_REGISTRY.values():
            if key in entry.hf_ids:
                return dict(entry.defaults)
        # 2. Direct _class_name match.
        if key in PIPELINE_REGISTRY:
            return dict(PIPELINE_REGISTRY[key].defaults)
        # 3. Local path — defer to PipelineLoader's resolution logic.
        class_name = AutoPipeline._detect_from_checkpoint(key)
        return dict(PIPELINE_REGISTRY[class_name].defaults)

    @set_api_status("prototype")
    def __init__(
        self,
        model: Union[str, Path],
        args: Optional[VisualGenArgs] = None,
    ):
        self.model = str(model)
        self.args = (args or VisualGenArgs()).model_copy(update={"model": self.model})

        # In external-launch mode (torchrun/srun), ranks 1..N-1 run as pure
        # workers and never return to user code.
        ext = _detect_external_launch()
        if ext is not None:
            rank, local_rank, world_size, master_addr, master_port = ext
            n_workers = self.args.parallel_config.n_workers
            if world_size != n_workers:
                raise ValueError(
                    f"Launcher world_size ({world_size}) does not match "
                    f"n_workers ({n_workers}). "
                    "Launch exactly n_workers tasks."
                )
            if rank != 0:
                logger.info(
                    f"VisualGen: rank {rank}/{world_size}, local_rank {local_rank} — "
                    "starting as worker (external launch mode)"
                )
                run_diffusion_worker(
                    rank=rank,
                    world_size=n_workers,
                    master_addr=master_addr,
                    master_port=master_port,
                    request_queue_addr=None,  # unused: non-zero ranks receive requests via dist.broadcast_object_list
                    response_queue_addr=None,  # unused: only rank 0 sends responses over ZMQ
                    visual_gen_args=self.args,
                    req_hmac_key=None,
                    resp_hmac_key=None,
                    local_rank=local_rank,
                )
                sys.exit(0)
            logger.info(
                f"VisualGen: rank 0/{world_size} — coordinator + worker (external launch mode)"
            )

        self.executor = DiffusionRemoteClient(
            args=self.args,
        )
        self._req_counter = itertools.count()

        atexit.register(VisualGen._atexit_shutdown, weakref.ref(self))

    @property
    def extra_param_specs(self) -> Dict[str, "ExtraParamSchema"]:
        """Returns extra param specs for the loaded pipeline.

        Use this to discover types, ranges, and descriptions of
        model-specific parameters passed via ``extra_params``.
        """
        return self.executor.extra_param_specs

    @property
    def default_params(self) -> "VisualGenParams":
        """Returns a ``VisualGenParams`` with all defaults resolved for the loaded pipeline.

        Universal fields (height, width, etc.) are filled from the
        pipeline's defaults.  All declared ``extra_params`` keys are
        included with their defaults (``None`` for params without one).

        Use this to inspect what the model will use, then modify and
        pass to ``generate()``::

            params = visual_gen.default_params
            params.extra_params["stg_scale"] = 0.5
            params.height = 1024
            output = visual_gen.generate(inputs="a cat", params=params)
        """
        kwargs = dict(self.executor.default_generation_params)
        extra = {}

        for key, spec in self.executor.extra_param_specs.items():
            extra[key] = spec.default

        if extra:
            kwargs["extra_params"] = extra

        return VisualGenParams(**kwargs)

    @set_api_status("prototype")
    def generate(
        self,
        inputs: Union[str, List[str]],
        params: Optional[VisualGenParams] = None,
    ) -> Union[VisualGenOutput, List[VisualGenOutput]]:
        """Synchronous generation. Blocks until complete.

        Args:
            inputs: Text prompt or list of prompts. A list triggers batch
                inference and returns one :class:`VisualGenOutput` per prompt.
            params: Single :class:`VisualGenParams` shared by every prompt
                in the batch. A list of params is not yet supported.

        Returns:
            For a single prompt, a :class:`VisualGenOutput`. For a list of
            prompts, ``List[VisualGenOutput]`` of the same length.

        Raises:
            RuntimeError: Single-prompt path on underlying-request failure.
                The batch path never raises on per-item or whole-batch
                failure; failed items carry ``error != None``.
            NotImplementedError: ``params`` is a list (per-item parameters
                are not yet supported).
        """
        return self.generate_async(inputs=inputs, params=params).result(timeout=None)

    @set_api_status("prototype")
    def generate_async(
        self,
        inputs: Union[str, List[str]],
        params: Optional[VisualGenParams] = None,
    ) -> VisualGenResult:
        """Async generation. Returns a :class:`VisualGenResult` handle.

        ``await`` on the handle (or :meth:`VisualGenResult.aresult`) resolves
        to a :class:`VisualGenOutput` for single-prompt input or a
        ``List[VisualGenOutput]`` for batch input.

        Args:
            inputs: Text prompt or list of prompts.
            params: Single :class:`VisualGenParams` shared by every prompt.
                A list of params is not yet supported.

        Raises:
            ValueError: ``inputs`` is empty or contains non-strings.
            NotImplementedError: ``params`` is a list.
        """
        if isinstance(params, list):
            raise NotImplementedError(
                "Per-item params (List[VisualGenParams]) are not supported in this "
                "release; pass a single VisualGenParams shared across the batch."
            )

        req_id = next(self._req_counter)

        # Normalize inputs to List[str] and remember whether the caller
        # passed a single prompt so the handle resolves to the right shape.
        if isinstance(inputs, str):
            prompt = [inputs]
            batch_size: Optional[int] = None
        elif isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("Batch inputs must contain at least one item")
            if not all(isinstance(item, str) for item in inputs):
                raise ValueError("Batch inputs must contain only strings (prompt text)")
            prompt = list(inputs)
            batch_size = len(prompt)
        else:
            raise ValueError(f"Invalid inputs type: {type(inputs)}")

        # Snapshot caller-provided params so later mutations don't affect
        # the queued request (the dispatcher thread serializes it lazily).
        # When the caller passed no params, materialize a default
        # :class:`VisualGenParams` from the loaded pipeline's
        # declared defaults + extra-param specs (cached on the executor
        # from the READY signal) and skip validation — there's nothing
        # user-supplied to validate against.
        if params is not None:
            resolved_params = params.model_copy(deep=True)
            # Raising in the caller's process means ``ValueError`` reaches
            # the user as a natural Python exception; the worker only has
            # to deal with genuine runtime failures from ``pipeline.infer()``.
            validate_visual_gen_params(
                resolved_params,
                declared_defaults=self.executor.default_generation_params,
                extra_param_specs=self.executor.extra_param_specs,
            )
        else:
            resolved_params = self.default_params

        # Materialize the seed once, here at the public Python boundary,
        # so every downstream layer (executor, broadcast, pipeline) sees
        # a concrete int. Drawing on the coordinator process and
        # broadcasting the resolved value keeps multi-rank parallelism
        # (cfg_size, ulysses_size) deterministic.
        if resolved_params.seed is None:
            resolved_params.seed = secrets.randbits(63)

        request = DiffusionRequest(
            request_id=req_id,
            prompt=prompt,
            params=resolved_params,
        )

        self.executor.enqueue_requests([request])
        return VisualGenResult(req_id, self.executor, batch_size=batch_size)

    @staticmethod
    def _atexit_shutdown(self_ref):
        instance = self_ref()
        if instance is not None:
            instance.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        del exc_value, traceback
        self.shutdown()
        return False

    def __del__(self):
        self.shutdown()

    @set_api_status("prototype")
    def shutdown(self):
        """Shutdown executor and cleanup."""
        if not hasattr(self, "executor") or self.executor is None:
            return
        logger.info("VisualGen: Shutting down")
        self.executor.shutdown()
        self.executor = None
