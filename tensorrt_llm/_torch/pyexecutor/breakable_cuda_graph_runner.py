# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import torch
from torch import nn
from torch.utils._pytree import tree_map

from tensorrt_llm.logger import logger

from ..attention.backends.interface import AttentionMetadata
from ..models.modeling_utils import timing_metric
from ..nccl_window_graph import abandon_nccl_window_graph_owner, release_nccl_window_graph_owner
from ..nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref
from .breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from .resource_manager import ResourceManager


class BreakableCUDAGraphRunnerState(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    CAPTURE = "capture"
    REPLAY = "replay"


def _copy_inputs(destination: Any, source: Any) -> bool:
    """Refresh a captured input tree only when its execution layout matches."""
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor) or (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            return False
        if destination.device.type == "cpu":
            # Host tensor values may have selected Python branches at capture.
            return torch.equal(destination, source)
        destination.copy_(source)
        return True
    if type(destination) is not type(source):
        return False
    if isinstance(destination, dict):
        return destination.keys() == source.keys() and all(
            _copy_inputs(value, source[key]) for key, value in destination.items()
        )
    if isinstance(destination, (tuple, list)):
        return len(destination) == len(source) and all(
            _copy_inputs(dst, src) for dst, src in zip(destination, source)
        )
    if isinstance(destination, (str, int, float, bool, type(None))):
        return destination == source
    # These engine-owned objects refresh their tensor state at the eager
    # execution boundaries. Other mutable host objects can hide captured
    # Python decisions and must use the eager body instead.
    return isinstance(destination, (AttentionMetadata, SpecMetadata, ResourceManager)) and (
        destination is source
    )


class BreakableCUDAGraphRunner:
    """Capture and replay prefill model bodies as breakable CUDA graphs."""

    _WARMUP_STEPS = 2

    CUDA_GRAPH_WARMUP_METRIC = "cuda_graph_warmup_seconds"
    CUDA_GRAPH_CAPTURE_METRIC = "cuda_graph_capture_seconds"

    def __init__(
        self,
        layer_model: nn.Module,
        prepare_inputs: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self.layer_model = layer_model
        self._prepare_inputs = prepare_inputs
        self._input_signature = inspect.signature(layer_model.forward) if prepare_inputs else None
        self._inputs: dict[int, dict[str, Any]] = {}
        self._graphs: dict[int, BreakableCUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._memory_pool = None
        self._capture_stream = torch.cuda.Stream()
        self._shared_output: Optional[torch.Tensor] = None
        self._state = BreakableCUDAGraphRunnerState.IDLE
        self._active_graph: Optional[BreakableCUDAGraph] = None
        self._active_num_tokens: Optional[int] = None
        self._metrics: dict[str, float] = {}

    def _prepare_body_inputs(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        assert self._input_signature is not None and self._prepare_inputs is not None
        bound = self._input_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = {}
        for name, value in bound.arguments.items():
            kind = self._input_signature.parameters[name].kind
            if kind == inspect.Parameter.VAR_KEYWORD:
                inputs.update(value)
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                if value:
                    raise TypeError("Multimodal BCG body inputs require named parameters")
            else:
                inputs[name] = value
        return self._prepare_inputs(inputs)

    @property
    def metrics(self) -> dict[str, float]:
        """Warmup and capture timings for the most recent token bucket."""
        return self._metrics

    @property
    def state(self) -> BreakableCUDAGraphRunnerState:
        return self._state

    @property
    def is_warming_up(self) -> bool:
        return self._state == BreakableCUDAGraphRunnerState.WARMUP

    @property
    def is_capturing(self) -> bool:
        return self._state == BreakableCUDAGraphRunnerState.CAPTURE

    def has_graph(self, num_tokens: int) -> bool:
        return num_tokens in self._graphs

    def warmup(
        self,
        engine_forward: Callable[[], Any],
        scope_inputs: Any = None,
        steps: int = _WARMUP_STEPS,
    ) -> None:
        """Run the complete eager engine forward under the warmup state.
        model_engine.forward will use state to determine what forward to do."""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot warm up BCG while runner is {self._state.value}")
        self._state = BreakableCUDAGraphRunnerState.WARMUP
        original_forward = self.layer_model.forward
        if self._prepare_inputs is not None:

            def warmup_forward(*args, **kwargs):
                return original_forward(**self._prepare_body_inputs(args, kwargs))

            self.layer_model.forward = warmup_forward
        try:
            for _ in range(steps):
                with discard_nccl_window_tensor_outputs(scope_inputs):
                    engine_forward()
        finally:
            if self._prepare_inputs is not None:
                self.layer_model.forward = original_forward
            self._state = BreakableCUDAGraphRunnerState.IDLE

    def capture(
        self, num_tokens: int, engine_forward: Callable[[], Any], scope_inputs: Any = None
    ) -> None:
        """Warm up eagerly, then capture one prefill token bucket.

        Record each phase in metrics, replacing the previous bucket's timings.
        Both phases run on the capture stream and synchronize before timing ends.
        """
        self._metrics.clear()
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot capture BCG while runner is {self._state.value}")
        if num_tokens in self._graphs:
            raise ValueError(f"BCG for num_tokens={num_tokens} is already captured")

        current_stream = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current_stream)
        graph = None
        created_memory_pool = False
        try:
            with torch.cuda.stream(self._capture_stream):
                with timing_metric(self.CUDA_GRAPH_WARMUP_METRIC, self._metrics):
                    self.warmup(engine_forward, scope_inputs)
                    torch.cuda.synchronize()

                with timing_metric(self.CUDA_GRAPH_CAPTURE_METRIC, self._metrics):
                    # Every segment in the first BCG bucket must receive the same
                    # explicit pool handle. Passing None lets each CUDAGraph create
                    # its own private pool, which multiplies the model workspace by
                    # the number of eager breaks.
                    if self._memory_pool is None:
                        self._memory_pool = torch.cuda.graph_pool_handle()
                        created_memory_pool = True

                    self._state = BreakableCUDAGraphRunnerState.CAPTURE
                    graph = BreakableCUDAGraph()
                    self._active_graph = graph
                    self._active_num_tokens = num_tokens
                    output = engine_forward()
                    torch.cuda.synchronize()

            current_stream.wait_stream(self._capture_stream)
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Breakable prefill capture requires a tensor body output, got {type(output)}"
                )
            assert graph is not None
            self._graphs[num_tokens] = graph
            self._outputs[num_tokens] = make_weak_ref(output)
        except Exception:
            if graph is not None:
                graph.reset()
            self._inputs.pop(num_tokens, None)
            if created_memory_pool and not self._graphs:
                # Capture failure can be rank-local; do not return this pool to reuse.
                abandon_nccl_window_graph_owner(self._memory_pool)
                self._memory_pool = None
                self._shared_output = None
            raise
        finally:
            self._active_graph = None
            self._active_num_tokens = None
            self._state = BreakableCUDAGraphRunnerState.IDLE

    @contextlib.contextmanager
    def capture_context(self) -> Iterator[None]:
        """Open the segmented CUDA graph capture for the active bucket."""
        if not self.is_capturing or self._active_graph is None:
            raise RuntimeError("BCG capture context requested outside capture")
        with (
            enable_breakable_cuda_graph(),
            BreakableCUDAGraphCapture(
                self._active_graph, pool=self._memory_pool, stream=self._capture_stream
            ),
        ):
            yield

    def capture_output(self, output: torch.Tensor) -> torch.Tensor:
        """Route all bucket outputs through the largest capture's buffer."""

        if not self.is_capturing or self._active_num_tokens is None:
            raise RuntimeError("BCG output registered outside capture")
        num_tokens = self._active_num_tokens
        if self._shared_output is None:
            self._shared_output = make_weak_ref(output)
            return self._shared_output
        if num_tokens > self._shared_output.shape[0]:
            raise ValueError(
                "BCG buckets must be captured in descending order: "
                f"{num_tokens} exceeds shared output size "
                f"{self._shared_output.shape[0]}"
            )
        self._shared_output[:num_tokens].copy_(output[:num_tokens])
        return self._shared_output[:num_tokens]

    def capture_model_body(self, outer_forward: Callable[[], Any]) -> Any:
        """Run the outer model while capturing only its decoder body.
        model_engine.forward is too broad and may pollute the CUDA stream
        before the actual model forward. We want to reuse the functions
        in forward that prepare the data and set the relevant flags."""
        if not self.is_capturing:
            raise RuntimeError("BCG body capture requested outside capture")

        original_body_forward = self.layer_model.forward
        captured_output = None

        def capture_forward(*args, **kwargs):
            nonlocal captured_output
            if self._prepare_inputs is not None:
                assert self._active_num_tokens is not None
                inputs = self._prepare_body_inputs(args, kwargs)
                static_inputs = tree_map(
                    lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
                    inputs,
                )
                self._inputs[self._active_num_tokens] = static_inputs
                args, kwargs = (), static_inputs
            with self.capture_context():
                captured_output = self.capture_output(original_body_forward(*args, **kwargs))
            return captured_output

        self.layer_model.forward = capture_forward
        try:
            outer_forward()
            if captured_output is None:
                raise RuntimeError("BCG capture did not execute the model body")
            return captured_output
        finally:
            self.layer_model.forward = original_body_forward

    def replay(self, num_tokens: int) -> torch.Tensor:
        if num_tokens not in self._graphs:
            raise KeyError(f"No BCG captured for num_tokens={num_tokens}")
        self._graphs[num_tokens].replay()
        return self._outputs[num_tokens]

    def run_eager(self, outer_forward: Callable[[], Any]) -> Any:
        """Keep padded attention outputs during multimodal body warmup or fallback."""
        if self._prepare_inputs is None:
            return outer_forward()
        original_forward = self.layer_model.forward

        def eager_forward(*args, **kwargs):
            with enable_breakable_cuda_graph():
                return original_forward(*args, **kwargs)

        self.layer_model.forward = eager_forward
        try:
            return outer_forward()
        finally:
            self.layer_model.forward = original_forward

    def execute(self, num_tokens: int, outer_forward: Callable[[], Any]) -> Any:
        """Patch the body with replay while preserving the outer forward.
        this function reuse DecoderRunner._forward_step to set flags.
        and just patch the body model forward"""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot execute BCG while runner is {self._state.value}")
        if num_tokens not in self._graphs:
            raise KeyError(f"No BCG captured for num_tokens={num_tokens}")

        original_forward = self.layer_model.forward

        def replay_forward(*args, **kwargs):
            if self._prepare_inputs is not None:
                inputs = self._prepare_body_inputs(args, kwargs)
                with enable_breakable_cuda_graph():
                    if not _copy_inputs(self._inputs[num_tokens], inputs):
                        logger.debug("BCG input layout changed; running the decoder body eagerly")
                        return original_forward(**inputs)
                    return self.replay(num_tokens)
            return self.replay(num_tokens)

        self._state = BreakableCUDAGraphRunnerState.REPLAY
        self.layer_model.forward = replay_forward
        try:
            if self._prepare_inputs is not None:
                return outer_forward()
            with enable_breakable_cuda_graph():
                return outer_forward()
        finally:
            self.layer_model.forward = original_forward
            self._state = BreakableCUDAGraphRunnerState.IDLE

    def clear(self, *, release_nccl_window_owners: bool = True) -> None:
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot clear BCG while runner is {self._state.value}")
        for graph in self._graphs.values():
            graph.reset()
        if release_nccl_window_owners and self._memory_pool is not None:
            release_nccl_window_graph_owner(self._memory_pool)
        self._graphs.clear()
        self._outputs.clear()
        self._inputs.clear()
        self._shared_output = None
        self._memory_pool = None
