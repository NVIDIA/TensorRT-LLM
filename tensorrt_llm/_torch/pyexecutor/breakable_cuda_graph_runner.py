# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bisect
import contextlib
import inspect
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from tensorrt_llm.logger import logger

from ..attention.backends.interface import AttentionMetadata
from ..models.modeling_utils import timing_metric
from ..nccl_window_graph import (
    abandon_nccl_window_graph_owner,
    promote_synchronized_nccl_window_releases,
    release_nccl_window_graph_owner,
)
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


def _find_input_mismatch(destination: Any, source: Any, path: str = "inputs") -> Optional[str]:
    """Return the first input whose execution layout differs from the captured tree, or None.

    Reads only, so every rank can decide before any rank copies (see _copy_inputs).
    """
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor) or (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            return path
        # Host tensor values may have selected Python branches at capture.
        if destination.device.type == "cpu" and not torch.equal(destination, source):
            return path
        return None
    if type(destination) is not type(source):
        return path
    if isinstance(destination, dict):
        if destination.keys() != source.keys():
            return path
        children = ((f"{path}[{key!r}]", value, source[key]) for key, value in destination.items())
    elif isinstance(destination, (tuple, list)):
        if len(destination) != len(source):
            return path
        children = (
            (f"{path}[{index}]", dst, src)
            for index, (dst, src) in enumerate(zip(destination, source))
        )
    elif isinstance(destination, (str, int, float, bool, type(None))):
        return None if destination == source else path
    elif isinstance(destination, (AttentionMetadata, SpecMetadata, ResourceManager)):
        # These engine-owned objects refresh their tensor state at the eager
        # execution boundaries.
        return None if destination is source else path
    else:
        # Other mutable host objects can hide captured Python decisions and
        # must use the eager body instead.
        return path
    for child_path, dst, src in children:
        mismatch = _find_input_mismatch(dst, src, child_path)
        if mismatch is not None:
            return mismatch
    return None


def _stage_inputs(inputs: Any, larger: Any) -> Any:
    """Copy the body's input tensors into storage that the bucket graph owns.

    A device tensor takes the leading rows of the same input of ``larger``, a
    larger bucket's staged inputs, when the rest of its shape matches, as bucket
    outputs share the largest output. Only one bucket graph replays at a time and
    each replay copies all of its inputs first, so buckets can share this storage.
    Both must be contiguous, so the view keeps the layout a clone would have.
    Other tensors get a clone.
    """
    values, spec = tree_flatten(inputs)
    larger_values = [None] * len(values)
    if larger is not None:
        flat_larger, larger_spec = tree_flatten(larger)
        if larger_spec == spec:
            larger_values = flat_larger
    staged = []
    for value, larger_value in zip(values, larger_values):
        if isinstance(value, torch.Tensor):
            if (
                isinstance(larger_value, torch.Tensor)
                and value.device.type != "cpu"
                and value.dim() > 0
                and larger_value.dim() == value.dim()
                and value.is_contiguous()
                and larger_value.is_contiguous()
                and larger_value.device == value.device
                and larger_value.dtype == value.dtype
                and larger_value.shape[1:] == value.shape[1:]
                and larger_value.shape[0] >= value.shape[0]
            ):
                view = larger_value[: value.shape[0]]
                view.copy_(value)
                value = view
            else:
                value = value.clone()
        staged.append(value)
    return tree_unflatten(staged, spec)


def _copy_inputs(destination: Any, source: Any) -> None:
    """Copy the device tensors of a matching input tree into the captured tree."""
    if isinstance(destination, torch.Tensor):
        if destination.device.type != "cpu":
            destination.copy_(source)
    elif isinstance(destination, dict):
        for key, value in destination.items():
            _copy_inputs(value, source[key])
    elif isinstance(destination, (tuple, list)):
        for dst, src in zip(destination, source):
            _copy_inputs(dst, src)


class BreakableCUDAGraphRunner:
    """Capture and replay prefill model bodies as breakable CUDA graphs."""

    _WARMUP_STEPS = 2

    CUDA_GRAPH_WARMUP_METRIC = "cuda_graph_warmup_seconds"
    CUDA_GRAPH_CAPTURE_METRIC = "cuda_graph_capture_seconds"

    def __init__(
        self,
        layer_model: nn.Module,
        prepare_replay: Callable[[dict[str, Any]], None] | None = None,
        prepare_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        agree_replay: Callable[[bool], bool] | None = None,
    ) -> None:
        self.layer_model = layer_model
        # Refreshes state the captured body reads but computes outside capture, from the
        # body's keyword inputs.
        self._prepare_replay = prepare_replay
        # Normalizes the body's inputs; capture then stages their tensors in storage the
        # bucket graph owns, and replay copies the current values into it.
        self._prepare_inputs = prepare_inputs
        # Under attention DP, turns this rank's staged-input match into the decision of
        # all ranks, so they all replay or all run the body eagerly.
        self._agree_replay = agree_replay
        self._input_signature = inspect.signature(layer_model.forward) if prepare_inputs else None
        self._inputs: dict[int, dict[str, Any]] = {}
        self._graphs: dict[int, BreakableCUDAGraph] = {}
        self._captured_num_tokens: list[int] = []
        self._outputs: dict[int, torch.Tensor] = {}
        self._memory_pool = None
        # Break graphs (see eager_on_graph) get a pool of their own: they are
        # captured while a bucket graph replays, when its capture-time tensors
        # hold live data in blocks that pool's allocator counts as free.
        self._break_graph_pool = None
        self._capturing_break_graphs = False
        self._capture_stream = torch.cuda.Stream()
        self._shared_output: Optional[torch.Tensor] = None
        self._state = BreakableCUDAGraphRunnerState.IDLE
        self._active_graph: Optional[BreakableCUDAGraph] = None
        self._active_num_tokens: Optional[int] = None
        self._metrics: dict[str, float] = {}
        # Executions whose staged inputs did not match the capture, so the body ran
        # eagerly. The bucket graphs alone cannot tell a replay from this fallback.
        self.num_eager_fallbacks = 0

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
                    raise TypeError("Staged BCG body inputs require named parameters")
            else:
                inputs[name] = value
        return self._prepare_inputs(inputs)

    @property
    def stages_inputs(self) -> bool:
        """Whether replay refreshes the body's inputs from the current call."""
        return self._prepare_inputs is not None

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

    @property
    def is_capturing_break_graphs(self) -> bool:
        """Whether a capture_break_graphs warmup forward is running."""
        return self._capturing_break_graphs

    @property
    def captured_num_tokens(self) -> list[int]:
        """Token buckets with a captured graph, in ascending order."""
        return self._captured_num_tokens

    def has_graphable_breaks(self, num_tokens: int) -> bool:
        """Whether the bucket's graph has breaks that can replay as CUDA graphs."""
        graph = self._graphs.get(num_tokens)
        return graph is not None and graph.has_graphable_breaks

    @property
    def num_break_graphs(self) -> int:
        """CUDA graphs captured for graphable breaks, over all buckets."""
        return sum(graph.num_break_graphs for graph in self._graphs.values())

    def warmup(
        self,
        engine_forward: Callable[[], Any],
        scope_inputs: Any = None,
        steps: int = _WARMUP_STEPS,
    ) -> None:
        """Run the complete eager engine forward under the warmup state.
        model_engine.forward will use state to determine what forward to do.

        The BCG flag covers the whole forward, as in capture and replay, so code
        outside the captured body takes the same op path in all three."""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot warm up BCG while runner is {self._state.value}")
        self._state = BreakableCUDAGraphRunnerState.WARMUP
        original_forward = self.layer_model.forward
        if self._prepare_inputs is not None:
            # Warm up the body on the inputs that capture and replay give it.
            self.layer_model.forward = lambda *args, **kwargs: original_forward(
                **self._prepare_body_inputs(args, kwargs)
            )
        try:
            with enable_breakable_cuda_graph():
                for _ in range(steps):
                    with discard_nccl_window_tensor_outputs(scope_inputs):
                        engine_forward()
        finally:
            if self._prepare_inputs is not None:
                self.layer_model.forward = original_forward
            self._state = BreakableCUDAGraphRunnerState.IDLE

    def capture(
        self,
        num_tokens: int,
        engine_forward: Callable[[], Any],
        scope_inputs: Any = None,
        can_capture: Callable[[], bool] | None = None,
    ) -> bool:
        """Warm up eagerly, then capture one prefill token bucket.

        Record each phase in metrics, replacing the previous bucket's timings.
        Both phases run on the capture stream and synchronize before timing ends.
        """
        self._metrics.clear()
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot capture BCG while runner is {self._state.value}")
        if num_tokens in self._graphs:
            raise ValueError(f"BCG for num_tokens={num_tokens} is already captured")

        promote_synchronized_nccl_window_releases()
        current_stream = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current_stream)
        graph = None
        created_memory_pool = False
        try:
            with torch.cuda.stream(self._capture_stream):
                with timing_metric(self.CUDA_GRAPH_WARMUP_METRIC, self._metrics):
                    self.warmup(engine_forward, scope_inputs)
                    torch.cuda.synchronize()
                if can_capture is not None and not can_capture():
                    return False

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
                    with enable_breakable_cuda_graph():
                        output = engine_forward()
                    torch.cuda.synchronize()

            current_stream.wait_stream(self._capture_stream)
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Breakable prefill capture requires a tensor body output, got {type(output)}"
                )
            assert graph is not None
            self._graphs[num_tokens] = graph
            bisect.insort(self._captured_num_tokens, num_tokens)
            self._outputs[num_tokens] = make_weak_ref(output)
            return True
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
            # Hold the largest bucket's output strongly. A weak reference would
            # return its block to the shared pool, where a later bucket's output
            # could land at an offset inside it and the copy below would then
            # overlap its own source.
            self._shared_output = output
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
                inputs = self._prepare_body_inputs(args, kwargs)
                # Capture runs the largest bucket first (see capture_output).
                largest = self._inputs[max(self._inputs)] if self._inputs else None
                static_inputs = _stage_inputs(inputs, largest)
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

    def capture_break_graphs(
        self, engine_forward: Callable[[], Any], scope_inputs: Any = None
    ) -> None:
        """Run a warmup engine forward whose bucket graph replay captures a CUDA
        graph for every graphable break key it meets without one (see
        eager_on_graph). Other replays only replay break graphs."""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot capture break graphs while runner is {self._state.value}")
        if self._break_graph_pool is None:
            self._break_graph_pool = torch.cuda.graph_pool_handle()
        current_stream = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current_stream)
        self._capturing_break_graphs = True
        try:
            with (
                torch.cuda.stream(self._capture_stream),
                discard_nccl_window_tensor_outputs(scope_inputs),
            ):
                engine_forward()
        finally:
            self._capturing_break_graphs = False
        current_stream.wait_stream(self._capture_stream)

    def replay(self, num_tokens: int) -> torch.Tensor:
        if num_tokens not in self._graphs:
            raise KeyError(f"No BCG captured for num_tokens={num_tokens}")
        self._graphs[num_tokens].replay(
            self._break_graph_pool if self._capturing_break_graphs else None
        )
        return self._outputs[num_tokens]

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
                static_inputs = self._inputs[num_tokens]
                mismatch = _find_input_mismatch(static_inputs, inputs)
                replayable = mismatch is None
                if self._agree_replay is not None:
                    replayable = self._agree_replay(replayable)
                if not replayable:
                    self.num_eager_fallbacks += 1
                    reason = (
                        f"{mismatch} differs from the capture"
                        if mismatch is not None
                        else "another attention DP rank's inputs differ from the capture"
                    )
                    logger.info_once(
                        f"Breakable prefill CUDA graph: the {num_tokens}-token body runs "
                        f"eagerly because {reason} (logged once per token bucket)",
                        key=f"breakable_prefill_eager_fallback_{num_tokens}",
                    )
                    return original_forward(**inputs)
                _copy_inputs(static_inputs, inputs)
                kwargs = static_inputs
            del args
            if self._prepare_replay is not None:
                # After the engine's in-place input corrections, as an eager body sees them.
                self._prepare_replay(kwargs)
            return self.replay(num_tokens)

        self._state = BreakableCUDAGraphRunnerState.REPLAY
        self.layer_model.forward = replay_forward
        try:
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
        self._captured_num_tokens.clear()
        self._outputs.clear()
        self._inputs.clear()
        self._shared_output = None
        self._memory_pool = None
        self._break_graph_pool = None
