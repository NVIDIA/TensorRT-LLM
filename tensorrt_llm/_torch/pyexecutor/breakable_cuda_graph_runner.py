# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import torch
from torch import nn

from ..utils import make_weak_ref
from .breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)


class BreakableCUDAGraphRunnerState(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    CAPTURE = "capture"
    REPLAY = "replay"


class BreakableCUDAGraphRunner:
    """Capture and replay prefill model bodies as breakable CUDA graphs."""

    _WARMUP_STEPS = 2

    def __init__(self, layer_model: nn.Module, logits_processor: nn.Module) -> None:
        self.layer_model = layer_model
        self.logits_processor = logits_processor
        self._graphs: dict[int, BreakableCUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._memory_pool = None
        self._capture_stream = torch.cuda.Stream()
        self._shared_output: Optional[torch.Tensor] = None
        self._state = BreakableCUDAGraphRunnerState.IDLE
        self._active_graph: Optional[BreakableCUDAGraph] = None
        self._active_num_tokens: Optional[int] = None

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

    def warmup(self, engine_forward: Callable[[], Any], steps: int = _WARMUP_STEPS) -> None:
        """Run the complete eager engine forward under the warmup state.
            model_engine.forward will use state to determine what forward to do."""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot warm up BCG while runner is {self._state.value}")
        self._state = BreakableCUDAGraphRunnerState.WARMUP
        try:
            for _ in range(steps):
                engine_forward()
        finally:
            self._state = BreakableCUDAGraphRunnerState.IDLE

    def capture(self, num_tokens: int, engine_forward: Callable[[], Any]) -> None:
        """Warm up eagerly, then capture one prefill token bucket."""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot capture BCG while runner is {self._state.value}")
        if num_tokens in self._graphs:
            raise ValueError(f"BCG for num_tokens={num_tokens} is already captured")

        current_stream = torch.cuda.current_stream()
        self._capture_stream.wait_stream(current_stream)
        graph = None
        try:
            with torch.cuda.stream(self._capture_stream):
                self.warmup(engine_forward)

                self._state = BreakableCUDAGraphRunnerState.CAPTURE
                graph = BreakableCUDAGraph()
                self._active_graph = graph
                self._active_num_tokens = num_tokens
                output = engine_forward()

            current_stream.wait_stream(self._capture_stream)
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Breakable prefill capture requires a tensor body output, got {type(output)}"
                )
            assert graph is not None
            self._graphs[num_tokens] = graph
            self._outputs[num_tokens] = make_weak_ref(output)
            if self._memory_pool is None:
                self._memory_pool = graph.pool()
        except Exception:
            if graph is not None:
                graph.reset()
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
        """Route all bucket outputs through the largest capture's buffer. """

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
        original_logits_forward = self.logits_processor.forward
        captured_output = None

        def capture_forward(*args, **kwargs):
            nonlocal captured_output
            with self.capture_context():
                captured_output = self.capture_output(
                    original_body_forward(*args, **kwargs))
            return captured_output

        def passthrough_forward(hidden_states, *args, **kwargs):
            del args, kwargs
            return hidden_states

        self.layer_model.forward = capture_forward
        self.logits_processor.forward = passthrough_forward
        try:
            outer_forward()
            if captured_output is None:
                raise RuntimeError("BCG capture did not execute the model body")
            return captured_output
        finally:
            self.logits_processor.forward = original_logits_forward
            self.layer_model.forward = original_body_forward

    def replay(self, num_tokens: int) -> torch.Tensor:
        if num_tokens not in self._graphs:
            raise KeyError(f"No BCG captured for num_tokens={num_tokens}")
        self._graphs[num_tokens].replay()
        return self._outputs[num_tokens]

    def execute(self, num_tokens: int, outer_forward: Callable[[], Any]) -> Any:
        """Patch the body with replay while preserving the outer forward.
        this function reuse model_engine._forward_step to set flags.
        and just patch the body model forward"""
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot execute BCG while runner is {self._state.value}")
        if num_tokens not in self._graphs:
            raise KeyError(f"No BCG captured for num_tokens={num_tokens}")

        original_forward = self.layer_model.forward

        def replay_forward(*args, **kwargs):
            del args, kwargs
            return self.replay(num_tokens)

        self._state = BreakableCUDAGraphRunnerState.REPLAY
        self.layer_model.forward = replay_forward
        try:
            with enable_breakable_cuda_graph():
                return outer_forward()
        finally:
            self.layer_model.forward = original_forward
            self._state = BreakableCUDAGraphRunnerState.IDLE

    def clear(self) -> None:
        if self._state != BreakableCUDAGraphRunnerState.IDLE:
            raise RuntimeError(f"Cannot clear BCG while runner is {self._state.value}")
        for graph in self._graphs.values():
            graph.reset()
        self._graphs.clear()
        self._outputs.clear()
        self._shared_output = None
        self._memory_pool = None
