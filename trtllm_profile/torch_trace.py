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
"""PyTorch dispatch tracing using HIP/CUDA events, never NVIDIA-only profiler APIs."""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from .session import _COMPONENT, Profiler


class DeviceSpan:
    def __init__(
        self,
        device: int,
        stream: torch.cuda.Stream,
        previous: torch.cuda.Event | None,
        origin: torch.cuda.Event,
        origin_ns: int,
    ) -> None:
        self.device = device
        self.stream = stream
        self.previous = previous
        self.origin = origin
        self.origin_ns = origin_ns
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record(stream)

    def resolve(self) -> tuple[float, float, float, int, int]:
        return (
            self.start.elapsed_time(self.end) * 1000,
            max(0.0, self.previous.elapsed_time(self.start) * 1000) if self.previous else 0.0,
            self.origin_ns / 1000 + self.origin.elapsed_time(self.start) * 1000,
            self.device,
            self.stream.cuda_stream,
        )


class EventRecorder:
    def __init__(self) -> None:
        self.devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self._last: dict[tuple[int, int], torch.cuda.Event] = {}
        self._origins: dict[tuple[int, int], tuple[torch.cuda.Event, int]] = {}
        self._lock = threading.Lock()

    def begin(self, device: int | None) -> DeviceSpan | None:
        if device is None:
            return None
        with torch.cuda.device(device), self._lock:
            stream = torch.cuda.current_stream(device)
            key = (device, stream.cuda_stream)
            if key not in self._origins:
                origin = torch.cuda.Event(enable_timing=True)
                timestamp = time.perf_counter_ns()
                origin.record(stream)
                self._origins[key] = (origin, timestamp)
            origin, timestamp = self._origins[key]
            return DeviceSpan(device, stream, self._last.get(key), origin, timestamp)

    def end(self, span: DeviceSpan) -> None:
        with torch.cuda.device(span.device), self._lock:
            span.end.record(span.stream)
            self._last[(span.device, span.stream.cuda_stream)] = span.end

    def synchronize(self) -> None:
        # Wait once at reporting/calibration boundaries, never once per operation.
        for event in self._last.values():
            event.synchronize()

    def clear(self) -> None:
        self._last.clear()
        self._origins.clear()

    def metadata(self) -> list[dict]:
        metadata = []
        for device in self.devices:
            entry = {
                "logical_index": device,
                "backend": "HIP" if torch.version.hip else "CUDA",
                "torch_version": str(torch.__version__),
                "hip_version": torch.version.hip,
            }
            try:
                properties = torch.cuda.get_device_properties(device)
                entry.update(
                    name=properties.name, architecture=getattr(properties, "gcnArchName", None)
                )
                free, total = torch.cuda.mem_get_info(device)
                entry.update(total_bytes=total, free_bytes_at_report=free)
                for field, getter in (
                    ("process_allocated_bytes", torch.cuda.memory_allocated),
                    ("process_reserved_bytes", torch.cuda.memory_reserved),
                    ("process_lifetime_peak_allocated_bytes", torch.cuda.max_memory_allocated),
                    ("process_lifetime_peak_reserved_bytes", torch.cuda.max_memory_reserved),
                ):
                    entry[field] = getter(device)
            except RuntimeError as error:
                entry["telemetry_error"] = str(error)
            metadata.append(entry)
        return metadata


def classify_operation(name: str) -> str:
    """Classify unannotated operations; model hooks override these heuristics."""
    name = name.lower()
    groups = (
        ("transfer", ("_to_copy", "copy_", "pin_memory")),
        ("attention", ("attention", "flash_attn")),
        ("recurrent", ("scan", "rnn", "lstm", "gru", "mamba")),
        ("norms", ("norm",)),
        ("embed", ("embedding",)),
        ("ffn-activate", ("silu", "gelu", "relu", "gated_activation")),
        ("head+sample", ("multinomial", "topk", "argmax", "sort", "softmax")),
        ("attention-mix", ("rotary", "rope", "cat", "split", "chunk")),
        ("bias", ("addmm",)),
        ("projections", ("matmul", "mm.", "bmm", "linear")),
        ("residual", ("add.",)),
    )
    for component, fragments in groups:
        if any(fragment in name for fragment in fragments):
            return component
    return "other"


def _source_location() -> tuple[str, int]:
    frame = sys._getframe(1)
    profiler_root = str(Path(__file__).parent)
    torch_root = str(Path(torch.__file__).parent)
    while frame is not None:
        path = frame.f_code.co_filename
        if not path.startswith((profiler_root, torch_root)):
            return path, frame.f_lineno
        frame = frame.f_back
    return "<unknown>", 0


def _operation_device(args: tuple, kwargs: dict) -> int | None:
    target = kwargs.get("device")
    if target is not None:
        target = torch.device(target)
        if target.type == "cuda":
            return target.index if target.index is not None else torch.cuda.current_device()
    for value in tree_leaves((args, kwargs)):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            return value.device.index
    return None


class TensorTrace(TorchDispatchMode):
    def __init__(self, profiler: Profiler) -> None:
        super().__init__()
        self.profiler = profiler

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs: dict | None = None):
        kwargs = {} if kwargs is None else kwargs
        if self.profiler.pid != os.getpid() or not self.profiler.reserve():
            return func(*args, **kwargs)
        guessed = classify_operation(str(func))
        name = guessed if guessed == "transfer" else (_COMPONENT.get() or guessed)
        source = _source_location()
        if str(func).startswith("trtllm_rdna4."):
            kernel_module = sys.modules.get("tensorrt_llm.rocm.kernels")
            if kernel_module is not None:
                source = (str(Path(kernel_module.__file__).parent / "csrc" / "rdna4Ops.hip"), 0)
        started = self.profiler.begin(_operation_device(args, kwargs))
        success = False
        try:
            result = func(*args, **kwargs)
            success = True
            return result
        finally:
            self.profiler.end(started, name, source, failed=not success)


def _module_component(name: str, module: torch.nn.Module) -> str | None:
    kind = type(module).__name__.lower()
    if "lm_head" in name:
        return "head+sample"
    if "norm" in kind:
        return "norms"
    if isinstance(module, torch.nn.Linear):
        return "projections"
    if isinstance(module, torch.nn.Embedding):
        return "embed"
    if "attention" in kind or "attn" in kind:
        return "attention"
    if any(fragment in kind for fragment in ("silu", "gelu", "relu")):
        return "ffn-activate"
    if any(fragment in kind for fragment in ("rnn", "lstm", "gru", "mamba", "recurrent")):
        return "recurrent"
    if "rotary" in kind:
        return "attention-mix"
    return None


@contextmanager
def instrument_model(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily attach thread-local, exception-safe component attribution hooks."""
    handles = []

    def attach(module: torch.nn.Module, label: str) -> None:
        tokens = ContextVar(f"profile_module_{id(module)}", default=())

        def before(_module, _args) -> None:
            tokens.set((*tokens.get(), _COMPONENT.set(label)))

        def after(_module, _args, _output) -> None:
            stack = tokens.get()
            if stack:
                _COMPONENT.reset(stack[-1])
                tokens.set(stack[:-1])

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    for name, module in model.named_modules():
        label = _module_component(name, module)
        if label is not None:
            attach(module, label)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
