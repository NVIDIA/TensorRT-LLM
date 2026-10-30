# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN frontend (graph API) convolutions with per-shape plan selection.

``F.conv2d``/``F.conv3d`` pick an engine through cuDNN's legacy heuristics (or torch's
benchmark mode over the legacy algorithm list). The graph API exposes every engine the
heuristics rank. This module builds one graph per (shape, dtype, layout), times every plan
the heuristics offer on first use, keeps the fastest that produces a finite output, and
remembers the choice on disk so later processes skip the search.

Tensors are channels-last (``torch.channels_last`` / ``channels_last_3d``) bf16 or fp16;
the bias (optional) is added in fp32 inside the graph and the output rounds once, like torch.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib
import time

import torch


@functools.lru_cache(maxsize=None)
def cudnn_frontend_available() -> bool:
    try:
        import cudnn  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _cache_file() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(base) / "tensorrt_llm" / "vae_cudnn_plans_v2.json"


@functools.lru_cache(maxsize=None)
def _handle(device: int):
    import cudnn

    torch.cuda.set_device(device)
    return cudnn.create_handle()


@functools.lru_cache(maxsize=None)
def _device_key(device: int) -> str:
    props = torch.cuda.get_device_properties(device)
    import cudnn

    return f"{props.name}|{props.multi_processor_count}|cudnn{cudnn.backend_version()}"


_plan_choices: dict[str, int] | None = None


def _load_choices() -> dict[str, int]:
    global _plan_choices
    if _plan_choices is None:
        try:
            _plan_choices = json.loads(_cache_file().read_text())
        except (OSError, ValueError):
            _plan_choices = {}
    return _plan_choices


def _save_choice(key: str, index: int) -> None:
    choices = _load_choices()
    choices[key] = index
    path = _cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(choices, indent=0, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        pass


class _Conv:
    """One compiled graph: the input/weight/bias/output tensors' shapes and strides are baked in."""

    def __init__(
        self,
        device: int,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        has_bias: bool,
        padding: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
    ) -> None:
        import cudnn

        dt = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}[
            x.dtype
        ]
        self.handle = _handle(device)
        self.graph = cudnn.pygraph(
            io_data_type=dt,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.x = self.graph.tensor(
            name="X", dim=list(x.shape), stride=list(x.stride()), data_type=dt
        )
        self.w = self.graph.tensor(
            name="W", dim=list(w.shape), stride=list(w.stride()), data_type=dt
        )
        y_out = y
        y = self.graph.conv_fprop(
            image=self.x,
            weight=self.w,
            padding=list(padding),
            stride=list(stride),
            dilation=list(dilation),
        )
        self.b = None
        if has_bias:
            cout = w.shape[0]
            bias_dim = [1, cout] + [1] * (x.dim() - 2)
            self.b = self.graph.tensor(
                name="B", dim=bias_dim, stride=self._bias_stride(bias_dim), data_type=dt
            )
            y = self.graph.bias(input=y, bias=self.b)
        y.set_output(True).set_data_type(dt).set_dim(list(y_out.shape)).set_stride(
            list(y_out.stride())
        )
        self.y = y
        self.graph.validate()
        self.graph.build_operation_graph()
        self.graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
        self.graph.check_support()
        self.plan_count = self.graph.get_execution_plan_count()
        self.plan: int | None = None
        self.workspace: torch.Tensor | None = None
        self.out_shape = None

    @staticmethod
    def _bias_stride(dim: list[int]) -> list[int]:
        stride = [1] * len(dim)
        acc = 1
        for i in reversed(range(len(dim))):
            stride[i] = acc
            acc *= dim[i]
        return stride

    def _build(self, index: int) -> torch.Tensor:
        self.graph.build_plan_at_index(index)
        size = max(int(self.graph.get_workspace_size_plan_at_index(index)), 16)
        return torch.empty(size, dtype=torch.uint8, device=self.x_device)

    def _run(self, index: int, workspace: torch.Tensor, x, w, b, y) -> None:
        import cudnn

        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        feeds = {self.x: x, self.w: w, self.y: y}
        if self.b is not None:
            feeds[self.b] = b
        self.graph.execute_plan_at_index(feeds, workspace, index, handle=self.handle)

    def select(self, key: str, x, w, b, y) -> None:
        """Pick the plan: the remembered one, else time every plan once and keep the fastest."""
        self.x_device = x.device
        remembered = _load_choices().get(key)
        if remembered is not None and remembered < self.plan_count:
            try:
                self.workspace = self._build(remembered)
                self.plan = remembered
                return
            except Exception:  # noqa: BLE001 - a stale plan index from another cuDNN build
                pass
        best = None
        for index in range(self.plan_count):
            try:
                workspace = self._build(index)
                # NaN-filled first: a plan that reports success but writes nothing, or only
                # part of the output, leaves NaNs behind; a plan that computes garbage makes
                # its own. One finiteness check covers all three.
                y.fill_(float("nan"))
                self._run(index, workspace, x, w, b, y)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001 - plans the engine cannot build/run are skipped
                continue
            if not torch.isfinite(y).all():
                continue
            # warm, then time a few launches
            for _ in range(2):
                self._run(index, workspace, x, w, b, y)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                self._run(index, workspace, x, w, b, y)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5
            if best is None or dt < best[0]:
                best = (dt, index, workspace)
        if best is None:
            raise RuntimeError("cuDNN frontend: no plan produced a finite output")
        _, self.plan, self.workspace = best
        _save_choice(key, self.plan)
        # The caller's output must hold the chosen plan's result, not the last candidate's.
        self._run(self.plan, self.workspace, x, w, b, y)


_convs: dict[tuple, _Conv] = {}


def conv_frontend(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``F.conv2d``/``F.conv3d`` (groups 1) through the cuDNN graph API with the fastest plan for
    this shape. ``x`` and ``w`` channels-last; ``bias`` [Cout] or None; ``out`` an optional
    destination of the output shape (any strides, e.g. one slice of a stacked buffer)."""
    dims = x.dim() - 2
    device = x.device.index
    out_shape = [x.shape[0], w.shape[0]] + [
        (x.shape[2 + i] + 2 * padding[i] - dilation[i] * (w.shape[2 + i] - 1) - 1) // stride[i] + 1
        for i in range(dims)
    ]
    if out is None:
        memory_format = torch.channels_last if dims == 2 else torch.channels_last_3d
        y = torch.empty(out_shape, dtype=x.dtype, device=x.device).contiguous(
            memory_format=memory_format
        )
    else:
        if list(out.shape) != out_shape or out.dtype != x.dtype:
            raise ValueError("out must have the conv's output shape and dtype")
        y = out
    key_t = (
        device,
        tuple(x.shape),
        tuple(x.stride()),
        tuple(w.shape),
        tuple(w.stride()),
        tuple(y.stride()),
        x.dtype,
        bias is not None,
        stride,
        padding,
        dilation,
    )
    entry = _convs.get(key_t)
    b = bias.reshape([1, -1] + [1] * dims).contiguous() if bias is not None else None
    if entry is None:
        entry = _Conv(device, x, w, y, bias is not None, padding, stride, dilation)
        key = hashlib.sha256(
            ("|".join(map(str, key_t[1:])) + "|" + _device_key(device)).encode()
        ).hexdigest()[:24]
        entry.select(key, x, w, b, y)
        _convs[key_t] = entry
    entry._run(entry.plan, entry.workspace, x, w, b, y)
    return y
