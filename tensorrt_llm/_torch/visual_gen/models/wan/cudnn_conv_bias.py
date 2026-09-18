# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native cuDNN convolution with a BF16 boundary before the bias add.

The cache holds descriptors and handles, never tensor values or data pointers.
Unsupported inputs and unsupported graph construction use the original PyTorch
convolution. Execution errors propagate to the caller.
"""

import importlib
import os
import threading
import weakref
from collections import OrderedDict
from functools import lru_cache
from typing import Any

import torch

# These complete signatures have operator numerical and timing evidence on B200.
# Expanding this set requires the same full-call numerical/performance checks.
_QUALIFIED = frozenset(
    {
        ((1, 256, 6, 352, 640), (256, 256, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((1, 512, 6, 176, 320), (512, 512, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((4, 512, 352, 640), (512, 512, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((4, 1024, 176, 320), (1024, 1024, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 512, 4, 352, 640), (256, 512, 1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 512, 6, 352, 640), (256, 512, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((1, 1024, 4, 176, 320), (512, 1024, 1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 1024, 4, 88, 160), (2048, 1024, 3, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
        ((1, 256, 3, 352, 640), (256, 256, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((1, 512, 352, 640), (512, 512, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 1024, 3, 88, 160), (1024, 1024, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((1, 1024, 176, 320), (1024, 1024, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 512, 3, 352, 640), (256, 512, 3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1)),
        ((1, 1024, 88, 160), (1024, 1024, 3, 3), (1, 1), (1, 1), (1, 1)),
    }
)
_CACHE_LIMIT = 64
_WORKSPACE_LIMIT = 1 << 30
_CACHE: OrderedDict[tuple, "_Plan | None"] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_BUILDS = 0
_EXECUTIONS = 0


def _channels_last_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    strides[1] = 1
    size = shape[1]
    for axis in reversed(range(2, len(shape))):
        strides[axis] = size
        size *= shape[axis]
    strides[0] = size
    return tuple(strides)


def _eligible(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    training: bool,
) -> bool:
    # Parameters retain requires_grad=True after eval(); no_grad is sufficient.
    if torch.compiler.is_compiling() or training or torch.is_grad_enabled():
        return False
    if type(x) is not torch.Tensor or type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return False
    if type(bias) not in (torch.Tensor, torch.nn.Parameter):
        return False
    if torch.is_autocast_enabled("cuda") or not x.is_cuda:
        return False
    if x.device != weight.device or x.device != bias.device:
        return False
    if groups != 1 or x.dtype != torch.bfloat16 or weight.dtype != x.dtype or bias.dtype != x.dtype:
        return False
    signature = (tuple(x.shape), tuple(weight.shape), stride, padding, dilation)
    if signature not in _QUALIFIED:
        return False
    if tuple(x.stride()) != _channels_last_stride(tuple(x.shape)):
        return False
    if tuple(weight.stride()) != _channels_last_stride(tuple(weight.shape)):
        return False
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0] or not bias.is_contiguous():
        return False
    if x.device.index != torch.cuda.current_device():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_capability(x.device) != (10, 0):
        return False
    if "B200" not in torch.cuda.get_device_name(x.device):
        return False
    if not torch.backends.cudnn.enabled or torch.backends.cudnn.deterministic:
        return False
    if torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark:
        return False
    if not torch.backends.cudnn.allow_tf32:
        return False
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").strip().lower() not in (
        "0",
        "false",
        "off",
        "",
    ):
        return False
    return all(
        not t.is_neg() and not t.is_conj() and t.data_ptr() % 16 == 0 for t in (x, weight, bias)
    )


@lru_cache(maxsize=1)
def _runtime() -> Any | None:
    # Bind the descriptor API and backend semantics to the evaluated runtime.
    if torch.version.cuda != "13.4" or torch.backends.cudnn.version() != 92501:
        return None
    if not (torch.__version__.startswith("2.14.0a0+") and torch.__version__.endswith(".nv26.08")):
        return None
    try:
        cudnn = importlib.import_module("cudnn")
    except ModuleNotFoundError as error:
        if error.name == "cudnn":
            return None
        raise
    return cudnn if cudnn.__version__ == "1.27.0" and cudnn.backend_version() == 92501 else None


def _destroy_handle(cudnn: Any, handle: Any, device: int) -> None:
    with torch.cuda.device(device):
        cudnn.destroy_handle(handle)


class _Plan:
    def __init__(
        self,
        cudnn: Any,
        handle: Any,
        device: int,
        graph: Any,
        uids: tuple[int, ...],
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        workspace_bytes: int,
    ) -> None:
        self.graph = graph
        self.handle = handle
        self.uids = uids
        self.shape = shape
        self.strides = strides
        self.workspace_bytes = workspace_bytes
        self.lock = threading.Lock()
        self._finalizer = weakref.finalize(self, _destroy_handle, cudnn, handle, device)
        # CUDA teardown order at interpreter exit is not defined.
        self._finalizer.atexit = False


def _boundary(
    cudnn: Any, graph: Any, intermediate: Any, output: Any, *, lowered: bool = False
) -> tuple:
    if lowered:
        intermediate, output = (graph._cpp_tensors[t.get_uid()] for t in (intermediate, output))
    tensors = (intermediate, output)
    state = tuple(
        (t.get_data_type(), t.get_is_virtual(), tuple(t.get_dim()), tuple(t.get_stride()))
        for t in tensors
    )
    if (
        state[0][0] != cudnn.data_type.BFLOAT16
        or state[1][0] != cudnn.data_type.BFLOAT16
        or state[0][1] is not True
        or state[1][1] is not False
        or graph.get_node("conv_bf16_boundary").compute_data_type != cudnn.data_type.FLOAT
        or graph.get_node("bias_after_bf16_boundary").compute_data_type != cudnn.data_type.FLOAT
        or graph.context.compute_data_type != cudnn.data_type.FLOAT
        or graph.context.intermediate_data_type != cudnn.data_type.BFLOAT16
    ):
        raise RuntimeError("cuDNN convolution lost its BF16 bias boundary")
    return state


def _build(cudnn: Any, key: tuple) -> _Plan | None:
    device, x_meta, w_meta, b_meta, stride, padding, dilation = key
    x_shape, x_stride = x_meta
    w_shape, w_stride = w_meta
    b_shape, b_stride = b_meta
    shape = (
        x_shape[0],
        w_shape[0],
        *(
            (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
            for size, kernel, pad, dil, step in zip(
                x_shape[2:], w_shape[2:], padding, dilation, stride
            )
        ),
    )
    strides = _channels_last_stride(shape)
    handle = cudnn.create_handle()
    owned = True
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream(device).cuda_stream)
        graph = cudnn.pygraph(
            handle=handle,
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.BFLOAT16,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        X, W, B = (
            graph.tensor(
                dim=list(dims), stride=list(steps), data_type=cudnn.data_type.BFLOAT16, name=name
            )
            for dims, steps, name in (
                (x_shape, x_stride, "x"),
                (w_shape, w_stride, "w"),
                (b_shape, b_stride, "bias"),
            )
        )
        C = graph.conv_fprop(
            image=X,
            weight=W,
            pre_padding=padding,
            post_padding=padding,
            stride=stride,
            dilation=dilation,
            compute_data_type=cudnn.data_type.FLOAT,
            name="conv_bf16_boundary",
        )
        C.set_data_type(cudnn.data_type.BFLOAT16).set_dim(list(shape)).set_stride(list(strides))
        Y = graph.add(
            a=C, b=B, compute_data_type=cudnn.data_type.FLOAT, name="bias_after_bf16_boundary"
        )
        Y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(list(shape)).set_stride(
            list(strides)
        )
        before = _boundary(cudnn, graph, C, Y)
        graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        if graph.selected_engine is not None or graph._data_bindings:
            raise RuntimeError("cuDNN convolution must use native plans without tensor bindings")
        if _boundary(cudnn, graph, C, Y, lowered=True) != before:
            raise RuntimeError("cuDNN lowering changed the BF16 convolution boundary")
        workspace_bytes = graph.get_workspace_size()
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise RuntimeError("Invalid cuDNN convolution workspace size")
        if workspace_bytes > _WORKSPACE_LIMIT:
            return None
        plan = _Plan(
            cudnn,
            handle,
            device,
            graph,
            tuple(t.get_uid() for t in (X, W, B, Y)),
            shape,
            strides,
            workspace_bytes,
        )
        owned = False
        return plan
    except cudnn.cudnnGraphNotSupportedError:
        return None
    finally:
        if owned:
            _destroy_handle(cudnn, handle, device)


def _plan(cudnn: Any, key: tuple) -> _Plan | None:
    global _BUILDS
    # Build under the cache lock: no duplicate handle ownership or winner race.
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        plan = _build(cudnn, key)
        _BUILDS += 1
        _CACHE[key] = plan
        if len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
        return plan


def _cache_info() -> dict[str, Any]:
    """Internal diagnostics; reading this snapshot does not build or run a graph."""
    with _CACHE_LOCK:
        return {
            "builds": _BUILDS,
            "executions": _EXECUTIONS,
            "size": len(_CACHE),
            "ready": sum(plan is not None for plan in _CACHE.values()),
            "unsupported": sum(plan is None for plan in _CACHE.values()),
        }


def try_conv_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    *,
    training: bool,
) -> torch.Tensor | None:
    """Return the fused output, or None before execution when native is required."""
    global _EXECUTIONS
    if not _eligible(x, weight, bias, stride, padding, dilation, groups, training):
        return None
    cudnn = _runtime()
    if cudnn is None:
        return None
    bias_view = bias.view(1, -1, *([1] * (x.ndim - 2)))
    key = (
        x.device.index,
        (tuple(x.shape), tuple(x.stride())),
        (tuple(weight.shape), tuple(weight.stride())),
        (tuple(bias_view.shape), tuple(bias_view.stride())),
        stride,
        padding,
        dilation,
    )
    with torch.cuda.device(x.device):
        plan = _plan(cudnn, key)
        if plan is None:
            return None
        # Allocations are on the caller's current stream. The caching allocator
        # preserves their lifetime until work queued on that stream completes.
        output = torch.empty_strided(plan.shape, plan.strides, dtype=x.dtype, device=x.device)
        workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=x.device)
        with plan.lock:
            cudnn.set_stream(plan.handle, torch.cuda.current_stream(x.device).cuda_stream)
            plan.graph.execute(
                dict(zip(plan.uids, (x, weight, bias_view, output))), workspace, handle=plan.handle
            )
    with _CACHE_LOCK:
        _EXECUTIONS += 1
    return output
