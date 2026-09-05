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

import sys
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from types import TracebackType
from typing import Any

import torch

from .utils import Fp4QuantizedTensor, MxFp8QuantizedTensor


def _cuda_tensors(
    value: Any,
    tensors: list[torch.Tensor],
    visited: set[int] | None = None,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            tensors.append(value)
        return

    if visited is None:
        visited = set()
    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)

    if isinstance(value, Fp4QuantizedTensor):
        _cuda_tensors(value.fp4_tensor, tensors, visited)
        _cuda_tensors(value.scaling_factor, tensors, visited)
        _cuda_tensors(value.unquantized_hidden_states, tensors, visited)
    elif isinstance(value, MxFp8QuantizedTensor):
        _cuda_tensors(value.fp8_tensor, tensors, visited)
        _cuda_tensors(value.scaling_factor, tensors, visited)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            # ``init=False`` fields need not exist until the dataclass initializes them.
            _cuda_tensors(getattr(value, field.name, None), tensors, visited)
    elif isinstance(value, dict):
        for item in value.values():
            _cuda_tensors(item, tensors, visited)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _cuda_tensors(item, tensors, visited)


class _NCCLWindowTensorScope:
    """Deterministically release temporary NCCL-window tensor leases.

    CUDA tensor inputs are adopted when the scope begins. Window tensors
    allocated inside the scope are tracked by the allocator. Values passed to
    :meth:`escape` transfer to the surrounding scope; all other leases are
    released on the current stream when the scope exits.
    """

    def __init__(self, inputs: Any):
        self._inputs: list[torch.Tensor] = []
        _cuda_tensors(inputs, self._inputs)
        self._outputs: list[torch.Tensor] = []

    def __enter__(self) -> "_NCCLWindowTensorScope":
        if self._inputs:
            torch.ops.trtllm.begin_nccl_window_tensor_scope(self._inputs)
        return self

    def escape(self, outputs: Any) -> Any:
        _cuda_tensors(outputs, self._outputs)
        return outputs

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._inputs:
            return

        failed = exc_type is not None
        torch.ops.trtllm.end_nccl_window_tensor_scope(self._inputs, self._outputs, failed)


def nccl_window_tensor_scope(inputs: Any) -> _NCCLWindowTensorScope:
    return _NCCLWindowTensorScope(inputs)


def install_eager_nccl_window_tensor_scopes(
    model: torch.nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install decoder-layer scopes without exposing Python hooks to Dynamo."""
    from .modules.decoder_layer import DecoderLayer

    handles = []
    for layer in model.modules():
        if not isinstance(layer, DecoderLayer):
            continue

        active_scopes: ContextVar[tuple[_NCCLWindowTensorScope, ...]] = ContextVar(
            f"nccl_window_scopes_{id(layer)}", default=()
        )

        def begin_scope(_module, args, kwargs, *, active_scopes=active_scopes):
            scope = nccl_window_tensor_scope((args, kwargs))
            scope.__enter__()
            active_scopes.set((*active_scopes.get(), scope))

        def end_scope(
            _module,
            _args,
            _kwargs,
            output,
            *,
            active_scopes=active_scopes,
        ):
            scopes = active_scopes.get()
            if not scopes:
                return output
            scope = scopes[-1]
            active_scopes.set(scopes[:-1])
            exception = sys.exc_info()
            if exception[0] is not None:
                scope.__exit__(*exception)
                return output
            try:
                return scope.escape(output)
            finally:
                scope.__exit__(*sys.exc_info())

        handles.append(layer.register_forward_pre_hook(begin_scope, with_kwargs=True))
        handles.append(layer.register_forward_hook(end_scope, with_kwargs=True, always_call=True))
    return handles
