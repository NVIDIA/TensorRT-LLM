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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import lru_cache
from types import TracebackType
from typing import Any, Iterable, Iterator, TypeVar

import torch

from .utils import Fp4QuantizedTensor, MxFp8QuantizedTensor

_SCOPED_MODULE_MARKER = "_trtllm_nccl_window_tensor_scoped"
_ModuleType = TypeVar("_ModuleType", bound=type[torch.nn.Module])


def nccl_window_tensor_scoped_module(module_type: _ModuleType) -> _ModuleType:
    """Mark a module type for eager and compiler-inserted tensor scopes.

    Python hooks disappear after Dynamo tracing, so the compiler backend uses
    this marker to insert opaque begin/end operations into the captured graph.
    """
    setattr(module_type, _SCOPED_MODULE_MARKER, True)
    return module_type


def is_nccl_window_tensor_scoped_module(module_type: type[torch.nn.Module]) -> bool:
    return getattr(module_type, _SCOPED_MODULE_MARKER, False)


# Exact built-in scalar types cannot contain tensor fields. Subclasses still
# use the generic traversal, including dataclass subclasses.
_SCALAR_TYPES = frozenset((type(None), bool, int, float, complex, str, bytes))


@lru_cache(maxsize=256)
def _dataclass_field_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))


def _cuda_tensors(
    value: Any,
    tensors: list[torch.Tensor],
    visited: set[int] | None = None,
) -> None:
    if type(value) in _SCALAR_TYPES:
        return

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
        for name in _dataclass_field_names(type(value)):
            # Computed metadata properties can require initialization or read CUDA values.
            if isinstance(getattr(type(value), name, None), property):
                continue
            # ``init=False`` fields need not exist until the dataclass initializes them.
            _cuda_tensors(getattr(value, name, None), tensors, visited)
    elif isinstance(value, dict):
        for item in value.values():
            if type(item) not in _SCALAR_TYPES:
                _cuda_tensors(item, tensors, visited)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if type(item) not in _SCALAR_TYPES:
                _cuda_tensors(item, tensors, visited)


# Argument types a scoped module call may receive window tensors through.
_MODULE_CALL_TENSOR_ARGUMENT_TYPES = (
    torch.Tensor,
    Fp4QuantizedTensor,
    MxFp8QuantizedTensor,
    list,
    tuple,
    dict,
)


def _module_call_cuda_tensors(args: tuple, kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """Collect CUDA tensors passed directly to a scoped module call.

    Activations flow between scoped modules as direct arguments: tensors,
    quantized-tensor wrappers, and plain containers of them. Per-forward
    metadata objects (attention, speculative-decoding, Mamba metadata, ...) are
    shared by every scoped module in the forward pass, so one module must not
    adopt and release leases they hold; their enclosing scope owns them.

    Skipping those objects also bounds the per-call cost by the number of
    arguments instead of the batch size. The eager hooks run once per scoped
    module, i.e. once per decoder layer per iteration, so walking the
    per-request lists inside attention metadata dominated eager iterations.
    """
    tensors: list[torch.Tensor] = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, _MODULE_CALL_TENSOR_ARGUMENT_TYPES):
            _cuda_tensors(value, tensors)
    return tensors


class _NCCLWindowTensorScope:
    """Deterministically release temporary NCCL-window tensor leases.

    CUDA tensor inputs are adopted when the scope begins. Window tensors
    allocated inside the scope are tracked by the allocator. Values passed to
    :meth:`escape` transfer to the surrounding scope; all other leases are
    released on the current stream when the scope exits.

    Exceptional exit also unwinds unfinished compiled scopes back to this
    scope's entry depth, quarantining their unescaped leases.
    """

    def __init__(self, inputs: Any = None, *, input_tensors: list[torch.Tensor] | None = None):
        if input_tensors is None:
            input_tensors = []
            _cuda_tensors(inputs, input_tensors)
        self._inputs: list[torch.Tensor] = input_tensors
        self._outputs: list[torch.Tensor] = []
        self._entry_depth = 0

    def __enter__(self) -> "_NCCLWindowTensorScope":
        if self._inputs:
            self._entry_depth = torch.ops.trtllm.begin_nccl_window_tensor_scope.tracked(
                self._inputs
            )
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
        torch.ops.trtllm.end_nccl_window_tensor_scope.tracked(
            self._inputs, self._outputs, failed, self._entry_depth
        )


def nccl_window_tensor_scope(inputs: Any) -> _NCCLWindowTensorScope:
    return _NCCLWindowTensorScope(inputs)


@contextmanager
def discard_nccl_window_tensor_outputs(inputs: Any) -> Iterator[None]:
    """Release window tensors produced while preserving borrowed inputs."""
    with nccl_window_tensor_scope(inputs) as scope:
        scope.escape(inputs)
        yield


def install_module_nccl_window_tensor_scopes(
    modules: Iterable[torch.nn.Module],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install deterministic lease scopes around eager module calls."""
    handles = []
    for layer in modules:
        active_scopes: ContextVar[tuple[_NCCLWindowTensorScope, ...]] = ContextVar(
            f"nccl_window_scopes_{id(layer)}", default=()
        )

        def begin_scope(_module, args, kwargs, *, active_scopes=active_scopes):
            scope = _NCCLWindowTensorScope(input_tensors=_module_call_cuda_tensors(args, kwargs))
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


def install_eager_nccl_window_tensor_scopes(
    model: torch.nn.Module,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install eager model scopes without exposing Python hooks to Dynamo."""
    modules = (
        module for module in model.modules() if is_nccl_window_tensor_scoped_module(type(module))
    )
    return install_module_nccl_window_tensor_scopes(modules)
