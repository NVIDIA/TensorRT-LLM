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
from functools import lru_cache, wraps
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


# Exact built-in scalar types cannot contain tensors. Subclasses still take the
# generic path, including dataclass subclasses.
_SCALAR_TYPES = frozenset((type(None), bool, int, float, complex, str, bytes))


@lru_cache(maxsize=256)
def _dataclass_field_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))


def _cuda_tensors(
    value: Any,
    tensors: list[torch.Tensor],
    visited: set[int] | None = None,
    include_dataclasses: bool = True,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            tensors.append(value)
        return
    if type(value) in _SCALAR_TYPES:
        return

    if visited is None:
        visited = set()
    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)

    if isinstance(value, Fp4QuantizedTensor):
        _cuda_tensors(value.fp4_tensor, tensors, visited, include_dataclasses)
        _cuda_tensors(value.scaling_factor, tensors, visited, include_dataclasses)
        _cuda_tensors(value.unquantized_hidden_states, tensors, visited, include_dataclasses)
    elif isinstance(value, MxFp8QuantizedTensor):
        _cuda_tensors(value.fp8_tensor, tensors, visited, include_dataclasses)
        _cuda_tensors(value.scaling_factor, tensors, visited, include_dataclasses)
    elif is_dataclass(value) and not isinstance(value, type):
        if not include_dataclasses:
            return
        value_type = type(value)
        for name in _dataclass_field_names(value_type):
            # Computed metadata properties can require initialization or read CUDA values.
            if isinstance(getattr(value_type, name, None), property):
                continue
            # ``init=False`` fields need not exist until the dataclass initializes them.
            _cuda_tensors(getattr(value, name, None), tensors, visited, include_dataclasses)
    elif isinstance(value, dict):
        for item in value.values():
            if type(item) not in _SCALAR_TYPES:
                _cuda_tensors(item, tensors, visited, include_dataclasses)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if type(item) not in _SCALAR_TYPES:
                _cuda_tensors(item, tensors, visited, include_dataclasses)


def _module_call_cuda_tensors(args: tuple, kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """Collect a module call's CUDA tensor inputs, excluding shared metadata.

    Equivalent to ``_cuda_tensors((args, kwargs), tensors, include_dataclasses=False)``
    with direct tensor arguments handled without recursion. Shared per-forward
    metadata belongs to the enclosing scope, including when nested inside
    activation containers.
    """
    tensors: list[torch.Tensor] = []
    visited: set[int] | None = None
    for container in (args, kwargs):
        if visited is not None:
            identity = id(container)
            if identity in visited:
                continue
            visited.add(identity)
        values = args if container is args else kwargs.values()
        for value in values:
            if type(value) is torch.Tensor:
                if value.is_cuda:
                    tensors.append(value)
            elif type(value) not in _SCALAR_TYPES:
                if visited is None:
                    visited = {id(args), id(container)}
                _cuda_tensors(value, tensors, visited, include_dataclasses=False)
    return tensors


def _output_cuda_tensors(value: Any, tensors: list[torch.Tensor]) -> None:
    """Equivalent to ``_cuda_tensors(value, tensors)``; flat tensor tuples skip recursion."""
    if type(value) is tuple or type(value) is list:
        visited: set[int] | None = None
        for item in value:
            if type(item) is torch.Tensor:
                if item.is_cuda:
                    tensors.append(item)
            elif type(item) not in _SCALAR_TYPES:
                if visited is None:
                    visited = {id(value)}
                _cuda_tensors(item, tensors, visited)
        return
    _cuda_tensors(value, tensors)


class _NCCLWindowTensorScope:
    """Deterministically release temporary NCCL-window tensor leases.

    CUDA tensor inputs are adopted when the scope begins. Window tensors
    allocated inside the scope are tracked by the allocator. Values passed to
    :meth:`escape` transfer to the surrounding scope; all other leases are
    released on the current stream when the scope exits.

    Exceptional exit also unwinds unfinished compiled scopes back to this
    scope's entry depth, quarantining their unescaped leases.
    """

    def __init__(self, inputs: Any, *, include_dataclasses: bool = True) -> None:
        self._inputs: list[torch.Tensor] = []
        _cuda_tensors(inputs, self._inputs, include_dataclasses=include_dataclasses)
        self._outputs: list[torch.Tensor] = []
        self._entry_depth = 0

    @classmethod
    def _for_module_call(cls, args: tuple, kwargs: dict[str, Any]) -> "_NCCLWindowTensorScope":
        """Scope one eager module call; its shared metadata stays with the enclosing scope."""
        scope = cls.__new__(cls)
        scope._inputs = _module_call_cuda_tensors(args, kwargs)
        scope._outputs = []
        scope._entry_depth = 0
        return scope

    def __enter__(self) -> "_NCCLWindowTensorScope":
        if self._inputs:
            self._entry_depth = torch.ops.trtllm.begin_nccl_window_tensor_scope.tracked(
                self._inputs
            )
        return self

    def escape(self, outputs: Any) -> Any:
        _output_cuda_tensors(outputs, self._outputs)
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
            # Shared metadata belongs to the enclosing forward, including when
            # nested inside activation containers. Outputs still escape recursively.
            scope = _NCCLWindowTensorScope._for_module_call(args, kwargs)
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


class _ForwardScopeHandle:
    """Restore a forward replaced by :func:`install_eager_nccl_window_tensor_scopes`."""

    def __init__(self, module: torch.nn.Module, previous: Any, wrapper: Any) -> None:
        self._module = module
        self._previous = previous
        self._wrapper = wrapper

    def remove(self) -> None:
        module, self._module = self._module, None
        if module is None or module.__dict__.get("forward") is not self._wrapper:
            return
        if self._previous is None:
            del module.forward
        else:
            module.forward = self._previous


def _install_forward_scope(module: torch.nn.Module) -> _ForwardScopeHandle:
    # A per-instance forward keeps nn.Module calls on their hook-free fast path; eager
    # decoder layers run once per layer per iteration.
    previous = module.__dict__.get("forward")
    forward = module.forward

    @wraps(forward)
    def forward_in_scope(*args: Any, **kwargs: Any) -> Any:
        scope = _NCCLWindowTensorScope._for_module_call(args, kwargs)
        with scope:
            return scope.escape(forward(*args, **kwargs))

    module.forward = forward_in_scope
    return _ForwardScopeHandle(module, previous, forward_in_scope)


def install_eager_nccl_window_tensor_scopes(
    model: torch.nn.Module,
) -> list[_ForwardScopeHandle]:
    """Install eager model scopes without exposing Python hooks to Dynamo."""
    return [
        _install_forward_scope(module)
        for module in model.modules()
        if is_nccl_window_tensor_scoped_module(type(module))
    ]
