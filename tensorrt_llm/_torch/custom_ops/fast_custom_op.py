# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead replacement for ``@torch.library.custom_op``.

``@torch.library.custom_op`` is ergonomic (auto schema inference from Python
type hints, a ``register_fake`` method on the returned op) but its wrapper
imposes a ~6-7us per-call dispatcher tax (schema re-validation, Python-level
``DispatchKeySet`` traversal, auto-functionalization bookkeeping, etc.).

``fast_custom_op`` preserves the ergonomics while bypassing that tax by
registering the op directly through the low-level
``torch.library.Library.define + impl`` API — same path as built-in ATen ops.

Usage is almost identical to ``@torch.library.custom_op``::

    from tensorrt_llm._torch.custom_ops.fast_custom_op import fast_custom_op

    @fast_custom_op("trtllm::nvfp4_gemm", mutates_args=())
    def nvfp4_gemm(x: torch.Tensor, ...) -> torch.Tensor:
        ...

    @nvfp4_gemm.register_fake
    def _(x: torch.Tensor, ...) -> torch.Tensor:
        ...

Caveats (inherited from using the low-level API):

* No autograd support (register a separate autograd kernel if the op is
  differentiable — but in practice this is for stateless inference kernels).
* ``mutates_args`` must be a concrete tuple; ``"unknown"`` auto-functionalization
  is not supported.
* ``device_types`` defaults to ``"CUDA"``. Pass a different string or a tuple
  of strings to register on other backends.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple, Union

import torch
from torch.library import Library, infer_schema, register_fake

_LIBS: dict[tuple[str, str], Library] = {}


def _get_library(namespace: str, kind: str = "FRAGMENT") -> Library:
    key = (namespace, kind)
    lib = _LIBS.get(key)
    if lib is None:
        lib = Library(namespace, kind)
        _LIBS[key] = lib
    return lib


def fast_custom_op(
    qualname: str,
    *,
    mutates_args: Union[Iterable[str], str] = (),
    device_types: Union[str, Tuple[str, ...]] = "CUDA",
) -> Callable[[Callable], "FastCustomOp"]:
    """Register a Python function as a fast custom torch op.

    Parameters mirror ``torch.library.custom_op``:
      qualname: ``"<namespace>::<op_name>"`` identifier.
      mutates_args: names of arguments that are mutated in-place; empty tuple
        means the op is pure.
      device_types: backend(s) to register the impl on (default ``"CUDA"``).
    """
    if "::" not in qualname:
        raise ValueError(f"qualname must be '<ns>::<name>', got {qualname!r}")
    namespace, op_name = qualname.split("::", 1)

    if isinstance(mutates_args, str) and mutates_args != "unknown":
        raise TypeError("mutates_args must be an iterable of names or 'unknown'")
    mutates_args_tuple = mutates_args if isinstance(mutates_args, str) else tuple(mutates_args)

    dev_types = (device_types,) if isinstance(device_types, str) else tuple(device_types)

    def decorator(fn: Callable) -> "FastCustomOp":
        schema = infer_schema(fn, op_name=op_name, mutates_args=mutates_args_tuple)
        lib = _get_library(namespace)
        lib.define(schema)
        for dt in dev_types:
            lib.impl(op_name, fn, dt)
        return FastCustomOp(qualname=qualname, namespace=namespace, op_name=op_name, python_fn=fn)

    return decorator


class FastCustomOp:
    """Handle returned by :func:`fast_custom_op`.

    Behaves like ``@torch.library.custom_op``'s return value: callable and
    exposes ``register_fake``. The call path goes through the C++ dispatcher
    (``torch.ops.<ns>.<name>``), bypassing the Python wrapper layer of
    ``@custom_op``.
    """

    __slots__ = ("qualname", "namespace", "op_name", "_python_fn", "_op")

    def __init__(self, qualname: str, namespace: str, op_name: str, python_fn: Callable):
        self.qualname = qualname
        self.namespace = namespace
        self.op_name = op_name
        self._python_fn = python_fn
        self._op = getattr(getattr(torch.ops, namespace), op_name)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def register_fake(self, fake_fn: Callable) -> Callable:
        register_fake(self.qualname, fake_fn)
        return fake_fn

    @property
    def python_impl(self) -> Callable:
        """The original un-wrapped Python function (for tests/introspection)."""
        return self._python_fn
