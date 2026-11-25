# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Low-overhead replacement for ``@torch.library.custom_op``.

Calling an op made by ``@torch.library.custom_op`` costs a few microseconds of
host time on top of the kernel: the Python ADInplaceOrView kernel bumps the
version counter of every mutated argument, outputs are checked for aliasing
inputs, and the body runs below autograd. ``fast_custom_op`` registers the same
function through the low-level ``torch.library.Library.define`` and ``impl``
API instead, the path built-in ATen ops take, and keeps the parts of the
``custom_op`` interface the tree uses: schema inference from type hints,
``register_fake`` and ``device_types``.

Usage is almost identical to ``@torch.library.custom_op``::

    from tensorrt_llm._torch.custom_ops.fast_custom_op import fast_custom_op

    @fast_custom_op("trtllm::nvfp4_gemm", mutates_args=())
    def nvfp4_gemm(x: torch.Tensor, ...) -> torch.Tensor:
        ...

    @nvfp4_gemm.register_fake
    def _(x: torch.Tensor, ...) -> torch.Tensor:
        ...

Differences from ``@torch.library.custom_op``:

* There is no autograd kernel, so the body runs with grad mode as the caller
  set it. Pass ``disable_grad=True`` when the body needs grad mode off, for
  example because it calls an ``out=`` overload.
* The version counters of mutated arguments are not bumped.
* Outputs are not checked for aliasing inputs. An op that returns an alias of
  an input without declaring it silently corrupts that input instead of
  raising. Set ``TLLM_VALIDATE_CUSTOM_OPS=1`` before the ops are imported to
  register every ``fast_custom_op`` through ``@torch.library.custom_op``
  instead, which restores the check.
* ``device_types`` defaults to ``"CUDA"``. Pass ``None`` to register one
  device-agnostic kernel, as ``custom_op`` does when ``device_types`` is
  omitted; such an op must register a fake, because without one FakeTensor
  tracing runs the real body on fake tensors.
* ``mutates_args="unknown"`` is not supported.

Port an op only when it is called outside CUDA graphs on a default
configuration often enough for the per-call saving to matter, it has a fake,
and every tensor it writes is either declared in ``mutates_args`` or freshly
allocated.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterable

import torch
from torch.library import Library, infer_schema, register_fake

_LIBS: dict[tuple[str, str], Library] = {}

# Every op registered through the fast path, keyed by qualname.
_FAST_OPS: dict[str, FastCustomOp] = {}

# Dispatch keys that ``custom_op`` spells as device types.
_DEVICE_TYPE_FOR_DISPATCH_KEY = {"CUDA": "cuda", "CPU": "cpu", "Meta": "meta"}


def _validate_custom_ops() -> bool:
    """Whether ``fast_custom_op`` should fall back to ``custom_op``.

    The fallback reintroduces the per-call overhead, so it is meant for tests
    and for bisecting a suspected miscompare, never for production.
    """
    return os.getenv("TLLM_VALIDATE_CUSTOM_OPS", "0") == "1"


def _dispatch_key(device: str) -> str:
    """Map a device type ("cuda") to its dispatch key ("CUDA").

    ``Library.impl`` takes dispatch keys while ``custom_op`` takes device
    types, and both spellings are used in the tree. Dispatch keys pass through.
    """
    try:
        return torch._C._dispatch_key_for_device(device)
    except (AttributeError, RuntimeError):
        return device


def _run_without_grad(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # no_grad is only entered when needed, which keeps it off the
        # inference-mode path.
        if torch.is_grad_enabled():
            with torch.no_grad():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


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
    mutates_args: Iterable[str] | str = (),
    device_types: str | tuple[str, ...] | None = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
    disable_grad: bool = False,
) -> Callable[[Callable], FastCustomOp | torch.library.CustomOpDef]:
    """Register a Python function as a fast custom torch op.

    Parameters mirror ``torch.library.custom_op``:
      qualname: ``"<namespace>::<op_name>"`` identifier.
      mutates_args: names of arguments that are mutated in-place; empty tuple
        means the op is pure.
      device_types: device types ("cuda") or dispatch keys ("CUDA") to
        register the kernel for; the default is ``"CUDA"``. ``None`` registers
        one device-agnostic kernel, matching ``custom_op`` with
        ``device_types`` omitted, so porting such an op keeps its devices.
      tags: ``torch.Tag`` values to attach to the op's schema in addition to
        ``pt2_compliant_tag``, which ``custom_op`` always adds.
      disable_grad: run the body with grad mode off when the caller has it on,
        as ``custom_op`` does for every op.

    With ``TLLM_VALIDATE_CUSTOM_OPS=1`` set when the decorator runs, this
    returns a ``torch.library.custom_op`` instead.
    """
    if "::" not in qualname:
        raise ValueError(f"qualname must be '<ns>::<name>', got {qualname!r}")
    namespace, op_name = qualname.split("::", 1)

    if isinstance(mutates_args, str) and mutates_args != "unknown":
        raise TypeError("mutates_args must be an iterable of names or 'unknown'")
    mutates_args_tuple = mutates_args if isinstance(mutates_args, str) else tuple(mutates_args)

    if device_types is None:
        dispatch_keys: tuple[str, ...] = ("CompositeExplicitAutograd",)
    else:
        names = (device_types,) if isinstance(device_types, str) else tuple(device_types)
        dispatch_keys = tuple(_dispatch_key(name) for name in names)
    all_tags = (torch.Tag.pt2_compliant_tag,) + tuple(
        tag for tag in tags if tag != torch.Tag.pt2_compliant_tag
    )

    def decorator(fn: Callable) -> FastCustomOp | torch.library.CustomOpDef:
        if _validate_custom_ops():
            return torch.library.custom_op(
                qualname,
                fn,
                mutates_args=mutates_args_tuple,
                device_types=None
                if device_types is None
                else tuple(_DEVICE_TYPE_FOR_DISPATCH_KEY.get(key, key) for key in dispatch_keys),
                tags=all_tags,
            )

        schema = infer_schema(fn, op_name=op_name, mutates_args=mutates_args_tuple)
        lib = _get_library(namespace)
        lib.define(schema, tags=all_tags)
        kernel = _run_without_grad(fn) if disable_grad else fn
        for key in dispatch_keys:
            lib.impl(op_name, kernel, key)
        op = FastCustomOp(qualname=qualname, lib=lib, python_fn=fn)
        _FAST_OPS[qualname] = op
        return op

    return decorator


class FastCustomOp:
    """Handle returned by :func:`fast_custom_op`.

    Behaves like ``@torch.library.custom_op``'s return value: callable and
    exposes ``register_fake``. A call goes straight to the op's ``default``
    overload in the C++ dispatcher, which is also the target torch.compile
    records for it, so graph passes can match the op as they would any other.
    """

    __slots__ = ("qualname", "namespace", "op_name", "_python_fn", "_op", "_lib", "_abstract_fn")

    def __init__(self, qualname: str, lib: Library, python_fn: Callable):
        self.qualname = qualname
        self.namespace, self.op_name = qualname.split("::", 1)
        self._python_fn = python_fn
        self._op = getattr(getattr(torch.ops, self.namespace), self.op_name).default
        # Holding the Library keeps its finalizer from unregistering the op.
        self._lib = lib
        # Set by the register_fake method, under the name custom_op uses, so
        # code that reads the fake from there works on either kind of op.
        self._abstract_fn: Callable | None = None

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def register_fake(self, fake_fn: Callable) -> Callable:
        register_fake(self.qualname, fake_fn)
        self._abstract_fn = fake_fn
        return fake_fn

    @property
    def python_impl(self) -> Callable:
        """The original un-wrapped Python function (for tests/introspection)."""
        return self._python_fn
