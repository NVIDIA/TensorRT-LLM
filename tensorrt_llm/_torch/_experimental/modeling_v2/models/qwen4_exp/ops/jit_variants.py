# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile a Triton kernel's launch variants before the step that would first need one.

A launcher that picks constexprs or launch options from the step's shape compiles one variant per
value; a served step whose value the engine's warm-up did not reach would compile inside a timed
request. `warm_variants` compiles the launcher's whole finite set at its first eager call for a
geometry, with that call's arguments and only the bucketed values changed. It only compiles
(`JITFunction.warmup`): nothing is launched, so results cannot move. Calls inside a CUDA-graph
capture defer to the next eager call.
"""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Mapping, Sequence

import torch

_WARMED: set = set()


def first_time(name: str, key: Hashable) -> bool:
    """True once per `(name, key)` in a process; False inside a CUDA-graph capture (the key
    stays unmarked, so the next eager call warms it)."""
    if (name, key) in _WARMED:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    _WARMED.add((name, key))
    return True


def compile_only(kernel, args: Sequence[Any], **kwargs: Any) -> None:
    """Compile `kernel`'s variant for `args` and `kwargs` without launching it. A stand-in
    without `warmup` (a test's launch recorder) is skipped."""
    warmup = getattr(kernel, "warmup", None)
    if warmup is not None:
        warmup(*args, grid=(1,), **kwargs)


def warm_variants(
    kernel,
    key: Hashable,
    args: Sequence[Any],
    variants: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> None:
    """Compile `kernel` once per `key` for every mapping of `variants` laid over `kwargs` (the
    call's constexprs and launch options)."""
    name = getattr(getattr(kernel, "fn", None), "__name__", None) or repr(kernel)
    if first_time(name, key):
        for variant in variants:
            compile_only(kernel, args, **{**kwargs, **variant})
