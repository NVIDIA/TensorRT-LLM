# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host hooks: what kernel host code needs from the integration layer, provided by it.

Kernels, families and ops never import the integration layer. It registers each hook as a zero-argument loader
when it is imported; a host module asks for the hook by name. get() calls the loader on every call, so a re-imported
provider module is never served stale.
"""

from __future__ import annotations

from collections.abc import Callable

STEP_META = (
    "step_meta_insert_kwargs"  # (layer_name, index_prefix) -> the front's step-metadata kwargs
)
PLAN_ERROR = "plan_error"  # the exception class a build-time refusal raises
TOPP_PATCH = "topp_split_patch"  # the module that pins vLLM's split top-p (stock_module / refusal / host_for)
NAMES = (STEP_META, PLAN_ERROR, TOPP_PATCH)

_HOOKS: dict[str, Callable[[], object]] = {}


def provide(name: str, loader: Callable[[], object]) -> None:
    """Register `loader`, a zero-argument callable that returns the hook (it imports its provider statically)."""
    if name not in NAMES:
        raise KeyError(f"unknown host hook {name!r}; known {NAMES}")
    if not callable(loader):
        raise TypeError(f"host hook {name}: loader {loader!r} is not callable")
    _HOOKS[name] = loader


def get(name: str):
    loader = _HOOKS.get(name)
    if loader is None:
        raise LookupError(
            f"host hook {name!r} is not provided: the integration layer registers it when imported "
            "(import it before building a chain)"
        )
    return loader()


def provided() -> dict[str, Callable[[], object]]:
    return dict(_HOOKS)


__all__ = ["STEP_META", "PLAN_ERROR", "TOPP_PATCH", "NAMES", "provide", "get", "provided"]
