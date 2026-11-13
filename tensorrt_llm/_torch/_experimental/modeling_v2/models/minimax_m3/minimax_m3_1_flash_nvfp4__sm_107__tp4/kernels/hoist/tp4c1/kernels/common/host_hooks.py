# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host hooks: what kernel host code needs from the integration layer, provided by it.

Kernels, families and ops never import tp4c1.vllm. The plugin registers each
hook by "module:attribute" when it is imported (tp4c1/vllm/__init__.py); a host module asks for it by name.
get() resolves the name on every call, so a re-imported provider module is never served stale.
"""

from __future__ import annotations

import importlib

STEP_META = (
    "step_meta_insert_kwargs"  # (layer_name, index_prefix) -> the front's step-metadata kwargs
)
PLAN_ERROR = "plan_error"  # the exception class a build-time refusal raises
TOPP_PATCH = "topp_split_patch"  # the module that pins vLLM's split top-p (stock_module / refusal / host_for)
NAMES = (STEP_META, PLAN_ERROR, TOPP_PATCH)

_HOOKS: dict[str, str] = {}


def provide(name: str, target: str) -> None:
    if name not in NAMES:
        raise KeyError(f"unknown host hook {name!r}; known {NAMES}")
    if target.count(":") != 1:
        raise ValueError(f"host hook {name}: target {target!r} is not 'module:attribute'")
    _HOOKS[name] = target


def get(name: str):
    target = _HOOKS.get(name)
    if target is None:
        raise LookupError(
            f"host hook {name!r} is not provided: the integration layer registers it when imported "
            "(import tp4c1.vllm before building a chain)"
        )
    mod, attr = target.split(":")
    m = importlib.import_module(mod)
    if not hasattr(m, attr):  # a submodule hook ("pkg:submodule") not imported yet
        return importlib.import_module(f"{mod}.{attr}")
    return getattr(m, attr)


def provided() -> dict[str, str]:
    return dict(_HOOKS)


__all__ = ["STEP_META", "PLAN_ERROR", "TOPP_PATCH", "NAMES", "provide", "get", "provided"]
