# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-time helpers the target's op builds share.

An op that owns its fallback is built once at load into a handle: the compiled kernels it
launches (None where a build failed, with the reason), the weights it needs, and the route
counter it reports to (`Count`, the core's `_count`). A build that fails is not an error: the
op runs its fallback on every call, and the warning says which kernel is missing.
"""

from __future__ import annotations

from typing import Any, Callable

from tensorrt_llm.logger import logger

#: `count(route, hit)`: one call of `route`, on its CuTe DSL kernel (`hit`) or on the fallback.
Count = Callable[[str, bool], None]


def no_count(route: str, hit: bool) -> None:
    """A counter that keeps nothing (handles built outside a core)."""
    return None


def build_or_reason(what: str, build: Callable[[], Any]) -> tuple[Any, str]:
    """`(build(), "")`, or `(None, reason)` with a warning: the op then runs its fallback."""
    try:
        return build(), ""
    except Exception as e:  # noqa: BLE001 -- the op keeps its fallback
        logger.warning(f"modeling_v2 GLM-5.2: {what} build failed ({e}); using the stock op")
        return None, f"{what} build failed: {e}"


def try_build(what: str, build: Callable[[], Any]) -> Any:
    """`build()`, or None with a warning: the caller's call sites then keep the stock op."""
    return build_or_reason(what, build)[0]
