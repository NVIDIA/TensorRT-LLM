# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Late-compile detector: report kernel compiles that happen after the executor finished its warmup.

A compile inside a served request costs that request its whole compile time, and in-memory compiles (CuTe DSL
handles, Triton specializations) leave no file in the JIT cache directories, so cache-file checks cannot see them.
After ``mark_server_ready()`` every ``cutlass.cute.compile`` call and every Triton JIT compile logs one warning line
``[late-compile] op=<op> key=<key>``. The executor arms it only for a model whose RuntimeHooks set
late_compile_detect. Kill switch: TRTLLM_LATE_COMPILE_DETECT=0.

Not covered here: the TRTLLM-Gen FMHA runtime NVRTC compile (C++, an in-memory module cache warmed for generation
kernels only), whose warmup and persistence fix is tracked with that kernel library, and the fmha_sm100 package's
on-disk AOT cache, whose post-warmup "[aot_cache] Saved" lines are its own signal.
"""

import os
from typing import Any

from tensorrt_llm.logger import logger

_ENABLED = os.environ.get("TRTLLM_LATE_COMPILE_DETECT", "1") != "0"
_ready = [False]


def note(op: str, key: Any) -> None:
    """Log a compile of ``op`` for ``key`` when it happens after warmup."""
    if _ready[0]:
        logger.warning(f"[late-compile] op={op} key={key}")


class _CuteCompile:
    """Stands in for ``cutlass.cute.compile``: notes each call, then forwards it (``cute.compile[...]`` included)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else None
        note("cute.compile", getattr(target, "__qualname__", type(target).__name__))
        return self._inner(*args, **kwargs)

    def __getitem__(self, item: Any) -> "_CuteCompile":
        return _CuteCompile(self._inner[item])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _hook_cute() -> None:
    try:
        import cutlass.cute as cute
    except ImportError:
        return
    if not isinstance(cute.compile, _CuteCompile):
        cute.compile = _CuteCompile(cute.compile)


def _hook_triton() -> None:
    try:
        import triton
    except ImportError:
        return
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    if runtime is None or not hasattr(runtime, "jit_post_compile_hook"):
        logger.info(
            "[late-compile] Triton post-compile hook unavailable; Triton compiles are not reported"
        )
        return
    previous = runtime.jit_post_compile_hook

    def hook(*args: Any, **kwargs: Any) -> Any:
        try:
            fn = kwargs.get("fn")
            name = getattr(getattr(fn, "fn", fn), "__qualname__", "?")
            note("triton", f"{name} {kwargs.get('key', '')}")
        except Exception:  # noqa: BLE001  (the detector must never break a compile)
            pass
        return previous(*args, **kwargs) if previous is not None else None

    runtime.jit_post_compile_hook = hook


def mark_server_ready() -> None:
    """Arm the detector once the executor's warmup is done (idempotent)."""
    if not _ENABLED or _ready[0]:
        return
    _hook_cute()
    _hook_triton()
    _ready[0] = True
    logger.info("[late-compile] detector armed after warmup")
