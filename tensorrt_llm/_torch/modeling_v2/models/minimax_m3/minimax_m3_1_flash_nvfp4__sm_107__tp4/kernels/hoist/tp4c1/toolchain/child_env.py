# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-process environment for legacy torch-extension builds: cpp_extension's subprocess module is wrapped so every
explicit env= gets PATH prefixed and CUDA_HOME set by default. Calls without env= pass through; os.environ is untouched.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
OVERRIDES: dict[str, str] = {}  # "PATH_PREFIX" / "CUDA_HOME"


def merge_env(env: dict) -> dict:
    """The env a build child receives: the caller's env (torch: a copy of the process env) plus the overrides."""
    out = dict(env)
    pre = OVERRIDES.get("PATH_PREFIX")
    if pre:
        cur = out.get("PATH", "")
        if not (cur == pre or cur.startswith(pre + ":")):
            out["PATH"] = pre + ":" + cur
    if OVERRIDES.get("CUDA_HOME"):
        out.setdefault(
            "CUDA_HOME", OVERRIDES["CUDA_HOME"]
        )  # setdefault: an explicit user CUDA_HOME wins
    return out


class _SubprocessProxy:
    """Module proxy for torch.utils.cpp_extension.subprocess: env= of run/check_output/check_call/Popen is merged."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, k):
        return getattr(self._mod, k)

    def _wrap(self, fn, args, kw):
        if kw.get("env") is not None:
            kw["env"] = merge_env(kw["env"])
        return fn(*args, **kw)

    def run(self, *a, **kw):
        return self._wrap(self._mod.run, a, kw)

    def check_output(self, *a, **kw):
        return self._wrap(self._mod.check_output, a, kw)

    def check_call(self, *a, **kw):
        return self._wrap(self._mod.check_call, a, kw)

    def Popen(self, *a, **kw):  # noqa: N802
        return self._wrap(self._mod.Popen, a, kw)


def child_env(cuda_home: str | None = None, path_prefix: str | None = None) -> None:
    """Register the overrides (process-wide) and wrap cpp_extension's subprocess. Also points cpp_extension at
    cuda_home for the nvcc it writes into the ninja file."""
    import torch.utils.cpp_extension as ce

    with _LOCK:
        if path_prefix:
            OVERRIDES["PATH_PREFIX"] = path_prefix
        if cuda_home:
            OVERRIDES.setdefault("CUDA_HOME", cuda_home)
            ce.CUDA_HOME = OVERRIDES["CUDA_HOME"]
        if not isinstance(ce.subprocess, _SubprocessProxy):
            ce.subprocess = _SubprocessProxy(ce.subprocess)


__all__ = ["OVERRIDES", "child_env", "merge_env"]
