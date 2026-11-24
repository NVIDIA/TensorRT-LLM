# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common CuTeDSL compilation and caller-stream handling for FP4 MLA."""

import contextlib
import sys
from collections.abc import Callable
from typing import Any

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL

from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    _CUTEDSL_COMPILE_LOCK,
    _cutedsl_verbose_compile_enabled,
)

_PYIR_STDOUT_LINES = frozenset(
    {
        "Enabling PyIR, it was False",
        "Enabling PyIR, it is now True",
        "Disabling PyIR, it was True",
        "Disabling PyIR, it is now False",
    }
)


class _PyIRStdoutFilter:
    def __init__(self, output: Any) -> None:
        self._output = output
        self._pending = ""

    def write(self, text: str) -> int:
        lines = (self._pending + text).split("\n")
        self._pending = lines.pop()
        for line in lines:
            if line.rstrip("\r") not in _PYIR_STDOUT_LINES:
                self._output.write(f"{line}\n")
        return len(text)

    def flush(self) -> None:
        self._output.flush()

    def finish(self) -> None:
        if self._pending and self._pending.rstrip("\r") not in _PYIR_STDOUT_LINES:
            self._output.write(self._pending)
        self._pending = ""
        self._output.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._output, name)


def _compile_cutedsl(*args: object, **kwargs: object) -> Callable[..., object]:
    """Compile the FP4 MLA kernels with their required PyIR frontend, one compilation at a time."""
    with _CUTEDSL_COMPILE_LOCK:
        if _cutedsl_verbose_compile_enabled():
            with BaseDSL.enable_pyir():
                return cute.compile(*args, **kwargs)
        stdout_filter = _PyIRStdoutFilter(sys.stdout)
        try:
            with contextlib.redirect_stdout(stdout_filter), BaseDSL.enable_pyir():
                return cute.compile(*args, **kwargs)
        finally:
            stdout_filter.finish()


def _current_cu_stream() -> cuda.CUstream:
    """Use the caller's stream for launch ordering and CUDA graph capture."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
