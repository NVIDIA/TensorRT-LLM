# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common CuTeDSL compilation and caller-stream handling for FP4 MLA."""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL


def _skip_schedule_hint() -> None:
    """Stand in for a scheduling hint the installed CuTe DSL does not provide."""


# Instruction-scheduling hints for ptxas. They do not change results, and the
# public nvidia-cutlass-dsl release does not provide them, so they resolve to a
# no-op when absent. When present, kernels call the DSL functions directly.
cfence: Callable[[], None] = getattr(cute.nvgpu, "cfence", _skip_schedule_hint)
warp_switch: Callable[[], None] = getattr(cute.nvgpu, "warp_switch", _skip_schedule_hint)


def _compile_cutedsl(*args: object, **kwargs: object) -> Callable[..., object]:
    """Compile the FP4 MLA kernels with their required PyIR frontend."""
    with BaseDSL.enable_pyir():
        return cute.compile(*args, **kwargs)


def _current_cu_stream() -> cuda.CUstream:
    """Use the caller's stream for launch ordering and CUDA graph capture."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
