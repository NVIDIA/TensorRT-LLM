# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common CuTeDSL compilation and caller-stream handling for FP4 MLA."""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.base_dsl.dsl import BaseDSL


def _compile_cutedsl(*args: object, **kwargs: object) -> Callable[..., object]:
    """Compile the FP4 MLA kernels with their required PyIR frontend."""
    with BaseDSL.enable_pyir():
        return cute.compile(*args, **kwargs)


def _current_cu_stream() -> cuda.CUstream:
    """Use the caller's stream for launch ordering and CUDA graph capture."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
