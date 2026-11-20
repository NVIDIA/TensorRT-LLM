# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL compilation and caller-stream handling of the FP4 MLA backend, for the K3 copies.

The backend serializes the compilations of a process with one lock, so the family uses the
backend's helpers instead of a copy of them.
"""

from tensorrt_llm._torch.attention.backends.fp4_mla.cute_dsl_utils import (
    _compile_cutedsl,
    _current_cu_stream,
)

__all__ = ["_compile_cutedsl", "_current_cu_stream"]
