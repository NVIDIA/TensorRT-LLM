# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops.

The loader imports this module right after it opens the family library, so a
fake for an op the library does not define fails the load. Every op the
target's C++ registers has a fake here; the native claims test checks that by
reading both sides.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__package__)


@torch.library.register_fake(f"{_NS}::scale_add")
def _(x, alpha, beta):
    return torch.empty_like(x)
