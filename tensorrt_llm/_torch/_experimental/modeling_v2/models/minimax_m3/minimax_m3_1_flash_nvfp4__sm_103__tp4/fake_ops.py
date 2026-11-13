# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family library, so a fake for an op the
library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


def _returns_none(*args, **kwargs):
    return None


def _in_place(name: str) -> None:
    """An op that only writes its tensor arguments in place returns nothing."""
    torch.library.register_fake(f"{_NS}::{name}")(_returns_none)


_in_place("minimax_m31_tp4_sparse_decode_lc")
