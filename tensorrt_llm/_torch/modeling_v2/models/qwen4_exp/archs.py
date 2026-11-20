# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPUs the qwen4_exp targets serve, and the GPUs each fast path launches on.

A fast path is a GPU-specific kernel with a general kernel beside it in the same op; the op
takes the general kernel where `fast_path` is false. A kernel with no general kernel beside it
(the row-tile pass) runs on every GPU in `SERVED`.
"""

from __future__ import annotations

from typing import Literal

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

SM_103 = (10, 3)
SM_107 = (10, 7)

#: The GPUs routing and the targets admit.
SERVED = frozenset({SM_103, SM_107})
#: `ARCHS` of the family's op entries.
SERVED_ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

FastPath = Literal["qsa_nvfp4_verify", "qsa_index_scores", "qsa_grouped_decode", "cute_dsl_moe"]

FAST_PATHS: dict[FastPath, frozenset[tuple[int, int]]] = {
    "qsa_nvfp4_verify": SERVED,
    "qsa_index_scores": SERVED,
    "qsa_grouped_decode": SERVED,
    # Its grouped-GEMM runners are built for sm_100 and sm_103 only.
    "cute_dsl_moe": frozenset({SM_103}),
}


def served(capability: tuple[int, int]) -> bool:
    """Whether `capability` (a `torch.cuda.get_device_capability` pair) is a GPU the family
    serves."""
    return tuple(capability) in SERVED


def fast_path(name: FastPath, capability: tuple[int, int]) -> bool:
    """Whether fast path `name` launches on `capability`; where it does not, its op runs the
    general kernel."""
    return tuple(capability) in FAST_PATHS[name]


def served_names() -> str:
    """The served GPUs as `sm_<major><minor>`, for messages."""
    return ", ".join(f"sm_{major}{minor}" for major, minor in sorted(SERVED))
