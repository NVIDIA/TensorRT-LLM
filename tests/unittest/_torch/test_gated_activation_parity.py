# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python and C++ gated-activation lists must agree."""

import re
from pathlib import Path

from tensorrt_llm._torch.utils import ActivationType, is_gated_activation

_HEADER = (
    Path(__file__).parents[3]
    / "cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"
)


def test_is_gated_activation_matches_the_cutlass_header():
    """``is_gated_activation`` decides ``intermediate_size_expand_ratio`` and
    therefore the whole FC1 weight geometry, while the kernel validates the
    shapes it receives against the C++ list. When the two disagree, a MoE
    allocates w3_w1 for the wrong geometry and the loader pads/truncates into
    it without complaint -- ``torch.nn.functional.pad`` with a negative pad
    truncates rather than raising -- so the first sign is a shape check deep
    inside the kernel, or nothing at all.

    The comment above the Python function already asks for this alignment.
    This asserts it.
    """
    body = re.search(
        r"constexpr bool isGatedActivation\(ActivationType activation_type\)\s*\{(.*?)\}",
        _HEADER.read_text(),
        re.DOTALL,
    )
    assert body is not None, f"isGatedActivation not found in {_HEADER}"
    cpp_gated = set(re.findall(r"ActivationType::(\w+)", body.group(1)))

    python_gated = {a.name for a in ActivationType if is_gated_activation(a)}
    assert python_gated == cpp_gated
