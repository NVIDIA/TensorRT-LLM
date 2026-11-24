# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-sided all-to-all dispatch kernel's expert-to-rank rule, ported for the tests.

``kernel_target_rank`` is ``compute_target_rank_id`` (moeAlltoAllKernels.cu) line for line: the
tests of ``a2a_expert_window`` / ``a2a_expert_owner`` and of the dep16 target's expert windows
compare against it, so the target's Python rule and the compiled kernel's are checked against one
port.
"""


def kernel_target_rank(expert: int, num_experts: int, ep_size: int) -> int:
    """The EP rank the dispatch kernel sends ``expert`` to: contiguous windows, ranks below
    ``num_experts % ep_size`` own ``num_experts // ep_size + 1`` experts, the rest one fewer."""
    base = num_experts // ep_size
    remainder = num_experts - base * ep_size
    if remainder == 0:
        return expert // base
    split = remainder * (base + 1)
    if expert < split:
        return expert // (base + 1)
    return remainder + (expert - split) // base
