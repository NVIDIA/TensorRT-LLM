# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family library, so a fake for an op the
library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


def _s1b_fold_fake(partials, bias, logits_out, ids, weights, route_scale, pdl):
    """m31_s1b_fold (S1B_FOLD): writes ids / weights (and logits_out) in place; returns nothing."""
    return None


def _s2_moe_tail_fake(
    fc2,
    scales,
    perm,
    shared,
    residual,
    norm_weight,
    workspace,
    rank,
    nranks,
    eps,
    trigger_completion_at_end=True,
):
    """m31_s2_moe_tail: (norm_out, residual_out) like shared / residual, the MXFP8 activation [tokens, hidden] and
    its swizzled scale bytes (rows padded to 128, scale columns to 4)."""
    tokens, hidden = shared.shape
    sf_size = (tokens + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4)
    return (
        torch.empty_like(shared),
        torch.empty_like(residual),
        shared.new_empty((tokens, hidden), dtype=torch.float8_e4m3fn),
        shared.new_empty((sf_size,), dtype=torch.uint8),
    )


def _ar_prepushed_fake(
    rank,
    workspace,
    allreduce_in,
    residual_in,
    residual_out,
    norm_out,
    gamma,
    eps,
    trigger_completion_at_end,
    mx_out=None,
    mx_sf=None,
):
    """m31_ar_prepushed (OPROJ_PUSH_GEMM): writes residual_out, norm_out (and mx_out / mx_sf) in place."""
    return None


torch.library.register_fake(f"{_NS}::minimax_m31_s1b_fold")(_s1b_fold_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_s2_moe_tail")(_s2_moe_tail_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_ar_prepushed")(_ar_prepushed_fake)
