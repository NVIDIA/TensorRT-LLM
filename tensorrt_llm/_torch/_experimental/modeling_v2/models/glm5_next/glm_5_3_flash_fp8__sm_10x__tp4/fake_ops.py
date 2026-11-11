# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family
library, so a fake for an op the library does not define fails the load. Both ops
write their outputs in place and return nothing. Each fake is named ``fake_<op>``
and takes the op's schema arguments by their schema names, in schema order; the
target's fake-op test checks both against the library.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::decode_moe")
def fake_decode_moe(
    x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2, out, pdl
):
    return None


@torch.library.register_fake(f"{_NS}::mhc_ar_boundary")
def fake_mhc_ar_boundary(
    partial,
    workspace,
    rank,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    sinkhorn_iters,
    norm_eps,
    hc_eps,
    sinkhorn_eps,
    post_mult,
    layer_input_eps,
    residual_out,
    post_out,
    comb_out,
    layer_input,
    pdl,
):
    return None
