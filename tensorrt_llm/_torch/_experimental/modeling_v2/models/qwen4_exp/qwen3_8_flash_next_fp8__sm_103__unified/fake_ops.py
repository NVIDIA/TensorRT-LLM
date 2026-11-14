# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family
library, so a fake for an op the library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


def _host_query(qualname: str):
    """Ops without tensor arguments answer a host-side question at setup; tracing never runs them."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{qualname} is a setup-time query; call it outside tracing")

    return torch.library.register_fake(qualname)(fake)


_host_query(f"{_NS}::fp8_block_moe_tpb_workspace_bytes")


@torch.library.register_fake(f"{_NS}::fp8_block_moe_tpb")
def _(
    logits,
    xq,
    xsf,
    w1,
    s1,
    w2,
    s2,
    rank,
    workspace,
    enable_pdl=False,
    sh_x=None,
    sh_gu=None,
    sh_gl=None,
    sh_dn=None,
):
    return logits.new_empty((logits.shape[0], xq.shape[1]))


@torch.library.register_fake(f"{_NS}::hc_allreduce_inject_norm")
def _(
    partial,
    workspace,
    rank,
    nranks,
    hc,
    inject,
    shared,
    gate,
    norm_weight,
    eps,
    program_warps,
    trigger_completion_at_end,
):
    # Two new contiguous [rows, hc_count * hidden] tensors (`hc` may carry a padded row stride).
    shape = (hc.shape[0], hc.shape[1])
    return [hc.new_empty(shape), hc.new_empty(shape)]


@torch.library.register_fake(f"{_NS}::fp8_block_scale_moe_front")
def _(
    routing_logits,
    hidden_states_bf16,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts,
    top_k,
    intermediate_size,
    local_expert_offset,
    local_num_experts,
    routing_method_type,
    tile_n,
    config,
):
    # A new [rows, hidden] bf16 tensor; hidden_states / hidden_states_scale are written in place.
    return hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)
