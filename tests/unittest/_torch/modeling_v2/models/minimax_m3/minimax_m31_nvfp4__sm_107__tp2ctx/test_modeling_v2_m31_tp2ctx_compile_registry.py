# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp2ctx target's in-place ops are in the compile passes' name-keyed registries.

The target defines its own copies of the built-in model's in-place torch ops (trtllm::m31_tp2ctx_sm107_*). The compile
passes find in-place ops by name: an op missing from inplace_info() keeps the functionalization clones of every
tensor it mutates (here the whole paged KV / index-K / KV-scale caches: the A0 gate's 16.4 GiB clones), and an
attention op missing from the piecewise boundaries is not split out of the compiled graph. A rename of either op that
forgets the registries fails here.
"""

import torch

import tensorrt_llm._torch.models.modeling_minimaxm3  # noqa: F401 (the built-in ops compared against; models load lazily)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (  # noqa: F401 (registers the target's ops)
    modeling,
)
from tensorrt_llm._torch.compilation.piecewise_optimizer import _piecewise_boundary_ops
from tensorrt_llm._torch.compilation.utils import inplace_info


def test_the_targets_inplace_ops_are_reinplaced_like_the_builtin_ones():
    info = inplace_info()
    producer = torch.ops.trtllm.m31_tp2ctx_sm107_fused_sparse_qkv_producer.default
    attn = torch.ops.trtllm.m31_tp2ctx_sm107_attn_custom_op_inplace.default
    assert info[producer] == {2: "kv_cache", 3: "index_k_cache", 4: "kv_scale_cache"}
    assert info[attn] == {1: "output"}
    # Every mutable argument of the op schema, in order (remove_copy_pass rebuilds the call from these names only).
    mutable = [
        a.name
        for a in producer._schema.arguments
        if a.alias_info is not None and a.alias_info.is_write
    ]
    assert list(info[producer].values()) == mutable
    assert info[attn] == info[torch.ops.trtllm.minimax_m3_attn_custom_op_inplace.default]


def test_the_targets_attention_op_is_a_piecewise_boundary():
    assert (
        torch.ops.trtllm.m31_tp2ctx_sm107_attn_custom_op_inplace.default
        in _piecewise_boundary_ops()
    )
