# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Transform: fuse FP8 1x128 activation quantization across shared consumers.

For any BF16 (or FP16) node whose users include at least one
`trtllm_finegrained_fp8_linear` and at least one
`trtllm_quant_finegrained_fp8_moe_fused`, insert a single
`fp8_quantize_1x128` on that shared input and rewire both consumers to
pre-quantized variants:

    shared_bf16                                shared_bf16
      ├─ fp8_linear   (router)                   ├─ fp8_quantize_1x128 ──┐
      └─ fp8_moe_fused                           │                      │
                                                 ├─ fp8_linear_prequant (router)
                                                 └─ fp8_moe_fused_prequant

Net per-layer savings: one fp8_quantize_1x128 executes instead of being
re-run internally by each consumer (each internal call is
`direct_copy + scale_1x128_kernel` ≈ 5–6 µs).

This transform complements `fuse_rmsnorm_fp8_finegrained`, which handles the
analogous pattern where the shared source is a `flashinfer_rms_norm`. Here
the source is arbitrary (e.g., a view of `trtllm_fused_allreduce_residual_rmsnorm`'s
output), so we emit an explicit `fp8_quantize_1x128` instead of replacing the
parent op.
"""

import operator
from typing import List, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _is_fp8_linear(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.trtllm_finegrained_fp8_linear)


def _is_fp8_moe(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.trtllm_quant_finegrained_fp8_moe_fused)


def _src_last_dim(src: Node) -> int:
    meta = src.meta.get("tensor_meta") or src.meta.get("val")
    if meta is None:
        return -1
    try:
        return int(meta.shape[-1])
    except Exception:
        return -1


@TransformRegistry.register("fuse_fp8_prequant_shared")
class FuseFP8PrequantShared(BaseTransform):
    """Share a single fp8_quantize_1x128 across FP8 linear + MoE users.

    Targets the post-attention residual path in DeepSeek-V3 / MiniMax-V3
    style MoE blocks: the same BF16 tensor (view of
    `trtllm_fused_allreduce_residual_rmsnorm`) feeds both the router FP8
    linear and the fused FP8 MoE op.
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        cnt = 0

        # Collect candidate source nodes (any node with ≥2 FP8 users where at
        # least one is MoE — that's the savings target; pure linear-only is
        # already handled by fuse_rmsnorm_fp8_finegrained where applicable).
        seen_src: set = set()
        for node in list(graph.nodes):
            for user in node.users:
                if _is_fp8_moe(user) and user.args[0] is node:
                    seen_src.add(node)
                    break

        # When True, also fuses the FP8 linear (router) users into the shared
        # pre-quant — this is the "all-fused" path that saves ~10 kernels/iter.
        # See the module docstring for the rank-timing trade-off observed in
        # synthetic nsys traces; the real-workload effect is what end-to-end
        # benchmarking measures.
        _FUSE_LINEAR_USERS = True

        for src in seen_src:
            # Split users by op type; ignore non-consuming users (e.g., views
            # to other paths) to stay conservative.
            linear_users: List[Node] = []
            moe_users: List[Node] = []
            for u in list(src.users):
                if _is_fp8_linear(u) and u.args[0] is src:
                    linear_users.append(u)
                elif _is_fp8_moe(u) and u.args[0] is src:
                    moe_users.append(u)

            if not _FUSE_LINEAR_USERS:
                linear_users = []

            # Need at least one linear + one MoE user to see savings; or
            # a MoE user alone (still saves the MoE's internal quant).
            if not moe_users:
                continue

            k_dim = _src_last_dim(src)
            if k_dim <= 0 or k_dim % 128 != 0:
                continue

            # Insert the explicit fp8_quantize_1x128 before the earliest consumer.
            earliest = min(linear_users + moe_users, key=lambda n: list(graph.nodes).index(n))
            with graph.inserting_before(earliest):
                # fp8_quantize_1x128 returns (fp8, sf)
                quant_node = graph.call_function(
                    torch.ops.trtllm.fp8_quantize_1x128.default,
                    args=(src,),
                )
                fp8_node = graph.call_function(operator.getitem, args=(quant_node, 0))
                sf_node = graph.call_function(operator.getitem, args=(quant_node, 1))

            # Rewrite linear users → prequant variant.
            for lin in linear_users:
                _, weight_arg, bias_arg, ws_arg = extract_op_args(
                    lin, "input", "weight", "bias", "weight_scale"
                )
                with graph.inserting_before(lin):
                    prequant = graph.call_function(
                        torch.ops.auto_deploy.trtllm_finegrained_fp8_linear_prequant.default,
                        args=(fp8_node, sf_node, weight_arg, bias_arg, ws_arg),
                    )
                lin.replace_all_uses_with(prequant)
                graph.erase_node(lin)
                cnt += 1

            # Rewrite MoE users → prequant variant (pass src as shape/dtype hint).
            for moe in moe_users:
                args = list(moe.args)
                kwargs = dict(moe.kwargs)
                # Build prequant call: replace positional arg 0 (x) with (fp8, sf, hint)
                # and keep the rest.
                new_args = (fp8_node, sf_node, src) + tuple(args[1:])
                with graph.inserting_before(moe):
                    prequant = graph.call_function(
                        torch.ops.auto_deploy.trtllm_quant_finegrained_fp8_moe_fused_prequant.default,
                        args=new_args,
                        kwargs=kwargs,
                    )
                moe.replace_all_uses_with(prequant)
                graph.erase_node(moe)
                cnt += 1

        if cnt > 0:
            eliminate_dead_code(gm)
        gm.recompile()
        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
