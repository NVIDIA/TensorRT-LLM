# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform: fuse flashinfer_rms_norm + trtllm_finegrained_fp8_linear.

Replaces the pattern:

    %norm = flashinfer_rms_norm(%x, %weight, eps)
    %q    = trtllm_finegrained_fp8_linear(%norm, %wq, None, %wsq)
    %k    = trtllm_finegrained_fp8_linear(%norm, %wk, None, %wsk)
    ...   (any number of finegrained FP8 linears sharing %norm as input)

with:

    %fused = rms_norm_fp8_1x128(%x, %weight, eps)
    %bf16  = getitem(%fused, 0)   # BF16 norm, replaces %norm for non-linear users
    %fp8   = getitem(%fused, 1)   # FP8 pre-quantized activations
    %sf    = getitem(%fused, 2)   # [K//128, M] activation scale factors
    %q     = trtllm_finegrained_fp8_linear_prequant(%fp8, %sf, %wq, None, %wsq)
    %k     = trtllm_finegrained_fp8_linear_prequant(%fp8, %sf, %wk, None, %wsk)
    ...

This eliminates the direct_copy + scale_1x128_kernel overhead that
trtllm_finegrained_fp8_linear pays internally for each call to quantize its
BF16 input on the fly. With N FP8 linears sharing one norm output, we save
2*(N-1) kernel launches and replace 1 BF16 quant with the Triton fused op.

Conditions for fusion:
  1. The norm op is flashinfer_rms_norm.
  2. ALL direct users of the norm output are trtllm_finegrained_fp8_linear.
     (If any non-linear user exists, fusion is skipped to avoid BF16/FP8 split.)
  3. K (the last dimension of the norm output) is divisible by 128.
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


def _is_finegrained_fp8_linear(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.trtllm_finegrained_fp8_linear)


def _extract_finegrained_linear_args(node: Node):
    """Return (input, weight, bias, weight_scale) for trtllm_finegrained_fp8_linear."""
    return extract_op_args(node, "input", "weight", "bias", "weight_scale")


def _norm_k_dim(norm_node: Node) -> int:
    """Return the K (last) dimension of a norm node's output, or -1 if unknown."""
    meta = norm_node.meta.get("tensor_meta") or norm_node.meta.get("val")
    if meta is None:
        return -1
    try:
        shape = meta.shape
        if len(shape) == 0:
            return -1
        k = int(shape[-1])
        return k
    except Exception:
        return -1


def _collect_fp8_linear_users(norm_node: Node) -> Tuple[List[Node], bool]:
    """Collect all direct trtllm_finegrained_fp8_linear users of norm_node.

    Returns:
        (users, all_fp8): list of matching FP8 linear nodes, and whether ALL
        direct users are FP8 linears (fusion safety check).
    """
    users = list(norm_node.users.keys())
    fp8_users = [u for u in users if _is_finegrained_fp8_linear(u)]
    all_fp8 = len(fp8_users) == len(users) and len(fp8_users) > 0
    return fp8_users, all_fp8


@TransformRegistry.register("fuse_rmsnorm_fp8_finegrained")
class FuseRMSNormFP8Finegrained(BaseTransform):
    """Fuse flashinfer_rms_norm + N×trtllm_finegrained_fp8_linear.

    This transform runs in the post_load_fusion stage and eliminates the
    repeated dynamic FP8 quantization (direct_copy + scale_1x128_kernel)
    that trtllm_finegrained_fp8_linear performs per call. When multiple FP8
    linears share one norm output (e.g. Q/K/V projections), a single Triton
    kernel now produces both the BF16 norm and pre-quantized FP8+scales.
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
        original_nodes = list(graph.nodes)
        node_order = {n: i for i, n in enumerate(original_nodes)}

        for norm_node in original_nodes:
            if not is_op(norm_node, torch.ops.auto_deploy.flashinfer_rms_norm):
                continue

            fp8_users, all_fp8 = _collect_fp8_linear_users(norm_node)
            if not all_fp8:
                # Non-linear users exist — skip to avoid BF16/FP8 split issues.
                continue

            # Check K is divisible by 128
            k_dim = _norm_k_dim(norm_node)
            if k_dim <= 0 or k_dim % 128 != 0:
                continue

            # Extract norm inputs: (input, weight, eps)
            norm_input, norm_weight, norm_eps = extract_op_args(norm_node, "input", "weight", "eps")

            # Find insertion point: before the earliest FP8 linear
            earliest = min(fp8_users, key=lambda n: node_order.get(n, float("inf")))

            # Insert fused op + getitems
            with graph.inserting_before(earliest):
                fused_node = graph.call_function(
                    torch.ops.auto_deploy.rms_norm_fp8_1x128.default,
                    args=(norm_input, norm_weight, norm_eps),
                )
                bf16_node = graph.call_function(operator.getitem, args=(fused_node, 0))
                fp8_node = graph.call_function(operator.getitem, args=(fused_node, 1))
                sf_node = graph.call_function(operator.getitem, args=(fused_node, 2))

            # Replace each finegrained FP8 linear with the prequant variant
            for fp8_linear in fp8_users:
                _, weight_arg, bias_arg, ws_arg = _extract_finegrained_linear_args(fp8_linear)
                with graph.inserting_before(fp8_linear):
                    prequant_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_finegrained_fp8_linear_prequant.default,
                        args=(fp8_node, sf_node, weight_arg, bias_arg, ws_arg),
                    )
                fp8_linear.replace_all_uses_with(prequant_node)
                graph.erase_node(fp8_linear)
                cnt += 1

            # Replace all remaining uses of norm_node with bf16_node
            norm_node.replace_all_uses_with(bf16_node)
            if len(norm_node.users) == 0:
                graph.erase_node(norm_node)

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
