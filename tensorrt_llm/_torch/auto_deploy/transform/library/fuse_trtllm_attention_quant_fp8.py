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

"""Prepare TRTLLM attention output FP8 quantization for downstream FP8 linears.

This transform runs in the pattern_matcher stage and precomputes graph wiring required by
the TRTLLM attention backend when all terminal consumers are FP8 linear ops that share
the same input scale.
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.attention.trtllm_attention import (
    clear_trtllm_attention_fp8_input_scale,
    set_trtllm_attention_fp8_input_scale,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    get_shared_input_scale_for_fp8_linears,
    is_op,
    set_op_args,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _get_out_dtype_str(attn_node: Node) -> str:
    val = attn_node.meta.get("val")
    if val is None:
        raise ValueError(
            f"{attn_node.format_node()} missing meta['val']; cannot determine out_dtype."
        )
    if not hasattr(val, "dtype"):
        raise ValueError(
            f"{attn_node.format_node()} meta['val'] has no dtype; cannot determine out_dtype."
        )
    if val.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"{attn_node.format_node()} has unsupported dtype {val.dtype}; "
            "expected float16/bfloat16/float32."
        )
    return str(val.dtype).replace("torch.", "")


@TransformRegistry.register("fuse_trtllm_attn_quant_fp8")
class FuseTrtllmAttentionQuantFP8(BaseTransform):
    """Prepare attention->FP8 linear path so TRTLLM attention can emit FP8 directly."""

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
        cnt = 0

        for attn_node in list(gm.graph.nodes):
            if not is_op(attn_node, torch.ops.auto_deploy.torch_attention.default):
                continue

            terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
                attn_node, max_traversal_nodes=256
            )
            fp8_users, first_scale = get_shared_input_scale_for_fp8_linears(terminal_users)
            if not (traversal_ok and fp8_users and len(fp8_users) == len(terminal_users)):
                clear_trtllm_attention_fp8_input_scale(attn_node)
                continue

            out_dtype_str = _get_out_dtype_str(attn_node)
            for user in fp8_users:
                set_op_args(user, out_dtype=out_dtype_str)

            if first_scale is not None:
                if first_scale.op == "get_attr":
                    attn_node.prepend(first_scale)
                set_trtllm_attention_fp8_input_scale(attn_node, first_scale)
                cnt += 1

        if cnt > 0:
            gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
