# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Move the FP8 quantize into the RMSNorm that feeds it.

Where every reader of a norm takes an already-quantized activation, the conversion belongs
in the norm's epilogue rather than in a pass of its own. That is what nanojet runs natively,
and it is why the native layer has no standalone quantize kernel.

Runs after ``fuse_fp8_linear`` so the readers are the TensorRT LLM FP8 op, and before the
gated SwiGLU GEMM, which needs an FP8 activation to apply at all.
"""

from typing import Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ....nanojet_utils import ensure_tune_configs, nanojet_supports
from ...custom_ops.normalization.nanojet_rmsnorm_fp8 import register
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.nanojet_graph import accepts_out_dtype, fp8_linear_ops, per_tensor_scale, set_val_meta
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    extract_op_args,
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


# Consumers that read an already-quantized activation, and where each keeps the scale it
# dequantizes by. Resolved lazily: the nanojet ops only exist once nanojet is installed, and
# this module is imported unconditionally by the library's package scan.
def _fp8_consumer_ops():
    """The ops that can take this norm's output already quantized.

    They all spell the argument ``input_scale``, so only the op identities are listed here —
    the position is looked up from each op's schema rather than tabulated by hand.
    """
    consumers = list(fp8_linear_ops())
    # Registered only if the QKV fusion is also enabled; absent simply means this norm has no
    # such consumer to consider.
    fused_qkv = getattr(torch.ops.auto_deploy, "nanojet_fused_qkv_gemm_norm_rope", None)
    if fused_qkv is not None:
        consumers.append(fused_qkv)
    return consumers


def _shared_quantize_scale(gm: GraphModule, normed: Node) -> Optional[float]:
    """The one dequant scale every reader of ``normed`` uses, if they all take FP8.

    Quantizing in the epilogue only works when nothing downstream wants the BF16 value and
    everyone agrees on the scale; otherwise the saved kernel reappears elsewhere, or a
    reader silently sees a differently-scaled tensor.

    ``get_shared_input_scale_for_fp8_linears`` answers the same question for linears alone;
    the readers here also include nanojet's fused QKV, which it does not know about.
    """
    readers, traversal_ok = collect_terminal_users_through_passthrough(normed)
    if not traversal_ok or not readers:
        return None
    consumer_ops = _fp8_consumer_ops()
    scales = set()
    for reader in readers:
        if not any(is_op(reader, op) for op in consumer_ops):
            return None
        scale = per_tensor_scale(gm, extract_op_args(reader, "input_scale")[0])
        if scale is None:
            return None
        scales.add(scale)
    return scales.pop() if len(scales) == 1 else None


class FuseNanojetRMSNormFP8Config(TransformConfig):
    """Configuration for folding the FP8 quantize into nanojet's RMSNorm."""


@TransformRegistry.register("fuse_nanojet_rmsnorm_fp8")
class FuseNanojetRMSNormFP8(BaseTransform):
    """Replace ``rmsnorm`` + downstream quantize with one nanojet norm emitting e4m3."""

    config: FuseNanojetRMSNormFP8Config

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNanojetRMSNormFP8Config

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled or not register():
            return gm, TransformInfo(skipped=True, num_matches=0)

        ensure_tune_configs(factory)

        graph = gm.graph
        num_matches = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_rmsnorm) or len(node.args) < 3:
                continue
            hidden_states, weight, eps = node.args[0], node.args[1], node.args[2]
            if not isinstance(hidden_states, Node) or not isinstance(weight, Node):
                continue
            if not isinstance(eps, float):
                continue
            value = node.meta.get("val")
            if value is None or value.dtype != torch.bfloat16:
                continue
            if not nanojet_supports(
                "unified_rmsnorm",
                hidden_size=int(value.shape[-1]),
                zero_centered_weight=False,
                multiply_in_fp32=False,
                input_dtype=value.dtype,
            ):
                continue

            quantize_scale = _shared_quantize_scale(gm, node)
            if quantize_scale is None:
                continue

            output_dtype = str(value.dtype).rsplit(".", 1)[-1]
            with graph.inserting_before(node):
                fused = graph.call_function(
                    torch.ops.auto_deploy.nanojet_rmsnorm_fp8.default,
                    args=(hidden_states, weight, eps, 1.0 / quantize_scale),
                )
            # An FP8 activation carries no hint of what the linear should emit.
            readers, _ = collect_terminal_users_through_passthrough(node)
            for reader in readers:
                if accepts_out_dtype(reader):
                    set_op_args(reader, out_dtype=output_dtype)
            set_val_meta(fused, node, dtype=torch.float8_e4m3fn)
            node.replace_all_uses_with(fused)
            num_matches += 1

        if num_matches:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(
                f"fuse_nanojet_rmsnorm_fp8: {num_matches} norms quantizing in the epilogue"
            )

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=True,
        )
