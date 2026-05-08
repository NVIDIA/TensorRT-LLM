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

"""Graph transform for fusing SiLU+Mul activation after GEMM fusion.

After fuse_gemms_mixed_children or fuse_fp8_gemms fuses gate+up projections into a
single GEMM, the graph contains:

    fused_out = gemm(x, gate_up_weight)
    gate = narrow(fused_out, -1, 0, intermediate_size)
    up = narrow(fused_out, -1, intermediate_size, intermediate_size)
    hidden = silu(gate) * up

This transform replaces the narrow+silu+mul pattern with a single fused op:

    fused_out = gemm(x, gate_up_weight)
    hidden = silu_and_mul(fused_out)

Two backends are supported:
- ``flashinfer`` (default): Uses FlashInfer's fused kernel.
- ``trtllm``: Uses TRT-LLM's Triton kernel, which is faster and supports fused
  FP8 quantization (eliminating a separate ``scaleMatrixPerTensorVec`` kernel when
  the consumer is an FP8 linear).
"""

import operator
from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

# Import to ensure the custom ops are registered
from ...custom_ops.linear.silu_mul import flashinfer_silu_and_mul, trtllm_silu_and_mul  # noqa: F401
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# FP8 linear ops whose input quantization can be fused into silu_and_mul.
# Only trtllm variant is supported (it has out_dtype param for FP8 input handling).
_FP8_LINEAR_OPS = {
    torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
}


def _get_narrow_info(node: Node) -> Optional[Tuple[Node, int, int]]:
    """Extract (parent, offset, length) from a narrow or split+getitem on dim=-1.

    Handles two patterns:
    1. ``torch.narrow(input, -1, start, length)``
    2. ``getitem(split_or_split_output(input, sizes, -1), idx)``

    Returns ``(parent, offset, length)`` or None.
    """
    # Pattern 1: torch.narrow(input, dim, start, length)
    if is_op(node, torch.narrow):
        if len(node.args) < 4:
            return None
        parent = node.args[0]
        dim = node.args[1]
        offset = node.args[2]
        length = node.args[3]
        if not isinstance(parent, Node) or dim != -1:
            return None
        if not isinstance(offset, int) or not isinstance(length, int):
            return None
        return parent, offset, length

    # Pattern 2: getitem(split_output/split_with_sizes(input, sizes, dim), idx)
    if is_op(node, operator.getitem):
        split_node = node.args[0]
        idx = node.args[1]
        if not isinstance(split_node, Node) or not isinstance(idx, int):
            return None
        if split_node.op != "call_function":
            return None
        # Check for aten.split_with_sizes(input, sizes, dim)
        if is_op(split_node, torch.ops.aten.split_with_sizes.default):
            if len(split_node.args) < 3:
                return None
            parent = split_node.args[0]
            sizes = split_node.args[1]
            dim = split_node.args[2]
        else:
            # Custom split_output(input) from GEMM fusion — get sizes from meta
            parent = split_node.args[0]
            if not isinstance(parent, Node):
                return None
            # Infer sizes from the getitem users' meta["val"] shapes
            sizes = []
            for user in split_node.users:
                if is_op(user, operator.getitem):
                    val = user.meta.get("val")
                    if val is not None:
                        sizes.append((user.args[1], val.shape[-1]))
            if not sizes:
                return None
            sizes.sort(key=lambda x: x[0])
            sizes = [s for _, s in sizes]
            dim = -1

        if not isinstance(parent, Node) or dim != -1:
            return None
        if not isinstance(sizes, (list, tuple)) or idx >= len(sizes):
            return None
        if not all(isinstance(s, int) for s in sizes):
            return None

        offset = sum(sizes[:idx])
        length = sizes[idx]
        return parent, offset, length

    return None


class FuseSiluMulConfig(TransformConfig):
    """Configuration for the SiLU+Mul fusion transform."""

    backend: str = Field(
        default="flashinfer",
        description=(
            "Backend for fused SiLU+Mul kernel. "
            "'flashinfer' (default) or 'trtllm' (faster, supports fused FP8 quant)."
        ),
    )


@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMul(BaseTransform):
    """Fuse narrow+silu+mul into a single silu_and_mul op after GEMM fusion.

    Detects the pattern:
        gate = narrow(x, -1, 0, size)
        up = narrow(x, -1, size, size)
        hidden = silu(gate) * up

    And replaces it with:
        hidden = silu_and_mul(x)

    When ``backend='trtllm'``, also detects if the sole consumer is an FP8 linear
    and fuses quantization into the kernel (eliminating a separate scaleMatrix pass).

    This runs as a post_load_fusion pass, after GEMM fusion has combined gate+up
    projections.
    """

    config: FuseSiluMulConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseSiluMulConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled:
            return gm, TransformInfo(skipped=True, num_matches=0)

        backend = self.config.backend
        if backend == "trtllm":
            target_op = torch.ops.auto_deploy.trtllm_silu_and_mul.default
        else:
            target_op = torch.ops.auto_deploy.flashinfer_silu_and_mul.default

        graph = gm.graph
        cnt = 0

        for node in list(graph.nodes):
            if not is_op(node, torch.ops.aten.mul.Tensor):
                continue

            result = self._try_fuse_mul(node)
            if result is None:
                continue

            fused_parent, half_size = result

            # Create fused silu_and_mul node
            with graph.inserting_before(node):
                fused_node = graph.call_function(
                    target_op,
                    args=(fused_parent,),
                )
                # Propagate shape metadata
                ref_val = node.meta.get("val")
                if ref_val is not None:
                    fused_node.meta["val"] = torch.empty(
                        ref_val.shape, dtype=ref_val.dtype, device="meta"
                    )

            node.replace_all_uses_with(fused_node)
            cnt += 1

            # Try to fuse FP8 quantization into silu_and_mul (trtllm backend only).
            if backend == "trtllm":
                self._try_fuse_fp8_quant(graph, fused_node)

        if cnt > 0:
            # Clean up dead code (narrow, silu nodes that are no longer used)
            gm.graph.eliminate_dead_code()
            gm.recompile()

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info

    @staticmethod
    def _try_fuse_fp8_quant(graph: torch.fx.Graph, silu_node: Node) -> None:
        """Try to fuse FP8 input quantization from a downstream FP8 linear.

        If silu_node has exactly one user that is an FP8 linear op, extract its
        input_scale argument and fold it into the silu_and_mul call so the Triton
        kernel quantizes the output in-kernel.  The FP8 linear's fast-path
        (``if input.dtype == fp8: skip quant``) then avoids the separate
        ``scaleMatrixPerTensorVec`` kernel entirely.
        """
        users = list(silu_node.users.keys())
        if len(users) != 1:
            return

        user = users[0]
        if user.target not in _FP8_LINEAR_OPS:
            return

        # The FP8 linear's input_scale is arg[3] (input, weight, bias, input_scale, ...)
        if len(user.args) < 4:
            return

        input_scale = user.args[3]
        if input_scale is None or not isinstance(input_scale, Node):
            return

        # input_scale is typically a get_attr for a static buffer (e.g.
        # ``down_proj_input_scale``).  It may be inserted later in the graph
        # than silu_node, so create a fresh get_attr in the right position.
        # If input_scale is anything else (e.g. a dynamic compute), bail out
        # rather than reorder existing nodes — fusion is optional.
        if input_scale.op != "get_attr":
            return
        with graph.inserting_before(silu_node):
            scale_node = graph.create_node("get_attr", input_scale.target)
        scale_node.meta = dict(input_scale.meta)

        # Rewrite: silu_and_mul(x) → silu_and_mul(x, scale=scale_node, out_dtype="float8_e4m3fn")
        silu_node.args = (silu_node.args[0], scale_node, "float8_e4m3fn")

        # Update meta to reflect FP8 output dtype
        ref_val = silu_node.meta.get("val")
        if ref_val is not None:
            silu_node.meta["val"] = torch.empty(
                ref_val.shape, dtype=torch.float8_e4m3fn, device="meta"
            )

        # The FP8 linear now receives FP8 input and needs out_dtype to determine
        # the output dtype (it can no longer infer it from the input dtype).
        # Infer the original input dtype from the silu_and_mul's input tensor.
        silu_input = silu_node.args[0]
        silu_input_val = silu_input.meta.get("val") if isinstance(silu_input, Node) else None
        orig_dtype = silu_input_val.dtype if silu_input_val is not None else torch.bfloat16
        dtype_str = str(orig_dtype).removeprefix("torch.")

        # Set out_dtype on the FP8 linear (arg[5] for trtllm, not present for torch variant)
        user_args = list(user.args)
        if user.target == torch.ops.auto_deploy.trtllm_quant_fp8_linear.default:
            # trtllm_quant_fp8_linear(input, weight, bias, input_scale, weight_scale, out_dtype)
            while len(user_args) < 6:
                user_args.append(None)
            user_args[5] = dtype_str
        user.args = tuple(user_args)

    @staticmethod
    def _try_fuse_mul(mul_node: Node) -> Optional[Tuple[Node, int]]:
        """Try to match the narrow+silu+mul pattern on a mul node.

        Returns (fused_parent, half_size) if the pattern matches, None otherwise.
        """
        left, right = mul_node.args[0], mul_node.args[1]
        if not isinstance(left, Node) or not isinstance(right, Node):
            return None

        # Try both orderings: silu(narrow(x)) * narrow(x)  or  narrow(x) * silu(narrow(x))
        for silu_candidate, up_candidate in [(left, right), (right, left)]:
            result = FuseSiluMul._match_silu_narrow_mul(silu_candidate, up_candidate)
            if result is not None:
                return result

        return None

    @staticmethod
    def _match_silu_narrow_mul(
        silu_candidate: Node, up_candidate: Node
    ) -> Optional[Tuple[Node, int]]:
        """Check if silu_candidate = silu(narrow(x, 0, size)) and up_candidate = narrow(x, size, size).

        Returns (x, size) on match, None otherwise.
        """
        # silu_candidate must be a silu op
        if not is_op(silu_candidate, torch.ops.aten.silu.default):
            return None

        # silu's input must be a narrow
        silu_input = silu_candidate.args[0]
        if not isinstance(silu_input, Node):
            return None

        gate_info = _get_narrow_info(silu_input)
        if gate_info is None:
            return None
        gate_parent, gate_offset, gate_size = gate_info

        # up_candidate must also be a narrow
        up_info = _get_narrow_info(up_candidate)
        if up_info is None:
            return None
        up_parent, up_offset, up_size = up_info

        # Both narrows must come from the same parent
        if gate_parent is not up_parent:
            return None

        # gate must be at offset 0, up must be at offset gate_size, both same size
        if gate_offset != 0:
            return None
        if up_offset != gate_size:
            return None
        if gate_size != up_size:
            return None

        # Only fuse when the two narrows together cover the parent's full last
        # dim (i.e. parent.shape[-1] == 2 * gate_size).  Otherwise the fused
        # silu_and_mul kernel — which always splits its input at half — would
        # operate on a different slice than the two narrows describe.
        parent_val = gate_parent.meta.get("val")
        if parent_val is None:
            return None
        parent_last_dim = parent_val.shape[-1]
        if not isinstance(parent_last_dim, int) or parent_last_dim != 2 * gate_size:
            return None

        return gate_parent, gate_size
