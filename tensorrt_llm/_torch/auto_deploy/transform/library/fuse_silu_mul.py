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

The fused kernel avoids materializing narrow views and uses FlashInfer's optimized
silu_and_mul kernel when available.
"""

from typing import Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

# Import to ensure the custom op is registered
from ...custom_ops.linear.silu_mul import silu_and_mul  # noqa: F401
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


def _get_narrow_info(node: Node) -> Optional[Tuple[Node, int, int]]:
    """Extract (parent, offset, length) from a torch.narrow call on dim=-1.

    Returns None if the node is not a narrow on the last dimension.
    """
    if not is_op(node, torch.narrow):
        return None

    # torch.narrow(input, dim, start, length)
    if len(node.args) < 4:
        return None

    parent = node.args[0]
    dim = node.args[1]
    offset = node.args[2]
    length = node.args[3]

    if not isinstance(parent, Node):
        return None
    if dim != -1:
        return None
    if not isinstance(offset, int) or not isinstance(length, int):
        return None

    return parent, offset, length


@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMul(BaseTransform):
    """Fuse narrow+silu+mul into a single silu_and_mul op after GEMM fusion.

    Detects the pattern:
        gate = narrow(x, -1, 0, size)
        up = narrow(x, -1, size, size)
        hidden = silu(gate) * up

    And replaces it with:
        hidden = silu_and_mul(x)

    This runs as a post_load_fusion pass, after GEMM fusion has combined gate+up
    projections.
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
        if not self.config.enabled:
            return gm, TransformInfo(skipped=True, num_matches=0)

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
                    torch.ops.auto_deploy.silu_and_mul.default,
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

        return gate_parent, gate_size
