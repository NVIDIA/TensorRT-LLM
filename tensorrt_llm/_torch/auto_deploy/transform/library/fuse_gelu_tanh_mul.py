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

"""Graph transform for fusing GELU-tanh+Mul activation after GEMM fusion.

Targets models using gelu_pytorch_tanh activation (e.g. Gemma4 dense MLP).

After fuse_gemms fuses gate+up projections into a single GEMM, the graph contains:

    fused_out = gemm(x, gate_up_weight)
    gate = narrow(fused_out, -1, 0, intermediate_size)
    up = narrow(fused_out, -1, intermediate_size, intermediate_size)
    hidden = gelu(gate, approximate='tanh') * up

This transform replaces it with a single fused op:

    hidden = gelu_tanh_and_mul(fused_out)

The fused kernel avoids materializing narrow views and uses FlashInfer's optimized
gelu_tanh_and_mul kernel. The transform is a no-op when FlashInfer is not available.
"""

import operator
from contextlib import nullcontext
from typing import Optional, Tuple, Type

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx import GraphModule, Node

from ...custom_ops.linear.gelu_tanh_mul import HAS_FUSED_GELU_TANH_AND_MUL
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import lift_to_meta, placeholders_on_meta, run_shape_prop
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_HALF_SIZE = 256


# ---- Variant 1: narrow+gelu_tanh+mul (pattern matcher) --------------------------------


def _gelu_tanh_mul_pattern(x: torch.Tensor) -> torch.Tensor:
    """Search pattern: narrow+gelu_tanh+mul after non-quantized GEMM fusion."""
    gate = torch.narrow(x, -1, 0, _HALF_SIZE)
    up = torch.narrow(x, -1, _HALF_SIZE, _HALF_SIZE)
    return torch.ops.aten.mul.Tensor(torch.nn.functional.gelu(gate, approximate="tanh"), up)


def _gelu_tanh_mul_replacement(x: torch.Tensor) -> torch.Tensor:
    """Replacement: fused gelu_tanh_and_mul custom op."""
    return torch.ops.auto_deploy.gelu_tanh_and_mul.default(x)


def _symbolic_trace_fn(fn, _args):
    """Trace preserving torch.narrow."""
    return torch.fx.symbolic_trace(fn)


def _check_narrow_constraints(match: Match) -> bool:
    """Verify the two narrow ops form a valid gate/up split on dim -1."""
    narrow_nodes = [n for n in match.nodes if n.op == "call_function" and n.target == torch.narrow]
    if len(narrow_nodes) != 2:
        return False

    for n in narrow_nodes:
        if n.args[1] != -1:
            return False

    narrow_nodes.sort(key=lambda n: n.args[2])
    gate_narrow, up_narrow = narrow_nodes

    gate_offset, gate_length = gate_narrow.args[2], gate_narrow.args[3]
    up_offset, up_length = up_narrow.args[2], up_narrow.args[3]

    return gate_offset == 0 and up_offset == gate_length and gate_length == up_length


# ---- Variant 2: getitem+gelu_tanh+mul (graph walk) ------------------------------------


def _match_getitem_gelu_tanh_mul(mul_node: Node) -> Optional[Node]:
    """Match gelu_tanh(getitem(split, 0)) * getitem(split, 1) and return the split input."""
    left, right = mul_node.args[0], mul_node.args[1]
    if not isinstance(left, Node) or not isinstance(right, Node):
        return None

    for gelu_candidate, up_candidate in [(left, right), (right, left)]:
        # gelu_candidate must be aten.gelu with approximate='tanh'
        if not is_op(gelu_candidate, torch.ops.aten.gelu.default):
            continue
        if gelu_candidate.kwargs.get("approximate") != "tanh":
            continue

        gelu_input = gelu_candidate.args[0]
        if not isinstance(gelu_input, Node):
            continue

        if gelu_input.op != "call_function" or gelu_input.target != operator.getitem:
            continue
        if up_candidate.op != "call_function" or up_candidate.target != operator.getitem:
            continue

        gate_parent, gate_idx = gelu_input.args[0], gelu_input.args[1]
        up_parent, up_idx = up_candidate.args[0], up_candidate.args[1]

        if gate_parent is not up_parent:
            continue
        if not isinstance(gate_parent, Node):
            continue

        if gate_idx != 0 or up_idx != 1:
            continue

        if gate_parent.op != "call_function" or len(gate_parent.args) < 1:
            continue

        fused_linear_out = gate_parent.args[0]
        if not isinstance(fused_linear_out, Node):
            continue

        return fused_linear_out

    return None


def _fuse_getitem_gelu_tanh_mul(gm: GraphModule) -> int:
    """Fuse getitem+gelu_tanh+mul patterns into gelu_tanh_and_mul."""
    graph = gm.graph
    cnt = 0

    for node in list(graph.nodes):
        if not is_op(node, torch.ops.aten.mul.Tensor):
            continue

        fused_linear_out = _match_getitem_gelu_tanh_mul(node)
        if fused_linear_out is None:
            continue

        with graph.inserting_before(node):
            fused_node = graph.call_function(
                torch.ops.auto_deploy.gelu_tanh_and_mul.default,
                args=(fused_linear_out,),
            )
            ref_val = node.meta.get("val")
            if ref_val is not None:
                fused_node.meta["val"] = torch.empty(
                    ref_val.shape, dtype=ref_val.dtype, device="meta"
                )

        node.replace_all_uses_with(fused_node)
        cnt += 1

    if cnt > 0:
        graph.eliminate_dead_code()
        gm.recompile()

    return cnt


# ---- Transform class -----------------------------------------------------------------


@TransformRegistry.register("fuse_gelu_tanh_mul")
class FuseGeluTanhMul(BaseTransform):
    """Fuse gelu_tanh+mul into a single gelu_tanh_and_mul op after GEMM fusion.

    Handles two variants:
    - Variant 1 (narrow split): matched via ADPatternMatcherPass
    - Variant 2 (getitem split): matched via direct graph walk

    Targets gelu_pytorch_tanh activation (Gemma4 dense MLP).
    Runs as a post_load_fusion pass, after GEMM fusion has combined gate+up projections.
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
        if not self.config.enabled or not HAS_FUSED_GELU_TANH_AND_MUL:
            return gm, TransformInfo(skipped=True, num_matches=0)

        cnt = 0

        # Variant 1: narrow+gelu_tanh+mul (pattern matcher)
        has_narrow_ops = any(
            n.op == "call_function" and n.target == torch.narrow for n in gm.graph.nodes
        )
        if has_narrow_ops:
            with lift_to_meta(gm) if placeholders_on_meta(gm) else nullcontext():
                run_shape_prop(gm)

            patterns = ADPatternMatcherPass()
            register_ad_pattern(
                search_fn=_gelu_tanh_mul_pattern,
                replace_fn=_gelu_tanh_mul_replacement,
                patterns=patterns,
                dummy_args=[torch.randn(2, _HALF_SIZE * 2, device="meta")],
                trace_fn=_symbolic_trace_fn,
                op_ignore_types={torch.narrow: (int,)},
                extra_check=_check_narrow_constraints,
            )
            cnt += patterns.apply(gm.graph)
            if cnt > 0:
                gm.recompile()

        # Variant 2: getitem+gelu_tanh+mul (direct graph walk for opaque split closures)
        cnt += _fuse_getitem_gelu_tanh_mul(gm)

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )
