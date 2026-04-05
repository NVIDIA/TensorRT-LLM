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
single GEMM, the graph contains one of two splitting patterns:

Variant 1 (non-quantized, torch.narrow):

    fused_out = gemm(x, gate_up_weight)
    gate = narrow(fused_out, -1, 0, intermediate_size)
    up = narrow(fused_out, -1, intermediate_size, intermediate_size)
    hidden = silu(gate) * up

Variant 2 (quantized, split+getitem):

    fused_out = quant_gemm(x, fused_weight, ...)
    parts = split_output(fused_out)
    gate = getitem(parts, 0)
    up = getitem(parts, 1)
    hidden = silu(gate) * up

This transform replaces both patterns with a single fused op:

    hidden = silu_and_mul(fused_out)

The fused kernel avoids materializing narrow views and uses FlashInfer's optimized
silu_and_mul kernel. The transform is a no-op when FlashInfer is not available.
"""

import operator
from contextlib import nullcontext
from typing import Optional, Tuple, Type

import torch
from torch._inductor.pattern_matcher import Match
from torch.fx import GraphModule, Node

from ...custom_ops.linear.silu_mul import HAS_FUSED_SILU_AND_MUL
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface

# It is important to import ADPatternMatcherPass from pattern_matcher.py, not from
# torch._inductor.pattern_matcher
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

# Dummy half-size for tracing the pattern (actual values are ignored via op_ignore_types).
_HALF_SIZE = 256


# ---- Variant 1: narrow+silu+mul (pattern matcher) -----------------------------------


def _silu_mul_pattern(x: torch.Tensor) -> torch.Tensor:
    """Search pattern: narrow+silu+mul after non-quantized GEMM fusion.

    Uses explicit aten ops for silu/mul to match the export-level graph, and torch.narrow
    to match the nodes injected by GEMM fusion.
    """
    gate = torch.narrow(x, -1, 0, _HALF_SIZE)
    up = torch.narrow(x, -1, _HALF_SIZE, _HALF_SIZE)
    return torch.ops.aten.mul.Tensor(torch.ops.aten.silu.default(gate), up)


def _silu_mul_replacement(x: torch.Tensor) -> torch.Tensor:
    """Replacement: fused silu_and_mul custom op."""
    return torch.ops.auto_deploy.silu_and_mul.default(x)


def _symbolic_trace_fn(fn, _args):
    """Trace preserving torch.narrow (torch_export_to_gm would decompose it to aten.slice)."""
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


# ---- Variant 2: getitem+silu+mul (graph walk) ---------------------------------------
# The quantized GEMM fusion path splits via a closure (split_output) + getitem.
# The getitem+silu+mul structure IS matchable by the pattern matcher, but the
# replacement needs the pre-split tensor (input to the split call), which is not
# capturable as a pattern input: CallFunction requires a specific target, so the
# split call (whose target varies per fusion site) cannot be included in the pattern.
# We use direct graph walking to access split_node.args[0] for the replacement.


def _match_getitem_silu_mul(mul_node: Node) -> Optional[Node]:
    """Match silu(getitem(split, 0)) * getitem(split, 1) and return the split input.

    Returns the fused linear output (pre-split tensor) if the pattern matches, None otherwise.
    """
    left, right = mul_node.args[0], mul_node.args[1]
    if not isinstance(left, Node) or not isinstance(right, Node):
        return None

    for silu_candidate, up_candidate in [(left, right), (right, left)]:
        if not is_op(silu_candidate, torch.ops.aten.silu.default):
            continue

        silu_input = silu_candidate.args[0]
        if not isinstance(silu_input, Node):
            continue

        # Both silu_input (gate) and up_candidate must be getitem from the same parent
        if silu_input.op != "call_function" or silu_input.target != operator.getitem:
            continue
        if up_candidate.op != "call_function" or up_candidate.target != operator.getitem:
            continue

        gate_parent, gate_idx = silu_input.args[0], silu_input.args[1]
        up_parent, up_idx = up_candidate.args[0], up_candidate.args[1]

        if gate_parent is not up_parent:
            continue
        if not isinstance(gate_parent, Node):
            continue

        # gate must be index 0, up must be index 1
        if gate_idx != 0 or up_idx != 1:
            continue

        # The parent is the split_output call; its first arg is the fused linear output
        if gate_parent.op != "call_function" or len(gate_parent.args) < 1:
            continue

        fused_linear_out = gate_parent.args[0]
        if not isinstance(fused_linear_out, Node):
            continue

        return fused_linear_out

    return None


def _fuse_getitem_silu_mul(gm: GraphModule) -> int:
    """Fuse getitem+silu+mul patterns from quantized GEMM fusion into silu_and_mul."""
    graph = gm.graph
    cnt = 0

    for node in list(graph.nodes):
        if not is_op(node, torch.ops.aten.mul.Tensor):
            continue

        fused_linear_out = _match_getitem_silu_mul(node)
        if fused_linear_out is None:
            continue

        with graph.inserting_before(node):
            fused_node = graph.call_function(
                torch.ops.auto_deploy.silu_and_mul.default,
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


# ---- Transform class ----------------------------------------------------------------


@TransformRegistry.register("fuse_silu_mul")
class FuseSiluMul(BaseTransform):
    """Fuse silu+mul into a single silu_and_mul op after GEMM fusion.

    Handles two variants:
    - Variant 1 (narrow split): matched via ADPatternMatcherPass
    - Variant 2 (getitem split): matched via direct graph walk (opaque split closures)

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
        if not self.config.enabled or not HAS_FUSED_SILU_AND_MUL:
            return gm, TransformInfo(skipped=True, num_matches=0)

        cnt = 0

        # Variant 1: narrow+silu+mul (pattern matcher, requires shape metadata).
        # Only run if torch.narrow ops exist — avoids expensive run_shape_prop otherwise.
        has_narrow_ops = any(
            n.op == "call_function" and n.target == torch.narrow for n in gm.graph.nodes
        )
        if has_narrow_ops:
            with lift_to_meta(gm) if placeholders_on_meta(gm) else nullcontext():
                run_shape_prop(gm)

            patterns = ADPatternMatcherPass()
            register_ad_pattern(
                search_fn=_silu_mul_pattern,
                replace_fn=_silu_mul_replacement,
                patterns=patterns,
                dummy_args=[torch.randn(2, _HALF_SIZE * 2, device="meta")],
                trace_fn=_symbolic_trace_fn,
                op_ignore_types={torch.narrow: (int,)},
                extra_check=_check_narrow_constraints,
            )
            cnt += patterns.apply(gm.graph)
            if cnt > 0:
                gm.recompile()

        # Variant 2: getitem+silu+mul (direct graph walk for opaque split closures)
        cnt += _fuse_getitem_silu_mul(gm)

        return gm, TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )
