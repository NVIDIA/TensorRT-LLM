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

"""Walker-driven elementwise fusion pass.

Mirrors :func:`..decompose.run_decomposition`: discovery computes the maximal
fusible subgraphs (greedy union-find + codegen-constraint partitioning — see
:mod:`.subgraph_discovery`), and an xDSL ``RewritePattern`` driven by
``PatternRewriteWalker`` performs the per-subgraph codegen + replacement.  This
keeps the top-level pipeline consistent with decomposition (both go through
xDSL's pass infrastructure) while leaving the domain-specific discovery
algorithm — placement-conflict detection, the 64-input split, and hash-stable
topological ordering — untouched.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from ..codegen.kernel_cache import KernelCache
from ..codegen.triton_emitter import generate_kernel_from_subgraph
from .subgraph_discovery import FusibleSubgraph, discover_fusible_subgraphs
from .subgraph_replace import replace_subgraph_with_fused_op

logger = logging.getLogger(__name__)


@dataclass
class FusionStats:
    """Summary of a :func:`run_fusion` invocation."""

    num_subgraphs: int = 0
    num_replaced: int = 0
    num_skipped_low_rank: int = 0


def _max_input_rank(sg: FusibleSubgraph) -> int:
    return max(
        (len(inp.type.get_shape()) for inp in sg.inputs if isinstance(inp.type, TensorType)),
        default=0,
    )


def _min_output_rank(sg: FusibleSubgraph) -> int:
    return min(
        (len(out.type.get_shape()) for out in sg.outputs if isinstance(out.type, TensorType)),
        default=0,
    )


class _FuseSubgraphPattern(RewritePattern):
    """Replace each pre-discovered subgraph with a fused op, walker-driven.

    The pattern matches on each subgraph's *anchor* — its topologically-first op
    (``sg.ops[0]``).  Because discovered subgraphs are op-disjoint, anchors are
    unique and survive until their own subgraph is fused.  A forward walk visits
    partition anchors in dependency order, so for subgraphs split across the
    64-input limit (where a later partition consumes an earlier one's output),
    ``refresh_inputs()`` picks up the already-redirected operands — matching the
    behavior of the previous ahead-of-time discover-then-replace loop.
    """

    def __init__(
        self,
        subgraphs: List[FusibleSubgraph],
        metadata: Dict,
        log_warning: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        # Anchor op -> subgraph. Discovered subgraphs are op-disjoint, so each
        # anchor maps to exactly one subgraph.
        self._by_anchor: Dict[Operation, FusibleSubgraph] = {
            sg.ops[0]: sg for sg in subgraphs if sg.ops
        }
        self._metadata = metadata
        self._log_warning = log_warning
        self.num_replaced = 0
        self.num_skipped_low_rank = 0

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        sg = self._by_anchor.get(op)
        if sg is None:
            return

        # Skip subgraphs where all inputs are 1D or lower — these are pure
        # weight-space ops (e.g., weight + 1.0) that don't benefit from fusion
        # and the row-based Triton kernel can't handle them correctly.
        if _max_input_rank(sg) < 2 or _min_output_rank(sg) < 2:
            self.num_skipped_low_rank += 1
            return

        try:
            # Refresh inputs: earlier subgraph replacements may have redirected
            # operands via SSAValue.replace_by(), making the inputs list computed
            # at discovery time stale.
            sg.refresh_inputs()
            kernel_fn = generate_kernel_from_subgraph(sg)
            if kernel_fn is None:
                return
            sg_hash = KernelCache.hash_subgraph(sg)
            replace_subgraph_with_fused_op(
                sg, kernel_fn, sg_hash, self._metadata, rewriter=rewriter
            )
            self.num_replaced += 1
        except (ValueError, NotImplementedError) as e:
            if self._log_warning is not None:
                self._log_warning(f"Skipping subgraph (unsupported pattern): {e}")


def run_fusion(
    mlir_module: ModuleOp,
    metadata: Dict,
    log_warning: Optional[Callable[[str], None]] = None,
) -> FusionStats:
    """Discover fusible subgraphs and replace each with a generated fused op.

    Args:
        mlir_module: The module to fuse in place.
        metadata: The ``FXToMLIRConverter`` metadata side-table; one entry is
            added per fused op so MLIR-to-FX can reconstruct the kernel call.
        log_warning: Optional callback for per-subgraph skip diagnostics.

    Returns:
        A :class:`FusionStats` with discovery/replacement counts.
    """
    subgraphs = discover_fusible_subgraphs(mlir_module)
    if not subgraphs:
        return FusionStats()

    pattern = _FuseSubgraphPattern(subgraphs, metadata, log_warning)
    # apply_recursively=False: a single forward pass; each anchor is matched once
    # and the inserted fused op is never re-matched.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([pattern]),
        apply_recursively=False,
    )
    walker.rewrite_module(mlir_module)

    return FusionStats(
        num_subgraphs=len(subgraphs),
        num_replaced=pattern.num_replaced,
        num_skipped_low_rank=pattern.num_skipped_low_rank,
    )
