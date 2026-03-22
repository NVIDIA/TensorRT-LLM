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

"""MLIR fusion pattern: ``ad.add`` + ``ad.rmsnorm`` → ``ad.fused_add_rmsnorm``.

Supports two input patterns:
    Pattern 1 (no cast):  ad.add → ad.rmsnorm
    Pattern 2 (with cast): ad.add → ad.to_dtype → ad.rmsnorm

Both are fused into ``ad.fused_add_rmsnorm`` which returns ``(norm_result, add_result)``.
The MLIR SSA form naturally handles multi-user nodes — ``replace_all_uses_with`` rewires
all consumers automatically.
"""

from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    InsertPoint,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.pattern_rewriter import PatternRewriteWalker as _RewriteDriver

from ..dialect import AdAdd, AdFusedAddRMSNorm, AdRMSNorm, AdToDtype


class FuseAddRMSNormPattern(RewritePattern):
    """Fuse ``ad.add`` (+ optional ``ad.to_dtype``) + ``ad.rmsnorm`` into ``ad.fused_add_rmsnorm``.

    Matches on ``ad.rmsnorm`` ops and walks backward to find the add pattern.
    """

    def __init__(self):
        super().__init__()
        self.num_matches = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AdRMSNorm, rewriter: PatternRewriter) -> None:
        input_val = op.input
        input_op = input_val.owner

        cast_op = None

        # Check for optional ad.to_dtype between add and rmsnorm
        if isinstance(input_op, AdToDtype):
            cast_op = input_op
            input_val = cast_op.input
            input_op = input_val.owner

        # The (possibly unwrapped) input must be an ad.add
        if not isinstance(input_op, AdAdd):
            return

        add_op = input_op

        # Build the fused op
        fused = AdFusedAddRMSNorm.build(
            operands=[add_op.lhs, add_op.rhs, op.weight],
            attributes={"eps": op.eps},
            result_types=[op.output.type, add_op.output.type],
        )

        # Insert fused op before the rmsnorm
        rewriter.insert_op(fused, InsertPoint.before(op))

        # Replace rmsnorm output with fused norm_result
        op.output.replace_by(fused.norm_result)

        # Replace add output with fused add_result (rewires all consumers)
        add_op.output.replace_by(fused.add_result)

        # Erase the matched rmsnorm
        rewriter.erase_op(op)

        # Handle cast if present
        if cast_op is not None:
            if not cast_op.output.uses:
                # Cast has no remaining users — safe to erase
                rewriter.erase_op(cast_op)
            else:
                # Cast still has users (e.g., graph output references it).
                # Replace it with a new cast after the fused op to maintain
                # valid SSA dominance (the old cast is before the fused op).
                new_cast = AdToDtype.build(
                    operands=[fused.add_result],
                    attributes={"target_dtype": cast_op.target_dtype},
                    result_types=[cast_op.output.type],
                )
                rewriter.insert_op(new_cast, InsertPoint.after(fused))
                cast_op.output.replace_by(new_cast.output)
                rewriter.erase_op(cast_op)

        # Erase the original add (all its uses have been replaced)
        rewriter.erase_op(add_op)

        self.num_matches += 1


def run_fusion_patterns(mlir_module: ModuleOp) -> int:
    """Run all fusion patterns on the given MLIR module.

    Returns the number of pattern matches applied.
    """
    pattern = FuseAddRMSNormPattern()
    driver = _RewriteDriver(GreedyRewritePatternApplier([pattern]))
    driver.rewrite_module(mlir_module)
    return pattern.num_matches
