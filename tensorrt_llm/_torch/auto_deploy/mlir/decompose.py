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

"""Decomposition pass: expand high-level ops into sequences of primitives.

Uses xDSL's ``PatternRewriter`` infrastructure for robust IR mutation.
Each decomposition rule is registered via a decorator and applied greedily
across the module.
"""

from typing import Callable, Dict, List, Tuple

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    InsertPoint,
    PatternRewriter,
    RewritePattern,
)
from xdsl.pattern_rewriter import PatternRewriteWalker as _RewriteDriver

# Registry: MLIR op class -> decomposition function
_DECOMP_REGISTRY: Dict[type, Callable] = {}


def decomposition(op_cls):
    """Decorator to register a decomposition rule for an MLIR op class.

    The decorated function receives an op instance and returns a tuple of
    ``(new_ops, result_val)`` where ``new_ops`` is a list of new operations
    to insert before the original op and ``result_val`` is the SSA value
    that replaces the original op's output.
    """

    def decorator(
        fn: Callable[[Operation], Tuple[List[Operation], SSAValue]],
    ) -> Callable:
        _DECOMP_REGISTRY[op_cls] = fn
        return fn

    return decorator


class _DecompPattern(RewritePattern):
    """Generic rewrite pattern that dispatches to registered decomposition rules."""

    def __init__(self):
        super().__init__()
        self.num_matches = 0

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        rule = _DECOMP_REGISTRY.get(type(op))
        if rule is None:
            return

        result = rule(op)

        # A decomposition may return None to skip (e.g. unsupported group_size).
        if result is None:
            return

        new_ops, result_val = result

        # Insert all new ops before the original
        for new_op in new_ops:
            rewriter.insert_op(new_op, InsertPoint.before(op))

        # Replace original op's output with the decomposed result
        op.output.replace_by(result_val)

        # Erase the original op
        rewriter.erase_op(op)
        self.num_matches += 1


def run_decomposition(mlir_module: ModuleOp) -> int:
    """Decompose all decomposable ops in the module.

    Returns the number of ops decomposed.
    """
    # Import rules to trigger registration via decorators
    from . import decompose_rules  # noqa: F401

    pattern = _DecompPattern()
    driver = _RewriteDriver(GreedyRewritePatternApplier([pattern]))
    driver.rewrite_module(mlir_module)
    return pattern.num_matches
