# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transform to fuse A_log into A for Mamba/NemotronH models."""

import operator
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch._inductor.pattern_matcher import (
    CallFunction,
    CallMethod,
    KeywordArg,
    Match,
    register_graph_pattern,
)
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import del_attr_by_name, get_attr_by_name, set_attr_by_name
from ...utils.logger import ad_logger
from ...utils.pattern_matcher import ADPatternMatcherPass
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

_PATTERN_INPUT_NAME = "a_log_like"


def _find_a_log_attr(node: Optional[Node]) -> Optional[Node]:
    """Walk backwards through up to `max_backtrack_steps` unary nodes to find the A_log attribute."""
    current = node
    max_backtrack_steps = 4
    for _ in range(max_backtrack_steps):
        if current is None:
            return None
        if current.op == "get_attr":
            return current
        inputs = list(current.all_input_nodes)
        if len(inputs) != 1:
            return None
        current = inputs[0]
    return current if current and current.op == "get_attr" else None


def _ensure_a_fused_param(gm: GraphModule, param_name: str) -> Optional[str]:
    """Create (if missing) the fused parameter that replaces A_log usage."""
    if not param_name.endswith("A_log"):
        return None

    new_param_name = param_name.replace("A_log", "A_fused")
    try:
        get_attr_by_name(gm, new_param_name)
        return new_param_name
    except AttributeError:
        pass

    try:
        a_log = get_attr_by_name(gm, param_name)
    except AttributeError:
        ad_logger.warning(f"Could not find attribute {param_name} in gm.")
        return None

    with torch.no_grad():
        a_fused = -torch.exp(a_log.float())

    set_attr_by_name(
        gm,
        new_param_name,
        nn.Parameter(a_fused, requires_grad=False),
    )
    return new_param_name


def _remove_unused_a_log_params(gm: GraphModule) -> bool:
    """Remove detached A_log parameters after fusion."""

    def _is_a_log_node(node: Node) -> bool:
        return (
            node.op == "get_attr" and isinstance(node.target, str) and node.target.endswith("A_log")
        )

    used_a_log_targets = {str(node.target) for node in gm.graph.nodes if _is_a_log_node(node)}
    removed = False

    def _maybe_remove(name: str) -> None:
        nonlocal removed
        if not name.endswith("A_log") or name in used_a_log_targets:
            return
        try:
            del_attr_by_name(gm, name)
            removed = True
        except AttributeError:
            ad_logger.warning(f"Failed to delete unused parameter {name} from GraphModule.")

    for name, _ in list(gm.named_parameters()):
        _maybe_remove(name)
    for name, _ in list(gm.named_buffers()):
        _maybe_remove(name)

    return removed


def _has_a_log_attr(match: Match) -> bool:
    node = match.kwargs.get(_PATTERN_INPUT_NAME)
    attr_node = _find_a_log_attr(node if isinstance(node, Node) else None)
    return bool(
        attr_node and isinstance(attr_node.target, str) and attr_node.target.endswith("A_log")
    )


def _fuse_a_log_handler(match: Match, a_log_like: Node) -> None:
    graph = match.graph
    gm = graph.owning_module
    if gm is None:
        ad_logger.warning("Pattern matched but owning GraphModule is missing.")
        return

    neg_node = match.output_node()
    exp_node = neg_node.args[0] if neg_node.args else None
    if not isinstance(exp_node, Node):
        ad_logger.warning("Unexpected exp node structure; skipping fusion.")
        return

    attr_node = _find_a_log_attr(a_log_like)
    if attr_node is None or not isinstance(attr_node.target, str):
        ad_logger.warning("Could not trace back to A_log attribute; skipping fusion.")
        return

    fused_name = _ensure_a_fused_param(gm, attr_node.target)
    if fused_name is None:
        return

    new_attr_node = graph.get_attr(fused_name)
    neg_node.replace_all_uses_with(new_attr_node)
    match.erase_nodes()


def _register_fuse_a_log_patterns(patterns: ADPatternMatcherPass) -> None:
    """Register neg(exp(.)) patterns that should be folded into fused constants."""

    def _register(pattern):
        register_graph_pattern(
            pattern,
            extra_check=_has_a_log_attr,
            pass_dict=patterns,
        )(_fuse_a_log_handler)

    exp_call_function_targets = (
        torch.exp,
        torch.ops.aten.exp.default,
    )
    neg_call_function_targets = (
        operator.neg,
        torch.neg,
        torch.ops.aten.neg.default,
    )
    neg_call_method_targets = ("neg",)
    for exp_target in exp_call_function_targets:
        exp_expr = CallFunction(exp_target, KeywordArg(_PATTERN_INPUT_NAME))
        for neg_target in neg_call_function_targets:
            _register(CallFunction(neg_target, exp_expr))
        for neg_target in neg_call_method_targets:
            _register(CallMethod(neg_target, exp_expr))

    exp_call_method_targets = ("exp",)
    for exp_target in exp_call_method_targets:
        exp_expr = CallMethod(exp_target, KeywordArg(_PATTERN_INPUT_NAME))
        for neg_target in neg_call_function_targets:
            _register(CallFunction(neg_target, exp_expr))
        for neg_target in neg_call_method_targets:
            _register(CallMethod(neg_target, exp_expr))


@TransformRegistry.register("fuse_mamba_a_log")
class FuseMambaALog(BaseTransform):
    """Fuse A_log parameter into A constant/parameter.

    Replaces:
        A = -torch.exp(self.A_log.float())
    With:
        A = self.A_fused
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        patterns = ADPatternMatcherPass()
        _register_fuse_a_log_patterns(patterns)
        num_matches = patterns.apply(gm.graph)

        if num_matches > 0:
            gm.graph.eliminate_dead_code()
            _remove_unused_a_log_params(gm)

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
