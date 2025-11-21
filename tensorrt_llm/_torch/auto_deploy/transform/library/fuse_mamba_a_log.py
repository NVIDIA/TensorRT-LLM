"""Transform to fuse A_log into A for Mamba/NemotronH models."""

import operator
from typing import Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _get_attr_by_name(obj, name):
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _set_attr_by_name(obj, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


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
        num_matches = 0

        # Candidates for operations
        exp_ops = {torch.exp, torch.ops.aten.exp.default, "exp"}
        neg_ops = {operator.neg, torch.neg, torch.ops.aten.neg.default, "neg"}

        # We search bottom-up starting from A_log parameters to be more robust
        # pattern: A_log -> [optional cast] -> exp -> neg

        # Snapshot nodes to avoid modification issues during iteration
        nodes = list(gm.graph.nodes)

        for node in nodes:
            if node.op != "get_attr":
                continue

            if not node.target.endswith("A_log"):
                continue
            # Found an A_log node. Check its usage.
            users = list(node.users.keys())
            for user in users:
                # 1. Check for optional Cast
                current_node = user

                # Skip cast/to nodes
                exp_node = None

                # Walk forward looking for exp
                cursor = current_node
                for _ in range(3):  # Max depth for casts
                    if (cursor.op == "call_function" and cursor.target in exp_ops) or (
                        cursor.op == "call_method" and cursor.target == "exp"
                    ):
                        exp_node = cursor
                        break

                    if len(cursor.users) != 1:
                        break
                    cursor = list(cursor.users.keys())[0]

                if not exp_node:
                    continue

                # 2. Check for Neg
                if len(exp_node.users) != 1:
                    continue

                neg_node = list(exp_node.users.keys())[0]
                is_neg = (neg_node.op == "call_function" and neg_node.target in neg_ops) or (
                    neg_node.op == "call_method" and neg_node.target == "neg"
                )

                if not is_neg:
                    continue
                # Found the pattern: node -> ... -> exp_node -> neg_node
                num_matches += 1

                # Perform Fusion
                param_name = node.target
                try:
                    a_log = _get_attr_by_name(gm, param_name)
                except AttributeError:
                    ad_logger.warning(f"Could not find attribute {param_name} in gm.")
                    continue

                # Compute A_fused
                with torch.no_grad():
                    # Replicate the logic: -exp(a_log.float())
                    a_fused = -torch.exp(a_log.float())

                new_param_name = param_name.replace("A_log", "A_fused")

                # Check if we already created this param (if A_log used multiple times)
                try:
                    _get_attr_by_name(gm, new_param_name)
                except AttributeError:
                    _set_attr_by_name(
                        gm, new_param_name, nn.Parameter(a_fused, requires_grad=False)
                    )

                # Replace usage
                with gm.graph.inserting_before(neg_node):
                    new_node = gm.graph.create_node("get_attr", new_param_name)

                neg_node.replace_all_uses_with(new_node)

        if num_matches > 0:
            gm.graph.eliminate_dead_code()

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=True,
        )
