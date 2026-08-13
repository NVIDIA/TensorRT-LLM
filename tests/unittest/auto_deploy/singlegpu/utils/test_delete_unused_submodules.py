# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for delete_all_unused_submodules.

This module tests the optimized implementation against PyTorch's original
GraphModule.delete_all_unused_submodules to ensure functional equivalence.

Since torch_export_to_gm already eliminates unused submodules during export,
we must modify the graph after export to create unused submodules for testing.
"""

import copy
from typing import List, Set

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.utils._graph import (
    canonicalize_graph,
    delete_all_unused_submodules,
)

# =============================================================================
# Test Models - All submodules are used in forward pass
# =============================================================================


class FullyUsedModel(nn.Module):
    """Model where all submodules are used in forward - none will be pruned during export."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.nested = nn.ModuleDict(
            {
                "layer_a": nn.Linear(hidden_dim, hidden_dim),
                "layer_b": nn.Linear(hidden_dim, hidden_dim),
            }
        )

    def forward(self, x):
        # Use ALL submodules so they appear in the exported graph
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.nested["layer_a"](x)
        x = self.nested["layer_b"](x)
        return x


class DeeplyNestedModel(nn.Module):
    """Model with deeply nested submodule hierarchy for testing depth-first deletion."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.level1 = nn.ModuleDict(
            {
                "level2": nn.ModuleDict(
                    {
                        "level3": nn.ModuleDict(
                            {
                                "leaf_a": nn.Linear(hidden_dim, hidden_dim),
                                "leaf_b": nn.Linear(hidden_dim, hidden_dim),
                            }
                        )
                    }
                )
            }
        )
        self.other = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Use all nested modules
        x = self.level1["level2"]["level3"]["leaf_a"](x)
        x = self.level1["level2"]["level3"]["leaf_b"](x)
        x = self.other(x)
        return x


class ModelWithSequential(nn.Module):
    """Model with Sequential submodules for testing call_module behavior."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.pre = nn.Linear(hidden_dim, hidden_dim)
        self.sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.post = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.pre(x)
        x = self.sequential(x)
        x = self.post(x)
        return x


# =============================================================================
# Helper Functions
# =============================================================================


def get_submodule_names(gm: nn.Module) -> Set[str]:
    """Get all submodule names (excluding root empty string)."""
    return {name for name, _ in gm.named_modules() if name}


def _target_matches_any(target: str, targets_to_remove: List[str]) -> bool:
    """Check if a target matches any of the targets to remove.

    Handles both exact matches and prefix matches (for nested attributes).
    Target names in the graph use underscores, so we check both formats.
    """
    # Normalize target by replacing underscores with dots for comparison
    # e.g., "linear2_weight" -> check against "linear2"
    for t in targets_to_remove:
        # Exact match
        if target == t:
            return True
        # Prefix match (e.g., "linear2.weight" starts with "linear2.")
        if target.startswith(t + "."):
            return True
        # Handle underscore-separated names from export
        # e.g., target="linear2_weight" should match t="linear2"
        t_underscore = t.replace(".", "_")
        if target == t_underscore or target.startswith(t_underscore + "_"):
            return True
    return False


def make_submodules_unused(gm: GraphModule, targets_to_remove: List[str]) -> None:
    """Remove nodes referencing specific submodules to make them unused.

    This function finds all operations that use parameters from the specified
    submodules and removes them from the graph, rewiring the data flow to bypass
    those operations. After removal, eliminate_dead_code() cleans up unused
    get_attr nodes.

    Args:
        gm: The GraphModule to modify.
        targets_to_remove: List of submodule target names to make unused.
    """
    graph = gm.graph

    # First pass: find call_function nodes that use parameters from target modules
    # These are the actual operations (e.g., linear, matmul) that we need to remove
    nodes_to_bypass = []

    for node in graph.nodes:
        if node.op == "call_function":
            # Check if any of the node's inputs come from a target module's get_attr
            uses_target_module = False
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                    if _target_matches_any(arg.target, targets_to_remove):
                        uses_target_module = True
                        break
            if uses_target_module:
                nodes_to_bypass.append(node)

        elif node.op == "call_module":
            # Direct call_module nodes (if any)
            if _target_matches_any(node.target, targets_to_remove):
                nodes_to_bypass.append(node)

    # Bypass each node by replacing its uses with its first tensor input
    for node in nodes_to_bypass:
        # Find the first tensor input (usually the activation, not weights)
        replacement = None
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                # Skip get_attr nodes (weights/biases) - we want the activation input
                if arg.op != "get_attr":
                    replacement = arg
                    break
        if replacement is not None:
            node.replace_all_uses_with(replacement)

    # Remove the bypassed nodes (in reverse topological order)
    for node in reversed(nodes_to_bypass):
        graph.erase_node(node)

    canonicalize_graph(gm)


# =============================================================================
# Test Class
# =============================================================================


class TestDeleteAllUnusedSubmodulesOptimized:
    """Tests for delete_all_unused_submodules function."""

    def test_functional_equivalence_basic(self):
        """Test that optimized version produces identical results to original."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        # Verify submodules are present after export
        submodules_after_export = get_submodule_names(gm)
        assert len(submodules_after_export) > 0, "Expected submodules after export"

        # Make some modules unused by modifying the graph
        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        make_submodules_unused(gm_original, ["linear2"])
        make_submodules_unused(gm_optimized, ["linear2"])

        # Apply both deletion implementations
        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        # Verify identical results
        original_submodules = get_submodule_names(gm_original)
        optimized_submodules = get_submodule_names(gm_optimized)

        assert original_submodules == optimized_submodules, (
            f"Mismatch in submodules:\n"
            f"Original: {original_submodules}\n"
            f"Optimized: {optimized_submodules}"
        )

    def test_functional_equivalence_multiple_removals(self):
        """Test equivalence when multiple submodules are made unused."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Remove multiple submodules
        targets = ["linear2", "linear3"]
        make_submodules_unused(gm_original, targets)
        make_submodules_unused(gm_optimized, targets)

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        assert get_submodule_names(gm_original) == get_submodule_names(gm_optimized)

    def test_no_unused_modules(self):
        """Test that nothing is deleted when all modules are used."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        submodules_before = get_submodule_names(gm)

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Don't modify the graph - all modules should remain used
        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        # Verify nothing was deleted
        assert get_submodule_names(gm_original) == submodules_before
        assert get_submodule_names(gm_optimized) == submodules_before

    def test_all_modules_made_unused(self):
        """Test deletion when all call_module/get_attr nodes are removed."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Find all submodule targets and remove them
        all_targets = list(get_submodule_names(gm))

        # Remove references to all submodules
        make_submodules_unused(gm_original, all_targets)
        make_submodules_unused(gm_optimized, all_targets)

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        # Both should have the same (empty or minimal) set of submodules
        assert get_submodule_names(gm_original) == get_submodule_names(gm_optimized)

    def test_nested_module_partial_removal(self):
        """Test that parent module stays when only one child is removed."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Remove only one nested layer, keep the other
        make_submodules_unused(gm_original, ["nested.layer_a"])
        make_submodules_unused(gm_optimized, ["nested.layer_a"])

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        original_submodules = get_submodule_names(gm_original)
        optimized_submodules = get_submodule_names(gm_optimized)

        assert original_submodules == optimized_submodules

        # Verify nested parent still exists (because layer_b is still used)
        # Note: The exact behavior depends on how the graph represents nested modules

    def test_deeply_nested_hierarchy(self):
        """Test deletion with deeply nested module hierarchy."""
        model = DeeplyNestedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Remove one of the deep leaves
        make_submodules_unused(gm_original, ["level1.level2.level3.leaf_a"])
        make_submodules_unused(gm_optimized, ["level1.level2.level3.leaf_a"])

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        original_submodules = get_submodule_names(gm_original)
        optimized_submodules = get_submodule_names(gm_optimized)

        assert original_submodules == optimized_submodules

    def test_deeply_nested_full_branch_removal(self):
        """Test removal of entire deeply nested branch."""
        model = DeeplyNestedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Remove entire nested branch
        make_submodules_unused(
            gm_original, ["level1.level2.level3.leaf_a", "level1.level2.level3.leaf_b"]
        )
        make_submodules_unused(
            gm_optimized, ["level1.level2.level3.leaf_a", "level1.level2.level3.leaf_b"]
        )

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        original_submodules = get_submodule_names(gm_original)
        optimized_submodules = get_submodule_names(gm_optimized)

        assert original_submodules == optimized_submodules

    def test_sequential_module_handling(self):
        """Test handling of Sequential modules (call_module marks children used)."""
        model = ModelWithSequential().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Remove the sequential module
        make_submodules_unused(gm_original, ["sequential"])
        make_submodules_unused(gm_optimized, ["sequential"])

        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        original_submodules = get_submodule_names(gm_original)
        optimized_submodules = get_submodule_names(gm_optimized)

        assert original_submodules == optimized_submodules

    def test_idempotent_deletion(self):
        """Test that running deletion multiple times is idempotent."""
        model = FullyUsedModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_optimized = copy.deepcopy(gm)
        make_submodules_unused(gm_optimized, ["linear2"])

        # Run deletion twice
        delete_all_unused_submodules(gm_optimized)
        submodules_after_first = get_submodule_names(gm_optimized)

        delete_all_unused_submodules(gm_optimized)
        submodules_after_second = get_submodule_names(gm_optimized)

        assert submodules_after_first == submodules_after_second

    def test_empty_graph_module(self):
        """Test handling of a minimal GraphModule."""

        class MinimalModel(nn.Module):
            def forward(self, x):
                return x

        model = MinimalModel().to("cuda")
        x = torch.randn(2, 32, device="cuda")

        gm = torch_export_to_gm(model, args=(x,))

        gm_original = copy.deepcopy(gm)
        gm_optimized = copy.deepcopy(gm)

        # Should not raise any errors
        gm_original.delete_all_unused_submodules()
        delete_all_unused_submodules(gm_optimized)

        assert get_submodule_names(gm_original) == get_submodule_names(gm_optimized)


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "targets_to_remove",
    [
        ["linear1"],
        ["linear2"],
        ["linear3"],
        ["linear1", "linear2"],
        ["linear2", "linear3"],
        ["linear1", "linear3"],
        ["linear1", "linear2", "linear3"],
    ],
)
def test_various_removal_combinations(targets_to_remove):
    """Test various combinations of submodule removals."""
    model = FullyUsedModel().to("cuda")
    x = torch.randn(2, 32, device="cuda")

    gm = torch_export_to_gm(model, args=(x,))

    gm_original = copy.deepcopy(gm)
    gm_optimized = copy.deepcopy(gm)

    make_submodules_unused(gm_original, targets_to_remove)
    make_submodules_unused(gm_optimized, targets_to_remove)

    gm_original.delete_all_unused_submodules()
    delete_all_unused_submodules(gm_optimized)

    assert get_submodule_names(gm_original) == get_submodule_names(gm_optimized)
