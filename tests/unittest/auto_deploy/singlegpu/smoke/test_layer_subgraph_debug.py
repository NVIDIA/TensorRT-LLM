# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Diagnostic test for Eagle-style layer subgraph partitioning.

Export Llama and Mistral4 as Eagle-style targets (inputs_embeds, no embedding op)
with 3 hidden layers, then run get_all_layer_subgraphs and collect_residual_add_nodes to print
every op in every subgraph for comparison.

Usage:
    pytest tests/unittest/auto_deploy/singlegpu/smoke/test_layer_subgraph_debug.py -sv
    pytest tests/unittest/auto_deploy/singlegpu/smoke/test_layer_subgraph_debug.py::test_llama_subgraphs -sv
    pytest tests/unittest/auto_deploy/singlegpu/smoke/test_layer_subgraph_debug.py::test_mistral4_subgraphs -sv
"""

from pathlib import Path

import pytest
import torch
from _model_test_utils import get_small_model_config
from test_common.llm_data import hf_id_to_local_model_dir

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.hidden_states import (
    DetectHiddenStatesForCapture,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    get_all_layer_subgraphs,
    identify_regions_between_residuals,
)

NUM_HIDDEN_LAYERS = 3
SEQ_LEN = 8


def _print_all_graph_nodes(gm, label=""):
    """Print every node in the FX graph with op, target, name."""
    print(f"\n{'=' * 80}")
    print(f"  ALL GRAPH NODES {label}")
    print(f"{'=' * 80}")
    for i, node in enumerate(gm.graph.nodes):
        target_str = str(node.target) if node.op == "call_function" else node.target
        print(f"  [{i:3d}] op={node.op:<16s} name={node.name:<60s} target={target_str}")
    print(f"{'=' * 80}\n")


def _print_subgraph_detail(layer_subgraphs, label=""):
    """Print detailed info about each layer subgraph."""
    print(f"\n{'=' * 80}")
    print(f"  LAYER SUBGRAPHS {label}")
    print(f"{'=' * 80}")
    for i, sg in enumerate(layer_subgraphs):
        print(f"\n--- Subgraph {i}: type={sg.layer_type} ---")
        print(f"  Opening nodes ({len(sg.opening_nodes)}):")
        for n in sg.opening_nodes:
            print(f"    {n.name}")
        print(f"  Terminating node: {sg.terminating_node.name if sg.terminating_node else None}")
        print(f"  Interior nodes ({len(sg.subgraph_nodes)}):")
        for n in sg.subgraph_nodes:
            if n.op == "call_function":
                print(f"    {n.name:<60s} target={n.target}")
            else:
                print(f"    {n.name:<60s} op={n.op}")
    print(f"{'=' * 80}\n")


def _print_residual_boundaries(gm, label=""):
    """Print the boundary nodes returned by identify_regions_between_residuals."""
    residuals = identify_regions_between_residuals(gm)
    print(f"\n{'=' * 80}")
    print(f"  RESIDUAL BOUNDARIES {label}")
    print(f"{'=' * 80}")
    for i, node in enumerate(residuals):
        print(f"  [{i}] name={node.name:<60s} op={node.op}")
    print(f"  Total boundary nodes: {len(residuals)}")
    if len(residuals) == 2:
        print("  WARNING: Only input+output — no residual adds found (no embedding op?)")
    print(f"{'=' * 80}\n")


def _export_as_eagle_target(model, hidden_size):
    """Export model using inputs_embeds (like Eagle target), not input_ids.

    This means no aten.embedding in the graph — matching what detect_hidden_states_for_capture sees.
    """
    inputs_embeds = torch.randn(1, SEQ_LEN, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int64).unsqueeze(0)
    gm = torch_export_to_gm(
        model,
        kwargs={"inputs_embeds": inputs_embeds, "position_ids": position_ids},
    )
    return gm


def _export_with_input_ids(model):
    """Export model using input_ids (standard path with embedding op)."""
    input_ids = torch.ones((1, SEQ_LEN), dtype=torch.int64)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int64).unsqueeze(0)
    gm = torch_export_to_gm(
        model,
        args=(input_ids, position_ids),
    )
    return gm


def _build_llama_model(num_hidden_layers):
    """Build a small Llama model."""
    config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    config["args"]["model_kwargs"]["num_hidden_layers"] = num_hidden_layers

    from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry

    factory_cls = ModelFactoryRegistry.get("AutoModelForCausalLM")
    factory = factory_cls(
        model=config["args"]["model"],
        model_kwargs=config["args"]["model_kwargs"],
        skip_loading_weights=True,
    )
    model = factory.build_model("meta")
    hidden_size = config["args"]["model_kwargs"]["hidden_size"]
    return model, hidden_size


def _build_mistral4_model(num_hidden_layers):
    """Build a small Mistral4 model."""
    model_hub_id = "mistralai/Mistral-Small-4-119B-2603"
    model_path = hf_id_to_local_model_dir(model_hub_id)
    if model_path is None or not Path(model_path).is_dir():
        pytest.skip(f"Target model path does not exist: {model_path}")

    config = get_small_model_config(model_hub_id)
    small_dims = dict(config["args"]["model_kwargs"]["text_config"])
    small_dims["num_hidden_layers"] = num_hidden_layers

    from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry

    factory_cls = ModelFactoryRegistry.get("Mistral3ForConditionalGeneration")
    factory = factory_cls(
        model=model_path,
        model_kwargs={"text_config": small_dims},
        skip_loading_weights=True,
    )
    model = factory.build_model("meta")
    hidden_size = small_dims["hidden_size"]
    return model, hidden_size


def _run_diagnostics(gm, label):
    """Run all diagnostic prints on a GraphModule."""
    _print_all_graph_nodes(gm, label)
    _print_residual_boundaries(gm, label)

    print(f"\n--- Running get_all_layer_subgraphs {label} ---")
    try:
        layer_subgraphs, unprocessed = get_all_layer_subgraphs(gm)
        _print_subgraph_detail(layer_subgraphs, label)
        print(f"Unprocessed linear nodes: {[n.name for n in unprocessed]}")
    except Exception as e:
        print(f"get_all_layer_subgraphs FAILED: {type(e).__name__}: {e}")
        return

    print(f"\n--- Running collect_residual_add_nodes {label} ---")
    try:
        transform = DetectHiddenStatesForCapture(
            config=TransformConfig(
                stage="pattern_matcher",
                eagle3_layers_to_capture={-1},
            )
        )
        residual_add_nodes = transform.collect_residual_add_nodes(gm)
        print(f"  Found residual add nodes for layers: {sorted(residual_add_nodes.keys())}")
        for layer_num, node in sorted(residual_add_nodes.items()):
            print(f"    layer {layer_num}: {node.name}")
    except Exception as e:
        print(f"collect_residual_add_nodes FAILED: {type(e).__name__}: {e}")


def test_llama_subgraphs():
    """Diagnostic: Llama 3-layer as Eagle target (inputs_embeds, no embedding).

    Also shows the standard input_ids path for comparison.
    """
    model, hidden_size = _build_llama_model(NUM_HIDDEN_LAYERS)

    # Standard path (with embedding) — for comparison
    print("\n" + "#" * 80)
    print("# LLAMA — STANDARD (input_ids, with embedding)")
    print("#" * 80)
    gm_standard = _export_with_input_ids(model)
    _run_diagnostics(gm_standard, "[Llama standard]")

    # Eagle target path (inputs_embeds, no embedding)
    print("\n" + "#" * 80)
    print("# LLAMA — EAGLE TARGET (inputs_embeds, no embedding)")
    print("#" * 80)
    gm_eagle = _export_as_eagle_target(model, hidden_size)
    _run_diagnostics(gm_eagle, "[Llama eagle-target]")


def test_mistral4_subgraphs():
    """Diagnostic: Mistral4 3-layer as Eagle target (inputs_embeds, no embedding)."""
    model, hidden_size = _build_mistral4_model(NUM_HIDDEN_LAYERS)

    # Eagle target path (inputs_embeds, no embedding)
    print("\n" + "#" * 80)
    print("# MISTRAL4 — EAGLE TARGET (inputs_embeds, no embedding)")
    print("#" * 80)
    gm_eagle = _export_as_eagle_target(model, hidden_size)
    _run_diagnostics(gm_eagle, "[Mistral4 eagle-target]")


if __name__ == "__main__":
    import sys

    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if test_name in ("llama", "all"):
        test_llama_subgraphs()
    if test_name in ("mistral4", "all"):
        test_mistral4_subgraphs()
