# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped MiniMax-M3.1-Flash tactic bundle against this build's native runners.

Every bundled entry names its tactic (``get_config_name``); the name must still be one of the tactics the runner
offers for that entry's shape, or the AutoTuner drops the entry and tunes the shape again.
"""

import ast
import json

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import m31_pins
from tensorrt_llm._utils import get_sm_version


def _entries():
    for rank_file in sorted((m31_pins.BUNDLE_ROOT / "tp2").glob("rank_*.json")):
        for entry in json.loads(rank_file.read_text())["autotuner"]:
            yield pytest.param(entry, id=f"{rank_file.stem}-{entry['runner']}-{entry['tactic']}")


def _names(entry):
    """The names of the tactics the entry's runner offers for its shape."""
    import tensorrt_llm  # noqa: F401  (registers the native runners)

    if entry["runner"] == "MXFP8GemmRunner":
        # runner_key is the runner's unique_id: "(torch.<output dtype>, <sm>)".
        dtype_name = entry["runner_key"].strip("()").split(",")[0].strip()
        runner = torch.classes.trtllm.MXFP8GemmRunner(getattr(torch, dtype_name.split(".")[-1]))
        return {"fallback"} | {runner.get_config_name(i) for i in range(runner.get_num_configs())}
    assert entry["runner"] == "FP4BlockScaleMoERunner"
    top_k, intermediate_size, local_num_experts, act_type = ast.literal_eval(entry["runner_key"])
    runner = torch.classes.trtllm.FP4BlockScaleMoERunner(act_type)
    num_tokens, packed_hidden = entry["shapes"][2]
    configs = runner.get_valid_configs(
        top_k, 2 * packed_hidden, intermediate_size, local_num_experts, num_tokens
    )
    return {"fallback"} | {runner.get_config_name(tile_n, config) for tile_n, config in configs}


@pytest.mark.skipif(get_sm_version() != 107, reason="the bundle is tuned for sm_107")
@pytest.mark.parametrize("entry", list(_entries()))
def test_bundled_tactic_name_resolves_on_this_build(entry):
    assert entry["tactic_name"] in _names(entry)
