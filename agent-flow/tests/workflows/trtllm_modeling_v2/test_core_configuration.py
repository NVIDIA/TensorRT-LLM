# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The core configuration: batch sizes are global totals, block reuse is off unless asked."""

from __future__ import annotations

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import check_config


def test_an_attention_dp_rank_serves_its_share_of_the_global_batch():
    options = {
        "max_batch_size": 128,
        "cuda_graph_config": {
            "max_batch_size": 128,
            "batch_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128],
        },
    }
    check_config.share_batch(options, "dep4")
    assert options["max_batch_size"] == 32
    assert options["cuda_graph_config"] == {
        "max_batch_size": 32,
        "batch_sizes": [1, 2, 4, 8, 16, 24, 32],
    }


@pytest.mark.parametrize("member", ["tp4", "tep4", "dep1", None])
def test_a_member_without_attention_dp_keeps_the_batch_it_is_given(member):
    options = {"max_batch_size": 128, "cuda_graph_config": {"max_batch_size": 128}}
    check_config.share_batch(options, member)
    assert options == {"max_batch_size": 128, "cuda_graph_config": {"max_batch_size": 128}}


def test_a_share_rounds_up_and_a_graph_list_ends_at_it():
    options = {"max_batch_size": 130, "cuda_graph_config": {"batch_sizes": [1, 8, 64]}}
    check_config.share_batch(options, "dep4")
    assert options["max_batch_size"] == 33
    assert options["cuda_graph_config"]["batch_sizes"] == [1, 8, 33]


def test_block_reuse_is_off_in_a_launch_that_does_not_ask_for_it(tmp_path):
    core = tmp_path / "core.yaml"
    core.write_text("kv_cache_config: {dtype: fp8}\n")
    task = {
        "features": [],
        "runtime": {},
        "accuracy": {"llm_api_options": {"core": str(core)}},
    }
    (launch,) = [k for k in check_config.launch_kv_caches(task) if k.unit == "core"]
    assert launch.reuse is False
