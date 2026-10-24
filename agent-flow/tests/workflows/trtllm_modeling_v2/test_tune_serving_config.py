# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the serving config a perf measurement derives from its workload."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import serving_config as sc
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec


def test_the_measured_campaign_settings_are_what_the_rules_give():
    """ISL 8192 at concurrency 128: a dense capture list and a 16384 token budget won."""
    config = sc.derive_serving_config(isl=8192, osl=1024, concurrency=[1, 128])
    assert config["max_batch_size"] == 128
    assert config["max_seq_len"] == 8192 + 1024
    assert config["max_num_tokens"] == 16384
    assert not config["enable_chunked_prefill"]
    assert config["cuda_graph_config"]["batch_sizes"] == list(range(1, 129))
    assert config["cuda_graph_config"]["enable_padding"]


def test_a_short_workload_keeps_the_default_token_budget():
    config = sc.derive_serving_config(isl=1024, osl=1024, concurrency=[8])
    assert config["max_num_tokens"] == sc.MIN_MAX_NUM_TOKENS


def test_a_prompt_longer_than_the_budget_turns_on_chunked_prefill():
    config = sc.derive_serving_config(isl=65536, osl=1024, concurrency=[4])
    assert config["max_num_tokens"] == sc.MAX_MAX_NUM_TOKENS
    assert config["enable_chunked_prefill"]


def test_drafts_count_toward_the_budget_and_the_sequence_limit():
    plain = sc.derive_serving_config(isl=8000, osl=100, concurrency=[64])
    spec = sc.derive_serving_config(isl=8000, osl=100, concurrency=[64], draft_len=3)
    assert spec["max_seq_len"] == plain["max_seq_len"] + 3
    assert spec["max_num_tokens"] > plain["max_num_tokens"]


def test_large_batches_capture_powers_of_two_past_the_dense_limit():
    sizes = sc.capture_batch_sizes(1000)
    assert sizes[: sc.DENSE_CAPTURE_LIMIT] == list(range(1, sc.DENSE_CAPTURE_LIMIT + 1))
    assert sizes[sc.DENSE_CAPTURE_LIMIT :] == [512, 1000]


def test_a_perf_block_without_configs_runs_at_the_derived_one(tmp_path):
    task = {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        "features": ["mtp"],
        "perf": {"workload": {"isl": 2048, "osl": 256, "concurrency": [1, 32]}},
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    (config,) = spec.identity_configs
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert data["max_batch_size"] == 32
    # No options file declares a depth, so the speculator drafts one token.
    assert data["max_seq_len"] == 2048 + 256 + 1
    assert data["modeling_v2"] == "require"


def test_every_derived_perf_config_serves_the_target_unless_it_says_otherwise(tmp_path):
    workload = {"isl": 8192, "osl": 1024, "concurrency": [1, 16]}
    task = {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        "perf": {
            "configs": [
                {"name": "tp1", "workload": workload},
                {"name": "stock", "workload": workload, "llm_api_options": {"modeling_v2": "off"}},
            ]
        },
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    modes = {
        part.name: yaml.safe_load(part.configs[0].read_text(encoding="utf-8"))["modeling_v2"]
        for part in spec.parts
    }
    assert modes == {"tp1": "require", "stock": "off"}
