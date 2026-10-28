# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cp_config with context_parallel_size 1 serves requests like no cp_config."""

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import CpConfig

PROMPTS = ["The capital of France is", "Hello, my name is"]


def _generate_token_ids(cp_config):
    with LLM(
        model=str(llm_models_root() / "Qwen3" / "Qwen3-0.6B"),
        cp_config=cp_config,
        kv_cache_config=KvCacheConfig(max_tokens=4096),
    ) as llm:
        outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=8, temperature=0.0))
    return [output.outputs[0].token_ids for output in outputs]


@pytest.fixture(scope="module")
def reference_token_ids():
    return _generate_token_ids(cp_config=None)


@pytest.mark.parametrize(
    "cp_config",
    [CpConfig(), CpConfig(cp_type="RING")],
    ids=["default", "ring"],
)
def test_single_cp_rank_cp_config_matches_no_cp_config(reference_token_ids, cp_config):
    assert _generate_token_ids(cp_config) == reference_token_ids
