# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise logical-to-physical expert routing with real weight banks."""

import json
from pathlib import Path

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import MoeConfig, MoeLoadBalancerConfig

from ._engine import shortcut_text_engine

_PROMPTS = ("The capital of France is", "Continue this sequence: one, two, three,")


def _engine(checkpoint, **options):
    return shortcut_text_engine(
        checkpoint,
        max_batch_size=1,
        max_num_tokens=256,
        max_seq_len=256,
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=False),
        **options,
    )


def _generate(llm, prompt):
    sampling = SamplingParams(temperature=0, top_k=1, max_tokens=16)
    return list(llm.generate(prompt, sampling).outputs[0].token_ids)


@pytest.fixture(scope="module")
def baseline_tokens():
    """Greedy tokens with the default placement, generated once per checkpoint."""
    cache = {}

    def tokens(checkpoint):
        if checkpoint not in cache:
            with _engine(checkpoint) as llm:
                cache[checkpoint] = [_generate(llm, prompt) for prompt in _PROMPTS]
        return cache[checkpoint]

    return tokens


@pytest.mark.skip_less_device(4)
def test_hot_expert_replica_uses_migrated_weight_planes(llm_venv):
    from tensorrt_llm._utils import get_sm_version

    from ....common import venv_mpi_check_call

    if get_sm_version() != 103:
        pytest.skip("Online placement uses host-accessible GB300 device weights")
    worker = Path(__file__).resolve().with_name("_expert_migration.py")
    venv_mpi_check_call(llm_venv, ["mpirun", "--allow-run-as-root", "-n", "4"], [str(worker)])


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize(
    ("checkpoint", "online"),
    [("bf16", False), ("mxfp8", False), ("nvfp4", False), ("nvfp4", True)],
    indirect=["checkpoint"],
    ids=["bf16-static", "mxfp8-static", "nvfp4-static", "nvfp4-online"],
)
def test_expert_placement_preserves_generation(checkpoint, online, baseline_tokens):
    if not checkpoint.is_dir():
        pytest.skip(f"Checkpoint unavailable: {checkpoint}")
    cfg = json.loads((checkpoint / "config.json").read_text())["text_config"]
    experts = cfg["n_routed_experts"]
    # Reverse the rank ownership and give each rank a replicated slot.
    placement = list(reversed(range(experts))) + [0, 1]
    assignments = {
        i: placement for i, kind in enumerate(cfg["layer_types"]) if kind.startswith("shortcut_moe")
    }
    config = MoeLoadBalancerConfig(
        num_slots=len(placement),
        initial_global_assignments=assignments,
        layer_updates_per_iter=int(online),
    )
    expected = baseline_tokens(checkpoint)
    assert all(expected)
    with _engine(checkpoint, moe_config=MoeConfig(load_balancer=config)) as llm:
        for prompt, tokens in zip(_PROMPTS, expected, strict=True):
            assert _generate(llm, prompt) == tokens
