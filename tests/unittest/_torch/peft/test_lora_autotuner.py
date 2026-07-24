# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.peft.lora import layer as lora_layer


def _make_runner(
    layer_idx: int = 3,
    input_hidden_size: int = 256,
) -> lora_layer._LoraGroupedGemmRunner:
    layer = lora_layer.LoraLayer(
        [
            lora_layer.LoraModuleType.ATTENTION_Q,
            lora_layer.LoraModuleType.ATTENTION_K,
        ],
        [128, 64],
    )
    return lora_layer._LoraGroupedGemmRunner(
        layer=layer,
        layer_idx=layer_idx,
        input_hidden_size=input_hidden_size,
        max_rank=16,
        max_lora_size=4,
        problem_count=8,
        dtype=torch.float16,
    )


def test_lora_split_k_runner_identity_is_layer_specific():
    runner = _make_runner(layer_idx=3)
    other_layer_runner = _make_runner(layer_idx=4)

    assert runner.unique_id() != other_layer_runner.unique_id()


def test_lora_split_k_runner_uses_token_buckets():
    runner = _make_runner()
    spec = runner.tuning_config.dynamic_tensor_specs[0]

    AutoTuner._find_nearest_profile.cache_clear()
    profile = AutoTuner._find_nearest_profile(
        (torch.Size((7, runner.input_hidden_size)),),
        runner.tuning_config.dynamic_tensor_specs,
        runner.tuning_config.constraint_specs,
        runner.tuning_config.tune_max_num_tokens,
    )

    assert spec.input_idx == 0
    assert spec.dim_idx == 0
    assert profile[0][0] == 4


def test_lora_split_k_runner_prunes_splits_larger_than_k_tiles():
    runner = _make_runner(input_hidden_size=256)

    assert runner.get_valid_tactics([], None) == [1, 2, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lora_autotuner_hook_builds_single_active_slot():
    runner = _make_runner()
    num_tokens = 8
    carrier = torch.randn(
        num_tokens,
        runner.input_hidden_size,
        dtype=runner.dtype,
        device="cuda",
    )

    inputs = runner._prepare_synthetic_inputs([carrier])
    slot_counts, slot_ranks = inputs[1], inputs[2]
    slot_offsets_full = inputs[3]
    b_ptrs, b_prime_ptrs = inputs[4], inputs[5]
    sorted_ids, output_hidden_sizes = inputs[6], inputs[7]

    assert slot_counts.tolist() == [num_tokens, 0, 0, 0]
    assert slot_ranks.tolist() == [runner.max_rank, 0, 0, 0]
    assert slot_offsets_full.tolist() == [
        0,
        num_tokens,
        num_tokens,
        num_tokens,
        num_tokens,
    ]
    assert sorted_ids.tolist() == list(range(num_tokens))
    assert output_hidden_sizes.tolist() == [128, 64]
    assert torch.all(b_ptrs[:, 0] != 0)
    assert torch.all(b_prime_ptrs[:, 0] != 0)
    assert torch.all(b_ptrs[:, 1:] == 0)
    assert torch.all(b_prime_ptrs[:, 1:] == 0)
