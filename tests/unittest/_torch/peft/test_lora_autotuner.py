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


def test_lora_layer_reuses_runner_across_cuda_graph_warmups(monkeypatch):
    layer = lora_layer.LoraLayer(
        [lora_layer.LoraModuleType.ATTENTION_Q],
        [128],
    )
    layer_idx = 3
    layer_key = lora_layer.CudaGraphLoraParams.LoraLayerKey(
        layer_idx=layer_idx,
        module_ids=tuple(layer.lora_module_types),
    )

    class FakeLayerParams:
        d_b_ptrs = None
        d_b_prime_ptrs = None
        d_output_sizes = None
        d_output_sizes_offset = None

    class FakeCudaGraphParams:
        layer_info = {layer_key: object()}
        max_rank = 16
        max_lora_size = 4
        slot_counts = None
        slot_ranks = None
        slot_offsets_full = None
        sorted_ids = None

        def get_layer_params(self, key):
            assert key == layer_key
            return FakeLayerParams()

        def get_problem_count(self, key):
            assert key == layer_key
            return 4

    runners_created = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.tuning_config = object()
            self.kwargs = kwargs
            self.calls = []
            runners_created.append(self)

        def __call__(self, inputs, *, tactic):
            self.calls.append((inputs, tactic))
            return inputs[0]

    tuned_runners = []

    class FakeTuner:
        def choose_one(self, custom_op, runners, tuning_config, inputs):
            assert custom_op == "trtllm::lora_grouped_gemm_cuda_graph"
            assert tuning_config is runners[0].tuning_config
            tuned_runners.append(runners[0])
            return runners[0], 1

    monkeypatch.setattr(lora_layer, "_LoraGroupedGemmRunner", FakeRunner)
    monkeypatch.setattr(lora_layer.AutoTuner, "get", staticmethod(lambda: FakeTuner()))

    lora_params = {"cuda_graph_params": FakeCudaGraphParams()}
    for batch_size in (4, 8):
        x = torch.empty(batch_size, 256)
        assert layer._forward_cuda_graph_mode(x, lora_params, layer_idx) is x

    assert len(runners_created) == 1
    assert layer._split_k_runner is runners_created[0]
    assert tuned_runners == [runners_created[0], runners_created[0]]
    assert len(runners_created[0].calls) == 2


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
