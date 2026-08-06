# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

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


def test_lora_split_k_runner_uses_default_tactic(monkeypatch):
    runner = _make_runner()
    runner.lora_params = {}
    split_ks = []

    def fake_forward_impl(x, lora_params, layer_idx, split_k):
        del lora_params, layer_idx
        split_ks.append(split_k)
        return x

    monkeypatch.setattr(runner.layer, "_forward_cuda_graph_mode_impl", fake_forward_impl)

    x = torch.empty(2, runner.input_hidden_size)
    assert runner([x], tactic=-1) is x
    assert split_ks == [lora_layer._LORA_DEFAULT_SPLIT_K]


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
        def __init__(self, max_lora_size: int = 4):
            self.d_b_ptrs = torch.ones((1, max_lora_size), dtype=torch.int64)
            self.d_b_prime_ptrs = torch.full((1, max_lora_size), 2, dtype=torch.int64)
            self.d_output_sizes = torch.tensor([128])
            self.d_output_sizes_offset = torch.tensor([0])
            self.h_output_sizes = torch.tensor([128])

    class FakeCudaGraphParams:
        def __init__(self):
            self.max_rank = 16
            self.max_lora_size = 4
            self.layer_info = {layer_key: object()}
            self.layer_params = {layer_key: FakeLayerParams(self.max_lora_size)}
            self.slot_counts = torch.tensor([4, 0, 0, 0])
            self.slot_ranks = torch.tensor([16, 0, 0, 0])
            self.slot_offsets_full = torch.tensor([0, 4, 4, 4, 4])
            self.sorted_ids = torch.arange(8)

        def get_layer_params(self, key):
            assert key == layer_key
            return self.layer_params.get(key)

        def get_problem_count(self, key):
            assert key == layer_key
            return 4

    tuned_runners = []

    class FakeTuner:
        def choose_one(self, custom_op, runners, tuning_config, inputs):
            assert custom_op == "trtllm::lora_grouped_gemm_cuda_graph"
            assert tuning_config is runners[0].tuning_config
            assert len(inputs) == 1
            tuned_runners.append(runners[0])
            return runners[0], 1

    monkeypatch.setattr(lora_layer.AutoTuner, "get", staticmethod(lambda: FakeTuner()))

    parameter_fill_calls = []

    def fake_parameter_fill(*args):
        parameter_fill_calls.append(args)

    monkeypatch.setattr(
        torch.ops.trtllm,
        "lora_group_gemm_param_fill_row_reorder_fusion",
        fake_parameter_fill,
    )
    operator_split_ks = []

    def fake_grouped_gemm(*args):
        operator_split_ks.append(args[-1])

    monkeypatch.setattr(torch.ops.trtllm, "lora_grouped_gemm_cuda_graph", fake_grouped_gemm)

    cuda_graph_params = FakeCudaGraphParams()
    original_layer_params = cuda_graph_params.get_layer_params(layer_key)
    original_slot_counts = cuda_graph_params.slot_counts
    original_b_ptrs = original_layer_params.d_b_ptrs
    lora_params = {"cuda_graph_params": cuda_graph_params}
    warmup_lora_params = []
    for batch_size in (4, 8):
        x = torch.empty(batch_size, 256)
        output = layer._forward_cuda_graph_mode(x, lora_params, layer_idx)
        assert output.shape == (batch_size, 128)
        warmup_lora_params.append(layer._split_k_runner.lora_params)

    runner = layer._split_k_runner
    assert runner is not None
    assert tuned_runners == [runner, runner]
    assert operator_split_ks == [1, 1]
    assert warmup_lora_params[0] is not warmup_lora_params[1]

    runner_lora_params = warmup_lora_params[-1]
    assert runner_lora_params is not None
    runner_cuda_graph_params = runner_lora_params["cuda_graph_params"]
    runner_layer_params = runner_cuda_graph_params.get_layer_params(layer_key)
    assert runner_lora_params is not lora_params
    assert runner_cuda_graph_params is not cuda_graph_params
    assert runner_cuda_graph_params.layer_params is not cuda_graph_params.layer_params
    assert runner_layer_params is not original_layer_params

    synthetic_inputs = runner._prepare_synthetic_inputs([torch.empty(2, 256)])
    runner(synthetic_inputs, tactic=2)
    assert operator_split_ks == [1, 1, 2]

    synthetic_fill_args = parameter_fill_calls[-1]
    assert synthetic_fill_args[16] is synthetic_inputs[1]
    assert synthetic_fill_args[21] is synthetic_inputs[4]
    assert runner_cuda_graph_params.slot_counts is original_slot_counts
    assert runner_layer_params.d_b_ptrs is original_b_ptrs
    assert cuda_graph_params.slot_counts is original_slot_counts
    assert original_layer_params.d_b_ptrs is original_b_ptrs


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
    layer_params = SimpleNamespace(
        d_b_ptrs=torch.zeros((2, 4), dtype=torch.int64, device="cuda"),
        d_b_prime_ptrs=torch.zeros((2, 4), dtype=torch.int64, device="cuda"),
        d_output_sizes=torch.tensor([128, 64], dtype=torch.int32, device="cuda"),
        d_output_sizes_offset=torch.tensor([0, 128], dtype=torch.int64, device="cuda"),
    )
    cuda_graph_params = SimpleNamespace(
        slot_counts=torch.zeros(4, dtype=torch.int32, device="cuda"),
        slot_ranks=torch.zeros(4, dtype=torch.int32, device="cuda"),
        slot_offsets_full=torch.zeros(5, dtype=torch.int64, device="cuda"),
        layer_params={runner.layer_key: layer_params},
    )
    cuda_graph_params.get_layer_params = cuda_graph_params.layer_params.get
    runner.lora_params = {"cuda_graph_params": cuda_graph_params}

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
