# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.qwen4_exp_hyper_connection import Qwen4ExpHyperConnection


def _reference_mix(module, hyper_input):
    hc, hs = module.hc_count, module.hidden_size
    normed = module._normed_bundle(hyper_input)
    if module.use_combine:
        packed = module.input_mix_weight_down_block_inject(normed)
        down, injection, _ = packed.split((module.hc_lowrank, hc, module.input_mix_padding), dim=-1)
    else:
        down = module.input_mix_weight_down(normed)
        injection = None
    gate = torch.nn.functional.silu(down / hc)
    gate = torch.sigmoid(module.input_mix_weight_up(gate).float()).unflatten(-1, (hc, hs))
    mixed = (gate * normed.float().unflatten(-1, (hc, hs))).mean(dim=-2)
    return mixed.to(module.params_dtype), (hyper_input, normed, injection)


def _reference_combine(block_output, residual, hc_count, hidden_size):
    hyper_input, _, injection = residual
    assert injection is not None
    streams = hyper_input.float().unflatten(-1, (hc_count, hidden_size))
    gate = 2.0 * torch.sigmoid(injection.float() / hc_count)
    return (streams + block_output.float().unsqueeze(-2) * gate.unsqueeze(-1)).flatten(-2)


def _initialize_test_weights(*modules):
    """Initialize TRT-LLM Linear parameters that checkpoints normally populate."""
    for module in modules:
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -0.01, 0.01)


@pytest.mark.parametrize("rows", [0, 2])
def test_combine_and_mix_matches_unfused_reference_cpu(rows):
    torch.manual_seed(42)
    hc_count, hidden_size, lowrank = 4, 8, 4
    previous = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.float32,
    ).eval()
    current = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.float32,
    ).eval()
    _initialize_test_weights(previous, current)
    hyper_input = torch.randn(rows, hc_count * hidden_size)
    block_output = torch.randn(rows, hidden_size)

    _, previous_residual = _reference_mix(previous, hyper_input)
    expected_hidden = _reference_combine(
        block_output,
        previous_residual,
        hc_count,
        hidden_size,
    )
    expected_mixed, expected_residual = _reference_mix(current, expected_hidden)

    actual_hidden, actual_mixed, actual_residual = current.combine_and_mix(
        block_output,
        previous_residual,
    )
    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_mixed, expected_mixed)
    torch.testing.assert_close(actual_residual[0], expected_residual[0])
    torch.testing.assert_close(actual_residual[1], expected_residual[1])
    torch.testing.assert_close(actual_residual[2], expected_residual[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 4, 16, 128])
@torch.inference_mode()
def test_fused_hyper_connection_matches_unfused_reference_and_graph(rows, monkeypatch):
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_DIRECT_SKINNY_GEMM", "1")
    torch.manual_seed(42)
    hc_count, hidden_size, lowrank = 4, 2560, 320
    previous = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    current = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    _initialize_test_weights(previous, current)
    hyper_input = torch.randn(
        rows,
        hc_count * hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    block_output = torch.randn(
        rows,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    expected_previous_mixed, expected_previous_residual = _reference_mix(
        previous,
        hyper_input,
    )
    expected_hidden = _reference_combine(
        block_output,
        expected_previous_residual,
        hc_count,
        hidden_size,
    ).to(torch.bfloat16)
    expected_mixed, expected_residual = _reference_mix(current, expected_hidden)

    actual_previous_mixed, actual_previous_residual = previous.mix(hyper_input)
    actual_hidden, actual_mixed, actual_residual = current.combine_and_mix(
        block_output,
        actual_previous_residual,
    )
    torch.testing.assert_close(
        actual_previous_mixed,
        expected_previous_mixed,
        rtol=1e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(actual_mixed, expected_mixed, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(actual_residual[1], expected_residual[1], rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(actual_residual[2], expected_residual[2], rtol=1e-2, atol=5e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_previous_mixed, graph_previous_residual = previous.mix(hyper_input)
        graph_hidden, graph_mixed, graph_residual = current.combine_and_mix(
            block_output,
            graph_previous_residual,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_previous_mixed,
        actual_previous_mixed,
        rtol=1e-2,
        atol=5e-3,
    )
    torch.testing.assert_close(graph_hidden, actual_hidden, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(graph_mixed, actual_mixed, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(graph_residual[2], actual_residual[2], rtol=1e-2, atol=5e-3)
