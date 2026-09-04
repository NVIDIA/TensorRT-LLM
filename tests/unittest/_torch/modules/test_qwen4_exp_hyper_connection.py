# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.qwen4_exp.hyper_connection import Qwen4ExpHyperConnection
from tensorrt_llm._torch.modules.qwen4_exp.hyper_connection_kernels import hc_combine_norm


def _reference_mix(module, hyper_input):
    hc, hs = module.hc_count, module.hidden_size
    normed = module._normed_bundle(hyper_input)
    if module.use_combine:
        packed = module.input_mix_weight_down_block_inject(normed)
        down = packed[..., : module.hc_lowrank]
        injection = packed[
            ...,
            module.input_mix_injection_offset : module.input_mix_injection_offset + hc,
        ]
    else:
        down = module.input_mix_weight_down(normed)
        injection = None
    gate = torch.nn.functional.silu(down / hc)
    gate = torch.sigmoid(module.input_mix_weight_up(gate).float()).unflatten(-1, (hc, hs))
    mixed = (gate * normed.float().unflatten(-1, (hc, hs))).mean(dim=-2)
    return mixed.to(module.params_dtype), (hyper_input, injection)


def _reference_combine(block_output, residual, hc_count, hidden_size):
    hyper_input, injection = residual
    assert injection is not None
    streams = hyper_input.float().unflatten(-1, (hc_count, hidden_size))
    gate = 2.0 * torch.sigmoid(injection.float() / hc_count)
    return (streams + block_output.float().unsqueeze(-2) * gate.unsqueeze(-1)).flatten(-2)


def _initialize_test_weights(*modules):
    """Initialize TRT-LLM Linear parameters that checkpoints normally populate."""
    for module in modules:
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -0.01, 0.01)


def test_from_config_preserves_shared_norm_layout() -> None:
    module = Qwen4ExpHyperConnection.from_config(
        SimpleNamespace(
            hc_count=2,
            hidden_size=8,
            hc_lowrank=4,
            rms_norm_eps=1e-6,
            hc_per_branch_norm=False,
        ),
        dtype=torch.float32,
    )

    assert not module.hc_per_branch_norm
    assert module.hc_norm.weight.shape == (8,)


def test_multi_dimensional_input_uses_the_unfused_path() -> None:
    """Only flattened two-dimensional operands are eligible for Triton fusion."""
    torch.manual_seed(42)
    module = Qwen4ExpHyperConnection(
        hc_count=2,
        hidden_size=4,
        hc_lowrank=2,
        dtype=torch.float32,
    ).eval()
    _initialize_test_weights(module)
    hyper_input = torch.randn(2, 3, 8)
    block_output = torch.randn(2, 3, 4)

    mixed, residual = module.mix(hyper_input)
    combined = module.combine(block_output, residual)

    assert mixed.shape == (2, 3, 4)
    assert combined.shape == hyper_input.shape


def test_combine_only_module_uses_caller_owned_residual() -> None:
    module = Qwen4ExpHyperConnection(
        hc_count=2,
        hidden_size=4,
        hc_lowrank=2,
        dtype=torch.float32,
        use_mix=False,
    ).eval()
    hyper_input = torch.randn(3, 8)
    injection_logits = torch.randn(3, 2)
    block_output = torch.randn(3, 4)

    actual = module.combine(block_output, (hyper_input, injection_logits))
    expected = _reference_combine(block_output, (hyper_input, injection_logits), 2, 4)

    torch.testing.assert_close(actual, expected)


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


def test_packed_down_projection_has_fixed_alignment_cpu() -> None:
    torch.manual_seed(42)
    module = Qwen4ExpHyperConnection(
        hc_count=2,
        hidden_size=4,
        hc_lowrank=6,
        dtype=torch.float32,
    ).eval()
    _initialize_test_weights(module)
    small_input = torch.randn(2, 8)
    large_input = torch.randn(8192, 8)

    small_actual = module._packed_down_and_injection(small_input)
    large_actual = module._packed_down_and_injection(large_input)

    assert module.input_mix_weight_down_block_inject.weight.shape == (16, 8)
    assert module.input_mix_injection_offset == 6
    assert module.input_mix_padding == 8
    assert small_actual.shape == (2, 16)
    assert large_actual.shape == (8192, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 4, 16, 128, 320])
@torch.inference_mode()
def test_fused_hyper_connection_matches_unfused_reference_and_graph(rows):
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
    torch.testing.assert_close(graph_residual[1], actual_residual[1], rtol=1e-2, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# 2560 is the Qwen4-Exp hidden size and takes the widest single tile; 640 is
# narrower than that tile; 4608 is wider than it, so it still has to tile.
@pytest.mark.parametrize("hidden_size", [2560, 640, 4608])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("per_branch_norm", [True, False])
@torch.inference_mode()
def test_combine_norm_matches_fp32_reference(rows, hidden_size, per_branch_norm):
    """The combine+norm kernel's tiling must not depend on the row width."""
    torch.manual_seed(42)
    hc_count, eps = 4, 1e-6
    residual = torch.randn(rows, hc_count * hidden_size, dtype=torch.bfloat16, device="cuda")
    block_output = torch.randn(rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    injection_logits = torch.randn(rows, hc_count, dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.randn(
        hc_count * hidden_size if per_branch_norm else hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    expected_output = _reference_combine(
        block_output,
        (residual, injection_logits),
        hc_count,
        hidden_size,
    ).to(torch.bfloat16)
    grouped = expected_output.float().unflatten(-1, (hc_count, hidden_size))
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected_normed = (grouped + grouped * norm_weight.float().reshape(-1, hidden_size)).flatten(-2)

    output, normed = hc_combine_norm(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )
    torch.testing.assert_close(output, expected_output, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(normed.float(), expected_normed, rtol=1e-2, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_combine_norm_rejects_a_reshaped_norm_weight():
    """The kernel accepts a parameter vector, not an equal-sized 2-D view."""
    residual = torch.empty((1, 16), dtype=torch.bfloat16, device="cuda")
    block_output = torch.empty((1, 4), dtype=torch.bfloat16, device="cuda")
    injection_logits = torch.empty((1, 4), dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.empty((2, 2), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="norm weight"):
        hc_combine_norm(
            residual,
            block_output,
            injection_logits,
            norm_weight,
            1e-6,
            4,
        )
