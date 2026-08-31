# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.fused_shared_expert import (
    PendingSharedExpertGate,
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch.modules.qwen4_exp_hyper_connection import Qwen4ExpHyperConnection
from tensorrt_llm._torch.modules.qwen4_exp_hyper_connection_kernels import (
    hc_combine,
    hc_combine_norm,
)
from tensorrt_llm._utils import is_sm_100f

_skip_non_sm10x = pytest.mark.skipif(
    torch.cuda.device_count() == 0 or not is_sm_100f(),
    reason="requires SM10x GPU (SM100/SM103)",
)


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
    assert actual_residual[1] is None
    torch.testing.assert_close(actual_residual[2], expected_residual[2])


def test_fused_layout_slices_padding_only_for_large_prefill_cpu(monkeypatch):
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_FUSED_MIX", "1")
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

    small_full = module.input_mix_weight_down_block_inject(small_input)
    small_actual, small_gate_fused = module._packed_down_and_injection(small_input)
    large_full = module.input_mix_weight_down_block_inject(large_input)
    large_actual, large_gate_fused = module._packed_down_and_injection(large_input)

    assert module.input_mix_weight_down_block_inject.weight.shape == (128, 8)
    assert module.input_mix_fallback_rows == 16
    assert module.input_mix_injection_offset == 6
    assert small_actual.shape == (2, 128)
    assert large_actual.shape == (8192, 16)
    # Only the direct decode kernel folds the gate activation into its epilogue.
    assert not small_gate_fused and not large_gate_fused
    torch.testing.assert_close(small_actual, small_full)
    torch.testing.assert_close(large_actual, large_full[:, :16])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 4, 16, 128, 320])
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
    assert actual_previous_residual[1] is None
    assert actual_residual[1] is None
    torch.testing.assert_close(actual_residual[2], expected_residual[2], rtol=1e-2, atol=5e-3)

    # The direct decode kernel must actually be the one that applied the gate
    # activation, otherwise this comparison silently stops covering its epilogue.
    _, gate_fused = current._packed_down_and_injection(current._normed_bundle(expected_hidden))
    assert gate_fused == (rows == 1)

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
    assert graph_previous_residual[1] is None
    assert graph_residual[1] is None
    torch.testing.assert_close(graph_residual[2], actual_residual[2], rtol=1e-2, atol=5e-3)


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
        (residual, None, injection_logits),
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
@pytest.mark.parametrize("rows", [1, 4, 16])
@torch.inference_mode()
def test_cute_fused_mix_matches_reference_and_graph(rows, monkeypatch):
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_FUSED_MIX", "1")
    torch.manual_seed(42)
    module = Qwen4ExpHyperConnection(
        hc_count=4,
        hidden_size=2560,
        hc_lowrank=320,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    _initialize_test_weights(module)
    hyper_input = torch.randn(
        rows,
        10240,
        dtype=torch.bfloat16,
        device="cuda",
    )

    expected_mixed, expected_residual = _reference_mix(module, hyper_input)
    actual_mixed, actual_residual = module.mix(hyper_input)
    torch.testing.assert_close(actual_mixed, expected_mixed, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(
        actual_residual[2],
        expected_residual[2],
        rtol=1e-2,
        atol=5e-3,
    )
    assert module.input_mix_weight_down_block_inject.weight.shape == (384, 10240)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_mixed, graph_residual = module.mix(hyper_input)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_mixed, actual_mixed, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(
        graph_residual[2],
        actual_residual[2],
        rtol=1e-2,
        atol=5e-3,
    )


@_skip_non_sm10x
@pytest.mark.parametrize("rows,engages", [(1, True), (4, False)])
@torch.inference_mode()
def test_gated_stream_mix_matches_the_unfused_projection(rows, engages):
    """The fused gate GEMM reproduces up projection + sigmoid gate + stream mean.

    It is a decode-row-count path: wider batches must decline it and keep the
    unfused pair, which is what ``mix`` falls back to.
    """
    torch.manual_seed(42)
    hc_count, hidden_size, lowrank = 4, 2560, 320
    module = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    _initialize_test_weights(module)
    normed = torch.randn(
        rows,
        hc_count * hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = torch.randn(rows, lowrank, dtype=torch.bfloat16, device="cuda")

    fused = module._gated_stream_mix(normed, gate)
    assert (fused is not None) == engages
    if not engages:
        return

    projected = torch.sigmoid(module.input_mix_weight_up(gate).float())
    expected = (
        projected.unflatten(-1, (hc_count, hidden_size))
        * normed.float().unflatten(-1, (hc_count, hidden_size))
    ).mean(dim=-2)
    torch.testing.assert_close(fused.float(), expected, rtol=1e-2, atol=5e-3)


@_skip_non_sm10x
@torch.inference_mode()
def test_gated_stream_mix_reads_the_packed_down_projection_slice(monkeypatch):
    """``mix`` feeds the fused gate the packed row slice, stride and all."""
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_DIRECT_SKINNY_GEMM", "1")
    torch.manual_seed(42)
    hc_count, hidden_size, lowrank = 4, 2560, 320
    module = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    _initialize_test_weights(module)
    hyper_input = torch.randn(1, hc_count * hidden_size, dtype=torch.bfloat16, device="cuda")

    expected_mixed, _ = _reference_mix(module, hyper_input)
    actual_mixed, _ = module.mix(hyper_input)

    normed = module._normed_bundle(hyper_input)
    packed, gate_fused = module._packed_down_and_injection(normed)
    gate = packed[..., :lowrank]
    assert gate_fused and gate.stride(0) == module.input_mix_fallback_rows
    assert module._gated_stream_mix(normed, gate) is not None
    torch.testing.assert_close(actual_mixed, expected_mixed, rtol=1e-2, atol=5e-3)


def _shared_expert_operands(rows, hidden_size, dtype=torch.bfloat16, device="cuda"):
    """A MoE block's routed output plus its shared-expert branch."""
    routed = torch.randn(rows, hidden_size, dtype=dtype, device=device)
    gate_logits = torch.randn(rows, 1, dtype=dtype, device=device)
    shared = torch.randn(rows, hidden_size, dtype=dtype, device=device)
    return PendingSharedExpertGate(routed, gate_logits, shared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_combine_kernels_fold_the_shared_expert_gate_exactly(rows):
    """Applying the shared-expert gate in the combine prologue changes no value.

    Both paths round the gated sum to the routed tensor's dtype before the
    injection reads it, so the results must agree bit for bit, not just closely.
    """
    torch.manual_seed(42)
    hc_count, hidden_size, eps = 4, 2560, 1e-6
    pending = _shared_expert_operands(rows, hidden_size)
    residual = torch.randn(rows, hc_count * hidden_size, dtype=torch.bfloat16, device="cuda")
    injection_logits = torch.randn(rows, hc_count, dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.randn(hc_count * hidden_size, dtype=torch.bfloat16, device="cuda")
    block_output = fused_sigmoid_gate_mul_add(
        pending.routed.clone(),
        pending.gate_logits,
        pending.shared,
    )

    expected_output, expected_normed = hc_combine_norm(
        residual, block_output, injection_logits, norm_weight, eps, hc_count
    )
    output, normed = hc_combine_norm(
        residual,
        pending.routed,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
        pending.shared,
        pending.gate_logits,
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(normed, expected_normed, rtol=0, atol=0)

    expected_combined = hc_combine(residual, block_output, injection_logits, hc_count)
    combined = hc_combine(
        residual,
        pending.routed,
        injection_logits,
        hc_count,
        pending.shared,
        pending.gate_logits,
    )
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_combine_takes_a_pending_shared_expert_gate_and_graphs(rows, monkeypatch):
    """A deferred gate reaches the kernels and never runs its own launch."""
    monkeypatch.setenv("TRTLLM_QWEN4_EXP_HC_DIRECT_SKINNY_GEMM", "1")
    torch.manual_seed(42)
    hc_count, hidden_size, lowrank = 4, 2560, 320
    module = Qwen4ExpHyperConnection(
        hc_count,
        hidden_size,
        lowrank,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ).eval()
    _initialize_test_weights(module)
    pending = _shared_expert_operands(rows, hidden_size)
    hyper_input = torch.randn(rows, hc_count * hidden_size, dtype=torch.bfloat16, device="cuda")
    _, previous_residual = module.mix(hyper_input)
    block_output = fused_sigmoid_gate_mul_add(
        pending.routed.clone(),
        pending.gate_logits,
        pending.shared,
    )

    evaluated = []

    def _record_evaluate(self):
        evaluated.append(self)
        return fused_sigmoid_gate_mul_add(self.routed.clone(), self.gate_logits, self.shared)

    monkeypatch.setattr(PendingSharedExpertGate, "evaluate", _record_evaluate)

    expected_hidden, expected_mixed, expected_residual = module.combine_and_mix(
        block_output, previous_residual
    )
    hidden, mixed, residual = module.combine_and_mix(pending, previous_residual)
    assert not evaluated
    torch.testing.assert_close(hidden, expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)
    torch.testing.assert_close(residual[2], expected_residual[2], rtol=0, atol=0)
    torch.testing.assert_close(
        module.combine(pending, previous_residual),
        module.combine(block_output, previous_residual),
        rtol=0,
        atol=0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hidden, graph_mixed, _ = module.combine_and_mix(pending, previous_residual)
    graph.replay()
    torch.cuda.synchronize()
    assert not evaluated
    torch.testing.assert_close(graph_hidden, hidden, rtol=0, atol=0)
    torch.testing.assert_close(graph_mixed, mixed, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_pending_shared_expert_gate_evaluates_to_the_moe_block_output():
    """The unfused paths' materialization is what the MoE block would return."""
    torch.manual_seed(42)
    pending = _shared_expert_operands(2, 2560)
    expected = fused_sigmoid_gate_mul_add(
        pending.routed.clone(),
        pending.gate_logits,
        pending.shared,
    )
    torch.testing.assert_close(pending.evaluate(), expected, rtol=0, atol=0)
    # A block output that never deferred its gate passes straight through.
    assert Qwen4ExpHyperConnection._evaluated_block_output(expected) is expected
